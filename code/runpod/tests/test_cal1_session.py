"""pod/cal1_session.sh: CAL-1 generated on one fresh pod, unattended (serve bf16, smoke gate, provenance, plan, run, verify).

No GPU, no network, no pod. The script runs from a copy of pod/ whose /workspace is a temp folder and whose steps
(stop_server.sh, serve_olala.sh, smoke.sh, quant/gptq/pod_cal1.sh, the python that checks `datasets`) are stubs that
record their arguments, and whether results/cal1.done existed at that moment, in one calls log: the ORDER of the steps and
the moment cal1.done appears are what the tests read.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
import time
from pathlib import Path

import pytest
from conftest import REPO, RUNPOD, run_script, stub_calls

POD = RUNPOD / "pod"
SHA_A = "a" * 40
SHA_B = "b" * 40
IMAGE_TXT = (
    "image=docker.io/ovhaillm/olala-vllm:ebbadef53\n"
    "digest=sha256:842ca5b0e9ae6f7166a9d6fcfd79d4f6fe33d9a20d14d55220cc95b11f5c3c34\n"
    "CUDA_VERSION=13.0.3\n"
)
WEIGHTS = {
    "config.json": b'{"model_type": "olala"}\n',
    "model-00001-of-00002.safetensors": b"first shard bytes",
    "model-00002-of-00002.safetensors": b"second shard bytes",
    "model.safetensors.index.json": b'{"weight_map": {}}\n',
}
SERVE_COMMAND = "vllm serve /workspace/models/olala-bf16 --served-model-name olala --port 8000 --max-model-len 32688"
MARKERS = ("CAL1_SESSION_DONE", "CAL1_SESSION_FAILED")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def weights_identity(files: dict[str, bytes]) -> str:
    """The identity hash the session documents: sha256 of the sorted `sha256  name` listing of the shards."""
    listing = "".join(
        f"{sha(data)}  {name}\n"
        for name, data in sorted(files.items())
        if name.endswith(".safetensors")
    )
    return hashlib.sha256(listing.encode()).hexdigest()


def stub_script(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


class Session:
    """A pod's /workspace in a temp folder with every step of cal1_session.sh replaced by a recording stub."""

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.work = tmp / "ws"
        self.pod = self.work / "bundle" / "pod"
        self.gptq = self.work / "bundle" / "quant" / "gptq"
        self.calls = tmp / "calls.log"
        self.calls.write_text("")
        self.pylog = tmp / "python.log"
        self.pylog.write_text("")
        shutil.copytree(POD, self.pod)
        script = self.pod / "cal1_session.sh"
        script.write_text(script.read_text().replace("/workspace", str(self.work)))
        self.gptq.mkdir(parents=True)
        shutil.copy(
            REPO / "quant" / "gptq" / "cal1_revisions.txt", self.gptq / "cal1_revisions.txt"
        )
        done = self.results / "cal1.done"
        c, w = str(self.calls), str(self.work)

        def rec(name: str) -> str:
            return f'echo "{name} $* ::done=$([ -e "{done}" ] && echo y || echo n)" >> "{c}"'

        stub_script(
            self.pod / "stop_server.sh", f'{rec("stop_server.sh")}\nexit "${{STOP_RC:-0}}"\n'
        )
        stub_script(
            self.pod / "serve_olala.sh",
            f"""{rec("serve_olala.sh")}
n=$(cat "{tmp}/serve.count" 2>/dev/null || echo 0); n=$((n + 1)); echo $n > "{tmp}/serve.count"
mkdir -p "{w}/logs/$1"
if [ -z "${{SERVE_NO_CMD:-}}" ]; then
  {{ echo "utc=2026-11-20T10:00:0$n"; echo "arm=$1"; echo "MAX_JOBS=$n"
    echo "command: ${{SERVE_COMMAND:-{SERVE_COMMAND}}}"; echo "VLLM_FOO=bar"; }} > "{w}/logs/$1/server-0000$n.cmd"
fi
echo "started $1"
exit "${{SERVE_RC:-0}}"
""",
        )
        stub_script(
            self.pod / "smoke.sh",
            f'{rec("smoke.sh")}\necho "smoke verdict"\nexit "${{SMOKE_RC:-0}}"\n',
        )
        stub_script(
            self.gptq / "pod_cal1.sh",
            f"""{rec("pod_cal1.sh")}
step=$1
{{ echo "CAL1_REVISIONS=${{CAL1_REVISIONS-}}"; echo "CAL1_PROVENANCE=${{CAL1_PROVENANCE-}}"
  echo "CAL1_OUT=${{CAL1_OUT-}}"; echo "PYTHON=${{PYTHON-}}"; }} > "{w}/env-$step.txt"
out=${{CAL1_OUT:-{w}/calib-cal1}}
mkdir -p "{w}/logs/cal1"
case $step in
  plan) echo "plan output"; echo CAL1_PLAN_DONE; exit "${{PLAN_RC:-0}}" ;;
  run)
    n=$(wc -l < "$out/generations.jsonl" 2> /dev/null | tr -d ' '); n=${{n:-0}}
    echo "$n" >> "{w}/run-sees.txt"
    echo "tool log of the run $$" > "{w}/logs/cal1/generate-$$.log"
    mkdir -p "$out"; echo "row $((n + 1))" >> "$out/generations.jsonl"
    if [ -n "${{LEAK:-}}" ]; then
      echo '{{"ok": false, "n_hits": 2}}' > "$out/leakage_audit.json"; rm -f "$out/manifest.json"
      echo "REFUSED (leakage): 2 prompts share an 8-gram with an item" >&2; echo CAL1_RUN_FAILED; exit 1
    fi
    [ "${{RUN_RC:-0}}" = 0 ] || {{ echo CAL1_RUN_FAILED; exit "$RUN_RC"; }}
    [ -n "${{RUN_NOMANIFEST:-}}" ] || echo '{{"n_windows": 3}}' > "$out/manifest.json"
    echo CAL1_RUN_DONE ;;
  verify) echo CAL1_VERIFY_DONE; exit "${{VERIFY_RC:-0}}" ;;
esac
""",
        )
        self.bin = tmp / "bin"
        self.bin.mkdir()
        stub_script(self.bin / "nvidia-smi", "echo 'NVIDIA RTX PRO 4500 Blackwell, 580.95'\n")
        # a cp that notes whether cal1.done exists, then copies: nothing may be copied after cal1.done
        stub_script(
            self.bin / "cp",
            f'echo "cp ::done=$([ -e "{done}" ] && echo y || echo n)" >> "{c}"\nexec /bin/cp "$@"\n',
        )
        if shutil.which("sha256sum") is None:
            stub_script(self.bin / "sha256sum", 'exec shasum -a 256 "$@"\n')
        # the python of pod_cal1.sh: `-c 'import datasets'` answers DS_RC; `-m venv` makes a venv whose python works
        stub_script(
            self.bin / "pystub",
            f"""echo "pystub $*" >> "{self.pylog}"
case "$1" in
  -c) exit "${{DS_RC:-0}}" ;;
  -m) if [ "$2" = venv ]; then
        [ "${{VENV_RC:-0}}" = 0 ] || exit "$VENV_RC"
        mkdir -p "$4/bin"
        printf '#!/bin/bash\\necho "venvpy $*" >> "{self.pylog}"\\n[ "$2" = pip ] && exit "${{PIP_RC:-0}}"\\nexit 0\\n' > "$4/bin/python3"
        chmod +x "$4/bin/python3"
        exit 0
      fi ;;
esac
exit 0
""",
        )
        self.weights = self.work / "models" / "olala-bf16"
        self.weights.mkdir(parents=True)
        for name, data in WEIGHTS.items():
            (self.weights / name).write_bytes(data)
        (self.work / "logs" / "setup").mkdir(parents=True)
        (self.work / "logs" / "setup" / "image.txt").write_text(IMAGE_TXT)
        (self.work / "results").mkdir(parents=True, exist_ok=True)

    # --- paths -----------------------------------------------------------------------------------------------------
    @property
    def results(self) -> Path:
        return self.work / "results"

    @property
    def rdir(self) -> Path:
        return self.results / "cal1"

    @property
    def done(self) -> Path:
        return self.results / "cal1.done"

    @property
    def out(self) -> Path:
        return self.work / "calib-cal1"

    @property
    def revfile(self) -> Path:
        return self.gptq / "cal1_revisions.txt"

    # --- running ---------------------------------------------------------------------------------------------------
    def run(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(self.pod / "cal1_session.sh"), *args],
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "CAL1_STOP_WAIT": "0",
                "PYTHON": str(self.bin / "pystub"),
                **env,
                "PATH": f"{self.bin}:{os.environ['PATH']}",
            },
            check=False,
            timeout=60,
        )

    def steps(self) -> list[str]:
        return [ln for ln in self.calls.read_text().splitlines() if ln and not ln.startswith("cp ")]

    def names(self) -> list[str]:
        return [s.split("::")[0].strip() for s in self.steps()]

    def plain(self) -> list[str]:
        return [
            n.split()[0] + (" " + n.split()[1] if n.startswith("pod_cal1") else "")
            for n in self.names()
        ]

    def done_flags(self, prefix: str) -> list[str]:
        return [
            ln.rsplit("::done=", 1)[1]
            for ln in self.calls.read_text().splitlines()
            if ln.startswith(prefix)
        ]

    def env_of(self, step: str) -> dict[str, str]:
        lines = (self.work / f"env-{step}.txt").read_text().splitlines()
        return dict(ln.split("=", 1) for ln in lines)

    def provenance(self, step: str = "run") -> str:
        return self.env_of(step)["CAL1_PROVENANCE"]


@pytest.fixture
def session(tmp_path):
    return Session(tmp_path)


def last_line(out: subprocess.CompletedProcess[str]) -> str:
    return out.stdout.strip().splitlines()[-1]


def tree(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def results_files(session: Session) -> list[Path]:
    return [p for p in session.rdir.rglob("*") if p.is_file()]


# --- the script itself -------------------------------------------------------------------------------------------------


def test_the_script_exists_and_parses():
    path = POD / "cal1_session.sh"
    assert path.exists()
    assert subprocess.run(["bash", "-n", str(path)], check=False).returncode == 0


# --- dry run -----------------------------------------------------------------------------------------------------------


def test_dry_run_prints_the_plan_and_touches_nothing(session):
    before = tree(session.work)
    out = session.run("--dry-run")
    assert out.returncode == 0, out.stdout + out.stderr
    text = out.stdout
    assert "serve_olala.sh bf16" in text
    assert "smoke.sh bf16 --timeout 900" in text
    assert "pod_cal1.sh plan" in text and "pod_cal1.sh run" in text and "pod_cal1.sh verify" in text
    assert "math_train=21a5633873b6a120296cce3e2df9d5550074f4a3" in text  # the bundled revisions
    assert "cal1.done" in text and "LAST" in text.upper()
    assert not last_line(out).startswith("CAL1_SESSION_FAILED")
    assert session.steps() == [] and tree(session.work) == before


def test_dry_run_uses_no_network_secret_or_gpu_tool(session, tmp_path):
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    for tool in ("security", "curl", "ssh", "scp", "nvidia-smi", "pgrep", "vllm"):
        stub_script(stubs / tool, f'echo "{tool} $*" >> "{stubs}/calls.log"\nexit 99\n')
    out = run_script(
        session.pod / "cal1_session.sh",
        "--dry-run",
        env={"PATH": f"{stubs}:{os.environ['PATH']}", "HOME": str(tmp_path)},
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert stub_calls(stubs) == ""


def test_dry_run_finds_the_revisions_of_the_repo_when_there_is_no_bundle(tmp_path):
    # the Mac has no /workspace/bundle: the dry run reads quant/gptq/cal1_revisions.txt of the repo (launch_cal1.sh)
    out = run_script("pod/cal1_session.sh", "--dry-run", env={"PATH": os.environ["PATH"]})
    assert out.returncode == 0, out.stdout + out.stderr
    assert "mgsm_train_parallel=" in out.stdout


# --- argument and revision refusal --------------------------------------------------------------------------------------


@pytest.mark.parametrize("args", [("--bogus",), ("extra",), ("--dry-run", "extra")])
def test_bad_arguments_are_refused_with_exit_2_before_anything_runs(session, args):
    out = session.run(*args)
    assert out.returncode == 2, out.stdout + out.stderr
    assert last_line(out).startswith("CAL1_SESSION_FAILED") and "usage" in out.stderr.lower()
    assert session.steps() == [] and not session.rdir.exists() and not session.done.exists()


@pytest.mark.parametrize("out_dir", ["relative/dir", "/tmp/a b", "/tmp/x;rm", "/tmp/../etc", "/"])
def test_an_unsafe_output_directory_is_refused(session, out_dir):
    out = session.run(CAL1_OUT=out_dir)
    assert out.returncode == 2, out.stdout + out.stderr
    assert "CAL1_OUT" in out.stdout + out.stderr and session.steps() == []


def test_the_output_directory_may_not_be_the_results_directory_it_is_copied_into(session):
    for bad in (session.rdir, session.rdir / "sub", session.results, session.work):
        out = session.run(CAL1_OUT=str(bad))
        assert out.returncode == 2 and "CAL1_OUT" in out.stdout + out.stderr
    assert session.steps() == []


BAD_REVISIONS = {
    "missing file": None,
    "empty file": "",
    "only comments": "# nothing\n\n",
    "39 hex": f"math_train={'a' * 39}\n",
    "41 hex": f"math_train={'a' * 41}\n",
    "upper case hex": f"math_train={'A' * 40}\n",
    "not hex": f"math_train={'g' * 40}\n",
    "no equals": f"math_train {SHA_A}\n",
    "empty sha": "math_train=\n",
    "branch name": "math_train=main\n",
    "second source bad": f"math_train={SHA_A}\nmgsm_train_parallel={'b' * 39}\n",
    "extra token": f"math_train={SHA_A} extra\n",
    "duplicate source": f"math_train={SHA_A}\nmath_train={SHA_B}\n",
}


@pytest.mark.parametrize("case", list(BAD_REVISIONS))
def test_a_missing_or_malformed_revisions_file_is_refused_before_a_server_is_started(session, case):
    content = BAD_REVISIONS[case]
    if content is None:
        session.revfile.unlink()
    else:
        session.revfile.write_text(content)
    out = session.run()
    assert out.returncode == 2, out.stdout + out.stderr
    assert last_line(out).startswith("CAL1_SESSION_FAILED") and "cal1_revisions" in last_line(out)
    assert session.steps() == [] and not session.rdir.exists() and not session.done.exists()
    assert session.run("--dry-run").returncode == 2  # the Mac-side dry run refuses the same way


def test_comments_blank_lines_and_surrounding_spaces_in_the_revisions_file_are_accepted(session):
    session.revfile.write_text(
        f"# pinned 2026-11-20\n\n  math_train={SHA_A}  \nmgsm_train_parallel={SHA_B}\r\n# end\n"
    )
    out = session.run()
    assert out.returncode == 0, out.stdout + out.stderr
    assert (
        session.env_of("run")["CAL1_REVISIONS"] == f"math_train={SHA_A} mgsm_train_parallel={SHA_B}"
    )


def test_the_bundled_revisions_file_of_the_repo_passes():
    out = run_script("pod/cal1_session.sh", "--dry-run", env={"PATH": os.environ["PATH"]})
    assert out.returncode == 0, out.stdout + out.stderr


# --- order of the steps ------------------------------------------------------------------------------------------------


def test_the_whole_session_in_order_with_the_expected_arguments(session):
    out = session.run()
    assert out.returncode == 0, out.stdout + out.stderr
    assert last_line(out) == "CAL1_SESSION_DONE"
    assert session.names() == [
        "stop_server.sh",
        "serve_olala.sh bf16",
        "smoke.sh bf16 --timeout 900",
        "pod_cal1.sh plan",
        "pod_cal1.sh run",
        "pod_cal1.sh verify",
        "stop_server.sh",
    ]


@pytest.mark.parametrize("rc", ["0", "2"])
def test_a_smoke_exit_0_or_2_is_accepted(session, rc):
    out = session.run(SMOKE_RC=rc)
    assert out.returncode == 0 and last_line(out) == "CAL1_SESSION_DONE"
    assert "pod_cal1.sh run" in session.names()


@pytest.mark.parametrize(
    ("env", "reason", "last_step"),
    [
        ({"SERVE_RC": "1"}, "serve", "serve_olala.sh bf16"),
        ({"SMOKE_RC": "1"}, "smoke gate", "smoke.sh bf16 --timeout 900"),
        ({"SMOKE_RC": "3"}, "smoke gate", "smoke.sh bf16 --timeout 900"),
        ({"PLAN_RC": "2"}, "plan", "pod_cal1.sh plan"),
        ({"RUN_RC": "2"}, "run", "pod_cal1.sh run"),
        ({"VERIFY_RC": "1"}, "verify", "pod_cal1.sh verify"),
    ],
)
def test_a_failed_step_stops_the_session_names_the_reason_stops_the_server_and_runs_nothing_after(
    session, env, reason, last_step
):
    out = session.run(**env)
    assert out.returncode == 1, out.stdout + out.stderr
    assert last_line(out).startswith("CAL1_SESSION_FAILED") and reason in last_line(out)
    names = session.names()
    assert names[-2] == last_step and names[-1] == "stop_server.sh"
    assert names.count(last_step) == 1


def test_a_failed_serve_does_not_reach_the_smoke_gate(session):
    session.run(SERVE_RC="1")
    assert "smoke.sh bf16 --timeout 900" not in session.names()


def test_the_final_stop_failing_does_not_turn_a_finished_session_into_a_failure(session):
    out = session.run(STOP_RC="1")
    assert out.returncode == 0 and last_line(out) == "CAL1_SESSION_DONE"


def test_the_server_is_stopped_before_the_results_are_copied(session):
    session.run()
    lines = session.calls.read_text().splitlines()
    last_stop = max(i for i, ln in enumerate(lines) if ln.startswith("stop_server.sh"))
    first_cp = min(i for i, ln in enumerate(lines) if ln.startswith("cp "))
    assert last_stop < first_cp


def test_the_tool_gets_exactly_the_steps_and_no_extra_arguments(session):
    session.run()
    assert [n for n in session.names() if n.startswith("pod_cal1")] == [
        "pod_cal1.sh plan",
        "pod_cal1.sh run",
        "pod_cal1.sh verify",
    ]


# --- what the tool is given ---------------------------------------------------------------------------------------------


def test_the_revisions_of_the_bundled_file_reach_plan_and_run_in_file_order(session):
    session.run()
    expected = " ".join(
        ln.strip()
        for ln in session.revfile.read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    )
    assert len(expected.split()) == 4
    for step in ("plan", "run"):
        assert session.env_of(step)["CAL1_REVISIONS"] == expected


def test_the_output_directory_is_passed_to_every_step_and_defaults_to_calib_cal1(session):
    session.run()
    for step in ("plan", "run", "verify"):
        assert session.env_of(step)["CAL1_OUT"] == str(session.out)


def test_a_custom_output_directory_is_used_and_copied(session):
    custom = session.work / "elsewhere"
    out = session.run(CAL1_OUT=str(custom))
    assert out.returncode == 0, out.stdout + out.stderr
    assert session.env_of("run")["CAL1_OUT"] == str(custom)
    assert (session.rdir / "calib-cal1" / "manifest.json").exists()
    assert not session.out.exists()


# --- provenance --------------------------------------------------------------------------------------------------------


def test_the_provenance_names_the_image_the_weights_and_the_server_flags(session):
    assert session.run().returncode == 0
    prov = session.provenance()
    assert "docker.io/ovhaillm/olala-vllm:ebbadef53" in prov
    assert "sha256:842ca5b0e9ae6f7166a9d6fcfd79d4f6fe33d9a20d14d55220cc95b11f5c3c34" in prov
    assert sha(WEIGHTS["config.json"]) in prov
    assert weights_identity(WEIGHTS) in prov
    assert SERVE_COMMAND in prov  # the flags come from the server's own .cmd file
    assert "olala-bf16" in prov


def test_the_provenance_is_one_line_without_timestamps_or_per_start_noise(session):
    assert session.run().returncode == 0
    prov = session.provenance()
    assert "\n" not in prov and prov.strip() == prov
    assert "utc=" not in prov and "2026-11-20T10" not in prov and "VLLM_FOO" not in prov
    assert "server-0000" not in prov  # the log's name carries the start time


def test_the_provenance_is_the_same_on_a_rerun_because_the_tool_refuses_a_changed_one(session):
    assert session.run(RUN_RC="2").returncode == 1
    first = session.provenance()
    assert session.run().returncode == 0
    assert session.provenance() == first
    assert session.env_of("plan")["CAL1_PROVENANCE"] == first


def test_changed_weights_or_changed_flags_change_the_provenance(session, tmp_path):
    assert session.run().returncode == 0
    base = session.provenance()
    (tmp_path / "other").mkdir()
    (tmp_path / "flags").mkdir()
    other = Session(tmp_path / "other")
    (other.weights / "model-00002-of-00002.safetensors").write_bytes(b"different")
    assert other.run().returncode == 0 and other.provenance() != base
    flags = Session(tmp_path / "flags")
    assert flags.run(SERVE_COMMAND="vllm serve X --max-model-len 1").returncode == 0
    assert flags.provenance() != base and "--max-model-len 1" in flags.provenance()


def test_the_provenance_is_kept_in_the_results_with_its_details(session):
    assert session.run().returncode == 0
    assert (session.rdir / "provenance.txt").read_text() == session.provenance()
    details = next(session.rdir.glob("provenance-details-*.txt")).read_text()
    assert IMAGE_TXT.splitlines()[0] in details
    assert (
        f"{sha(WEIGHTS['model-00001-of-00002.safetensors'])}  model-00001-of-00002.safetensors"
        in details
    )
    assert SERVE_COMMAND in details and "NVIDIA RTX PRO 4500" in details
    assert (session.rdir / "server-00001.cmd").exists()  # the server's own record of its flags


@pytest.mark.parametrize(
    ("what", "reason"),
    [
        ("config", "config.json"),
        ("shards", "safetensors"),
        ("image", "image"),
    ],
)
def test_missing_provenance_inputs_fail_before_a_server_is_started(session, what, reason):
    if what == "config":
        (session.weights / "config.json").unlink()
    elif what == "shards":
        for p in session.weights.glob("*.safetensors"):
            p.unlink()
    else:
        (session.work / "logs" / "setup" / "image.txt").write_text("digest=sha256:x\n")
    out = session.run()
    assert out.returncode == 1 and last_line(out).startswith("CAL1_SESSION_FAILED")
    assert reason in last_line(out)
    assert "serve_olala.sh bf16" not in session.names() and "pod_cal1.sh run" not in session.names()
    assert session.done.read_text().startswith("CAL1_SESSION_FAILED")


def test_a_server_without_a_command_record_fails_after_the_smoke_gate_with_the_server_stopped(
    session,
):
    out = session.run(SERVE_NO_CMD="1")
    assert out.returncode == 1 and "provenance" in last_line(out) and ".cmd" in last_line(out)
    names = session.names()
    assert "pod_cal1.sh plan" not in names and names[-1] == "stop_server.sh"


def test_the_newest_command_record_is_the_one_used(session):
    old = session.work / "logs" / "bf16"
    old.mkdir(parents=True)
    (old / "server-00000.cmd").write_text("utc=x\ncommand: vllm serve OLD --stale\n")
    assert session.run().returncode == 0
    assert "--stale" not in session.provenance() and SERVE_COMMAND in session.provenance()


# --- datasets on the pod -----------------------------------------------------------------------------------------------


def test_with_datasets_importable_no_environment_is_created(session):
    assert session.run().returncode == 0
    assert "venv" not in session.pylog.read_text()
    assert session.env_of("run")["PYTHON"] == str(session.bin / "pystub")


def test_without_datasets_an_isolated_venv_is_made_with_the_pinned_version_and_used_by_the_tool(
    session,
):
    out = session.run(DS_RC="1")
    assert out.returncode == 0, out.stdout + out.stderr
    venv = session.work / "venv-cal1"
    py = session.pylog.read_text()
    assert f"-m venv --system-site-packages {venv}" in py
    assert "venvpy -m pip install" in py and "datasets==5.2.0" in py
    assert session.env_of("run")["PYTHON"] == str(venv / "bin" / "python3")


@pytest.mark.parametrize("env", [{"VENV_RC": "1"}, {"PIP_RC": "1"}])
def test_when_datasets_cannot_be_provided_the_session_fails_before_the_server_starts(session, env):
    out = session.run(DS_RC="1", **env)
    assert out.returncode == 1 and "datasets" in last_line(out)
    assert "serve_olala.sh bf16" not in session.names()


# --- the result ----------------------------------------------------------------------------------------------------------


def test_the_results_directory_holds_the_output_the_provenance_the_logs_and_the_outputs_of_every_step(
    session,
):
    assert session.run().returncode == 0
    d = session.rdir
    assert (d / "calib-cal1" / "manifest.json").read_text() == '{"n_windows": 3}\n'
    assert (d / "calib-cal1" / "generations.jsonl").read_text() == "row 1\n"
    for prefix in ("plan", "run", "verify", "smoke", "serve", "session"):
        assert list(d.glob(f"{prefix}-*")), prefix
    assert "CAL1_RUN_DONE" in next(d.glob("run-*")).read_text()
    assert "smoke verdict" in next(d.glob("smoke-*")).read_text()
    assert list((d / "pod_cal1-logs").glob("generate-*.log"))  # the tool's own logs


def test_cal1_done_holds_the_success_marker_and_is_written_last(session):
    out = session.run()
    assert out.returncode == 0
    assert session.done.read_text() == "CAL1_SESSION_DONE\n"
    for prefix in ("stop_server.sh", "serve_olala.sh", "smoke.sh", "pod_cal1.sh", "cp "):
        flags = session.done_flags(prefix)
        assert flags and set(flags) == {"n"}, prefix  # absent at every step and at every copy
    newest = max(p.stat().st_mtime_ns for p in results_files(session))
    assert session.done.stat().st_mtime_ns >= newest
    assert last_line(out) == "CAL1_SESSION_DONE"


def test_cal1_done_is_not_left_half_written_next_to_the_collectors_name(session):
    session.run()
    assert sorted(p.name for p in session.results.iterdir()) == ["cal1", "cal1.done"]


def test_a_stale_cal1_done_of_an_earlier_attempt_is_removed_before_the_first_step(session):
    session.done.write_text("CAL1_SESSION_FAILED: earlier\n")
    out = session.run()
    assert out.returncode == 0 and session.done.read_text() == "CAL1_SESSION_DONE\n"
    assert set(session.done_flags("stop_server.sh")) == {"n"}


def test_a_session_that_is_already_done_is_not_started_again(session):
    assert session.run().returncode == 0
    session.calls.write_text("")
    out = session.run()
    assert out.returncode == 2 and "already" in out.stdout + out.stderr
    assert session.steps() == [] and session.done.read_text() == "CAL1_SESSION_DONE\n"


def test_an_output_directory_without_a_manifest_is_a_failure_even_when_the_tool_says_ok(session):
    out = session.run(RUN_NOMANIFEST="1", VERIFY_RC="0")
    assert out.returncode == 1 and "manifest" in last_line(out)
    assert session.done.read_text().startswith("CAL1_SESSION_FAILED")


# --- failures keep the evidence and still release the collector -------------------------------------------------------


@pytest.mark.parametrize(
    "env",
    [{"SMOKE_RC": "1"}, {"PLAN_RC": "2"}, {"RUN_RC": "2"}, {"VERIFY_RC": "1"}, {"LEAK": "1"}],
)
def test_every_failure_writes_the_marker_into_cal1_done_last_so_the_collector_pulls_the_evidence(
    session, env
):
    out = session.run(**env)
    assert out.returncode == 1
    marker = last_line(out)
    assert marker.startswith("CAL1_SESSION_FAILED:")
    assert session.done.read_text() == marker + "\n"
    assert set(session.done_flags("cp ")) <= {"n"}
    assert list(session.rdir.glob("session-*.log"))
    assert session.done.stat().st_mtime_ns >= max(
        p.stat().st_mtime_ns for p in results_files(session)
    )


def test_a_leakage_refusal_is_a_failed_session_with_the_audit_output_kept(session):
    out = session.run(LEAK="1")
    assert out.returncode == 1
    assert last_line(out).startswith("CAL1_SESSION_FAILED") and "leak" in last_line(out).lower()
    assert "pod_cal1.sh verify" not in session.names()
    assert session.names()[-1] == "stop_server.sh"
    assert "REFUSED (leakage)" in next(session.rdir.glob("run-*")).read_text()
    assert '"n_hits": 2' in (session.rdir / "calib-cal1" / "leakage_audit.json").read_text()
    assert (session.out / "leakage_audit.json").exists()  # nothing is deleted at the source either
    assert (session.rdir / "calib-cal1" / "generations.jsonl").exists()


def test_a_plain_run_failure_is_not_called_a_leakage_refusal(session):
    out = session.run(RUN_RC="2")
    assert "leak" not in last_line(out).lower() and "run" in last_line(out)


def test_a_failure_before_any_server_keeps_the_session_log_and_still_writes_the_marker(session):
    (session.weights / "config.json").unlink()
    assert session.run().returncode == 1
    assert list(session.rdir.glob("session-*.log")) and session.done.exists()


# --- resume ------------------------------------------------------------------------------------------------------------


def test_a_second_run_continues_and_deletes_nothing_of_the_first(session):
    assert session.run(RUN_RC="2").returncode == 1
    first_files = {p.name: p.read_text() for p in session.rdir.glob("*-*") if p.is_file()}
    (session.rdir / "earlier-evidence.txt").write_text("keep me")
    assert (session.out / "generations.jsonl").read_text() == "row 1\n"
    out = session.run()
    assert out.returncode == 0, out.stdout + out.stderr
    # the tool saw the earlier rows: its output directory was not cleared, so it resumed
    assert (session.work / "run-sees.txt").read_text().split() == ["0", "1"]
    assert (session.out / "generations.jsonl").read_text() == "row 1\nrow 2\n"
    assert (session.rdir / "calib-cal1" / "generations.jsonl").read_text() == "row 1\nrow 2\n"
    assert (session.rdir / "earlier-evidence.txt").read_text() == "keep me"
    for (
        name,
        text,
    ) in first_files.items():  # every log of the first attempt is still there, unchanged
        assert (session.rdir / name).read_text() == text, name
    assert (
        len(list(session.rdir.glob("run-*"))) == 2
        and len(list(session.rdir.glob("session-*.log"))) == 2
    )


def test_a_rerun_after_a_failed_session_does_not_start_over_the_generation(session):
    session.run(RUN_RC="2")
    session.run()
    assert (session.work / "run-sees.txt").read_text().split()[-1] == "1"


def test_the_earlier_provenance_file_is_kept_when_the_provenance_changed(session):
    assert session.run(RUN_RC="2").returncode == 1
    (session.weights / "model-00002-of-00002.safetensors").write_bytes(b"changed")
    session.run(RUN_RC="2")
    kept = [p.read_text() for p in session.rdir.glob("provenance-before-*.txt")]
    assert len(kept) == 1 and weights_identity(WEIGHTS) in kept[0]


# --- one session at a time ---------------------------------------------------------------------------------------------


def test_a_second_session_while_one_is_running_is_refused_and_touches_no_server(session):
    run_dir = session.work / "run"
    run_dir.mkdir(parents=True)
    other = subprocess.Popen(["bash", "-c", "sleep 30; :", "cal1_session.sh"])
    try:
        (run_dir / "cal1_session.pid").write_text(f"{other.pid}\n")
        time.sleep(0.2)
        out = session.run()
        assert out.returncode == 2 and "running" in out.stdout + out.stderr
        assert session.steps() == [] and not session.done.exists()
    finally:
        other.kill()
        other.wait()


def test_a_stale_pid_file_is_ignored_and_the_pid_file_is_removed_at_the_end(session):
    run_dir = session.work / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "cal1_session.pid").write_text("999999\n")
    assert session.run().returncode == 0
    assert not (run_dir / "cal1_session.pid").exists()


def test_the_pid_file_is_removed_after_a_failure_too(session):
    session.run(SMOKE_RC="1")
    assert not (session.work / "run" / "cal1_session.pid").exists()
