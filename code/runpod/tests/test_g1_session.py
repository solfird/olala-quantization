"""pod/g1_session.sh: arm G1 (GPTQ routed experts from CAL-1, the rest bf16) built and gated on one long-lived pod.

No GPU, no network, no pod. The script runs from a copy of pod/ whose /workspace is a temp folder and whose steps
(stop_server.sh, serve_olala.sh, smoke.sh, run_accuracy.sh, quant/gptq/pod_g1lite.sh, evals/fidelity.py, the loader patch)
are stubs that record their arguments, and whether results/g1.done existed at that moment, in one calls log: the ORDER of
the steps, the arithmetic of the gate and the moment g1.done appears are what the tests read.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import time
from pathlib import Path

import pytest
from conftest import REPO, RUNPOD
from shellcheck_helper import shellcheck_command
from test_g1_tools import calib_dir

POD = RUNPOD / "pod"
SCRIPT = POD / "g1_session.sh"
STEPS = ("check", "feasibility", "build", "verify", "fidelity", "gate", "accuracy")
KEEP_ARG = ("pod_g1lite.sh", "fidelity", "serve_olala.sh", "smoke.sh", "run_accuracy.sh")
FINAL = ("G1_SESSION_DONE", "G1_SESSION_GATE_FAILED", "G1_SESSION_FAILED")


def stub_script(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


class Session:
    """A pod's /workspace in a temp folder with every step of g1_session.sh replaced by a recording stub."""

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        tmp.mkdir(parents=True, exist_ok=True)
        self.work = tmp / "ws"
        self.bundle = self.work / "bundle"
        self.pod = self.bundle / "pod"
        self.gptq = self.bundle / "quant" / "gptq"
        self.calls = tmp / "calls.log"
        self.calls.write_text("")
        self.offset = tmp / "time.offset"
        shutil.copytree(POD, self.pod)
        for name in ("g1_session.sh", "job_common.sh"):  # they hard-code /workspace
            path = self.pod / name
            path.write_text(path.read_text().replace("/workspace", str(self.work)))
        done = self.results / "g1.done"
        c, w = str(self.calls), str(self.work)
        flag = f'$([ -e "{done}" ] && echo y || echo n)'

        def rec(name: str) -> str:
            return f'echo "{name} $* ::done={flag}" >> "{c}"'

        with (self.pod / "job_common.sh").open("a") as f:  # the loader patch needs the real vLLM
            f.write(
                f'\napply_patch() {{ echo "apply_patch ::done={flag}" >> "{c}"; return "${{PATCH_RC:-0}}"; }}\n'
            )
        stub_script(
            self.pod / "stop_server.sh", f'{rec("stop_server.sh")}\nexit "${{STOP_RC:-0}}"\n'
        )
        stub_script(
            self.pod / "serve_olala.sh",
            f"""{rec("serve_olala.sh")}
if [ "$1" = ckpt ]; then name=ckpt-$(basename "$2"); else name=$1; fi
mkdir -p "{w}/logs/$name"
printf '%s' "${{SERVER_LOG_TEXT:-INFO Using MARLIN NvFp4 MoE backend}}" > "{w}/logs/$name/server-1.log"
echo "command: vllm serve" > "{w}/logs/$name/server-1.cmd"
echo "started $1"
case "$1" in bf16) exit "${{REF_SERVE_RC:-0}}" ;; *) exit "${{SERVE_RC:-0}}" ;; esac
""",
        )
        stub_script(
            self.pod / "smoke.sh",
            f"""{rec("smoke.sh")}
echo "smoke verdict $1"
case "$1" in bf16) exit "${{REF_SMOKE_RC:-0}}" ;; *) exit "${{SMOKE_RC:-0}}" ;; esac
""",
        )
        stub_script(
            self.pod / "run_accuracy.sh",
            f"""{rec("run_accuracy.sh")}
mkdir -p "{w}/results/raw/$1"
if [ -z "${{ACC_NO_INFO:-}}" ]; then echo '{{"status": "'"${{ACC_STATUS:-complete}}"'"}}' > "{w}/results/raw/$1/run_info.json"; fi
echo RUN_ACCURACY_DONE
exit "${{ACC_RC:-0}}"
""",
        )
        self.gptq.mkdir(parents=True)
        stub_script(
            self.gptq / "pod_g1lite.sh",
            f"""{rec("pod_g1lite.sh")}
echo "env G1_CALIB=${{G1_CALIB-}} G1_BASE_SELECT=${{G1_BASE_SELECT-}} G1_DST=${{G1_DST-}}" >> "{tmp}/g1env.log"
step=$1
mkdir -p "{w}/logs/g1lite"
rc=0
[ "${{FAIL_STEP:-}}" != "$step" ] || rc=1
case $step in
  check) marker=CHECK; echo "versions ok" ;;
  feasibility)
    marker=FEASIBILITY
    echo "every layer that was run (forward only, or captured):"
    echo "layer  captured  forward_s  mixer_s  mlp_s  peak_cuda_GB  max|mixer_out|  max|mlp_out|"
    for l in 2 3 4; do
      echo "    $l      True      ${{FEAS_FWD:-5.0}}      2.0    1.0  ${{FEAS_PEAK:-20.00}}       1.5e+03          2.5"
    done
    echo "calibrated layers:"
    echo "layer  hessian_s  solve_s  total_s  peak_after_solve_GB  gptq/rtn  mean_loss_ratio_up  mean_heldout_ratio"
    echo "    2       10.0     20.0     30.0                10.00  250/6  0.5  0.9" ;;
  build)
    marker=BUILD
    mkdir -p "$G1_DST"
    for n in model.safetensors.index.json config.json quantization_info.json; do echo "{{\\"n\\": \\"$n\\"}}" > "$G1_DST/$n"; done
    head -c 4096 /dev/zero > "$G1_DST/model-00001-of-00001.safetensors"
    echo "built" ;;
  verify) marker=VERIFY; echo '{{"ok": true}}' > "{w}/logs/g1lite/verify-1.json"; echo "verify ok" ;;
esac
if [ -n "${{BUMP_AFTER:-}}" ] && [ "$BUMP_AFTER" = "$step" ]; then echo "${{BUMP_SECONDS:-0}}" > "{self.offset}"; fi
if [ $rc -eq 0 ] && [ "${{NO_MARKER:-}}" != "$step" ]; then echo "G1LITE_${{marker}}_DONE"; else echo "G1LITE_${{marker}}_FAILED"; fi
exit $rc
""",
        )
        evals = self.bundle / "evals"
        evals.mkdir(parents=True)
        (evals / "fidelity.py").write_text(
            f"""import json, os, pathlib, sys
a = sys.argv[1:]
open({c!r}, "a").write("fidelity " + " ".join(a) + " ::cwd=" + os.getcwd() + " ::done=" + ("y" if os.path.exists({str(done)!r}) else "n") + "\\n")
if a[0] == "collect":
    out = pathlib.Path(a[a.index("--out") + 1])
    if os.environ.get("COLLECT_RC", "0") != "0" and out.name.startswith("g1_"):
        print("collect boom"); sys.exit(1)
    if os.environ.get("REF_COLLECT_RC", "0") != "0" and out.name.startswith("b0_"):
        print("reference collect boom"); sys.exit(1)
    if os.environ.get("REPEAT_COLLECT_RC", "0") != "0" and out.name.startswith("b0repeat"):
        print("repeat collect boom"); sys.exit(1)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not (os.environ.get("COLLECT_NO_FILE") and out.name.startswith("g1_")):
        out.write_bytes(b"npz:" + out.name.encode())
    print("wrote " + str(out))
else:
    if os.environ.get("COMPARE_RC", "0") != "0":
        print("compare boom"); sys.exit(1)
    kl = float(os.environ.get("FID_KL", "0.0150")); ppl = float(os.environ.get("FID_PPL", "1.0100"))
    def iv(v): return {{"value": v, "lo": v * 0.9, "hi": v * 1.1}}
    data = {{"ppl_ref": iv(14.6), "ppl_var": iv(14.7), "ppl_ratio": iv(ppl), "kld_mean": iv(kl), "kld_median": iv(0.004),
             "kld_p99": iv(0.2), "top1_agreement": iv(0.93), "n_chunks": 64, "n_positions": 32704, "k": 20,
             "n_boot": 1000, "seed": 0, "confidence": 0.95}}
    data["ppl_ratio"]["value"] = ppl
    data["kld_mean"]["value"] = kl
    pathlib.Path(a[a.index("--json") + 1]).write_text(json.dumps(data))
    print("### Logit fidelity")
"""
        )
        self.bin = tmp / "bin"
        self.bin.mkdir()
        stub_script(self.bin / "nvidia-smi", "echo 'NVIDIA RTX PRO 4500 Blackwell, 580.95'\n")
        stub_script(
            self.bin / "cp",
            f'echo "cp ::done={flag}" >> "{c}"\nexec /bin/cp "$@"\n',
        )
        # a clock the tests can move: `date +%s` is the real time plus the offset a stub writes
        stub_script(
            self.bin / "date",
            f"""if [ "${{1:-}}" = "+%s" ]; then
  off=$(cat "{self.offset}" 2>/dev/null || echo 0)
  if [ -n "${{BUMP_ON_GATE:-}}" ] && [ -e "{self.rdir}/gate.json" ]; then off=$((off + BUMP_ON_GATE)); fi
  echo $(( $(/bin/date +%s) + off ))
else exec /bin/date "$@"; fi
""",
        )
        # an mv that records its arguments: g1.done must arrive by a rename of a finished temporary file
        stub_script(
            self.bin / "mv",
            f'echo "mv $*" >> "{tmp}/mv.log"\nexec /bin/mv "$@"\n',
        )
        if shutil.which("sha256sum") is None:
            stub_script(self.bin / "sha256sum", 'exec shasum -a 256 "$@"\n')
        (self.work / "models" / "olala-bf16").mkdir(parents=True)
        (self.work / "models" / "olala-bf16" / "config.json").write_text("{}")
        (self.work / "logs" / "setup").mkdir(parents=True)
        (self.work / "logs" / "setup" / "evals-env-1.log").write_text("EVALS_ENV_DONE\n")
        self.results.mkdir(parents=True, exist_ok=True)
        (self.work / "cal1").mkdir(parents=True)
        self.calib = calib_dir(self.work / "cal1")  # /workspace/cal1/calib-cal1
        self.calib_files = {p.name: p.read_bytes() for p in self.calib.iterdir()}

    # --- paths -----------------------------------------------------------------------------------------------------
    @property
    def results(self) -> Path:
        return self.work / "results"

    @property
    def rdir(self) -> Path:
        return self.results / "g1"

    @property
    def done(self) -> Path:
        return self.results / "g1.done"

    @property
    def ckpt(self) -> Path:
        return self.work / "ckpt" / "g1lite"

    @property
    def ref(self) -> Path:
        return self.results / "raw" / "b0_wikitext2.npz"

    # --- running ---------------------------------------------------------------------------------------------------
    def make_reference(self) -> None:
        self.ref.parent.mkdir(parents=True, exist_ok=True)
        self.ref.write_bytes(b"a reference collected earlier on this pod")

    def deadline_in(self, minutes: float) -> str:
        return str(int(time.time() + minutes * 60))

    def run(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        full = {
            **os.environ,
            "G1_STOP_WAIT": "0",
            "G1_EVALS_POLL": "0",
            "G1_POD_DEADLINE_EPOCH": self.deadline_in(300),
            **env,
            "PATH": f"{self.bin}:{os.environ['PATH']}",
        }
        return subprocess.run(
            ["bash", str(self.pod / "g1_session.sh"), *args],
            capture_output=True,
            text=True,
            env=full,
            check=False,
            timeout=120,
        )

    def steps(self) -> list[str]:
        return [ln for ln in self.calls.read_text().splitlines() if ln and not ln.startswith("cp ")]

    def names(self) -> list[str]:
        """The calls without their done flag; most keep their first argument."""
        out = []
        for s in self.steps():
            head = s.split("::")[0].strip().split()
            keep = 2 if head[0] in KEEP_ARG else 1
            out.append(" ".join(head[:keep]))
        return out

    def done_flags(self) -> list[str]:
        return [
            ln.rsplit("::done=", 1)[1]
            for ln in self.calls.read_text().splitlines()
            if "::done=" in ln
        ]

    def done_lines(self) -> list[str]:
        return self.done.read_text().splitlines() if self.done.exists() else []

    def gate(self) -> dict:
        return json.loads((self.rdir / "gate.json").read_text())


@pytest.fixture
def s(tmp_path):
    return Session(tmp_path)


def last_line(out: subprocess.CompletedProcess[str]) -> str:
    return out.stdout.strip().splitlines()[-1]


def all_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file()]


FULL_PASS = [
    "stop_server.sh",
    "pod_g1lite.sh check",
    "pod_g1lite.sh feasibility",
    "pod_g1lite.sh build",
    "pod_g1lite.sh verify",
    "stop_server.sh",
    "serve_olala.sh bf16",
    "smoke.sh bf16",
    "fidelity collect",
    "fidelity collect",
    "stop_server.sh",
    "fidelity compare",
    "apply_patch",
    "serve_olala.sh ckpt",
    "smoke.sh ckpt-g1lite",
    "fidelity collect",
    "fidelity compare",
    "run_accuracy.sh G1-v2",
    "stop_server.sh",
]


# --- the script itself ---------------------------------------------------------------------------------------------


def test_the_script_exists_and_parses():
    assert SCRIPT.exists()
    assert subprocess.run(["bash", "-n", str(SCRIPT)], check=False).returncode == 0


@pytest.mark.skipif(shellcheck_command() is None, reason="neither shellcheck nor uvx is installed")
def test_the_script_is_shellcheck_clean():
    out = subprocess.run(
        [*shellcheck_command(), "-x", "-P", "SCRIPTDIR", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stdout


def code_lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text().splitlines() if not ln.lstrip().startswith("#")]


def test_the_thresholds_live_in_one_place_with_their_source():
    text = SCRIPT.read_text()
    code = "\n".join(code_lines(SCRIPT))
    assert len(re.findall(r"^G1_GATE_KL_MAX=0\.0194\b", code, re.M)) == 1
    assert len(re.findall(r"^G1_GATE_PPL_RATIO_MAX=1\.0197\b", code, re.M)) == 1
    assert code.count("0.0194") == 1  # no second copy of the KL threshold anywhere in the code
    lines = text.splitlines()
    at = next(i for i, ln in enumerate(lines) if ln.startswith("G1_GATE_KL_MAX="))
    assert any("amendment 22" in ln for ln in lines[max(0, at - 5) : at + 1])  # the source is named
    # the number is not in the tool (it takes it as an argument) and not in the launcher
    for other in (POD / "g1_tools.py", RUNPOD / "launch_g1.sh"):
        if other.exists():
            assert "0.0194" not in other.read_text(), other


def test_the_header_states_the_projection_formula():
    head = SCRIPT.read_text().split("set -u")[0]
    assert "72.75" in head and "36" in head and "291" in head
    assert "G1_BUILD_BUDGET_HOURS" in head and "30 GiB" in head


def test_fidelity_uses_the_same_chunks_and_tool_arguments_as_the_ablation_arm():
    mine = SCRIPT.read_text()
    theirs = (POD / "ablation_arm.sh").read_text()
    for needle in (
        "data/wikitext2_test_64x512.json",
        "fidelity.py collect --chunks",
        "--base-url http://127.0.0.1:8000/v1 --model olala",
        "fidelity.py compare",
        "b0_wikitext2.npz",
    ):
        assert needle in mine and needle in theirs, needle


# --- dry run ---------------------------------------------------------------------------------------------------------


def tree(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def test_dry_run_prints_the_plan_and_touches_nothing(s):
    before = tree(s.work)
    out = s.run("--dry-run", G1_POD_DEADLINE_EPOCH="")
    assert out.returncode == 0, out.stdout + out.stderr
    assert tree(s.work) == before and s.steps() == [] and not s.done.exists()
    text = out.stdout
    positions = [text.index(f"step {n}") for n in STEPS]
    assert positions == sorted(positions)
    for needle in (
        "0.0194",
        "1.0197",
        "72.75",
        "G1_GATE_PASS",
        "G1_SESSION_GATE_FAILED",
        "20 minutes",
    ):
        assert needle in text, needle
    assert "pod_g1lite.sh check" in text and "run_accuracy.sh G1-v2 --effort medium" in text


@pytest.mark.parametrize("args", [("extra",), ("--bogus",), ("--dry-run", "x")])
def test_bad_arguments_are_refused_and_write_nothing(s, args):
    out = s.run(*args)
    assert out.returncode == 2 and "usage" in out.stderr.lower()
    assert s.steps() == [] and not s.done.exists()


@pytest.mark.parametrize("deadline", ["", "soon", "-5", "12.5"])
def test_a_missing_or_bad_pod_deadline_is_refused_before_anything_runs(s, deadline):
    out = s.run(G1_POD_DEADLINE_EPOCH=deadline)
    assert out.returncode == 2 and "G1_POD_DEADLINE_EPOCH" in out.stdout + out.stderr
    assert s.steps() == []


@pytest.mark.parametrize("budget", ["0", "-1", "abc", "4h"])
def test_a_bad_build_budget_is_refused_before_anything_runs(s, budget):
    out = s.run(G1_BUILD_BUDGET_HOURS=budget)
    assert out.returncode == 2 and "G1_BUILD_BUDGET_HOURS" in out.stdout + out.stderr
    assert s.steps() == []


# --- the full pass ---------------------------------------------------------------------------------------------------


def test_the_whole_pass_runs_the_steps_in_order_and_ends_done(s):
    out = s.run()
    assert out.returncode == 0, out.stdout + out.stderr
    assert s.names() == FULL_PASS
    assert last_line(out) == "G1_SESSION_DONE"
    assert s.done_lines()[0] == "G1_SESSION_DONE"
    assert "G1_GATE_PASS" in out.stdout and "G1_GATE_FAIL" not in out.stdout
    for step in STEPS:
        assert f"G1_STEP_{step.upper()}_DONE" in out.stdout, step
        assert list(s.rdir.glob(f"{step}-*.log")), f"no log of step {step}"


def test_pod_g1lite_gets_the_cal1_directory_the_experts_base_and_the_checkpoint_path(s):
    assert s.run().returncode == 0
    env_lines = (s.tmp / "g1env.log").read_text().splitlines()
    assert len(env_lines) == 4
    for line in env_lines:
        assert f"G1_CALIB={s.work}/cal1/calib-cal1" in line
        assert "G1_BASE_SELECT=experts" in line
        assert f"G1_DST={s.ckpt}" in line


def test_the_accuracy_run_is_the_arm_pipeline_call_all_five_tasks_effort_medium(s):
    assert s.run().returncode == 0
    call = next(x for x in s.steps() if x.startswith("run_accuracy.sh"))
    assert call.split("::")[0].strip() == "run_accuracy.sh G1-v2 --effort medium"
    arm_pipeline = (POD / "arm_pipeline.sh").read_text()
    assert 'run_accuracy.sh" "$run" --effort medium' in arm_pipeline  # the call kept as there
    assert "--tasks" not in call  # the five tasks of the primary metric are the default


def test_the_same_server_serves_the_fidelity_run_and_the_accuracy_run(s):
    assert s.run().returncode == 0
    assert s.names().count("serve_olala.sh ckpt") == 1
    names = s.names()
    assert names.index("fidelity compare") < names.index("run_accuracy.sh G1-v2")
    # no stop between the checkpoint's fidelity collection and the accuracy run
    between = names[names.index("serve_olala.sh ckpt") : names.index("run_accuracy.sh G1-v2")]
    assert "stop_server.sh" not in between


def test_the_results_hold_fidelity_gate_checkpoint_record_and_provenance(s):
    assert s.run().returncode == 0
    fid = json.loads((s.rdir / "fidelity.json").read_text())
    for key in ("mean_kl", "median_kl", "p99_kl", "ppl_ratio", "top1_agreement"):
        assert {"value", "lo", "hi"} <= set(fid[key]), key
    gate = s.gate()
    assert gate["pass"] is True and gate["kl"] == 0.015 and gate["ppl_ratio"] == 1.01
    assert gate["thresholds"] == {
        "kl_max": 0.0194,
        "ppl_ratio_max": 1.0197,
        "source": gate["thresholds"]["source"],
    }
    assert "amendment 22" in gate["thresholds"]["source"]
    assert gate["n5_reference"] == {
        "kl": 0.0243,
        "ppl_ratio": 1.0197,
        "from": "results/wave1/ablation-abl-experts",
    }
    ck = json.loads((s.rdir / "checkpoint.json").read_text())
    assert ck["size_bytes"] == sum(p.stat().st_size for p in s.ckpt.iterdir())
    for name in ("model.safetensors.index.json", "config.json", "quantization_info.json"):
        assert ck["sha256"][name] == hashlib.sha256((s.ckpt / name).read_bytes()).hexdigest()
        assert ck["sha256"][name] in (s.rdir / "provenance.txt").read_text()
    assert (s.rdir / "feasibility.json").exists() and (s.rdir / "fidelity_vs_b0.json").exists()
    assert json.loads((s.rdir / "verify.json").read_text()) == {"ok": True}  # the verify report
    assert (s.rdir / "calibration.json").exists()


def test_the_checkpoint_never_leaves_the_pod_only_its_metadata_is_in_the_results(s):
    assert s.run().returncode == 0
    for path in all_files(s.rdir):
        assert path.suffix != ".safetensors", path
        assert path.stat().st_size < 1_000_000, path
    assert s.ckpt.exists() and (s.ckpt / "model-00001-of-00001.safetensors").exists()
    assert not (s.rdir / "g1lite").exists()


def test_the_feasibility_log_states_the_formula_and_the_numbers(s):
    out = s.run()
    assert "projected build seconds = 72.75 x" in out.stdout
    log = next(s.rdir.glob("feasibility-*.log")).read_text()
    assert "72.75" in log and "projected build" in log


# --- gate arithmetic through the script ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kl", "ppl", "passes"),
    [
        ("0.0194", "1.0197", True),
        ("0.01941", "1.0197", False),
        ("0.0194", "1.01971", False),
        ("0.01939", "1.01969", True),
        ("0.0243", "1.0197", False),
    ],
)
def test_the_gate_arithmetic_at_the_boundaries(s, kl, ppl, passes):
    out = s.run(FID_KL=kl, FID_PPL=ppl)
    assert out.returncode == 0, out.stdout + out.stderr
    assert s.gate()["pass"] is passes
    assert ("G1_GATE_PASS" in out.stdout) is passes and ("G1_GATE_FAIL" in out.stdout) is not passes
    assert last_line(out) == ("G1_SESSION_DONE" if passes else "G1_SESSION_GATE_FAILED")
    assert ("run_accuracy.sh G1-v2" in " ".join(s.names())) is passes


def test_a_gate_fail_stops_before_the_accuracy_run_and_is_a_legitimate_negative_result(s):
    out = s.run(FID_KL="0.03")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "run_accuracy.sh" not in " ".join(s.names())
    assert s.names()[-1] == "stop_server.sh"  # the server is stopped, nothing is left running
    assert (
        s.done_lines()[0] == "G1_SESSION_GATE_FAILED" and last_line(out) == "G1_SESSION_GATE_FAILED"
    )
    assert "0.03" in "\n".join(s.done_lines()[1:])  # the numbers are in the .done too
    assert (s.rdir / "gate.json").exists() and not (s.rdir / "steps" / "accuracy.ok").exists()
    assert "G1_STEP_ACCURACY" not in out.stdout


def test_an_accuracy_run_that_is_not_complete_fails_the_session(s):
    out = s.run(ACC_STATUS="running")
    assert out.returncode == 1 and last_line(out).startswith("G1_SESSION_FAILED: ")
    assert "accuracy" in last_line(out)
    assert s.done_lines()[0] == last_line(out)


def test_an_accuracy_run_exit_status_is_checked(s):
    out = s.run(ACC_RC="1")
    assert out.returncode == 1 and "accuracy" in last_line(out)


def test_an_incomplete_run_directory_of_an_earlier_attempt_is_never_overwritten_or_deleted(s):
    run = s.results / "raw" / "G1-v2"
    run.mkdir(parents=True)
    (run / "samples.jsonl").write_text("partial\n")
    out = s.run()
    assert out.returncode == 1 and "G1-v2" in last_line(out)
    assert "run_accuracy.sh" not in " ".join(s.names())
    assert (run / "samples.jsonl").read_text() == "partial\n"


def test_a_complete_run_directory_is_taken_as_done(s):
    run = s.results / "raw" / "G1-v2"
    run.mkdir(parents=True)
    (run / "run_info.json").write_text('{"status": "complete"}')
    out = s.run()
    assert out.returncode == 0 and last_line(out) == "G1_SESSION_DONE"
    assert "run_accuracy.sh" not in " ".join(s.names())


def test_the_evals_environment_is_waited_for_and_a_failed_one_fails_the_session(s):
    (s.work / "logs" / "setup" / "evals-env-1.log").write_text("EVALS_ENV_FAILED: no uv\n")
    out = s.run()
    assert out.returncode == 1 and "evals environment" in last_line(out)
    assert "run_accuracy.sh" not in " ".join(s.names())
    assert (s.rdir / "gate.json").exists()  # the gate had been decided before


def test_an_evals_environment_that_never_finishes_fails_the_session(s):
    (s.work / "logs" / "setup" / "evals-env-1.log").write_text("still going\n")
    out = s.run(G1_EVALS_TRIES="2")
    assert out.returncode == 1 and "evals environment" in last_line(out)


# --- G1_ACCURACY_ALWAYS: the accuracy run happens whatever the gate says -----------------------------------------------


def test_with_the_always_switch_a_gate_fail_still_runs_the_accuracy_and_ends_done(s):
    out = s.run(FID_KL="0.03", G1_ACCURACY_ALWAYS="1")
    assert out.returncode == 0, out.stdout + out.stderr
    assert last_line(out) == "G1_SESSION_DONE" and s.done_lines()[0] == "G1_SESSION_DONE"
    reason = " ".join(s.done_lines()[1:])
    assert "gate: fail" in reason and "0.03" in reason and "accuracy run G1-v2 complete" in reason
    assert "G1_GATE_FAIL" in out.stdout and "G1_GATE_PASS" not in out.stdout
    assert "G1_SESSION_GATE_FAILED" not in out.stdout + "".join(s.done_lines())
    gate = s.gate()
    assert gate["pass"] is False and gate["kl_ok"] is False and gate["kl"] == 0.03
    call = next(x for x in s.steps() if x.startswith("run_accuracy.sh"))
    assert call.split("::")[0].strip() == "run_accuracy.sh G1-v2 --effort medium"
    assert s.names().count("serve_olala.sh ckpt") == 1 and s.names()[-1] == "stop_server.sh"
    assert (s.rdir / "steps" / "accuracy.ok").exists() and "G1_STEP_ACCURACY_DONE" in out.stdout


def test_with_the_always_switch_a_pass_says_gate_pass_in_the_reason(s):
    out = s.run(G1_ACCURACY_ALWAYS="1")
    assert last_line(out) == "G1_SESSION_DONE"
    reason = " ".join(s.done_lines()[1:])
    assert "gate: pass" in reason and "accuracy run G1-v2 complete" in reason
    assert s.gate()["pass"] is True


def test_the_boundaries_still_decide_the_verdict_with_the_always_switch(s):
    out = s.run(FID_KL="0.01941", FID_PPL="1.0197", G1_ACCURACY_ALWAYS="1")
    assert s.gate()["pass"] is False and "gate: fail" in " ".join(s.done_lines()[1:])
    assert "G1_GATE_FAIL" in out.stdout


def test_the_switch_at_0_behaves_as_before_a_gate_fail_stops_before_accuracy(s):
    out = s.run(FID_KL="0.03", G1_ACCURACY_ALWAYS="0")
    assert last_line(out) == "G1_SESSION_GATE_FAILED"
    assert "run_accuracy.sh" not in " ".join(s.names())


@pytest.mark.parametrize("value", ["2", "yes", "-1"])
def test_a_bad_always_switch_is_refused_before_anything_runs(s, value):
    out = s.run(G1_ACCURACY_ALWAYS=value)
    assert out.returncode == 2 and "G1_ACCURACY_ALWAYS" in out.stdout + out.stderr
    assert s.steps() == [] and not s.done.exists()


def test_the_deadline_guard_still_stops_the_accuracy_after_a_gate_fail_with_the_switch(s):
    out = s.run(
        FID_KL="0.03",
        G1_ACCURACY_ALWAYS="1",
        G1_POD_DEADLINE_EPOCH=s.deadline_in(120),
        BUMP_ON_GATE=str(110 * 60),
    )
    assert out.returncode == 1 and "deadline" in last_line(out) and "accuracy" in last_line(out)
    assert "run_accuracy.sh" not in " ".join(s.names())
    assert s.gate()["pass"] is False  # the verdict is kept


def test_a_failed_accuracy_run_still_fails_the_session_with_the_switch(s):
    out = s.run(FID_KL="0.03", G1_ACCURACY_ALWAYS="1", ACC_STATUS="running")
    assert out.returncode == 1 and last_line(out).startswith("G1_SESSION_FAILED: ")


def test_the_feasibility_refusal_is_unchanged_by_the_switch(s):
    out = s.run(FEAS_FWD="5.5", G1_ACCURACY_ALWAYS="1")
    assert out.returncode == 1 and "feasibility" in last_line(out)
    assert "pod_g1lite.sh build" not in s.names()


def test_a_rerun_is_refused_after_a_done_that_followed_a_gate_fail_with_the_switch(s):
    assert s.run(FID_KL="0.03", G1_ACCURACY_ALWAYS="1").returncode == 0
    again = s.run(G1_ACCURACY_ALWAYS="1")
    assert again.returncode == 2 and s.done_lines()[0] == "G1_SESSION_DONE"


def test_the_dry_run_names_the_always_switch(s):
    out = s.run("--dry-run", G1_POD_DEADLINE_EPOCH="")
    assert "G1_ACCURACY_ALWAYS" in out.stdout


def test_the_always_switch_defaults_to_0_in_the_script():
    code = "\n".join(code_lines(SCRIPT))
    assert "G1_ACCURACY_ALWAYS-0" in code


# --- feasibility refusal -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "reason"),
    [
        ({"FEAS_FWD": "5.5"}, "budget"),  # 72.75 * 36 * 5.5 = 14,404.5 s > 4 h
        ({"FEAS_FWD": "50.0"}, "budget"),
        ({"FEAS_PEAK": "32.3"}, "GiB"),  # 32.3 GB = 30.08 GiB
        ({"FEAS_PEAK": "80.00"}, "GiB"),
    ],
)
def test_the_feasibility_refusal_stops_before_the_build_with_a_failed_marker_and_done(
    s, env, reason
):
    out = s.run(**env)
    assert out.returncode == 1, out.stdout + out.stderr
    assert last_line(out).startswith("G1_SESSION_FAILED: ") and "feasibility" in last_line(out)
    assert reason in last_line(out)
    names = s.names()
    assert names[-1] == "pod_g1lite.sh feasibility"
    for later in ("build", "verify"):
        assert f"pod_g1lite.sh {later}" not in names
    assert "serve_olala.sh bf16" not in names and "serve_olala.sh ckpt" not in names
    assert s.done_lines()[0] == last_line(out)
    assert (s.rdir / "feasibility.json").exists()
    assert not (s.rdir / "steps" / "feasibility.ok").exists()
    assert "G1_STEP_FEASIBILITY_FAILED" in out.stdout


def test_the_feasibility_boundaries_just_inside_the_budget_and_the_memory_limit_pass(s):
    out = s.run(FEAS_FWD="5.4", FEAS_PEAK="32.2")  # 14,144 s = 3.93 h and 29.99 GiB
    assert out.returncode == 0, out.stdout + out.stderr
    assert last_line(out) == "G1_SESSION_DONE"


def test_the_budget_comes_from_the_environment(s):
    assert s.run(FEAS_FWD="5.5", G1_BUILD_BUDGET_HOURS="4.5").returncode == 0
    t2 = Session(s.tmp / "second")
    assert t2.run(FEAS_FWD="5.5", G1_BUILD_BUDGET_HOURS="3.9").returncode == 1


def test_the_feasibility_measurement_is_reused_by_a_rerun_but_the_decision_is_taken_again(s):
    assert s.run(FEAS_FWD="5.5").returncode == 1  # refused at the default budget of 4 h
    n_feas = s.names().count("pod_g1lite.sh feasibility")
    out = s.run(FEAS_FWD="5.5", G1_BUILD_BUDGET_HOURS="5")
    assert out.returncode == 0, out.stdout + out.stderr
    assert s.names().count("pod_g1lite.sh feasibility") == n_feas  # not measured again
    again = Session(s.tmp / "third")
    assert again.run(FEAS_FWD="5.5").returncode == 1
    assert (
        again.run(FEAS_FWD="5.5").returncode == 1
    )  # still refused at 4 h: the decision is not cached


# --- a failing step --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("step", ["check", "feasibility", "build", "verify"])
def test_a_failing_pod_g1lite_step_fails_the_session_and_stops(s, step):
    out = s.run(FAIL_STEP=step)
    assert out.returncode == 1 and last_line(out).startswith("G1_SESSION_FAILED: ")
    assert step in last_line(out)
    names = s.names()
    assert names[-1] == f"pod_g1lite.sh {step}"
    assert s.done_lines()[0] == last_line(out)
    assert f"G1_STEP_{step.upper()}_FAILED" in out.stdout


@pytest.mark.parametrize("step", ["check", "feasibility", "build", "verify"])
def test_an_exit_zero_without_the_done_marker_is_a_failure(s, step):
    out = s.run(NO_MARKER=step)
    assert out.returncode == 1 and step in last_line(out)
    assert not (s.rdir / "steps" / f"{step}.ok").exists()


# --- the fidelity step -----------------------------------------------------------------------------------------------


def test_the_bf16_reference_is_made_once_when_absent_with_the_ablation_arms_arguments(s):
    assert s.run().returncode == 0
    fid = [x for x in s.steps() if x.startswith("fidelity")]
    chunks = (
        "--chunks data/wikitext2_test_64x512.json --base-url http://127.0.0.1:8000/v1 --model olala"
    )
    assert (
        fid[0].split(" ::")[0]
        == f"fidelity collect {chunks} --out {s.work}/results/raw/b0_wikitext2.npz"
    )
    assert (
        fid[1].split(" ::")[0]
        == f"fidelity collect {chunks} --out {s.work}/results/raw/b0repeat_wikitext2.npz"
    )
    assert fid[2].split(" ::")[0] == (  # the noise floor: the two collections of the same server
        f"fidelity compare {s.work}/results/raw/b0_wikitext2.npz "
        f"{s.work}/results/raw/b0repeat_wikitext2.npz --json {s.rdir}/reference/noise_floor.json"
    )
    assert (
        fid[3].split(" ::")[0]
        == f"fidelity collect {chunks} --out {s.work}/results/raw/g1_wikitext2.npz"
    )
    cmp = fid[4].split(" ::")[0]
    assert cmp == (
        f"fidelity compare {s.work}/results/raw/b0_wikitext2.npz {s.work}/results/raw/g1_wikitext2.npz "
        f"--json {s.rdir}/fidelity_vs_b0.json"
    )
    assert all(
        f"::cwd={s.bundle}/evals" in x for x in fid
    )  # run from the evals directory, as the ablation does
    assert (s.rdir / "reference" / "b0_wikitext2.npz").exists()
    assert (s.rdir / "reference" / "noise_floor.json").exists()


def test_an_existing_reference_is_used_and_bf16_is_not_served(s):
    s.make_reference()
    assert s.run().returncode == 0
    names = s.names()
    assert "serve_olala.sh bf16" not in names and "smoke.sh bf16" not in names
    assert names.count("fidelity collect") == 1
    assert s.ref.read_bytes() == b"a reference collected earlier on this pod"


@pytest.mark.parametrize(
    ("env", "reason"),
    [
        ({"REF_SERVE_RC": "1"}, "reference"),
        ({"REF_SMOKE_RC": "1"}, "reference"),
        ({"REF_COLLECT_RC": "1"}, "reference"),
        ({"PATCH_RC": "1"}, "loader patch"),
        ({"SERVE_RC": "1"}, "serve"),
        ({"SMOKE_RC": "1"}, "smoke"),
        ({"COLLECT_RC": "1"}, "collect"),
        ({"COLLECT_NO_FILE": "1"}, "collect"),
        ({"COMPARE_RC": "1"}, "compare"),
    ],
)
def test_a_failing_fidelity_sub_step_fails_the_session_and_stops_the_server(s, env, reason):
    out = s.run(**env)
    assert out.returncode == 1 and last_line(out).startswith("G1_SESSION_FAILED: ")
    assert reason in last_line(out), last_line(out)
    names = s.names()
    assert names[-1] == (
        "apply_patch" if "PATCH_RC" in env else "stop_server.sh"
    )  # no server is left up
    assert "run_accuracy.sh" not in " ".join(names)
    assert s.done_lines()[0] == last_line(out)
    assert not (s.rdir / "steps" / "fidelity.ok").exists()


@pytest.mark.parametrize("rc", ["0", "2"])
def test_a_smoke_gate_exit_0_or_2_is_accepted(s, rc):
    assert s.run(SMOKE_RC=rc, REF_SMOKE_RC=rc).returncode == 0


def test_a_failed_noise_floor_collection_is_a_warning_not_a_failure(s):
    out = s.run(REPEAT_COLLECT_RC="1")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "warning" in out.stdout and not (s.rdir / "reference" / "noise_floor.json").exists()
    assert (s.rdir / "reference" / "b0_wikitext2.npz").exists()


# --- resume ------------------------------------------------------------------------------------------------------------


def test_a_rerun_after_a_failure_skips_finished_steps_and_never_deletes(s):
    (s.ckpt).mkdir(parents=True)
    (s.ckpt / "sentinel.txt").write_text("a partial build of an earlier attempt\n")
    first = s.run(FAIL_STEP="verify")
    assert first.returncode == 1
    earlier = {p: p.read_bytes() for p in all_files(s.rdir) if p.name.endswith((".log", ".json"))}
    second = s.run()
    assert second.returncode == 0, second.stdout + second.stderr
    assert last_line(second) == "G1_SESSION_DONE"
    calls = s.names()
    for skipped in ("pod_g1lite.sh check", "pod_g1lite.sh feasibility", "pod_g1lite.sh build"):
        assert calls.count(skipped) == 1, skipped  # run once, by the first attempt only
    assert calls.count("pod_g1lite.sh verify") == 2
    assert "G1_STEP_CHECK_SKIPPED" in second.stdout and "G1_STEP_BUILD_SKIPPED" in second.stdout
    assert (s.ckpt / "sentinel.txt").exists() and (
        s.ckpt / "model-00001-of-00001.safetensors"
    ).exists()
    for (
        path,
        data,
    ) in earlier.items():  # nothing of the first attempt was deleted, the step logs are unchanged
        assert path.exists(), path
        if re.match(r"(check|feasibility|build)-", path.name):
            assert path.read_bytes() == data, path
    assert len(list(s.rdir.glob("session-*.log"))) == 2  # a log per attempt, none overwritten


def test_a_rerun_after_a_failed_build_runs_the_build_again_on_the_same_directory(s):
    s.ckpt.mkdir(parents=True)
    (s.ckpt / "sentinel.txt").write_text("partial\n")
    assert s.run(FAIL_STEP="build").returncode == 1
    assert s.run().returncode == 0
    assert s.names().count("pod_g1lite.sh build") == 2
    assert (s.ckpt / "sentinel.txt").read_text() == "partial\n"


def test_a_rerun_is_refused_when_the_session_already_reached_a_final_verdict(s):
    assert s.run().returncode == 0
    n = len(s.steps())
    again = s.run()
    assert again.returncode == 2 and "G1_SESSION_DONE" in again.stdout + again.stderr
    assert len(s.steps()) == n and s.done_lines()[0] == "G1_SESSION_DONE"


def test_a_rerun_is_refused_after_a_gate_fail_too(s):
    assert s.run(FID_KL="0.03").returncode == 0
    n = len(s.steps())
    again = s.run()
    assert again.returncode == 2 and len(s.steps()) == n
    assert s.done_lines()[0] == "G1_SESSION_GATE_FAILED"


def test_a_stale_done_of_a_failed_attempt_is_removed_when_a_rerun_starts(s):
    assert s.run(FAIL_STEP="build").returncode == 1
    assert s.done_lines()[0].startswith("G1_SESSION_FAILED")
    assert s.run().returncode == 0
    assert set(s.done_flags()) == {"n"}  # no step of any attempt ever saw a done file


def test_two_sessions_at_the_same_time_are_refused(s):
    other = subprocess.Popen(["bash", "-c", "sleep 30; true", "g1_session.sh"])
    try:
        (s.work / "run").mkdir(parents=True, exist_ok=True)
        (s.work / "run" / "g1_session.pid").write_text(str(other.pid))
        out = s.run()
        assert out.returncode == 2 and "another" in out.stdout + out.stderr
        assert s.steps() == [] and not s.done.exists()
    finally:
        other.kill()


def test_a_pid_file_of_a_process_that_is_not_a_session_is_stale(s):
    other = subprocess.Popen(["sleep", "30"])
    try:
        (s.work / "run").mkdir(parents=True, exist_ok=True)
        (s.work / "run" / "g1_session.pid").write_text(str(other.pid))
        assert s.run().returncode == 0
    finally:
        other.kill()


def test_a_rerun_restarts_the_server_the_same_way_for_the_accuracy_run(s):
    first = s.run(G1_POD_DEADLINE_EPOCH=s.deadline_in(120), BUMP_ON_GATE=str(110 * 60))
    assert first.returncode == 1 and "accuracy" in last_line(first)
    second = s.run()
    assert second.returncode == 0, second.stdout + second.stderr
    names = s.names()
    assert names.count("serve_olala.sh ckpt") == 2 and names.count("apply_patch") == 2
    assert (
        names.count("fidelity collect") == 3
    )  # the fidelity step was not repeated (2 reference + g1)
    assert names.count("run_accuracy.sh G1-v2") == 1 and names[-1] == "stop_server.sh"
    assert names.index("serve_olala.sh ckpt", names.index("serve_olala.sh ckpt") + 1) < names.index(
        "run_accuracy.sh G1-v2"
    )


def test_the_noise_floor_collection_can_be_switched_off(s):
    assert s.run(G1_NOISE="0").returncode == 0
    assert s.names().count("fidelity collect") == 2
    assert not (s.rdir / "reference" / "noise_floor.json").exists()


# --- .done is written last, with the marker as its first line, in every path ------------------------------------------

FAILURES = [
    {"FAIL_STEP": "check"},
    {"FAIL_STEP": "feasibility"},
    {"FEAS_FWD": "9.0"},
    {"FAIL_STEP": "build"},
    {"FAIL_STEP": "verify"},
    {"REF_SERVE_RC": "1"},
    {"SERVE_RC": "1"},
    {"SMOKE_RC": "1"},
    {"COMPARE_RC": "1"},
    {"ACC_RC": "1"},
    {"ACC_STATUS": "running"},
]


@pytest.mark.parametrize(
    "env",
    [*FAILURES, {}, {"FID_KL": "0.05"}, {"FID_KL": "0.05", "G1_ACCURACY_ALWAYS": "1"}],
    ids=str,
)
def test_done_is_written_last_and_no_step_or_copy_ever_sees_it(s, env):
    out = s.run(**env)
    assert out.returncode in (0, 1)
    assert s.done.exists()
    assert set(s.done_flags()) == {
        "n"
    }  # not a single stub call, nor a cp, ran after g1.done appeared
    first = s.done_lines()[0]
    assert first.split(":")[0] in FINAL
    assert last_line(out) == first
    # nothing in the results is newer than the .done
    done_mtime = s.done.stat().st_mtime_ns
    assert all(p.stat().st_mtime_ns <= done_mtime for p in all_files(s.rdir))


def test_done_is_written_atomically_with_no_temporary_left(s):
    assert s.run().returncode == 0
    renames = [ln for ln in (s.tmp / "mv.log").read_text().splitlines() if ln.endswith("g1.done")]
    assert renames == [f"mv {s.results}/g1.done.tmp {s.results}/g1.done"]  # complete, then visible
    assert not list(s.results.glob("g1.done.*"))


def test_a_failure_before_any_step_still_writes_done(s):
    shutil.rmtree(s.calib)
    out = s.run()
    assert out.returncode == 1 and last_line(out).startswith("G1_SESSION_FAILED: ")
    assert "calibration" in last_line(out) and s.done_lines()[0] == last_line(out)
    assert s.steps() == []


@pytest.mark.parametrize("victim", ["windows.json", "calib_text.jsonl", "manifest.json"])
def test_a_calibration_file_that_differs_from_the_manifest_is_refused_before_any_step(s, victim):
    if victim == "manifest.json":
        (s.calib / victim).write_text("{}")
    else:
        (s.calib / victim).write_bytes(b"tampered\n")
    out = s.run()
    assert out.returncode == 1 and "calibration" in last_line(out)
    assert s.steps() == []


def test_missing_bundle_parts_or_weights_fail_before_any_step(s):
    (s.gptq / "pod_g1lite.sh").unlink()
    out = s.run()
    assert out.returncode == 1 and "pod_g1lite.sh" in last_line(out) and s.steps() == []
    s2 = Session(s.tmp / "other")
    (s2.work / "models" / "olala-bf16" / "config.json").unlink()
    out = s2.run()
    assert out.returncode == 1 and "weights" in last_line(out) and s2.steps() == []


# --- the deadline guard ------------------------------------------------------------------------------------------------


def test_nothing_starts_when_less_than_20_minutes_of_the_pod_lifetime_remain(s):
    out = s.run(G1_POD_DEADLINE_EPOCH=s.deadline_in(19))
    assert out.returncode == 1 and last_line(out).startswith("G1_SESSION_FAILED: ")
    assert "deadline" in last_line(out) and "check" in last_line(out)
    assert not [x for x in s.names() if x.startswith("pod_g1lite")]
    assert s.done_lines()[0] == last_line(out)


def test_a_step_starts_when_21_minutes_remain(s):
    out = s.run(G1_POD_DEADLINE_EPOCH=s.deadline_in(21))
    assert out.returncode == 0, out.stdout + out.stderr


def test_the_guard_is_checked_before_every_step_not_only_the_first(s):
    # the build "takes" four hours: the clock jumps after it, 10 minutes of the pod's life remain for verify
    out = s.run(
        G1_POD_DEADLINE_EPOCH=s.deadline_in(60 * 4 + 10),
        BUMP_AFTER="build",
        BUMP_SECONDS=str(4 * 3600),
    )
    assert out.returncode == 1, out.stdout + out.stderr
    assert "deadline" in last_line(out) and "verify" in last_line(out)
    names = s.names()
    assert "pod_g1lite.sh build" in names and "pod_g1lite.sh verify" not in names
    assert s.done_lines()[0] == last_line(out)


def test_the_guard_also_stops_the_accuracy_run_from_starting(s):
    # 2 hours of life at the start; the clock jumps 110 minutes once the gate has been decided
    out = s.run(G1_POD_DEADLINE_EPOCH=s.deadline_in(120), BUMP_ON_GATE=str(110 * 60))
    assert out.returncode == 1, out.stdout + out.stderr
    assert "deadline" in last_line(out) and "accuracy" in last_line(out)
    assert "run_accuracy.sh" not in " ".join(s.names())
    assert s.names()[-1] == "stop_server.sh"  # the server of the fidelity step does not stay up
    assert s.gate()["pass"] is True  # and the verdict of the gate is kept


# --- the repo's own pieces ---------------------------------------------------------------------------------------------


def test_the_stub_of_pod_g1lite_matches_the_real_scripts_markers_and_environment():
    real = (REPO / "quant" / "gptq" / "pod_g1lite.sh").read_text()
    assert "G1LITE_$1_DONE" in real  # finish() prints G1LITE_<STEP>_DONE
    for var in ("G1_CALIB", "G1_BASE_SELECT", "G1_DST"):
        assert var in real
    assert "every layer that was run" in real and "layer  captured  forward_s" in real
