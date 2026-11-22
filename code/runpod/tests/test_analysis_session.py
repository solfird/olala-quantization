"""pod/analysis_session.sh: weight statistics, default prompts and activation probe on one pod, unattended.

No GPU, no network, no pod. The script runs against a fake /workspace (OLALA_WORK in a temp folder) whose bundle holds
stand-ins for olala_quant.weight_stats, analysis/activation_probe.py, analysis/build_prompts.py and the vendored model
loader; `uv`, `nvidia-smi` and `python3 -m pip` are shims that record their arguments in one calls log, so that the ORDER
and the ARGUMENTS of the steps are what the tests read.
"""

from __future__ import annotations

import hashlib
import os
import stat
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from conftest import RUNPOD

POD = RUNPOD / "pod"
SCRIPT = POD / "analysis_session.sh"
CONFIG = '{"model_type": "olala"}\n'
INDEX = '{"weight_map": {"a": "model-00001.safetensors"}}\n'
GIT = "0123456789abcdef0123456789abcdef01234567"
STEPS = ("weights", "prompts", "probe")


def sha(text: str | bytes) -> str:
    data = text.encode() if isinstance(text, str) else text
    return hashlib.sha256(data).hexdigest()


def shim(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


FAKE_WEIGHT_STATS = r"""import os, pathlib, sys, time
a = sys.argv[1:]
log = os.environ["FAKE_CALLS"]
open(log, "a").write("weight_stats " + " ".join(a) + "\n")
done = pathlib.Path(os.environ["OLALA_WORK"]) / "results" / "analysis.done"
open(log, "a").write("weight_stats sees .done: %s\n" % done.exists())
if os.environ.get("FAKE_WS_SLEEP"):
    time.sleep(float(os.environ["FAKE_WS_SLEEP"]))
print("weight_stats progress line")
if os.environ.get("FAKE_WS_RC", "0") != "0":
    print("weight_stats boom", file=sys.stderr); sys.exit(int(os.environ["FAKE_WS_RC"]))
if not os.environ.get("FAKE_WS_NOFILES"):
    out = pathlib.Path(a[a.index("--out") + 1]); out.mkdir(parents=True, exist_ok=True)
    (out / "weight_stats.json").write_text("{}")
    (out / "weight_stats.md").write_text("# stats\n")
"""

FAKE_PROBE = r"""import json, os, pathlib, sys
a = sys.argv[1:]
log = os.environ["FAKE_CALLS"]
open(log, "a").write("activation_probe " + " ".join(a) + " | OLALA_ATTN_IMPL=" + os.environ.get("OLALA_ATTN_IMPL", "") + "\n")
out = pathlib.Path(a[a.index("--out") + 1])
counter = out.parent / ".fake_probe_attempts"
n = int(counter.read_text()) + 1 if counter.exists() else 1
counter.parent.mkdir(parents=True, exist_ok=True); counter.write_text(str(n))
print("probe 1/80 progress")
if os.environ.get("FAKE_PROBE_RC", "0") != "0" or (os.environ.get("FAKE_PROBE_FAIL_FIRST") and n == 1):
    print("probe boom", file=sys.stderr); sys.exit(int(os.environ.get("FAKE_PROBE_RC", "1")))
if os.environ.get("FAKE_PROBE_NOFILES"):
    sys.exit(0)
impl = a[a.index("--attn-impl") + 1] if "--attn-impl" in a else os.environ.get("FAKE_ATTN_IMPL", "flex")
out.mkdir(parents=True, exist_ok=True)
meta = {"attn_impl": impl, "peak_gpu_gib": 21.5, "max_tokens_in_a_prompt": 3900, "n_prompts": 80,
        "n_tokens_total": 91234}
(out / "activation_probe.json").write_text(json.dumps({"meta": meta, "prompts": [{"id": "x", "truncated": True}, {"id": "y", "truncated": False}]}))
(out / "activation_probe.md").write_text("# probe\n")
"""

FAKE_BUILD_PROMPTS = r"""import os, pathlib, sys
a = sys.argv[1:]
log = os.environ["FAKE_CALLS"]
open(log, "a").write("build_prompts " + " ".join(a) + " | HF_HUB_OFFLINE=" + os.environ.get("HF_HUB_OFFLINE", "") + "\n")
print("build_prompts progress")
if os.environ.get("FAKE_BP_RC", "0") != "0":
    print("build_prompts boom", file=sys.stderr); sys.exit(int(os.environ["FAKE_BP_RC"]))
out = pathlib.Path(a[a.index("--out") + 1]); out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("".join('{"id": "p%d"}\n' % i for i in range(int(os.environ.get("FAKE_PROMPT_LINES", "80")))))
"""

FAKE_OLALA_MODEL = r"""import os, types
def load_olala_modules():
    if os.environ.get("FAKE_DETECT_FAIL"):
        raise ImportError("no flash, no flex")
    open(os.environ["FAKE_CALLS"], "a").write(
        "detect OLALA_TORCH_MOE=%s OLALA_TORCH_MAMBA3=%s\n"
        % (os.environ.get("OLALA_TORCH_MOE"), os.environ.get("OLALA_TORCH_MAMBA3")))
    return types.SimpleNamespace(ATTN_IMPL=os.environ.get("FAKE_ATTN_IMPL", "flex")), None
"""


class Session:
    """A pod's /workspace in a temp folder with a fake bundle and recording shims."""

    def __init__(self, tmp: Path) -> None:
        self.work = tmp / "ws"
        self.bundle = self.work / "bundle"
        self.calls = tmp / "calls.log"
        self.calls.write_text("")
        self.results = self.work / "results"
        self.res = self.results / "analysis"
        self.logs = self.work / "logs" / "analysis"
        self.model = self.work / "models" / "olala-bf16"
        for rel, text in {
            "quant/olala_quant/__init__.py": "",
            "quant/olala_quant/weight_stats.py": FAKE_WEIGHT_STATS,
            "quant/gptq/olala_model.py": FAKE_OLALA_MODEL,
            "analysis/activation_probe.py": FAKE_PROBE,
            "analysis/build_prompts.py": FAKE_BUILD_PROMPTS,
            "analysis/pyproject.toml": "[project]\n",
            "evals/pyproject.toml": "[project]\n",
        }.items():
            path = self.bundle / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        (self.bundle / "BUNDLE_MANIFEST.txt").write_text(
            f"# olala-quants bundle, built 2026-11-20T10:00:00Z; git: {GIT}\n# analysis: 4 files\n"
        )
        self.model.mkdir(parents=True)
        (self.model / "config.json").write_text(CONFIG)
        (self.model / "model.safetensors.index.json").write_text(INDEX)
        self.evals_log = self.work / "logs" / "setup" / "evals-env-1.log"
        self.evals_log.parent.mkdir(parents=True)
        self.evals_log.write_text("== done\nEVALS_ENV_DONE\n")
        self.bin = tmp / "bin"
        self.bin.mkdir()
        c = str(self.calls)
        shim(
            self.bin / "uv",
            f"""echo "uv cwd=$PWD UV_PROJECT_ENVIRONMENT=${{UV_PROJECT_ENVIRONMENT:-}} HF_HOME=${{HF_HOME:-}} $*" >> "{c}"
[ "$1" = run ] || exit 0
shift
while [ $# -gt 0 ] && [ "$1" != python ]; do shift; done
shift
exec "{sys.executable}" "$@"
""",
        )
        shim(
            self.bin / "nvidia-smi",
            f'echo "nvidia-smi $*" >> "{c}"\necho "NVIDIA RTX PRO 4500 Blackwell, 580.95, 32623 MiB"\n',
        )
        # the pod's python3: a shim that records pip and answers `import einops`; everything else is the real python
        self.python = self.bin / "podpython"
        shim(
            self.python,
            f"""if [ "$1" = -m ] && [ "$2" = pip ]; then echo "pip $*" >> "{c}"; exit "${{FAKE_PIP_RC:-0}}"; fi
if [ "$1" = -c ] && [ "$2" = "import einops" ]; then exit "${{FAKE_EINOPS_RC:-0}}"; fi
exec "{sys.executable}" "$@"
""",
        )
        self.env = {
            **os.environ,
            "OLALA_WORK": str(self.work),
            "FAKE_CALLS": str(self.calls),
            "PYTHON": str(self.python),
            "ANALYSIS_EVALS_POLL": "0",
            "ANALYSIS_EVALS_TRIES": "3",
            "PATH": f"{self.bin}:{os.environ['PATH']}",
        }
        for var in (
            "HF_TOKEN",
            "HF_HUB_OFFLINE",
            "HF_DATASETS_OFFLINE",
            "OLALA_ATTN_IMPL",
            "OLALA_TORCH_MOE",
            "OLALA_TORCH_MAMBA3",
        ):
            self.env.pop(var, None)

    def run(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            capture_output=True,
            text=True,
            env={**self.env, **env},
            check=False,
            timeout=60,
        )

    def seed_prompts(self) -> None:
        """The prompt file a finished prompts step leaves (a probe rerun alone needs it)."""
        path = self.res / "prompts" / "prompts_default.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f'{{"id": "p{i}"}}\n' for i in range(80)))

    def lines(self) -> list[str]:
        return [ln for ln in self.calls.read_text().splitlines() if ln]

    def calls_of(self, prefix: str) -> list[str]:
        return [ln for ln in self.lines() if ln.startswith(prefix + " ")]

    def tree(self) -> list[str]:
        return (
            sorted(str(p.relative_to(self.work)) for p in self.work.rglob("*"))
            if self.work.exists()
            else []
        )


@pytest.fixture
def pod(tmp_path):
    return Session(tmp_path)


def markers(out: subprocess.CompletedProcess[str]) -> list[str]:
    return [ln for ln in out.stdout.splitlines() if ln.startswith("ANALYSIS_")]


def untouched(pod: Session) -> list[str]:
    """The tree of a fresh pod, before any run (the bundle, the weights and the evals log)."""
    return sorted(str(p.relative_to(pod.work)) for p in pod.work.rglob("*"))


# ------------------------------------------------------------------------------------------------ the script exists
def test_the_script_exists_and_parses():
    assert SCRIPT.exists()
    assert subprocess.run(["bash", "-n", str(SCRIPT)], check=False).returncode == 0


# ------------------------------------------------------------------------------------------------ refusals and dry run
@pytest.mark.parametrize(
    "args",
    [
        (),
        ("magic",),
        ("weights", "magic"),
        ("weights", "weights"),
        ("all", "probe"),
        ("--bogus", "all"),
        ("--dry-run",),
        ("WEIGHTS",),
    ],
)
def test_bad_arguments_are_refused_with_a_failed_marker_and_nothing_is_written(pod, args):
    before = untouched(pod)
    out = pod.run(*args)
    assert out.returncode == 2, out.stdout + out.stderr
    assert out.stdout.strip().splitlines()[-1].startswith("ANALYSIS_FAILED: "), out.stdout
    assert "usage" in out.stderr
    assert pod.tree() == before
    assert pod.lines() == []


def test_a_dry_run_prints_the_plan_and_runs_and_writes_nothing(pod):
    before = untouched(pod)
    out = pod.run("--dry-run", "all")
    assert out.returncode == 0, out.stdout + out.stderr
    text = out.stdout
    assert "olala_quant.weight_stats" in text and "--device cuda" in text
    assert "build_prompts.py" in text and "--online" in text
    assert "activation_probe.py" in text and "--max-tokens" in text
    assert str(pod.res / "weight_stats") in text and str(pod.res / "probe") in text
    assert str(pod.results / "analysis.done") in text
    assert not [ln for ln in text.splitlines() if ln.startswith(("ANALYSIS_STEP", "ANALYSIS_DONE"))]
    assert pod.tree() == before
    assert pod.lines() == []


def test_a_dry_run_of_one_step_plans_only_that_step(pod):
    out = pod.run("--dry-run", "probe")
    assert out.returncode == 0
    assert "activation_probe.py" in out.stdout
    assert "olala_quant.weight_stats" not in out.stdout and "build_prompts.py" not in out.stdout


def test_a_dry_run_needs_no_weights_and_no_bundle(tmp_path):
    env = {**os.environ, "OLALA_WORK": str(tmp_path / "empty")}
    out = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run", "all"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert not (tmp_path / "empty").exists()


# ------------------------------------------------------------------------------------------------ the full run
def test_a_full_run_runs_the_three_steps_in_order_with_their_arguments_and_ends_done(pod):
    out = pod.run("all")
    assert out.returncode == 0, out.stdout + out.stderr
    assert markers(out) == [
        "ANALYSIS_STEP_DONE:weights",
        "ANALYSIS_STEP_DONE:prompts",
        "ANALYSIS_STEP_DONE:probe",
        "ANALYSIS_DONE",
    ]
    assert out.stdout.strip().splitlines()[-1] == "ANALYSIS_DONE"
    first = {
        k: next(i for i, ln in enumerate(pod.lines()) if ln.startswith(k + " "))
        for k in ("weight_stats", "uv", "activation_probe")
    }
    assert first["weight_stats"] < first["uv"] < first["activation_probe"]
    (ws,) = [ln for ln in pod.calls_of("weight_stats") if "--src" in ln]
    assert ws == f"weight_stats --src {pod.model} --out {pod.res / 'weight_stats'} --device cuda"
    (bp,) = pod.calls_of("build_prompts")
    assert bp == (
        f"build_prompts --out {pod.res / 'prompts' / 'prompts_default.jsonl'} --evals-dir {pod.bundle / 'evals'} --online | HF_HUB_OFFLINE="
    )
    (pr,) = pod.calls_of("activation_probe")
    assert pr == (
        f"activation_probe --model-dir {pod.model} --prompts {pod.res / 'prompts' / 'prompts_default.jsonl'} "
        f"--out {pod.res / 'probe'} --device cuda --max-tokens 4096 | OLALA_ATTN_IMPL="
    )


def test_every_step_has_its_own_log_and_its_result_folder(pod):
    assert pod.run("all").returncode == 0
    for step in STEPS:
        assert (pod.logs / f"{step}.log").is_file(), step
    assert "weight_stats progress line" in (pod.logs / "weights.log").read_text()
    assert "build_prompts progress" in (pod.logs / "prompts.log").read_text()
    assert "probe 1/80 progress" in (pod.logs / "probe.log").read_text()
    assert (pod.res / "weight_stats" / "weight_stats.json").is_file()
    assert (pod.res / "weight_stats" / "weight_stats.md").is_file()
    assert (pod.res / "prompts" / "prompts_default.jsonl").is_file()
    assert (pod.res / "probe" / "activation_probe.json").is_file()
    assert (pod.res / "probe" / "activation_probe.md").is_file()
    for folder in ("weight_stats", "prompts", "probe"):  # collect.sh pulls only results/analysis
        assert (pod.res / folder / "provenance.txt").is_file(), folder
        assert (pod.res / folder / "step.log").is_file(), folder
        assert (pod.res / folder / "status.txt").read_text().startswith("done")


def test_the_prompts_step_uses_the_evals_environment_and_the_hub(pod):
    assert pod.run("prompts").returncode == 0
    (uv,) = pod.calls_of("uv")
    assert f"cwd={pod.bundle / 'evals'}" in uv
    assert f"UV_PROJECT_ENVIRONMENT={pod.work / 'venv-evals'}" in uv
    assert f"HF_HOME={pod.work / 'hf_cache'}" in uv
    assert " run --frozen --python 3.12 python " in uv


def test_one_step_alone_runs_only_that_step_and_still_ends_done(pod):
    out = pod.run("weights")
    assert out.returncode == 0, out.stdout + out.stderr
    assert markers(out) == ["ANALYSIS_STEP_DONE:weights", "ANALYSIS_DONE"]
    assert pod.calls_of("build_prompts") == [] and pod.calls_of("activation_probe") == []
    assert (pod.results / "analysis.done").is_file()


def test_the_steps_run_in_the_canonical_order_whatever_the_order_given(pod):
    out = pod.run("probe", "prompts", "weights")
    assert out.returncode == 0, out.stdout + out.stderr
    assert markers(out)[:3] == [f"ANALYSIS_STEP_DONE:{s}" for s in STEPS]


# ------------------------------------------------------------------------------------------------ independence
FAILS = {
    "weights": {"FAKE_WS_RC": "3"},
    "prompts": {"FAKE_BP_RC": "4"},
    "probe": {"FAKE_PROBE_RC": "5", "FAKE_ATTN_IMPL": "eager"},
}


@pytest.mark.parametrize("failing", STEPS)
def test_a_failed_step_does_not_stop_the_others_and_the_final_marker_says_failed(pod, failing):
    out = pod.run("all", **FAILS[failing])
    assert out.returncode == 1, out.stdout + out.stderr
    got = markers(out)
    assert f"ANALYSIS_STEP_FAILED:{failing}" in got
    for step in STEPS:
        assert any(
            m.startswith(f"ANALYSIS_STEP_{'FAILED' if step == failing else 'DONE'}:{step}")
            for m in got
        ) or (failing == "prompts" and step == "probe"), (step, got)
    # all three steps were attempted: weights and probe ran even though another step failed
    assert pod.calls_of("weight_stats") and pod.calls_of("build_prompts")
    last = out.stdout.strip().splitlines()[-1]
    assert last.startswith("ANALYSIS_FAILED: ") and failing in last
    assert "ANALYSIS_DONE" not in got


def test_a_failed_prompts_step_leaves_the_probe_to_fail_on_its_own_and_weights_still_done(pod):
    out = pod.run("all", FAKE_BP_RC="4")
    got = markers(out)
    assert got[:3] == [
        "ANALYSIS_STEP_DONE:weights",
        "ANALYSIS_STEP_FAILED:prompts",
        "ANALYSIS_STEP_FAILED:probe",
    ]
    assert pod.calls_of("activation_probe") == []  # no prompt file: the probe is not run on nothing
    assert "prompt file" in (pod.logs / "probe.log").read_text()


def test_a_step_that_exits_zero_without_its_outputs_has_failed(pod):
    out = pod.run("all", FAKE_WS_NOFILES="1")
    assert "ANALYSIS_STEP_FAILED:weights" in markers(out)
    assert "ANALYSIS_STEP_DONE:probe" in markers(out)


def test_a_probe_that_exits_zero_without_its_outputs_has_failed(pod):
    pod.seed_prompts()
    out = pod.run("probe", FAKE_PROBE_NOFILES="1", FAKE_ATTN_IMPL="eager")
    assert "ANALYSIS_STEP_FAILED:probe" in markers(out)


def test_an_output_left_by_an_earlier_failed_attempt_does_not_make_a_step_done(pod):
    stale = pod.res / "weight_stats"
    stale.mkdir(parents=True)
    (stale / "weight_stats.json").write_text("{}")
    (stale / "weight_stats.md").write_text("# stale\n")
    (stale / "status.txt").write_text("failed: exit 3\n")
    out = pod.run("weights", FAKE_WS_NOFILES="1")
    assert "ANALYSIS_STEP_FAILED:weights" in markers(out)


def test_a_short_prompt_file_fails_the_prompts_step(pod):
    out = pod.run("prompts", FAKE_PROMPT_LINES="79")
    assert "ANALYSIS_STEP_FAILED:prompts" in markers(out)
    assert out.returncode == 1


def test_two_failed_steps_are_both_named_in_the_final_marker(pod):
    out = pod.run("all", FAKE_WS_RC="3", FAKE_PROBE_RC="5", FAKE_ATTN_IMPL="eager")
    assert out.returncode == 1
    last = out.stdout.strip().splitlines()[-1]
    assert last == "ANALYSIS_FAILED: step(s) failed: weights, probe"
    assert "ANALYSIS_STEP_DONE:prompts" in markers(out)


def test_the_failed_step_records_its_status_and_its_reason(pod):
    pod.run("all", FAKE_WS_RC="3")
    status = (pod.res / "weight_stats" / "status.txt").read_text()
    assert status.startswith("failed") and "exit 3" in status
    assert "weight_stats boom" in (pod.logs / "weights.log").read_text()
    assert (pod.res / "weight_stats" / "step.log").is_file()


# ------------------------------------------------------------------------------------------------ the .done file
def test_the_done_file_is_written_last_and_says_how_it_ended(pod):
    out = pod.run("all")
    assert out.returncode == 0
    done = pod.results / "analysis.done"
    assert done.read_text().startswith("ANALYSIS_DONE")
    newest_other = max(p.stat().st_mtime_ns for p in pod.res.rglob("*") if p.is_file())
    assert done.stat().st_mtime_ns >= newest_other
    assert done.stat().st_mtime_ns >= (pod.logs / "session.log").stat().st_mtime_ns


def test_the_done_file_is_written_after_a_failure_too_so_that_partial_results_are_pulled(pod):
    out = pod.run("all", FAKE_WS_RC="3")
    assert out.returncode == 1
    text = (pod.results / "analysis.done").read_text()
    assert text.startswith("ANALYSIS_FAILED: step(s) failed: weights")
    assert (
        pod.res / "probe" / "activation_probe.json"
    ).is_file()  # the other results are in place before it


def test_no_done_file_exists_while_the_steps_run_and_a_stale_one_is_removed(pod):
    pod.results.mkdir(parents=True)
    (pod.results / "analysis.done").write_text("ANALYSIS_DONE (an earlier attempt)\n")
    assert pod.run("weights").returncode == 0
    assert "weight_stats sees .done: False" in pod.lines()
    assert "earlier attempt" not in (pod.results / "analysis.done").read_text()


def test_no_done_file_after_a_refusal_or_a_dry_run(pod):
    pod.run("magic")
    pod.run("--dry-run", "all")
    assert not (pod.results / "analysis.done").exists()


# ------------------------------------------------------------------------------------------------ rerun rules
def test_a_step_that_is_done_is_never_overwritten(pod):
    assert pod.run("weights").returncode == 0
    n = len(pod.calls_of("weight_stats"))
    out = pod.run("all")
    assert out.returncode == 2 and out.stdout.strip().splitlines()[-1].startswith(
        "ANALYSIS_FAILED: "
    )
    assert "weights" in out.stdout
    assert len(pod.calls_of("weight_stats")) == n and pod.calls_of("build_prompts") == []


def test_a_failed_step_can_be_rerun_alone(pod):
    pod.run("all", FAKE_WS_RC="3")
    out = pod.run("weights")
    assert out.returncode == 0, out.stdout + out.stderr
    assert (pod.res / "weight_stats" / "status.txt").read_text().startswith("done at ")
    assert (pod.results / "analysis.done").read_text().startswith("ANALYSIS_DONE")


# ------------------------------------------------------------------------------------------------ evals environment
def test_the_prompts_step_waits_for_the_evals_environment_like_arm_pipeline(pod):
    pod.evals_log.write_text("== sync running\n")
    threading.Timer(0.6, lambda: pod.evals_log.write_text("== sync\nEVALS_ENV_DONE\n")).start()
    out = pod.run("prompts", ANALYSIS_EVALS_POLL="0.1", ANALYSIS_EVALS_TRIES="50")
    assert out.returncode == 0, out.stdout + out.stderr
    assert pod.calls_of("build_prompts")


@pytest.mark.parametrize(
    ("log_text", "reason"),
    [
        ("== sync\nEVALS_ENV_FAILED: uv sync failed\n", "failed"),
        ("== sync still running\n", "not ready"),
    ],
)
def test_the_prompts_step_fails_without_calling_the_builder_when_the_evals_environment_is_not_usable(
    pod, log_text, reason
):
    pod.evals_log.write_text(log_text)
    out = pod.run("all", ANALYSIS_EVALS_TRIES="2")
    got = markers(out)
    assert "ANALYSIS_STEP_FAILED:prompts" in got
    assert "ANALYSIS_STEP_DONE:weights" in got  # the other steps do not need that environment
    assert pod.calls_of("build_prompts") == [] and pod.calls_of("uv") == []
    assert reason in (pod.logs / "prompts.log").read_text()


# ------------------------------------------------------------------------------------------------ probe attention choice
@pytest.mark.parametrize(
    ("impl", "tokens"),
    [("fa3", "4096"), ("fa2", "4096"), ("flex", "4096"), ("eager", "2048")],
)
def test_the_probe_uses_the_flash_or_flex_backend_when_the_vendored_model_has_one_else_2048_tokens(
    pod, impl, tokens
):
    pod.seed_prompts()
    out = pod.run("probe", FAKE_ATTN_IMPL=impl)
    assert out.returncode == 0, out.stdout + out.stderr
    (pr,) = pod.calls_of("activation_probe")
    assert f"--max-tokens {tokens}" in pr
    assert (
        "--attn-impl" not in pr
    )  # the model picks fa3 > fa2 > flex; only eager needs no choice either
    prov = (pod.res / "probe" / "provenance.txt").read_text()
    assert f"attention_detected={impl}" in prov and f"max_tokens={tokens}" in prov


def test_the_detection_runs_with_the_pure_pytorch_paths_forced_like_the_probe_itself(pod):
    pod.seed_prompts()
    assert pod.run("probe").returncode == 0
    assert "detect OLALA_TORCH_MOE=1 OLALA_TORCH_MAMBA3=1" in pod.lines()


def test_a_detection_that_fails_is_treated_as_eager(pod):
    pod.seed_prompts()
    out = pod.run("probe", FAKE_DETECT_FAIL="1")
    assert out.returncode == 0, out.stdout + out.stderr
    (pr,) = pod.calls_of("activation_probe")
    assert "--max-tokens 2048" in pr
    assert "attention_detected=unknown" in (pod.res / "probe" / "provenance.txt").read_text()


def test_the_provenance_says_why_the_choice_was_made_and_what_the_run_really_used(pod):
    pod.seed_prompts()
    assert pod.run("probe", FAKE_ATTN_IMPL="flex").returncode == 0
    prov = (pod.res / "probe" / "provenance.txt").read_text()
    assert "estimate_memory_gib" in prov
    assert "attempts=1 final_max_tokens=4096 final_attn_impl_arg=auto" in prov
    assert (
        "result: attn_impl=flex peak_gpu_gib=21.5 max_tokens_in_a_prompt=3900 n_prompts=80 n_truncated=1"
        in prov
    )


def test_a_failing_flash_or_flex_run_is_retried_once_with_eager_and_2048_tokens(pod):
    pod.seed_prompts()
    out = pod.run("probe", FAKE_ATTN_IMPL="flex", FAKE_PROBE_FAIL_FIRST="1")
    assert out.returncode == 0, out.stdout + out.stderr
    first, second = pod.calls_of("activation_probe")
    assert "--max-tokens 4096" in first and "--attn-impl" not in first
    assert "--max-tokens 2048" in second and "--attn-impl eager" in second
    prov = (pod.res / "probe" / "provenance.txt").read_text()
    assert "attempt 1" in prov and "attempt 2" in prov and "result: attn_impl=eager" in prov
    assert "attempts=2 final_max_tokens=2048 final_attn_impl_arg=eager" in prov
    assert "attempts=2 final_max_tokens=2048" in (pod.res / "summary.txt").read_text()
    assert "ANALYSIS_STEP_DONE:probe" in markers(out)


def test_an_eager_run_that_fails_is_not_retried(pod):
    pod.seed_prompts()
    out = pod.run("probe", FAKE_ATTN_IMPL="eager", FAKE_PROBE_RC="5")
    assert "ANALYSIS_STEP_FAILED:probe" in markers(out)
    assert len(pod.calls_of("activation_probe")) == 1


def test_a_probe_that_fails_twice_is_failed(pod):
    pod.seed_prompts()
    out = pod.run("probe", FAKE_ATTN_IMPL="flex", FAKE_PROBE_RC="5")
    assert "ANALYSIS_STEP_FAILED:probe" in markers(out)
    assert len(pod.calls_of("activation_probe")) == 2


# ------------------------------------------------------------------------------------------------ environment installs
def test_einops_is_not_installed_when_it_is_there(pod):
    pod.seed_prompts()
    assert pod.run("probe").returncode == 0
    assert pod.calls_of("pip") == []


def test_a_missing_einops_is_installed_with_the_pinned_no_deps_command(tmp_path):
    p = Session(tmp_path)
    p.seed_prompts()
    assert p.run("probe", FAKE_EINOPS_RC="1").returncode == 0
    assert p.calls_of("pip") == ["pip -m pip install --no-deps einops==0.8.1"]


def test_a_failed_einops_install_fails_the_probe_only(tmp_path):
    p = Session(tmp_path)
    out = p.run("all", FAKE_EINOPS_RC="1", FAKE_PIP_RC="1")
    got = markers(out)
    assert "ANALYSIS_STEP_FAILED:probe" in got and "ANALYSIS_STEP_DONE:weights" in got
    assert p.calls_of("activation_probe") == []


def test_the_script_installs_nothing_unpinned_and_fetches_nothing_by_itself():
    text = SCRIPT.read_text()
    for ln in text.splitlines():
        code = ln.split("#", 1)[0]
        if "pip install" in code:
            assert "==" in code and "--no-deps" in code, ln
        if "uv run" in code or "uv sync" in code:
            assert "--frozen" in code or "$UV_RUN_ARGS" in code, ln
    assert "curl " not in text and "wget " not in text and "git clone" not in text
    assert "HF_TOKEN" not in text.replace("unset HF_TOKEN", "")


# ------------------------------------------------------------------------------------------------ provenance
def test_the_provenance_of_every_step_records_the_machine_the_weights_the_bundle_and_the_command(
    pod,
):
    assert pod.run("all").returncode == 0
    for folder, needle in (
        ("weight_stats", "olala_quant.weight_stats --src"),
        ("prompts", "build_prompts.py --out"),
        ("probe", "activation_probe.py --model-dir"),
    ):
        prov = (pod.res / folder / "provenance.txt").read_text()
        assert "NVIDIA RTX PRO 4500 Blackwell, 580.95" in prov, folder
        assert "python: Python 3" in prov, folder
        assert "torch:" in prov, folder
        assert f"config.json sha256={sha(CONFIG)}" in prov, folder
        assert f"model.safetensors.index.json sha256={sha(INDEX)}" in prov, folder
        assert f"bundle_git={GIT}" in prov, folder
        assert "bundle_manifest_sha256=" in prov, folder
        assert "command: " in prov and needle in prov, folder


def test_a_single_file_checkpoint_is_identified_by_the_hash_of_its_safetensors_header(pod):
    (pod.model / "model.safetensors.index.json").unlink()
    header = b'{"w": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}'
    (pod.model / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + b"\x00" * 4
    )
    assert pod.run("weights").returncode == 0
    prov = (pod.res / "weight_stats" / "provenance.txt").read_text()
    assert f"model.safetensors header sha256={sha(header)}" in prov
    assert "model.safetensors.index.json" not in prov


def test_an_unknown_bundle_commit_is_said_so(pod):
    (pod.bundle / "BUNDLE_MANIFEST.txt").write_text(
        "# olala-quants bundle, built X; git: no-commit-yet\n"
    )
    assert pod.run("weights").returncode == 0
    assert "bundle_git=unknown" in (pod.res / "weight_stats" / "provenance.txt").read_text()


def test_missing_weights_fail_the_weights_and_probe_steps_and_not_the_prompts_step(pod):
    (pod.model / "config.json").unlink()
    (pod.model / "model.safetensors.index.json").unlink()
    out = pod.run("all")
    got = markers(out)
    assert got[:3] == [
        "ANALYSIS_STEP_FAILED:weights",
        "ANALYSIS_STEP_DONE:prompts",
        "ANALYSIS_STEP_FAILED:probe",
    ]
    assert pod.calls_of("weight_stats") == [] and pod.calls_of("activation_probe") == []
    assert "no config.json" in (pod.logs / "weights.log").read_text()


def test_a_bundle_without_analysis_fails_the_prompts_and_probe_steps_with_the_way_out(pod):
    (pod.bundle / "analysis" / "activation_probe.py").unlink()
    (pod.bundle / "analysis" / "build_prompts.py").unlink()
    out = pod.run("all")
    got = markers(out)
    assert got[:3] == [
        "ANALYSIS_STEP_DONE:weights",
        "ANALYSIS_STEP_FAILED:prompts",
        "ANALYSIS_STEP_FAILED:probe",
    ]
    assert "pod_setup.sh" in (pod.logs / "probe.log").read_text()


# ------------------------------------------------------------------------------------------------ the pod's lifetime
def test_a_step_that_is_still_running_at_the_deadline_is_stopped_and_the_rest_is_not_started(pod):
    start = time.monotonic()
    out = pod.run("all", FAKE_WS_SLEEP="30", ANALYSIS_DEADLINE_SECONDS="2")
    assert time.monotonic() - start < 25
    assert out.returncode == 1, out.stdout + out.stderr
    got = markers(out)
    assert got[:3] == [
        "ANALYSIS_STEP_FAILED:weights",
        "ANALYSIS_STEP_FAILED:prompts",
        "ANALYSIS_STEP_FAILED:probe",
    ]
    assert "deadline" in (pod.logs / "weights.log").read_text()
    assert pod.calls_of("build_prompts") == [] and pod.calls_of("activation_probe") == []
    assert (pod.results / "analysis.done").read_text().startswith("ANALYSIS_FAILED")
