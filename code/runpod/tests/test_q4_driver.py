"""pod/q4_driver.sh: what a Q4 pod runs, in order. No GPU, no network, no pod.

q4_prepare.sh, arm_pipeline.sh and bench_gguf.sh are stubs in a copy of runpod/pod that record their arguments; the
quantized file is a small file whose sha256 the tests know.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
from conftest import RUNPOD, stub_calls, stub_env

POD_SRC = RUNPOD / "pod"
SCRIPT = "q4_driver.sh"
VARIANT = "q4_k_m-ovhrecipe"
Q4_NAME = "Olala-v1.0-mini-7A1B-q4_k_m-ovhrecipe.gguf"
Q4_BYTES = b"the quantized file, identical on every pod"
Q4_SHA = hashlib.sha256(Q4_BYTES).hexdigest()
IMATRIX_SHA = "9" * 64
TASKS = "mgsm aime2025"

STUBS = {
    "q4_prepare.sh": r"""#!/bin/bash
echo "q4_prepare.sh $*" >> "$STUB_LOG"
mkdir -p "$OLALA_WORK/results/q4" "$OLALA_WORK/models/olala-gguf"
if [ -n "${STUB_PREPARE_FAIL:-}" ]; then
  echo "Q4_PREPARE_FAILED:$STUB_PREPARE_FAIL" > "$OLALA_WORK/results/q4.done"; echo "Q4_PREPARE_FAILED:$STUB_PREPARE_FAIL"
  exit "${STUB_PREPARE_RC:-1}"
fi
printf '%s' "$STUB_Q4_CONTENT" > "$OLALA_WORK/models/olala-gguf/Olala-v1.0-mini-7A1B-q4_k_m-ovhrecipe.gguf"
echo "Q4_PREPARE_DONE" > "$OLALA_WORK/results/q4.done"
echo "Q4_PREPARE_DONE"
""",
    "arm_pipeline.sh": r"""#!/bin/bash
echo "arm_pipeline.sh $*" >> "$STUB_LOG"
for a in "$@"; do echo "arg=[$a]" >> "$STUB_LOG"; done
[ "${STUB_PIPELINE_RC:-0}" = 0 ] || { echo ARM_PIPELINE_FAILED; exit "$STUB_PIPELINE_RC"; }
echo ARM_PIPELINE_DONE
""",
    "bench_gguf.sh": r"""#!/bin/bash
echo "bench_gguf.sh $*" >> "$STUB_LOG"
for a in "$@"; do echo "arg=[$a]" >> "$STUB_LOG"; done
exit "${STUB_BENCH_RC:-0}"
""",
}


def write_exec(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


class Driver:
    def __init__(self, tmp: Path) -> None:
        self.work = tmp / "ws"
        self.pod = tmp / "pod"
        self.log = tmp / "calls.log"
        self.log.write_text("")
        shutil.copytree(POD_SRC, self.pod, ignore=shutil.ignore_patterns("__pycache__"))
        for name, body in STUBS.items():
            write_exec(self.pod / name, body)
        self.env = {
            "OLALA_WORK": str(self.work),
            "STUB_LOG": str(self.log),
            "STUB_Q4_CONTENT": Q4_BYTES.decode(),
        }

    @property
    def done(self) -> Path:
        return self.work / "results" / "q4-driver.done"

    @property
    def sha_check(self) -> Path:
        return self.work / "results" / "q4" / "sha_check.json"

    def run(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(self.pod / SCRIPT), *args],
            capture_output=True,
            text=True,
            env={**os.environ, **self.env, **env},
            check=False,
            timeout=120,
        )

    def calls(self) -> list[str]:
        return [ln for ln in self.log.read_text().splitlines() if ln and not ln.startswith("arg=")]

    def args_of(self, script: str) -> list[str]:
        lines = self.log.read_text().splitlines()
        start = next(i for i, ln in enumerate(lines) if ln.startswith(script + " "))
        out = []
        for ln in lines[start + 1 :]:
            if not ln.startswith("arg=["):
                break
            out.append(ln[5:-1])
        return out

    def names(self) -> list[str]:
        return [c.split()[0] for c in self.calls()]


@pytest.fixture
def drv(tmp_path):
    return Driver(tmp_path)


MAKE = ["--imatrix", "make"]
FILE = ["--imatrix", "/workspace/q4/olala.imatrix", "--imatrix-sha", IMATRIX_SHA]


def args(
    *extra: str, run: str = "Q4a", tasks: str = TASKS, imatrix: list[str] | None = None
) -> list[str]:
    return [run, tasks, *(imatrix if imatrix is not None else MAKE), *extra]


def test_the_script_parses():
    assert subprocess.run(["bash", "-n", str(POD_SRC / SCRIPT)], check=False).returncode == 0


@pytest.mark.parametrize(
    "bad",
    [
        [],
        ["Q4a"],
        ["Q4a", TASKS],
        ["a/b", TASKS, *MAKE],
        ["Q4a", "", *MAKE],
        ["Q4a", TASKS, "--imatrix", "make", "--imatrix-sha", "a" * 64],
        ["Q4a", TASKS, *MAKE, "--expect-sha", "xyz"],
        ["Q4a", TASKS, *MAKE, "--bogus"],
        ["Q4a", TASKS, "--imatrix", "/x"],
    ],
)
def test_bad_arguments_are_refused_before_anything_is_run(drv, bad):
    out = drv.run(*bad)
    assert out.returncode == 2 and "Q4_DRIVER_FAILED:BAD_ARGUMENTS" in out.stdout
    assert drv.calls() == [] and not drv.work.exists()


def test_the_dry_run_prints_the_order_and_runs_nothing(drv, tmp_path, stub_bin):
    out = drv.run(
        "--dry-run",
        *args("--expect-sha", Q4_SHA, "--bench"),
        **stub_env(stub_bin, tmp_path / "home"),
    )
    assert out.returncode == 0, out.stderr
    assert drv.calls() == [] and stub_calls(stub_bin) == "" and not drv.work.exists()
    text = out.stdout
    order = ["q4_prepare.sh", "compare", "arm_pipeline.sh", "bench_gguf.sh"]
    positions = [text.index(w) for w in order]
    assert positions == sorted(positions)
    assert f"bench_gguf.sh Q4a {VARIANT},q5_k_m" in text
    assert text.strip().splitlines()[-1] == "Q4_DRIVER_DRY_RUN (nothing was run)"


def test_the_steps_run_in_order_prepare_then_the_pipeline_then_the_bench(drv):
    out = drv.run(*args("--bench"))
    assert out.returncode == 0, out.stdout + out.stderr
    assert drv.names() == ["q4_prepare.sh", "arm_pipeline.sh", "bench_gguf.sh"]
    assert out.stdout.strip().splitlines()[-1] == "Q4_DRIVER_DONE"


def test_no_bench_unless_asked(drv):
    assert drv.run(*args()).returncode == 0
    assert drv.names() == ["q4_prepare.sh", "arm_pipeline.sh"]


def test_the_imatrix_arguments_go_to_prepare_unchanged(drv):
    assert drv.run(*args(imatrix=MAKE)).returncode == 0
    assert drv.calls()[0] == "q4_prepare.sh --imatrix make"
    drv.log.write_text("")
    assert drv.run(*args(imatrix=FILE)).returncode == 0
    assert (
        drv.calls()[0]
        == f"q4_prepare.sh --imatrix /workspace/q4/olala.imatrix --imatrix-sha {IMATRIX_SHA}"
    )


def test_the_pipeline_gets_the_run_the_q4_variant_and_the_tasks_as_one_argument(drv):
    assert drv.run(*args()).returncode == 0
    assert drv.args_of("arm_pipeline.sh") == ["Q4a", f"gguf:{VARIANT}", "--tasks", TASKS]


def test_the_bench_label_is_the_run_name_with_both_variants(drv):
    assert drv.run(*args("--bench", run="Q4b")).returncode == 0
    assert drv.args_of("bench_gguf.sh") == ["Q4b", f"{VARIANT},q5_k_m"]


# --- the identity of the three files --------------------------------------------------------------------------


def test_a_file_with_the_expected_sha_goes_on_to_serving_and_the_check_is_recorded(drv):
    out = drv.run(*args("--expect-sha", Q4_SHA))
    assert out.returncode == 0, out.stdout + out.stderr
    assert drv.names() == ["q4_prepare.sh", "arm_pipeline.sh"]
    check = json.loads(drv.sha_check.read_text())
    assert check == {"expected": Q4_SHA, "got": Q4_SHA, "identical": True}


def test_the_expected_sha_may_be_written_in_capitals(drv):
    assert drv.run(*args("--expect-sha", Q4_SHA.upper())).returncode == 0


def test_a_file_with_another_sha_fails_the_pod_before_anything_is_served(drv):
    other = "0" * 64
    out = drv.run(*args("--bench", "--expect-sha", other))
    assert out.returncode == 1
    assert f"Q4_SHA_MISMATCH: expected {other} got {Q4_SHA}" in out.stdout
    assert drv.names() == ["q4_prepare.sh"]  # no pipeline, no server, no bench
    assert drv.done.read_text().splitlines()[0].startswith("Q4_SHA_MISMATCH")
    check = json.loads(drv.sha_check.read_text())
    assert check["identical"] is False and check["expected"] == other and check["got"] == Q4_SHA


def test_without_an_expected_sha_the_sha_is_recorded_for_the_other_pods(drv):
    assert drv.run(*args()).returncode == 0
    check = json.loads(drv.sha_check.read_text())
    assert check == {"expected": None, "got": Q4_SHA, "identical": None}


# --- failures -------------------------------------------------------------------------------------------------


def test_a_failed_preparation_stops_the_driver_with_the_reason(drv):
    out = drv.run(*args("--bench"), STUB_PREPARE_FAIL="IMATRIX_SHA_MISMATCH:x")
    assert (
        out.returncode == 1
        and "Q4_DRIVER_FAILED:PREPARE:Q4_PREPARE_FAILED:IMATRIX_SHA_MISMATCH:x" in out.stdout
    )
    assert drv.names() == ["q4_prepare.sh"]
    assert drv.done.read_text().splitlines()[0].startswith("Q4_DRIVER_FAILED:PREPARE")


def test_a_prepare_that_exits_0_without_a_done_marker_is_a_failure(drv):
    (drv.pod / "q4_prepare.sh").write_text("#!/bin/bash\necho nothing\n")
    out = drv.run(*args())
    assert out.returncode == 1 and "Q4_DRIVER_FAILED:PREPARE" in out.stdout and drv.names() == []


def test_a_failed_pipeline_stops_before_the_bench(drv):
    out = drv.run(*args("--bench"), STUB_PIPELINE_RC="1")
    assert out.returncode == 1 and "Q4_DRIVER_FAILED:PIPELINE" in out.stdout
    assert drv.names() == ["q4_prepare.sh", "arm_pipeline.sh"]
    assert drv.done.read_text().splitlines()[0].startswith("Q4_DRIVER_FAILED:PIPELINE")


def test_a_bench_with_failed_points_is_done_but_says_so(drv):
    out = drv.run(*args("--bench"), STUB_BENCH_RC="3")
    assert out.returncode == 0 and out.stdout.strip().splitlines()[-1] == "Q4_DRIVER_DONE"
    assert "bench_exit=3" in drv.done.read_text().splitlines()


def test_a_failed_bench_fails_the_driver(drv):
    out = drv.run(*args("--bench"), STUB_BENCH_RC="1")
    assert out.returncode == 1 and "Q4_DRIVER_FAILED:BENCH" in out.stdout


def test_the_done_file_is_written_last_and_an_old_one_is_removed_first(drv):
    drv.done.parent.mkdir(parents=True)
    drv.done.write_text("Q4_DRIVER_DONE\nstale\n")
    out = drv.run(*args("--bench", "--expect-sha", "0" * 64))
    assert out.returncode == 1 and "stale" not in drv.done.read_text()
    assert drv.done.stat().st_mtime_ns >= drv.sha_check.stat().st_mtime_ns
    ok = drv.run(*args("--bench"))
    assert ok.returncode == 0 and drv.done.read_text().splitlines()[0] == "Q4_DRIVER_DONE"
    assert drv.done.stat().st_mtime_ns >= drv.sha_check.stat().st_mtime_ns
