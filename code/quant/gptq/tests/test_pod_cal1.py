"""pod_cal1.sh: dry-run prints the exact commands and touches nothing; refuses missing inputs; valid bash."""

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "pod_cal1.sh"
REVS = "math_train=" + "a" * 40 + " mgsm_train_parallel=" + "b" * 40


def run(*args: str, work: Path, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = {"PATH": "/usr/bin:/bin", "OLALA_WORK": str(work), **(env_extra or {})}
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, env=env, check=False)


def test_the_script_is_valid_bash() -> None:
    assert subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, check=False).returncode == 0


def test_dry_run_plan_prints_the_tool_dry_run_and_creates_nothing(tmp_path: Path) -> None:
    r = run("--dry-run", "plan", work=tmp_path, env_extra={"CAL1_REVISIONS": REVS})
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "cal1_generate.py" in out and "--dry-run" in out and "--model olala" in out
    assert "--base-url http://127.0.0.1:8000/v1" in out and f"--out {tmp_path}/calib-cal1" in out
    assert f"--revision math_train={'a' * 40}" in out and f"--revision mgsm_train_parallel={'b' * 40}" in out
    assert f"--tokenizer {tmp_path}/models/olala-bf16" in out and "--window 2048" in out
    assert "CAL1_PLAN_DRY_RUN" in out
    assert list(tmp_path.iterdir()) == []


def test_dry_run_run_and_verify(tmp_path: Path) -> None:
    env = {"CAL1_REVISIONS": REVS, "CAL1_PROVENANCE": "vllm 0.0 olala-bf16"}
    out = run("--dry-run", "run", "--max-windows", "64", work=tmp_path, env_extra=env).stdout
    assert "--server-provenance vllm 0.0 olala-bf16" in out and "--max-windows 64" in out
    assert "--dry-run" not in out.split("# log")[0]
    out = run("--dry-run", "verify", work=tmp_path, env_extra=env).stdout
    assert "--verify" in out and f"--out {tmp_path}/calib-cal1" in out


def test_a_real_run_without_provenance_or_weights_is_refused_before_doing_anything(tmp_path: Path) -> None:
    r = run("run", work=tmp_path, env_extra={"CAL1_REVISIONS": REVS})
    assert r.returncode == 1 and "CAL1_PROVENANCE" in r.stderr
    r = run("run", work=tmp_path, env_extra={"CAL1_REVISIONS": REVS, "CAL1_PROVENANCE": "x"})
    assert r.returncode == 1 and "olala-bf16" in r.stderr
    assert not (tmp_path / "calib-cal1").exists()


def test_unknown_step_is_refused(tmp_path: Path) -> None:
    assert run("--dry-run", "frobnicate", work=tmp_path).returncode == 2
