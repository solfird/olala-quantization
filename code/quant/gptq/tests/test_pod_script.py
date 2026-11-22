"""pod_g1lite.sh: dry-run prints the exact commands and touches nothing; refuses missing inputs; valid bash."""

import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "pod_g1lite.sh"


def run(*args: str, work: Path, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = {"PATH": "/usr/bin:/bin", "OLALA_WORK": str(work), **(env_extra or {})}
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, env=env, check=False)


def test_the_script_is_valid_bash() -> None:
    assert subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, check=False).returncode == 0


def test_dry_run_feasibility_prints_the_commands_and_creates_nothing(tmp_path: Path) -> None:
    r = run("--dry-run", "feasibility", work=tmp_path)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "build_g1lite.py" in out and "--layers 2,4,6" in out and "--calibrate-only" in out
    assert "--max-windows 4" in out and "--device cuda" in out and "--attn-impl eager" in out
    assert f"{tmp_path}/models/olala-bf16" in out and f"{tmp_path}/logs/g1lite/" in out
    assert "OLALA_TORCH_MOE=1" in out and "OLALA_TORCH_MAMBA3=1" in out and "HF_HUB_OFFLINE=1" in out
    assert list(tmp_path.iterdir()) == []  # nothing created


def test_dry_run_build_verify_and_tokenize(tmp_path: Path) -> None:
    out = run("--dry-run", "build", "--layers", "2,3", work=tmp_path).stdout
    assert "--layers 2,3" in out and "--propagate bf16" in out and f"--dst {tmp_path}/ckpt/g1lite" in out
    out = run("--dry-run", "verify", work=tmp_path).stdout
    assert "olala_quant.verify" in out and "--calibrated" in out
    out = run("--dry-run", "tokenize", work=tmp_path).stdout
    assert "calib_data.py tokenize" in out and "--n-tokens 65536" in out and "--window 2048" in out
    assert "leakage_audit.py" in out


def test_unknown_step_is_refused(tmp_path: Path) -> None:
    r = run("--dry-run", "frobnicate", work=tmp_path)
    assert r.returncode == 2


@pytest.mark.parametrize("step", ["build", "feasibility", "tokenize"])
def test_a_real_run_without_the_weights_is_refused_before_doing_anything(tmp_path: Path, step: str) -> None:
    r = run(step, work=tmp_path)
    assert r.returncode == 1
    assert "olala-bf16" in r.stderr
    assert not (tmp_path / "ckpt").exists()


# ------------------------------------------------------------------------- base selection and CAL-1 windows


def test_by_default_no_base_select_and_no_select_are_passed(tmp_path: Path) -> None:
    for step in ("feasibility", "build", "verify"):
        out = run("--dry-run", step, work=tmp_path).stdout
        assert "--base-select" not in out and "--select" not in out, step


@pytest.mark.parametrize("step", ["feasibility", "build"])
def test_g1_base_select_is_passed_to_build_g1lite(tmp_path: Path, step: str) -> None:
    out = run("--dry-run", step, work=tmp_path, env_extra={"G1_BASE_SELECT": "experts"}).stdout
    assert "build_g1lite.py" in out and "--base-select experts" in out


def test_g1_base_select_is_required_of_the_checkpoint_by_verify(tmp_path: Path) -> None:
    out = run("--dry-run", "verify", work=tmp_path, env_extra={"G1_BASE_SELECT": "experts:layers=12-23"}).stdout
    assert "olala_quant.verify" in out and "--calibrated" in out and "--select experts:layers=12-23" in out


@pytest.mark.parametrize("step", ["feasibility", "build"])
def test_g1_calib_is_passed_as_the_calibration_manifest(tmp_path: Path, step: str) -> None:
    out = run("--dry-run", step, work=tmp_path, env_extra={"G1_CALIB": "/data/calib-cal1"}).stdout
    assert "--calib /data/calib-cal1/manifest.json" in out
    assert "--calib" in run("--dry-run", step, work=tmp_path).stdout  # the default stays calib-lite
    assert f"{tmp_path}/bundle/quant/gptq/data/calib-lite/manifest.json" in run("--dry-run", step, work=tmp_path).stdout


def cal1_dir(tmp_path: Path, windows: bool = True) -> Path:
    d = tmp_path / "calib-cal1"
    d.mkdir()
    win = (
        '  "windows": {\n    "file": "windows.json",\n    "sha256": "x"\n  },\n' if windows else '  "windows": null,\n'
    )
    (d / "manifest.json").write_text('{\n  "schema": "cal1/1",\n' + win + '  "audit": {"ok": true}\n}\n')
    return d


def test_tokenize_is_not_needed_and_not_run_when_cal1_windows_are_given(tmp_path: Path) -> None:
    d = cal1_dir(tmp_path)
    r = run("--dry-run", "tokenize", work=tmp_path / "w", env_extra={"G1_CALIB": str(d)})
    assert r.returncode == 0, r.stderr
    assert "calib_data.py tokenize" not in r.stdout and "leakage_audit.py" not in r.stdout
    assert "not needed" in r.stdout and "CAL-1" in r.stdout
    r = run("--dry-run", "all", work=tmp_path / "w", env_extra={"G1_CALIB": str(d), "G1_BASE_SELECT": "experts"})
    assert r.returncode == 0, r.stderr
    assert "calib_data.py tokenize" not in r.stdout
    assert "build_g1lite.py" in r.stdout and "--base-select experts" in r.stdout and "olala_quant.verify" in r.stdout


def test_a_real_tokenize_step_on_cal1_windows_does_nothing_to_them(tmp_path: Path) -> None:
    d = cal1_dir(tmp_path)
    before = (d / "manifest.json").read_bytes()
    model = tmp_path / "w" / "models" / "olala-bf16"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (tmp_path / "w" / "bundle" / "quant" / "gptq").mkdir(parents=True)
    r = run("tokenize", work=tmp_path / "w", env_extra={"G1_CALIB": str(d)})
    assert r.returncode == 0, r.stderr
    assert "G1LITE_TOKENIZE_DONE" in r.stdout and "calib_data.py tokenize" not in r.stdout
    assert (d / "manifest.json").read_bytes() == before


def test_tokenize_still_runs_for_a_manifest_without_windows_or_not_cal1(tmp_path: Path) -> None:
    d = cal1_dir(tmp_path, windows=False)
    out = run("--dry-run", "tokenize", work=tmp_path / "w", env_extra={"G1_CALIB": str(d)}).stdout
    assert "calib_data.py tokenize" in out
    out = run("--dry-run", "all", work=tmp_path / "w2").stdout  # the default calib-lite
    assert "calib_data.py tokenize" in out


def test_tokenize_still_runs_for_a_tokenized_lite_bundle_that_is_not_cal1(tmp_path: Path) -> None:
    d = cal1_dir(tmp_path)
    m = (d / "manifest.json").read_text().replace("cal1/1", "g1lite-calib/1")
    (d / "manifest.json").write_text(m)
    out = run("--dry-run", "tokenize", work=tmp_path / "w", env_extra={"G1_CALIB": str(d)}).stdout
    assert "calib_data.py tokenize" in out  # only a CAL-1 output is exempt: its windows come from cal1_generate.py
