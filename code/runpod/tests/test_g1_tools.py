"""pod/g1_tools.py: the parsing and arithmetic of the G1 session (feasibility projection, gate, fidelity, checkpoint record).

The thresholds are arguments (the session script holds them, in ONE place); these tests feed the boundaries.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import RUNPOD

TOOLS = RUNPOD / "pod" / "g1_tools.py"
SCALE = "72.75"
SECONDS_PER_4H = 4 * 3600


def tool(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOLS), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def feasibility_text(forward: list[float], peaks: list[float], solve_peak: float = 1.0) -> str:
    """The output of `pod_g1lite.sh feasibility` (its python table), with noise around it."""
    lines = [
        "+ env OLALA_TORCH_MOE=1 python3 build_g1lite.py ...   # log: /x/feasibility-1.log",
        "some library warning",
        "every layer that was run (forward only, or captured):",
        "layer  captured  forward_s  mixer_s  mlp_s  peak_cuda_GB  max|mixer_out|  max|mlp_out|",
    ]
    for i, (f, p) in enumerate(zip(forward, peaks, strict=True)):
        cap = i % 2 == 0
        lines.append(
            f"{i + 2:>5}  {cap!s:>8}  {f:9.1f}  {f / 2:7.1f}  {f / 4:5.1f}  {p:12.2f}  "
            f"{1.5e3:13.3g}  {2.5:11.3g}"
        )
    lines += [
        "calibrated layers:",
        "layer  hessian_s  solve_s  total_s  peak_after_solve_GB  gptq/rtn  mean_loss_ratio_up  mean_heldout_ratio",
        f"    2        10.0     20.0     30.0  {solve_peak:19.2f}  250/6  0.5  0.9",
        "G1LITE_FEASIBILITY_DONE",
    ]
    return "\n".join(lines) + "\n"


def run_feasibility(
    tmp: Path, text: str, *extra: str, budget: str = "4.0", peak_max: str = "30"
) -> tuple[subprocess.CompletedProcess[str], dict]:
    log = tmp / "feas.txt"
    log.write_text(text)
    out = tmp / "feasibility.json"
    res = tool(
        "feasibility",
        str(log),
        "--out",
        str(out),
        "--scale",
        SCALE,
        "--n-layers",
        "36",
        "--budget-hours",
        budget,
        "--peak-gib-max",
        peak_max,
        *extra,
    )
    return res, json.loads(out.read_text()) if out.exists() else {}


# --- feasibility: projection ---------------------------------------------------------------------------------------


def test_the_projection_is_scale_times_the_36_layer_sum_of_forward_seconds(tmp_path):
    res, data = run_feasibility(
        tmp_path, feasibility_text([10.0, 20.0, 30.0], [5.0, 6.0, 7.0]), budget="20"
    )
    assert res.returncode == 0, res.stderr
    assert data["measured_layers"] == 3
    assert data["mean_forward_s"] == pytest.approx(20.0)
    assert data["sum_forward_s_36_layers"] == pytest.approx(720.0)
    assert data["projected_build_seconds"] == pytest.approx(72.75 * 720.0)
    assert data["projected_build_hours"] == pytest.approx(72.75 * 720.0 / 3600)
    assert "72.75" in data["formula"] and "36" in data["formula"]
    assert "projected" in res.stdout.lower() and "72.75" in res.stdout  # the log states the formula


def test_the_projection_boundary_of_the_budget_is_strict(tmp_path):
    # 5.4 s per layer: 72.75 * 36 * 5.4 = 14,144 s < 14,400 (4 h); 5.5 s: 14,404.5 s > 14,400
    ok, data_ok = run_feasibility(tmp_path, feasibility_text([5.4] * 3, [5.0] * 3))
    assert ok.returncode == 0 and data_ok["ok"] is True
    assert data_ok["projected_build_seconds"] < SECONDS_PER_4H
    over, data_over = run_feasibility(tmp_path, feasibility_text([5.5] * 3, [5.0] * 3))
    assert over.returncode == 3 and data_over["ok"] is False
    assert data_over["projected_build_seconds"] > SECONDS_PER_4H
    assert any("projected build" in r for r in data_over["refusals"])


def test_a_projection_exactly_at_the_budget_is_not_refused(tmp_path):
    # scale 100 x 36 layers x 4.0 s = 14,400 s = 4 h exactly, and 4.1 s is over
    log = tmp_path / "feas.txt"
    for seconds, rc in ((4.0, 0), (4.1, 3)):
        log.write_text(feasibility_text([seconds] * 3, [5.0] * 3))
        res = tool(
            *("feasibility", str(log), "--out", str(tmp_path / "o.json")),
            *("--scale", "100", "--n-layers", "36", "--budget-hours", "4", "--peak-gib-max", "30"),
        )
        assert res.returncode == rc, (seconds, res.stdout)


def test_a_peak_exactly_at_the_memory_limit_is_not_refused(tmp_path):
    # 2.147483648 GB = 2 GiB exactly
    for peak, rc in (("2.147483648", 0), ("2.147483649", 3)):
        log = tmp_path / "feas.txt"
        log.write_text(
            "layer  captured  forward_s  mixer_s  mlp_s  peak_cuda_GB  max|mixer_out|  max|mlp_out|\n"
            f"    2      True        1.0      0.5    0.2  {peak}  1.5  2.5\n"
        )
        res = tool(
            *("feasibility", str(log), "--out", str(tmp_path / "o.json")),
            *(
                "--scale",
                "72.75",
                "--n-layers",
                "36",
                "--budget-hours",
                "40",
                "--peak-gib-max",
                "2",
            ),
        )
        assert res.returncode == rc, (peak, res.stdout)


def test_the_budget_is_an_argument_not_a_constant(tmp_path):
    text = feasibility_text([5.5] * 3, [5.0] * 3)
    res, data = run_feasibility(tmp_path, text, budget="4.5")
    assert res.returncode == 0 and data["ok"] is True
    res, data = run_feasibility(tmp_path, text, budget="3.9")
    assert res.returncode == 3 and data["budget_hours"] == 3.9


def test_the_peak_memory_limit_is_30_gib_not_30_gb(tmp_path):
    # the table prints GB of 1e9 bytes: 30 GiB = 32.2122 GB
    ok, data = run_feasibility(tmp_path, feasibility_text([1.0], [32.2]))
    assert ok.returncode == 0 and data["ok"] is True
    assert data["peak_gib"] == pytest.approx(32.2e9 / 2**30)
    over, data = run_feasibility(tmp_path, feasibility_text([1.0], [32.3]))
    assert over.returncode == 3 and any("peak" in r for r in data["refusals"])


def test_the_peak_after_solve_counts_as_peak_memory_too(tmp_path):
    res, data = run_feasibility(tmp_path, feasibility_text([1.0], [5.0], solve_peak=40.0))
    assert res.returncode == 3 and data["peak_gb"] == pytest.approx(40.0)


def test_both_refusals_are_listed(tmp_path):
    res, data = run_feasibility(tmp_path, feasibility_text([50.0], [40.0]))
    assert res.returncode == 3 and len(data["refusals"]) == 2


def test_a_log_without_the_table_is_an_error_not_a_pass(tmp_path):
    res, data = run_feasibility(tmp_path, "nothing useful\nG1LITE_FEASIBILITY_DONE\n")
    assert res.returncode == 2 and "forward" in res.stderr
    assert data.get("ok") is not True


def test_an_unreadable_row_is_an_error(tmp_path):
    text = feasibility_text([1.0, 2.0], [1.0, 1.0]).replace("      2.0", "      nan?", 1)
    res, _ = run_feasibility(tmp_path, text)
    assert res.returncode == 2


# --- gate ----------------------------------------------------------------------------------------------------------


def fidelity_file(tmp: Path, kl: object, ppl: object) -> Path:
    path = tmp / "fidelity.json"
    path.write_text(json.dumps({"mean_kl": {"value": kl}, "ppl_ratio": {"value": ppl}}))
    return path


def run_gate(tmp: Path, kl: object, ppl: object, **over: str) -> tuple[int, dict]:
    out = tmp / "gate.json"
    out.unlink(missing_ok=True)
    args = {
        "--kl-max": "0.0194",
        "--ppl-ratio-max": "1.0197",
        "--n5-kl": "0.0243",
        "--n5-ppl-ratio": "1.0197",
        "--n5-from": "results/wave1/ablation-abl-experts",
        "--source": "PROTOCOL.md amendment 22",
    }
    args.update(over)
    flat = [x for kv in args.items() for x in kv]
    res = tool("gate", str(fidelity_file(tmp, kl, ppl)), "--out", str(out), *flat)
    return res.returncode, json.loads(out.read_text()) if out.exists() else {}


@pytest.mark.parametrize(
    ("kl", "ppl", "passes"),
    [
        (0.0194, 1.0197, True),  # both exactly at the threshold pass
        (0.01941, 1.0197, False),
        (0.0194, 1.01971, False),
        (0.01939, 1.01969, True),
        (0.0, 1.0, True),
        (0.0243, 1.0197, False),  # N5 itself fails the KL threshold
        (0.0100, 1.5, False),
        (0.5, 1.0, False),
    ],
)
def test_the_gate_arithmetic_at_the_boundaries(tmp_path, kl, ppl, passes):
    rc, data = run_gate(tmp_path, kl, ppl)
    assert rc == (0 if passes else 1)
    assert data["pass"] is passes
    assert data["kl"] == kl and data["ppl_ratio"] == ppl


def test_the_gate_json_carries_the_thresholds_their_source_and_the_n5_reference(tmp_path):
    rc, data = run_gate(tmp_path, 0.015, 1.01)
    assert rc == 0
    assert data["thresholds"] == {
        "kl_max": 0.0194,
        "ppl_ratio_max": 1.0197,
        "source": "PROTOCOL.md amendment 22",
    }
    assert data["n5_reference"] == {
        "kl": 0.0243,
        "ppl_ratio": 1.0197,
        "from": "results/wave1/ablation-abl-experts",
    }


def test_the_thresholds_are_the_arguments_not_constants_of_the_tool(tmp_path):
    rc, data = run_gate(tmp_path, 0.03, 1.1, **{"--kl-max": "0.05", "--ppl-ratio-max": "1.2"})
    assert rc == 0 and data["pass"] is True and data["thresholds"]["kl_max"] == 0.05


@pytest.mark.parametrize("bad", [None, "x", float("nan"), float("inf"), -0.1, True, []])
def test_a_missing_or_non_finite_value_never_passes(tmp_path, bad):
    rc, data = run_gate(tmp_path, bad, 1.0)
    assert rc == 2 and data.get("pass") is not True
    rc, data = run_gate(tmp_path, 0.01, bad)
    assert rc == 2 and data.get("pass") is not True


def test_a_gate_without_threshold_arguments_is_refused(tmp_path):
    res = tool("gate", str(fidelity_file(tmp_path, 0.01, 1.0)), "--out", str(tmp_path / "g.json"))
    assert res.returncode == 2 and not (tmp_path / "g.json").exists()


# --- fidelity.json -------------------------------------------------------------------------------------------------

COMPARE = {
    "ppl_ref": {"value": 14.6, "lo": 13.3, "hi": 15.8},
    "ppl_var": {"value": 14.9, "lo": 13.6, "hi": 16.1},
    "ppl_ratio": {"value": 1.0197, "lo": 1.016, "hi": 1.023},
    "kld_mean": {"value": 0.0242, "lo": 0.022, "hi": 0.026},
    "kld_median": {"value": 0.0096, "lo": 0.0093, "hi": 0.0099},
    "kld_p99": {"value": 0.257, "lo": 0.233, "hi": 0.291},
    "top1_agreement": {"value": 0.912, "lo": 0.909, "hi": 0.915},
    "n_chunks": 64,
    "n_positions": 32704,
    "k": 20,
    "n_boot": 1000,
    "seed": 0,
    "confidence": 0.95,
}


def test_fidelity_json_has_mean_median_p99_ppl_top1_with_intervals(tmp_path):
    src = tmp_path / "fidelity_vs_b0.json"
    src.write_text(json.dumps(COMPARE))
    ref = tmp_path / "ref.npz"
    ref.write_bytes(b"reference")
    out = tmp_path / "fidelity.json"
    res = tool("fidelity", str(src), "--reference", str(ref), "--out", str(out))
    assert res.returncode == 0, res.stderr
    data = json.loads(out.read_text())
    assert data["mean_kl"] == COMPARE["kld_mean"]
    assert data["median_kl"] == COMPARE["kld_median"]
    assert data["p99_kl"] == COMPARE["kld_p99"]
    assert data["ppl_ratio"] == COMPARE["ppl_ratio"]
    assert data["top1_agreement"] == COMPARE["top1_agreement"]
    assert data["n_chunks"] == 64 and data["n_positions"] == 32704 and data["confidence"] == 0.95
    assert data["reference"]["sha256"] == hashlib.sha256(b"reference").hexdigest()
    assert data["reference"]["path"] == str(ref)


def test_fidelity_json_refuses_a_comparison_without_the_headline_numbers(tmp_path):
    bad = {k: v for k, v in COMPARE.items() if k != "kld_mean"}
    src = tmp_path / "f.json"
    src.write_text(json.dumps(bad))
    res = tool("fidelity", str(src), "--reference", str(src), "--out", str(tmp_path / "o.json"))
    assert res.returncode == 2 and not (tmp_path / "o.json").exists()


# --- checkpoint record ---------------------------------------------------------------------------------------------


def test_the_checkpoint_record_has_sha256_of_the_three_files_and_the_size_in_bytes(tmp_path):
    ck = tmp_path / "ck"
    ck.mkdir()
    files = {
        "model.safetensors.index.json": b'{"a": 1}',
        "config.json": b'{"b": 2}',
        "quantization_info.json": b'{"c": 3}',
        "model-00001-of-00002.safetensors": b"x" * 1000,
        "model-00002-of-00002.safetensors": b"y" * 234,
    }
    for name, data in files.items():
        (ck / name).write_bytes(data)
    out = tmp_path / "checkpoint.json"
    res = tool("checkpoint", str(ck), "--out", str(out))
    assert res.returncode == 0, res.stderr
    data = json.loads(out.read_text())
    assert data["size_bytes"] == sum(len(v) for v in files.values())
    for name in ("model.safetensors.index.json", "config.json", "quantization_info.json"):
        assert data["sha256"][name] == hashlib.sha256(files[name]).hexdigest()
    assert data["n_shards"] == 2 and data["path"] == str(ck)


def test_a_checkpoint_without_its_metadata_is_refused(tmp_path):
    ck = tmp_path / "ck"
    ck.mkdir()
    (ck / "config.json").write_text("{}")
    res = tool("checkpoint", str(ck), "--out", str(tmp_path / "c.json"))
    assert res.returncode == 2 and not (tmp_path / "c.json").exists()


# --- run status ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("status", "rc"), [("complete", 0), ("running", 1), ("failed", 1)])
def test_run_status_is_zero_only_for_a_complete_run(tmp_path, status, rc):
    info = tmp_path / "run_info.json"
    info.write_text(json.dumps({"status": status}))
    assert tool("run-status", str(info)).returncode == rc


def test_run_status_of_a_missing_or_broken_file_is_not_complete(tmp_path):
    assert tool("run-status", str(tmp_path / "none.json")).returncode == 1
    (tmp_path / "bad.json").write_text("{")
    assert tool("run-status", str(tmp_path / "bad.json")).returncode == 1


# --- calibration files ---------------------------------------------------------------------------------------------


def calib_dir(tmp: Path, **over: object) -> Path:
    """A CAL-1 output directory with the three files the build needs and a manifest whose files block hashes them."""
    root = tmp / "calib-cal1"
    root.mkdir()
    windows = b"[[1, 2, 3]]\n"
    text = b'{"text": "a trace"}\n'
    (root / "windows.json").write_bytes(windows)
    (root / "calib_text.jsonl").write_bytes(text)
    manifest = {
        "schema": "cal1/1",
        "n_windows": 1,
        "window": 3,
        "windows": {"file": "windows.json", "sha256": hashlib.sha256(windows).hexdigest()},
        "files": {
            "windows.json": hashlib.sha256(windows).hexdigest(),
            "calib_text.jsonl": hashlib.sha256(text).hexdigest(),
            "generations.jsonl": "0" * 64,
        },
        "audit": {"ok": True, "n_hits": 0},
    }
    manifest.update(over)
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_calib_check_accepts_a_directory_whose_files_match_the_manifest_files_block(tmp_path):
    root = calib_dir(tmp_path)
    out = tmp_path / "calibration.json"
    res = tool("calib-check", str(root), "--out", str(out), "--list")
    assert res.returncode == 0, res.stderr
    record = json.loads(out.read_text())
    assert set(record["sha256"]) == {"manifest.json", "windows.json", "calib_text.jsonl"}
    for name, digest in record["sha256"].items():
        assert digest == hashlib.sha256((root / name).read_bytes()).hexdigest()
        assert f"{digest}  {name}" in res.stdout.splitlines()  # the list `sha256sum -c` reads


@pytest.mark.parametrize("victim", ["windows.json", "calib_text.jsonl"])
def test_calib_check_refuses_a_file_that_differs_from_the_manifest(tmp_path, victim):
    root = calib_dir(tmp_path)
    (root / victim).write_bytes(b"tampered\n")
    res = tool("calib-check", str(root))
    assert res.returncode == 2 and victim in res.stderr


@pytest.mark.parametrize("victim", ["windows.json", "calib_text.jsonl", "manifest.json"])
def test_calib_check_refuses_a_missing_file(tmp_path, victim):
    root = calib_dir(tmp_path)
    (root / victim).unlink()
    assert tool("calib-check", str(root)).returncode == 2


@pytest.mark.parametrize(
    "over",
    [
        {"schema": "other/1"},
        {"audit": {"ok": False, "n_hits": 0}},
        {"audit": {"ok": True, "n_hits": 3}},
        {"files": {"windows.json": "0" * 64}},
        {"windows": {"file": "windows.json", "sha256": "0" * 64}},
    ],
)
def test_calib_check_refuses_a_manifest_that_is_not_an_audited_cal1_set(tmp_path, over):
    root = calib_dir(tmp_path, **over)
    assert tool("calib-check", str(root)).returncode == 2
