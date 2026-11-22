"""Regression: a build WITHOUT ``--base-select`` is what it was before the option existed, byte for byte.

The golden values were measured on the unmodified ``build_g1lite.py`` (tiny random Olala seed 0, ``make_calibration``
seed 0, torch 2.14.0 on the CPU, the pinned ``uv.lock``). Shards, index and ``config.json`` are compared by sha256; the
``quantization_info.json`` by the sha256 of its content without the keys that vary from run to run (time stamps, tool
versions, paths, the sha256 of the manifest and of the report, wall-clock seconds).
"""

import hashlib
import json
from pathlib import Path

import pytest

import build_g1lite as bg
import tiny_model
from tests_support import make_calibration

pytestmark = pytest.mark.slow

GOLDEN = {
    "model-00001-of-00001.safetensors": "65f92bc92f9817c4ef6c786aa1c9f80df1c9d6ce281662d96bcf49f4401922d3",
    "config.json": "e6bd6bd21c19c10193394ac9ea12c3807f0d7d65e6904f231f2a07efc2a9a935",
    "model.safetensors.index.json": "3aaa650a71ce3264fc5e91e97ee850f343907c0f4640d237398bac260db9d93d",
    "quantization_info.json (normalized)": "f35898226ff287d5dd74f3113252a6b516d616f5406dbb680dc472db8eb7061a",
}


def digest(dst: Path) -> dict[str, str]:
    out = {}
    for f in [*sorted(dst.glob("model-*.safetensors")), dst / "config.json", dst / "model.safetensors.index.json"]:
        out[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    info = json.loads((dst / "quantization_info.json").read_text())
    for k in ("created_utc", "tool"):
        info.pop(k)
    info["source"].pop("directory")
    cal = info["calibration"]
    cal["manifest"].pop("path")
    cal["manifest"].pop("sha256")
    cal.pop("report")
    cal["totals"].pop("wall_seconds")
    cal["totals"].pop("seconds_per_layer")
    out["quantization_info.json (normalized)"] = hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()
    return out


def test_a_default_build_is_byte_identical_to_the_one_before_the_option_existed(tiny_dir: Path, tmp_path: Path) -> None:
    calib = make_calibration(tmp_path)
    ignore = tmp_path / "ignore.json"
    ignore.write_text(json.dumps(list(tiny_model.TINY_IGNORE)))
    dst = tmp_path / "g1"
    args = ["--src", str(tiny_dir), "--calib", str(calib), "--dst", str(dst), "--ignore-file", str(ignore)]
    assert bg.main([*args, "--heldout-every", "4", "--batch-windows", "4"]) == 0
    assert digest(dst) == GOLDEN
    info = json.loads((dst / "quantization_info.json").read_text())
    assert "selection" not in info and "base_select" not in info["calibration"]
