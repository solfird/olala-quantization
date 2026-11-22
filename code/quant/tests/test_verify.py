"""Tests for olala_quant.verify: a clean conversion passes, every kind of damage is caught."""

import json
from collections.abc import Callable
from pathlib import Path

import fakes
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from olala_quant import convert as conv
from olala_quant import verify

IGNORE = ("model.layers.0.mlp.*", "model.layers.1.mixer_proj.weight")


def _make(tmp_path: Path, recipe: str) -> tuple[Path, Path]:
    src, dst = tmp_path / "src", tmp_path / "dst"
    fakes.write_fake_checkpoint(src)
    conv.convert(src, dst, recipe, IGNORE, max_shard_bytes=80_000)
    return src, dst


def _mutate(dst: Path, fn: Callable[[dict[str, torch.Tensor]], None]) -> None:
    """Load every output tensor, let ``fn`` damage them, write the shards back unchanged in layout."""
    index = json.loads((dst / "model.safetensors.index.json").read_text())
    by_file: dict[str, dict[str, torch.Tensor]] = {}
    for f in sorted(set(index["weight_map"].values())):
        with safe_open(dst / f, framework="pt") as h:
            by_file[f] = {k: h.get_tensor(k) for k in h.keys()}  # noqa: SIM118
    merged = {k: v for part in by_file.values() for k, v in part.items()}
    fn(merged)
    known = {k for part in by_file.values() for k in part}
    extras = {k: v for k, v in merged.items() if k not in known}
    for n, (f, part) in enumerate(by_file.items()):
        out = {k: merged[k] for k in part if k in merged}
        if n == 0:
            out.update(extras)  # tensors added by the damage function land in the first shard
        save_file(out, str(dst / f), metadata={"format": "pt"})


def _checks(report: verify.VerifyReport) -> set[str]:
    return {v.check for v in report.violations}


@pytest.mark.parametrize("recipe", ["fp8", "nvfp4"])
def test_clean_conversion_passes(tmp_path: Path, recipe: str) -> None:
    src, dst = _make(tmp_path, recipe)
    report = verify.verify(src, dst)
    assert report.violations == []
    assert report.n_tensors_checked == len(fakes.make_tensors())
    # 4 layers x 7 quantizable tensors + 1 attention gate_proj, minus 7 ignored (layer 0 mlp x6,
    # layer 1 mixer_proj)
    assert report.stats["n_quantized"] == 4 * 7 + 1 - 7
    assert report.stats["n_ignored"] == 7
    scales = report.stats["scales"]
    if recipe == "fp8":
        assert set(scales) == {"min_scale", "max_scale"}
        assert 0 < scales["min_scale"] <= scales["max_scale"]
    else:
        assert set(scales) == {"n_flushed_blocks", "max_block_scale", "min_nonzero_block_scale"}
        assert scales["max_block_scale"] == 448.0
        assert scales["n_flushed_blocks"] == 0
    assert verify.main(["--src", str(src), "--dst", str(dst)]) == 0


def test_bit_flip_in_ignored_tensor_is_caught(tmp_path: Path) -> None:
    src, dst = _make(tmp_path, "fp8")

    def damage(t: dict[str, torch.Tensor]) -> None:
        w = t["model.layers.0.mlp.shared_experts.fc_1.weight"].clone()
        w.view(torch.uint8)[0, 0] ^= 1
        t["model.layers.0.mlp.shared_experts.fc_1.weight"] = w

    _mutate(dst, damage)
    report = verify.verify(src, dst)
    assert "bit_identical" in _checks(report)
    assert verify.main(["--src", str(src), "--dst", str(dst)]) == 1


def test_bit_flip_in_unquantized_tensor_is_caught(tmp_path: Path) -> None:
    src, dst = _make(tmp_path, "nvfp4")

    def damage(t: dict[str, torch.Tensor]) -> None:
        w = t["lm_head.weight"].clone()
        w.view(torch.uint8)[3, 3] ^= 0x10
        t["lm_head.weight"] = w

    _mutate(dst, damage)
    assert "bit_identical" in _checks(verify.verify(src, dst))


def test_missing_extra_wrong_shape_and_wrong_dtype(tmp_path: Path) -> None:
    src, dst = _make(tmp_path, "fp8")

    def damage(t: dict[str, torch.Tensor]) -> None:
        del t["model.layers.2.mixer_proj.weight_scale"]
        t["model.layers.9.junk"] = torch.zeros(2)
        t["model.layers.3.mixer_proj.weight"] = t["model.layers.3.mixer_proj.weight"][:, :-1].contiguous()
        t["model.layers.2.gate_proj.weight_scale"] = t["model.layers.2.gate_proj.weight_scale"].to(
            torch.bfloat16
        )

    _mutate(dst, damage)
    checks = _checks(verify.verify(src, dst))
    assert {"missing_tensor", "unexpected_tensor", "shape", "dtype"} <= checks


def test_destroyed_weights_exceed_the_error_bound(tmp_path: Path) -> None:
    src, dst = _make(tmp_path, "fp8")

    def damage(t: dict[str, torch.Tensor]) -> None:
        t["model.layers.2.mlp.experts.experts.weight"] = torch.zeros_like(
            t["model.layers.2.mlp.experts.experts.weight"]
        )

    _mutate(dst, damage)
    assert "reconstruction_error" in _checks(verify.verify(src, dst))


def test_nvfp4_scrambled_packed_values_exceed_the_error_bound(tmp_path: Path) -> None:
    src, dst = _make(tmp_path, "nvfp4")

    def damage(t: dict[str, torch.Tensor]) -> None:
        w = t["model.layers.2.mixer_proj.weight_packed"]
        t["model.layers.2.mixer_proj.weight_packed"] = w.flip(0).contiguous()

    _mutate(dst, damage)
    assert "reconstruction_error" in _checks(verify.verify(src, dst))


def test_nan_and_invalid_scales_are_caught(tmp_path: Path) -> None:
    src, dst = _make(tmp_path, "nvfp4")

    def damage(t: dict[str, torch.Tensor]) -> None:
        s = t["model.layers.2.mixer_proj.weight_scale"].view(torch.uint8).clone()
        s[0, 0] = 0x7F  # e4m3 NaN
        t["model.layers.2.mixer_proj.weight_scale"] = s.view(torch.float8_e4m3fn)
        g = t["model.layers.3.mixer_proj.weight_global_scale"].clone()
        g[0] = -1.0
        t["model.layers.3.mixer_proj.weight_global_scale"] = g

    _mutate(dst, damage)
    checks = _checks(verify.verify(src, dst))
    assert "finite" in checks
    assert "scale_range" in checks


def test_fp8_scale_must_be_positive_and_finite(tmp_path: Path) -> None:
    src, dst = _make(tmp_path, "fp8")

    def damage(t: dict[str, torch.Tensor]) -> None:
        s = t["model.layers.2.mixer_proj.weight_scale"].clone()
        s[0, 0] = float("inf")
        s[1, 0] = 0.0
        t["model.layers.2.mixer_proj.weight_scale"] = s

    _mutate(dst, damage)
    checks = _checks(verify.verify(src, dst))
    assert {"finite", "scale_range"} <= checks


def test_quantized_tensor_where_the_ignore_list_says_bf16_is_caught(tmp_path: Path) -> None:
    src, dst = _make(tmp_path, "fp8")

    def damage(t: dict[str, torch.Tensor]) -> None:
        w = t["model.layers.1.mixer_proj.weight"]  # ignored: must stay bf16
        t["model.layers.1.mixer_proj.weight"] = w.to(torch.float32).to(torch.float8_e4m3fn)

    _mutate(dst, damage)
    checks = _checks(verify.verify(src, dst))
    assert {"dtype", "bit_identical"} & checks


def test_config_index_and_info_are_cross_checked(tmp_path: Path) -> None:
    src, dst = _make(tmp_path, "fp8")
    cfg = json.loads((dst / "config.json").read_text())
    cfg["quantization_config"]["ignore"] = [e for e in cfg["quantization_config"]["ignore"] if e != "lm_head"]
    (dst / "config.json").write_text(json.dumps(cfg))
    assert "config" in _checks(verify.verify(src, dst))

    src2, dst2 = _make(tmp_path / "two", "fp8")
    info = json.loads((dst2 / "quantization_info.json").read_text())
    info["tensors"]["model.layers.2.mixer_proj.weight"]["rel_fro_error"] *= 2
    info["source"]["identity_sha256"] = "0" * 64
    (dst2 / "quantization_info.json").write_text(json.dumps(info))
    checks = _checks(verify.verify(src2, dst2))
    assert {"info_error_mismatch", "info_source_hash"} <= checks

    src3, dst3 = _make(tmp_path / "three", "fp8")
    (dst3 / "tokenizer_config.json").write_text("changed")
    idx = json.loads((dst3 / "model.safetensors.index.json").read_text())
    idx["weight_map"].pop("lm_head.weight")
    (dst3 / "model.safetensors.index.json").write_text(json.dumps(idx))
    checks = _checks(verify.verify(src3, dst3))
    assert {"auxiliary_file", "index"} <= checks


def test_usage_errors_exit_2(tmp_path: Path) -> None:
    assert verify.main(["--src", str(tmp_path / "nope"), "--dst", str(tmp_path / "nope2")]) == 2


@pytest.mark.parametrize("recipe", ["fp8", "nvfp4"])
def test_relative_error_bound_is_enforced_on_its_own(tmp_path: Path, recipe: str) -> None:
    # An intact conversion, judged against an absurdly tight bound: only the Frobenius bound can fail
    # here (every element is within its own half-step bound).
    src, dst = _make(tmp_path, recipe)
    tight = verify.Bounds(fp8_rel_fro=1e-4, nvfp4_rel_fro=1e-4)
    report = verify.verify(src, dst, tight)
    assert _checks(report) == {"reconstruction_error"}
    assert all("relative Frobenius" in v.message for v in report.violations)
