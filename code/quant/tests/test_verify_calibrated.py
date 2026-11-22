"""Tests of the calibrated mode of olala_quant.verify and of the convert hooks that G1-lite uses."""

import json
from collections.abc import Callable
from pathlib import Path

import fakes
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from olala_quant import convert as conv
from olala_quant import nvfp4, verify
from olala_quant.metrics import ErrorAccumulator

IGNORE = ("model.layers.0.mlp.*", "model.layers.1.mixer_proj.weight")
ROUTED = (
    "model.layers.2.mlp.experts.experts.weight",
    "model.layers.2.mlp.experts.output_experts.weight",
)


def _perturb(
    packed: torch.Tensor, frac: float, seed: int, skip_experts: tuple[int, ...] = ()
) -> torch.Tensor:
    """Valid e2m1 codes that differ from the RTN ones: move the magnitude code of some elements one step."""
    codes = nvfp4.unpack_fp4(packed)
    g = torch.Generator().manual_seed(seed)
    move = torch.rand(codes.shape, generator=g) < frac
    for e in skip_experts:
        move[e] = False
    mag = (codes & 7).to(torch.int16)
    mag = torch.where(move, torch.where(mag >= 7, mag - 1, mag + 1), mag)
    sign = codes & 8
    new = mag.to(torch.uint8) | torch.where(mag == 0, torch.zeros_like(sign), sign)
    return nvfp4.pack_fp4(new)


def _override(skip_experts: tuple[int, ...] = ()) -> Callable:
    def fn(name: str, tensor: torch.Tensor) -> tuple[dict, dict] | None:
        if name not in ROUTED:
            return None
        r = nvfp4.quantize_nvfp4(tensor)
        packed = _perturb(r.packed, 0.05, seed=len(name), skip_experts=skip_experts)
        acc = ErrorAccumulator()
        acc.update(tensor, nvfp4.dequantize_nvfp4(packed, r.block_scale, r.global_scale))
        err = acc.finish()
        base = name.removesuffix(".weight")
        outs = {
            f"{base}.weight_packed": packed,
            f"{base}.weight_scale": r.block_scale,
            f"{base}.weight_global_scale": r.global_scale,
        }
        rec = {
            "scale_stats": r.scale_stats.as_dict(),
            "global_scale_min": float(r.global_scale.min()),
            "global_scale_max": float(r.global_scale.max()),
            "shape": list(tensor.shape),
            "max_abs_error": err.max_abs,
            "rel_fro_error": err.rel_fro,
        }
        return outs, rec

    return fn


def _extra(rtn_experts: dict[str, list[int]] | None = None, tensors: tuple[str, ...] = ROUTED) -> dict:
    return {"calibration": {"calibrated_tensors": list(tensors), "rtn_experts": rtn_experts or {}}}


def _make(tmp_path: Path, skip: tuple[int, ...] = (), extra: dict | None = None) -> tuple[Path, Path]:
    src, dst = tmp_path / "src", tmp_path / "dst"
    fakes.write_fake_checkpoint(src)
    rtn = {n: list(skip) for n in ROUTED} if skip else None
    conv.convert(
        src,
        dst,
        "nvfp4",
        IGNORE,
        max_shard_bytes=80_000,
        override=_override(skip),
        info_extra=extra if extra is not None else _extra(rtn),
    )
    return src, dst


def _mutate(dst: Path, fn: Callable[[dict[str, torch.Tensor]], None]) -> None:
    index = json.loads((dst / "model.safetensors.index.json").read_text())
    by_file: dict[str, dict[str, torch.Tensor]] = {}
    for f in sorted(set(index["weight_map"].values())):
        with safe_open(dst / f, framework="pt") as h:
            by_file[f] = {k: h.get_tensor(k) for k in h.keys()}  # noqa: SIM118
    merged = {k: v for part in by_file.values() for k, v in part.items()}
    fn(merged)
    for f, part in by_file.items():
        save_file({k: merged[k] for k in part}, str(dst / f), metadata={"format": "pt"})


def _raw(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().reshape(-1).view(torch.uint8)


def _checks(report: verify.VerifyReport) -> set[str]:
    return {v.check for v in report.violations}


def test_convert_override_replaces_the_tensor_and_info_extra_lands_in_the_info(tmp_path: Path) -> None:
    _src, dst = _make(tmp_path)
    info = json.loads((dst / "quantization_info.json").read_text())
    assert info["calibration"]["calibrated_tensors"] == list(ROUTED)
    assert info["recipe"]["name"] == "nvfp4"


def test_a_calibrated_checkpoint_passes_the_calibrated_mode_and_fails_the_strict_one(tmp_path: Path) -> None:
    src, dst = _make(tmp_path)
    ok = verify.verify(src, dst, calibrated=True)
    assert ok.violations == []
    strict = verify.verify(src, dst)
    assert _checks(strict) == {"calibrated_checkpoint"}


def test_calibrated_codes_really_differ_from_n2_but_scales_do_not(tmp_path: Path) -> None:
    src, dst = _make(tmp_path)
    n2 = tmp_path / "n2"
    conv.convert(src, n2, "nvfp4", IGNORE, max_shard_bytes=80_000)

    def read(d: Path) -> dict[str, torch.Tensor]:
        out = {}
        for f in sorted(d.glob("model-*.safetensors")):
            with safe_open(f, framework="pt") as h:
                out.update({k: h.get_tensor(k) for k in h.keys()})  # noqa: SIM118
        return out

    a, b = read(dst), read(n2)
    assert set(a) == set(b)
    for name in ROUTED:
        base = name.removesuffix(".weight")
        assert not torch.equal(a[f"{base}.weight_packed"], b[f"{base}.weight_packed"])
        assert torch.equal(
            a[f"{base}.weight_scale"].view(torch.uint8), b[f"{base}.weight_scale"].view(torch.uint8)
        )
        assert torch.equal(a[f"{base}.weight_global_scale"], b[f"{base}.weight_global_scale"])
    untouched = [k for k in a if not k.startswith("model.layers.2.mlp.experts.")]
    assert all(torch.equal(_raw(a[k]), _raw(b[k])) for k in untouched)


def test_a_plain_n2_checkpoint_also_passes_the_calibrated_mode(tmp_path: Path) -> None:
    src, dst = tmp_path / "src", tmp_path / "n2"
    fakes.write_fake_checkpoint(src)
    conv.convert(src, dst, "nvfp4", IGNORE, max_shard_bytes=80_000)
    assert verify.verify(src, dst, calibrated=True).violations == []


def test_a_changed_block_scale_is_caught(tmp_path: Path) -> None:
    src, dst = _make(tmp_path)

    def damage(t: dict[str, torch.Tensor]) -> None:
        s = t[ROUTED[0].replace(".weight", ".weight_scale")]
        raw = s.view(torch.uint8).clone()
        raw[0, 0, 0] = raw[0, 0, 0] - 1 if raw[0, 0, 0] > 1 else raw[0, 0, 0] + 1
        t[ROUTED[0].replace(".weight", ".weight_scale")] = raw.view(torch.float8_e4m3fn)

    _mutate(dst, damage)
    assert "scale_equals_n2" in _checks(verify.verify(src, dst, calibrated=True))


def test_codes_of_a_tensor_that_is_not_calibrated_must_equal_n2(tmp_path: Path) -> None:
    src, dst = _make(tmp_path)
    name = "model.layers.2.mlp.shared_experts.fc_1.weight_packed"

    def damage(t: dict[str, torch.Tensor]) -> None:
        t[name] = _perturb(t[name], 0.2, seed=3)

    _mutate(dst, damage)
    assert "codes_equal_n2" in _checks(verify.verify(src, dst, calibrated=True))


def test_experts_declared_rtn_must_carry_the_rtn_codes(tmp_path: Path) -> None:
    src, dst = _make(tmp_path, skip=(0, 3))
    assert verify.verify(src, dst, calibrated=True).violations == []
    name = ROUTED[0].replace(".weight", ".weight_packed")

    def damage(t: dict[str, torch.Tensor]) -> None:
        t[name] = _perturb(t[name], 0.3, seed=9)  # now expert 0 and 3 differ from RTN too

    _mutate(dst, damage)
    assert "rtn_expert_codes" in _checks(verify.verify(src, dst, calibrated=True))


def test_a_reconstruction_error_far_beyond_n2_is_caught(tmp_path: Path) -> None:
    src, dst = _make(tmp_path)
    name = ROUTED[1].replace(".weight", ".weight_packed")

    def damage(t: dict[str, torch.Tensor]) -> None:
        t[name] = torch.full_like(t[name], 0x77)  # every element is +6 * scale

    _mutate(dst, damage)
    assert "reconstruction_error" in _checks(verify.verify(src, dst, calibrated=True))


def test_the_calibrated_error_factor_is_the_documented_bound() -> None:
    assert verify.Bounds().nvfp4_calibrated_factor == 2.0
    # a looser factor lets a larger error through: the bound is read from the Bounds object
    assert verify.Bounds(nvfp4_calibrated_factor=50.0).nvfp4_calibrated_factor == 50.0


def test_a_larger_factor_accepts_what_the_default_rejects(tmp_path: Path) -> None:
    src, dst = _make(tmp_path)

    def damage(t: dict[str, torch.Tensor]) -> None:
        t[ROUTED[1].replace(".weight", ".weight_packed")] = _perturb(
            t[ROUTED[1].replace(".weight", ".weight_packed")], 0.15, seed=1
        )

    _mutate(dst, damage)
    tight = verify.verify(src, dst, verify.Bounds(nvfp4_calibrated_factor=1.0), calibrated=True)
    loose = verify.verify(src, dst, verify.Bounds(nvfp4_calibrated_factor=100.0), calibrated=True)
    assert "reconstruction_error" in _checks(tight)
    assert "reconstruction_error" not in _checks(
        loose
    )  # (the recorded error in the info no longer matches: ignored)


def test_calibration_info_must_name_quantized_routed_tensors_only(tmp_path: Path) -> None:
    src, dst = _make(tmp_path, extra=_extra(tensors=(*ROUTED, "model.layers.2.mlp.down_proj.weight")))
    assert "calibration_info" in _checks(verify.verify(src, dst, calibrated=True))
    src2 = tmp_path / "src2"
    dst2 = tmp_path / "dst2"
    fakes.write_fake_checkpoint(src2)
    conv.convert(
        src2,
        dst2,
        "nvfp4",
        IGNORE,
        max_shard_bytes=80_000,
        override=_override(),
        info_extra=_extra(tensors=("model.layers.0.mlp.experts.experts.weight",)),
    )
    assert "calibration_info" in _checks(verify.verify(src2, dst2, calibrated=True))


def test_cli_calibrated_flag(tmp_path: Path) -> None:
    src, dst = _make(tmp_path)
    assert verify.main(["--src", str(src), "--dst", str(dst)]) == 1
    assert verify.main(["--src", str(src), "--dst", str(dst), "--calibrated"]) == 0


def test_the_strict_mode_is_unchanged_for_a_data_free_checkpoint(tmp_path: Path) -> None:
    src, dst = tmp_path / "src", tmp_path / "n2"
    fakes.write_fake_checkpoint(src)
    conv.convert(src, dst, "nvfp4", IGNORE, max_shard_bytes=80_000)
    assert verify.verify(src, dst).violations == []
