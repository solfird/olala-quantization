"""Tests for olala_quant.weight_stats (why is this model hard to quantize at the weight level).

Conventions under test, all stated in the module docstring:

* statistics have known answers on constructed tensors (two-point +-1 distribution: excess kurtosis exactly
  -2; one planted outlier: max-abs / RMS and the fraction above 6 RMS exact; Gaussian: kurtosis near 0);
* NVFP4 and FP8 error are the repo's own quantizers, cross-checked against ``convert.quantize_tensor`` and an
  independent dequantize-and-compare, tolerance 1e-6 relative (float32 dequantization against float64 norms);
* INT4 (symmetric, group 128, fp16 scale, compressed-tensors convention) and MXFP4 (OCP MX v1.0, floor rule)
  are checked against hand-computed answers and against an independent numpy re-implementation.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from olala_quant import convert, fp8, nvfp4, recipes
from olala_quant import weight_stats as ws
from olala_quant.checkpoint_io import header_from_json

DATA = Path(__file__).parent / "data" / "olala_header.json"
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


# --------------------------------------------------------------------------- classification


def _real_header() -> dict:
    return header_from_json(json.loads(DATA.read_text()))


def test_classification_of_the_real_header() -> None:
    header = _real_header()
    att = ws.attention_layers(header)
    assert att == {4, 9, 14, 19, 24, 29, 34}
    counts: dict[str, int] = {}
    params: dict[str, int] = {}
    layers: dict[str, set[int | None]] = {}
    skipped = 0
    for name, meta in header.items():
        wc = ws.classify_weight(name, att)
        if wc is None:
            skipped += 1
            continue
        counts[wc.cls] = counts.get(wc.cls, 0) + 1
        params[wc.cls] = params.get(wc.cls, 0) + meta.numel
        layers.setdefault(wc.cls, set()).add(wc.layer)
    assert counts == {
        "routed_up": 36,
        "routed_down": 36,
        "shared_expert": 72,
        "latent_proj": 72,
        "attn_o": 7,
        "mamba_out_proj": 29,
        "attn_gate": 7,
        "attn_q": 7,
        "attn_kv": 14,
        "attn_small": 35,
        "mamba_in_proj": 29,
        "mamba_other": 29,
        "router": 36,
        "embedding": 1,
        "lm_head": 1,
    }
    assert skipped == len(header) - 411
    # recon/quantizability.md A.2: routed experts hold 5,435,817,984 parameters
    assert params["routed_up"] + params["routed_down"] == 5_435_817_984
    assert params["embedding"] == params["lm_head"] == 233_373_696
    assert layers["embedding"] == {None} and layers["lm_head"] == {None}
    assert layers["attn_o"] == att
    assert layers["mamba_out_proj"] == set(range(36)) - att


def test_classes_follow_the_names_plan_uses() -> None:
    att = {4}
    assert ws.classify_weight("model.layers.3.mlp.experts.experts.weight", att) == ws.WeightClass(
        "routed_up", 3
    )
    assert ws.classify_weight("model.layers.3.mlp.experts.output_experts.weight", att) == ws.WeightClass(
        "routed_down", 3
    )
    assert ws.classify_weight("model.layers.4.mixer_proj.weight", att) == ws.WeightClass("attn_o", 4)
    assert ws.classify_weight("model.layers.5.mixer_proj.weight", att) == ws.WeightClass("mamba_out_proj", 5)
    assert ws.classify_weight("model.layers.5.mlp.moe_gate.weight", att) == ws.WeightClass("router", 5)
    # not weight matrices: biases, norms, scalars, unknown names
    for name in (
        "model.layers.5.mlp.expert_bias",
        "model.layers.5.mixer.D",
        "model.layers.5.mixer.B_bias",
        "model.layers.5.mixer.B_norm.norm.weight",
        "model.layers.5.geodesic_mlp.scale",
        "model.layers.4.mixer.softmax_scaler",
        "something.else.weight",
    ):
        assert ws.classify_weight(name, att) is None, name
    assert "expert_bias" in ws.skip_reason("model.layers.5.mlp.expert_bias")
    assert "unclassified" in ws.skip_reason("something.else.weight")


# --------------------------------------------------------------------------- distribution statistics


def test_gaussian_has_zero_excess_kurtosis() -> None:
    g = torch.Generator().manual_seed(0)
    w = torch.randn(2048, 1024, generator=g)
    pooled, units = ws.compute_distribution([w])
    assert len(units) == 1 and units[0] == pooled
    assert abs(pooled["excess_kurtosis"]) < 0.05
    assert pooled["rms"] == pytest.approx(1.0, abs=0.01)
    assert 4.5 < pooled["max_over_rms"] < 6.5
    assert pooled["frac_gt_6rms"] < 1e-5
    assert pooled["max_abs"] == pytest.approx(float(w.abs().max()))


def test_two_point_distribution_has_kurtosis_minus_two_exactly() -> None:
    w = torch.tensor([-1.0, 1.0]).repeat(2048).reshape(64, 64)  # balanced: mean exactly 0
    pooled, _ = ws.compute_distribution([w])
    assert pooled["excess_kurtosis"] == pytest.approx(-2.0, abs=1e-12)
    assert pooled["rms"] == pytest.approx(1.0, abs=1e-12)
    assert pooled["max_over_rms"] == pytest.approx(1.0, abs=1e-12)
    assert pooled["frac_gt_6rms"] == 0.0
    assert pooled["channel_rms_range"] == pytest.approx(1.0, abs=1e-12)


def test_planted_outlier_ratio_is_exact() -> None:
    n = 1000
    w = torch.ones(10, 100)
    w[3, 7] = -100.0
    pooled, _ = ws.compute_distribution([w])
    rms = math.sqrt(((n - 1) * 1.0 + 100.0**2) / n)
    assert pooled["rms"] == pytest.approx(rms, rel=1e-12)
    assert pooled["max_abs"] == 100.0
    assert pooled["max_over_rms"] == pytest.approx(100.0 / rms, rel=1e-12)
    assert pooled["frac_gt_6rms"] == pytest.approx(1 / n, rel=1e-12)
    ref = w.double().numpy().ravel()
    m = ref.mean()
    expected = ((ref - m) ** 4).mean() / ((ref - m) ** 2).mean() ** 2 - 3.0
    assert pooled["excess_kurtosis"] == pytest.approx(expected, rel=1e-9)
    assert pooled["excess_kurtosis"] > 900  # one spike: kurtosis near n - 3


def test_outlier_fraction_uses_six_rms_not_five() -> None:
    torch.manual_seed(11)
    w = torch.distributions.StudentT(3.0).sample((200, 100)).float()  # heavy tails: many in 5..6 RMS
    ref = w.double().numpy().ravel()
    rms = float(np.sqrt((ref**2).mean()))
    frac6, frac5 = float((np.abs(ref) > 6 * rms).mean()), float((np.abs(ref) > 5 * rms).mean())
    assert frac5 > frac6 > 0  # the band between 5 and 6 RMS is not empty
    assert ws.compute_distribution([w])[0]["frac_gt_6rms"] == pytest.approx(frac6, abs=1e-15)


def test_channel_rms_range_is_max_over_min_row_rms() -> None:
    w = torch.ones(4, 64)
    w[1] *= 4.0
    w[2, ::2] *= -1.0
    w[3] *= 2.0
    pooled, _ = ws.compute_distribution([w])
    assert pooled["channel_rms_range"] == pytest.approx(4.0, rel=1e-12)
    w[0] = 0.0
    assert ws.compute_distribution([w])[0]["channel_rms_range"] is None  # min RMS is zero: undefined


def test_group_amax_max_over_median() -> None:
    w = torch.full((1, 64), 0.1)
    for j, a in enumerate((1.0, 2.0, 4.0, 8.0)):  # one spike per group of 16
        w[0, 16 * j] = a
    pooled, _ = ws.compute_distribution([w])
    # g16 group maxima [1, 2, 4, 8], median 3; g32 group maxima [2, 8], median 5 (true median, not lower)
    assert pooled["g16_amax_max_over_median"] == pytest.approx(8.0 / 3.0, rel=1e-12)
    assert pooled["g32_amax_max_over_median"] == pytest.approx(8.0 / 5.0, rel=1e-12)


def test_group_ratio_undefined_when_width_not_divisible() -> None:
    pooled, _ = ws.compute_distribution([torch.randn(4, 20)])
    assert pooled["g16_amax_max_over_median"] is None
    assert pooled["g32_amax_max_over_median"] is None


def test_per_unit_and_pooled_stats_of_a_fused_expert_tensor() -> None:
    g = torch.Generator().manual_seed(2)
    e0 = torch.randn(32, 64, generator=g)
    e1 = 10.0 * torch.randn(32, 64, generator=g)
    pooled, units = ws.compute_distribution([e0, e1])
    assert len(units) == 2
    assert units[0]["rms"] == pytest.approx(float(e0.double().pow(2).mean().sqrt()), rel=1e-12)
    assert units[1]["rms"] == pytest.approx(float(e1.double().pow(2).mean().sqrt()), rel=1e-12)
    assert abs(units[0]["excess_kurtosis"]) < 0.5 and abs(units[1]["excess_kurtosis"]) < 0.5
    both = torch.cat([e0, e1]).double().numpy().ravel()
    m = both.mean()
    expected = ((both - m) ** 4).mean() / ((both - m) ** 2).mean() ** 2 - 3.0
    assert pooled["excess_kurtosis"] == pytest.approx(expected, rel=1e-9)
    assert pooled["excess_kurtosis"] > 0.5  # a mixture of two scales is heavy tailed
    assert pooled["rms"] == pytest.approx(float(np.sqrt((both**2).mean())), rel=1e-12)
    assert pooled["max_abs"] == float(max(e0.abs().max(), e1.abs().max()))


def test_distribution_does_not_depend_on_chunking(monkeypatch: pytest.MonkeyPatch) -> None:
    g = torch.Generator().manual_seed(3)
    units = [torch.randn(40, 64, generator=g) * (1 + i) for i in range(3)]
    ref = ws.compute_distribution(units)
    monkeypatch.setattr(ws, "_STAT_CHUNK", 200)
    small = ws.compute_distribution(units)
    for a, b in zip([ref[0], *ref[1]], [small[0], *small[1]], strict=True):
        for k, v in a.items():
            assert b[k] == pytest.approx(v, rel=1e-9), k


def test_non_finite_weights_are_refused() -> None:
    w = torch.randn(4, 32)
    w[1, 1] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        ws.compute_distribution([w])


# --------------------------------------------------------------------------- INT4 g128 (fp16 scale)


def test_int4_constants() -> None:
    assert ws.INT4_GROUP == 128
    assert (ws.INT4_QMIN, ws.INT4_QMAX) == (-8, 7)
    assert ws.INT4_SCALE_DIVISOR == 7.5


def test_int4_known_answer() -> None:
    # amax 7.5 -> scale exactly 1; integers in [-7, 7] are exact; +7.5 rounds to 8, clamps to 7 (error 0.5),
    # -7.5 rounds to -8 (error 0.5)
    x = torch.zeros(1, 128)
    x[0, :15] = torch.arange(-7, 8, dtype=torch.float32)
    x[0, 15] = 7.5
    x[0, 16] = -7.5
    deq, counts = ws.int4_g128_roundtrip(x)
    err = (deq - x).double()
    assert float(err[0, 15]) == -0.5 and float(err[0, 16]) == -0.5
    assert float(err.pow(2).sum()) == pytest.approx(0.5, abs=1e-12)
    assert counts == {"underflow": 0, "flushed": 0, "saturate": 0}


def test_int4_scale_underflow_flush_and_saturation_counts() -> None:
    x = torch.zeros(4, 128)
    x[0, 0] = 1.0  # normal
    x[1, 0] = 1e-7  # scale 1.3e-8 rounds to 0 in fp16: flushed (and underflow)
    x[2, 0] = 3e-4  # scale 4e-5: below fp16 min normal 6.1e-5, still nonzero: underflow only
    x[3, 0] = 1e6  # scale 1.3e5 > 65504: saturate
    deq, counts = ws.int4_g128_roundtrip(x)
    assert counts == {"underflow": 2 * 128, "flushed": 128, "saturate": 128}
    assert torch.isfinite(deq).all()
    assert float(deq[1].abs().sum()) == 0.0  # flushed group dequantizes to zero


def test_int4_matches_independent_numpy_reimplementation() -> None:
    g = torch.Generator().manual_seed(4)
    x = torch.randn(8, 256, generator=g) * torch.rand(8, 1, generator=g) * 0.1
    deq, _ = ws.int4_g128_roundtrip(x)
    ref = np.empty(x.shape, dtype=np.float64)
    xn = x.numpy().astype(np.float64)
    for r in range(8):
        for c0 in range(0, 256, 128):
            blk = xn[r, c0 : c0 + 128]
            scale = float(np.float16(np.abs(blk).max() / 7.5))
            q = np.clip(np.round(blk.astype(np.float32) / np.float32(scale)), -8, 7)
            ref[r, c0 : c0 + 128] = q * scale
    assert np.allclose(deq.double().numpy(), ref, rtol=1e-6, atol=1e-9)


# --------------------------------------------------------------------------- MXFP4 (g32, e8m0)


def test_mxfp4_constants() -> None:
    assert ws.MXFP4_GROUP == 32
    assert ws.E2M1_EMAX == 2
    assert (ws.E8M0_MIN_EXP, ws.E8M0_MAX_EXP) == (-127, 127)


def test_mxfp4_known_answers_for_the_floor_rule() -> None:
    # amax 6: floor(log2 6) = 2, exponent 2 - 2 = 0, scale 1, 6 is representable: exact
    x = torch.zeros(3, 32)
    x[0, :2] = torch.tensor([6.0, -3.0])
    # amax 8: exponent 3 - 2 = 1, scale 2, 8 / 2 = 4 representable: exact
    x[1, :2] = torch.tensor([8.0, 1.0])
    # amax 7: exponent 0, scale 1, 7 saturates to 6 (the OCP floor rule clips): error 1
    x[2, 0] = 7.0
    deq, counts = ws.mxfp4_roundtrip(x)
    assert torch.equal(deq[0], x[0])
    assert float(deq[1, 0]) == 8.0 and float(deq[1, 1]) == 1.0  # 1 / 2 = 0.5 is representable
    assert float(deq[2, 0]) == 6.0
    assert counts == {"underflow": 0, "flushed": 0, "saturate": 0}


def test_mxfp4_rule_uses_exact_frexp_not_log2() -> None:
    # 2**k - tiny: floor(log2) must be k - 1 even where float32 log2 would round up to k
    amax = torch.tensor([np.nextafter(np.float32(8.0), np.float32(0.0))], dtype=torch.float32)
    exp, _, _ = ws.mxfp4_exponent(amax)
    assert int(exp[0]) == 0  # floor(log2(7.9999995)) = 2, minus emax 2
    exp, _, _ = ws.mxfp4_exponent(torch.tensor([8.0]))
    assert int(exp[0]) == 1


def test_mxfp4_exponent_clamp_flags() -> None:
    amax = torch.tensor([2.0**-130, 2.0**-125, 2.0**130, 1.0], dtype=torch.float64)
    exp, under, sat = ws.mxfp4_exponent(amax)
    assert exp.tolist() == [-127, -127, 127, -2]  # 2^-125 -> exp -127: representable, not clamped
    assert under.tolist() == [True, False, False, False]
    assert sat.tolist() == [False, False, True, False]


def test_mxfp4_underflow_counts_in_roundtrip() -> None:
    x = torch.zeros(2, 32)
    x[0, 0] = 2.0**-130
    x[1, 0] = 1.0
    deq, counts = ws.mxfp4_roundtrip(x)
    assert counts == {"underflow": 32, "flushed": 0, "saturate": 0}
    assert torch.isfinite(deq).all()


def test_mxfp4_matches_independent_numpy_reimplementation() -> None:
    g = torch.Generator().manual_seed(5)
    x = torch.randn(6, 96, generator=g) * torch.rand(6, 1, generator=g)
    deq, _ = ws.mxfp4_roundtrip(x)
    xn = x.numpy().astype(np.float64)
    ref = np.empty_like(xn)
    for r in range(6):
        for c0 in range(0, 96, 32):
            blk = xn[r, c0 : c0 + 32]
            scale = 2.0 ** (np.floor(np.log2(np.abs(blk).max())) - 2)
            v = np.clip(np.abs(blk) / scale, 0, 6.0)
            nearest = E2M1[np.abs(v[:, None] - E2M1[None, :]).argmin(axis=1)]
            ref[r, c0 : c0 + 32] = np.sign(blk) * nearest * scale
    assert np.allclose(deq.double().numpy(), ref, rtol=1e-6, atol=1e-12)


# --------------------------------------------------------------------------- per-format errors on a tensor


def _as_recipe_record(name: str, w: torch.Tensor, recipe_name: str) -> dict:
    return convert.quantize_tensor(name, w, recipes.RECIPES[recipe_name])[1]


def _independent_rel_fro(w: torch.Tensor, deq: torch.Tensor) -> float:
    w64, d64 = w.double(), deq.double()
    return float(((w64 - d64).pow(2).sum() / w64.pow(2).sum()).sqrt())


def test_nvfp4_and_fp8_error_match_olala_quant_conversion_2d() -> None:
    g = torch.Generator().manual_seed(6)
    w = (torch.randn(48, 128, generator=g) * 0.03).to(torch.bfloat16)
    w[5, 9] = 0.7  # one outlier so scales are not trivial
    row = ws.measure_tensor("model.layers.0.mixer_proj.weight", w, "mamba_out_proj", 0, device="cpu")
    for fmt, recipe in (("nvfp4", "nvfp4"), ("fp8", "fp8")):
        got = row["formats"][fmt]["rel_fro"]
        rec = _as_recipe_record("model.layers.0.mixer_proj.weight", w, recipe)
        assert got == pytest.approx(rec["rel_fro_error"], rel=1e-9)  # same code path
    outs, _ = convert.quantize_tensor("m.weight", w, recipes.RECIPES["nvfp4"])
    deq = nvfp4.dequantize_nvfp4(
        outs["m.weight_packed"], outs["m.weight_scale"], outs["m.weight_global_scale"]
    )
    assert row["formats"]["nvfp4"]["rel_fro"] == pytest.approx(_independent_rel_fro(w, deq), rel=1e-6)
    outs, _ = convert.quantize_tensor("m.weight", w, recipes.RECIPES["fp8"])
    deq = fp8.dequantize_fp8(outs["m.weight"], outs["m.weight_scale"])
    assert row["formats"]["fp8"]["rel_fro"] == pytest.approx(_independent_rel_fro(w, deq), rel=1e-6)


def test_fused_expert_error_is_pooled_over_per_expert_quantization() -> None:
    g = torch.Generator().manual_seed(7)
    spread = torch.tensor([0.2, 0.5, 1.0, 2.0, 3.0, 4.0]).reshape(6, 1, 1)
    w = (torch.randn(6, 32, 128, generator=g) * 0.03 * spread).to(torch.bfloat16)
    row = ws.measure_tensor("model.layers.2.mlp.experts.experts.weight", w, "routed_up", 2, device="cpu")
    for fmt, recipe in (("nvfp4", "nvfp4"), ("fp8", "fp8")):
        rec = _as_recipe_record("e.weight", w, recipe)  # one call on the whole 3D tensor
        assert row["formats"][fmt]["rel_fro"] == pytest.approx(rec["rel_fro_error"], rel=1e-9)
    pe = row["per_expert"]
    assert set(pe["rel_fro"]) == set(ws.FORMATS)
    for q in pe["rel_fro"].values():
        assert set(q) == {"min", "p05", "p50", "p95", "max"}
        assert q["min"] <= q["p05"] <= q["p50"] <= q["p95"] <= q["max"]
    # per-expert rel_fro of expert e equals a 1-expert call on that expert
    solo = [fp8.quantize_fp8(w[e : e + 1]).error.rel_fro for e in range(6)]
    assert pe["rel_fro"]["fp8"]["min"] == pytest.approx(min(solo), rel=1e-9)
    assert pe["rel_fro"]["fp8"]["max"] == pytest.approx(max(solo), rel=1e-9)
    assert pe["rms"]["max"] / pe["rms"]["min"] > 3  # the planted per-expert spread is visible
    for key in ("excess_kurtosis", "max_over_rms", "frac_gt_6rms", "channel_rms_range"):
        assert key in pe


def test_2d_tensor_has_no_per_expert_block_and_fp_fields() -> None:
    w = (torch.randn(16, 128) * 0.02).to(torch.bfloat16)
    row = ws.measure_tensor("lm_head.weight", w, "lm_head", None, device="cpu")
    assert row["per_expert"] is None
    assert row["shape"] == [16, 128] and row["numel"] == 16 * 128 and row["layer"] is None
    assert row["class"] == "lm_head"
    assert set(row["formats"]) == set(ws.FORMATS)
    for f in row["formats"].values():
        assert set(f) >= {"rel_fro", "frac_scale_underflow", "frac_scale_flushed", "frac_scale_saturate"}
        assert 0 < f["rel_fro"] < 1


def test_format_not_applicable_when_width_does_not_divide_the_group() -> None:
    w = (torch.randn(8, 64) * 0.02).to(torch.bfloat16)  # 64 % 128 != 0 -> INT4 g128 not applicable
    row = ws.measure_tensor("x.weight", w, "router", 0, device="cpu")
    assert row["formats"]["int4_g128"] is None
    assert "int4_g128" in row["not_applicable"]
    assert row["formats"]["mxfp4"] is not None and row["formats"]["nvfp4"] is not None


def test_nvfp4_scale_underflow_fraction_counts_blocks_of_16() -> None:
    # one huge block drives the global scale (448 * 6 / 100): the other blocks get a scale of
    # 26.88 * 1e-5 / 6 = 4.5e-5, below half the smallest e4m3 subnormal (2^-10): flushed to zero
    w = torch.full((1, 64), 1e-5)
    w[0, :16] = 1e2
    row = ws.measure_tensor("x.weight", w, "router", 0, device="cpu")
    f = row["formats"]["nvfp4"]
    assert f["frac_scale_flushed"] == pytest.approx(48 / 64)
    assert f["frac_scale_underflow"] >= f["frac_scale_flushed"]
    assert f["frac_scale_saturate"] == 0.0
    assert f["rel_fro"] == pytest.approx(math.sqrt(48 * 1e-10 / (16 * 1e4 + 48 * 1e-10)), rel=0.01)


def test_nvfp4_subnormal_block_scale_counts_as_underflow_but_not_flushed() -> None:
    # global scale 448 * 6 / 100; small blocks get scale 26.88 * 1e-3 / 6 = 4.5e-3: a nonzero e4m3 subnormal
    # (multiples of 2^-9 below 2^-6), so the block survives but with a coarse scale
    w = torch.full((1, 64), 1e-3)
    w[0, :16] = 1e2
    f = ws.measure_tensor("x.weight", w, "router", 0, device="cpu")["formats"]["nvfp4"]
    assert f["frac_scale_flushed"] == 0.0
    assert f["frac_scale_underflow"] == pytest.approx(48 / 64)


def test_fp8_row_floor_counts_as_underflow() -> None:
    w = torch.zeros(4, 128)
    w[0, 0] = 1.0
    w[1, 0] = 1e-3  # amax / 448 = 2.2e-6 < 1 / (448 * 512) = 4.4e-6 : clamped to the floor
    # rows 2 and 3 stay all zero: not counted
    row = ws.measure_tensor("x.weight", w, "router", 0, device="cpu")
    f = row["formats"]["fp8"]
    assert f["frac_scale_underflow"] == pytest.approx(128 / (4 * 128))
    assert f["frac_scale_flushed"] == 0.0 and f["frac_scale_saturate"] == 0.0


# --------------------------------------------------------------------------- summary


def _row(name: str, cls: str, layer: int | None, numel: int, errs: dict[str, float | None]) -> dict:
    return {
        "name": name,
        "class": cls,
        "layer": layer,
        "numel": numel,
        "shape": [numel],
        "formats": {
            f: None
            if errs.get(f) is None
            else {
                "rel_fro": errs[f],
                "frac_scale_underflow": 0.0,
                "frac_scale_flushed": 0.0,
                "frac_scale_saturate": 0.0,
            }
            for f in ws.FORMATS
        },
        "stats": {
            "excess_kurtosis": 1.0,
            "max_over_rms": 10.0,
            "frac_gt_6rms": 0.0,
            "channel_rms_range": 2.0,
            "g16_amax_max_over_median": 3.0,
            "g32_amax_max_over_median": 3.0,
        },
    }


def test_summary_is_parameter_weighted_with_worst_layer_and_ratio_to_best() -> None:
    rows = [
        _row("a", "attn_q", 4, 100, {"nvfp4": 0.10, "fp8": 0.02, "int4_g128": 0.12, "mxfp4": 0.20}),
        _row("b", "attn_q", 9, 300, {"nvfp4": 0.20, "fp8": 0.04, "int4_g128": 0.14, "mxfp4": 0.30}),
        _row("c", "attn_q", 14, 100, {"nvfp4": 0.15, "fp8": 0.06, "int4_g128": 0.10, "mxfp4": 0.25}),
    ]
    s = {(r["class"], r["format"]): r for r in ws.summarize(rows)}
    nv = s[("attn_q", "nvfp4")]
    assert nv["n_tensors"] == 3 and nv["params"] == 500
    assert nv["wmean_rel_fro"] == pytest.approx((100 * 0.10 + 300 * 0.20 + 100 * 0.15) / 500)
    assert nv["worst_layer"] == 9 and nv["worst_rel_fro"] == 0.20
    f8 = s[("attn_q", "fp8")]
    assert f8["wmean_rel_fro"] == pytest.approx((100 * 0.02 + 300 * 0.04 + 100 * 0.06) / 500)
    assert f8["ratio_to_best"] == pytest.approx(1.0)
    assert nv["ratio_to_best"] == pytest.approx(nv["wmean_rel_fro"] / f8["wmean_rel_fro"])
    assert s[("attn_q", "mxfp4")]["ratio_to_best"] > nv["ratio_to_best"] > 1.0


def test_summary_skips_not_applicable_formats() -> None:
    rows = [_row("a", "router", 0, 50, {"nvfp4": 0.1, "fp8": 0.05, "int4_g128": None, "mxfp4": 0.2})]
    out = {(r["class"], r["format"]): r for r in ws.summarize(rows)}
    assert ("router", "int4_g128") not in out
    assert out[("router", "fp8")]["ratio_to_best"] == 1.0


# --------------------------------------------------------------------------- json and layer filter


def test_json_clean_replaces_non_finite_numbers() -> None:
    cleaned = ws.json_clean({"a": float("inf"), "b": [float("nan"), 1.0, {"c": -math.inf}], "d": 3})
    assert cleaned == {"a": None, "b": [None, 1.0, {"c": None}], "d": 3}
    json.dumps(cleaned, allow_nan=False)


def test_parse_layers() -> None:
    assert ws.parse_layers("2:5") == (2, 5)
    assert ws.parse_layers("3:") == (3, None)
    assert ws.parse_layers(":4") == (None, 4)
    for bad in ("5", "a:b", "5:2", "1:2:3", ""):
        with pytest.raises(ValueError):
            ws.parse_layers(bad)
    assert ws.layer_selected(2, (2, 5)) and ws.layer_selected(4, (2, 5))
    assert not ws.layer_selected(5, (2, 5)) and not ws.layer_selected(1, (2, 5))
    assert not ws.layer_selected(None, (0, 36))  # embedding and lm_head are dropped under a layer filter
    assert ws.layer_selected(None, None)


# --------------------------------------------------------------------------- end to end


def _write_checkpoint(directory: Path) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(8)

    def bf(*shape: int) -> torch.Tensor:
        return (torch.randn(shape, generator=g) * 0.05).to(torch.bfloat16)

    t: dict[str, torch.Tensor] = {
        "model.embedding.weight": bf(128, 128),
        "lm_head.weight": bf(128, 128),
    }
    for i, kind in enumerate("MV"):
        p = f"model.layers.{i}"
        t[f"{p}.geodesic_mixer.scale"] = torch.tensor(1.0)
        t[f"{p}.mixer_proj.weight"] = bf(128, 256 if kind == "M" else 128)
        if kind == "M":
            t[f"{p}.mixer.in_proj.weight"] = bf(128, 128)
            t[f"{p}.mixer.in_proj_dyn.weight"] = bf(128, 128)
            t[f"{p}.mixer.D"] = torch.randn(4, generator=g)
        else:
            t[f"{p}.mixer.c_q.weight"] = bf(128, 128)
            t[f"{p}.mixer.W_B_k.weight"] = bf(128, 128)
            t[f"{p}.mixer.W_A_k.weight"] = bf(128, 128)
            t[f"{p}.gate_proj.weight"] = bf(128, 128)
        t[f"{p}.mlp.experts.experts.weight"] = bf(5, 128, 128)
        t[f"{p}.mlp.experts.output_experts.weight"] = bf(5, 128, 128)
        t[f"{p}.mlp.shared_experts.fc_1.weight"] = bf(128, 128)
        t[f"{p}.mlp.shared_experts.fc_2.weight"] = bf(128, 128)
        t[f"{p}.mlp.down_proj.weight"] = bf(128, 128)
        t[f"{p}.mlp.up_proj.weight"] = bf(128, 128)
        t[f"{p}.mlp.moe_gate.weight"] = torch.randn(5, 128, generator=g)
        t[f"{p}.mlp.expert_bias"] = torch.randn(5, generator=g)
    directory.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    for shard, pick in (
        ("part-0.safetensors", lambda n: n.startswith("model.layers.0.") or n == "model.embedding.weight"),
        (
            "part-1.safetensors",
            lambda n: not (n.startswith("model.layers.0.") or n == "model.embedding.weight"),
        ),
    ):
        part = {n: v for n, v in t.items() if pick(n)}
        save_file(part, directory / shard, metadata={"format": "pt"})
        weight_map.update({n: shard for n in part})
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}, indent=1)
    )
    return t


def _expected_measured(t: dict[str, torch.Tensor]) -> set[str]:
    att = ws.attention_layers(dict.fromkeys(t))
    return {n for n in t if ws.classify_weight(n, att) is not None}


def _load_strict(path: Path) -> dict:
    def refuse(c: str) -> None:
        raise AssertionError(f"non-standard JSON constant {c}")

    return json.loads(path.read_text(), parse_constant=refuse)


def test_end_to_end_writes_json_and_markdown_with_index_hash(tmp_path: Path) -> None:
    src, out = tmp_path / "src", tmp_path / "out"
    t = _write_checkpoint(src)
    assert ws.main(["--src", str(src), "--out", str(out), "--device", "cpu"]) == 0
    doc = _load_strict(out / "weight_stats.json")
    md = (out / "weight_stats.md").read_text()
    digest = hashlib.sha256((src / "model.safetensors.index.json").read_bytes()).hexdigest()
    assert doc["checkpoint"]["sha256"] == digest and doc["checkpoint"]["kind"] == "index_json"
    assert digest in "\n".join(md.splitlines()[:12])  # in the header, before any table
    assert {r["name"] for r in doc["rows"]} == _expected_measured(t)
    assert {s["name"] for s in doc["skipped"]} == set(t) - _expected_measured(t)
    by = {r["name"]: r for r in doc["rows"]}
    routed = by["model.layers.1.mlp.experts.experts.weight"]
    assert routed["class"] == "routed_up" and routed["layer"] == 1 and routed["shape"] == [5, 128, 128]
    assert routed["numel"] == 5 * 128 * 128
    assert set(routed["per_expert"]["rel_fro"]) == set(ws.FORMATS)
    assert by["model.layers.0.mixer_proj.weight"]["class"] == "mamba_out_proj"
    assert by["model.layers.1.mixer_proj.weight"]["class"] == "attn_o"
    assert by["lm_head.weight"]["layer"] is None
    for r in doc["rows"]:
        assert set(r["stats"]) >= {
            "rms",
            "max_abs",
            "max_over_rms",
            "excess_kurtosis",
            "frac_gt_6rms",
            "channel_rms_range",
            "g16_amax_max_over_median",
            "g32_amax_max_over_median",
        }
        assert all(r["formats"][f] is not None for f in ws.FORMATS)
    assert doc["scope"] == {"layers": None, "max_tensors": None, "partial": False}
    for cls in {r["class"] for r in doc["rows"]}:
        assert f"`{cls}`" in md
    assert "NVFP4" in md and "MXFP4" in md and "INT4" in md and "FP8" in md
    assert "partial" not in md.lower().split("## ")[0]


def test_layer_filter_and_max_tensors_mark_the_run_partial(tmp_path: Path) -> None:
    src, out = tmp_path / "src", tmp_path / "out"
    _write_checkpoint(src)
    assert ws.main(["--src", str(src), "--out", str(out), "--device", "cpu", "--layers", "1:2"]) == 0
    doc = _load_strict(out / "weight_stats.json")
    assert {r["layer"] for r in doc["rows"]} == {1}  # no embedding, no layer 0
    assert doc["scope"]["partial"] is True and doc["scope"]["layers"] == [1, 2]
    assert "PARTIAL" in (out / "weight_stats.md").read_text()
    out2 = tmp_path / "out2"
    assert ws.main(["--src", str(src), "--out", str(out2), "--device", "cpu", "--max-tensors", "3"]) == 0
    doc2 = _load_strict(out2 / "weight_stats.json")
    assert len(doc2["rows"]) == 3 and doc2["scope"]["max_tensors"] == 3 and doc2["scope"]["partial"]


def test_single_file_checkpoint_hashes_the_safetensors_header(tmp_path: Path) -> None:
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    save_file({"lm_head.weight": (torch.randn(16, 128) * 0.02).to(torch.bfloat16)}, src / "model.safetensors")
    assert ws.main(["--src", str(src), "--out", str(out), "--device", "cpu"]) == 0
    doc = _load_strict(out / "weight_stats.json")
    assert doc["checkpoint"]["kind"] == "safetensors_header" and len(doc["checkpoint"]["sha256"]) == 64


def test_cuda_request_without_cuda_fails_cleanly(tmp_path: Path) -> None:
    if torch.cuda.is_available():
        pytest.skip("cuda present")
    src = tmp_path / "src"
    _write_checkpoint(src)
    with pytest.raises(SystemExit):
        ws.main(["--src", str(src), "--out", str(tmp_path / "o"), "--device", "cuda"])


def test_markdown_tables_report_ratio_to_best_and_worst_layer(tmp_path: Path) -> None:
    src, out = tmp_path / "src", tmp_path / "out"
    _write_checkpoint(src)
    ws.main(["--src", str(src), "--out", str(out), "--device", "cpu"])
    md = (out / "weight_stats.md").read_text()
    assert "ratio to best" in md.lower() and "worst layer" in md.lower()
    assert "1.00x" in md  # the best format of every class is exactly 1.00x
