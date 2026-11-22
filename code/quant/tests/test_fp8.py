"""Tests for olala_quant.fp8 (per-output-channel FP8 e4m3 weight quantization).

Format source: vLLM online PTPC path, online/fp8.py:86-106 (scale = amax/448 clamped
below at 1/(448*512), true division, clamp to +-448, cast to e4m3fn).
"""

import pytest
import torch

from olala_quant import fp8

FP8_MAX = 448.0


def _all_finite_e4m3() -> torch.Tensor:
    bytes_ = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    return bytes_[torch.isfinite(bytes_)]


def test_constants_match_vllm_online_path() -> None:
    assert fp8.FP8_MAX == FP8_MAX
    assert pytest.approx(1.0 / (448.0 * 512.0)) == fp8.MIN_SCALE


def test_representable_values_round_trip_exactly() -> None:
    # A row whose amax is exactly 448 gets scale 1.0, so every e4m3 value maps to itself.
    values = _all_finite_e4m3()
    row = values.unique()
    assert row.abs().max() == FP8_MAX
    w = row.unsqueeze(0).repeat(3, 1)
    result = fp8.quantize_fp8(w)
    assert result.scale.shape == (3, 1)
    assert torch.all(result.scale == 1.0)
    deq = fp8.dequantize_fp8(result.weight, result.scale)
    assert torch.equal(deq, w)
    assert result.error.max_abs == 0.0
    assert result.error.rel_fro == 0.0


def test_scale_is_amax_over_448_per_row_and_shapes() -> None:
    torch.manual_seed(0)
    w = torch.randn(8, 64)
    result = fp8.quantize_fp8(w)
    assert result.weight.dtype == torch.float8_e4m3fn
    assert result.weight.shape == w.shape
    assert result.scale.dtype == torch.float32
    assert result.scale.shape == (8, 1)
    expected = w.abs().amax(dim=1, keepdim=True) / FP8_MAX
    assert torch.allclose(result.scale, expected)
    # Largest element of each row lands on +-448 (saturating cast keeps it there).
    assert torch.all(result.weight.float().abs().amax(dim=1) == FP8_MAX)


def test_zero_row_gets_min_scale_and_zero_weights() -> None:
    w = torch.zeros(2, 32)
    w[1, 3] = 1.0
    result = fp8.quantize_fp8(w)
    assert result.scale[0, 0].item() == pytest.approx(fp8.MIN_SCALE)
    assert torch.all(result.weight[0].float() == 0.0)
    assert result.scale[1, 0].item() == pytest.approx(1.0 / FP8_MAX)


def test_error_bound_on_random_tensor_against_slow_reference() -> None:
    torch.manual_seed(1)
    w = (torch.randn(16, 128) * 0.02).to(torch.bfloat16)
    result = fp8.quantize_fp8(w)
    deq = fp8.dequantize_fp8(result.weight, result.scale)
    ref = _slow_reference_fp8(w.float())
    assert torch.equal(deq, ref)
    # e4m3 has 3 mantissa bits: relative step 2^-3, so |err| <= 2^-4 |x| for normal values.
    err = (deq - w.float()).abs()
    assert result.error.max_abs == pytest.approx(err.max().item(), rel=1e-6)
    rel = err.sum().item() / w.float().abs().sum().item()
    assert rel < 0.04
    assert 0.0 < result.error.rel_fro < 0.04


def _slow_reference_fp8(w: torch.Tensor) -> torch.Tensor:
    """Row loop with explicit e4m3 nearest-value search (ties to even mantissa code)."""
    table = []
    for byte in range(256):
        if byte & 0x7F == 0x7F:
            continue
        sign = -1.0 if byte & 0x80 else 1.0
        exp = (byte >> 3) & 0xF
        man = byte & 0x7
        value = man / 8.0 * 2.0**-6 if exp == 0 else (1.0 + man / 8.0) * 2.0 ** (exp - 7)
        table.append((sign * value, byte))
    out = torch.empty_like(w)
    for r in range(w.shape[0]):
        amax = w[r].abs().max().item()
        scale = max(amax / FP8_MAX, 1.0 / (448.0 * 512.0))
        scale32 = torch.tensor(scale, dtype=torch.float32)
        for c in range(w.shape[1]):
            x = float(torch.clamp(w[r, c] / scale32, -FP8_MAX, FP8_MAX))
            best = min(table, key=lambda t, x=x: (abs(t[0] - x), t[1] & 1))
            out[r, c] = best[0] * scale32
    return out


def test_fused_3d_tensor_matches_per_expert_2d_and_is_chunk_invariant() -> None:
    torch.manual_seed(2)
    w = torch.randn(5, 24, 48) * torch.rand(5, 1, 1) * 3
    whole = fp8.quantize_fp8(w, expert_chunk=5)
    for chunk in (1, 2, 3):
        part = fp8.quantize_fp8(w, expert_chunk=chunk)
        assert torch.equal(part.weight.view(torch.uint8), whole.weight.view(torch.uint8))
        assert torch.equal(part.scale, whole.scale)
        assert part.error.max_abs == whole.error.max_abs
        assert part.error.rel_fro == pytest.approx(whole.error.rel_fro, rel=1e-6)
    assert whole.weight.shape == (5, 24, 48)
    assert whole.scale.shape == (5, 24, 1)
    for e in range(5):
        single = fp8.quantize_fp8(w[e])
        assert torch.equal(single.weight.view(torch.uint8), whole.weight[e].view(torch.uint8))
        assert torch.equal(single.scale, whole.scale[e])


def test_rejects_non_finite_and_bad_rank() -> None:
    bad = torch.zeros(2, 16)
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        fp8.quantize_fp8(bad)
    with pytest.raises(ValueError, match="rank"):
        fp8.quantize_fp8(torch.zeros(16))


def test_stats_record_scale_range() -> None:
    w = torch.tensor([[1.0, -2.0, 0.5, 0.25], [0.0, 0.0, 0.0, 0.0]])
    result = fp8.quantize_fp8(w)
    assert result.scale_stats.max_scale == pytest.approx(2.0 / FP8_MAX)
    assert result.scale_stats.min_scale == pytest.approx(fp8.MIN_SCALE)
    assert result.scale_stats.n_zero_rows == 1
    assert result.scale_stats.n_rows == 2
