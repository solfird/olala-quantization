"""Tests for olala_quant.nvfp4 (NVFP4 weight quantization, compressed-tensors convention).

Conventions under test, all read from the vLLM fork (commit ebbadef):

* e2m1 value table and nibble layout: quantization/utils/nvfp4_emulation_utils.py:23-25 and
  break_fp4_bytes (:546-563); low nibble is the even element, high nibble the odd one.
* round-to-nearest thresholds: cast_to_fp4 (:695-706), ties to the even code.
* the stored global scale is the ENCODE scale 448*6/amax (a divisor of the block scale), and
  dequantization is fp4 * (block_scale * (1 / global_scale))
  (compressed_tensors_w4a4_nvfp4.py:110-112, nvfp4_emulation_utils.py:576-630 and :633-690).
"""

import pytest
import torch

from olala_quant import nvfp4

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def test_constants() -> None:
    assert nvfp4.GROUP_SIZE == 16
    assert nvfp4.FP4_MAX == 6.0
    assert nvfp4.E4M3_MAX == 448.0
    assert list(nvfp4.E2M1_VALUES) == E2M1


def test_encode_decode_round_trip_on_every_representable_value() -> None:
    values = torch.tensor(E2M1 + [-v for v in E2M1[1:]])
    codes = nvfp4.encode_fp4(values)
    assert codes.dtype == torch.uint8
    assert torch.equal(nvfp4.decode_fp4(codes), values)
    # canonical encoding: sign bit 0x8 only for non-zero negatives, magnitude in the low 3 bits
    assert nvfp4.encode_fp4(torch.tensor([-0.0, 0.0, -0.1])).tolist() == [0, 0, 0]
    assert nvfp4.encode_fp4(torch.tensor([-1.0, -6.0, 6.0, 0.5])).tolist() == [
        0x8 | 2,
        0x8 | 7,
        7,
        1,
    ]


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (0.25, 0.0),  # tie 0 / 0.5 goes to code 0
        (0.2500001, 0.5),
        (0.75, 1.0),  # tie 0.5 (code 1) / 1.0 (code 2) goes to even code 2
        (1.25, 1.0),  # tie 1.0 (code 2) / 1.5 (code 3) goes to code 2
        (1.75, 2.0),  # tie 1.5 / 2.0 goes to code 4
        (2.5, 2.0),  # tie 2 (code 4) / 3 (code 5) goes to code 4
        (3.5, 4.0),  # tie 3 (code 5) / 4 (code 6) goes to code 6
        (5.0, 4.0),  # tie 4 (code 6) / 6 (code 7) goes to code 6
        (5.0000005, 6.0),
        (100.0, 6.0),  # saturates
        (-3.5, -4.0),
        (-0.2, 0.0),
    ],
)
def test_rounding_thresholds_match_vllm_cast_to_fp4(x: float, expected: float) -> None:
    got = nvfp4.decode_fp4(nvfp4.encode_fp4(torch.tensor([x], dtype=torch.float32)))
    assert got.item() == expected


def test_pack_unpack_low_nibble_is_even_element() -> None:
    codes = torch.tensor([[1, 2, 3, 4, 0xF, 0x0, 7, 8]], dtype=torch.uint8)
    packed = nvfp4.pack_fp4(codes)
    assert packed.dtype == torch.uint8
    assert packed.shape == (1, 4)
    assert packed.tolist() == [[0x21, 0x43, 0x0F, 0x87]]
    assert torch.equal(nvfp4.unpack_fp4(packed), codes)


def test_pack_rejects_odd_width() -> None:
    with pytest.raises(ValueError, match="even"):
        nvfp4.pack_fp4(torch.zeros(2, 3, dtype=torch.uint8))


def _representable_block_weight() -> tuple[torch.Tensor, float]:
    """A [4, 32] weight made of exact e2m1 values times powers of two per block."""
    torch.manual_seed(3)
    idx = torch.randint(0, 8, (4, 32))
    sign = torch.where(torch.rand(4, 32) < 0.5, -1.0, 1.0)
    base = torch.tensor(E2M1)[idx] * sign
    base[:, 0::16] = 6.0  # every block contains the max so the block scale is exact
    return base, 1.0


def test_exact_round_trip_on_representable_values() -> None:
    base, _ = _representable_block_weight()
    # s = 7/32 makes the tensor amax 6s = 1.3125, so the global scale is 2688/1.3125 = 2048 (a power
    # of two) and every block scale is exactly 448: nothing in the pipeline rounds.
    w = base * 0.21875
    result = nvfp4.quantize_nvfp4(w)
    deq = nvfp4.dequantize_nvfp4(result.packed, result.block_scale, result.global_scale)
    assert torch.equal(deq, w)
    assert result.error.max_abs == 0.0
    assert result.error.rel_fro == 0.0


def test_shapes_dtypes_and_global_scale_convention() -> None:
    torch.manual_seed(4)
    w = torch.randn(8, 64) * 0.05
    result = nvfp4.quantize_nvfp4(w)
    assert result.packed.dtype == torch.uint8
    assert result.packed.shape == (8, 32)
    assert result.block_scale.dtype == torch.float8_e4m3fn
    assert result.block_scale.shape == (8, 4)
    assert result.global_scale.dtype == torch.float32
    assert result.global_scale.shape == (1,)
    # compressed-tensors stores the ENCODE scale: 448 * 6 / amax(tensor)
    assert result.global_scale.item() == pytest.approx(448.0 * 6.0 / w.abs().max().item())
    # the largest block's e4m3 scale is exactly 448 (= amax_block/6 * global_scale)
    assert result.block_scale.float().max().item() == 448.0


def test_error_bounds_and_bitwise_match_with_slow_reference() -> None:
    torch.manual_seed(5)
    w = (torch.randn(6, 48) * 0.03).to(torch.bfloat16)
    result = nvfp4.quantize_nvfp4(w)
    ref_codes, ref_scale_bytes, ref_global = _slow_reference_nvfp4(w.float())
    assert torch.equal(nvfp4.unpack_fp4(result.packed), ref_codes)
    assert torch.equal(result.block_scale.view(torch.uint8), ref_scale_bytes)
    assert result.global_scale.item() == pytest.approx(ref_global)
    deq = nvfp4.dequantize_nvfp4(result.packed, result.block_scale, result.global_scale)
    # NVFP4 on gaussian weights: relative Frobenius error of about 0.09 to 0.12
    assert 0.05 < result.error.rel_fro < 0.2
    # per element: at most half the largest e2m1 step (1.0 on the 4..6 interval) times block scale
    blocks = w.float().reshape(6, 3, 16)
    block_amax = blocks.abs().amax(dim=-1, keepdim=True)
    bound = (block_amax / 6.0 * 1.0 * 1.0626).expand(-1, -1, 16).reshape(6, 48)  # 1.0626: e4m3 scale slack
    assert torch.all((deq - w.float()).abs() <= bound + 1e-7)
    assert result.error.max_abs == pytest.approx((deq - w.float()).abs().max().item(), rel=1e-6)


def _e4m3_table() -> list[tuple[float, int]]:
    table = []
    for byte in range(128):  # positive finite values only
        if byte == 0x7F:
            continue
        exp, man = (byte >> 3) & 0xF, byte & 0x7
        value = man / 8.0 * 2.0**-6 if exp == 0 else (1.0 + man / 8.0) * 2.0 ** (exp - 7)
        table.append((value, byte))
    return table


def _slow_reference_nvfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Pure-Python loops: explicit e4m3 table search and explicit fp4 thresholds."""
    table = _e4m3_table()
    amax = float(w.abs().max())
    g = torch.tensor(448.0 * 6.0, dtype=torch.float32) / torch.tensor(amax, dtype=torch.float32)
    recip6 = torch.tensor(1.0 / 6.0, dtype=torch.float32)
    rows, cols = w.shape
    codes = torch.zeros(rows, cols, dtype=torch.uint8)
    scale_bytes = torch.zeros(rows, cols // 16, dtype=torch.uint8)
    for r in range(rows):
        for b in range(cols // 16):
            block = [float(v) for v in w[r, b * 16 : (b + 1) * 16]]
            bmax = max(abs(v) for v in block)
            s = float(g * (torch.tensor(bmax, dtype=torch.float32) * recip6))
            s = min(s, 448.0)
            best = min(table, key=lambda t, s=s: (abs(t[0] - s), t[1] & 1))
            scale_bytes[r, b] = best[1]
            sq = torch.tensor(best[0], dtype=torch.float32)
            for i, v in enumerate(block):
                if best[1] == 0:
                    code = 0
                else:
                    inv = 1.0 / (sq * (1.0 / g))
                    x = float(torch.tensor(v, dtype=torch.float32) * inv)
                    x = max(-6.0, min(6.0, x))
                    a = abs(x)
                    code = int(
                        (a > 0.25)
                        + (a >= 0.75)
                        + (a > 1.25)
                        + (a >= 1.75)
                        + (a > 2.5)
                        + (a >= 3.5)
                        + (a > 5.0)
                    )
                    if x < 0 and code != 0:
                        code |= 0x8
                codes[r, b * 16 + i] = code
    return codes, scale_bytes, float(g)


def test_fused_3d_per_expert_global_scale_and_chunk_invariance() -> None:
    torch.manual_seed(6)
    w = torch.randn(5, 16, 32) * (torch.rand(5, 1, 1) * 10 + 0.01)
    whole = nvfp4.quantize_nvfp4(w, expert_chunk=5)
    assert whole.packed.shape == (5, 16, 16)
    assert whole.block_scale.shape == (5, 16, 2)
    assert whole.global_scale.shape == (5,)
    for e in range(5):
        single = nvfp4.quantize_nvfp4(w[e])
        assert torch.equal(single.packed, whole.packed[e])
        assert torch.equal(single.block_scale.view(torch.uint8), whole.block_scale[e].view(torch.uint8))
        assert single.global_scale.item() == whole.global_scale[e].item()
    for chunk in (1, 2):
        part = nvfp4.quantize_nvfp4(w, expert_chunk=chunk)
        assert torch.equal(part.packed, whole.packed)
        assert part.error.rel_fro == pytest.approx(whole.error.rel_fro, rel=1e-6)
        assert part.scale_stats.n_blocks == whole.scale_stats.n_blocks


def test_scale_stats_report_max_min_and_flushed_blocks() -> None:
    # one block ten orders of magnitude below the tensor max: its e4m3 scale underflows to zero
    w = torch.zeros(2, 32)
    w[0, :16] = torch.linspace(-1.0, 1.0, 16)  # the tensor max block
    w[1, :16] = 1e-9 * torch.linspace(-1.0, 1.0, 16)  # flushed
    result = nvfp4.quantize_nvfp4(w)
    s = result.scale_stats
    assert s.n_blocks == 4
    assert s.n_zero_blocks == 2  # the two all-zero blocks
    assert s.n_flushed == 1  # non-zero block whose scale rounds to e4m3 zero
    assert s.n_overflow == 0
    assert s.max_block_scale == 448.0
    assert s.min_nonzero_block_scale > 0.0
    deq = nvfp4.dequantize_nvfp4(result.packed, result.block_scale, result.global_scale)
    assert torch.all(deq[1] == 0.0)  # flushed block reconstructs to zero, loudly counted


def test_block_scale_never_overflows_even_at_the_tensor_max() -> None:
    torch.manual_seed(7)
    w = torch.randn(32, 64) * 1000.0
    result = nvfp4.quantize_nvfp4(w)
    assert result.scale_stats.n_overflow == 0
    scales = result.block_scale.float()
    assert torch.isfinite(scales).all()
    assert scales.max().item() <= 448.0


def test_all_zero_tensor_is_safe() -> None:
    result = nvfp4.quantize_nvfp4(torch.zeros(4, 32))
    assert result.global_scale.item() == 1.0
    assert torch.all(result.packed == 0)
    assert torch.all(result.block_scale.view(torch.uint8) == 0)
    assert result.scale_stats.n_zero_blocks == 8
    assert result.error.rel_fro == 0.0


def test_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="multiple of 16"):
        nvfp4.quantize_nvfp4(torch.zeros(2, 24))
    with pytest.raises(ValueError, match="rank"):
        nvfp4.quantize_nvfp4(torch.zeros(32))
    bad = torch.zeros(2, 32)
    bad[1, 5] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        nvfp4.quantize_nvfp4(bad)
