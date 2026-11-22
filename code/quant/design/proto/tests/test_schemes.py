"""Known-answer tests for the two quantization grids (NVFP4 g16 and symmetric INT4)."""

import torch
from olala_quant.nvfp4 import E2M1_VALUES, dequantize_nvfp4, quantize_nvfp4

from moegptq.schemes import Int4Scheme, Nvfp4Scheme, rtn_quantize


def test_nvfp4_rtn_matches_olala_quant_reference_per_expert():
    torch.manual_seed(0)
    w = torch.randn(3, 24, 64) * torch.tensor([0.5, 1.0, 3.0]).view(3, 1, 1)
    ref = quantize_nvfp4(w)
    ref_dq = dequantize_nvfp4(ref.packed, ref.block_scale, ref.global_scale)
    for e in range(3):
        q, _ = rtn_quantize(w[e], Nvfp4Scheme())
        assert torch.equal(q, ref_dq[e]), f"expert {e} differs from olala_quant.nvfp4"


def test_nvfp4_values_lie_on_e2m1_grid_times_representable_block_scale():
    torch.manual_seed(1)
    w = torch.randn(8, 48)
    scheme = Nvfp4Scheme()
    q, scales = rtn_quantize(w, scheme)
    grid = torch.tensor(sorted(set(E2M1_VALUES) | {-v for v in E2M1_VALUES}))
    assert scales.shape == (8, 3)
    for gi in range(3):
        s = scales[:, gi : gi + 1]
        ratio = q[:, gi * 16 : (gi + 1) * 16] / s
        nearest = (ratio.unsqueeze(-1) - grid).abs().amin(dim=-1)
        assert torch.all(nearest < 1e-5), "value off the e2m1 grid"


def test_int4_rtn_keeps_representable_weights_exactly():
    s = torch.tensor([[0.25], [0.5]])
    ints = torch.tensor([[7, -7, 3, 0] * 32, [-7, 7, 1, -2] * 32], dtype=torch.float32)
    w = ints * s
    q, scales = rtn_quantize(w, Int4Scheme(group_size=128))
    assert torch.allclose(q, w, atol=1e-7)
    assert torch.allclose(scales, s)


def test_int4_error_bounded_by_half_a_step_and_groups_are_independent():
    torch.manual_seed(2)
    w = torch.randn(4, 256)
    w[:, 128:] *= 10.0  # second group has a 10x larger scale
    q, scales = rtn_quantize(w, Int4Scheme(group_size=128))
    assert scales.shape == (4, 2)
    assert torch.all(scales[:, 1] > 5 * scales[:, 0])
    for gi in range(2):
        err = (q[:, gi * 128 : (gi + 1) * 128] - w[:, gi * 128 : (gi + 1) * 128]).abs()
        assert torch.all(err <= scales[:, gi : gi + 1] / 2 + 1e-6)


def test_int4_group_larger_than_row_means_one_group_per_row():
    w = torch.randn(5, 48)
    _, scales = rtn_quantize(w, Int4Scheme(group_size=128))
    assert scales.shape == (5, 1)


def test_zero_weights_stay_zero():
    for scheme in (Nvfp4Scheme(), Int4Scheme(group_size=128)):
        q, _ = rtn_quantize(torch.zeros(4, 128), scheme)
        assert torch.count_nonzero(q) == 0


def test_int4_round_clamps_to_minus_eight_and_seven():
    wg = torch.tensor([[-8.0, 8.0, 7.4, -7.6, 3.5, -3.5]])
    s = torch.ones(1, 1)
    out = Int4Scheme(group_size=128).round(wg, s)
    assert out.tolist() == [[-8.0, 7.0, 7.0, -8.0, 4.0, -4.0]]
