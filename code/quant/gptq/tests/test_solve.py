"""Tests of the batched NVFP4 GPTQ solver (solve.py): known answers, prototype cross-check, guards."""

import pytest
import torch
from moegptq.gptq import gptq_quantize as proto_gptq
from moegptq.schemes import Nvfp4Scheme
from olala_quant import nvfp4

import solve


def _weights(e: int = 4, r: int = 32, c: int = 64, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    spread = torch.rand(e, 1, 1, generator=g) * 3 + 0.3
    return (torch.randn(e, r, c, generator=g) * 0.05 * spread).to(torch.bfloat16)


def _correlated_hessians(e: int, c: int, n: int = 400, seed: int = 1) -> torch.Tensor:
    """Sum of x x^T over n correlated tokens per expert (float32, positive definite)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(e):
        mix = torch.randn(c, c, generator=g) / c**0.5
        mix = mix * (torch.arange(1, c + 1) ** -0.5)[None, :]
        x = torch.randn(n, c, generator=g) @ mix
        out.append(x.T @ x)
    return torch.stack(out)


def test_static_state_is_exactly_the_data_free_n2_scales() -> None:
    w = _weights()
    st = solve.static_nvfp4(w)
    ref = nvfp4.quantize_nvfp4(w)
    assert torch.equal(st.block_scale.view(torch.uint8), ref.block_scale.view(torch.uint8))
    assert torch.equal(st.global_scale, ref.global_scale)
    assert torch.equal(st.rtn_packed, ref.packed)


def test_dequantize_codes_equals_dequantize_nvfp4_bitwise() -> None:
    w = _weights()
    st = solve.static_nvfp4(w)
    ref = nvfp4.dequantize_nvfp4(st.rtn_packed, st.block_scale, st.global_scale)
    assert torch.equal(solve.dequantize_codes(solve.rtn_codes(st), st), ref)


@pytest.mark.parametrize("act_order", [False, True])
def test_diagonal_hessian_returns_rtn_codes(act_order: bool) -> None:
    w = _weights()
    st = solve.static_nvfp4(w)
    g = torch.Generator().manual_seed(5)
    h = torch.diag_embed(torch.rand(4, 64, generator=g) + 0.1)
    res = solve.solve_experts(w, h, st, solve.SolverConfig(act_order=act_order))
    assert torch.equal(res.codes, solve.rtn_codes(st))
    assert not res.fallback.any()


def test_identity_hessian_returns_rtn_codes() -> None:
    w = _weights()
    st = solve.static_nvfp4(w)
    h = torch.eye(64).expand(4, 64, 64).contiguous()
    res = solve.solve_experts(w, h, st, solve.SolverConfig())
    assert torch.equal(res.codes, solve.rtn_codes(st))


def test_correlated_hessian_changes_codes_and_lowers_weighted_error() -> None:
    w = _weights()
    st = solve.static_nvfp4(w)
    h = _correlated_hessians(4, 64)
    res = solve.solve_experts(w, h, st, solve.SolverConfig())
    assert not torch.equal(res.codes, solve.rtn_codes(st))
    err_g = solve.weighted_error(w, solve.dequantize_codes(res.codes, st), h)
    err_r = solve.weighted_error(w, solve.dequantize_codes(solve.rtn_codes(st), st), h)
    assert bool((err_g < err_r).all())


@pytest.mark.parametrize("act_order", [False, True])
def test_matches_the_prototype_solver_per_expert(act_order: bool) -> None:
    w = _weights(e=3)
    st = solve.static_nvfp4(w)
    h = _correlated_hessians(3, 64)
    cfg = solve.SolverConfig(act_order=act_order, blocksize=32)
    res = solve.solve_experts(w, h, st, cfg)
    deq = solve.dequantize_codes(res.codes, st)
    for e in range(3):
        ref, info = proto_gptq(
            w[e].float(), h[e], Nvfp4Scheme(), percdamp=0.01, blocksize=32, dynamic=False, act_order=act_order
        )
        assert not info["fallback"]
        agree = float((deq[e] == ref).float().mean())
        assert agree > 0.999, agree
        assert float(res.loss[e]) == pytest.approx(info["loss"], rel=1e-6)


def test_expert_chunking_does_not_change_the_result() -> None:
    w = _weights(e=5)
    st = solve.static_nvfp4(w)
    h = _correlated_hessians(5, 64)
    a = solve.solve_experts(w, h, st, solve.SolverConfig(expert_chunk=1))
    b = solve.solve_experts(w, h, st, solve.SolverConfig(expert_chunk=5))
    assert torch.equal(a.codes, b.codes)


def test_codes_are_nibbles_without_negative_zero() -> None:
    w = _weights()
    st = solve.static_nvfp4(w)
    res = solve.solve_experts(w, _correlated_hessians(4, 64), st, solve.SolverConfig())
    assert res.codes.dtype == torch.uint8
    assert int(res.codes.max()) <= 15
    assert not bool((res.codes == 8).any())


def test_act_order_changes_the_solution_on_a_nontrivial_hessian() -> None:
    w = _weights(e=2)
    st = solve.static_nvfp4(w)
    h = _correlated_hessians(2, 64)
    a = solve.solve_experts(w, h, st, solve.SolverConfig(act_order=False))
    b = solve.solve_experts(w, h, st, solve.SolverConfig(act_order=True))
    assert not torch.equal(a.codes, b.codes)


def test_dead_columns_are_rounded_not_zeroed() -> None:
    w = _weights(e=1)
    st = solve.static_nvfp4(w)
    h = _correlated_hessians(1, 64)
    h[0, 10, :] = 0
    h[0, :, 10] = 0
    res = solve.solve_experts(w, h, st, solve.SolverConfig())
    assert not res.fallback.any()
    assert torch.equal(res.codes[0, :, 10], solve.rtn_codes(st)[0, :, 10])


@pytest.mark.parametrize("act_order", [False, True])
def test_a_dead_column_is_treated_like_the_prototype_does_even_at_a_tiny_hessian_scale(act_order: bool) -> None:
    """A zero diagonal gets a unit entry before damping, so a tiny H is dominated by it: a scale-dependent rule."""
    w = _weights(e=2)
    st = solve.static_nvfp4(w)
    h = _correlated_hessians(2, 64) * 1e-6
    h[:, 10, :] = 0
    h[:, :, 10] = 0
    cfg = solve.SolverConfig(act_order=act_order, blocksize=32)
    res = solve.solve_experts(w, h, st, cfg)
    deq = solve.dequantize_codes(res.codes, st)
    for e in range(2):
        ref, info = proto_gptq(
            w[e].float(), h[e], Nvfp4Scheme(), percdamp=0.01, blocksize=32, dynamic=False, act_order=act_order
        )
        assert not info["fallback"]
        assert float((deq[e] == ref).float().mean()) > 0.999
        assert float(res.loss[e]) == pytest.approx(info["loss"], rel=1e-6)


@pytest.mark.parametrize(
    ("damage", "why"),
    [
        ("nan", "hessian_not_finite"),
        ("negative_diag", "hessian_negative_diagonal"),
        ("not_psd", "hessian_not_positive_definite"),
    ],
)
def test_unusable_hessian_falls_back_to_rtn_and_says_why(damage: str, why: str) -> None:
    w = _weights(e=2)
    st = solve.static_nvfp4(w)
    h = _correlated_hessians(2, 64)
    if damage == "nan":
        h[1, 3, 3] = float("nan")
    elif damage == "negative_diag":
        h[1, 3, 3] = -1.0
    else:
        h[1] = 2.0 * torch.ones(64, 64) - torch.eye(64)  # positive diagonal, indefinite
    res = solve.solve_experts(w, h, st, solve.SolverConfig())
    assert res.fallback.tolist() == [False, True]
    assert res.reason[0] == ""
    assert res.reason[1] == why
    assert torch.equal(res.codes[1], solve.rtn_codes(st)[1])


def test_diagonal_ratio_guard_sends_a_spiky_hessian_to_rtn() -> None:
    w = _weights(e=2)
    st = solve.static_nvfp4(w)
    h = _correlated_hessians(2, 64)
    h[1, 5, 5] *= 1e9
    res = solve.solve_experts(w, h, st, solve.SolverConfig(diag_ratio_max=1e6))
    assert res.fallback.tolist() == [False, True]
    assert "diag_ratio" in res.reason[1]
    assert res.diag_ratio[1] > 1e6 > res.diag_ratio[0]
    relaxed = solve.solve_experts(w, h, st, solve.SolverConfig(diag_ratio_max=1e30))
    assert not relaxed.fallback[1]


def test_inactive_experts_get_rtn_codes_and_a_reason() -> None:
    w = _weights(e=3)
    st = solve.static_nvfp4(w)
    h = _correlated_hessians(3, 64)
    active = torch.tensor([True, False, True])
    res = solve.solve_experts(w, h, st, solve.SolverConfig(), active=active)
    assert res.fallback.tolist() == [False, True, False]
    assert res.reason[1] == "inactive"
    assert torch.equal(res.codes[1], solve.rtn_codes(st)[1])


def test_result_does_not_depend_on_the_scale_of_the_hessian() -> None:
    w = _weights(e=2)
    st = solve.static_nvfp4(w)
    h = _correlated_hessians(2, 64)
    a = solve.solve_experts(w, h, st, solve.SolverConfig())
    b = solve.solve_experts(w, h * 1000.0, st, solve.SolverConfig())
    assert float((a.codes == b.codes).float().mean()) > 0.999


def test_rejects_widths_that_are_not_a_multiple_of_the_group() -> None:
    w = torch.randn(1, 8, 24).to(torch.bfloat16)
    with pytest.raises(ValueError):
        solve.static_nvfp4(w)


def _expert(x: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    return torch.relu(x @ up.T).square() @ down.T


def test_expert_output_error_matches_an_explicit_computation() -> None:
    g = torch.Generator().manual_seed(3)
    up, down = torch.randn(16, 8, generator=g), torch.randn(8, 16, generator=g)
    upq, downq = up + 0.01, down - 0.02
    x = torch.randn(5, 8, generator=g)
    want = float(((_expert(x, upq, downq) - _expert(x, up, down)) ** 2).sum())
    assert solve.expert_output_error(up, down, upq, downq, x) == pytest.approx(want, rel=1e-6)
    assert solve.expert_output_error(up, down, up, down, x) == 0.0


def test_heldout_guard_keeps_gptq_only_when_it_is_better_on_heldout_rows() -> None:
    g = torch.Generator().manual_seed(4)
    e, d, hid = 2, 16, 32
    up = torch.randn(e, hid, d, generator=g) * 0.1
    down = torch.randn(e, d, hid, generator=g) * 0.1
    rtn_up, rtn_down = up + 0.001, down + 0.001
    good_up, good_down = up + 0.0001, down + 0.0001  # closer to the reference than RTN
    bad_up, bad_down = up + 0.05, down + 0.05
    held = [torch.randn(20, d, generator=g), torch.randn(20, d, generator=g)]
    gptq_up = torch.stack([good_up[0], bad_up[1]])
    gptq_down = torch.stack([good_down[0], bad_down[1]])
    keep = solve.heldout_guard(up, down, gptq_up, gptq_down, rtn_up, rtn_down, held, min_rows=4)
    assert keep.keep.tolist() == [True, False]
    assert keep.err_gptq[0] < keep.err_rtn[0]
    assert keep.err_gptq[1] > keep.err_rtn[1]
    assert keep.reason == ["", "heldout_worse"]


def test_heldout_guard_refuses_to_judge_an_expert_with_too_few_heldout_rows() -> None:
    g = torch.Generator().manual_seed(4)
    up, down = torch.randn(1, 8, 4, generator=g), torch.randn(1, 4, 8, generator=g)
    held = [torch.randn(2, 4, generator=g)]
    keep = solve.heldout_guard(up, down, up, down, up + 0.1, down + 0.1, held, min_rows=4)
    assert keep.keep.tolist() == [False]
    assert keep.reason == ["heldout_too_few"]
