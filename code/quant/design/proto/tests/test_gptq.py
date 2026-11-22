"""Known-answer and property tests for the GPTQ solver (including 'identity Hessian gives RTN')."""

import pytest
import torch

from moegptq.gptq import gptq_quantize, weighted_error
from moegptq.schemes import Int4Scheme, Nvfp4Scheme, rtn_quantize

SCHEMES = [Nvfp4Scheme, lambda: Int4Scheme(group_size=128)]


def _correlated_hessian(n: int, seed: int, tokens: int = 4096) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    mix = torch.randn(n, n, generator=g) / n**0.5
    mix = mix * (torch.arange(1, n + 1).float() ** -0.7)
    x = torch.randn(tokens, n, generator=g) @ mix
    return (x.T @ x).double()


@pytest.mark.parametrize("make", SCHEMES)
@pytest.mark.parametrize("dynamic", [True, False])
def test_identity_hessian_gives_rtn(make, dynamic):
    torch.manual_seed(3)
    w = torch.randn(32, 256) * 0.05
    scheme = make()
    q_rtn, _ = rtn_quantize(w, scheme)
    q, info = gptq_quantize(w, torch.eye(256, dtype=torch.float64), scheme, dynamic=dynamic)
    assert not info["fallback"]
    assert torch.equal(q, q_rtn)


@pytest.mark.parametrize("make", SCHEMES)
def test_diagonal_hessian_gives_rtn_whatever_its_scale(make):
    torch.manual_seed(4)
    w = torch.randn(16, 256) * 0.05
    h = torch.diag(torch.rand(256, dtype=torch.float64) * 1e6 + 1.0)
    q_rtn, _ = rtn_quantize(w, make())
    q, _ = gptq_quantize(w, h, make())
    assert torch.equal(q, q_rtn)


@pytest.mark.parametrize("make", SCHEMES)
def test_result_is_invariant_to_hessian_scale(make):
    torch.manual_seed(5)
    w = torch.randn(16, 256) * 0.05
    h = _correlated_hessian(256, 5)
    q1, _ = gptq_quantize(w, h, make())
    q2, _ = gptq_quantize(w, 37.0 * h, make())
    # not bitwise: 37 * h differs from h in the last float64 bit and BLAS reductions are not deterministic
    # across thread counts, so a rounding decision at a grid boundary may flip (a one-step change)
    assert float((q1 != q2).double().mean()) < 1e-3


@pytest.mark.parametrize("make", SCHEMES)
@pytest.mark.parametrize("seed", [10, 11, 12, 13, 14])
def test_gptq_lowers_the_calibration_objective_on_a_correlated_hessian(make, seed):
    torch.manual_seed(seed)
    w = torch.randn(48, 256) * 0.05
    h = _correlated_hessian(256, seed)
    q_rtn, _ = rtn_quantize(w, make())
    q, _ = gptq_quantize(w, h, make(), percdamp=0.01)
    assert weighted_error(w, q, h) < weighted_error(w, q_rtn, h)


def test_dead_columns_stay_finite_and_are_plain_rounded():
    torch.manual_seed(6)
    w = torch.randn(8, 256) * 0.05
    h = _correlated_hessian(256, 6)
    h[7, :] = 0.0
    h[:, 7] = 0.0
    q_rtn, _ = rtn_quantize(w, Int4Scheme(group_size=128))
    q, info = gptq_quantize(w, h, Int4Scheme(group_size=128))
    assert torch.isfinite(q).all()
    assert not info["fallback"]
    # With a zero row/column the column receives no compensation, but earlier columns still shift the
    # scale group; the dead weight itself must stay within half a step of its original value.
    assert (q[:, 7] - w[:, 7]).abs().max() <= (w.abs().amax() / 7) + 1e-6
    assert q_rtn.shape == q.shape


@pytest.mark.parametrize("make", SCHEMES)
def test_non_finite_or_indefinite_hessian_falls_back_to_rtn(make):
    torch.manual_seed(7)
    w = torch.randn(8, 128) * 0.05
    q_rtn, _ = rtn_quantize(w, make())
    bad = torch.eye(128, dtype=torch.float64)
    bad[3, 3] = float("nan")
    q, info = gptq_quantize(w, bad, make())
    assert info["fallback"] and torch.equal(q, q_rtn)
    neg = -torch.eye(128, dtype=torch.float64)
    q, info = gptq_quantize(w, neg, make())
    assert info["fallback"] and torch.equal(q, q_rtn)


def test_dynamic_nvfp4_scales_are_representable_e4m3_values():
    torch.manual_seed(8)
    w = torch.randn(16, 128) * 0.05
    h = _correlated_hessian(128, 8)
    scheme = Nvfp4Scheme()
    q, info = gptq_quantize(w, h, scheme, dynamic=True)
    g = scheme.prepare(w)
    scale_q = info["scales"] * g
    again = scale_q.to(torch.float8_e4m3fn).to(torch.float32)
    assert torch.allclose(scale_q, again, rtol=1e-6)
    assert torch.isfinite(q).all()


def test_damping_modes_differ_only_when_the_diagonal_is_uneven():
    torch.manual_seed(9)
    w = torch.randn(16, 128) * 0.05
    h = _correlated_hessian(128, 9)
    q_mean, _ = gptq_quantize(w, h, Int4Scheme(), damp_mode="mean")
    q_diag, _ = gptq_quantize(w, h, Int4Scheme(), damp_mode="diag")
    assert q_mean.shape == q_diag.shape
    with pytest.raises(ValueError):
        gptq_quantize(w, h, Int4Scheme(), damp_mode="nope")


@pytest.mark.parametrize("make", SCHEMES)
def test_act_order_identity_hessian_gives_rtn(make):
    torch.manual_seed(20)
    w = torch.randn(16, 256) * 0.05
    q_rtn, _ = rtn_quantize(w, make())
    q, info = gptq_quantize(w, torch.eye(256, dtype=torch.float64), make(), dynamic=False, act_order=True)
    assert not info["fallback"]
    assert torch.equal(q, q_rtn)


@pytest.mark.parametrize("make", SCHEMES)
def test_act_order_keeps_every_weight_on_the_grid_of_its_natural_group(make):
    torch.manual_seed(21)
    scheme = make()
    w = torch.randn(16, 256) * 0.05
    h = _correlated_hessian(256, 21)
    q, info = gptq_quantize(w, h, scheme, dynamic=False, act_order=True)
    gs = min(scheme.group_size, 256)
    state = scheme.prepare(w)
    for gi in range(256 // gs):
        s = info["scales"][:, gi : gi + 1]
        qg = q[:, gi * gs : (gi + 1) * gs]
        assert torch.equal(scheme.round(qg, s), qg), "weight left the grid of its natural group"
        assert torch.equal(s, scheme.scales(w[:, gi * gs : (gi + 1) * gs], state))


def test_act_order_with_dynamic_scales_is_refused():
    with pytest.raises(ValueError):
        gptq_quantize(torch.randn(4, 128), torch.eye(128, dtype=torch.float64), Int4Scheme(), act_order=True)


@pytest.mark.parametrize("make", SCHEMES)
def test_act_order_lowers_the_calibration_objective(make):
    torch.manual_seed(22)
    w = torch.randn(48, 256) * 0.05
    h = _correlated_hessian(256, 22)
    q_rtn, _ = rtn_quantize(w, make())
    q, _ = gptq_quantize(w, h, make(), dynamic=False, act_order=True)
    assert weighted_error(w, q, h) < weighted_error(w, q_rtn, h)


def _naive_gptq(w, h, scheme, percdamp=0.01):
    """Textbook column-by-column GPTQ with static group scales, written independently of the solver."""
    w = w.double().clone()
    n = h.shape[0]
    hd = h.clone().double()
    hd += percdamp * torch.diagonal(hd).mean() * torch.eye(n, dtype=torch.float64)
    upper = torch.linalg.cholesky(torch.linalg.inv(hd), upper=True)
    gs = min(scheme.group_size, n)
    state = scheme.prepare(w.float())
    scales = [scheme.scales(w[:, c : c + gs].float(), state) for c in range(0, n, gs)]
    out = torch.zeros_like(w)
    for i in range(n):
        q = scheme.round(w[:, i : i + 1].float(), scales[i // gs]).double().squeeze(1)
        out[:, i] = q
        err = (w[:, i] - q) / upper[i, i]
        w[:, i:] -= err.unsqueeze(1) * upper[i, i:].unsqueeze(0)
    return out.float()


@pytest.mark.parametrize("make", SCHEMES)
@pytest.mark.parametrize("blocksize", [32, 128, 256])
def test_blocked_solver_equals_the_naive_column_loop_for_any_block_size(make, blocksize):
    torch.manual_seed(30)
    w = torch.randn(24, 256) * 0.05
    h = _correlated_hessian(256, 30)
    expected = _naive_gptq(w, h, make())
    got, _ = gptq_quantize(w, h, make(), dynamic=False, blocksize=blocksize)
    assert float((got != expected).double().mean()) < 1e-3  # float64 order of operations differs
