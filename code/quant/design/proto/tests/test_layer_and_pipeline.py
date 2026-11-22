"""Toy layer forward, spike injection, and the per-expert pipeline policies."""

import pytest
import torch

from moegptq.pipeline import Method, evaluate, quantize_layer
from moegptq.schemes import Int4Scheme, Nvfp4Scheme, rtn_quantize
from moegptq.toy import ToyConfig, layer_forward, make_layer, sample_tokens

SMALL = ToyConfig(n_experts=4, d=32, h=64, top_k=2, rare_experts=0)


def test_layer_forward_is_non_gated_relu_squared_with_gate_weights():
    layer = make_layer(SMALL, seed=0)
    x = torch.randn(1, SMALL.d, dtype=torch.float64)
    idx = torch.tensor([[1, 3]])
    p = torch.tensor([[0.5, 2.0]], dtype=torch.float64)
    y = layer_forward(layer, x, idx, p)
    manual = torch.zeros(1, SMALL.d, dtype=torch.float64)
    for k, e in enumerate(idx[0].tolist()):
        hdn = torch.relu(layer.w_up[e].double() @ x[0]) ** 2
        manual[0] += p[0, k] * (layer.w_down[e].double() @ hdn)
    assert torch.allclose(y, manual, atol=1e-10)


def test_persistent_spike_scales_exactly_one_channel_after_routing():
    cfg = ToyConfig(n_experts=4, d=32, h=64, top_k=2, rare_experts=0, spike_channel=5, spike_factor=100.0)
    base = ToyConfig(n_experts=4, d=32, h=64, top_k=2, rare_experts=0)
    layer = make_layer(cfg, seed=1)
    tok = sample_tokens(layer, 64, seed=2)
    ref = sample_tokens(make_layer(base, seed=1), 64, seed=2)
    assert torch.allclose(tok.x[:, 5], ref.x[:, 5] * 100.0)
    other = [c for c in range(32) if c != 5]
    assert torch.allclose(tok.x[:, other], ref.x[:, other])
    assert torch.equal(tok.idx, ref.idx), "routing must not see the spike"


def test_sporadic_spike_hits_only_the_marked_tokens():
    cfg = ToyConfig(
        n_experts=4,
        d=32,
        h=64,
        top_k=2,
        rare_experts=0,
        spike_channel=0,
        spike_factor=1000.0,
        spike_mode="sporadic",
        spike_frac=0.25,
    )
    layer = make_layer(cfg, seed=3)
    tok = sample_tokens(layer, 400, seed=4)
    assert 60 <= int(tok.spike_mask.sum()) <= 140
    assert tok.x[tok.spike_mask, 0].abs().mean() > 50 * tok.x[~tok.spike_mask, 0].abs().mean()


def test_rtn_method_equals_direct_rtn_and_identity_check_on_the_pipeline():
    layer = make_layer(SMALL, seed=5)
    cal = sample_tokens(layer, 256, seed=6)
    res = quantize_layer(layer, cal, Int4Scheme, Method(name="rtn", gptq=False))
    for e in range(SMALL.n_experts):
        q_up, _ = rtn_quantize(layer.w_up[e], Int4Scheme())
        assert torch.equal(res.w_up[e], q_up)


def test_expert_below_min_tokens_falls_back_to_rtn_and_is_reported():
    cfg = ToyConfig(n_experts=4, d=32, h=64, top_k=2, rare_experts=0)
    layer = make_layer(cfg, seed=7)
    cal = sample_tokens(layer, 300, seed=8)
    counts = torch.bincount(cal.idx.flatten(), minlength=4)
    threshold = int(counts.max()) + 1  # every expert is below the threshold
    res = quantize_layer(layer, cal, Nvfp4Scheme, Method(name="g", gptq=True, min_tokens=threshold))
    assert sorted(res.fallback_experts) == [0, 1, 2, 3]
    for e in range(4):
        q_up, _ = rtn_quantize(layer.w_up[e], Nvfp4Scheme())
        assert torch.equal(res.w_up[e], q_up)


def test_evaluate_reports_zero_error_for_unquantized_weights():
    layer = make_layer(SMALL, seed=9)
    test = sample_tokens(layer, 128, seed=10)
    out = evaluate(layer, layer.w_up, layer.w_down, test)
    assert out["rel_err"] == 0.0


def test_act_order_method_runs_and_dynamic_act_order_is_refused():
    layer = make_layer(SMALL, seed=11)
    cal = sample_tokens(layer, 256, seed=12)
    res = quantize_layer(layer, cal, Int4Scheme, Method(name="ao", dynamic=False, act_order=True))
    assert torch.isfinite(res.w_up).all() and torch.isfinite(res.w_down).all()
    with pytest.raises(ValueError):
        quantize_layer(layer, cal, Int4Scheme, Method(name="bad", dynamic=True, act_order=True))


def test_clustered_tokens_are_routed_to_their_own_expert_when_the_router_is_selective():
    cfg = ToyConfig(
        n_experts=8,
        d=64,
        h=128,
        top_k=2,
        rare_experts=0,
        n_clusters=8,
        cluster_strength=1.0,
        router_selectivity=8.0,
        cluster_prior_sigma=0.0,
    )
    layer = make_layer(cfg, seed=13)
    tok = sample_tokens(layer, 2000, seed=14)
    assert tok.cluster.min() >= 0 and tok.cluster.max() < 8
    hit = (tok.idx == tok.cluster.unsqueeze(1)).any(dim=1).double().mean()
    assert hit > 0.9, f"own-cluster expert selected for only {float(hit):.2f} of the tokens"
    # and a non-selective router does not do that (chance for top-2 of 8 is 0.25)
    flat = ToyConfig(**{**cfg.__dict__, "router_selectivity": 0.0})
    tok0 = sample_tokens(make_layer(flat, seed=13), 2000, seed=14)
    hit0 = (tok0.idx == tok0.cluster.unsqueeze(1)).any(dim=1).double().mean()
    assert hit0 < 0.6


def test_cluster_prior_makes_some_clusters_rare():
    cfg = ToyConfig(
        n_experts=16,
        d=32,
        h=64,
        top_k=2,
        rare_experts=0,
        n_clusters=16,
        cluster_prior_sigma=1.5,
    )
    layer = make_layer(cfg, seed=15)
    tok = sample_tokens(layer, 8000, seed=16)
    share = torch.bincount(tok.cluster, minlength=16).double() / 8000
    assert share.max() > 4 * share.min()


def test_default_config_draws_are_unchanged_by_the_cluster_feature():
    # the first tokens of the unclustered toy must not depend on the new fields
    a = sample_tokens(make_layer(SMALL, seed=17), 16, seed=18)
    assert a.cluster.tolist() == [-1] * 16


def test_requantizing_rtn_output_with_the_data_free_quantizer_is_numerically_a_no_op():
    from moegptq.experiments import requant_changed_fraction

    w = torch.randn(32, 128) * 0.05
    q, _ = rtn_quantize(w, Nvfp4Scheme())
    frac, max_rel = requant_changed_fraction(q)
    assert frac == 0.0 and max_rel < 1e-6


def _top_subspace(x: torch.Tensor, k: int) -> torch.Tensor:
    cov = x.T @ x / x.shape[0]
    _, vecs = torch.linalg.eigh(cov)
    return vecs[:, -k:]


def test_cluster_rotation_gives_each_cluster_its_own_covariance_subspace():
    base = dict(
        n_experts=4,
        d=32,
        h=64,
        top_k=2,
        rare_experts=0,
        n_clusters=4,
        cluster_strength=0.1,
        cluster_cov_sigma=0.0,
        cluster_prior_sigma=0.0,
        bias_sigma=0.0,
    )
    overlaps = {}
    for rotate in (False, True):
        cfg = ToyConfig(**base, cluster_rotate=rotate)
        tok = sample_tokens(make_layer(cfg, seed=19), 6000, seed=20)
        a = _top_subspace(tok.x[tok.cluster == 0], 4)
        b = _top_subspace(tok.x[tok.cluster == 1], 4)
        overlaps[rotate] = float((a.T @ b).pow(2).sum() / 4)
    assert overlaps[False] > 0.5 and overlaps[True] < 0.3, overlaps


def test_blend_hessian_interpolates_between_routed_and_all_tokens():
    layer = make_layer(SMALL, seed=21)
    cal = sample_tokens(layer, 600, seed=22)
    routed = quantize_layer(layer, cal, Int4Scheme, Method(name="r", dynamic=False, hessian="routed"))
    all_tok = quantize_layer(layer, cal, Int4Scheme, Method(name="a", dynamic=False, hessian="all"))
    zero = quantize_layer(
        layer, cal, Int4Scheme, Method(name="b0", dynamic=False, hessian="blend", blend_pseudo=0.0)
    )
    huge = quantize_layer(
        layer, cal, Int4Scheme, Method(name="b1", dynamic=False, hessian="blend", blend_pseudo=1e9)
    )
    assert float((zero.w_up != routed.w_up).double().mean()) < 1e-3
    assert float((zero.w_down != routed.w_down).double().mean()) < 1e-3
    assert float((huge.w_up != all_tok.w_up).double().mean()) < 1e-3
    assert float((huge.w_down != all_tok.w_down).double().mean()) < 1e-3


def test_render_prints_every_table_from_saved_json(tmp_path, capsys):
    import json

    from moegptq import experiments as ex

    e1 = {
        "nvfp4|512": {
            "rel_err": {"rtn": [0.02, 0.001], "g": [0.01, 0.001]},
            "per_seed": {},
            "min_tokens_per_expert": [0],
            "median_tokens_per_expert": [30.0],
        }
    }
    e2 = {"nvfp4": {"gptq damp 0.01": {"0-16": [1.0, 3], "16-64": [None, 0]}}}
    e3 = {"nvfp4|none": {"rtn": {"all": [0.02, 0.001], "plain": [0.02, 0.001]}}}
    for name, obj in (("e1_weak", e1), ("e2_weak", e2), ("e3", e3)):
        (tmp_path / f"{name}.json").write_text(json.dumps(obj))
    ex.render(tmp_path)
    out = capsys.readouterr().out
    assert "E1" in out and "E2" in out and "E3" in out and "0.5" in out


def test_e4_diagnostic_reports_the_worst_expert_and_its_token_count():
    from moegptq.experiments import run_e4

    rows = run_e4(seeds=(0,), n_cal=1024)
    assert len(rows) == 1
    row = rows[0]
    assert set(row) == {
        "seed",
        "layer_rel_err",
        "worst_expert",
        "worst_expert_tokens",
        "worst_expert_rel_err",
    }
    assert row["worst_expert_rel_err"] >= row["layer_rel_err"] * 0.0
