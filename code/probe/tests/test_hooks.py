"""Tests of probe_hooks.Probe on the tiny random Olala: values against independent captures, planted spikes, hygiene."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from conftest import TINY_SPECIAL

import probe_hooks as ph

SITES = {"mixer_out", "moe_out", "shared_out", "routed_out", "routed_latent", "moe_pre", "moe_post", "resid_out"}


def _run(model, ids, probe_kwargs=None, pre_hooks=None, prompt=0):
    """Plant (pre_hooks), attach a Probe, run one prompt; return (probe, hidden)."""
    for fn in pre_hooks or []:
        fn(model)
    probe = ph.Probe(model, max_len=ids.shape[1], **(probe_kwargs or {}))
    toks = ids[0].numpy()
    probe.begin_prompt(prompt, toks, np.isin(toks, TINY_SPECIAL))
    with torch.no_grad():
        hidden = model.model(input_ids=ids).last_hidden_state
    return probe, hidden


def _plant_head(layer, head, scale, position=None):
    def plant(model):
        blk = model.model.layers[layer]
        n_heads = blk.mixer.nheads

        def hook(_m, _a, out):
            y = out[0].clone()
            v = y.view(y.shape[0], y.shape[1], n_heads, -1)
            if position is None:
                v[:, :, head, :] *= scale
            else:
                v[:, position, head, :] *= scale
            return (y, *out[1:])

        blk.mixer.register_forward_hook(hook)

    return plant


def test_all_sites_exist_with_zero_based_layer_indices(tiny, tokens):
    probe, _ = _run(tiny, tokens)
    res = probe.result()
    assert [row["layer"] for row in res["layers"]] == [0, 1, 2, 3]
    assert [row["type"] for row in res["layers"]] == ["M", "M", "V", "M"]
    for row in res["layers"]:
        assert set(row["sites"]) == SITES
        assert (row["heads"] is not None) == (row["type"] == "M")
    probe.close()


def test_probe_does_not_change_the_forward_pass(tiny, tokens):
    with torch.no_grad():
        ref = tiny.model(input_ids=tokens).last_hidden_state
    probe, hidden = _run(tiny, tokens)
    assert torch.equal(ref, hidden)
    probe.close()


def test_site_maxima_match_independent_captures(tiny, tokens):
    cap = {}
    for i, blk in enumerate(tiny.model.layers):
        blk.mixer_proj.register_forward_hook(lambda _m, _a, o, i=i: cap.__setitem__((i, "mixer_out"), o))
        blk.mlp.register_forward_hook(lambda _m, _a, o, i=i: cap.__setitem__((i, "moe_out"), o))
        blk.mlp.shared_experts.register_forward_hook(lambda _m, _a, o, i=i: cap.__setitem__((i, "shared_out"), o))
        blk.mlp.up_proj.register_forward_hook(lambda _m, _a, o, i=i: cap.__setitem__((i, "routed_out"), o))
        blk.mlp.experts.register_forward_hook(lambda _m, _a, o, i=i: cap.__setitem__((i, "routed_latent"), o))
        blk.register_forward_hook(lambda _m, _a, o, i=i: cap.__setitem__((i, "resid_out"), o))
    probe, _ = _run(tiny, tokens)
    res = probe.result()
    for row in res["layers"]:
        for kind, expected in ((k, v) for (i, k), v in cap.items() if i == row["layer"]):
            got = row["sites"][kind]
            assert got["max_abs"] == pytest.approx(float(expected.abs().max()), rel=1e-6), (row["layer"], kind)
            assert got["rms"] == pytest.approx(float(expected.double().pow(2).mean().sqrt()), rel=1e-5)
    probe.close()


def test_planted_head_is_found_in_the_right_layer_and_head(tiny, tokens):
    probe, _ = _run(tiny, tokens, pre_hooks=[_plant_head(1, 5, 1e6)])
    res = probe.result()
    top = {row["layer"]: row["heads"]["top_head"] for row in res["layers"] if row["heads"]}
    assert top[1] == 5
    ratios = {row["layer"]: row["heads"]["dominance_ratio"] for row in res["layers"] if row["heads"]}
    assert ratios[1] > 1e4
    assert max(ratios[0], ratios[3]) < 100
    maxima = {row["layer"]: row["sites"]["mixer_out"]["max_abs"] for row in res["layers"] if row["heads"]}
    assert max(maxima, key=maxima.get) == 1
    assert maxima[1] > 1e3 * max(maxima[0], maxima[3])
    probe.close()


def test_planted_token_is_found_at_the_right_position_and_channel_range(tiny, tokens):
    probe, _ = _run(tiny, tokens, pre_hooks=[_plant_head(3, 2, 1e7, position=17)])
    row = probe.result()["layers"][3]
    assert row["heads"]["top_head"] == 2
    arg = row["heads"]["top_head_argmax"]
    assert arg["position"] == 17 and arg["prompt"] == 0
    assert 0 <= arg["channel"] < 16  # the channel is counted inside the head (16 channels per head)
    assert row["sites"]["mixer_out"]["argmax"]["position"] == 17
    assert row["sites"]["mixer_out"]["argmax"]["token_id"] == int(tokens[0, 17])
    probe.close()


def test_special_and_first_token_flags_follow_the_planted_position(tiny, tokens):
    probe, _ = _run(tiny, tokens, pre_hooks=[_plant_head(1, 0, 1e7, position=9)])
    arg = probe.result()["layers"][1]["sites"]["mixer_out"]["argmax"]
    assert arg["position"] == 9 and arg["is_special"] is True and arg["is_first"] is False
    probe.close()


def test_first_token_flag(tiny, tokens):
    probe, _ = _run(tiny, tokens, pre_hooks=[_plant_head(1, 0, 1e7, position=0)])
    arg = probe.result()["layers"][1]["sites"]["mixer_out"]["argmax"]
    assert arg["position"] == 0 and arg["is_first"] is True and arg["is_special"] is True
    probe.close()


def test_negative_spike_is_found_by_absolute_value(tiny, tokens):
    probe, _ = _run(tiny, tokens, pre_hooks=[_plant_head(1, 4, -1e6)])
    row = probe.result()["layers"][1]
    assert row["heads"]["top_head"] == 4
    assert row["heads"]["dominance_ratio"] > 1e4
    probe.close()


def test_sibling_erasure_fraction_is_high_when_one_head_dominates_and_low_otherwise(tiny, tiny_base, tokens):
    probe, _ = _run(tiny, tokens, pre_hooks=[_plant_head(1, 5, 1e9)])
    planted = probe.result()["layers"][1]["mixer_erasure"]
    assert planted["fraction_bit_identical"] > 0.9
    assert planted["n_total"] == tokens.shape[1] * tiny.config.hidden_size
    probe.close()
    import copy

    natural = copy.deepcopy(tiny_base)
    probe2, _ = _run(natural, tokens)
    assert probe2.result()["layers"][1]["mixer_erasure"]["fraction_bit_identical"] < 0.5
    probe2.close()


def test_moe_activation_sites_match_a_brute_force_recomputation(tiny, tokens):
    captured = {}
    for i, blk in enumerate(tiny.model.layers):
        blk.mlp.experts.register_forward_pre_hook(
            lambda _m, a, i=i: captured.__setitem__(i, tuple(t.clone() for t in a))
        )
    probe, _ = _run(tiny, tokens)
    res = probe.result()
    for i, blk in enumerate(tiny.model.layers):
        x0, _p, idx = captured[i]
        w_in = blk.mlp.experts.experts.weight
        best_pre = best_post = 0.0
        best_expert = -1
        per_expert_max = {}
        for t in range(x0.shape[0]):
            for e in idx[t].tolist():
                pre = w_in[e] @ x0[t]
                post = F.relu(pre).square()
                best_pre = max(best_pre, float(pre.detach().abs().max()))
                if float(post.detach().max()) > best_post:
                    best_post, best_expert = float(post.detach().max()), e
                per_expert_max[e] = max(per_expert_max.get(e, 0.0), float(post.detach().max()))
        sites = res["layers"][i]["sites"]
        assert sites["moe_pre"]["max_abs"] == pytest.approx(best_pre, rel=1e-5)
        assert sites["moe_post"]["max_abs"] == pytest.approx(best_post, rel=1e-5)
        ex = res["layers"][i]["experts"]
        assert ex["top_expert_by_post"] == best_expert
        assert ex["max_post_per_expert"][best_expert] == pytest.approx(best_post, rel=1e-5)
        for e, v in per_expert_max.items():
            assert ex["max_post_per_expert"][e] == pytest.approx(v, rel=1e-5)
    probe.close()


def test_planted_expert_is_the_top_of_the_expert_histogram(tiny, tokens):
    with torch.no_grad():
        tiny.model.layers[2].mlp.experts.experts.weight[7] *= 1e4
    probe, _ = _run(tiny, tokens)
    ex = probe.result()["layers"][2]["experts"]
    assert ex["top_expert_by_post"] == 7
    assert ex["top_pairs"][0]["expert"] == 7
    entry = next(r for r in ex["expert_hist_top_pairs"] if r["expert"] == 7)
    n_pairs_of_7 = ex["n_pairs_per_expert"][7]
    assert entry["count"] == min(n_pairs_of_7, 32) and entry["max_post"] > 1e6
    other = probe.result()["layers"][0]["experts"]
    assert other["top_expert_by_post"] != 7 or other["max_post_per_expert"][7] < 1e6
    probe.close()


def test_moe_up_projection_amplification_is_recorded(tiny, tokens):
    probe, _ = _run(tiny, tokens)
    for row in probe.result()["layers"]:
        s = row["sites"]
        assert s["routed_out"]["max_abs"] > 0 and s["routed_latent"]["max_abs"] > 0


def test_close_restores_the_model(tiny, tokens):
    orig = [blk.mlp.experts.activation for blk in tiny.model.layers]
    probe, _ = _run(tiny, tokens)
    probe.close()
    assert [blk.mlp.experts.activation for blk in tiny.model.layers] == orig
    for blk in tiny.model.layers:
        assert not blk._forward_hooks and not blk.mixer_proj._forward_pre_hooks and not blk.mlp._forward_hooks
        assert not blk.mlp.experts._forward_pre_hooks


def test_several_prompts_are_pooled(tiny, tokens):
    probe = ph.Probe(tiny, max_len=tokens.shape[1])
    for k in range(3):
        toks = tokens[0].numpy()
        probe.begin_prompt(k, toks, np.isin(toks, TINY_SPECIAL))
        with torch.no_grad():
            tiny.model(input_ids=tokens)
    res = probe.result()
    assert res["layers"][0]["sites"]["resid_out"]["n_prompts"] == 3
    assert res["layers"][0]["sites"]["resid_out"]["n_tokens"] == 3 * tokens.shape[1]
    probe.close()


def test_non_torch_moe_is_refused(tiny):
    tiny.model.layers[0].mlp.experts = torch.nn.Identity()
    with pytest.raises(RuntimeError, match="TorchMoEMLP"):
        ph.Probe(tiny, max_len=8)
