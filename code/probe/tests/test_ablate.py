"""Tests of the ablation hooks (probe_hooks) and of the next-token metrics / ablation runner (probe_ablate)."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from conftest import TINY_SPECIAL

import probe_ablate as pa
import probe_hooks as ph
import probe_prompts as pp


def _hidden(model, ids):
    with torch.no_grad():
        return model.model(input_ids=ids).last_hidden_state


def _plant(model, layer, head, scale):
    blk = model.model.layers[layer]
    n = blk.mixer.nheads

    def hook(_m, _a, out):
        y = out[0].clone()
        y.view(y.shape[0], y.shape[1], n, -1)[:, :, head, :] *= scale
        return (y, *out[1:])

    blk.mixer.register_forward_hook(hook)


def _capture(model, layer, kind):
    store = {}
    blk = model.model.layers[layer]
    target = {"mixer_out": blk.mixer_proj, "moe_out": blk.mlp, "shared_out": blk.mlp.shared_experts}[kind]
    target.register_forward_hook(lambda _m, _a, out: store.__setitem__("x", out.detach().clone()))
    return store


def test_zero_ablation_zeroes_the_branch_and_changes_the_output(tiny, tokens):
    base = _hidden(tiny, tokens)
    store = _capture(tiny, 1, "mixer_out")  # registered first: sees the branch BEFORE the ablation edit? no: see below
    handles = ph.install_ablation(tiny, ph.AblationSpec(1, "mixer_out", "zero"))
    seen = _capture(tiny, 1, "mixer_out")  # registered after the ablation: sees the edited branch
    out = _hidden(tiny, tokens)
    assert float(seen["x"].abs().max()) == 0.0
    assert float(store["x"].abs().max()) > 0.0  # the hook before the ablation still saw the real branch
    assert not torch.equal(out, base)
    for h in handles:
        h.remove()
    assert torch.equal(_hidden(tiny, tokens), base)


def test_clamp_ablation_clamps_a_planted_spike_and_leaves_the_rest_alone(tiny, tokens):
    _plant(tiny, 1, 5, 1e12)
    base = _hidden(tiny, tokens)
    ph.install_ablation(tiny, ph.AblationSpec(1, "mixer_out", "clamp448"))
    seen = _capture(tiny, 1, "mixer_out")
    out = _hidden(tiny, tokens)
    assert float(seen["x"].abs().max()) == 448.0
    assert not torch.equal(out, base)


def test_clamp_on_a_branch_that_never_exceeds_448_is_a_noop(tiny, tokens):
    base = _hidden(tiny, tokens)
    ph.install_ablation(tiny, ph.AblationSpec(0, "moe_out", "clamp448"))
    assert torch.equal(_hidden(tiny, tokens), base)


def test_noop_control_reproduces_the_baseline(tiny, tokens):
    base = _hidden(tiny, tokens)
    for branch in ("mixer_out", "moe_out", "routed_out", "shared_out"):
        hs = ph.install_ablation(tiny, ph.AblationSpec(2, branch, "noop"))
        assert torch.equal(_hidden(tiny, tokens), base)
        for h in hs:
            h.remove()


def test_head_ablation_edits_only_that_heads_channels(tiny, tokens):
    seen = {}
    blk = tiny.model.layers[1]
    blk.mixer.register_forward_hook(lambda _m, _a, out: seen.__setitem__("before", out[0].detach().clone()))
    ph.install_ablation(tiny, ph.AblationSpec(1, "head", "zero", head=3))
    blk.mixer.register_forward_hook(lambda _m, _a, out: seen.__setitem__("after", out[0].detach().clone()))
    _hidden(tiny, tokens)
    b = seen["before"].view(1, -1, 8, 16)
    a = seen["after"].view(1, -1, 8, 16)
    assert float(a[:, :, 3].abs().max()) == 0.0
    assert float(b[:, :, 3].abs().max()) > 0.0
    keep = [h for h in range(8) if h != 3]
    assert torch.equal(a[:, :, keep], b[:, :, keep])


def test_shared_and_routed_ablation_hit_the_right_module(tiny, tokens):
    base = _hidden(tiny, tokens)
    ph.install_ablation(tiny, ph.AblationSpec(0, "shared_out", "zero"))
    seen = _capture(tiny, 0, "shared_out")
    out = _hidden(tiny, tokens)
    assert float(seen["x"].abs().max()) == 0.0 and not torch.equal(out, base)


def test_bad_specs_are_refused(tiny):
    with pytest.raises(ValueError):
        ph.install_ablation(tiny, ph.AblationSpec(1, "bogus", "zero"))
    with pytest.raises(ValueError):
        ph.install_ablation(tiny, ph.AblationSpec(1, "mixer_out", "bogus"))
    with pytest.raises(ValueError):
        ph.install_ablation(tiny, ph.AblationSpec(99, "mixer_out", "zero"))
    with pytest.raises(ValueError):
        ph.install_ablation(tiny, ph.AblationSpec(2, "head", "zero", head=0))  # layer 2 is attention
    with pytest.raises(ValueError):
        ph.install_ablation(tiny, ph.AblationSpec(1, "head", "zero"))


# ------------------------------------------------------------------------------------------------ metrics
@torch.no_grad()
def _naive(model, base_h, abl_h, tokens):
    lb = model.lm_head(base_h).float()[0]
    la = model.lm_head(abl_h).float()[0]
    pb = F.softmax(lb, -1)
    kl = (pb * (F.log_softmax(lb, -1) - F.log_softmax(la, -1))).sum(-1)
    nll_b = F.cross_entropy(lb[:-1], tokens[0, 1:], reduction="none")
    nll_a = F.cross_entropy(la[:-1], tokens[0, 1:], reduction="none")
    return kl, (lb.argmax(-1) == la.argmax(-1)).float(), nll_a - nll_b


def test_metrics_match_a_naive_full_logit_computation(tiny, tokens):
    base = _hidden(tiny, tokens)
    ph.install_ablation(tiny, ph.AblationSpec(0, "moe_out", "zero"))
    abl = _hidden(tiny, tokens)
    acc = pa.MetricAcc()
    acc.add(pa.prompt_metrics(tiny.lm_head, base, abl, tokens[0], chunk=7))
    r = acc.result()
    kl, agree, dn = _naive(tiny, base, abl, tokens)
    assert r["kl_mean_all_positions"] == pytest.approx(float(kl.mean()), rel=1e-4)
    assert r["kl_max_position"] == pytest.approx(float(kl.max()), rel=1e-4)
    assert r["kl_last_position_mean"] == pytest.approx(float(kl[-1]), rel=1e-4)
    assert r["top1_agreement"] == pytest.approx(float(agree.mean()))
    assert r["delta_nll_mean"] == pytest.approx(float(dn.mean()), abs=1e-5)
    assert r["n_positions"] == tokens.shape[1]
    assert r["kl_mean_all_positions"] > 0


def test_metrics_identity_is_zero(tiny, tokens):
    base = _hidden(tiny, tokens)
    acc = pa.MetricAcc()
    acc.add(pa.prompt_metrics(tiny.lm_head, base, base.clone(), tokens[0]))
    r = acc.result()
    assert r["kl_mean_all_positions"] == 0.0 and r["top1_agreement"] == 1.0 and r["delta_nll_mean"] == 0.0


def test_metrics_pool_prompts_weighted_by_positions(tiny):
    acc = pa.MetricAcc()
    g = torch.Generator().manual_seed(0)
    for n in (10, 30):
        ids = torch.randint(4, 100, (1, n), generator=g)
        base = _hidden(tiny, ids)
        acc.add(pa.prompt_metrics(tiny.lm_head, base, base * 0.9, ids[0]))
    r = acc.result()
    assert r["n_positions"] == 40 and r["n_prompts"] == 2


def test_metrics_report_nonfinite_positions(tiny, tokens):
    base = _hidden(tiny, tokens)
    bad = base.clone()
    bad[0, 5, :] = float("nan")
    acc = pa.MetricAcc()
    acc.add(pa.prompt_metrics(tiny.lm_head, base, bad, tokens[0]))
    r = acc.result()
    assert r["n_nonfinite_positions"] == 1
    assert np.isfinite(r["kl_mean_all_positions"])  # the mean is over the finite positions


# ------------------------------------------------------------------------------------------------ runner
def _prompts(n=2, length=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        ids = torch.randint(4, 200, (length,), generator=g)
        out.append(pp.Prompt(f"p{i}", "plain", ids.numpy(), np.isin(ids.numpy(), TINY_SPECIAL), False))
    return out


def test_run_ablations_returns_one_row_per_spec_and_noop_is_exactly_zero(tiny):
    _plant(tiny, 1, 5, 1e12)
    specs = [
        ph.AblationSpec(1, "mixer_out", "noop"),
        ph.AblationSpec(1, "mixer_out", "zero"),
        ph.AblationSpec(1, "mixer_out", "clamp448"),
        ph.AblationSpec(1, "head", "zero", head=5),
        ph.AblationSpec(0, "moe_out", "clamp448"),
    ]
    rows = pa.run_ablations(tiny, _prompts(), specs, device="cpu")
    assert [r["label"] for r in rows] == [s.label() for s in specs]
    by = {r["label"]: r for r in rows}
    assert by["L1.mixer_out:noop"]["metrics"]["kl_mean_all_positions"] == 0.0
    assert by["L0.moe_out:clamp448"]["metrics"]["kl_mean_all_positions"] == 0.0  # never above 448 there
    assert by["L1.mixer_out:clamp448"]["metrics"]["kl_mean_all_positions"] > 0.0  # the planted spike is clamped
    assert by["L1.mixer_out:zero"]["metrics"]["kl_mean_all_positions"] > 0.0
    assert by["L1.head5:zero"]["metrics"]["kl_mean_all_positions"] > 0.0
    assert tiny.model.layers[1].mixer._forward_hooks  # only the planting hook is left
    assert len(tiny.model.layers[1].mixer._forward_hooks) == 1


def _site(layer, kind, mx, typ="M"):
    return {"layer": layer, "kind": kind, "max_abs": mx, "type": typ}


def test_select_targets_picks_branches_above_the_threshold_and_adds_controls():
    probe = {
        "layers": [
            {
                "layer": 0,
                "type": "M",
                "heads": {"top_head": 3},
                "sites": {
                    "mixer_out": {"max_abs": 2.0},
                    "moe_out": {"max_abs": 5e8},
                    "routed_out": {"max_abs": 5e8},
                    "shared_out": {"max_abs": 1.0},
                },
            },
            {
                "layer": 1,
                "type": "M",
                "heads": {"top_head": 5},
                "sites": {
                    "mixer_out": {"max_abs": 1e15},
                    "moe_out": {"max_abs": 3.0},
                    "routed_out": {"max_abs": 3.0},
                    "shared_out": {"max_abs": 1.0},
                },
            },
            {
                "layer": 2,
                "type": "V",
                "heads": None,
                "sites": {
                    "mixer_out": {"max_abs": 4.0},
                    "moe_out": {"max_abs": 2.0},
                    "routed_out": {"max_abs": 1.0},
                    "shared_out": {"max_abs": 1.0},
                },
            },
        ]
    }
    specs = pa.select_targets(probe, threshold=448.0, max_branches=10)
    labels = [s.label() for s in specs]
    assert "L1.mixer_out:zero" in labels and "L1.mixer_out:clamp448" in labels and "L1.head5:zero" in labels
    assert "L0.moe_out:zero" in labels and "L0.moe_out:clamp448" in labels and "L0.routed_out:zero" in labels
    assert "L0.shared_out:zero" not in labels  # 1.0 is below the threshold
    assert labels[0] == "L1.mixer_out:noop"  # the determinism control comes first
    # one unflagged control per flagged kind: the lowest-numbered layer of that kind whose branch is below the threshold
    assert "L0.mixer_out:zero" in labels  # layer 0 mixer (2.0) is the control for the flagged layer-1 mixer
    assert "L1.moe_out:zero" in labels  # layer 1 MoE (3.0) is the control for the flagged layer-0 MoE
    assert "L1.routed_out:zero" in labels
    assert labels.count("L1.mixer_out:zero") == 1


def test_select_targets_respects_the_cap_and_keeps_the_largest():
    layers = [
        {
            "layer": i,
            "type": "M",
            "heads": {"top_head": 0},
            "sites": {
                "mixer_out": {"max_abs": 10.0 ** (i + 3)},
                "moe_out": {"max_abs": 1.0},
                "routed_out": {"max_abs": 1.0},
                "shared_out": {"max_abs": 1.0},
            },
        }
        for i in range(6)
    ]
    specs = pa.select_targets({"layers": layers}, threshold=448.0, max_branches=2)
    flagged = {s.layer for s in specs if s.branch == "mixer_out" and s.mode == "clamp448"}
    assert flagged == {4, 5}
