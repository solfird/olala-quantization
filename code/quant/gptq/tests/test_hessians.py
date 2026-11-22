"""Tests of hessians.py: the Hessian of an expert is the explicit sum over its routed tokens; guards; hooks; runner."""

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from moegptq.routing import route
from olala_quant.checkpoint_io import SourceCheckpoint

import hessians as hs
import tiny_model

E, D, M, K = 4, 8, 16, 2


def _synthetic(n: int = 240, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, D, generator=g)
    idx = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(n)])
    window = torch.arange(n) // 20  # 12 windows of 20 tokens
    w_up = torch.randn(E, M, D, generator=g) * 0.3
    w_down = torch.randn(E, D, M, generator=g) * 0.3
    return hs.LayerCapture(x, idx, window), w_up, w_down


def _brute_force(cap, w_up, w_down, e, keep=None):
    """Explicit python sums over (token, slot) pairs routed to expert e."""
    h_up = torch.zeros(D, D, dtype=torch.float64)
    h_down = torch.zeros(M, M, dtype=torch.float64)
    n = 0
    for t in range(cap.x.shape[0]):
        if keep is not None and not keep[t]:
            continue
        for s in range(cap.idx.shape[1]):
            if int(cap.idx[t, s]) == e:
                x = cap.x[t].double()
                h = torch.relu(w_up[e].double() @ x) ** 2
                h_up += torch.outer(x, x)
                h_down += torch.outer(h, h)
                n += 1
    return h_up, h_down, n


def _cfg(**kw):
    base = {"mode": "routed", "token_clip": None, "heldout_every": 0}
    base.update(kw)
    return hs.HessianConfig(**base)


def test_routed_hessian_equals_the_explicit_sum_over_routed_tokens() -> None:
    cap, w_up, w_down = _synthetic()
    eh = hs.build_hessians(cap, w_up, w_down, _cfg())
    assert eh.h_up.dtype == torch.float32 and eh.h_down.dtype == torch.float32
    for e in range(E):
        bu, bd, n = _brute_force(cap, w_up, w_down, e)
        assert int(eh.n_tokens[e]) == n
        torch.testing.assert_close(eh.h_up[e].double(), bu, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(eh.h_down[e].double(), bd, rtol=1e-4, atol=1e-4)


def test_heldout_windows_are_excluded_from_the_hessian_and_returned_as_rows() -> None:
    cap, w_up, w_down = _synthetic()
    mask = hs.heldout_windows(12, every=4)
    assert mask.nonzero().flatten().tolist() == [3, 7, 11]
    eh = hs.build_hessians(cap, w_up, w_down, _cfg(heldout_every=4))
    keep = ~mask[cap.window]
    for e in range(E):
        bu, _bd, n = _brute_force(cap, w_up, w_down, e, keep)
        assert int(eh.n_tokens[e]) == n
        torch.testing.assert_close(eh.h_up[e].double(), bu, rtol=1e-4, atol=1e-4)
        rows = cap.x[mask[cap.window] & (cap.idx == e).any(dim=1)]
        assert torch.equal(eh.held_x[e], rows)
        assert int(eh.n_heldout[e]) == rows.shape[0]


def test_no_heldout_when_disabled() -> None:
    assert not hs.heldout_windows(12, every=0).any()
    assert not hs.heldout_windows(3, every=4).any()


def test_token_clipping_caps_a_spike_token_at_three_times_the_median_norm() -> None:
    cap, w_up, w_down = _synthetic()
    spike = 5
    cap.x[spike] *= 200.0
    e = int(cap.idx[spike, 0])
    plain = hs.build_hessians(cap, w_up, w_down, _cfg())
    clipped = hs.build_hessians(cap, w_up, w_down, _cfg(token_clip=3.0))
    rows = (cap.idx == e).any(dim=1)
    norms = cap.x[rows].norm(dim=1)
    cap_norm = 3.0 * norms.median()
    cap.x[spike]
    # the clipped Hessian = the sum over the other tokens plus the spike scaled by (cap / norm)^2 (float64 oracle)
    xr = cap.x[rows].double()
    scale = torch.where(norms > cap_norm, (cap_norm / norms).square(), torch.ones_like(norms)).double()
    expected = ((xr * scale[:, None]).T @ xr).float()
    others = rows.clone()
    others[spike] = False
    # other tokens above 3x median (none expected in Gaussian data at this size) would also be clipped
    assert float(norms.max()) > float(cap_norm)  # the spike is clipped
    assert float((norms > cap_norm).sum()) == 1
    torch.testing.assert_close(clipped.h_up[e], expected, rtol=1e-3, atol=1e-3)
    assert float(clipped.h_up[e].diagonal().max()) < 0.01 * float(plain.h_up[e].diagonal().max())
    assert float(clipped.clip_frac_up[e]) == pytest.approx(1 / int(plain.n_tokens[e]))


def test_all_token_mode_uses_every_token_for_every_expert() -> None:
    cap, w_up, w_down = _synthetic()
    eh = hs.build_hessians(cap, w_up, w_down, _cfg(mode="all"))
    allx = cap.x.double()
    for e in range(E):
        torch.testing.assert_close(eh.h_up[e].double(), allx.T @ allx, rtol=1e-4, atol=1e-4)
        hid = torch.relu(allx @ w_up[e].double().T) ** 2
        torch.testing.assert_close(eh.h_down[e].double(), hid.T @ hid, rtol=1e-4, atol=1e-3)


def test_blend_mode_adds_one_pseudo_token_per_dimension_of_the_all_token_moment() -> None:
    cap, w_up, w_down = _synthetic()
    eh = hs.build_hessians(cap, w_up, w_down, _cfg(mode="blend", blend_pseudo=1.0))
    allx = cap.x.double()
    n = allx.shape[0]
    for e in range(E):
        bu, bd, _ = _brute_force(cap, w_up, w_down, e)
        hid = torch.relu(allx @ w_up[e].double().T) ** 2
        torch.testing.assert_close(eh.h_up[e].double(), bu + D * (allx.T @ allx) / n, rtol=1e-4, atol=1e-3)
        torch.testing.assert_close(eh.h_down[e].double(), bd + M * (hid.T @ hid) / n, rtol=1e-4, atol=1e-3)


def test_blend_excludes_heldout_windows_from_the_all_token_moment() -> None:
    cap, w_up, w_down = _synthetic()
    mask = hs.heldout_windows(12, every=4)
    keep = ~mask[cap.window]
    eh = hs.build_hessians(cap, w_up, w_down, _cfg(mode="blend", heldout_every=4))
    xk = cap.x[keep].double()
    e = 1
    bu, _, _ = _brute_force(cap, w_up, w_down, e, keep)
    torch.testing.assert_close(eh.h_up[e].double(), bu + D * (xk.T @ xk) / xk.shape[0], rtol=1e-4, atol=1e-3)


def test_minimum_token_rule_depends_on_the_mode() -> None:
    cap, w_up, w_down = _synthetic(n=40)
    routed = hs.build_hessians(cap, w_up, w_down, _cfg(mode="routed", min_tokens=1000))
    assert not routed.active.any()
    assert set(routed.inactive_reason) == {"fewer than 1000 routed tokens"}
    blend = hs.build_hessians(cap, w_up, w_down, _cfg(mode="blend", min_tokens=1))
    assert blend.active.all()
    none = hs.LayerCapture(cap.x, torch.zeros_like(cap.idx), cap.window)  # every token to expert 0 only (k=2: dup)
    eh = hs.build_hessians(none, w_up, w_down, _cfg(mode="blend", min_tokens=1))
    assert eh.active.tolist() == [True, False, False, False]
    assert eh.inactive_reason[1] == "no routed calibration token"


def test_default_minimum_is_sixteen_for_routed_and_one_otherwise() -> None:
    assert hs.HessianConfig(mode="routed").effective_min_tokens() == 16
    assert hs.HessianConfig(mode="blend").effective_min_tokens() == 1
    assert hs.HessianConfig(mode="all").effective_min_tokens() == 1
    assert hs.HessianConfig(mode="routed", min_tokens=5).effective_min_tokens() == 5


def test_unknown_mode_is_rejected() -> None:
    cap, w_up, w_down = _synthetic()
    with pytest.raises(ValueError):
        hs.build_hessians(cap, w_up, w_down, _cfg(mode="bogus"))


# ---------------------------------------------------------------------------------------------- hooks on the model


def _block_and_windows(tiny_modules, tiny_dir: Path, layer: int):
    modeling, config_mod = tiny_modules
    cfg = config_mod.OlalaConfig(**json.loads((tiny_dir / "config.json").read_text()))
    src = SourceCheckpoint(tiny_dir)
    return modeling, cfg, src, hs.load_block(modeling, cfg, src, layer, "cpu")


def test_capture_hook_records_the_latent_input_and_routing_of_the_expert_loop(tiny_modules, tiny_dir) -> None:
    _modeling, cfg, _src, block = _block_and_windows(tiny_modules, tiny_dir, 2)
    hidden = F.normalize(torch.randn(3, tiny_model.TINY_WINDOW, cfg.hidden_size).to(torch.bfloat16), dim=-1)
    seen: dict = {}
    h1 = block.mlp.register_forward_pre_hook(lambda m, a: seen.update(x=a[0].detach().clone()))
    cap = hs.MoECapture(block.mlp.experts)
    cap.set_windows(torch.tensor([4, 5, 6]), tiny_model.TINY_WINDOW)
    with torch.no_grad():
        block(hidden, position_ids=torch.arange(tiny_model.TINY_WINDOW)[None])
    h1.remove()
    got = cap.finish()
    moe = block.mlp
    xin = seen["x"].reshape(-1, cfg.hidden_size)
    # independent oracle: the router of the design study's prototype, and the latent projection
    logits = xin.float() @ moe.moe_gate.weight.float().T
    idx_ref, _ = route(logits, moe.expert_bias, cfg.moe_num_active_experts, cfg.moe_routed_scaling_factor)
    assert torch.equal(got.idx.sort(dim=1).values, idx_ref.sort(dim=1).values)
    assert torch.equal(got.x, moe.down_proj(xin).to(xin.dtype))
    assert got.window.tolist() == [4] * 64 + [5] * 64 + [6] * 64
    cap.close()


def test_capture_does_not_change_the_block_output(tiny_modules, tiny_dir) -> None:
    _modeling, cfg, _src, block = _block_and_windows(tiny_modules, tiny_dir, 1)
    hidden = F.normalize(torch.randn(2, tiny_model.TINY_WINDOW, cfg.hidden_size).to(torch.bfloat16), dim=-1)
    pos = torch.arange(tiny_model.TINY_WINDOW)[None]
    with torch.no_grad():
        ref = block(hidden, position_ids=pos)
        cap = hs.MoECapture(block.mlp.experts)
        cap.set_windows(torch.tensor([0, 1]), tiny_model.TINY_WINDOW)
        out = block(hidden, position_ids=pos)
        cap.close()
    assert torch.equal(ref, out)


def test_hessian_of_a_hooked_tiny_expert_equals_the_explicit_sum(tiny_modules, tiny_dir) -> None:
    _modeling, cfg, src, block = _block_and_windows(tiny_modules, tiny_dir, 3)
    windows = tiny_model.random_windows(4)
    hidden = hs.embed_windows(cfg, src.get_tensor("model.embedding.weight"), windows)
    cap = hs.MoECapture(block.mlp.experts)
    cap.set_windows(torch.arange(4), tiny_model.TINY_WINDOW)
    with torch.no_grad():
        block(hidden, position_ids=torch.arange(tiny_model.TINY_WINDOW)[None])
    got = cap.finish()
    cap.close()
    w_up = block.mlp.experts.experts.weight.detach()
    w_down = block.mlp.experts.output_experts.weight.detach()
    eh = hs.build_hessians(got, w_up, w_down, _cfg())
    for e in (0, 5, 11):
        h_up = torch.zeros(32, 32, dtype=torch.float64)
        h_dn = torch.zeros(64, 64, dtype=torch.float64)
        n = 0
        for t in range(got.x.shape[0]):
            for s in range(got.idx.shape[1]):
                if int(got.idx[t, s]) == e:
                    x = got.x[t]
                    h = F.linear(x, w_up[e].to(x.dtype))
                    h = F.relu(h).square().double()
                    h_up += torch.outer(x.double(), x.double())
                    h_dn += torch.outer(h, h)
                    n += 1
        assert int(eh.n_tokens[e]) == n > 0
        torch.testing.assert_close(eh.h_up[e].double(), h_up, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(eh.h_down[e].double(), h_dn, rtol=1e-3, atol=1e-8)


# ---------------------------------------------------------------------------------------------- sequential runner


class _Recorder:
    """Handler that captures the layers it is told to and records what the runner hands it."""

    def __init__(self, layers: set[int], weights=None) -> None:
        self.layers, self.infos, self.weights = layers, {}, weights or {}

    def needs_capture(self, layer: int) -> bool:
        return layer in self.layers

    def process(self, info):
        self.infos[info.layer] = info
        return self.weights.get(info.layer)


def _full_model_hidden(tiny_modules, tiny_dir: Path, windows: torch.Tensor) -> tuple[torch.Tensor, ...]:
    modeling, config_mod = tiny_modules
    cfg = config_mod.OlalaConfig(**json.loads((tiny_dir / "config.json").read_text()))
    model = modeling.OlalaForCausalLM(cfg)
    src = SourceCheckpoint(tiny_dir)
    sd = {k: src.get_tensor(k) for k in src.names()}
    model.load_state_dict(sd, strict=False, assign=True)
    model.eval()
    with torch.no_grad():
        out = model.model(input_ids=windows, output_hidden_states=True)
    return out.hidden_states


def test_runner_boundary_states_equal_the_full_model_forward(tiny_modules, tiny_dir) -> None:
    modeling, config_mod = tiny_modules
    cfg = config_mod.OlalaConfig(**json.loads((tiny_dir / "config.json").read_text()))
    windows = tiny_model.random_windows(4)
    ref = _full_model_hidden(tiny_modules, tiny_dir, windows)
    rec = _Recorder({1, 2, 3})
    src = SourceCheckpoint(tiny_dir)
    final = hs.run_sequential(src, cfg, windows, rec, device="cpu", batch_windows=2, modeling=modeling)
    assert set(rec.infos) == {1, 2, 3}
    for _layer, info in rec.infos.items():
        assert info.capture is not None and info.capture.x.shape[0] == 4 * tiny_model.TINY_WINDOW
        assert info.forward_seconds > 0
    torch.testing.assert_close(final.float(), ref[-1].float(), rtol=2e-2, atol=2e-2)


def test_runner_only_captures_the_layers_the_handler_asks_for(tiny_modules, tiny_dir) -> None:
    modeling, config_mod = tiny_modules
    cfg = config_mod.OlalaConfig(**json.loads((tiny_dir / "config.json").read_text()))
    rec = _Recorder({2})
    hs.run_sequential(
        SourceCheckpoint(tiny_dir),
        cfg,
        tiny_model.random_windows(2),
        rec,
        device="cpu",
        batch_windows=2,
        modeling=modeling,
    )
    assert set(rec.infos) == {2}


def test_runner_stops_after_last_layer(tiny_modules, tiny_dir) -> None:
    modeling, config_mod = tiny_modules
    cfg = config_mod.OlalaConfig(**json.loads((tiny_dir / "config.json").read_text()))
    rec = _Recorder({1, 2, 3})
    hs.run_sequential(
        SourceCheckpoint(tiny_dir),
        cfg,
        tiny_model.random_windows(2),
        rec,
        device="cpu",
        batch_windows=2,
        modeling=modeling,
        last_layer=1,
    )
    assert set(rec.infos) == {1}


def test_runner_resumes_from_the_saved_boundary_state(tiny_modules, tiny_dir, tmp_path) -> None:
    modeling, config_mod = tiny_modules
    cfg = config_mod.OlalaConfig(**json.loads((tiny_dir / "config.json").read_text()))
    windows = tiny_model.random_windows(2)
    src = SourceCheckpoint(tiny_dir)
    full = _Recorder({1, 2, 3})
    ref = hs.run_sequential(src, cfg, windows, full, device="cpu", batch_windows=2, modeling=modeling)

    class Boom(_Recorder):
        def process(self, info):
            if info.layer == 2:
                raise RuntimeError("interrupted")
            return super().process(info)

    state = tmp_path / "state"
    first = Boom({1, 2, 3})
    with pytest.raises(RuntimeError):
        hs.run_sequential(src, cfg, windows, first, device="cpu", batch_windows=2, modeling=modeling, state_dir=state)
    assert set(first.infos) == {1}
    progress = json.loads((state / "progress.json").read_text())
    assert progress["next_layer"] == 2
    second = _Recorder({2, 3})
    final = hs.run_sequential(
        src, cfg, windows, second, device="cpu", batch_windows=2, modeling=modeling, state_dir=state
    )
    assert set(second.infos) == {2, 3}  # layer 1 was not recomputed
    assert torch.equal(final, ref)
    assert json.loads((state / "progress.json").read_text())["next_layer"] == 4


def test_resume_refuses_a_state_made_for_other_windows(tiny_modules, tiny_dir, tmp_path) -> None:
    modeling, config_mod = tiny_modules
    cfg = config_mod.OlalaConfig(**json.loads((tiny_dir / "config.json").read_text()))
    src = SourceCheckpoint(tiny_dir)
    state = tmp_path / "state"
    hs.run_sequential(
        src,
        cfg,
        tiny_model.random_windows(2, seed=1),
        _Recorder({1}),
        device="cpu",
        batch_windows=2,
        modeling=modeling,
        state_dir=state,
        last_layer=1,
    )
    with pytest.raises(hs.ResumeError):
        hs.run_sequential(
            src,
            cfg,
            tiny_model.random_windows(2, seed=2),
            _Recorder({2}),
            device="cpu",
            batch_windows=2,
            modeling=modeling,
            state_dir=state,
        )


def test_quantized_propagation_feeds_the_next_layer_with_the_handler_weights(tiny_modules, tiny_dir) -> None:
    modeling, config_mod = tiny_modules
    cfg = config_mod.OlalaConfig(**json.loads((tiny_dir / "config.json").read_text()))
    windows = tiny_model.random_windows(2)
    src = SourceCheckpoint(tiny_dir)
    zero = hs.ExpertWeights(
        torch.zeros(16, 64, 32, dtype=torch.bfloat16), torch.zeros(16, 32, 64, dtype=torch.bfloat16)
    )
    same = _Recorder({1, 2})
    a = hs.run_sequential(src, cfg, windows, same, device="cpu", batch_windows=2, modeling=modeling, last_layer=2)
    changed = _Recorder({1, 2}, weights={1: zero})
    b = hs.run_sequential(
        src,
        cfg,
        windows,
        changed,
        device="cpu",
        batch_windows=2,
        modeling=modeling,
        last_layer=2,
        propagate="quantized",
    )
    assert not torch.equal(a, b)
    # with propagate='bf16' the handler's weights are ignored
    c = hs.run_sequential(
        src,
        cfg,
        windows,
        _Recorder({1, 2}, weights={1: zero}),
        device="cpu",
        batch_windows=2,
        modeling=modeling,
        last_layer=2,
        propagate="bf16",
    )
    assert torch.equal(a, c)


def test_runner_records_timings_and_branch_magnitudes_for_every_layer_it_runs(tiny_modules, tiny_dir, tmp_path) -> None:
    modeling, config_mod = tiny_modules
    cfg = config_mod.OlalaConfig(**json.loads((tiny_dir / "config.json").read_text()))
    state = tmp_path / "state"
    hs.run_sequential(
        SourceCheckpoint(tiny_dir),
        cfg,
        tiny_model.random_windows(2),
        _Recorder({2}),
        device="cpu",
        batch_windows=2,
        modeling=modeling,
        state_dir=state,
        last_layer=2,
    )
    rows = [json.loads(line) for line in (state / "forward_stats.jsonl").read_text().splitlines()]
    assert [r["layer"] for r in rows] == [0, 1, 2]  # layers 0 and 1 are only run forward, and still measured
    for r in rows:
        assert r["forward_seconds"] > 0 and r["mixer_seconds"] > 0 and r["mlp_seconds"] > 0
        assert r["max_abs_mixer_out"] > 0 and r["max_abs_mlp_out"] > 0
        assert r["captured"] == (r["layer"] == 2)


def test_check_vendored_matches_reports_missing_and_changed_files_as_different(tmp_path: Path) -> None:
    import shutil

    import olala_model

    empty = tmp_path / "empty"
    empty.mkdir()
    assert olala_model.check_vendored_matches(empty) == {"modeling_olala.py": False, "configuration_olala.py": False}
    same = tmp_path / "same"
    same.mkdir()
    for name in olala_model.VENDORED_FILES:
        shutil.copy2(olala_model.VENDOR_DIR / name, same / name)
    assert all(olala_model.check_vendored_matches(same).values())
    (same / "modeling_olala.py").write_text("# other\n")
    assert olala_model.check_vendored_matches(same) == {"modeling_olala.py": False, "configuration_olala.py": True}
