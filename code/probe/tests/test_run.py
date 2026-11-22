"""Tests of run_probe / loading / the command line, on the tiny random Olala (CPU)."""

import json

import numpy as np
import pytest
import torch
from conftest import TINY_SPECIAL

import activation_probe as ap
import probe_prompts as pp


def _prompts(n=3, length=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        ids = torch.randint(4, 200, (length,), generator=g)
        ids[0] = 1
        out.append(pp.Prompt(f"p{i}", "plain", ids.numpy(), np.isin(ids.numpy(), TINY_SPECIAL), False))
    return out


def _plant(model, layer, head, scale):
    blk = model.model.layers[layer]
    n = blk.mixer.nheads

    def hook(_m, _a, out):
        y = out[0].clone()
        y.view(y.shape[0], y.shape[1], n, -1)[:, :, head, :] *= scale
        return (y, *out[1:])

    blk.mixer.register_forward_hook(hook)


def test_run_probe_finds_the_planted_spike_and_ablates_it(tiny):
    _plant(tiny, 1, 6, 1e12)
    res = ap.run_probe(tiny, _prompts(), "cpu", pp.ByteChatTokenizer(), ablation_prompts=2, max_ablate_branches=4)
    meta = res["meta"]
    assert meta["attention_layers_0based"] == [2] and "0-based" in meta["layer_indexing"]
    assert meta["mamba_layers_0based"] == [0, 1, 3] and meta["mamba_heads_per_layer"] == 8
    row = res["layers"][1]
    assert row["heads"]["top_head"] == 6
    labels = [a["label"] for a in res["ablations"]]
    assert "L1.mixer_out:zero" in labels and "L1.mixer_out:clamp448" in labels and "L1.head6:zero" in labels
    assert labels[0].endswith(":noop")
    noop = res["ablations"][0]["metrics"]
    assert noop["kl_mean_all_positions"] == 0.0
    assert res["ablation_prompt_ids"] == ["p0", "p2"]
    assert res["token_text"]  # decoded text of the tokens that the result refers to
    assert meta["memory_estimate_gib_4096_tokens"]["flash"]["total"] > 1.0  # the tiny model: the CUDA context dominates
    json.dumps(ap._jsonable(res), allow_nan=False)  # strictly serialisable


def test_run_probe_without_spikes_runs_no_ablation(tiny_base):
    import copy

    res = ap.run_probe(copy.deepcopy(tiny_base), _prompts(2), "cpu", None, spike_threshold=1e30)
    assert res["ablations"] == [] and "token_text" not in res


def test_jsonable_handles_nonfinite_and_numpy():
    out = ap._jsonable({"a": float("nan"), "b": [float("inf"), -float("inf")], "c": np.float32(1.5), "d": np.arange(2)})
    assert out == {"a": "NaN", "b": ["Infinity", "-Infinity"], "c": 1.5, "d": [0, 1]}


def test_memory_estimate_is_monotonic_and_fits_the_real_config_on_a_32_gb_card():
    import olala_model

    _, configuration = olala_model.load_olala_modules()
    cfg = configuration.OlalaConfig(**json.loads((ap.GPTQ_DIR / "vendor" / "config.json").read_text()))
    small, big = ap.estimate_memory_gib(cfg, 1024), ap.estimate_memory_gib(cfg, 4096)
    assert big["total"] > small["total"]
    assert 12.0 < big["weights"] < 13.5  # 13.6 GB on disk = 12.7 GiB
    assert big["total"] < 29.0  # fits a 32 GB card with flash / flex attention
    eager = ap.estimate_memory_gib(cfg, 4096, attn="eager")
    assert eager["eager_attention_scores"] > 5 and eager["total"] > big["total"] + 5


# ------------------------------------------------------------------------------------------------ loading
@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    import tiny_model

    saved = dict(tiny_model.TINY_OVERRIDES)
    tiny_model.TINY_OVERRIDES["mamba_headdim"] = ap.TINY_HEADDIM
    try:
        return tiny_model.write_tiny_checkpoint(tmp_path_factory.mktemp("tiny-ckpt"), seed=0)
    finally:
        tiny_model.TINY_OVERRIDES.clear()
        tiny_model.TINY_OVERRIDES.update(saved)


def test_load_model_keeps_checkpoint_dtypes_and_values(tiny_dir):
    from safetensors import safe_open

    model = ap.load_model(tiny_dir, "cpu")
    with safe_open(str(tiny_dir / "model.safetensors"), "pt") as f:
        for name in f.keys():  # noqa: SIM118
            t = f.get_tensor(name)
            got = model.state_dict()[name]
            assert got.dtype == t.dtype, name
            assert torch.equal(got.reshape(t.shape), t), name
    assert not [n for n, t in [*model.named_parameters(), *model.named_buffers()] if t.is_meta]
    ids = torch.randint(4, 100, (1, 20))
    with torch.no_grad():
        h = model.model(input_ids=ids).last_hidden_state
    assert torch.isfinite(h.float()).all()


def test_load_model_refuses_a_modified_modeling_file(tiny_dir, tmp_path):
    import shutil

    dst = tmp_path / "ckpt"
    shutil.copytree(tiny_dir, dst)
    (dst / "modeling_olala.py").write_text((dst / "modeling_olala.py").read_text() + "\n# edited\n")
    with pytest.raises(RuntimeError, match="differ"):
        ap.load_model(dst, "cpu")


def test_load_model_refuses_missing_tensors(tiny_dir, tmp_path):
    import shutil

    from safetensors.torch import load_file, save_file

    dst = tmp_path / "ckpt"
    shutil.copytree(tiny_dir, dst)
    state = load_file(str(dst / "model.safetensors"))
    state.pop("lm_head.weight")
    save_file(state, str(dst / "model.safetensors"), metadata={"format": "pt"})
    with pytest.raises(RuntimeError, match="missing"):
        ap.load_model(dst, "cpu")


# ------------------------------------------------------------------------------------------------ command line
def test_cli_tiny_writes_strict_json_and_markdown(tmp_path):
    rc = ap.main(
        [
            "--tiny",
            "--out",
            str(tmp_path),
            "--quiet",
            "--spike-threshold",
            "0.0001",
            "--ablation-prompts",
            "2",
            "--max-ablate-branches",
            "2",
        ]
    )
    assert rc == 0
    data = json.loads((tmp_path / "activation_probe.json").read_text(), parse_constant=lambda c: pytest.fail(c))
    md = (tmp_path / "activation_probe.md").read_text()
    assert data["meta"]["tiny"] is True and data["meta"]["n_prompts"] == 4
    assert "## Claim by claim" in md and "0-based" in md
    assert data["ablations"] and data["ablations"][0]["label"].endswith(":noop")


def test_cli_requires_model_dir_and_prompts_without_tiny(tmp_path):
    with pytest.raises(SystemExit):
        ap.main(["--out", str(tmp_path)])


def test_cli_refuses_cuda_when_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert ap.main(["--tiny", "--device", "cuda", "--out", str(tmp_path), "--quiet"]) == 2


def test_cli_max_tokens_truncates(tmp_path):
    assert ap.main(["--tiny", "--out", str(tmp_path), "--quiet", "--max-tokens", "12", "--ablation-prompts", "1"]) == 0
    data = json.loads((tmp_path / "activation_probe.json").read_text())
    assert data["meta"]["max_tokens_in_a_prompt"] == 12
    assert all(p["n_tokens"] <= 12 for p in data["prompts"]) and any(p["truncated"] for p in data["prompts"])


def test_cli_reads_a_custom_prompt_file(tmp_path):
    f = tmp_path / "p.jsonl"
    pp.write_prompts(f, [{"id": "only", "kind": "plain", "text": "hello world " * 5}])
    assert ap.main(["--tiny", "--prompts", str(f), "--out", str(tmp_path / "o"), "--quiet"]) == 0
    data = json.loads((tmp_path / "o" / "activation_probe.json").read_text())
    assert [p["id"] for p in data["prompts"]] == ["only"]
