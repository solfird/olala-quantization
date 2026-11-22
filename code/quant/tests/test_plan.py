"""Tests for recipes and the conversion plan, on the REAL safetensors header of the model.

The header (945 tensors, shapes and dtypes only) is copied from recon/src/hf/header.json, so
no weight is needed. Expected numbers come from recon/quantizability.md section A.
"""

import json
from pathlib import Path

import pytest

from olala_quant import recipes
from olala_quant.checkpoint_io import TensorMeta, header_from_json
from olala_quant.plan import IgnoreListError, build_plan, load_ignore_file

DATA = Path(__file__).parent / "data"
TOTAL_PARAMS_INCL_SCALARS = 6_773_124_360  # recon A: header sum (hub counts 216 fewer 0-dim scalars)
QUANTIZABLE_PARAMS = 5_884_084_224  # routed + shared + latent + mixer_proj + attn gate_proj


@pytest.fixture(scope="module")
def header() -> dict[str, TensorMeta]:
    return header_from_json(json.loads((DATA / "olala_header.json").read_text()))


def test_header_totals_match_recon(header: dict[str, TensorMeta]) -> None:
    assert len(header) == 945
    assert sum(m.numel for m in header.values()) == TOTAL_PARAMS_INCL_SCALARS


def test_recipes_are_data_with_declared_sets() -> None:
    assert set(recipes.RECIPES) >= {"fp8", "nvfp4", "nvfp4_w4a4"}
    fp8, nvfp4, w4a4 = (recipes.RECIPES[k] for k in ("fp8", "nvfp4", "nvfp4_w4a4"))
    assert fp8.implemented and nvfp4.implemented and not w4a4.implemented
    assert fp8.quantizer == "fp8" and nvfp4.quantizer == "nvfp4"
    assert nvfp4.activations == "bf16" and w4a4.activations == "nvfp4"
    assert w4a4.calibration_needs, "the W4A4 recipe must say precisely what it still needs"
    assert fp8.suffixes == ("weight", "weight_scale")
    assert nvfp4.suffixes == ("weight_packed", "weight_scale", "weight_global_scale")
    with pytest.raises(KeyError):
        recipes.get_recipe("int4")


def test_quantizable_set_without_ignore_list(header: dict[str, TensorMeta]) -> None:
    plan = build_plan(header, recipes.RECIPES["fp8"], ())
    quantized = [t for t in plan.tensors if t.category == "quantized"]
    assert len(quantized) == 259
    assert sum(t.meta.numel for t in quantized) == QUANTIZABLE_PARAMS
    by_role: dict[str, int] = {}
    for t in quantized:
        by_role[t.role] = by_role.get(t.role, 0) + 1
    assert by_role == {
        "routed_up": 36,
        "routed_down": 36,
        "shared_up": 36,
        "shared_down": 36,
        "latent_reduce": 36,
        "latent_expand": 36,
        "mixer_proj": 36,
        "attn_gate_proj": 7,
    }
    assert plan.ignored_quantizable == ()
    # 86.9 % of all parameters (recon A.2)
    assert pytest.approx(0.8688, abs=1e-4) == QUANTIZABLE_PARAMS / TOTAL_PARAMS_INCL_SCALARS


def test_every_kept_tensor_has_a_named_category(header: dict[str, TensorMeta]) -> None:
    plan = build_plan(header, recipes.RECIPES["fp8"], ())
    counts: dict[str, int] = {}
    for t in plan.tensors:
        counts[t.role] = counts.get(t.role, 0) + 1
    assert "unclassified" not in counts
    assert counts["embedding"] == 1
    assert counts["lm_head"] == 1
    assert counts["mamba_in_proj"] == 29
    assert counts["mamba_in_proj_dyn"] == 29
    assert counts["attn_c_q"] == 7
    assert counts["attn_w_b"] == 14
    assert counts["attn_small"] == 35
    assert counts["router"] == 36
    assert counts["expert_bias"] == 36
    assert counts["geodesic_scalar"] == 216
    assert sum(counts.values()) == 945


def test_default_ignore_list_i_resolves_to_concrete_names(header: dict[str, TensorMeta]) -> None:
    plan = build_plan(header, recipes.RECIPES["fp8"], recipes.DEFAULT_IGNORE_I)
    assert set(plan.ignored_quantizable) >= {f"model.layers.{i}.mixer_proj.weight" for i in (1, 7, 23)}
    assert len(plan.ignored_quantizable) == 3 + 4 * 6  # 3 mixer_proj + 6 quantizable tensors x 4 layers
    for layer in (0, 1, 5, 19):
        for tail in (
            "experts.experts.weight",
            "experts.output_experts.weight",
            "shared_experts.fc_1.weight",
            "shared_experts.fc_2.weight",
            "down_proj.weight",
            "up_proj.weight",
        ):
            assert f"model.layers.{layer}.mlp.{tail}" in plan.ignored_quantizable
    # the patterns also hit non-quantizable tensors (router, expert_bias): resolved, not quantizable
    assert "model.layers.0.mlp.moe_gate.weight" in plan.ignored_resolved
    assert "model.layers.0.mlp.moe_gate.weight" not in plan.ignored_quantizable
    quantized = [t for t in plan.tensors if t.category == "quantized"]
    assert len(quantized) == 259 - 27
    kept_ignored = [t for t in plan.tensors if t.category == "ignored"]
    assert len(kept_ignored) == 27


def test_vllm_ignore_list_uses_module_names_and_regex_for_experts(
    header: dict[str, TensorMeta],
) -> None:
    plan = build_plan(header, recipes.RECIPES["fp8"], recipes.DEFAULT_IGNORE_I)
    ign = set(plan.vllm_ignore)
    assert "lm_head" in ign
    assert "model.layers.1.mixer_proj" in ign
    for layer in (0, 1, 5, 19):
        p = f"model.layers.{layer}.mlp"
        assert f"{p}.fc1_latent_proj" in ign
        assert f"{p}.fc2_latent_proj" in ign
        assert f"{p}.shared_experts.up_proj" in ign
        assert f"{p}.shared_experts.down_proj" in ign
        # RoutedExperts is matched by compressed-tensors through "<layer>.<expert>.<proj>"
        # names (compressed_tensors_moe.py:36-49), and plain entries are exact-match only
        # (config_utils.py:443-452), so the experts need a regex entry.
        assert f"re:model\\.layers\\.{layer}\\.mlp\\.experts\\..*" in ign
    assert "model.layers.2.mixer_proj" not in ign
    assert not any(n.endswith(".weight") for n in ign)


def test_no_ignore_means_only_lm_head_in_vllm_ignore(header: dict[str, TensorMeta]) -> None:
    plan = build_plan(header, recipes.RECIPES["nvfp4"], ())
    assert plan.vllm_ignore == ("lm_head",)


def test_half_of_a_routed_pair_is_expanded_because_vllm_cannot_split_it(
    header: dict[str, TensorMeta],
) -> None:
    only_down = "model.layers.3.mlp.experts.output_experts.weight"
    plan = build_plan(header, recipes.RECIPES["fp8"], (only_down,))
    assert set(plan.ignored_quantizable) == {
        only_down,
        "model.layers.3.mlp.experts.experts.weight",
    }
    assert plan.forced["model.layers.3.mlp.experts.experts.weight"].startswith("vLLM RoutedExperts")


def test_ignore_list_that_matches_nothing_is_refused(header: dict[str, TensorMeta]) -> None:
    with pytest.raises(IgnoreListError, match="matches no tensor"):
        build_plan(header, recipes.RECIPES["fp8"], ("model.layers.99.mixer_proj.weight",))
    with pytest.raises(IgnoreListError, match="matches no tensor"):
        build_plan(
            header,
            recipes.RECIPES["fp8"],
            ("model.layers.1.mixer_proj.weight", "typo.layers.1.mixer_proj.weight"),
        )


def test_ignore_list_that_hits_no_quantizable_tensor_is_refused(
    header: dict[str, TensorMeta],
) -> None:
    with pytest.raises(IgnoreListError, match="no tensor that would be quantized"):
        build_plan(header, recipes.RECIPES["fp8"], ("model.layers.0.mixer.in_proj.weight",))


def test_fp8_output_tensors_and_sizes(header: dict[str, TensorMeta]) -> None:
    plan = build_plan(header, recipes.RECIPES["fp8"], recipes.DEFAULT_IGNORE_I)
    out = {o.name: o for o in plan.output_tensors()}
    w = out["model.layers.2.mlp.experts.experts.weight"]
    s = out["model.layers.2.mlp.experts.experts.weight_scale"]
    assert (w.dtype, w.shape) == ("F8_E4M3", (256, 768, 384))
    assert (s.dtype, s.shape) == ("F32", (256, 768, 1))
    s2 = out["model.layers.4.gate_proj.weight_scale"]
    assert (s2.dtype, s2.shape) == ("F32", (4608, 1))
    # ignored layer stays bf16 and unrenamed
    assert out["model.layers.0.mlp.experts.experts.weight"].dtype == "BF16"
    assert "model.layers.0.mlp.experts.experts.weight_scale" not in out
    assert "model.layers.2.mlp.experts.experts.weight" in out  # name kept, dtype changed
    # independent size computation from the header
    quant_names = {t.name for t in plan.tensors if t.category == "quantized"}
    expected = 0
    for name, meta in header.items():
        if name in quant_names:
            rows = meta.numel // meta.shape[-1]
            expected += meta.numel + 4 * rows
        else:
            expected += meta.nbytes
    assert plan.output_bytes() == expected


def test_nvfp4_output_tensors_and_sizes(header: dict[str, TensorMeta]) -> None:
    plan = build_plan(header, recipes.RECIPES["nvfp4"], recipes.DEFAULT_IGNORE_I)
    out = {o.name: o for o in plan.output_tensors()}
    base = "model.layers.2.mlp.experts.experts"
    assert "model.layers.2.mlp.experts.experts.weight" not in out
    assert (out[f"{base}.weight_packed"].dtype, out[f"{base}.weight_packed"].shape) == (
        "U8",
        (256, 768, 192),
    )
    assert (out[f"{base}.weight_scale"].dtype, out[f"{base}.weight_scale"].shape) == (
        "F8_E4M3",
        (256, 768, 24),
    )
    assert (out[f"{base}.weight_global_scale"].dtype, out[f"{base}.weight_global_scale"].shape) == (
        "F32",
        (256,),
    )
    g = out["model.layers.2.mixer_proj.weight_global_scale"]
    assert (g.dtype, g.shape) == ("F32", (1,))
    p = out["model.layers.2.mixer_proj.weight_packed"]
    assert p.shape == (1536, 1536)
    quant_names = {t.name for t in plan.tensors if t.category == "quantized"}
    expected = 0
    for name, meta in header.items():
        if name in quant_names:
            n_exp = meta.shape[0] if len(meta.shape) == 3 else 1
            expected += meta.numel // 2 + meta.numel // 16 + 4 * n_exp
        else:
            expected += meta.nbytes
    assert plan.output_bytes() == expected
    total = sum(m.numel for m in header.values())
    assert 7.0 < plan.output_bytes() * 8 / total < 7.3  # ignore list I keeps 4 MoE layers in bf16
    everything = build_plan(header, recipes.RECIPES["nvfp4"], ())
    # no ignore list: 86.9 % of the weights at 4.5 bits (4 + 8/16 scale bits), the rest bf16/f32
    assert 5.9 < everything.output_bytes() * 8 / total < 6.05


def test_category_accounting_sums_to_total(header: dict[str, TensorMeta]) -> None:
    plan = build_plan(header, recipes.RECIPES["fp8"], recipes.DEFAULT_IGNORE_I)
    acct = plan.category_accounting()
    assert sum(v["count"] for v in acct.values()) == 945
    assert sum(v["params"] for v in acct.values()) == TOTAL_PARAMS_INCL_SCALARS
    assert sum(v["bytes_out"] for v in acct.values()) == plan.output_bytes()
    assert acct["routed_up"]["count"] == 36 - 4


def test_quantizable_name_with_unexpected_dtype_or_shape_is_an_error(
    header: dict[str, TensorMeta],
) -> None:
    bad = dict(header)
    name = "model.layers.2.mixer_proj.weight"
    bad[name] = TensorMeta("F32", header[name].shape)
    with pytest.raises(ValueError, match="expected BF16"):
        build_plan(bad, recipes.RECIPES["fp8"], ())
    bad[name] = TensorMeta("BF16", (1536, 1535))
    with pytest.raises(ValueError, match="multiple of 16"):
        build_plan(bad, recipes.RECIPES["nvfp4"], ())


def test_load_ignore_file_accepts_list_or_object(tmp_path: Path) -> None:
    f1 = tmp_path / "a.json"
    f1.write_text(json.dumps(["a.weight", "b.*"]))
    f2 = tmp_path / "b.json"
    f2.write_text(json.dumps({"patterns": ["c.weight"]}))
    f3 = tmp_path / "c.json"
    f3.write_text(json.dumps({"nope": 1}))
    assert load_ignore_file(f1) == ("a.weight", "b.*")
    assert load_ignore_file(f2) == ("c.weight",)
    with pytest.raises(ValueError, match="patterns"):
        load_ignore_file(f3)


def test_quantization_config_blocks(header: dict[str, TensorMeta]) -> None:
    plan8 = build_plan(header, recipes.RECIPES["fp8"], recipes.DEFAULT_IGNORE_I)
    cfg8 = recipes.RECIPES["fp8"].quantization_config(plan8.vllm_ignore)
    assert cfg8["quant_method"] == "compressed-tensors"
    assert cfg8["format"] == "float-quantized"
    group = cfg8["config_groups"]["group_0"]
    assert group["targets"] == ["Linear"]
    assert group["weights"] == {
        "num_bits": 8,
        "type": "float",
        "strategy": "channel",
        "symmetric": True,
        "dynamic": False,
        "group_size": None,
        "block_structure": None,
        "actorder": None,
        "observer": "minmax",
        "observer_kwargs": {},
    }
    assert group["input_activations"]["strategy"] == "token"
    assert group["input_activations"]["dynamic"] is True
    assert cfg8["ignore"] == list(plan8.vllm_ignore)

    plan4 = build_plan(header, recipes.RECIPES["nvfp4"], ())
    cfg4 = recipes.RECIPES["nvfp4"].quantization_config(plan4.vllm_ignore)
    assert cfg4["format"] == "nvfp4-pack-quantized"
    w4 = cfg4["config_groups"]["group_0"]["weights"]
    assert (w4["num_bits"], w4["type"], w4["strategy"], w4["group_size"], w4["symmetric"]) == (
        4,
        "float",
        "tensor_group",
        16,
        True,
    )
    assert cfg4["config_groups"]["group_0"]["input_activations"] is None
