"""Tests for fork-patch/olala_quant_loader.py, the pure logic of the vLLM loader patch.

The module is loaded BY PATH and must not import vllm. Fake parameters mirror the semantics of
``RoutedExperts.weight_loader`` as read from the fork (routed_experts.py:688-1002) for one rank.
"""

import importlib.util
import json
import sys
from pathlib import Path

import fakes
import pytest
import torch
from safetensors import safe_open

from olala_quant import convert as conv
from olala_quant import recipes
from olala_quant.checkpoint_io import header_from_json
from olala_quant.plan import build_plan

LOADER_PATH = Path(__file__).parents[1] / "fork-patch" / "olala_quant_loader.py"
DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def ql():
    assert "vllm" not in sys.modules, "the logic module must be testable without vllm"
    spec = importlib.util.spec_from_file_location("olala_quant_loader", LOADER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["olala_quant_loader"] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    assert "vllm" not in sys.modules
    return module


MOE = frozenset(range(36))


def test_routes_for_bf16_and_dense_quantized_names(ql) -> None:
    r = ql.route_checkpoint_name("model.layers.3.mlp.down_proj.weight", MOE)
    assert (r.kind, r.target) == ("rename", "model.layers.3.mlp.fc1_latent_proj.weight")
    r = ql.route_checkpoint_name("model.layers.3.mlp.up_proj.weight_packed", MOE)
    assert r.target == "model.layers.3.mlp.fc2_latent_proj.weight_packed"
    r = ql.route_checkpoint_name("model.layers.3.mlp.shared_experts.fc_1.weight_scale", MOE)
    assert r.target == "model.layers.3.mlp.shared_experts.up_proj.weight_scale"
    r = ql.route_checkpoint_name("model.layers.3.mlp.shared_experts.fc_2.weight_global_scale", MOE)
    assert r.target == "model.layers.3.mlp.shared_experts.down_proj.weight_global_scale"
    r = ql.route_checkpoint_name("model.layers.3.mlp.moe_gate.weight", MOE)
    assert r.target == "model.layers.3.mlp.gate.weight"
    r = ql.route_checkpoint_name("model.layers.3.mlp.expert_bias", MOE)
    assert (r.kind, r.target) == ("expert_bias", "model.layers.3.mlp.gate.e_score_correction_bias")
    assert ql.route_checkpoint_name("model.layers.3.mlp.tokens_per_expert", MOE).kind == "drop"
    # names that are already vLLM names are passed through
    for n in (
        "model.layers.3.mixer_proj.weight_packed",
        "model.layers.4.gate_proj.weight_scale",
        "lm_head.weight",
    ):
        r = ql.route_checkpoint_name(n, MOE)
        assert (r.kind, r.target) == ("direct", n)
    # a dense MLP layer (not in moe_layers) is not renamed
    assert ql.route_checkpoint_name("model.layers.3.mlp.down_proj.weight", frozenset()).kind == "direct"


@pytest.mark.parametrize(
    ("tail", "fused_param", "shard"),
    [
        ("experts.experts.weight", "w13_weight", "w1"),
        ("experts.experts.weight_scale", "w13_weight_scale", "w1"),
        ("experts.experts.weight_packed", "w13_weight_packed", "w1"),
        ("experts.experts.weight_global_scale", "w13_weight_global_scale", "w1"),
        ("experts.experts.input_global_scale", "w13_input_global_scale", "w1"),
        ("experts.output_experts.weight", "w2_weight", "w2"),
        ("experts.output_experts.weight_scale", "w2_weight_scale", "w2"),
        ("experts.output_experts.weight_packed", "w2_weight_packed", "w2"),
        ("experts.output_experts.weight_global_scale", "w2_weight_global_scale", "w2"),
        ("experts.output_experts.input_global_scale", "w2_input_global_scale", "w2"),
    ],
)
def test_fused_expert_routes(ql, tail: str, fused_param: str, shard: str) -> None:
    r = ql.route_checkpoint_name(f"model.layers.5.mlp.{tail}", MOE)
    assert r.kind == "fused_experts"
    assert (r.layer, r.fused_param, r.shard_id) == (5, fused_param, shard)
    # RoutedExperts.weight_loader dispatches on substrings of this name (routed_experts.py:828-1000)
    assert r.loader_name == f"model.layers.5.mlp.experts.{fused_param}"


def test_unknown_suffix_on_fused_module_is_flagged_not_guessed(ql) -> None:
    r = ql.route_checkpoint_name("model.layers.5.mlp.experts.experts.weight_zero_point", MOE)
    assert r.kind == "unsupported_fused"


def test_legacy_experts_int8_scale_keeps_its_own_parameter(ql) -> None:
    r = ql.route_checkpoint_name("model.layers.5.mlp.experts.experts.weight_scale", MOE)
    assert r.legacy_param == "w13_scale"
    r = ql.route_checkpoint_name("model.layers.5.mlp.experts.output_experts.weight_scale", MOE)
    assert r.legacy_param == "w2_scale"


def test_dtype_contract(ql) -> None:
    f8, u8, bf, f32 = torch.float8_e4m3fn, torch.uint8, torch.bfloat16, torch.float32
    ql.check_dtype_contract("a", f8, f8)
    ql.check_dtype_contract("a", u8, u8)
    ql.check_dtype_contract("a", bf, bf)
    ql.check_dtype_contract("a", bf, f32)  # existing bf16 -> fp32 widening (router, expert_bias)
    for loaded, param in ((bf, f8), (f8, bf), (u8, f8), (f8, u8), (f32, f8), (u8, bf)):
        with pytest.raises(ql.OlalaQuantLoadError, match=r"quantization_config\.ignore"):
            ql.check_dtype_contract("model.layers.1.mixer_proj.weight", loaded, param)


def test_slice_shape_checks(ql) -> None:
    ql.check_slice_shape("n", (768, 384), "w13_weight", (256, 768, 384))
    ql.check_slice_shape("n", (768, 1), "w13_weight_scale", (256, 768, 1))
    ql.check_slice_shape("n", (), "w13_weight_global_scale", (256, 1))
    ql.check_slice_shape("n", (), "w2_input_global_scale", (256,))
    with pytest.raises(ql.OlalaQuantLoadError, match="shape"):
        ql.check_slice_shape("n", (768,), "w13_weight_scale", (256, 768, 1))  # [out] vs [out, 1]
    with pytest.raises(ql.OlalaQuantLoadError, match="shape"):
        ql.check_slice_shape("n", (384, 768), "w13_weight", (256, 768, 384))
    # backends that round sizes up allocate larger parameters; the loader narrows them
    # (routed_experts.py:494-527), so a smaller checkpoint slice is accepted, a larger one is not
    ql.check_slice_shape("n", (768, 384), "w13_weight", (256, 896, 384))
    ql.check_slice_shape("n", (768, 24), "w13_weight_scale", (256, 768, 32))
    with pytest.raises(ql.OlalaQuantLoadError, match="shape"):
        ql.check_slice_shape("n", (768, 384), "w13_weight", (256, 640, 384))
    with pytest.raises(ql.OlalaQuantLoadError, match="shape"):
        ql.check_slice_shape("n", (768, 384, 1), "w13_weight", (256, 768, 384))
    # tensor parallel: shapes legitimately differ, the check is skipped
    ql.check_slice_shape("n", (768, 384), "w13_weight", (256, 384, 384), tp_size=2)


class FakeRoutedParam:
    """A parameter whose weight_loader mirrors RoutedExperts.weight_loader for tp=1, non-gated."""

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype, quant_method: str | None = None) -> None:
        self.data = (
            torch.zeros(shape, dtype=dtype)
            if dtype != torch.float8_e4m3fn
            else torch.zeros(shape, dtype=torch.float32).to(dtype)
        )
        self.quant_method = quant_method
        self.calls: list[tuple[str, int]] = []

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    def weight_loader(
        self,
        param: "FakeRoutedParam",
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        assert param is self
        assert shard_id in ("w1", "w2", "w3")
        self.calls.append((weight_name, expert_id))
        expert_data = self.data[expert_id]
        if "scale" in weight_name:
            if self.quant_method == "channel":
                if shard_id == "w2":
                    expert_data.copy_(loaded_weight)
                else:
                    expert_data.narrow(0, 0, expert_data.shape[0]).copy_(loaded_weight)
            elif self.quant_method == "group":
                expert_data.copy_(loaded_weight)
            elif self.quant_method == "tensor":
                if shard_id == "w2":
                    self.data[expert_id] = loaded_weight.reshape(())
                else:
                    self.data[expert_id][0] = loaded_weight.reshape(())
            else:
                raise ValueError("quant method must be one of tensor, channel, group, block")
        elif "weight" in weight_name:
            expert_data.copy_(loaded_weight)
        else:
            return False if return_success else None
        return True if return_success else None


def _expected_params(recipe: str, n_exp: int = fakes.N_EXPERTS) -> dict[str, FakeRoutedParam]:
    """Routed-expert parameters as created by the fork's CT FP8 / NVFP4 MoE methods."""
    h, i = fakes.LATENT, fakes.EXPERT_FF  # hidden of the experts is the latent size
    if recipe == "fp8":  # compressed_tensors_moe_w8a8_fp8.py:143-209 (channel strategy)
        return {
            "w13_weight": FakeRoutedParam((n_exp, i, h), torch.float8_e4m3fn),
            "w2_weight": FakeRoutedParam((n_exp, h, i), torch.float8_e4m3fn),
            "w13_weight_scale": FakeRoutedParam((n_exp, i, 1), torch.float32, "channel"),
            "w2_weight_scale": FakeRoutedParam((n_exp, h, 1), torch.float32, "channel"),
        }
    return {  # compressed_tensors_moe_w4a4_nvfp4.py:89-189 (non-gated: one shard, a16 variant)
        "w13_weight_packed": FakeRoutedParam((n_exp, i, h // 2), torch.uint8),
        "w2_weight_packed": FakeRoutedParam((n_exp, h, i // 2), torch.uint8),
        "w13_weight_scale": FakeRoutedParam((n_exp, i, h // 16), torch.float8_e4m3fn, "group"),
        "w2_weight_scale": FakeRoutedParam((n_exp, h, i // 16), torch.float8_e4m3fn, "group"),
        "w13_weight_global_scale": FakeRoutedParam((n_exp, 1), torch.float32, "tensor"),
        "w2_weight_global_scale": FakeRoutedParam((n_exp,), torch.float32, "tensor"),
        "w13_input_global_scale": FakeRoutedParam((n_exp, 1), torch.float32, "tensor"),
        "w2_input_global_scale": FakeRoutedParam((n_exp,), torch.float32, "tensor"),
    }


def _convert(tmp_path: Path, recipe: str) -> tuple[Path, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    src, dst = tmp_path / "src", tmp_path / "dst"
    orig = fakes.write_fake_checkpoint(src)
    conv.convert(src, dst, recipe, ("model.layers.0.mlp.*",))
    out: dict[str, torch.Tensor] = {}
    for f in dst.glob("model-*.safetensors"):
        with safe_open(f, framework="pt") as h:
            out.update({k: h.get_tensor(k) for k in h.keys()})  # noqa: SIM118
    return dst, orig, out


@pytest.mark.parametrize("recipe", ["fp8", "nvfp4"])
def test_scatter_of_converted_expert_tensors_fills_every_parameter(ql, tmp_path: Path, recipe: str) -> None:
    _, _, out = _convert(tmp_path, recipe)
    params = _expected_params(recipe)
    layer = 2
    moe = frozenset(range(4))
    seen = set()
    for name, tensor in out.items():
        if not name.startswith(f"model.layers.{layer}.mlp.experts."):
            continue
        route = ql.route_checkpoint_name(name, moe)
        assert route.kind == "fused_experts", name
        param = params[route.fused_param]
        accepted = ql.scatter_fused_expert_tensor(
            param, tensor, route, num_experts=fakes.N_EXPERTS, tp_size=1
        )
        assert accepted == fakes.N_EXPERTS
        seen.add(route.fused_param)
        # the parameter now holds exactly the checkpoint tensor (all experts, right shard)
        stored = param.data
        if route.fused_param.endswith("global_scale"):
            assert torch.equal(stored.reshape(-1), tensor.reshape(-1))
        else:
            assert torch.equal(stored.reshape(-1).view(torch.uint8), tensor.reshape(-1).view(torch.uint8))
    expected = set(params) - (
        {"w13_input_global_scale", "w2_input_global_scale"} if recipe == "nvfp4" else set()
    )
    assert seen == expected
    if recipe == "nvfp4":
        assert set(ql.UNUSED_A16_PARAMS) == {"w13_input_global_scale", "w2_input_global_scale"}


def test_scatter_rejects_wrong_dtype_wrong_shape_and_wrong_expert_count(ql) -> None:
    params = _expected_params("fp8")
    route = ql.route_checkpoint_name("model.layers.2.mlp.experts.experts.weight", frozenset(range(4)))
    bf16 = torch.zeros(fakes.N_EXPERTS, fakes.EXPERT_FF, fakes.LATENT, dtype=torch.bfloat16)
    with pytest.raises(ql.OlalaQuantLoadError, match="ignore"):
        ql.scatter_fused_expert_tensor(params["w13_weight"], bf16, route, num_experts=fakes.N_EXPERTS)
    wrong_shape = torch.zeros(fakes.N_EXPERTS, fakes.LATENT, fakes.EXPERT_FF).to(torch.float8_e4m3fn)
    with pytest.raises(ql.OlalaQuantLoadError, match="shape"):
        ql.scatter_fused_expert_tensor(params["w13_weight"], wrong_shape, route, num_experts=fakes.N_EXPERTS)
    few = torch.zeros(3, fakes.EXPERT_FF, fakes.LATENT).to(torch.float8_e4m3fn)
    with pytest.raises(ql.OlalaQuantLoadError, match="experts"):
        ql.scatter_fused_expert_tensor(params["w13_weight"], few, route, num_experts=fakes.N_EXPERTS)


def test_scatter_raises_when_no_expert_is_accepted(ql) -> None:
    class Declining(FakeRoutedParam):
        def weight_loader(self, *args: object, **kwargs: object) -> bool:
            return False

    p = Declining((fakes.N_EXPERTS, fakes.EXPERT_FF, fakes.LATENT), torch.float8_e4m3fn)
    route = ql.route_checkpoint_name("model.layers.2.mlp.experts.experts.weight", frozenset(range(4)))
    w = torch.zeros(fakes.N_EXPERTS, fakes.EXPERT_FF, fakes.LATENT).to(torch.float8_e4m3fn)
    with pytest.raises(ql.OlalaQuantLoadError, match="no expert"):
        ql.scatter_fused_expert_tensor(p, w, route, num_experts=fakes.N_EXPERTS)


def test_load_tracker_lists_every_unconsumed_name_and_honours_the_allow_list(ql) -> None:
    t = ql.LoadTracker()
    t.consume("a")
    t.reject("model.layers.1.mlp.down_proj.weight_scale", "no parameter of that name")
    t.reject("model.layers.2.mixer.weird", "no parameter of that name")
    with pytest.raises(ql.OlalaQuantLoadError) as exc:
        t.raise_if_unconsumed(allow_regex=None)
    msg = str(exc.value)
    assert "2 checkpoint tensor(s)" in msg
    assert "model.layers.1.mlp.down_proj.weight_scale" in msg and "model.layers.2.mixer.weird" in msg
    assert "OLALA_ALLOW_UNCONSUMED" in msg
    # allow-list: the matching name is tolerated, the other still raises
    with pytest.raises(ql.OlalaQuantLoadError) as exc2:
        t.raise_if_unconsumed(allow_regex=r".*\.weird$")
    assert "1 checkpoint tensor(s)" in str(exc2.value)
    t.raise_if_unconsumed(allow_regex=r".*")  # everything allowed: no error


def test_unloaded_quantized_parameters_are_found_modulo_the_routed_experts_alias(ql) -> None:
    expected = [
        "model.layers.1.mixer_proj.weight_packed",
        "model.layers.1.mixer_proj.weight_scale",
        "model.layers.1.mlp.experts.routed_experts.w13_weight_packed",
        "model.layers.1.mlp.experts.routed_experts.w13_weight_global_scale",
    ]
    loaded = {
        "model.layers.1.mixer_proj.weight_packed",
        "model.layers.1.mlp.experts.w13_weight_packed",  # the loader records the alias without it
    }
    assert ql.unloaded_params(expected, loaded) == [
        "model.layers.1.mixer_proj.weight_scale",
        "model.layers.1.mlp.experts.routed_experts.w13_weight_global_scale",
    ]


@pytest.mark.parametrize("recipe", ["fp8", "nvfp4"])
def test_contract_between_converter_output_and_loader_on_the_real_header(ql, recipe: str) -> None:
    """Every tensor the converter writes for the REAL model is routed to a parameter that the
    fork's quantization methods create, and every such parameter is fed (a16 input scales aside)."""
    header = header_from_json(json.loads((DATA / "olala_header.json").read_text()))
    plan = build_plan(header, recipes.RECIPES[recipe], recipes.DEFAULT_IGNORE_I)
    moe = frozenset(range(36))
    # parameters created per module (fork source lines in the comments)
    dense_params = (
        {
            "weight",
            "weight_scale",
        }  # compressed_tensors_w8a8_fp8.py:117-130, dynamic: no input_scale
        if recipe == "fp8"
        else {"weight_packed", "weight_scale", "weight_global_scale"}  # ..._w4a4_nvfp4.py:53-91 a16
    )
    fused_params = (
        {"w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"}
        if recipe == "fp8"
        else {
            f"{s}_{p}"
            for s in ("w13", "w2")
            for p in ("weight_packed", "weight_scale", "weight_global_scale")
        }
    )
    targets: set[str] = set()
    for o in plan.output_tensors():
        r = ql.route_checkpoint_name(o.name, moe)
        assert r.kind in {"direct", "rename", "expert_bias", "fused_experts"}, o.name
        if r.kind == "fused_experts":
            targets.add(f"{r.layer}:{r.fused_param}")
        else:
            targets.add(r.target)
    quantized_layers = {t.layer for t in plan.tensors if t.category == "quantized" and t.role == "routed_up"}
    assert quantized_layers == set(range(36)) - {0, 1, 5, 19}
    for layer in quantized_layers:
        for p in fused_params:
            assert f"{layer}:{p}" in targets, (layer, p)
        for mod in (
            "mlp.fc1_latent_proj",
            "mlp.fc2_latent_proj",
            "mlp.shared_experts.up_proj",
            "mlp.shared_experts.down_proj",
        ):
            for p in dense_params:
                assert f"model.layers.{layer}.{mod}.{p}" in targets
    for layer in (0, 1, 5, 19):  # ignored layers keep bf16 names
        assert f"{layer}:w13_weight" in targets and f"{layer}:w2_weight" in targets
        assert f"{layer}:w13_weight_scale" not in targets
    assert not [
        t for t in targets if t.endswith(".weight_packed") and "mixer_proj" in t and ".layers.1." in t
    ]
    assert "model.layers.7.mixer_proj.weight" in targets  # ignored mixer_proj stays bf16 and unquantized


# vLLM parameter / buffer names of the bf16 model, from reading the fork (cited per line):
VLLM_NAME_PATTERNS = {
    "embedding": r"model\.embedding\.weight",  # olala.py:1016 area, VocabParallelEmbedding
    "lm_head": r"lm_head\.weight",  # olala.py:1192-1197
    "geodesic": r"model\.layers\.\d+\.geodesic_(mixer|mlp)\.(scale|bias|prosres_scalar)",  # :710-712
    "block_proj": r"model\.layers\.\d+\.(mixer_proj|gate_proj)\.weight",  # :828-857
    "mamba_linear": r"model\.layers\.\d+\.mixer\.(in_proj|in_proj_dyn)\.weight",  # mamba3.py:416-432
    "mamba_param": (
        r"model\.layers\.\d+\.mixer\.(B_bias|C_bias|in_proj_mimo_x|in_proj_mimo_z|out_proj_mimo|dt_bias|D)"
    ),  # mamba3.py:445-495
    "norm": r"model\.layers\.\d+\.mixer\.(B_norm|C_norm|q_norm|k_norm)\.norm\.weight",  # mamba3.py:461,
    # diff_tpa.py:354-355
    "attn_linear": (
        r"model\.layers\.\d+\.mixer\.(c_q|W_A_k|W_A_v|W_B_k|W_B_v|lambda_proj|shift_proj_k|shift_proj_v)\.weight"
    ),  # diff_tpa.py:301-340
    "softmax_scaler": r"model\.layers\.\d+\.mixer\.softmax_scaler",  # diff_tpa.py:275
    "router": r"model\.layers\.\d+\.mlp\.gate\.(weight|e_score_correction_bias)",  # olala.py:242-251
    "latent_shared": (
        r"model\.layers\.\d+\.mlp\.(fc1_latent_proj|fc2_latent_proj|shared_experts\.up_proj"
        r"|shared_experts\.down_proj)\.weight"
    ),  # olala.py:161-175, 267-280
    "routed": r"\d+:(w13_weight|w2_weight)",  # RoutedExperts parameters (bf16 / UnquantizedFusedMoEMethod)
}


def test_every_real_checkpoint_name_lands_on_a_vllm_parameter(ql) -> None:
    """Static name audit on the real header: no HF tensor may fall through to 'not found'."""
    import re

    header = header_from_json(json.loads((DATA / "olala_header.json").read_text()))
    moe = frozenset(range(36))
    compiled = {k: re.compile(v) for k, v in VLLM_NAME_PATTERNS.items()}
    used: dict[str, int] = dict.fromkeys(compiled, 0)
    unmatched = []
    for name in header:
        r = ql.route_checkpoint_name(name, moe)
        assert r.kind != "unsupported_fused", name
        key = f"{r.layer}:{r.fused_param}" if r.kind == "fused_experts" else r.target
        hit = [k for k, rx in compiled.items() if rx.fullmatch(key or "")]
        if not hit:
            unmatched.append(name)
        for k in hit[:1]:
            used[k] += 1
    assert unmatched == []
    assert all(n > 0 for n in used.values()), used  # no dead pattern
    assert sum(used.values()) == 945
