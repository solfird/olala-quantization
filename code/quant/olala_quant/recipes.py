"""Quantization recipes, as data.

A recipe says which tensors are quantized and in which on-disk layout. The quantized set is what
the OVH vLLM fork (commit ebbadef) can consume, from ``recon/quantizability.md`` D.2:

* routed experts: the fused 3D tensors ``mlp.experts.experts.weight`` ``[256, 768, 384]`` and
  ``mlp.experts.output_experts.weight`` ``[256, 384, 768]`` (``RoutedExperts``, olala.py:296-326);
* shared expert ``fc_1`` / ``fc_2`` (olala.py:145-182), latent ``down_proj`` / ``up_proj``
  (olala.py:267-280), ``mixer_proj`` of every layer (olala.py:849-857) and the attention
  ``gate_proj`` (olala.py:828-835).

Everything else (Mamba ``in_proj*``, attention q/k/v-side projections, router, embedding,
``lm_head``, norms, all F32 tensors) is copied unchanged: the fork builds those modules without a
``quant_config`` (mamba3.py:416-432, diff_tpa.py:284-306) or never quantizes them.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from olala_quant.checkpoint_io import TensorMeta

# Default ignore list I (patterns on HF tensor names, fnmatch syntax): the output projection of
# the Mamba-3 mixers of layers 1, 7, 23 and every MLP tensor of layers 0, 1, 5, 19
# (recon/quantizability.md section C).
DEFAULT_IGNORE_I: tuple[str, ...] = (
    "model.layers.1.mixer_proj.weight",
    "model.layers.7.mixer_proj.weight",
    "model.layers.23.mixer_proj.weight",
    "model.layers.0.mlp.*",
    "model.layers.1.mlp.*",
    "model.layers.5.mlp.*",
    "model.layers.19.mlp.*",
)


@dataclass(frozen=True)
class RoleSpec:
    """A quantizable tensor family.

    Attributes:
        role: Short name used in reports.
        pattern: Regex over the HF tensor name; group 1 is the layer index.
        ndim: Expected rank of the weight (2, or 3 for the fused experts).
        vllm_module: Template of the vLLM module that owns the tensor (``{L}`` is the layer).
        routed: True for the two fused expert tensors, which share one vLLM quant method.
    """

    role: str
    pattern: str
    ndim: int
    vllm_module: str
    routed: bool = False


QUANT_ROLES: tuple[RoleSpec, ...] = (
    RoleSpec(
        "routed_up",
        r"^model\.layers\.(\d+)\.mlp\.experts\.experts\.weight$",
        3,
        "model.layers.{L}.mlp.experts",
        True,
    ),
    RoleSpec(
        "routed_down",
        r"^model\.layers\.(\d+)\.mlp\.experts\.output_experts\.weight$",
        3,
        "model.layers.{L}.mlp.experts",
        True,
    ),
    RoleSpec(
        "shared_up",
        r"^model\.layers\.(\d+)\.mlp\.shared_experts\.fc_1\.weight$",
        2,
        "model.layers.{L}.mlp.shared_experts.up_proj",
    ),
    RoleSpec(
        "shared_down",
        r"^model\.layers\.(\d+)\.mlp\.shared_experts\.fc_2\.weight$",
        2,
        "model.layers.{L}.mlp.shared_experts.down_proj",
    ),
    RoleSpec(
        "latent_reduce",
        r"^model\.layers\.(\d+)\.mlp\.down_proj\.weight$",
        2,
        "model.layers.{L}.mlp.fc1_latent_proj",
    ),
    RoleSpec(
        "latent_expand",
        r"^model\.layers\.(\d+)\.mlp\.up_proj\.weight$",
        2,
        "model.layers.{L}.mlp.fc2_latent_proj",
    ),
    RoleSpec(
        "mixer_proj",
        r"^model\.layers\.(\d+)\.mixer_proj\.weight$",
        2,
        "model.layers.{L}.mixer_proj",
    ),
    RoleSpec(
        "attn_gate_proj",
        r"^model\.layers\.(\d+)\.gate_proj\.weight$",
        2,
        "model.layers.{L}.gate_proj",
    ),
)

# Tensors that are never quantized, in match order. Unknown names get the role "unclassified" and
# are copied unchanged (the converter reports them).
KEPT_ROLES: tuple[tuple[str, str], ...] = (
    ("embedding", r"^model\.embedding\.weight$"),
    ("lm_head", r"^lm_head\.weight$"),
    ("mamba_in_proj", r"^model\.layers\.\d+\.mixer\.in_proj\.weight$"),
    ("mamba_in_proj_dyn", r"^model\.layers\.\d+\.mixer\.in_proj_dyn\.weight$"),
    ("attn_c_q", r"^model\.layers\.\d+\.mixer\.c_q\.weight$"),
    ("attn_w_b", r"^model\.layers\.\d+\.mixer\.W_B_[kv]\.weight$"),
    ("attn_small", r"^model\.layers\.\d+\.mixer\.(W_A_[kv]|shift_proj_[kv]|lambda_proj)\.weight$"),
    ("router", r"^model\.layers\.\d+\.mlp\.moe_gate\.weight$"),
    ("expert_bias", r"^model\.layers\.\d+\.mlp\.expert_bias$"),
    ("geodesic_scalar", r"^model\.layers\.\d+\.geodesic_(mixer|mlp)\.(scale|bias|prosres_scalar)$"),
    (
        "mamba_param",
        r"^model\.layers\.\d+\.mixer\.(B_bias|C_bias|in_proj_mimo_x|in_proj_mimo_z|out_proj_mimo|D|dt_bias)$",
    ),
    ("norm", r"^model\.layers\.\d+\.mixer\.(B_norm|C_norm|q_norm|k_norm)\.norm\.weight$"),
    ("softmax_scaler", r"^model\.layers\.\d+\.mixer\.softmax_scaler$"),
)

_COMPILED_QUANT = tuple((spec, re.compile(spec.pattern)) for spec in QUANT_ROLES)
_COMPILED_KEPT = tuple((role, re.compile(p)) for role, p in KEPT_ROLES)


def classify_tensor(name: str) -> tuple[str, RoleSpec | None, int | None]:
    """Return ``(role, quant_spec_or_None, layer_index_or_None)`` for an HF tensor name."""
    for spec, rx in _COMPILED_QUANT:
        m = rx.match(name)
        if m:
            return spec.role, spec, int(m.group(1))
    for role, rx in _COMPILED_KEPT:
        if rx.match(name):
            return role, None, None
    return "unclassified", None, None


@dataclass(frozen=True)
class OutputSpec:
    """One tensor written for a quantized weight: name suffix, safetensors dtype, shape."""

    suffix: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class Recipe:
    """A named quantization recipe.

    Attributes:
        name: CLI name.
        arms: Study arms (PROTOCOL.md) this recipe can produce.
        description: One-paragraph summary.
        quantizer: ``fp8`` or ``nvfp4`` (module of :mod:`olala_quant` that does the math).
        weights: Weight format label.
        activations: Activation handling at serve time.
        implemented: False when the recipe is declared but needs data that is not available yet.
        suffixes: Tensor-name suffixes written per quantized weight.
        calibration_needs: What an unimplemented recipe still requires.
    """

    name: str
    arms: tuple[str, ...]
    description: str
    quantizer: str
    weights: str
    activations: str
    implemented: bool
    suffixes: tuple[str, ...]
    calibration_needs: tuple[str, ...] = ()

    def output_specs(self, meta: TensorMeta) -> list[OutputSpec]:
        """Describe the tensors written for a quantized weight of the given shape."""
        shape = meta.shape
        if self.quantizer == "fp8":
            return [
                OutputSpec("weight", "F8_E4M3", shape),
                OutputSpec("weight_scale", "F32", (*shape[:-1], 1)),
            ]
        n_global = shape[0] if len(shape) == 3 else 1
        return [
            OutputSpec("weight_packed", "U8", (*shape[:-1], shape[-1] // 2)),
            OutputSpec("weight_scale", "F8_E4M3", (*shape[:-1], shape[-1] // 16)),
            OutputSpec("weight_global_scale", "F32", (n_global,)),
        ]

    def quantization_config(self, ignore: Sequence[str]) -> dict[str, Any]:
        """Build the ``quantization_config`` block of ``config.json``.

        ``ignore`` holds vLLM module names (and ``re:`` regexes), see
        :attr:`olala_quant.plan.Plan.vllm_ignore`.
        """
        if self.quantizer == "fp8":
            weights = _qargs(8, "channel", None, dynamic=False, observer="minmax")
            inputs: dict[str, Any] | None = _qargs(8, "token", None, dynamic=True, observer=None)
            fmt = "float-quantized"
        else:
            weights = _qargs(4, "tensor_group", 16, dynamic=False, observer="minmax")
            inputs = None
            fmt = "nvfp4-pack-quantized"
        return {
            "quant_method": "compressed-tensors",
            "format": fmt,
            "quantization_status": "compressed",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": weights,
                    "input_activations": inputs,
                    "output_activations": None,
                }
            },
            "kv_cache_scheme": None,
            "ignore": list(ignore),
        }


def _qargs(
    num_bits: int, strategy: str, group_size: int | None, *, dynamic: bool, observer: str | None
) -> dict[str, Any]:
    """Fields of a compressed-tensors ``QuantizationArgs`` for a symmetric float quantizer."""
    return {
        "num_bits": num_bits,
        "type": "float",
        "strategy": strategy,
        "symmetric": True,
        "dynamic": dynamic,
        "group_size": group_size,
        "block_structure": None,
        "actorder": None,
        "observer": observer,
        "observer_kwargs": {},
    }


RECIPES: dict[str, Recipe] = {
    "fp8": Recipe(
        name="fp8",
        arms=("F2",),
        description=(
            "FP8 e4m3 weights with one fp32 scale per output channel; activations are quantized "
            "dynamically per token at serve time (llm-compressor FP8_DYNAMIC layout). No data needed."
        ),
        quantizer="fp8",
        weights="fp8_e4m3_per_output_channel",
        activations="fp8_dynamic_per_token",
        implemented=True,
        suffixes=("weight", "weight_scale"),
    ),
    "nvfp4": Recipe(
        name="nvfp4",
        arms=("N3",),
        description=(
            "NVFP4 weights (e2m1, group 16, e4m3 block scales, one fp32 global scale per tensor, "
            "per expert for the fused expert tensors); activations stay bf16 (W4A16). No data needed."
        ),
        quantizer="nvfp4",
        weights="nvfp4_g16",
        activations="bf16",
        implemented=True,
        suffixes=("weight_packed", "weight_scale", "weight_global_scale"),
    ),
    "nvfp4_w4a4": Recipe(
        name="nvfp4_w4a4",
        arms=("N1", "N2"),
        description=(
            "NVFP4 weights as in `nvfp4` plus NVFP4 activations. Declared, not implemented: the "
            "static activation global scales need calibration data."
        ),
        quantizer="nvfp4",
        weights="nvfp4_g16",
        activations="nvfp4",
        implemented=False,
        suffixes=("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale"),
        calibration_needs=(
            "input_global_scale (fp32, shape [1]) for every quantized Linear: 448*6/amax of the "
            "activation entering it over the calibration set (compressed-tensors encode convention, "
            "compressed_tensors_w4a4_nvfp4.py:124-135)",
            "w13_input_global_scale [E] and w2_input_global_scale [E] for every routed-expert layer, "
            "PER EXPERT, from the tokens the router sends to each expert: needs a forward pass "
            "through the real model (router included) with hooks on the latent-expert input and on "
            "the ReLU^2 output (compressed_tensors_moe_w4a4_nvfp4.py:170-189)",
            "calibration text per PROTOCOL.md section 5: 512 samples of up to 2048 tokens, chat "
            "template with reasoning_effort=medium, disjoint from the test items",
            "an unfused or hooked forward of the OVH modeling code (GPU, scattermoe) to read the "
            "activations; a quantized layer whose activation amax is dominated by the Appendix C "
            "spikes (1e7 to 1e16) gets block scales that underflow e4m3, which is what the ignore "
            "list and arm N1 test",
        ),
    ),
}


def get_recipe(name: str) -> Recipe:
    """Return the recipe called ``name`` (KeyError with the valid names otherwise)."""
    try:
        return RECIPES[name]
    except KeyError:
        raise KeyError(f"unknown recipe {name!r}; choose one of {sorted(RECIPES)}") from None
