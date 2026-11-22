# SPDX-License-Identifier: Apache-2.0
"""Pure logic of the Olala quantized-checkpoint loader (no vLLM import, unit-testable on CPU).

Used by ``OlalaForCausalLM.load_weights`` in ``vllm/model_executor/models/olala.py``. The checkpoint
keeps the Hugging Face tensor names; this module decides, for every checkpoint name:

* which vLLM parameter receives it (dense modules are renamed, the fused routed-expert tensors are
  scattered expert by expert into ``w13_*`` / ``w2_*`` parameters of ``RoutedExperts``);
* whether the dtype and shape agree with that parameter (vLLM's own loaders ``copy_`` silently
  cast bf16 into an fp8 parameter, which would load garbage without any error);
* which checkpoint tensors nothing consumed and which quantized parameters nothing filled.

Layouts it supports (written by ``olala_quant.convert``):

* FP8 per output channel (compressed-tensors ``float-quantized``): ``<module>.weight`` float8_e4m3fn
  and ``<module>.weight_scale`` float32 ``[out, 1]``; fused experts ``[E, out, in]`` and
  ``[E, out, 1]``.
* NVFP4 weight-only (compressed-tensors ``nvfp4-pack-quantized``): ``<module>.weight_packed`` uint8
  ``[out, in/2]``, ``weight_scale`` float8_e4m3fn ``[out, in/16]``, ``weight_global_scale`` float32
  ``[1]`` (``[E]`` for the fused experts, one per expert).
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import Any

import torch

# Tail renames inside ``model.layers.N.mlp.`` (olala.py:1319-1325 generalised from the ``.weight``
# suffix to every suffix, so ``down_proj.weight_scale`` follows ``down_proj.weight``).
MLP_MODULE_RENAMES = {
    "moe_gate": "gate",
    "down_proj": "fc1_latent_proj",
    "up_proj": "fc2_latent_proj",
    "shared_experts.fc_1": "shared_experts.up_proj",
    "shared_experts.fc_2": "shared_experts.down_proj",
}
# Fused routed-expert modules -> (parameter stem, shard id for RoutedExperts.weight_loader).
FUSED_MODULES = {
    "experts.experts": ("w13", "w1"),
    "experts.output_experts": ("w2", "w2"),
}
# Checkpoint suffix -> parameter-name tail of RoutedExperts (compressed_tensors_moe_w8a8_fp8.py:
# 143-252 and compressed_tensors_moe_w4a4_nvfp4.py:89-189 create exactly these).
FUSED_SUFFIXES = {
    "weight": "weight",
    "weight_scale": "weight_scale",
    "weight_packed": "weight_packed",
    "weight_global_scale": "weight_global_scale",
    "input_global_scale": "input_global_scale",
    "input_scale": "input_scale",
}
# The serialized experts_int8 layout keeps its scales in w13_scale / w2_scale (olala.py:1372-1392).
LEGACY_FUSED_SCALE = {"w13": "w13_scale", "w2": "w2_scale"}
# Created by CompressedTensorsW4A4Nvfp4MoEMethod even for weight-only checkpoints
# (compressed_tensors_moe_w4a4_nvfp4.py:170-189) and read, unconditionally, by
# process_weights_after_loading (:248-253). A weight-only checkpoint cannot carry them.
UNUSED_A16_PARAMS = ("w13_input_global_scale", "w2_input_global_scale")
QUANT_STORAGE_DTYPES = frozenset({torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8, torch.int8})
_MLP_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.(.+)$")


class OlalaQuantLoadError(RuntimeError):
    """A quantized Olala checkpoint cannot be loaded as given (always raised, never swallowed)."""


@dataclass(frozen=True)
class Route:
    """Where one checkpoint tensor goes.

    Attributes:
        kind: ``direct`` (same name), ``rename`` (vLLM name in ``target``), ``expert_bias``,
            ``drop`` (non-persistent buffer), ``fused_experts`` (scatter into a ``RoutedExperts``
            parameter) or ``unsupported_fused`` (a fused-expert suffix this loader does not know).
        target: Parameter or buffer name (unused for ``fused_experts`` and ``drop``).
        layer: Layer index for MoE-layer tensors.
        fused_param: ``RoutedExperts`` attribute for ``fused_experts`` (e.g. ``w13_weight_packed``).
        shard_id: ``w1`` or ``w2`` for ``fused_experts``.
        loader_name: Name handed to ``RoutedExperts.weight_loader``, which dispatches on substrings
            of it (``weight``, ``scale``, ``input_scale`` ...; routed_experts.py:826-1000).
        legacy_param: Alternative ``RoutedExperts`` attribute used by the serialized experts_int8
            layout when ``fused_param`` does not exist.
    """

    kind: str
    target: str | None = None
    layer: int | None = None
    fused_param: str | None = None
    shard_id: str | None = None
    loader_name: str | None = None
    legacy_param: str | None = None


def route_checkpoint_name(name: str, moe_layers: Collection[int]) -> Route:
    """Classify a checkpoint tensor name (see :class:`Route`)."""
    m = _MLP_RE.match(name)
    if m is None or int(m.group(1)) not in moe_layers:
        return Route("direct", target=name)
    layer, tail = int(m.group(1)), m.group(2)
    prefix = f"model.layers.{layer}.mlp"
    if tail == "tokens_per_expert":
        return Route("drop", layer=layer)
    if tail == "expert_bias":
        return Route("expert_bias", target=f"{prefix}.gate.e_score_correction_bias", layer=layer)
    module, _, suffix = tail.rpartition(".")
    if module in FUSED_MODULES:
        stem, shard = FUSED_MODULES[module]
        if suffix not in FUSED_SUFFIXES:
            return Route("unsupported_fused", target=name, layer=layer)
        param = f"{stem}_{FUSED_SUFFIXES[suffix]}"
        return Route(
            "fused_experts",
            target=name,
            layer=layer,
            fused_param=param,
            shard_id=shard,
            loader_name=f"{prefix}.experts.{param}",
            legacy_param=LEGACY_FUSED_SCALE[stem] if suffix == "weight_scale" else None,
        )
    if module in MLP_MODULE_RENAMES:
        return Route("rename", target=f"{prefix}.{MLP_MODULE_RENAMES[module]}.{suffix}", layer=layer)
    return Route("direct", target=name, layer=layer)


def check_dtype_contract(name: str, loaded: torch.dtype, param: torch.dtype) -> None:
    """Refuse to copy across the quantized-storage boundary.

    ``Tensor.copy_`` converts silently, so an fp8 or packed tensor loaded into a bf16 parameter (or
    a bf16 tensor into an fp8 parameter) would 'work' and produce garbage. A quantized storage dtype
    must match exactly; a bf16/fp32 tensor may go into any non-quantized parameter.
    """
    loaded_q, param_q = loaded in QUANT_STORAGE_DTYPES, param in QUANT_STORAGE_DTYPES
    if loaded_q != param_q or (loaded_q and loaded != param):
        raise OlalaQuantLoadError(
            f"{name}: the checkpoint tensor is {loaded} but the model parameter is {param}. "
            "config.json's quantization_config and the tensors disagree: a tensor stored quantized "
            "must belong to a module the quantization_config quantizes, and a bf16 tensor to a "
            "module listed in quantization_config.ignore (exact vLLM module names; the routed "
            "experts need a 're:' regex entry because they are matched as '<layer>.<expert>.<proj>')."
        )


def expected_slice_shape(fused_param: str, param_shape: tuple[int, ...]) -> tuple[int, ...]:
    """Shape of one expert's slice of a fused checkpoint tensor, from the parameter shape.

    Scalars (global and input scales) arrive as one number per expert; everything else keeps the
    parameter's trailing dimensions (``[out, in]``, ``[out, 1]``, ``[out, in/16]``...).
    """
    if fused_param.endswith(("global_scale", "input_scale")):
        return ()
    return tuple(param_shape[1:])


def check_slice_shape(
    name: str,
    slice_shape: tuple[int, ...],
    fused_param: str,
    param_shape: tuple[int, ...],
    tp_size: int = 1,
) -> None:
    """Check one expert slice against the parameter. Skipped for tensor parallel (shapes differ).

    The rank must match and no dimension may exceed the parameter's. A smaller dimension is accepted:
    backends that round sizes up allocate padded parameters and the loader narrows them
    (routed_experts.py:494-527). A trailing dimension of 1 against none, or the reverse, is a rank
    mismatch and is refused (``weight_scale`` ``[out]`` against ``[out, 1]`` would otherwise
    broadcast or fail inside the loader).
    """
    if tp_size != 1:
        return
    expected = expected_slice_shape(fused_param, tuple(param_shape))
    got = tuple(slice_shape)
    if len(got) != len(expected) or any(g > e for g, e in zip(got, expected, strict=True)):
        raise OlalaQuantLoadError(
            f"{name}: per-expert shape {got} but parameter {fused_param} expects "
            f"{expected} (parameter shape {tuple(param_shape)})."
        )


def scatter_fused_expert_tensor(
    param: Any,
    loaded_weight: torch.Tensor,
    route: Route,
    *,
    num_experts: int,
    tp_size: int = 1,
) -> int:
    """Scatter a fused ``[E, ...]`` checkpoint tensor into a ``RoutedExperts`` parameter.

    Each expert goes through ``param.weight_loader(param, slice, route.loader_name, shard_id=...,
    expert_id=..., return_success=True)``, the loader's own dispatch for weights, group scales,
    channel scales and per-tensor scales.

    Returns:
        The number of experts the loader accepted (fewer than ``num_experts`` under expert
        parallelism, where each rank keeps its own).

    Raises:
        OlalaQuantLoadError: On a dtype or shape mismatch, a wrong expert count, or when the loader
            accepted no expert at all.
    """
    name = route.target or route.loader_name or "<fused tensor>"
    check_dtype_contract(name, loaded_weight.dtype, param.dtype)
    if loaded_weight.ndim == 0 or loaded_weight.shape[0] != num_experts:
        raise OlalaQuantLoadError(
            f"{name}: {tuple(loaded_weight.shape)} does not hold {num_experts} experts on dim 0."
        )
    assert route.fused_param is not None and route.shard_id is not None
    check_slice_shape(name, tuple(loaded_weight.shape[1:]), route.fused_param, tuple(param.shape), tp_size)
    accepted = 0
    for expert_id in range(num_experts):
        accepted += bool(
            param.weight_loader(
                param,
                loaded_weight[expert_id],
                route.loader_name,
                shard_id=route.shard_id,
                expert_id=expert_id,
                return_success=True,
            )
        )
    if not accepted:
        raise OlalaQuantLoadError(
            f"{name}: no expert was accepted by the loader of {route.fused_param} "
            "(with expert parallelism at least one expert must land on each rank)."
        )
    return accepted


class LoadTracker:
    """Collect checkpoint tensors that no parameter consumed, then fail loudly with all of them."""

    def __init__(self) -> None:
        """Start empty."""
        self.consumed: set[str] = set()
        self.rejected: list[tuple[str, str]] = []

    def consume(self, name: str) -> None:
        """Record a tensor that was loaded."""
        self.consumed.add(name)

    def reject(self, name: str, reason: str) -> None:
        """Record a tensor that nothing could consume."""
        self.rejected.append((name, reason))

    def raise_if_unconsumed(self, allow_regex: str | None) -> list[str]:
        """Raise listing every unconsumed tensor not matched by ``allow_regex``.

        Returns:
            The names that were tolerated because they matched ``allow_regex``.
        """
        pattern = re.compile(allow_regex) if allow_regex else None
        tolerated = [n for n, _ in self.rejected if pattern and pattern.fullmatch(n)]
        bad = [(n, r) for n, r in self.rejected if not (pattern and pattern.fullmatch(n))]
        if bad:
            lines = "\n".join(f"  {n}: {r}" for n, r in bad)
            raise OlalaQuantLoadError(
                f"{len(bad)} checkpoint tensor(s) were not consumed by any model parameter:\n{lines}\n"
                "A quantized tensor with no matching parameter usually means the module is listed in "
                "quantization_config.ignore (or not targeted) while the checkpoint stores it "
                "quantized. To tolerate known-harmless names set OLALA_ALLOW_UNCONSUMED to a regex "
                "that fully matches them."
            )
        return tolerated


def _normalise(name: str) -> str:
    return name.replace(".routed_experts.", ".")


def unloaded_params(expected: Iterable[str], loaded: Collection[str]) -> list[str]:
    """Expected parameter names with no loaded counterpart.

    ``RoutedExperts`` parameters may be reported under ``...experts.routed_experts.w13_weight`` or
    ``...experts.w13_weight`` depending on which module path reaches them first (olala.py:264-266),
    so the ``routed_experts`` segment is ignored on both sides.
    """
    done = {_normalise(n) for n in loaded}
    return [n for n in expected if _normalise(n) not in done]
