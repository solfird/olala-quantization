"""Conversion plan: which tensors are quantized, ignored or copied, and what gets written.

The plan is built from a safetensors header only (names, dtypes, shapes), so it can be computed,
tested and budgeted without any weight data.
"""

from __future__ import annotations

import fnmatch
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from olala_quant.checkpoint_io import DTYPE_BYTES, TensorMeta
from olala_quant.recipes import QUANT_ROLES, Recipe, RoleSpec, classify_tensor

_ROUTED_ROLES = {"routed_up", "routed_down"}
_FORCED_REASON = (
    "vLLM RoutedExperts fuses w13 and w2 under one quant method (olala.py:296-326 builds one "
    "FusedMoEFactory per layer; compressed_tensors_moe.py:36-57 needs one scheme for all its "
    "projections), so ignoring one routed tensor of a layer ignores both"
)


class IgnoreListError(ValueError):
    """The ignore list does not do what it says (typo, or it excludes nothing)."""


@dataclass(frozen=True)
class OutputTensor:
    """A tensor of the output checkpoint: name, safetensors dtype, shape."""

    name: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        """Size in bytes."""
        n = 1
        for d in self.shape:
            n *= d
        return n * DTYPE_BYTES[self.dtype]


@dataclass(frozen=True)
class PlannedTensor:
    """One source tensor and what happens to it.

    Attributes:
        name: HF tensor name in the source checkpoint.
        meta: Source dtype and shape.
        role: Tensor family (see :mod:`olala_quant.recipes`).
        category: ``quantized``, ``ignored`` (quantizable but kept bf16 by the ignore list) or
            ``kept`` (never quantized).
        vllm_module: vLLM module name for quantizable tensors, else None.
        layer: Layer index for quantizable tensors, else None.
    """

    name: str
    meta: TensorMeta
    role: str
    category: str
    vllm_module: str | None
    layer: int | None


@dataclass(frozen=True)
class Plan:
    """Everything the converter needs to know, derived from a header.

    Attributes:
        recipe: The recipe applied.
        ignore_patterns: The patterns given (empty for ``--no-ignore``).
        ignored_resolved: Source tensor names matched by any pattern (quantizable or not).
        ignored_quantizable: Quantizable tensors left in bf16 (matched or forced).
        forced: Names added to ``ignored_quantizable`` by a vLLM constraint, with the reason.
        tensors: Every source tensor in name order.
        vllm_ignore: ``quantization_config.ignore`` entries in vLLM module naming.
    """

    recipe: Recipe
    ignore_patterns: tuple[str, ...]
    ignored_resolved: tuple[str, ...]
    ignored_quantizable: tuple[str, ...]
    forced: Mapping[str, str]
    tensors: tuple[PlannedTensor, ...]
    vllm_ignore: tuple[str, ...]

    def outputs_of(self, t: PlannedTensor) -> list[OutputTensor]:
        """The tensors written for source tensor ``t`` (itself, unless it is quantized)."""
        if t.category != "quantized":
            return [OutputTensor(t.name, t.meta.dtype, t.meta.shape)]
        base = t.name.removesuffix(".weight")
        return [
            OutputTensor(f"{base}.{spec.suffix}", spec.dtype, spec.shape)
            for spec in self.recipe.output_specs(t.meta)
        ]

    def output_tensors(self) -> list[OutputTensor]:
        """Every tensor of the output checkpoint, in source-name order."""
        return [o for t in self.tensors for o in self.outputs_of(t)]

    def output_bytes(self) -> int:
        """Total size of the output tensors in bytes (safetensors headers excluded)."""
        return sum(o.nbytes for o in self.output_tensors())

    def total_params(self) -> int:
        """Number of source elements."""
        return sum(t.meta.numel for t in self.tensors)

    def category_accounting(self) -> dict[str, dict[str, int]]:
        """Counts, parameters and bytes (in and out) per role; ignored tensors under ``ignored:``."""
        acct: dict[str, dict[str, int]] = {}
        for t in self.tensors:
            key = f"ignored:{t.role}" if t.category == "ignored" else t.role
            row = acct.setdefault(key, {"count": 0, "params": 0, "bytes_in": 0, "bytes_out": 0})
            row["count"] += 1
            row["params"] += t.meta.numel
            row["bytes_in"] += t.meta.nbytes
            row["bytes_out"] += sum(o.nbytes for o in self.outputs_of(t))
        return acct


def load_ignore_file(path: Path) -> tuple[str, ...]:
    """Read an ignore list: a JSON list of patterns or ``{"patterns": [...]}``."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        if "patterns" not in data:
            raise ValueError(f"{path}: an object ignore file needs a 'patterns' list")
        data = data["patterns"]
    if not isinstance(data, list) or not all(isinstance(p, str) for p in data):
        raise ValueError(f"{path}: ignore patterns must be a list of strings")
    return tuple(data)


def _vllm_ignore_entries(chosen: Sequence[PlannedTensor]) -> tuple[str, ...]:
    """Map ignored quantizable tensors to ``quantization_config.ignore`` entries.

    Plain entries are exact module names. RoutedExperts gets a regex as well: compressed-tensors
    resolves it through the names ``<layer>.<expert>.<proj>`` (compressed_tensors_moe.py:36-49)
    and plain entries only match exactly (utils/config_utils.py:443-452).
    """
    entries = {"lm_head"}
    for t in chosen:
        assert t.vllm_module is not None
        entries.add(t.vllm_module)
        if t.role in _ROUTED_ROLES:
            entries.add("re:" + re.escape(t.vllm_module) + r"\..*")
    return tuple(sorted(entries))


def build_plan(
    header: Mapping[str, TensorMeta], recipe: Recipe, ignore_patterns: Sequence[str] | None
) -> Plan:
    """Classify every tensor of ``header`` for ``recipe`` under ``ignore_patterns``.

    Args:
        header: Source tensor names, dtypes and shapes.
        recipe: The recipe to apply.
        ignore_patterns: fnmatch patterns over HF tensor names; ``None`` or empty keeps nothing
            in bf16 on purpose.

    Raises:
        IgnoreListError: If a pattern matches no tensor, or the list matches no tensor that would
            otherwise be quantized.
        ValueError: If a quantizable tensor has an unexpected dtype, rank or width.
    """
    patterns = tuple(ignore_patterns or ())
    names = sorted(header)
    matched: set[str] = set()
    for pattern in patterns:
        hits = [n for n in names if fnmatch.fnmatchcase(n, pattern)]
        if not hits:
            raise IgnoreListError(f"ignore pattern {pattern!r} matches no tensor of the source")
        matched.update(hits)

    classified = {n: classify_tensor(n) for n in names}
    ignored_q = {n for n in matched if classified[n][1] is not None}
    if patterns and not ignored_q:
        raise IgnoreListError(
            f"the ignore list {list(patterns)} matches no tensor that would be quantized; "
            "it would change nothing (use --no-ignore to quantize everything on purpose)"
        )

    forced: dict[str, str] = {}
    by_layer: dict[int, dict[str, str]] = {}
    for n in ignored_q:
        role, spec, layer = classified[n]
        if spec is not None and spec.routed and layer is not None:
            by_layer.setdefault(layer, {})[role] = n
    for layer, found in by_layer.items():
        if len(found) == 1:
            missing_role = "routed_down" if "routed_up" in found else "routed_up"
            missing = next(n for n in names if classified[n][0] == missing_role and classified[n][2] == layer)
            forced[missing] = _FORCED_REASON
            ignored_q.add(missing)

    planned: list[PlannedTensor] = []
    for n in names:
        meta = header[n]
        role, spec, layer = classified[n]
        if spec is None:
            planned.append(PlannedTensor(n, meta, role, "kept", None, None))
            continue
        module = spec.vllm_module.format(L=layer)
        if n in ignored_q:
            planned.append(PlannedTensor(n, meta, role, "ignored", module, layer))
            continue
        _check_quantizable(n, meta, spec, recipe)
        planned.append(PlannedTensor(n, meta, role, "quantized", module, layer))

    chosen = [t for t in planned if t.category == "ignored"]
    return Plan(
        recipe=recipe,
        ignore_patterns=patterns,
        ignored_resolved=tuple(sorted(matched)),
        ignored_quantizable=tuple(sorted(ignored_q)),
        forced=forced,
        tensors=tuple(planned),
        vllm_ignore=_vllm_ignore_entries(chosen),
    )


def _check_quantizable(name: str, meta: TensorMeta, spec: RoleSpec, recipe: Recipe) -> None:
    """Raise ValueError if ``meta`` is not a bf16 weight of the expected rank and width."""
    if meta.dtype != "BF16":
        raise ValueError(f"{name}: expected BF16, found {meta.dtype}")
    if len(meta.shape) != spec.ndim:
        raise ValueError(f"{name}: expected rank {spec.ndim}, found shape {meta.shape}")
    if recipe.quantizer == "nvfp4" and meta.shape[-1] % 16:
        raise ValueError(f"{name}: input width {meta.shape[-1]} is not a multiple of 16")


__all__ = [
    "QUANT_ROLES",
    "IgnoreListError",
    "OutputTensor",
    "Plan",
    "PlannedTensor",
    "build_plan",
    "load_ignore_file",
]
