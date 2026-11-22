"""Component ablation: quantize only a chosen part of the model, keep the rest in bf16.

The question this answers is "which part of the model causes the NVFP4 loss". A *selection* names a
component class (and optionally a layer range); every quantizable tensor outside it is added to the
ignore list, exactly as list I is, so the existing machinery does the rest: the plan keeps those tensors
bf16, ``quantization_config.ignore`` names their vLLM modules (and the ``re:`` regex of the routed
experts), the verifier re-derives the same plan from ``quantization_info.json``.

Classes (the five *components* partition the 259 quantizable tensors of the model):

==============  ===================================================================  ==========
name            tensors                                                              components
==============  ===================================================================  ==========
experts         routed experts, ``mlp.experts.experts`` and ``mlp.experts.output_..``  experts
shared-experts  ``mlp.shared_experts.fc_1`` and ``fc_2``                             shared-experts
latent          ``mlp.down_proj`` and ``mlp.up_proj`` (the latent reduce and expand)  latent
attention       ``mixer_proj`` and ``gate_proj`` of the attention (V) layers          attention
mamba           ``mixer_proj`` of the Mamba-3 (M) layers                              mamba
non-experts     shared-experts, latent, attention and mamba together                 four of them
all             the five                                                              every one
==============  ===================================================================  ==========

A spec is ``NAME`` or ``NAME:layers=RANGES`` where RANGES is a comma list of ``N`` or ``A-B`` (both ends
included), for example ``experts:layers=12-23``. List I (``recipes.DEFAULT_IGNORE_I``, or the base list
the caller gives) is ALWAYS kept in bf16, whatever the selection: it is the base of the ignore list and a
selection only adds to it.

One selection is the deliberate exception, ``experts-all`` (PROTOCOL.md amendment 25, candidates C1 and C2):
it quantizes the routed experts (up and down tensors) of EVERY layer, including the MoE layers 0, 1, 5 and 19
that list I protects. It therefore LIFTS the patterns of the base list that cover a routed-expert tensor (the
four ``model.layers.{0,1,5,19}.mlp.*``) and keeps the others (the mixer projections of layers 1, 7, 23). What
the lifted patterns also protected (the shared experts and the latent projections of those four layers) stays
bf16, because those tensors are not selected and are therefore added to the ignore list by exact name, as for
every selection. ``quantization_info.json`` says so in its ``selection`` block: ``list_i_lifted``,
``lifted_ignore_patterns`` and ``kept_ignore_patterns``. These keys exist only for a selection that lifts; the
block of every other selection is what it always was. ``experts-all`` is kept out of :data:`SELECT_NAMES` (the
grammar of the older pod scripts) in :data:`EXTENDED_SELECTIONS`.

Which layers are Mamba and which attention is read from the header (``mixer.in_proj`` against
``mixer.c_q``), not assumed. :func:`expected_counts` predicts the number printed by the server
(``Quantizing N layers of types: ...``, one count per vLLM module, the two routed-expert tensors of a
layer being one module) and the number of tensors that stay bf16; :func:`parse_quantizing_line` reads the
server's line back. A server that loads a checkpoint (not online quantization) does not print that line
(smoke.json of arm N2: ``quantized_layers`` null), so ``check`` also counts what is on disk.

Usage::

    python -m olala_quant.ablation expected (--src BF16_DIR | --header HEADER.json) --select SPEC \\
        [--ignore-file F]
    python -m olala_quant.ablation check --src BF16_DIR --dst CKPT_DIR --select SPEC [--server-log FILE]
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from olala_quant.checkpoint_io import INDEX_NAME, SourceCheckpoint, TensorMeta, header_from_json
from olala_quant.plan import IgnoreListError, Plan, build_plan, load_ignore_file
from olala_quant.recipes import DEFAULT_IGNORE_I, RECIPES, Recipe, classify_tensor

COMPONENTS: tuple[str, ...] = ("experts", "shared-experts", "latent", "attention", "mamba")
SELECT_NAMES: dict[str, frozenset[str]] = {
    "all": frozenset(COMPONENTS),
    "experts": frozenset({"experts"}),
    "non-experts": frozenset(COMPONENTS) - {"experts"},
    "shared-experts": frozenset({"shared-experts"}),
    "latent": frozenset({"latent"}),
    "attention": frozenset({"attention"}),
    "mamba": frozenset({"mamba"}),
}
# Selections that quantize a part of the model list I protects: they lift the base patterns that cover a
# routed-expert tensor (see the module docstring). Same grammar as SELECT_NAMES (NAME or NAME:layers=RANGES).
EXTENDED_SELECTIONS: dict[str, frozenset[str]] = {
    "experts-all": frozenset({"experts"}),
}
_ROLE_COMPONENT = {
    "routed_up": "experts",
    "routed_down": "experts",
    "shared_up": "shared-experts",
    "shared_down": "shared-experts",
    "latent_reduce": "latent",
    "latent_expand": "latent",
    "attn_gate_proj": "attention",
}
# ``mixer_proj`` is the output projection of every layer: attention or Mamba by the kind of its layer.
_MIXER_ROLE = "mixer_proj"
_ROUTED_ROLES = frozenset({"routed_up", "routed_down"})
_LAYER_TENSOR = re.compile(r"^model\.layers\.(\d+)\.")
_MAMBA_MARK = re.compile(r"^model\.layers\.(\d+)\.mixer\.in_proj\.weight$")
_ATTENTION_MARK = re.compile(r"^model\.layers\.(\d+)\.mixer\.c_q\.weight$")
_RANGE = re.compile(r"^(\d+)(?:-(\d+))?$")
_QUANTIZING = re.compile(r"Quantizing (\d+) layers of types: (.*)")
_TYPE_COUNT = re.compile(r"([\w.]+): (\d+) \(")


class SelectionError(ValueError):
    """A selection spec that is malformed, outside the model, or that would quantize nothing."""


@dataclass(frozen=True)
class Selection:
    """A parsed spec.

    Attributes:
        name: A key of :data:`SELECT_NAMES`.
        components: The components it quantizes (a subset of :data:`COMPONENTS`).
        intervals: Sorted, disjoint, non-adjacent ``(first, last)`` layer intervals, both ends
            included; ``None`` for every layer.
        lifts_routed: True for a selection that lifts the base-list patterns covering routed experts.
    """

    name: str
    components: frozenset[str]
    intervals: tuple[tuple[int, int], ...] | None
    lifts_routed: bool = False

    @property
    def spec(self) -> str:
        """The canonical spec: merged and sorted intervals, one spelling per selection."""
        if self.intervals is None:
            return self.name
        parts = [str(a) if a == b else f"{a}-{b}" for a, b in self.intervals]
        return f"{self.name}:layers={','.join(parts)}"

    def has_layer(self, layer: int) -> bool:
        """Whether ``layer`` is in the selected layer ranges (always True without ranges)."""
        return self.intervals is None or any(a <= layer <= b for a, b in self.intervals)


def _merge(ranges: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    merged: list[tuple[int, int]] = []
    for a, b in sorted(ranges):
        if merged and a <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return tuple(merged)


def parse_select(spec: str) -> Selection:
    """Parse ``NAME`` or ``NAME:layers=RANGES``; anything else is a :class:`SelectionError`."""
    name, *options = spec.split(":")
    if name not in SELECT_NAMES and name not in EXTENDED_SELECTIONS:
        known = sorted([*SELECT_NAMES, *EXTENDED_SELECTIONS])
        raise SelectionError(f"unknown selection {name!r} in {spec!r}; choose one of {known}")
    intervals: tuple[tuple[int, int], ...] | None = None
    seen: set[str] = set()
    for option in options:
        key, eq, value = option.partition("=")
        if not eq or not value:
            raise SelectionError(f"{spec!r}: option {option!r} is not key=value")
        if key != "layers":
            raise SelectionError(f"{spec!r}: unknown option {key!r}; the only option is layers=RANGES")
        if key in seen:
            raise SelectionError(f"{spec!r}: option {key!r} given twice")
        seen.add(key)
        ranges = []
        for part in value.split(","):
            m = _RANGE.match(part)
            if m is None:
                raise SelectionError(f"{spec!r}: {part!r} is not a layer N or a range A-B")
            first = int(m.group(1))
            last = int(m.group(2)) if m.group(2) is not None else first
            if last < first:
                raise SelectionError(f"{spec!r}: the range {part!r} runs backwards")
            ranges.append((first, last))
        intervals = _merge(ranges)
    if name in EXTENDED_SELECTIONS:
        return Selection(name, EXTENDED_SELECTIONS[name], intervals, lifts_routed=True)
    return Selection(name, SELECT_NAMES[name], intervals)


def layer_kinds(header: Mapping[str, TensorMeta]) -> dict[int, str]:
    """Map every layer index to ``M`` (Mamba-3) or ``V`` (attention), read from the tensor names."""
    layers = {int(m.group(1)) for n in header if (m := _LAYER_TENSOR.match(n))}
    kinds: dict[int, str] = {}
    for layer in sorted(layers):
        mamba = f"model.layers.{layer}.mixer.in_proj.weight" in header
        attention = f"model.layers.{layer}.mixer.c_q.weight" in header
        if mamba == attention:
            raise SelectionError(
                f"layer {layer} has {'both' if mamba else 'neither'} a Mamba in_proj and an attention c_q: "
                "cannot tell which kind of layer it is"
            )
        kinds[layer] = "M" if mamba else "V"
    return kinds


def component_of(role: str, layer: int, kinds: Mapping[int, str]) -> str:
    """The component (one of :data:`COMPONENTS`) of a quantizable tensor."""
    if role == _MIXER_ROLE:
        return "attention" if kinds[layer] == "V" else "mamba"
    try:
        return _ROLE_COMPONENT[role]
    except KeyError:
        raise SelectionError(
            f"role {role!r} belongs to no component; add it to ablation._ROLE_COMPONENT"
        ) from None


@dataclass(frozen=True)
class ResolvedSelection:
    """A selection applied to a header.

    Attributes:
        selection: The parsed spec.
        base_patterns: The base ignore list as the caller gave it (list I).
        patterns: The base patterns that are kept, followed by one exact name per deselected tensor; this is
        the
            ignore list to give :func:`olala_quant.plan.build_plan`. It equals ``base_patterns`` followed by
            the deselected names unless the selection lifts part of the base list.
        lifted: Base patterns the selection dropped because they cover a routed-expert tensor (empty unless
        the
            selection lifts, see :data:`EXTENDED_SELECTIONS`).
        deselected: Quantizable tensors the selection leaves in bf16 that the base list does not already
            hold, sorted.
        plan: The plan these patterns produce (FP8 recipe unless one was given: only the classification
            is used here).
    """

    selection: Selection
    base_patterns: tuple[str, ...]
    patterns: tuple[str, ...]
    deselected: tuple[str, ...]
    plan: Plan
    lifted: tuple[str, ...] = ()

    def record(self) -> dict[str, Any]:
        """The block written to ``quantization_info.json`` under ``selection``."""
        record: dict[str, Any] = {
            "spec": self.selection.spec,
            "name": self.selection.name,
            "components": sorted(self.selection.components),
            "layers": None
            if self.selection.intervals is None
            else [list(i) for i in self.selection.intervals],
            "base_ignore_patterns": list(self.base_patterns),
            "deselected_tensors": list(self.deselected),
            "expected": counts_from_plan(self.plan, self.selection.spec),
        }
        # these keys are absent for every other selection: their block stays what it always was
        if self.selection.lifts_routed:
            record["list_i_lifted"] = True
            record["lifted_ignore_patterns"] = list(self.lifted)
            record["kept_ignore_patterns"] = [p for p in self.base_patterns if p not in self.lifted]
        return record


def resolve_selection(
    header: Mapping[str, TensorMeta],
    spec: str,
    base_patterns: Sequence[str] | None,
    recipe: Recipe | None = None,
) -> ResolvedSelection:
    """Apply ``spec`` to ``header`` on top of the always-bf16 ``base_patterns``.

    Raises:
        SelectionError: For a malformed spec, a layer range outside the model, or a selection that
            quantizes no tensor once the base list has taken its share.
    """
    selection = parse_select(spec)
    kinds = layer_kinds(header)
    if selection.intervals is not None:
        lo, hi = min(kinds), max(kinds)
        for a, b in selection.intervals:
            if a < lo or b > hi:
                raise SelectionError(f"{selection.spec!r}: the model has layers {lo} to {hi}, not {a}-{b}")
    base = tuple(base_patterns or ())
    lifted: tuple[str, ...] = ()
    if selection.lifts_routed:
        routed = [n for n in header if classify_tensor(n)[0] in _ROUTED_ROLES]
        lifted = tuple(p for p in base if any(fnmatch.fnmatchcase(n, p) for n in routed))
    kept = tuple(p for p in base if p not in lifted)
    held = {n for n in header if any(fnmatch.fnmatchcase(n, p) for p in kept)}
    selected: set[str] = set()
    quantizable: list[str] = []
    for name in sorted(header):
        role, spec_role, layer = classify_tensor(name)
        if spec_role is None or layer is None:
            continue
        quantizable.append(name)
        if component_of(role, layer, kinds) in selection.components and selection.has_layer(layer):
            selected.add(name)
    deselected = tuple(n for n in quantizable if n not in selected and n not in held)
    patterns = kept + tuple(glob.escape(n) for n in deselected)
    try:
        plan = build_plan(header, recipe or RECIPES["fp8"], patterns)
    except IgnoreListError as e:  # cannot happen for a non-empty pattern list; keep the cause readable
        raise SelectionError(f"{selection.spec!r}: {e}") from e
    if not any(t.category == "quantized" for t in plan.tensors):
        raise SelectionError(
            f"selection {selection.spec!r} quantizes no tensor: what it selects is held in bf16 by the "
            "base ignore list (list I) or does not exist in this model"
        )
    return ResolvedSelection(selection, base, patterns, deselected, plan, lifted)


def _type_of(module: str, layer: int) -> str:
    return module.removeprefix(f"model.layers.{layer}.")


def counts_from_plan(plan: Plan, spec: str = "") -> dict[str, Any]:
    """Counts of a plan in the units of the server's ``Quantizing N layers`` line and of the tensors."""
    quantized = [t for t in plan.tensors if t.category == "quantized"]
    ignored = [t for t in plan.tensors if t.category == "ignored"]
    modules = {(t.layer, t.vllm_module) for t in quantized}
    by_type: dict[str, int] = {}
    for layer, module in modules:
        assert layer is not None and module is not None
        key = _type_of(module, layer)
        by_type[key] = by_type.get(key, 0) + 1
    all_modules = {(t.layer, t.vllm_module) for t in quantized + ignored}
    return {
        "spec": spec,
        "quantized_tensors": len(quantized),
        "ignored_tensors": len(ignored),
        "quantized_layers": len(modules),
        "quantized_layers_by_type": dict(sorted(by_type.items())),
        "ignored_layers": len(all_modules) - len(modules),
        "total_quantizable_layers": len(all_modules),
        "ignore_entries": len(plan.vllm_ignore),
    }


def expected_counts(
    header: Mapping[str, TensorMeta],
    spec: str,
    base_patterns: Sequence[str] | None = DEFAULT_IGNORE_I,
    recipe: Recipe | None = None,
) -> dict[str, Any]:
    """What a conversion with ``spec`` quantizes and what the server must report.

    ``quantized_layers`` is the N of the server's ``Quantizing N layers of types: ...`` line (online
    quantization), ``quantized_layers_by_type`` its per-type counts; ``quantized_tensors`` and
    ``ignored_tensors`` count the source tensors (a routed-expert layer is two tensors, one module).
    """
    resolved = resolve_selection(header, spec, base_patterns, recipe)
    return counts_from_plan(resolved.plan, resolved.selection.spec)


def parse_quantizing_line(text: str) -> tuple[int, dict[str, int]]:
    """Read ``Quantizing N layers of types: a: 1 (...); b: 2 (...)`` into ``(N, {type: count})``."""
    m = _QUANTIZING.search(text)
    if m is None:
        raise SelectionError("no 'Quantizing N layers of types' line in the text")
    total = int(m.group(1))
    by_type = {t: int(c) for t, c in _TYPE_COUNT.findall(m.group(2))}
    if sum(by_type.values()) != total:
        raise SelectionError(f"the per-type counts {by_type} do not sum to the announced {total}")
    return total, by_type


# ---------------------------------------------------------------------------------------------- CLI


def _base_patterns(args: argparse.Namespace) -> tuple[str, ...]:
    if args.ignore_file is not None:
        return load_ignore_file(args.ignore_file)
    return DEFAULT_IGNORE_I


def check_checkpoint(
    src: Path, dst: Path, spec: str, base_patterns: Sequence[str], server_log: Path | None = None
) -> list[str]:
    """Compare a converted checkpoint (and a server log) with the prediction; return the problems."""
    problems: list[str] = []
    header = SourceCheckpoint(src).header()
    resolved = resolve_selection(header, spec, base_patterns)
    want = counts_from_plan(resolved.plan, resolved.selection.spec)
    info = json.loads((dst / "quantization_info.json").read_text())
    sel = info.get("selection")
    if sel is None:
        problems.append("quantization_info.json has no selection block: not an ablation checkpoint")
    else:
        if sel.get("spec") != resolved.selection.spec:
            problems.append(
                f"selection recorded as {sel.get('spec')!r}, asked for {resolved.selection.spec!r}"
            )
        if sel.get("expected") != want:
            problems.append(f"recorded expected counts {sel.get('expected')} differ from {want}")
    weight_map = json.loads((dst / INDEX_NAME).read_text())["weight_map"]
    n_on_disk = sum(1 for name in weight_map if name.endswith(".weight_scale"))
    if n_on_disk != want["quantized_tensors"]:
        problems.append(f"{n_on_disk} quantized tensors on disk, expected {want['quantized_tensors']}")
    cfg = json.loads((dst / "config.json").read_text())
    if cfg.get("quantization_config", {}).get("ignore") != list(resolved.plan.vllm_ignore):
        problems.append("config.json quantization_config.ignore differs from the one the selection implies")
    if server_log is not None:
        try:
            n, by_type = parse_quantizing_line(server_log.read_text(errors="replace"))
        except SelectionError:
            pass  # a server that loads a checkpoint does not print the line: the disk check above stands
        else:
            if (n, by_type) != (want["quantized_layers"], want["quantized_layers_by_type"]):
                problems.append(
                    f"server quantizes {n} layers {by_type}, expected "
                    f"{want['quantized_layers']} {want['quantized_layers_by_type']}"
                )
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: ``expected`` prints the prediction, ``check`` compares a checkpoint."""
    p = argparse.ArgumentParser(prog="python -m olala_quant.ablation", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("expected", help="print the expected counts as JSON")
    where = exp.add_mutually_exclusive_group(required=True)
    where.add_argument("--src", type=Path, help="bf16 checkpoint directory")
    where.add_argument("--header", type=Path, help="a safetensors header as JSON (no weights needed)")
    chk = sub.add_parser("check", help="check a converted checkpoint (and a server log)")
    chk.add_argument("--src", type=Path, required=True, help="bf16 checkpoint directory")
    chk.add_argument("--dst", type=Path, required=True, help="converted checkpoint directory")
    chk.add_argument("--server-log", type=Path, help="vLLM server log (its Quantizing line is compared)")
    for s in (exp, chk):
        s.add_argument("--select", required=True, metavar="SPEC")
        s.add_argument("--ignore-file", type=Path, help="base ignore list (default: list I)")
    args = p.parse_args(argv)
    try:
        base = _base_patterns(args)
        if args.command == "expected":
            header = (
                header_from_json(json.loads(args.header.read_text()))
                if args.header is not None
                else SourceCheckpoint(args.src).header()
            )
            counts = expected_counts(header, args.select, base)
            print(json.dumps(counts, indent=2))
            return 0
        problems = check_checkpoint(args.src, args.dst, args.select, base, args.server_log)
    except (SelectionError, IgnoreListError, FileNotFoundError, KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for line in problems:
        print(f"MISMATCH: {line}", file=sys.stderr)
    if problems:
        return 1
    print(f"ABLATION_CHECK_OK {args.select}")
    return 0


__all__ = [
    "COMPONENTS",
    "EXTENDED_SELECTIONS",
    "SELECT_NAMES",
    "ResolvedSelection",
    "Selection",
    "SelectionError",
    "check_checkpoint",
    "component_of",
    "counts_from_plan",
    "expected_counts",
    "layer_kinds",
    "parse_quantizing_line",
    "parse_select",
    "resolve_selection",
]


if __name__ == "__main__":
    raise SystemExit(main())
