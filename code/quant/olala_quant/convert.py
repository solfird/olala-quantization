"""Convert a bf16 Olala checkpoint to FP8 or NVFP4, shard by shard, with a full audit record.

Usage::

    python -m olala_quant.convert --src DIR --dst DIR --recipe fp8|nvfp4 \\
        [--ignore-file FILE | --no-ignore] [--select SPEC] [--scale-search] [--device cpu|cuda] [--dry-run]

Memory: tensors are read one at a time from memory-mapped safetensors; the only buffer that grows
is the current output shard (at most ``--max-shard-gb`` GB, 5 by default) plus a few tens of
megabytes of scratch per expert chunk. The output shard layout is planned from shapes before any
data is read, so the number of shards in the file names is known up front.

``--scale-search`` (NVFP4 only, off by default, PROTOCOL.md amendment 25) chooses every group's e4m3 block
scale by the MSE-optimal search of :mod:`olala_quant.nvfp4`; the global scales, the tensor names, shapes and
dtypes, the file sizes and the ``quantization_config`` are the plain conversion's. It is recorded as
``scale_search: {factors, version}`` in ``quantization_info.json`` (the key is absent from every conversion
made without it).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import safetensors
import torch
from safetensors.torch import save_file

import olala_quant
from olala_quant import ablation, fp8, nvfp4
from olala_quant.checkpoint_io import SourceCheckpoint
from olala_quant.plan import IgnoreListError, Plan, build_plan, load_ignore_file
from olala_quant.recipes import DEFAULT_IGNORE_I, RECIPES, Recipe, get_recipe

DEFAULT_MAX_SHARD_BYTES = 5_000_000_000
INDEX_FILE = "model.safetensors.index.json"
INFO_FILE = "quantization_info.json"
_SKIP_COPY_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")
_MAX_AUX_BYTES = 100 * 1024 * 1024
_DEFAULT_CHUNK = {"fp8": 32, "nvfp4": 16}


def plan_shards(sizes: Sequence[tuple[str, int]], max_bytes: int) -> list[list[str]]:
    """Greedily pack ``(name, nbytes)`` items, in order, into shards of at most ``max_bytes``.

    An item is never split; an item larger than ``max_bytes`` gets a shard of its own.
    """
    shards: list[list[str]] = []
    current: list[str] = []
    used = 0
    for name, size in sizes:
        if current and used + size > max_bytes:
            shards.append(current)
            current, used = [], 0
        current.append(name)
        used += size
    if current:
        shards.append(current)
    return shards


def quantize_tensor(
    name: str,
    tensor: torch.Tensor,
    recipe: Recipe,
    *,
    device: str = "cpu",
    expert_chunk: int | None = None,
    scale_search: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Quantize one weight (``scale_search``: NVFP4 only, see :func:`olala_quant.nvfp4.quantize_nvfp4`).

    Returns:
        ``({output_name: tensor}, record)`` where ``record`` holds the error and scale statistics.
    """
    base = name.removesuffix(".weight")
    chunk = expert_chunk or _DEFAULT_CHUNK[recipe.quantizer]
    x = tensor.to(device)
    if scale_search and recipe.quantizer != "nvfp4":
        raise ValueError(
            f"the MSE-optimal scale search applies to the nvfp4 recipe only, not {recipe.name!r}"
        )
    if recipe.quantizer == "fp8":
        r8 = fp8.quantize_fp8(x, expert_chunk=chunk)
        outputs = {f"{base}.weight": r8.weight.cpu(), f"{base}.weight_scale": r8.scale.cpu()}
        record: dict[str, Any] = {"scale_stats": r8.scale_stats.as_dict()}
        err = r8.error
    else:
        r4 = nvfp4.quantize_nvfp4(x, expert_chunk=chunk, scale_search=scale_search)
        outputs = {
            f"{base}.weight_packed": r4.packed.cpu(),
            f"{base}.weight_scale": r4.block_scale.cpu(),
            f"{base}.weight_global_scale": r4.global_scale.cpu(),
        }
        record = {
            "scale_stats": r4.scale_stats.as_dict(),
            "global_scale_min": float(r4.global_scale.min()),
            "global_scale_max": float(r4.global_scale.max()),
        }
        err = r4.error
    record.update(
        {
            "shape": list(tensor.shape),
            "max_abs_error": err.max_abs,
            "rel_fro_error": err.rel_fro,
        }
    )
    return outputs, record


def _write_shard(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    """Write ``tensors`` to ``path`` atomically (temporary name, then rename)."""
    tmp = path.with_name(path.name + ".partial")
    save_file(tensors, str(tmp), metadata={"format": "pt"})
    os.replace(tmp, path)


def list_auxiliary_files(src: Path) -> list[Path]:
    """Files next to the weights that travel with the model (tokenizer, template, remote code...)."""
    return [
        f
        for f in sorted(src.iterdir())
        if f.is_file()
        and not f.name.startswith(".")
        and f.name not in ("config.json", INDEX_FILE, INFO_FILE)
        and not f.name.endswith(_SKIP_COPY_SUFFIXES)
        and f.stat().st_size <= _MAX_AUX_BYTES
    ]


def _copy_auxiliary_files(src: Path, dst: Path) -> list[str]:
    """Copy tokenizer, chat template, remote code, license and generation files."""
    copied = []
    for f in list_auxiliary_files(src):
        shutil.copy2(f, dst / f.name)
        copied.append(f.name)
    return copied


def _tool_versions() -> dict[str, str]:
    return {
        "olala_quant": olala_quant.__version__,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "safetensors": safetensors.__version__,
        "platform": platform.platform(),
    }


def _scale_summary(records: dict[str, dict[str, Any]], recipe: Recipe) -> dict[str, Any]:
    """Aggregate scale statistics over all quantized tensors."""
    if recipe.quantizer == "fp8":
        stats = [r["scale_stats"] for r in records.values()]
        return {
            "n_rows": sum(s["n_rows"] for s in stats),
            "n_zero_rows": sum(s["n_zero_rows"] for s in stats),
            "max_scale": max((s["max_scale"] for s in stats), default=0.0),
            "min_scale": min((s["min_scale"] for s in stats), default=0.0),
        }
    stats = [r["scale_stats"] for r in records.values()]
    mins = [s["min_nonzero_block_scale"] for s in stats if s["min_nonzero_block_scale"] > 0]
    return {
        "n_blocks": sum(s["n_blocks"] for s in stats),
        "n_zero_blocks": sum(s["n_zero_blocks"] for s in stats),
        "n_overflow": sum(s["n_overflow"] for s in stats),
        "n_flushed": sum(s["n_flushed"] for s in stats),
        "n_subnormal": sum(s["n_subnormal"] for s in stats),
        "max_block_scale": max((s["max_block_scale"] for s in stats), default=0.0),
        "min_nonzero_block_scale": min(mins) if mins else 0.0,
    }


def _build_info(
    plan: Plan,
    source: SourceCheckpoint,
    header_total: int,
    shard_layout: list[list[str]],
    records: dict[str, dict[str, Any]],
    files: list[dict[str, Any]],
    aux: list[str],
    *,
    dry_run: bool,
    selection: dict[str, Any] | None = None,
    scale_search: bool = False,
) -> dict[str, Any]:
    recipe = plan.recipe
    kind, digest = source.identity_hash()
    out_bytes = plan.output_bytes()
    quant_params = sum(t.meta.numel for t in plan.tensors if t.category == "quantized")
    quant_bytes = sum(
        sum(o.nbytes for o in plan.outputs_of(t)) for t in plan.tensors if t.category == "quantized"
    )
    by_cat: dict[str, dict[str, int]] = {}
    for t in plan.tensors:
        row = by_cat.setdefault(t.category, {"count": 0, "params": 0, "bytes_in": 0, "bytes_out": 0})
        row["count"] += 1
        row["params"] += t.meta.numel
        row["bytes_in"] += t.meta.nbytes
        row["bytes_out"] += (
            sum(o.nbytes for o in plan.outputs_of(t)) if t.category == "quantized" else t.meta.nbytes
        )
    info = {
        "schema_version": 1,
        "dry_run": dry_run,
        "created_utc": datetime.now(UTC).isoformat(),
        "tool": _tool_versions(),
        "recipe": {
            "name": recipe.name,
            "arms": list(recipe.arms),
            "description": recipe.description,
            "weights": recipe.weights,
            "activations": recipe.activations,
            "suffixes": list(recipe.suffixes),
        },
        "quantization_config": recipe.quantization_config(plan.vllm_ignore),
        "ignore": {
            "patterns": list(plan.ignore_patterns),
            "resolved": list(plan.ignored_resolved),
            "quantizable_ignored": list(plan.ignored_quantizable),
            "forced": dict(plan.forced),
            "vllm_ignore": list(plan.vllm_ignore),
        },
        "source": {
            "directory": str(source.directory),
            "identity_kind": kind,
            "identity_sha256": digest,
            "files": source.shard_files(),
            "n_tensors": len(plan.tensors),
            "total_params": header_total,
        },
        "output": {
            "n_tensors": len(plan.output_tensors()),
            "tensor_bytes": out_bytes,
            "file_bytes": sum(f["bytes"] for f in files),
            "n_shards": len(shard_layout),
            "bits_per_weight": out_bytes * 8 / header_total,
            "bits_per_quantized_weight": quant_bytes * 8 / quant_params if quant_params else None,
            "files": files,
            "auxiliary_files": aux,
        },
        "categories": plan.category_accounting(),
        "totals_by_category": by_cat,
        "unclassified": [t.name for t in plan.tensors if t.role == "unclassified"],
        "tensors": records,
        "scale_summary": _scale_summary(records, recipe) if records else None,
    }
    if selection is not None:  # absent from a default conversion, which stays what it always was
        info["selection"] = selection
    if scale_search:  # likewise
        info["scale_search"] = {
            "factors": list(nvfp4.SCALE_SEARCH_FACTORS),
            "version": nvfp4.SCALE_SEARCH_VERSION,
        }
    return info


def convert(
    src: Path,
    dst: Path,
    recipe_name: str,
    ignore_patterns: Sequence[str] | None,
    *,
    device: str = "cpu",
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
    expert_chunk: int | None = None,
    dry_run: bool = False,
    log: Any = print,
    override: Callable[[str, torch.Tensor], tuple[dict[str, torch.Tensor], dict[str, Any]] | None]
    | None = None,
    info_extra: dict[str, Any] | None = None,
    select: str | None = None,
    scale_search: bool = False,
) -> dict[str, Any]:
    """Convert the checkpoint in ``src`` and write the result to ``dst``.

    Args:
        src: Directory with ``model.safetensors`` (or an indexed shard set) and ``config.json``.
        dst: Output directory; must not exist or be empty.
        recipe_name: ``fp8`` or ``nvfp4``.
        ignore_patterns: fnmatch patterns over HF tensor names to keep in bf16; ``()`` or ``None``
            quantizes everything the recipe can.
        device: Torch device for the quantization math.
        max_shard_bytes: Cap per output shard (a single larger tensor gets its own shard).
        expert_chunk: Experts per chunk for fused tensors (default depends on the recipe).
        dry_run: Plan and project sizes only; write nothing.
        log: Callable used for progress lines.
        override: Optional hook ``(tensor name, tensor) -> (outputs, record) | None`` called for every
            tensor the plan quantizes. ``None`` keeps the data-free quantization; a pair replaces it (the
            outputs must have the names, dtypes and shapes of the recipe, ``record`` the keys of
            :func:`quantize_tensor`). The calibrated build (G1) uses it to write codes chosen by a solver.
        info_extra: Optional top-level keys merged into ``quantization_info.json`` (e.g. ``calibration``).
        select: Optional component-ablation spec (see :mod:`olala_quant.ablation`, e.g. ``experts`` or
            ``experts:layers=12-23``): only that part is quantized, every other quantizable tensor is
            added to the ignore list; ``ignore_patterns`` (list I) stays in bf16 whatever the selection.
            The selection is recorded under ``selection`` in ``quantization_info.json``. ``experts-all`` is
            the one selection that lifts part of list I (the MoE patterns), and its record says which
            patterns.
        scale_search: NVFP4 only: MSE-optimal block scales (:mod:`olala_quant.nvfp4`), recorded as
        ``scale_search`` in
            ``quantization_info.json``. Not available with ``override`` (a calibrated build keeps the plain
            scales).

    Returns:
        The content of ``quantization_info.json``.

    Raises:
        NotImplementedError: For a recipe that is declared but not implemented.
        ValueError: For ``scale_search`` with a recipe other than NVFP4 or with ``override``; also, below, for
            non-finite source values or unexpected tensor dtype or shape.
        IgnoreListError: If the ignore list matches nothing.
        SelectionError: If ``select`` is malformed, outside the model, or quantizes nothing.
        FileExistsError: If ``dst`` is not empty.
    """
    recipe = get_recipe(recipe_name)
    if scale_search and recipe.quantizer != "nvfp4":
        raise ValueError(
            f"the MSE-optimal scale search applies to the nvfp4 recipe only, not {recipe.name!r}"
        )
    if scale_search and override is not None:
        raise ValueError(
            "the scale search cannot be combined with an override (a calibrated build keeps plain scales)"
        )
    if not recipe.implemented:
        raise NotImplementedError(
            f"recipe {recipe.name!r} is declared but not implemented; it still needs: "
            + "; ".join(recipe.calibration_needs)
        )
    source = SourceCheckpoint(src)
    header = source.header()
    selection_record: dict[str, Any] | None = None
    if select is not None:
        resolved = ablation.resolve_selection(header, select, ignore_patterns, recipe)
        ignore_patterns = resolved.patterns
        selection_record = resolved.record()
    plan = build_plan(header, recipe, ignore_patterns)
    header_total = sum(m.numel for m in header.values())
    out_by_source = {t.name: sum(o.nbytes for o in plan.outputs_of(t)) for t in plan.tensors}
    layout = plan_shards([(t.name, out_by_source[t.name]) for t in plan.tensors], max_shard_bytes)
    if dry_run:
        return _build_info(
            plan,
            source,
            header_total,
            layout,
            {},
            [],
            [],
            dry_run=True,
            selection=selection_record,
            scale_search=scale_search,
        )
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f"destination {dst} exists and is not empty")
    cfg_path = src / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"{cfg_path} not found")
    dst.mkdir(parents=True, exist_ok=True)

    by_name = {t.name: t for t in plan.tensors}
    n_shards = len(layout)
    weight_map: dict[str, str] = {}
    records: dict[str, dict[str, Any]] = {}
    files: list[dict[str, Any]] = []
    written: list[Path] = []
    try:
        for i, group in enumerate(layout, start=1):
            fname = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
            buffer: dict[str, torch.Tensor] = {}
            for name in group:
                t = by_name[name]
                tensor = source.get_tensor(name)
                if t.category == "quantized":
                    replaced = override(name, tensor) if override is not None else None
                    outs, rec = (
                        replaced
                        if replaced is not None
                        else quantize_tensor(
                            name,
                            tensor,
                            recipe,
                            device=device,
                            expert_chunk=expert_chunk,
                            scale_search=scale_search,
                        )
                    )
                    rec["role"] = t.role
                    records[name] = rec
                    buffer.update(outs)
                else:
                    buffer[name] = tensor.contiguous()
            _write_shard(dst / fname, buffer)
            written.append(dst / fname)
            for k in buffer:
                weight_map[k] = fname
            files.append(
                {
                    "name": fname,
                    "bytes": (dst / fname).stat().st_size,
                    "n_tensors": len(buffer),
                }
            )
            log(f"[{i}/{n_shards}] {fname}: {len(buffer)} tensors, {files[-1]['bytes'] / 1e9:.2f} GB")
            del buffer
        cfg = json.loads(cfg_path.read_text())
        cfg["quantization_config"] = recipe.quantization_config(plan.vllm_ignore)
        (dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        aux = _copy_auxiliary_files(src, dst)
        info = _build_info(
            plan,
            source,
            header_total,
            layout,
            records,
            files,
            aux,
            dry_run=False,
            selection=selection_record,
            scale_search=scale_search,
        )
        if info_extra:
            info.update(info_extra)
        index = {
            "metadata": {"total_size": info["output"]["tensor_bytes"]},
            "weight_map": dict(sorted(weight_map.items())),
        }
        (dst / INFO_FILE).write_text(json.dumps(info, indent=2) + "\n")
        (dst / INDEX_FILE).write_text(json.dumps(index, indent=2) + "\n")
    except BaseException:
        for f in written:
            f.unlink(missing_ok=True)
        for leftover in dst.glob("*.partial"):
            leftover.unlink(missing_ok=True)
        raise
    finally:
        source.close()
    return info


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    implemented = sorted(n for n, r in RECIPES.items() if r.implemented)
    p = argparse.ArgumentParser(
        prog="python -m olala_quant.convert",
        description="Data-free FP8 / NVFP4 conversion of an Olala bf16 checkpoint.",
    )
    p.add_argument("--src", type=Path, required=True, help="bf16 checkpoint directory")
    p.add_argument("--dst", type=Path, required=True, help="output directory (empty or new)")
    p.add_argument("--recipe", required=True, choices=implemented)
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--ignore-file",
        type=Path,
        help="JSON list of fnmatch patterns over HF tensor names kept in bf16 "
        "(default: ignore list I, see olala_quant.recipes.DEFAULT_IGNORE_I)",
    )
    group.add_argument("--no-ignore", action="store_true", help="quantize everything the recipe can")
    p.add_argument(
        "--select",
        metavar="SPEC",
        help="component ablation: quantize only this part and keep the rest in bf16, always with list I "
        "kept in bf16 too (except experts-all, which lifts the MoE half of list I). SPEC is NAME or "
        "NAME:layers=A-B[,N...] with NAME one of "
        f"{', '.join([*ablation.SELECT_NAMES, *ablation.EXTENDED_SELECTIONS])} (see olala_quant.ablation); "
        "recorded in quantization_info.json",
    )
    p.add_argument(
        "--scale-search",
        action="store_true",
        help="nvfp4 only: choose each group's e4m3 block scale by the MSE-optimal search of PROTOCOL.md "
        "amendment 25 (amax scale x 1.00, 0.95, 0.90, 0.85, 0.80, 0.75, smallest squared error); "
        "same layout and global scales; "
        "recorded as scale_search in quantization_info.json",
    )
    p.add_argument("--device", default="cpu", help="torch device for the math (cpu or cuda)")
    p.add_argument("--max-shard-gb", type=float, default=DEFAULT_MAX_SHARD_BYTES / 1e9)
    p.add_argument("--expert-chunk", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="plan and project sizes, write nothing")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point. Returns 0 on success, 2 on a refused or failed run."""
    args = _parse_args(argv)
    if args.select is not None and args.no_ignore:
        print(
            "error: --select always keeps list I in bf16; it cannot be combined with --no-ignore "
            "(use --ignore-file to give another base list)",
            file=sys.stderr,
        )
        return 2
    if args.no_ignore:
        patterns: tuple[str, ...] = ()
    elif args.ignore_file is not None:
        patterns = load_ignore_file(args.ignore_file)
    else:
        patterns = DEFAULT_IGNORE_I
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("error: --device cuda requested but CUDA is not available", file=sys.stderr)
        return 2
    try:
        info = convert(
            args.src,
            args.dst,
            args.recipe,
            patterns,
            device=args.device,
            max_shard_bytes=int(args.max_shard_gb * 1e9),
            expert_chunk=args.expert_chunk,
            dry_run=args.dry_run,
            select=args.select,
            scale_search=args.scale_search,
        )
    except (
        IgnoreListError,
        ablation.SelectionError,
        FileExistsError,
        FileNotFoundError,
        NotImplementedError,
        ValueError,
    ) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = info["output"]
    print(
        f"{'projected' if args.dry_run else 'wrote'}: {out['n_tensors']} tensors, "
        f"{out['tensor_bytes'] / 1e9:.3f} GB, {out['bits_per_weight']:.3f} bits per weight, "
        f"{out['n_shards']} shards"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
