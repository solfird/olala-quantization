"""Verify a converted checkpoint against its bf16 source. Exit status 0 only if nothing is wrong.

Usage::

    python -m olala_quant.verify --src BF16_DIR --dst CONVERTED_DIR [--report FILE.json] [--calibrated] \\
        [--select SPEC] [--scale-search | --no-scale-search]

The expected output is re-derived from the source header, the recipe and the ignore patterns
recorded in ``quantization_info.json`` (not from the recorded tensor list), then every tensor is
checked:

* dtype and shape contract, no missing or extra tensors, index and shard headers agree;
* tensors that are ignored or never quantized are identical to the source bit for bit;
* quantized tensors dequantize within per-element and relative-Frobenius error bounds, hold no
  NaN or infinity, have scales in range (FP8: positive and no larger than the row needs;
  NVFP4: e4m3 block scales in [0, 448], one global scale per tensor or expert that puts the
  largest block of the tensor exactly on 448);
* ``config.json`` carries the expected ``quantization_config`` and is otherwise the source config;
* auxiliary files are byte-identical copies; ``quantization_info.json`` agrees with what is on disk.

``--calibrated`` is for checkpoints whose NVFP4 codes were chosen by an error-compensating solver (G1, see
``quant/gptq``). The half-step element bound does not hold for them; it is replaced by: block and global
scales bit-equal to the data-free ones, untouched tensors and experts declared round-to-nearest bit-equal to
the data-free codes, and a relative Frobenius error of at most ``Bounds.nvfp4_calibrated_factor`` times the
data-free one. Without the flag, such a checkpoint is refused.

A checkpoint whose ``quantization_info.json`` records ``scale_search`` (MSE-optimal block scales, PROTOCOL.md
amendment 25, :mod:`olala_quant.nvfp4`) is NOT flagged. Its block scales are not the amax-derived ones, so the
plain checks that assume them (the largest block scale of a tensor is exactly 448; every element within half a
step of the plain grid) are replaced by checks that hold for it: the recorded block is the one this tool knows
(factors and version), the global scales are unchanged (``448 * 6 / amax`` of the source), the largest block
scale of a tensor (expert) is at most 448 and at least the smallest factor's candidate for the block that
holds the amax, every stored block scale is one of the six candidates of its group (the amax-derived scale
times a factor, rounded to e4m3), and no group has a larger squared error than the plain scale gives it.
``--scale-search`` makes the record a requirement and ``--no-scale-search`` its absence (so that a build is
checked to be the candidate that was asked for); neither can be combined with ``--calibrated`` (a calibrated
checkpoint keeps the plain scales).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from olala_quant import ablation, fp8, nvfp4
from olala_quant.checkpoint_io import (
    INDEX_NAME,
    SourceCheckpoint,
    TensorMeta,
    header_from_json,
    read_safetensors_header,
)
from olala_quant.convert import INFO_FILE, list_auxiliary_files
from olala_quant.metrics import ErrorAccumulator
from olala_quant.plan import IgnoreListError, Plan, PlannedTensor, build_plan
from olala_quant.recipes import get_recipe

_EXPERT_CHUNK = 16


@dataclass(frozen=True)
class Bounds:
    """Thresholds of the reconstruction checks (guesses, see DECISIONS.md).

    ``nvfp4_calibrated_factor`` (calibrated mode only, a GUESS): a calibrated tensor may have a relative
    Frobenius reconstruction error up to this multiple of the data-free (round-to-nearest) error of the
    same tensor. Error compensation moves codes away from the nearest grid point on purpose, so the
    half-step bound does not hold; the weight-space error of GPTQ-style solutions is typically a small
    multiple of RTN's, and 2.0 leaves room while still catching a broken solver (a random assignment of
    codes has a far larger error than RTN).
    """

    fp8_rel_fro: float = 0.04
    nvfp4_rel_fro: float = 0.25
    nvfp4_calibrated_factor: float = 2.0
    info_rtol: float = 1e-3


@dataclass(frozen=True)
class Violation:
    """One failed check."""

    check: str
    tensor: str | None
    message: str


@dataclass
class VerifyReport:
    """Result of :func:`verify`."""

    violations: list[Violation] = field(default_factory=list)
    n_tensors_checked: int = 0
    stats: dict[str, Any] = field(default_factory=dict)

    def add(self, check: str, tensor: str | None, message: str) -> None:
        """Record a violation."""
        self.violations.append(Violation(check, tensor, message))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "ok": not self.violations,
            "n_tensors_checked": self.n_tensors_checked,
            "stats": self.stats,
            "violations": [vars(v) for v in self.violations],
        }


def _raw(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().reshape(-1).view(torch.uint8)


def _inventory(dst: Path) -> dict[str, tuple[str, TensorMeta]]:
    """Map every tensor found in the output shard headers to ``(file, meta)``."""
    inv: dict[str, tuple[str, TensorMeta]] = {}
    for f in sorted(dst.glob("model-*.safetensors")):
        raw, _ = read_safetensors_header(f)
        for name, meta in header_from_json(raw).items():
            inv[name] = (f.name, meta)
    return inv


@dataclass
class _Calibrated:
    """Calibrated-mode context: which tensors and experts are allowed to differ from the data-free codes."""

    tensors: frozenset[str]
    rtn_experts: dict[str, list[int]]
    rtn_acc: ErrorAccumulator = field(default_factory=ErrorAccumulator)


def verify(
    src: Path,
    dst: Path,
    bounds: Bounds | None = None,
    device: str = "cpu",
    *,
    calibrated: bool = False,
    select: str | None = None,
    scale_search: bool | None = None,
) -> VerifyReport:
    """Check ``dst`` against ``src``; see the module docstring for the list of checks.

    With ``calibrated=True`` the NVFP4 element bound (half an e2m1 step) is replaced by checks that hold for
    error-compensated codes: block and global scales bit-equal to the data-free ones, tensors and experts that
    were not calibrated bit-equal to the data-free codes, and a relative Frobenius error of at most
    ``bounds.nvfp4_calibrated_factor`` times the data-free error of the same tensor. Without it, a checkpoint
    whose ``quantization_info.json`` has a ``calibration`` block is refused (``calibrated_checkpoint``).

    A checkpoint with a ``selection`` block (component ablation, :mod:`olala_quant.ablation`) has it
    re-derived from the source header and its recorded base ignore list; any difference with what is recorded
    is an ``info_selection`` violation. With ``select`` the checkpoint must in addition have been made with
    that selection (compared in canonical spelling).
    """
    bounds = bounds or Bounds()
    report = VerifyReport()
    info_path = dst / INFO_FILE
    if not info_path.is_file():
        report.add("info", None, f"{INFO_FILE} is missing")
        return report
    info = json.loads(info_path.read_text())
    source = SourceCheckpoint(src)
    header = source.header()
    recipe = get_recipe(info["recipe"]["name"])
    if "calibration" in info and not calibrated:
        report.add(
            "calibrated_checkpoint",
            None,
            "quantization_info.json has a calibration block: its codes are not round-to-nearest, "
            "verify it with calibrated mode (--calibrated)",
        )
        return report
    try:
        plan = build_plan(header, recipe, info["ignore"]["patterns"])
    except (IgnoreListError, ValueError) as e:
        report.add("plan", None, f"cannot rebuild the plan from the source and the recorded recipe: {e}")
        return report

    inv = _inventory(dst)
    bad_outputs = _check_contract(report, plan, inv)
    _check_index(report, dst, inv)
    _check_config(report, src, dst, plan, info)
    _check_aux(report, src, dst)
    _check_info(report, source, plan, info)
    _check_selection(report, header, plan, info, select)
    searched = _check_scale_search(report, info, scale_search, calibrated)
    if not (dst / INDEX_NAME).is_file():
        report.add("index", None, f"{INDEX_NAME} is missing")
    dst_ck = SourceCheckpoint.from_shard_headers(dst)
    calib_info = info.get("calibration", {}) if calibrated else {}
    calib_names = frozenset(calib_info.get("calibrated_tensors", []))
    if calibrated:
        quantized_routed = {
            t.name for t in plan.tensors if t.category == "quantized" and t.role.startswith("routed")
        }
        for bad in sorted(calib_names - quantized_routed):
            report.add(
                "calibration_info", bad, "listed as calibrated but it is not a quantized routed-expert tensor"
            )

    worst: list[tuple[float, str]] = []
    n_q = n_i = n_k = 0
    scale_stats: dict[str, Any] = (
        {"min_scale": None, "max_scale": 0.0}
        if recipe.quantizer == "fp8"
        else {"n_flushed_blocks": 0, "max_block_scale": 0.0, "min_nonzero_block_scale": None}
    )
    for t in plan.tensors:
        report.n_tensors_checked += 1
        names = [o.name for o in plan.outputs_of(t)]
        if any(n not in inv or n in bad_outputs for n in names):
            continue  # already reported as missing or malformed; values cannot be compared
        if t.category == "quantized":
            n_q += 1
            ctx = (
                _Calibrated(calib_names, {k: list(v) for k, v in calib_info.get("rtn_experts", {}).items()})
                if calibrated and recipe.quantizer == "nvfp4"
                else None
            )
            rel = _check_quantized(
                report, source, dst_ck, plan, t, info, bounds, device, scale_stats, ctx, searched
            )
            if rel is not None:
                worst.append((rel, t.name))
        else:
            n_i += t.category == "ignored"
            n_k += t.category == "kept"
            _check_copied(report, source, dst_ck, t)
    worst.sort(reverse=True)
    report.stats.update(
        {
            "n_quantized": n_q,
            "n_ignored": n_i,
            "n_kept": n_k,
            "worst_rel_fro": [{"tensor": n, "rel_fro": r} for r, n in worst[:5]],
            "scales": scale_stats,
        }
    )
    source.close()
    dst_ck.close()
    return report


def _check_contract(report: VerifyReport, plan: Plan, inv: dict[str, tuple[str, TensorMeta]]) -> set[str]:
    """Check names, dtypes and shapes; return the output names that violate the contract."""
    bad: set[str] = set()
    expected = {o.name: o for o in plan.output_tensors()}
    for name in sorted(set(expected) - set(inv)):
        report.add("missing_tensor", name, "expected in the output but absent")
    for name in sorted(set(inv) - set(expected)):
        report.add("unexpected_tensor", name, "present in the output but not in the plan")
    for name, o in expected.items():
        if name not in inv:
            continue
        meta = inv[name][1]
        if meta.dtype != o.dtype:
            report.add("dtype", name, f"dtype {meta.dtype}, expected {o.dtype}")
            bad.add(name)
        if meta.shape != o.shape:
            report.add("shape", name, f"shape {meta.shape}, expected {o.shape}")
            bad.add(name)
    return bad


def _check_index(report: VerifyReport, dst: Path, inv: dict[str, tuple[str, TensorMeta]]) -> None:
    path = dst / INDEX_NAME
    if not path.is_file():
        return
    index = json.loads(path.read_text())
    wmap = index["weight_map"]
    actual = {n: f for n, (f, _) in inv.items()}
    if wmap != actual:
        diff = sorted(set(wmap.items()) ^ set(actual.items()))[:5]
        report.add("index", None, f"weight_map differs from the shard headers, e.g. {diff}")
    total = sum(m.nbytes for _, m in inv.values())
    if index.get("metadata", {}).get("total_size") != total:
        report.add("index", None, f"metadata.total_size is not {total}")


def _check_config(report: VerifyReport, src: Path, dst: Path, plan: Plan, info: dict[str, Any]) -> None:
    path = dst / "config.json"
    if not path.is_file():
        report.add("config", None, "config.json is missing")
        return
    cfg = json.loads(path.read_text())
    qc = cfg.pop("quantization_config", None)
    expected = plan.recipe.quantization_config(plan.vllm_ignore)
    if qc != expected:
        report.add(
            "config",
            None,
            "quantization_config differs from the one derived from recipe and ignore list",
        )
    if qc != info.get("quantization_config"):
        report.add("config", None, "quantization_config differs from quantization_info.json")
    if cfg != json.loads((src / "config.json").read_text()):
        report.add("config", None, "config.json differs from the source config outside quantization_config")


def _check_aux(report: VerifyReport, src: Path, dst: Path) -> None:
    for f in list_auxiliary_files(src):
        copy = dst / f.name
        if not copy.is_file():
            report.add("auxiliary_file", f.name, "not copied")
        elif copy.read_bytes() != f.read_bytes():
            report.add("auxiliary_file", f.name, "differs from the source file")


def _check_info(report: VerifyReport, source: SourceCheckpoint, plan: Plan, info: dict[str, Any]) -> None:
    kind, digest = source.identity_hash()
    rec = info.get("source", {})
    if rec.get("identity_kind") != kind or rec.get("identity_sha256") != digest:
        report.add("info_source_hash", None, "recorded source hash does not match the source on disk")
    if info.get("ignore", {}).get("quantizable_ignored") != list(plan.ignored_quantizable):
        report.add("info_ignore", None, "recorded resolved ignore list differs from the re-derived one")


def _check_selection(
    report: VerifyReport, header: dict[str, TensorMeta], plan: Plan, info: dict[str, Any], select: str | None
) -> None:
    """The recorded selection must be what its spec and base list derive from the source header."""
    recorded = info.get("selection")
    if select is not None:
        try:
            wanted = ablation.parse_select(select).spec
        except ablation.SelectionError as e:
            report.add("info_selection", None, f"the selection required ({select!r}) is not valid: {e}")
            return
        found = recorded.get("spec") if isinstance(recorded, dict) else None
        if found != wanted:
            report.add(
                "info_selection", None, f"checkpoint made with selection {found!r}, required {wanted!r}"
            )
            return
    if recorded is None:
        return
    try:
        resolved = ablation.resolve_selection(
            header, recorded["spec"], recorded["base_ignore_patterns"], plan.recipe
        )
    except (ablation.SelectionError, KeyError, TypeError) as e:
        report.add("info_selection", None, f"the recorded selection cannot be re-derived: {e!r}")
        return
    if resolved.patterns != tuple(info.get("ignore", {}).get("patterns", ())):
        report.add(
            "info_selection",
            None,
            "the recorded ignore patterns are not the base list plus the deselected tensors",
        )
    rederived = resolved.record()
    for key in sorted(set(rederived) | set(recorded)):
        if rederived.get(key) != recorded.get(key):
            report.add(
                "info_selection", None, f"recorded selection differs from the re-derived one in {key!r}"
            )


def _check_scale_search(
    report: VerifyReport, info: dict[str, Any], required: bool | None, calibrated: bool
) -> bool:
    """Check the recorded ``scale_search`` block; True when the tensors must be verified as searched."""
    recorded = info.get("scale_search")
    if required is True and "scale_search" not in info:
        report.add(
            "info_scale_search", None, "the checkpoint was required to record scale_search and does not"
        )
    if required is False and "scale_search" in info:
        report.add(
            "info_scale_search",
            None,
            "the checkpoint was required to be made without scale_search and records it",
        )
    if recorded is None:
        return False
    known = {"factors": list(nvfp4.SCALE_SEARCH_FACTORS), "version": nvfp4.SCALE_SEARCH_VERSION}
    if recorded != known:
        report.add(
            "info_scale_search",
            None,
            f"recorded scale_search {recorded!r} is not the method this tool verifies ({known!r})",
        )
        return False
    if calibrated:
        report.add(
            "info_scale_search",
            None,
            "a checkpoint with searched scales cannot be verified as calibrated (--calibrated)",
        )
    return True


def _check_copied(
    report: VerifyReport, source: SourceCheckpoint, dst: SourceCheckpoint, t: PlannedTensor
) -> None:
    a, b = source.get_tensor(t.name), dst.get_tensor(t.name)
    if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(_raw(a), _raw(b)):
        report.add("bit_identical", t.name, f"{t.category} tensor differs from the source")
        return
    if a.is_floating_point() and not torch.isfinite(a.float()).all():
        report.add("finite", t.name, "NaN or infinity in a tensor that is copied unchanged")


def _check_quantized(
    report: VerifyReport,
    source: SourceCheckpoint,
    dst: SourceCheckpoint,
    plan: Plan,
    t: PlannedTensor,
    info: dict[str, Any],
    bounds: Bounds,
    device: str,
    scale_stats: dict[str, Any],
    calibrated: _Calibrated | None = None,
    searched: bool = False,
) -> float | None:
    orig = source.get_tensor(t.name)
    base = t.name.removesuffix(".weight")
    outs = {s: dst.get_tensor(f"{base}.{s}") for s in plan.recipe.suffixes}
    n_before = len(report.violations)
    acc = ErrorAccumulator()
    if plan.recipe.quantizer == "fp8":
        _verify_fp8(report, t.name, orig, outs, acc, scale_stats)
        limit = bounds.fp8_rel_fro
    else:
        _verify_nvfp4(report, t.name, orig, outs, acc, scale_stats, calibrated, searched)
        limit = bounds.nvfp4_rel_fro
    if len(report.violations) != n_before and any(
        v.check == "finite" and v.tensor == t.name for v in report.violations[n_before:]
    ):
        return None
    err = acc.finish()
    if calibrated is not None and t.name in calibrated.tensors:
        limit = min(limit, bounds.nvfp4_calibrated_factor * calibrated.rtn_acc.finish().rel_fro)
    if not err.rel_fro <= limit:
        report.add(
            "reconstruction_error",
            t.name,
            f"relative Frobenius error {err.rel_fro:.4g} exceeds {limit}",
        )
    rec = info.get("tensors", {}).get(t.name)
    if rec is not None:
        for key, value in (("rel_fro_error", err.rel_fro), ("max_abs_error", err.max_abs)):
            if abs(rec[key] - value) > bounds.info_rtol * max(abs(value), 1e-12):
                report.add(
                    "info_error_mismatch",
                    t.name,
                    f"{key}: recorded {rec[key]:.6g}, recomputed {value:.6g}",
                )
    return err.rel_fro


def _chunks(n: int, step: int) -> list[slice]:
    return [slice(i, i + step) for i in range(0, n, step)]


def _verify_fp8(
    report: VerifyReport,
    name: str,
    orig: torch.Tensor,
    outs: dict[str, torch.Tensor],
    acc: ErrorAccumulator,
    stats: dict[str, Any],
) -> None:
    w, s = outs["weight"], outs["weight_scale"]
    if not torch.isfinite(w.to(torch.float32)).all():
        report.add("finite", name, "NaN or infinity in the fp8 weight")
    if not torch.all(s > 0):
        report.add("scale_range", name, "a weight scale is not positive")
    if not torch.isfinite(s).all():
        report.add("finite", name, "NaN or infinity in the weight scale")
        return
    if not torch.isfinite(w.to(torch.float32)).all():
        return
    step = _EXPERT_CHUNK if orig.ndim == 3 else max(1, orig.shape[0])
    for sl in _chunks(orig.shape[0], step):
        x = orig[sl].to(torch.float32)
        scale = s[sl]
        deq = fp8.dequantize_fp8(w[sl], scale)
        amax = x.abs().amax(dim=-1, keepdim=True)
        need = (amax / fp8.FP8_MAX).clamp_min(fp8.MIN_SCALE)
        if not torch.all(scale >= fp8.MIN_SCALE * (1 - 1e-6)):
            report.add("scale_range", name, "a weight scale is below 1/(448*512) or not positive")
        if not torch.all(scale <= need * (1 + 1e-5) + 1e-30):
            report.add("scale_range", name, "a weight scale is larger than the row needs (amax/448)")
        row_err = (deq - x).abs().amax(dim=-1, keepdim=True)
        if not torch.all(row_err <= 16.0 * scale * 1.001 + 1e-30):
            report.add(
                "reconstruction_error",
                name,
                "an element error exceeds half the e4m3 step at the row max",
            )
        acc.update(x, deq)
    lo, hi = float(s.min()), float(s.max())
    stats["min_scale"] = lo if stats["min_scale"] is None else min(stats["min_scale"], lo)
    stats["max_scale"] = max(stats["max_scale"], hi)


def _verify_nvfp4(
    report: VerifyReport,
    name: str,
    orig: torch.Tensor,
    outs: dict[str, torch.Tensor],
    acc: ErrorAccumulator,
    stats: dict[str, Any],
    calibrated: _Calibrated | None = None,
    searched: bool = False,
) -> None:
    packed, bs, gs = outs["weight_packed"], outs["weight_scale"], outs["weight_global_scale"]
    bsf = bs.to(torch.float32)
    bad = False
    if not torch.isfinite(bsf).all():
        report.add("finite", name, "NaN or infinity in the e4m3 block scales")
        bad = True
    if not torch.isfinite(gs).all():
        report.add("finite", name, "NaN or infinity in the global scale")
        bad = True
    if bad:
        return
    if not torch.all(bsf >= 0) or not torch.all(bsf <= nvfp4.E4M3_MAX):
        report.add("scale_range", name, "an e4m3 block scale is outside [0, 448]")
    if not torch.all(gs > 0):
        report.add("scale_range", name, "a global scale is not positive")
        return
    is3d = orig.ndim == 3
    n_exp = orig.shape[0] if is3d else 1
    if gs.shape != (n_exp,):
        return
    o3 = orig if is3d else orig.unsqueeze(0)
    p3 = packed if is3d else packed.unsqueeze(0)
    b3 = bs if is3d else bs.unsqueeze(0)
    for sl in _chunks(n_exp, _EXPERT_CHUNK):
        x = o3[sl].to(torch.float32)
        g = gs[sl]
        amax = x.abs().amax(dim=(1, 2))
        expected_g = torch.where(
            amax > 0, nvfp4.E4M3_MAX * nvfp4.FP4_MAX / amax.clamp_min(1e-38), torch.ones_like(amax)
        )
        if not torch.allclose(g, expected_g, rtol=1e-5):
            report.add("scale_range", name, "a global scale is not 448*6/amax of its tensor (expert)")
        top = b3[sl].to(torch.float32).amax(dim=(1, 2))
        if searched:
            # searched scales: the block that holds the amax chose a factor, so its scale is 448 or a little
            # below
            floor = nvfp4.E4M3_MAX * min(nvfp4.SCALE_SEARCH_FACTORS) * (1.0 - 2.0**-4)
            if torch.any((amax > 0) & ((top > nvfp4.E4M3_MAX) | (top < floor))):
                report.add(
                    "scale_range",
                    name,
                    "the largest block scale of a tensor (expert) is outside [0.75 x 448, 448]",
                )
        elif torch.any((amax > 0) & (top != nvfp4.E4M3_MAX)):
            report.add("scale_range", name, "the largest block scale of a tensor (expert) is not 448")
        deq = nvfp4.dequantize_nvfp4(p3[sl], b3[sl], g)
        if searched:
            _check_searched_scales(report, name, x, b3[sl], deq, g)
        elif calibrated is not None:
            _check_against_data_free(report, name, o3[sl], p3[sl], b3[sl], g, sl.start, calibrated)
        else:
            err = (deq - x).abs().reshape(*x.shape[:-1], -1, nvfp4.GROUP_SIZE).amax(dim=-1)
            s_eff = b3[sl].to(torch.float32) * (1.0 / g).reshape(-1, 1, 1)
            flush_bound = 6.0 * 2.0**-10 / g.reshape(-1, 1, 1)
            bound = torch.where(s_eff > 0, s_eff * 1.001, flush_bound * 1.001) + 1e-30
            if torch.any(err > bound):
                report.add("reconstruction_error", name, "a block error exceeds the e2m1 half-step bound")
        flushed = (b3[sl].to(torch.float32) == 0) & (
            x.abs().reshape(*x.shape[:-1], -1, nvfp4.GROUP_SIZE).amax(dim=-1) > 0
        )
        stats["n_flushed_blocks"] += int(flushed.sum())
        acc.update(x, deq)
    stats["max_block_scale"] = max(stats["max_block_scale"], float(bsf.max()))
    nz = bsf[bsf > 0]
    if nz.numel():
        lo = float(nz.min())
        prev = stats["min_nonzero_block_scale"]
        stats["min_nonzero_block_scale"] = lo if prev is None else min(prev, lo)


def _check_searched_scales(
    report: VerifyReport,
    name: str,
    x: torch.Tensor,
    block: torch.Tensor,
    deq: torch.Tensor,
    gscale: torch.Tensor,
) -> None:
    """Checks of a chunk of experts whose block scales come from the scale search (see the module docstring).

    ``x`` is the float32 source, ``block`` the stored e4m3 scales, ``deq`` the dequantized checkpoint,
    ``gscale`` the stored global scales. Every stored scale must be one of the candidates ``e4m3(amax-derived
    scale * factor)`` of its group, and no group may have a larger squared error than the plain scale gives
    it.
    """
    e, rows, cols = x.shape
    blocks = x.reshape(e, rows, cols // nvfp4.GROUP_SIZE, nvfp4.GROUP_SIZE)
    base = gscale.reshape(e, 1, 1, 1) * (blocks.abs().amax(dim=-1) * (1.0 / nvfp4.FP4_MAX)).unsqueeze(-1)
    stored = block.view(torch.uint8).unsqueeze(-1)
    is_candidate = torch.zeros_like(stored, dtype=torch.bool)
    for factor in nvfp4.SCALE_SEARCH_FACTORS:
        cand = (base * factor).clamp(min=0.0, max=nvfp4.E4M3_MAX).to(torch.float8_e4m3fn)
        is_candidate |= cand.view(torch.uint8) == stored
    if not bool(is_candidate.all()):
        report.add("scale_search_scale", name, "a block scale is none of the six candidates of its group")
    plain = nvfp4.quantize_nvfp4(x)
    deq_plain = nvfp4.dequantize_nvfp4(plain.packed, plain.block_scale, plain.global_scale)
    err = ((deq - x) ** 2).reshape(e, rows, -1, nvfp4.GROUP_SIZE).sum(dim=-1)
    err_plain = ((deq_plain - x) ** 2).reshape(e, rows, -1, nvfp4.GROUP_SIZE).sum(dim=-1)
    if not bool((err <= err_plain * (1.0 + 1e-4)).all()):
        report.add(
            "scale_search_error", name, "a group has a larger squared error than the plain scale gives it"
        )


def _check_against_data_free(
    report: VerifyReport,
    name: str,
    orig: torch.Tensor,
    packed: torch.Tensor,
    block: torch.Tensor,
    gscale: torch.Tensor,
    first_expert: int,
    ctx: _Calibrated,
) -> None:
    """Calibrated-mode checks for a chunk of experts: scales and untouched codes equal the data-free ones.

    ``orig`` is the chunk of the source (any float dtype, as the converter saw it). The data-free quantization
    of the same chunk is recomputed; its block and global scales must match the checkpoint bit for bit; the
    packed codes must match for every tensor that was not calibrated and for every expert declared
    round-to-nearest; its reconstruction error is accumulated so that the caller can bound the calibrated
    error by it.
    """
    ref = nvfp4.quantize_nvfp4(orig)
    if not torch.equal(ref.block_scale.view(torch.uint8), block.view(torch.uint8)):
        report.add("scale_equals_n2", name, "an e4m3 block scale differs from the data-free one")
    if not torch.equal(ref.global_scale, gscale):
        report.add("scale_equals_n2", name, "a global scale differs from the data-free one")
    if name not in ctx.tensors:
        if not torch.equal(ref.packed, packed):
            report.add(
                "codes_equal_n2", name, "a tensor that was not calibrated differs from the data-free codes"
            )
    else:
        for e in ctx.rtn_experts.get(name, []):
            local = e - first_expert
            if 0 <= local < packed.shape[0] and not torch.equal(ref.packed[local], packed[local]):
                report.add(
                    "rtn_expert_codes", name, f"expert {e} is declared round-to-nearest but its codes differ"
                )
    ctx.rtn_acc.update(
        orig.to(torch.float32), nvfp4.dequantize_nvfp4(ref.packed, ref.block_scale, ref.global_scale)
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m olala_quant.verify",
        description="Verify a converted Olala checkpoint against its bf16 source.",
    )
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--report", type=Path, help="write the full report as JSON here")
    p.add_argument("--max-rel-fro-fp8", type=float, default=Bounds.fp8_rel_fro)
    p.add_argument("--max-rel-fro-nvfp4", type=float, default=Bounds.nvfp4_rel_fro)
    p.add_argument(
        "--calibrated",
        action="store_true",
        help="verify a calibrated (error-compensated) NVFP4 checkpoint: scales equal the data-free ones, "
        "uncalibrated codes equal the data-free codes, calibrated error bounded by --calibrated-factor x RTN",
    )
    p.add_argument("--calibrated-factor", type=float, default=Bounds.nvfp4_calibrated_factor)
    p.add_argument(
        "--select",
        metavar="SPEC",
        help="require the checkpoint to have been made with this component-ablation selection",
    )
    search = p.add_mutually_exclusive_group()
    search.add_argument(
        "--scale-search",
        dest="scale_search",
        action="store_const",
        const=True,
        default=None,
        help="require the checkpoint to record scale_search (MSE-optimal block scales)",
    )
    search.add_argument(
        "--no-scale-search",
        dest="scale_search",
        action="store_const",
        const=False,
        help="require the checkpoint to have been made without the scale search",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: 0 clean, 1 violations found, 2 unusable inputs."""
    args = _parse_args(argv)
    for d in (args.src, args.dst):
        if not d.is_dir():
            print(f"error: {d} is not a directory", file=sys.stderr)
            return 2
    try:
        report = verify(
            args.src,
            args.dst,
            Bounds(
                fp8_rel_fro=args.max_rel_fro_fp8,
                nvfp4_rel_fro=args.max_rel_fro_nvfp4,
                nvfp4_calibrated_factor=args.calibrated_factor,
            ),
            calibrated=args.calibrated,
            select=args.select,
            scale_search=args.scale_search,
        )
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print(f"error: cannot read the checkpoints: {e!r}", file=sys.stderr)
        return 2
    if args.report:
        args.report.write_text(json.dumps(report.as_dict(), indent=2) + "\n")
    if report.violations:
        print(f"FAILED: {len(report.violations)} violation(s)", file=sys.stderr)
        for v in report.violations[:50]:
            print(f"  [{v.check}] {v.tensor or '-'}: {v.message}", file=sys.stderr)
        return 1
    print(f"OK: {report.n_tensors_checked} tensors checked, no violation. {json.dumps(report.stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
