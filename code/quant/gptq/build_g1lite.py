"""G1-lite: calibrated NVFP4 codes for the ROUTED EXPERTS of Olala-v1.0-mini-7A1B, in exactly the N2 checkpoint format.

Usage::

    python build_g1lite.py --src DIR --calib MANIFEST --dst DIR [--device cuda] [--layers 2,3 | all]
        [--hessian blend|routed|all] [--propagate bf16|quantized] [--calibrate-only]

What it does (``GPTQ_DESIGN.md`` section 6, milestone M4; decision D1):

1. reads the audited calibration manifest (``calib_data.py``): refuses a manifest without a passing leakage audit;
2. runs the vendored HF blocks layer by layer over the calibration windows (``hessians.run_sequential``), captures the
   inputs of the expert loop, builds per-expert Hessians (routed second moment plus pseudo-tokens of the all-token
   moment; float32; token-norm clip; held-out windows kept out) and solves every expert's up- and down-projection with
   GPTQ on the static NVFP4 grid (``solve.solve_experts``): block scales and global scales are those of the data-free
   N2 build, only the codes change;
3. falls back to round-to-nearest, per expert and with a reason, for too few tokens, an unusable or spiky Hessian, or
   a held-out error worse than round-to-nearest's;
4. writes the checkpoint through ``olala_quant.convert.convert`` (same tensor names, packing, ``config.json``) with the
   solver's codes for the routed tensors, ``quantization_info.json`` extended with a ``calibration`` block, and
   ``g1lite_report.json`` (per layer and expert: token counts, diagonal ratios, loss ratio against RTN, fallbacks,
   wall time per layer).

Everything else (list I in bf16, dense linears and shared experts) is exactly what the N2 recipe does, unless
``--base-select SPEC`` is given: with ``--base-select experts`` ONLY the routed experts of the layers outside the ignore
list are quantized (calibrated) and every other tensor is copied bf16 and listed in ``quantization_config.ignore``,
exactly as ``python -m olala_quant.convert --select experts`` does (the ``selection`` block of the info is the one
convert writes; the ``calibration`` block records ``base_select``). Any spec of the ablation grammar that quantizes
only routed experts works (``experts``, ``experts:layers=12-23``); a spec that would quantize any other tensor is
refused, because only routed experts are calibrated. Without the option nothing changes.

``--calib`` is a ``manifest.json`` or the directory that holds it (the lite bundle, or the output directory of
``cal1_generate.py``); either way ``calib_data.read_windows`` refuses a manifest without a passing leakage audit. The
report says how many windows (tokens, window equivalents) of each CAL-1 domain were used (``window_domains.py``).

Verify with ``python -m olala_quant.verify --src SRC --dst DST --calibrated [--select SPEC]``.

Resume: every finished layer is saved under ``--work`` (default ``DST.work``); a rerun continues after the last
finished layer. ``--layers`` restricts the calibration to some layers (the pod feasibility session); the other routed
tensors then keep their round-to-nearest codes and the info says ``partial: true``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from olala_quant import ablation, nvfp4
from olala_quant import convert as conv
from olala_quant.checkpoint_io import SourceCheckpoint
from olala_quant.metrics import ErrorAccumulator
from olala_quant.plan import IgnoreListError, Plan, build_plan, load_ignore_file
from olala_quant.recipes import DEFAULT_IGNORE_I, get_recipe

import calib_data
import hessians as hs
import solve
import window_domains
from olala_model import check_vendored_matches, load_olala_modules, sha256_file

REPORT_FILE = "g1lite_report.json"
ROUTED_SUFFIX = {"up": "mlp.experts.experts.weight", "down": "mlp.experts.output_experts.weight"}
SCHEMA = "g1lite-report/1"
ROUTED_ROLES = ("routed_up", "routed_down")


class BaseSelectError(ValueError):
    """A ``--base-select`` that would quantize something other than the routed experts."""


def require_routed_only(plan: Plan, spec: str) -> None:
    """Refuse a plan that quantizes any tensor that is not a routed-expert matrix (only those are calibrated).

    Raises:
        BaseSelectError: Naming the first offending tensors.
    """
    bad = [t.name for t in plan.tensors if t.category == "quantized" and t.role not in ROUTED_ROLES]
    if bad:
        raise BaseSelectError(
            f"--base-select {spec!r} would quantize {len(bad)} tensor(s) that are not routed experts "
            f"(e.g. {bad[:3]}): only routed experts are calibrated, so the base may only quantize them "
            "(use 'experts' or 'experts:layers=...')"
        )


@dataclass(frozen=True)
class BuildSettings:
    """Everything that changes the result (and therefore the resume fingerprint)."""

    hessian: hs.HessianConfig = field(default_factory=hs.HessianConfig)
    solver: solve.SolverConfig = field(default_factory=solve.SolverConfig)
    min_heldout_rows: int = 4
    propagate: str = "bf16"
    max_windows: int | None = None

    def describe(self) -> dict[str, Any]:
        """JSON view recorded in the info and the report."""
        h, s = self.hessian, self.solver
        return {
            "hessian": {
                "mode": h.mode,
                "blend_pseudo": h.blend_pseudo,
                "token_clip": h.token_clip,
                "heldout_every": h.heldout_every,
                "min_tokens": h.effective_min_tokens(),
                "accumulation_dtype": "float32",
            },
            "solver": {
                "algorithm": "GPTQ, static NVFP4 scales (decision D1), static act-order, no g_idx",
                "percdamp": s.percdamp,
                "blocksize": s.blocksize,
                "act_order": s.act_order,
                "diag_ratio_max": s.diag_ratio_max,
                "expert_chunk": s.expert_chunk,
                "dtype": str(s.dtype).removeprefix("torch."),
            },
            "guard": {"heldout_min_rows": self.min_heldout_rows, "fallback": "round-to-nearest, per expert"},
            "propagate": self.propagate,
            "max_windows": self.max_windows,
        }


def parse_layers(spec: str, n_layers: int) -> set[int] | None:
    """``all`` -> None; otherwise a set from ``2,4-6``.

    Raises:
        ValueError: For a malformed spec or a layer outside the model.
    """
    if spec == "all":
        return None
    out: set[int] = set()
    for part in spec.split(","):
        try:
            lo, _, hi = part.partition("-")
            out.update(range(int(lo), int(hi or lo) + 1))
        except ValueError as e:
            raise ValueError(f"bad --layers entry {part!r}") from e
    bad = sorted(i for i in out if not 0 <= i < n_layers)
    if bad:
        raise ValueError(f"layers {bad} are outside 0..{n_layers - 1}")
    return out


@dataclass
class LayerResult:
    """What is kept of one calibrated layer."""

    layer: int
    tensors: dict[str, dict[str, torch.Tensor]]
    report: dict[str, Any]
    rtn_experts: dict[str, list[int]]
    weights: hs.ExpertWeights


def describe_per_domain(per_domain: dict[str, Any]) -> list[str]:
    """Log lines for the windows used per domain (a domain without any token in the used windows is flagged)."""
    if per_domain["method"] == "unavailable":
        return [f"windows per domain: not available ({per_domain['reason']})"]
    lines = [
        f"windows per domain ({per_domain['method']}, {per_domain['n_windows_used']} windows of {per_domain['window']}"
        + (f"; {per_domain['reason']}" if per_domain.get("reason") else "")
        + "):"
    ]
    for name, r in per_domain["domains"].items():
        touched = r["windows_touched"]
        lines.append(
            f"  {name}: {r['window_equivalents']:.1f} window equivalents"
            + (f", {touched} windows touched" if touched is not None else "")
            + (" (NONE of this domain is in the windows used)" if not r["window_equivalents"] else "")
        )
    return lines


def _num(x: float) -> float | None:
    """JSON-safe float (NaN and infinity become None)."""
    return float(x) if x == x and abs(x) != float("inf") else None


def solve_layer(info: hs.LayerInfo, settings: BuildSettings, device: torch.device | str) -> LayerResult:
    """Hessians, GPTQ, held-out guard and bookkeeping for the experts of one captured layer."""
    assert info.capture is not None
    t0 = time.perf_counter()
    experts = info.block.mlp.experts
    w_up = experts.experts.weight.detach()
    w_down = experts.output_experts.weight.detach()
    n_exp = w_up.shape[0]
    eh = hs.build_hessians(info.capture, w_up, w_down, settings.hessian, device)
    t1 = time.perf_counter()
    st_up, st_dn = solve.static_nvfp4(w_up), solve.static_nvfp4(w_down)
    res_up = solve.solve_experts(w_up, eh.h_up, st_up, settings.solver, active=eh.active)
    res_dn = solve.solve_experts(w_down, eh.h_down, st_dn, settings.solver, active=eh.active)
    t2 = time.perf_counter()
    rtn_up, rtn_dn = solve.rtn_codes(st_up), solve.rtn_codes(st_dn)
    deq_rtn_up, deq_rtn_dn = solve.dequantize_codes(rtn_up, st_up), solve.dequantize_codes(rtn_dn, st_dn)
    deq_g_up, deq_g_dn = solve.dequantize_codes(res_up.codes, st_up), solve.dequantize_codes(res_dn.codes, st_dn)
    guard = solve.heldout_guard(
        w_up, w_down, deq_g_up, deq_g_dn, deq_rtn_up, deq_rtn_dn, eh.held_x, min_rows=settings.min_heldout_rows
    )
    both_fallback = (res_up.fallback & res_dn.fallback).cpu()
    use_gptq = (~both_fallback) & guard.keep.cpu()
    sel = use_gptq.to(w_up.device)[:, None, None]
    codes_up = torch.where(sel, res_up.codes, rtn_up)
    codes_dn = torch.where(sel, res_dn.codes, rtn_dn)
    deq_up, deq_dn = solve.dequantize_codes(codes_up, st_up), solve.dequantize_codes(codes_dn, st_dn)
    werr = {
        "up": (solve.weighted_error(w_up, deq_up, eh.h_up), solve.weighted_error(w_up, deq_rtn_up, eh.h_up)),
        "down": (solve.weighted_error(w_down, deq_dn, eh.h_down), solve.weighted_error(w_down, deq_rtn_dn, eh.h_down)),
    }
    t3 = time.perf_counter()
    experts_report, reasons = [], Counter()
    rtn_experts: dict[str, list[int]] = {"up": [], "down": []}
    ratios = {"up": [], "down": []}
    ho_ratio = []
    for e in range(n_exp):
        gptq = bool(use_gptq[e])
        if gptq:
            reason = ""
        elif bool(both_fallback[e]):
            reason = res_up.reason[e] or res_dn.reason[e]
        else:
            reason = guard.reason[e]
        reasons[reason or "gptq"] += 1
        fb_up = (not gptq) or bool(res_up.fallback[e])
        fb_dn = (not gptq) or bool(res_dn.fallback[e])
        if (not gptq) or bool(res_up.fallback[e]):
            rtn_experts["up"].append(e)
        if (not gptq) or bool(res_dn.fallback[e]):
            rtn_experts["down"].append(e)
        lr = {}
        for k in ("up", "down"):
            g, r = float(werr[k][0][e]), float(werr[k][1][e])
            lr[k] = _num(g / r) if r > 0 else None
            if gptq and lr[k] is not None:
                ratios[k].append(lr[k])
        eg, er = guard.err_gptq[e], guard.err_rtn[e]
        if gptq and er > 0:
            ho_ratio.append(eg / er)
        experts_report.append(
            {
                "expert": e,
                "n_tokens": int(eh.n_tokens[e]),
                "n_heldout": int(eh.n_heldout[e]),
                "diag_ratio_up": _num(float(res_up.diag_ratio[e])),
                "diag_ratio_down": _num(float(res_dn.diag_ratio[e])),
                "clip_frac_up": _num(float(eh.clip_frac_up[e])),
                "clip_frac_down": _num(float(eh.clip_frac_down[e])),
                "loss_ratio_up": lr["up"],
                "loss_ratio_down": lr["down"],
                "heldout_err_gptq": _num(eg),
                "heldout_err_rtn": _num(er),
                "fallback_up": fb_up,
                "fallback_down": fb_dn,
                "final": "gptq" if gptq else "rtn",
                "reason": reason,
            }
        )

    def mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    report = {
        "layer": info.layer,
        "n_tokens_total": int(info.capture.x.shape[0]),
        "wall_seconds": {
            "forward": info.forward_seconds,
            "mixer": info.mixer_seconds,
            "mlp": info.mlp_seconds,
            "hessians": t1 - t0,
            "solve": t2 - t1,
            "guard_and_bookkeeping": t3 - t2,
            "total": info.forward_seconds + (t3 - t0),
        },
        "memory": {
            "peak_cuda_allocated_bytes_forward": info.peak_memory_bytes,
            "peak_cuda_allocated_bytes_after_solve": torch.cuda.max_memory_allocated()
            if str(device).startswith("cuda")
            else None,
            "process_max_rss_bytes": info.rss_bytes,
        },
        "max_abs": {"mixer_out": _num(info.max_abs_mixer_out), "mlp_out": _num(info.max_abs_mlp_out)},
        "summary": {
            "n_gptq": int(use_gptq.sum()),
            "n_rtn": int(n_exp - use_gptq.sum()),
            "reasons": dict(reasons),
            "mean_loss_ratio_up": mean(ratios["up"]),
            "mean_loss_ratio_down": mean(ratios["down"]),
            "mean_heldout_err_ratio": mean(ho_ratio),
            "min_tokens": int(eh.n_tokens.min()),
            "median_tokens": float(eh.n_tokens.float().median()),
            "max_tokens": int(eh.n_tokens.max()),
        },
        "experts": experts_report,
    }
    tensors = {
        "up": {
            "packed": nvfp4.pack_fp4(codes_up).cpu(),
            "block_scale": st_up.block_scale.cpu(),
            "global_scale": st_up.global_scale.cpu(),
        },
        "down": {
            "packed": nvfp4.pack_fp4(codes_dn).cpu(),
            "block_scale": st_dn.block_scale.cpu(),
            "global_scale": st_dn.global_scale.cpu(),
        },
    }
    return LayerResult(info.layer, tensors, report, rtn_experts, hs.ExpertWeights(deq_up, deq_dn))


def _dequantize_saved(entry: dict[str, torch.Tensor]) -> torch.Tensor:
    return nvfp4.dequantize_nvfp4(entry["packed"], entry["block_scale"], entry["global_scale"], torch.bfloat16)


class G1Handler:
    """Layer handler: solves the captured layers, saves each result, and restores finished ones on resume."""

    def __init__(
        self,
        targets: set[int],
        work: Path,
        settings: BuildSettings,
        device: torch.device | str,
        fingerprint: str,
        log: Callable[[str], None],
    ) -> None:
        """Bind the target layers, the work directory and the settings."""
        self.targets, self.work, self.settings, self.device = targets, work, settings, device
        self.fingerprint, self.log = fingerprint, log

    def path(self, layer: int) -> Path:
        """File holding the result of ``layer``."""
        return self.work / f"layer_{layer:02d}.pt"

    def load(self, layer: int) -> dict[str, Any]:
        """Load a saved layer result, refusing one made with other settings."""
        data = torch.load(self.path(layer), weights_only=True)
        if data["fingerprint"] != self.fingerprint:
            raise hs.ResumeError(f"{self.path(layer)} was made with other settings; delete {self.work}")
        return data

    def needs_capture(self, layer: int) -> bool:
        """True for a target layer without a saved result."""
        return layer in self.targets and not self.path(layer).is_file()

    def restored_weights(self, layer: int) -> hs.ExpertWeights | None:
        """Dequantized experts of a finished layer (for quantized propagation on resume)."""
        if layer not in self.targets or not self.path(layer).is_file():
            return None
        t = self.load(layer)["tensors"]
        return hs.ExpertWeights(_dequantize_saved(t["up"]), _dequantize_saved(t["down"]))

    def process(self, info: hs.LayerInfo) -> hs.ExpertWeights | None:
        """Solve a captured layer, save its result, return its dequantized experts."""
        res = solve_layer(info, self.settings, self.device)
        tmp = self.path(info.layer).with_name(self.path(info.layer).name + ".partial")
        torch.save(
            {
                "fingerprint": self.fingerprint,
                "layer": res.layer,
                "tensors": res.tensors,
                "report": res.report,
                "rtn_experts": res.rtn_experts,
            },
            tmp,
        )
        tmp.replace(self.path(info.layer))
        s = res.report["summary"]
        self.log(
            f"layer {info.layer}: {s['n_gptq']} GPTQ / {s['n_rtn']} RTN experts, "
            f"loss ratio up {s['mean_loss_ratio_up']}, {res.report['wall_seconds']['total']:.1f} s"
        )
        return res.weights if self.settings.propagate == "quantized" else None


def write_checkpoint(
    src_dir: Path,
    dst: Path,
    patterns: Sequence[str],
    handler: G1Handler,
    layers_done: list[int],
    calibration_info: dict[str, Any],
    device: str,
    log: Callable[[str], None],
    select: str | None = None,
) -> dict[str, Any]:
    """Write the N2-format checkpoint with the solver's codes for the routed tensors of ``layers_done``.

    With ``select`` (a component-ablation spec) only that part is quantized and every other tensor stays bf16, as in
    ``olala_quant.convert`` (which records the selection in the info).
    """
    cache: dict[int, dict[str, Any]] = {}

    def override(name: str, tensor: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, Any]] | None:
        for kind, suffix in ROUTED_SUFFIX.items():
            if name.endswith("." + suffix):
                layer = int(name.split(".")[2])
                if layer not in layers_done:
                    return None
                if layer not in cache:
                    cache.clear()
                    cache[layer] = handler.load(layer)
                saved = cache[layer]["tensors"][kind]
                ref = nvfp4.quantize_nvfp4(tensor.to(device))
                if not torch.equal(
                    ref.block_scale.cpu().view(torch.uint8), saved["block_scale"].view(torch.uint8)
                ) or not torch.equal(ref.global_scale.cpu(), saved["global_scale"]):
                    raise RuntimeError(f"{name}: the solver's scales differ from the data-free ones (decision D1)")
                acc = ErrorAccumulator()
                for i in range(0, tensor.shape[0], 16):
                    deq = nvfp4.dequantize_nvfp4(
                        saved["packed"][i : i + 16], saved["block_scale"][i : i + 16], saved["global_scale"][i : i + 16]
                    )
                    acc.update(tensor[i : i + 16], deq)
                err = acc.finish()
                base = name.removesuffix(".weight")
                outs = {
                    f"{base}.weight_packed": saved["packed"],
                    f"{base}.weight_scale": saved["block_scale"],
                    f"{base}.weight_global_scale": saved["global_scale"],
                }
                rec = {
                    "scale_stats": ref.scale_stats.as_dict(),
                    "global_scale_min": float(ref.global_scale.min()),
                    "global_scale_max": float(ref.global_scale.max()),
                    "shape": list(tensor.shape),
                    "max_abs_error": err.max_abs,
                    "rel_fro_error": err.rel_fro,
                }
                return outs, rec
        return None

    return conv.convert(
        src_dir,
        dst,
        "nvfp4",
        patterns,
        device=device,
        override=override,
        info_extra={"calibration": calibration_info},
        select=select,
        log=log,
    )


def _forward_stats(path: Path) -> list[dict[str, Any]]:
    """The last record of every layer the runner ran (list-I layers and other forward-only layers included)."""
    if not path.is_file():
        return []
    rows: dict[int, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        rows[row["layer"]] = row
    return [rows[k] for k in sorted(rows)]


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="build_g1lite.py", description=__doc__.split("\n")[0])
    p.add_argument("--src", type=Path, required=True, help="bf16 checkpoint directory (with the remote-code files)")
    p.add_argument(
        "--calib",
        type=Path,
        required=True,
        help="audited calibration manifest.json, or the directory that holds it (calib_data.py, cal1_generate.py)",
    )
    p.add_argument(
        "--base-select",
        default="",
        metavar="SPEC",
        help="quantize ONLY the routed experts selected by SPEC ('experts', 'experts:layers=12-23') and keep every "
        "other tensor bf16 (listed in the ignore list), as olala_quant.convert --select does; default: empty = the "
        "N2 base (non-experts quantized data-free)",
    )
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--layers", default="all", help="'all' or e.g. '2,3,10-12' (others keep RTN codes)")
    p.add_argument("--hessian", choices=["routed", "all", "blend"], default="blend")
    p.add_argument("--blend-pseudo", type=float, default=1.0)
    p.add_argument("--percdamp", type=float, default=0.01)
    p.add_argument("--no-act-order", action="store_true")
    p.add_argument("--token-clip", type=float, default=3.0, help="0 disables the clip")
    p.add_argument("--heldout-every", type=int, default=10, help="every n-th window is held out; 0 disables")
    p.add_argument("--min-tokens", type=int, default=None)
    p.add_argument("--min-heldout-rows", type=int, default=4)
    p.add_argument("--diag-ratio-max", type=float, default=1e6)
    p.add_argument("--expert-chunk", type=int, default=32)
    p.add_argument("--solve-dtype", choices=["float64", "float32"], default="float64")
    p.add_argument("--propagate", choices=["bf16", "quantized"], default="bf16")
    p.add_argument("--batch-windows", type=int, default=2)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--attn-impl", choices=["eager", "flex", "fa2", "fa3"], default=None)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--ignore-file", type=Path)
    group.add_argument("--no-ignore", action="store_true")
    p.add_argument("--work", type=Path, default=None, help="resume directory (default: DST.work)")
    p.add_argument("--restart", action="store_true", help="delete the work directory first")
    p.add_argument("--calibrate-only", action="store_true", help="write the report only, no checkpoint")
    p.add_argument("--allow-modeling-mismatch", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI. Returns 0 on success and 2 when it refuses (unaudited manifest, bad layers, other settings, ...)."""
    args = _parse_args(argv)
    log = lambda s: print(s, flush=True)  # noqa: E731
    try:
        return _run(args, log)
    except (
        calib_data.ManifestError,
        hs.ResumeError,
        IgnoreListError,
        FileExistsError,
        FileNotFoundError,
        ValueError,  # includes ablation.SelectionError and BaseSelectError
    ) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace, log: Callable[[str], None]) -> int:
    t_start = time.perf_counter()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("--device cuda requested but CUDA is not available")
    if args.attn_impl:
        import os

        os.environ["OLALA_ATTN_IMPL"] = args.attn_impl
    args.calib = calib_data.resolve_manifest(args.calib)  # a manifest.json, or the directory that holds it
    windows = calib_data.read_windows(args.calib)  # refuses an unaudited or modified manifest
    if args.max_windows:
        windows = windows[: args.max_windows]
    per_domain = window_domains.windows_per_domain(args.calib, windows)
    base_select = args.base_select.strip()
    match = check_vendored_matches(args.src)
    if not all(match.values()) and not args.allow_modeling_mismatch:
        raise ValueError(
            f"the modeling files next to the weights differ from the vendored OVH copy: {match}; the Hessians would "
            "come from a different function (use --allow-modeling-mismatch only knowingly)"
        )
    if args.dst.exists() and any(args.dst.iterdir()):
        raise FileExistsError(f"destination {args.dst} exists and is not empty")
    modeling, config_mod = load_olala_modules()
    config = config_mod.OlalaConfig(**json.loads((args.src / "config.json").read_text()))
    n_layers = len(config.layers_config)
    requested = parse_layers(args.layers, n_layers)
    patterns: tuple[str, ...] = (
        () if args.no_ignore else load_ignore_file(args.ignore_file) if args.ignore_file else DEFAULT_IGNORE_I
    )
    source = SourceCheckpoint(args.src)
    if base_select:
        if args.no_ignore:
            raise ValueError(
                "--base-select always keeps the base ignore list in bf16; it cannot be combined with --no-ignore "
                "(use --ignore-file to give another base list)"
            )
        resolved = ablation.resolve_selection(source.header(), base_select, patterns, get_recipe("nvfp4"))
        require_routed_only(resolved.plan, base_select)
        plan = resolved.plan
        base_select = resolved.selection.spec  # one spelling, the one convert records
    else:
        plan = build_plan(source.header(), get_recipe("nvfp4"), patterns)
    quantized_layers = sorted(
        {t.layer for t in plan.tensors if t.category == "quantized" and t.role in ("routed_up", "routed_down")}
    )
    why_skipped = (
        "its routed tensors are not quantized under the base selection or are in the ignore list"
        if base_select
        else "all routed tensors of the layer are in the ignore list"
    )
    skipped = {str(i): why_skipped for i in sorted(requested or set()) if i not in quantized_layers}
    targets = set(quantized_layers if requested is None else requested) & set(quantized_layers)
    if not targets:
        raise ValueError(
            "no layer to calibrate: every requested layer is in the ignore list or outside the base selection"
        )
    for i, why in skipped.items():
        log(f"skipping layer {i}: {why}")
    for line in describe_per_domain(per_domain):
        log(line)
    settings = BuildSettings(
        hessian=hs.HessianConfig(
            mode=args.hessian,
            blend_pseudo=args.blend_pseudo,
            token_clip=args.token_clip or None,
            heldout_every=args.heldout_every,
            min_tokens=args.min_tokens,
        ),
        solver=solve.SolverConfig(
            percdamp=args.percdamp,
            act_order=not args.no_act_order,
            diag_ratio_max=args.diag_ratio_max,
            expert_chunk=args.expert_chunk,
            dtype=getattr(torch, args.solve_dtype),
        ),
        min_heldout_rows=args.min_heldout_rows,
        propagate=args.propagate,
        max_windows=args.max_windows,
    )
    manifest_sha = sha256_file(args.calib)
    fingerprint_extra = json.dumps(settings.describe(), sort_keys=True) + manifest_sha
    fingerprint = hashlib.sha256(fingerprint_extra.encode()).hexdigest()
    work = args.work or args.dst.with_name(args.dst.name + ".work")
    if args.restart and work.exists():
        import shutil

        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    handler = G1Handler(targets, work, settings, args.device, fingerprint, log)
    progress = work / "state" / "progress.json"
    if progress.is_file():
        nxt = json.loads(progress.read_text())["next_layer"]
        missing = sorted(t for t in targets if t < nxt and not handler.path(t).is_file())
        if missing:
            raise hs.ResumeError(f"the saved state is past layers {missing} that have no result; use --restart")
    hs.run_sequential(
        source,
        config,
        windows,
        handler,
        device=args.device,
        batch_windows=args.batch_windows,
        modeling=modeling,
        last_layer=max(targets),
        propagate=args.propagate,
        state_dir=work / "state",
        fingerprint_extra=fingerprint_extra,
        log=log,
    )
    layers = sorted(targets)
    results = {i: handler.load(i) for i in layers}
    report = {
        "schema": SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "settings": settings.describe(),
        "model": {"src": str(args.src), "n_layers": n_layers, "device": args.device, "torch": torch.__version__},
        "base_select": base_select or None,
        "calibration": {
            "manifest": str(args.calib),
            "manifest_sha256": manifest_sha,
            "n_windows": int(windows.shape[0]),
            "window": int(windows.shape[1]),
            "per_domain": per_domain,
        },
        "layers": {str(i): results[i]["report"] for i in layers},
        "forward_stats": _forward_stats(work / "state" / "forward_stats.jsonl"),
    }
    summaries = [r["report"]["summary"] for r in results.values()]
    report["totals"] = {
        "n_layers": len(layers),
        "n_experts": sum(s["n_gptq"] + s["n_rtn"] for s in summaries),
        "n_gptq": sum(s["n_gptq"] for s in summaries),
        "n_rtn": sum(s["n_rtn"] for s in summaries),
        "reasons": dict(sum((Counter(s["reasons"]) for s in summaries), Counter())),
        "seconds_per_layer": {str(i): results[i]["report"]["wall_seconds"]["total"] for i in layers},
    }
    args.dst.mkdir(parents=True, exist_ok=True)
    if args.calibrate_only:
        (args.dst / REPORT_FILE).write_text(json.dumps(report, indent=2) + "\n")
        log(f"wrote {args.dst / REPORT_FILE} (calibrate-only)")
        return 0
    report_path = work / REPORT_FILE
    report["totals"]["wall_seconds"] = time.perf_counter() - t_start
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    rtn_experts = {
        f"model.layers.{i}.{ROUTED_SUFFIX[k]}": results[i]["rtn_experts"][k] for i in layers for k in ("up", "down")
    }
    modeling_hash = sha256_file(Path(modeling.__file__))
    calibration_info = {
        "method": "GPTQ with static NVFP4 scales and static act-order (G1-lite)",
        "scope": "routed experts only",
        "decision": "D1: block scales and per-expert global scales equal the data-free N2 ones; only codes change",
        **settings.describe(),
        "calibrated_layers": layers,
        "skipped_layers": skipped,
        "partial": layers != quantized_layers,
        "calibrated_tensors": sorted(rtn_experts),
        "rtn_experts": rtn_experts,
        "manifest": {
            "path": str(args.calib),
            "sha256": manifest_sha,
            "n_tokens": int(windows.numel()),
            "n_windows": int(windows.shape[0]),
            "window": int(windows.shape[1]),
            **({"per_domain": per_domain} if per_domain["method"] != "unavailable" else {}),
        },
        **(
            {"base_select": base_select, "non_expert_tensors": "bf16, listed in quantization_config.ignore"}
            if base_select
            else {}
        ),
        "modeling_olala_sha256": modeling_hash,
        "report": {"file": REPORT_FILE, "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest()},
        "totals": report["totals"],
    }
    # an empty destination directory made above is allowed by convert(); the report is copied in afterwards
    write_checkpoint(
        args.src, args.dst, patterns, handler, layers, calibration_info, args.device, log, select=base_select or None
    )
    (args.dst / REPORT_FILE).write_bytes(report_path.read_bytes())
    log(f"wrote {args.dst}: {len(layers)} layer(s) calibrated in {time.perf_counter() - t_start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
