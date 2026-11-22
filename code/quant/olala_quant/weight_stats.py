"""Weight-level quantizability statistics of a checkpoint.

Reads a bf16 safetensors checkpoint tensor by tensor (never the whole model at once, shard by shard, one
tensor resident at a time) and, for every weight matrix, records

* the distribution shape: RMS, max-abs, max-abs / RMS, excess kurtosis, fraction of ``|w| > 6 RMS``, the
  dynamic range of the per-output-channel RMS, and the spread of the per-group amax for groups of 16 and
  32
  along the input dimension;
* the round-to-nearest relative Frobenius error ``||W - Q(W)||_F / ||W||_F`` and the fractions of weights
  whose block scale underflows or saturates, for four weight formats.

Fused 3D routed-expert tensors ``[E, out, in]`` are measured per expert and reduced to quantiles
(min, p05, p50, p95, max) so that the output stays small; the tensor-level numbers pool all experts.

Definitions (every one is a choice; they are repeated in the JSON header ``format_definitions``)
-------------------------------------------------------------------------------------------------
Classes
    Derived from the roles of :mod:`olala_quant.recipes` (the names ``plan.py`` uses). ``mixer_proj`` is
    ``attn_o`` in the attention layers (those that own ``mixer.c_q.weight``) and ``mamba_out_proj``
    elsewhere. Tensors that are not weight matrices (norms, biases, scalars, F32 Mamba parameters,
    ``expert_bias``) are listed under ``skipped`` with a reason and not measured.

Statistics (population moments, float64 accumulation, two sweeps so that central moments are exact)
    * ``rms = sqrt(mean(w^2))``; ``max_over_rms = max|w| / rms``.
    * ``excess_kurtosis = m4 / m2^2 - 3`` with central moments about the mean (Gaussian: 0, two-point
      +-1: -2).
    * ``frac_gt_6rms = mean(|w| > 6 rms)`` with the RMS of the unit measured (the tensor for the
      tensor-level number, the expert for the per-expert numbers).
    * ``channel_rms_range = max_r rms_r / min_r rms_r`` over output rows (all rows of all experts for a
      fused tensor); ``None`` when the smallest row RMS is zero.
    * ``g{16,32}_amax_max_over_median``: the amax of every group of 16 (32) consecutive input elements;
      the ratio of the largest group amax to the median group amax (true median: mean of the two middle
      values). This is the spread the block scales have to span. ``None`` when the width is not a multiple
      of the group or when the median is zero.

Formats (round to nearest, data free, on the bf16 values)
    * ``nvfp4``: :func:`olala_quant.nvfp4.quantize_nvfp4` itself (e2m1, group 16, e4m3 block scale, fp32
      global scale ``448*6/amax`` per tensor, per expert for 3D). Underflow = non-zero block whose e4m3
      scale is zero or subnormal (below 2^-6); flushed = scale exactly zero (the block is lost); saturate
      = scale above 448 before the clamp (the block clips). Zero by construction for the saturation, since
      the global scale comes from the exact amax.
    * ``fp8``: :func:`olala_quant.fp8.quantize_fp8` itself (e4m3, one fp32 scale per output row, per
      expert row for 3D). Underflow = non-zero row whose scale is clamped to the floor ``1 / (448 * 512)``
      (the amax is below 1/512); flushed = 0; saturate = 0 (the scale is fp32 and exact).
    * ``int4_g128``: symmetric integer, group 128 along the input dim, fp16 scale. The convention is that
      of compressed-tensors ``pack-quantized`` / AutoGPTQ symmetric: ``scale = amax / 7.5`` (``(qmax -
      qmin) / 2``), ``q = clamp(round(w / scale), -8, 7)`` (ties to even), ``w ~ q * scale`` with the
      scale stored in fp16. The extreme +amax therefore rounds to 8 and clamps to 7 (an error of half a
      step). Underflow = non-zero group whose fp16 scale is below the fp16 minimum normal 2^-14 (zero or
      subnormal); flushed = scale zero (the group is lost); saturate = ``amax / 7.5 > 65504``.
    * ``mxfp4``: OCP Microscaling v1.0 MXFP4: e2m1 elements, group 32, e8m0 (power of two) scale. Rounding
      rule (the OCP reference, "floor"): ``shared_exp = floor(log2(amax)) - emax_elem`` with ``emax_elem =
      2`` for e2m1, ``scale = 2^shared_exp``, elements ``clamp(w / scale, -6, 6)`` rounded to the nearest
      e2m1 value with the thresholds of :func:`olala_quant.nvfp4.encode_fp4` (ties to even code). Because
      the exponent is floored, a group whose amax / scale lands in (6, 8) clips its largest elements; no
      "ceil" correction is applied. ``floor(log2)`` comes from ``frexp`` (exact, no log rounding). The
      exponent clamps to [-127, 127]; underflow = exponent below -127 (clamped), saturate = above 127
      (impossible for finite float32 input), flushed = 0 (e8m0 has no zero).

    A format is not applicable (``None``) to a tensor whose input width is not a multiple of its group.

Summary (markdown)
    Per class and format: parameter-weighted mean of the per-tensor ``rel_fro`` (weights: element counts),
    the
    worst tensor (its layer), the ratio of that mean to the best format of the class, and the parameter-
    weighted
    fractions of weights whose scale underflows or saturates.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from olala_quant import fp8, nvfp4
from olala_quant.checkpoint_io import SourceCheckpoint, read_safetensors_header
from olala_quant.recipes import classify_tensor

FORMATS = ("nvfp4", "fp8", "int4_g128", "mxfp4")
FORMAT_LABELS = {
    "nvfp4": "NVFP4 (g16, e4m3 scale, fp32 global)",
    "fp8": "FP8 e4m3 per-output-channel",
    "int4_g128": "INT4 sym g128 (fp16 scale)",
    "mxfp4": "MXFP4 (g32, e8m0 scale)",
}
FORMAT_DEFINITIONS = {
    "nvfp4": "olala_quant.nvfp4.quantize_nvfp4: e2m1, group 16, e4m3 block scale, fp32 global scale "
    "448*6/amax (per tensor, per expert for 3D); underflow = nonzero block with scale zero or below 2^-6, "
    "flushed = scale zero, saturate = scale above 448 before the clamp",
    "fp8": "olala_quant.fp8.quantize_fp8: e4m3, one fp32 scale per output row (per expert row for 3D), scale "
    "floor 1/(448*512); underflow = nonzero row clamped to the floor",
    "int4_g128": "symmetric int4, group 128, scale = amax/7.5 stored fp16, q = clamp(round(w/scale), -8, 7), "
    "ties to even; underflow = fp16 scale below 2^-14, flushed = zero, saturate = amax/7.5 > 65504",
    "mxfp4": "OCP MX v1.0: e2m1, group 32, e8m0 scale 2^(floor(log2 amax) - 2) clamped to [-127, 127], "
    "elements clamped to +-6 then rounded to nearest e2m1 (ties to even code); underflow = exponent < -127",
}
INT4_GROUP = 128
INT4_QMIN = -8
INT4_QMAX = 7
INT4_SCALE_DIVISOR = 7.5
FP16_MAX = 65504.0
FP16_MIN_NORMAL = 2.0**-14
MXFP4_GROUP = 32
E2M1_EMAX = 2
E8M0_MIN_EXP = -127
E8M0_MAX_EXP = 127
QUANTILES = (("min", 0.0), ("p05", 0.05), ("p50", 0.5), ("p95", 0.95), ("max", 1.0))
PER_EXPERT_METRICS = (
    "rms",
    "max_over_rms",
    "excess_kurtosis",
    "frac_gt_6rms",
    "channel_rms_range",
    "g16_amax_max_over_median",
    "g32_amax_max_over_median",
)
CLASS_ORDER = (
    "routed_up",
    "routed_down",
    "shared_expert",
    "latent_proj",
    "attn_q",
    "attn_kv",
    "attn_small",
    "attn_gate",
    "attn_o",
    "mamba_in_proj",
    "mamba_other",
    "mamba_out_proj",
    "router",
    "embedding",
    "lm_head",
)
_ROLE_CLASS = {
    "routed_up": "routed_up",
    "routed_down": "routed_down",
    "shared_up": "shared_expert",
    "shared_down": "shared_expert",
    "latent_reduce": "latent_proj",
    "latent_expand": "latent_proj",
    "attn_gate_proj": "attn_gate",
    "attn_c_q": "attn_q",
    "attn_w_b": "attn_kv",
    "attn_small": "attn_small",
    "mamba_in_proj": "mamba_in_proj",
    "mamba_in_proj_dyn": "mamba_other",
    "router": "router",
    "embedding": "embedding",
    "lm_head": "lm_head",
}
_FLOAT_DTYPES = {"BF16", "F16", "F32"}
_LAYER_RX = re.compile(r"^model\.layers\.(\d+)\.")
_ATTN_RX = re.compile(r"^model\.layers\.(\d+)\.mixer\.c_q\.weight$")
# Elements per float32/float64 scratch block of the statistics sweeps (about 32 MB in float64).
_STAT_CHUNK = 4 * 1024 * 1024


# --------------------------------------------------------------------------- classification


@dataclass(frozen=True)
class WeightClass:
    """Class of a measured weight matrix and its layer (None for embedding and lm_head)."""

    cls: str
    layer: int | None


def attention_layers(names: Iterable[str]) -> set[int]:
    """Layers that own an attention mixer, i.e. a ``mixer.c_q.weight`` (the others are Mamba layers)."""
    return {int(m.group(1)) for n in names if (m := _ATTN_RX.match(n))}


def classify_weight(name: str, attention: set[int]) -> WeightClass | None:
    """Class and layer of a weight matrix, or None when the tensor is not one that gets measured."""
    role, _spec, layer = classify_tensor(name)
    if role == "mixer_proj":
        return WeightClass("attn_o" if layer in attention else "mamba_out_proj", layer)
    cls = _ROLE_CLASS.get(role)
    if cls is None:
        return None
    if layer is None and (m := _LAYER_RX.match(name)):
        layer = int(m.group(1))
    return WeightClass(cls, layer)


def skip_reason(name: str) -> str:
    """Why a tensor is not measured."""
    role, _spec, _layer = classify_tensor(name)
    return f"not a weight matrix (role: {role})"


# --------------------------------------------------------------------------- distribution statistics


def _row_blocks(w: torch.Tensor) -> Iterator[torch.Tensor]:
    """Yield row blocks of a 2D tensor holding about ``_STAT_CHUNK`` elements each."""
    rows, cols = w.shape
    step = max(1, _STAT_CHUNK // max(1, cols))
    for i in range(0, rows, step):
        yield w[i : i + step]


def _max_over_median(parts: Sequence[torch.Tensor]) -> float | None:
    """Largest value over the true median of a set of group maxima (None if undefined)."""
    if not parts:
        return None
    v = torch.sort(torch.cat([p.flatten() for p in parts])).values
    n = v.numel()
    median = 0.5 * (float(v[(n - 1) // 2]) + float(v[n // 2]))
    if median <= 0.0:
        return None
    return float(v[-1]) / median


@dataclass
class _Sweep1:
    """First sweep over one unit: raw sums, row sums of squares and group maxima."""

    n: int
    cols: int
    s1: float
    s2: float
    max_abs: float
    row_ss: torch.Tensor
    group_amax: dict[int, list[torch.Tensor]]


def _sweep1(w: torch.Tensor) -> _Sweep1:
    rows, cols = w.shape
    if rows * cols == 0:
        raise ValueError("empty tensor")
    s1 = s2 = max_abs = 0.0
    row_ss: list[torch.Tensor] = []
    group_amax: dict[int, list[torch.Tensor]] = {16: [], 32: []}
    for blk in _row_blocks(w):
        x = blk.to(torch.float32)
        if not bool(torch.isfinite(x).all()):
            raise ValueError("weight_stats: tensor holds non-finite values")
        x64 = x.double()
        sq = x64 * x64
        s1 += float(x64.sum())
        s2 += float(sq.sum())
        max_abs = max(max_abs, float(x.abs().max()))
        row_ss.append(sq.sum(dim=1))
        for g, parts in group_amax.items():
            if cols % g == 0:
                parts.append(x.abs().reshape(x.shape[0], cols // g, g).amax(dim=-1).flatten())
    return _Sweep1(rows * cols, cols, s1, s2, max_abs, torch.cat(row_ss), group_amax)


def _sweep2(w: torch.Tensor, specs: Sequence[tuple[float, float]]) -> list[tuple[float, float, int]]:
    """Central sums ``(sum d^2, sum d^4, count |w| > threshold)`` about each ``(mean, threshold)`` pair."""
    acc = [[0.0, 0.0, 0] for _ in specs]
    for blk in _row_blocks(w):
        x64 = blk.to(torch.float32).double()
        ax = x64.abs()
        for a, (mean, threshold) in zip(acc, specs, strict=True):
            d2 = (x64 - mean) ** 2
            a[0] += float(d2.sum())
            a[1] += float((d2 * d2).sum())
            a[2] += int((ax > threshold).sum())
    return [(a[0], a[1], a[2]) for a in acc]


def _stats_dict(
    n: int,
    cols: int,
    s2: float,
    max_abs: float,
    central: tuple[float, float, int],
    row_ss: torch.Tensor,
    g16: Sequence[torch.Tensor],
    g32: Sequence[torch.Tensor],
) -> dict[str, float | None]:
    m2, m4, count = central
    rms = math.sqrt(s2 / n)
    row_rms = (row_ss / cols).sqrt()
    row_min = float(row_rms.min())
    return {
        "rms": rms,
        "max_abs": max_abs,
        "max_over_rms": max_abs / rms if rms > 0 else None,
        "excess_kurtosis": (m4 / n) / (m2 / n) ** 2 - 3.0 if m2 > 0 else None,
        "frac_gt_6rms": count / n,
        "channel_rms_range": float(row_rms.max()) / row_min if row_min > 0 else None,
        "g16_amax_max_over_median": _max_over_median(g16),
        "g32_amax_max_over_median": _max_over_median(g32),
    }


def compute_distribution(units: Sequence[torch.Tensor]) -> tuple[dict, list[dict]]:
    """Distribution statistics of 2D units (a matrix, or the experts of a fused tensor).

    Returns:
        ``(pooled, per_unit)``. ``pooled`` treats all units as one population (one unit: ``pooled`` equals
        its only entry); ``per_unit`` holds one dictionary per unit. See the module docstring for the keys.

    Raises:
        ValueError: If a unit is empty or holds NaN or infinity.
    """
    sweeps = [_sweep1(u) for u in units]
    n_all = sum(s.n for s in sweeps)
    pooled_mean = sum(s.s1 for s in sweeps) / n_all
    pooled_rms = math.sqrt(sum(s.s2 for s in sweeps) / n_all)
    per_unit: list[dict] = []
    pooled_central = [0.0, 0.0, 0]
    for u, s in zip(units, sweeps, strict=True):
        own = (s.s1 / s.n, 6.0 * math.sqrt(s.s2 / s.n))
        specs = [own] if len(units) == 1 else [own, (pooled_mean, 6.0 * pooled_rms)]
        res = _sweep2(u, specs)
        per_unit.append(
            _stats_dict(s.n, s.cols, s.s2, s.max_abs, res[0], s.row_ss, s.group_amax[16], s.group_amax[32])
        )
        if len(units) > 1:
            for i in range(3):
                pooled_central[i] += res[1][i]
    if len(units) == 1:
        return dict(per_unit[0]), per_unit
    cols = sweeps[0].cols
    pooled = _stats_dict(
        n_all,
        cols,
        sum(s.s2 for s in sweeps),
        max(s.max_abs for s in sweeps),
        (pooled_central[0], pooled_central[1], int(pooled_central[2])),
        torch.cat([s.row_ss for s in sweeps]),
        [p for s in sweeps for p in s.group_amax[16]],
        [p for s in sweeps for p in s.group_amax[32]],
    )
    return pooled, per_unit


# --------------------------------------------------------------------------- INT4 and MXFP4


def int4_g128_roundtrip(x: torch.Tensor) -> tuple[torch.Tensor, dict[str, int]]:
    """Round trip through symmetric INT4, group 128 along the last dim, fp16 scale.

    Returns:
        The dequantized float32 tensor and the element counts of groups whose scale underflows, is flushed
        to zero or saturates (see the module docstring).
    """
    x = x.to(torch.float32)
    *lead, cols = x.shape
    blocks = x.reshape(*lead, cols // INT4_GROUP, INT4_GROUP)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    raw = amax / INT4_SCALE_DIVISOR
    saturate = raw > FP16_MAX
    scale = raw.clamp(max=FP16_MAX).to(torch.float16).to(torch.float32)
    nonzero = amax > 0
    live = scale > 0
    q = torch.round(blocks / torch.where(live, scale, torch.ones_like(scale))).clamp(INT4_QMIN, INT4_QMAX)
    deq = torch.where(live, q * scale, torch.zeros_like(blocks)).reshape(x.shape)
    counts = {
        "underflow": int((nonzero & (scale < FP16_MIN_NORMAL)).sum()) * INT4_GROUP,
        "flushed": int((nonzero & ~live).sum()) * INT4_GROUP,
        "saturate": int(saturate.sum()) * INT4_GROUP,
    }
    return deq, counts


def mxfp4_exponent(amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared e8m0 exponent ``floor(log2(amax)) - 2`` clamped to [-127, 127], with underflow/saturate masks.

    ``floor(log2)`` is taken from ``frexp`` (``amax = m * 2^e`` with ``m`` in [0.5, 1) gives ``e - 1``).
    Zero amax gives a harmless exponent and no flag.
    """
    _, e = torch.frexp(amax)
    raw = e.to(torch.int64) - 1 - E2M1_EMAX
    nonzero = amax > 0
    underflow = nonzero & (raw < E8M0_MIN_EXP)
    saturate = nonzero & (raw > E8M0_MAX_EXP)
    return raw.clamp(E8M0_MIN_EXP, E8M0_MAX_EXP), underflow, saturate


def mxfp4_roundtrip(x: torch.Tensor) -> tuple[torch.Tensor, dict[str, int]]:
    """Round trip through MXFP4 (group 32 along the last dim, e8m0 scale, OCP floor rule).

    Returns:
        The dequantized float32 tensor and the element counts of groups whose exponent underflows or
        saturates (``flushed`` is always 0: e8m0 has no zero).
    """
    x = x.to(torch.float32)
    *lead, cols = x.shape
    blocks = x.reshape(*lead, cols // MXFP4_GROUP, MXFP4_GROUP)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    exp, underflow, saturate = mxfp4_exponent(amax)
    scale = torch.ldexp(torch.ones_like(amax), exp.to(torch.int32))
    v = (blocks / scale).clamp(-nvfp4.FP4_MAX, nvfp4.FP4_MAX)
    deq = (nvfp4.decode_fp4(nvfp4.encode_fp4(v)) * scale).reshape(x.shape)
    counts = {
        "underflow": int(underflow.sum()) * MXFP4_GROUP,
        "flushed": 0,
        "saturate": int(saturate.sum()) * MXFP4_GROUP,
    }
    return deq, counts


# --------------------------------------------------------------------------- per-format evaluation


@dataclass(frozen=True)
class _FormatResult:
    """Squared error and scale-failure element counts of one unit in one format."""

    err_ss: float
    underflow: int
    flushed: int
    saturate: int


def _err_ss(rel_fro: float, ref_ss: float) -> float:
    return 0.0 if ref_ss == 0.0 else rel_fro * rel_fro * ref_ss


def _eval_nvfp4(q_in: torch.Tensor, ref_ss: float) -> _FormatResult | None:
    if q_in.shape[-1] % nvfp4.GROUP_SIZE:
        return None
    r = nvfp4.quantize_nvfp4(q_in)
    st, g = r.scale_stats, nvfp4.GROUP_SIZE
    return _FormatResult(
        _err_ss(r.error.rel_fro, ref_ss),
        (st.n_flushed + st.n_subnormal) * g,
        st.n_flushed * g,
        st.n_overflow * g,
    )


def _eval_fp8(q_in: torch.Tensor, ref_ss: float) -> _FormatResult | None:
    r = fp8.quantize_fp8(q_in)
    floor = torch.tensor(fp8.MIN_SCALE, dtype=torch.float32, device=r.scale.device)
    clamped = int((r.scale <= floor).sum()) - r.scale_stats.n_zero_rows
    return _FormatResult(_err_ss(r.error.rel_fro, ref_ss), clamped * q_in.shape[-1], 0, 0)


def _eval_grouped(
    unit: torch.Tensor, group: int, roundtrip: Callable[[torch.Tensor], tuple[torch.Tensor, dict[str, int]]]
) -> _FormatResult | None:
    if unit.shape[-1] % group:
        return None
    err = 0.0
    counts = {"underflow": 0, "flushed": 0, "saturate": 0}
    for blk in _row_blocks(unit):
        x = blk.to(torch.float32)
        deq, c = roundtrip(x)
        err += float(((deq - x).double() ** 2).sum())
        for k, v in c.items():
            counts[k] += v
    return _FormatResult(err, counts["underflow"], counts["flushed"], counts["saturate"])


def _eval_format(fmt: str, unit: torch.Tensor, q_in: torch.Tensor, ref_ss: float) -> _FormatResult | None:
    if fmt == "nvfp4":
        return _eval_nvfp4(q_in, ref_ss)
    if fmt == "fp8":
        return _eval_fp8(q_in, ref_ss)
    if fmt == "int4_g128":
        return _eval_grouped(unit, INT4_GROUP, int4_g128_roundtrip)
    if fmt == "mxfp4":
        return _eval_grouped(unit, MXFP4_GROUP, mxfp4_roundtrip)
    raise ValueError(f"unknown format {fmt!r}")


def _rel(err_ss: float, ref_ss: float) -> float:
    return 0.0 if ref_ss == 0.0 else math.sqrt(err_ss / ref_ss)


def _quantiles(values: Sequence[float | None]) -> dict[str, float] | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    t = torch.tensor(vals, dtype=torch.float64)
    q = torch.quantile(t, torch.tensor([p for _, p in QUANTILES], dtype=torch.float64))
    return {name: float(v) for (name, _), v in zip(QUANTILES, q, strict=True)}


def measure_tensor(
    name: str, tensor: torch.Tensor, cls: str, layer: int | None, *, device: str = "cpu"
) -> dict[str, Any]:
    """Measure one weight: distribution statistics and the four format errors.

    A 3D tensor ``[E, out, in]`` is treated as E experts: each is quantized on its own (so NVFP4 gets one
    global scale per expert, as in the checkpoint), the tensor-level numbers pool them, and ``per_expert``
    holds quantiles over experts.

    Raises:
        ValueError: For a rank other than 2 or 3, or non-finite values.
    """
    if tensor.ndim not in (2, 3):
        raise ValueError(f"{name}: expected rank 2 or 3, got shape {tuple(tensor.shape)}")
    w = tensor.to(device)
    fused = w.ndim == 3
    units = [w[e] for e in range(w.shape[0])] if fused else [w]
    q_inputs = [w[e : e + 1] for e in range(w.shape[0])] if fused else [w]
    pooled, per_unit = compute_distribution(units)
    ref_ss = [(d["rms"] ** 2) * u.numel() for d, u in zip(per_unit, units, strict=True)]
    numel = int(w.numel())
    formats: dict[str, dict[str, float] | None] = {}
    per_expert_rel: dict[str, dict[str, float] | None] = {}
    not_applicable: list[str] = []
    for fmt in FORMATS:
        results = [_eval_format(fmt, u, qi, rs) for u, qi, rs in zip(units, q_inputs, ref_ss, strict=True)]
        if any(r is None for r in results):
            formats[fmt] = None
            not_applicable.append(fmt)
            continue
        done: list[_FormatResult] = [r for r in results if r is not None]
        formats[fmt] = {
            "rel_fro": _rel(sum(r.err_ss for r in done), sum(ref_ss)),
            "frac_scale_underflow": sum(r.underflow for r in done) / numel,
            "frac_scale_flushed": sum(r.flushed for r in done) / numel,
            "frac_scale_saturate": sum(r.saturate for r in done) / numel,
        }
        if fused:
            per_expert_rel[fmt] = _quantiles([_rel(r.err_ss, rs) for r, rs in zip(done, ref_ss, strict=True)])
    per_expert: dict[str, Any] | None = None
    if fused:
        per_expert = {m: _quantiles([d[m] for d in per_unit]) for m in PER_EXPERT_METRICS}
        per_expert["rel_fro"] = per_expert_rel
    return {
        "name": name,
        "class": cls,
        "layer": layer,
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "numel": numel,
        "stats": pooled,
        "formats": formats,
        "not_applicable": not_applicable,
        "per_expert": per_expert,
    }


# --------------------------------------------------------------------------- summary


def _class_rank(cls: str) -> tuple[int, str]:
    return (CLASS_ORDER.index(cls), cls) if cls in CLASS_ORDER else (len(CLASS_ORDER), cls)


def summarize(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per class and format: parameter-weighted mean error, worst tensor, ratio to the best format."""
    out: list[dict[str, Any]] = []
    for cls in sorted({r["class"] for r in rows}, key=_class_rank):
        in_class = [r for r in rows if r["class"] == cls]
        entries: list[dict[str, Any]] = []
        for fmt in FORMATS:
            have = [r for r in in_class if r["formats"][fmt] is not None]
            if not have:
                continue
            params = sum(r["numel"] for r in have)
            worst = max(have, key=lambda r, f=fmt: r["formats"][f]["rel_fro"])
            entries.append(
                {
                    "class": cls,
                    "format": fmt,
                    "n_tensors": len(have),
                    "params": params,
                    "wmean_rel_fro": sum(r["numel"] * r["formats"][fmt]["rel_fro"] for r in have) / params,
                    "worst_layer": worst["layer"],
                    "worst_name": worst["name"],
                    "worst_rel_fro": worst["formats"][fmt]["rel_fro"],
                    "wmean_frac_scale_underflow": sum(
                        r["numel"] * r["formats"][fmt]["frac_scale_underflow"] for r in have
                    )
                    / params,
                    "wmean_frac_scale_saturate": sum(
                        r["numel"] * r["formats"][fmt]["frac_scale_saturate"] for r in have
                    )
                    / params,
                }
            )
        best = min(e["wmean_rel_fro"] for e in entries)
        for e in entries:
            e["ratio_to_best"] = e["wmean_rel_fro"] / best if best > 0 else None
        out.extend(entries)
    return out


def summarize_distribution(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per class: weighted kurtosis and outlier fraction, worst max/RMS, channel and group ranges."""

    def wmean(rs: Sequence[Mapping[str, Any]], key: str) -> float | None:
        have = [r for r in rs if r["stats"][key] is not None]
        total = sum(r["numel"] for r in have)
        return sum(r["numel"] * r["stats"][key] for r in have) / total if total else None

    def worst(rs: Sequence[Mapping[str, Any]], key: str) -> tuple[float | None, int | None]:
        have = [r for r in rs if r["stats"][key] is not None]
        if not have:
            return None, None
        r = max(have, key=lambda x: x["stats"][key])
        return r["stats"][key], r["layer"]

    out = []
    for cls in sorted({r["class"] for r in rows}, key=_class_rank):
        rs = [r for r in rows if r["class"] == cls]
        mor, mor_layer = worst(rs, "max_over_rms")
        out.append(
            {
                "class": cls,
                "n_tensors": len(rs),
                "params": sum(r["numel"] for r in rs),
                "wmean_excess_kurtosis": wmean(rs, "excess_kurtosis"),
                "wmean_frac_gt_6rms": wmean(rs, "frac_gt_6rms"),
                "worst_max_over_rms": mor,
                "worst_max_over_rms_layer": mor_layer,
                "worst_channel_rms_range": worst(rs, "channel_rms_range")[0],
                "worst_g16_amax_max_over_median": worst(rs, "g16_amax_max_over_median")[0],
                "worst_g32_amax_max_over_median": worst(rs, "g32_amax_max_over_median")[0],
            }
        )
    return out


# --------------------------------------------------------------------------- output


def json_clean(obj: Any) -> Any:
    """Replace NaN and infinity by None (recursively) so that the output is standard JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Mapping):
        return {k: json_clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [json_clean(v) for v in obj]
    return obj


def _num(x: float | None, spec: str = ".4g") -> str:
    return "-" if x is None else format(x, spec)


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.3g}%"


def render_markdown(doc: Mapping[str, Any]) -> str:
    """Render the summary tables, with the checkpoint identity hash in the header."""
    ck, scope = doc["checkpoint"], doc["scope"]
    lines = [
        "# Weight-level quantizability statistics",
        "",
        f"Checkpoint index sha256 (`{ck['kind']}`): `{ck['sha256']}`",
        "",
        f"Source: `{doc['src']}`. Device: {doc['device']}. torch {doc['torch']}. "
        f"Tensors measured: {len(doc['rows'])}; not weight matrices (skipped): {len(doc['skipped'])}.",
    ]
    if scope["partial"]:
        lines += [
            "",
            f"**PARTIAL RUN** (layers {scope['layers']}, max tensors {scope['max_tensors']}): "
            "these tables are not a whole-model result.",
        ]
    lines += ["", "## Formats", ""]
    lines += [f"* **{FORMAT_LABELS[f]}** (`{f}`): {FORMAT_DEFINITIONS[f]}" for f in FORMATS]
    lines += [
        "",
        "## Round-to-nearest error per class and format",
        "",
        "Mean error is the parameter-weighted mean of the per-tensor relative Frobenius error "
        "`||W-Q(W)||/||W||`; the worst layer is the layer of the worst tensor of the class; ratio to best is "
        "the mean error over that of the best format of the class (1.00x is the best). Scale underflow and "
        "saturation are the parameter-weighted fractions of weights in blocks whose scale underflows or "
        "saturates.",
        "",
        "| class | format | tensors | params | mean rel. error | worst layer (error) | ratio to best "
        "| scale underflow | scale saturate |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for e in doc["summary"]:
        layer = "n/a" if e["worst_layer"] is None else f"L{e['worst_layer']}"
        ratio = "-" if e["ratio_to_best"] is None else f"{e['ratio_to_best']:.2f}x"
        lines.append(
            f"| `{e['class']}` | {e['format']} | {e['n_tensors']} | {e['params']:,} | "
            f"{_num(e['wmean_rel_fro'])} | {layer} ({_num(e['worst_rel_fro'])}) | {ratio} | "
            f"{_pct(e['wmean_frac_scale_underflow'])} | {_pct(e['wmean_frac_scale_saturate'])} |"
        )
    lines += [
        "",
        "## Distribution shape per class",
        "",
        "| class | tensors | params | mean excess kurtosis | mean frac > 6 RMS | worst max/RMS (layer) "
        "| worst channel RMS range | worst g16 amax max/median | worst g32 amax max/median |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for d in doc["distribution_summary"]:
        layer = "n/a" if d["worst_max_over_rms_layer"] is None else f"L{d['worst_max_over_rms_layer']}"
        lines.append(
            f"| `{d['class']}` | {d['n_tensors']} | {d['params']:,} | {_num(d['wmean_excess_kurtosis'])} | "
            f"{_num(d['wmean_frac_gt_6rms'])} | {_num(d['worst_max_over_rms'])} ({layer}) | "
            f"{_num(d['worst_channel_rms_range'])} | {_num(d['worst_g16_amax_max_over_median'])} | "
            f"{_num(d['worst_g32_amax_max_over_median'])} |"
        )
    return "\n".join(lines) + "\n"


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


# --------------------------------------------------------------------------- driver


def parse_layers(spec: str) -> tuple[int | None, int | None]:
    """Parse ``A:B`` (half-open, either side optional) into ``(start, end)``."""
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(f"--layers expects A:B, got {spec!r}")
    try:
        start = int(parts[0]) if parts[0] else None
        end = int(parts[1]) if parts[1] else None
    except ValueError:
        raise ValueError(f"--layers expects integers in A:B, got {spec!r}") from None
    if start is None and end is None:
        raise ValueError(f"--layers needs at least one bound, got {spec!r}")
    if start is not None and end is not None and start >= end:
        raise ValueError(f"--layers {spec!r}: start must be below end")
    return start, end


def layer_selected(layer: int | None, layers: tuple[int | None, int | None] | None) -> bool:
    """Whether ``layer`` is in the half-open range (layerless tensors only when there is no range)."""
    if layers is None:
        return True
    if layer is None:
        return False
    start, end = layers
    return (start is None or layer >= start) and (end is None or layer < end)


def run(
    src: Path,
    out: Path,
    *,
    device: str,
    layers: tuple[int | None, int | None] | None = None,
    max_tensors: int | None = None,
) -> dict[str, Any]:
    """Measure the checkpoint in ``src``; write ``weight_stats.json`` and ``weight_stats.md`` to ``out``."""
    ckpt = SourceCheckpoint(src)
    kind, digest = ckpt.identity_hash()
    header = ckpt.header()
    names = set(header)
    attention = attention_layers(names)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    start = time.monotonic()
    done = False
    for shard in ckpt.shard_files():
        raw, _ = read_safetensors_header(src / shard)
        for name in sorted(k for k in raw if k != "__metadata__" and k in names):
            wc = classify_weight(name, attention)
            meta = header[name]
            if wc is None:
                skipped.append({"name": name, "reason": skip_reason(name)})
                continue
            if len(meta.shape) not in (2, 3) or meta.dtype not in _FLOAT_DTYPES:
                skipped.append({"name": name, "reason": f"unsupported rank/dtype {meta.shape} {meta.dtype}"})
                continue
            if done or not layer_selected(wc.layer, layers):
                continue
            t0 = time.monotonic()
            tensor = ckpt.get_tensor(name)
            rows.append(measure_tensor(name, tensor, wc.cls, wc.layer, device=device))
            del tensor
            if device == "cuda":
                torch.cuda.empty_cache()
            print(
                f"[{len(rows)}] {name} {wc.cls} {list(meta.shape)} {time.monotonic() - t0:.1f}s "
                f"(total {time.monotonic() - start:.0f}s)",
                file=sys.stderr,
                flush=True,
            )
            if max_tensors is not None and len(rows) >= max_tensors:
                done = True
        ckpt.close()
    rows.sort(key=lambda r: r["name"])
    skipped.sort(key=lambda s: s["name"])
    doc: dict[str, Any] = {
        "schema": 1,
        "tool": "olala_quant.weight_stats",
        "checkpoint": {"kind": kind, "sha256": digest},
        "src": str(src),
        "device": device,
        "torch": torch.__version__,
        "scope": {
            "layers": None if layers is None else list(layers),
            "max_tensors": max_tensors,
            "partial": layers is not None or max_tensors is not None,
        },
        "format_definitions": FORMAT_DEFINITIONS,
        "rows": rows,
        "skipped": skipped,
        "summary": summarize(rows),
        "distribution_summary": summarize_distribution(rows),
    }
    doc = json_clean(doc)
    out.mkdir(parents=True, exist_ok=True)
    _write_atomic(out / "weight_stats.json", json.dumps(doc, indent=1, allow_nan=False) + "\n")
    _write_atomic(out / "weight_stats.md", render_markdown(doc))
    return doc


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Weight-level quantizability statistics of a bf16 checkpoint.")
    p.add_argument(
        "--src", type=Path, required=True, help="checkpoint directory (safetensors, indexed or single)"
    )
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--device", choices=("cuda", "cpu"), default=None, help="default: cuda when available")
    p.add_argument("--layers", default=None, help="half-open layer range A:B (drops embedding and lm_head)")
    p.add_argument("--max-tensors", type=int, default=None, help="stop after this many measured tensors")
    args = p.parse_args(argv)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("--device cuda requested but CUDA is not available")
    if args.layers is not None:
        try:
            args.layers = parse_layers(args.layers)
        except ValueError as exc:
            p.error(str(exc))
    if args.max_tensors is not None and args.max_tensors < 1:
        p.error("--max-tensors must be at least 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point."""
    args = _parse_args(argv)
    run(args.src, args.out, device=args.device, layers=args.layers, max_tensors=args.max_tensors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
