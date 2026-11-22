"""Per-site activation statistics and their accumulation over prompts (pure torch / numpy, no model code).

A *site* is one tensor that a hook sees (a branch output, the residual stream, the post-ReLU-squared activation, ...).
:func:`summarize` reduces ONE tensor from ONE forward pass to a small :class:`Summary`; :class:`SiteAcc` pools the
summaries of all prompts into the numbers of the report. :func:`head_summary` / :class:`HeadAcc` do the same for the
per-head outputs of a Mamba-3 mixer.

Conventions that the tests pin down:

* ``max_abs`` is the maximum of ``|x|`` over the FINITE entries; NaN and infinities are counted separately
  (``n_nan``, ``n_inf``) and ``n_ge_f32max`` counts entries ``>= 3.4e38`` (infinities included).
* The root mean square is taken over all entries in float64 (squares of 1e16 overflow neither float32 sums of one
  row nor the float64 total), non-finite entries counted as zero.
* ``p999`` is the ``ceil(0.999 * N)``-th smallest ``|x|`` of one prompt (an order statistic, not interpolated).
* A *token maximum* is ``max_c |x[t, c]|``; fractions of tokens above 448 / 65504 are fractions of token maxima
  strictly above the threshold (448 = float8 e4m3 max, 65504 = float16 max).
* Positions are 0-based; position 0 is the first token of the sequence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

E4M3_MAX = 448.0
FP16_MAX = 65504.0
F32_MAX_ISH = 3.4e38
P_QUANTILE = 0.999
HIST_LO, HIST_HI, HIST_STEP = -8.0, 24.0, 0.1  # log10 bins of the token maxima
REL_LEVELS = (0.5, 0.1, 0.01, 0.0001)
LOG_FLOOR = 1e-30
MIN_POSITIONS_FOR_EXPONENT = 8


@dataclass
class Summary:
    """The reduction of one tensor of one forward pass."""

    max_abs: float
    sumsq: float
    n_elems: int
    p999: float
    n_nan: int
    n_inf: int
    n_ge_f32max: int
    n_tok_gt_448: int
    n_tok_gt_65504: int
    tok_max: np.ndarray  # (num_tokens,) float64
    arg_token: int
    arg_channel: int
    arg_row: int


def summarize(x: torch.Tensor, *, token_of: torch.Tensor | None = None, num_tokens: int | None = None) -> Summary:
    """Reduce ``x`` (rows by channels; any leading dims are flattened) to a :class:`Summary`.

    Args:
        x: The activations. One row per token, unless ``token_of`` is given.
        token_of: Optional ``[rows]`` int tensor mapping each row to a token (several rows may share a token, as the
            (token, expert) pairs of a mixture of experts do); the token maximum is then the maximum over its rows.
        num_tokens: Number of tokens (required with ``token_of``; defaults to the number of rows otherwise).
    """
    xf = x.detach().reshape(-1, x.shape[-1]).float()
    rows = xf.shape[0]
    finite = torch.isfinite(xf)
    n_nan = int(torch.isnan(xf).sum())
    n_inf = int(torch.isinf(xf).sum())
    absx = xf.abs()
    n_ge = int((absx >= F32_MAX_ISH).sum())
    a = torch.where(finite, absx, torch.zeros_like(absx))
    n = a.numel()
    k = max(1, math.ceil(P_QUANTILE * n))
    p999 = float(torch.kthvalue(a.reshape(-1), k).values)
    sumsq = float(a.double().pow(2).sum())
    row_max, row_arg = a.max(dim=1)
    if token_of is None:
        num = rows if num_tokens is None else num_tokens
        tok = row_max
    else:
        if num_tokens is None:
            raise ValueError("num_tokens is required with token_of")
        num = num_tokens
        tok = torch.zeros(num, dtype=row_max.dtype, device=row_max.device).scatter_reduce(
            0, token_of.to(row_max.device), row_max, reduce="amax", include_self=True
        )
    top = int(tok.argmax())
    if token_of is None:
        arg_row = top
    else:
        hit = ((token_of.to(row_max.device) == top) & (row_max == tok[top])).nonzero().flatten()
        arg_row = int(hit[0]) if hit.numel() else 0
    return Summary(
        max_abs=float(tok.max()) if tok.numel() else 0.0,
        sumsq=sumsq,
        n_elems=n,
        p999=p999,
        n_nan=n_nan,
        n_inf=n_inf,
        n_ge_f32max=n_ge,
        n_tok_gt_448=int((tok > E4M3_MAX).sum()),
        n_tok_gt_65504=int((tok > FP16_MAX).sum()),
        tok_max=tok.double().cpu().numpy(),
        arg_token=top,
        arg_channel=int(row_arg[arg_row]),
        arg_row=arg_row,
    )


def growth_exponent(mean_log10: np.ndarray, counts: np.ndarray, lo: int = 16) -> float | None:
    """Slope of ``mean_log10`` against ``log10(position + 1)`` over the positions ``>= lo`` that have data.

    ``None`` when fewer than 8 positions qualify. A magnitude ``~ t**a`` gives ``a``.
    """
    idx = np.arange(len(mean_log10))
    ok = (counts > 0) & (idx >= lo)
    if int(ok.sum()) < MIN_POSITIONS_FOR_EXPONENT:
        return None
    x = np.log10(idx[ok] + 1.0)
    y = np.asarray(mean_log10)[ok]
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc * xc).sum())


def _sample_positions(n: int) -> list[int]:
    out = [0, 1, 2]
    p = 4
    while p < n:
        out.append(p)
        p *= 2
    return [p for p in out if p < n]


class _Profile:
    """Mean of log10(token max) per position, over the prompts long enough to have that position."""

    def __init__(self, max_len: int, width: int = 1) -> None:
        self.sum = np.zeros((max_len, width))
        self.cnt = np.zeros(max_len)

    def add(self, tok_max: np.ndarray) -> None:
        n = min(len(tok_max), self.sum.shape[0])
        t = np.log10(np.maximum(tok_max[:n], LOG_FLOOR))
        self.sum[:n] += t.reshape(n, -1)
        self.cnt[:n] += 1

    def mean(self) -> np.ndarray:
        return self.sum / np.maximum(self.cnt, 1)[:, None]


def _argmax_info(prompt: int, pos: int, tokens: np.ndarray, special: np.ndarray, channel: int) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "position": pos,
        "token_id": int(tokens[pos]),
        "channel": channel,
        "is_special": bool(special[pos]),
        "is_first": pos == 0,
    }


class SiteAcc:
    """Pools the :class:`Summary` of every prompt for one site (a layer and a kind of tensor)."""

    def __init__(self, layer: int, kind: str, max_len: int, top_n: int = 0) -> None:
        """Create an empty accumulator; ``max_len`` bounds the per-position profile, ``top_n`` the token list kept."""
        self.layer, self.kind, self.max_len, self.top_n = layer, kind, max_len, top_n
        self.top: list[tuple[float, int, int, int]] = []  # (token max, prompt, position, token id)
        self.n_prompts = self.n_tokens = self.n_elems = 0
        self.sumsq = 0.0
        self.max_abs = 0.0
        self.argmax: dict[str, Any] | None = None
        self.p999: list[float] = []
        self.n_nan = self.n_inf = self.n_ge_f32max = 0
        self.n_tok_gt_448 = self.n_tok_gt_65504 = 0
        self.argmax_special = self.argmax_first = 0
        self.profile = _Profile(max_len)
        self.hist = np.zeros(round((HIST_HI - HIST_LO) / HIST_STEP), dtype=np.int64)

    def add(self, s: Summary, prompt: int, tokens: np.ndarray, special: np.ndarray) -> None:
        """Fold one prompt's summary in (``tokens`` and ``special``: its token ids and special-token flags)."""
        self.n_prompts += 1
        self.n_tokens += len(s.tok_max)
        self.n_elems += s.n_elems
        self.sumsq += s.sumsq
        self.p999.append(s.p999)
        self.n_nan += s.n_nan
        self.n_inf += s.n_inf
        self.n_ge_f32max += s.n_ge_f32max
        self.n_tok_gt_448 += s.n_tok_gt_448
        self.n_tok_gt_65504 += s.n_tok_gt_65504
        is_special = bool(special[s.arg_token])
        self.argmax_special += int(is_special)
        self.argmax_first += int(s.arg_token == 0)
        if self.argmax is None or s.max_abs > self.max_abs:
            self.max_abs = s.max_abs
            self.argmax = _argmax_info(prompt, s.arg_token, tokens, special, s.arg_channel)
        self.profile.add(s.tok_max)
        if self.top_n:
            k = min(self.top_n, len(s.tok_max))
            for pos in np.argpartition(-s.tok_max, k - 1)[:k]:
                self.top.append((float(s.tok_max[pos]), prompt, int(pos), int(tokens[pos])))
            self.top = sorted(self.top, key=lambda r: -r[0])[: self.top_n]
        lg = np.log10(np.maximum(s.tok_max, LOG_FLOOR))
        b = np.clip(np.floor((lg - HIST_LO) / HIST_STEP).astype(np.int64), 0, len(self.hist) - 1)
        np.add.at(self.hist, b, 1)

    def _frac_ge_rel(self) -> dict[str, float]:
        out = {}
        for rel in REL_LEVELS:
            thr = math.log10(max(self.max_abs, LOG_FLOOR)) + math.log10(rel)
            lower = HIST_LO + HIST_STEP * np.arange(len(self.hist))
            out[f"{rel:g}"] = float(self.hist[lower >= thr - 1e-9].sum() / max(self.n_tokens, 1))
        return out

    def result(self) -> dict[str, Any]:
        """The pooled numbers of this site, JSON-ready."""
        rms = math.sqrt(self.sumsq / self.n_elems) if self.n_elems else 0.0
        mean = self.profile.mean()[:, 0]
        cnt = self.profile.cnt
        later = mean[1:][cnt[1:] > 0]
        pos0_min = bool(cnt[0] > 0 and (later.size == 0 or mean[0] <= later.min() + 1e-12))
        out = {
            "layer": self.layer,
            "kind": self.kind,
            "n_prompts": self.n_prompts,
            "n_tokens": self.n_tokens,
            "max_abs": self.max_abs,
            "rms": rms,
            "max_over_rms": self.max_abs / rms if rms > 0 else None,
            "p999_median_over_prompts": float(np.median(self.p999)) if self.p999 else None,
            "p999_max_over_prompts": max(self.p999) if self.p999 else None,
            "argmax": self.argmax,
            "frac_prompts_argmax_special": self.argmax_special / max(self.n_prompts, 1),
            "frac_prompts_argmax_first": self.argmax_first / max(self.n_prompts, 1),
            "frac_tokens_gt_448": self.n_tok_gt_448 / max(self.n_tokens, 1),
            "frac_tokens_gt_65504": self.n_tok_gt_65504 / max(self.n_tokens, 1),
            "frac_tokens_ge_rel_max": self._frac_ge_rel(),
            "n_nan": self.n_nan,
            "n_inf": self.n_inf,
            "n_ge_3p4e38": self.n_ge_f32max,
            "growth_exponent": growth_exponent(mean, cnt),
            "position0_is_smallest": pos0_min,
            "profile_log10": {str(p): float(mean[p]) for p in _sample_positions(self.max_len) if cnt[p] > 0},
        }
        if self.top_n:
            out["top_tokens"] = [
                {"value": v, "prompt": pr, "position": pos, "token_id": tid} for v, pr, pos, tid in self.top
            ]
        return out


@dataclass
class HeadSummary:
    """Per-head reduction of a Mamba-3 mixer output ``[1, L, n_heads * head_dim]``."""

    head_max: torch.Tensor  # (H,) float32
    head_sumsq: torch.Tensor  # (H,) float64
    head_arg_pos: torch.Tensor  # (H,)
    head_arg_channel: torch.Tensor  # (H,)
    head_tok_max: torch.Tensor  # (L, H)
    n_elems_per_head: int
    n_nan: int
    n_inf: int


def head_summary(y: torch.Tensor, n_heads: int) -> HeadSummary:
    """Per-head maxima, square sums and argmax positions / channels of ``y`` (non-finite entries counted as zero)."""
    yf = y.detach().reshape(-1, y.shape[-1]).float()
    length = yf.shape[0]
    a = yf.abs()
    n_nan = int(torch.isnan(a).sum())
    n_inf = int(torch.isinf(a).sum())
    a = torch.where(torch.isfinite(a), a, torch.zeros_like(a)).view(length, n_heads, -1)
    tok_max, _ = a.max(dim=-1)  # (L, H)
    head_max, arg_pos = tok_max.max(dim=0)
    picked = a[arg_pos, torch.arange(n_heads, device=a.device)]  # (H, P)
    return HeadSummary(
        head_max=head_max,
        head_sumsq=a.double().pow(2).sum(dim=(0, 2)),
        head_arg_pos=arg_pos,
        head_arg_channel=picked.argmax(dim=-1),
        head_tok_max=tok_max,
        n_elems_per_head=length * a.shape[-1],
        n_nan=n_nan,
        n_inf=n_inf,
    )


class HeadAcc:
    """Pools :class:`HeadSummary` over prompts for one Mamba-3 layer: which head dominates, and by how much."""

    def __init__(self, layer: int, n_heads: int, max_len: int, head_dim: int) -> None:
        """Create an empty accumulator for a layer of ``n_heads`` heads."""
        self.layer, self.n_heads, self.max_len, self.head_dim = layer, n_heads, max_len, head_dim
        self.head_max = np.zeros(n_heads)
        self.sumsq = np.zeros(n_heads)
        self.n_elems = 0
        self.info: list[dict[str, Any] | None] = [None] * n_heads
        self.profile = _Profile(max_len, n_heads)
        self.n_nan = self.n_inf = 0
        self.n_prompts = 0

    def add(self, h: HeadSummary, prompt: int, tokens: np.ndarray, special: np.ndarray) -> None:
        """Fold one prompt in."""
        self.n_prompts += 1
        self.n_elems += h.n_elems_per_head
        self.sumsq += h.head_sumsq.cpu().numpy()
        self.n_nan += h.n_nan
        self.n_inf += h.n_inf
        hm = h.head_max.double().cpu().numpy()
        pos = h.head_arg_pos.cpu().numpy()
        ch = h.head_arg_channel.cpu().numpy()
        for j in range(self.n_heads):
            if self.info[j] is None or hm[j] > self.head_max[j]:
                self.head_max[j] = hm[j]
                self.info[j] = _argmax_info(prompt, int(pos[j]), tokens, special, int(ch[j]))
        self.profile.add(h.head_tok_max.double().cpu().numpy())

    def result(self) -> dict[str, Any]:
        """The pooled per-head numbers of this layer, JSON-ready."""
        top = int(self.head_max.argmax())
        others = np.delete(self.head_max, top)
        rms = np.sqrt(self.sumsq / max(self.n_elems, 1))
        second = float(others.max()) if others.size else 0.0
        mean = self.profile.mean()
        return {
            "layer": self.layer,
            "n_heads": self.n_heads,
            "top_head": top,
            "top_head_max": float(self.head_max[top]),
            "second_head_max": second,
            "median_sibling_max": float(np.median(others)) if others.size else 0.0,
            "median_sibling_rms": float(np.median(np.delete(rms, top))) if others.size else 0.0,
            "dominance_ratio": float(self.head_max[top] / second) if second > 0 else None,
            "top_head_argmax": self.info[top],
            "top_head_growth_exponent": growth_exponent(mean[:, top], self.profile.cnt),
            "head_max": self.head_max.tolist(),
            "head_rms": rms.tolist(),
            "n_nan": self.n_nan,
            "n_inf": self.n_inf,
        }
