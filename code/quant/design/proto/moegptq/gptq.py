"""GPTQ (Frantar et al., 2022) for one 2D weight, written for the schemes of ``schemes.py``.

Differences from the reference implementation that matter for this study, each stated here:

* Group scales are either computed once from the ORIGINAL weights (``dynamic=False``, what llm-compressor
  does: its observer scales are fixed before the solve) or at the start of each group from the weights as
  updated by the error of the earlier columns (``dynamic=True``, the original GPTQ behaviour).
* Dead columns (zero Hessian diagonal) are NOT zeroed: a channel that never fired in calibration may fire
  at serving time, so the weight is rounded like any other and receives no compensation.
* A Hessian that is not finite or not positive definite after damping falls back to round-to-nearest and
  says so in the returned info (``fallback``).
* Damping is ``percdamp * mean(diag H)`` (reference) or ``percdamp * diag(H)`` (``damp_mode='diag'``,
  invariant to per-channel rescaling, a candidate mitigation for huge channels).
"""

from __future__ import annotations

from typing import Any

import torch

from moegptq.schemes import Int4Scheme, Nvfp4Scheme, effective_group, rtn_quantize


def weighted_error(w: torch.Tensor, q: torch.Tensor, h: torch.Tensor) -> float:
    """``trace(dW H dW^T)``, the layer-output squared error on the calibration set that H describes."""
    dw = (w - q).double()
    return float(((dw @ h.double()) * dw).sum())


def _damped_inverse_factor(h: torch.Tensor, percdamp: float, damp_mode: str) -> torch.Tensor | None:
    """Upper Cholesky factor of the inverse of the damped Hessian, or None if it cannot be computed."""
    if not bool(torch.isfinite(h).all()):
        return None
    n = h.shape[0]
    diag = torch.diagonal(h).clone()
    if bool((diag < 0).any()):
        return None
    h = h.clone()
    dead = diag == 0
    if bool(dead.any()):
        h[dead, dead] = 1.0
        diag[dead] = 1.0
    if damp_mode == "mean":
        h += percdamp * diag.mean() * torch.eye(n, dtype=h.dtype)
    else:
        h += percdamp * torch.diag(diag)
    chol, info = torch.linalg.cholesky_ex(h)
    if int(info) != 0 or not bool(torch.isfinite(chol).all()):
        return None
    inv = torch.cholesky_inverse(chol)
    upper, info = torch.linalg.cholesky_ex(inv, upper=True)
    if int(info) != 0 or not bool(torch.isfinite(upper).all()):
        return None
    return upper


def gptq_quantize(
    w: torch.Tensor,
    h: torch.Tensor,
    scheme: Nvfp4Scheme | Int4Scheme,
    *,
    percdamp: float = 0.01,
    damp_mode: str = "mean",
    blocksize: int = 128,
    dynamic: bool = True,
    act_order: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Quantize ``w [rows, cols]`` with the Hessian ``h [cols, cols]`` of its inputs.

    Args:
        w: Weight to quantize (float32 or bfloat16 values; the solve runs in float64).
        h: ``sum_t x_t x_t^T`` over the calibration tokens that reach this weight.
        scheme: Grid (NVFP4 or INT4).
        percdamp: Damping fraction.
        damp_mode: ``'mean'`` or ``'diag'`` (see module docstring).
        blocksize: Columns per lazy-update block (rounded to a multiple of the group size).
        dynamic: Group scales from the updated weights (True) or from the original weights (False).
        act_order: Process columns by descending Hessian diagonal. The groups stay the natural contiguous
            groups with scales fixed from the original weights ("static groups", llm-compressor's default
            ``actorder='static'``): the stored format needs no ``g_idx``. Requires ``dynamic=False``.

    Returns:
        The dequantized weight (dtype of ``w``) and an info dict: ``loss`` (the GPTQ error sum),
        ``fallback`` (True if RTN was used because the Hessian was unusable) and ``scales``.
    """
    if damp_mode not in ("mean", "diag"):
        raise ValueError(f"damp_mode must be 'mean' or 'diag', got {damp_mode!r}")
    if act_order and dynamic:
        raise ValueError("act_order needs static group scales (dynamic=False)")
    _rows, cols = w.shape
    gs = effective_group(scheme, cols)
    h = h.to(torch.float64)
    perm = torch.argsort(torch.diagonal(h), descending=True) if act_order else torch.arange(cols)
    if act_order:
        h = h[perm][:, perm]
    upper = _damped_inverse_factor(h, percdamp, damp_mode)
    if upper is None:
        q, scales = rtn_quantize(w, scheme)
        return q, {"loss": float("nan"), "fallback": True, "scales": scales}
    bs = max(gs, blocksize - blocksize % gs)
    wc = w.to(torch.float64).clone()
    state = scheme.prepare(w)
    static = None
    if not dynamic:
        static = [scheme.scales(wc[:, c : c + gs].float(), state) for c in range(0, cols, gs)]
    wc = wc[:, perm]
    out = torch.zeros_like(wc)
    scale_list: list[torch.Tensor] = [torch.empty(0)] * (cols // gs)
    loss = 0.0
    for i1 in range(0, cols, bs):
        i2 = min(i1 + bs, cols)
        w1 = wc[:, i1:i2].clone()
        err1 = torch.zeros_like(w1)
        hinv1 = upper[i1:i2, i1:i2]
        s = None
        for i in range(i2 - i1):
            col = i1 + i
            if act_order:
                s = static[int(perm[col]) // gs]
            elif col % gs == 0:
                gi = col // gs
                s = static[gi] if static is not None else scheme.scales(w1[:, i : i + gs].float(), state)
                scale_list[gi] = s
            wi = w1[:, i]
            qi = scheme.round(wi.float().unsqueeze(1), s).squeeze(1).to(torch.float64)
            out[:, col] = qi
            d = hinv1[i, i]
            err = (wi - qi) / d
            loss += float((err**2).sum()) / 2.0
            w1[:, i:] -= err.unsqueeze(1) * hinv1[i, i:].unsqueeze(0)
            err1[:, i] = err
        wc[:, i2:] -= err1 @ upper[i1:i2, i2:]
    if act_order:
        scale_list = static
        restored = torch.empty_like(out)
        restored[:, perm] = out
        out = restored
    return out.to(w.dtype), {"loss": loss, "fallback": False, "scales": torch.cat(scale_list, dim=1)}
