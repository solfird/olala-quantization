"""Per-expert GPTQ on the NVFP4 grid with STATIC scales, batched over experts (decision D1 of the design study).

The block scales (e4m3, one per 16 input columns) and the per-expert fp32 global scale are exactly those that
``olala_quant.nvfp4.quantize_nvfp4`` derives from the original weights; the solver only chooses the 4-bit CODES. The
codes it returns are written as they are (never re-quantized from float values: re-quantizing changes them, E5 of the
design study), so G1 and the data-free N2 build share every scale and differ only in the rounding decisions.

The algorithm is the one of ``moegptq.gptq.gptq_quantize`` (static scales, static act-order = columns solved by
descending Hessian diagonal with the natural groups, no ``g_idx`` in the file) vectorised over a batch of experts: the
column loop is shared and every operation is a batched tensor operation, so a layer of 256 experts costs
``cols`` Python iterations, not ``256 x cols``. The arithmetic is float64 by default (the Hessians are accumulated
in float32 elsewhere; the Cholesky and the error propagation are where precision is lost).

Guards, each reported per expert with a reason string:

* unusable Hessian (not finite, negative diagonal, not positive definite after damping): round-to-nearest;
* diagonal ratio ``max(diag H) / median(diag H)`` above ``diag_ratio_max``: round-to-nearest (spike guard);
* experts masked ``inactive`` by the caller (no calibration tokens): round-to-nearest;
* a held-out guard, :func:`heldout_guard`, applied after the solve by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from olala_quant import nvfp4

GROUP = nvfp4.GROUP_SIZE
FP4_MAX = nvfp4.FP4_MAX


@dataclass(frozen=True)
class SolverConfig:
    """Solver settings.

    Attributes:
        percdamp: Damping as a fraction of the mean Hessian diagonal (``0.01`` is the reference GPTQ value).
        blocksize: Columns per lazy-update block (rounded to a multiple of the 16-column group).
        act_order: Solve columns by descending Hessian diagonal (static groups, no ``g_idx``).
        diag_ratio_max: Diagonal-ratio spike guard; larger ratios go to round-to-nearest.
        expert_chunk: Experts solved at once (memory: about ``chunk x cols^2 x 8 B x 4``).
        dtype: Solve dtype (float64 keeps the Cholesky stable; float32 is faster on consumer GPUs).
    """

    percdamp: float = 0.01
    blocksize: int = 128
    act_order: bool = True
    diag_ratio_max: float = 1e6
    expert_chunk: int = 32
    dtype: torch.dtype = torch.float64


@dataclass(frozen=True)
class StaticNvfp4:
    """The data-free (N2) quantization of a ``[E, R, C]`` weight: scales, and the RTN codes.

    Attributes:
        rtn_packed: ``uint8 [E, R, C/2]`` round-to-nearest codes (what N2 wrote).
        block_scale: ``float8_e4m3fn [E, R, C/16]``.
        global_scale: ``float32 [E]`` encode scale ``448*6/amax``.
        dq_scale: ``float32 [E, R, C/16]`` dequantization scale ``block_scale * (1 / global_scale)``.
    """

    rtn_packed: torch.Tensor
    block_scale: torch.Tensor
    global_scale: torch.Tensor
    dq_scale: torch.Tensor


@dataclass
class SolveResult:
    """Output of :func:`solve_experts`.

    Attributes:
        codes: ``uint8 [E, R, C]`` e2m1 nibbles (RTN codes for the experts that fell back).
        loss: ``float64 [E]`` the GPTQ internal error sum (NaN for fallbacks).
        fallback: ``bool [E]`` True where round-to-nearest was used.
        reason: Per-expert reason for the fallback (empty string when GPTQ was used).
        diag_ratio: ``float64 [E]`` ``max(diag H) / median(diag H)`` (inf when the median is 0).
    """

    codes: torch.Tensor
    loss: torch.Tensor
    fallback: torch.Tensor
    reason: list[str] = field(default_factory=list)
    diag_ratio: torch.Tensor | None = None


def static_nvfp4(w: torch.Tensor, *, expert_chunk: int = 8) -> StaticNvfp4:
    """Quantize ``w [E, R, C]`` with the data-free quantizer and keep its scales (the N2 scales).

    Raises:
        ValueError: If ``w`` is not rank 3 or ``C`` is not a multiple of 16 (raised by ``quantize_nvfp4``).
    """
    if w.ndim != 3:
        raise ValueError(f"static_nvfp4 needs [E, R, C], got {tuple(w.shape)}")
    r = nvfp4.quantize_nvfp4(w, expert_chunk=expert_chunk)
    dq = r.block_scale.to(torch.float32) * (1.0 / r.global_scale.reshape(-1, 1, 1))
    return StaticNvfp4(r.packed, r.block_scale, r.global_scale, dq)


def rtn_codes(state: StaticNvfp4) -> torch.Tensor:
    """The round-to-nearest codes of ``state`` as ``uint8 [E, R, C]`` nibbles."""
    return nvfp4.unpack_fp4(state.rtn_packed)


def dequantize_codes(codes: torch.Tensor, state: StaticNvfp4, device: torch.device | str | None = None) -> torch.Tensor:
    """Dequantize nibbles with the static scales.

    ``decode(code) * dq_scale`` in float32, bitwise equal to ``olala_quant.nvfp4.dequantize_nvfp4`` of the packed codes.
    """
    e, r, c = codes.shape
    dq = state.dq_scale if device is None else state.dq_scale.to(device)
    values = nvfp4.decode_fp4(codes if device is None else codes.to(device))
    return (values.reshape(e, r, c // GROUP, GROUP) * dq.unsqueeze(-1)).reshape(e, r, c)


def weighted_error(w: torch.Tensor, deq: torch.Tensor, h: torch.Tensor, chunk: int = 32) -> torch.Tensor:
    """``trace(dW H dW^T)`` per expert, float64: the squared output error on the tokens that H describes."""
    out = []
    for i in range(0, w.shape[0], chunk):
        dw = w[i : i + chunk].to(torch.float64) - deq[i : i + chunk].to(torch.float64)
        out.append(((dw @ h[i : i + chunk].to(torch.float64)) * dw).sum(dim=(1, 2)))
    return torch.cat(out)


def _damped_inverse_factor(h: torch.Tensor, percdamp: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched upper Cholesky factor of the inverse of the damped Hessians.

    Returns:
        ``(upper [E, C, C], ok [E] bool, reason_code [E] int)`` with reason codes 0 ok, 1 not finite,
        2 negative diagonal, 3 not positive definite. Failed experts get an identity factor so that the batch goes on.
    """
    e, c, _ = h.shape
    eye = torch.eye(c, dtype=h.dtype, device=h.device)
    finite = torch.isfinite(h).flatten(1).all(dim=1)
    h = torch.where(finite[:, None, None], h, eye.expand_as(h))
    diag = torch.diagonal(h, dim1=1, dim2=2)
    neg = (diag < 0).any(dim=1)
    dead = diag == 0
    diag = torch.where(dead, torch.ones_like(diag), diag)
    h = h + torch.diag_embed(dead.to(h.dtype))  # dead columns get a unit diagonal, as in the prototype
    h = h + (percdamp * diag.mean(dim=1))[:, None, None] * eye
    h = torch.where(neg[:, None, None], eye.expand_as(h), h)
    chol, info = torch.linalg.cholesky_ex(h)
    pd_ok = info == 0
    chol = torch.where(pd_ok[:, None, None], chol, eye.expand_as(chol))
    inv = torch.cholesky_inverse(chol)
    upper, info2 = torch.linalg.cholesky_ex(inv, upper=True)
    ok2 = (info2 == 0) & torch.isfinite(upper).flatten(1).all(dim=1)
    upper = torch.where(ok2[:, None, None], upper, eye.expand_as(upper))
    code = torch.zeros(e, dtype=torch.int64, device=h.device)
    code = torch.where(~(pd_ok & ok2), torch.full_like(code, 3), code)
    code = torch.where(neg, torch.full_like(code, 2), code)
    code = torch.where(~finite, torch.ones_like(code), code)
    return upper, code == 0, code


_REASONS = {1: "hessian_not_finite", 2: "hessian_negative_diagonal", 3: "hessian_not_positive_definite"}


def _diag_ratio(h: torch.Tensor) -> torch.Tensor:
    """``max(diag) / median(diag)`` per matrix (inf when the median is zero and the max is not)."""
    diag = torch.diagonal(h, dim1=1, dim2=2).to(torch.float64)
    mx = diag.amax(dim=1)
    med = diag.median(dim=1).values
    ratio = mx / med.clamp_min(1e-300)
    return torch.where(med > 0, ratio, torch.where(mx > 0, torch.full_like(mx, float("inf")), torch.ones_like(mx)))


def _solve_chunk(
    w: torch.Tensor, h: torch.Tensor, dq: torch.Tensor, cfg: SolverConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPTQ for a chunk. Returns ``(codes [Ec,R,C] uint8, loss [Ec], ok [Ec], code [Ec])``."""
    ec, rows, cols = w.shape
    dtype = cfg.dtype
    h = h.to(dtype)
    diag0 = torch.diagonal(h, dim1=1, dim2=2)
    if cfg.act_order:
        perm = torch.argsort(diag0, dim=1, descending=True, stable=True)
    else:
        perm = torch.arange(cols, device=w.device).expand(ec, cols).contiguous()
    hp = torch.gather(h, 1, perm[:, :, None].expand(ec, cols, cols))
    hp = torch.gather(hp, 2, perm[:, None, :].expand(ec, cols, cols))
    upper, ok, code = _damped_inverse_factor(hp, cfg.percdamp)
    wc = torch.gather(w.to(dtype), 2, perm[:, None, :].expand(ec, rows, cols))
    # per-column dequantization scale in solve order [Ec, R, C]
    s_nat = dq.unsqueeze(-1).expand(ec, rows, cols // GROUP, GROUP).reshape(ec, rows, cols)
    s_perm = torch.gather(s_nat, 2, perm[:, None, :].expand(ec, rows, cols))
    out = torch.zeros(ec, rows, cols, dtype=torch.uint8, device=w.device)
    loss = torch.zeros(ec, dtype=dtype, device=w.device)
    bs = max(GROUP, cfg.blocksize - cfg.blocksize % GROUP)
    for i1 in range(0, cols, bs):
        i2 = min(i1 + bs, cols)
        w1 = wc[:, :, i1:i2].clone()
        err1 = torch.zeros_like(w1)
        hinv1 = upper[:, i1:i2, i1:i2]
        for i in range(i2 - i1):
            s = s_perm[:, :, i1 + i : i1 + i + 1]  # [Ec, R, 1] float32
            wi = w1[:, :, i]
            x = wi.to(torch.float32).unsqueeze(-1)
            positive = s > 0
            out_scale = torch.where(positive, 1.0 / torch.where(positive, s, torch.ones_like(s)), torch.zeros_like(s))
            codes_i = nvfp4.encode_fp4((x * out_scale).clamp(-FP4_MAX, FP4_MAX))
            qi = (nvfp4.decode_fp4(codes_i) * s).squeeze(-1).to(dtype)
            out[:, :, i1 + i] = codes_i.squeeze(-1)
            d = hinv1[:, i, i].unsqueeze(-1)
            err = (wi - qi) / d
            loss += (err**2).sum(dim=1) / 2.0
            w1[:, :, i:] -= err.unsqueeze(-1) * hinv1[:, i, i:].unsqueeze(1)
            err1[:, :, i] = err
        wc[:, :, i2:] -= torch.bmm(err1, upper[:, i1:i2, i2:])
    codes = torch.empty_like(out)
    codes.scatter_(2, perm[:, None, :].expand(ec, rows, cols), out)
    return codes, loss, ok, code


def solve_experts(
    w: torch.Tensor,
    h: torch.Tensor,
    state: StaticNvfp4,
    cfg: SolverConfig | None = None,
    *,
    active: torch.Tensor | None = None,
) -> SolveResult:
    """Choose the NVFP4 codes of every expert of ``w [E, R, C]`` given its Hessian ``h [E, C, C]``.

    Args:
        w: The original weights (bf16 or float32), on the device where the solve should run.
        h: ``sum_t x_t x_t^T`` per expert (any float dtype; converted to ``cfg.dtype``).
        state: Output of :func:`static_nvfp4` for the SAME ``w`` (same device as ``w``).
        cfg: Solver settings.
        active: Optional ``bool [E]``; False marks experts that must be rounded to nearest (no tokens).

    Returns:
        :class:`SolveResult`; fallback experts carry the RTN codes of ``state``.
    """
    cfg = cfg or SolverConfig()
    e, _rows, cols = w.shape
    if cols % GROUP:
        raise ValueError(f"width {cols} is not a multiple of {GROUP}")
    if h.shape != (e, cols, cols):
        raise ValueError(f"Hessian shape {tuple(h.shape)} does not match weights {tuple(w.shape)}")
    rtn = rtn_codes(state).to(w.device)
    ratio = _diag_ratio(h)
    codes = rtn.clone()
    loss = torch.full((e,), float("nan"), dtype=torch.float64, device=w.device)
    fallback = torch.ones(e, dtype=torch.bool, device=w.device)
    reason = ["inactive"] * e
    act = torch.ones(e, dtype=torch.bool, device=w.device) if active is None else active.to(w.device)
    step = max(1, cfg.expert_chunk)
    for i in range(0, e, step):
        sl = slice(i, min(i + step, e))
        c, ls, ok, rcode = _solve_chunk(w[sl], h[sl].to(w.device), state.dq_scale[sl].to(w.device), cfg)
        for j in range(c.shape[0]):
            k = i + j
            if not bool(act[k]):
                continue
            if not bool(ok[j]):
                reason[k] = _REASONS[int(rcode[j])]
            elif float(ratio[k]) > cfg.diag_ratio_max:
                reason[k] = f"diag_ratio {float(ratio[k]):.3g} > {cfg.diag_ratio_max:.3g}"
            else:
                codes[k] = c[j]
                loss[k] = ls[j].to(torch.float64)
                fallback[k] = False
                reason[k] = ""
    return SolveResult(codes, loss, fallback, reason, ratio.cpu())


def expert_output_error(
    w_up: torch.Tensor, w_down: torch.Tensor, up_q: torch.Tensor, down_q: torch.Tensor, x: torch.Tensor
) -> float:
    """Squared error ``sum ||f_q(x) - f(x)||^2`` of one expert ``f(x) = relu(x W_up^T)^2 W_down^T`` on rows ``x``."""
    xf = x.to(device=w_up.device, dtype=torch.float32)

    def f(up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
        return torch.relu(xf @ up.to(torch.float32).T).square() @ down.to(torch.float32).T

    return float(((f(up_q, down_q) - f(w_up, w_down)).to(torch.float64) ** 2).sum())


@dataclass
class HeldoutDecision:
    """Result of :func:`heldout_guard`: per expert, keep GPTQ (True) or use RTN (False)."""

    keep: torch.Tensor
    err_gptq: list[float]
    err_rtn: list[float]
    reason: list[str]


def heldout_guard(
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    gptq_up: torch.Tensor,
    gptq_down: torch.Tensor,
    rtn_up: torch.Tensor,
    rtn_down: torch.Tensor,
    held_x: list[torch.Tensor],
    *,
    min_rows: int = 4,
) -> HeldoutDecision:
    """Keep GPTQ for an expert only if its output error on HELD-OUT routed rows is not worse than RTN's.

    All weights are ``[E, ., .]`` dequantized float tensors (``w_*`` are the originals); ``held_x[e]`` holds the rows
    of the latent input that were routed to expert ``e`` and kept out of every Hessian. An expert with fewer than
    ``min_rows`` held-out rows cannot be judged and goes to RTN (a calibrated result must be shown to help).
    """
    e = w_up.shape[0]
    keep, eg, er, reason = [], [], [], []
    for k in range(e):
        x = held_x[k]
        if x.shape[0] < min_rows:
            keep.append(False)
            eg.append(float("nan"))
            er.append(float("nan"))
            reason.append("heldout_too_few")
            continue
        g = expert_output_error(w_up[k], w_down[k], gptq_up[k], gptq_down[k], x)
        r = expert_output_error(w_up[k], w_down[k], rtn_up[k], rtn_down[k], x)
        eg.append(g)
        er.append(r)
        worse = g > r
        keep.append(not worse)
        reason.append("heldout_worse" if worse else "")
    return HeldoutDecision(torch.tensor(keep, dtype=torch.bool), eg, er, reason)
