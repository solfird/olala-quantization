"""Per-expert quantization of a fused layer with routing-aware Hessians, and its evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from moegptq.gptq import gptq_quantize
from moegptq.schemes import Int4Scheme, Nvfp4Scheme, rtn_quantize
from moegptq.toy import Tokens, ToyLayer, expert_forward, layer_forward


@dataclass(frozen=True)
class Method:
    """One calibration recipe.

    Attributes:
        name: Label used in tables.
        gptq: False means plain round-to-nearest.
        dynamic: Group scales from the updated weights (True) or the original weights (False).
        hessian: ``'routed'`` (tokens the router sent to the expert), ``'all'`` (every calibration token,
            what llm-compressor's ``calibrate_all_experts`` does) or ``'blend'`` (routed second moment plus
            ``blend_pseudo * dim`` pseudo-tokens of the all-token mean second moment, a shrinkage estimator).
        gate_weighted: Weight each routed token by its squared gate weight ``p^2``.
        percdamp: Damping fraction.
        damp_mode: ``'mean'`` or ``'diag'``.
        min_tokens: An expert with fewer routed calibration tokens than this is quantized by RTN.
        true_sequential: Build the down-projection Hessian from the QUANTIZED up projection's output.
        token_clip: If set, tokens whose input norm exceeds ``token_clip`` times the expert's median norm
            are down-weighted to that norm (their weight becomes ``(clip * median / norm)^2``).
        act_order: Columns processed by descending Hessian diagonal with static natural groups (needs
            ``dynamic=False``); the stored format is unchanged.
        blend_pseudo: Prior strength of ``'blend'`` in units of the Hessian's dimension.
    """

    name: str
    gptq: bool = True
    dynamic: bool = True
    hessian: str = "routed"
    gate_weighted: bool = False
    percdamp: float = 0.01
    damp_mode: str = "mean"
    min_tokens: int = 0
    true_sequential: bool = False
    token_clip: float | None = None
    act_order: bool = False
    blend_pseudo: float = 1.0


@dataclass
class QuantResult:
    """Quantized weights of a layer plus bookkeeping."""

    w_up: torch.Tensor
    w_down: torch.Tensor
    counts: torch.Tensor
    fallback_experts: list[int] = field(default_factory=list)
    info: dict = field(default_factory=dict)


def accumulate_hessians(
    x: torch.Tensor,
    idx: torch.Tensor,
    n_experts: int,
    mode: str,
    token_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-expert ``sum_t w_t x_t x_t^T`` over the tokens the router sent to each expert.

    Args:
        x: ``[N, d]`` expert inputs.
        idx: ``[N, k]`` selected experts.
        n_experts: Number of experts.
        mode: ``'routed'`` or ``'all'`` (every token for every expert).
        token_weight: Optional ``[N]`` weights.

    Returns:
        ``(H [E, d, d] float64, counts [E])`` where ``counts`` is the number of tokens used per expert.
    """
    x = x.double()
    w = torch.ones(x.shape[0], dtype=torch.float64) if token_weight is None else token_weight.double()
    h = torch.zeros(n_experts, x.shape[1], x.shape[1], dtype=torch.float64)
    counts = torch.zeros(n_experts, dtype=torch.long)
    for e in range(n_experts):
        rows = torch.arange(x.shape[0]) if mode == "all" else torch.nonzero((idx == e).any(dim=1)).flatten()
        xe = x[rows]
        h[e] = (xe * w[rows].unsqueeze(1)).T @ xe
        counts[e] = rows.numel()
    return h, counts


def _clip_weights(inputs: torch.Tensor, clip: float | None) -> torch.Tensor:
    """Token weights that cap the influence of tokens whose norm is far above the median."""
    if clip is None or inputs.shape[0] == 0:
        return torch.ones(inputs.shape[0], dtype=torch.float64)
    norm = inputs.norm(dim=1)
    cap = clip * norm.median()
    return torch.clamp(cap / norm.clamp_min(1e-30), max=1.0) ** 2


def quantize_layer(
    layer: ToyLayer,
    cal: Tokens,
    scheme_factory: Callable[[], Nvfp4Scheme | Int4Scheme],
    method: Method,
) -> QuantResult:
    """Quantize every expert's up and down projection with ``method``."""
    cfg = layer.cfg
    e_total = cfg.n_experts
    w_up_q = torch.empty_like(layer.w_up)
    w_down_q = torch.empty_like(layer.w_down)
    counts = torch.zeros(e_total, dtype=torch.long)
    fallback: list[int] = []
    losses = {"up": 0.0, "down": 0.0}
    n_inverse_failures = 0
    n_all = cal.x.shape[0]
    h_all_up = (cal.x.T @ cal.x) / n_all if method.hessian == "blend" else None
    for e in range(e_total):
        if method.hessian == "all":
            rows = torch.arange(cal.x.shape[0])
        else:
            rows = torch.nonzero((cal.idx == e).any(dim=1)).flatten()
        counts[e] = rows.numel()
        floor = 0 if method.hessian == "blend" else 1
        use_gptq = method.gptq and rows.numel() >= max(floor, method.min_tokens)
        if method.gptq and not use_gptq:
            fallback.append(e)
        if not use_gptq:
            w_up_q[e], _ = rtn_quantize(layer.w_up[e], scheme_factory())
            w_down_q[e], _ = rtn_quantize(layer.w_down[e], scheme_factory())
            continue
        xe = cal.x[rows]
        wgt = _clip_weights(xe, method.token_clip)
        if method.gate_weighted:
            slot_p = (cal.p[rows] * (cal.idx[rows] == e)).sum(dim=1)
            wgt = wgt * slot_p**2 if method.hessian == "routed" else wgt
        h_up = (xe * wgt.unsqueeze(1)).T @ xe
        if h_all_up is not None:
            h_up = h_up + method.blend_pseudo * cfg.d * h_all_up
        q_up, info = gptq_quantize(
            layer.w_up[e],
            h_up,
            scheme_factory(),
            percdamp=method.percdamp,
            damp_mode=method.damp_mode,
            dynamic=method.dynamic,
            act_order=method.act_order,
        )
        n_inverse_failures += int(info["fallback"])
        losses["up"] += info["loss"] if not info["fallback"] else 0.0
        w_up_q[e] = q_up
        up_for_hidden = (q_up if method.true_sequential else layer.w_up[e]).double()
        hid = torch.relu(xe @ up_for_hidden.T) ** 2
        wgt_dn = _clip_weights(hid, method.token_clip)
        if method.gate_weighted and method.hessian == "routed":
            wgt_dn = wgt_dn * (cal.p[rows] * (cal.idx[rows] == e)).sum(dim=1) ** 2
        h_dn = (hid * wgt_dn.unsqueeze(1)).T @ hid
        if h_all_up is not None:
            hid_all = torch.relu(cal.x @ up_for_hidden.T) ** 2
            h_dn = h_dn + method.blend_pseudo * cfg.h * (hid_all.T @ hid_all) / n_all
        q_dn, info = gptq_quantize(
            layer.w_down[e],
            h_dn,
            scheme_factory(),
            percdamp=method.percdamp,
            damp_mode=method.damp_mode,
            dynamic=method.dynamic,
            act_order=method.act_order,
        )
        n_inverse_failures += int(info["fallback"])
        losses["down"] += info["loss"] if not info["fallback"] else 0.0
        w_down_q[e] = q_dn
    return QuantResult(
        w_up_q, w_down_q, counts, fallback, {"n_hessian_fallbacks": n_inverse_failures, **losses}
    )


def evaluate(layer: ToyLayer, w_up_q: torch.Tensor, w_down_q: torch.Tensor, test: Tokens) -> dict:
    """Layer-output and per-expert errors of quantized weights on held-out routed tokens.

    ``rel_err`` is ``||y_q - y||^2 / ||y||^2`` of the gate-weighted layer output; ``rel_err_plain`` the same
    on tokens without a spike; ``expert_rel_err[e]`` is the relative squared error of expert ``e``'s own
    output (before gating) on the held-out tokens routed to it (NaN if none).
    """
    y = layer_forward(layer, test.x, test.idx, test.p)
    yq = layer_forward(layer, test.x, test.idx, test.p, w_up_q, w_down_q)
    rel = float(((yq - y) ** 2).sum() / (y**2).sum().clamp_min(1e-300))
    plain = ~test.spike_mask
    if bool(plain.any()) and bool(test.spike_mask.any()):
        rel_plain = float(((yq[plain] - y[plain]) ** 2).sum() / (y[plain] ** 2).sum().clamp_min(1e-300))
    else:
        rel_plain = rel
    per_expert = []
    for e in range(layer.cfg.n_experts):
        rows = torch.nonzero((test.idx == e).any(dim=1)).flatten()
        if rows.numel() == 0:
            per_expert.append(float("nan"))
            continue
        ref = expert_forward(layer.w_up[e], layer.w_down[e], test.x[rows])
        got = expert_forward(w_up_q[e], w_down_q[e], test.x[rows])
        per_expert.append(float(((got - ref) ** 2).sum() / (ref**2).sum().clamp_min(1e-300)))
    return {"rel_err": rel, "rel_err_plain": rel_plain, "expert_rel_err": per_expert}
