"""Ablations of the spiking branches and the next-token metrics that say how much the model depends on them.

For one ablation (a :class:`probe_hooks.AblationSpec`) every prompt is run again with the edit in place and compared
with the unedited run on the SAME tokens: the Kullback-Leibler divergence ``KL(p_base || p_ablated)`` of the next-token
distributions (nats, at every position and at the last position, which is where generation starts), the top-1
agreement, and the change of the teacher-forced negative log-likelihood of the prompt's own next token. Logits are
computed from the final hidden states in chunks so that a 4,096-token pass never holds more than one chunk of the
151,936-way vocabulary at a time.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

import probe_hooks as ph
import probe_prompts as pp

FLAG_KINDS = ("mixer_out", "moe_out", "routed_out", "shared_out")


@dataclass
class PromptMetrics:
    """Sums over one prompt (so that prompts can be pooled with position weights)."""

    n_pos: int
    n_finite: int
    n_nonfinite: int
    kl_sum: float
    kl_max: float
    kl_last: float  # nan when the last position is not finite
    top1_sum: float
    n_nll: int
    nll_base_sum: float
    nll_abl_sum: float


@torch.no_grad()
def prompt_metrics(
    lm_head: Callable[[torch.Tensor], torch.Tensor],
    base_hidden: torch.Tensor,
    abl_hidden: torch.Tensor,
    tokens: torch.Tensor,
    chunk: int = 256,
) -> PromptMetrics:
    """Compare the next-token distributions of two final hidden states ``[1, L, D]`` of the same ``tokens`` ``[L]``."""
    length = base_hidden.shape[1]
    dtype = getattr(getattr(lm_head, "weight", None), "dtype", base_hidden.dtype)
    kl_sum = top1 = nll_b = nll_a = 0.0
    kl_max = 0.0
    kl_last = math.nan
    n_fin = n_nll = 0
    for c0 in range(0, length, chunk):
        c1 = min(c0 + chunk, length)
        lb = lm_head(base_hidden[0, c0:c1].to(dtype)).float()
        la = lm_head(abl_hidden[0, c0:c1].to(dtype)).float()
        ok = torch.isfinite(lb).all(-1) & torch.isfinite(la).all(-1)
        lb, la = lb[ok], la[ok]
        pos = torch.arange(c0, c1, device=ok.device)[ok]
        n_fin += int(ok.sum())
        if not pos.numel():
            continue
        logp_b, logp_a = F.log_softmax(lb, -1), F.log_softmax(la, -1)
        kl = (logp_b.exp() * (logp_b - logp_a)).sum(-1)
        kl_sum += float(kl.sum())
        kl_max = max(kl_max, float(kl.max()))
        top1 += float((lb.argmax(-1) == la.argmax(-1)).sum())
        if int(pos[-1]) == length - 1:
            kl_last = float(kl[-1])
        lab = pos < length - 1
        if bool(lab.any()):
            target = tokens[pos[lab] + 1].to(lb.device)
            nll_b += float(F.nll_loss(logp_b[lab], target, reduction="sum"))
            nll_a += float(F.nll_loss(logp_a[lab], target, reduction="sum"))
            n_nll += int(lab.sum())
    return PromptMetrics(length, n_fin, length - n_fin, kl_sum, kl_max, kl_last, top1, n_nll, nll_b, nll_a)


class MetricAcc:
    """Pools :class:`PromptMetrics`."""

    def __init__(self) -> None:
        """Create an empty pool."""
        self.rows: list[PromptMetrics] = []

    def add(self, m: PromptMetrics) -> None:
        """Add one prompt."""
        self.rows.append(m)

    def result(self) -> dict[str, Any]:
        """Pooled metrics (means weighted by the number of finite positions), JSON-ready."""
        n_fin = sum(r.n_finite for r in self.rows)
        n_nll = sum(r.n_nll for r in self.rows)
        lasts = [r.kl_last for r in self.rows if not math.isnan(r.kl_last)]
        nb = sum(r.nll_base_sum for r in self.rows) / n_nll if n_nll else None
        na = sum(r.nll_abl_sum for r in self.rows) / n_nll if n_nll else None
        return {
            "n_prompts": len(self.rows),
            "n_positions": sum(r.n_pos for r in self.rows),
            "n_nonfinite_positions": sum(r.n_nonfinite for r in self.rows),
            "kl_mean_all_positions": sum(r.kl_sum for r in self.rows) / n_fin if n_fin else None,
            "kl_max_position": max((r.kl_max for r in self.rows), default=None),
            "kl_last_position_mean": sum(lasts) / len(lasts) if lasts else None,
            "top1_agreement": sum(r.top1_sum for r in self.rows) / n_fin if n_fin else None,
            "nll_base_mean": nb,
            "nll_ablated_mean": na,
            "delta_nll_mean": (na - nb) if (na is not None and nb is not None) else None,
        }


def run_ablations(
    model: Any,
    prompts: Sequence[pp.Prompt],
    specs: Sequence[ph.AblationSpec],
    device: str,
    chunk: int = 256,
    log: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Run every prompt once unedited, then once per spec, and return one metrics row per spec (in order)."""
    ids = [torch.as_tensor(p.tokens, device=device).unsqueeze(0) for p in prompts]
    with torch.no_grad():
        base = [model.model(input_ids=i).last_hidden_state.detach() for i in ids]
        rows = []
        for spec in specs:
            handles = ph.install_ablation(model, spec)
            try:
                acc = MetricAcc()
                for i, b in zip(ids, base, strict=True):
                    h = model.model(input_ids=i).last_hidden_state
                    acc.add(prompt_metrics(model.lm_head, b, h, i[0], chunk))
            finally:
                for h_ in handles:
                    h_.remove()
            rows.append(
                {
                    "label": spec.label(),
                    "layer": spec.layer,
                    "branch": spec.branch,
                    "mode": spec.mode,
                    "head": spec.head,
                    "metrics": acc.result(),
                }
            )
            if log:
                log(f"ablation {spec.label()}: KL {rows[-1]['metrics']['kl_mean_all_positions']}")
    return rows


def select_targets(probe: dict[str, Any], threshold: float, max_branches: int) -> list[ph.AblationSpec]:
    """Choose what to ablate from a probe result.

    Every branch site (``mixer_out``, ``moe_out``, ``routed_out``, ``shared_out``) whose maximum exceeds ``threshold``
    is flagged; the ``max_branches`` largest are kept. Each gets a zero and a clamp-at-448 ablation, the flagged Mamba
    mixers also a zero ablation of their largest head, one control (zero ablation of the lowest-numbered UNflagged
    branch of the same kind) is added per flagged kind, and the list starts with a no-op control that must reproduce the
    baseline exactly. Empty when nothing is flagged.
    """
    rows = {r["layer"]: r for r in probe["layers"]}
    flagged = sorted(
        (
            (r["sites"][k]["max_abs"], r["layer"], k)
            for r in rows.values()
            for k in FLAG_KINDS
            if k in r["sites"] and r["sites"][k]["max_abs"] > threshold
        ),
        reverse=True,
    )[:max_branches]
    if not flagged:
        return []
    specs: list[ph.AblationSpec] = [ph.AblationSpec(flagged[0][1], flagged[0][2], "noop")]
    for _, layer, kind in flagged:
        specs.append(ph.AblationSpec(layer, kind, "zero"))
        specs.append(ph.AblationSpec(layer, kind, "clamp448"))
        row = rows[layer]
        if kind == "mixer_out" and row.get("heads"):
            specs.append(ph.AblationSpec(layer, "head", "zero", head=int(row["heads"]["top_head"])))
    flagged_set = {(layer, kind) for _, layer, kind in flagged}
    for kind in dict.fromkeys(k for _, _, k in flagged):
        for layer in sorted(rows):
            r = rows[layer]
            eligible = kind in r["sites"] and (kind != "mixer_out" or r["type"] == "M")
            if eligible and (layer, kind) not in flagged_set and r["sites"][kind]["max_abs"] <= threshold:
                specs.append(ph.AblationSpec(layer, kind, "zero"))
                break
    return list(dict.fromkeys(specs))
