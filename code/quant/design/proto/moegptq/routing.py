"""Router of the Olala MoE (``modeling_olala.py:1518-1527``): sigmoid scores, bias for selection only."""

from __future__ import annotations

import torch


def route(
    logits: torch.Tensor, expert_bias: torch.Tensor, top_k: int, scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select ``top_k`` experts per token and return their gate weights.

    The sigmoid score is used for the gate weight; ``expert_bias`` only changes which experts are
    selected. The selected scores are renormalised to sum to one and multiplied by ``scale``.

    Args:
        logits: ``[N, E]`` router logits.
        expert_bias: ``[E]`` selection bias.
        top_k: Experts per token.
        scale: Routed scaling factor (2.45 in Olala).

    Returns:
        ``(idx [N, k], p [N, k])`` with ``p`` rows summing to ``scale``.
    """
    scores = torch.sigmoid(logits.float())
    _, idx = torch.topk(scores + expert_bias.float(), k=top_k, dim=1)
    picked = torch.gather(scores, 1, idx)
    probs = picked / (picked.sum(dim=-1, keepdim=True) + 1e-20) if top_k > 1 else picked
    return idx, probs * scale
