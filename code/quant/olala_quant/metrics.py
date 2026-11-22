"""Streaming reconstruction-error statistics shared by the quantizers and the verifier."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class ErrorStats:
    """Reconstruction error of a dequantized tensor against its source.

    Attributes:
        max_abs: Largest absolute element error.
        rel_fro: Frobenius norm of the error divided by the Frobenius norm of the source
            (0.0 when both are zero, infinity when only the source is zero).
        n_elements: Number of elements compared.
    """

    max_abs: float
    rel_fro: float
    n_elements: int

    def as_dict(self) -> dict[str, float | int]:
        """Return a JSON-serializable view."""
        return {"max_abs": self.max_abs, "rel_fro": self.rel_fro, "n_elements": self.n_elements}


@dataclass
class ErrorAccumulator:
    """Accumulate error statistics over chunks of one tensor."""

    max_abs: float = 0.0
    sum_sq_err: float = 0.0
    sum_sq_ref: float = 0.0
    n_elements: int = 0
    _finished: bool = field(default=False, repr=False)

    def update(self, reference: torch.Tensor, reconstructed: torch.Tensor) -> None:
        """Add one chunk (any shape, same shape on both sides)."""
        ref = reference.detach().to(torch.float64)
        err = reconstructed.detach().to(torch.float64) - ref
        if err.numel() == 0:
            return
        self.max_abs = max(self.max_abs, float(err.abs().max()))
        self.sum_sq_err += float((err * err).sum())
        self.sum_sq_ref += float((ref * ref).sum())
        self.n_elements += err.numel()

    def finish(self) -> ErrorStats:
        """Return the final statistics."""
        if self.sum_sq_ref == 0.0:
            rel = 0.0 if self.sum_sq_err == 0.0 else math.inf
        else:
            rel = math.sqrt(self.sum_sq_err / self.sum_sq_ref)
        return ErrorStats(max_abs=self.max_abs, rel_fro=rel, n_elements=self.n_elements)
