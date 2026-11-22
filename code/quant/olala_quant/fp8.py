"""Per-output-channel FP8 (e4m3fn) weight quantization, data free.

The scale formula and the rounding follow vLLM's online PTPC path so that an offline
checkpoint and ``--quantization fp8_per_channel`` agree:

* ``scale = max(amax_row / 448, 1 / (448 * 512))``
  (``vllm/model_executor/layers/quantization/online/fp8.py:86-88``)
* ``q = clamp(w / scale, -448, 448).to(float8_e4m3fn)`` with a true division
  (``online/fp8.py:96-106``).
* Dequantization is ``q.float() * scale``.

Works on a 2D ``[out, in]`` weight (scale ``[out, 1]``) and on a fused 3D ``[E, out, in]``
tensor (scale ``[E, out, 1]``), processed in chunks of experts so a 256 x 768 x 384 tensor
never needs more than a few tens of megabytes of float32 scratch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from olala_quant.metrics import ErrorAccumulator, ErrorStats

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0
MIN_SCALE = 1.0 / (FP8_MAX * 512.0)
# Elements per float32 scratch chunk when slicing a 2D weight into row blocks (~64 MB).
_CHUNK_ELEMS = 16 * 1024 * 1024


@dataclass(frozen=True)
class Fp8ScaleStats:
    """Range of the per-row scales of one tensor."""

    n_rows: int
    n_zero_rows: int
    max_scale: float
    min_scale: float

    def as_dict(self) -> dict[str, float | int]:
        """Return a JSON-serializable view."""
        return {
            "n_rows": self.n_rows,
            "n_zero_rows": self.n_zero_rows,
            "max_scale": self.max_scale,
            "min_scale": self.min_scale,
        }


@dataclass(frozen=True)
class Fp8Result:
    """Output of :func:`quantize_fp8`.

    Attributes:
        weight: ``float8_e4m3fn`` tensor with the shape of the input.
        scale: ``float32`` per-row scales, shape ``input.shape[:-1] + (1,)``.
        error: Reconstruction error against the (float32 view of the) input.
        scale_stats: Range of the scales and number of all-zero rows.
    """

    weight: torch.Tensor
    scale: torch.Tensor
    error: ErrorStats
    scale_stats: Fp8ScaleStats


def _quantize_rows(
    w: torch.Tensor, out_w: torch.Tensor, out_s: torch.Tensor, acc: ErrorAccumulator
) -> tuple[float, float, int]:
    """Quantize ``w`` ([..., rows, cols], float32) into the given output views."""
    amax = w.abs().amax(dim=-1, keepdim=True)
    scale = (amax / FP8_MAX).clamp_min(MIN_SCALE)
    q = (w / scale).clamp_(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    out_w.copy_(q)
    out_s.copy_(scale)
    acc.update(w, q.to(torch.float32) * scale)
    return float(scale.max()), float(scale.min()), int((amax == 0).sum())


def quantize_fp8(weight: torch.Tensor, *, expert_chunk: int = 16) -> Fp8Result:
    """Quantize a 2D or 3D weight to FP8 e4m3 with one fp32 scale per output row.

    Args:
        weight: ``[out, in]`` or ``[E, out, in]`` tensor in any float dtype.
        expert_chunk: Number of experts (leading dim) processed at once for 3D input; for
            2D input the weight is cut into row blocks of about 64 MB of float32.

    Returns:
        The quantized weight, scales and error statistics.

    Raises:
        ValueError: If the rank is not 2 or 3 or the tensor holds NaN or infinity.
    """
    if weight.ndim not in (2, 3):
        raise ValueError(f"quantize_fp8 needs a rank 2 or 3 tensor, got rank {weight.ndim}")
    out_w = torch.empty(weight.shape, dtype=FP8_DTYPE, device=weight.device)
    out_s = torch.empty((*weight.shape[:-1], 1), dtype=torch.float32, device=weight.device)
    acc = ErrorAccumulator()
    max_s, min_s, zero_rows = 0.0, float("inf"), 0
    if weight.ndim == 3:
        step = max(1, expert_chunk)
        slices = [slice(i, i + step) for i in range(0, weight.shape[0], step)]
    else:
        rows = max(1, _CHUNK_ELEMS // max(1, weight.shape[-1]))
        slices = [slice(i, i + rows) for i in range(0, weight.shape[0], rows)]
    for sl in slices:
        w = weight[sl].to(torch.float32)
        if not torch.isfinite(w).all():
            raise ValueError("quantize_fp8: source tensor holds non-finite values")
        hi, lo, zeros = _quantize_rows(w, out_w[sl], out_s[sl], acc)
        max_s, min_s, zero_rows = max(max_s, hi), min(min_s, lo), zero_rows + zeros
    stats = Fp8ScaleStats(n_rows=out_s.numel(), n_zero_rows=zero_rows, max_scale=max_s, min_scale=min_s)
    return Fp8Result(weight=out_w, scale=out_s, error=acc.finish(), scale_stats=stats)


def dequantize_fp8(
    weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Dequantize ``weight`` (e4m3) with per-row ``scale`` of shape ``[..., rows, 1]``."""
    return (weight.to(torch.float32) * scale.to(torch.float32)).to(dtype)
