"""Quantization grids: NVFP4 (group 16, e4m3 block scale, fp32 global scale) and symmetric INT4.

The NVFP4 arithmetic repeats the float32 operation order of ``olala_quant.nvfp4._quantize_chunk`` (and
its rounding thresholds through ``encode_fp4``), so round-to-nearest here equals the data-free quantizer
that produced the checkpoint already served on the pod. A scheme exposes three operations:

* ``prepare(w)``: per-tensor state (the NVFP4 global scale of one expert; nothing for INT4);
* ``scales(wg, state)``: the dequantization scale of every row of a group of columns ``wg [rows, gs]``;
* ``round(wg, s)``: round-to-nearest of ``wg`` on the grid with scale ``s [rows, 1]``.
"""

from __future__ import annotations

import torch
from olala_quant.nvfp4 import E4M3_MAX, FP4_MAX, decode_fp4, encode_fp4

INT4_MAX = 7.0  # symmetric range used here: scale = amax / 7, codes clamped to [-8, 7]


class Nvfp4Scheme:
    """NVFP4: e2m1 values, one e4m3 scale per 16 input columns, one fp32 global scale per tensor."""

    name = "nvfp4_g16"
    group_size = 16

    def prepare(self, w: torch.Tensor) -> torch.Tensor:
        """Return the encode global scale ``448 * 6 / amax`` of the whole tensor (1.0 if all zero)."""
        amax = w.to(torch.float32).abs().amax()
        top = torch.tensor(E4M3_MAX * FP4_MAX, dtype=torch.float32)
        return torch.where(amax > 0, top / torch.where(amax > 0, amax, torch.ones_like(amax)), torch.ones(()))

    def scales(self, wg: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Dequantization scale ``e4m3(g * amax / 6) / g`` per row; 0 where the e4m3 scale flushes to 0."""
        g = state.to(torch.float32)
        vec_max = wg.to(torch.float32).abs().amax(dim=-1, keepdim=True)
        scale = g * (vec_max * (1.0 / FP4_MAX))
        scale_q = scale.clamp(min=0.0, max=E4M3_MAX).to(torch.float8_e4m3fn).to(torch.float32)
        return scale_q * (1.0 / g)

    def round(self, wg: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Round to the e2m1 grid scaled by ``s`` (rows whose scale is 0 become 0)."""
        x = wg.to(torch.float32)
        positive = s > 0
        out_scale = torch.where(
            positive, 1.0 / torch.where(positive, s, torch.ones_like(s)), torch.zeros_like(s)
        )
        codes = encode_fp4((x * out_scale).clamp(-FP4_MAX, FP4_MAX))
        return (decode_fp4(codes) * s).to(wg.dtype)


class Int4Scheme:
    """Symmetric INT4 with one scale per ``group_size`` input columns (a short row is one group)."""

    name = "int4_g128"

    def __init__(self, group_size: int = 128) -> None:
        """Set the group size (the layer's input width is used instead when it is smaller)."""
        self.group_size = group_size

    def prepare(self, w: torch.Tensor) -> None:
        """INT4 has no per-tensor state."""
        return None

    def scales(self, wg: torch.Tensor, state: None) -> torch.Tensor:
        """Scale ``amax / 7`` per row of the group (1.0 for an all-zero row)."""
        amax = wg.to(torch.float32).abs().amax(dim=-1, keepdim=True)
        return torch.where(amax > 0, amax / INT4_MAX, torch.ones_like(amax))

    def round(self, wg: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Round to the integer grid ``-8..7`` times ``s``."""
        x = wg.to(torch.float32)
        return (torch.round(x / s).clamp(-8, 7) * s).to(wg.dtype)


def effective_group(scheme: object, cols: int) -> int:
    """Group size actually used for a matrix with ``cols`` input columns."""
    gs = min(scheme.group_size, cols)  # type: ignore[attr-defined]
    if cols % gs:
        raise ValueError(f"input width {cols} is not a multiple of the group size {gs}")
    return gs


def rtn_quantize(w: torch.Tensor, scheme: Nvfp4Scheme | Int4Scheme) -> tuple[torch.Tensor, torch.Tensor]:
    """Round-to-nearest of a 2D weight ``[rows, cols]``.

    Returns:
        The dequantized weight (same dtype as ``w``) and the scales ``[rows, cols // group]``.
    """
    _rows, cols = w.shape
    gs = effective_group(scheme, cols)
    state = scheme.prepare(w)
    q = torch.empty_like(w)
    scales = []
    for c in range(0, cols, gs):
        wg = w[:, c : c + gs]
        s = scheme.scales(wg, state)
        q[:, c : c + gs] = scheme.round(wg, s)
        scales.append(s)
    return q, torch.cat(scales, dim=1)
