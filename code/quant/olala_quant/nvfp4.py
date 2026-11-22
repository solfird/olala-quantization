"""NVFP4 weight quantization (e2m1 values, e4m3 block scales, fp32 global scale), data free.

Format contract, taken from the OVH vLLM fork (commit ebbadef) and the compressed-tensors
``nvfp4-pack-quantized`` checkpoint layout that its ``CompressedTensorsW4A4Fp4`` scheme loads:

* Values are e2m1 (``0, .5, 1, 1.5, 2, 3, 4, 6`` and their negatives), two per ``uint8`` with the
  EVEN element in the LOW nibble (``nvfp4_emulation_utils.py``: ``break_fp4_bytes`` ``:546-563``, Triton
  decode ``:120-122``, value table ``:23-25``).
* One e4m3 scale per 16 consecutive input elements (``compressed_tensors_w4a4_nvfp4.py:72-84``).
* One fp32 global scale per tensor. compressed-tensors stores the ENCODE scale
  ``448 * 6 / amax`` (the vLLM loader inverts it: ``compressed_tensors_w4a4_nvfp4.py:110-112``,
  MoE ``compressed_tensors_moe_w4a4_nvfp4.py:248,252``), so that
  ``block_scale_e4m3 = amax_block / 6 * global_scale`` fits in e4m3 (max 448) and
  ``w ~ fp4 * (block_scale_e4m3 * (1 / global_scale))``.
* Rounding to e2m1 uses the thresholds of ``cast_to_fp4`` (``nvfp4_emulation_utils.py:695-706``);
  ties go to the even code. The block scale follows ``ref_nvfp4_quant`` (``:709-722``):
  ``scale = global * (vec_max * (1/6))`` clamped to +-448 and cast to e4m3.

For a fused 3D expert tensor ``[E, out, in]`` one global scale is computed per expert (shape
``[E]``), which is what the vLLM MoE parameters ``w13_weight_global_scale`` / ``w2_weight_global_scale``
hold (``compressed_tensors_moe_w4a4_nvfp4.py:149-168``).

MSE-optimal block scales (``scale_search=True``, PROTOCOL.md amendment 25, off by default)
---------------------------------------------------------------------------------------------
The plain scale of a group of 16 is derived from its amax, so the largest weight of the group lands exactly on
6 (the top of the e2m1 grid) and the group's other weights are rounded relative to it. That is not the scale
with the smallest reconstruction error: a slightly smaller scale clips the largest weight a little and gives
the rest of the group a finer grid. The search tries, for every group, the amax-derived scale multiplied by
each factor of :data:`SCALE_SEARCH_FACTORS` (1.00, 0.95, 0.90, 0.85, 0.80, 0.75), rounds each candidate to
e4m3 EXACTLY as the plain scale is (``scale = global * (vec_max / 6) * factor``, clamped to [0, 448], then
cast), quantizes and dequantizes the group with it, and keeps the candidate with the smallest squared error of
the dequantized group against the source. Ties go to the LARGEST factor (candidates are tried from 1.00 down
and a later one must be strictly better), so a group that no factor improves keeps the plain scale bit for
bit, and no group can end with a larger squared error than the plain scale gives it (1.00 is one of the
candidates). Data free: only the weights are read.

How it interacts with the global scale: not at all. The per-tensor (2D) or per-expert (3D) global scale is
still ``448 * 6 / amax`` of the ORIGINAL tensor (expert), computed before any search, so the format, the
dequantization order (``fp4 * (block_scale * (1 / global_scale))``) and the vLLM loader are unchanged and a
checkpoint built with the search loads like any other. Every candidate is at most the plain scale (factors are
at most 1), so none can overflow e4m3: the block that holds the tensor's amax is stored as 448 (factor 1.00)
or a little below (a smaller factor); it is no longer always exactly 448, which is why
:mod:`olala_quant.verify` has a dedicated mode for it.

Memory: the search never holds more than :data:`_SEARCH_CHUNK_ELEMS` = 4,194,304 weights (a ``[rows, in]``
slice of the current expert chunk, at least one row) in its float32 temporaries, 16.8 MB each, whatever
``expert_chunk`` is. A ``[256, 768, 384]`` routed tensor (75.5 million weights) is therefore processed as 16
expert chunks of 16 experts (the converter's default for NVFP4; 18.9 MB as float32, two row slices per chunk).
Measured on a Mac CPU for one such tensor in bf16: peak resident memory 1.25 GB without the search and 1.31 GB
with it (the 151 MB input and the three outputs dominate; the search adds about 60 MB), 0.3 s against 1.3 s:
six candidate passes cost about four times the plain quantization. A 32 GB GPU is nowhere near its limit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from olala_quant.metrics import ErrorAccumulator, ErrorStats

GROUP_SIZE = 16
FP4_MAX = 6.0
E4M3_MAX = 448.0
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E4M3_MIN_NORMAL = 2.0**-6
_RECIP_FP4_MAX = 1.0 / FP4_MAX
# The max block of a tensor lands exactly on 448 up to float32 rounding (about 1e-7 relative);
# only a scale above this bound counts as a real overflow (a clip).
_OVERFLOW_BOUND = E4M3_MAX * (1.0 + 1e-6)
_CHUNK_ELEMS = 8 * 1024 * 1024
# MSE-optimal block scales (amendment 25): factors applied to the amax-derived scale, largest first (the
# tie-break relies on the order), and the version recorded in quantization_info.json (bump it when the method
# changes).
SCALE_SEARCH_FACTORS: tuple[float, ...] = (1.00, 0.95, 0.90, 0.85, 0.80, 0.75)
SCALE_SEARCH_VERSION = 1
_SEARCH_CHUNK_ELEMS = 4 * 1024 * 1024  # weights per candidate pass of the search (see the module docstring)


@dataclass(frozen=True)
class Nvfp4ScaleStats:
    """Range and clipping counters of the e4m3 block scales of one tensor.

    Attributes:
        n_blocks: Number of 16-element blocks.
        n_zero_blocks: Blocks whose 16 source values are all zero.
        n_overflow: Blocks whose scale exceeded 448 (beyond float32 rounding noise) before the clamp,
            i.e. blocks that clip. Zero by construction when the global scale comes from the exact amax.
        n_flushed: Non-zero blocks whose e4m3 scale rounded to zero (their values are lost).
        n_subnormal: Blocks with a non-zero e4m3 scale below the e4m3 normal minimum 2^-6.
        max_block_scale: Largest stored e4m3 block scale.
        min_nonzero_block_scale: Smallest non-zero stored e4m3 block scale (0.0 if none).
    """

    n_blocks: int
    n_zero_blocks: int
    n_overflow: int
    n_flushed: int
    n_subnormal: int
    max_block_scale: float
    min_nonzero_block_scale: float

    def merge(self, other: Nvfp4ScaleStats) -> Nvfp4ScaleStats:
        """Combine the statistics of two chunks of the same tensor."""
        mins = [m for m in (self.min_nonzero_block_scale, other.min_nonzero_block_scale) if m > 0]
        return Nvfp4ScaleStats(
            n_blocks=self.n_blocks + other.n_blocks,
            n_zero_blocks=self.n_zero_blocks + other.n_zero_blocks,
            n_overflow=self.n_overflow + other.n_overflow,
            n_flushed=self.n_flushed + other.n_flushed,
            n_subnormal=self.n_subnormal + other.n_subnormal,
            max_block_scale=max(self.max_block_scale, other.max_block_scale),
            min_nonzero_block_scale=min(mins) if mins else 0.0,
        )

    def as_dict(self) -> dict[str, float | int]:
        """Return a JSON-serializable view."""
        return {
            "n_blocks": self.n_blocks,
            "n_zero_blocks": self.n_zero_blocks,
            "n_overflow": self.n_overflow,
            "n_flushed": self.n_flushed,
            "n_subnormal": self.n_subnormal,
            "max_block_scale": self.max_block_scale,
            "min_nonzero_block_scale": self.min_nonzero_block_scale,
        }


_EMPTY_STATS = Nvfp4ScaleStats(0, 0, 0, 0, 0, 0.0, 0.0)


@dataclass(frozen=True)
class Nvfp4Result:
    """Output of :func:`quantize_nvfp4`.

    Attributes:
        packed: ``uint8`` ``[..., out, in // 2]``, even element in the low nibble.
        block_scale: ``float8_e4m3fn`` ``[..., out, in // 16]``.
        global_scale: ``float32`` encode scale ``448 * 6 / amax``; shape ``[1]`` for a 2D weight
            and ``[E]`` (one per expert) for a 3D weight.
        error: Reconstruction error against the float32 view of the input.
        scale_stats: Block-scale range and clipping counters.
    """

    packed: torch.Tensor
    block_scale: torch.Tensor
    global_scale: torch.Tensor
    error: ErrorStats
    scale_stats: Nvfp4ScaleStats


def encode_fp4(x: torch.Tensor) -> torch.Tensor:
    """Round float values (already divided by their scale) to e2m1 nibbles.

    Magnitudes above 6 saturate. The nibble is ``sign << 3 | magnitude_code``; the sign bit is
    set only for non-zero negative results so that zero has a single encoding.
    """
    a = x.abs()
    code = (
        (a > 0.25).to(torch.uint8)
        + (a >= 0.75).to(torch.uint8)
        + (a > 1.25).to(torch.uint8)
        + (a >= 1.75).to(torch.uint8)
        + (a > 2.5).to(torch.uint8)
        + (a >= 3.5).to(torch.uint8)
        + (a > 5.0).to(torch.uint8)
    )
    negative = ((x < 0) & (code != 0)).to(torch.uint8)
    return code | (negative << 3)


def decode_fp4(codes: torch.Tensor) -> torch.Tensor:
    """Decode e2m1 nibbles (``uint8``, one per element) to float32."""
    table = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=codes.device)
    magnitude = table[(codes & 0x7).to(torch.long)]
    return torch.where((codes & 0x8) != 0, -magnitude, magnitude)


def pack_fp4(codes: torch.Tensor) -> torch.Tensor:
    """Pack nibbles two per byte along the last dim; the even element goes in the low nibble."""
    if codes.shape[-1] % 2:
        raise ValueError("pack_fp4 needs an even last dimension")
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_fp4`: ``[..., n]`` bytes to ``[..., 2n]`` nibbles."""
    return torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(*packed.shape[:-1], -1)


def _encode_with_scale(
    blocks: torch.Tensor, scale_e4m3: torch.Tensor, g4: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize ``blocks`` ``[Ec, rows, groups, 16]`` with the given e4m3 scales ``[Ec, rows, groups, 1]``.

    Returns ``(codes, dequantized, scale_q, positive)``: the e2m1 nibbles, the dequantized weights (the
    ``fp4 * (scale * (1 / global))`` of the loader), the scales as float32 and the mask of non-zero scales.
    """
    scale_q = scale_e4m3.to(torch.float32)
    dq_scale = scale_q * (1.0 / g4)
    positive = scale_q > 0
    out_scale = torch.where(positive, 1.0 / torch.where(positive, dq_scale, torch.ones_like(dq_scale)), 0.0)
    codes = encode_fp4((blocks * out_scale).clamp_(-FP4_MAX, FP4_MAX))
    return codes, decode_fp4(codes) * dq_scale, scale_q, positive


def _candidate_scale(base: torch.Tensor, factor: float) -> torch.Tensor:
    """The e4m3 scale of candidate ``factor``: ``base`` times it, rounded like the plain scale."""
    return (base * factor).clamp(min=0.0, max=E4M3_MAX).to(torch.float8_e4m3fn)


def _group_error(blocks: torch.Tensor, scale_e4m3: torch.Tensor, g4: torch.Tensor) -> torch.Tensor:
    """Squared error ``[Ec, rows, groups, 1]`` of every group quantized with ``scale_e4m3``."""
    _, deq, _, _ = _encode_with_scale(blocks, scale_e4m3, g4)
    diff = deq - blocks
    return (diff * diff).sum(dim=-1, keepdim=True)


def _search_scales(blocks: torch.Tensor, g4: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """The e4m3 scale of every group with the smallest squared error among the candidates of the search.

    ``base`` is the amax-derived scale ``global * (vec_max / 6)`` of every group. Candidates are tried from
    the largest factor down and replace the best so far only when their error is STRICTLY smaller, so ties
    keep the larger factor. Rows are processed in slices of at most :data:`_SEARCH_CHUNK_ELEMS` weights.
    """
    ec, rows, groups, _ = blocks.shape
    step = max(1, _SEARCH_CHUNK_ELEMS // max(1, ec * groups * GROUP_SIZE))
    best = torch.empty((ec, rows, groups, 1), dtype=torch.float8_e4m3fn, device=blocks.device)
    for r0 in range(0, rows, step):
        sl = slice(r0, r0 + step)
        chunk, b = blocks[:, sl], base[:, sl]
        best_scale = _candidate_scale(b, SCALE_SEARCH_FACTORS[0])
        best_err = _group_error(chunk, best_scale, g4)
        for factor in SCALE_SEARCH_FACTORS[1:]:
            scale = _candidate_scale(b, factor)
            err = _group_error(chunk, scale, g4)
            better = err < best_err
            best_scale = torch.where(better, scale.view(torch.uint8), best_scale.view(torch.uint8)).view(
                torch.float8_e4m3fn
            )
            best_err = torch.where(better, err, best_err)
        best[:, sl] = best_scale
    return best


def _quantize_chunk(
    w: torch.Tensor, g: torch.Tensor, acc: ErrorAccumulator, scale_search: bool = False
) -> tuple[torch.Tensor, torch.Tensor, Nvfp4ScaleStats]:
    """Quantize ``w`` ``[Ec, rows, cols]`` (float32) with encode scales ``g`` ``[Ec]``."""
    ec, rows, cols = w.shape
    blocks = w.reshape(ec, rows, cols // GROUP_SIZE, GROUP_SIZE)
    g4 = g.reshape(ec, 1, 1, 1)
    vec_max = blocks.abs().amax(dim=-1, keepdim=True)
    scale = g4 * (vec_max * _RECIP_FP4_MAX)
    overflow = scale > _OVERFLOW_BOUND
    if scale_search:
        scale_e4m3 = _search_scales(blocks, g4, scale)
    else:
        scale_e4m3 = scale.clamp(min=0.0, max=E4M3_MAX).to(torch.float8_e4m3fn)
    codes, deq, scale_q, positive = _encode_with_scale(blocks, scale_e4m3, g4)
    acc.update(w, deq.reshape(ec, rows, cols))
    nonzero = scale_q[positive]
    stats = Nvfp4ScaleStats(
        n_blocks=int(vec_max.numel()),
        n_zero_blocks=int((vec_max == 0).sum()),
        n_overflow=int(overflow.sum()),
        n_flushed=int(((vec_max > 0) & ~positive).sum()),
        n_subnormal=int(((scale_q > 0) & (scale_q < E4M3_MIN_NORMAL)).sum()),
        max_block_scale=float(scale_q.max()) if scale_q.numel() else 0.0,
        min_nonzero_block_scale=float(nonzero.min()) if nonzero.numel() else 0.0,
    )
    return pack_fp4(codes.reshape(ec, rows, cols)), scale_e4m3.reshape(ec, rows, -1), stats


def _global_scale(amax: torch.Tensor) -> torch.Tensor:
    """Encode scale ``448 * 6 / amax`` per entry; 1.0 for an all-zero tensor."""
    top = torch.tensor(E4M3_MAX * FP4_MAX, dtype=torch.float32, device=amax.device)
    g = torch.where(amax > 0, top / torch.where(amax > 0, amax, torch.ones_like(amax)), 1.0)
    if not torch.isfinite(g).all():
        raise ValueError("quantize_nvfp4: tensor amax is too small for a finite global scale")
    return g


def quantize_nvfp4(weight: torch.Tensor, *, expert_chunk: int = 8, scale_search: bool = False) -> Nvfp4Result:
    """Quantize a 2D or 3D weight to NVFP4.

    Args:
        weight: ``[out, in]`` or ``[E, out, in]`` (``in`` a multiple of 16), any float dtype.
        expert_chunk: Experts processed at once for 3D input. A 2D weight is processed in row
            blocks sized to keep the float32 scratch near 32 MB.
        scale_search: Choose every group's e4m3 block scale by the MSE-optimal search of the module docstring
            instead of the amax-derived scale. The global scale and the layout are unchanged; the default
            (False) is bit for bit the plain data-free quantization.

    Returns:
        Packed values, e4m3 block scales, per-tensor (2D) or per-expert (3D) encode global
        scales, error statistics and scale statistics.

    Raises:
        ValueError: For a wrong rank, an input width that is not a multiple of 16, or NaN/inf.
    """
    if weight.ndim not in (2, 3):
        raise ValueError(f"quantize_nvfp4 needs a rank 2 or 3 tensor, got rank {weight.ndim}")
    if weight.shape[-1] % GROUP_SIZE:
        raise ValueError(f"input width {weight.shape[-1]} is not a multiple of 16")
    w3 = weight.unsqueeze(0) if weight.ndim == 2 else weight
    n_exp, rows, cols = w3.shape
    packed = torch.empty((n_exp, rows, cols // 2), dtype=torch.uint8, device=w3.device)
    block = torch.empty((n_exp, rows, cols // GROUP_SIZE), dtype=torch.float8_e4m3fn, device=w3.device)
    gscale = torch.empty(n_exp, dtype=torch.float32, device=w3.device)
    acc = ErrorAccumulator()
    stats = _EMPTY_STATS
    if weight.ndim == 3:
        jobs = [
            (slice(i, i + max(1, expert_chunk)), slice(None)) for i in range(0, n_exp, max(1, expert_chunk))
        ]
    else:
        step = max(1, _CHUNK_ELEMS // max(1, cols))
        jobs = [(slice(0, 1), slice(i, i + step)) for i in range(0, rows, step)]
        amax = w3.to(torch.float32).abs().amax().reshape(1)
        if not torch.isfinite(amax).all():
            raise ValueError("quantize_nvfp4: source tensor holds non-finite values")
        gscale[:] = _global_scale(amax)
    for esl, rsl in jobs:
        w = w3[esl, rsl].to(torch.float32)
        if not torch.isfinite(w).all():
            raise ValueError("quantize_nvfp4: source tensor holds non-finite values")
        if weight.ndim == 3:
            gscale[esl] = _global_scale(w.abs().amax(dim=(1, 2)))
        p, s, st = _quantize_chunk(w, gscale[esl], acc, scale_search)
        packed[esl, rsl] = p
        block[esl, rsl] = s
        stats = stats.merge(st)
    if weight.ndim == 2:
        return Nvfp4Result(packed[0], block[0], gscale, acc.finish(), stats)
    return Nvfp4Result(packed, block, gscale, acc.finish(), stats)


def dequantize_nvfp4(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    global_scale: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize with the vLLM ordering ``fp4 * (block_scale * (1 / global_scale))``.

    ``global_scale`` is the stored encode scale: ``[1]`` for 2D input, ``[E]`` for 3D input.
    """
    values = decode_fp4(unpack_fp4(packed))
    blocks = values.reshape(*values.shape[:-1], -1, GROUP_SIZE)
    inv_g = (1.0 / global_scale.to(torch.float32)).reshape(-1, *([1] * (blocks.ndim - 1)))
    if packed.ndim == 2:
        inv_g = inv_g.reshape(1, 1, 1)
    scale = block_scale.to(torch.float32).unsqueeze(-1) * inv_g
    return (blocks * scale).reshape(values.shape).to(dtype)
