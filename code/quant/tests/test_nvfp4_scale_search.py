"""Tests for the MSE-optimal block scales of NVFP4 (``quantize_nvfp4(..., scale_search=True)``).

PROTOCOL.md amendment 25 fixes the method: for every group of 16 weights, the amax-derived scale is multiplied
by each of six fixed factors (1.00, 0.95, 0.90, 0.85, 0.80, 0.75), every candidate is rounded to e4m3 exactly
as the existing scale is, and the candidate whose dequantized group has the smallest squared error is kept
(ties: the largest factor). The global scale and the on-disk layout do not change.

The reference implementations below are written here, not imported: one loops over single groups, the other
works on the whole tensor at once without any chunking.
"""

import hashlib
import itertools

import pytest
import torch

from olala_quant import nvfp4

FACTORS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75)

# Characterization of the data-free path BEFORE the scale search existed (torch 2.14.0): sha256 of the packed
# codes, the e4m3 block scales and the global scales of a fixed random tensor. The default must keep producing
# them.
GOLDEN_2D = "e9adf800c628f458dcccb65f1954e72ab610bfff3daac19dd18797cf3c7f8a8b"
GOLDEN_3D = "150872b7de5057d810dad47109c75ad4de949fc6d3ae115a474a46477c1f1328"


def _digest(r: nvfp4.Nvfp4Result) -> str:
    blob = (
        r.packed.numpy().tobytes()
        + r.block_scale.view(torch.uint8).numpy().tobytes()
        + r.global_scale.numpy().tobytes()
    )
    return hashlib.sha256(blob).hexdigest()


def _golden_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(1234)
    w2 = (torch.randn(48, 80, generator=g) * 0.05).to(torch.bfloat16)
    noise = torch.randn(5, 32, 64, generator=g)  # the draw order is part of the characterization
    w3 = (noise * 0.03 * (torch.rand(5, 1, 1, generator=g) * 4 + 0.2)).to(torch.bfloat16)
    return w2, w3


def _tensor(kind: str, seed: int, shape: tuple[int, ...]) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(shape, generator=g)
    if kind == "gaussian":
        w = z
    elif kind == "student_t2":  # infinite variance: the heaviest tail that still has a mean
        w = z / (torch.randn(shape, generator=g).abs() + 1e-3)
    elif kind == "laplace":
        u = torch.rand(shape, generator=g) - 0.5
        w = -torch.sign(u) * torch.log1p(-2 * u.abs())
    elif kind == "lognormal":
        w = torch.sign(z) * torch.exp(torch.randn(shape, generator=g) * 1.5)
    elif kind == "outliers":  # a gaussian bulk with one weight in 200 a hundred times larger
        w = z * 0.02
        mask = torch.rand(shape, generator=g) < 0.005
        w = torch.where(mask, w * 100, w)
    elif kind == "sparse":
        w = torch.where(torch.rand(shape, generator=g) < 0.9, torch.zeros(shape), z)
    else:
        raise AssertionError(kind)
    return w.to(torch.bfloat16)


KINDS = ["gaussian", "student_t2", "laplace", "lognormal", "outliers", "sparse"]


def _group_errors(weight: torch.Tensor, r: nvfp4.Nvfp4Result) -> torch.Tensor:
    """Squared error of every group of 16 of the dequantized result against the float32 source."""
    deq = nvfp4.dequantize_nvfp4(r.packed, r.block_scale, r.global_scale)
    diff = deq - weight.to(torch.float32)
    return (diff * diff).reshape(*diff.shape[:-1], -1, 16).sum(dim=-1)


def reference_candidates(
    weight: torch.Tensor, gscale: torch.Tensor
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """For every factor ``(e4m3 scale [E, rows, groups], squared error [E, rows, groups])``, whole tensor."""
    w = weight.to(torch.float32)
    w3 = w.unsqueeze(0) if w.ndim == 2 else w
    e, rows, cols = w3.shape
    blocks = w3.reshape(e, rows, cols // 16, 16)
    g = gscale.reshape(e, 1, 1, 1)
    base = g * (blocks.abs().amax(dim=-1, keepdim=True) * (1.0 / 6.0))
    out = []
    for f in FACTORS:
        s = (base * f).clamp(0.0, 448.0).to(torch.float8_e4m3fn)
        sq = s.to(torch.float32)
        dq = sq * (1.0 / g)
        mult = torch.where(sq > 0, 1.0 / torch.where(sq > 0, dq, torch.ones_like(dq)), torch.zeros_like(dq))
        codes = nvfp4.encode_fp4((blocks * mult).clamp(-6.0, 6.0))
        err = ((nvfp4.decode_fp4(codes) * dq - blocks) ** 2).sum(dim=-1)
        out.append((s.squeeze(-1), err))
    return out


# ------------------------------------------------------------------------------------------ the default is
# untouched


def test_the_factors_and_the_version_are_the_ones_of_amendment_25() -> None:
    assert nvfp4.SCALE_SEARCH_FACTORS == FACTORS
    assert nvfp4.SCALE_SEARCH_VERSION == 1
    assert all(a > b for a, b in itertools.pairwise(FACTORS)), "descending: the tie-break relies on it"


def test_the_default_path_is_byte_identical_to_what_it_was_before_the_search_existed() -> None:
    w2, w3 = _golden_inputs()
    assert _digest(nvfp4.quantize_nvfp4(w2)) == GOLDEN_2D
    assert _digest(nvfp4.quantize_nvfp4(w3, expert_chunk=2)) == GOLDEN_3D
    assert _digest(nvfp4.quantize_nvfp4(w3, expert_chunk=8)) == GOLDEN_3D
    # asking for "no search" explicitly is the same call
    assert _digest(nvfp4.quantize_nvfp4(w2, scale_search=False)) == GOLDEN_2D
    assert _digest(nvfp4.quantize_nvfp4(w3, scale_search=False)) == GOLDEN_3D


def test_the_search_changes_the_data_it_is_supposed_to_change() -> None:
    w2, _ = _golden_inputs()
    assert _digest(nvfp4.quantize_nvfp4(w2, scale_search=True)) != GOLDEN_2D


# ------------------------------------------------------------------------------------------ the method itself


def test_it_matches_a_group_by_group_reference_written_with_plain_loops() -> None:
    """Independent of the vectorized code: 96 groups, every candidate built and scored one group at a time."""
    torch.manual_seed(5)
    w = (torch.randn(3, 8, 64) * torch.rand(3, 1, 1).add(0.3) * 0.04).to(torch.bfloat16)
    res = nvfp4.quantize_nvfp4(w, scale_search=True)
    plain = nvfp4.quantize_nvfp4(w)
    assert torch.equal(res.global_scale, plain.global_scale)
    for e in range(3):
        g = plain.global_scale[e]
        for r in range(8):
            for k in range(4):
                vec = w[e, r, 16 * k : 16 * (k + 1)].to(torch.float32)
                base = g * (vec.abs().max() * (1.0 / 6.0))
                best_err, best_scale = None, None
                for f in FACTORS:
                    s = (base * f).clamp(0.0, 448.0).to(torch.float8_e4m3fn)
                    sq = s.to(torch.float32)
                    dq = sq * (1.0 / g)
                    mult = 1.0 / dq if sq > 0 else torch.tensor(0.0)
                    deq = nvfp4.decode_fp4(nvfp4.encode_fp4((vec * mult).clamp(-6.0, 6.0))) * dq
                    err = float(((deq - vec) ** 2).sum())
                    if best_err is None or err < best_err:
                        best_err, best_scale = err, s
                got = res.block_scale[e, r, k]
                assert got.view(torch.uint8) == best_scale.view(torch.uint8), (e, r, k)


@pytest.mark.parametrize("kind", KINDS)
def test_it_matches_a_whole_tensor_reference_and_picks_the_best_candidate_of_every_group(kind: str) -> None:
    for shape in ((40, 96), (3, 24, 64)):
        w = _tensor(kind, 11, shape)
        res = nvfp4.quantize_nvfp4(w, scale_search=True)
        plain = nvfp4.quantize_nvfp4(w)
        cands = reference_candidates(w, plain.global_scale)
        chosen = res.block_scale.view(torch.uint8).reshape(cands[0][0].shape)
        # 1. every stored scale is one of the six candidates of its group (rounding order and factors are
        # right)
        is_candidate = torch.zeros_like(chosen, dtype=torch.bool)
        for s, _ in cands:
            is_candidate |= s.view(torch.uint8) == chosen
        assert bool(is_candidate.all()), kind
        # 2. the error of the chosen scale is the smallest of the six (the reference's own errors, 1e-6
        # relative)
        errs = torch.stack([e for _, e in cands])
        chosen_err = _group_errors(w, res).reshape(errs.shape[1:])
        best = errs.min(dim=0).values
        assert bool((chosen_err <= best * (1 + 1e-5) + 1e-12).all()), kind
        # 3. and it is the same choice as the reference's in all but (rare) float-summation near ties
        ref_pick = torch.full_like(chosen, 0)
        decided = torch.zeros_like(chosen, dtype=torch.bool)
        best_so_far = torch.full(best.shape, float("inf"))
        for s, e in cands:
            better = e < best_so_far
            ref_pick = torch.where(better, s.view(torch.uint8), ref_pick)
            best_so_far = torch.where(better, e, best_so_far)
            decided |= better
        assert (ref_pick != chosen).float().mean() < 1e-3, kind


def test_the_global_scale_and_the_layout_are_the_ones_of_the_plain_path() -> None:
    w = _tensor("gaussian", 2, (4, 32, 64))
    res, plain = nvfp4.quantize_nvfp4(w, scale_search=True), nvfp4.quantize_nvfp4(w)
    assert torch.equal(res.global_scale, plain.global_scale)  # 448 * 6 / amax of the ORIGINAL tensor (expert)
    amax = w.to(torch.float32).abs().amax(dim=(1, 2))
    assert torch.allclose(res.global_scale, 448.0 * 6.0 / amax, rtol=1e-6)
    for field in ("packed", "block_scale", "global_scale"):
        a, b = getattr(res, field), getattr(plain, field)
        assert (a.dtype, a.shape) == (b.dtype, b.shape), field
    assert float(res.block_scale.to(torch.float32).max()) <= 448.0
    assert res.scale_stats.n_overflow == 0
    w2 = _tensor("gaussian", 2, (64, 96))
    assert torch.equal(
        nvfp4.quantize_nvfp4(w2, scale_search=True).global_scale, nvfp4.quantize_nvfp4(w2).global_scale
    )


# ------------------------------------------------------------------------------------------ never worse,
# better on average


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("kind", KINDS)
def test_property_no_group_and_no_tensor_has_a_larger_squared_error_than_the_plain_scale(
    kind: str, seed: int
) -> None:
    w = _tensor(kind, seed, (2, 48, 128))
    res, plain = nvfp4.quantize_nvfp4(w, scale_search=True), nvfp4.quantize_nvfp4(w)
    se, sp = _group_errors(w, res), _group_errors(w, plain)
    assert bool((se <= sp * (1 + 1e-5) + 1e-12).all()), (kind, seed, float((se - sp).max()))
    assert float(se.sum()) <= float(sp.sum()) * (1 + 1e-6)
    assert res.error.rel_fro <= plain.error.rel_fro * (1 + 1e-6)


def _mse_reduction(kind: str, seed: int, shape: tuple[int, ...]) -> float:
    w = _tensor(kind, seed, shape)
    se = float(_group_errors(w, nvfp4.quantize_nvfp4(w, scale_search=True)).sum())
    sp = float(_group_errors(w, nvfp4.quantize_nvfp4(w)).sum())
    return 1.0 - se / sp


# Measured (torch 2.14.0, CPU, three seeds of a [3, 128, 256] tensor, relative reduction of the total squared
# error of the dequantized weights): gaussian 13.8 to 14.1 %, laplace 11.5 to 12.0 %, sparse 13.3 to 14.9 %,
# lognormal(1.5) 3.8 to 4.6 %, student-t with 2 degrees of freedom 1.4 to 2.7 %, a gaussian bulk with 0.5 % of
# x100 outliers 1.2 to 3.9 %. The floors below sit about 15 % under the smallest value seen.
MIN_GAUSSIAN_GAIN = 0.12
MIN_HEAVY_GAIN: dict[str, float] = {"laplace": 0.10, "lognormal": 0.03, "student_t2": 0.01, "outliers": 0.008}


def test_it_lowers_the_squared_error_of_a_gaussian_tensor_by_a_measured_amount() -> None:
    gains = [_mse_reduction("gaussian", s, (3, 128, 256)) for s in range(3)]
    assert min(gains) >= MIN_GAUSSIAN_GAIN, gains


@pytest.mark.parametrize("kind", sorted(MIN_HEAVY_GAIN))
def test_it_lowers_the_squared_error_of_heavy_tailed_tensors_too(kind: str) -> None:
    gains = [_mse_reduction(kind, s, (3, 128, 256)) for s in range(3)]
    assert min(gains) >= MIN_HEAVY_GAIN[kind], (kind, gains)


# ------------------------------------------------------------------------------------------ determinism,
# chunking


def test_it_is_deterministic() -> None:
    w = _tensor("student_t2", 4, (4, 64, 96))
    assert _digest(nvfp4.quantize_nvfp4(w, scale_search=True)) == _digest(
        nvfp4.quantize_nvfp4(w, scale_search=True)
    )


def test_the_result_does_not_depend_on_how_the_tensor_is_chunked(monkeypatch: pytest.MonkeyPatch) -> None:
    w = _tensor("gaussian", 6, (6, 64, 128))
    whole = _digest(nvfp4.quantize_nvfp4(w, expert_chunk=6, scale_search=True))
    assert _digest(nvfp4.quantize_nvfp4(w, expert_chunk=1, scale_search=True)) == whole
    assert _digest(nvfp4.quantize_nvfp4(w, expert_chunk=4, scale_search=True)) == whole
    monkeypatch.setattr(nvfp4, "_SEARCH_CHUNK_ELEMS", 16 * 128)  # a few rows at a time
    assert _digest(nvfp4.quantize_nvfp4(w, expert_chunk=6, scale_search=True)) == whole
    monkeypatch.setattr(nvfp4, "_SEARCH_CHUNK_ELEMS", 1)  # one row at a time, the floor
    assert _digest(nvfp4.quantize_nvfp4(w, expert_chunk=3, scale_search=True)) == whole


def test_the_search_never_holds_more_than_the_documented_number_of_weights_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert nvfp4._SEARCH_CHUNK_ELEMS == 4 * 1024 * 1024  # the chunk size stated in the module docstring
    limit = 8 * 1024
    monkeypatch.setattr(nvfp4, "_SEARCH_CHUNK_ELEMS", limit)
    seen: list[int] = []
    real = nvfp4._group_error

    def spy(blocks: torch.Tensor, scale: torch.Tensor, g4: torch.Tensor) -> torch.Tensor:
        seen.append(blocks.numel())
        return real(blocks, scale, g4)

    monkeypatch.setattr(nvfp4, "_group_error", spy)
    w = _tensor("gaussian", 8, (4, 96, 256))  # 98,304 weights, 12 times the limit
    nvfp4.quantize_nvfp4(w, expert_chunk=4, scale_search=True)
    assert seen and max(seen) <= limit
    assert len(seen) >= 12 * len(FACTORS)


# ------------------------------------------------------------------------------------------ tie-break and
# order


def _fake_errors(monkeypatch: pytest.MonkeyPatch, per_call: list[float]) -> list[torch.Tensor]:
    """Replace the error function: the i-th candidate of every group gets the error ``per_call[i]``."""
    scales: list[torch.Tensor] = []

    def fake(blocks: torch.Tensor, scale: torch.Tensor, g4: torch.Tensor) -> torch.Tensor:
        scales.append(scale.clone())
        return torch.full(scale.shape, per_call[len(scales) - 1])

    monkeypatch.setattr(nvfp4, "_group_error", fake)
    return scales


def _plain_and_candidates(w: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    plain = nvfp4.quantize_nvfp4(w)
    cands = [s for s, _ in reference_candidates(w, plain.global_scale)]
    return plain.block_scale, cands


def test_candidates_are_tried_from_the_largest_factor_down(monkeypatch: pytest.MonkeyPatch) -> None:
    w = _tensor("gaussian", 9, (2, 8, 32))
    scales = _fake_errors(monkeypatch, [1.0] * 6)
    nvfp4.quantize_nvfp4(w, scale_search=True)
    _, cands = _plain_and_candidates(w)
    assert len(scales) == 6
    for got, want in zip(scales, cands, strict=True):
        assert torch.equal(got.reshape(want.shape).view(torch.uint8), want.view(torch.uint8))


def test_on_a_tie_the_largest_factor_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    w = _tensor("gaussian", 9, (2, 8, 32))
    plain_scale, cands = _plain_and_candidates(w)
    _fake_errors(monkeypatch, [1.0] * 6)  # all six tie: the plain scale (factor 1.00) must be kept
    res = nvfp4.quantize_nvfp4(w, scale_search=True)
    assert torch.equal(res.block_scale.view(torch.uint8), plain_scale.view(torch.uint8))
    monkeypatch.undo()
    _fake_errors(monkeypatch, [5.0, 4.0, 3.0, 2.0, 1.0, 1.0])  # 0.80 and 0.75 tie as the best: 0.80 wins
    res = nvfp4.quantize_nvfp4(w, scale_search=True)
    assert torch.equal(res.block_scale.view(torch.uint8).reshape(cands[4].shape), cands[4].view(torch.uint8))
    monkeypatch.undo()
    _fake_errors(monkeypatch, [5.0, 4.0, 3.0, 2.0, 2.0, 2.0])  # a three-way tie among the last three: 0.85
    res = nvfp4.quantize_nvfp4(w, scale_search=True)
    assert torch.equal(res.block_scale.view(torch.uint8).reshape(cands[3].shape), cands[3].view(torch.uint8))


def test_a_strictly_smaller_error_beats_a_larger_factor(monkeypatch: pytest.MonkeyPatch) -> None:
    w = _tensor("gaussian", 9, (2, 8, 32))
    _, cands = _plain_and_candidates(w)
    _fake_errors(monkeypatch, [1.0, 0.9, 1.0, 1.0, 1.0, 1.0])
    res = nvfp4.quantize_nvfp4(w, scale_search=True)
    assert torch.equal(res.block_scale.view(torch.uint8).reshape(cands[1].shape), cands[1].view(torch.uint8))
    monkeypatch.undo()
    _fake_errors(monkeypatch, [1.0, 1.0, 1.0, 1.0, 1.0, 0.5])
    res = nvfp4.quantize_nvfp4(w, scale_search=True)
    assert torch.equal(res.block_scale.view(torch.uint8).reshape(cands[5].shape), cands[5].view(torch.uint8))


# ------------------------------------------------------------------------------------------ round trip, edge
# cases


@pytest.mark.parametrize("shape", [(40, 96), (3, 24, 64)])
def test_the_repos_dequantizer_reproduces_the_error_the_quantizer_reports(shape: tuple[int, ...]) -> None:
    w = _tensor("student_t2", 12, shape)
    res, plain = nvfp4.quantize_nvfp4(w, scale_search=True), nvfp4.quantize_nvfp4(w)
    deq = nvfp4.dequantize_nvfp4(res.packed, res.block_scale, res.global_scale)
    x = w.to(torch.float32)
    rel = float(torch.linalg.vector_norm(deq - x) / torch.linalg.vector_norm(x))
    assert rel == pytest.approx(res.error.rel_fro, rel=1e-5)
    assert float((deq - x).abs().max()) == pytest.approx(res.error.max_abs, rel=1e-5)
    assert rel <= plain.error.rel_fro


def test_all_zero_and_partly_zero_tensors_are_handled_like_the_plain_path() -> None:
    zeros = torch.zeros(4, 32, dtype=torch.bfloat16)
    res = nvfp4.quantize_nvfp4(zeros, scale_search=True)
    assert _digest(res) == _digest(nvfp4.quantize_nvfp4(zeros))
    w = _tensor("sparse", 13, (2, 16, 64))
    w[:, :, :16] = 0  # a whole column of zero groups
    res = nvfp4.quantize_nvfp4(w, scale_search=True)
    assert bool(torch.isfinite(nvfp4.dequantize_nvfp4(res.packed, res.block_scale, res.global_scale)).all())
    assert res.scale_stats.n_zero_blocks >= 2 * 16


def test_the_scale_statistics_describe_the_stored_scales() -> None:
    w = _tensor("gaussian", 14, (2, 32, 64))
    res = nvfp4.quantize_nvfp4(w, scale_search=True)
    stored = res.block_scale.to(torch.float32)
    assert res.scale_stats.max_block_scale == float(stored.max())
    assert res.scale_stats.n_blocks == stored.numel()
    assert res.scale_stats.min_nonzero_block_scale == float(stored[stored > 0].min())


def test_invalid_inputs_are_refused_as_before() -> None:
    with pytest.raises(ValueError, match="rank"):
        nvfp4.quantize_nvfp4(torch.zeros(16), scale_search=True)
    with pytest.raises(ValueError, match="multiple of 16"):
        nvfp4.quantize_nvfp4(torch.zeros(4, 20), scale_search=True)
    bad = torch.zeros(2, 32)
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        nvfp4.quantize_nvfp4(bad, scale_search=True)
