"""Tests for the FidelityData container, npz round trip and the reference-vs-variant comparison."""

import math

import numpy as np
import pytest

from fidelity import (
    FidelityData,
    compare_fidelity,
    load_fidelity,
    render_comparison_markdown,
    save_fidelity,
)

VOCAB = np.array([f"token_id:{i}" for i in range(8)])


def make_data(
    probs: np.ndarray,
    target_lp: np.ndarray | None = None,
    *,
    vocab: np.ndarray = VOCAB,
    chunks_sha256: str = "abc",
) -> FidelityData:
    """Build data from a full distribution ``probs[C, P, V]`` (keep every token, K = V)."""
    c, p, v = probs.shape
    order = np.argsort(-probs, axis=2, kind="stable")
    top_lp = np.log(np.take_along_axis(probs, order, axis=2)).astype(np.float32)
    if target_lp is None:
        target_lp = np.full((c, p), math.log(1 / v), dtype=np.float32)
    return FidelityData(
        vocab=vocab,
        top_ids=order.astype(np.int32),
        top_lp=top_lp,
        target_lp=target_lp.astype(np.float32),
        meta={"chunks_sha256": chunks_sha256, "model": "m"},
    )


def random_probs(c: int = 4, p: int = 6, v: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.random((c, p, v)) + 0.05
    return x / x.sum(axis=2, keepdims=True)


def test_npz_round_trip_preserves_every_field(tmp_path):
    data = make_data(random_probs())
    path = tmp_path / "ref.npz"
    save_fidelity(path, data)
    back = load_fidelity(path)
    assert back.vocab.tolist() == data.vocab.tolist()
    np.testing.assert_array_equal(back.top_ids, data.top_ids)
    np.testing.assert_array_equal(back.top_lp, data.top_lp)
    np.testing.assert_array_equal(back.target_lp, data.target_lp)
    assert back.meta == data.meta


def test_identical_files_have_zero_divergence_and_full_agreement():
    data = make_data(random_probs())
    res = compare_fidelity(data, data, n_boot=50, seed=0)
    assert res["kld_mean"]["value"] == 0.0
    assert res["kld_median"]["value"] == 0.0
    assert res["kld_p99"]["value"] == 0.0
    assert res["top1_agreement"]["value"] == 1.0
    assert res["ppl_ratio"]["value"] == pytest.approx(1.0)
    assert res["n_chunks"] == 4
    assert res["n_positions"] == 24


def test_a_different_vocabulary_order_in_the_variant_file_does_not_change_the_result():
    probs = random_probs()
    ref = make_data(probs)
    shuffled_vocab = VOCAB[::-1].copy()  # id i of the variant file means token (7 - i)
    var = make_data(probs[:, :, ::-1], vocab=shuffled_vocab)
    res = compare_fidelity(ref, var, n_boot=20, seed=0)
    assert res["kld_mean"]["value"] == pytest.approx(0.0, abs=1e-12)
    assert res["top1_agreement"]["value"] == 1.0


def test_perplexity_ratio_follows_a_uniform_shift_of_the_target_log_probs():
    probs = random_probs()
    base = np.full((4, 6), math.log(0.25))
    ref = make_data(probs, base)
    var = make_data(probs, base - 0.1)  # the variant is 0.1 nats worse on every token
    res = compare_fidelity(ref, var, n_boot=50, seed=0)
    assert res["ppl_ref"]["value"] == pytest.approx(4.0, rel=1e-6)
    assert res["ppl_var"]["value"] == pytest.approx(4.0 * math.exp(0.1), rel=1e-6)
    assert res["ppl_ratio"]["value"] == pytest.approx(math.exp(0.1), rel=1e-6)
    lo, hi = res["ppl_ratio"]["lo"], res["ppl_ratio"]["hi"]
    assert lo == pytest.approx(math.exp(0.1), rel=1e-6)
    assert hi == pytest.approx(math.exp(0.1), rel=1e-6)


def test_one_diverging_position_out_of_100_gives_known_mean_median_and_p99():
    ref_p = np.full((1, 100, 2), 0.5)
    var_p = ref_p.copy()
    var_p[0, 17] = [0.1, 0.9]  # KL((.5,.5)||(.1,.9)) = ln(5/3) at one position only
    ref = make_data(ref_p)
    var = make_data(var_p)
    res = compare_fidelity(ref, var, n_boot=20, seed=0)
    v = math.log(5 / 3)
    assert res["kld_mean"]["value"] == pytest.approx(v / 100, rel=1e-6)
    assert res["kld_median"]["value"] == 0.0
    # np.percentile (linear interpolation) of 99 zeros and one v at q=99 is 0.01 * v
    assert res["kld_p99"]["value"] == pytest.approx(0.01 * v, rel=1e-6)
    assert res["top1_agreement"]["value"] == pytest.approx(0.99)  # tie (.5,.5) resolved by order
    for key in ("kld_mean", "kld_median", "kld_p99", "top1_agreement", "ppl_ref", "ppl_var"):
        assert res[key]["lo"] <= res[key]["value"] <= res[key]["hi"] or key == "kld_median"


def test_max_k_truncation_is_recorded_and_changes_the_divergence_estimate():
    probs = random_probs(v=8)
    ref = make_data(probs)
    var = make_data(random_probs(v=8, seed=1))
    full = compare_fidelity(ref, var, n_boot=10, seed=0)
    cut = compare_fidelity(ref, var, max_k=3, n_boot=10, seed=0)
    assert full["k"] == 8
    assert cut["k"] == 3
    assert cut["kld_mean"]["value"] != full["kld_mean"]["value"]


def test_files_made_from_different_chunks_are_refused():
    probs = random_probs()
    with pytest.raises(ValueError, match="chunks"):
        compare_fidelity(
            make_data(probs, chunks_sha256="a"), make_data(probs, chunks_sha256="b"), n_boot=5
        )


def test_files_with_different_shapes_are_refused():
    with pytest.raises(ValueError, match="shape"):
        compare_fidelity(make_data(random_probs(c=4)), make_data(random_probs(c=3)), n_boot=5)


def test_markdown_report_contains_the_numbers_of_the_result_dict():
    ref = make_data(random_probs())
    var = make_data(random_probs(seed=2))
    res = compare_fidelity(ref, var, n_boot=20, seed=0)
    md = render_comparison_markdown(res)
    assert f"{res['kld_mean']['value']:.5f}" in md
    assert f"{res['ppl_ref']['value']:.3f}" in md
    assert "KL(ref || variant)" in md
    assert "approximation" in md.lower()
