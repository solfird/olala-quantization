"""Known-answer tests for the KL / perplexity / agreement maths on synthetic distributions."""

import math

import numpy as np
import pytest

from fidelity_math import kl_topk_union, perplexity, top1_agreement

NEG_INF = -np.inf


def _row(ids: list[int], probs: list[float], width: int) -> tuple[np.ndarray, np.ndarray]:
    """Pad one distribution to ``width`` entries (id -1, logprob -inf)."""
    pad = width - len(ids)
    return (
        np.array([ids + [-1] * pad], dtype=np.int32),
        np.array([[math.log(p) for p in probs] + [NEG_INF] * pad], dtype=np.float64),
    )


def test_identical_distributions_have_exactly_zero_kl():
    ids, lp = _row([0, 1, 2], [0.5, 0.25, 0.125], 4)  # 0.125 left in the 'other' bucket
    assert kl_topk_union(ids, lp, ids, lp)[0] == 0.0


def test_two_token_distributions_match_the_closed_form():
    # KL((.5,.5) || (.9,.1)) = .5 ln(.5/.9) + .5 ln(.5/.1) = ln(5/3)
    ref = _row([0, 1], [0.5, 0.5], 2)
    var = _row([0, 1], [0.9, 0.1], 2)
    assert kl_topk_union(*ref, *var)[0] == pytest.approx(math.log(5 / 3), abs=1e-12)


def test_entry_order_inside_a_row_does_not_matter():
    ref = _row([0, 1], [0.5, 0.5], 2)
    var = _row([1, 0], [0.1, 0.9], 2)  # same distribution as above, listed in another order
    assert kl_topk_union(*ref, *var)[0] == pytest.approx(math.log(5 / 3), abs=1e-12)


def test_remaining_mass_is_lumped_into_one_other_bucket():
    # Same top-2 tokens; the 'other' mass is 0.25 for the reference and 0.375 for the variant.
    # KL = .5 ln 1 + .25 ln(.25/.125) + .25 ln(.25/.375) = .25 ln(4/3)
    ref = _row([0, 1], [0.5, 0.25], 2)
    var = _row([0, 1], [0.5, 0.125], 2)
    assert kl_topk_union(*ref, *var)[0] == pytest.approx(0.25 * math.log(4 / 3), abs=1e-12)


def test_lumping_lower_bounds_the_full_vocabulary_kl_when_the_top_sets_coincide():
    p = np.array([0.40, 0.30, 0.15, 0.10, 0.03, 0.02])
    q = np.array([0.35, 0.35, 0.10, 0.15, 0.04, 0.01])
    full = float(np.sum(p * np.log(p / q)))
    ref = _row([0, 1, 2, 3], list(p[:4]), 4)
    var = _row([0, 1, 2, 3], list(q[:4]), 4)
    lumped = kl_topk_union(*ref, *var)[0]
    four = float(np.sum(p[:4] * np.log(p[:4] / q[:4])))  # the 'other' buckets are both 0.05
    assert lumped == pytest.approx(four, abs=1e-12)
    assert 0.0 < lumped < full


def test_k_equal_to_vocabulary_size_gives_the_exact_kl():
    p = np.array([0.4, 0.3, 0.2, 0.1])
    q = np.array([0.1, 0.2, 0.3, 0.4])
    exact = float(np.sum(p * np.log(p / q)))
    ref = _row([0, 1, 2, 3], list(p), 4)
    var = _row([0, 1, 2, 3], list(q), 4)
    assert kl_topk_union(*ref, *var)[0] == pytest.approx(exact, abs=1e-12)


def test_disjoint_top_sets_use_the_documented_imputation():
    # Reference tokens A,B = .6,.3 (0.1 unseen); variant tokens C,D = .6,.3 (0.1 unseen).
    # A token missing from a list is imputed min(k-th probability, unseen mass / n_missing),
    # here min(.3, .1/2) = .05, so the union is {A,B,C,D,other} with
    #   p = (.6, .3, .05, .05, 0)   q = (.05, .05, .6, .3, 0)
    ref = _row([0, 1], [0.6, 0.3], 2)
    var = _row([2, 3], [0.6, 0.3], 2)
    expected = (
        0.6 * math.log(0.6 / 0.05)
        + 0.3 * math.log(0.3 / 0.05)
        + 0.05 * math.log(0.05 / 0.6)
        + 0.05 * math.log(0.05 / 0.3)
    )
    assert kl_topk_union(*ref, *var)[0] == pytest.approx(expected, abs=1e-12)


def test_padding_entries_are_ignored_and_rows_are_independent():
    r0 = _row([0, 1], [0.5, 0.5], 3)
    v0 = _row([0, 1], [0.9, 0.1], 3)
    r1 = _row([4, 5, 6], [0.5, 0.3, 0.2], 3)
    v1 = _row([4, 5, 6], [0.5, 0.3, 0.2], 3)
    out = kl_topk_union(
        np.vstack([r0[0], r1[0]]),
        np.vstack([r0[1], r1[1]]),
        np.vstack([v0[0], v1[0]]),
        np.vstack([v0[1], v1[1]]),
    )
    assert out[0] == pytest.approx(math.log(5 / 3), abs=1e-12)
    assert out[1] == 0.0


def test_kl_is_never_negative_and_finite_for_noisy_rounded_logprobs():
    rng = np.random.default_rng(0)
    n, k = 200, 20
    logits = rng.normal(size=(n, 50)) * 3
    noisy = logits + rng.normal(size=(n, 50)) * 0.3

    def topk(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lp = x - np.log(np.exp(x).sum(axis=1, keepdims=True))
        ids = np.argsort(-lp, axis=1)[:, :k]
        return ids.astype(np.int32), np.round(np.take_along_axis(lp, ids, axis=1), 6)

    kl = kl_topk_union(*topk(logits), *topk(noisy))
    assert np.all(np.isfinite(kl))
    assert np.all(kl >= 0.0)


def test_perplexity_of_a_uniform_model_is_the_vocabulary_size():
    assert perplexity(np.full((3, 5), -math.log(8))) == pytest.approx(8.0)


def test_perplexity_is_the_exponential_of_the_mean_negative_log_likelihood():
    target = np.array([[math.log(0.5), math.log(0.25)]])
    assert perplexity(target) == pytest.approx(2**1.5)


def test_top1_agreement_uses_the_highest_probability_entry_not_the_first_one():
    ref_ids = np.array([[3, 1], [7, 2]], dtype=np.int32)
    ref_lp = np.log(np.array([[0.6, 0.3], [0.2, 0.7]]))  # top-1: 3 and 2
    var_ids = np.array([[3, 1], [9, 2]], dtype=np.int32)
    var_lp = np.log(np.array([[0.5, 0.4], [0.1, 0.8]]))  # top-1: 3 and 2
    assert top1_agreement(ref_ids, ref_lp, var_ids, var_lp).tolist() == [True, True]
    var_lp2 = np.log(np.array([[0.5, 0.4], [0.9, 0.05]]))  # top-1 of the second row is now 9
    assert top1_agreement(ref_ids, ref_lp, var_ids, var_lp2).tolist() == [True, False]
