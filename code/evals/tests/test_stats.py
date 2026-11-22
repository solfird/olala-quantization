"""Tests for the bootstrap helpers in stats.py (known answers, determinism, pairing)."""

import numpy as np
import pytest

from stats import bootstrap_mean_ci, cluster_bootstrap_ci, paired_bootstrap_diff_ci


def test_constant_sample_has_degenerate_ci():
    mean, lo, hi = bootstrap_mean_ci([1.0] * 10, n_boot=500, seed=0)
    assert (mean, lo, hi) == (1.0, 1.0, 1.0)


def test_ci_brackets_the_mean_and_is_deterministic_for_a_seed():
    x = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    a = bootstrap_mean_ci(x, n_boot=2000, seed=7)
    b = bootstrap_mean_ci(x, n_boot=2000, seed=7)
    assert a == b
    mean, lo, hi = a
    assert mean == pytest.approx(5 / 8)
    assert lo <= mean <= hi


def test_ci_width_matches_the_normal_approximation_for_a_bernoulli_sample():
    # p = 0.5, n = 400 -> standard error 0.025 -> 95% interval width about 0.098.
    x = np.array([0.0, 1.0] * 200)
    _, lo, hi = bootstrap_mean_ci(x, n_boot=5000, seed=1)
    assert 0.085 < hi - lo < 0.110


def test_paired_diff_of_identical_vectors_is_exactly_zero():
    a = [0.0, 1.0, 1.0, 0.0]
    assert paired_bootstrap_diff_ci(a, a, n_boot=500, seed=0) == (0.0, 0.0, 0.0)


def test_paired_diff_of_a_constant_shift_is_that_shift():
    a = np.array([0.1, 0.5, 0.9, 0.3])
    diff, lo, hi = paired_bootstrap_diff_ci(a, a + 0.1, n_boot=500, seed=0)
    assert diff == pytest.approx(0.1)
    assert lo == pytest.approx(0.1)
    assert hi == pytest.approx(0.1)


def test_pairing_gives_a_narrower_interval_than_independent_resampling():
    rng = np.random.default_rng(3)
    a = (rng.random(400) < 0.5).astype(float)
    flip = rng.random(400) < 0.04
    b = np.where(flip, 1.0 - a, a)  # b agrees with a on 96% of the items
    _, lo, hi = paired_bootstrap_diff_ci(a, b, n_boot=4000, seed=2)
    # Independent samples of this size would give a width of about 0.14.
    assert hi - lo < 0.06


def test_paired_diff_rejects_vectors_of_different_length():
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_diff_ci([1.0, 0.0], [1.0], n_boot=10, seed=0)


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        bootstrap_mean_ci([], n_boot=10, seed=0)


def test_cluster_bootstrap_resamples_whole_clusters():
    # Two clusters with very different means: a statistic of the pooled sample can only
    # take the values (all-A), (all-B) or a mixture, never anything outside the data range.
    clusters = [np.zeros(50), np.ones(50)]
    stat, lo, hi = cluster_bootstrap_ci(clusters, np.mean, n_boot=1000, seed=0)
    assert stat == pytest.approx(0.5)
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0
    assert hi - lo > 0.4  # cluster resampling makes the interval wide


def test_cluster_bootstrap_supports_two_column_arrays_for_paired_statistics():
    # column 0 = reference NLL, column 1 = variant NLL; statistic = difference of means.
    clusters = [np.column_stack([np.full(10, 1.0), np.full(10, 1.5)]) for _ in range(8)]

    def stat(v: np.ndarray) -> float:
        return float(v[:, 1].mean() - v[:, 0].mean())

    value, lo, hi = cluster_bootstrap_ci(clusters, stat, n_boot=200, seed=0)
    assert (value, lo, hi) == (pytest.approx(0.5), pytest.approx(0.5), pytest.approx(0.5))


def test_multi_statistic_bootstrap_matches_single_statistic_calls_with_the_same_seed():
    from stats import cluster_bootstrap_multi

    rng = np.random.default_rng(0)
    clusters = [rng.normal(size=(30, 2)) for _ in range(12)]
    stats_fns = {"mean0": lambda v: float(v[:, 0].mean()), "max1": lambda v: float(v[:, 1].max())}
    multi = cluster_bootstrap_multi(clusters, stats_fns, n_boot=100, seed=5)
    assert set(multi) == {"mean0", "max1"}
    # Same resamples are shared by all statistics, so each equals the single-statistic result.
    for name, fn in stats_fns.items():
        assert multi[name] == pytest.approx(cluster_bootstrap_ci(clusters, fn, n_boot=100, seed=5))
