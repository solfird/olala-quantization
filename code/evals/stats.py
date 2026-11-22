"""Percentile-bootstrap helpers shared by compare.py and fidelity.py.

All intervals are two-sided percentile bootstrap intervals. Every function takes an explicit
``seed`` so that a reported interval can be regenerated bit for bit from the result files.
"""

from collections.abc import Callable, Mapping, Sequence

import numpy as np

CONFIDENCE = 0.95


def _bounds(samples: np.ndarray, confidence: float) -> tuple[float, float]:
    """Return the (lower, upper) percentile bounds of ``samples``."""
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(samples, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def bootstrap_mean_ci(
    values: Sequence[float] | np.ndarray,
    n_boot: int = 10_000,
    seed: int = 0,
    confidence: float = CONFIDENCE,
) -> tuple[float, float, float]:
    """Mean of ``values`` with a percentile bootstrap interval over its elements.

    Args:
        values: One number per independent unit (e.g. per item).
        n_boot: Number of bootstrap resamples.
        seed: Seed of the resampling generator.
        confidence: Two-sided coverage of the interval.

    Returns:
        ``(mean, lower, upper)``.

    Raises:
        ValueError: If ``values`` is empty.
    """
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    lo, hi = _bounds(x[idx].mean(axis=1), confidence)
    return float(x.mean()), lo, hi


def paired_bootstrap_diff_ci(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    n_boot: int = 10_000,
    seed: int = 0,
    confidence: float = CONFIDENCE,
) -> tuple[float, float, float]:
    """Mean of ``b - a`` with a paired percentile bootstrap interval.

    The same resampled unit indices are applied to both vectors, so the interval reflects
    the disagreement between the two systems on the same items, not their marginal spread.

    Args:
        a: Per-item scores of the reference system.
        b: Per-item scores of the variant, in the same item order as ``a``.
        n_boot: Number of bootstrap resamples.
        seed: Seed of the resampling generator.
        confidence: Two-sided coverage of the interval.

    Returns:
        ``(mean(b) - mean(a), lower, upper)``.

    Raises:
        ValueError: If the vectors differ in length or are empty.
    """
    xa = np.asarray(a, dtype=float)
    xb = np.asarray(b, dtype=float)
    if xa.shape != xb.shape:
        raise ValueError("paired vectors must have the same length")
    return bootstrap_mean_ci(xb - xa, n_boot=n_boot, seed=seed, confidence=confidence)


def cluster_bootstrap_multi(
    clusters: Sequence[np.ndarray],
    statistics: Mapping[str, Callable[[np.ndarray], float]],
    n_boot: int = 1_000,
    seed: int = 0,
    confidence: float = CONFIDENCE,
) -> dict[str, tuple[float, float, float]]:
    """Bootstrap several statistics of pooled observations on the same cluster resamples.

    Used for the fidelity metrics, where the independent unit is a text chunk and a
    statistic such as the 99th percentile is computed over all positions of the resample.

    Args:
        clusters: One array per cluster; arrays are concatenated along axis 0.
        statistics: Named functions mapping the pooled array of a resample to a float.
        n_boot: Number of bootstrap resamples.
        seed: Seed of the resampling generator.
        confidence: Two-sided coverage of the intervals.

    Returns:
        For each name, ``(statistic on the original sample, lower, upper)``.

    Raises:
        ValueError: If ``clusters`` is empty.
    """
    n = len(clusters)
    if n == 0:
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    pooled = np.concatenate(list(clusters), axis=0)
    values = {name: float(fn(pooled)) for name, fn in statistics.items()}
    draws = {name: np.empty(n_boot) for name in statistics}
    for i in range(n_boot):
        pick = rng.integers(0, n, size=n)
        sample = np.concatenate([clusters[j] for j in pick], axis=0)
        for name, fn in statistics.items():
            draws[name][i] = fn(sample)
    return {name: (values[name], *_bounds(draws[name], confidence)) for name in statistics}


def cluster_bootstrap_ci(
    clusters: Sequence[np.ndarray],
    statistic: Callable[[np.ndarray], float],
    n_boot: int = 1_000,
    seed: int = 0,
    confidence: float = CONFIDENCE,
) -> tuple[float, float, float]:
    """Bootstrap one statistic of pooled observations by resampling whole clusters.

    Args:
        clusters: One array per cluster; arrays are concatenated along axis 0.
        statistic: Maps the pooled array of a resample to a float.
        n_boot: Number of bootstrap resamples.
        seed: Seed of the resampling generator.
        confidence: Two-sided coverage of the interval.

    Returns:
        ``(statistic on the original sample, lower, upper)``.
    """
    out = cluster_bootstrap_multi(clusters, {"s": statistic}, n_boot, seed, confidence)
    return out["s"]
