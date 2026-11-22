"""Pure numpy maths for the logit-fidelity comparison (no I/O, no network).

Every function works on *top-k truncated* next-token distributions, as returned by an
OpenAI-compatible ``/v1/completions`` endpoint with ``logprobs=k``:

* ``ids``: ``int`` array ``[N, K]`` of token identifiers, ``-1`` marks padding;
* ``lp``: ``float`` array ``[N, K]`` of natural-log probabilities, ``-inf`` marks padding.

KL divergence approximation (see DECISIONS.md for the discussion)
-----------------------------------------------------------------
For one position let ``R`` be the reference top-k tokens and ``V`` the variant top-k tokens.
The divergence ``KL(P_ref || Q_var)`` is evaluated on the union ``R | V`` plus one bucket
``other`` that lumps every remaining token. Probabilities that the truncated output does not
reveal are imputed:

* a token of the union that is missing from a distribution's top-k list gets
  ``min(p_k, unseen / n_missing)`` where ``p_k`` is the smallest probability that
  distribution lists (an upper bound for any unlisted token) and ``unseen = 1 - sum(listed)``
  is the mass the list does not account for;
* ``other`` receives what is left of ``unseen`` after the imputation.

Both distributions are renormalised over the union plus ``other`` so they sum to one, which
keeps the divergence non-negative.
"""

import numpy as np

EPS = 1e-12
"""Floor applied to the variant probability inside a logarithm."""

UNSEEN_TOL = 1e-6
"""Unseen mass below this is treated as zero (rounding noise of the printed log-probs)."""

_BLOCK = 4096
"""Rows processed at once; bounds the ``[rows, K, K]`` match tensor."""


def _unseen(mass: np.ndarray) -> np.ndarray:
    """Mass not accounted for by the listed tokens, with rounding noise removed."""
    rest = 1.0 - mass
    return np.where(rest < UNSEEN_TOL, 0.0, rest)


def _impute(floor: np.ndarray, unseen: np.ndarray, n_missing: np.ndarray) -> np.ndarray:
    """Per-token probability given to the tokens a list does not show."""
    share = unseen / np.maximum(n_missing, 1)
    return np.where(n_missing > 0, np.minimum(floor, share), 0.0)


def _kl_block(
    ref_ids: np.ndarray, ref_lp: np.ndarray, var_ids: np.ndarray, var_lp: np.ndarray
) -> np.ndarray:
    """KL for one block of rows; see :func:`kl_topk_union`."""
    valid_r = (ref_ids >= 0) & np.isfinite(ref_lp)
    valid_v = (var_ids >= 0) & np.isfinite(var_lp)
    p = np.where(valid_r, np.exp(np.where(valid_r, ref_lp, 0.0)), 0.0)
    q = np.where(valid_v, np.exp(np.where(valid_v, var_lp, 0.0)), 0.0)

    match = (ref_ids[:, :, None] == var_ids[:, None, :]) & valid_r[:, :, None] & valid_v[:, None, :]
    in_var = match.any(axis=2)
    in_ref = match.any(axis=1)
    matched = match.astype(np.float64)
    q_of_ref = np.einsum("nij,nj->ni", matched, q)
    ref_only = valid_r & ~in_var
    var_only = valid_v & ~in_ref
    n_ref_only = ref_only.sum(axis=1)
    n_var_only = var_only.sum(axis=1)

    mass_p, mass_q = p.sum(axis=1), q.sum(axis=1)
    unseen_p, unseen_q = _unseen(mass_p), _unseen(mass_q)
    floor_p = np.where(valid_r, p, np.inf).min(axis=1, initial=np.inf)
    floor_q = np.where(valid_v, q, np.inf).min(axis=1, initial=np.inf)
    floor_p = np.where(np.isfinite(floor_p), floor_p, 0.0)
    floor_q = np.where(np.isfinite(floor_q), floor_q, 0.0)
    impute_q = _impute(floor_q, unseen_q, n_ref_only)  # variant prob of reference-only tokens
    impute_p = _impute(floor_p, unseen_p, n_var_only)  # reference prob of variant-only tokens
    other_p = unseen_p - n_var_only * impute_p
    other_q = unseen_q - n_ref_only * impute_q
    norm_p = mass_p + n_var_only * impute_p + other_p
    norm_q = mass_q + n_ref_only * impute_q + other_q

    p_ref = p / norm_p[:, None]
    q_ref = np.where(in_var, q_of_ref, impute_q[:, None]) / norm_q[:, None]
    term_ref = np.where(
        valid_r & (p_ref > 0),
        p_ref * np.log(np.where(p_ref > 0, p_ref, 1.0) / np.maximum(q_ref, EPS)),
        0.0,
    )

    p_var = (impute_p / norm_p)[:, None]
    q_var = q / norm_q[:, None]
    term_var = np.where(
        var_only & (p_var > 0),
        p_var * np.log(np.where(p_var > 0, p_var, 1.0) / np.maximum(q_var, EPS)),
        0.0,
    )

    po, qo = other_p / norm_p, other_q / norm_q
    term_other = np.where(po > 0, po * np.log(np.where(po > 0, po, 1.0) / np.maximum(qo, EPS)), 0.0)
    return np.maximum(term_ref.sum(axis=1) + term_var.sum(axis=1) + term_other, 0.0)


def kl_topk_union(
    ref_ids: np.ndarray, ref_lp: np.ndarray, var_ids: np.ndarray, var_lp: np.ndarray
) -> np.ndarray:
    """Per-position ``KL(P_ref || Q_var)`` in nats on the union of the two top-k lists.

    Args:
        ref_ids: Reference token ids ``[N, K1]`` (``-1`` = padding).
        ref_lp: Reference natural-log probabilities ``[N, K1]`` (``-inf`` = padding).
        var_ids: Variant token ids ``[N, K2]``, in the same id space as ``ref_ids``.
        var_lp: Variant natural-log probabilities ``[N, K2]``.

    Returns:
        ``float64`` array ``[N]`` of non-negative divergences. This is an approximation:
        see the module docstring.
    """
    ref_ids, var_ids = np.asarray(ref_ids), np.asarray(var_ids)
    ref_lp = np.asarray(ref_lp, dtype=np.float64)
    var_lp = np.asarray(var_lp, dtype=np.float64)
    out = np.empty(ref_ids.shape[0], dtype=np.float64)
    for start in range(0, ref_ids.shape[0], _BLOCK):
        sl = slice(start, start + _BLOCK)
        out[sl] = _kl_block(ref_ids[sl], ref_lp[sl], var_ids[sl], var_lp[sl])
    return out


def perplexity(target_lp: np.ndarray) -> float:
    """Perplexity ``exp(-mean(log p(target)))`` of the actual next tokens.

    Args:
        target_lp: Natural-log probability of each actual token, any shape.

    Returns:
        The perplexity as a float.
    """
    return float(np.exp(-np.mean(np.asarray(target_lp, dtype=np.float64))))


def top1_agreement(
    ref_ids: np.ndarray, ref_lp: np.ndarray, var_ids: np.ndarray, var_lp: np.ndarray
) -> np.ndarray:
    """Boolean ``[N]``: do both distributions put their highest probability on the same token?

    Args:
        ref_ids: Reference token ids ``[N, K1]``.
        ref_lp: Reference log-probabilities ``[N, K1]``.
        var_ids: Variant token ids ``[N, K2]``.
        var_lp: Variant log-probabilities ``[N, K2]``.

    Returns:
        Boolean array, ``True`` where the arg-max tokens are identical.
    """
    rows = np.arange(np.asarray(ref_ids).shape[0])
    ref_top = np.asarray(ref_ids)[rows, np.asarray(ref_lp).argmax(axis=1)]
    var_top = np.asarray(var_ids)[rows, np.asarray(var_lp).argmax(axis=1)]
    return ref_top == var_top
