"""What the result rows say about retried requests and working time (one definition for all tools).

Since 2026-11-21 ``run_tasks.py`` writes into every row of ``samples.jsonl``

* ``model_attempts``: the HTTP requests the sample needed (1 if the first one was answered);
* ``working_time``: seconds of the sample's working time as inspect_ai measures it (may be null),

and the per-task and total counters of ``run_info.json`` are computed from the rows by
:func:`retry_stats`. Rows of runs made before that carry neither field (legacy rows): they say
nothing about retries and are never counted as zero.
"""

from collections.abc import Iterable, Mapping
from typing import Any


def has_attempts(row: Mapping[str, Any]) -> bool:
    """Whether the row records its ``model_attempts`` (a legacy row does not)."""
    value = row.get("model_attempts")
    return isinstance(value, int) and not isinstance(value, bool)


def is_retried(row: Mapping[str, Any]) -> bool:
    """Whether the row records more than one request for its sample."""
    return has_attempts(row) and row["model_attempts"] > 1


def retry_stats(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Counters of ``run_info.json`` computed from rows.

    Args:
        rows: Rows of ``samples.jsonl`` (of one task, or of the whole run).

    Returns:
        ``samples_with_retries`` (rows with ``model_attempts`` above 1) if any row carries
        ``model_attempts``, and ``max_working_time`` (the largest non-null ``working_time``, null
        if there is none) if any row carries ``working_time``. A key is absent when no row carries
        the field it comes from: legacy rows are not counted as zero.
    """
    rows = list(rows)
    out: dict[str, Any] = {}
    if any(has_attempts(r) for r in rows):
        out["samples_with_retries"] = sum(is_retried(r) for r in rows)
    if any("working_time" in r for r in rows):
        times = [r["working_time"] for r in rows if isinstance(r.get("working_time"), int | float)]
        out["max_working_time"] = max(times) if times else None
    return out


def run_client_timeout(tasks: Mapping[str, Mapping[str, Any]]) -> float | None:
    """The client timeout of a whole run: the smallest of its tasks, if every task states one.

    Args:
        tasks: The ``tasks`` block of ``run_info.json``.

    Returns:
        The smallest ``client_timeout``, or ``None`` if there is no task or one has none (a legacy
        task ran with the provider's 600 s default, which no run-wide value can describe).
    """
    values = [t.get("client_timeout") for t in tasks.values()]
    if not values or any(not isinstance(v, int | float) or isinstance(v, bool) for v in values):
        return None
    return min(values)


def apply_run_counters(info: dict[str, Any], rows: Iterable[Mapping[str, Any]]) -> None:
    """Set the run-wide ``client_timeout``, ``samples_with_retries`` and ``max_working_time``.

    The three keys are recomputed from the rows and the ``tasks`` block of ``info`` (already
    final): a stale value copied from a source run is removed, never kept.

    Args:
        info: ``run_info.json`` content, changed in place.
        rows: Every row of the run.
    """
    for key in ("client_timeout", "samples_with_retries", "max_working_time"):
        info.pop(key, None)
    info.update(retry_stats(rows))
    timeout = run_client_timeout(info.get("tasks") or {})
    if timeout is not None:
        info["client_timeout"] = timeout
