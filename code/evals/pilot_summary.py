"""Summarise a pilot run of ``run_tasks.py`` and project the cost of the full protocol.

Usage::

    uv run python pilot_summary.py ../results/raw/pilot --tok-per-s 2500 [--json summary.json]

For each task of the run directory it prints the items and epochs run, the mean output tokens per
sample, the total output tokens, truncations, request errors and accuracy (scored as in
``compare.py``: a request error or a truncated answer is a failure). Then it projects, for the
frozen protocol's item set (``PROTOCOL_ITEMS`` x ``PROTOCOL_EPOCHS``), the output tokens as the
pilot's mean tokens per sample times the number of samples, and the hours at the aggregate decode
throughput given by ``--tok-per-s`` (output tokens per second summed over all concurrent requests,
as read from the server log or a measurement at the same concurrency).

The projection is a linear extrapolation of the pilot's mean: it ignores prefill, scoring time
(code execution), the long tail of the slowest request and any difference between the pilot's
items and the full set, so read the hours as a planning figure, not a promise.
"""

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from compare import _per_item, load_run

PROTOCOL_ITEMS: dict[str, int] = {
    "mgsm": 550,
    "humaneval_plus": 164,
    "mbpp_plus": 378,
    "ifeval": 541,
    "aime2025": 30,
}
"""Items per task of the primary accuracy metric (PROTOCOL.md sections 4 and 6): MGSM 50 per
language over 11 languages, the full HumanEval+, MBPP+, IFEval and AIME 2025 sets."""

PROTOCOL_EPOCHS: dict[str, int] = {
    "mgsm": 1,
    "humaneval_plus": 1,
    "mbpp_plus": 1,
    "ifeval": 1,
    "aime2025": 5,
}
"""Samples per item (PROTOCOL.md section 6); ``run_tasks.py`` has the same defaults."""


def _task_stats(rows: list[dict[str, Any]], run_epochs: int | None) -> dict[str, Any]:
    """Statistics of the rows of one task (projection fields are filled by the caller)."""
    items = _per_item(rows)
    tokens = [r["output_tokens"] for r in rows if r.get("output_tokens") is not None]
    epochs = run_epochs or max(it["epochs"] for it in items.values())
    return {
        "items": len(items),
        "epochs": epochs,
        "samples": len(rows),
        "mean_output_tokens": float(np.mean(tokens)) if tokens else None,
        "total_output_tokens": int(sum(tokens)),
        "truncations": sum(bool(r.get("truncated")) for r in rows),
        "request_errors": sum(r.get("error") is not None for r in rows),
        "accuracy": float(np.mean([it["score"] for it in items.values()])),
        "projected_samples": None,
        "projected_output_tokens": None,
        "projected_hours": None,
    }


def summarise(run_dir: str | Path, tok_per_s: float | None = None) -> dict[str, Any]:
    """Per-task statistics of a run directory and the projection to the full protocol.

    Args:
        run_dir: Directory written by ``run_tasks.py``.
        tok_per_s: Aggregate decode throughput in output tokens per second, for the hours.

    Returns:
        ``{"run_dir", "status", "started_utc", "ended_utc", "tasks": {name: stats},
        "projection": {...}}``.

    Raises:
        ValueError: If ``tok_per_s`` is given and is not positive.
    """
    if tok_per_s is not None and tok_per_s <= 0:
        raise ValueError("--tok-per-s must be positive")
    rows, info = load_run(run_dir)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    tasks: dict[str, dict[str, Any]] = {}
    for name, task_rows in by_task.items():
        run_epochs = (info.get("tasks", {}).get(name) or {}).get("epochs")
        stats = _task_stats(task_rows, run_epochs)
        if name in PROTOCOL_ITEMS and stats["mean_output_tokens"] is not None:
            samples = PROTOCOL_ITEMS[name] * PROTOCOL_EPOCHS[name]
            stats["projected_samples"] = samples
            stats["projected_output_tokens"] = stats["mean_output_tokens"] * samples
            if tok_per_s:
                stats["projected_hours"] = stats["projected_output_tokens"] / tok_per_s / 3600
        tasks[name] = stats
    total = float(sum(t["projected_output_tokens"] or 0 for t in tasks.values()))
    projection = {
        "tok_per_s": tok_per_s,
        "output_tokens": total,
        "hours": total / tok_per_s / 3600 if tok_per_s else None,
        "samples": sum(t["projected_samples"] or 0 for t in tasks.values()),
        "missing": [t for t in PROTOCOL_ITEMS if t not in tasks],
        "not_projected": [t for t in tasks if t not in PROTOCOL_ITEMS],
    }
    return {
        "run_dir": str(run_dir),
        "status": info.get("status"),
        "started_utc": info.get("started_utc"),
        "ended_utc": info.get("ended_utc"),
        "tasks": tasks,
        "projection": projection,
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.0f}"


def render(summary: dict[str, Any]) -> str:
    """Render a summary as markdown.

    Args:
        summary: The result of ``summarise``.

    Returns:
        A table per task and the projection paragraph.
    """
    lines = [
        f"Pilot: {summary['run_dir']} (status {summary['status']}, "
        f"{summary['started_utc']} to {summary['ended_utc']})",
        "",
        "| Task | Items | Epochs | Samples | Mean output tokens / sample | Total output tokens "
        "| Truncations | Request errors | Accuracy | Projected output tokens (protocol) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, t in summary["tasks"].items():
        lines.append(
            f"| {name} | {t['items']} | {t['epochs']} | {t['samples']} "
            f"| {_fmt(t['mean_output_tokens'])} | {t['total_output_tokens']:,} "
            f"| {t['truncations']} | {t['request_errors']} | {100 * t['accuracy']:.1f}% "
            f"| {_fmt(t['projected_output_tokens'])} |"
        )
    p = summary["projection"]
    protocol = ", ".join(f"{k} {v} x {PROTOCOL_EPOCHS[k]}" for k, v in PROTOCOL_ITEMS.items())
    full = sum(v * PROTOCOL_EPOCHS[k] for k, v in PROTOCOL_ITEMS.items())
    lines += [
        "",
        f"Protocol item set: {protocol} = {full:,} samples.",
        f"Projected output tokens for the piloted protocol tasks: {_fmt(p['output_tokens'])} "
        f"({p['samples']:,} samples).",
    ]
    if p["hours"] is not None:
        lines.append(
            f"At {p['tok_per_s']:,.0f} tokens/s aggregate decode throughput: "
            f"{p['hours']:.2f} hours (linear extrapolation; prefill, scoring time and the tail "
            "of slow requests are not included)."
        )
    else:
        lines.append("No --tok-per-s given: hours not projected.")
    if p["missing"]:
        lines.append(
            f"NOT in the pilot, so NOT in the projection: {', '.join(p['missing'])}. "
            "The totals above are an undercount."
        )
    if p["not_projected"]:
        others = ", ".join(p["not_projected"])
        lines.append(f"Run but outside the protocol's primary set: {others}.")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Print the summary of a pilot run directory.

    Args:
        argv: Arguments (default: ``sys.argv[1:]``).

    Returns:
        ``0``.

    Raises:
        SystemExit: If the directory holds no run or the throughput is invalid.
    """
    ap = argparse.ArgumentParser(prog="pilot_summary.py", description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", help="result directory written by run_tasks.py")
    ap.add_argument("--tok-per-s", type=float, help="aggregate decode throughput, output tokens/s")
    ap.add_argument("--json", help="also write the summary to this file")
    args = ap.parse_args(argv)
    run = Path(args.run_dir)
    if not (run / "samples.jsonl").exists() or not (run / "run_info.json").exists():
        raise SystemExit(f"{run} has no samples.jsonl and run_info.json: not a run_tasks.py output")
    try:
        summary = summarise(run, args.tok_per_s)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(render(summary), end="")
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
