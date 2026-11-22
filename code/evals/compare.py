"""Compare two run_tasks.py result directories with a paired bootstrap; print a markdown table.

Usage::

    uv run python compare.py ../results/bf16 ../results/fp8 [--by-stratum] [--out report.md]

Scoring rules, all applied here and nowhere else:

* the score of one (item, epoch) row is its scorer value; a request error or a truncated
  answer counts as 0 (a failure), whatever the scorer said;
* the score of an item is the mean over its epochs (pass@1 averaged over samples);
* accuracy is the mean over the items present in *both* runs; intervals are percentile
  bootstrap intervals over items, paired for the difference.
"""

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from stats import bootstrap_mean_ci, paired_bootstrap_diff_ci

COMPARED_SETTINGS = (
    "model",
    "inspect_model",
    "endpoint",
    "sampling",
    "reasoning_effort",
    "chat_template_kwargs",
    "seed",
    "max_connections",
    "versions",
    "git",
    "code_sha256",
)
"""Top-level run_info.json keys whose differences are reported."""

LIST_LIMIT = 20


def load_run(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read ``samples.jsonl`` and ``run_info.json`` of a result directory.

    Args:
        path: Directory written by ``run_tasks.py``.

    Returns:
        ``(rows, run_info)``.
    """
    path = Path(path)
    rows = [json.loads(line) for line in (path / "samples.jsonl").read_text().splitlines() if line]
    info = json.loads((path / "run_info.json").read_text())
    return rows, info


def effective_score(row: dict[str, Any]) -> float:
    """Score of one row with request errors and truncations counted as failures.

    Args:
        row: A row of ``samples.jsonl``.

    Returns:
        ``0.0`` for an errored, truncated or unscored row, otherwise the scorer value.
    """
    if row.get("error") is not None or row.get("truncated") or row.get("score") is None:
        return 0.0
    return float(row["score"])


def _flatten(prefix: str, value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            out.update(_flatten(f"{prefix}.{k}" if prefix else str(k), v))
        return out
    return {prefix: value}


def setting_differences(info_a: dict[str, Any], info_b: dict[str, Any]) -> list[dict[str, Any]]:
    """List the compared run settings that differ between two ``run_info.json`` records."""
    fa = _flatten("", {k: info_a.get(k) for k in COMPARED_SETTINGS})
    fb = _flatten("", {k: info_b.get(k) for k in COMPARED_SETTINGS})
    return [
        {"key": key, "a": fa.get(key), "b": fb.get(key)}
        for key in sorted(set(fa) | set(fb))
        if fa.get(key) != fb.get(key)
    ]


def _per_item(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group the rows of one task by item: mean effective score, stratum, epochs."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[r["sample_id"]].append(r)
    return {
        sid: {
            "score": float(np.mean([effective_score(r) for r in rs])),
            "stratum": rs[0].get("stratum"),
            "epochs": len(rs),
        }
        for sid, rs in grouped.items()
    }


def _failures(rows: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, list[str]]]:
    errors = sorted({r["sample_id"] for r in rows if r.get("error") is not None})
    truncs = sorted({r["sample_id"] for r in rows if r.get("truncated") and r.get("error") is None})
    counts = {
        "request_errors": sum(r.get("error") is not None for r in rows),
        "truncations": sum(bool(r.get("truncated")) for r in rows),
        "rows": len(rows),
    }
    return counts, {"request_errors": errors, "truncations": truncs}


def _interval(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> dict[str, Any]:
    ma, la, ha = bootstrap_mean_ci(a, n_boot=n_boot, seed=seed)
    mb, lb, hb = bootstrap_mean_ci(b, n_boot=n_boot, seed=seed)
    md, ld, hd = paired_bootstrap_diff_ci(a, b, n_boot=n_boot, seed=seed)
    return {
        "acc_a": {"value": ma, "lo": la, "hi": ha},
        "acc_b": {"value": mb, "lo": lb, "hi": hb},
        "diff": {"value": md, "lo": ld, "hi": hd},
    }


def compare_runs(
    a_dir: str | Path, b_dir: str | Path, n_boot: int = 10_000, seed: int = 0
) -> dict[str, Any]:
    """Compare two result directories task by task.

    Args:
        a_dir: Reference run (e.g. bf16).
        b_dir: Variant run (e.g. a quantized model); differences are ``B - A``.
        n_boot: Bootstrap resamples over items.
        seed: Bootstrap seed.

    Returns:
        A JSON-serialisable dict with per-task accuracies, intervals, failure counts,
        unpaired-item counts, setting differences and an optional per-stratum breakdown.
    """
    rows_a, info_a = load_run(a_dir)
    rows_b, info_b = load_run(b_dir)
    tasks_a = sorted({r["task"] for r in rows_a})
    tasks_b = sorted({r["task"] for r in rows_b})
    result: dict[str, Any] = {
        "a_dir": str(a_dir),
        "b_dir": str(b_dir),
        "n_boot": n_boot,
        "seed": seed,
        "confidence": 0.95,
        "only_in_a": sorted(set(tasks_a) - set(tasks_b)),
        "only_in_b": sorted(set(tasks_b) - set(tasks_a)),
        "info_a": {k: info_a.get(k) for k in ("model", "endpoint", "reasoning_effort")},
        "info_b": {k: info_b.get(k) for k in ("model", "endpoint", "reasoning_effort")},
        "setting_differences": setting_differences(info_a, info_b),
        "tasks": {},
    }
    for task in sorted(set(tasks_a) & set(tasks_b)):
        ta = [r for r in rows_a if r["task"] == task]
        tb = [r for r in rows_b if r["task"] == task]
        ia, ib = _per_item(ta), _per_item(tb)
        common = sorted(set(ia) & set(ib))
        va = np.array([ia[i]["score"] for i in common])
        vb = np.array([ib[i]["score"] for i in common])
        counts_a, failed_a = _failures(ta)
        counts_b, failed_b = _failures(tb)
        sha_a = (info_a.get("tasks", {}).get(task) or {}).get("items_sha256")
        sha_b = (info_b.get("tasks", {}).get(task) or {}).get("items_sha256")
        entry: dict[str, Any] = {
            "n_items": len(common),
            "only_a": len(set(ia) - set(ib)),
            "only_b": len(set(ib) - set(ia)),
            "epochs_a": max(v["epochs"] for v in ia.values()),
            "epochs_b": max(v["epochs"] for v in ib.values()),
            "same_items_hash": None if sha_a is None or sha_b is None else sha_a == sha_b,
            "failures_a": counts_a,
            "failures_b": counts_b,
            "failed_ids_a": failed_a,
            "failed_ids_b": failed_b,
            **_interval(va, vb, n_boot, seed),
        }
        labels = {ia[i]["stratum"] for i in common if ia[i]["stratum"] is not None}
        if labels:
            entry["strata"] = {}
            for label in sorted(labels):
                idx = [k for k, i in enumerate(common) if ia[i]["stratum"] == label]
                entry["strata"][label] = {
                    "n_items": len(idx),
                    **_interval(va[idx], vb[idx], n_boot, seed),
                }
        result["tasks"][task] = entry
    return result


def _ci(m: dict[str, float], fmt: str = ".1f") -> str:
    return f"{100 * m['value']:{fmt}} [{100 * m['lo']:{fmt}}, {100 * m['hi']:{fmt}}]"


def _excludes_zero(m: dict[str, float]) -> str:
    return "yes" if m["lo"] > 0 or m["hi"] < 0 else "no"


def render_markdown(result: dict[str, Any], by_stratum: bool = False) -> str:
    """Render :func:`compare_runs` output as markdown (every number comes from ``result``).

    Args:
        result: Output of :func:`compare_runs`.
        by_stratum: Also print per-stratum rows (e.g. MGSM languages) for stratified tasks.

    Returns:
        Markdown text.
    """
    lines = [
        "## Comparison of two runs",
        "",
        f"- A (reference): `{result['a_dir']}`, model {result['info_a']['model']}, "
        f"endpoint {result['info_a']['endpoint']}",
        f"- B (variant): `{result['b_dir']}`, model {result['info_b']['model']}, "
        f"endpoint {result['info_b']['endpoint']}",
        "- Accuracy is pass@1 averaged over the epochs of each item, in percent. A request error "
        "or a truncated answer counts as a failure (0).",
        f"- Intervals: {result['confidence']:.0%} percentile bootstrap over items "
        f"({result['n_boot']} resamples, seed {result['seed']}); the difference is paired "
        "(B minus A, in percentage points) over the items present in both runs.",
        "",
        "| Task | Paired items | Epochs A/B | A accuracy % [CI] | B accuracy % [CI] "
        "| B minus A, pp [CI] | CI excludes 0 |",
        "|---|---|---|---|---|---|---|",
    ]
    tasks = result["tasks"]
    for name, t in tasks.items():
        lines.append(
            f"| {name} | {t['n_items']} | {t['epochs_a']}/{t['epochs_b']} | {_ci(t['acc_a'])} "
            f"| {_ci(t['acc_b'])} | {_ci(t['diff'], '+.1f')} | {_excludes_zero(t['diff'])} |"
        )

    lines += [
        "",
        "### Request errors and truncations (already counted as failures above)",
        "",
        "| Task | Rows A | Request errors (A) | Truncations (A) "
        "| Rows B | Request errors (B) | Truncations (B) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, t in tasks.items():
        fa, fb = t["failures_a"], t["failures_b"]
        lines.append(
            f"| {name} | {fa['rows']} | {fa['request_errors']} | {fa['truncations']} "
            f"| {fb['rows']} | {fb['request_errors']} | {fb['truncations']} |"
        )
    listed = False
    for name, t in tasks.items():
        for side, key in (("A", "failed_ids_a"), ("B", "failed_ids_b")):
            for kind, label in (
                ("request_errors", "Request errors"),
                ("truncations", "Truncations"),
            ):
                ids = t[key][kind]
                if ids:
                    if not listed:
                        lines += ["", f"Items concerned (first {LIST_LIMIT} per list):", ""]
                        listed = True
                    more = f" (+{len(ids) - LIST_LIMIT} more)" if len(ids) > LIST_LIMIT else ""
                    lines.append(
                        f"- {name}, {label} in {side}: {', '.join(ids[:LIST_LIMIT])}{more}"
                    )

    warnings = []
    for name, t in tasks.items():
        if t["same_items_hash"] is False:
            warnings.append(f"{name}: different item sets in the two runs (items_sha256 differs)")
        if t["only_a"] or t["only_b"]:
            warnings.append(
                f"{name}: {t['only_a']} item(s) only in A and {t['only_b']} only in B "
                "are not paired"
            )
        if t["epochs_a"] != t["epochs_b"]:
            warnings.append(
                f"{name}: different number of epochs ({t['epochs_a']} vs {t['epochs_b']})"
            )
    for label, key in (("A", "only_in_a"), ("B", "only_in_b")):
        if result[key]:
            warnings.append(f"tasks only in {label} (not compared): {', '.join(result[key])}")
    if warnings:
        lines += ["", "### Warnings", ""] + [f"- {w}" for w in warnings]

    lines += ["", "### Settings that differ between the runs", ""]
    diffs = result["setting_differences"]
    if diffs:
        lines += ["| Setting | A | B |", "|---|---|---|"]
        lines += [f"| {d['key']} | {d['a']} | {d['b']} |" for d in diffs]
    else:
        lines.append("None of the compared settings differ.")

    if by_stratum:
        for name, t in tasks.items():
            if "strata" not in t:
                continue
            lines += [
                "",
                f"### {name} by stratum",
                "",
                "| Stratum | Paired items | A accuracy % [CI] | B accuracy % [CI] "
                "| B minus A, pp [CI] |",
                "|---|---|---|---|---|",
            ]
            for label, s in t["strata"].items():
                lines.append(
                    f"| {label} | {s['n_items']} | {_ci(s['acc_a'])} | {_ci(s['acc_b'])} "
                    f"| {_ci(s['diff'], '+.1f')} |"
                )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Arguments (default: ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    ap = argparse.ArgumentParser(prog="compare.py", description=__doc__.split("\n")[0])
    ap.add_argument("a_dir", help="reference run directory")
    ap.add_argument("b_dir", help="variant run directory")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--by-stratum", action="store_true", help="per-language rows for MGSM")
    ap.add_argument("--out", help="also write the markdown here")
    ap.add_argument("--json", help="also write the full result as JSON")
    args = ap.parse_args(argv)
    result = compare_runs(args.a_dir, args.b_dir, n_boot=args.n_boot, seed=args.seed)
    md = render_markdown(result, by_stratum=args.by_stratum)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
