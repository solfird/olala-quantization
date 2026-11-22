"""Summarize the throughput sweeps (`vllm bench serve` JSON files, one folder per arm) into a markdown table.

Usage: python3 bench_summary.py --bench-dir ../results/bench --out ../results/bench-summary.md

Each arm folder `bench-LABEL/` holds `{short,long}_c{N}_r{K}.json` (two repeats per point). For every point the
median of the repeats of `output_throughput` (output tokens per second) is reported with the min-max range, next
to the same point of bf16 (the pooled median of the bf16 folders). Failed requests per point are listed and a point
with failures is flagged. `median_tpot_ms` is NOT reported: for the long workload it disagrees with the throughput
(0.4 ms per token against 230 tokens/s at concurrency 1), so its definition in this tool version is not trusted here.
No statistic beyond medians and ranges; differences smaller than the spread between the two independent bf16 pods
are marked "within pod spread".
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ARM_ORDER = ["bf16", "bf16-pod2", "F1", "F2", "N2", "N5"]
POINTS = [("short", 1), ("short", 16), ("short", 32), ("short", 64), ("short", 128), ("long", 1), ("long", 16), ("long", 32)]


def load_arm(folder: Path) -> dict[tuple[str, int], list[dict]]:
    """Result files of one arm: {(workload, concurrency): [json, ...]}."""
    pts: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for f in sorted(folder.glob("*.json")):
        name, conc, _rep = f.stem.split("_")
        pts[(name, int(conc[1:]))].append(json.loads(f.read_text()))
    return pts


def point_stats(runs: list[dict]) -> dict[str, float]:
    tps = [r["output_throughput"] for r in runs]
    return {
        "median": st.median(tps),
        "min": min(tps),
        "max": max(tps),
        "ttft_ms": st.median(r["median_ttft_ms"] for r in runs),
        "completed": sum(r["completed"] for r in runs),
        "failed": sum(r.get("failed", 0) for r in runs),
        "n": len(runs),
    }


def summarize(bench_dir: Path) -> str:
    arms = {}
    for label in ARM_ORDER:
        folder = bench_dir / f"bench-{label}"
        if folder.is_dir():
            arms[label] = {k: point_stats(v) for k, v in load_arm(folder).items()}
    bf = [a for a in ("bf16", "bf16-pod2") if a in arms]
    lines = ["# Throughput sweeps (output tokens per second, synthetic random-token workload, RTX PRO 4500)", ""]
    lines.append("Median of the repeats per point (min-max). `x bf16` = ratio to the mean of the bf16 medians; "
                 "`~` marks a ratio inside the spread of the two bf16 pods at that point. Failed requests: 0 unless listed.")
    lines.append("")
    head = "| arm | " + " | ".join(f"{w} c={c}" for w, c in POINTS) + " |"
    lines += [head, "|" + "---|" * (len(POINTS) + 1)]
    ref = {p: st.mean(arms[a][p]["median"] for a in bf) for p in POINTS if all(p in arms[a] for a in bf)}
    spread = {p: abs(arms[bf[0]][p]["median"] - arms[bf[1]][p]["median"]) / ref[p] for p in ref} if len(bf) == 2 else {}
    for label, d in arms.items():
        cells = []
        for p in POINTS:
            s = d.get(p)
            if not s:
                cells.append("n/a")
                continue
            cell = f"{s['median']:.0f} ({s['min']:.0f}-{s['max']:.0f})"
            if label not in ("bf16", "bf16-pod2") and p in ref:
                ratio = s["median"] / ref[p]
                near = "~" if abs(ratio - 1) <= spread.get(p, 0) else ""
                cell += f" x{ratio:.2f}{near}"
            if s["failed"]:
                cell += f" **{s['failed']} failed**"
            cells.append(cell)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "bf16 pod-to-pod spread at each point (|pod1 - pod2| / mean): "
              + ", ".join(f"{w} c={c} {spread[(w, c)] * 100:.1f}%" for w, c in POINTS if (w, c) in spread)]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.write_text(summarize(a.bench_dir))
    print(a.out.read_text())


if __name__ == "__main__":
    main()
