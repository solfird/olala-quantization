"""Summarize the llama.cpp throughput sweeps of the Q4 stand-in and OVH's Q5_K_M (amendment 24), valid points only.

    python3 analysis/gguf_bench_summary.py --out results/bench-gguf-summary.md --json results/bench-gguf-summary.json \
        --sweep Q4sw2=results/wave1 [--sweep Q4-a=results/wave1]

Each sweep is a label LABEL; its folders are `bench-LABEL-VARIANT/` with `c<C>_r<K>.json` (vllm bench serve output, completions
endpoint). A repeat with any failed request is EXCLUDED (the pre-registered rule: its output tokens are not counted, so its
throughput is biased low) and listed. For every variant and concurrency the median of the valid repeats over all sweeps is reported
with the number of valid repeats, then the ratio Q4 / Q5 where both have at least one valid repeat. Same engine, same flags, same card
type: the ratio compares the two files, not engines. Not comparable in absolute terms with the vLLM sweeps.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

VARIANTS = {"q4_k_m-ovhrecipe": "Q4_K_M (stand-in)", "q5_k_m": "Q5_K_M (OVH)"}
CONCS = [1, 4, 12]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="append", required=True, help="LABEL=DIR holding bench-LABEL-VARIANT folders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    points: dict[tuple[str, int], list[dict]] = {}
    excluded, missing = [], []
    for spec in args.sweep:
        label, _, root = spec.partition("=")
        for variant in VARIANTS:
            folder = Path(root) / f"bench-{label}-{variant}"
            if not folder.is_dir():
                missing.append(str(folder))
                continue
            for f in sorted(folder.glob("c*_r*.json")):
                d = json.loads(f.read_text())
                c = int(f.name.split("_")[0][1:])
                rec = {"sweep": label, "file": f.name, "completed": d["completed"], "failed": d["failed"],
                       "tok_s": d["output_throughput"], "tokens": d["total_output_tokens"]}
                if d["failed"]:
                    excluded.append({"variant": variant, "c": c, **rec})
                else:
                    points.setdefault((variant, c), []).append(rec)

    rows, result = [], {}
    for variant, name in VARIANTS.items():
        for c in CONCS:
            v = points.get((variant, c), [])
            result[f"{variant}:{c}"] = {"valid": len(v), "median": st.median([p["tok_s"] for p in v]) if v else None,
                                       "values": [p["tok_s"] for p in v]}
    lines = ["# llama.cpp throughput of the Q4 stand-in and OVH's Q5_K_M (output tokens per second)", "",
             "Synthetic random-token requests, 512 in and 512 out, `--ignore-eos`, one RTX PRO 4500 card per sweep, 12 server slots. "
             "A repeat with any failed request is excluded. Values are the median of the valid repeats (their number in brackets).", "",
             "| Concurrent requests | " + " | ".join(VARIANTS.values()) + " | Q4 / Q5 |", "|---|---|---|---|"]
    ratios = {}
    for c in CONCS:
        cells = []
        for variant in VARIANTS:
            r = result[f"{variant}:{c}"]
            cells.append(f"{r['median']:.0f} ({r['valid']})" if r["median"] else "no valid repeat")
        q4, q5 = result[f"q4_k_m-ovhrecipe:{c}"]["median"], result[f"q5_k_m:{c}"]["median"]
        ratio = (q4 / q5) if q4 and q5 else None
        ratios[c] = ratio
        lines.append(f"| {c} | " + " | ".join(cells) + f" | {ratio:.2f} |" if ratio else f"| {c} | " + " | ".join(cells) + " | n/a |")
    lines += ["", "## Excluded repeats (failed requests)", ""]
    lines += [f"- {e['variant']} c={e['c']} {e['file']} (sweep {e['sweep']}): {e['failed']} of {e['completed'] + e['failed']} requests failed, throughput {e['tok_s']:.0f} not used"
              for e in excluded] or ["none"]
    if missing:
        lines += ["", "## Missing sweep folders", ""] + [f"- {m}" for m in missing]
    Path(args.out).write_text("\n".join(lines) + "\n")
    Path(args.json).write_text(json.dumps({"result": result, "ratios": ratios, "excluded": excluded, "missing": missing}, indent=1))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
