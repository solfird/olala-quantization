r"""Audit the nature of truncated answers (stop_reason max_tokens) in the raw inspect logs of every arm.

THIS SCRIPT DOCUMENTS HOW THE AUDIT WAS MADE. IT NEEDS THE HELD-BACK RAW INSPECT LOGS (``results/wave1/<run>/logs/*.eval``), WHICH THIS REPOSITORY DOES NOT
HOLD (they embed the benchmark prompts under their own licences and are gigabytes). Its output is shipped: ``results/truncation-audit.json`` and ``.md``.
With the logs put back in place and ``inspect_ai`` installed (``uv sync`` in ``code/evals``):

    cd code/evals && uv run python ../../analysis/truncation_audit.py --out ../../results/truncation-audit.md --json ../../results/truncation-audit.json

For every truncated sample the reasoning text is read from the .eval log (the harness strips it from samples.jsonl) and four things
are measured, none of them a judgement call:

* ``compress``: zlib size / raw size of the last 8,000 characters of the reasoning. Natural reasoning compresses to about 0.3 to 0.45; text
  that repeats itself compresses far lower. A sample is called a LOOP when ``compress`` is below LOOP_THRESHOLD (0.12) OR when fewer than
  LOOP_UNIQUE (0.25) of the last 200 non-empty lines are distinct. Otherwise it is LONG: still producing new text when the cap cut it off.
* ``target_in``: the correct answer (as a whole token) appears anywhere in the reasoning; ``target_late``: in its last quarter.
* ``chars``: length of the reasoning.
* ``text_chars``: length of the visible answer text (0 for nearly every truncated sample: the cap fell inside the reasoning).

The thresholds were set after looking at the compression of the truncated samples of the two bf16 runs and of Q4-a (2026-11-21), where loops compress to 0.004 to
0.01 and natural reasoning to 0.30 or more. Over all ten builds (before B0g) the gap is narrower (loops up to 0.097, everything else from 0.131), and 65 of the 424 truncated
samples have a compression between 0.05 and 0.25. The share of answers still producing text is 71.5 to 72.9 percent (about 72) for any cutoff from 0.05 to 0.15, 68 percent
at 0.20 and 58 percent at 0.25, so the split is stable below 0.15 and not above it. (The first version of this note said that any cutoff from 0.05 to 0.25 gives the
same split. That was true of the samples looked at first and false over all builds; corrected 2026-11-22. The calibrated build G1 was added to RUNS on 2026-11-22: the nine
earlier rows are unchanged; with nine builds the figures were 378 truncated samples, 53 between 0.05 and 0.25, 70.9 to 72.5 percent for cutoffs 0.05 to 0.15, 69 and 59 percent.
The same-engine control B0g, OVH's bf16 GGUF on llama.cpp, was added to RUNS on 2026-11-22: the ten earlier rows and their per-sample records are unchanged (checked sample by sample);
with eleven arms the figures are 458 truncated samples, 71 between 0.05 and 0.25 (loops still up to 0.097, everything else from 0.131), 71.0 to 72.3 percent for cutoffs
0.05 to 0.15, 68 and 57 percent at 0.20 and 0.25.)
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics as st
import zlib
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log

RUNS = {
    "bf16 (run a)": ["B0a-v2"], "bf16 (run b)": ["B0b-v2"], "FP8 online": ["F1-v2"], "FP8 offline": ["F2-v2"],
    "NVFP4 all linear (N2)": ["N2-v2"], "NVFP4 experts only (N5)": ["N5-v2"], "NVFP4 C2 (this build)": ["SEL-v2"],
    "OVH Q5_K_M": ["Q5v2-mgsm-he", "Q5v2-ifeval-mbpp", "Q5v2-aime"], "Q4_K_M stand-in": ["Q4-a", "Q4-b", "Q4-c"],
    "NVFP4 experts, calibrated (G1)": ["G1-v2"],
    "bf16 GGUF (llama.cpp)": ["B0g-mgsm-he", "B0g-ifeval-mbpp", "B0g-aime"],
}
LOOP_THRESHOLD, LOOP_UNIQUE = 0.12, 0.25


def measure(reasoning: str, text: str, target: str) -> dict:
    tail = reasoning[-8000:]
    comp = len(zlib.compress(tail.encode(), 6)) / max(1, len(tail.encode()))
    lines = [ln.strip() for ln in reasoning.splitlines() if ln.strip()][-200:]
    uniq = len(set(lines)) / max(1, len(lines))
    tok = re.compile(rf"(?<![\w.]){re.escape(str(target).strip())}(?![\w.])") if str(target).strip() else None
    return {"chars": len(reasoning), "text_chars": len(text), "compress": comp, "unique_lines": uniq,
            "target_in": bool(tok and tok.search(reasoning)), "target_late": bool(tok and tok.search(reasoning[int(len(reasoning) * 0.75):]))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", default=str(Path(__file__).resolve().parents[1] / "results" / "wave1"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()
    per: dict[str, list[dict]] = defaultdict(list)
    totals: dict[str, int] = defaultdict(int)
    for arm, dirs in RUNS.items():
        for d in dirs:
            for f in sorted(glob.glob(f"{args.wave}/{d}/logs/*.eval")):
                log = read_eval_log(f)
                task = (log.eval.task or "").replace("-", "_")
                for s in log.samples or []:
                    totals[arm] += 1
                    out = s.output
                    if not out or out.stop_reason != "max_tokens" or not out.choices:
                        continue
                    msg = out.choices[0].message
                    blocks = msg.content if isinstance(msg.content, list) else []
                    reasoning = "".join(getattr(b, "reasoning", "") or "" for b in blocks if getattr(b, "type", "") == "reasoning")
                    text = "".join(getattr(b, "text", "") or "" for b in blocks if getattr(b, "type", "") == "text") if blocks else (msg.content or "")
                    m = measure(reasoning, text, str(s.target if not isinstance(s.target, list) else s.target[0]))
                    m.update({"task": task, "sample_id": str(s.id), "tokens": out.usage.output_tokens if out.usage else None})
                    per[arm].append(m)

    def is_loop(m: dict, thr: float = LOOP_THRESHOLD) -> bool:
        return m["compress"] < thr or m["unique_lines"] < LOOP_UNIQUE

    lines = ["# Nature of the truncated answers (stop_reason max_tokens)", "",
             f"A truncated sample is called a LOOP when the last 8,000 characters of its reasoning compress below {LOOP_THRESHOLD} (zlib) or fewer than "
             f"{LOOP_UNIQUE:.0%} of its last 200 lines are distinct; otherwise LONG (still producing new text). Counts at 0.10 and 0.15 are in the JSON.", "",
             "| Arm | Samples | Truncated | Loops | Long | Visible answer text | Correct answer in the reasoning | Correct answer in its last quarter | Median reasoning (k chars) |",
             "|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for arm in RUNS:
        t = per.get(arm, [])
        n = len(t)
        loops = sum(is_loop(m) for m in t)
        row = {"samples": totals[arm], "truncated": n, "loops": loops, "long": n - loops,
               "with_text": sum(m["text_chars"] > 0 for m in t), "target_in": sum(m["target_in"] for m in t), "target_late": sum(m["target_late"] for m in t),
               "median_chars": int(st.median([m["chars"] for m in t])) if t else 0,
               "loops_at_010": sum(is_loop(m, 0.10) for m in t), "loops_at_015": sum(is_loop(m, 0.15) for m in t)}
        summary[arm] = row
        lines.append(f"| {arm} | {row['samples']} | {n} | {loops} | {n - loops} | {row['with_text']} | {row['target_in']} | {row['target_late']} | {row['median_chars'] / 1000:.0f} |")
    by_task: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for arm, t in per.items():
        for m in t:
            by_task[arm][m["task"]] += 1
    lines += ["", "## Truncations per task", ""] + [f"- {arm}: " + ", ".join(f"{k} {v}" for k, v in sorted(d.items())) for arm, d in by_task.items()]
    Path(args.out).write_text("\n".join(lines) + "\n")
    Path(args.json).write_text(json.dumps({"summary": summary, "samples": per, "thresholds": {"compress": LOOP_THRESHOLD, "unique_lines": LOOP_UNIQUE}}, indent=1))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
