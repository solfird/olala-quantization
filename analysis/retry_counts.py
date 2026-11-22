"""Count samples that needed more than one request, per run and task, from the saved eval logs.

THIS SCRIPT DOCUMENTS HOW THE COUNTS WERE MADE: it needs the held-back raw inspect logs (``results/wave1/<run>/logs/*.eval``). Its output is shipped
as ``results/retry-counts.json``. With the logs in place:  cd code/evals && uv run python ../../analysis/retry_counts.py
A sample's requests are the sum over its model events of 1 + the event's own retries (the OpenAI client logs its
retries inside one event, so counting events undercounts). Output: results/retry-counts.json.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

from inspect_ai.log import read_eval_log

ROOT = Path(__file__).resolve().parents[1]
W1 = ROOT / "results" / "wave1"
RUNS = {
    "wave1_full": ["B0a", "B0b", "F1", "F2", "N2", "F1n"],
    "wave1_aime_rerun": ["B0a-aime2", "B0b-aime3", "F1-aime2", "F2-aime2", "N2-aime2", "F1n-aime2"],
    "wave2": ["B0a-v2", "B0b-v2", "F1-v2", "F2-v2", "N2-v2", "N5-v2", "G1-v2"],
    "q5_wave1": ["Q5-mgsm-he"],
    "q5_wave2": ["Q5v2-mgsm-he", "Q5v2-ifeval-mbpp", "Q5v2-aime"],
}


def count_run(run_dir: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for f in sorted(glob.glob(str(run_dir / "logs" / "*.eval"))):
        log = read_eval_log(f)
        task = log.eval.task.split("/")[-1]
        n = multi = events_multi = 0
        max_wt = 0.0
        for s in log.samples or []:
            events = [e for e in s.events if e.event == "model"]
            requests = sum(1 + (getattr(e, "retries", None) or 0) for e in events)
            n += 1
            multi += requests > 1
            events_multi += len(events) > 1
            max_wt = max(max_wt, s.working_time or 0.0)
        out[task] = {"samples": n, "more_than_one_request": multi, "more_than_one_event": events_multi,
                     "max_working_time_s": round(max_wt, 1)}
    return out


def main() -> None:
    result = {group: {r: count_run(W1 / r) for r in runs if (W1 / r).is_dir()} for group, runs in RUNS.items()}
    (ROOT / "results" / "retry-counts.json").write_text(json.dumps(result, indent=1) + "\n")
    for group, runs in result.items():
        for r, tasks in runs.items():
            print(group, r, {t: f"{v['more_than_one_request']}/{v['samples']}" for t, v in tasks.items()})


if __name__ == "__main__":
    main()
