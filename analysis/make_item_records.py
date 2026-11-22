r"""Make the per-item records of this repository from the private ones: ``samples.jsonl`` without benchmark text.

THIS SCRIPT DOCUMENTS HOW THE SHIPPED ``samples.jsonl`` FILES WERE MADE. IT NEEDS THE PRIVATE WORKING COPY OF ``results/wave1`` AND THE HELD-BACK RAW
INSPECT LOGS OF ONE RUN (the prompts of the five tasks are read from them), so it cannot run on this repository alone.

    cd code/evals && uv run python ../../analysis/make_item_records.py --src-wave PRIVATE/results/wave1 --out-wave ../../results/wave1

A private row of ``samples.jsonl`` holds the item id, the score, the flags, the token counts, the ``target`` (the reference answer) and the ``completion``
(the visible answer of the model, without its reasoning). The rule applied to every row of every run folder:

* Item id, epoch, stratum, score, scores, error, truncated, stop_reason, token counts, times and attempts are kept unchanged.
* ``target`` is kept for MGSM and AIME 2025, where it is a short number. It is dropped for HumanEval+ and MBPP+ (the reference solution, code) and for
  IFEval (empty). The tests and the instruction text were never in the row.
* ``completion`` is dropped for HumanEval+ and MBPP+: it is code, and about half of the HumanEval+ answers repeat the function signature and the docstring
  of the prompt.
* ``completion`` is dropped for IFEval too: it is free text, about 7 percent of the answers repeat a stretch of the prompt, and the model invents names,
  addresses, URLs and e-mail addresses in it.
* ``completion`` of MGSM and AIME 2025 is cut to its first 1,000 characters (``completion_cut_at`` says so), and withheld (null, with
  ``completion_withheld`` set) when any 30 consecutive characters of the kept part also occur in the prompt of the item.
* Nothing else is added. The counts per task are written to ``results/item-records.json``.

Every number of the report is computed from the fields that are kept (``score``, ``truncated``, ``stop_reason``, token counts); none reads ``target`` or ``completion``.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

KEEP_TARGET = {"mgsm", "aime2025"}
DROP_COMPLETION = {"humaneval_plus", "mbpp_plus", "ifeval"}
CUT = 1000
WINDOW = 30


def norm_task(name: str) -> str:
    """Task name of an inspect log (``inspect_evals/aime2025``) in the form of ``samples.jsonl``."""
    return name.split("/")[-1].replace("-", "_")


def prompts_of(run_dir: Path) -> dict[tuple[str, str], str]:
    """``{(task, item id): prompt text}`` from the raw logs of one run (the prompts are the same for every run)."""
    from inspect_ai.log import read_eval_log

    out: dict[tuple[str, str], str] = {}
    for f in sorted(glob.glob(str(run_dir / "logs" / "*.eval"))):
        log = read_eval_log(f)
        task = norm_task(log.eval.task or "")
        for s in log.samples or []:
            inp = s.input
            text = inp if isinstance(inp, str) else " ".join(m.content if isinstance(m.content, str) else "" for m in inp)
            out[(task, str(s.id))] = text
    return out


def shares_text(piece: str, prompt: str) -> bool:
    """True if some ``WINDOW`` consecutive characters of ``piece`` occur in ``prompt``."""
    return any(piece[i : i + WINDOW] in prompt for i in range(max(1, len(piece) - WINDOW + 1)) if len(piece[i : i + WINDOW]) == WINDOW)


def clean_row(row: dict, prompts: dict[tuple[str, str], str], stats: dict) -> dict:
    """The shipped form of one private row."""
    task = row["task"]
    st = stats[task]
    st["rows"] += 1
    out = {k: v for k, v in row.items() if k not in ("target", "completion")}
    if task in KEEP_TARGET:
        out["target"] = row.get("target")
    if task in DROP_COMPLETION:
        st["completion_dropped"] += 1
        return out
    comp = row.get("completion") or ""
    piece = comp[:CUT]
    prompt = prompts.get((task, str(row["sample_id"])))
    if prompt is None or shares_text(piece, prompt):
        out["completion"] = None
        out["completion_withheld"] = "shares text with the prompt" if prompt is not None else "prompt unknown"
        st["completion_withheld"] += 1
        return out
    out["completion"] = piece
    if len(comp) > CUT:
        out["completion_cut_at"] = CUT
        st["completion_cut"] += 1
    st["completion_kept"] += 1
    return out


def main() -> int:
    """Rewrite every ``samples.jsonl`` under ``--src-wave`` into ``--out-wave`` and write the counts."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-wave", type=Path, required=True, help="private results/wave1")
    ap.add_argument("--out-wave", type=Path, required=True, help="results/wave1 of this repository")
    ap.add_argument("--prompts-from-run", default="B0a-v2", help="run folder of --src-wave whose raw logs give the prompts")
    ap.add_argument("--counts", type=Path, default=None, help="default: <out-wave>/../item-records.json")
    args = ap.parse_args()
    prompts = prompts_of(args.src_wave / args.prompts_from_run)
    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    files = 0
    for src in sorted(args.src_wave.rglob("samples.jsonl")):
        dst = args.out_wave / src.relative_to(args.src_wave)
        rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("".join(json.dumps(clean_row(r, prompts, stats), ensure_ascii=False) + "\n" for r in rows))
        files += 1
    counts = {"files": files, "policy": "see the docstring of analysis/make_item_records.py: target kept for MGSM and AIME only; completion dropped for HumanEval+, MBPP+ and IFEval; for MGSM and AIME cut to 1,000 characters and withheld on a 30-character overlap with the prompt", "window_chars": WINDOW, "cut_chars": CUT,
              "prompt_items_known": len(prompts), "per_task": {t: dict(v) for t, v in sorted(stats.items())}}
    (args.counts or args.out_wave.parent / "item-records.json").write_text(json.dumps(counts, indent=1) + "\n")
    print(json.dumps(counts, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
