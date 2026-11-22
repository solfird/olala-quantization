"""Fetch every dataset and NLTK resource the six accuracy tasks need, into the pod's caches.

Run inside the evals environment (``uv run python prefetch_evals_data.py --evals-dir DIR``). It
builds each task of ``run_tasks.TASKS`` through the harness's own factory, which downloads the
dataset exactly as a real run would (MGSM from the simple-evals blob, EvalPlus HumanEval+ and
MBPP+, GSM8K, IFEval and AIME 2025 from the Hub, into ``HF_HOME``), counts its items against the
known sizes (a short count is a truncated download), then makes sure the NLTK tokenizers of the
IFEval scorer are in ``~/.cache/nltk_data``, where that scorer looks for them. No dataset content
is executed; nothing is written outside the caches.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXPECTED_ITEMS: dict[str, int] = {
    "mgsm": 2750,
    "humaneval_plus": 164,
    "mbpp_plus": 378,
    "gsm8k": 1319,
    "ifeval": 541,
    "aime2025": 30,
}
"""Items of each task's dataset (MGSM 250 x 11 languages), counted 2026-11-20."""


def count_problems(counts: dict[str, int]) -> list[str]:
    """Compare the item counts of the fetched datasets with the known sizes.

    Args:
        counts: Items found per task name.

    Returns:
        One message per task that is missing or has an unexpected count (empty when all is well).
    """
    problems = []
    for name, expected in EXPECTED_ITEMS.items():
        if name not in counts:
            problems.append(f"{name}: not fetched")
        elif counts[name] != expected:
            problems.append(f"{name}: {counts[name]} items, expected {expected}")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Fetch the datasets and the NLTK data; return 0 when every count and resource is right.

    Args:
        argv: Arguments (default: ``sys.argv[1:]``).

    Returns:
        ``0`` on success, ``1`` if a count or an NLTK resource is wrong.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--evals-dir", required=True, help="the evals/ folder of the bundle")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(Path(args.evals_dir).resolve()))
    import run_tasks

    counts: dict[str, int] = {}
    for name, spec in run_tasks.TASKS.items():
        task = spec.factory(sandbox="local") if spec.takes_sandbox else spec.factory()
        counts[name] = len(list(task.dataset))
        print(f"{name}: {counts[name]} items", flush=True)
    problems = count_problems(counts)

    import nltk
    from instruction_following_eval.evaluation import ensure_nltk_resource

    ensure_nltk_resource()
    nltk_dir = str((Path.home() / ".cache" / "nltk_data").absolute())
    for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
        try:
            nltk.data.find(resource, paths=[nltk_dir])
            print(f"nltk {resource}: present in {nltk_dir}", flush=True)
        except LookupError:
            problems.append(f"nltk {resource}: not found in {nltk_dir} after the download")
    for problem in problems:
        print(f"PROBLEM {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
