"""Validate plus_tasks.py: the EvalPlus canonical solutions must all pass the sandboxed scorer.

A canonical solution that fails means the harness (prompt assembly, sandbox, timeout), not the
model, is at fault. Needs the Hugging Face datasets, and Docker unless ``--sandbox local`` is
given; results go to a JSON file that documents the check::

    uv run python check_plus_canonical.py --out validation/plus_canonical.json
    uv run python check_plus_canonical.py --sandbox local --out validation/plus_canonical_local.json
"""

import argparse
import json
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def canonical_completion(task: str, prompt: str, target: str) -> str:
    """Wrap the reference solution in a markdown code block as a model would answer.

    Args:
        task: ``humaneval_plus`` or ``mbpp_plus``.
        prompt: The dataset prompt (signature and docstring for HumanEval+).
        target: The reference solution (body for HumanEval+, whole code for MBPP+).

    Returns:
        The completion text.
    """
    code = prompt + target if task == "humaneval_plus" else target
    return f"```python\n{code}\n```"


def summarise(results: Sequence[tuple[str, str | None, bool]]) -> dict[str, Any]:
    """Count outcomes of ``(sample_id, score_value, has_error)`` triples.

    Args:
        results: One triple per sample.

    Returns:
        ``n``, ``correct``, ``errors`` and the sorted ids that did not pass.
    """
    failing = sorted(sid for sid, value, err in results if err or value != "C")
    return {
        "n": len(results),
        "correct": sum(1 for _, value, err in results if not err and value == "C"),
        "errors": sum(1 for _, _, err in results if err),
        "failing_ids": failing,
    }


def build_parser() -> argparse.ArgumentParser:
    """The command-line parser."""
    ap = argparse.ArgumentParser(prog="check_plus_canonical.py", description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", nargs="+", default=["humaneval_plus", "mbpp_plus"])
    ap.add_argument("--max-samples", type=int, default=8, help="parallel sandboxes")
    ap.add_argument(
        "--sandbox", choices=["docker", "local"], default="docker", help="where the code runs"
    )
    ap.add_argument("--out", required=True)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """Run the canonical solutions through the scorers and write a JSON summary.

    Args:
        argv: Arguments (default: ``sys.argv[1:]``).

    Returns:
        ``0`` if every canonical solution passed, ``1`` otherwise.
    """
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import ModelOutput
    from inspect_ai.solver import Generate, TaskState, solver

    import plus_tasks

    args = build_parser().parse_args(argv)

    report: dict[str, Any] = {
        "checked_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "datasets": {
            "humaneval_plus": [plus_tasks.HUMANEVAL_PLUS_PATH, plus_tasks.HUMANEVAL_PLUS_REVISION],
            "mbpp_plus": [plus_tasks.MBPP_PLUS_PATH, plus_tasks.MBPP_PLUS_REVISION],
        },
        "sandbox": args.sandbox,
        "verify_timeout_s": plus_tasks.VERIFY_TIMEOUT,
        "tasks": {},
    }
    for name in args.tasks:

        @solver
        def canonical(task_name: str = name):  # noqa: ANN202
            async def solve(state: TaskState, generate: Generate) -> TaskState:
                prompt = state.metadata.get("prompt", "")
                text = canonical_completion(task_name, prompt, state.target.text)
                state.output = ModelOutput.from_content("canonical", text)
                return state

            return solve

        log = inspect_eval(
            getattr(plus_tasks, name)(solver=canonical(), sandbox_mode=args.sandbox),
            model="mockllm/model",
            max_samples=args.max_samples,
            display="none",
            log_dir=tempfile.mkdtemp(prefix="plus-canonical-"),
            fail_on_error=False,
        )[0]
        triples = [
            (
                str(s.id),
                next(iter(s.scores.values())).value if s.scores else None,
                s.error is not None,
            )
            for s in log.samples or []
        ]
        report["tasks"][name] = summarise(triples)
        print(name, {k: v for k, v in report["tasks"][name].items() if k != "failing_ids"})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if all(t["correct"] == t["n"] for t in report["tasks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
