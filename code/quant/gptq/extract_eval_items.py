"""Write the item texts of the five evaluation tasks to JSON, for the leakage audit. Never runs a model.

It imports the harness's own task registry (``evals/run_tasks.py``: ``TASKS``) and calls each task factory, which only
builds the ``inspect_ai`` ``Task`` and loads its dataset (from the Hugging Face cache; ``HF_HUB_OFFLINE=1`` is set
unless ``--online`` is given). It then reads ``task.dataset``: the prompt, the target and every string in the sample
metadata (canonical solutions, tests, instruction texts). No model, no sandbox, no scorer is started: tasks that
execute model code are built with ``sandbox="local"`` and never run.

Run it inside the harness environment, which has ``inspect_ai`` and ``inspect_evals``::

    uv run --project ../../evals python quant/gptq/extract_eval_items.py --out eval_items.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "eval-items/1"
DEFAULT_TASKS = ("mgsm", "humaneval_plus", "mbpp_plus", "ifeval", "aime2025")
DEFAULT_EVALS_DIR = Path(__file__).resolve().parents[2] / "evals"


def _text_of(value: Any) -> str | None:
    """Flatten a sample input (string, or a list of chat messages whose content is a string or a list of parts)."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        parts = [t for t in (_text_of(v) for v in value) if t]
        return "\n".join(parts) if parts else None
    for attr in ("text", "content"):
        inner = getattr(value, attr, None)
        if inner is not None:
            return _text_of(inner)
    return None


def _flatten(prefix: str, value: Any, out: dict[str, str]) -> None:
    """Add every string found in a JSON-like value to ``out`` under a dotted key."""
    if isinstance(value, str):
        out[prefix] = value
    elif isinstance(value, Mapping):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}", v, out)
    elif isinstance(value, Sequence):
        for i, v in enumerate(value):
            _flatten(f"{prefix}.{i}", v, out)


def items_from_samples(samples: Iterable[Any]) -> list[dict[str, Any]]:
    """Turn ``inspect_ai`` samples (or look-alikes) into ``[{"id": ..., "fields": {name: text}}]``."""
    items = []
    for pos, s in enumerate(samples):
        fields: dict[str, str] = {}
        text = _text_of(getattr(s, "input", None))
        if text:
            fields["input"] = text
        target = getattr(s, "target", None)
        for i, t in enumerate(target if isinstance(target, Sequence) and not isinstance(target, str) else [target]):
            if isinstance(t, str) and t:
                fields["target" if i == 0 else f"target.{i}"] = t
        choices = getattr(s, "choices", None)
        if choices:
            for i, c in enumerate(choices):
                _flatten(f"choices.{i}", c, fields)
        meta = getattr(s, "metadata", None)
        if meta:
            _flatten("metadata", meta, fields)
        sid = getattr(s, "id", None)
        items.append({"id": str(pos if sid is None else sid), "fields": fields})
    return items


def extract(tasks: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    """Build the item file for ``names`` from a task registry (``name -> spec`` with ``factory`` and ``takes_sandbox``).

    Raises:
        KeyError: For a task name that is not in the registry.
        ValueError: If a task has no item at all.
    """
    out: dict[str, Any] = {}
    for name in names:
        spec = tasks[name]
        task = spec.factory(sandbox="local") if getattr(spec, "takes_sandbox", False) else spec.factory()
        items = items_from_samples(task.dataset)
        if not items:
            raise ValueError(f"task {name!r} has no item")
        out[name] = {"n_items": len(items), "items": items}
    return {"schema": SCHEMA, "created_utc": datetime.now(UTC).isoformat(), "tasks": out}


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="extract_eval_items.py", description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--evals-dir", type=Path, default=DEFAULT_EVALS_DIR, help="directory of evals/run_tasks.py")
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    p.add_argument("--online", action="store_true", help="allow dataset downloads (default: cache only)")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI. Needs ``inspect_ai`` and ``inspect_evals`` importable (the harness environment)."""
    args = _parse_args(argv)
    if not args.online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    sys.path.insert(0, str(args.evals_dir))
    try:
        import run_tasks

        data = extract(run_tasks.TASKS, args.tasks)
    except (ImportError, KeyError, ValueError, OSError) as e:
        print(f"error: {e!r}", file=sys.stderr)
        return 2
    data["harness"] = {
        "run_tasks": str(args.evals_dir / "run_tasks.py"),
        "run_tasks_sha256": hashlib.sha256((args.evals_dir / "run_tasks.py").read_bytes()).hexdigest(),
    }
    args.out.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {args.out}: " + ", ".join(f"{k}={v['n_items']}" for k, v in data["tasks"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
