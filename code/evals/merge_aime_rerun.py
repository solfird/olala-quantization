r"""Merge a one-task rerun into a full run: the analysis directory of PROTOCOL amendments 10, 11.

Usage::

    uv run python merge_aime_rerun.py --original DIR --rerun DIR --out NEWDIR \
        [--task aime2025] [--allow-incomplete-original]

The new directory holds ``samples.jsonl``, ``items.json``, ``run_info.json`` and, if the original
has one, its ``quantization_info.json``. The rows of every other task are copied from the original
unchanged; the rows of the replaced task come from the rerun unchanged (the original rows of that
task are not copied: they stay in the original directory). ``run_info.json`` is the original's with
the counts recomputed from the merged rows, ``status`` ``complete`` and a ``merged`` block that
records both sources (paths and sha256), the reason, the settings that differ and the counts of
the replaced rows. The replaced task's entry comes from the rerun (with its ``client_timeout``),
its ``samples_with_retries`` and ``max_working_time`` are recomputed from the rerun's rows, and the
run-wide ``client_timeout`` (the smallest of the tasks'), ``samples_with_retries`` and
``max_working_time`` are recomputed from the merged rows (see ``attempts.py``): a counter appears
only if rows say it, and counts only the rows that carry the field (legacy rows are not zeros).
Nothing is changed in the two source directories.

The tool refuses (exit code 2, a message on stderr, nothing written) if the rerun is not complete
or does not hold exactly the original's item ids for the task, if the sampling parameters, the
reasoning effort, the sandbox or the epochs of the task differ, if ``--out`` exists, or if the task
is absent from either run. It does not hide request errors of the rerun: the gate of
``analyze_wave.py`` still applies to the merged directory.
"""

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import analyze_wave as aw
import attempts

DEFAULT_TASK = "aime2025"
REASON = "PROTOCOL amendment 10/11: AIME client-timeout artifact, rerun with non-binding settings"
MUST_MATCH = (
    "sampling.temperature",
    "sampling.top_p",
    "sampling.top_k",
    "sampling.max_tokens",
    "sampling.seed",
    "reasoning_effort",
    "chat_template_kwargs",
    "sandbox.mode",
)
"""``run_info.json`` keys (dotted) that must be equal in the original and the rerun.

The client timeout and the concurrency are the settings the rerun changes on purpose and are
recorded, not compared; ``sandbox.tasks`` lists the tasks of the run, so only whether the replaced
task is in it is compared.
"""


class MergeError(Exception):
    """The merge is refused; the message says why."""


def _get(info: Mapping[str, Any], dotted: str) -> Any:
    node: Any = info
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_text(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=False) + "\n"


def _read_lines(run: aw.Run) -> list[tuple[str, str]]:
    """``(task, raw line)`` of every row of ``samples.jsonl``, the line kept exactly as written."""
    text = (run.path / "samples.jsonl").read_text(encoding="utf-8")
    return [(json.loads(line).get("task"), line) for line in text.splitlines() if line.strip()]


def _load(path: Path, which: str) -> aw.Run:
    run = aw.load_run_dir(path)
    if run.problems:
        raise MergeError(f"the {which} directory {path} is unreadable: {'; '.join(run.problems)}")
    return run


def _epochs(run: aw.Run, task: str) -> int | None:
    value = _get(run.info, f"tasks.{task}.epochs")
    return value if isinstance(value, int) else None


def _check_rerun(rerun: aw.Run, task: str) -> None:
    """Refuse a rerun that is not a complete, exact run of ``task``."""
    status = rerun.info.get("status")
    if status != "complete":
        raise MergeError(f"the rerun is not complete (run_info status is {status!r})")
    epochs = _epochs(rerun, task)
    if epochs is None:
        raise MergeError(
            f"the rerun's run_info has no epochs for {task}: the rerun is not complete"
        )
    ids = [str(i) for i in rerun.items[task]]
    seen: dict[tuple[str, int], int] = defaultdict(int)
    n_rows = 0
    for name, line in _read_lines(rerun):
        if name != task:
            continue
        row = json.loads(line)
        seen[(str(row["sample_id"]), int(row["epoch"]))] += 1
        n_rows += 1
    wanted = {(i, e) for i in ids for e in range(1, epochs + 1)}
    missing = sorted(wanted - set(seen))
    extra = sorted(set(seen) - wanted)
    dupes = sorted(k for k, n in seen.items() if n > 1)
    if missing or extra or dupes or n_rows != len(wanted):
        raise MergeError(
            f"the rerun is not complete for {task}: {n_rows} rows, expected {len(wanted)} "
            f"({len(missing)} missing, {len(extra)} unexpected, {len(dupes)} duplicated)"
        )


def _check_same_run_conditions(original: aw.Run, rerun: aw.Run, task: str) -> None:
    """Refuse a rerun made with other sampling, effort, sandbox or epochs than the original."""
    for key in MUST_MATCH:
        a, b = _get(original.info, key), _get(rerun.info, key)
        if a != b:
            raise MergeError(f"the rerun differs from the original in {key}: {a!r} against {b!r}")
    sandboxed = [task in (_get(r.info, "sandbox.tasks") or []) for r in (original, rerun)]
    if sandboxed[0] != sandboxed[1]:
        raise MergeError(
            f"the rerun differs from the original in sandbox for {task}: "
            f"in the original sandbox.tasks {sandboxed[0]}, in the rerun {sandboxed[1]}"
        )
    a, b = _epochs(original, task), _epochs(rerun, task)
    if a != b:
        raise MergeError(f"the rerun differs from the original in epochs of {task}: {a} and {b}")


def _check_other_tasks_full(original: aw.Run, task: str, lines: list[tuple[str, str]]) -> None:
    """Refuse an incomplete original whose other tasks are not all at their expected count."""
    counts: dict[str, int] = defaultdict(int)
    for name, _ in lines:
        counts[name] += 1
    for other, ids in original.items.items():
        if other == task:
            continue
        epochs = _epochs(original, other)
        if epochs is None:
            raise MergeError(
                f"the original's run_info has no epochs for {other}: cannot tell if it is full"
            )
        expected = len(ids) * epochs
        if counts[other] != expected:
            raise MergeError(
                f"the original is incomplete and {other}: {counts[other]} rows, expected {expected}"
            )


def _settings(info: Mapping[str, Any], task: str) -> dict[str, Any]:
    return {
        "max_connections_per_task": aw.task_connections(info, task),
        "timeout": _get(info, "sampling.timeout"),
    }


def _client_timeout(info: Mapping[str, Any], task: str) -> Any:
    """The client timeout a run used for ``task`` (its entry's, else the run's), or ``None``."""
    value = _get(info, f"tasks.{task}.client_timeout")
    return value if value is not None else info.get("client_timeout")


def _task_retries(rows: Sequence[Mapping[str, Any]]) -> int | None:
    """Rows of a task that needed more than one request, or ``None`` if its rows do not say."""
    return attempts.retry_stats(rows).get("samples_with_retries")


def _counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "n_samples": len(rows),
        "request_errors": sum(r.get("error") is not None for r in rows),
        "truncations": sum(bool(r.get("truncated")) for r in rows),
    }


def merge_runs(
    original_dir: str | Path,
    rerun_dir: str | Path,
    out_dir: str | Path,
    task: str = DEFAULT_TASK,
    *,
    allow_incomplete_original: bool = False,
) -> dict[str, Any]:
    """Write ``out_dir``: the original run with ``task`` replaced by its rerun.

    Args:
        original_dir: A run directory of ``run_tasks.py`` with every task.
        rerun_dir: A complete run directory of ``task`` alone (other tasks in it are ignored).
        out_dir: The new directory (must not exist).
        task: The task to replace.
        allow_incomplete_original: Accept an original whose status is not ``complete`` if every
            other task has exactly its expected number of rows.

    Returns:
        The merged ``run_info.json`` content.

    Raises:
        MergeError: If any condition of the module docstring fails; nothing is written then.
    """
    original_dir, rerun_dir, out_dir = Path(original_dir), Path(rerun_dir), Path(out_dir)
    if out_dir.exists():
        raise MergeError(f"{out_dir} exists: choose a new --out")
    if original_dir.resolve() == rerun_dir.resolve():
        raise MergeError("the original and the rerun are the same directory")
    original, rerun = _load(original_dir, "original"), _load(rerun_dir, "rerun")
    if task not in original.items:
        raise MergeError(f"{task} is absent from the original (items.json)")
    if task not in rerun.items:
        raise MergeError(f"{task} is absent from the rerun (items.json)")
    _check_rerun(rerun, task)
    if sorted(map(str, original.items[task])) != sorted(map(str, rerun.items[task])):
        raise MergeError(f"the item ids of {task} differ between the original and the rerun")
    _check_same_run_conditions(original, rerun, task)

    original_lines, rerun_lines = _read_lines(original), _read_lines(rerun)
    if original.info.get("status") != "complete":
        if not allow_incomplete_original:
            raise MergeError(
                f"the original's status is {original.info.get('status')!r}, not 'complete' "
                "(--allow-incomplete-original accepts it if every other task is full)"
            )
        _check_other_tasks_full(original, task, original_lines)

    kept = [line for name, line in original_lines if name != task]
    new = [line for name, line in rerun_lines if name == task]
    rows = [json.loads(line) for line in [*kept, *new]]
    original_task = [json.loads(line) for name, line in original_lines if name == task]
    rerun_task = [json.loads(line) for line in new]

    items = {**original.items, task: list(rerun.items[task])}
    info = json.loads(json.dumps(original.info))
    info.update(_counts(rows))
    info["status"] = "complete"
    task_info = json.loads(json.dumps(rerun.info["tasks"][task]))
    task_info.update(_counts(rerun_task))
    for key in ("samples_with_retries", "max_working_time"):
        task_info.pop(key, None)  # never the rerun's stale value: recomputed from its rows below
    task_info.update(attempts.retry_stats(rerun_task))
    task_info["n_items"] = len(items[task])
    task_info["items_sha256"] = aw.sha256_of_ids(items[task])
    info["tasks"] = {**(info.get("tasks") or {}), task: task_info}
    attempts.apply_run_counters(info, rows)

    before, after = _settings(original.info, task), _settings(rerun.info, task)
    quant = {
        "original": original.sha256.get("quantization_info"),
        "rerun": rerun.sha256.get("quantization_info"),
    }
    info["merged"] = {
        "reason": REASON,
        "task": task,
        "original_dir": str(original_dir.resolve()),
        "rerun_dir": str(rerun_dir.resolve()),
        "sha256": {
            "original_samples": original.sha256["samples"],
            "original_run_info": original.sha256["run_info"],
            "rerun_samples": rerun.sha256["samples"],
            "rerun_run_info": rerun.sha256["run_info"],
        },
        "settings_differing": {
            k: {"original": before[k], "rerun": after[k]} for k in before if before[k] != after[k]
        },
        "task_rows": {"original": len(original_task), "rerun": len(rerun_task)},
        "original_task_request_errors": _counts(original_task)["request_errors"],
        "original_task_truncations": _counts(original_task)["truncations"],
        "rerun_task_request_errors": _counts(rerun_task)["request_errors"],
        "rerun_task_truncations": _counts(rerun_task)["truncations"],
        "original_task_samples_with_retries": _task_retries(original_task),
        "rerun_task_samples_with_retries": _task_retries(rerun_task),
        "client_timeout": {
            "original": _client_timeout(original.info, task),
            "rerun": _client_timeout(rerun.info, task),
        },
        "quantization_info_sha256": quant,
    }
    if quant["original"] != quant["rerun"]:
        print(
            "WARNING: quantization_info.json differs between the original and the rerun "
            "(the original's is copied): check that both served the same checkpoint",
            file=sys.stderr,
        )

    partial = out_dir.parent / f".{out_dir.name}.partial"
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    try:
        (partial / "samples.jsonl").write_text(
            "".join(line + "\n" for line in [*kept, *new]), encoding="utf-8"
        )
        (partial / "items.json").write_text(_json_text(items), encoding="utf-8")
        (partial / "run_info.json").write_text(_json_text(info), encoding="utf-8")
        if quant["original"] is not None:
            shutil.copyfile(
                original.path / "quantization_info.json", partial / "quantization_info.json"
            )
        partial.rename(out_dir)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return info


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="merge_aime_rerun.py", description=__doc__.split("\n")[0])
    ap.add_argument("--original", required=True, help="full run directory")
    ap.add_argument("--rerun", required=True, help="complete run directory of the one task")
    ap.add_argument("--out", required=True, help="new merged directory (must not exist)")
    ap.add_argument("--task", default=DEFAULT_TASK, help="task to replace (default aime2025)")
    ap.add_argument(
        "--allow-incomplete-original",
        action="store_true",
        help="accept an original whose status is not complete if every other task has exactly "
        "its expected number of rows",
    )
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """Run the merge.

    Args:
        argv: Command-line arguments (default: ``sys.argv[1:]``).

    Returns:
        ``0`` on success, ``2`` if the merge is refused (nothing is written then).
    """
    args = _build_parser().parse_args(argv)
    try:
        info = merge_runs(
            args.original,
            args.rerun,
            args.out,
            args.task,
            allow_incomplete_original=args.allow_incomplete_original,
        )
    except MergeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    block = info["merged"]
    print(
        f"merged {args.out}: {args.task} replaced ({block['task_rows']['original']} original rows "
        f"with {block['original_task_request_errors']} request errors and "
        f"{block['original_task_truncations']} truncations, kept in {args.original}; "
        f"{block['task_rows']['rerun']} rerun rows with {block['rerun_task_request_errors']} "
        f"request errors); {info['n_samples']} samples, {info['request_errors']} request errors, "
        f"{info['truncations']} truncations, status {info['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
