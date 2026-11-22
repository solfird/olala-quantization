r"""Combine per-task run directories of ONE arm into one analysable run directory.

Usage::

    uv run python combine_task_runs.py --run DIR:TASK[,TASK...] [--run ...] --out NEWDIR \
        [--allow-partial-tasks]

An arm can be scored on several pods at once, each pod running a task group with ``run_tasks.py``
(one run directory per pod: for example mgsm and humaneval_plus on run A, ifeval and mbpp_plus on
run B, aime2025 on run C). ``analyze_wave.py`` wants one directory per arm with all five protocol
tasks. For each ``--run`` only the listed tasks are taken: their rows of ``samples.jsonl`` are
copied as the exact text lines of the source, so the output holds the union of the tasks with one
source per task. Nothing is changed in the source directories.

The new directory holds ``samples.jsonl``, ``items.json``, ``run_info.json`` and, copied from the
FIRST ``--run`` if it has them, ``server_provenance.txt``, ``server.cmd`` and
``quantization_info.json``. ``run_info.json`` is the first input's with ``status`` ``complete``,
``n_samples``, ``request_errors`` and ``truncations`` recomputed from the rows, the per-task
entries (``n_items``, ``epochs``, ``n_samples``, ``request_errors``, ``truncations``,
``items_sha256``) recomputed from the rows and ``items.json``, ``sandbox.tasks`` and
``max_connections_per_task`` taken from the source of each task, ``client_timeout`` (kept per
task, the run-wide value being the smallest, and only if every task states one),
``samples_with_retries`` and ``max_working_time`` recomputed from the rows (see ``attempts.py``:
legacy rows are not counted as zeros; a counter appears only if rows say it), and a ``combined``
block: every source directory with the sha256 of its ``samples.jsonl``, ``run_info.json`` and
``items.json``, its tasks and row counts, its ``max_connections_per_task``, ``timeout`` and
``client_timeout`` (these are not compared: they are non-binding settings), its model and
endpoint, its arm and the sha256 of its provenance files.

The tool refuses (exit code 2, a message on stderr, nothing written: the directory is built in
``.NAME.partial`` and renamed) if

* a task is listed twice, the same directory is given twice, a listed task is not one of the five
  protocol tasks or is absent from the ``items.json`` of its input;
* an input is unreadable or its ``run_info.json`` status is not ``complete`` (even if the tasks
  taken from it are finished: a run with another task unfinished is not accepted);
* the rows of a listed task are not exactly the ``(item, epoch)`` set that its ``items.json`` and
  its epochs imply (missing, unexpected or duplicated rows);
* the sampling parameters, the reasoning effort, the chat-template arguments, the sandbox mode or
  the item-selection seed differ between inputs (the list of ``merge_aime_rerun.py`` plus ``seed``);
* the inputs did not serve the same arm: ``server_provenance.txt`` (arm, llama.cpp commit, GGUF
  size and sha256, Hugging Face repository and revision) or ``quantization_info.json`` differ, or
  one is present in some inputs and missing in others;
* the union of the tasks is not exactly the five protocol tasks, unless ``--allow-partial-tasks``
  is given (the output and its ``combined`` block then say it is partial);
* ``--out`` exists.

The gate of ``analyze_wave.py`` still applies to the combined directory (request errors, item sets,
sample counts): this tool does not hide anything.
"""

import argparse
import copy
import json
import re
import shutil
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import analyze_wave as aw
import attempts
import merge_aime_rerun as mar

PROTOCOL_TASKS: tuple[str, ...] = tuple(aw.DEFAULT_EXPECTED)
"""The five tasks of the protocol, in the order of ``analyze_wave.DEFAULT_EXPECTED``."""

MUST_MATCH: tuple[str, ...] = (*mar.MUST_MATCH, "seed")
"""``run_info.json`` keys (dotted) equal in every input: ``merge_aime_rerun``'s list plus the
item-selection seed. The client timeout and the concurrency are recorded, not compared."""

PROVENANCE_MUST_MATCH: tuple[str, ...] = (
    "arm",
    "llama_cpp_repo",
    "llama_cpp_commit",
    "gguf_size",
    "gguf_sha256_pinned",
    "gguf_sha256_verified_by_setup",
    "hf_repo",
    "hf_revision",
)
"""Keys of ``server_provenance.txt`` that identify the served arm: the same in every input.

The time of the copy, the server log, the pod's command line and the serving shape (``np``,
``ctx``) are not compared.
"""

PROVENANCE_FILES: tuple[str, ...] = (
    "server_provenance.txt",
    "server.cmd",
    "quantization_info.json",
)
"""Files copied from the first input if it has them, and hashed for every input."""

REASON = "one arm scored on several pods, one task group per pod, combined for analyze_wave.py"
_KEY = re.compile(r"^[A-Za-z0-9_]+$")


class CombineError(Exception):
    """The combination is refused; the message says why."""


@dataclass
class Source:
    """One input: a run directory and the tasks taken from it.

    Attributes:
        run: The loaded run directory.
        tasks: The tasks taken from it, in the order given.
        lines: ``{task: [raw lines]}`` of the listed tasks, exactly as written in the source.
        provenance: Parsed ``server_provenance.txt``, or ``None`` if there is none.
    """

    run: aw.Run
    tasks: list[str]
    lines: dict[str, list[str]]
    provenance: dict[str, str] | None


def parse_run_arg(spec: str) -> tuple[Path, list[str]]:
    """Parse ``DIR:TASK[,TASK...]``.

    Args:
        spec: The text given to ``--run``.

    Returns:
        ``(directory, [tasks])``.

    Raises:
        CombineError: If the directory or any task is empty or there is no ``:``.
    """
    directory, sep, names = spec.rpartition(":")
    tasks = names.split(",") if sep else []
    if not sep or not directory or any(not t.strip() for t in tasks) or not tasks:
        raise CombineError(f"invalid --run {spec!r}: use DIR:TASK[,TASK...]")
    return Path(directory), [t.strip() for t in tasks]


def _canonical(tasks: Sequence[str]) -> list[str]:
    """``tasks`` in protocol order."""
    return sorted(tasks, key=PROTOCOL_TASKS.index)


def _parse_provenance(path: Path) -> dict[str, str]:
    """``key=value`` lines of a ``server_provenance.txt`` (the first line of a key wins)."""
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and _KEY.match(key) and key not in out:
            out[key] = value
    return out


def _check_status(run: aw.Run) -> None:
    status = run.info.get("status")
    if status != "complete":
        raise CombineError(
            f"{run.path} is not complete (run_info status is {status!r}): a run with a task "
            "unfinished is not accepted even if the tasks taken from it are finished"
        )


def _check_task_rows(run: aw.Run, task: str, lines: Sequence[str]) -> None:
    """Refuse a task whose rows are not exactly ``items.json`` times its epochs."""
    ids = [str(i) for i in run.items[task]]
    if len(set(ids)) != len(ids):
        raise CombineError(
            f"{run.path}: {task}: items.json lists {len(ids)} ids ({len(set(ids))} distinct)"
        )
    epochs = mar._epochs(run, task)
    if epochs is None:
        raise CombineError(f"the run_info of {run.path} has no epochs for {task}")
    seen: dict[tuple[str, int], int] = defaultdict(int)
    for line in lines:
        try:
            row = json.loads(line)
            seen[(str(row["sample_id"]), int(row["epoch"]))] += 1
        except (KeyError, TypeError, ValueError) as exc:
            raise CombineError(f"{run.path}: {task}: malformed row ({exc!r}): {line[:80]}") from exc
    wanted = {(i, e) for i in ids for e in range(1, epochs + 1)}
    missing = wanted - set(seen)
    extra = set(seen) - wanted
    dupes = [k for k, n in seen.items() if n > 1]
    if missing or extra or dupes or len(lines) != len(wanted):
        raise CombineError(
            f"{run.path}: {task}: {len(lines)} rows, expected {len(wanted)} "
            f"({len(missing)} missing, {len(extra)} unexpected, {len(dupes)} duplicated)"
        )


def _load_source(directory: Path, tasks: list[str]) -> Source:
    """Read one input and validate the tasks taken from it."""
    run = aw.load_run_dir(directory)
    if run.problems:
        raise CombineError(f"the input {directory} is unreadable: {'; '.join(run.problems)}")
    _check_status(run)
    for task in tasks:
        if task not in run.items:
            raise CombineError(f"{task} is absent from items.json of {directory}")
    try:
        by_task: dict[str, list[str]] = defaultdict(list)
        for name, line in mar._read_lines(run):
            by_task[name].append(line)
    except (AttributeError, ValueError) as exc:
        raise CombineError(f"{directory}: malformed row in samples.jsonl: {exc!r}") from exc
    lines = {task: by_task.get(task, []) for task in tasks}
    for task in tasks:
        _check_task_rows(run, task, lines[task])
    prov_file = directory / "server_provenance.txt"
    provenance = _parse_provenance(prov_file) if prov_file.is_file() else None
    return Source(run=run, tasks=tasks, lines=lines, provenance=provenance)


def _check_settings(sources: Sequence[Source], warnings: list[str]) -> None:
    """Refuse inputs whose binding settings differ; warn about the non-binding ones."""
    first = sources[0].run
    for src in sources[1:]:
        for key in MUST_MATCH:
            a, b = mar._get(first.info, key), mar._get(src.run.info, key)
            if a != b:
                raise CombineError(
                    f"{src.run.path} differs from the first input in {key}: {b!r} against {a!r}"
                )
        for key in aw.WARN_IF_DIFFERENT:
            if key == "sampling.timeout":
                continue  # recorded per source, expected to differ
            a, b = mar._get(first.info, key), mar._get(src.run.info, key)
            if a != b:
                warnings.append(f"{key} differs: {a!r} in {first.path}, {b!r} in {src.run.path}")
        if first.info.get("model") != src.run.info.get("model"):
            warnings.append(
                f"model label differs: {first.info.get('model')!r} in {first.path}, "
                f"{src.run.info.get('model')!r} in {src.run.path}"
            )


def _check_provenance(sources: Sequence[Source], warnings: list[str]) -> dict[str, str] | None:
    """Refuse inputs that did not serve the same arm; return the compared values of the first."""
    first = sources[0]
    for name in ("server_provenance.txt", "quantization_info.json"):
        present = [(src.run.path / name).is_file() for src in sources]
        if any(present) and not all(present):
            have = next(s.run.path for s, p in zip(sources, present, strict=True) if p)
            lack = next(s.run.path for s, p in zip(sources, present, strict=True) if not p)
            raise CombineError(
                f"{name} is missing from {lack} (present in {have}): cannot tell "
                "that the inputs served the same arm"
            )
    if first.provenance is None:
        warnings.append(
            "no server_provenance.txt in any input: the arm served by the pods is not checked"
        )
    else:
        for src in sources[1:]:
            for key in PROVENANCE_MUST_MATCH:
                a, b = first.provenance.get(key), (src.provenance or {}).get(key)
                if a != b:
                    raise CombineError(
                        f"server_provenance.txt differs in {key}: {a!r} in {first.run.path}, "
                        f"{b!r} in {src.run.path}: the inputs did not serve the same arm"
                    )
        if first.provenance.get("gguf_sha256_verified_by_setup") == "unverified":
            warnings.append(
                "the GGUF file's sha256 was 'unverified' by the setup (only the pinned value "
                "was compared)"
            )
    hashes = [s.run.sha256.get("quantization_info") for s in sources]
    if len(set(hashes)) > 1:
        raise CombineError("quantization_info.json differs between the inputs")
    if first.provenance is None:
        return None
    return {k: first.provenance[k] for k in PROVENANCE_MUST_MATCH if k in first.provenance}


def _hash_provenance_files(run: aw.Run) -> dict[str, str | None]:
    return {
        name: aw._sha256_file(run.path / name) if (run.path / name).is_file() else None
        for name in PROVENANCE_FILES
    }


def _source_block(src: Source) -> dict[str, Any]:
    run, info = src.run, src.run.info
    return {
        "dir": str(run.path.resolve()),
        "tasks": list(src.tasks),
        "status": info.get("status"),
        "sha256": {
            "samples": run.sha256["samples"],
            "run_info": run.sha256["run_info"],
            "items": run.sha256["items"],
        },
        "task_rows": {t: len(src.lines[t]) for t in src.tasks},
        "max_connections_per_task": {t: aw.task_connections(info, t) for t in src.tasks},
        "timeout": mar._get(info, "sampling.timeout"),
        "client_timeout": {t: mar._client_timeout(info, t) for t in src.tasks},
        "model": info.get("model"),
        "endpoint": info.get("endpoint"),
        "arm": (src.provenance or {}).get("arm"),
        "provenance_sha256": _hash_provenance_files(run),
        "code_sha256": info.get("code_sha256"),
        "git_commit": mar._get(info, "git.commit"),
        "started_utc": info.get("started_utc"),
        "ended_utc": info.get("ended_utc"),
        "argv": info.get("argv"),
    }


def combine_runs(
    runs: Sequence[tuple[Path, Sequence[str]]],
    out_dir: str | Path,
    *,
    allow_partial_tasks: bool = False,
) -> dict[str, Any]:
    """Write ``out_dir``: the union of the listed tasks of the input runs.

    Args:
        runs: ``(run directory, tasks to take from it)``, in the order of the command line; the
            first one gives ``run_info.json`` and the provenance files.
        out_dir: The new directory (must not exist).
        allow_partial_tasks: Accept a union that is not exactly the five protocol tasks.

    Returns:
        The combined ``run_info.json`` content.

    Raises:
        CombineError: If any condition of the module docstring fails; nothing is written then.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise CombineError(f"{out_dir} exists: choose a new --out")
    if not runs:
        raise CombineError("no --run given")
    dirs = [Path(d).resolve() for d, _ in runs]
    if len(set(dirs)) != len(dirs):
        raise CombineError("the same directory is given twice: list all its tasks in one --run")
    listed = [t for _, tasks in runs for t in tasks]
    for task in listed:
        if listed.count(task) > 1:
            raise CombineError(f"{task} is listed twice: each task has exactly one source")
        if task not in PROTOCOL_TASKS:
            raise CombineError(
                f"{task} is not a protocol task (the tasks are {', '.join(PROTOCOL_TASKS)})"
            )
    sources = [_load_source(Path(d), list(tasks)) for d, tasks in runs]
    warnings: list[str] = []
    _check_settings(sources, warnings)
    compared = _check_provenance(sources, warnings)
    tasks = _canonical(listed)
    missing = [t for t in PROTOCOL_TASKS if t not in tasks]
    if missing and not allow_partial_tasks:
        raise CombineError(
            f"the union of tasks is not the five protocol tasks: missing {', '.join(missing)} "
            "(--allow-partial-tasks accepts a partial union)"
        )

    owner = {t: src for src in sources for t in src.tasks}
    lines = [line for t in tasks for line in owner[t].lines[t]]
    rows = [json.loads(line) for line in lines]
    items = {t: list(owner[t].run.items[t]) for t in tasks}
    info = copy.deepcopy(sources[0].run.info)
    info.update(mar._counts(rows))
    info["status"] = "complete"
    info["tasks"] = {}
    for t in tasks:
        entry = copy.deepcopy(owner[t].run.info["tasks"][t])
        task_rows = [json.loads(x) for x in owner[t].lines[t]]
        entry.update(mar._counts(task_rows))
        for key in ("samples_with_retries", "max_working_time"):
            entry.pop(key, None)  # never the source's stale value: recomputed from its rows
        entry.update(attempts.retry_stats(task_rows))
        entry["n_items"] = len(items[t])
        entry["items_sha256"] = aw.sha256_of_ids(items[t])
        info["tasks"][t] = entry
    attempts.apply_run_counters(info, rows)
    info["sandbox"] = {
        **(info.get("sandbox") or {}),
        "tasks": [t for t in tasks if t in (mar._get(owner[t].run.info, "sandbox.tasks") or [])],
    }
    info["max_connections_per_task"] = {t: aw.task_connections(owner[t].run.info, t) for t in tasks}
    info["skipped_tasks"] = {
        k: v for k, v in (info.get("skipped_tasks") or {}).items() if k not in tasks
    }
    info["combined"] = {
        "reason": REASON,
        "allow_partial_tasks": allow_partial_tasks,
        "missing_tasks": missing,
        "task_sources": {t: str(owner[t].run.path.resolve()) for t in tasks},
        "sources": [_source_block(s) for s in sources],
        "provenance_compared": compared,
        "warnings": warnings,
    }
    for text in warnings:
        print(f"WARNING: {text}", file=sys.stderr)

    partial = out_dir.parent / f".{out_dir.name}.partial"
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    try:
        (partial / "samples.jsonl").write_text("".join(x + "\n" for x in lines), encoding="utf-8")
        (partial / "items.json").write_text(mar._json_text(items), encoding="utf-8")
        (partial / "run_info.json").write_text(mar._json_text(info), encoding="utf-8")
        for name in PROVENANCE_FILES:
            if (sources[0].run.path / name).is_file():
                shutil.copyfile(sources[0].run.path / name, partial / name)
        partial.rename(out_dir)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return info


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="combine_task_runs.py", description=__doc__.split("\n")[0])
    ap.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="DIR:TASK[,TASK...]",
        help="a run directory and the tasks taken from it (repeat; the first gives run_info.json)",
    )
    ap.add_argument("--out", required=True, help="new combined directory (must not exist)")
    ap.add_argument(
        "--allow-partial-tasks",
        action="store_true",
        help="accept a union that is not exactly the five protocol tasks (it is marked partial)",
    )
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """Run the combination.

    Args:
        argv: Command-line arguments (default: ``sys.argv[1:]``).

    Returns:
        ``0`` on success, ``2`` if it is refused (nothing is written then).
    """
    args = _build_parser().parse_args(argv)
    try:
        specs = [parse_run_arg(s) for s in args.run]
        info = combine_runs(specs, args.out, allow_partial_tasks=args.allow_partial_tasks)
    except CombineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    block = info["combined"]
    print(
        f"combined {args.out}: {len(block['sources'])} sources, tasks {', '.join(info['tasks'])}; "
        f"{info['n_samples']} samples, {info['request_errors']} request errors, "
        f"{info['truncations']} truncations, status {info['status']}"
    )
    if block["missing_tasks"]:
        print(
            f"PARTIAL: the protocol tasks {', '.join(block['missing_tasks'])} are not in the "
            "combined directory (--allow-partial-tasks); analyze_wave.py will refuse it"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
