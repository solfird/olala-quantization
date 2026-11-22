"""Mutation check of the retry accounting and the client timeout: weaken the code one way at a time.

Usage::

    uv run python mutation_check_retries.py [--json out.json]

Covers the client timeout given to inspect_ai's provider (``run_tasks.py``), the ``model_attempts``
and ``working_time`` of the rows and the counters of ``run_info.json`` (``attempts.py``,
``run_tasks.py``, ``merge_aime_rerun.py``, ``combine_task_runs.py``) and the gate rules of
``analyze_wave.py`` that read them. The sources and the tests are copied to a temporary directory;
each mutation is applied to the COPY (the pattern must match exactly once, or the run stops), the
tests are run under a timeout with byte-code writing off, and a mutation counts as killed if any
test fails. The original files are never touched. Exit code 0 if every mutation was killed, 1
otherwise. ``Mutation`` and ``apply`` are those of ``mutation_check_analyze_wave.py``.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from mutation_check_analyze_wave import Mutation, apply

HERE = Path(__file__).resolve().parent
SOURCES = (
    "attempts.py",
    "run_tasks.py",
    "plus_tasks.py",
    "merge_aime_rerun.py",
    "combine_task_runs.py",
    "analyze_wave.py",
    "compare.py",
    "stats.py",
    "pyproject.toml",
)
TESTS = (
    "fake_runs.py",
    "fake_server.py",
    "test_attempts.py",
    "test_analyze_wave_gate.py",
    "test_analyze_wave_cli.py",
    "test_merge_aime_rerun.py",
    "test_combine_task_runs.py",
    "test_run_tasks_unit.py",
    "test_client_timeout.py",
    "test_run_tasks_e2e.py",
)
TIMEOUT_S = 900

R = "run_tasks.py"
T = "attempts.py"
M = "merge_aime_rerun.py"
C = "combine_task_runs.py"
A = "analyze_wave.py"
MUTATIONS: tuple[Mutation, ...] = (
    # ---- the client timeout reaches the provider
    Mutation(
        "the timeout is not passed to the provider (inspect_ai's 600 s default stays)",
        R,
        "        client_timeout=timeout,\n        memoize=False,\n",
        "        memoize=False,\n",
    ),
    Mutation(
        "the provider argument has the wrong name",
        R,
        "        client_timeout=timeout,\n",
        "        request_timeout=timeout,\n",
    ),
    Mutation(
        "main passes a fixed 600 s instead of --timeout",
        R,
        "model = build_model(args.base_url, args.model, args.timeout)",
        "model = build_model(args.base_url, args.model, 600)",
    ),
    Mutation(
        "build_model passes half of the timeout",
        R,
        "        client_timeout=timeout,\n",
        "        client_timeout=timeout / 2,\n",
    ),
    # ---- rows and run_info.json written by the harness
    Mutation(
        "attempts count the model events, not the requests",
        R,
        "return sum(1 + (e.retries or 0) for e in sample.events if isinstance(e, ModelEvent))",
        "return sum(1 for e in sample.events if isinstance(e, ModelEvent))",
    ),
    Mutation(
        "attempts forget the first request",
        R,
        "return sum(1 + (e.retries or 0) for e in sample.events if isinstance(e, ModelEvent))",
        "return sum((e.retries or 0) for e in sample.events if isinstance(e, ModelEvent))",
    ),
    Mutation(
        "attempts count events that are not model calls",
        R,
        "return sum(1 + (e.retries or 0) for e in sample.events if isinstance(e, ModelEvent))",
        "return sum(1 + (getattr(e, 'retries', None) or 0) for e in sample.events)",
    ),
    Mutation(
        "the row's working_time is the total time",
        R,
        '        "working_time": sample.working_time,\n',
        '        "working_time": sample.total_time,\n',
    ),
    Mutation(
        "run_info does not record the client timeout",
        R,
        '        "client_timeout": args.timeout,\n        "samples_with_retries": 0,\n',
        '        "samples_with_retries": 0,\n',
    ),
    Mutation(
        "the task entries do not record the client timeout",
        R,
        '                    "client_timeout": args.timeout,\n'
        "                    **retry_stats(rows),\n",
        "                    **retry_stats(rows),\n",
    ),
    Mutation(
        "the run-wide counters are never updated",
        R,
        "                info.update(retry_stats(all_rows))\n",
        "                pass\n",
    ),
    Mutation(
        "the run-wide counters cover the last task only",
        R,
        "                all_rows += rows\n",
        "                all_rows = rows\n",
    ),
    # ---- the shared counters
    Mutation(
        "a sample with exactly one attempt counts as retried",
        T,
        '    return has_attempts(row) and row["model_attempts"] > 1\n',
        '    return has_attempts(row) and row["model_attempts"] >= 1\n',
    ),
    Mutation(
        "only a sample with three attempts or more counts as retried",
        T,
        '    return has_attempts(row) and row["model_attempts"] > 1\n',
        '    return has_attempts(row) and row["model_attempts"] > 2\n',
    ),
    Mutation(
        "legacy rows are counted as zero retries",
        T,
        "    if any(has_attempts(r) for r in rows):\n",
        "    if True:\n",
    ),
    Mutation(
        "the run-wide client timeout is the largest, not the smallest",
        T,
        "    return min(values)\n",
        "    return max(values)\n",
    ),
    Mutation(
        "stale run-wide counters are kept",
        T,
        '    for key in ("client_timeout", "samples_with_retries", "max_working_time"):\n'
        "        info.pop(key, None)\n",
        "    pass\n",
    ),
    # ---- merge_aime_rerun.py
    Mutation(
        "the merge drops the retry counters of the replaced task",
        M,
        "    task_info.update(attempts.retry_stats(rerun_task))\n",
        "",
    ),
    Mutation(
        "the merge does not recompute the run-wide counters",
        M,
        "    attempts.apply_run_counters(info, rows)\n",
        "",
    ),
    Mutation(
        "the merge only overwrites the run-wide counters it can recompute",
        M,
        "    attempts.apply_run_counters(info, rows)\n",
        "    info.update(attempts.retry_stats(rows))\n",
    ),
    Mutation(
        "the merge adds the fields to legacy rows",
        M,
        '            "".join(line + "\\n" for line in [*kept, *new]), encoding="utf-8"\n',
        '            "".join(\n'
        '                json.dumps({**json.loads(x), "model_attempts": 1}) + "\\n"\n'
        "                for x in [*kept, *new]\n"
        '            ),\n            encoding="utf-8",\n',
    ),
    # ---- combine_task_runs.py
    Mutation(
        "the combination keeps the stale retry counters of the source entries",
        C,
        "        entry.update(attempts.retry_stats(task_rows))\n",
        "",
    ),
    Mutation(
        "the combination does not recompute the run-wide counters",
        C,
        "    attempts.apply_run_counters(info, rows)\n",
        "",
    ),
    # ---- the gate
    Mutation(
        "retry threshold off by one (exactly 0.5% refused)",
        A,
        "    if n * 1000 > len(rows) * RETRIED_SAMPLES_MAX_PER_MILLE:\n",
        "    if n * 1000 >= len(rows) * RETRIED_SAMPLES_MAX_PER_MILLE:\n",
    ),
    Mutation(
        "retry threshold doubled",
        A,
        "RETRIED_SAMPLES_MAX_PER_MILLE = 5\n",
        "RETRIED_SAMPLES_MAX_PER_MILLE = 10\n",
    ),
    Mutation(
        "retried samples counted as those with more than two attempts",
        A,
        "    n = sum(attempts.is_retried(r) for r in rows)\n",
        '    n = sum(1 for r in rows if r.get("model_attempts", 1) > 2)\n',
    ),
    Mutation(
        "rows with the field in only some rows tolerated",
        A,
        "    if len(known) != len(rows):\n",
        "    if False:\n",
    ),
    Mutation(
        "legacy AIME accepted without the option",
        A,
        'LEGACY_STRICT_TASKS = ("aime2025",)\n',
        "LEGACY_STRICT_TASKS = ()\n",
    ),
    Mutation(
        "the option is ignored (legacy AIME always accepted)",
        A,
        "        elif accept_legacy_aime:\n",
        "        elif True:\n",
    ),
    Mutation(
        "every legacy task is refused like AIME",
        A,
        "        if task not in LEGACY_STRICT_TASKS:\n",
        "        if False:\n",
    ),
    Mutation(
        "the warning for a legacy non-AIME task is not emitted",
        A,
        '            warnings.append(f"{task}: {LEGACY_RETRY_NOTE}")\n',
        "            pass\n",
    ),
    Mutation(
        "the gate drops the warnings of the runs",
        A,
        '        gate.warnings += [f"[{r.path}] {w}" for w in run_warnings]\n',
        "",
    ),
    Mutation(
        "the analysis does not give the option to the gate",
        A,
        "            accept_legacy_aime_retries=cfg.accept_legacy_aime_retries,\n        )\n",
        "            accept_legacy_aime_retries=False,\n        )\n",
    ),
    Mutation(
        "the command line does not give the option to the configuration",
        A,
        "            accept_legacy_aime_retries=ns.accept_legacy_aime_retries,\n",
        "            accept_legacy_aime_retries=False,\n",
    ),
    Mutation(
        "a client timeout equal to the timeout is refused",
        A,
        "        if _is_number(value) and value < timeout\n",
        "        if _is_number(value) and value <= timeout\n",
    ),
    Mutation(
        "the run-wide client timeout is not checked",
        A,
        '    found = [("run_info", run.info.get("client_timeout"))]\n',
        "    found = []\n",
    ),
    Mutation(
        "the client timeouts of the tasks are not checked",
        A,
        '    found += [(t, (tasks_info.get(t) or {}).get("client_timeout")) for t in expected]\n',
        "",
    ),
    Mutation(
        "run_info retry counters are not cross-checked with the rows",
        A,
        "        if key in tinfo and tinfo[key] != value:",
        "        if False:",
    ),
)


def run_tests(work: Path) -> tuple[bool, str]:
    """Run the tests in ``work``.

    Returns:
        ``(passed, first failing test or a note)``.
    """
    cmd = [sys.executable, "-m", "pytest", "-q", "-x", "--tb=no", "-p", "no:cacheprovider", "-rf"]
    cmd += [f"tests/{t}" for t in TESTS if t.startswith("test_")]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        done = subprocess.run(  # noqa: S603
            cmd, cwd=work, env=env, capture_output=True, text=True, timeout=TIMEOUT_S, check=False
        )
    except subprocess.TimeoutExpired:
        return False, "timeout (counts as killed)"
    if done.returncode == 0:
        return True, ""
    failed = [ln for ln in done.stdout.splitlines() if ln.startswith(("FAILED", "ERROR"))]
    return False, failed[0] if failed else f"exit code {done.returncode}"


def stage(work: Path) -> None:
    """Copy the sources and tests into ``work`` (fresh)."""
    shutil.rmtree(work, ignore_errors=True)
    (work / "tests").mkdir(parents=True)
    for name in SOURCES:
        shutil.copy(HERE / name, work / name)
    for name in TESTS:
        shutil.copy(HERE / "tests" / name, work / "tests" / name)


def main(argv: Sequence[str] | None = None) -> int:
    """Run every mutation and print which were killed.

    Args:
        argv: Arguments (default: ``sys.argv[1:]``).

    Returns:
        0 if all mutations were killed, 1 if any survived or the baseline fails.
    """
    ap = argparse.ArgumentParser(prog="mutation_check_retries.py", description=__doc__)
    ap.add_argument("--json", help="write the results here")
    args = ap.parse_args(argv)
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "evals"
        stage(work)
        passed, note = run_tests(work)
        if not passed:
            sys.stdout.write(f"baseline FAILS before any mutation: {note}\n")
            return 1
        sys.stdout.write(f"baseline: tests pass ({len(MUTATIONS)} mutations to try)\n")
        for m in MUTATIONS:
            stage(work)
            target = work / m.file
            target.write_text(apply(target.read_text(encoding="utf-8"), m), encoding="utf-8")
            passed, note = run_tests(work)
            status = "SURVIVED" if passed else "killed"
            results.append(
                {"mutation": m.name, "file": m.file, "status": status, "caught_by": note}
            )
            sys.stdout.write(f"{status:9} {m.name}  <- {note}\n")
            sys.stdout.flush()
    survivors = [r for r in results if r["status"] == "SURVIVED"]
    sys.stdout.write(f"{len(results) - len(survivors)} of {len(results)} mutations killed\n")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 1 if survivors else 0


if __name__ == "__main__":
    raise SystemExit(main())
