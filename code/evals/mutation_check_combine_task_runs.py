"""Mutation check of the tests of ``combine_task_runs.py``: weaken the tool one way at a time.

Usage::

    uv run python mutation_check_combine_task_runs.py [--json out.json]

The sources and the combine_task_runs tests are copied to a temporary directory; each mutation is
applied to the COPY (the pattern must match exactly once, or the run stops), the tests are run
under a timeout with byte-code writing off, and a mutation counts as killed if any test fails.
The original files are never touched. Exit code 0 if every mutation was killed, 1 otherwise.
``Mutation`` and ``apply`` are those of ``mutation_check_analyze_wave.py``.
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
    "combine_task_runs.py",
    "attempts.py",
    "merge_aime_rerun.py",
    "analyze_wave.py",
    "compare.py",
    "stats.py",
    "pyproject.toml",
)
TESTS = ("fake_runs.py", "test_combine_task_runs.py")
TIMEOUT_S = 600

C = "combine_task_runs.py"
MUTATIONS: tuple[Mutation, ...] = (
    Mutation(
        "a task listed twice accepted",
        C,
        "        if listed.count(task) > 1:\n",
        "        if False:\n",
    ),
    Mutation(
        "the same directory given twice accepted",
        C,
        "    if len(set(dirs)) != len(dirs):\n",
        "    if False:\n",
    ),
    Mutation(
        "missing rows tolerated",
        C,
        "    if missing or extra or dupes or len(lines) != len(wanted):\n",
        "    if extra or dupes:\n",
    ),
    Mutation(
        "unexpected rows tolerated",
        C,
        "    if missing or extra or dupes or len(lines) != len(wanted):\n",
        "    if missing or dupes:\n",
    ),
    Mutation(
        "duplicated rows tolerated",
        C,
        "    if missing or extra or dupes or len(lines) != len(wanted):\n",
        "    if missing or extra:\n",
    ),
    Mutation(
        "settings that must match not compared",
        C,
        "        for key in MUST_MATCH:\n",
        "        for key in ():\n",
    ),
    Mutation(
        "only the second input compared with the first",
        C,
        "    for src in sources[1:]:\n        for key in MUST_MATCH:\n",
        "    for src in sources[1:2]:\n        for key in MUST_MATCH:\n",
    ),
    Mutation(
        "item-selection seed not compared",
        C,
        'MUST_MATCH: tuple[str, ...] = (*mar.MUST_MATCH, "seed")',
        "MUST_MATCH: tuple[str, ...] = (*mar.MUST_MATCH,)",
    ),
    Mutation(
        "run status not checked",
        C,
        '    if status != "complete":\n',
        "    if False:\n",
    ),
    Mutation(
        "run counts not recomputed",
        C,
        "    info.update(mar._counts(rows))\n",
        "    pass\n",
    ),
    Mutation(
        "per-task counts not recomputed",
        C,
        "        entry.update(mar._counts(task_rows))\n",
        "        pass\n",
    ),
    Mutation(
        "items_sha256 not recomputed",
        C,
        '        entry["items_sha256"] = aw.sha256_of_ids(items[t])\n',
        "        pass\n",
    ),
    Mutation(
        "sources not hashed",
        C,
        '            "samples": run.sha256["samples"],\n',
        '            "samples": None,\n',
    ),
    Mutation(
        "provenance mismatch tolerated",
        C,
        "                a, b = first.provenance.get(key), (src.provenance or {}).get(key)\n"
        "                if a != b:\n",
        "                a, b = first.provenance.get(key), (src.provenance or {}).get(key)\n"
        "                if False:\n",
    ),
    Mutation(
        "GGUF sha256 not part of the provenance comparison",
        C,
        '    "gguf_sha256_pinned",\n    "gguf_sha256_verified_by_setup",\n',
        "",
    ),
    Mutation(
        "arm not part of the provenance comparison",
        C,
        '    "arm",\n    "llama_cpp_repo",\n',
        '    "llama_cpp_repo",\n',
    ),
    Mutation(
        "provenance present in some inputs only tolerated",
        C,
        "        if any(present) and not all(present):\n",
        "        if False:\n",
    ),
    Mutation(
        "quantization_info.json mismatch tolerated",
        C,
        "    if len(set(hashes)) > 1:\n",
        "    if False:\n",
    ),
    Mutation(
        "partial union accepted by default",
        C,
        "    if missing and not allow_partial_tasks:\n",
        "    if False:\n",
    ),
    Mutation(
        "rows modified (not byte-identical)",
        C,
        "    lines = [line for t in tasks for line in owner[t].lines[t]]\n",
        '    lines = [line.replace(", ", ",") for t in tasks for line in owner[t].lines[t]]\n',
    ),
    Mutation(
        "output directory created before the checks (written on refusal)",
        C,
        "    sources = [_load_source(Path(d), list(tasks)) for d, tasks in runs]\n",
        "    out_dir.mkdir(parents=True, exist_ok=True)\n"
        "    sources = [_load_source(Path(d), list(tasks)) for d, tasks in runs]\n",
    ),
    Mutation(
        ".partial directory left behind after a failed write",
        C,
        "    except BaseException:\n        shutil.rmtree(partial, ignore_errors=True)\n",
        "    except BaseException:\n        pass\n",
    ),
    Mutation(
        "sandbox.tasks kept from the first input",
        C,
        '        "tasks": [t for t in tasks if t in '
        '(mar._get(owner[t].run.info, "sandbox.tasks") or [])],\n',
        '        "tasks": (info.get("sandbox") or {}).get("tasks"),\n',
    ),
    Mutation(
        "max_connections_per_task kept from the first input",
        C,
        '    info["max_connections_per_task"] = '
        "{t: aw.task_connections(owner[t].run.info, t) for t in tasks}\n",
        "    pass\n",
    ),
    Mutation(
        "provenance files not copied",
        C,
        "        for name in PROVENANCE_FILES:\n"
        "            if (sources[0].run.path / name).is_file():\n",
        "        for name in PROVENANCE_FILES:\n            if False:\n",
    ),
)


def run_tests(work: Path) -> tuple[bool, str]:
    """Run the combine_task_runs tests in ``work``.

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
    ap = argparse.ArgumentParser(prog="mutation_check_combine_task_runs.py", description=__doc__)
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
    survivors = [r for r in results if r["status"] == "SURVIVED"]
    sys.stdout.write(f"{len(results) - len(survivors)} of {len(results)} mutations killed\n")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 1 if survivors else 0


if __name__ == "__main__":
    raise SystemExit(main())
