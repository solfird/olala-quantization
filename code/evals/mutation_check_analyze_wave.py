"""Mutation check of the tests of ``analyze_wave.py``: weaken the script one way at a time.

Usage::

    uv run python mutation_check_analyze_wave.py [--json out.json]

The sources and the analyze_wave tests are copied to a temporary directory; each mutation is
applied to the COPY (the pattern must match exactly once, or the run stops), the tests are run
under a timeout with byte-code writing off, and a mutation counts as killed if any test fails.
The original files are never touched. Exit code 0 if every mutation was killed, 1 otherwise.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = (
    "analyze_wave.py",
    "attempts.py",
    "compare.py",
    "stats.py",
    "run_tasks.py",
    "plus_tasks.py",
    "pyproject.toml",
)
TESTS = (
    "fake_runs.py",
    "test_analyze_wave_gate.py",
    "test_analyze_wave_stats.py",
    "test_analyze_wave_analysis.py",
    "test_analyze_wave_cli.py",
)
TIMEOUT_S = 600


@dataclass(frozen=True)
class Mutation:
    """One deliberate weakening of the code.

    Attributes:
        name: Short description of the weakening.
        file: Source file (relative to ``evals/``) to change.
        old: Exact text to replace (must occur once).
        new: Replacement text.
    """

    name: str
    file: str
    old: str
    new: str


A = "analyze_wave.py"
MUTATIONS: tuple[Mutation, ...] = (
    Mutation("skip the gate (never refuse)", A, "    if refusals:\n", "    if False:\n"),
    Mutation(
        "do not check the run status",
        A,
        'if status != "complete":',
        "if False:",
    ),
    Mutation("do not detect missing requests", A, "    if missing:\n", "    if False:\n"),
    Mutation("accept different item sets", A, "        if ia != ib:\n", "        if False:\n"),
    Mutation(
        "request-error limit at 1% instead of 0.5%",
        A,
        "REQUEST_ERROR_MAX_PER_MILLE = 5",
        "REQUEST_ERROR_MAX_PER_MILLE = 10",
    ),
    Mutation(
        "request errors at exactly 0.5% refused (>= for >)",
        A,
        "n_errors * 1000 > n_rows",
        "n_errors * 1000 >= n_rows",
    ),
    Mutation(
        "top_k allowed to differ between arms",
        A,
        '    "sampling.top_k",\n    "sampling.max_tokens",',
        '    "sampling.max_tokens",',
    ),
    Mutation(
        "the same run listed twice is accepted",
        A,
        "if key in seen_dirs or (digest is not None and digest in seen_hashes):",
        "if False:",
    ),
    Mutation(
        "wrong sign of the margin in the p-value",
        A,
        "np.asarray(draws) <= margin",
        "np.asarray(draws) <= -margin",
    ),
    Mutation("no Holm adjustment", A, "adj = holm_adjust(raw)", "adj = dict(raw)"),
    Mutation("wrong alpha (0.05 for 0.025)", A, "ALPHA = 0.025", "ALPHA = 0.05"),
    Mutation(
        "pooling off (first run only)",
        A,
        "np.mean([s[i] for s, _ in per_run])",
        "np.mean([s[i] for s, _ in per_run[:1]])",
    ),
    Mutation(
        "wrong clustering unit (each sample its own item)",
        A,
        "grouped[sid].append(compare.effective_score(r))",
        "grouped[f\"{sid}#{r['epoch']}\"].append(compare.effective_score(r))",
    ),
    Mutation(
        "truncations not counted as failures",
        "compare.py",
        'or row.get("truncated") or',
        "or",
    ),
    Mutation(
        "macro-average weighted by items instead of tasks",
        A,
        'draws["macro"] = np.mean([draws[t] for t in t_list], axis=0)',
        'draws["macro"] = np.average([draws[t] for t in t_list], axis=0, '
        'weights=[summary["tasks"][t]["n_items"] for t in t_list])',
    ),
    Mutation(
        "all tasks share one random stream",
        A,
        'key = int.from_bytes(hashlib.sha256(task.encode()).digest()[:8], "big")',
        "key = 0",
    ),
    Mutation(
        "pairing broken (arm y resampled with other items)",
        A,
        "yv[idx].mean(axis=1)",
        "yv[idx[::-1]].mean(axis=1)",
    ),
    Mutation(
        "seed ignored by the bootstrap",
        A,
        "np.random.SeedSequence([seed, key])",
        "np.random.SeedSequence([0, key])",
    ),
    Mutation("interval at 90%", A, "CONFIDENCE = stats.CONFIDENCE", "CONFIDENCE = 0.90"),
    Mutation(
        "one-sided bound from the wrong tail",
        A,
        "np.quantile(d, alpha)",
        "np.quantile(d, 1 - alpha)",
    ),
    Mutation("WORSE also at the margin (<= for <)", A, "if upper < margin:", "if upper <= margin:"),
    Mutation(
        "NON-INFERIOR needs p strictly below alpha",
        A,
        "if p_holm <= alpha:",
        "if p_holm < alpha:",
    ),
    Mutation(
        "sensitivity margins dropped",
        A,
        "margins = [cfg.margin, *[m for m in cfg.sensitivity_margins if m != cfg.margin]]",
        "margins = [cfg.margin]",
    ),
    Mutation(
        "exploratory arms enter the Holm family",
        A,
        '            result["exploratory"][a] = block\n',
        '            result["exploratory"][a] = block\n'
        '            macro_draws[a] = draws["macro"]\n',
    ),
    Mutation(
        "exploratory pairs are refused like the others",
        A,
        'soft = kind.startswith("exploratory")',
        "soft = False",
    ),
    Mutation(
        "truncation threshold at 2 points",
        A,
        "TRUNCATION_FLAG_PP = 1.0",
        "TRUNCATION_FLAG_PP = 2.0",
    ),
    Mutation(
        "a difference of exactly 1 point flagged",
        A,
        "limit = TRUNCATION_FLAG_PP + 1e-9",
        "limit = TRUNCATION_FLAG_PP - 1e-9",
    ),
    Mutation(
        "truncation counted over the first run of a pooled arm only",
        A,
        "for run in arm.runs for r in run.rows",
        "for run in arm.runs[:1] for r in run.rows",
    ),
    Mutation("task-flag threshold at -10", A, "TASK_FLAG_PP = -5.0", "TASK_FLAG_PP = -10.0"),
    Mutation(
        "task flag on the point estimate boundary (<= for <)",
        A,
        'if v["diff"]["hi"] < threshold',
        'if v["diff"]["hi"] <= threshold',
    ),
    Mutation(
        "AIME secondary keeps truncated samples",
        A,
        '        if r.get("truncated"):\n            c[2] += 1',
        "        if False:\n            c[2] += 1",
    ),
    Mutation(
        "calibration always reported as containing 0",
        A,
        'block["contains_zero"] = bool(macro["lo"] <= 0.0 <= macro["hi"])',
        'block["contains_zero"] = True',
    ),
    Mutation("exit code 1 instead of 3 on refusal", A, "        return 3\n", "        return 1\n"),
    Mutation(
        "items hash not checked against items.json",
        A,
        'if tinfo.get("items_sha256") != sha256_of_ids(ids):',
        "if False:",
    ),
    Mutation(
        "headline bootstrap ignores the resamples option",
        A,
        "compare_arms(x, y, tasks, cfg.n_boot, cfg.seed)",
        "compare_arms(x, y, tasks, 1000, cfg.seed)",
    ),
    Mutation(
        "AIME secondary ignores the seed",
        A,
        '"aime_non_truncated": aime_non_truncated(x, y, cfg.n_boot, cfg.seed),',
        '"aime_non_truncated": aime_non_truncated(x, y, cfg.n_boot, 0),',
    ),
    Mutation(
        "MGSM table ignores the seed",
        A,
        '"mgsm_languages": mgsm_languages(x, y, cfg.n_boot, cfg.seed),',
        '"mgsm_languages": mgsm_languages(x, y, cfg.n_boot, 0),',
    ),
    Mutation(
        "command line ignores --seed", A, "            seed=ns.seed,\n", "            seed=0,\n"
    ),
    Mutation(
        "command line ignores --resamples",
        A,
        "            n_boot=ns.resamples,\n",
        "            n_boot=10_000,\n",
    ),
    Mutation(
        "command line ignores --margin",
        A,
        "            margin=ns.margin,\n",
        "            margin=-3.0,\n",
    ),
    Mutation(
        "command line ignores --expected",
        A,
        "expected=load_expected(ns.expected),",
        "expected=load_expected(None),",
    ),
    Mutation(
        "per-task truncation scopes dropped from the flag",
        A,
        "        t for t, d in by_task.items() if abs(d) > limit\n",
        "        t for t, d in by_task.items() if False\n",
    ),
    Mutation(
        "code hash and version differences not warned",
        A,
        'WARN_IF_DIFFERENT = ("code_sha256", "versions", "git.commit", "sampling.timeout")',
        "WARN_IF_DIFFERENT = ()",
    ),
    Mutation(
        "pooled runs of one arm not compared with the others",
        A,
        "    for r in usable[1:]:\n",
        "    for r in usable[1:2]:\n",
    ),
    Mutation("a positive margin accepted", A, "        if margin >= 0:\n", "        if False:\n"),
    Mutation(
        "calibration width computed as a sum",
        A,
        'block["width_pp"] = macro["hi"] - macro["lo"]',
        'block["width_pp"] = macro["hi"] + macro["lo"]',
    ),
    Mutation(
        "sections no longer open with the conclusion",
        A,
        'return [f"## {title}", "", f"**Conclusion:** {conclusion}", "", *body, ""]',
        'return [f"## {title}", "", *body, "", f"**Conclusion:** {conclusion}", ""]',
    ),
    Mutation(
        "quantization recipe not read",
        A,
        '"quantization_recipe": ((run.quantization or {}).get("recipe") or {}).get("name"),',
        '"quantization_recipe": None,',
    ),
)


def apply(text: str, m: Mutation) -> str:
    """Apply a mutation, failing unless its pattern occurs exactly once.

    Args:
        text: Source text.
        m: The mutation.

    Returns:
        The mutated text.

    Raises:
        SystemExit: If the pattern does not occur exactly once.
    """
    n = text.count(m.old)
    if n != 1:
        raise SystemExit(f"mutation {m.name!r}: pattern found {n} times in {m.file}, expected 1")
    return text.replace(m.old, m.new)


def run_tests(work: Path) -> tuple[bool, str]:
    """Run the analyze_wave tests in ``work``.

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
    ap = argparse.ArgumentParser(prog="mutation_check_analyze_wave.py", description=__doc__)
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
