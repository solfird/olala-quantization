r"""Pre-registered analysis of the accuracy wave (PROTOCOL.md 7, 8 and amendments 5 to 9).

Usage::

    uv run python analyze_wave.py --run B0=DIR_A,DIR_B --run F1=DIR ... \
        --reference B0 --candidates F1,F2,N2 --out report.md --json report.json

Every number comes from the run directories written by ``run_tasks.py``. The script first applies
the gate (identical item sets, full sample counts, no missing request, complete status, identical
settings, request errors at most 0.5%, at most 0.5% of the samples of a task that needed more than
one request, no client timeout below the run's timeout; a run that does not record its retries
is warned about, and refused for AIME unless ``--accept-legacy-aime-retries`` is given); a pair
that fails it is refused (exit code 3, nothing is written) unless it involves an ``--exploratory``
arm, where the failure is a warning in the report.
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import attempts
import compare
import stats

DEFAULT_EXPECTED: dict[str, dict[str, int]] = {
    "mgsm": {"items": 550, "epochs": 1},
    "humaneval_plus": {"items": 164, "epochs": 1},
    "mbpp_plus": {"items": 378, "epochs": 1},
    "ifeval": {"items": 541, "epochs": 1},
    "aime2025": {"items": 30, "epochs": 5},
}
"""Items and samples per item of a full run (PROTOCOL.md section 6); 1,783 samples in all."""

REQUEST_ERROR_MAX_PER_MILLE = 5
"""A run with more than 0.5% request errors (5 per 1,000 samples) makes its pair invalid."""

RETRIED_SAMPLES_MAX_PER_MILLE = 5
"""A task in which more than 0.5% of the samples needed more than one request is invalid.

A retry is a request the client cancelled (timeout, dropped connection, 5xx) and sent again: the
generation that was under way is thrown away and redrawn, which biases the long generations out.
"""

LEGACY_RETRY_NOTE = "retry count unknown (client timeout 600 s legacy default)"
"""Said of a task whose rows do not carry ``model_attempts``: it ran with inspect_ai's 600 s client
timeout and nothing in its rows tells how many requests were cancelled and redrawn."""

LEGACY_STRICT_TASKS = ("aime2025",)
"""Tasks whose legacy rows are refused (not just warned about) unless the option accepts them: AIME
requests are the ones that run past 600 s."""

LEGACY_AIME_MEASURED = (
    "measured on the six saved vLLM AIME reruns: 3 to 9 percent of the samples (4 to 13 of 150) "
    "had a model call that inspect_ai logged as retried, and 11 to 21 percent (16 to 31 of 150) "
    "had a request cancelled and redrawn once retries inside a model call are counted"
)

MUST_MATCH = (
    "sampling.temperature",
    "sampling.top_p",
    "sampling.top_k",
    "sampling.max_tokens",
    "sampling.seed",
    "reasoning_effort",
    "chat_template_kwargs",
    "sandbox.mode",
    "sandbox.tasks",
    "seed",
)
"""``run_info.json`` keys (dotted) that must be equal in every run of a compared pair."""

WARN_IF_DIFFERENT = ("code_sha256", "versions", "git.commit", "sampling.timeout")
"""Keys whose difference is reported as a warning, not a refusal.

The client timeout and the per-task concurrency (``max_connections_per_task``, compared task by
task in ``_compare_runs``) are deliberately not in ``MUST_MATCH``: they are non-binding settings
that must not change an output unless a request errors (PROTOCOL.md amendment 10), but a reader
should see that two arms did not use the same ones.
"""


# --------------------------------------------------------------------------- syntax


def parse_run_spec(spec: str) -> tuple[str, list[Path]]:
    """Parse ``NAME=DIR`` or ``NAME=DIR1,DIR2`` (two completed runs of one arm, pooled).

    Args:
        spec: The text given to ``--run``.

    Returns:
        ``(name, [directories])``.

    Raises:
        ValueError: If the name or any directory is empty or there is no ``=``.
    """
    name, sep, dirs = spec.partition("=")
    parts = dirs.split(",") if sep else []
    if not name or not sep or not parts or any(not p for p in parts):
        raise ValueError(f"invalid run spec {spec!r}: use NAME=DIR or NAME=DIR1,DIR2")
    return name, [Path(p) for p in parts]


def total_samples(expected: Mapping[str, Mapping[str, int]]) -> int:
    """Samples of a full run: the sum over tasks of ``items * epochs``."""
    return sum(v["items"] * v["epochs"] for v in expected.values())


def load_expected(path: str | Path | None) -> dict[str, dict[str, int]]:
    """Read the expected items and epochs per task (default: the protocol's).

    Args:
        path: JSON file ``{task: {"items": n, "epochs": e}}``, or ``None`` for the default.

    Returns:
        The mapping.

    Raises:
        ValueError: If an entry lacks ``items`` or ``epochs`` or they are not positive integers.
    """
    if path is None:
        return {k: dict(v) for k, v in DEFAULT_EXPECTED.items()}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for task, spec in data.items():
        ok = isinstance(spec, dict) and all(
            isinstance(spec.get(k), int) and spec[k] > 0 for k in ("items", "epochs")
        )
        if not ok:
            raise ValueError(
                f"expected counts for {task!r} must be {{'items': n, 'epochs': e}} (positive ints)"
            )
    if not data:
        raise ValueError("expected counts are empty")
    return {task: {"items": spec["items"], "epochs": spec["epochs"]} for task, spec in data.items()}


def sha256_of_ids(ids: Sequence[Any]) -> str:
    """Order-independent hash of a set of item ids (the formula of ``run_tasks.py``)."""
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


# --------------------------------------------------------------------------- loading


@dataclass
class Run:
    """One run directory as read from disk.

    Attributes:
        path: The directory.
        rows: Rows of ``samples.jsonl``.
        info: Content of ``run_info.json``.
        items: Content of ``items.json`` (item ids per task).
        sha256: Hashes of ``run_info.json``, ``samples.jsonl`` and ``items.json``.
        quantization: Content of ``quantization_info.json`` if present.
        problems: Reasons the directory could not be read.
    """

    path: Path
    rows: list[dict[str, Any]] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    items: dict[str, list[str]] = field(default_factory=dict)
    sha256: dict[str, str] = field(default_factory=dict)
    quantization: dict[str, Any] | None = None
    problems: list[str] = field(default_factory=list)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run_dir(path: str | Path) -> Run:
    """Read a result directory of ``run_tasks.py``; read failures become ``problems``.

    Args:
        path: The directory.

    Returns:
        The run (possibly with empty content and a list of problems).
    """
    path = Path(path)
    run = Run(path=path)
    if not path.is_dir():
        run.problems.append(f"{path} is not a directory")
        return run
    for key, name in (
        ("samples", "samples.jsonl"),
        ("items", "items.json"),
        ("run_info", "run_info.json"),
    ):
        f = path / name
        if not f.is_file():
            run.problems.append(f"missing file {name}")
            continue
        try:
            text = f.read_text(encoding="utf-8")
            if key == "samples":
                run.rows = [json.loads(line) for line in text.splitlines() if line.strip()]
            elif key == "items":
                run.items = json.loads(text)
            else:
                run.info = json.loads(text)
        except (ValueError, OSError) as exc:
            run.problems.append(f"cannot read {name}: {exc}")
            continue
        run.sha256[key] = _sha256_file(f)
    qf = path / "quantization_info.json"
    if qf.is_file():
        try:
            run.quantization = json.loads(qf.read_text(encoding="utf-8"))
            run.sha256["quantization_info"] = _sha256_file(qf)
        except (ValueError, OSError) as exc:
            run.problems.append(f"cannot read quantization_info.json: {exc}")
    return run


# --------------------------------------------------------------------------- gate


def _dotted(info: Mapping[str, Any], key: str) -> Any:
    node: Any = info
    for part in key.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _check_retries(
    task: str, rows: Sequence[Mapping[str, Any]], accept_legacy_aime: bool, warnings: list[str]
) -> list[str]:
    """Problems and warnings about the requests that were cancelled and redrawn in one task."""
    if not rows:
        return []  # missing rows are reported by the other checks
    known = [r for r in rows if attempts.has_attempts(r)]
    if not known:
        if task not in LEGACY_STRICT_TASKS:
            warnings.append(f"{task}: {LEGACY_RETRY_NOTE}")
        elif accept_legacy_aime:
            warnings.append(
                f"{task}: {LEGACY_RETRY_NOTE}; accepted with --accept-legacy-aime-retries, "
                f"{LEGACY_AIME_MEASURED}"
            )
        else:
            return [
                f"{task}: {LEGACY_RETRY_NOTE}; its rows are refused because a request that "
                "lasted more than 600 s was cancelled and redrawn (pass "
                "--accept-legacy-aime-retries to accept them with a warning)"
            ]
        return []
    problems: list[str] = []
    if len(known) != len(rows):
        problems.append(
            f"{task}: model_attempts is missing from some rows ({len(rows) - len(known)} of "
            f"{len(rows)}): a task is either all legacy or all new"
        )
    n = sum(attempts.is_retried(r) for r in rows)
    if n * 1000 > len(rows) * RETRIED_SAMPLES_MAX_PER_MILLE:
        problems.append(
            f"{task}: {n} of {len(rows)} samples ({n / len(rows):.2%}) needed more than one "
            "request (a request was cancelled and redrawn), above 0.5% of the task's samples"
        )
    return problems


def _check_client_timeout(run: Run, expected: Mapping[str, Mapping[str, int]]) -> list[str]:
    """Problems when a client timeout is below the run's own timeout (a request cut short)."""
    timeout = _dotted(run.info, "sampling.timeout")
    if not _is_number(timeout):
        return []
    found = [("run_info", run.info.get("client_timeout"))]
    tasks_info = run.info.get("tasks") or {}
    found += [(t, (tasks_info.get(t) or {}).get("client_timeout")) for t in expected]
    return [
        f"{where}: client_timeout {value} is below the run's timeout {timeout}: a request longer "
        f"than {value} s was cancelled and redrawn before the harness timeout"
        for where, value in found
        if _is_number(value) and value < timeout
    ]


def check_run(
    run: Run,
    expected: Mapping[str, Mapping[str, int]],
    warnings: list[str] | None = None,
    *,
    accept_legacy_aime_retries: bool = False,
) -> list[str]:
    """Check one run against the gate's per-run conditions.

    Args:
        run: A loaded run.
        expected: Items and epochs per task.
        warnings: If given, receives what is worth reading but does not invalidate the run (the
            retry count of a legacy task is unknown).
        accept_legacy_aime_retries: Accept legacy AIME rows (no ``model_attempts``), with a
            warning, instead of refusing them.

    Returns:
        The reasons the run is not a full, complete, valid run (empty if it is).
    """
    warnings = [] if warnings is None else warnings
    if run.problems:
        return list(run.problems)
    problems: list[str] = []
    status = run.info.get("status")
    if status != "complete":
        problems.append(f"run_info status is {status!r}, not 'complete'")
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in run.rows:
        by_task[r.get("task")].append(r)
    n_rows = 0
    n_errors = 0
    for task, spec in expected.items():
        ids = run.items.get(task)
        if not ids:
            problems.append(f"{task}: absent from items.json")
            continue
        rows = by_task.get(task, [])
        n_rows += len(rows)
        n_errors += sum(r.get("error") is not None for r in rows)
        problems += _check_task(run, task, spec, ids, rows)
        problems += _check_retries(task, rows, accept_legacy_aime_retries, warnings)
    if n_rows and n_errors * 1000 > n_rows * REQUEST_ERROR_MAX_PER_MILLE:
        problems.append(
            f"request errors {n_errors}/{n_rows} = {n_errors / n_rows:.2%} exceed 0.5% of samples"
        )
    problems += _check_client_timeout(run, expected)
    return problems


def _check_task(
    run: Run,
    task: str,
    spec: Mapping[str, int],
    ids: list[str],
    rows: list[dict[str, Any]],
) -> list[str]:
    problems: list[str] = []
    if len(set(ids)) != len(ids) or len(ids) != spec["items"]:
        problems.append(
            f"{task}: items.json lists {len(ids)} items ({len(set(ids))} distinct), "
            f"expected {spec['items']}"
        )
    tinfo = (run.info.get("tasks") or {}).get(task) or {}
    if tinfo.get("epochs") != spec["epochs"]:
        problems.append(f"{task}: run_info epochs {tinfo.get('epochs')}, expected {spec['epochs']}")
    seen: dict[tuple[str, int], int] = defaultdict(int)
    for r in rows:
        seen[(str(r["sample_id"]), int(r["epoch"]))] += 1
    wanted = {(str(i), e) for i in ids for e in range(1, spec["epochs"] + 1)}
    missing = sorted(wanted - set(seen))
    extra = sorted(set(seen) - wanted)
    dupes = sorted(k for k, n in seen.items() if n > 1)
    if missing:
        listed = ", ".join(f"{i} epoch {e}" for i, e in missing[:5])
        problems.append(f"{task}: {len(missing)} missing request(s): {listed}")
    if extra:
        problems.append(
            f"{task}: {len(extra)} row(s) for items or epochs not expected: {extra[:3]}"
        )
    if dupes:
        problems.append(f"{task}: duplicate rows for {dupes[:3]}")
    if len(rows) != spec["items"] * spec["epochs"]:
        problems.append(f"{task}: {len(rows)} samples, expected {spec['items'] * spec['epochs']}")
    unscored = [r["sample_id"] for r in rows if r.get("error") is None and r.get("score") is None]
    if unscored:
        problems.append(f"{task}: {len(unscored)} unscored row(s) without a request error")
    counted = {
        "n_samples": len(rows),
        "request_errors": sum(r.get("error") is not None for r in rows),
        "truncations": sum(bool(r.get("truncated")) for r in rows),
    }
    for key, value in counted.items():
        if tinfo.get(key) != value:
            problems.append(
                f"{task}: run_info {key} {tinfo.get(key)} disagrees with {value} in samples.jsonl"
            )
    for key, value in attempts.retry_stats(rows).items():
        if key in tinfo and tinfo[key] != value:  # a legacy task has neither the key nor the rows
            problems.append(
                f"{task}: run_info {key} {tinfo[key]} disagrees with {value} in samples.jsonl"
            )
    if tinfo.get("items_sha256") != sha256_of_ids(ids):
        problems.append(f"{task}: run_info items_sha256 does not match the ids in items.json")
    return problems


@dataclass
class Gate:
    """Outcome of the gate for one compared pair.

    Attributes:
        problems: Reasons that make the pair invalid.
        warnings: Differences worth reading that do not invalidate the pair.
    """

    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether the pair passed."""
        return not self.problems


def gate_pair(
    runs_a: Sequence[Run],
    runs_b: Sequence[Run],
    expected: Mapping[str, Mapping[str, int]],
    *,
    accept_legacy_aime_retries: bool = False,
) -> Gate:
    """Apply the gate to two arms (each one run, or several runs to be pooled).

    Args:
        runs_a: Runs of the first arm.
        runs_b: Runs of the second arm.
        expected: Items and epochs per task.
        accept_legacy_aime_retries: Accept AIME rows without ``model_attempts`` (a legacy run,
            made with the 600 s client timeout) with a warning instead of refusing them.

    Returns:
        The gate's problems and warnings.
    """
    gate = Gate()
    every = [*runs_a, *runs_b]
    for r in every:
        run_warnings: list[str] = []
        found = check_run(
            r, expected, run_warnings, accept_legacy_aime_retries=accept_legacy_aime_retries
        )
        gate.problems += [f"[{r.path}] {p}" for p in found]
        gate.warnings += [f"[{r.path}] {w}" for w in run_warnings]
    for arm in (runs_a, runs_b):
        seen_dirs: set[Path] = set()
        seen_hashes: set[str] = set()
        for r in arm:
            key = r.path.resolve()
            digest = r.sha256.get("samples")
            if key in seen_dirs or (digest is not None and digest in seen_hashes):
                gate.problems.append(
                    f"[{r.path}] the same run directory is listed twice in one arm"
                )
            seen_dirs.add(key)
            if digest:
                seen_hashes.add(digest)
    usable = [r for r in every if not r.problems]
    if len(usable) < 2:
        return gate
    first = usable[0]
    for r in usable[1:]:
        gate.problems += _compare_runs(first, r, expected, gate.warnings)
    return gate


def task_connections(info: Mapping[str, Any], task: str) -> Any:
    """Requests in flight a run used for ``task``: its per-task value, else its global one."""
    per_task = info.get("max_connections_per_task")
    if isinstance(per_task, Mapping) and task in per_task:
        return per_task[task]
    return info.get("max_connections")


def _compare_runs(
    a: Run, b: Run, expected: Mapping[str, Mapping[str, int]], warnings: list[str]
) -> list[str]:
    problems: list[str] = []
    keys = list(MUST_MATCH)
    for task in expected:
        keys += [f"tasks.{task}.epochs", f"tasks.{task}.items_sha256"]
    for key in keys:
        va, vb = _dotted(a.info, key), _dotted(b.info, key)
        if va != vb:
            problems.append(f"{key} differs: {va!r} in {a.path}, {vb!r} in {b.path}")
    for task in expected:
        ia, ib = set(a.items.get(task, [])), set(b.items.get(task, []))
        if ia != ib:
            problems.append(
                f"{task}: item sets differ ({len(ia - ib)} only in {a.path}, "
                f"{len(ib - ia)} only in {b.path})"
            )
    for key in WARN_IF_DIFFERENT:
        va, vb = _dotted(a.info, key), _dotted(b.info, key)
        if va != vb:
            warnings.append(f"{key} differs: {va!r} in {a.path}, {vb!r} in {b.path}")
    for task in expected:
        if task not in (a.info.get("tasks") or {}) or task not in (b.info.get("tasks") or {}):
            continue
        ca, cb = task_connections(a.info, task), task_connections(b.info, task)
        if ca != cb:
            warnings.append(
                f"max_connections_per_task differs for {task}: {ca!r} in {a.path}, "
                f"{cb!r} in {b.path}"
            )
    return problems


# --------------------------------------------------------------------------- scores and pooling


@dataclass
class Arm:
    """One arm: its runs and its item scores, pooled over the runs.

    Attributes:
        name: Arm label (``B0``, ``F1`` ...).
        runs: The completed runs of the arm.
        scores: ``{task: {item id: score in [0, 1]}}``, the mean over the runs of each item's
            score (itself the mean over the item's epochs, failures counted as 0).
        strata: ``{task: {item id: stratum}}`` (the language for MGSM, else ``None``).
    """

    name: str
    runs: list[Run]
    scores: dict[str, dict[str, float]]
    strata: dict[str, dict[str, str | None]]


def run_item_scores(run: Run, task: str) -> tuple[dict[str, float], dict[str, str | None]]:
    """Score of each item of ``task`` in one run: the mean over its epochs.

    A request error, a truncated answer or an unscored row counts as 0
    (``compare.effective_score``).

    Args:
        run: A loaded run.
        task: Task name.

    Returns:
        ``({item id: score}, {item id: stratum})``.
    """
    grouped: dict[str, list[float]] = defaultdict(list)
    strata: dict[str, str | None] = {}
    for r in run.rows:
        if r.get("task") != task:
            continue
        sid = str(r["sample_id"])
        grouped[sid].append(compare.effective_score(r))
        strata.setdefault(sid, r.get("stratum"))
    return {sid: float(np.mean(v)) for sid, v in grouped.items()}, strata


def build_arm(name: str, runs: Sequence[Run], tasks: Sequence[str]) -> Arm:
    """Pool the runs of an arm by averaging the item scores (PROTOCOL.md amendment 9).

    Args:
        name: Arm label.
        runs: Completed runs of the arm.
        tasks: Tasks to score.

    Returns:
        The arm. Only items present in every run are kept (the gate makes them all of them).
    """
    scores: dict[str, dict[str, float]] = {}
    strata: dict[str, dict[str, str | None]] = {}
    for task in tasks:
        per_run = [run_item_scores(r, task) for r in runs]
        per_run = [(s, st) for s, st in per_run if s]
        if not per_run:
            continue
        common = set.intersection(*(set(s) for s, _ in per_run))
        scores[task] = {i: float(np.mean([s[i] for s, _ in per_run])) for i in sorted(common)}
        strata[task] = {i: per_run[0][1].get(i) for i in sorted(common)}
    return Arm(name=name, runs=list(runs), scores=scores, strata=strata)


# --------------------------------------------------------------------------- bootstrap


ALPHA = 0.025
"""One-sided significance level of the confirmatory family (Holm-adjusted)."""

CONFIDENCE = stats.CONFIDENCE
"""Two-sided coverage of every reported interval (95%)."""


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values.

    Args:
        p_values: Raw p-values by hypothesis name.

    Returns:
        Adjusted p-values (monotone, at most 1), same keys.
    """
    order = sorted(p_values, key=lambda k: p_values[k])
    m = len(order)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, name in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[name]))
        adjusted[name] = running
    return adjusted


def outcome_label(p_holm: float, upper: float, margin: float, alpha: float = ALPHA) -> str:
    """Outcome of a non-inferiority test.

    Args:
        p_holm: Holm-adjusted one-sided p-value.
        upper: Upper bound of the two-sided 95% interval, in points.
        margin: Non-inferiority margin, in points (negative).
        alpha: One-sided level.

    Returns:
        ``NON-INFERIOR`` if Holm rejects, ``WORSE`` if the whole interval is below the margin,
        otherwise ``INCONCLUSIVE``.
    """
    if p_holm <= alpha:
        return "NON-INFERIOR"
    if upper < margin:
        return "WORSE"
    return "INCONCLUSIVE"


def resample_indices(seed: int, task: str, n_items: int, n_boot: int) -> np.ndarray:
    """Indices of ``n_boot`` bootstrap resamples of the items of one task.

    Each task has its own random stream, derived from the seed and the task name, so tasks are
    resampled independently and the draws of a task do not depend on which other tasks exist.

    Args:
        seed: Bootstrap seed.
        task: Task name.
        n_items: Number of items (the resampling unit).
        n_boot: Number of resamples.

    Returns:
        Integer array of shape ``(n_boot, n_items)``.
    """
    key = int.from_bytes(hashlib.sha256(task.encode()).digest()[:8], "big")
    rng = np.random.default_rng(np.random.SeedSequence([seed, key]))
    return rng.integers(0, n_items, size=(n_boot, n_items))


def interval(draws: np.ndarray) -> tuple[float, float]:
    """Two-sided 95% percentile interval of bootstrap draws (``stats`` percentile rule)."""
    return stats._bounds(np.asarray(draws, dtype=float), CONFIDENCE)


def _ci(value: float, draws: np.ndarray) -> dict[str, float]:
    lo, hi = interval(draws)
    return {"value": float(value), "lo": lo, "hi": hi}


def compare_arms(
    x: Arm, y: Arm, tasks: Sequence[str], n_boot: int, seed: int
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Paired bootstrap of ``x - y`` per task and for the equally weighted macro-average.

    Items are resampled independently within each task, the same items in both arms; each
    resample's task means are then averaged with equal weight (PROTOCOL.md section 7).

    Args:
        x: The arm under test.
        y: The comparison arm (reference).
        tasks: The tasks of the macro-average.
        n_boot: Number of resamples.
        seed: Bootstrap seed.

    Returns:
        ``(summary, draws)``. ``summary`` holds, per task and for ``macro`` (``None`` unless
        every task could be paired), accuracies in percent and differences in points, each
        as ``{"value", "lo", "hi"}``. ``draws`` maps each task and ``"macro"`` to the
        ``n_boot`` bootstrap differences in points.
    """
    summary: dict[str, Any] = {"n_boot": n_boot, "seed": seed, "tasks": {}, "macro": None}
    draws: dict[str, np.ndarray] = {}
    x_draws: dict[str, np.ndarray] = {}
    y_draws: dict[str, np.ndarray] = {}
    for task in tasks:
        sx, sy = x.scores.get(task, {}), y.scores.get(task, {})
        ids = sorted(set(sx) & set(sy))
        if not ids:
            continue
        xv = 100.0 * np.array([sx[i] for i in ids])
        yv = 100.0 * np.array([sy[i] for i in ids])
        idx = resample_indices(seed, task, len(ids), n_boot)
        x_draws[task], y_draws[task] = xv[idx].mean(axis=1), yv[idx].mean(axis=1)
        draws[task] = x_draws[task] - y_draws[task]
        summary["tasks"][task] = {
            "n_items": len(ids),
            "only_x": len(set(sx) - set(sy)),
            "only_y": len(set(sy) - set(sx)),
            "x": _ci(xv.mean(), x_draws[task]),
            "y": _ci(yv.mean(), y_draws[task]),
            "diff": _ci(xv.mean() - yv.mean(), draws[task]),
        }
    if summary["tasks"] and all(t in summary["tasks"] for t in tasks):
        t_list = list(tasks)
        macro_x = np.mean([x_draws[t] for t in t_list], axis=0)
        macro_y = np.mean([y_draws[t] for t in t_list], axis=0)
        draws["macro"] = np.mean([draws[t] for t in t_list], axis=0)
        point = {k: float(np.mean([summary["tasks"][t][k]["value"] for t in t_list])) for k in "xy"}
        summary["macro"] = {
            "x": _ci(point["x"], macro_x),
            "y": _ci(point["y"], macro_y),
            "diff": _ci(point["x"] - point["y"], draws["macro"]),
        }
    return summary, draws


def p_value(draws: np.ndarray, margin: float) -> float:
    """Bootstrap one-sided p-value: the fraction of resamples at or below the margin."""
    return float(np.mean(np.asarray(draws) <= margin))


def confirmatory_tests(
    macro_draws: Mapping[str, np.ndarray], margins: Sequence[float], alpha: float = ALPHA
) -> dict[float, dict[str, dict[str, Any]]]:
    """One-sided non-inferiority tests of the candidates, Holm-adjusted, at each margin.

    Args:
        macro_draws: Bootstrap draws (points) of each candidate's macro-average difference.
        margins: Margins in points; each must be negative.
        alpha: One-sided level.

    Returns:
        ``{margin: {candidate: {p_raw, p_holm, lower_bound_one_sided, interval, outcome}}}``.

    Raises:
        ValueError: If a margin is not negative.
    """
    out: dict[float, dict[str, dict[str, Any]]] = {}
    for margin in margins:
        if margin >= 0:
            raise ValueError(f"margin must be negative (a loss in points), got {margin}")
        raw = {name: p_value(d, margin) for name, d in macro_draws.items()}
        adj = holm_adjust(raw)
        out[margin] = {}
        for name, d in macro_draws.items():
            lo, hi = interval(d)
            out[margin][name] = {
                "p_raw": raw[name],
                "p_holm": adj[name],
                "lower_bound_one_sided": float(np.quantile(d, alpha)),
                "interval": [lo, hi],
                "outcome": outcome_label(adj[name], hi, margin, alpha),
            }
    return out


# --------------------------------------------------------------------------- flags, secondary


TASK_FLAG_PP = -5.0
"""A task whose 95% interval upper bound is below this many points is a 'task-level regression'."""

TRUNCATION_FLAG_PP = 1.0
"""Truncation rates differing by more than this many percentage points are flagged."""

AIME = "aime2025"
MGSM = "mgsm"


def task_level_regressions(
    summary: Mapping[str, Any], threshold: float = TASK_FLAG_PP
) -> list[str]:
    """Tasks whose two-sided 95% interval for the difference lies entirely below ``threshold``.

    Args:
        summary: The ``summary`` of :func:`compare_arms`.
        threshold: Points (default -5, PROTOCOL.md section 7).

    Returns:
        Task names, in the summary's order.
    """
    return [t for t, v in summary["tasks"].items() if v["diff"]["hi"] < threshold]


def _rows_of(arm: Arm, tasks: Sequence[str]) -> list[dict[str, Any]]:
    wanted = set(tasks)
    return [r for run in arm.runs for r in run.rows if r.get("task") in wanted]


def _rate(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole else 0.0


def _failure_counts(rows: Sequence[Mapping[str, Any]], tasks: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"by_task": {}}
    for task in tasks:
        rs = [r for r in rows if r["task"] == task]
        trunc = sum(bool(r.get("truncated")) for r in rs)
        out["by_task"][task] = {
            "samples": len(rs),
            "truncations": trunc,
            "rate_pp": _rate(trunc, len(rs)),
        }
    n = len(rows)
    trunc = sum(bool(r.get("truncated")) for r in rows)
    errors = sum(r.get("error") is not None for r in rows)
    out.update(
        samples=n,
        truncations=trunc,
        request_errors=errors,
        rate_pp=_rate(trunc, n),
        request_error_pp=_rate(errors, n),
    )
    return out


def truncation_summary(x: Arm, y: Arm, tasks: Sequence[str]) -> dict[str, Any]:
    """Truncation and request-error counts of two arms and the flag of PROTOCOL.md section 8.1.

    The flag is raised when the overall truncation rates, or the rates of any one task, differ
    by more than one percentage point. It is a result to report, not a reason to discard.

    Args:
        x: First arm (pooled over its runs).
        y: Second arm.
        tasks: Tasks counted.

    Returns:
        Counts and rates per arm, the overall and per-task differences (``x - y``, in
        percentage points), ``flag`` and the scopes that triggered it.
    """
    cx, cy = _failure_counts(_rows_of(x, tasks), tasks), _failure_counts(_rows_of(y, tasks), tasks)
    diff = cx["rate_pp"] - cy["rate_pp"]
    by_task = {
        t: cx["by_task"][t]["rate_pp"] - cy["by_task"][t]["rate_pp"]
        for t in tasks
        if cx["by_task"][t]["samples"] and cy["by_task"][t]["samples"]
    }
    limit = TRUNCATION_FLAG_PP + 1e-9
    scopes = (["overall"] if abs(diff) > limit else []) + [
        t for t, d in by_task.items() if abs(d) > limit
    ]
    return {
        "x": cx,
        "y": cy,
        "diff_pp": diff,
        "by_task_diff_pp": by_task,
        "flagged_scopes": scopes,
        "flag": bool(scopes),
        "threshold_pp": TRUNCATION_FLAG_PP,
    }


def _aime_counts(arm: Arm) -> dict[str, list[float]]:
    """Per problem: correct non-truncated samples, non-truncated samples, truncated, all."""
    counts: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    for r in _rows_of(arm, [AIME]):
        c = counts[str(r["sample_id"])]
        c[3] += 1
        if r.get("truncated"):
            c[2] += 1
        else:
            c[1] += 1
            c[0] += compare.effective_score(r)
    return counts


def _ratio_ci(value: float, draws: np.ndarray) -> dict[str, float | None]:
    ok = draws[~np.isnan(draws)]
    if ok.size == 0:
        return {"value": value, "lo": None, "hi": None}
    lo, hi = interval(ok)
    return {"value": float(value), "lo": lo, "hi": hi}


def aime_non_truncated(x: Arm, y: Arm, n_boot: int, seed: int) -> dict[str, Any] | None:
    """AIME accuracy among samples that were not truncated (secondary, next to the headline).

    A request error stays a failure; only truncated samples leave the denominator. Problems
    are resampled (with their samples) on the same indices as the headline AIME bootstrap.

    Args:
        x: First arm.
        y: Second arm.
        n_boot: Number of resamples.
        seed: Bootstrap seed.

    Returns:
        Per arm ``{value, lo, hi, truncated, non_truncated, samples}`` in percent and the
        difference ``diff`` (``x - y``, points); ``None`` if AIME is absent or an arm has no
        non-truncated sample.
    """
    cx, cy = _aime_counts(x), _aime_counts(y)
    ids = sorted(set(cx) & set(cy))
    if not ids:
        return None
    ax = np.array([cx[i] for i in ids])
    ay = np.array([cy[i] for i in ids])
    if ax[:, 1].sum() == 0 or ay[:, 1].sum() == 0:
        return None
    idx = resample_indices(seed, AIME, len(ids), n_boot)

    def ratio(a: np.ndarray, rows: np.ndarray | slice = slice(None)) -> np.ndarray:
        num, den = a[:, 0][rows].sum(axis=-1), a[:, 1][rows].sum(axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return 100.0 * num / den

    dx, dy = ratio(ax, idx), ratio(ay, idx)
    vx, vy = float(ratio(ax)), float(ratio(ay))
    out: dict[str, Any] = {"n_items": len(ids)}
    for key, v, d, a in (("x", vx, dx, ax), ("y", vy, dy, ay)):
        out[key] = {
            **_ratio_ci(v, d),
            "truncated": int(a[:, 2].sum()),
            "non_truncated": int(a[:, 1].sum()),
            "samples": int(a[:, 3].sum()),
        }
    out["diff"] = _ratio_ci(vx - vy, dx - dy)
    return out


def mgsm_languages(x: Arm, y: Arm, n_boot: int, seed: int) -> dict[str, Any] | None:
    """Per-language MGSM accuracies and difference (``x - y``) with their own intervals.

    Args:
        x: First arm (its strata give the languages).
        y: Second arm.
        n_boot: Number of resamples.
        seed: Bootstrap seed (each language gets its own derived seed).

    Returns:
        ``{language: {n_items, x, y, diff}}`` in percent and points, languages sorted;
        ``None`` if MGSM is absent.
    """
    sx, sy = x.scores.get(MGSM, {}), y.scores.get(MGSM, {})
    ids = sorted(set(sx) & set(sy))
    if not ids:
        return None
    by_lang: dict[str, list[str]] = defaultdict(list)
    for i in ids:
        by_lang[str(x.strata[MGSM].get(i))].append(i)
    out: dict[str, Any] = {}
    for lang in sorted(by_lang):
        key = int.from_bytes(hashlib.sha256(f"{MGSM}/{lang}".encode()).digest()[:4], "big")
        s = seed + key
        xv = 100.0 * np.array([sx[i] for i in by_lang[lang]])
        yv = 100.0 * np.array([sy[i] for i in by_lang[lang]])
        mx, lx, hx = stats.bootstrap_mean_ci(xv, n_boot=n_boot, seed=s)
        my, ly, hy = stats.bootstrap_mean_ci(yv, n_boot=n_boot, seed=s)
        md, ld, hd = stats.paired_bootstrap_diff_ci(yv, xv, n_boot=n_boot, seed=s)
        out[lang] = {
            "n_items": len(by_lang[lang]),
            "x": {"value": mx, "lo": lx, "hi": hx},
            "y": {"value": my, "lo": ly, "hi": hy},
            "diff": {"value": md, "lo": ld, "hi": hd},
        }
    return out


# --------------------------------------------------------------------------- the whole wave


class GateRefused(Exception):
    """The gate refused a comparison pair that is not exploratory.

    Attributes:
        problems: Every reason, each with the pairs it concerns.
    """

    def __init__(self, problems: list[str]) -> None:
        """Keep the reasons."""
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass
class Config:
    """Everything the analysis needs (the command line, parsed).

    Attributes:
        runs: Arm name to its run directories (several: pooled by averaging item scores).
        reference: Arm every candidate is compared with.
        candidates: The confirmatory family, Holm-adjusted.
        exploratory: Arms reported descriptively (a failed gate is a warning for them).
        contrasts: ``(A, B)`` pairs for descriptive contrasts ``A - B``.
        calibration: Two single runs of the reference ``((name_a, dir), (name_b, dir))``.
        expected: Items and epochs per task.
        margin: Non-inferiority margin in points (negative).
        sensitivity_margins: Other margins to show.
        seed: Bootstrap seed.
        n_boot: Bootstrap resamples.
        accept_legacy_aime_retries: Accept AIME rows of a run that does not record its retries
            (``model_attempts``), with a warning, instead of refusing them.
    """

    runs: dict[str, list[Path]]
    reference: str
    candidates: list[str]
    exploratory: list[str] = field(default_factory=list)
    contrasts: list[tuple[str, str]] = field(default_factory=list)
    calibration: tuple[tuple[str, Path], tuple[str, Path]] | None = None
    expected: dict[str, dict[str, int]] = field(default_factory=lambda: load_expected(None))
    margin: float = -3.0
    sensitivity_margins: tuple[float, ...] = (-2.0, -5.0)
    seed: int = 0
    n_boot: int = 10_000
    accept_legacy_aime_retries: bool = False


def _validate(cfg: Config) -> None:
    known = set(cfg.runs)
    if cfg.reference not in known:
        raise ValueError(f"reference {cfg.reference!r} is not among the --run arms")
    if not cfg.candidates:
        raise ValueError("no candidates given")
    for name in [*cfg.candidates, *cfg.exploratory, *(n for c in cfg.contrasts for n in c)]:
        if name not in known:
            raise ValueError(f"arm {name!r} is not among the --run arms")
    both = sorted(set(cfg.candidates) & set(cfg.exploratory))
    if both:
        raise ValueError(f"{', '.join(both)} listed both as candidate and as exploratory")
    if cfg.reference in cfg.candidates or cfg.reference in cfg.exploratory:
        raise ValueError("the reference cannot be a candidate or exploratory arm")
    if cfg.calibration:
        names = [n for n, _ in cfg.calibration]
        if len(set(names)) != 2 or set(names) & known:
            raise ValueError("calibration names must be two new names, different from the arms")
    if cfg.margin >= 0 or any(m >= 0 for m in cfg.sensitivity_margins):
        raise ValueError("margin must be negative (a loss in points)")
    if cfg.n_boot < 1:
        raise ValueError("resamples must be at least 1")


def summarize_run(run: Run, expected: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    """Settings, hashes and failure counts of one run, as read from its files."""
    info = run.info
    rows = [r for r in run.rows if r.get("task") in expected]
    per_task = {}
    for task in expected:
        rs = [r for r in rows if r["task"] == task]
        per_task[task] = {
            "samples": len(rs),
            "request_errors": sum(r.get("error") is not None for r in rs),
            "truncations": sum(bool(r.get("truncated")) for r in rs),
        }
    errors = sum(v["request_errors"] for v in per_task.values())
    trunc = sum(v["truncations"] for v in per_task.values())
    tasks_info = info.get("tasks") or {}
    sampling = info.get("sampling") or {}
    return {
        "dir": str(run.path),
        "run_info_sha256": run.sha256.get("run_info"),
        "samples_sha256": run.sha256.get("samples"),
        "items_sha256": run.sha256.get("items"),
        "quantization_info_sha256": run.sha256.get("quantization_info"),
        "status": info.get("status"),
        "model": info.get("model"),
        "endpoint": info.get("endpoint"),
        "settings": {
            "reasoning_effort": info.get("reasoning_effort"),
            "sampling": {
                k: sampling.get(k) for k in ("temperature", "top_p", "top_k", "max_tokens", "seed")
            },
            "sandbox": (info.get("sandbox") or {}).get("mode"),
            "epochs": {t: (tasks_info.get(t) or {}).get("epochs") for t in expected},
            "items_seed": info.get("seed"),
            "git_commit": (info.get("git") or {}).get("commit"),
            "git_dirty": (info.get("git") or {}).get("dirty"),
            "code_sha256": info.get("code_sha256"),
            "started_utc": info.get("started_utc"),
            "ended_utc": info.get("ended_utc"),
        },
        "quantization_recipe": ((run.quantization or {}).get("recipe") or {}).get("name"),
        "counts": {
            "samples": len(rows),
            "request_errors": errors,
            "truncations": trunc,
            "request_error_pct": _rate(errors, len(rows)),
            "truncation_pct": _rate(trunc, len(rows)),
        },
        "per_task": per_task,
    }


def _pair_block(
    x: Arm, y: Arm, kind: str, cfg: Config
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    tasks = list(cfg.expected)
    summary, draws = compare_arms(x, y, tasks, cfg.n_boot, cfg.seed)
    trunc = truncation_summary(x, y, tasks)
    block: dict[str, Any] = {
        "kind": kind,
        "x": x.name,
        "y": y.name,
        "summary": summary,
        "truncation": trunc,
        "aime_non_truncated": aime_non_truncated(x, y, cfg.n_boot, cfg.seed),
        "mgsm_languages": mgsm_languages(x, y, cfg.n_boot, cfg.seed),
        "flags": {
            "task_level_regression": task_level_regressions(summary),
            "truncation_rate_difference": trunc["flag"],
            "truncation_scopes": trunc["flagged_scopes"],
        },
    }
    return block, draws


def analyze(cfg: Config) -> dict[str, Any]:
    """Run the pre-registered analysis of a wave.

    Args:
        cfg: The configuration.

    Returns:
        A JSON-serialisable dict with every number of the report.

    Raises:
        ValueError: For inconsistent arm names or margins.
        GateRefused: If the gate fails for a pair that is not exploratory.
    """
    _validate(cfg)
    loaded: dict[str, list[Run]] = {
        name: [load_run_dir(d) for d in dirs] for name, dirs in cfg.runs.items()
    }
    roles = {cfg.reference: "reference"}
    roles |= dict.fromkeys(cfg.candidates, "candidate") | dict.fromkeys(
        cfg.exploratory, "exploratory"
    )
    pairs: list[tuple[str, str, str, str]] = [
        (f"{c} vs {cfg.reference}", "candidate", c, cfg.reference) for c in cfg.candidates
    ]
    pairs += [(f"{e} vs {cfg.reference}", "exploratory", e, cfg.reference) for e in cfg.exploratory]
    for a, b in cfg.contrasts:
        exploratory = "exploratory" in (roles.get(a), roles.get(b))
        pairs.append((f"{a} vs {b}", "exploratory-contrast" if exploratory else "contrast", a, b))
    if cfg.calibration:
        (na, da), (nb, db) = cfg.calibration
        loaded[na], loaded[nb] = [load_run_dir(da)], [load_run_dir(db)]
        pairs.append((f"{na} vs {nb}", "calibration", na, nb))

    gates: dict[str, Any] = {}
    refusals: dict[str, list[str]] = defaultdict(list)
    for label, kind, a, b in pairs:
        gate = gate_pair(
            loaded[a],
            loaded[b],
            cfg.expected,
            accept_legacy_aime_retries=cfg.accept_legacy_aime_retries,
        )
        soft = kind.startswith("exploratory")
        status = "pass" if gate.ok else ("warn-exploratory" if soft else "fail")
        gates[label] = {
            "kind": kind,
            "status": status,
            "problems": gate.problems,
            "warnings": gate.warnings,
        }
        if status == "fail":
            for p in gate.problems:
                refusals[p].append(label)
    if refusals:
        raise GateRefused([f"{p} (pairs: {', '.join(ls)})" for p, ls in refusals.items()])

    tasks = list(cfg.expected)
    arms = {name: build_arm(name, runs, tasks) for name, runs in loaded.items()}
    result: dict[str, Any] = {
        "script": {"file": Path(__file__).name, "sha256": _sha256_file(Path(__file__))},
        "bootstrap": {"seed": cfg.seed, "resamples": cfg.n_boot, "confidence": CONFIDENCE},
        "protocol": {
            "alpha_one_sided": ALPHA,
            "task_flag_threshold_pp": TASK_FLAG_PP,
            "truncation_flag_pp": TRUNCATION_FLAG_PP,
            "request_error_max_pct": REQUEST_ERROR_MAX_PER_MILLE / 10,
            "tasks": tasks,
        },
        "expected": cfg.expected,
        "expected_total_samples": total_samples(cfg.expected),
        "reference": cfg.reference,
        "arms": {
            name: {
                "role": roles.get(name, "calibration"),
                "n_runs": len(runs),
                "runs": [summarize_run(r, cfg.expected) for r in runs],
            }
            for name, runs in loaded.items()
        },
        "gate": {"pairs": gates, "accept_legacy_aime_retries": cfg.accept_legacy_aime_retries},
        "candidates": {},
        "exploratory": {},
        "contrasts": {},
        "calibration": None,
    }
    macro_draws: dict[str, np.ndarray] = {}
    for label, kind, a, b in pairs:
        block, draws = _pair_block(arms[a], arms[b], kind, cfg)
        block["gate_status"] = gates[label]["status"]
        if kind == "candidate":
            result["candidates"][a] = block
            macro_draws[a] = draws["macro"]
        elif kind == "exploratory":
            result["exploratory"][a] = block
        elif kind.endswith("contrast"):
            result["contrasts"][label] = block
        else:
            macro = block["summary"]["macro"]["diff"]
            block["contains_zero"] = bool(macro["lo"] <= 0.0 <= macro["hi"])
            block["width_pp"] = macro["hi"] - macro["lo"]
            result["calibration"] = block
    margins = [cfg.margin, *[m for m in cfg.sensitivity_margins if m != cfg.margin]]
    tests = confirmatory_tests(macro_draws, margins)
    result["confirmatory"] = {
        "primary_margin": cfg.margin,
        "margins": margins,
        "alpha_one_sided": ALPHA,
        "family": list(cfg.candidates),
        "tests": {f"{m:g}": per for m, per in tests.items()},
    }
    for name in cfg.candidates:
        result["candidates"][name]["tests"] = {f"{m:g}": tests[m][name] for m in margins}
        result["candidates"][name]["outcome"] = tests[cfg.margin][name]["outcome"]
    return result


# --------------------------------------------------------------------------- report


def _pp(d: Mapping[str, float | None] | None) -> str:
    """Difference in points with its interval: ``+1.2 [-0.5, +2.9]``."""
    if not d or d.get("lo") is None:
        return "n/a" if not d else f"{d['value']:+.1f} [n/a]"
    return f"{d['value']:+.1f} [{d['lo']:+.1f}, {d['hi']:+.1f}]"


def _acc(d: Mapping[str, float | None] | None) -> str:
    """Accuracy in percent with its interval: ``71.2 [68.0, 74.1]``."""
    if not d or d.get("lo") is None:
        return "n/a" if not d else f"{d['value']:.1f} [n/a]"
    return f"{d['value']:.1f} [{d['lo']:.1f}, {d['hi']:.1f}]"


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


def _outcome_cell(test: Mapping[str, Any]) -> str:
    return f"{test['outcome']} ({test['p_holm']:.4f})"


def _macro_pp(block: Mapping[str, Any]) -> str:
    macro = block["summary"]["macro"]
    return _pp(macro["diff"]) if macro else "n/a"


def _width(macro: Mapping[str, Any] | None) -> float | None:
    return None if not macro else macro["diff"]["hi"] - macro["diff"]["lo"]


def _section(title: str, conclusion: str, body: Sequence[str]) -> list[str]:
    return [f"## {title}", "", f"**Conclusion:** {conclusion}", "", *body, ""]


def _summary_section(res: dict[str, Any]) -> list[str]:
    conf = res["confirmatory"]
    m = f"{conf['primary_margin']:g}"
    cands = res["candidates"]
    outcomes = "; ".join(f"{n}: {c['outcome']}" for n, c in cands.items())
    concl = (
        f"At the margin of {m} points on the macro-average "
        f"(one-sided alpha {conf['alpha_one_sided']}, "
        f"Holm over {len(cands)} candidate(s)), {outcomes}."
    )
    rows = []
    for n, c in cands.items():
        flags = c["flags"]
        marks = []
        if flags["task_level_regression"]:
            marks.append("task-level regression")
        if flags["truncation_rate_difference"]:
            marks.append("truncation-rate difference")
        rows.append(
            [
                n,
                _pp(c["summary"]["macro"]["diff"]),
                c["outcome"],
                f"{c['summary']['macro']['y']['value']:.1f}",
                f"{c['summary']['macro']['x']['value']:.1f}",
                "; ".join(marks) or "none",
            ]
        )
    body = _table(
        [
            "Candidate",
            "Macro difference, pp [95% interval]",
            "Outcome",
            f"{res['reference']} macro %",
            "Candidate macro %",
            "Flags",
        ],
        rows,
    )
    single = [n for n, a in res["arms"].items() if a["n_runs"] < 2 and a["role"] != "calibration"]
    body += [""]
    if single:
        body.append(
            f"Arms with a single run, so provisional under PROTOCOL.md amendment 9 (stage 1): "
            f"{', '.join(single)}."
        )
    else:
        body.append("Every arm has two or more runs, pooled by averaging item scores.")
    body.append(
        "Estimand: difference in the equally weighted macro-average of the five task scores, in "
        "percentage points, over the fixed items; request errors and truncations count as 0."
    )
    cal = res["calibration"]
    if cal:
        verdict = "contains 0" if cal["contains_zero"] else "does NOT contain 0"
        body.append(
            f"Calibration ({cal['x']} versus {cal['y']}): the interval {verdict}, width "
            f"{cal['width_pp']:.1f} pp."
        )
    return _section("1. Summary", concl, body)


def _gate_section(res: dict[str, Any]) -> list[str]:
    pairs = res["gate"]["pairs"]
    failed = [k for k, g in pairs.items() if g["status"] == "warn-exploratory"]
    flagged = [n for n, c in res["candidates"].items() if c["flags"]["truncation_rate_difference"]]
    concl = f"all {len(pairs)} comparison pair(s) passed the gate"
    if failed:
        concl = (
            f"every confirmatory pair passed the gate; gate failed for exploratory pair(s) "
            f"{', '.join(failed)}, computed anyway on the common items and marked as such"
        )
    if flagged:
        concl += f"; truncation rates differ by more than 1 point for {', '.join(flagged)}"
    concl += "."
    body = ["### Pairs", ""]
    body += _table(
        ["Pair", "Kind", "Status", "Problems", "Warnings"],
        [
            [
                k,
                g["kind"],
                "gate failed (exploratory, computed anyway)"
                if g["status"] == "warn-exploratory"
                else g["status"],
                len(g["problems"]),
                len(g["warnings"]),
            ]
            for k, g in pairs.items()
        ],
    )
    for k, g in pairs.items():
        for p in g["problems"]:
            body.append(f"- {k}: {p}")
        for w in g["warnings"]:
            body.append(f"- {k}, warning: {w}")
    body += ["", "### Settings per run", ""]
    srows, crows = [], []
    for name, arm in res["arms"].items():
        for r in arm["runs"]:
            s = r["settings"]
            samp = s["sampling"]
            srows.append(
                [
                    name,
                    f"`{r['dir']}`",
                    r["status"],
                    s["reasoning_effort"],
                    f"T {samp['temperature']}, top-p {samp['top_p']}, top-k {samp['top_k']}, "
                    f"max tokens {samp['max_tokens']}",
                    s["sandbox"],
                    ", ".join(f"{t} {e}" for t, e in s["epochs"].items()),
                    s["git_commit"],
                    s["code_sha256"],
                    r["quantization_recipe"] or "none",
                ]
            )
            c = r["counts"]
            crows.append(
                [
                    name,
                    f"`{r['dir']}`",
                    c["samples"],
                    c["request_errors"],
                    c["truncations"],
                    f"{c['request_error_pct']:.2f}%",
                    f"{c['truncation_pct']:.2f}%",
                ]
            )
    body += _table(
        [
            "Arm",
            "Run",
            "Status",
            "Reasoning effort",
            "Sampling",
            "Sandbox",
            "Epochs per task",
            "Git commit",
            "Code sha256",
            "Quantization recipe",
        ],
        srows,
    )
    body += ["", "### Request errors and truncations per run (counted as failures)", ""]
    body += _table(
        ["Arm", "Run", "Samples", "Request errors", "Truncations", "Errors %", "Truncated %"],
        crows,
    )
    return _section("2. Gate and integrity", concl, body)


def _confirmatory_section(res: dict[str, Any]) -> list[str]:
    conf = res["confirmatory"]
    prim = f"{conf['primary_margin']:g}"
    tests = conf["tests"][prim]
    ok = [n for n, t in tests.items() if t["outcome"] == "NON-INFERIOR"]
    concl = (
        f"at the margin of {prim} points, {len(ok)} of {len(tests)} candidate(s) "
        f"{'are' if len(ok) != 1 else 'is'} NON-INFERIOR after Holm adjustment"
        + (f" ({', '.join(ok)})" if ok else "")
        + "; outcomes: "
        + "; ".join(f"{n} {t['outcome']}" for n, t in tests.items())
        + "."
    )
    rows = []
    for n, t in tests.items():
        d = res["candidates"][n]["summary"]["macro"]["diff"]
        rows.append(
            [
                n,
                _pp(d),
                f"{d['hi'] - d['lo']:.1f}",
                f"{t['lower_bound_one_sided']:+.2f}",
                f"{t['p_raw']:.4f}",
                f"{t['p_holm']:.4f}",
                t["outcome"],
            ]
        )
    body = [
        f"One-sided non-inferiority against {prim} points; bootstrap p = fraction of "
        f"{res['bootstrap']['resamples']} resamples with a macro difference at or below the "
        f"margin; Holm over {len(tests)} test(s) at one-sided alpha "
        f"{conf['alpha_one_sided']}. The one-sided "
        "97.5% lower bound equals the lower end of the two-sided 95% interval.",
        "",
        *_table(
            [
                "Candidate",
                "Macro difference, pp [two-sided 95% interval]",
                "Interval width, pp",
                "One-sided 97.5% lower bound",
                "p (raw)",
                "p (Holm)",
                "Outcome",
            ],
            rows,
        ),
        "",
        "### Sensitivity to the margin (outcome, Holm-adjusted p)",
        "",
    ]
    margins = conf["margins"]
    srows = []
    for n in tests:
        srows.append([n] + [_outcome_cell(conf["tests"][f"{m:g}"][n]) for m in margins])
    body += _table(["Candidate", *[f"margin {m:g}" for m in margins]], srows)
    cal = res["calibration"]
    if cal:
        body += [
            "",
            f"Width of pure run-to-run noise ({cal['x']} versus {cal['y']}, one run each): "
            f"{cal['width_pp']:.1f} pp, against the candidates' widths above. It is not used "
            "in any test.",
        ]
    return _section("3. Confirmatory tests", concl, body)


def _per_task_section(res: dict[str, Any]) -> list[str]:
    regress = {n: c["flags"]["task_level_regression"] for n, c in res["candidates"].items()}
    hit = {n: t for n, t in regress.items() if t}
    thr = res["protocol"]["task_flag_threshold_pp"]
    concl = (
        "task-level regression (interval upper bound below "
        f"{thr:g} points) for " + "; ".join(f"{n} ({', '.join(t)})" for n, t in hit.items())
        if hit
        else f"no candidate has a task whose interval upper bound is below {thr:g} points."
    )
    if hit:
        concl += "."
    body: list[str] = []
    for n, c in res["candidates"].items():
        s = c["summary"]
        body += [f"### {n} versus {res['reference']}", ""]
        rows = [
            [
                t,
                v["n_items"],
                _acc(v["y"]),
                _acc(v["x"]),
                _pp(v["diff"]),
                "task-level regression" if t in regress[n] else "",
            ]
            for t, v in s["tasks"].items()
        ]
        m = s["macro"]
        rows.append(["macro-average", "", _acc(m["y"]), _acc(m["x"]), _pp(m["diff"]), ""])
        body += _table(
            [
                "Task",
                "Items",
                f"{res['reference']} %",
                f"{n} %",
                "Difference, pp [95% interval]",
                "Flag",
            ],
            rows,
        )
        body.append("")
    return _section("4. Per-task differences", concl, body)


def _flag_lines(res: dict[str, Any], blocks: Mapping[str, Any]) -> list[str]:
    lines = []
    for n, c in blocks.items():
        f, t = c["flags"], c["truncation"]
        if f["task_level_regression"]:
            lines.append(
                f"- {n}: task-level regression in {', '.join(f['task_level_regression'])}."
            )
        if f["truncation_rate_difference"]:
            lines.append(
                f"- {n}: truncation-rate difference above {t['threshold_pp']:g} pp "
                f"(scopes: {', '.join(t['flagged_scopes'])}); {n} {t['x']['rate_pp']:.2f}% "
                f"({t['x']['truncations']}/{t['x']['samples']}) against {t['y']['rate_pp']:.2f}% "
                f"({t['y']['truncations']}/{t['y']['samples']}) for {res['reference']}. A result "
                "to report, not a reason to discard; truncated answers count as failures."
            )
    return lines


def _flags_section(res: dict[str, Any]) -> list[str]:
    allb = {**res["candidates"], **res["exploratory"]}
    reg = [n for n, c in allb.items() if c["flags"]["task_level_regression"]]
    tr = [n for n, c in allb.items() if c["flags"]["truncation_rate_difference"]]
    concl = (
        f"task-level regression: {', '.join(reg) or 'none'}; "
        "truncation-rate difference above 1 pp: "
        f"{', '.join(tr) or 'none'}."
    )
    body = _flag_lines(res, allb) or ["No flag raised."]
    return _section("5. Flags", concl, body)


def _aime_section(res: dict[str, Any]) -> list[str]:
    rows, parts = [], []
    for n, c in res["candidates"].items():
        a = c["aime_non_truncated"]
        head = c["summary"]["tasks"].get(AIME)
        if not a or not head:
            continue
        parts.append(f"{n} {_pp(a['diff'])}")
        rows.append(
            [
                n,
                _pp(head["diff"]),
                _pp(a["diff"]),
                f"{a['x']['truncated']} / {a['y']['truncated']}",
                _acc(a["x"]),
                _acc(a["y"]),
            ]
        )
    concl = (
        "AIME difference among non-truncated samples only (points): " + "; ".join(parts) + ". "
        "The headline counts truncated samples as failures."
        if parts
        else "AIME is not part of this analysis."
    )
    body = _table(
        [
            "Candidate",
            "Headline AIME difference (truncations = failures)",
            "Difference among non-truncated samples",
            f"Truncated samples (arm / {res['reference']})",
            "Arm accuracy among non-truncated %",
            f"{res['reference']} accuracy among non-truncated %",
        ],
        rows,
    )
    return _section("6. AIME accuracy among non-truncated samples (secondary)", concl, body)


def _mgsm_section(res: dict[str, Any]) -> list[str]:
    body: list[str] = []
    worst = []
    for n, c in res["candidates"].items():
        langs = c["mgsm_languages"]
        if not langs:
            continue
        lang, v = min(langs.items(), key=lambda kv: kv[1]["diff"]["value"])
        worst.append(f"{n}: {lang} {_pp(v['diff'])}")
        body += [f"### {n} versus {res['reference']}", ""]
        body += _table(
            [
                "Language",
                "Items",
                f"{res['reference']} %",
                f"{n} %",
                "Difference, pp [95% interval]",
            ],
            [
                [lang_, v_["n_items"], _acc(v_["y"]), _acc(v_["x"]), _pp(v_["diff"])]
                for lang_, v_ in langs.items()
            ],
        )
        body.append("")
    concl = (
        "descriptive only; the largest drop in a language is " + "; ".join(worst) + "."
        if worst
        else "MGSM is not part of this analysis."
    )
    return _section("7. MGSM per language", concl, body)


def _calibration_section(res: dict[str, Any]) -> list[str]:
    cal = res["calibration"]
    if not cal:
        return _section(
            "8. Calibration of the interval",
            "not provided (no --calibration), so the width of pure run-to-run noise is not shown.",
            ["Give the two runs of the reference as separate inputs with --calibration."],
        )
    d = cal["summary"]["macro"]["diff"]
    verdict = "contains 0" if cal["contains_zero"] else "does NOT contain 0"
    concl = (
        f"the interval of the macro-average difference between the two reference runs "
        f"({cal['x']} minus {cal['y']}) {verdict}: {_pp(d)}, width {cal['width_pp']:.1f} pp."
    )
    widths = ", ".join(
        f"{n} {_width(c['summary']['macro']):.1f}" for n, c in res["candidates"].items()
    )
    body = [
        "The check expects the interval to contain 0; it does not alter any test.",
        f"Candidate interval widths (pp, each against the pooled reference): {widths}.",
        "",
        *_table(
            ["Task", "Difference, pp [95% interval]"],
            [[t, _pp(v["diff"])] for t, v in cal["summary"]["tasks"].items()]
            + [["macro-average", _pp(d)]],
        ),
    ]
    if not cal["contains_zero"]:
        body.append("")
        body.append(
            "CALIBRATION FAILED: the interval excludes 0, so the intervals of this report may be "
            "too narrow for the run-to-run noise. Read every outcome with that in mind."
        )
    return _section("8. Calibration of the interval", concl, body)


def _exploratory_section(res: dict[str, Any]) -> list[str]:
    ex = res["exploratory"]
    if not ex:
        return _section(
            "9. Exploratory arms (descriptive only)",
            "none given: no exploratory arm was requested.",
            [
                "Arms are listed with --exploratory; they get no outcome label and no Holm "
                "membership."
            ],
        )
    tasks = res["protocol"]["tasks"]
    pairs = res["gate"]["pairs"]
    failed = [n for n in ex if pairs[f"{n} vs {res['reference']}"]["status"] == "warn-exploratory"]
    concl = (
        "descriptive only (difference and interval, no outcome label, no Holm membership): "
        + "; ".join(
            f"{n} {_pp(c['summary']['macro']['diff']) if c['summary']['macro'] else 'n/a'}"
            for n, c in ex.items()
        )
        + "."
    )
    if failed:
        concl += f" Gate failed for {', '.join(failed)} (see below); computed on the common items."
    rows = []
    for n, c in ex.items():
        s = c["summary"]
        rows.append(
            [n]
            + [_pp(s["tasks"][t]["diff"]) if t in s["tasks"] else "n/a" for t in tasks]
            + [_pp(s["macro"]["diff"]) if s["macro"] else "n/a"]
        )
    body = _table(["Arm", *tasks, "Macro-average"], rows)
    body.append("")
    for n in ex:
        g = pairs[f"{n} vs {res['reference']}"]
        if g["status"] == "warn-exploratory":
            body.append(f"Gate failed for {n} (exploratory, so a warning only):")
            body += [f"- {p}" for p in g["problems"]]
    body += _flag_lines(res, ex)
    return _section("9. Exploratory arms (descriptive only)", concl, body)


def _contrast_section(res: dict[str, Any]) -> list[str]:
    cs = res["contrasts"]
    if not cs:
        return _section(
            "10. Contrasts between arms (descriptive)",
            "none given: no --contrast was requested.",
            ["Pairs are given as --contrast A,B and report A minus B."],
        )
    tasks = res["protocol"]["tasks"]
    concl = (
        "descriptive differences with 95% intervals, no outcome label: "
        + "; ".join(f"{c['x']} versus {c['y']} {_macro_pp(c)}" for c in cs.values())
        + "."
    )
    rows = []
    for c in cs.values():
        s = c["summary"]
        rows.append(
            [f"{c['x']} versus {c['y']}"]
            + [_pp(s["tasks"][t]["diff"]) if t in s["tasks"] else "n/a" for t in tasks]
            + [_pp(s["macro"]["diff"]) if s["macro"] else "n/a"]
        )
    body = _table(["Contrast (first minus second)", *tasks, "Macro-average"], rows)
    return _section("10. Contrasts between arms (descriptive)", concl, body)


def _provenance_section(res: dict[str, Any]) -> list[str]:
    b = res["bootstrap"]
    concl = (
        f"bootstrap seed {b['seed']}, {b['resamples']} resamples; every number comes from the run "
        "files hashed below and from this script, whose own hash is given."
    )
    body = [
        f"Bootstrap: seed {b['seed']}, {b['resamples']} resamples, {b['confidence']:.0%} two-sided "
        "percentile intervals; items are resampled independently within each task and the "
        "five task "
        "means are then averaged with equal weight; the reference is pooled over its runs.",
        f"Script: `{res['script']['file']}`, sha256 `{res['script']['sha256']}`.",
        f"Expected full run: {res['expected_total_samples']} samples ("
        + ", ".join(f"{t} {v['items']} x {v['epochs']}" for t, v in res["expected"].items())
        + ").",
        "",
        "Limits (PROTOCOL.md section 7): MGSM translations of one problem are separate items "
        "and are "
        "not clustered by source problem, so the MGSM interval is somewhat too narrow; AIME has 30 "
        "problems, so its interval and the macro interval are wide and the bootstrap coverage is "
        "approximate. A bootstrap p-value is a fraction of resamples: its resolution is "
        "1/resamples.",
        "",
        *_table(
            [
                "Arm",
                "Run directory",
                "run_info.json sha256",
                "samples.jsonl sha256",
                "items.json sha256",
            ],
            [
                [
                    name,
                    f"`{r['dir']}`",
                    r["run_info_sha256"],
                    r["samples_sha256"],
                    r["items_sha256"],
                ]
                for name, arm in res["arms"].items()
                for r in arm["runs"]
            ],
        ),
    ]
    return _section("11. Provenance and limits", concl, body)


def render_markdown(res: dict[str, Any]) -> str:
    """Render :func:`analyze` output as markdown; every number comes from ``res``.

    Args:
        res: The analysis result.

    Returns:
        The report. Each section opens with its conclusion.
    """
    lines = [
        "# Accuracy wave: pre-registered analysis",
        "",
        f"Reference `{res['reference']}`; confirmatory family "
        f"{', '.join(res['confirmatory']['family'])}. "
        "Outcome labels are NON-INFERIOR, WORSE or INCONCLUSIVE, always read with the interval.",
        "",
    ]
    for build in (
        _summary_section,
        _gate_section,
        _confirmatory_section,
        _per_task_section,
        _flags_section,
        _aime_section,
        _mgsm_section,
        _calibration_section,
        _exploratory_section,
        _contrast_section,
        _provenance_section,
    ):
        lines += build(res)
    return "\n".join(lines).rstrip("\n") + "\n"


# --------------------------------------------------------------------------- command line


def _arg_run(text: str) -> tuple[str, list[Path]]:
    try:
        return parse_run_spec(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _arg_contrast(text: str) -> tuple[str, str]:
    parts = text.split(",")
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError(f"invalid contrast {text!r}: use A,B")
    return parts[0], parts[1]


def _arg_calibration(text: str) -> tuple[tuple[str, Path], tuple[str, Path]]:
    parts = [p.partition("=") for p in text.split(",")]
    if len(parts) != 2 or any(not n or not sep or not d for n, sep, d in parts):
        raise argparse.ArgumentTypeError(f"invalid calibration {text!r}: use NAME_A=DIR,NAME_B=DIR")
    return (parts[0][0], Path(parts[0][2])), (parts[1][0], Path(parts[1][2]))


def _arg_names(text: str) -> list[str]:
    names = text.split(",")
    if not all(names):
        raise argparse.ArgumentTypeError(f"invalid list {text!r}: use NAME,NAME")
    return names


class _Extend(argparse.Action):
    """Append the values of a comma-separated list to the destination list."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        setattr(namespace, self.dest, [*(getattr(namespace, self.dest) or []), *values])


def build_parser() -> argparse.ArgumentParser:
    """The command-line parser."""
    ap = argparse.ArgumentParser(
        prog="analyze_wave.py",
        description="Pre-registered analysis of the accuracy wave: gate, paired bootstrap on the "
        "macro-average, Holm-adjusted non-inferiority tests, flags, calibration, report.",
        epilog="Exit codes: 0 done; 2 invalid arguments; "
        "3 the gate refused a pair (nothing written).",
    )
    ap.add_argument(
        "--run",
        action="append",
        type=_arg_run,
        required=True,
        metavar="NAME=DIR[,DIR2...]",
        help="an arm and its run directory; several directories mean several completed runs of the "
        "same arm, pooled by averaging item scores; repeatable",
    )
    ap.add_argument("--reference", required=True, help="the reference arm (for example B0)")
    ap.add_argument(
        "--candidates",
        required=True,
        type=_arg_names,
        help="comma-separated confirmatory arms, compared with the reference and Holm-adjusted",
    )
    ap.add_argument(
        "--exploratory",
        action=_Extend,
        type=_arg_names,
        default=[],
        metavar="NAME[,NAME...]",
        help="arms reported descriptively (no label, no Holm); a failed gate only warns for them; "
        "repeatable",
    )
    ap.add_argument(
        "--contrast",
        action="append",
        dest="contrasts",
        type=_arg_contrast,
        default=[],
        metavar="A,B",
        help="descriptive contrast A minus B between two arms (for example F2,F1); repeatable",
    )
    ap.add_argument(
        "--calibration",
        type=_arg_calibration,
        metavar="NAME_A=DIR,NAME_B=DIR",
        help="two separate runs of the reference: report their paired macro difference and the "
        "width of its interval (never used in a test)",
    )
    ap.add_argument(
        "--expected",
        help="JSON {task: {items, epochs}}; default the protocol's full run (1,783 samples: mgsm "
        "550, humaneval_plus 164, mbpp_plus 378, ifeval 541, aime2025 30 x 5)",
    )
    ap.add_argument(
        "--margin", type=float, default=-3.0, help="non-inferiority margin in points (negative)"
    )
    ap.add_argument(
        "--accept-legacy-aime-retries",
        action="store_true",
        help="accept the AIME rows of a run made before the harness recorded retries "
        "(client timeout 600 s, no model_attempts in its rows): they are refused by default "
        "because a request longer than 600 s was cancelled and redrawn; with this option the gate "
        "only warns, stating what was measured",
    )
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed (recorded)")
    ap.add_argument("--resamples", type=int, default=10_000, help="bootstrap resamples")
    ap.add_argument("--out", required=True, help="markdown report to write")
    ap.add_argument("--json", required=True, help="JSON with every number and the provenance")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Arguments (default: ``sys.argv[1:]``).

    Returns:
        0 on success, 3 if the gate refused (nothing is written, the reasons go to stderr).
    """
    ap = build_parser()
    ns = ap.parse_args(argv)
    runs: dict[str, list[Path]] = {}
    for name, dirs in ns.run:
        if name in runs:
            ap.error(f"arm {name!r} given twice with --run (use NAME=DIR1,DIR2 to pool two runs)")
        runs[name] = dirs
    try:
        cfg = Config(
            runs=runs,
            reference=ns.reference,
            candidates=ns.candidates,
            exploratory=ns.exploratory,
            contrasts=ns.contrasts,
            calibration=ns.calibration,
            expected=load_expected(ns.expected),
            margin=ns.margin,
            seed=ns.seed,
            n_boot=ns.resamples,
            accept_legacy_aime_retries=ns.accept_legacy_aime_retries,
        )
        result = analyze(cfg)
    except GateRefused as exc:
        lines = ["analyze_wave.py: REFUSED by the gate (PROTOCOL.md section 8; amendment 9):"]
        lines += [f"  - {p}" for p in exc.problems]
        lines.append("Nothing was written.")
        sys.stderr.write("\n".join(lines) + "\n")
        return 3
    except (ValueError, OSError) as exc:
        ap.error(str(exc))
    md = render_markdown(result)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    Path(ns.out).write_text(md, encoding="utf-8")
    Path(ns.json).write_text(payload, encoding="utf-8")
    sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
