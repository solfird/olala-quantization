"""Builder of fake run directories in the format ``run_tasks.py`` writes (no model, no server).

A run directory holds ``samples.jsonl``, ``items.json``, ``run_info.json`` and, for a quantized
arm, ``quantization_info.json``. The builder reproduces the fields ``analyze_wave.py`` reads, with
the same names and types as the real harness, so a test can break one thing at a time.
"""

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SMALL_EXPECTED: dict[str, dict[str, int]] = {
    "mgsm": {"items": 20, "epochs": 1},
    "humaneval_plus": {"items": 10, "epochs": 1},
    "mbpp_plus": {"items": 12, "epochs": 1},
    "ifeval": {"items": 14, "epochs": 1},
    "aime2025": {"items": 6, "epochs": 5},
}
"""A scaled-down protocol (same five tasks, same epochs) so tests stay fast."""

LANGS = ("en", "fr", "de", "ja")


def sha256_of_ids(ids: list[str]) -> str:
    """Same formula as ``run_tasks.sha256_of_ids`` (a test pins that they agree)."""
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def task_item_ids(task: str, n: int) -> list[str]:
    """Item ids of a fake task; MGSM ids carry their language."""
    if task == "mgsm":
        return [f"mgsm-{LANGS[i % len(LANGS)]}-{i // len(LANGS):02d}" for i in range(n)]
    return [f"{task}-{i:03d}" for i in range(n)]


def build_run(
    path: Path,
    *,
    expected: dict[str, dict[str, int]] | None = None,
    label: str = "arm",
    p: float | dict[str, float] = 0.7,
    shift: float = 0.0,
    run_seed: int = 0,
    status: str = "complete",
    scores: dict[tuple[str, str, int], float] | None = None,
    error_rows: set[tuple[str, str, int]] | None = None,
    truncated_rows: set[tuple[str, str, int]] | None = None,
    drop_rows: set[tuple[str, str, int]] | None = None,
    duplicate_rows: set[tuple[str, str, int]] | None = None,
    items_override: dict[str, list[str]] | None = None,
    info_overrides: dict[str, Any] | None = None,
    task_info_overrides: dict[str, dict[str, Any]] | None = None,
    quant_recipe: str | None = None,
    skip_files: tuple[str, ...] = (),
    legacy: bool = False,
    legacy_tasks: tuple[str, ...] = (),
    attempts: dict[tuple[str, str, int], int] | None = None,
    working_times: dict[tuple[str, str, int], float | None] | None = None,
    client_timeout: int | None = None,
) -> Path:
    """Write one fake run directory.

    Item difficulty is shared by every run built with the same ``expected`` (so two runs are
    paired the way two arms on the same items are); sampling noise depends on ``run_seed``.

    Args:
        path: New directory.
        expected: ``{task: {"items": n, "epochs": e}}`` (default :data:`SMALL_EXPECTED`).
        label: Arm label, written to ``model`` and ``endpoint`` (expected to differ).
        p: Mean accuracy, one number or per task.
        shift: Added to every item's success probability (an arm effect).
        run_seed: Seed of the sampling noise.
        status: ``run_info.json`` status.
        scores: Explicit ``{(task, id, epoch): score}`` overriding the random draw.
        error_rows: Rows that are request errors (no score).
        truncated_rows: Rows with ``stop_reason`` ``max_tokens``.
        drop_rows: Rows left out of ``samples.jsonl`` (missing requests).
        duplicate_rows: Rows written twice.
        items_override: Replacement item ids per task (in ``items.json`` and the rows).
        info_overrides: Dotted-path overrides of ``run_info.json`` (``"sampling.top_k": 50``).
        task_info_overrides: Per-task overrides of ``run_info.json`` ``tasks.<task>``.
        quant_recipe: If given, write a ``quantization_info.json`` with this recipe name.
        skip_files: Names of files not to write.
        legacy: Write the format of before 2026-11-21: no ``model_attempts`` or ``working_time`` in
            the rows and no ``client_timeout``, ``samples_with_retries`` or ``max_working_time`` in
            ``run_info.json`` (the client timeout was then inspect_ai's 600 s default).
        legacy_tasks: The same, for these tasks only.
        attempts: ``{(task, id, epoch): model_attempts}``; the default is 1 (no retry).
        working_times: ``{(task, id, epoch): working_time}``; the default is 1.0 (may be None).
        client_timeout: ``run_info.json`` ``client_timeout``; the default is the run's
            ``sampling.timeout`` (the harness passes the same value to both).

    Returns:
        ``path``.
    """
    expected = expected or SMALL_EXPECTED
    scores = scores or {}
    error_rows = error_rows or set()
    truncated_rows = truncated_rows or set()
    drop_rows = drop_rows or set()
    duplicate_rows = duplicate_rows or set()
    attempts = attempts or {}
    working_times = working_times or {}
    timeout = (info_overrides or {}).get("sampling.timeout", 3600)
    client_timeout = timeout if client_timeout is None else client_timeout
    rng = np.random.default_rng(run_seed)
    rows: list[dict[str, Any]] = []
    items: dict[str, list[str]] = {}
    tasks_info: dict[str, dict[str, Any]] = {}
    for task, spec in expected.items():
        ids = (items_override or {}).get(task) or task_item_ids(task, spec["items"])
        items[task] = ids
        base = np.random.default_rng(zlib_seed(task)).uniform(-0.25, 0.25, size=len(ids))
        mean = p[task] if isinstance(p, dict) else p
        task_rows: list[dict[str, Any]] = []
        for k, sid in enumerate(ids):
            for epoch in range(1, spec["epochs"] + 1):
                key = (task, sid, epoch)
                q = float(np.clip(mean + shift + base[k], 0.02, 0.98))
                draw = float(rng.random() < q)
                is_error = key in error_rows
                row = {
                    "task": task,
                    "sample_id": sid,
                    "epoch": epoch,
                    "stratum": sid.split("-")[1] if task == "mgsm" else None,
                    "score": None if is_error else scores.get(key, draw),
                    "scores": {},
                    "error": "ConnectError: boom" if is_error else None,
                    "truncated": key in truncated_rows,
                    "stop_reason": "max_tokens" if key in truncated_rows else "stop",
                    "input_tokens": 10,
                    "output_tokens": 100,
                    "total_time": 1.0,
                    "target": "",
                    "completion": f"{label}-{run_seed}",
                }
                if not (legacy or task in legacy_tasks):
                    row["model_attempts"] = attempts.get(key, 1)
                    row["working_time"] = working_times.get(key, 1.0)
                if key in drop_rows:
                    continue
                task_rows.append(row)
                if key in duplicate_rows:
                    task_rows.append(dict(row))
        rows += task_rows
        tasks_info[task] = {
            "epochs": spec["epochs"],
            "n_items": len(ids),
            "items_sha256": sha256_of_ids(ids),
            "n_samples": len(task_rows),
            "request_errors": sum(r["error"] is not None for r in task_rows),
            "truncations": sum(bool(r["truncated"]) for r in task_rows),
            "eval_status": "success",
            "inspect_log": f"{task}.eval",
        }
        if not (legacy or task in legacy_tasks):
            tasks_info[task] |= {"client_timeout": client_timeout, **retry_info(task_rows)}
        tasks_info[task].update((task_info_overrides or {}).get(task, {}))
    info: dict[str, Any] = {
        "status": status,
        "endpoint": f"http://{label}/v1",
        "model": f"olala-{label}",
        "inspect_model": f"openai-api/vllm/olala-{label}",
        "git": {"commit": "abc123", "dirty": False, "note": None},
        "code_sha256": "c" * 64,
        "versions": {"python": "3.12.1", "inspect_ai": "0.3.291"},
        "sampling": {
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 40,
            "max_tokens": None,
            "seed": None,
            "max_retries": 5,
            "timeout": 3600,
        },
        "reasoning_effort": "medium",
        "chat_template_kwargs": {"reasoning_effort": "medium"},
        "max_connections": 96,
        "seed": 0,
        "sandbox": {"mode": "local", "tasks": ["humaneval_plus", "mbpp_plus"]},
        "epochs_requested": {"default": None, "per_task": {}},
        "started_utc": "2026-11-21T10:00:00Z",
        "ended_utc": "2026-11-21T12:00:00Z",
        "skipped_tasks": {},
        "tasks": tasks_info,
        "n_samples": len(rows),
        "request_errors": sum(r["error"] is not None for r in rows),
        "truncations": sum(bool(r["truncated"]) for r in rows),
    }
    if not legacy:
        info["client_timeout"] = client_timeout
        info |= retry_info(rows)
    for dotted, value in (info_overrides or {}).items():
        node = info
        *parents, last = dotted.split(".")
        for part in parents:
            node = node[part]
        node[last] = copy.deepcopy(value)
    path.mkdir(parents=True)
    files = {
        "samples.jsonl": "".join(json.dumps(r) + "\n" for r in rows),
        "items.json": json.dumps(items, indent=2),
        "run_info.json": json.dumps(info, indent=2),
    }
    if quant_recipe is not None:
        files["quantization_info.json"] = json.dumps(
            {"schema_version": 1, "recipe": {"name": quant_recipe, "weights": "fp8"}}
        )
    for name, text in files.items():
        if name not in skip_files:
            (path / name).write_text(text, encoding="utf-8")
    return path


def retry_info(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """``samples_with_retries`` and ``max_working_time`` of the rows that carry the new fields."""
    carrying = [r for r in rows if "model_attempts" in r]
    times = [r["working_time"] for r in rows if r.get("working_time") is not None]
    return {
        "samples_with_retries": sum(r["model_attempts"] > 1 for r in carrying),
        "max_working_time": max(times) if times else None,
    }


def zlib_seed(text: str) -> int:
    """Stable small integer derived from a string (the item difficulty seed of a task)."""
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def det(expected, fn):
    """Deterministic scores ``{(task, id, epoch): fn(task, k, epoch)}``."""
    return {
        (t, sid, e): float(fn(t, k, e))
        for t, spec in expected.items()
        for k, sid in enumerate(task_item_ids(t, spec["items"]))
        for e in range(1, spec["epochs"] + 1)
    }


def ref_fn(t, k, e):
    return k % 4 != 0  # 75% of the items are solved


def same_fn(t, k, e):
    return ref_fn(t, k, e)


def worse_fn(t, k, e):
    return k % 4 == 1  # 25%: about 50 points below the reference on every task


def edge_fn(t, k, e):
    # the reference, except two MGSM items: about -2 points on the macro-average, wide interval
    return ref_fn(t, k, e) and not (t == "mgsm" and k in (1, 2))


def make(tmp_path, name, fn=ref_fn, expected=SMALL_EXPECTED, **kw):
    return build_run(tmp_path / name, expected=expected, label=name, scores=det(expected, fn), **kw)
