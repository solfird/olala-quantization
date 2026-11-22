"""Thin CLI wrapper around inspect_ai: run benchmark tasks against an OpenAI-compatible endpoint.

Writes, into ``--out``:

* ``samples.jsonl``: one row per (task, item, epoch);
* ``items.json``: the exact item ids that were run, per task (reuse with ``--items-from``);
* ``run_info.json``: every setting of the run, versions, request errors and truncations;
* ``logs/``: the raw inspect_ai ``.eval`` logs.

Use ``compare.py A_dir B_dir`` to compare two such directories.
"""

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from inspect_ai import Task
from inspect_ai.event import ModelEvent
from inspect_ai.log import EvalSample
from inspect_ai.model import GenerateConfig, Model
from inspect_ai.scorer import value_to_float

from attempts import retry_stats

INSPECT_PROVIDER = "openai-api/vllm/"
"""inspect_ai's generic OpenAI-compatible provider, named ``vllm``.

The dedicated ``vllm/`` provider keeps a process-wide registry keyed by model name, so a second
endpoint for the same model name in one process would silently reuse the first URL.
"""

TRUNCATION_REASONS = frozenset({"max_tokens", "model_length"})
"""inspect_ai stop reasons that mean the answer was cut short."""

DEFAULT_MAX_CONNECTIONS = 32
"""Requests in flight per task when neither a bare value nor a built-in per-task value applies."""

TASK_MAX_CONNECTIONS: dict[str, int] = {"aime2025": 24}
"""Built-in per-task concurrency (beats a bare ``--max-connections N``, loses to ``TASK=N``).

AIME 2025 requests can run to the 32k-token context, and the vLLM server's KV cache holds only
about 23 such sequences. With more requests in flight the surplus queues on the server and the
client timeout counts that queueing: on 2026-11-20 96 in flight produced 34 client timeouts in the
arms that reason longer and none in the others (PROTOCOL.md amendment 10). 24 is about what the
cache holds (a queue of one at most), so a request's clock measures its own generation.
"""

DEFAULT_TIMEOUT = 14400
"""Client timeout in seconds per request, retries included (4 hours).

It must exceed the worst-case generation time of one request so that it never turns queueing
or long reasoning into a request error; it is a safety net against a hung connection, not a limit
of the benchmark. It was 3,600 s until 2026-11-20 (see PROTOCOL.md amendment 10).

It is applied twice, with the same value: as ``GenerateConfig.timeout`` and as the provider's own
HTTP client timeout, the model argument ``client_timeout`` (:func:`build_model`). The second one
matters: inspect_ai 0.3.291's OpenAI-compatible provider cancels any request that lasts longer
than ``client_timeout`` (600 s unless given) and sends it again, whatever ``GenerateConfig.timeout``
says, so a long generation was thrown away and redrawn without a request error.
"""


# --------------------------------------------------------------------------- item selection


def sha256_of_ids(ids: Sequence[Any]) -> str:
    """Order-independent hash of a set of item ids.

    Args:
        ids: Item identifiers.

    Returns:
        Hex SHA-256 of the sorted, newline-joined string forms.
    """
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def _quotas(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Split ``n`` draws as evenly as possible over strata of the given sizes.

    Strata smaller than their share are taken whole and the shortfall is redistributed; any
    remainder goes to the first strata in sorted order.
    """
    quotas: dict[str, int] = {}
    active = sorted(sizes)
    remaining = n
    while active:
        share, extra = divmod(remaining, len(active))
        small = [s for s in active if sizes[s] <= share]
        if not small:
            for i, s in enumerate(active):
                quotas[s] = share + (1 if i < extra else 0)
            break
        for s in small:
            quotas[s] = sizes[s]
            remaining -= sizes[s]
        active = [s for s in active if s not in small]
    return quotas


def select_items(
    ids: Sequence[str],
    strata: Sequence[str] | None,
    n: int | None,
    seed: int,
) -> list[str]:
    """Pick a reproducible subset of item ids, optionally stratified.

    The result depends only on the *set* of ids, the strata, ``n`` and ``seed``, never on the
    order in which the dataset happens to list them, so two runs select identical items.

    Args:
        ids: Unique item identifiers (as strings).
        strata: Stratum label of each id (e.g. the MGSM language) or ``None``.
        n: Number of items wanted; ``None`` or at least ``len(ids)`` selects everything.
        seed: Seed of the per-stratum random draws.

    Returns:
        The chosen ids, in the order of ``ids``.

    Raises:
        ValueError: If ``n`` is not positive or ids are not unique.
    """
    if len(set(ids)) != len(ids):
        raise ValueError("item ids must be unique")
    if n is None or n >= len(ids):
        return list(ids)
    if n <= 0:
        raise ValueError("the number of items must be positive")
    labels = list(strata) if strata is not None else [""] * len(ids)
    groups: dict[str, list[str]] = defaultdict(list)
    for item, label in zip(ids, labels, strict=True):
        groups[label].append(item)
    quotas = _quotas({k: len(v) for k, v in groups.items()}, n)
    chosen: set[str] = set()
    for label, members in groups.items():
        rng = random.Random(f"{seed}|{label}")  # noqa: S311  (reproducible sampling, not crypto)
        chosen.update(rng.sample(sorted(members), quotas[label]))
    return [i for i in ids if i in chosen]


def parse_n_items(values: Sequence[str] | None) -> tuple[int | None, dict[str, int]]:
    """Parse repeated ``--n-items`` values: ``N`` (default) and ``task=N`` (override).

    Args:
        values: Raw option strings, e.g. ``["200", "mgsm=550"]``.

    Returns:
        ``(default_n, {task: n})``.

    Raises:
        ValueError: If a value is not ``N`` or ``task=N`` with a positive integer ``N``.
    """
    default: int | None = None
    per_task: dict[str, int] = {}
    for raw in values or []:
        name, sep, num = raw.rpartition("=")
        try:
            count = int(num)
        except ValueError:
            raise ValueError(f"invalid --n-items value {raw!r} (use N or task=N)") from None
        if count <= 0 or (sep and not name):
            raise ValueError(f"invalid --n-items value {raw!r} (use N or task=N)")
        if sep:
            per_task[name] = count
        else:
            default = count
    return default, per_task


def parse_epochs(
    values: Sequence[str] | None, known_tasks: Sequence[str]
) -> tuple[int | None, dict[str, int]]:
    """Parse repeated ``--epochs`` values: ``N`` (every task) and ``task=N`` (one task).

    A task value beats the bare value, whatever their order on the command line; a task without
    either keeps the epochs of its registry entry (the frozen protocol's).

    Args:
        values: Raw option strings, e.g. ``["1", "aime2025=5"]``.
        known_tasks: Names accepted on the left of ``=``.

    Returns:
        ``(default_epochs, {task: epochs})``.

    Raises:
        ValueError: If a value is not ``N`` or ``task=N`` with a positive integer ``N``, or names
            a task that does not exist.
    """
    default: int | None = None
    per_task: dict[str, int] = {}
    for raw in values or []:
        name, sep, num = raw.rpartition("=")
        message = f"invalid --epochs value {raw!r} (use N or task=N with a positive integer N)"
        try:
            count = int(num)
        except ValueError:
            raise ValueError(message) from None
        if count <= 0 or (sep and not name):
            raise ValueError(message)
        if not sep:
            default = count
        elif name not in known_tasks:
            raise ValueError(f"unknown task {name!r} in --epochs (known: {', '.join(known_tasks)})")
        else:
            per_task[name] = count
    return default, per_task


def parse_max_connections(
    values: Sequence[str] | None, known_tasks: Sequence[str]
) -> tuple[int | None, dict[str, int]]:
    """Parse repeated ``--max-connections`` values: ``N`` (bare) and ``task=N`` (one task).

    Args:
        values: Raw option strings, e.g. ``["96", "aime2025=12"]``.
        known_tasks: Names accepted on the left of ``=``.

    Returns:
        ``(bare_n, {task: n})``.

    Raises:
        ValueError: If a value is not ``N`` or ``task=N`` with a positive integer ``N``, or names
            a task that does not exist.
    """
    bare: int | None = None
    per_task: dict[str, int] = {}
    for raw in values or []:
        name, sep, num = raw.rpartition("=")
        message = (
            f"invalid --max-connections value {raw!r} (use N or task=N with a positive integer N)"
        )
        try:
            count = int(num)
        except ValueError:
            raise ValueError(message) from None
        if count <= 0 or (sep and not name):
            raise ValueError(message)
        if not sep:
            bare = count
        elif name not in known_tasks:
            raise ValueError(
                f"unknown task {name!r} in --max-connections (known: {', '.join(known_tasks)})"
            )
        else:
            per_task[name] = count
    return bare, per_task


def resolve_max_connections(
    tasks: Sequence[str], bare: int | None, per_task: Mapping[str, int]
) -> dict[str, int]:
    """Concurrency of each task: explicit ``TASK=N``, else built-in, else bare ``N``, else 32.

    Args:
        tasks: Names of the tasks that run.
        bare: The bare ``--max-connections N``, or ``None``.
        per_task: The explicit ``TASK=N`` values.

    Returns:
        ``{task: requests in flight}`` for the given tasks.
    """
    fallback = DEFAULT_MAX_CONNECTIONS if bare is None else bare
    return {t: per_task.get(t, TASK_MAX_CONNECTIONS.get(t, fallback)) for t in tasks}


# --------------------------------------------------------------------------- rows


def model_attempts(sample: EvalSample) -> int:
    """HTTP requests the sample needed: 1 if the first was answered, more if any was redrawn.

    inspect_ai logs one ``ModelEvent`` per call of its own retry loop and records the retries of
    the OpenAI client (a request cancelled by the client timeout, a 5xx, a dropped connection) as
    ``ModelEvent.retries`` of that same event, so the number of events undercounts: a sample whose
    first request was cancelled at the client timeout and whose second one was answered has ONE
    event with ``retries`` 1. The requests are ``sum(1 + retries)`` over the model events.

    Args:
        sample: The sample from an inspect_ai eval log.

    Returns:
        The number of requests (0 if the log has no model event for the sample).
    """
    return sum(1 + (e.retries or 0) for e in sample.events if isinstance(e, ModelEvent))


def row_from_sample(
    sample: EvalSample,
    task: str,
    primary_metric: str | None,
    stratum_field: str | None,
) -> dict[str, Any]:
    """Flatten one inspect_ai sample into a JSON-serialisable result row.

    Args:
        sample: The sample from an inspect_ai eval log.
        task: Task name stored in the row.
        primary_metric: Key to read when the scorer returns a dict (e.g. IFEval).
        stratum_field: Metadata key giving the sample's stratum (e.g. ``language``).

    Returns:
        A dict with the numeric ``score`` (``None`` for a request error), the error text,
        the truncation flag, token usage, the timings (``total_time``, ``working_time``), the
        number of requests the sample needed (``model_attempts``) and the model completion.

    Raises:
        ValueError: If the score is a dict and ``primary_metric`` is not given.
    """
    score: float | None = None
    scores: dict[str, Any] = {}
    if sample.scores:
        scores = {name: s.value for name, s in sample.scores.items()}
        first = next(iter(sample.scores.values()))
        if isinstance(first.value, dict):
            if primary_metric is None:
                raise ValueError(f"task {task!r} returns dict scores: set its primary_metric")
            score = float(first.value[primary_metric])
        else:
            score = float(value_to_float()(first.value))
    stop_reason = sample.output.stop_reason if sample.output and sample.output.choices else None
    usage = sample.output.usage if sample.output else None
    target = sample.target if isinstance(sample.target, str) else "|".join(sample.target)
    stratum = (sample.metadata or {}).get(stratum_field) if stratum_field else None
    return {
        "task": task,
        "sample_id": str(sample.id),
        "epoch": sample.epoch,
        "stratum": stratum,
        "score": score,
        "scores": scores,
        "error": sample.error.message if sample.error else None,
        "truncated": stop_reason in TRUNCATION_REASONS,
        "stop_reason": stop_reason,
        "input_tokens": usage.input_tokens if usage else None,
        "output_tokens": usage.output_tokens if usage else None,
        "total_time": sample.total_time,
        "working_time": sample.working_time,
        "model_attempts": model_attempts(sample),
        "target": target,
        "completion": sample.output.completion if sample.output and sample.output.choices else "",
    }


# --------------------------------------------------------------------------- provenance


def _git(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        ["git", "-C", str(path), *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )


def git_info(path: str | Path) -> dict[str, Any]:
    """Describe the git state of ``path`` (commit, dirtiness of that path).

    Args:
        path: A directory inside the repository to describe.

    Returns:
        ``{"commit": sha | None, "dirty": bool | None, "note": str | None}``. A repository
        without commits, or no repository at all, yields ``commit=None`` and a note.
    """
    path = Path(path)
    if _git(path, "rev-parse", "--show-toplevel").returncode != 0:
        return {"commit": None, "dirty": None, "note": "not a git repository"}
    head = _git(path, "rev-parse", "HEAD")
    dirty = bool(_git(path, "status", "--porcelain", "--", ".").stdout.strip())
    if head.returncode != 0:
        return {"commit": None, "dirty": dirty, "note": "repository has no commit yet"}
    return {"commit": head.stdout.strip(), "dirty": dirty, "note": None}


# --------------------------------------------------------------------------- task registry


@dataclass(frozen=True)
class TaskSpec:
    """How to build and score one benchmark.

    Attributes:
        name: Name used on the command line and in the output.
        factory: Builds the inspect_ai ``Task`` (lazily: it may download a dataset). Called with
            the keyword ``sandbox`` (``docker`` or ``local``) when ``takes_sandbox`` is true.
        epochs: Samples per item, fixed by PROTOCOL.md section 6 (5 for AIME 2025, 1 otherwise).
        primary_metric: Key to read when the scorer returns a dict of values.
        stratum_field: Sample-metadata key used to stratify item subsets.
        takes_sandbox: Whether the task executes model code and so needs a sandbox mode.
    """

    name: str
    factory: Callable[..., Task]
    epochs: int = 1
    primary_metric: str | None = None
    stratum_field: str | None = None
    takes_sandbox: bool = False


def _mgsm() -> Task:
    from inspect_evals.mgsm import mgsm

    return mgsm(languages="all")


def _gsm8k() -> Task:
    from inspect_evals.gsm8k import gsm8k

    return gsm8k(fewshot=0)  # the protocol is zero-shot; inspect_evals defaults to 10-shot


def _ifeval() -> Task:
    from inspect_evals.ifeval import ifeval

    return ifeval()


def _aime2025() -> Task:
    from inspect_evals.aime2025 import aime2025

    return aime2025()


def _humaneval_plus(sandbox: str = "docker") -> Task:
    from plus_tasks import humaneval_plus

    return humaneval_plus(sandbox_mode=sandbox)


def _mbpp_plus(sandbox: str = "docker") -> Task:
    from plus_tasks import mbpp_plus

    return mbpp_plus(sandbox_mode=sandbox)


TASKS: dict[str, TaskSpec] = {
    spec.name: spec
    for spec in (
        TaskSpec("mgsm", _mgsm, epochs=1, stratum_field="language"),
        TaskSpec("humaneval_plus", _humaneval_plus, epochs=1, takes_sandbox=True),
        TaskSpec("mbpp_plus", _mbpp_plus, epochs=1, takes_sandbox=True),
        TaskSpec("gsm8k", _gsm8k, epochs=1),
        TaskSpec("ifeval", _ifeval, epochs=1, primary_metric="prompt_level_strict"),
        TaskSpec("aime2025", _aime2025, epochs=5),
    )
}
"""Tasks of the protocol, in the order they are run by default."""

UNAVAILABLE: dict[str, str] = {
    "ifbench": (
        "IFBench is not in inspect_evals (checked version 0.29.0: no ifbench task); "
        "skipped, not reimplemented (it needs the 58 out-of-distribution constraint verifiers "
        "of allenai/IFBench)."
    ),
}
"""Tasks of the protocol that cannot be run, with the reason recorded in run_info.json."""

DEFAULT_TASKS = list(TASKS)


# --------------------------------------------------------------------------- run


def build_model(base_url: str, model: str, timeout: float, api_key: str | None = None) -> Model:
    """The inspect_ai model of a run, with its HTTP client timeout set.

    ``client_timeout`` is the name of the model argument of inspect_ai's OpenAI-compatible provider
    (``openai_compatible.py``, ``OpenAICompatibleAPI.__init__``). Without it the provider's client
    cancels a request after 600 s and silently sends it again.

    Args:
        base_url: The endpoint's ``/v1`` URL.
        model: The model name served by the endpoint.
        timeout: Seconds a request may last before the client gives up (``--timeout``).
        api_key: The API key (default: ``$VLLM_API_KEY``, else ``EMPTY``).

    Returns:
        The model, not memoized.
    """
    from inspect_ai.model import get_model

    # memoize=False: inspect_ai otherwise reuses a cached client (and base URL) for the same model
    # name across eval() calls in one process.
    return get_model(
        INSPECT_PROVIDER + model,
        base_url=base_url,
        api_key=api_key or os.environ.get("VLLM_API_KEY", "EMPTY"),
        client_timeout=timeout,
        memoize=False,
    )


def generate_config(args: argparse.Namespace, max_connections: int) -> GenerateConfig:
    """The sampling and client configuration of one task's eval call.

    inspect_ai's OpenAI-compatible providers do not forward ``GenerateConfig.top_k``, so top_k is
    sent (like the chat-template argument) as a vLLM extra body field.

    Args:
        args: The parsed command line.
        max_connections: Requests in flight for the task.

    Returns:
        The configuration.
    """
    return GenerateConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.sampling_seed,
        max_retries=args.max_retries,
        timeout=args.timeout,
        max_connections=max_connections,
        extra_body={
            "top_k": args.top_k,
            "chat_template_kwargs": {"reasoning_effort": args.reasoning_effort},
        },
    )


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def package_versions() -> dict[str, str]:
    """Versions of the packages that influence the results."""
    out = {"python": platform.python_version()}
    for dist in (
        "inspect_ai",
        "inspect_evals",
        "openai",
        "httpx",
        "datasets",
        "numpy",
        "instruction_following_eval",
    ):
        try:
            out[dist] = importlib_metadata.version(dist.replace("_", "-"))
        except importlib_metadata.PackageNotFoundError:
            out[dist] = "not installed"
    return out


def code_sha256() -> str:
    """Hash of the harness sources (the repository may have no commit to point at)."""
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def _dataset_items(
    task: Task, stratum_field: str | None
) -> tuple[dict[str, Any], list[str] | None]:
    """Map ``str(id) -> original id`` and the stratum labels of the task's dataset."""
    originals: dict[str, Any] = {}
    strata: list[str] | None = [] if stratum_field else None
    for sample in task.dataset:
        if sample.id is None:
            raise SystemExit("task has samples without an id; cannot select items reproducibly")
        originals[str(sample.id)] = sample.id
        if strata is not None:
            strata.append(str((sample.metadata or {}).get(stratum_field)))
    return originals, strata


def _choose_ids(
    name: str,
    originals: dict[str, Any],
    strata: list[str] | None,
    n: int | None,
    seed: int,
    items_from: dict[str, list[str]] | None,
) -> list[str]:
    if items_from is not None:
        if name not in items_from:
            raise SystemExit(f"--items-from file has no entry for task {name!r}")
        unknown = [i for i in items_from[name] if i not in originals]
        if unknown:
            raise SystemExit(f"--items-from lists ids unknown to task {name!r}: {unknown[:5]}")
        return list(items_from[name])
    return select_items(list(originals), strata, n, seed)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="run_tasks.py", description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS, help="default: the protocol list")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="olala", help="model name served by the endpoint")
    ap.add_argument("--out", required=True, help="new output directory")
    ap.add_argument("--seed", type=int, default=0, help="seed of the item selection")
    ap.add_argument(
        "--n-items",
        action="append",
        metavar="[TASK=]N",
        help="subset size per task (stratified for MGSM); repeatable; default: all items",
    )
    ap.add_argument("--items-from", help="items.json of another run: score exactly those items")
    ap.add_argument(
        "--epochs",
        action="append",
        metavar="[TASK=]N",
        help="samples per item; N applies to every task, TASK=N to one task; repeatable; "
        "default: the protocol's (aime2025 5, every other task 1)",
    )
    ap.add_argument(
        "--sandbox",
        choices=["docker", "local"],
        default="docker",
        help="where HumanEval+ and MBPP+ run model code: the Docker sandbox (isolated, default) "
        "or inspect_ai's local sandbox (a subprocess of this process: no isolation, for a "
        "disposable machine without Docker such as a RunPod pod)",
    )
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=None, help="default: server context limit")
    ap.add_argument(
        "--reasoning-effort",
        choices=["disabled", "low", "medium", "high"],
        default="high",
        help="sent as chat_template_kwargs.reasoning_effort (the Olala chat template)",
    )
    ap.add_argument("--sampling-seed", type=int, default=None, help="per-request seed (1 epoch)")
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="seconds per request incl. retries (default 14400). It must exceed the worst-case "
        "generation time so that it never counts server-side queueing or long reasoning as a "
        "request error; it only guards against a hung connection",
    )
    ap.add_argument(
        "--max-connections",
        action="append",
        metavar="[TASK=]N",
        help="requests in flight (also inspect_ai's max_samples); repeatable. Per task the "
        "precedence is: an explicit TASK=N, then the built-in per-task default (aime2025=24: the "
        "server's KV cache holds about 23 full-length sequences, more would queue on the server "
        "and the client timeout would count the queueing), then a bare N, then 32",
    )
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected tasks and write the result directory.

    Args:
        argv: Command-line arguments (default: ``sys.argv[1:]``).

    Returns:
        ``0`` if every task finished, ``1`` if any inspect_ai task ended in error.

    Raises:
        SystemExit: For invalid options (unknown task, existing output, bad item file, ...).
    """
    import inspect_ai
    from inspect_ai.log import read_eval_log

    argv = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(argv)
    try:
        default_n, per_task_n = parse_n_items(args.n_items)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    unknown = [t for t in args.tasks if t not in TASKS and t not in UNAVAILABLE]
    if unknown:
        raise SystemExit(
            f"unknown task {unknown[0]!r} (known: {', '.join([*TASKS, *UNAVAILABLE])})"
        )
    try:
        default_epochs, per_task_epochs = parse_epochs(args.epochs, list(TASKS))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        bare_connections, per_task_connections = parse_max_connections(
            args.max_connections, list(TASKS)
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    skipped = {t: UNAVAILABLE[t] for t in args.tasks if t in UNAVAILABLE}
    specs = [TASKS[t] for t in args.tasks if t in TASKS]
    connections = resolve_max_connections(
        [s.name for s in specs], bare_connections, per_task_connections
    )
    epochs = {s.name: per_task_epochs.get(s.name, default_epochs or s.epochs) for s in specs}
    if args.sampling_seed is not None and any(e > 1 for e in epochs.values()):
        raise SystemExit(
            "--sampling-seed needs --epochs 1: with a fixed per-request seed every epoch of an "
            "item would return the same sample"
        )

    out = Path(args.out)
    if (out / "run_info.json").exists():
        raise SystemExit(f"{out} already exists (run_info.json present); choose a new --out")
    out.mkdir(parents=True, exist_ok=True)
    items_from = json.loads(Path(args.items_from).read_text()) if args.items_from else None

    sampling = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "seed": args.sampling_seed,
        "max_retries": args.max_retries,
        "timeout": args.timeout,
    }
    info: dict[str, Any] = {
        "status": "running",
        "endpoint": args.base_url,
        "model": args.model,
        "inspect_model": INSPECT_PROVIDER + args.model,
        "git": git_info(Path(__file__).parent),
        "code_sha256": code_sha256(),
        "versions": package_versions(),
        "sampling": sampling,
        "reasoning_effort": args.reasoning_effort,
        "chat_template_kwargs": {"reasoning_effort": args.reasoning_effort},
        "max_connections": DEFAULT_MAX_CONNECTIONS
        if bare_connections is None
        else bare_connections,
        "max_connections_per_task": connections,
        "seed": args.seed,
        "n_items_requested": {"default": default_n, "per_task": per_task_n},
        "sandbox": {"mode": args.sandbox, "tasks": [s.name for s in specs if s.takes_sandbox]},
        "epochs_requested": {"default": default_epochs, "per_task": per_task_epochs},
        "items_from": args.items_from,
        "argv": argv,
        "started_utc": _now(),
        "ended_utc": None,
        "skipped_tasks": skipped,
        "tasks": {},
        "n_samples": 0,
        "request_errors": 0,
        "truncations": 0,
        "client_timeout": args.timeout,
        "samples_with_retries": 0,
        "max_working_time": None,
    }
    _write_json(out / "run_info.json", info)

    def run_all() -> int:
        """Run every task; return 0 if each inspect_ai eval succeeded."""
        items_out: dict[str, list[str]] = {}
        all_rows: list[dict[str, Any]] = []
        exit_code = 0
        with (out / "samples.jsonl").open("w", encoding="utf-8") as rows_file:
            for spec in specs:
                task = spec.factory(sandbox=args.sandbox) if spec.takes_sandbox else spec.factory()
                originals, strata = _dataset_items(task, spec.stratum_field)
                n = per_task_n.get(spec.name, default_n)
                chosen = _choose_ids(spec.name, originals, strata, n, args.seed, items_from)
                items_out[spec.name] = chosen
                _write_json(out / "items.json", items_out)

                model = build_model(args.base_url, args.model, args.timeout)
                logs = inspect_ai.eval(
                    task,
                    model=model,
                    sample_id=[originals[i] for i in chosen],
                    epochs=epochs[spec.name],
                    log_dir=str(out / "logs"),
                    display="none",
                    fail_on_error=False,
                    log_samples=True,
                    max_samples=connections[spec.name],
                    **generate_config(args, connections[spec.name]).model_dump(exclude_none=True),
                )
                log = logs[0]
                if log.samples is None:
                    log = read_eval_log(log.location)
                rows = [
                    row_from_sample(s, spec.name, spec.primary_metric, spec.stratum_field)
                    for s in (log.samples or [])
                ]
                for row in rows:
                    rows_file.write(json.dumps(row) + "\n")
                rows_file.flush()
                errors = sum(r["error"] is not None for r in rows)
                truncs = sum(bool(r["truncated"]) for r in rows)
                info["tasks"][spec.name] = {
                    "epochs": epochs[spec.name],
                    "n_items": len(chosen),
                    "items_sha256": sha256_of_ids(chosen),
                    "n_samples": len(rows),
                    "request_errors": errors,
                    "truncations": truncs,
                    "client_timeout": args.timeout,
                    **retry_stats(rows),
                    "eval_status": log.status,
                    "inspect_log": str(Path(log.location).name),
                }
                all_rows += rows
                info.update(retry_stats(all_rows))
                info["n_samples"] += len(rows)
                info["request_errors"] += errors
                info["truncations"] += truncs
                if log.status != "success":
                    exit_code = 1
                _write_json(out / "run_info.json", info)
        return exit_code

    try:
        exit_code = run_all()
    except Exception as exc:
        info["status"] = "failed"
        info["error"] = f"{type(exc).__name__}: {exc}"
        info["ended_utc"] = _now()
        _write_json(out / "run_info.json", info)
        raise

    info["status"] = "complete" if exit_code == 0 else "incomplete"
    info["ended_utc"] = _now()
    _write_json(out / "run_info.json", info)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
