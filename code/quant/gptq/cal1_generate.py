"""Build CAL-1: SELF-GENERATED bf16 reasoning traces on PUBLIC prompts, disjoint from the five test tasks.

Why: a reasoning model's activations on its own chat-formatted reasoning traces differ from those on plain public text
(``design/GPTQ_DESIGN.md`` sections 5.3 and 5.4), and the toy study showed that the calibration domain decides whether
GPTQ helps or hurts out of distribution. CAL-1 is therefore made of what the bf16 model writes in answer to public
prompts, rendered with the model's chat template (``Reasoning effort:`` line, analysis channel, final channel).

One command does everything and can be re-run at will (every stage is resumable or idempotent)::

    python cal1_generate.py --out DIR --eval-items data/eval_items.json --model NAME \
        --base-url http://127.0.0.1:8000/v1 --server-provenance "vllm X, olala-bf16 sha256 ..." \
        --tokenizer /workspace/models/olala-bf16 --revision math_train=SHA --revision mgsm_train_parallel=SHA \
        --revision code_oss_python=SHA --revision if_personas=SHA

1. **Prompts** (``prompts.jsonl`` + ``prompts_manifest.json``): per domain a fixed quota (default 240 prompts, the
   weights of the five tasks in the macro-average: math = AIME 2025 1, multilingual math = MGSM 1, code = HumanEval+ and
   MBPP+ 2, instruction following = IFEval 1), drawn with a seed from the public sources of :data:`DEFAULT_SOURCES` at
   a PINNED dataset revision. Prompts that share an 8-gram with an evaluation item are skipped (and listed) at
   draw time.
2. **Generation** (``generations.jsonl``, append-only): the OpenAI-compatible endpoint is called with the harness
   settings (temperature 0.8, top-p 0.9, ``top_k`` 40 and ``chat_template_kwargs.reasoning_effort`` in the request
   body, max output 8192 tokens). A truncated trace is KEPT and flagged, a failure is recorded as a row, a re-run asks
   only for the prompts without a successful row.
3. **Audit**: the 8-gram leakage audit (``leakage_audit.py``) runs over the prompts AND over the generated text
   (reasoning plus answer). Any overlap refuses the build (exit 1, no manifest, no windows; ``--drop-flagged`` removes
   the flagged traces instead and lists them).
4. **Windows**: each trace is rendered with the chat template, tokenized, concatenated with the EOS id and cut into
   windows of ``--window`` tokens exactly as ``calib_data.pack_windows`` does; ``manifest.json`` follows the schema that
   ``calib_data.read_windows`` accepts and records the sha256 of every file, the prompt-source revisions, the sampling,
   the server provenance and the counts per domain.

``--dry-run`` prints the plan and counts without a server (and writes nothing); ``--verify`` re-checks a finished
directory (every hash, the windows, the audit). Exit status: 0 ok, 1 leakage refusal, 2 any other refusal.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import calib_data
import leakage_audit as la

SCHEMA = "cal1/1"
PROMPTS_SCHEMA = "cal1-prompts/1"
PROMPTS_FILE = "prompts.jsonl"
PROMPTS_MANIFEST = "prompts_manifest.json"
GENERATIONS_FILE = "generations.jsonl"
GEN_CONFIG = "generation_config.json"
MANIFEST = "manifest.json"
TEXT_FILE = la.BUNDLE_TEXT
WINDOWS_FILE = "windows.json"
EVAL_ITEMS_COPY = "eval_items.json"
AUDIT_REPORT = "leakage_audit.json"
AUDIT_N = 8
MAX_CONCURRENCY = 128
MIN_PROMPT_CHARS = 20
TRUNCATION_FINISH = frozenset({"length", "max_tokens", "model_length"})
HEX40 = re.compile(r"^[0-9a-f]{40}$")
DEFAULT_EVAL_ITEMS = Path(__file__).resolve().parent / "data" / "eval_items.json"
HASHED_FILES = (
    PROMPTS_FILE,
    PROMPTS_MANIFEST,
    GENERATIONS_FILE,
    GEN_CONFIG,
    TEXT_FILE,
    EVAL_ITEMS_COPY,
    AUDIT_REPORT,
    WINDOWS_FILE,
)


class Cal1Error(RuntimeError):
    """A CAL-1 input, resume state, response or output is missing, inconsistent or refused."""


# ---------------------------------------------------------------------------------------------------- sources


@dataclass(frozen=True)
class SourceDef:
    """A public prompt source: Hugging Face dataset id, licence, split, columns. The revision is given at run time."""

    name: str
    domain: str
    repo: str
    license: str
    split: str = "train"
    configs: tuple[str | None, ...] = (None,)
    column: str = "problem"
    lang_column: str | None = None
    filters: tuple[tuple[str, tuple[str, ...]], ...] = ()
    weight: float = 1.0
    revision: str = ""
    note: str = ""
    extract: str = ""  # name of a text transform in EXTRACTORS ("" keeps the column text unchanged)


_ALPACA_START = "### Instruction:"
_ALPACA_END = "### Response"


def _alpaca_instruction(text: str) -> str | None:
    """The question inside an Alpaca-style wrapper ('### Instruction:' ... '### Response'), else None."""
    start = text.find(_ALPACA_START)
    if start < 0:
        return None
    body = text[start + len(_ALPACA_START) :]
    end = body.find(_ALPACA_END)
    body = (body if end < 0 else body[:end]).strip()
    return body or None


EXTRACTORS: dict[str, Callable[[str], str | None]] = {"alpaca_instruction": _alpaca_instruction}


# Domain weights = weight of the tasks in the macro-average of the five tasks (PROTOCOL section 6: each task equal).
DEFAULT_WEIGHTS: dict[str, float] = {"math": 1, "multilingual_math": 1, "code": 2, "instruction": 1}

_MATH_SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)

DEFAULT_SOURCES: tuple[SourceDef, ...] = (
    SourceDef(
        "math_train",
        "math",
        "EleutherAI/hendrycks_math",
        "mit",
        configs=_MATH_SUBJECTS,
        column="problem",
        filters=(("level", ("Level 4", "Level 5")),),
        note="MATH train split, levels 4 and 5 (competition problems, published 2021: AIME 2025 cannot be in it)",
    ),
    SourceDef(
        "mgsm_train_parallel",
        "multilingual_math",
        "Mathoctopus/GSM8KInstruct_Parallel",
        "apache-2.0",
        column="prompt",
        extract="alpaca_instruction",
        note="GSM8K TRAIN questions translated into bn, de, es, fr, ja, ru, sw, th, zh (+ en); MGSM is GSM8K TEST. "
        "Columns are chosen, prompt, reject (read on the pod 2026-11-20; `query` does not exist); `prompt` is an "
        "Alpaca wrapper, the question between '### Instruction:' and '### Response' is the prompt.",
    ),
    SourceDef(
        "code_oss_python",
        "code",
        "ise-uiuc/Magicoder-OSS-Instruct-75K",
        "mit",
        column="problem",
        filters=(("lang", ("python",)),),
        note="Python problems written from open-source code snippets, not from HumanEval or MBPP",
    ),
    SourceDef(
        "if_personas",
        "instruction",
        "allenai/tulu-3-sft-personas-instruction-following",
        "odc-by",
        column="prompt",
        note="synthetic persona prompts with IFEval-style format constraints, not IFEval's own prompts",
    ),
)


@dataclass(frozen=True)
class PoolRow:
    """One candidate prompt of a source."""

    doc_id: str
    text: str
    lang: str = ""


@dataclass(frozen=True)
class Prompt:
    """One prompt of the manifest."""

    id: str
    domain: str
    source: str
    doc_id: str
    text: str
    lang: str = ""


PoolLoader = Callable[[SourceDef], list[PoolRow]]


def compute_quotas(total: int, weights: Mapping[str, float]) -> dict[str, int]:
    """Split ``total`` over the domains in proportion to ``weights`` (largest remainder, ties in mapping order).

    Raises:
        Cal1Error: For a non-positive total or weight.
    """
    if total <= 0 or not weights or any(w <= 0 for w in weights.values()):
        raise Cal1Error("the total and every domain weight must be positive")
    wsum = sum(weights.values())
    exact = {d: total * w / wsum for d, w in weights.items()}
    quotas = {d: math.floor(v + 1e-9) for d, v in exact.items()}
    order = sorted(weights, key=lambda d: (-(exact[d] - quotas[d]), list(weights).index(d)))
    for d in order[: total - sum(quotas.values())]:
        quotas[d] += 1
    return quotas


def load_index(eval_items: Path) -> la.NgramIndex:
    """The 8-gram index of the evaluation items; refuses an item file that lacks one of the five tasks."""
    items = la.load_eval_items(eval_items)
    missing = [t for t in la.REQUIRED_TASKS if t not in {i.task for i in items}]
    if missing:
        raise Cal1Error(f"{eval_items} lacks the evaluation tasks {missing}")
    return la.build_index(items, AUDIT_N)


def rows_from_records(src: SourceDef, records: Sequence[Mapping[str, Any]], config_label: str) -> list[PoolRow]:
    """Candidate rows of ``records`` after the source's column and filters (the doc id is the original row number).

    Raises:
        Cal1Error: If a record lacks the text column or a filter column (the message lists the columns found).
    """
    rows = []
    for i, rec in enumerate(records):
        for col in (src.column, *(f for f, _ in src.filters)):
            if col not in rec:
                raise Cal1Error(f"{src.name}: column {col!r} not found; the columns of {src.repo} are {sorted(rec)}")
        if any(rec[f] not in allowed for f, allowed in src.filters):
            continue
        text = rec[src.column]
        if isinstance(text, str) and src.extract:
            if src.extract not in EXTRACTORS:
                raise Cal1Error(f"{src.name}: unknown extract {src.extract!r}; known: {sorted(EXTRACTORS)}")
            text = EXTRACTORS[src.extract](text)
            if text is None:
                continue
        if isinstance(text, str):
            lang = str(rec.get(src.lang_column, "")) if src.lang_column else ""
            rows.append(PoolRow(f"{config_label}-{i}", text, lang))
    return rows


def _check_revision(src: SourceDef, revision: str | None) -> str:
    if not revision or not HEX40.match(revision):
        raise Cal1Error(
            f"source {src.name!r} ({src.repo}): needs a pinned 40-hex dataset revision "
            f"(--revision {src.name}=SHA), got {revision!r}"
        )
    return revision


def hf_pool_loader(revisions: Mapping[str, str]) -> PoolLoader:
    """Loader reading each source with ``datasets.load_dataset(repo, config, split=, revision=)`` (cache or network)."""

    def load(src: SourceDef) -> list[PoolRow]:
        rev = _check_revision(src, revisions.get(src.name))
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise Cal1Error("the `datasets` package is needed (uv sync --extra data), or use --pool-dir") from e
        rows: list[PoolRow] = []
        for cfg in src.configs:
            ds = (
                load_dataset(src.repo, cfg, split=src.split, revision=rev)
                if cfg
                else load_dataset(src.repo, split=src.split, revision=rev)
            )
            rows.extend(rows_from_records(src, list(ds), cfg or "default"))
        return rows

    return load


def dir_pool_loader(root: Path) -> PoolLoader:
    """Loader reading ``ROOT/<source name>.jsonl`` (rows ``{"id", "text", "lang"?}``, already filtered): offline."""

    def load(src: SourceDef) -> list[PoolRow]:
        path = Path(root) / f"{src.name}.jsonl"
        if not path.is_file():
            raise Cal1Error(f"{path} not found (pool of source {src.name!r})")
        recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [PoolRow(str(r["id"]), r["text"], r.get("lang", "")) for r in recs]

    return load


# ---------------------------------------------------------------------------------------------------- the draw


@dataclass
class PromptSet:
    """The drawn prompts, the per-source accounting and the prompts skipped by the audit."""

    prompts: list[Prompt]
    sources: list[dict[str, Any]]
    dropped: list[dict[str, Any]]


def _dedup_key(text: str) -> str:
    return " ".join(la.normalize(text))


def draw_prompts(
    sources: Sequence[SourceDef],
    loader: PoolLoader,
    quotas: Mapping[str, int],
    seed: int,
    index: la.NgramIndex,
    revisions: Mapping[str, str],
) -> PromptSet:
    """Seeded draw of ``quotas[domain]`` prompts per domain; skips duplicates and 8-gram overlaps with the items.

    Raises:
        Cal1Error: For a missing or unpinned revision, a domain without source, or a pool too small for its quota.
    """
    for s in sources:
        _check_revision(s, revisions.get(s.name))
    unknown = [d for d in quotas if not any(s.domain == d for s in sources)]
    if unknown:
        raise Cal1Error(f"no prompt source for the domain(s) {unknown}")
    chosen: list[Prompt] = []
    accounting: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for domain, quota in quotas.items():
        members = [s for s in sources if s.domain == domain]
        shares = (
            compute_quotas(quota, {s.name: s.weight for s in members}) if quota > 0 else {s.name: 0 for s in members}
        )
        for src in members:
            want = shares[src.name]
            pool = loader(src)
            order = list(pool)
            random.Random(f"{seed}|{src.name}").shuffle(order)
            taken: list[Prompt] = []
            n_dropped = 0
            for row in order:
                if len(taken) >= want:
                    break
                key = _dedup_key(row.text)
                if len(row.text.strip()) < MIN_PROMPT_CHARS or key in seen:
                    continue
                pid = f"{src.name}/{row.doc_id}"
                hits = la.scan_text(pid, row.text, index)
                if hits:
                    n_dropped += 1
                    dropped.append(
                        {
                            "source": src.name,
                            "doc_id": row.doc_id,
                            "task": hits[0].task,
                            "item_id": hits[0].item_id,
                            "n_hits": len(hits),
                        }
                    )
                    continue
                seen.add(key)
                taken.append(Prompt(pid, domain, src.name, row.doc_id, row.text, row.lang))
            if len(taken) < want:
                raise Cal1Error(
                    f"source {src.name!r}: {len(taken)} usable prompts in a pool of {len(pool)} "
                    f"({n_dropped} skipped by the audit), {want} needed for the quota of {domain!r}"
                )
            chosen.extend(taken)
            accounting.append(
                {
                    "name": src.name,
                    "domain": domain,
                    "repo": src.repo,
                    "revision": revisions[src.name],
                    "license": src.license,
                    "split": src.split,
                    "configs": [c for c in src.configs if c],
                    "column": src.column,
                    "filters": {f: list(v) for f, v in src.filters},
                    "note": src.note,
                    "pool_size": len(pool),
                    "n_selected": len(taken),
                    "n_dropped_by_audit": n_dropped,
                    "languages": dict(Counter(p.lang for p in taken)),
                }
            )
    random.Random(f"{seed}|order").shuffle(chosen)  # a partial run then covers every domain
    return PromptSet(chosen, accounting, dropped)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def write_prompt_set(
    out: Path,
    ps: PromptSet,
    *,
    seed: int,
    weights: Mapping[str, float],
    quotas: Mapping[str, int],
    eval_items_sha256: str,
) -> dict[str, Any]:
    """Write ``prompts.jsonl`` and ``prompts_manifest.json`` (quotas, weights, seed, sources, sha256)."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / PROMPTS_FILE
    path.write_text(
        "".join(
            json.dumps(
                {"id": p.id, "domain": p.domain, "source": p.source, "doc": p.doc_id, "lang": p.lang, "text": p.text},
                ensure_ascii=False,
            )
            + "\n"
            for p in ps.prompts
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema": PROMPTS_SCHEMA,
        "created_utc": _now(),
        "seed": seed,
        "weights": dict(weights),
        "quotas": dict(quotas),
        "n_prompts": len(ps.prompts),
        "sources": ps.sources,
        "dropped_prompts": ps.dropped,
        "audit": {"n": AUDIT_N, "eval_items_sha256": eval_items_sha256, "n_dropped": len(ps.dropped)},
        "prompts": {"file": PROMPTS_FILE, "sha256": la.sha256_file(path)},
    }
    (out / PROMPTS_MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def load_prompt_set(out: Path) -> tuple[list[Prompt], dict[str, Any]]:
    """Read the prompt manifest, refusing a changed prompt file or one that does not meet the declared quotas."""
    mpath = out / PROMPTS_MANIFEST
    if not mpath.is_file():
        raise Cal1Error(f"{mpath} not found")
    m = json.loads(mpath.read_text(encoding="utf-8"))
    if m.get("schema") != PROMPTS_SCHEMA:
        raise Cal1Error(f"{mpath}: schema {m.get('schema')!r}, expected {PROMPTS_SCHEMA!r}")
    path = out / m["prompts"]["file"]
    if la.sha256_file(path) != m["prompts"]["sha256"]:
        raise Cal1Error(f"{path}: sha256 differs from {PROMPTS_MANIFEST}")
    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            prompts.append(Prompt(r["id"], r["domain"], r["source"], r["doc"], r["text"], r.get("lang", "")))
    if len({p.id for p in prompts}) != len(prompts):
        raise Cal1Error(f"{path}: duplicate prompt ids")
    counts = Counter(p.domain for p in prompts)
    for domain, quota in m["quotas"].items():
        if counts.get(domain, 0) != quota:
            raise Cal1Error(f"{path}: {counts.get(domain, 0)} prompts for domain {domain!r}, the quota is {quota}")
    if set(counts) - set(m["quotas"]):
        raise Cal1Error(f"{path}: prompts of domains without a quota: {sorted(set(counts) - set(m['quotas']))}")
    return prompts, m


# ---------------------------------------------------------------------------------------------------- requests


@dataclass(frozen=True)
class Sampling:
    """The harness sampling (PROTOCOL section 6) and the output cap of the generation."""

    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 40
    reasoning_effort: str = "medium"
    max_tokens: int = 8192


def request_body(model: str, prompt: str, s: Sampling) -> dict[str, Any]:
    """The chat-completions body: top_k and the chat-template kwargs at top level, like the harness ``extra_body``."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": s.temperature,
        "top_p": s.top_p,
        "max_tokens": s.max_tokens,
        "top_k": s.top_k,
        "chat_template_kwargs": {"reasoning_effort": s.reasoning_effort},
    }


def config_sha(sampling: Sampling, model: str) -> str:
    """Hash of everything that must not change between the rows of one corpus."""
    blob = json.dumps({"sampling": dataclasses.asdict(sampling), "model": model}, sort_keys=True)
    return _sha256_bytes(blob.encode())


def parse_completion(resp: Mapping[str, Any]) -> dict[str, Any]:
    """Reasoning, answer, finish reason, truncation flag and usage of a chat completion.

    Raises:
        Cal1Error: For a response without choice, without any text, or with a finish reason other than stop or length.
    """
    try:
        choice = resp["choices"][0]
        message = choice["message"]
        reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
        answer = message.get("content") or ""
        finish = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError, AttributeError) as e:
        raise Cal1Error(f"unexpected response shape: {e!r}") from e
    if not str(reasoning).strip() and not str(answer).strip():
        raise Cal1Error(f"empty completion (finish_reason {finish!r})")
    if finish != "stop" and finish not in TRUNCATION_FINISH:
        raise Cal1Error(f"unexpected finish_reason {finish!r}")
    usage = resp.get("usage") or {}
    return {
        "reasoning": str(reasoning),
        "answer": str(answer),
        "finish_reason": finish,
        "truncated": finish in TRUNCATION_FINISH,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def http_call(
    base_url: str, model: str, sampling: Sampling, *, api_key: str, timeout: float
) -> Callable[[Prompt], dict[str, Any]]:
    """A function sending one prompt to ``BASE_URL/chat/completions``; transport problems become :class:`Cal1Error`."""
    url = base_url.rstrip("/") + "/chat/completions"

    def call(p: Prompt) -> dict[str, Any]:
        data = json.dumps(request_body(model, p.text, sampling)).encode()
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise Cal1Error(f"HTTP {e.code}: {e.read()[:200]!r}") from e
        except (urllib.error.URLError, OSError, TimeoutError, json.JSONDecodeError) as e:
            raise Cal1Error(f"request failed: {e!r}") from e

    return call


def server_models(base_url: str, api_key: str, timeout: float = 30) -> list[str]:
    """Ids served by ``BASE_URL/models`` (recorded in the manifest as part of the server provenance)."""
    req = urllib.request.Request(base_url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return [str(m["id"]) for m in json.loads(r.read())["data"]]
    except (urllib.error.URLError, OSError, TimeoutError, KeyError, TypeError, json.JSONDecodeError) as e:
        raise Cal1Error(f"cannot read {base_url}/models: {e!r}") from e


# ---------------------------------------------------------------------------------------------------- generation


class GenStore:
    """The append-only ``generations.jsonl``: loads existing rows, repairs a line cut by a kill, refuses duplicates."""

    def __init__(self, path: Path, repair: bool = True) -> None:
        """Load ``path`` (absent is empty). With ``repair=False`` a cut last line is ignored and the file left alone."""
        self.path = Path(path)
        self.lock = threading.Lock()
        self.rows: list[dict[str, Any]] = []
        self.ok_ids: set[str] = set()
        self.failures: Counter[str] = Counter()
        if self.path.is_file():
            self._load(repair)

    def _load(self, repair: bool) -> None:
        raw = self.path.read_bytes()
        lines = raw.split(b"\n")
        tail = lines.pop()  # text after the last newline: empty, or a line cut by an interruption
        for n, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                self._add(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                raise Cal1Error(f"{self.path}: line {n} is not a valid generation row ({e!r})") from e
        if tail.strip():
            try:
                self._add(json.loads(tail))
                good = len(raw)
                if repair:
                    with self.path.open("ab") as f:
                        f.write(b"\n")
            except (json.JSONDecodeError, KeyError, TypeError):
                good = len(raw) - len(tail)
                if repair:
                    with self.path.open("r+b") as f:
                        f.truncate(good)

    def _add(self, row: dict[str, Any]) -> None:
        pid, status = row["id"], row["status"]
        if status == "ok":
            if pid in self.ok_ids:
                raise Cal1Error(f"{self.path}: a second successful row for prompt {pid!r}")
            self.ok_ids.add(pid)
        else:
            self.failures[pid] += 1
        self.rows.append(row)

    def append(self, row: dict[str, Any]) -> None:
        """Append one row (flushed to disk); refuses a second ``ok`` row for the same prompt."""
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._add(row)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())

    def final_rows(self) -> dict[str, dict[str, Any]]:
        """Per prompt: its ``ok`` row, else its last failed row."""
        final: dict[str, dict[str, Any]] = {}
        for r in self.rows:
            if r["id"] not in final or final[r["id"]]["status"] != "ok":
                final[r["id"]] = r
        return final


def pending_prompts(prompts: Sequence[Prompt], store: GenStore, max_attempts: int) -> list[Prompt]:
    """Prompts without a successful row that have failed fewer than ``max_attempts`` times."""
    return [p for p in prompts if p.id not in store.ok_ids and store.failures[p.id] < max_attempts]


def _prompt_sha(p: Prompt) -> str:
    return _sha256_bytes(p.text.encode())


def generate(
    prompts: Sequence[Prompt],
    store: GenStore,
    call: Callable[[Prompt], dict[str, Any]],
    *,
    concurrency: int,
    max_attempts: int,
    config_sha: str,
) -> dict[str, int]:
    """Ask ``call`` for every pending prompt (bounded concurrency); append one row per attempt, ok or failed.

    Raises:
        Cal1Error: For a concurrency outside ``1..MAX_CONCURRENCY``, rows of prompts not in ``prompts`` or whose
            text changed, or rows made with another configuration.
    """
    if not 1 <= concurrency <= MAX_CONCURRENCY:
        raise Cal1Error(f"concurrency must be between 1 and {MAX_CONCURRENCY}, got {concurrency}")
    by_id = {p.id: p for p in prompts}
    for r in store.rows:
        if r["id"] not in by_id:
            raise Cal1Error(f"{store.path}: a row for prompt {r['id']!r}, which is not in the prompt manifest")
    for r in store.rows:
        if r.get("config_sha256") != config_sha:
            raise Cal1Error(
                f"{store.path}: rows were made with another configuration (sampling or model) than this run"
            )
    for p in prompts:
        for r in store.rows:
            if r["id"] == p.id and r.get("prompt_sha256") != _prompt_sha(p):
                raise Cal1Error(f"{store.path}: the text of prompt {p.id!r} changed since its row was written")
    todo = pending_prompts(prompts, store, max_attempts)
    stats = {
        "ok": 0,
        "failed": 0,
        "skipped_done": len(store.ok_ids),
        "gave_up": sum(1 for p in prompts if p.id not in store.ok_ids and store.failures[p.id] >= max_attempts),
    }
    counter = threading.Lock()
    done = 0

    def work(p: Prompt) -> None:
        nonlocal done
        t0 = time_now()
        base = {
            "id": p.id,
            "domain": p.domain,
            "attempt": store.failures[p.id] + 1,
            "prompt_sha256": _prompt_sha(p),
            "config_sha256": config_sha,
        }
        try:
            parsed = parse_completion(call(p))
            row = {**base, "status": "ok", **parsed, "latency_s": round(time_now() - t0, 2), "utc": _now()}
        except Cal1Error as e:
            row = {**base, "status": "failed", "error": str(e), "utc": _now()}
        store.append(row)
        with counter:
            stats["ok" if row["status"] == "ok" else "failed"] += 1
            done += 1
            if done % 20 == 0 or done == len(todo):
                print(f"generated {done}/{len(todo)} ({stats['failed']} failed)", file=sys.stderr, flush=True)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(work, todo))
    return stats


def time_now() -> float:
    """Monotonic seconds."""
    return time.monotonic()


# ---------------------------------------------------------------------------------------------------- rendering


class ChatRenderer(Protocol):
    """Renders prompt, reasoning and answer with the model's chat template and tokenizes the result."""

    name: str
    eos_id: int
    template_sha256: str

    def render_text(self, prompt: str, reasoning: str, answer: str, effort: str) -> str:
        """The chat-formatted text of one finished exchange."""
        ...

    def encode(self, prompt: str, reasoning: str, answer: str, effort: str) -> list[int]:
        """Token ids of :meth:`render_text` (no extra special tokens)."""
        ...


class ByteChatRenderer:
    """Byte tokens over a minimal text with the Olala channel markers; for tests and the dry-run plumbing."""

    name = "byte"
    eos_id = 256
    template_sha256 = "byte-minimal-olala-layout"

    def render_text(self, prompt: str, reasoning: str, answer: str, effort: str) -> str:
        """Olala-shaped text: effort line, user turn, analysis channel (reasoning), final channel (answer)."""
        return (
            "<|endoftext|><|im_start|>system<|channel_start|>instructions<|content|>You are Olala.\n"
            f"Reasoning effort: {effort}<|channel_end|><|im_end|>"
            f"<|im_start|>user<|channel_start|>text<|content|>{prompt}<|channel_end|><|im_end|>"
            f"<|im_start|>assistant<|channel_start|>analysis<|content|>{reasoning}<|channel_end|>"
            f"<|channel_start|>final<|content|>{answer}<|channel_end|><|im_end|>"
        )

    def encode(self, prompt: str, reasoning: str, answer: str, effort: str) -> list[int]:
        """UTF-8 bytes of the rendered text."""
        return list(self.render_text(prompt, reasoning, answer, effort).encode("utf-8"))


class HFChatRenderer:
    """A ``transformers`` tokenizer with the model's own chat template (``reasoning_effort`` as template kwarg)."""

    def __init__(self, tok: Any, eos_id: int, name: str) -> None:
        """Wrap a tokenizer that carries the chat template."""
        self.tok, self.eos_id, self.name = tok, eos_id, name
        self.template_sha256 = _sha256_bytes(str(getattr(tok, "chat_template", "")).encode())

    def render_text(self, prompt: str, reasoning: str, answer: str, effort: str) -> str:
        """The template applied to user prompt plus assistant turn (``reasoning_content`` = analysis channel)."""
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer, "reasoning_content": reasoning},
        ]
        return self.tok.apply_chat_template(messages, tokenize=False, reasoning_effort=effort)

    def encode(self, prompt: str, reasoning: str, answer: str, effort: str) -> list[int]:
        """Token ids of the rendered text, no special tokens added on top of those the template writes."""
        return list(self.tok.encode(self.render_text(prompt, reasoning, answer, effort), add_special_tokens=False))


def make_renderer(spec: str) -> ChatRenderer:
    """``byte`` or a model directory with tokenizer files and ``config.json`` (the template is the tokenizer's)."""
    if spec == "byte":
        return ByteChatRenderer()
    hf = calib_data.HFTokenizer(Path(spec))
    return HFChatRenderer(hf.tok, hf.eos_id, hf.name)


# ---------------------------------------------------------------------------------------------------- build


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_bundle(
    out: Path,
    *,
    renderer: ChatRenderer,
    window: int,
    eval_items: Path,
    seed: int,
    max_attempts: int,
    drop_flagged: bool = False,
    max_windows: int | None = None,
) -> dict[str, Any]:
    """Audit prompts and generated text, render and pack the traces, write the bundle and its manifest.

    The manifest and windows of an earlier run are deleted first, so a refusal never leaves a passing manifest behind.

    Raises:
        la.LeakageError: If any prompt or generated text overlaps an evaluation item (unless ``drop_flagged``).
        Cal1Error: If generation is incomplete, a domain has no trace, or the traces do not fill one window.
    """
    for name in (MANIFEST, WINDOWS_FILE):
        (out / name).unlink(missing_ok=True)
    prompts, pm = load_prompt_set(out)
    cfg = _read_json(out / GEN_CONFIG)
    sampling = cfg["sampling"]
    store = GenStore(out / GENERATIONS_FILE, repair=False)
    todo = pending_prompts(prompts, store, max_attempts)
    if todo:
        raise Cal1Error(
            f"generation is incomplete: {len(todo)} prompts still to generate (run again with the server up)"
        )
    index = load_index(eval_items)
    final = store.final_rows()
    by_id = {p.id: p for p in prompts}
    traces = [(by_id[i], r) for i, r in final.items() if r["status"] == "ok"]
    traces.sort(key=lambda t: [p.id for p in prompts].index(t[0].id))
    failures = [
        {"id": i, "domain": by_id[i].domain, "attempts": store.failures[i], "error": r.get("error", "")}
        for i, r in final.items()
        if r["status"] != "ok"
    ]
    n_ok = len(traces)
    for domain, quota in pm["quotas"].items():
        if quota > 0 and not any(p.domain == domain for p, _ in traces):
            raise Cal1Error(f"domain {domain!r} has no successful trace ({quota} prompts, all failed)")

    def docs_of(items: Sequence[tuple[Prompt, dict[str, Any]]]) -> list[tuple[str, str]]:
        d: list[tuple[str, str]] = []
        for p, r in items:
            d.append((f"prompt/{p.id}", p.text))
            d.append((f"generation/{p.id}", r["reasoning"] + "\n" + r["answer"]))
        return d

    report = la.audit_documents(docs_of(traces), index)
    dropped: list[dict[str, Any]] = []
    if not report.ok:
        if not drop_flagged:
            la.raise_if_leaky(report)
        flagged = {d.split("/", 1)[1] for d in report.flagged_docs}
        for h_id in sorted(flagged):
            hs = [h for h in report.hits if h.doc.split("/", 1)[1] == h_id]
            dropped.append(
                {
                    "id": h_id,
                    "docs": sorted({h.doc.split("/", 1)[0] for h in hs}),
                    "tasks": sorted({h.task for h in hs}),
                    "n_hits": len(hs),
                }
            )
        traces = [t for t in traces if t[0].id not in flagged]
        report = la.audit_documents(docs_of(traces), index)
        la.raise_if_leaky(report)  # the survivors must be clean on their own
    drop_ids = {d["id"] for d in dropped}
    out.mkdir(parents=True, exist_ok=True)
    with (out / TEXT_FILE).open("w", encoding="utf-8") as f:
        for p, r in traces:
            f.write(json.dumps({"source": "prompt", "doc": p.id, "text": p.text}, ensure_ascii=False) + "\n")
            f.write(
                json.dumps(
                    {"source": "generation", "doc": p.id, "text": r["reasoning"] + "\n" + r["answer"]},
                    ensure_ascii=False,
                )
                + "\n"
            )
    (out / EVAL_ITEMS_COPY).write_bytes(Path(eval_items).read_bytes())
    items_sha = la.sha256_file(out / EVAL_ITEMS_COPY)
    audit_json = report.as_dict() | {
        "created_utc": _now(),
        "eval_items_sha256": items_sha,
        "scope": "prompts and generated text",
    }
    (out / AUDIT_REPORT).write_text(json.dumps(audit_json, indent=2) + "\n")

    effort = sampling["reasoning_effort"]
    order = list(range(len(traces)))
    random.Random(f"{seed}|order").shuffle(order)
    token_lists: list[list[int]] = [[] for _ in traces]
    for i in order:
        p, r = traces[i]
        token_lists[i] = renderer.encode(p.text, r["reasoning"], r["answer"], effort)
    try:
        windows = calib_data.pack_windows(
            [token_lists[i] for i in order], renderer.eos_id, window, max_windows=max_windows
        )
    except ValueError as e:
        raise Cal1Error(str(e)) from e
    (out / WINDOWS_FILE).write_text(json.dumps(windows.tolist()))

    domains: dict[str, dict[str, Any]] = {}
    for d, quota in pm["quotas"].items():
        mine = [(p, r) for p, r in traces if p.domain == d]
        domains[d] = {
            "weight": pm["weights"].get(d),
            "quota": quota,
            "n_prompts": sum(1 for p in prompts if p.domain == d),
            "n_ok": sum(1 for p in prompts if p.domain == d and final.get(p.id, {}).get("status") == "ok"),
            "n_failed": sum(1 for f_ in failures if f_["domain"] == d),
            "n_truncated": sum(1 for _, r in mine if r["truncated"]),
            "n_dropped_leak": sum(1 for x in drop_ids if by_id[x].domain == d),
            "n_traces": len(mine),
            "n_tokens": sum(len(token_lists[traces.index(t)]) for t in mine),
        }
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": _now(),
        "seed": seed,
        "n_tokens": int(windows.numel()),
        "window": window,
        "n_windows": int(windows.shape[0]),
        "tokenizer": {"name": renderer.name, "eos_id": renderer.eos_id},
        "chat_template": {
            "reasoning_effort": effort,
            "renderer": renderer.name,
            "template_sha256": renderer.template_sha256,
        },
        "sampling": sampling,
        "server": {
            "provenance": cfg["server_provenance"],
            "models": cfg["server_models"],
            "base_url": cfg.get("base_url", ""),
            "model": cfg["model"],
        },
        "prompt_sources": pm["sources"],
        "quotas": pm["quotas"],
        "weights": pm["weights"],
        "domains": domains,
        "counts": {
            "n_prompts": len(prompts),
            "n_ok": n_ok,
            "n_failed": len(failures),
            "n_traces": len(traces),
            "n_truncated": sum(1 for _, r in traces if r["truncated"]),
            "n_dropped_leak": len(dropped),
            "n_dropped_prompts_at_draw": len(pm["dropped_prompts"]),
            "n_tokens_rendered": sum(len(t) for t in token_lists),
        },
        "failures": failures,
        "windows": {"file": WINDOWS_FILE, "sha256": la.sha256_file(out / WINDOWS_FILE)},
        "audit": {
            "ok": True,
            "n": AUDIT_N,
            "n_hits": 0,
            "n_docs": report.n_docs,
            "n_scanned_prompts": len(traces),
            "n_scanned_generations": len(traces),
            "n_eval_fields": report.n_eval_fields,
            "n_skipped_short": report.n_skipped_short,
            "n_eval_ngrams": report.n_ngrams,
            "eval_items_file": EVAL_ITEMS_COPY,
            "eval_items_sha256": items_sha,
            "dropped_traces": dropped,
        },
        "files": {name: la.sha256_file(out / name) for name in HASHED_FILES},
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def verify_bundle(out: Path) -> dict[str, Any]:
    """Re-check a finished directory: every file hash, the windows (``calib_data.read_windows``) and the audit.

    Raises:
        Cal1Error: For a missing manifest, a file whose sha256 differs, or windows that ``read_windows`` refuses.
        la.LeakageError: If the audited text overlaps the item file of the bundle.
    """
    mpath = out / MANIFEST
    if not mpath.is_file():
        raise Cal1Error(f"{mpath} not found: no finished CAL-1 build in {out}")
    m = _read_json(mpath)
    if m.get("schema") != SCHEMA:
        raise Cal1Error(f"{mpath}: schema {m.get('schema')!r}, expected {SCHEMA!r}")
    for name, sha in m["files"].items():
        path = out / name
        if not path.is_file() or la.sha256_file(path) != sha:
            raise Cal1Error(f"{name}: missing or sha256 differs from the manifest")
    try:
        calib_data.read_windows(mpath)
    except calib_data.ManifestError as e:
        raise Cal1Error(str(e)) from e
    la.raise_if_leaky(la.audit_documents(la.read_bundle(out), load_index(out / EVAL_ITEMS_COPY)))
    return m


# ---------------------------------------------------------------------------------------------------- CLI


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="cal1_generate.py", description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, required=True, help="the CAL-1 directory (created, resumed)")
    p.add_argument("--eval-items", type=Path, default=DEFAULT_EVAL_ITEMS)
    p.add_argument(
        "--pool-dir", type=Path, help="read candidate prompts from DIR/<source>.jsonl instead of Hugging Face"
    )
    p.add_argument(
        "--revision",
        action="append",
        default=[],
        metavar="SOURCE=SHA",
        help="pinned dataset revision (40 hex), per source",
    )
    p.add_argument(
        "--column", action="append", default=[], metavar="SOURCE=COLUMN", help="override a source's text column"
    )
    p.add_argument("--total", type=int, default=240)
    p.add_argument("--quota", action="append", default=[], metavar="DOMAIN=N", help="override one domain's quota")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base-url", default="", help="OpenAI-compatible endpoint, e.g. http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="", help="served model name")
    p.add_argument("--server-provenance", default="", help="free text: server version, weights identity (recorded)")
    p.add_argument("--concurrency", type=int, default=24)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--timeout", type=float, default=3600.0)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--reasoning-effort", default="medium", choices=["disabled", "low", "medium", "high"])
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument(
        "--tokenizer", default="byte", help="byte | a model directory (tokenizer, chat template, config.json)"
    )
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--max-windows", type=int)
    p.add_argument("--drop-flagged", action="store_true", help="drop traces that overlap an item instead of refusing")
    p.add_argument("--dry-run", action="store_true", help="print the plan and counts; no server, nothing written")
    p.add_argument("--verify", action="store_true", help="re-check a finished directory and exit")
    return p.parse_args(argv)


def _pairs(items: Sequence[str], what: str) -> dict[str, str]:
    out = {}
    for it in items:
        k, sep, v = it.partition("=")
        if not sep or not k or not v:
            raise Cal1Error(f"{what} must be NAME=VALUE, got {it!r}")
        out[k] = v
    return out


def _plan(
    args: argparse.Namespace, prompts: Sequence[Prompt], pm: Mapping[str, Any], store: GenStore, existing: bool
) -> str:
    lines = [
        "CAL-1 plan (dry run: nothing written, no server contacted; prompt manifest "
        + ("read from the output directory)" if existing else "drawn in memory)")
    ]
    lines.append(f"prompts: {len(prompts)} prompts in {len(pm['quotas'])} domains, seed {pm['seed']}")
    for d, q in pm["quotas"].items():
        lines.append(f"  {d}: {q} (weight {pm['weights'].get(d)})")
    lines.append("sources (public, revision pinned):")
    for s in pm["sources"]:
        lines.append(
            f"  {s['name']}: {s['repo']}@{s['revision']} license {s['license']} split {s['split']}; "
            f"pool {s['pool_size']}, "
            f"selected {s['n_selected']}, skipped by the audit {s['n_dropped_by_audit']}"
        )
    s = Sampling(args.temperature, args.top_p, args.top_k, args.reasoning_effort, args.max_tokens)
    lines.append(
        f"sampling: temperature {s.temperature}, top_p {s.top_p}, top_k {s.top_k}, "
        f"reasoning_effort {s.reasoning_effort}, "
        f"max_tokens {s.max_tokens}; concurrency {args.concurrency}, max attempts {args.max_attempts}"
    )
    todo = pending_prompts(prompts, store, args.max_attempts)
    gave_up = sum(1 for p in prompts if p.id not in store.ok_ids and store.failures[p.id] >= args.max_attempts)
    lines.append(
        f"generation: {len(store.ok_ids)} done, {len(todo)} to generate, {gave_up} gave up; "
        f"at most {len(todo) * s.max_tokens} new tokens"
    )
    lines.append(
        f"windows: window {args.window} tokens, tokenizer {args.tokenizer}; 8-gram audit of prompts and generated "
        f"text against {args.eval_items}"
    )
    return "\n".join(lines)


def _check_resume_config(out: Path, args: argparse.Namespace, sampling: Sampling, need_server: bool) -> dict[str, Any]:
    """Create or compare ``generation_config.json``; return it (the model and provenance of an earlier run are kept)."""
    path = out / GEN_CONFIG
    stored = _read_json(path) if path.is_file() else None
    model = args.model or (stored or {}).get("model", "")
    current = {"model": model, "sampling": dataclasses.asdict(sampling)}
    if stored:
        for key in ("model", "sampling"):
            if stored[key] != current[key]:
                raise Cal1Error(
                    f"{GEN_CONFIG}: {key} differs from the earlier run ({stored[key]!r} vs {current[key]!r}); "
                    "use a new --out"
                )
        if args.server_provenance and stored["server_provenance"] != args.server_provenance:
            raise Cal1Error(f"{GEN_CONFIG}: the server provenance differs from the earlier run; use a new --out")
        return stored
    if not need_server:
        raise Cal1Error(f"{path} not found: this directory has no generation record")
    if not (args.base_url and model and args.server_provenance):
        raise Cal1Error("a real run needs --base-url, --model and --server-provenance (recorded in the manifest)")
    return {
        "schema": "cal1-gen/1",
        **current,
        "config_sha256": config_sha(sampling, model),
        "server_provenance": args.server_provenance,
        "base_url": args.base_url,
        "server_models": [],
    }


def run(args: argparse.Namespace) -> int:
    """The pipeline behind :func:`main`; returns the exit status of a success (raises on refusals)."""
    out: Path = args.out
    index = load_index(args.eval_items)
    items_sha = la.sha256_file(args.eval_items)
    revisions = _pairs(args.revision, "--revision")
    columns = _pairs(args.column, "--column")
    sources = tuple(dataclasses.replace(s, column=columns.get(s.name, s.column)) for s in DEFAULT_SOURCES)
    unknown = (set(revisions) | set(columns)) - {s.name for s in sources}
    if unknown:
        raise Cal1Error(f"unknown source(s) {sorted(unknown)}; known: {[s.name for s in sources]}")
    quotas = compute_quotas(args.total, DEFAULT_WEIGHTS)
    for d, n in {k: int(v) for k, v in _pairs(args.quota, "--quota").items()}.items():
        if d not in quotas:
            raise Cal1Error(f"unknown domain {d!r}; known: {list(quotas)}")
        quotas[d] = n
    sampling = Sampling(args.temperature, args.top_p, args.top_k, args.reasoning_effort, args.max_tokens)
    existing = (out / PROMPTS_MANIFEST).is_file()
    if existing:
        prompts, pm = load_prompt_set(out)
        if pm["quotas"] != quotas or pm["seed"] != args.seed:
            raise Cal1Error(
                f"{out} holds prompts drawn with quotas {pm['quotas']} and seed {pm['seed']}, "
                f"not {quotas} and seed {args.seed}; use a new --out"
            )
        have = {s["name"]: s["revision"] for s in pm["sources"]}
        if any(have.get(n) != r for n, r in revisions.items()):
            raise Cal1Error(
                f"{out} holds prompts drawn at other dataset revisions than the --revision given; use a new --out"
            )
    else:
        loader = dir_pool_loader(args.pool_dir) if args.pool_dir else hf_pool_loader(revisions)
        ps = draw_prompts(sources, loader, quotas, args.seed, index, revisions)
        prompts = ps.prompts
        pm = {"quotas": quotas, "weights": DEFAULT_WEIGHTS, "seed": args.seed, "sources": ps.sources}
        if not args.dry_run:
            write_prompt_set(
                out, ps, seed=args.seed, weights=DEFAULT_WEIGHTS, quotas=quotas, eval_items_sha256=items_sha
            )
    store = GenStore(out / GENERATIONS_FILE, repair=not args.dry_run)
    if args.dry_run:
        print(_plan(args, prompts, pm, store, existing))
        return 0
    todo = pending_prompts(prompts, store, args.max_attempts)
    cfg = _check_resume_config(out, args, sampling, need_server=bool(todo))
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    if not (out / GEN_CONFIG).is_file():
        if todo:
            cfg["server_models"] = server_models(args.base_url, api_key)
        (out / GEN_CONFIG).write_text(
            json.dumps(cfg | {"max_attempts": args.max_attempts, "created_utc": _now()}, indent=2) + "\n"
        )
    if todo:
        call = http_call(
            cfg.get("base_url") or args.base_url, cfg["model"], sampling, api_key=api_key, timeout=args.timeout
        )
        stats = generate(
            prompts,
            store,
            call,
            concurrency=args.concurrency,
            max_attempts=args.max_attempts,
            config_sha=cfg["config_sha256"],
        )
        print(f"generation: {stats}", file=sys.stderr)
    manifest = build_bundle(
        out,
        renderer=make_renderer(args.tokenizer),
        window=args.window,
        eval_items=args.eval_items,
        seed=args.seed,
        max_attempts=args.max_attempts,
        drop_flagged=args.drop_flagged,
        max_windows=args.max_windows,
    )
    c = manifest["counts"]
    print(
        f"wrote {out / MANIFEST}: {manifest['n_windows']} windows of {manifest['window']} tokens, "
        f"{c['n_traces']} traces ({c['n_truncated']} truncated, {c['n_failed']} failed, "
        f"{c['n_dropped_leak']} dropped by the audit), audit clean"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI. Returns 0 on success, 1 on a leakage refusal, 2 on any other refusal."""
    args = _parse_args(argv)
    try:
        if args.verify:
            m = verify_bundle(args.out)
            print(f"OK: {args.out} is intact ({m['n_windows']} windows, audit clean)")
            return 0
        return run(args)
    except la.LeakageError as e:
        print(f"REFUSED (leakage): {e}", file=sys.stderr)
        return 1
    except (Cal1Error, calib_data.ManifestError, ValueError, OSError, ImportError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
