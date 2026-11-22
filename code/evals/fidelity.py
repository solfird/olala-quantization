"""Logit-level fidelity tool: collect top-k log-probs from an endpoint and compare two files.

Sub-commands (see ``--help`` and README.md):

* ``fetch-wikitext``: download wikitext-2 (test split) through ``datasets`` and write a text file;
* ``prepare``: tokenize a text file and cut it into fixed-length chunks (``chunks.json``);
* ``collect``: send the chunks to ``/v1/completions`` and store per-position top-k log-probs;
* ``compare``: compare a reference ``.npz`` with a variant ``.npz``.

The maths lives in :mod:`fidelity_math`; the bootstrap in :mod:`stats`.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from fidelity_math import kl_topk_union, top1_agreement
from stats import cluster_bootstrap_multi

APPROXIMATION_NOTE = (
    "KL is computed on the union of the two top-k lists plus one 'other' bucket; tokens a list "
    "does not show are imputed (see fidelity_math.py and DECISIONS.md). It is an approximation, "
    "not the full-vocabulary divergence."
)


@dataclass
class FidelityData:
    """Per-position top-k log-probabilities of one model over a fixed set of chunks.

    Attributes:
        vocab: Token strings (``"token_id:N"`` on vLLM) indexed by ``top_ids``.
        top_ids: ``int32 [chunks, positions, k]`` indices into ``vocab``; ``-1`` is padding.
        top_lp: ``float32 [chunks, positions, k]`` natural-log probabilities, descending.
        target_lp: ``float32 [chunks, positions]`` log-probability of the actual next token.
        meta: JSON-serialisable record of the exact collection settings.
    """

    vocab: np.ndarray
    top_ids: np.ndarray
    top_lp: np.ndarray
    target_lp: np.ndarray
    meta: dict[str, Any]


def save_fidelity(path: str | Path, data: FidelityData) -> None:
    """Write ``data`` to a compressed ``.npz`` (no pickled objects).

    Args:
        path: Destination file.
        data: The container to store.
    """
    np.savez_compressed(
        path,
        vocab=np.asarray(data.vocab, dtype=str),
        top_ids=data.top_ids,
        top_lp=data.top_lp,
        target_lp=data.target_lp,
        meta_json=np.array(json.dumps(data.meta, sort_keys=True)),
    )


def load_fidelity(path: str | Path) -> FidelityData:
    """Read a file written by :func:`save_fidelity`.

    Args:
        path: The ``.npz`` file.

    Returns:
        The stored container.
    """
    with np.load(path, allow_pickle=False) as z:
        return FidelityData(
            vocab=z["vocab"],
            top_ids=z["top_ids"],
            top_lp=z["top_lp"],
            target_lp=z["target_lp"],
            meta=json.loads(str(z["meta_json"])),
        )


def _to_common_ids(
    data: FidelityData, common: np.ndarray, max_k: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ids, lp)`` flattened to ``[N, K]`` with ids mapped into ``common``."""
    mapping = np.searchsorted(common, data.vocab)
    ids = np.where(data.top_ids >= 0, mapping[np.maximum(data.top_ids, 0)], -1)
    lp = data.top_lp
    if max_k is not None:
        ids, lp = ids[..., :max_k], lp[..., :max_k]
    k = ids.shape[-1]
    return ids.reshape(-1, k), lp.reshape(-1, k).astype(np.float64)


def compare_fidelity(
    ref: FidelityData,
    var: FidelityData,
    *,
    max_k: int | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare a quantized variant with its reference, with bootstrap CIs over chunks.

    Args:
        ref: Reference (e.g. bf16) results.
        var: Variant results over the same chunks.
        max_k: Optional truncation of both top-k lists, to measure how sensitive the KL
            approximation is to ``k``.
        n_boot: Bootstrap resamples (chunks are resampled, not positions).
        seed: Bootstrap seed.

    Returns:
        A dict with ``value``/``lo``/``hi`` entries for ``ppl_ref``, ``ppl_var``,
        ``ppl_ratio`` (variant over reference, paired), ``kld_mean``, ``kld_median``,
        ``kld_p99`` and ``top1_agreement`` plus counts and both files' metadata.

    Raises:
        ValueError: If the files were not made from the same chunks or differ in shape.
    """
    sha_ref, sha_var = ref.meta.get("chunks_sha256"), var.meta.get("chunks_sha256")
    if sha_ref is not None and sha_var is not None and sha_ref != sha_var:
        raise ValueError(f"files were made from different chunks ({sha_ref} vs {sha_var})")
    if ref.target_lp.shape != var.target_lp.shape:
        raise ValueError(f"shape mismatch: {ref.target_lp.shape} vs {var.target_lp.shape}")

    common = np.unique(np.concatenate([ref.vocab, var.vocab]))
    r_ids, r_lp = _to_common_ids(ref, common, max_k)
    v_ids, v_lp = _to_common_ids(var, common, max_k)
    n_chunks, n_pos = ref.target_lp.shape
    kl = kl_topk_union(r_ids, r_lp, v_ids, v_lp).reshape(n_chunks, n_pos)
    agree = top1_agreement(r_ids, r_lp, v_ids, v_lp).reshape(n_chunks, n_pos)

    # one cluster per chunk, columns: KL, top-1 agreement, NLL reference, NLL variant
    clusters = [
        np.column_stack([kl[c], agree[c], -ref.target_lp[c], -var.target_lp[c]])
        for c in range(n_chunks)
    ]
    statistics = {
        "kld_mean": lambda v: float(v[:, 0].mean()),
        "kld_median": lambda v: float(np.median(v[:, 0])),
        "kld_p99": lambda v: float(np.percentile(v[:, 0], 99)),
        "top1_agreement": lambda v: float(v[:, 1].mean()),
        "ppl_ref": lambda v: float(np.exp(v[:, 2].mean())),
        "ppl_var": lambda v: float(np.exp(v[:, 3].mean())),
        "ppl_ratio": lambda v: float(np.exp(v[:, 3].mean() - v[:, 2].mean())),
    }
    boot = cluster_bootstrap_multi(clusters, statistics, n_boot=n_boot, seed=seed)
    result: dict[str, Any] = {
        name: {"value": value, "lo": lo, "hi": hi} for name, (value, lo, hi) in boot.items()
    }
    result.update(
        n_chunks=n_chunks,
        n_positions=int(n_chunks * n_pos),
        k=int(r_ids.shape[1]),
        n_boot=n_boot,
        seed=seed,
        confidence=0.95,
        ref_meta=ref.meta,
        var_meta=var.meta,
        approximation=APPROXIMATION_NOTE,
    )
    return result


def render_comparison_markdown(result: dict[str, Any]) -> str:
    """Render the dict returned by :func:`compare_fidelity` as a markdown report.

    Args:
        result: Output of :func:`compare_fidelity`.

    Returns:
        Markdown text; every number is taken from ``result``.
    """

    def row(label: str, key: str, fmt: str) -> str:
        m = result[key]
        return f"| {label} | {m['value']:{fmt}} | [{m['lo']:{fmt}}, {m['hi']:{fmt}}] |"

    lines = [
        "### Logit fidelity",
        "",
        f"{result['n_chunks']} chunks, {result['n_positions']} scored positions, "
        f"top-{result['k']} log-probs, {result['n_boot']} bootstrap resamples over chunks "
        f"(seed {result['seed']}), {result['confidence']:.0%} percentile intervals.",
        "",
        "| metric | value | 95% CI |",
        "|---|---|---|",
        row("Perplexity, reference", "ppl_ref", ".3f"),
        row("Perplexity, variant", "ppl_var", ".3f"),
        row("Perplexity ratio (variant / reference)", "ppl_ratio", ".4f"),
        row("Mean KL(ref || variant), nats", "kld_mean", ".5f"),
        row("Median KL(ref || variant), nats", "kld_median", ".5f"),
        row("99th percentile KL, nats", "kld_p99", ".5f"),
        row("Same top-1 token (fraction)", "top1_agreement", ".4f"),
        "",
        f"Note: {result['approximation']}",
        "",
        "| file | model | endpoint | mode | k | collected (UTC) |",
        "|---|---|---|---|---|---|",
    ]
    for label, meta in (("reference", result["ref_meta"]), ("variant", result["var_meta"])):
        lines.append(
            f"| {label} | {meta.get('model', '?')} | {meta.get('endpoint', '?')} | "
            f"{meta.get('mode', '?')} | {meta.get('k', '?')} | {meta.get('collected_utc', '?')} |"
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- chunking

WIKITEXT_REPO = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
ZERO_PROB_LOGPROB = -9000.0
"""vLLM clamps ``-inf`` to -9999; anything at or below this is treated as probability zero."""


class RequestFailedError(RuntimeError):
    """A request to the endpoint failed for good (client error or retry budget exhausted)."""


@dataclass
class Chunks:
    """Fixed-length token chunks and their provenance.

    Attributes:
        token_ids: ``int64 [n_chunks, chunk_len]``.
        meta: Provenance written by :func:`prepare_chunks` (may be empty for hand-made files).
        sha256: Hash of ``token_ids``, recomputed on load; identifies the chunk set.
    """

    token_ids: np.ndarray
    meta: dict[str, Any]
    sha256: str


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ids_sha256(token_ids: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(token_ids, dtype="<i8").tobytes()).hexdigest()


def chunk_tokens(token_ids: Sequence[int], n_chunks: int, chunk_len: int) -> np.ndarray:
    """Cut the first ``n_chunks * chunk_len`` tokens into consecutive, non-overlapping chunks.

    Args:
        token_ids: The whole tokenized text.
        n_chunks: Number of chunks wanted.
        chunk_len: Tokens per chunk.

    Returns:
        ``int64`` array ``[n_chunks, chunk_len]``.

    Raises:
        ValueError: If the text has fewer than ``n_chunks * chunk_len`` tokens.
    """
    need = n_chunks * chunk_len
    if len(token_ids) < need:
        raise ValueError(f"needs {need} tokens but the text only has {len(token_ids)}")
    return np.asarray(token_ids[:need], dtype=np.int64).reshape(n_chunks, chunk_len)


def prepare_chunks(
    text_path: str | Path,
    tokenizer_path: str | Path,
    out_path: str | Path,
    n_chunks: int = 64,
    chunk_len: int = 512,
    name: str | None = None,
    add_special_tokens: bool = False,
) -> Chunks:
    """Tokenize a text file with a local ``tokenizer.json`` and write ``chunks.json``.

    Args:
        text_path: UTF-8 text file (e.g. wikitext-2 test, a French slice, a Python slice).
        tokenizer_path: A ``tokenizer.json`` file, or a directory containing one.
        out_path: Destination ``chunks.json``.
        n_chunks: Number of chunks.
        chunk_len: Tokens per chunk.
        name: Label stored in the file (defaults to the text file stem).
        add_special_tokens: Whether the tokenizer may add BOS/EOS tokens.

    Returns:
        The chunks that were written.
    """
    from tokenizers import Tokenizer  # lazy: keeps ``compare`` start-up fast

    text_path, out_path = Path(text_path), Path(out_path)
    tok_file = Path(tokenizer_path)
    if tok_file.is_dir():
        tok_file = tok_file / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tok_file))
    text = text_path.read_text(encoding="utf-8")
    ids = tokenizer.encode(text, add_special_tokens=add_special_tokens).ids
    arr = chunk_tokens(ids, n_chunks, chunk_len)
    meta = {
        "name": name or text_path.stem,
        "n_chunks": n_chunks,
        "chunk_len": chunk_len,
        "tokens_available": len(ids),
        "text_file": text_path.name,
        "text_sha256": _sha256_file(text_path),
        "tokenizer_file": str(tok_file),
        "tokenizer_sha256": _sha256_file(tok_file),
        "add_special_tokens": add_special_tokens,
        "layout": "consecutive chunks from the start of the text",
        "chunks_sha256": _ids_sha256(arr),
        "prepared_utc": _now(),
    }
    out_path.write_text(
        json.dumps({"meta": meta, "token_ids": arr.tolist()}, separators=(",", ":")),
        encoding="utf-8",
    )
    return Chunks(arr, meta, meta["chunks_sha256"])


def load_chunks(path: str | Path) -> Chunks:
    """Read a ``chunks.json`` file.

    Args:
        path: File written by :func:`prepare_chunks`.

    Returns:
        The chunks with a freshly computed hash.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    arr = np.asarray(raw["token_ids"], dtype=np.int64)
    return Chunks(arr, raw.get("meta", {}), _ids_sha256(arr))


def fetch_wikitext(out_path: str | Path, split: str = "test") -> dict[str, Any]:
    """Download wikitext-2 (raw) through ``datasets`` and write it as one text file.

    The rows are concatenated unchanged (each already ends with a newline), which reproduces
    the usual ``wiki.<split>.raw`` file. The dataset revision that was fetched is resolved to a
    commit hash and written, with a hash of the text, to ``<out_path>.meta.json``.

    Args:
        out_path: Destination text file.
        split: ``test``, ``validation`` or ``train``.

    Returns:
        The provenance record that was written next to the text.
    """
    import datasets  # lazy imports: only this command needs them
    from huggingface_hub import HfApi

    revision = HfApi().dataset_info(WIKITEXT_REPO).sha
    rows = datasets.load_dataset(WIKITEXT_REPO, WIKITEXT_CONFIG, split=split, revision=revision)
    text = "".join(rows["text"])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    record = {
        "repo": WIKITEXT_REPO,
        "config": WIKITEXT_CONFIG,
        "split": split,
        "revision": revision,
        "n_rows": len(rows),
        "n_chars": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "fetched_utc": _now(),
        "datasets_version": importlib_metadata.version("datasets"),
    }
    Path(str(out_path) + ".meta.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


# --------------------------------------------------------------------------- collection


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in ("httpx", "numpy", "tokenizers"):
        out[pkg] = importlib_metadata.version(pkg)
    return out


def _error_message(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        err = body.get("error", body) if isinstance(body, dict) else body
        return str(err.get("message", err) if isinstance(err, dict) else err)
    except ValueError:
        return resp.text[:300]


def _post_with_retries(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
    retries: int,
    backoff_s: float,
    retry_log: list[str],
) -> dict[str, Any]:
    """POST ``payload``; retry transport errors, 429 and 5xx; fail fast on other 4xx."""
    attempt = 0
    while True:
        try:
            resp = client.post(url, json=payload)
        except httpx.TransportError as exc:
            err = f"{type(exc).__name__}: {exc}"
        else:
            if resp.status_code < 400:
                return resp.json()
            err = f"HTTP {resp.status_code}: {_error_message(resp)}"
            if resp.status_code < 500 and resp.status_code != 429:
                raise RequestFailedError(err)
        if attempt >= retries:
            raise RequestFailedError(f"giving up after {attempt + 1} attempts: {err}")
        attempt += 1
        retry_log.append(err)
        time.sleep(backoff_s * 2 ** (attempt - 1))


def _clean_entries(entries: dict[str, float], k: int) -> list[tuple[str, float]]:
    """Sort by log-probability, drop zero-probability entries, keep the best ``k``."""
    ranked = sorted(entries.items(), key=lambda kv: -kv[1])
    return [(tok, lp) for tok, lp in ranked if lp > ZERO_PROB_LOGPROB][:k]


def parse_echo_choice(
    choice: dict[str, Any], chunk: np.ndarray, k: int
) -> tuple[list[list[tuple[str, float]]], list[float]]:
    """Parse a vLLM ``echo=true, logprobs=k`` completion into per-position top-k lists.

    ``top_logprobs[i]`` describes the distribution that predicts prompt token ``i`` given the
    tokens before it, so positions ``1..L-1`` are scored and position 0 is ignored.

    Args:
        choice: ``response["choices"][0]``.
        chunk: The prompt token ids that were sent.
        k: Number of entries to keep per position.

    Returns:
        ``(top, target_lp)`` with ``L - 1`` entries each.

    Raises:
        ValueError: If the echoed prompt tokens differ from ``chunk`` or logprobs are missing.
    """
    lp = choice.get("logprobs")
    if not lp:
        raise ValueError("response has no logprobs (is echo+logprobs supported by the server?)")
    tokens, tok_lp, tops = lp["tokens"], lp["token_logprobs"], lp["top_logprobs"]
    length = len(chunk)
    if len(tokens) < length:
        raise ValueError(f"server returned {len(tokens)} prompt tokens for {length} sent")
    for i, t in enumerate(chunk):
        if tokens[i].startswith("token_id:") and tokens[i] != f"token_id:{int(t)}":
            raise ValueError(
                f"prompt tokens differ from the ones sent at position {i}: "
                f"{tokens[i]} != token_id:{int(t)} (does the server add special tokens?)"
            )
    top = [_clean_entries(tops[i], k) for i in range(1, length)]
    target = [float(max(tok_lp[i], -9999.0)) for i in range(1, length)]
    return top, target


def parse_prompt_logprobs_choice(
    choice: dict[str, Any], chunk: np.ndarray, k: int
) -> tuple[list[list[tuple[str, float]]], list[float]]:
    """Parse a vLLM ``prompt_logprobs=k`` completion into per-position top-k lists.

    Args:
        choice: ``response["choices"][0]``.
        chunk: The prompt token ids that were sent.
        k: Number of entries to keep per position.

    Returns:
        ``(top, target_lp)`` with ``L - 1`` entries each.

    Raises:
        ValueError: If the actual prompt token is missing from a position's entries.
    """
    plp = choice.get("prompt_logprobs")
    if not plp or len(plp) < len(chunk):
        raise ValueError("response has no prompt_logprobs for every prompt token")
    top, target = [], []
    for i in range(1, len(chunk)):
        entries = {f"token_id:{tid}": float(v["logprob"]) for tid, v in plp[i].items()}
        actual = f"token_id:{int(chunk[i])}"
        if actual not in entries:
            raise ValueError(f"prompt tokens differ from the ones sent at position {i}")
        top.append(_clean_entries(entries, k))
        target.append(max(entries[actual], -9999.0))
    return top, target


def assemble(
    parsed: list[tuple[list[list[tuple[str, float]]], list[float]]], meta: dict[str, Any]
) -> FidelityData:
    """Stack per-chunk parse results into one :class:`FidelityData`.

    Args:
        parsed: One ``(top, target_lp)`` pair per chunk, in chunk order.
        meta: Run metadata to store.

    Returns:
        The assembled container with a per-file vocabulary.
    """
    vocab = sorted({tok for top, _ in parsed for pos in top for tok, _ in pos})
    index = {tok: i for i, tok in enumerate(vocab)}
    n_chunks, n_pos = len(parsed), len(parsed[0][1])
    width = max(len(pos) for top, _ in parsed for pos in top)
    ids = np.full((n_chunks, n_pos, width), -1, dtype=np.int32)
    lps = np.full((n_chunks, n_pos, width), -np.inf, dtype=np.float32)
    targets = np.empty((n_chunks, n_pos), dtype=np.float32)
    for c, (top, target) in enumerate(parsed):
        targets[c] = target
        for p, entries in enumerate(top):
            for j, (tok, lp) in enumerate(entries):
                ids[c, p, j] = index[tok]
                lps[c, p, j] = lp
    return FidelityData(np.array(vocab, dtype=str), ids, lps, targets, meta)


def _server_model_info(client: httpx.Client, base_url: str, model: str) -> dict[str, Any] | None:
    """Best-effort ``GET /models`` record of the served model (None when unavailable)."""
    try:
        resp = client.get(f"{base_url}/models")
        data = resp.json().get("data", [])
    except (httpx.HTTPError, ValueError, AttributeError):
        return None
    for entry in data:
        if entry.get("id") == model:
            return {k: entry[k] for k in ("id", "root", "max_model_len") if k in entry}
    return None


def collect(
    chunks_path: str | Path,
    base_url: str,
    model: str,
    out_path: str | Path,
    *,
    k: int = 20,
    mode: str = "echo",
    concurrency: int = 4,
    retries: int = 3,
    backoff_s: float = 2.0,
    timeout_s: float = 600.0,
    api_key: str | None = None,
) -> FidelityData:
    """Query ``/v1/completions`` for every chunk and store the top-k log-probs.

    Args:
        chunks_path: ``chunks.json`` written by :func:`prepare_chunks`.
        base_url: OpenAI-compatible base URL ending in ``/v1``.
        model: Model name sent in the request.
        out_path: Destination ``.npz``.
        k: Number of log-probs per position (vLLM caps it with ``--max-logprobs``).
        mode: ``"echo"`` (``echo=true, logprobs=k``) or ``"prompt_logprobs"`` (vLLM field).
        concurrency: Parallel requests.
        retries: Retries per request for transport errors, 429 and 5xx.
        backoff_s: First retry delay in seconds; doubles each retry.
        timeout_s: Per-request timeout.
        api_key: Bearer token (never written to the output).

    Returns:
        The collected data (also saved to ``out_path``).

    Raises:
        RequestFailedError: If a request fails for good.
        ValueError: If the server's response does not match the prompt that was sent.
    """
    if mode not in ("echo", "prompt_logprobs"):
        raise ValueError(f"unknown mode {mode!r}")
    chunks = load_chunks(chunks_path)
    base_url = base_url.rstrip("/")
    template: dict[str, Any] = {"model": model, "max_tokens": 1, "temperature": 0}
    if mode == "echo":
        template.update(echo=True, logprobs=k, return_tokens_as_token_ids=True)
    else:
        template.update(prompt_logprobs=k)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    parse = parse_echo_choice if mode == "echo" else parse_prompt_logprobs_choice
    retry_log: list[str] = []
    started = _now()
    lock = threading.Lock()
    abort = threading.Event()  # set by the first failing request so the others stop early

    with httpx.Client(timeout=timeout_s, headers=headers) as client:
        info = _server_model_info(client, base_url, model)

        def one(i: int) -> tuple[list[list[tuple[str, float]]], list[float]]:
            if abort.is_set():
                raise RequestFailedError("aborted after an earlier request failed")
            chunk = chunks.token_ids[i]
            payload = {**template, "prompt": [int(t) for t in chunk]}
            local_log: list[str] = []
            try:
                resp = _post_with_retries(
                    client, f"{base_url}/completions", payload, retries, backoff_s, local_log
                )
                usage = resp.get("usage", {})
                if usage.get("prompt_tokens") not in (None, len(chunk)):
                    raise ValueError(
                        f"server counted {usage['prompt_tokens']} prompt tokens "
                        f"for {len(chunk)} sent"
                    )
                return parse(resp["choices"][0], chunk, k)
            except Exception:
                abort.set()
                raise
            finally:
                with lock:
                    retry_log.extend(local_log)

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            parsed = list(pool.map(one, range(len(chunks.token_ids))))

    meta = {
        "tool": "fidelity.py collect",
        "endpoint": base_url,
        "model": model,
        "mode": mode,
        "k": k,
        "request": template,
        "concurrency": concurrency,
        "retries": retries,
        "request_retries": len(retry_log),
        "n_chunks": int(chunks.token_ids.shape[0]),
        "chunk_len": int(chunks.token_ids.shape[1]),
        "chunks_name": chunks.meta.get("name"),
        "chunks_sha256": chunks.sha256,
        "chunks_meta": chunks.meta,
        "server_model_info": info,
        "versions": _versions(),
        "started_utc": started,
        "collected_utc": _now(),
    }
    data = assemble(parsed, meta)
    save_fidelity(out_path, data)
    return data


# --------------------------------------------------------------------------- CLI


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="fidelity.py", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch-wikitext", help="download wikitext-2 raw (Salesforce/wikitext)")
    f.add_argument("--out", default="data/wikitext2_test.txt")
    f.add_argument("--split", default="test", choices=["test", "validation", "train"])

    p = sub.add_parser("prepare", help="tokenize a text file and cut it into chunks")
    p.add_argument("--text", required=True)
    p.add_argument("--tokenizer", required=True, help="tokenizer.json or a directory with one")
    p.add_argument("--out", required=True)
    p.add_argument("--n-chunks", type=int, default=64)
    p.add_argument("--chunk-len", type=int, default=512)
    p.add_argument("--name")
    p.add_argument("--add-special-tokens", action="store_true")

    c = sub.add_parser("collect", help="query an endpoint and store top-k log-probs")
    c.add_argument("--chunks", required=True)
    c.add_argument("--base-url", default="http://localhost:8000/v1")
    c.add_argument("--model", default="olala")
    c.add_argument("--out", required=True)
    c.add_argument("--k", type=int, default=20)
    c.add_argument("--mode", choices=["echo", "prompt_logprobs"], default="echo")
    c.add_argument("--concurrency", type=int, default=4)
    c.add_argument("--retries", type=int, default=3)
    c.add_argument("--backoff", type=float, default=2.0)
    c.add_argument("--timeout", type=float, default=600.0)
    c.add_argument("--api-key", default=None, help="default: $FIDELITY_API_KEY or $OPENAI_API_KEY")

    m = sub.add_parser("compare", help="compare a reference .npz with a variant .npz")
    m.add_argument("reference")
    m.add_argument("variant")
    m.add_argument("--max-k", type=int, default=None, help="truncate both top-k lists to k")
    m.add_argument("--n-boot", type=int, default=1000)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--json", default=None, help="also write the full result as JSON")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Arguments (default: ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    args = _build_parser().parse_args(argv)
    if args.command == "fetch-wikitext":
        record = fetch_wikitext(args.out, args.split)
        print(json.dumps(record, indent=2))
    elif args.command == "prepare":
        chunks = prepare_chunks(
            args.text,
            args.tokenizer,
            args.out,
            args.n_chunks,
            args.chunk_len,
            args.name,
            args.add_special_tokens,
        )
        print(f"wrote {args.out}: {chunks.token_ids.shape}, sha256 {chunks.sha256}")
    elif args.command == "collect":
        key = args.api_key or os.environ.get("FIDELITY_API_KEY") or os.environ.get("OPENAI_API_KEY")
        data = collect(
            args.chunks,
            args.base_url,
            args.model,
            args.out,
            k=args.k,
            mode=args.mode,
            concurrency=args.concurrency,
            retries=args.retries,
            backoff_s=args.backoff,
            timeout_s=args.timeout,
            api_key=key,
        )
        print(f"wrote {args.out}: {data.target_lp.shape} positions, k={data.top_ids.shape[-1]}")
    else:
        result = compare_fidelity(
            load_fidelity(args.reference),
            load_fidelity(args.variant),
            max_k=args.max_k,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        if args.json:
            Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
        sys.stdout.write(render_comparison_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
