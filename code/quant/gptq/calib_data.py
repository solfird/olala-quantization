"""Build the calibration token set of G1-lite from PUBLIC text only, with a manifest and a leakage audit.

Two stages, because the Olala tokenizer lives with the model weights (the pod), not on the Mac:

* ``build --text-only`` (anywhere): collect candidate documents from the sources, pick a reproducible subset whose
  size is estimated with a conservative characters-per-token tokenizer, run the 8-gram leakage audit against the item
  texts of the five evaluation tasks, and write the text bundle ``calib_text.jsonl`` plus ``manifest.json``. It
  REFUSES to write anything when a selected document overlaps an evaluation item (``--drop-flagged`` removes the
  flagged documents and reselects instead, and lists them in the manifest).
* ``tokenize`` (needs the real tokenizer): re-audits the bundle, tokenizes it, packs it into windows
  (documents concatenated with the end-of-sequence token, no document masking: the model was trained that way,
  ``config.json`` ``intra_doc_masking: false``) and records the sha256 of ``windows.json`` in the manifest.

``build`` without ``--text-only`` does both in one go when a real tokenizer is available. ``build_g1lite.py`` reads only
the manifest: it refuses a manifest without a passing audit, without windows, or whose window file changed.

Sources (``--source``, repeatable)::

    wikitext2_train:SHARE                     wikitext-2 raw TRAIN split through ``datasets`` (the local HF cache)
    pystdlib:NAME:SHARE                       the Python standard library of the running interpreter (public, PSF)
    textdir:NAME:PATH:.suffix[,.suffix]:SHARE every matching file of a directory is one document
    textfile:NAME:PATH:SHARE                  one file, split into documents on blank-line runs
    hfdataset:NAME:REPO:CONFIG:SPLIT:FIELD:SHARE   a text column of a Hugging Face dataset (cache or network)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import sysconfig
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import torch

import leakage_audit as la

MANIFEST = "manifest.json"
TEXT_FILE = la.BUNDLE_TEXT
WINDOWS_FILE = "windows.json"
EVAL_ITEMS_COPY = "eval_items.json"
SCHEMA = "g1lite-calib/1"


class ManifestError(RuntimeError):
    """A calibration manifest is missing, unaudited, incomplete or inconsistent with its files."""


# ---------------------------------------------------------------------------------------------------- tokenizers


class Tokenizer(Protocol):
    """What the builder needs from a tokenizer."""

    eos_id: int
    name: str

    def encode(self, text: str) -> list[int]:
        """Token ids of ``text`` (no special tokens added)."""
        ...


class ByteTokenizer:
    """One token per UTF-8 byte; for tests."""

    eos_id = 256
    name = "byte"

    def encode(self, text: str) -> list[int]:
        """The bytes of ``text``."""
        return list(text.encode("utf-8"))


class CharsPerToken:
    """Size estimator for stage 1: ``ceil(len(text) / chars)`` dummy tokens (3.0 is conservative for BPE)."""

    eos_id = 0

    def __init__(self, chars: float = 3.0) -> None:
        """Set the assumed characters per token."""
        self.chars = chars
        self.name = f"chars:{chars}"

    def encode(self, text: str) -> list[int]:
        """Dummy ids, only their number matters."""
        return [0] * math.ceil(len(text) / self.chars)


class HFTokenizer:
    """The tokenizer of a model directory (``transformers.AutoTokenizer``), EOS from its ``config.json``."""

    def __init__(self, path: Path) -> None:
        """Load the tokenizer files in ``path``."""
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        cfg = json.loads((Path(path) / "config.json").read_text())
        eos = cfg.get("eos_token_id", self.tok.eos_token_id)
        self.eos_id = int(eos[0] if isinstance(eos, list) else eos)
        digest = hashlib.sha256()
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
            if (Path(path) / name).is_file():
                digest.update((Path(path) / name).read_bytes())
        self.name = f"hf:{Path(path).name}:{digest.hexdigest()[:16]}"

    def encode(self, text: str) -> list[int]:
        """Token ids without special tokens."""
        return list(self.tok.encode(text, add_special_tokens=False))


def make_tokenizer(spec: str) -> Tokenizer:
    """``byte``, ``chars:K`` or a model directory with tokenizer files."""
    if spec == "byte":
        return ByteTokenizer()
    if spec.startswith("chars:"):
        return CharsPerToken(float(spec.split(":", 1)[1]))
    return HFTokenizer(Path(spec))


# ---------------------------------------------------------------------------------------------------- documents


@dataclass(frozen=True)
class Document:
    """One piece of calibration text."""

    source: str
    doc_id: str
    text: str


@dataclass
class SourceSpec:
    """A source of documents and its share of the token budget."""

    name: str
    docs: Sequence[Document]
    share: float
    kind: str = "documents"
    origin: str = ""
    revision: str = ""
    license: str = ""


_WIKI_TITLE = re.compile(r"^ = [^=].* = \n?$")


def split_wikitext(rows: Sequence[str]) -> list[str]:
    """Group wikitext rows into articles: a new document starts at each level-1 heading `` = Title = ``."""
    docs: list[list[str]] = []
    for row in rows:
        if _WIKI_TITLE.match(row) or not docs:
            docs.append([])
        docs[-1].append(row)
    return ["".join(d) for d in docs if "".join(d).strip()]


def load_wikitext2_train() -> SourceSpec:
    """Wikitext-2 raw TRAIN split (``Salesforce/wikitext``, ``wikitext-2-raw-v1``) from the local HF cache."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    texts = split_wikitext(list(ds["text"]))
    docs = [Document("wikitext2_train", f"article-{i}", t) for i, t in enumerate(texts) if len(t) > 2000]
    return SourceSpec(
        "wikitext2_train",
        docs,
        1.0,
        "hf_dataset",
        "Salesforce/wikitext wikitext-2-raw-v1 train",
        _hf_revision("Salesforce/wikitext"),
        "CC-BY-SA-3.0",
    )


def _hf_revision(repo: str) -> str:
    """Revision of the cached snapshot of a dataset repo (best effort; empty when it cannot be told)."""
    snaps = Path.home() / ".cache/huggingface/hub" / f"datasets--{repo.replace('/', '--')}" / "snapshots"
    names = sorted(p.name for p in snaps.iterdir()) if snaps.is_dir() else []
    return names[0] if len(names) == 1 else ""


def load_text_dir(name: str, root: Path, suffixes: Sequence[str], min_chars: int = 1000) -> list[Document]:
    """Every file under ``root`` with one of ``suffixes`` and at least ``min_chars`` characters, as documents."""
    docs = []
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and path.suffix in suffixes:
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if len(text) >= min_chars:
                docs.append(Document(name, str(path.relative_to(root)), text))
    return docs


def load_python_stdlib(name: str = "python") -> SourceSpec:
    """The Python standard library of this interpreter (public, PSF licence), tests and GUI code left out."""
    root = Path(sysconfig.get_paths()["stdlib"])
    skip = {"test", "tests", "idlelib", "lib2to3", "site-packages", "turtledemo", "__pycache__"}
    docs = [d for d in load_text_dir(name, root, (".py",), 1500) if not skip & set(Path(d.doc_id).parts)]
    return SourceSpec(name, docs, 1.0, "text_dir", f"python stdlib {sysconfig.get_python_version()}", "", "PSF-2.0")


def load_text_file(name: str, path: Path, min_chars: int = 1500) -> list[Document]:
    """Split a text file into documents on runs of blank lines, merging paragraphs up to ``min_chars``."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", Path(path).read_text(encoding="utf-8")) if p.strip()]
    docs, cur = [], ""
    for p in paras:
        cur = f"{cur}\n\n{p}" if cur else p
        if len(cur) >= min_chars:
            docs.append(Document(name, f"{Path(path).name}:{len(docs)}", cur))
            cur = ""
    return docs


def split_long(docs: Sequence[Document], max_chars: int) -> list[Document]:
    """Cut documents longer than ``max_chars`` into pieces (at line ends where possible) named ``id#k``.

    One huge file would otherwise overshoot its source's token share by far; the pieces rejoin with ``"\\n"``.
    """
    out: list[Document] = []
    for d in docs:
        if len(d.text) <= max_chars:
            out.append(d)
            continue
        pieces: list[str] = []
        cur: list[str] = []
        size = 0
        for line in d.text.split("\n"):
            while len(line) > max_chars:  # a line longer than the limit is cut hard
                if cur:
                    pieces.append("\n".join(cur))
                    cur, size = [], 0
                pieces.append(line[:max_chars])
                line = line[max_chars:]
            extra = len(line) + (1 if cur else 0)
            if cur and size + extra > max_chars:
                pieces.append("\n".join(cur))
                cur, size, extra = [], 0, len(line)
            cur.append(line)
            size += extra
        if cur:
            pieces.append("\n".join(cur))
        out.extend(Document(d.source, f"{d.doc_id}#{i}", p) for i, p in enumerate(pieces))
    return out


def parse_source(spec: str, max_doc_chars: int = 6000) -> SourceSpec:
    """Build a :class:`SourceSpec` from a ``--source`` string (see the module docstring)."""
    kind, *rest = spec.split(":")
    try:
        if kind == "wikitext2_train":
            src = load_wikitext2_train()
            src.share = float(rest[0])
        elif kind == "pystdlib":
            src = load_python_stdlib(rest[0])
            src.share = float(rest[1])
        elif kind == "textdir":
            name, path, suffixes, share = rest
            docs = load_text_dir(name, Path(path), tuple(suffixes.split(",")))
            src = SourceSpec(name, docs, float(share), "text_dir", path)
        elif kind == "textfile":
            name, path, share = rest
            src = SourceSpec(name, load_text_file(name, Path(path)), float(share), "text_file", path)
        elif kind == "hfdataset":
            name, repo, config, split, column, share = rest
            from datasets import load_dataset

            ds = load_dataset(repo, config, split=split)
            docs = [Document(name, f"row-{i}", t) for i, t in enumerate(ds[column]) if len(t) >= 1000]
            src = SourceSpec(name, docs, float(share), "hf_dataset", f"{repo} {config} {split}", _hf_revision(repo))
        else:
            raise ValueError(f"unknown source kind {kind!r}")
    except (ValueError, IndexError) as e:
        raise ValueError(f"cannot parse --source {spec!r}: {e}") from e
    src.docs = split_long(src.docs, max_doc_chars)
    return src


# ---------------------------------------------------------------------------------------------------- selection


@dataclass
class Selection:
    """Selected documents (shuffled, in packing order) and the per-source accounting."""

    docs: list[Document]
    sources: list[dict[str, Any]] = field(default_factory=list)


def select_documents(
    sources: Sequence[SourceSpec],
    tokenizer: Tokenizer,
    n_tokens: int,
    seed: int,
    exclude: frozenset[str] = frozenset(),
) -> Selection:
    """Pick, per source, shuffled documents until its share of ``n_tokens`` is reached (one separator per document).

    The result depends only on the sources, the budget and the seed. The last document of a source may overshoot its
    share (documents are not cut). Documents named in ``exclude`` (``source/doc_id``) are never taken.

    Raises:
        ValueError: For a non-positive share, or a source without enough text for its share.
    """
    if any(s.share <= 0 for s in sources):
        raise ValueError("every source share must be positive")
    total = sum(s.share for s in sources)
    chosen: list[Document] = []
    accounting = []
    for s in sources:
        budget = math.ceil(n_tokens * s.share / total)
        order = [d for d in s.docs if f"{d.source}/{d.doc_id}" not in exclude]
        random.Random(f"{seed}|{s.name}").shuffle(order)
        taken, count, chars = [], 0, 0
        for d in order:
            if count >= budget:
                break
            taken.append(d)
            count += len(tokenizer.encode(d.text)) + 1
            chars += len(d.text)
        if count < budget:
            raise ValueError(f"not enough text in source {s.name!r}: {count} tokens available, {budget} needed")
        chosen.extend(taken)
        digest = hashlib.sha256("\n".join(d.text for d in taken).encode()).hexdigest()
        accounting.append(
            {
                "name": s.name,
                "kind": s.kind,
                "origin": s.origin,
                "revision": s.revision,
                "license": s.license,
                "share": s.share / total,
                "n_docs": len(taken),
                "n_chars": chars,
                "n_tokens": count,
                "sha256": digest,
            }
        )
    random.Random(f"{seed}|order").shuffle(chosen)
    return Selection(chosen, accounting)


def pack_windows(
    token_lists: Sequence[Sequence[int]], eos_id: int, window: int, max_windows: int | None = None
) -> torch.Tensor:
    """Concatenate documents with ``eos_id`` after each, cut into full windows of ``window`` tokens (tail dropped).

    Raises:
        ValueError: If there are not enough tokens for one window.
    """
    flat: list[int] = []
    for toks in token_lists:
        flat.extend(toks)
        flat.append(eos_id)
    n = len(flat) // window
    if max_windows is not None:
        n = min(n, max_windows)
    if n == 0:
        raise ValueError(f"{len(flat)} tokens are not enough for one window of {window}")
    return torch.tensor(flat[: n * window], dtype=torch.int64).reshape(n, window)


# ---------------------------------------------------------------------------------------------------- build


def _write_text(path: Path, docs: Sequence[Document]) -> str:
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps({"source": d.source, "doc": d.doc_id, "text": d.text}, ensure_ascii=False) + "\n")
    return la.sha256_file(path)


def _read_text(path: Path) -> list[Document]:
    docs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            docs.append(Document(r["source"], r["doc"], r["text"]))
    return docs


def _audit(docs: Sequence[Document], eval_items: Path, n: int) -> la.AuditReport:
    items = la.load_eval_items(eval_items)
    missing = [t for t in la.REQUIRED_TASKS if t not in {i.task for i in items}]
    if missing:
        raise ManifestError(f"{eval_items} lacks the evaluation tasks {missing}")
    return la.audit_documents([(f"{d.source}/{d.doc_id}", d.text) for d in docs], la.build_index(items, n))


def build_calibration(
    sources: Sequence[SourceSpec],
    *,
    tokenizer: Tokenizer,
    out_dir: Path,
    n_tokens: int,
    window: int,
    seed: int,
    eval_items: Path,
    ngram: int = 8,
    drop_flagged: bool = False,
    text_only: bool = False,
) -> dict[str, Any]:
    """Select, audit and write a calibration set; return the manifest.

    Raises:
        LeakageError: If a selected document overlaps an evaluation item and ``drop_flagged`` is False.
        ManifestError: If the evaluation items lack one of the five tasks.
    """
    excluded: set[str] = set()
    dropped: list[str] = []
    for _attempt in range(10):
        sel = select_documents(sources, tokenizer, n_tokens, seed, frozenset(excluded))
        report = _audit(sel.docs, eval_items, ngram)
        if report.ok:
            break
        if not drop_flagged:
            la.raise_if_leaky(report)
        excluded.update(report.flagged_docs)
        dropped.extend(report.flagged_docs)
    else:
        raise la.LeakageError("still overlapping after 10 rounds of dropping flagged documents")
    out_dir.mkdir(parents=True, exist_ok=True)
    text_sha = _write_text(out_dir / TEXT_FILE, sel.docs)
    (out_dir / EVAL_ITEMS_COPY).write_bytes(Path(eval_items).read_bytes())
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": seed,
        "n_tokens": n_tokens,
        "window": window,
        "n_windows": n_tokens // window,
        "tokenizer": {"name": tokenizer.name, "eos_id": tokenizer.eos_id},
        "sources": sel.sources,
        "text": {"file": TEXT_FILE, "sha256": text_sha, "n_docs": len(sel.docs)},
        "windows": None,
        "audit": {
            "ok": True,
            "n": ngram,
            "n_hits": 0,
            "n_docs": report.n_docs,
            "n_eval_fields": report.n_eval_fields,
            "n_skipped_short": report.n_skipped_short,
            "n_eval_ngrams": report.n_ngrams,
            "eval_items_file": EVAL_ITEMS_COPY,
            "eval_items_sha256": la.sha256_file(Path(eval_items)),
            "dropped_docs": dropped,
        },
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    if not text_only:
        manifest = tokenize_bundle(out_dir, tokenizer, n_tokens, window)
    return manifest


def tokenize_bundle(out_dir: Path, tokenizer: Tokenizer, n_tokens: int, window: int) -> dict[str, Any]:
    """Stage 2: re-audit the text bundle, tokenize and pack it, record ``windows.json`` in the manifest.

    Raises:
        ManifestError: If the bundle changed since it was audited, or holds too few tokens.
        LeakageError: If the re-audit finds an overlap.
    """
    mpath = out_dir / MANIFEST
    manifest = json.loads(mpath.read_text())
    if la.sha256_file(out_dir / TEXT_FILE) != manifest["text"]["sha256"]:
        raise ManifestError("calib_text.jsonl changed since it was audited (sha256 differs)")
    items = out_dir / manifest["audit"]["eval_items_file"]
    if la.sha256_file(items) != manifest["audit"]["eval_items_sha256"]:
        raise ManifestError("the evaluation-item file of the bundle changed (sha256 differs)")
    docs = _read_text(out_dir / TEXT_FILE)
    la.raise_if_leaky(_audit(docs, items, manifest["audit"]["n"]))
    token_lists = [tokenizer.encode(d.text) for d in docs]
    avail = sum(len(t) + 1 for t in token_lists)
    if avail < n_tokens:
        raise ManifestError(
            f"the bundle holds {avail} tokens for this tokenizer, {n_tokens} needed: select again with a lower "
            "--tokenizer chars:K estimate or more text"
        )
    windows = pack_windows(token_lists, tokenizer.eos_id, window, max_windows=n_tokens // window)
    (out_dir / WINDOWS_FILE).write_text(json.dumps(windows.tolist()))
    manifest["tokenizer"] = {"name": tokenizer.name, "eos_id": tokenizer.eos_id}
    manifest["n_windows"], manifest["window"] = int(windows.shape[0]), window
    manifest["n_tokens"] = int(windows.numel())
    manifest["windows"] = {"file": WINDOWS_FILE, "sha256": la.sha256_file(out_dir / WINDOWS_FILE)}
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def resolve_manifest(path: Path) -> Path:
    """The manifest file for ``path``: the file itself, or ``manifest.json`` of a calibration output directory.

    Raises:
        ManifestError: If ``path`` is a directory without a ``manifest.json``, or does not exist.
    """
    path = Path(path)
    if path.is_dir():
        path = path / MANIFEST
    if not path.is_file():
        raise ManifestError(f"{path}: no calibration manifest (a manifest.json, or the directory that holds one)")
    return path


def read_windows(manifest_path: Path) -> torch.Tensor:
    """Read the token windows of a manifest, refusing anything that is not audited and intact.

    ``manifest_path`` is a ``manifest.json`` or the directory holding it (the lite bundle and the CAL-1 output of
    ``cal1_generate.py`` both qualify: ``audit.ok``, ``audit.n_hits == 0``, ``windows``, ``n_windows``, ``window``).

    Raises:
        ManifestError: If the audit is missing or not ok, the windows are missing, or a file hash differs.
    """
    manifest_path = resolve_manifest(manifest_path)
    m = json.loads(manifest_path.read_text())
    audit = m.get("audit")
    if not audit or audit.get("ok") is not True or audit.get("n_hits", 1) != 0:
        raise ManifestError(f"{manifest_path}: no passing leakage audit (audit missing, failed, or with hits)")
    if not m.get("windows"):
        raise ManifestError(f"{manifest_path}: no token windows yet (run `calib_data.py tokenize`)")
    wpath = manifest_path.parent / m["windows"]["file"]
    if la.sha256_file(wpath) != m["windows"]["sha256"]:
        raise ManifestError(f"{wpath}: sha256 differs from the manifest")
    windows = torch.tensor(json.loads(wpath.read_text()), dtype=torch.int64)
    if windows.shape != (m["n_windows"], m["window"]):
        raise ManifestError(f"{wpath}: shape {tuple(windows.shape)} differs from the manifest")
    return windows


# ---------------------------------------------------------------------------------------------------- CLI


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="calib_data.py", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="select, audit and write the calibration set")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--tokenizer", default="chars:3.0", help="byte | chars:K | a model directory")
    b.add_argument("--n-tokens", type=int, default=65536)
    b.add_argument("--window", type=int, default=2048)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--eval-items", type=Path, required=True)
    b.add_argument("--source", action="append", required=True, help="see the module docstring")
    b.add_argument("--n", type=int, default=8, help="n-gram length of the audit")
    b.add_argument("--max-doc-chars", type=int, default=6000, help="longer documents are split into pieces")
    b.add_argument("--drop-flagged", action="store_true", help="drop flagged documents instead of refusing")
    b.add_argument("--text-only", action="store_true", help="stage 1 only: no tokenization")
    t = sub.add_parser("tokenize", help="stage 2: tokenize an audited text bundle into windows")
    t.add_argument("--out", type=Path, required=True, help="the bundle directory")
    t.add_argument("--tokenizer", required=True)
    t.add_argument("--n-tokens", type=int, default=65536)
    t.add_argument("--window", type=int, default=2048)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI. Returns 0 on success, 1 on a leakage refusal, 2 on any other refusal."""
    args = _parse_args(argv)
    try:
        tok = make_tokenizer(args.tokenizer)
        if args.cmd == "build":
            manifest = build_calibration(
                [parse_source(s, args.max_doc_chars) for s in args.source],
                tokenizer=tok,
                out_dir=args.out,
                n_tokens=args.n_tokens,
                window=args.window,
                seed=args.seed,
                eval_items=args.eval_items,
                ngram=args.n,
                drop_flagged=args.drop_flagged,
                text_only=args.text_only,
            )
        else:
            manifest = tokenize_bundle(args.out, tok, args.n_tokens, args.window)
    except la.LeakageError as e:
        print(f"REFUSED (leakage): {e}", file=sys.stderr)
        return 1
    except (ManifestError, ValueError, OSError, ImportError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(
        f"wrote {args.out / MANIFEST}: {manifest['n_tokens']} tokens, windows={'yes' if manifest['windows'] else 'no'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
