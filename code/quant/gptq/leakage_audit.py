"""8-gram leakage audit of calibration text against the item texts of the five evaluation tasks.

The item texts come from a JSON file written by ``extract_eval_items.py`` from the harness's own task factories
(MGSM, HumanEval+, MBPP+, IFEval, AIME 2025). Every string field of every item (prompt, target, canonical solution,
tests...) with at least ``min_words`` words is indexed by its ``n``-grams of normalised words (lower case, punctuation
stripped, split on whitespace). A calibration document that shares ANY n-gram with an item is a hit, and any hit makes
the audit fail: the caller refuses to proceed (PROTOCOL section 5, invalid-result condition 5).

Fields shorter than ``min_words`` words (a gold answer such as ``42``) cannot be audited by n-grams; they are counted in
``n_skipped_short`` and reported, not silently ignored.

Usage::

    python leakage_audit.py --items eval_items.json --bundle CALIB_DIR [--n 8] [--report leakage_audit.json]

Exit status: 0 clean, 1 overlap found, 2 unusable input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

ITEMS_SCHEMA = "eval-items/1"
REQUIRED_TASKS = ("mgsm", "humaneval_plus", "mbpp_plus", "ifeval", "aime2025")
BUNDLE_TEXT = "calib_text.jsonl"
_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)


class LeakageError(RuntimeError):
    """Calibration text overlaps an evaluation item."""


@dataclass(frozen=True)
class EvalItem:
    """One string field of one evaluation item."""

    task: str
    item_id: str
    field: str
    text: str


@dataclass(frozen=True)
class Hit:
    """An n-gram of a calibration document that also occurs in an evaluation item."""

    doc: str
    task: str
    item_id: str
    field: str
    ngram: str


@dataclass
class NgramIndex:
    """Map from n-gram (tuple of words) to the ``(task, item_id, field)`` triples that contain it."""

    n: int
    grams: dict[tuple[str, ...], list[tuple[str, str, str]]] = field(default_factory=dict)
    n_items_indexed: int = 0
    n_skipped_short: int = 0


@dataclass
class AuditReport:
    """Result of :func:`audit_documents`."""

    n: int
    n_docs: int
    n_eval_fields: int
    n_skipped_short: int
    n_ngrams: int
    hits: list[Hit]
    by_task: dict[str, int]
    flagged_docs: list[str]

    @property
    def ok(self) -> bool:
        """True when no document shares an n-gram with an evaluation item."""
        return not self.hits

    def as_dict(self) -> dict:
        """JSON-serialisable view (at most 200 hits are listed)."""
        return {
            "ok": self.ok,
            "n": self.n,
            "n_docs": self.n_docs,
            "n_eval_fields": self.n_eval_fields,
            "n_skipped_short": self.n_skipped_short,
            "n_ngrams": self.n_ngrams,
            "n_hits": len(self.hits),
            "by_task": self.by_task,
            "flagged_docs": self.flagged_docs,
            "hits": [vars(h) for h in self.hits[:200]],
        }


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text: str) -> list[str]:
    """Lower-case, replace punctuation (anything but letters and digits) by spaces, split on whitespace."""
    return _NON_WORD.sub(" ", text.lower()).split()


def load_eval_items(path: Path) -> list[EvalItem]:
    """Read the file written by ``extract_eval_items.py``.

    Raises:
        ValueError: For another schema, or when no item field is found at all.
    """
    data = json.loads(Path(path).read_text())
    if data.get("schema") != ITEMS_SCHEMA:
        raise ValueError(f"{path}: schema {data.get('schema')!r}, expected {ITEMS_SCHEMA!r}")
    items = [
        EvalItem(task, str(it["id"]), fname, text)
        for task, body in data["tasks"].items()
        for it in body["items"]
        for fname, text in it["fields"].items()
    ]
    if not items:
        raise ValueError(f"{path}: no evaluation item text found")
    return items


def build_index(items: Iterable[EvalItem], n: int = 8, min_words: int = 8) -> NgramIndex:
    """Index the ``n``-grams of every item field with at least ``min_words`` words (and at least ``n``)."""
    index = NgramIndex(n=n)
    threshold = max(min_words, n)
    for it in items:
        words = normalize(it.text)
        if len(words) < threshold:
            index.n_skipped_short += 1
            continue
        index.n_items_indexed += 1
        for i in range(len(words) - n + 1):
            index.grams.setdefault(tuple(words[i : i + n]), []).append((it.task, it.item_id, it.field))
    return index


def scan_text(doc: str, text: str, index: NgramIndex) -> list[Hit]:
    """All distinct n-grams of ``text`` found in ``index``, one :class:`Hit` per (n-gram, item field)."""
    words = normalize(text)
    n = index.n
    seen: set[tuple[tuple[str, ...], tuple[str, str, str]]] = set()
    hits: list[Hit] = []
    for i in range(len(words) - n + 1):
        gram = tuple(words[i : i + n])
        for owner in index.grams.get(gram, ()):
            if (gram, owner) not in seen:
                seen.add((gram, owner))
                hits.append(Hit(doc, owner[0], owner[1], owner[2], " ".join(gram)))
    return hits


def audit_documents(docs: Iterable[tuple[str, str]], index: NgramIndex) -> AuditReport:
    """Audit ``(name, text)`` documents against ``index``."""
    hits: list[Hit] = []
    n_docs = 0
    for name, text in docs:
        n_docs += 1
        hits.extend(scan_text(name, text, index))
    by_task = dict(Counter(h.task for h in hits))
    return AuditReport(
        n=index.n,
        n_docs=n_docs,
        n_eval_fields=index.n_items_indexed + index.n_skipped_short,
        n_skipped_short=index.n_skipped_short,
        n_ngrams=len(index.grams),
        hits=hits,
        by_task=by_task,
        flagged_docs=sorted({h.doc for h in hits}),
    )


def read_bundle(bundle: Path) -> list[tuple[str, str]]:
    """Read ``calib_text.jsonl`` of a calibration bundle as ``(source/doc, text)`` pairs."""
    out = []
    for line in (bundle / BUNDLE_TEXT).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out.append((f"{row['source']}/{row['doc']}", row["text"]))
    return out


def raise_if_leaky(report: AuditReport) -> None:
    """Raise :class:`LeakageError` naming the flagged documents and tasks if the report has any hit."""
    if report.ok:
        return
    shown = ", ".join(report.flagged_docs[:10])
    raise LeakageError(
        f"{len(report.hits)} overlapping {report.n}-gram(s) with evaluation items {report.by_task}; "
        f"flagged documents: {shown}"
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="leakage_audit.py", description="8-gram leakage audit of calibration text.")
    p.add_argument("--items", type=Path, required=True, help="eval_items.json from extract_eval_items.py")
    p.add_argument("--bundle", type=Path, required=True, help="calibration directory with calib_text.jsonl")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--min-words", type=int, default=8)
    p.add_argument("--report", type=Path, help="write the audit report here")
    p.add_argument(
        "--allow-missing-tasks", action="store_true", help=f"do not require all of {', '.join(REQUIRED_TASKS)}"
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI. Returns 0 when clean, 1 on overlap, 2 on unusable input (including a missing task)."""
    args = _parse_args(argv)
    try:
        items = load_eval_items(args.items)
        docs = read_bundle(args.bundle)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    present = {i.task for i in items}
    missing = [t for t in REQUIRED_TASKS if t not in present]
    if missing and not args.allow_missing_tasks:
        print(f"error: evaluation tasks missing from {args.items}: {missing}", file=sys.stderr)
        return 2
    report = audit_documents(docs, build_index(items, args.n, args.min_words))
    out = report.as_dict() | {
        "created_utc": datetime.now(UTC).isoformat(),
        "eval_items_sha256": sha256_file(args.items),
        "tasks": sorted(present),
    }
    if args.report:
        args.report.write_text(json.dumps(out, indent=2) + "\n")
    if not report.ok:
        print(f"FAILED: {len(report.hits)} overlapping {args.n}-gram(s) {report.by_task}", file=sys.stderr)
        for h in report.hits[:20]:
            print(f"  {h.doc} ~ {h.task}/{h.item_id} [{h.field}]: {h.ngram!r}", file=sys.stderr)
        return 1
    print(f"OK: {report.n_docs} documents, {report.n_ngrams} eval {args.n}-grams, no overlap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
