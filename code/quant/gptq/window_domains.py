"""Which domain of a CAL-1 set the token windows come from (the "windows used per domain" of the build report).

CAL-1 windows are made by ``cal1_generate.build_bundle``: the rendered traces, in a seeded shuffle of the prompt order,
one EOS id after each, cut into windows of ``window`` tokens (the tail is dropped). The windows do not align with the
traces, so a window usually holds tokens of several domains. The attribution here is exact and needs no tokenizer:

* every rendered trace starts with the same tokens (the chat template's system prompt), so the starts of the traces are
  the places where the first ``MARKER_TOKENS`` tokens of the first window occur again;
* the order of the traces is the shuffle of ``cal1_generate`` (``random.Random(f"{seed}|order")`` over the prompts of
  ``calib_text.jsonl``, which are written in that prompt order), the domain of a trace is the domain of the source of
  its prompt id (``prompt_sources`` of the manifest).

It is checked, not trusted: the marker must occur at most once per trace, no domain may get more tokens than the
manifest says it rendered, and when every window is used the tokens left over (the dropped tail) must be fewer than one
window. On the real CAL-1 windows it reproduces the manifest's per-domain token counts exactly (three domains equal,
instruction following short by the dropped tail). When the files or the checks do not allow it, the result is a labelled
estimate by token share (``method: token_share_estimate`` with the reason), never a silent guess.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch

import calib_data

MARKER_TOKENS = 32
TEXT_FILE = "calib_text.jsonl"
CAL1_SCHEMA = "cal1/1"


def _unavailable(reason: str) -> dict[str, Any]:
    return {"method": "unavailable", "reason": reason}


def _slots(domain: dict[str, Any]) -> int:
    """Tokens of a domain in the stream: its rendered tokens plus the EOS after each trace."""
    return int(domain["n_tokens"]) + int(domain["n_traces"])


def _estimate(m: dict[str, Any], n_windows: int, window: int, reason: str) -> dict[str, Any]:
    total = sum(_slots(d) for d in m["domains"].values())
    doms = {}
    for name, d in m["domains"].items():
        share = _slots(d) / total if total else 0.0
        doms[name] = {
            "n_traces": int(d["n_traces"]),
            "tokens": None,
            "token_share": share,
            "window_equivalents": share * n_windows,
            "windows_touched": None,
        }
    return {
        "method": "token_share_estimate",
        "reason": reason,
        "window": window,
        "n_windows_used": n_windows,
        "domains": doms,
    }


def windows_per_domain(manifest_path: Path, windows: torch.Tensor) -> dict[str, Any]:
    """Tokens, window equivalents and windows touched per domain for ``windows`` (all or a prefix of the manifest's).

    Returns:
        ``{"method": "exact" | "token_share_estimate" | "unavailable", "reason"?, "window", "n_windows_used",
        "domains": {name: {"n_traces", "tokens", "token_share", "window_equivalents", "windows_touched"}}}``.
        ``window_equivalents`` is tokens divided by the window length, ``windows_touched`` the number of windows that
        hold at least one token of the domain. A manifest that is not a CAL-1 manifest gives ``unavailable``.

    Raises:
        calib_data.ManifestError: If ``calib_text.jsonl`` exists but its sha256 differs from the manifest's.
    """
    manifest_path = calib_data.resolve_manifest(manifest_path)
    m = json.loads(manifest_path.read_text())
    if m.get("schema") != CAL1_SCHEMA or not m.get("domains"):
        return _unavailable(f"manifest schema {m.get('schema')!r} has no per-domain trace record (needs 'domains')")
    n_used, window = int(windows.shape[0]), int(windows.shape[1])
    text = manifest_path.parent / TEXT_FILE
    if not text.is_file():
        return _estimate(m, n_used, window, f"{TEXT_FILE} is not next to the manifest")
    want = (m.get("files") or {}).get(TEXT_FILE)
    if want is not None and calib_data.la.sha256_file(text) != want:
        raise calib_data.ManifestError(f"{text}: sha256 differs from the manifest")
    ids = [r["doc"] for r in map(json.loads, text.read_text(encoding="utf-8").splitlines()) if r["source"] == "prompt"]
    source_domain = {s["name"]: s["domain"] for s in m["prompt_sources"]}
    try:
        trace_domain = [source_domain[i.split("/", 1)[0]] for i in ids]
    except KeyError as e:
        return _estimate(m, n_used, window, f"a prompt id of {TEXT_FILE} names an unknown source {e}")
    if m.get("counts", {}).get("n_traces") != len(ids) or sum(d["n_traces"] for d in m["domains"].values()) != len(ids):
        return _estimate(
            m, n_used, window, f"n_traces of the manifest differs from the {len(ids)} prompts of {TEXT_FILE}"
        )
    flat = windows.reshape(-1).tolist()
    k = min(MARKER_TOKENS, window)
    marker = flat[:k]
    starts = [i for i in range(len(flat) - k + 1) if flat[i] == marker[0] and flat[i : i + k] == marker]
    if starts[:1] != [0] or len(starts) > len(ids):
        return _estimate(m, n_used, window, f"the trace-start marker occurs {len(starts)} times for {len(ids)} traces")
    order = list(range(len(ids)))
    random.Random(f"{m['seed']}|order").shuffle(order)
    got = {name: {"n_traces": 0, "tokens": 0, "touched": set()} for name in m["domains"]}
    for j, s in enumerate(starts):
        e = starts[j + 1] if j + 1 < len(starts) else len(flat)
        rec = got[trace_domain[order[j]]]
        rec["n_traces"] += 1
        rec["tokens"] += e - s
        rec["touched"].update(range(s // window, (e - 1) // window + 1))
    for name, d in m["domains"].items():
        if got[name]["tokens"] > _slots(d):
            return _estimate(m, n_used, window, f"domain {name!r} gets more tokens than the manifest rendered")
    if n_used == m["n_windows"]:
        left = sum(_slots(d) - got[name]["tokens"] for name, d in m["domains"].items())
        if not 0 <= left < window:
            return _estimate(
                m, n_used, window, f"{left} tokens of the manifest are not in the windows (a tail is < {window})"
            )
    total = n_used * window
    doms = {
        name: {
            "n_traces": r["n_traces"],
            "tokens": r["tokens"],
            "token_share": r["tokens"] / total,
            "window_equivalents": r["tokens"] / window,
            "windows_touched": len(r["touched"]),
        }
        for name, r in got.items()
    }
    return {"method": "exact", "window": window, "n_windows_used": n_used, "domains": doms}
