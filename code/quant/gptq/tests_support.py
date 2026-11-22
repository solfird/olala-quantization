"""Helpers shared by the tests and ``tiny_e2e.py``: a calibration manifest built from toy documents."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import calib_data
import tiny_model

TASKS = ("mgsm", "humaneval_plus", "mbpp_plus", "ifeval", "aime2025")


def make_calibration(tmp: Path, n_windows: int = 16, seed: int = 0, name: str = "calib") -> Path:
    """Build an audited calibration manifest of ``n_windows`` byte-token windows from toy documents; return its path."""
    items = tmp / "items.json"
    items.write_text(
        json.dumps(
            {
                "schema": "eval-items/1",
                "tasks": {
                    t: {"items": [{"id": "1", "fields": {"input": "zzz qqq xxx vvv uuu ttt sss rrr ppp"}}]}
                    for t in TASKS
                },
            }
        )
    )
    out = tmp / name
    calib_data.build_calibration(
        [calib_data.SourceSpec("toy", tiny_model.word_docs(60, seed), share=1.0)],
        tokenizer=calib_data.ByteTokenizer(),
        out_dir=out,
        n_tokens=n_windows * tiny_model.TINY_WINDOW,
        window=tiny_model.TINY_WINDOW,
        seed=seed,
        eval_items=items,
    )
    return out / "manifest.json"


# ----------------------------------------------------------------------------------- CAL-1 layout (tiny, fake)

REAL_MANIFEST = Path(__file__).parent / "tests" / "fixtures" / "cal1_manifest_real.json"
_DOMAIN_SOURCES = {
    "math": "math_train",
    "multilingual_math": "mgsm_train_parallel",
    "code": "code_oss_python",
    "instruction": "if_personas",
}
CAL1_MARKER = tuple(range(3, 35))  # the "system prompt" every fake trace starts with (32 tokens, as the real marker)


@dataclass
class Cal1Layout:
    """A fake CAL-1 output directory plus the ground truth of which domain every window token comes from."""

    directory: Path
    manifest: Path
    labels: list[str]  # domain of every token of windows.json, in order
    window: int
    n_traces: dict[str, int]
    tokens: dict[str, int]  # rendered tokens per domain, EOS not counted (the manifest's ``n_tokens``)


def make_cal1_layout(
    tmp: Path,
    per_domain: dict[str, int] | None = None,
    *,
    window: int = 64,
    seed: int = 5,
    eos_id: int = 2,
    name: str = "calib-cal1",
) -> Cal1Layout:
    """Write a CAL-1 output directory in the layout of ``cal1_generate.py`` with a tiny fake windows file.

    The manifest starts from a verbatim copy of the real CAL-1 manifest (``tests/fixtures/cal1_manifest_real.json``)
    and only the fields that describe the fake windows are replaced. The windows are packed like
    ``cal1_generate.build_bundle`` does: traces in prompt order, shuffled with ``Random(f"{seed}|order")``, one EOS
    after each, cut into full windows (the tail is dropped).
    """
    per_domain = per_domain or {"math": 3, "multilingual_math": 3, "code": 4, "instruction": 3}
    rng = random.Random(f"fake-cal1|{seed}")
    out = tmp / name
    out.mkdir(parents=True)
    traces: list[tuple[str, str, list[int]]] = []  # (doc id, domain, tokens)
    for domain, n in per_domain.items():
        for _ in range(n):
            body = [rng.randint(40, 255) for _ in range(rng.randint(60, 140))]
            traces.append((f"{_DOMAIN_SOURCES[domain]}/default-{len(traces)}", domain, [*CAL1_MARKER, *body]))
    rng.shuffle(traces)  # the prompt order is not grouped by domain
    with (out / "calib_text.jsonl").open("w", encoding="utf-8") as f:
        for doc, _, _ in traces:
            f.write(json.dumps({"source": "prompt", "doc": doc, "text": "a prompt"}) + "\n")
            f.write(json.dumps({"source": "generation", "doc": doc, "text": "a trace"}) + "\n")
    order = list(range(len(traces)))
    random.Random(f"{seed}|order").shuffle(order)  # the rule of cal1_generate.build_bundle
    flat: list[int] = []
    labels: list[str] = []
    for i in order:
        _, domain, toks = traces[i]
        flat += [*toks, eos_id]
        labels += [domain] * (len(toks) + 1)
    n_win = len(flat) // window  # the tail (fewer than ``window`` tokens) is dropped, as pack_windows does
    flat, labels = flat[: n_win * window], labels[: n_win * window]
    (out / "windows.json").write_text(json.dumps([flat[i : i + window] for i in range(0, len(flat), window)]))
    sha = calib_data.la.sha256_file
    m = json.loads(REAL_MANIFEST.read_text())
    m.update(
        seed=seed,
        window=window,
        n_windows=n_win,
        n_tokens=len(flat),
        tokenizer={"name": "fake", "eos_id": eos_id},
        quotas=dict(per_domain),
    )
    m["prompt_sources"] = [{"name": s, "domain": d} for d, s in _DOMAIN_SOURCES.items()]
    n_tok = {d: sum(len(t) for _, dd, t in traces if dd == d) for d in per_domain}
    m["domains"] = {d: {"quota": n, "n_traces": n, "n_tokens": n_tok[d]} for d, n in per_domain.items()}
    m["counts"] = {**m["counts"], "n_traces": len(traces)}
    m["windows"] = {"file": "windows.json", "sha256": sha(out / "windows.json")}
    m["files"] = {"calib_text.jsonl": sha(out / "calib_text.jsonl"), "windows.json": sha(out / "windows.json")}
    (out / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")
    return Cal1Layout(out, out / "manifest.json", labels, window, dict(per_domain), n_tok)
