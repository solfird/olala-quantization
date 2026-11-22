"""Mutation check of the CAL-1 generator guards: weaken each guard in a throw-away copy, confirm the tests fail.

Usage: python tools/mutation_check_cal1.py [--out results/mutation_cal1_report.json] [--only SUBSTRING]

Same method as ``tools/mutation_check.py``: each mutation replaces ``old`` by ``new`` in ONE file of a temporary copy
of ``quant/gptq`` (the script stops when ``old`` is not found exactly once, so a mistyped pattern cannot yield a
meaningless 'survivor'); pytest runs the CAL-1 test files in the copy; 'caught' means pytest failed or timed out. The
real files are never touched. An unmutated baseline copy must pass first. Exit status 1 when a mutation survives.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

GPTQ = Path(__file__).resolve().parents[1]
TIMEOUT_S = 300
TESTS = ["tests/test_cal1_generate.py", "tests/test_pod_cal1.py"]
TEMPLATE = GPTQ.parents[1] / "recon" / "src" / "hf" / "chat_template.jinja"
SRC = "cal1_generate.py"


@dataclass(frozen=True)
class Mutation:
    """Replace ``old`` by ``new`` in ``path`` (relative to ``quant/gptq``)."""

    name: str
    old: str
    new: str
    path: str = SRC


MUTATIONS = [
    # ---- the leakage audit
    Mutation(
        "audit skipped over the generated traces",
        "    report = la.audit_documents(docs_of(traces), index)\n    dropped:",
        "    report = la.audit_documents([], index)\n    dropped:",
    ),
    Mutation(
        "overlap tolerated (refusal replaced by a pass)",
        "        if not drop_flagged:\n            la.raise_if_leaky(report)",
        "        if not drop_flagged:\n            pass",
    ),
    Mutation(
        "up to 3 overlapping 8-grams tolerated",
        "    if not report.ok:\n        if not drop_flagged:",
        "    if len(report.hits) > 3:\n        if not drop_flagged:",
    ),
    Mutation("audit n-gram length 13 instead of 8", "AUDIT_N = 8", "AUDIT_N = 13"),
    Mutation("prompts left out of the final audit", 'd.append((f"prompt/{p.id}", p.text))', "pass"),
    Mutation(
        "final answer left out of the audited generated text",
        'd.append((f"generation/{p.id}", r["reasoning"] + "\\n" + r["answer"]))',
        'd.append((f"generation/{p.id}", r["reasoning"]))',
    ),
    Mutation("prompt audit skipped at draw time", "hits = la.scan_text(pid, row.text, index)", "hits = []"),
    Mutation("stale manifest and windows kept on a refusal", "(out / name).unlink(missing_ok=True)", "pass"),
    Mutation(
        "verify does not re-run the audit",
        "    la.raise_if_leaky(la.audit_documents(la.read_bundle(out), load_index(out / EVAL_ITEMS_COPY)))",
        "    pass",
    ),
    Mutation(
        "flagged traces not dropped under --drop-flagged",
        "        traces = [t for t in traces if t[0].id not in flagged]",
        "        pass",
    ),
    # ---- resume, duplicates, failures, truncation
    Mutation("duplicate ok rows allowed", "            if pid in self.ok_ids:", "            if False:"),
    Mutation(
        "done prompts asked again on resume",
        "p.id not in store.ok_ids and store.failures[p.id] < max_attempts]",
        "store.failures[p.id] < max_attempts]",
    ),
    Mutation(
        "failed prompts retried without limit",
        "p.id not in store.ok_ids and store.failures[p.id] < max_attempts]",
        "p.id not in store.ok_ids and store.failures[p.id] < 10**9]",
    ),
    Mutation(
        "failures not written to the store",
        "        store.append(row)\n        with counter:",
        '        if row["status"] == "ok":\n            store.append(row)\n        with counter:',
    ),
    Mutation("truncation not flagged", '"truncated": finish in TRUNCATION_FINISH,', '"truncated": False,'),
    Mutation(
        "truncated traces dropped as failures",
        'if finish != "stop" and finish not in TRUNCATION_FINISH:',
        'if finish != "stop":',
    ),
    Mutation("cut last line of the store not repaired", "f.truncate(good)", "pass"),
    Mutation("rows of unknown prompts accepted on resume", '        if r["id"] not in by_id:', "        if False:"),
    Mutation(
        "rows of another configuration accepted on resume",
        '        if r.get("config_sha256") != config_sha:',
        "        if False:",
    ),
    Mutation(
        "changed prompt text accepted on resume",
        'if r["id"] == p.id and r.get("prompt_sha256") != _prompt_sha(p):',
        "if False:",
    ),
    Mutation(
        "unfinished generation accepted by the build",
        '    if todo:\n        raise Cal1Error(\n            f"generation is incomplete',
        '    if False:\n        raise Cal1Error(\n            f"generation is incomplete',
    ),
    Mutation(
        "a domain without any trace accepted",
        "        if quota > 0 and not any(p.domain == domain for p, _ in traces):",
        "        if False:",
    ),
    Mutation(
        "sampling change accepted on resume", "            if stored[key] != current[key]:", "            if False:"
    ),
    Mutation(
        "server provenance change accepted on resume",
        '        if args.server_provenance and stored["server_provenance"] != args.server_provenance:',
        "        if False:",
    ),
    Mutation(
        "provenance not required for a real run",
        "    if not (args.base_url and model and args.server_provenance):",
        "    if False:",
    ),
    Mutation(
        "prompts drawn with other quotas/seed accepted on resume",
        '        if pm["quotas"] != quotas or pm["seed"] != args.seed:',
        "        if False:",
    ),
    # ---- concurrency
    Mutation(
        "concurrency unbounded",
        "with ThreadPoolExecutor(max_workers=concurrency) as pool:",
        "with ThreadPoolExecutor(max_workers=len(todo) or 1) as pool:",
    ),
    Mutation("concurrency not validated", "    if not 1 <= concurrency <= MAX_CONCURRENCY:", "    if False:"),
    # ---- the request
    Mutation("top_k not sent", '        "top_k": s.top_k,\n', ""),
    Mutation(
        "chat_template_kwargs not sent",
        '        "chat_template_kwargs": {"reasoning_effort": s.reasoning_effort},\n',
        "",
    ),
    Mutation("temperature not sent", '        "temperature": s.temperature,\n', ""),
    Mutation("max_tokens not sent", '        "max_tokens": s.max_tokens,\n', ""),
    Mutation(
        "older reasoning field name ignored",
        'message.get("reasoning") or message.get("reasoning_content") or ""',
        'message.get("reasoning") or ""',
    ),
    # ---- the prompt draw and manifest
    Mutation(
        "quotas ignored (whole pool taken)",
        "                if len(taken) >= want:",
        "                if len(taken) >= len(pool):",
    ),
    Mutation(
        "duplicate prompts allowed",
        "if len(row.text.strip()) < MIN_PROMPT_CHARS or key in seen:",
        "if len(row.text.strip()) < MIN_PROMPT_CHARS:",
    ),
    Mutation(
        "seed ignored in the draw",
        'random.Random(f"{seed}|{src.name}").shuffle(order)',
        'random.Random(f"{src.name}").shuffle(order)',
    ),
    Mutation("dataset revision not pinned", "    if not revision or not HEX40.match(revision):", "    if False:"),
    Mutation(
        "revision not passed to datasets",
        "load_dataset(src.repo, cfg, split=src.split, revision=rev)",
        "load_dataset(src.repo, cfg, split=src.split)",
    ),
    Mutation("prompt-file hash not checked", '    if la.sha256_file(path) != m["prompts"]["sha256"]:', "    if False:"),
    Mutation(
        "declared quotas not checked against the prompts",
        "        if counts.get(domain, 0) != quota:",
        "        if False:",
    ),
    # ---- the bundle
    Mutation(
        "file hashes not checked by verify",
        "        if not path.is_file() or la.sha256_file(path) != sha:",
        "        if not path.is_file():",
    ),
    Mutation(
        "truncated count missing in the manifest",
        '"n_truncated": sum(1 for _, r in traces if r["truncated"]),',
        '"n_truncated": 0,',
    ),
    # ---- the chat template
    Mutation(
        "reasoning effort not given to the template", "tokenize=False, reasoning_effort=effort)", "tokenize=False)"
    ),
    Mutation(
        "reasoning left out of the rendered assistant turn",
        '{"role": "assistant", "content": answer, "reasoning_content": reasoning},',
        '{"role": "assistant", "content": answer},',
    ),
    # ---- the pod wrapper
    Mutation(
        "pod wrapper does not require the provenance", '[ -n "${CAL1_PROVENANCE:-}" ] ||', "true ||", "pod_cal1.sh"
    ),
    Mutation("pod wrapper plan is not a dry run", '$(common) --dry-run "$@"', '$(common) "$@"', "pod_cal1.sh"),
]


def apply(text: str, m: Mutation) -> str:
    """Return ``text`` with the mutation applied; stop if the pattern is not found exactly once."""
    n = text.count(m.old)
    if n != 1:
        raise SystemExit(f"mutation {m.name!r}: pattern found {n} times in {m.path}, expected exactly 1")
    return text.replace(m.old, m.new)


def run_one(m: Mutation, baseline: bool = False) -> dict:
    """Apply one mutation in a temporary copy and run the CAL-1 tests there."""
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "gptq"
        shutil.copytree(
            GPTQ,
            root,
            ignore=shutil.ignore_patterns(".venv", "data", "results", "__pycache__", ".pytest_cache", ".ruff_cache"),
        )
        target = root / m.path
        if not baseline:
            target.write_text(apply(target.read_text(), m))
        env = {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PATH": "/usr/bin:/bin",
            "HOME": str(Path.home()),
            "OLALA_CHAT_TEMPLATE": str(TEMPLATE),
        }
        try:
            proc = subprocess.run(
                [str(GPTQ / ".venv" / "bin" / "python"), "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *TESTS],
                cwd=root,
                timeout=TIMEOUT_S,
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
            code, tail = proc.returncode, (proc.stdout.strip().splitlines() or [""])[-1]
            failed = [ln for ln in proc.stdout.splitlines() if ln.startswith("FAILED")][:1]
        except subprocess.TimeoutExpired:
            code, tail, failed = 124, "timeout", []
    return {
        "passed": code == 0,
        "tail": tail,
        "name": m.name,
        "file": m.path,
        "caught": code != 0,
        "caught_by": failed[0].removeprefix("FAILED ").split(" - ")[0]
        if failed
        else ("timeout" if code == 124 else tail),
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    """Run the baseline and every mutation; exit 1 if any survives."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, default=GPTQ / "results" / "mutation_cal1_report.json")
    p.add_argument("--only", default="")
    args = p.parse_args()
    if not TEMPLATE.is_file():
        raise SystemExit(f"{TEMPLATE} not found: the real chat template is needed to catch the renderer mutations")
    base = run_one(Mutation("baseline", SRC, "", ""), baseline=True)
    if not base["passed"]:
        raise SystemExit(f"the unmutated copy fails its own tests: {base['tail']}")
    print(f"baseline: tests pass in the copy ({base['seconds']} s)", flush=True)
    results = []
    for m in MUTATIONS:
        if args.only in m.name:
            r = run_one(m)
            results.append(r)
            print(
                f"{'CAUGHT  ' if r['caught'] else 'SURVIVED'} {r['name']}  [{r['caught_by']}]  ({r['seconds']} s)",
                flush=True,
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"n": len(results), "caught": sum(r["caught"] for r in results), "results": results}, indent=2)
        + "\n"
    )
    survivors = [r for r in results if not r["caught"]]
    print(f"\n{len(results) - len(survivors)} of {len(results)} mutations caught")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
