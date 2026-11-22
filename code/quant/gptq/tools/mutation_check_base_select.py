"""Mutation check of the ``--base-select`` / CAL-1-directory / per-domain guards: weaken each in a throw-away copy.

Usage: python tools/mutation_check_base_select.py [--out results/mutation_base_select_report.json] [--only SUBSTRING]

Same method as ``tools/mutation_check_cal1.py``: each mutation replaces ``old`` by ``new`` in ONE file of a temporary
copy of ``quant/gptq`` (the script stops when ``old`` is not found exactly once, so a mistyped pattern cannot yield a
meaningless 'survivor'); pytest runs the test files of the feature in the copy; 'caught' means pytest failed or timed
out. The real files are never touched. An unmutated baseline copy must pass first. Exit status 1 when a mutation
survives.
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
TIMEOUT_S = 600
TESTS = [
    "tests/test_base_select.py",
    "tests/test_default_unchanged.py",
    "tests/test_window_domains.py",
    "tests/test_pod_script.py",
    "tests/test_tiny_e2e.py",
]
BUILD = "build_g1lite.py"


@dataclass(frozen=True)
class Mutation:
    """Replace ``old`` by ``new`` in ``path`` (relative to ``quant/gptq``)."""

    name: str
    old: str
    new: str
    path: str = BUILD


MUTATIONS = [
    # ---- a non-expert tensor must never be quantized under --base-select
    Mutation(
        "the base selection is not given to the converter (non-experts quantized under --base-select experts)",
        "        select=select,\n",
        "",
    ),
    Mutation(
        "the guard against quantizing non-experts is removed",
        '    bad = [t.name for t in plan.tensors if t.category == "quantized" and t.role not in ROUTED_ROLES]',
        "    bad = []",
    ),
    Mutation(
        "the guard accepts mixer projections (attention and mamba specs pass)",
        'ROUTED_ROLES = ("routed_up", "routed_down")',
        'ROUTED_ROLES = ("routed_up", "routed_down", "mixer_proj", "attn_gate_proj")',
    ),
    Mutation(
        "the guard accepts shared experts and latent projections",
        'ROUTED_ROLES = ("routed_up", "routed_down")',
        'ROUTED_ROLES = ("routed_up", "routed_down", "shared_up", "shared_down", "latent_reduce", "latent_expand")',
    ),
    Mutation(
        "the guard only looks at tensors that are not yet quantized (category test inverted)",
        't.category == "quantized" and t.role not in ROUTED_ROLES',
        't.category != "quantized" and t.role not in ROUTED_ROLES',
    ),
    # ---- the ignore list and the selection record
    Mutation(
        "the targets come from the default plan, not from the base selection (ignore list incomplete)",
        "        plan = resolved.plan\n",
        '        plan = build_plan(source.header(), get_recipe("nvfp4"), patterns)\n',
    ),
    Mutation(
        "the selection spec is recorded as typed, not in its canonical spelling",
        "        base_select = resolved.selection.spec  # one spelling, the one convert records\n",
        "",
    ),
    Mutation(
        "base_select is not recorded in the calibration block",
        '            {"base_select": base_select, "non_expert_tensors": "bf16, listed in quantization_config.ignore"}\n'
        "            if base_select\n",
        '            {"non_expert_tensors": "bf16, listed in quantization_config.ignore"}\n            if base_select\n',  # noqa: E501
    ),
    Mutation(
        "the non-expert state is not recorded in the calibration block",
        '            {"base_select": base_select, "non_expert_tensors": "bf16, listed in quantization_config.ignore"}\n',  # noqa: E501
        '            {"base_select": base_select}\n',
    ),
    Mutation("base_select is not recorded in the report", '        "base_select": base_select or None,\n', ""),
    Mutation(
        "--base-select combined with --no-ignore accepted",
        "        if args.no_ignore:\n            raise ValueError(",
        "        if False:\n            raise ValueError(",
    ),
    # ---- the default must stay what it was
    Mutation(
        "the default build uses the experts base",
        "select=base_select or None\n    )",
        'select=base_select or "experts"\n    )',
    ),
    Mutation(
        "the default build records an empty base_select in the calibration block",
        '            if base_select\n            else {}\n        ),\n        "modeling_olala_sha256"',
        '            if True\n            else {}\n        ),\n        "modeling_olala_sha256"',
    ),
    Mutation(
        "the default skip note is replaced by the base-selection one",
        '        if base_select\n        else "all routed tensors of the layer are in the ignore list"',
        '        if True\n        else "all routed tensors of the layer are in the ignore list"',
    ),
    # ---- CAL-1 directories and their audit
    Mutation(
        "a calibration directory is not accepted",
        "    if path.is_dir():\n        path = path / MANIFEST\n",
        "",
        "calib_data.py",
    ),
    Mutation(
        "a manifest whose audit failed is accepted",
        '    if not audit or audit.get("ok") is not True or audit.get("n_hits", 1) != 0:',
        "    if not audit:",
        "calib_data.py",
    ),
    Mutation(
        "an audit with hits is accepted",
        'audit.get("ok") is not True or audit.get("n_hits", 1) != 0:',
        'audit.get("ok") is not True:',
        "calib_data.py",
    ),
    Mutation(
        "a manifest with no audit block is accepted",
        '    if not audit or audit.get("ok") is not True or audit.get("n_hits", 1) != 0:',
        '    if audit and (audit.get("ok") is not True or audit.get("n_hits", 1) != 0):',
        "calib_data.py",
    ),
    Mutation(
        "changed windows accepted (sha256 not compared)",
        '    if la.sha256_file(wpath) != m["windows"]["sha256"]:',
        "    if False:",
        "calib_data.py",
    ),
    Mutation(
        "the build reads the windows of the file it was given, not of the resolved manifest",
        "    args.calib = calib_data.resolve_manifest(args.calib)  # a manifest.json, or the directory that holds it\n",
        "",
    ),
    # ---- windows per domain
    Mutation(
        "per-domain counts missing from the report", '            "per_domain": per_domain,\n        },', "        },"
    ),
    Mutation(
        "per-domain counts missing from the checkpoint info",
        '            **({"per_domain": per_domain} if per_domain["method"] != "unavailable" else {}),\n',
        "",
    ),
    Mutation(
        "the trace order ignores the seed",
        "random.Random(f\"{m['seed']}|order\").shuffle(order)",
        'random.Random("order").shuffle(order)',
        "window_domains.py",
    ),
    Mutation(
        "windows touched counts only the first window of a span",
        'rec["touched"].update(range(s // window, (e - 1) // window + 1))',
        'rec["touched"].add(s // window)',
        "window_domains.py",
    ),
    Mutation(
        "the sha256 of calib_text.jsonl is not checked",
        "    if want is not None and calib_data.la.sha256_file(text) != want:",
        "    if False:",
        "window_domains.py",
    ),
    Mutation(
        "a trace count that disagrees with the manifest is accepted as exact",
        '    if m.get("counts", {}).get("n_traces") != len(ids) or sum(d["n_traces"] for d in m["domains"].values()) != len(ids):',  # noqa: E501
        "    if False:",
        "window_domains.py",
    ),
    Mutation(
        "a domain with more tokens than the manifest rendered is accepted as exact",
        '        if got[name]["tokens"] > _slots(d):',
        "        if False:",
        "window_domains.py",
    ),
    Mutation(
        "a missing tail check (windows from other traces accepted as exact)",
        "        if not 0 <= left < window:",
        "        if False:",
        "window_domains.py",
    ),
    Mutation(
        "a marker that occurs more often than there are traces is accepted",
        "    if starts[:1] != [0] or len(starts) > len(ids):",
        "    if starts[:1] != [0]:",
        "window_domains.py",
    ),
    Mutation(
        "an unknown prompt source is not reported",
        "    except KeyError as e:\n        return _estimate(",
        "    except KeyError as e:\n        raise e  # noqa\n        return _estimate(",
        "window_domains.py",
    ),
    Mutation(
        "an estimate is labelled exact",
        '        "method": "token_share_estimate",\n        "reason": reason,',
        '        "method": "exact",\n        "reason": reason,',
        "window_domains.py",
    ),
    Mutation(
        "a non-CAL-1 manifest gets an attribution",
        '    if m.get("schema") != CAL1_SCHEMA or not m.get("domains"):',
        '    if not m.get("domains"):',
        "window_domains.py",
    ),
    # ---- the pod wrapper
    Mutation(
        "the build step does not pass --base-select",
        '--dst "$DST" \\\n    ${BASE_ARGS[@]+"${BASE_ARGS[@]}"} --device cuda',
        '--dst "$DST" \\\n    --device cuda',
        "pod_g1lite.sh",
    ),
    Mutation(
        "the feasibility step does not pass --base-select",
        '--dst "$WORK/ckpt/g1lite-feas" \\\n    ${BASE_ARGS[@]+"${BASE_ARGS[@]}"} --device cuda',
        '--dst "$WORK/ckpt/g1lite-feas" \\\n    --device cuda',
        "pod_g1lite.sh",
    ),
    Mutation(
        "the verify step does not require the selection",
        '    ${SELECT_ARGS[@]+"${SELECT_ARGS[@]}"} --report',
        "    --report",
        "pod_g1lite.sh",
    ),
    Mutation(
        "--base-select is passed even when empty (default changed)",
        'BASE_ARGS=(); [ -n "$BASE_SELECT" ] && BASE_ARGS=(--base-select "$BASE_SELECT")',
        'BASE_ARGS=(--base-select "$BASE_SELECT")',
        "pod_g1lite.sh",
    ),
    Mutation(
        "tokenize runs on a CAL-1 output",
        '  if cal1_windows_present; then\n    say "tokenize: not needed',
        '  if false; then\n    say "tokenize: not needed',
        "pod_g1lite.sh",
    ),
    Mutation(
        "tokenize skipped for any manifest with windows, not only CAL-1",
        '  [ -f "$CALIB/manifest.json" ] && grep -q \'"schema": "cal1/1"\' "$CALIB/manifest.json" \\\n    && grep',
        '  [ -f "$CALIB/manifest.json" ] && true \\\n    && grep',
        "pod_g1lite.sh",
    ),
    Mutation(
        "tokenize skipped for a CAL-1 manifest without windows",
        '    && grep -q \'"windows": {\' "$CALIB/manifest.json"',
        "    && true",
        "pod_g1lite.sh",
    ),
    # ---- the tiny comparison
    Mutation(
        "the RTN arm of the tiny comparison quantizes everything (N2 instead of N5)",
        "select=base_select or None, log=lambda _s: None",
        "select=None, log=lambda _s: None",
        "tiny_e2e.py",
    ),
    Mutation(
        "the G1 arm of the tiny comparison ignores the base selection",
        '                *(["--base-select", base_select] if base_select else []),\n',
        "",
        "tiny_e2e.py",
    ),
]


def apply(text: str, m: Mutation) -> str:
    """Return ``text`` with the mutation applied; stop if the pattern is not found exactly once."""
    n = text.count(m.old)
    if n != 1:
        raise SystemExit(f"mutation {m.name!r}: pattern found {n} times in {m.path}, expected exactly 1")
    return text.replace(m.old, m.new)


def run_one(m: Mutation, baseline: bool = False) -> dict:
    """Apply one mutation in a temporary copy and run the feature's tests there."""
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
        env = {"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin", "HOME": str(Path.home())}
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
    p.add_argument("--out", type=Path, default=GPTQ / "results" / "mutation_base_select_report.json")
    p.add_argument("--only", default="")
    args = p.parse_args()
    base = run_one(Mutation("baseline", BUILD, "", ""), baseline=True)
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
