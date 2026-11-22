"""Mutation check of the G1-lite guards: weaken each guard in a throw-away copy and confirm that the tests fail.

Usage: python tools/mutation_check.py [--out results/mutation_report.json] [--only SUBSTRING]

Each mutation replaces ``old`` by ``new`` in ONE file (the script fails when ``old`` is not found exactly once, so a
mistyped pattern cannot produce a meaningless 'survivor'), in a temporary copy of ``quant/gptq`` (or of ``quant``
for the changes to ``olala_quant``). The real files are never touched. pytest runs in the copy with a 600 s
timeout; the mutation is 'caught' when pytest fails or times out. A survivor is a guard without a test: the
script exits 1. An unmutated baseline copy must pass first.
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
QUANT = GPTQ.parent
TIMEOUT_S = 600


@dataclass(frozen=True)
class Mutation:
    """Replace ``old`` by ``new`` in ``path`` (relative to ``quant/gptq``, or ``quant/`` for ``project == 'quant'``)."""

    name: str
    path: str
    old: str
    new: str
    project: str = "gptq"


MUTATIONS = [
    # ---- solve.py
    Mutation(
        "solve: diagonal-ratio spike guard disabled",
        "solve.py",
        "elif float(ratio[k]) > cfg.diag_ratio_max:",
        "elif False:",
    ),
    Mutation(
        "solve: non-finite Hessian not detected",
        "solve.py",
        "finite = torch.isfinite(h).flatten(1).all(dim=1)",
        "finite = torch.ones(e, dtype=torch.bool, device=h.device)",
    ),
    Mutation(
        "solve: negative diagonal not detected",
        "solve.py",
        "neg = (diag < 0).any(dim=1)",
        "neg = torch.zeros(e, dtype=torch.bool, device=h.device)",
    ),
    Mutation(
        "solve: act-order reversed",
        "solve.py",
        "perm = torch.argsort(diag0, dim=1, descending=True, stable=True)",
        "perm = torch.argsort(diag0, dim=1, descending=False, stable=True)",
    ),
    Mutation(
        "solve: no damping",
        "solve.py",
        "h = h + (percdamp * diag.mean(dim=1))[:, None, None] * eye",
        "h = h + (0.0 * diag.mean(dim=1))[:, None, None] * eye",
    ),
    Mutation("solve: dead columns not repaired", "solve.py", "dead = diag == 0", "dead = diag < 0"),
    Mutation(
        "solve: error not propagated to later blocks",
        "solve.py",
        "wc[:, :, i2:] -= torch.bmm(err1, upper[:, i1:i2, i2:])",
        "pass",
    ),
    Mutation(
        "solve: static scale forgets 1/global",
        "solve.py",
        "dq = r.block_scale.to(torch.float32) * (1.0 / r.global_scale.reshape(-1, 1, 1))",
        "dq = r.block_scale.to(torch.float32)",
    ),
    Mutation(
        "solve: codes not mapped back to natural column order",
        "solve.py",
        "codes.scatter_(2, perm[:, None, :].expand(ec, rows, cols), out)",
        "codes = out",
    ),
    Mutation(
        "solve: inactive experts solved anyway",
        "solve.py",
        "if not bool(act[k]):\n                continue",
        "if False:\n                continue",
    ),
    Mutation("solve: held-out guard keeps GPTQ when worse", "solve.py", "worse = g > r", "worse = False"),
    Mutation("solve: held-out guard judges on too few rows", "solve.py", "if x.shape[0] < min_rows:", "if False:"),
    # ---- hessians.py
    Mutation(
        "hessians: token-norm clip disabled", "hessians.py", "cap = clip * norms.median()", "cap = clip * norms.max()"
    ),
    Mutation(
        "hessians: held-out windows leak into the Hessian",
        "hessians.py",
        "held_tok = held_win[cap.window.to(dev)]",
        "held_tok = torch.zeros_like(cap.window, dtype=torch.bool, device=dev)",
    ),
    Mutation(
        "hessians: blend prior dropped",
        "hessians.py",
        "h_up[e] += cfg.blend_pseudo * d * gram_all",
        "h_up[e] += 0.0 * gram_all",
    ),
    Mutation("hessians: minimum-token fallback disabled", "hessians.py", "elif n < min_tokens:", "elif False:"),
    Mutation(
        "hessians: Gram accumulated in float16",
        "hessians.py",
        "r = rows.to(torch.float32)",
        "r = rows.to(torch.float16).to(torch.float32)",
    ),
    Mutation(
        "hessians: weights share memory with the mmap'd shard",
        "hessians.py",
        ".to(device, copy=True) for n in src.names()",
        ".to(device) for n in src.names()",
    ),
    Mutation(
        "hessians: resume ignores a state made for other windows",
        "hessians.py",
        'if saved["fingerprint"] != fp:',
        "if False:",
    ),
    # ---- leakage_audit.py
    Mutation(
        "leakage: last n-gram of a document skipped",
        "leakage_audit.py",
        "for i in range(len(words) - n + 1):\n        gram",
        "for i in range(len(words) - n):\n        gram",
    ),
    Mutation(
        "leakage: case-sensitive",
        "leakage_audit.py",
        'return _NON_WORD.sub(" ", text.lower()).split()',
        'return _NON_WORD.sub(" ", text).split()',
    ),
    Mutation(
        "leakage: CLI exits 0 on overlap",
        "leakage_audit.py",
        '        return 1\n    print(f"OK:',
        '        return 0\n    print(f"OK:',
    ),
    Mutation(
        "leakage: missing evaluation task tolerated",
        "leakage_audit.py",
        "if missing and not args.allow_missing_tasks:",
        "if False:",
    ),
    # ---- calib_data.py
    Mutation(
        "calib_data: unaudited manifest accepted",
        "calib_data.py",
        'if not audit or audit.get("ok") is not True or audit.get("n_hits", 1) != 0:',
        "if False:",
    ),
    Mutation(
        "calib_data: modified window file accepted",
        "calib_data.py",
        'if la.sha256_file(wpath) != m["windows"]["sha256"]:',
        "if False:",
    ),
    Mutation(
        "calib_data: leaky selection is written anyway",
        "calib_data.py",
        "        if not drop_flagged:\n            la.raise_if_leaky(report)",
        "        if not drop_flagged:\n            pass",
    ),
    Mutation(
        "calib_data: documents not separated by EOS", "calib_data.py", "        flat.append(eos_id)", "        pass"
    ),
    # ---- build_g1lite.py
    Mutation(
        "build: solver scales not compared with the data-free ones",
        "build_g1lite.py",
        "                if not torch.equal(\n                    ref.block_scale.cpu().view(torch.uint8)",
        "                if False and not torch.equal(\n                    ref.block_scale.cpu().view(torch.uint8)",
    ),
    Mutation(
        "build: held-out guard ignored",
        "build_g1lite.py",
        "use_gptq = (~both_fallback) & guard.keep.cpu()",
        "use_gptq = ~both_fallback",
    ),
    Mutation(
        "build: modeling-file mismatch tolerated",
        "build_g1lite.py",
        "if not all(match.values()) and not args.allow_modeling_mismatch:",
        "if False:",
    ),
    Mutation(
        "build: layer outside the model accepted",
        "build_g1lite.py",
        "bad = sorted(i for i in out if not 0 <= i < n_layers)",
        "bad = []",
    ),
    Mutation(
        "build: a finished layer is not restored on resume",
        "build_g1lite.py",
        "return layer in self.targets and not self.path(layer).is_file()",
        "return layer in self.targets",
    ),
    # ---- olala_model.py
    Mutation(
        "olala_model: unknown modeling file treated as identical",
        "olala_model.py",
        "(model_dir / name).is_file() and sha256_file",
        "(not (model_dir / name).is_file()) or sha256_file",
    ),
    # ---- olala_quant (calibrated verify mode, convert hooks)
    Mutation(
        "verify: block scales not compared with the data-free ones",
        "olala_quant/verify.py",
        "if not torch.equal(ref.block_scale.view(torch.uint8), block.view(torch.uint8)):",
        "if False:",
        "quant",
    ),
    Mutation(
        "verify: untouched tensors may differ from the data-free codes",
        "olala_quant/verify.py",
        "if not torch.equal(ref.packed, packed):",
        "if False:",
        "quant",
    ),
    Mutation(
        "verify: experts declared RTN not compared",
        "olala_quant/verify.py",
        "and not torch.equal(ref.packed[local], packed[local]):",
        "and False:",
        "quant",
    ),
    Mutation(
        "verify: calibrated error bound not applied",
        "olala_quant/verify.py",
        "limit = min(limit, bounds.nvfp4_calibrated_factor * calibrated.rtn_acc.finish().rel_fro)",
        "pass",
        "quant",
    ),
    Mutation(
        "verify: strict mode accepts a calibrated checkpoint",
        "olala_quant/verify.py",
        'if "calibration" in info and not calibrated:',
        "if False:",
        "quant",
    ),
    Mutation(
        "verify: calibration info may name any tensor",
        "olala_quant/verify.py",
        "for bad in sorted(calib_names - quantized_routed):",
        "for bad in []:",
        "quant",
    ),
    Mutation(
        "convert: override ignored",
        "olala_quant/convert.py",
        "replaced = override(name, tensor) if override is not None else None",
        "replaced = None",
        "quant",
    ),
]


def apply(text: str, m: Mutation) -> str:
    """Return ``text`` with the mutation applied; raise if the pattern is not found exactly once."""
    n = text.count(m.old)
    if n != 1:
        raise SystemExit(f"mutation {m.name!r}: pattern found {n} times in {m.path}, expected exactly 1")
    return text.replace(m.old, m.new)


def run_one(m: Mutation, baseline: bool = False) -> dict:
    """Apply one mutation in a temporary copy and run the tests there."""
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        if m.project == "quant":
            root = Path(tmp) / "quant"
            shutil.copytree(
                QUANT,
                root,
                ignore=shutil.ignore_patterns(
                    ".venv", "gptq", "design", "work", "__pycache__", ".pytest_cache", ".ruff_cache"
                ),
            )
            python = QUANT / ".venv" / "bin" / "python"
            tests = ["tests/test_verify_calibrated.py", "tests/test_verify.py", "tests/test_convert.py"]
        else:
            root = Path(tmp) / "gptq"
            shutil.copytree(
                GPTQ,
                root,
                ignore=shutil.ignore_patterns(
                    ".venv", "data", "results", "__pycache__", ".pytest_cache", ".ruff_cache"
                ),
            )
            python = GPTQ / ".venv" / "bin" / "python"
            tests = ["tests"]
        target = root / m.path
        original = target.read_text()
        if not baseline:
            target.write_text(apply(original, m))
        env = {"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin", "HOME": str(Path.home())}
        try:
            proc = subprocess.run(
                [str(python), "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *tests],
                cwd=root,
                timeout=TIMEOUT_S,
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
            code, tail = proc.returncode, (proc.stdout.strip().splitlines() or [""])[-1]
            failed = [ln for ln in proc.stdout.splitlines() if ln.startswith("FAILED")][:1]
        except subprocess.TimeoutExpired:  # pragma: no cover - 'timeout' normally fires first
            code, tail, failed = 124, "timeout", []
    return {
        "passed": code == 0,
        "tail": tail,
        "name": m.name,
        "file": m.path if m.project == "gptq" else f"quant/{m.path}",
        "caught": code != 0,
        "caught_by": failed[0].removeprefix("FAILED ").split(" - ")[0]
        if failed
        else ("timeout" if code == 124 else tail),
        "seconds": round(time.time() - t0, 1),
    }


def baseline(project: str) -> None:
    """Run the tests of an UNMUTATED copy; a red baseline would make every mutation look 'caught'."""
    r = run_one(Mutation("baseline", "pyproject.toml", "", "", project), baseline=True)
    if not r["passed"]:
        raise SystemExit(f"the unmutated copy of {project!r} fails its own tests: {r['tail']}")
    print(f"baseline {project}: tests pass in the copy ({r['seconds']} s)", flush=True)


def main() -> int:
    """Run every mutation; exit 1 if any survives."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, default=GPTQ / "results" / "mutation_report.json")
    p.add_argument("--only", default="")
    args = p.parse_args()
    results = []
    for project in sorted({m.project for m in MUTATIONS if args.only in m.name}):
        baseline(project)
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
