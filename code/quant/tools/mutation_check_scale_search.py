# ruff: noqa: E501  (the strings below are source lines to be replaced verbatim)
"""Mutation check of the scale search (PROTOCOL.md amendment 25) and of ``experts-all``.

Usage: python tools/mutation_check_scale_search.py [--out report.json]

Each mutation edits a COPY of olala_quant/ and tests/ in a temporary directory (the real files are never
touched; the text to replace must occur exactly once or the run stops), runs the tests that concern the
feature and counts the mutation as caught when one of them fails. The whole suite is not run:
tools/mutation_check.py does that for the older guards. Exit 1 when a mutation survives.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = (
    "tests/test_nvfp4_scale_search.py",
    "tests/test_convert_scale_search.py",
    "tests/test_ablation.py",
    "tests/test_convert_select.py",
    "tests/test_nvfp4.py",
    "tests/test_verify.py",
    "tests/test_convert.py",
)
NV = "olala_quant/nvfp4.py"
AB = "olala_quant/ablation.py"
CV = "olala_quant/convert.py"
VF = "olala_quant/verify.py"


@dataclass(frozen=True)
class Mutation:
    """One deliberate defect: every ``(path, old, new)`` edit replaces exactly one occurrence."""

    name: str
    edits: tuple[tuple[str, str, str], ...]


def one(name: str, path: str, old: str, new: str) -> Mutation:
    """A mutation with a single edit."""
    return Mutation(name, ((path, old, new),))


MUTATIONS = [
    # --- experts-all: which part of list I is lifted
    # ------------------------------------------------------------
    one(
        "list I is not lifted at all (the MoE of layers 0, 1, 5, 19 stays bf16)",
        AB,
        "    kept = tuple(p for p in base if p not in lifted)",
        "    kept = base",
    ),
    one(
        "list I is lifted for every pattern, the mixers of layers 1, 7, 23 included",
        AB,
        "        lifted = tuple(p for p in base if any(fnmatch.fnmatchcase(n, p) for n in routed))",
        "        lifted = tuple(base)",
    ),
    one(
        "list I is lifted for the wrong layers (only the patterns of layers 0 and 1)",
        AB,
        "        lifted = tuple(p for p in base if any(fnmatch.fnmatchcase(n, p) for n in routed))",
        '        lifted = tuple(p for p in base if p.startswith(("model.layers.0.", "model.layers.1.m")) and any(fnmatch.fnmatchcase(n, p) for n in routed))',
    ),
    Mutation(
        "the mixer projections of layers 1, 7, 23 are quantized (list I lifted whole, mixers selected)",
        (
            (
                AB,
                "        lifted = tuple(p for p in base if any(fnmatch.fnmatchcase(n, p) for n in routed))",
                "        lifted = tuple(base)",
            ),
            (
                AB,
                '    "experts-all": frozenset({"experts"}),',
                '    "experts-all": frozenset({"experts", "mamba", "attention"}),',
            ),
        ),
    ),
    one(
        "experts-all also quantizes the shared experts of the lifted layers",
        AB,
        '    "experts-all": frozenset({"experts"}),',
        '    "experts-all": frozenset({"experts", "shared-experts"}),',
    ),
    one(
        "the selection block does not say that list I was lifted",
        AB,
        '            record["list_i_lifted"] = True',
        '            record["list_i_lifted"] = False',
    ),
    one(
        "the selection block does not record which patterns were lifted",
        AB,
        '            record["lifted_ignore_patterns"] = list(self.lifted)',
        '            record["lifted_ignore_patterns"] = []',
    ),
    one(
        "the lifted patterns are recorded for every selection (the older blocks change)",
        AB,
        "        if self.selection.lifts_routed:\n            record[",
        "        if True:\n            record[",
    ),
    # --- the scale search
    # -----------------------------------------------------------------------------------------
    one(
        "scale search factors changed (0.75 becomes 0.70)",
        NV,
        "SCALE_SEARCH_FACTORS: tuple[float, ...] = (1.00, 0.95, 0.90, 0.85, 0.80, 0.75)",
        "SCALE_SEARCH_FACTORS: tuple[float, ...] = (1.00, 0.95, 0.90, 0.85, 0.80, 0.70)",
    ),
    one(
        "scale search factors changed (a factor above 1 is tried)",
        NV,
        "SCALE_SEARCH_FACTORS: tuple[float, ...] = (1.00, 0.95, 0.90, 0.85, 0.80, 0.75)",
        "SCALE_SEARCH_FACTORS: tuple[float, ...] = (1.05, 1.00, 0.95, 0.90, 0.85, 0.80)",
    ),
    one(
        "rounding order wrong (the base scale is rounded to e4m3 first, then multiplied by the factor)",
        NV,
        "    return (base * factor).clamp(min=0.0, max=E4M3_MAX).to(torch.float8_e4m3fn)",
        "    return (base.clamp(min=0.0, max=E4M3_MAX).to(torch.float8_e4m3fn).to(torch.float32) * factor).to(torch.float8_e4m3fn)",
    ),
    one(
        "search tie-break wrong (a later candidate wins a tie)",
        NV,
        "            better = err < best_err",
        "            better = err <= best_err",
    ),
    one(
        "search keeps the WORST candidate",
        NV,
        "            better = err < best_err",
        "            better = err > best_err",
    ),
    one(
        "search picks by absolute error instead of squared error",
        NV,
        "    return (diff * diff).sum(dim=-1, keepdim=True)",
        "    return diff.abs().sum(dim=-1, keepdim=True)",
    ),
    one(
        "default output changed (the search is on unless asked off)",
        NV,
        "def quantize_nvfp4(weight: torch.Tensor, *, expert_chunk: int = 8, scale_search: bool = False)",
        "def quantize_nvfp4(weight: torch.Tensor, *, expert_chunk: int = 8, scale_search: bool = True)",
    ),
    one(
        "the global scale is derived from a smaller amax when searching",
        NV,
        "            gscale[esl] = _global_scale(w.abs().amax(dim=(1, 2)))",
        "            gscale[esl] = _global_scale(w.abs().amax(dim=(1, 2)) * (0.75 if scale_search else 1.0))",
    ),
    one(
        "the search chunk is far larger than documented",
        NV,
        "_SEARCH_CHUNK_ELEMS = 4 * 1024 * 1024",
        "_SEARCH_CHUNK_ELEMS = 1024 * 1024 * 1024",
    ),
    # --- the converter
    # ------------------------------------------------------------------------------------------------
    one(
        "the converter does not pass the flag to the quantizer",
        CV,
        "                            scale_search=scale_search,\n                        )",
        "                            scale_search=False,\n                        )",
    ),
    one(
        "the converter does not record the search in quantization_info.json",
        CV,
        "    if scale_search:  # likewise",
        "    if False:  # likewise",
    ),
    one(
        "the converter lets fp8 use the search",
        CV,
        '    recipe = get_recipe(recipe_name)\n    if scale_search and recipe.quantizer != "nvfp4":',
        '    recipe = get_recipe(recipe_name)\n    if False and recipe.quantizer != "nvfp4":',
    ),
    # --- the verifier
    # ---------------------------------------------------------------------------------------------------
    one(
        "verify flags a checkpoint made with the search (plain checks applied to it)",
        VF,
        '            "a checkpoint with searched scales cannot be verified as calibrated (--calibrated)",\n        )\n    return True',
        '            "a checkpoint with searched scales cannot be verified as calibrated (--calibrated)",\n        )\n    return False',
    ),
    one(
        "verify no longer checks that every scale is one of the six candidates",
        VF,
        "    if not bool(is_candidate.all()):",
        "    if False:",
    ),
    one(
        "verify no longer compares the error with the plain scale's",
        VF,
        "    if not bool((err <= err_plain * (1.0 + 1e-4)).all()):",
        "    if False:",
    ),
    one(
        "verify applies the plain top-scale rule (448) to a searched checkpoint",
        VF,
        "        if searched:\n            # searched scales",
        "        if False:\n            # searched scales",
    ),
    one(
        "verify ignores --scale-search / --no-scale-search",
        VF,
        '    if required is True and "scale_search" not in info:',
        '    if False and "scale_search" not in info:',
    ),
]


def run_mutation(m: Mutation) -> tuple[bool, str]:
    """Return ``(caught, first failing test or note)`` for one mutation."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name in ("olala_quant", "tests"):
            shutil.copytree(ROOT / name, work / name, ignore=shutil.ignore_patterns("__pycache__", "work"))
        shutil.copy(ROOT / "pyproject.toml", work / "pyproject.toml")
        for path, old, new in m.edits:
            target = work / path
            text = target.read_text()
            if text.count(old) != 1:
                return (
                    False,
                    f"MUTATION DID NOT APPLY: {text.count(old)} occurrences of {old[:70]!r} in {path}",
                )
            target.write_text(text.replace(old, new, 1))
        env = {"PYTHONPATH": str(work), "PATH": "/usr/bin:/bin", "HOME": str(work)}
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-x", "--no-header", "-p", "no:cacheprovider", *TESTS],
            cwd=work,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        out = proc.stdout + proc.stderr
        failed = [ln for ln in out.splitlines() if ln.startswith(("FAILED", "ERROR"))]
        if proc.returncode != 0:
            return True, failed[0] if failed else out.strip().splitlines()[-1]
        return False, "all tests passed with the defect in place"


def main() -> int:
    """Run every mutation and print a table."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    rows = []
    for m in MUTATIONS:
        caught, note = run_mutation(m)
        rows.append({"mutation": m.name, "caught": caught, "first_failure": note})
        print(f"{'CAUGHT  ' if caught else 'SURVIVED'} {m.name}: {note[:110]}")
    survivors = [r for r in rows if not r["caught"]]
    print(f"{len(rows) - len(survivors)} of {len(rows)} mutations caught")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2) + "\n")
    return 1 if survivors else 0


if __name__ == "__main__":
    raise SystemExit(main())
