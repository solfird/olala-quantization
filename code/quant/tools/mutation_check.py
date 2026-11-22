"""Mutation check: break each guard on a copy of the code and confirm that the tests notice.

Usage: python tools/mutation_check.py [--out mutation_report.json]

Every mutation edits a copy of olala_quant/, fork-patch/ and tests/ in a temporary directory (the
real files are never touched) and runs pytest there. A mutation is 'caught' when pytest fails.
A surviving mutation means a guard without a test; the script exits 1 in that case.
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


@dataclass(frozen=True)
class Mutation:
    """One deliberate defect: replace ``old`` by ``new`` in ``path`` (relative to the project)."""

    name: str
    path: str
    old: str
    new: str


MUTATIONS = [
    Mutation("fp8: wrong max (447)", "olala_quant/fp8.py", "FP8_MAX = 448.0", "FP8_MAX = 447.0"),
    Mutation("fp8: no minimum scale", "olala_quant/fp8.py", ".clamp_min(MIN_SCALE)", ".clamp_min(0.0)"),
    Mutation(
        "nvfp4: nibble order swapped",
        "olala_quant/nvfp4.py",
        "return codes[..., 0::2] | (codes[..., 1::2] << 4)",
        "return (codes[..., 0::2] << 4) | codes[..., 1::2]",
    ),
    Mutation(
        "nvfp4: tie at 0.25 rounds up",
        "olala_quant/nvfp4.py",
        "(a > 0.25).to(torch.uint8)",
        "(a >= 0.25).to(torch.uint8)",
    ),
    Mutation(
        "nvfp4: tie at 0.75 rounds down",
        "olala_quant/nvfp4.py",
        "(a >= 0.75).to(torch.uint8)",
        "(a > 0.75).to(torch.uint8)",
    ),
    Mutation(
        "nvfp4: global scale is the reciprocal",
        "olala_quant/nvfp4.py",
        "g = torch.where(amax > 0, top / torch.where(amax > 0, amax, torch.ones_like(amax)), 1.0)",
        "g = torch.where(amax > 0, torch.where(amax > 0, amax, torch.ones_like(amax)) / top, 1.0)",
    ),
    Mutation(
        "nvfp4: overflow counted without noise margin",
        "olala_quant/nvfp4.py",
        "_OVERFLOW_BOUND = E4M3_MAX * (1.0 + 1e-6)",
        "_OVERFLOW_BOUND = E4M3_MAX",
    ),
    Mutation(
        "nvfp4: dequant forgets the global scale",
        "olala_quant/nvfp4.py",
        "scale = block_scale.to(torch.float32).unsqueeze(-1) * inv_g",
        "scale = block_scale.to(torch.float32).unsqueeze(-1)",
    ),
    Mutation(
        "plan: routed pair not expanded",
        "olala_quant/plan.py",
        "ignored_q.add(missing)",
        "pass",
    ),
    Mutation(
        "plan: pattern matching nothing accepted",
        "olala_quant/plan.py",
        'raise IgnoreListError(f"ignore pattern {pattern!r} matches no tensor of the source")',
        "continue",
    ),
    Mutation(
        "plan: no regex for routed experts in vllm ignore",
        "olala_quant/plan.py",
        'entries.add("re:" + re.escape(t.vllm_module) + r"\\..*")',
        "pass",
    ),
    Mutation(
        "plan: ignore list that excludes nothing accepted",
        "olala_quant/plan.py",
        "if patterns and not ignored_q:",
        "if False:",
    ),
    Mutation(
        "convert: kept tensors are widened to float32",
        "olala_quant/convert.py",
        "buffer[name] = tensor.contiguous()",
        "buffer[name] = tensor.float().contiguous()",
    ),
    Mutation(
        "convert: config.json loses the quantization_config",
        "olala_quant/convert.py",
        'cfg["quantization_config"] = recipe.quantization_config(plan.vllm_ignore)',
        "pass",
    ),
    Mutation(
        "verify: bit-identity check disabled",
        "olala_quant/verify.py",
        "if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(_raw(a), _raw(b)):",
        "if False:",
    ),
    Mutation(
        "verify: error bound disabled",
        "olala_quant/verify.py",
        "if not err.rel_fro <= limit:",
        "if False:",
    ),
    Mutation(
        "verify: NaN scale check disabled",
        "olala_quant/verify.py",
        "if not torch.isfinite(bsf).all():",
        "if False:",
    ),
    Mutation(
        "loader: dtype contract is a no-op",
        "fork-patch/olala_quant_loader.py",
        "if loaded_q != param_q or (loaded_q and loaded != param):",
        "if False:",
    ),
    Mutation(
        "loader: w13 and w2 shard ids swapped",
        "fork-patch/olala_quant_loader.py",
        '"experts.experts": ("w13", "w1"),',
        '"experts.experts": ("w13", "w2"),',
    ),
    Mutation(
        "loader: slice shape never checked",
        "fork-patch/olala_quant_loader.py",
        "if len(got) != len(expected) or any(g > e for g, e in zip(got, expected, strict=True)):",
        "if False:",
    ),
    Mutation(
        "loader: unconsumed tensors never raise",
        "fork-patch/olala_quant_loader.py",
        "if bad:",
        "if False:",
    ),
    Mutation(
        "loader: down_proj suffixes not renamed",
        "fork-patch/olala_quant_loader.py",
        '"down_proj": "fc1_latent_proj",',
        '"down_proj": "down_proj",',
    ),
    Mutation(
        "patch: silent skip restored in the patch text",
        "fork-patch/olala-quant-loader.patch",
        "+        tolerated = tracker.raise_if_unconsumed(",
        "+        tolerated = tracker.raise_if_unconsumed_REMOVED(",
    ),
    # --- component ablation (olala_quant/ablation.py, --select) ---------------------------------------
    Mutation(
        "ablation: list I not kept (a selection replaces the base ignore list)",
        "olala_quant/ablation.py",
        "    patterns = base + tuple(glob.escape(n) for n in deselected)",
        "    patterns = tuple(glob.escape(n) for n in deselected)",
    ),
    Mutation(
        "ablation: deselected tensors not added to the ignore list (selection applied to nothing)",
        "olala_quant/convert.py",
        "        ignore_patterns = resolved.patterns\n",
        "        pass\n",
    ),
    Mutation(
        "ablation: ignore list incomplete for deselected modules (first one dropped)",
        "olala_quant/ablation.py",
        "    patterns = base + tuple(glob.escape(n) for n in deselected)",
        "    patterns = base + tuple(glob.escape(n) for n in deselected[1:])",
    ),
    Mutation(
        "ablation: selection not recorded in quantization_info.json",
        "olala_quant/convert.py",
        '        info["selection"] = selection\n',
        "        pass\n",
    ),
    Mutation(
        "ablation: default conversion output changed (selection block always written)",
        "olala_quant/convert.py",
        "    if selection is not None:  # absent from a default",
        "    if True:  # absent from a default",
    ),
    Mutation(
        "ablation: layer range off by one at the top (end excluded)",
        "olala_quant/ablation.py",
        "any(a <= layer <= b for a, b in self.intervals)",
        "any(a <= layer < b or a == b == layer for a, b in self.intervals)",
    ),
    Mutation(
        "ablation: layer range off by one at the bottom (start excluded)",
        "olala_quant/ablation.py",
        "any(a <= layer <= b for a, b in self.intervals)",
        "any(a < layer <= b or a == b == layer for a, b in self.intervals)",
    ),
    Mutation(
        "ablation: the routed down projection assigned to another component (pair split)",
        "olala_quant/ablation.py",
        '    "routed_down": "experts",',
        '    "routed_down": "latent",',
    ),
    Mutation(
        "ablation: attention and Mamba mixer_proj swapped",
        "olala_quant/ablation.py",
        '        return "attention" if kinds[layer] == "V" else "mamba"',
        '        return "mamba" if kinds[layer] == "V" else "attention"',
    ),
    Mutation(
        "ablation: non-experts includes the experts",
        "olala_quant/ablation.py",
        '    "non-experts": frozenset(COMPONENTS) - {"experts"},',
        '    "non-experts": frozenset(COMPONENTS),',
    ),
    Mutation(
        "ablation: adjacent layer ranges not merged in the canonical spec",
        "olala_quant/ablation.py",
        "        if merged and a <= merged[-1][1] + 1:",
        "        if merged and a <= merged[-1][1]:",
    ),
    Mutation(
        "ablation: reversed layer range accepted",
        "olala_quant/ablation.py",
        "            if last < first:",
        "            if False:",
    ),
    Mutation(
        "ablation: layers beyond the model accepted",
        "olala_quant/ablation.py",
        "            if a < lo or b > hi:",
        "            if False:",
    ),
    Mutation(
        "ablation: a selection that quantizes nothing accepted",
        "olala_quant/ablation.py",
        '    if not any(t.category == "quantized" for t in plan.tensors):',
        "    if False:",
    ),
    Mutation(
        "ablation: expected layer count counts tensors, not modules (routed pair counted twice)",
        "olala_quant/ablation.py",
        "    modules = {(t.layer, t.vllm_module) for t in quantized}",
        "    modules = {t.name for t in quantized}",
    ),
    Mutation(
        "ablation: --select allowed with --no-ignore (list I droppable)",
        "olala_quant/convert.py",
        "    if args.select is not None and args.no_ignore:",
        "    if False:",
    ),
    Mutation(
        "ablation: verify does not re-derive the recorded selection",
        "olala_quant/verify.py",
        "    _check_selection(report, header, plan, info, select)\n",
        "",
    ),
    Mutation(
        "ablation: verify does not compare the recorded ignore patterns with the re-derived ones",
        "olala_quant/verify.py",
        '    if resolved.patterns != tuple(info.get("ignore", {}).get("patterns", ())):',
        "    if False:",
    ),
    Mutation(
        "ablation: check ignores the number of quantized tensors on disk",
        "olala_quant/ablation.py",
        '    if n_on_disk != want["quantized_tensors"]:',
        "    if False:",
    ),
    Mutation(
        "ablation: check ignores the server's Quantizing line",
        "olala_quant/ablation.py",
        '            if (n, by_type) != (want["quantized_layers"], want["quantized_layers_by_type"]):',
        "            if False:",
    ),
]


def run_mutation(m: Mutation) -> tuple[bool, str]:
    """Return ``(caught, first failing test or note)`` for one mutation."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name in ("olala_quant", "fork-patch", "tests"):
            shutil.copytree(ROOT / name, work / name, ignore=shutil.ignore_patterns("__pycache__", "work"))
        shutil.copy(ROOT / "pyproject.toml", work / "pyproject.toml")
        target = work / m.path
        text = target.read_text()
        if m.old not in text:
            return False, f"MUTATION DID NOT APPLY: {m.old!r} not found in {m.path}"
        target.write_text(text.replace(m.old, m.new, 1))
        env = {
            "PYTHONPATH": str(work),
            "PATH": "/usr/bin:/bin",
            "HOME": str(work),
            "OLALA_FORK_DIR": str(ROOT.parent / "recon" / "src" / "vllm"),
        }
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-x", "--no-header", "-p", "no:cacheprovider"],
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
