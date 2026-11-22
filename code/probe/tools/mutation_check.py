"""Mutation check of the activation probe: weaken each guard in a throw-away copy and confirm that the tests fail.

Usage: python tools/mutation_check.py [--out results/mutation_report.json] [--only SUBSTRING]

Each mutation replaces ``old`` by ``new`` in ONE file of ``analysis/`` (the script fails when ``old`` is not found
exactly once, so a mistyped pattern cannot produce a meaningless 'survivor'), inside a temporary copy of ``analysis/``
that sees the real ``quant/gptq`` through a symlink. The real files are never touched. pytest runs in the copy with a
600 s timeout; the mutation is 'caught' when pytest fails or times out. A survivor is a guard without a test: the script
exits 1. An unmutated baseline copy must pass first.
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

ANALYSIS = Path(__file__).resolve().parents[1]
QUANT = ANALYSIS.parent / "quant"
TIMEOUT_S = 600


@dataclass(frozen=True)
class Mutation:
    """Replace ``old`` by ``new`` in ``path`` (relative to ``analysis/``)."""

    name: str
    path: str
    old: str
    new: str


MUTATIONS = [
    # ---- probe_stats.py
    Mutation(
        "stats: max over abs forgotten (signed max)",
        "probe_stats.py",
        "a = torch.where(finite, absx, torch.zeros_like(absx))",
        "a = torch.where(finite, xf, torch.zeros_like(xf))",
    ),
    Mutation(
        "stats: per-head max over abs forgotten",
        "probe_stats.py",
        "a = yf.abs()",
        "a = yf",
    ),
    Mutation(
        "stats: first-token flag always false",
        "probe_stats.py",
        '"is_first": pos == 0,',
        '"is_first": False,',
    ),
    Mutation(
        "stats: special-token flag missing",
        "probe_stats.py",
        '"is_special": bool(special[pos]),',
        '"is_special": False,',
    ),
    Mutation("stats: e4m3 threshold wrong", "probe_stats.py", "E4M3_MAX = 448.0", "E4M3_MAX = 480.0"),
    Mutation("stats: p99.9 is p99", "probe_stats.py", "P_QUANTILE = 0.999", "P_QUANTILE = 0.99"),
    Mutation(
        "stats: RMS divides by tokens instead of elements",
        "probe_stats.py",
        "rms = math.sqrt(self.sumsq / self.n_elems) if self.n_elems else 0.0",
        "rms = math.sqrt(self.sumsq / self.n_tokens) if self.n_elems else 0.0",
    ),
    Mutation(
        "stats: growth exponent against natural log",
        "probe_stats.py",
        "x = np.log10(idx[ok] + 1.0)",
        "x = np.log(idx[ok] + 1.0)",
    ),
    Mutation(
        "stats: non-finite entries leak into the maximum",
        "probe_stats.py",
        "a = torch.where(finite, absx, torch.zeros_like(absx))",
        "a = torch.nan_to_num(absx, nan=0.0)",
    ),
    Mutation(
        "stats: argmax kept from the first prompt instead of the largest",
        "probe_stats.py",
        "if self.argmax is None or s.max_abs > self.max_abs:",
        "if self.argmax is None:",
    ),
    # ---- probe_hooks.py
    Mutation(
        "hooks: layer index base wrong (1-based)",
        "probe_hooks.py",
        "layer = int(blk.layer_idx)\n        is_mamba",
        "layer = int(blk.layer_idx) + 1\n        is_mamba",
    ),
    Mutation(
        "hooks: mixer site hooked on the MoE module",
        "probe_hooks.py",
        'add(blk.mixer_proj.register_forward_hook(lambda _m, _a, out: self._site(rec, "mixer_out", out)))',
        'add(blk.mlp.register_forward_hook(lambda _m, _a, out: self._site(rec, "mixer_out", out)))',
    ),
    Mutation(
        "hooks: shared-expert site hooked on the routed up-projection",
        "probe_hooks.py",
        'add(mlp.shared_experts.register_forward_hook(lambda _m, _a, out: self._site(rec, "shared_out", out)))',
        'add(mlp.up_proj.register_forward_hook(lambda _m, _a, out: self._site(rec, "shared_out", out)))',
    ),
    Mutation(
        "hooks: post-ReLU^2 site records the pre-activation",
        "probe_hooks.py",
        "ps.summarize(post, token_of=tok, num_tokens=num), ctx.prompt",
        "ps.summarize(pre, token_of=tok, num_tokens=num), ctx.prompt",
    ),
    Mutation(
        "hooks: expert attribution always the first expert of the call",
        "probe_hooks.py",
        "torch.full((n,), state.seq[state.ptr], dtype=torch.long, device=x.device)",
        "torch.full((n,), state.seq[0], dtype=torch.long, device=x.device)",
    ),
    Mutation(
        "hooks: sibling erasure compares the output with itself",
        "probe_hooks.py",
        "rec.erasure.add(full_out, keep_out, top)",
        "rec.erasure.add(full_out, full_out, top)",
    ),
    Mutation(
        "hooks: sibling erasure keeps the wrong head",
        "probe_hooks.py",
        "keep[..., top * width : (top + 1) * width] = y[..., top * width : (top + 1) * width]",
        "keep[..., (top + 1) * width : (top + 2) * width] = y[..., (top + 1) * width : (top + 2) * width]",
    ),
    Mutation(
        "hooks: clamp ablation not applied",
        "probe_hooks.py",
        "return out.clamp(-ps.E4M3_MAX, ps.E4M3_MAX)",
        "return out",
    ),
    Mutation(
        "hooks: zero ablation not applied",
        "probe_hooks.py",
        "return torch.zeros_like(out)",
        "return out",
    ),
    Mutation(
        "hooks: head ablation edits the neighbouring head",
        "probe_hooks.py",
        "sl = slice(spec.head * width, (spec.head + 1) * width)",
        "sl = slice((spec.head + 1) * width, (spec.head + 2) * width)",
    ),
    Mutation(
        "hooks: ablation of the mixer branch hooked on the MoE branch",
        "probe_hooks.py",
        "return [blk.mixer_proj.register_forward_hook(lambda _m, _a, out: _edit(out, spec.mode))]",
        "return [blk.mlp.register_forward_hook(lambda _m, _a, out: _edit(out, spec.mode))]",
    ),
    Mutation(
        "hooks: probe hooks left in place by close()",
        "probe_hooks.py",
        "        for h in self._handles:\n            h.remove()\n",
        "        for h in self._handles:\n            pass\n",
    ),
    # ---- probe_ablate.py
    Mutation(
        "ablate: KL computed in the wrong direction",
        "probe_ablate.py",
        "kl = (logp_b.exp() * (logp_b - logp_a)).sum(-1)",
        "kl = (logp_a.exp() * (logp_a - logp_b)).sum(-1)",
    ),
    Mutation(
        "ablate: delta NLL sign flipped",
        "probe_ablate.py",
        '"delta_nll_mean": (na - nb) if',
        '"delta_nll_mean": (nb - na) if',
    ),
    Mutation(
        "ablate: ablated run compared with itself",
        "probe_ablate.py",
        "acc.add(prompt_metrics(model.lm_head, b, h, i[0], chunk))",
        "acc.add(prompt_metrics(model.lm_head, h, h, i[0], chunk))",
    ),
    Mutation(
        "ablate: unflagged control branch not added",
        "probe_ablate.py",
        'if eligible and (layer, kind) not in flagged_set and r["sites"][kind]["max_abs"] <= threshold:',
        "if False:",
    ),
    Mutation(
        "ablate: all flagged branches kept (cap ignored)",
        "probe_ablate.py",
        "reverse=True,\n    )[:max_branches]",
        "reverse=True,\n    )",
    ),
    # ---- probe_prompts.py
    Mutation(
        "prompts: no truncation to max_tokens",
        "probe_prompts.py",
        "arr = np.asarray(ids[:max_tokens], dtype=np.int64)",
        "arr = np.asarray(ids, dtype=np.int64)",
    ),
    Mutation(
        "prompts: sink token not prepended to plain text",
        "probe_prompts.py",
        'ids = ([bos_id] if bos_id is not None else []) + tok.encode_plain(r["text"])',
        'ids = tok.encode_plain(r["text"])',
    ),
    # ---- activation_probe.py
    Mutation(
        "probe: attention layers listed with a 1-based index",
        "activation_probe.py",
        '"attention_layers_0based": [i for i, c in enumerate(layers) if c == "V"],',
        '"attention_layers_0based": [i for i, c in enumerate(layers, 1) if c == "V"],',
    ),
    Mutation(
        "loader: modified modeling file accepted",
        "activation_probe.py",
        "if not all(same.values()):",
        "if False:",
    ),
    Mutation(
        "loader: missing / unexpected tensors accepted",
        "activation_probe.py",
        "if missing or unexpected:",
        "if False:",
    ),
    # ---- probe_claims.py
    Mutation(
        "claims: 1-based reading computed as +1",
        "probe_claims.py",
        "one_based = sorted(x - 1 for x in OVH_MAMBA_LAYERS)",
        "one_based = sorted(x + 1 for x in OVH_MAMBA_LAYERS)",
    ),
    Mutation(
        "claims: head identity always consistent",
        "probe_claims.py",
        "_tally([hd[layer] == OVH_HEADS[layer] for layer in OVH_MAMBA_LAYERS]),",
        "_tally([True]),",
    ),
    Mutation(
        "claims: residual bound not enforced",
        "probe_claims.py",
        "CONSISTENT if res_max[0] <= 0.695 else NOT_CONSISTENT",
        "CONSISTENT",
    ),
    Mutation(
        "claims: absent layers judged instead of not judged",
        "probe_claims.py",
        "        if absent:\n",
        "        if False:\n",
    ),
]


def run_pytest(cwd: Path) -> tuple[int, str]:
    """Run the tests in ``cwd`` with the analysis virtual environment; return (exit code, output tail)."""
    py = ANALYSIS / ".venv" / "bin" / "python"
    cmd = [str(py if py.exists() else sys.executable), "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "tests"]
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    return p.returncode, (p.stdout + p.stderr)[-600:]


def make_copy(dst: Path) -> Path:
    """Copy ``analysis/`` to ``dst/analysis`` and link the real ``quant`` as ``dst/quant`` (read only use)."""
    ignore = shutil.ignore_patterns(".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "results")
    shutil.copytree(ANALYSIS, dst / "analysis", ignore=ignore)
    (dst / "quant").symlink_to(QUANT, target_is_directory=True)
    return dst / "analysis"


def main() -> int:
    """Run every mutation; write the report; exit 1 on any survivor or a failing baseline."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ANALYSIS / "results" / "mutation_report.json")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    report: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as tmp:
        base = make_copy(Path(tmp) / "baseline")
        t0 = time.time()
        code, tail = run_pytest(base)
        print(f"baseline: exit {code} in {time.time() - t0:.0f}s")
        if code != 0:
            print(tail)
            return 1
        for m in MUTATIONS:
            if args.only and args.only not in m.name:
                continue
            root = Path(tmp) / f"m{len(report)}"
            work = make_copy(root)
            target = work / m.path
            text = target.read_text()
            if text.count(m.old) != 1:
                print(f"PATTERN ERROR {m.name}: found {text.count(m.old)} times in {m.path}")
                return 1
            target.write_text(text.replace(m.old, m.new))
            t1 = time.time()
            code, tail = run_pytest(work)
            caught = code != 0
            report.append(
                {"name": m.name, "file": m.path, "caught": caught, "exit": code, "seconds": round(time.time() - t1, 1)}
            )
            print(f"{'CAUGHT  ' if caught else 'SURVIVED'} {m.name} ({time.time() - t1:.0f}s)")
            shutil.rmtree(root, ignore_errors=True)
    survivors = [r["name"] for r in report if not r["caught"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"n": len(report), "caught": len(report) - len(survivors), "survivors": survivors, "mutations": report},
            indent=1,
        )
        + "\n"
    )
    print(f"{len(report) - len(survivors)}/{len(report)} caught; survivors: {survivors or 'none'}")
    return 1 if survivors else 0


if __name__ == "__main__":
    raise SystemExit(main())
