"""Mutation check of the proto tests: apply one edit at a time in a throw-away copy; every edit must make pytest fail."""

import pathlib
import shutil
import subprocess
import sys
import tempfile

SRC = pathlib.Path(__file__).resolve().parents[1]
MUTATIONS = [
    (
        "gptq.py",
        "            w1[:, i:] -= err.unsqueeze(1) * hinv1[i, i:].unsqueeze(0)\n",
        "            pass\n",
        "no in-block error propagation",
    ),
    (
        "gptq.py",
        "        wc[:, i2:] -= err1 @ upper[i1:i2, i2:]\n",
        "        pass\n",
        "no cross-block error propagation",
    ),
    (
        "gptq.py",
        "        restored[:, perm] = out\n",
        "        restored = out\n",
        "act-order permutation not undone",
    ),
    (
        "gptq.py",
        '    if damp_mode == "mean":\n        h += percdamp * diag.mean() * torch.eye(n, dtype=h.dtype)\n    else:\n        h += percdamp * torch.diag(diag)\n',
        "    h += 0.0 * torch.eye(n, dtype=h.dtype)\n",
        "no damping",
    ),
    (
        "gptq.py",
        "    if not bool(torch.isfinite(h).all()):\n        return None\n",
        "    if False:\n        return None\n",
        "non-finite Hessian accepted (equivalent mutant: the Cholesky check catches NaN)",
    ),
    (
        "pipeline.py",
        "torch.nonzero((idx == e).any(dim=1)).flatten()",
        "torch.arange(x.shape[0])",
        "routed Hessian uses every token",
    ),
    (
        "routing.py",
        "    picked = torch.gather(scores, 1, idx)\n",
        "    picked = torch.gather(scores + expert_bias.float(), 1, idx)\n",
        "bias leaks into gate weights",
    ),
    (
        "schemes.py",
        "return (torch.round(x / s).clamp(-8, 7) * s).to(wg.dtype)",
        "return (torch.round(x / s).clamp(-7, 7) * s).to(wg.dtype)",
        "int4 clamp changed",
    ),
    (
        "schemes.py",
        "scale_q = scale.clamp(min=0.0, max=E4M3_MAX).to(torch.float8_e4m3fn).to(torch.float32)",
        "scale_q = scale.clamp(min=0.0, max=E4M3_MAX)",
        "no e4m3 rounding of the block scale",
    ),
    (
        "pipeline.py",
        "use_gptq = method.gptq and rows.numel() >= max(floor, method.min_tokens)",
        "use_gptq = method.gptq",
        "min_tokens fallback ignored",
    ),
]
survived = []
for fname, old, new, label in MUTATIONS:
    work = pathlib.Path(tempfile.mkdtemp(prefix="mutwork_"))
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(
        SRC,
        work,
        ignore=shutil.ignore_patterns(".venv", "results", "__pycache__", ".pytest_cache", ".ruff_cache"),
    )
    f = work / "moegptq" / fname
    text = f.read_text()
    if old not in text:
        print(f"SKIPPED (pattern not found): {label}")
        survived.append(label + " [pattern not found]")
        continue
    f.write_text(text.replace(old, new, 1))
    r = subprocess.run(
        [
            "uv",
            "run",
            "--offline",
            "--project",
            str(SRC),
            "python",
            "-m",
            "pytest",
            "-q",
            "-x",
            "-p",
            "no:cacheprovider",
            str(work / "tests"),
        ],
        capture_output=True,
        text=True,
        cwd=work,
        env={**__import__("os").environ, "PYTHONPATH": str(work)},
    )
    caught = r.returncode != 0
    print(("CAUGHT   " if caught else "SURVIVED ") + label)
    if not caught:
        survived.append(label)
print("survivors:", survived)
sys.exit(1 if survived else 0)
