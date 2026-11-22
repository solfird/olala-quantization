"""Rounding error of random Gaussian weights, with the same code that measured the real weights.

Run from code/quant/ (needs torch, see its pyproject):  cd code/quant && uv run python ../../analysis/gaussian_reference.py
Writes results/gaussian-reference.json: the relative round-to-nearest error of a Gaussian matrix for each format, the
reference against which the per-class errors of the real checkpoint (results/wave1/analysis/weight_stats) are read.
"""

import json
from pathlib import Path

import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "quant"))
from olala_quant import weight_stats as ws  # noqa: E402

torch.manual_seed(0)
t = torch.randn(256, 768, 384, dtype=torch.float32).to(torch.bfloat16)
r = ws.measure_tensor("gaussian", t, "routed_up", 0)
out = {"shape": list(t.shape), "seed": 0, "rel_fro": {k: v["rel_fro"] for k, v in r["formats"].items() if v}}
p = Path(__file__).resolve().parents[1] / "results" / "gaussian-reference.json"
p.write_text(json.dumps(out, indent=1) + "\n")
print(out)
