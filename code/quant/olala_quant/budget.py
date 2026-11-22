"""Projected checkpoint sizes from a safetensors header, with and without ignore list I.

Usage::

    python -m olala_quant.budget --header tests/data/olala_header.json --out budget.json

Every size and bits-per-weight number quoted in the documentation comes from this script.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from olala_quant.checkpoint_io import header_from_json
from olala_quant.plan import build_plan
from olala_quant.recipes import DEFAULT_IGNORE_I, RECIPES


def project_all(header_path: Path) -> dict[str, Any]:
    """Project the output size of every implemented recipe, with no ignore list and with list I."""
    header = header_from_json(json.loads(Path(header_path).read_text()))
    total = sum(m.numel for m in header.values())
    source_bytes = sum(m.nbytes for m in header.values())
    rows = []
    for name in sorted(n for n, r in RECIPES.items() if r.implemented):
        for label, patterns in (("none", ()), ("I", DEFAULT_IGNORE_I)):
            plan = build_plan(header, RECIPES[name], patterns)
            out_bytes = plan.output_bytes()
            quantized = [t for t in plan.tensors if t.category == "quantized"]
            q_params = sum(t.meta.numel for t in quantized)
            q_bytes = sum(o.nbytes for t in quantized for o in plan.outputs_of(t))
            rows.append(
                {
                    "recipe": name,
                    "ignore": label,
                    "n_quantized": len(quantized),
                    "n_ignored_quantizable": len(plan.ignored_quantizable),
                    "quantized_params": q_params,
                    "quantized_fraction": q_params / total,
                    "output_bytes": out_bytes,
                    "output_gb": out_bytes / 1e9,
                    "bits_per_weight": out_bytes * 8 / total,
                    "bits_per_quantized_weight": q_bytes * 8 / q_params,
                }
            )
    return {
        "source": {
            "header": str(header_path),
            "n_tensors": len(header),
            "total_params": total,
            "source_bytes": source_bytes,
            "source_gb": source_bytes / 1e9,
            "bits_per_weight": source_bytes * 8 / total,
        },
        "ignore_list_I": list(DEFAULT_IGNORE_I),
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    p = argparse.ArgumentParser(prog="python -m olala_quant.budget", description=__doc__)
    p.add_argument("--header", type=Path, required=True, help="safetensors header JSON")
    p.add_argument("--out", type=Path, help="write the table as JSON")
    args = p.parse_args(argv)
    table = project_all(args.header)
    src = table["source"]
    print(
        f"source: {src['total_params']:,} params, {src['source_gb']:.3f} GB, "
        f"{src['bits_per_weight']:.2f} bits per weight"
    )
    head = ("recipe", "ignore", "quantized", "GB", "bits per weight", "bits per quantized weight")
    print(f"{head[0]:8} {head[1]:6} {head[2]:>9} {head[3]:>8} {head[4]:>16} {head[5]:>26}")
    for r in table["rows"]:
        print(
            f"{r['recipe']:8} {r['ignore']:6} {r['n_quantized']:>9} {r['output_gb']:>8.3f} "
            f"{r['bits_per_weight']:>16.3f} {r['bits_per_quantized_weight']:>26.3f}"
        )
    if args.out:
        args.out.write_text(json.dumps(table, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
