#!/usr/bin/env bash
# Regenerate every number quoted in README.md and DECISIONS.md (nothing there is typed by hand).
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m olala_quant.budget --header tests/data/olala_header.json --out reports/budget.json
uv run python tools/bench_quantize.py --device cpu > reports/bench_quantize_cpu.json
uv run python tools/mutation_check.py --out reports/mutation_report.json
