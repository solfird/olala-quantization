#!/usr/bin/env bash
# Rebuild every derived number file, table and figure of the report from the shipped result files and compare them with the shipped copies.
#
#   ./reproduce.sh              rebuild in a temporary folder; nothing in the repository is written
#   ./reproduce.sh --in-place   write the rebuilt files over the shipped ones (then `git diff --stat results/ report/ figures/` shows the change)
#
# Needs: bash, uv (https://docs.astral.sh/uv/) and the network once, to fetch Python 3.12, numpy and matplotlib. No GPU, no model, no key.
# The sources of the study did not pin the plotting library; this repository does: matplotlib 3.11.2 and numpy 2.5.5 drew the figures of the report.
# Exit status 0 when every number file is identical; PDF figures are never compared byte for byte (a PDF carries its build time) and PNG figures
# are compared by bytes first and by pixels second (a different matplotlib build or font rendering can change the bytes of an identical drawing).
set -euo pipefail
cd "$(dirname "$0")"

PYVER=3.12
NUMPY=numpy==2.5.5
MPL=matplotlib==3.11.2
PY="uv run --no-project --python $PYVER --with $NUMPY python -B"
PYMPL="uv run --no-project --python $PYVER --with $NUMPY --with $MPL python -B"
export PYTHONDONTWRITEBYTECODE=1

IN_PLACE=0
[ "${1:-}" = "--in-place" ] && IN_PLACE=1
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
if [ "$IN_PLACE" = 1 ]; then
  DATA=report; TABLES=report; FIGS=figures; RES=results
else
  DATA="$TMP/data"; TABLES="$TMP/tables"; FIGS="$TMP/figures"; RES="$TMP/results"
fi
mkdir -p "$DATA" "$TABLES" "$FIGS" "$RES"
fail=0

same() {  # same SHIPPED REBUILT LABEL
  if [ "$IN_PLACE" = 1 ]; then echo "written     $3"; return; fi
  if cmp -s "$1" "$2"; then echo "identical   $3"; else echo "DIFFERS     $3"; fail=1; fi
}

echo "== 1. report/data.json (every number of the text, from results/)"
$PY report/build_data.py --out "$DATA" > "$TMP/build_data.log"
head -n 1 "$TMP/build_data.log"
same report/data.json "$DATA/data.json" "report/data.json"

echo "== 2. report/tab_*.tex"
$PY report/make_tables.py --data "$DATA/data.json" --out "$TABLES" > /dev/null
for f in report/tab_*.tex; do same "$f" "$TABLES/$(basename "$f")" "$f"; done

echo "== 3. figures (matplotlib 3.11.2, numpy 2.5.5)"
$PYMPL report/make_figures.py --data "$DATA/data.json" --out "$FIGS" > /dev/null
if [ "$IN_PLACE" = 1 ]; then
  echo "written     figures/*.pdf figures/*.png"
else
  for n in teaser arch probe ablation speed; do
    if cmp -s "figures/fig_$n.png" "$FIGS/fig_$n.png"; then
      echo "identical   figures/fig_$n.png"
    else
      px=$($PYMPL - "figures/fig_$n.png" "$FIGS/fig_$n.png" <<'PYEOF'
import sys
import numpy as np
from PIL import Image
a, b = (np.asarray(Image.open(p).convert("RGB"), dtype=int) for p in sys.argv[1:3])
print("sizes differ" if a.shape != b.shape else int((np.abs(a - b).sum(axis=2) > 0).sum()))
PYEOF
)
      echo "DIFFERS     figures/fig_$n.png (bytes differ; pixels differing: $px)"; fail=1
    fi
  done
  echo "not compared  figures/*.pdf (a PDF carries its build time)"
fi

echo "== 4. results/review-sensitivity.json and .md"
$PY analysis/review_sensitivity.py --json "$RES/review-sensitivity.json" --md "$RES/review-sensitivity.md" > /dev/null
same results/review-sensitivity.json "$RES/review-sensitivity.json" "results/review-sensitivity.json"
same results/review-sensitivity.md "$RES/review-sensitivity.md" "results/review-sensitivity.md"

echo "== 5. results/b0g-control.json and .md"
$PY analysis/b0g_control.py --json "$RES/b0g-control.json" --md "$RES/b0g-control.md" > /dev/null
same results/b0g-control.json "$RES/b0g-control.json" "results/b0g-control.json"
same results/b0g-control.md "$RES/b0g-control.md" "results/b0g-control.md"

echo "== 6. throughput summaries"
$PY analysis/bench_summary.py --bench-dir results/bench --out "$RES/bench-summary.md" > /dev/null
same results/bench-summary.md "$RES/bench-summary.md" "results/bench-summary.md"
$PY analysis/gguf_bench_summary.py --sweep Q4sw2=results/wave1 --sweep Q4-a=results/wave1 --out "$RES/bench-gguf-summary.md" --json "$RES/bench-gguf-summary.json" > /dev/null
same results/bench-gguf-summary.json "$RES/bench-gguf-summary.json" "results/bench-gguf-summary.json"
same results/bench-gguf-summary.md "$RES/bench-gguf-summary.md" "results/bench-gguf-summary.md"

echo "== 7. results/wave2-*.json and results/wave1-analysis.json (analyze_wave.py, 10,000 resamples)"
# Compares every leaf. The hashes of the per-item files differ by design (the shipped files hold no benchmark text); see the docstring.
if [ "$IN_PLACE" = 1 ]; then
  echo "in-place mode does not rewrite these files: they keep the hashes of the private per-item files"
else
  $PY analysis/recompute_wave2.py || fail=1
fi

echo "== 8. tests of the analysis scripts"
if [ "$IN_PLACE" = 1 ] || uv run --no-project --python $PYVER --with $NUMPY --with pytest==9.2.0 python -B -m pytest -p no:cacheprovider -q analysis/tests > "$TMP/pytest.log" 2>&1; then
  [ "$IN_PLACE" = 1 ] || tail -n 1 "$TMP/pytest.log"
else
  tail -n 5 "$TMP/pytest.log"; fail=1
fi

echo "== 9. README links and numbers"
$PY tools/check_links.py README.md || fail=1
$PY tools/check_readme_numbers.py || fail=1

if [ "$fail" = 0 ]; then echo "RESULT: every check passed"; else echo "RESULT: at least one check failed"; exit 1; fi
