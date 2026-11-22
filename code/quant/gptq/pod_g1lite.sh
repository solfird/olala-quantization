#!/usr/bin/env bash
# Pod-side wrapper of G1-lite (calibrated NVFP4 routed experts of Olala-v1.0-mini-7A1B).
# Usage: pod_g1lite.sh [--dry-run] STEP [extra build_g1lite.py arguments]
#   STEP: check        versions, GPU, vendored modeling file identical to the weights' one (installs nothing but einops==0.8.1 --no-deps)
#         tokenize     calibration text bundle -> token windows with the model's tokenizer, then re-audit (8-gram leakage);
#                      NOT needed (nothing is run) when G1_CALIB is a CAL-1 output (manifest schema cal1/1 with windows)
#         feasibility  the 30-minute session: calibrate layers 2,4,6 on 4 windows, no checkpoint; seconds per layer + peak memory
#         build        the full G1-lite build into /workspace/ckpt/g1lite
#         verify       python -m olala_quant.verify --calibrated on the result
#         all          check, tokenize, build, verify
# Expects /workspace/models/olala-bf16 (weights, config, tokenizer, modeling files) and the bundle extracted at
# /workspace/bundle (runpod/build_bundle.sh puts quant/gptq in it, with data/calib-lite). Logs: /workspace/logs/g1lite/.
# --dry-run prints every command and runs nothing (it creates no directory). The last line of a real step is
# G1LITE_<STEP>_DONE or G1LITE_<STEP>_FAILED. No network, no pip install other than the pinned einops.
# Env: OLALA_WORK (default /workspace), G1_MODEL, G1_BUNDLE, G1_DST, G1_LAYERS_FEAS (default 2,4,6),
#      G1_CALIB      calibration directory, its manifest.json is passed as --calib (default: the bundle's data/calib-lite;
#                    for CAL-1 the directory written by cal1_generate.py, e.g. /workspace/cal1/calib-cal1),
#      G1_BASE_SELECT  spec for build_g1lite.py --base-select and, in verify, olala_quant.verify --select (default empty:
#                    the N2 base; 'experts' = routed experts calibrated, every other tensor bf16 = the N5/G1 base).
set -u

dry=0
if [ "${1:-}" = "--dry-run" ]; then dry=1; shift; fi
step=${1:-}
[ -n "$step" ] || { echo "usage: $0 [--dry-run] check|tokenize|feasibility|build|verify|all [args]" >&2; exit 2; }
shift

WORK=${OLALA_WORK:-/workspace}
MODEL=${G1_MODEL:-$WORK/models/olala-bf16}
BUNDLE=${G1_BUNDLE:-$WORK/bundle}
G=$BUNDLE/quant/gptq
CALIB=${G1_CALIB:-$G/data/calib-lite}
DST=${G1_DST:-$WORK/ckpt/g1lite}
LOGDIR=$WORK/logs/g1lite
PY=${PYTHON:-python3}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
BASE_SELECT=${G1_BASE_SELECT:-}
BASE_ARGS=(); [ -n "$BASE_SELECT" ] && BASE_ARGS=(--base-select "$BASE_SELECT")
SELECT_ARGS=(); [ -n "$BASE_SELECT" ] && SELECT_ARGS=(--select "$BASE_SELECT")
ENVV="OLALA_TORCH_MOE=1 OLALA_TORCH_MAMBA3=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$BUNDLE/quant:$G"

say() { echo "$*"; }
die() { echo "error: $*" >&2; exit 1; }

# run NAME CMD...: print the command; unless --dry-run run it under env, logging to LOGDIR/NAME-STAMP.log
run() {
  local name=$1; shift
  say "+ env $ENVV $*   # log: $LOGDIR/$name-$STAMP.log"
  [ $dry -eq 1 ] && return 0
  mkdir -p "$LOGDIR"
  # shellcheck disable=SC2086
  env $ENVV "$@" 2>&1 | tee -a "$LOGDIR/$name-$STAMP.log"
  return "${PIPESTATUS[0]}"
}

need_inputs() {
  [ $dry -eq 1 ] && return 0
  [ -f "$MODEL/config.json" ] || die "$MODEL/config.json not found: the bf16 weights are expected at $MODEL (olala-bf16)"
  [ -d "$G" ] || die "$G not found: extract the bundle at $BUNDLE"
}

finish() {  # STEP RC
  if [ $dry -eq 1 ]; then say "G1LITE_$1_DRY_RUN (nothing was run)"; return "$2"; fi
  if [ "$2" -eq 0 ]; then say "G1LITE_$1_DONE"; else say "G1LITE_$1_FAILED"; fi
  return "$2"
}

do_check() {
  need_inputs
  if [ $dry -eq 0 ]; then
    $PY -c 'import einops' 2> /dev/null || $PY -m pip install --no-deps einops==0.8.1 || return 1
  else
    say "+ $PY -c 'import einops' || $PY -m pip install --no-deps einops==0.8.1"
  fi
  run check nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
  run check $PY -c "import torch, transformers, einops, safetensors; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__, 'einops', einops.__version__, 'safetensors', safetensors.__version__)" || return 1
  run check $PY -c "import json,sys; sys.path.insert(0,'$G'); from olala_model import check_vendored_matches; m=check_vendored_matches(__import__('pathlib').Path('$MODEL')); print('vendored modeling files identical to the weights copy:', m); sys.exit(0 if all(m.values()) else 1)"
}

cal1_windows_present() {  # the manifest of a finished CAL-1 build (cal1_generate.py) with its windows
  [ -f "$CALIB/manifest.json" ] && grep -q '"schema": "cal1/1"' "$CALIB/manifest.json" \
    && grep -q '"windows": {' "$CALIB/manifest.json"
}

do_tokenize() {
  if cal1_windows_present; then
    say "tokenize: not needed, $CALIB is a CAL-1 output with its token windows (audited by cal1_generate.py); nothing run"
    return 0
  fi
  need_inputs
  run tokenize $PY "$G/calib_data.py" tokenize --out "$CALIB" --tokenizer "$MODEL" --n-tokens 65536 --window 2048 || return 1
  run tokenize $PY "$G/leakage_audit.py" --items "$CALIB/eval_items.json" --bundle "$CALIB" --report "$CALIB/leakage_audit.json"
}

do_feasibility() {
  need_inputs
  run feasibility $PY "$G/build_g1lite.py" --src "$MODEL" --calib "$CALIB/manifest.json" --dst "$WORK/ckpt/g1lite-feas" \
    ${BASE_ARGS[@]+"${BASE_ARGS[@]}"} --device cuda --layers "${G1_LAYERS_FEAS:-2,4,6}" --max-windows 4 --heldout-every 4 --batch-windows 1 \
    --attn-impl eager --calibrate-only --restart "$@" || return 1
  [ $dry -eq 1 ] && return 0
  $PY - "$WORK/ckpt/g1lite-feas/g1lite_report.json" << 'PYEOF'
import json, sys
rep = json.load(open(sys.argv[1]))
print("every layer that was run (forward only, or captured):")
print("layer  captured  forward_s  mixer_s  mlp_s  peak_cuda_GB  max|mixer_out|  max|mlp_out|")
for r in rep["forward_stats"]:
    peak = (r["peak_cuda_allocated_bytes"] or 0) / 1e9
    print(f"{r['layer']:>5}  {str(r['captured']):>8}  {r['forward_seconds']:9.1f}  {r['mixer_seconds']:7.1f}  {r['mlp_seconds']:5.1f}  {peak:12.2f}  {r['max_abs_mixer_out']:13.3g}  {r['max_abs_mlp_out']:11.3g}")
print("calibrated layers:")
print("layer  hessian_s  solve_s  total_s  peak_after_solve_GB  gptq/rtn  mean_loss_ratio_up  mean_heldout_ratio")
for i, lay in rep["layers"].items():
    w, s, m = lay["wall_seconds"], lay["summary"], lay["memory"]
    peak = (m["peak_cuda_allocated_bytes_after_solve"] or 0) / 1e9
    print(f"{i:>5}  {w['hessians']:9.1f}  {w['solve']:7.1f}  {w['total']:7.1f}  {peak:19.2f}  {s['n_gptq']:>3}/{s['n_rtn']:<3}  {s['mean_loss_ratio_up']}  {s['mean_heldout_err_ratio']}")
PYEOF
}

do_build() {
  need_inputs
  run build $PY "$G/build_g1lite.py" --src "$MODEL" --calib "$CALIB/manifest.json" --dst "$DST" \
    ${BASE_ARGS[@]+"${BASE_ARGS[@]}"} --device cuda \
    --batch-windows 4 --propagate bf16 --attn-impl eager "$@"
}

do_verify() {
  need_inputs
  run verify $PY -m olala_quant.verify --src "$MODEL" --dst "$DST" --calibrated \
    ${SELECT_ARGS[@]+"${SELECT_ARGS[@]}"} --report "$LOGDIR/verify-$STAMP.json"
}

case $step in
  check) do_check; finish CHECK $? ;;
  tokenize) do_tokenize; finish TOKENIZE $? ;;
  feasibility) do_feasibility "$@"; finish FEASIBILITY $? ;;
  build) do_build "$@"; finish BUILD $? ;;
  verify) do_verify; finish VERIFY $? ;;
  all) do_check && do_tokenize && do_build "$@" && do_verify; finish ALL $? ;;
  *) echo "unknown step: $step" >&2; exit 2 ;;
esac
