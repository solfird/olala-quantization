#!/usr/bin/env bash
# Pod-side wrapper of CAL-1 (self-generated reasoning traces of the bf16 model on public prompts, for calibrated NVFP4).
# Usage: pod_cal1.sh [--dry-run] STEP [extra cal1_generate.py arguments]
#   STEP: plan    the tool's own --dry-run: prompt draw, quotas, sources at their pinned revisions, sampling, progress; no server
#         run     draw (if not done), generate against the bf16 vLLM server (resumable), audit, tokenize, write the manifest
#         verify  re-check a finished directory: every sha256, the windows, the 8-gram audit
# Expects the bf16 server of the study serving the weights at CAL1_BASE_URL (runpod/pod/arms.json: port 8000, served name
# "olala"), the weights at /workspace/models/olala-bf16 (tokenizer and chat template) and the bundle at /workspace/bundle.
# The prompt datasets come from the Hugging Face cache or network (`datasets` needed), unless CAL1_POOL_DIR holds
# <source>.jsonl pools. --dry-run prints every command and runs nothing (it creates no directory). The last line of a real
# step is CAL1_<STEP>_DONE or CAL1_<STEP>_FAILED (exit 1 on a leakage refusal inside the tool is reported as FAILED too).
# Env: OLALA_WORK (default /workspace), G1_MODEL, G1_BUNDLE, CAL1_OUT (default $WORK/calib-cal1), CAL1_BASE_URL,
#      CAL1_MODEL (served name, default olala), CAL1_REVISIONS ("source=SHA source=SHA ...", 40-hex dataset revisions:
#      curl -s https://huggingface.co/api/datasets/REPO | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])'),
#      CAL1_PROVENANCE (required for `run`: server version and weights identity, recorded in the manifest),
#      CAL1_WINDOW (default 2048), CAL1_POOL_DIR.
set -u

dry=0
if [ "${1:-}" = "--dry-run" ]; then dry=1; shift; fi
step=${1:-}
[ -n "$step" ] || { echo "usage: $0 [--dry-run] plan|run|verify [args]" >&2; exit 2; }
shift

WORK=${OLALA_WORK:-/workspace}
MODEL_DIR=${G1_MODEL:-$WORK/models/olala-bf16}
BUNDLE=${G1_BUNDLE:-$WORK/bundle}
G=$BUNDLE/quant/gptq
OUT=${CAL1_OUT:-$WORK/calib-cal1}
URL=${CAL1_BASE_URL:-http://127.0.0.1:8000/v1}
SERVED=${CAL1_MODEL:-olala}
WINDOW=${CAL1_WINDOW:-2048}
LOGDIR=$WORK/logs/cal1
PY=${PYTHON:-python3}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ENVV="PYTHONPATH=$BUNDLE/quant:$G"

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

revargs() {
  local r
  for r in ${CAL1_REVISIONS:-}; do printf -- '--revision %s ' "$r"; done
}

need_inputs() {
  [ $dry -eq 1 ] && return 0
  [ -f "$MODEL_DIR/config.json" ] || die "$MODEL_DIR/config.json not found: the bf16 weights (tokenizer, chat template) are expected at $MODEL_DIR (olala-bf16)"
  [ -d "$G" ] || die "$G not found: extract the bundle at $BUNDLE"
}

finish() {  # STEP RC
  if [ $dry -eq 1 ]; then say "CAL1_$1_DRY_RUN (nothing was run)"; return "$2"; fi
  if [ "$2" -eq 0 ]; then say "CAL1_$1_DONE"; else say "CAL1_$1_FAILED"; fi
  return "$2"
}

common() {
  # shellcheck disable=SC2046
  echo --out "$OUT" --eval-items "$G/data/eval_items.json" --model "$SERVED" --base-url "$URL" \
    --tokenizer "$MODEL_DIR" --window "$WINDOW" $(revargs) ${CAL1_POOL_DIR:+--pool-dir "$CAL1_POOL_DIR"}
}

do_plan() {
  need_inputs
  # shellcheck disable=SC2046,SC2086
  run plan $PY "$G/cal1_generate.py" $(common) --dry-run "$@"
}

do_run() {
  [ -n "${CAL1_PROVENANCE:-}" ] || die "CAL1_PROVENANCE is required for a real run (server version and weights identity; recorded in the manifest)"
  need_inputs
  # shellcheck disable=SC2046,SC2086
  run generate $PY "$G/cal1_generate.py" $(common) --server-provenance "$CAL1_PROVENANCE" "$@"
}

do_verify() {
  need_inputs
  run verify $PY "$G/cal1_generate.py" --out "$OUT" --verify
}

case $step in
  plan) do_plan "$@"; finish PLAN $? ;;
  run) do_run "$@"; finish RUN $? ;;
  verify) do_verify; finish VERIFY $? ;;
  *) echo "unknown step: $step" >&2; exit 2 ;;
esac
