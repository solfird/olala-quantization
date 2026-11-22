#!/usr/bin/env bash
# The serving half of the gguf mode of arm_pipeline.sh: make sure the fork is built and the GGUF file is there (gguf_setup.sh
# is idempotent), start llama-server (serve_gguf.sh), pass the smoke gate (smoke_gguf.sh), then append the provenance
# (gguf_provenance.sh) to the run's provenance file. run_accuracy.sh is run by the caller.
# Usage: gguf_arm.sh RUNNAME VARIANT_OR_FILE PROVENANCE_FILE
#   VARIANT_OR_FILE  q5_k_m, q6_k, q8_0-safe2, bf16, or a GGUF file name / path (resolved against the pins)
# Env: GGUF_SERVE_ARGS extra options for serve_gguf.sh, e.g. "--np 16 --ctx 524288"; GGUF_SMOKE_ARGS for smoke_gguf.sh,
#      e.g. "--parallel 12". The smoke output goes to /workspace/logs/RUNNAME.smoke.txt.
# Exit 0 gate passed (smoke exit 0 or 2: warnings are read, not fatal, as in arm_pipeline.sh), 1 a step failed.
set -u
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=common.sh
. "$here/common.sh"
run=${1:?RUNNAME}; what=${2:?VARIANT_OR_FILE}; prov=${3:?PROVENANCE_FILE}
say() { echo "[$(date -u +%H:%M:%S)] $*"; }
fail() { say "FAILED: $*"; exit 1; }

variant=$(python3 -I "$here/gguf_tool.py" resolve "$what" 2>&1) || fail "$variant"
say "== $run: gguf variant $variant"
read -ra serve_args <<< "${GGUF_SERVE_ARGS:-}"
read -ra smoke_args <<< "${GGUF_SMOKE_ARGS:-}"

say "== setup (idempotent): fork build and GGUF file"
bash "$here/gguf_setup.sh" --variant "$variant" 2>&1 | tail -n 3
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "gguf_setup.sh"

say "== serve"
bash "$here/serve_gguf.sh" --variant "$variant" ${serve_args[@]+"${serve_args[@]}"} 2>&1 | tail -n 3
[ "${PIPESTATUS[0]}" -eq 0 ] || fail "serve_gguf.sh"

say "== smoke gate"
mkdir -p "$LOG_ROOT"
bash "$here/smoke_gguf.sh" --timeout 900 ${smoke_args[@]+"${smoke_args[@]}"} > "$LOG_ROOT/$run.smoke.txt" 2>&1
rc=$?
say "smoke exit $rc: $(head -n 2 "$LOG_ROOT/$run.smoke.txt" | tr '\n' ' ')"
{ [ $rc -eq 0 ] || [ $rc -eq 2 ]; } || { tail -n 12 "$LOG_ROOT/$run.smoke.txt"; fail "smoke gate"; }

bash "$here/gguf_provenance.sh" >> "$prov" || fail "gguf_provenance.sh"
say "provenance appended to $prov"
