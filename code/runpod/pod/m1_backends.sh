#!/usr/bin/env bash
# M1 second attempt (PROTOCOL amendment 5): the default MXFP4 MoE backend (Marlin) asserts on Olala's expert shapes
# (marlin_utils_fp4.py:602). Try the other backends selectable with --moe-backend, in order; the first that passes the
# gate is collected and compared with the bf16 of this pod (b0, collected by m1_gate.sh). Run detached.
set -u
. "$(dirname "$0")/job_common.sh"
for be in ${M1_BACKENDS:-triton_unfused flashinfer_cutlass emulation}; do
  log "== M1 online mxfp4, --moe-backend $be"
  bash "$B/pod/serve_olala.sh" mxfp4 -- --moe-backend "$be" 2>&1 | tail -n 1
  gate mxfp4 "m1_$be"; rc=$?
  grep -h -E 'MXFP4|mxfp4|moe.backend' "$(ls -t /workspace/logs/mxfp4/server-*.log | head -n 1)" | grep -v Warning | cut -c1-250 | head -n 6
  stop
  if [ $rc -eq 0 ] || [ $rc -eq 2 ]; then compare b0 "m1_$be"; log "M1_BACKEND_DONE $be"; exit 0; fi
done
log "M1_BACKEND_DONE none served"
