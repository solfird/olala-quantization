#!/usr/bin/env bash
# Arm M1 gate and same-pod references (exploratory, PROTOCOL amendment 5): online MXFP4 with list I, then bf16 and our
# offline NVFP4 weight-only (N2) on the SAME pod so that the three fidelity files share hardware and a time window.
# Usage: bash /workspace/bundle/pod/m1_gate.sh   (run detached; the log is the record)
set -u
. "$(dirname "$0")/job_common.sh"
IGN_I='["model.layers.1.mixer_proj","model.layers.7.mixer_proj","model.layers.23.mixer_proj","model.layers.0.mlp.*","model.layers.1.mlp.*","model.layers.5.mlp.*","model.layers.19.mlp.*"]'

log "== M1 online mxfp4 with list I"
bash "$B/pod/serve_olala.sh" --dry-run mxfp4 2>&1 | grep -E '^command' | cut -c1-700
bash "$B/pod/serve_olala.sh" mxfp4 2>&1 | tail -n 2
gate mxfp4 m1; m1rc=$?
grep -h -i -E 'MXFP4|mxfp4' "$(ls -t /workspace/logs/mxfp4/server-*.log | head -n 1)" | cut -c1-240 | head -n 12
stop

log "== bf16 reference on this pod"
bash "$B/pod/serve_olala.sh" bf16 2>&1 | tail -n 1
gate bf16 b0

log "== N2 offline NVFP4 weight-only (list I)"
stop
apply_patch || exit 1
(cd "$QU" && PYTHONPATH="$QU" python3 -m olala_quant.convert --src /workspace/models/olala-bf16 --dst /workspace/ckpt/nvfp4-I --recipe nvfp4 --device cuda 2>&1 | tail -n 2)
bash "$B/pod/serve_olala.sh" ckpt /workspace/ckpt/nvfp4-I 2>&1 | tail -n 1
gate ckpt-nvfp4-I n2

stop
log "== comparisons against the bf16 of this pod"
{ [ $m1rc -eq 0 ] || [ $m1rc -eq 2 ]; } && compare b0 m1
compare b0 n2
{ [ $m1rc -eq 0 ] || [ $m1rc -eq 2 ]; } && compare n2 m1
log "M1_JOB_DONE (m1 gate exit $m1rc)"
