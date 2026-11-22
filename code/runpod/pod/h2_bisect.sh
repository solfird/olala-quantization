#!/usr/bin/env bash
# Bisect of the H2 failure (PROTOCOL amendment 6): is the F1-versus-F2 gap in the dense layers or in the fused experts?
# Four fidelity passes with NO list I, so that only one family is quantized at a time:
#   E: experts only      online (--quantization fp8_per_channel + ignore dense)   vs offline (our converter, ignore dense)
#   D: dense layers only online (ignore experts)                                  vs offline (our converter, ignore experts)
# Usage: bash /workspace/bundle/pod/h2_bisect.sh   (run detached; the log is the record)
set -u
. "$(dirname "$0")/job_common.sh"

# vLLM module-name globs (online ignore list) and HF tensor-name globs (converter ignore file)
ONLINE_IGN_DENSE='"model.layers.*.mixer_proj","model.layers.*.gate_proj","model.layers.*.mlp.fc1_latent_proj","model.layers.*.mlp.fc2_latent_proj","model.layers.*.mlp.shared_experts.*"'
ONLINE_IGN_EXPERTS='"model.layers.*.mlp.experts"'
printf '%s\n' '["*mixer_proj.weight","*gate_proj.weight","*mlp.down_proj.weight","*mlp.up_proj.weight","*shared_experts.*"]' > /workspace/ign_dense.json
printf '%s\n' '["*mlp.experts.*"]' > /workspace/ign_experts.json

apply_patch || exit 1
log "== convert offline variants (no list I)"
(cd "$QU" && PYTHONPATH="$QU" python3 -m olala_quant.convert --src /workspace/models/olala-bf16 --dst /workspace/ckpt/fp8-E \
  --recipe fp8 --ignore-file /workspace/ign_dense.json --device cuda 2>&1 | tail -n 2)
(cd "$QU" && PYTHONPATH="$QU" python3 -m olala_quant.convert --src /workspace/models/olala-bf16 --dst /workspace/ckpt/fp8-D \
  --recipe fp8 --ignore-file /workspace/ign_experts.json --device cuda 2>&1 | tail -n 2)

serve_online() {  # NAME IGNORE_GLOBS
  bash "$B/pod/serve_olala.sh" fp8pc_noignore -- --quantization-config "{\"ignore\":[$2]}" 2>&1 | tail -n 1
  gate fp8pc_noignore "$1"
  grep -h 'Quantizing' "$(ls -t /workspace/logs/fp8pc_noignore/server-*.log | head -n 1)" | cut -c1-330 | head -n 2
  stop
}
serve_offline() {  # NAME DIR
  bash "$B/pod/serve_olala.sh" ckpt "$2" 2>&1 | tail -n 1
  gate "ckpt-$(basename "$2")" "$1"
  stop
}

log "== E: experts only"
serve_online onlineE "$ONLINE_IGN_DENSE"
serve_offline offlineE /workspace/ckpt/fp8-E
log "== D: dense only"
serve_online onlineD "$ONLINE_IGN_EXPERTS"
serve_offline offlineD /workspace/ckpt/fp8-D

log "== comparisons (online as reference, offline as variant)"
compare onlineE offlineE
compare onlineD offlineD
log "H2_BISECT_DONE"
