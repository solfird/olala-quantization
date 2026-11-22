#!/usr/bin/env bash
# One arm end to end on one pod: wait for the evals environment, (patch and convert for an offline checkpoint), serve,
# smoke-gate, record provenance, then run the accuracy harness. Detach it; the log is the record.
# Usage: arm_pipeline.sh RUNNAME online:ARM            e.g. F1  online:fp8pc
#        arm_pipeline.sh RUNNAME ckpt:DIR:RECIPE       e.g. F2  ckpt:/workspace/ckpt/fp8-I:fp8  (RECIPE fp8 or nvfp4)
#        arm_pipeline.sh RUNNAME ckptsel:DIR:RECIPE:SPEC  e.g. A1  ckptsel:/workspace/ckpt/abl-A1:nvfp4:experts:layers=0-11
#          like ckpt: but the converter runs with --select SPEC (a component ablation: NAME or NAME:layers=A-B[,N...], see
#          quant/olala_quant/ablation.py; SPEC may contain ':' '=' ',' '-', DIR and RECIPE are split off the front; DIR is
#          absolute, RECIPE fp8, nvfp4 or nvfp4_w4a4). Checked BEFORE anything else (even the wait for the evals environment):
#          the form of the mode, `olala_quant.ablation expected` (a bad selection is refused), and a DIR that already holds a
#          checkpoint must record the same selection (a stale checkpoint is refused). Then olala_quant.verify --select SPEC and
#          `olala_quant.ablation check` must pass before serving; a failure fails the pipeline. The provenance records the
#          selection, the expected counts and the sha256 of the checkpoint's index, config and quantization_info.json.
#        arm_pipeline.sh RUNNAME gguf:FILE             e.g. G5  gguf:q5_k_m  (OVH's published GGUF on OVH's llama.cpp fork;
#          FILE is a variant q5_k_m|q6_k|q8_0-safe2|bf16 or a pinned GGUF file name; gguf_arm.sh builds the fork and
#          fetches the file if missing, serves, smoke-gates and records the provenance; env GGUF_SERVE_ARGS="--np 16")
#        extra arguments after the mode go to run_accuracy.sh (e.g. --effort high --tasks "mgsm aime2025")
# Exit: 0 run done, 1 a gate or the run failed. The last line is ARM_PIPELINE_DONE or ARM_PIPELINE_FAILED.
set -u
. "$(dirname "$0")/job_common.sh"
run=${1:?RUNNAME}; mode=${2:?online:ARM or ckpt:DIR:RECIPE}; shift 2
extra=("$@")
fail() { log "FAILED: $*"; echo ARM_PIPELINE_FAILED; exit 1; }

if [[ $mode == ckptsel:* ]]; then
  # SPEC of ckptsel: (olala_quant/ablation.py): NAME or NAME:layers=A-B[,N...]; the names are those of ablation.SELECT_NAMES
  sel_re='(all|experts|non-experts|shared-experts|latent|attention|mamba)(:layers=[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*)?'
  ckptsel_re="^ckptsel:(/[^:[:space:]]+):(fp8|nvfp4|nvfp4_w4a4):($sel_re)\$"
  [[ $mode =~ $ckptsel_re ]] \
    || fail "bad ckptsel mode '$mode' (ckptsel:/ABS/DIR:fp8|nvfp4|nvfp4_w4a4:SPEC, SPEC = NAME or NAME:layers=A-B[,N...])"
  dir=${BASH_REMATCH[1]}; recipe=${BASH_REMATCH[2]}; selspec=${BASH_REMATCH[3]}
  ckptsel_precheck "$dir" "$selspec" || fail "$CKPTSEL_ERR"   # CPU only: nothing has touched the GPU yet
fi

log "== $run: wait for the evals environment"
for _ in $(seq 1 120); do
  grep -q EVALS_ENV_DONE /workspace/logs/setup/evals-env-*.log 2> /dev/null && break
  grep -q EVALS_ENV_FAILED /workspace/logs/setup/evals-env-*.log 2> /dev/null && fail "evals environment setup failed"
  sleep 10
done
grep -q EVALS_ENV_DONE /workspace/logs/setup/evals-env-*.log 2> /dev/null || fail "evals environment not ready after 20 minutes"

prov=/workspace/results/provenance-$run.txt
mkdir -p /workspace/results
{ echo "run=$run mode=$mode started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; } > "$prov"

bash "$B/pod/stop_server.sh" > /dev/null 2>&1; sleep 2   # idempotent: by pid file, nothing running is fine
case $mode in
  online:*)
    arm=${mode#online:}
    bash "$B/pod/serve_olala.sh" "$arm" 2>&1 | tail -n 2
    logname=$arm
    ;;
  ckpt:*)
    rest=${mode#ckpt:}; dir=${rest%%:*}; recipe=${rest##*:}
    apply_patch || fail "loader patch"
    if [ ! -f "$dir/model.safetensors.index.json" ]; then
      (cd "$QU" && PYTHONPATH="$QU" python3 -m olala_quant.convert --src /workspace/models/olala-bf16 --dst "$dir" \
        --recipe "$recipe" --device cuda 2>&1 | tail -n 2) || fail "convert"
    fi
    (cd "$dir" && sha256sum model.safetensors.index.json config.json quantization_info.json) >> "$prov"
    bash "$B/pod/serve_olala.sh" ckpt "$dir" 2>&1 | tail -n 2
    logname="ckpt-$(basename "$dir")"
    ;;
  ckptsel:*)
    # DIR, RECIPE and SPEC were split off the front, and the selection checked, before the wait for the evals environment
    ckptsel_provenance "$selspec" >> "$prov"
    tmpout=$(mktemp); trap 'rm -f "$tmpout"' EXIT
    apply_patch || fail "loader patch"
    if [ ! -f "$dir/model.safetensors.index.json" ]; then
      log "convert $dir ($recipe, --select $selspec)"
      (cd "$QU" && PYTHONPATH="$QU" python3 -m olala_quant.convert --src /workspace/models/olala-bf16 --dst "$dir" \
        --recipe "$recipe" --device cuda --select "$selspec" > "$tmpout" 2>&1) \
        || { tail -n 4 "$tmpout"; fail "convert (olala_quant.convert $recipe --select $selspec)"; }
      tail -n 2 "$tmpout"
    fi
    log "verify $dir"
    (cd "$QU" && PYTHONPATH="$QU" python3 -m olala_quant.verify --src /workspace/models/olala-bf16 --dst "$dir" \
      --report "/workspace/results/verify-$run.json" --select "$selspec" > "$tmpout" 2>&1) \
      || { tail -n 8 "$tmpout"; fail "verify (olala_quant.verify --select $selspec exit status not 0)"; }
    echo "verify: $(tail -n 1 "$tmpout" | cut -c1-300)" >> "$prov"
    log "ablation check $dir"
    (cd "$QU" && PYTHONPATH="$QU" python3 -m olala_quant.ablation check --src /workspace/models/olala-bf16 --dst "$dir" \
      --select "$selspec" > "$tmpout" 2>&1) \
      || { tail -n 6 "$tmpout"; fail "ablation check (olala_quant.ablation check: the checkpoint does not match the selection $selspec)"; }
    echo "ablation_check: $(tail -n 1 "$tmpout" | cut -c1-300)" >> "$prov"
    (cd "$dir" && sha256sum model.safetensors.index.json config.json quantization_info.json) >> "$prov" \
      || fail "provenance (sha256 of the checkpoint files)"
    bash "$B/pod/serve_olala.sh" ckpt "$dir" > "$tmpout" 2>&1 || { tail -n 4 "$tmpout"; fail "serve (serve_olala.sh ckpt $dir)"; }
    tail -n 2 "$tmpout"
    logname="ckpt-$(basename "$dir")"
    ;;
  gguf:*)
    # llama-server instead of vllm: its own smoke gate and provenance (commit, file sha256, flags); run_accuracy.sh
    # reads the GGUF path and the commit from the server's .cmd file (serve_gguf.sh writes them)
    bash "$B/pod/gguf_arm.sh" "$run" "${mode#gguf:}" "$prov" || fail "gguf serve, smoke gate or provenance"
    gguf_done=1
    ;;
  *) fail "unknown mode $mode" ;;
esac

if [ -z "${gguf_done:-}" ]; then
  gate_log="$logname"
  bash "$B/pod/smoke.sh" "$gate_log" --timeout 900 > "/workspace/logs/$run.smoke.txt" 2>&1
  rc=$?
  log "smoke exit $rc: $(head -n 2 "/workspace/logs/$run.smoke.txt" | tr '\n' ' ')"
  { [ $rc -eq 0 ] || [ $rc -eq 2 ]; } || { tail -n 8 "/workspace/logs/$run.smoke.txt"; fail "smoke gate"; }
  grep -h -E 'Quantizing [0-9]+ layers|Selected .*Kernel|Using .* MoE backend' "$(ls -t "/workspace/logs/$gate_log"/server-*.log | head -n 1)" \
    | cut -c1-300 >> "$prov"
fi
log "provenance in $prov"

log "== run_accuracy $run"
bash "$B/pod/run_accuracy.sh" "$run" --effort medium "${extra[@]}"
rc=$?
log "run_accuracy exit $rc"
[ $rc -eq 0 ] || fail "run_accuracy exit $rc"
echo ARM_PIPELINE_DONE
