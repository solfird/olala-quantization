#!/usr/bin/env bash
# One arm's throughput sweep end to end on a FRESH pod: (patch and convert and verify an offline checkpoint), serve,
# smoke-gate, record provenance, then bench_sweep.sh with nothing else on the GPU. Detach it; the log is the record.
# Usage: bench_arm.sh [--dry-run] LABEL online:ARM                    e.g. bf16-bench  online:bf16
#        bench_arm.sh [--dry-run] LABEL ckpt:/ABS/DIR:RECIPE          e.g. F2-bench    ckpt:/workspace/ckpt/fp8-I:fp8
#          (RECIPE fp8, nvfp4 or nvfp4_w4a4; DIR is converted from /workspace/models/olala-bf16 when it has no
#          model.safetensors.index.json, and ALWAYS checked with olala_quant.verify; gguf mode is not supported here)
#        bench_arm.sh [--dry-run] LABEL ckptsel:/ABS/DIR:RECIPE:SPEC   e.g. abl-bench  ckptsel:/workspace/ckpt/abl:nvfp4:experts:layers=0-11
#          (like ckpt: but only the component SPEC is quantized: NAME or NAME:layers=A-B[,N...], olala_quant/ablation.py; SPEC
#          may contain ':' '=' ',' '-', DIR and RECIPE are split off the front. Before anything else runs, `olala_quant.ablation
#          expected` must accept SPEC and a DIR that already holds a checkpoint must record that same selection in its
#          quantization_info.json (a stale checkpoint is refused). The converter and olala_quant.verify get --select SPEC, and
#          `olala_quant.ablation check` must pass BEFORE serving; the provenance records the selection, the expected counts and
#          the sha256 of the checkpoint's index, config and quantization_info.json)
# Env: BENCH_REPEATS (repeats per point, default 2), BENCH_VERIFY_ARGS (extra options of olala_quant.verify, for a
#      calibrated checkpoint: --calibrated), BENCH_STOP_WAIT (seconds to wait after the first stop_server, default 2).
# Files: /workspace/results/provenance-LABEL.txt (run header, GPU, checkpoint sha256, verify verdict, kernel banners),
#   /workspace/results/bench-LABEL/ (bench_sweep.sh's JSON per point, plus copies of the provenance, the smoke output and
#   the sweep output: collect.sh pulls only that directory), /workspace/logs/bench-LABEL.log (the sweep output and, only
#   when the sweep ran to its end and wrote results, the line BENCH_DONE, the line collect.sh waits for).
# Exit: 0 sweep done, 1 a step failed, 2 bad arguments. The last line is BENCH_PIPELINE_DONE or
#   BENCH_PIPELINE_FAILED: REASON. A sweep with failed points still gets BENCH_DONE (the partial data must be pulled) and
#   then BENCH_PIPELINE_FAILED.
# Shares code with arm_pipeline.sh by sourcing job_common.sh (log, apply_patch, B, QU, ckptsel_precheck). Its serve/convert
# steps are copied, with their exit status checked (arm_pipeline.sh's ckpt: mode pipes them into tail, which hides it).
set -u
here=$(cd "$(dirname "$0")" && pwd)
usage="usage: bench_arm.sh [--dry-run] LABEL online:ARM|ckpt:/ABS/DIR:fp8|nvfp4|nvfp4_w4a4|ckptsel:/ABS/DIR:RECIPE:SPEC   (env BENCH_REPEATS, BENCH_VERIFY_ARGS)"
refuse() { echo "$usage" >&2; echo "BENCH_PIPELINE_FAILED: $*"; exit 2; }

dry=0; positional=()
for a in "$@"; do
  case $a in
    --dry-run) dry=1 ;;
    -*) refuse "unknown option $a" ;;
    *) positional+=("$a") ;;
  esac
done
[ ${#positional[@]} -eq 2 ] || refuse "expected LABEL and MODE, got ${#positional[@]} argument(s)"
label=${positional[0]}; mode=${positional[1]}
[[ $label =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || refuse "bad LABEL '$label' (letters, digits, . _ -; it names a directory and a log)"
dir=""; recipe=""; arm=""; selspec=""
# SPEC of ckptsel: (olala_quant/ablation.py): NAME or NAME:layers=A-B[,N...]; the names are those of ablation.SELECT_NAMES
sel_re='(all|experts|non-experts|shared-experts|latent|attention|mamba)(:layers=[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*)?'
ckptsel_re="^ckptsel:(/[^:[:space:]]+):(fp8|nvfp4|nvfp4_w4a4):($sel_re)\$"
if [[ $mode =~ ^online:([A-Za-z0-9_]+)$ ]]; then
  arm=${BASH_REMATCH[1]}
elif [[ $mode =~ ^ckpt:(/[^:[:space:]]+):(fp8|nvfp4|nvfp4_w4a4)$ ]]; then
  dir=${BASH_REMATCH[1]}; recipe=${BASH_REMATCH[2]}
elif [[ $mode =~ $ckptsel_re ]]; then
  dir=${BASH_REMATCH[1]}; recipe=${BASH_REMATCH[2]}; selspec=${BASH_REMATCH[3]}
else
  refuse "bad MODE '$mode' (online:ARM or ckpt:/ABS/DIR:RECIPE or ckptsel:/ABS/DIR:RECIPE:SPEC, SPEC = NAME or NAME:layers=A-B[,N...])"
fi
sel_args=(); [ -z "$selspec" ] || sel_args=(--select "$selspec")
repeats=${BENCH_REPEATS-2}
[[ $repeats =~ ^[1-9][0-9]*$ ]] || refuse "BENCH_REPEATS must be a positive integer, got '$repeats'"
verify_args=${BENCH_VERIFY_ARGS-}
[[ $verify_args =~ ^[A-Za-z0-9\ =._/-]*$ ]] || refuse "BENCH_VERIFY_ARGS has characters outside A-Za-z0-9 =._/-"
read -r -a verify_extra <<< "$verify_args"

if [ -n "$arm" ]; then logname=$arm; else logname="ckpt-$(basename "$dir")"; fi
B=/workspace/bundle
files="model.safetensors.index.json config.json quantization_info.json"

if [ $dry -eq 1 ]; then
  echo "bench_arm.sh dry run: $label $mode (nothing is run, nothing is written)"
  echo "refuses when /workspace/results/bench-$label exists"
  echo "stop: bash $B/pod/stop_server.sh"
  if [ -n "$arm" ]; then
    echo "serve: bash $B/pod/serve_olala.sh $arm"
  else
    if [ -n "$selspec" ]; then
      echo "pre-check, on the pod before anything else (refuses a bad SPEC and a stale $dir): python3 -m olala_quant.ablation expected --src /workspace/models/olala-bf16 --select $selspec"
      echo "stale checkpoint: when $dir holds a checkpoint, its quantization_info.json selection must equal the canonical form of $selspec"
      echo "provenance: selection, selection_requested and expected_counts of that pre-check"
    fi
    echo "patch: apply_patch (olala-quant-loader.patch)"
    echo "convert if $dir/model.safetensors.index.json is missing: python3 -m olala_quant.convert --src /workspace/models/olala-bf16 --dst $dir --recipe $recipe --device cuda${sel_args[*]:+ ${sel_args[*]}}"
    echo "verify: python3 -m olala_quant.verify --src /workspace/models/olala-bf16 --dst $dir --report /workspace/results/verify-$label.json ${verify_extra[*]}${sel_args[*]:+ ${sel_args[*]}}"
    if [ -n "$selspec" ]; then
      echo "check, before serving: python3 -m olala_quant.ablation check --src /workspace/models/olala-bf16 --dst $dir --select $selspec"
    fi
    echo "provenance: (cd $dir && sha256sum $files) >> /workspace/results/provenance-$label.txt"
    echo "serve: bash $B/pod/serve_olala.sh ckpt $dir"
  fi
  echo "smoke gate (exit 0 or 2 accepted): bash $B/pod/smoke.sh $logname --timeout 900"
  echo "provenance: kernel banners of the newest /workspace/logs/$logname/server-*.log"
  echo "GPU exclusivity check, then: bash $B/pod/bench_sweep.sh $label --repeats $repeats"
  echo "log: /workspace/logs/bench-$label.log (ends with BENCH_DONE only after a complete sweep)"
  echo "stop: bash $B/pod/stop_server.sh"
  exit 0
fi

# shellcheck source=job_common.sh
. "$here/job_common.sh"
results=/workspace/results
benchlog=/workspace/logs/bench-$label.log
raw=/workspace/logs/bench-$label.sweep.raw
prov=$results/provenance-$label.txt
smokeout=/workspace/logs/bench-$label.smoke.txt
serving=""

stop_srv() { bash "$B/pod/stop_server.sh" > /dev/null 2>&1 || log "warning: stop_server.sh exit $?"; }
fail() { log "FAILED: $*"; [ -z "$serving" ] || stop_srv; echo "BENCH_PIPELINE_FAILED: $*"; exit 1; }

[ ! -e "$results/bench-$label" ] || refuse "$results/bench-$label exists (bench_sweep.sh never overwrites; use a new LABEL)"
if [ -n "$selspec" ]; then   # CPU only, before any server is stopped or started: a bad SPEC or a stale DIR costs nothing
  ckptsel_precheck "$dir" "$selspec" || refuse "$CKPTSEL_ERR"
fi
rm -f "$benchlog" "$raw"   # a BENCH_DONE of an earlier attempt must not let collect.sh pull an unfinished sweep
mkdir -p "$results"
log "== bench $label ($mode), $repeats repeat(s)"
{ echo "run=$label mode=$mode started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; } > "$prov"
[ -z "$selspec" ] || ckptsel_provenance "$selspec" >> "$prov"

bash "$B/pod/stop_server.sh" > /dev/null 2>&1; sleep "${BENCH_STOP_WAIT:-2}"   # idempotent: nothing running is fine
tmpout=$(mktemp)
if [ -n "$arm" ]; then
  serving=1
  bash "$B/pod/serve_olala.sh" "$arm" > "$tmpout" 2>&1 || { tail -n 4 "$tmpout"; fail "serve (serve_olala.sh $arm)"; }
else
  apply_patch || fail "loader patch"
  if [ ! -f "$dir/model.safetensors.index.json" ]; then
    log "convert $dir ($recipe)"
    (cd "$QU" && PYTHONPATH="$QU" python3 -m olala_quant.convert --src /workspace/models/olala-bf16 --dst "$dir" \
      --recipe "$recipe" --device cuda ${sel_args[@]+"${sel_args[@]}"} > "$tmpout" 2>&1) \
      || { tail -n 4 "$tmpout"; fail "convert (olala_quant.convert $recipe${selspec:+ --select $selspec})"; }
    tail -n 2 "$tmpout"
  fi
  log "verify $dir"
  (cd "$QU" && PYTHONPATH="$QU" python3 -m olala_quant.verify --src /workspace/models/olala-bf16 --dst "$dir" \
    --report "$results/verify-$label.json" ${verify_extra[@]+"${verify_extra[@]}"} ${sel_args[@]+"${sel_args[@]}"} > "$tmpout" 2>&1) \
    || { tail -n 8 "$tmpout"; fail "verify (olala_quant.verify exit status not 0)"; }
  echo "verify: $(tail -n 1 "$tmpout" | cut -c1-300)" >> "$prov"
  if [ -n "$selspec" ]; then   # the checkpoint must be the selection asked for, before anything is served
    log "ablation check $dir"
    (cd "$QU" && PYTHONPATH="$QU" python3 -m olala_quant.ablation check --src /workspace/models/olala-bf16 --dst "$dir" \
      --select "$selspec" > "$tmpout" 2>&1) \
      || { tail -n 6 "$tmpout"; fail "ablation check (olala_quant.ablation check: the checkpoint does not match the selection $selspec)"; }
    echo "ablation_check: $(tail -n 1 "$tmpout" | cut -c1-300)" >> "$prov"
  fi
  # shellcheck disable=SC2086  # $files is three names
  (cd "$dir" && sha256sum $files) >> "$prov" || fail "provenance (sha256 of the checkpoint files)"
  serving=1
  bash "$B/pod/serve_olala.sh" ckpt "$dir" > "$tmpout" 2>&1 || { tail -n 4 "$tmpout"; fail "serve (serve_olala.sh ckpt $dir)"; }
fi
tail -n 2 "$tmpout"

bash "$B/pod/smoke.sh" "$logname" --timeout 900 > "$smokeout" 2>&1
rc=$?
log "smoke exit $rc: $(head -n 2 "$smokeout" | tr '\n' ' ')"
{ [ $rc -eq 0 ] || [ $rc -eq 2 ]; } || { tail -n 8 "$smokeout"; fail "smoke gate (exit $rc)"; }
# shellcheck disable=SC2012  # newest server log by mtime, names are ours
serverlog=$(ls -t "/workspace/logs/$logname"/server-*.log 2> /dev/null | head -n 1)
banners=$(grep -h -E 'Quantizing [0-9]+ layers|Selected .*Kernel|Using .* MoE backend' "$serverlog" 2> /dev/null | cut -c1-300)
if [ -n "$banners" ]; then echo "$banners" >> "$prov"; else echo "warning: no kernel banner found in ${serverlog:-any server log}" >> "$prov"; fi
log "provenance in $prov"

# nothing else may use the GPU while the sweep measures it (an accuracy run, a second sweep, a fidelity collection)
if command -v pgrep > /dev/null 2>&1; then
  busy=$(pgrep -af 'run_tasks\.py|run_accuracy\.sh|arm_pipeline\.sh|vllm bench|bench_sweep\.sh|fidelity\.py')
  [ -z "$busy" ] || fail "GPU not exclusive, other jobs running: $(echo "$busy" | head -n 3 | tr '\n' ';' | cut -c1-300)"
  echo "gpu_exclusive_check=ok" >> "$prov"
else
  echo "gpu_exclusive_check=skipped (no pgrep on this pod)" >> "$prov"
fi

log "== bench_sweep $label --repeats $repeats"
bash "$B/pod/bench_sweep.sh" "$label" --repeats "$repeats" 2>&1 | tee "$raw" | grep -v -x --line-buffered BENCH_DONE
sweep_rc=${PIPESTATUS[0]}
grep -v -x BENCH_DONE "$raw" > "$benchlog"   # BENCH_DONE is ours: it is appended below, after the checks
echo "sweep_exit=$sweep_rc finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$prov"
if [ -d "$results/bench-$label" ]; then   # collect.sh pulls only this directory
  cp "$prov" "$results/bench-$label/provenance-$label.txt"
  cp "$smokeout" "$results/bench-$label/smoke.txt"
  cp "$benchlog" "$results/bench-$label/sweep.log"
fi
[ "$sweep_rc" -eq 0 ] || fail "sweep (bench_sweep.sh exit $sweep_rc)"
grep -q -x BENCH_DONE "$raw" || fail "sweep ended without BENCH_DONE"
ls "$results/bench-$label"/*.json > /dev/null 2>&1 || fail "sweep wrote no result file"
echo BENCH_DONE >> "$benchlog"   # the data is in place; collect.sh may pull it from now on
nfail=$(grep -c '^POINT FAILED' "$raw")
stop_srv
[ "$nfail" -eq 0 ] || { log "FAILED: $nfail point(s) failed"; echo "BENCH_PIPELINE_FAILED: $nfail point(s) failed in the sweep (partial data kept, BENCH_DONE written)"; exit 1; }
echo BENCH_PIPELINE_DONE
