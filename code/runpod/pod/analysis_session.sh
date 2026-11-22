#!/usr/bin/env bash
# Weight statistics, default prompts and activation probe on the REAL bf16 weights, on one fresh pod, unattended.
# Detach it; the log and the markers are the record.
# Usage: analysis_session.sh [--dry-run] all|STEP...        STEP: weights, prompts, probe (always run in that order)
#   weights  python -m olala_quant.weight_stats --src MODEL --out RES/weight_stats --device cuda
#   prompts  the default 80 prompts (64 task inputs + 16 plain texts) with analysis/build_prompts.py in the EVALS
#            environment (needs setup_evals_env.sh to be finished: it waits for EVALS_ENV_DONE like arm_pipeline.sh does;
#            the datasets load from the Hub at that moment, which is fine on a pod)
#   probe    analysis/activation_probe.py on the GPU with those prompts. Attention backend and length (decided at run
#            time, recorded in the provenance): when the vendored model finds FA3, FA2 or flex attention (it picks
#            fa3 > fa2 > flex by itself), --max-tokens 4096 (the probe's default); when it would fall back to eager,
#            --max-tokens 2048, because estimate_memory_gib (analysis/activation_probe.py, meta.memory_estimate) puts
#            eager at about 27 GiB at 4096 tokens (three float32 [heads, T, T] score tensors) against 20 GiB with flash
#            or flex, and about 20 GiB at 2048 tokens, on a card of 32 GB. A flash/flex run that fails is retried once
#            with --attn-impl eager --max-tokens 2048 (the first failure stays in the log and the provenance).
# Python: the pod's own python3 (PYTHON replaces it), like quant/gptq/pod_g1lite.sh and pod/bench_arm.sh: the image holds the
# torch the OVH fork was built with, which analysis/pyproject.toml pins too. The only install is `pip install --no-deps
# einops==0.8.1` when einops is missing (pod_g1lite.sh does the same). The prompts step uses the evals environment
# (uv run --frozen --python 3.12, /workspace/venv-evals). No other install, no token, no download of ours.
# Files: logs /workspace/logs/analysis/STEP.log (and session.log: the markers); results under
#   /workspace/results/analysis/{weight_stats,prompts,probe}/ each with provenance.txt (GPU, python and library versions, sha256
#   of config.json and of the safetensors index, or of the header of a single model.safetensors, git commit of the bundle when
#   known, the command lines, for the probe the attention backend and what the run really used), status.txt and step.log;
#   summary.txt next to them; at the very end /workspace/results/analysis.done (what collect.sh files mode waits for).
# Markers on stdout (one per line): ANALYSIS_STEP_DONE:STEP or ANALYSIS_STEP_FAILED:STEP after each step; a failed step does
#   not stop the others (a probe without prompts fails on its own); the last line is ANALYSIS_DONE, or ANALYSIS_FAILED: REASON
#   when any step is failed (refused arguments: exit 2). The .done file holds the last line, so partial results get pulled too.
# A done step is never overwritten (a refusal, exit 2); a failed one can be rerun alone.
# Env: ANALYSIS_DEADLINE_SECONDS (default 9600 = 160 minutes: a step still running then is stopped, the others are not started,
#   so that the results are written and pulled before the 3-hour watchdog), ANALYSIS_EVALS_TRIES / ANALYSIS_EVALS_POLL (default
#   120 x 10 s), OLALA_WORK (default /workspace), PYTHON, ANALYSIS_MODEL (default /workspace/models/olala-bf16).
set -u
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=evals_env.sh
. "$here/evals_env.sh"   # common.sh, EVALS_DIR, UV_RUN_ARGS, UV_PROJECT_ENVIRONMENT, HF_HOME, PATH with ~/.local/bin
unset OLALA_ATTN_IMPL    # the backend decision below must not depend on a stray variable of the shell

STEP_ORDER=(weights prompts probe)
usage="usage: analysis_session.sh [--dry-run] all|STEP...   (STEP: weights prompts probe)"
refuse() { echo "$usage" >&2; echo "ANALYSIS_FAILED: $*"; exit 2; }
step_dir() { case $1 in weights) echo weight_stats ;; *) echo "$1" ;; esac; }

dry=0; requested=()
while [ $# -gt 0 ]; do
  case $1 in
    --dry-run) dry=1 ;;
    -*) refuse "unknown option $1" ;;
    *) requested+=("$1") ;;
  esac
  shift
done
[ ${#requested[@]} -gt 0 ] || refuse "no step given"
selected=()
if [ "${requested[0]}" = all ]; then
  [ ${#requested[@]} -eq 1 ] || refuse "'all' stands alone, got: ${requested[*]}"
  selected=("${STEP_ORDER[@]}")
else
  for r in "${requested[@]}"; do
    case " ${STEP_ORDER[*]} " in *" $r "*) ;; *) refuse "unknown step '$r'" ;; esac
  done
  for s in "${STEP_ORDER[@]}"; do
    n=0; for r in "${requested[@]}"; do [ "$r" != "$s" ] || n=$((n + 1)); done
    [ $n -le 1 ] || refuse "step '$s' given twice"
    [ $n -eq 0 ] || selected+=("$s")
  done
fi

B=$OLALA_WORK/bundle
MODEL=${ANALYSIS_MODEL:-$MODEL_ROOT/olala-bf16}
RESULTS=$OLALA_WORK/results
RES=$RESULTS/analysis
DONE=$RESULTS/analysis.done
LOGDIR=$LOG_ROOT/analysis
PY=${PYTHON:-python3}
PROMPTS=$RES/prompts/prompts_default.jsonl
N_PROMPTS=80
ENVV=(HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)

weights_cmd=("$PY" -m olala_quant.weight_stats --src "$MODEL" --out "$RES/weight_stats" --device cuda)
# shellcheck disable=SC2206  # UV_RUN_ARGS is a list of words
prompts_cmd=(uv $UV_RUN_ARGS python "$B/analysis/build_prompts.py" --out "$PROMPTS" --evals-dir "$EVALS_DIR" --online)
probe_args() {  # TOKENS [ATTN_IMPL]
  PROBE_CMD=("$PY" "$B/analysis/activation_probe.py" --model-dir "$MODEL" --prompts "$PROMPTS" --out "$RES/probe"
    --device cuda --max-tokens "$1")
  [ -z "${2:-}" ] || PROBE_CMD+=(--attn-impl "$2")
}
quoted() { printf '%q ' "$@"; }

for s in "${selected[@]}"; do   # a step that is done is never overwritten
  if grep -q '^done' "$RES/$(step_dir "$s")/status.txt" 2> /dev/null; then
    refuse "step $s is already done ($RES/$(step_dir "$s")/status.txt): results are never overwritten; move that folder away first"
  fi
done

if [ $dry -eq 1 ]; then
  echo "analysis_session.sh dry run: ${selected[*]} (nothing is run, nothing is written)"
  echo "python: $PY; model: $MODEL; bundle: $B"
  echo "logs: $LOGDIR/STEP.log; results: $RES/{weight_stats,prompts,probe}/ each with provenance.txt, status.txt, step.log"
  echo "deadline: ${ANALYSIS_DEADLINE_SECONDS:-9600} s after the start (a step still running is stopped, the others are not started)"
  for s in "${selected[@]}"; do
    case $s in
      weights) echo "1 weights: (cd $B/quant && env ${ENVV[*]} PYTHONPATH=$B/quant $(quoted "${weights_cmd[@]}"))" ;;
      prompts)
        echo "2 prompts: wait for EVALS_ENV_DONE in $LOG_ROOT/setup/evals-env-*.log (${ANALYSIS_EVALS_TRIES:-120} polls of ${ANALYSIS_EVALS_POLL:-10} s), then"
        echo "    (cd $EVALS_DIR && $(quoted "${prompts_cmd[@]}"))   # the Hub is reached here; $N_PROMPTS lines expected"
        ;;
      probe)
        probe_args 4096
        echo "3 probe: python3 -c 'import einops' || $PY -m pip install --no-deps einops==0.8.1"
        echo "    detect the attention backend of the vendored model (fa3, fa2 or flex: --max-tokens 4096; eager or unknown: 2048):"
        echo "    env ${ENVV[*]} $(quoted "${PROBE_CMD[@]}")"
        echo "    with eager attention: ... --max-tokens 2048 (a failing flash/flex run is retried once with --attn-impl eager --max-tokens 2048)"
        ;;
    esac
  done
  echo "markers: ANALYSIS_STEP_DONE:STEP or ANALYSIS_STEP_FAILED:STEP after each step; at the end ANALYSIS_DONE or ANALYSIS_FAILED: REASON"
  echo "last of all: $DONE (written after the final marker; also when a step failed)"
  echo "ANALYSIS_DRY_RUN (nothing was run)"
  exit 0
fi

mkdir -p "$LOGDIR" "$RES" || refuse "cannot create $LOGDIR and $RES"
rm -f "$DONE"   # the .done of an earlier attempt must not let collect.sh pull an unfinished session
start=$(date +%s)
deadline=$((start + ${ANALYSIS_DEADLINE_SECONDS:-9600}))
out() { echo "$*"; echo "$*" >> "$LOGDIR/session.log"; }
sha256_of() { if command -v sha256sum > /dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
st_header_sha() {
  "$PY" - "$1" << 'PYEOF'
import hashlib, struct, sys
with open(sys.argv[1], "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    print(hashlib.sha256(f.read(n)).hexdigest())
PYEOF
}

# run_limited CMD...: the command under `timeout` until the deadline (exit 124 when stopped)
run_limited() {
  local left=$((deadline - $(date +%s)))
  if [ "$left" -lt 1 ]; then echo "the deadline of the session has passed: $* not started"; return 124; fi
  if command -v timeout > /dev/null 2>&1; then timeout -k 20 "$left" "$@"; else "$@"; fi
}

PROV=""
prov() { echo "$*" >> "$PROV"; }
provenance_header() {  # STEP
  echo "analysis_session step=$1 started=$(utc_iso) host=$(hostname 2> /dev/null || echo unknown)"
  echo "gpu: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2> /dev/null | head -n 1)"
  echo "gpu_memory_used_before: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2> /dev/null | head -n 1)"
  echo "python: $("$PY" --version 2>&1)"
  "$PY" - << 'PYEOF' 2>&1
import importlib.metadata as md
try:
    import torch
    print("torch:", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
except Exception as e:
    print("torch: not importable (%s)" % type(e).__name__)
for pkg in ("transformers", "numpy", "safetensors", "einops"):
    try:
        print(pkg + ":", md.version(pkg))
    except Exception:
        print(pkg + ": not installed")
PYEOF
  echo "model_dir=$MODEL"
  if [ -f "$MODEL/config.json" ]; then echo "config.json sha256=$(sha256_of "$MODEL/config.json")"; else echo "config.json: missing"; fi
  if [ -f "$MODEL/model.safetensors.index.json" ]; then
    echo "model.safetensors.index.json sha256=$(sha256_of "$MODEL/model.safetensors.index.json")"
  elif [ -f "$MODEL/model.safetensors" ]; then
    echo "model.safetensors header sha256=$(st_header_sha "$MODEL/model.safetensors")"
  else
    echo "weights: neither a safetensors index nor a model.safetensors in $MODEL"
  fi
  local commit
  commit=$(sed -n '1s/.*git: //p' "$B/BUNDLE_MANIFEST.txt" 2> /dev/null | tr -d '[:space:]')
  [[ $commit =~ ^[0-9a-f]{40}$ ]] || commit=unknown
  echo "bundle=$(readlink -f "$B" 2> /dev/null || echo "$B")"
  echo "bundle_git=$commit"
  if [ -f "$B/BUNDLE_MANIFEST.txt" ]; then echo "bundle_manifest_sha256=$(sha256_of "$B/BUNDLE_MANIFEST.txt")"; else echo "bundle_manifest_sha256=missing"; fi
}

need_weights() {
  [ -f "$MODEL/config.json" ] || { echo "no config.json in $MODEL: the bf16 weights are expected there (pod_setup.sh downloads them)"; return 1; }
}
need_bundle_file() {  # RELATIVE_PATH
  [ -f "$B/$1" ] || { echo "the bundle lacks $1: rebuild and upload it (pod_setup.sh NAME --bundle-only)"; return 1; }
}

step_weights() {
  need_weights || return 1
  need_bundle_file quant/olala_quant/weight_stats.py || return 1
  rm -f "$RES/weight_stats/weight_stats.json" "$RES/weight_stats/weight_stats.md"
  prov "command: (cd $B/quant && env ${ENVV[*]} PYTHONPATH=$B/quant $(quoted "${weights_cmd[@]}"))"
  (cd "$B/quant" && run_limited env "${ENVV[@]}" PYTHONPATH="$B/quant" "${weights_cmd[@]}")
  local rc=$?
  [ $rc -eq 0 ] || return $rc
  [ -s "$RES/weight_stats/weight_stats.json" ] && [ -s "$RES/weight_stats/weight_stats.md" ] \
    || { echo "weight_stats exited 0 but wrote no weight_stats.json / weight_stats.md in $RES/weight_stats"; return 1; }
}

step_prompts() {
  need_bundle_file analysis/build_prompts.py || return 1
  local i
  for i in $(seq 1 "${ANALYSIS_EVALS_TRIES:-120}"); do
    grep -q EVALS_ENV_DONE "$LOG_ROOT"/setup/evals-env-*.log 2> /dev/null && break
    if grep -q EVALS_ENV_FAILED "$LOG_ROOT"/setup/evals-env-*.log 2> /dev/null; then
      echo "evals environment setup failed (EVALS_ENV_FAILED in $LOG_ROOT/setup/evals-env-*.log): the prompts need that environment"
      return 1
    fi
    sleep "${ANALYSIS_EVALS_POLL:-10}"
  done
  grep -q EVALS_ENV_DONE "$LOG_ROOT"/setup/evals-env-*.log 2> /dev/null \
    || { echo "evals environment not ready after ${ANALYSIS_EVALS_TRIES:-120} polls: no EVALS_ENV_DONE in $LOG_ROOT/setup/evals-env-*.log"; return 1; }
  rm -f "$PROMPTS"
  mkdir -p "$RES/prompts"
  prov "command: (cd $EVALS_DIR && $(quoted "${prompts_cmd[@]}"))"
  (cd "$EVALS_DIR" && run_limited "${prompts_cmd[@]}")
  local rc=$?
  [ $rc -eq 0 ] || return $rc
  local n
  n=$(wc -l < "$PROMPTS" 2> /dev/null | tr -d ' ')
  [ "${n:-0}" -eq "$N_PROMPTS" ] || { echo "the prompt file has ${n:-0} lines, expected $N_PROMPTS: $PROMPTS"; return 1; }
  prov "result: $N_PROMPTS prompts, sha256 of the file $(sha256_of "$PROMPTS")"
}

detect_attention() {  # prints fa3, fa2, flex, eager or unknown: what the vendored model resolves to on this pod
  local impl
  impl=$(cd "$B/quant/gptq" 2> /dev/null && env OLALA_TORCH_MOE=1 OLALA_TORCH_MAMBA3=1 HF_HUB_OFFLINE=1 PYTHONPATH="$B/quant/gptq" \
    "$PY" -c 'import olala_model; m, _ = olala_model.load_olala_modules(); print("ATTN_IMPL=" + m.ATTN_IMPL)' 2>&1 \
    | sed -n 's/^ATTN_IMPL=//p' | tail -n 1)
  case $impl in fa3|fa2|flex|eager) echo "$impl" ;; *) echo unknown ;; esac
}

probe_result() {
  "$PY" - "$RES/probe/activation_probe.json" << 'PYEOF' 2>&1 || echo "result: activation_probe.json could not be read"
import json, sys
d = json.load(open(sys.argv[1]))
m = d["meta"]
n = sum(1 for p in d.get("prompts", []) if p.get("truncated"))
print("result: attn_impl=%s peak_gpu_gib=%s max_tokens_in_a_prompt=%s n_prompts=%s n_truncated=%d"
      % (m.get("attn_impl"), m.get("peak_gpu_gib"), m.get("max_tokens_in_a_prompt"), m.get("n_prompts"), n))
PYEOF
}

run_probe_once() {  # TOKENS [ATTN_IMPL]
  rm -f "$RES/probe/activation_probe.json" "$RES/probe/activation_probe.md"
  probe_args "$@"
  prov "command: env ${ENVV[*]} $(quoted "${PROBE_CMD[@]}")"
  run_limited env "${ENVV[@]}" "${PROBE_CMD[@]}"
}

step_probe() {
  need_weights || return 1
  need_bundle_file analysis/activation_probe.py || return 1
  [ -s "$PROMPTS" ] || { echo "no prompt file $PROMPTS: the prompts step has not produced it"; return 1; }
  if ! "$PY" -c 'import einops' 2> /dev/null; then
    echo "einops missing: installing einops==0.8.1 without dependencies (the pin of analysis/pyproject.toml)"
    "$PY" -m pip install --no-deps einops==0.8.1 || { echo "could not install einops==0.8.1"; return 1; }
  fi
  local impl tokens
  impl=$(detect_attention)
  case $impl in fa3|fa2|flex) tokens=4096 ;; *) tokens=2048 ;; esac
  prov "attention_detected=$impl"
  prov "max_tokens=$tokens"
  prov "choice: flash and flex attention keep no score tensors (estimate_memory_gib in analysis/activation_probe.py: about 20 GiB at 4096 tokens on a 32 GB card); eager keeps three float32 [heads, T, T] score tensors (about 27 GiB at 4096, about 20 GiB at 2048), so eager or an unknown backend runs with --max-tokens 2048"
  prov "attempt 1:"
  mkdir -p "$RES/probe"
  run_probe_once "$tokens"
  local rc=$? attempts=1 final_tokens=$tokens final_attn=auto
  if [ $rc -ne 0 ] && [ "$impl" != eager ] && [ "$impl" != unknown ] && [ $rc -ne 124 ]; then
    echo "attempt 1 (attention $impl, --max-tokens $tokens) failed with exit $rc: retrying once with eager attention and 2048 tokens"
    prov "attempt 1 failed with exit $rc; attempt 2:"
    run_probe_once 2048 eager
    rc=$?; attempts=2; final_tokens=2048; final_attn=eager
  fi
  prov "attempts=$attempts final_max_tokens=$final_tokens final_attn_impl_arg=$final_attn"
  [ $rc -eq 0 ] || return $rc
  [ -s "$RES/probe/activation_probe.json" ] && [ -s "$RES/probe/activation_probe.md" ] \
    || { echo "the probe exited 0 but wrote no activation_probe.json / activation_probe.md in $RES/probe"; return 1; }
  prov "$(probe_result)"
}

run_step() {  # STEP
  local name=$1 dir log rc reason status
  dir=$RES/$(step_dir "$name"); log=$LOGDIR/$name.log
  mkdir -p "$dir"; : > "$log"
  rm -f "$dir/status.txt"
  PROV=$dir/provenance.txt
  provenance_header "$name" > "$PROV" 2>&1
  out "== $(utc_iso) step $name: log $log"
  "step_$name" >> "$log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    status="done at $(utc_iso)"
  else
    reason="exit $rc"
    case $rc in 124|137) reason="exit $rc, stopped at the deadline of the session" ;; esac
    echo "$reason" >> "$log"
    status="failed: $reason; last line of the log: $(grep -v '^exit [0-9]' "$log" | tail -n 1 | cut -c1-200)"
  fi
  cp "$log" "$dir/step.log"
  echo "$status" > "$dir/status.txt"
  prov "finished=$(utc_iso) status=$status"
  if [ $rc -eq 0 ]; then
    out "ANALYSIS_STEP_DONE:$name"
  else
    out "step $name failed: $reason"
    tail -n 6 "$log" | sed -e 's/^/    /'
    out "ANALYSIS_STEP_FAILED:$name"
  fi
}

out "analysis_session $(utc_iso): steps ${selected[*]}, model $MODEL, bundle $B, deadline in $((deadline - start)) s"
for s in "${selected[@]}"; do run_step "$s"; done

failed=()
{
  echo "analysis_session summary, written $(utc_iso)"
  for s in "${STEP_ORDER[@]}"; do
    f=$RES/$(step_dir "$s")/status.txt
    if [ -f "$f" ]; then
      echo "$s: $(cat "$f")"
      grep -q '^failed' "$f" && failed+=("$s")
    else
      echo "$s: not run"
    fi
  done
  grep -h -E '^(attention_detected|max_tokens|attempts|result: attn_impl)' "$RES/probe/provenance.txt" 2> /dev/null
} > "$RES/summary.txt.tmp"
mv "$RES/summary.txt.tmp" "$RES/summary.txt"
# the loop above ran in a group, not a subshell: `failed` is set
if [ ${#failed[@]} -eq 0 ]; then
  final="ANALYSIS_DONE"
else
  final="ANALYSIS_FAILED: step(s) failed: $(IFS=,; echo "${failed[*]}" | sed 's/,/, /g')"
fi
out "$final"
printf '%s\nfinished=%s\n' "$final" "$(utc_iso)" > "$DONE.tmp" && mv "$DONE.tmp" "$DONE"   # last of all: collect.sh pulls from here on
[ ${#failed[@]} -eq 0 ]
