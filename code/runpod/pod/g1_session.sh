#!/usr/bin/env bash
# Arm G1 (PROTOCOL amendments 18 and 22: GPTQ routed experts from CAL-1, every other tensor bf16) built and tested end to end on
# ONE long-lived pod, unattended. The pre-registered stage 1 gate decides whether the accuracy run happens. Detach it; the log is
# the record.
# Usage: g1_session.sh [--dry-run]
# Steps, in order, each with its own log (results/g1/STEP-STAMP.log) and marker (G1_STEP_<STEP>_DONE, _FAILED: reason, _SKIPPED):
#   1 check        pod_g1lite.sh check
#   2 feasibility  pod_g1lite.sh feasibility (G1_BASE_SELECT=experts, G1_CALIB=/workspace/cal1/calib-cal1): per-layer forward
#                  seconds and peak GPU memory are parsed from its output (pod/g1_tools.py feasibility). The full build time is
#                  PROJECTED with the formula below; the session REFUSES to go on (G1_SESSION_FAILED, .done written) when the
#                  projection exceeds the budget or the peak memory exceeds 30 GiB.
#                  FORMULA: projected build seconds = 72.75 x (sum of the per-layer forward seconds over the 36 layers), the sum
#                  being the mean forward seconds of the measured layers x 36. 72.75 = 291 CAL-1 windows (the build) / 4 windows
#                  (the feasibility run, --max-windows 4). Budget: G1_BUILD_BUDGET_HOURS (default 4.0); "exceeds" is strict. Peak:
#                  the largest peak_cuda_GB (1e9 bytes) of the forward table and of the calibrated table, converted to GiB.
#   3 build        pod_g1lite.sh build into /workspace/ckpt/g1lite (resumable per layer: the tool continues from its work dir)
#   4 verify       pod_g1lite.sh verify = olala_quant.verify --calibrated --select experts
#   5 fidelity     the bf16 reference when absent (same code as pod/ablation_arm.sh), the checkpoint served as arm_pipeline.sh's
#                  ckpt mode does (loader patch, serve_olala.sh ckpt, smoke gate exit 0 or 2), top-20 logprobs of the same chunk
#                  file collected, compared with the reference by the same tool call as the ablation batch: results/g1/fidelity.json
#   6 gate         thresholds below (the ONE place they live); results/g1/gate.json; prints G1_GATE_PASS or G1_GATE_FAIL
#   7 accuracy     only on PASS: run_accuracy.sh G1-v2 --effort medium (the five tasks, as arm_pipeline.sh calls it) on the same
#                  server (restarted the same way when a rerun finds it down); on FAIL nothing more runs, UNLESS
#                  G1_ACCURACY_ALWAYS=1 (a pre-registered change: the accuracy run happens whatever the gate says; gate.json still
#                  holds the verdict, G1_GATE_PASS or G1_GATE_FAIL is still logged, and the session ends G1_SESSION_DONE with
#                  'gate: pass' or 'gate: fail' and the numbers in the reason; every refusal and the deadline guard are unchanged)
# Final markers (the last stdout line, and the FIRST line of /workspace/results/g1.done, which is written LAST in every path):
#   G1_SESSION_DONE         gate passed and the accuracy run is complete (exit 0)
#   G1_SESSION_GATE_FAILED  the gate failed (G1_ACCURACY_ALWAYS=0 only): a legitimate negative result with its numbers (exit 0)
#   G1_SESSION_FAILED: REASON  anything else, a feasibility refusal included (exit 1)
# Env: G1_POD_DEADLINE_EPOCH (required: epoch seconds at which the pod's watchdog deletes it; set by launch_g1.sh; no new step
#   starts when less than 20 minutes remain), G1_BUILD_BUDGET_HOURS (4.0), G1_STOP_WAIT (seconds after a stop, 3), G1_EVALS_POLL
#   (seconds between checks of the evals environment, 10) and G1_EVALS_TRIES (120), G1_NOISE (0 or 1, default 1: the second
#   collection of the bf16 server, the noise floor), G1_ACCURACY_ALWAYS (0 or 1, default 0 here; launch_g1.sh sets 1). Nothing else
#   is read.
# Inputs: the bf16 weights /workspace/models/olala-bf16, the bundle, and the three CAL-1 files manifest.json, windows.json,
#   calib_text.jsonl in /workspace/cal1/calib-cal1 (shipped by launch_g1.sh; checked here against the manifest's files block).
# The checkpoint /workspace/ckpt/g1lite stays on the pod: it is never copied into the results (its sha256 index, config and
#   quantization_info.json and its size in bytes are: results/g1/checkpoint.json and provenance.txt).
# Resume: a rerun continues. A step with results/g1/steps/STEP.ok is skipped; build resumes from its work directory; the
#   feasibility measurement is reused but its refusal decision is taken again; nothing of the checkpoint or of earlier results is
#   deleted. A session whose g1.done says DONE or GATE_FAILED is refused (exit 2); a FAILED one is continued.
# Exit: 0 done or gate failed, 1 failed, 2 bad arguments, environment, or a refused rerun.
# shellcheck disable=SC2329  # the step functions are called by name through run_step
set -u
here=$(cd "$(dirname "$0")" && pwd)

# ---- THE NUMBERS THAT DECIDE (the only place they live; g1_tools.py takes them as arguments) -------------------------------
# Stage 1 gate of arm G1, PROTOCOL.md amendment 22 (2026-11-21), fixed from N5's measured values on the same text set and tool:
G1_GATE_KL_MAX=0.0194          # mean KL against bf16 at most this (a reduction of at least 20% against N5's 0.0243)
G1_GATE_PPL_RATIO_MAX=1.0197   # perplexity ratio at most this (N5's own value)
G1_GATE_SOURCE="PROTOCOL.md amendment 22 (stage 1 gate of arm G1, fixed from the measured values of N5)"
N5_KL=0.0243                   # N5 (routed experts RTN, rest bf16): ablation batch result, for the record in gate.json
N5_PPL_RATIO=1.0197
N5_FROM="results/wave1/ablation-abl-experts"
# Feasibility projection (see the header): 291 CAL-1 windows / 4 windows of the feasibility run = 72.75
FEAS_SCALE=72.75
N_LAYERS=36
PEAK_GIB_MAX=30
GUARD_MIN=20                   # no new step starts when fewer minutes of the pod's lifetime remain
# ----------------------------------------------------------------------------------------------------------------------------

usage="usage: g1_session.sh [--dry-run]   (env G1_POD_DEADLINE_EPOCH, G1_BUILD_BUDGET_HOURS)"
refuse() { echo "$usage" >&2; echo "refused: $*" >&2; exit 2; }

dry=0
for a in "$@"; do
  case $a in
    --dry-run) dry=1 ;;
    -*) refuse "unknown option $a" ;;
    *) refuse "unexpected argument $a" ;;
  esac
done

B=/workspace/bundle
G=$B/quant/gptq
results=/workspace/results
rdir=$results/g1
sdir=$rdir/steps
donefile=$results/g1.done
weights=/workspace/models/olala-bf16
calib=/workspace/cal1/calib-cal1
ckpt=/workspace/ckpt/g1lite
raw=$results/raw
ref=$raw/b0_wikitext2.npz
run_name=G1-v2
logname="ckpt-g1lite"
chunks=data/wikitext2_test_64x512.json
g1logs=/workspace/logs/g1lite
pidfile=/workspace/run/g1_session.pid
tools=$here/g1_tools.py
budget=${G1_BUILD_BUDGET_HOURS-4.0}
stopwait=${G1_STOP_WAIT-3}
evals_poll=${G1_EVALS_POLL-10}
evals_tries=${G1_EVALS_TRIES-120}
noise=${G1_NOISE-1}
always=${G1_ACCURACY_ALWAYS-0}
guard_s=$((GUARD_MIN * 60))

if [ $dry -eq 1 ]; then
  echo "g1_session.sh dry run (nothing is run, nothing is written)"
  echo "refuses when $donefile already says G1_SESSION_DONE or G1_SESSION_GATE_FAILED, or another session runs (pid file $pidfile)"
  echo "constants: gate mean KL <= $G1_GATE_KL_MAX, perplexity ratio <= $G1_GATE_PPL_RATIO_MAX ($G1_GATE_SOURCE)"
  echo "  N5 reference $N5_KL / $N5_PPL_RATIO from $N5_FROM; projection scale $FEAS_SCALE x $N_LAYERS layers, peak memory limit $PEAK_GIB_MAX GiB"
  echo "env: G1_BUILD_BUDGET_HOURS=$budget G1_POD_DEADLINE_EPOCH=${G1_POD_DEADLINE_EPOCH:-(set by launch_g1.sh)}"
  echo "deadline guard: no new step starts when less than $GUARD_MIN minutes of the pod lifetime remain"
  echo "preflight: weights $weights, bundle ($G/pod_g1lite.sh, $tools), CAL-1 files in $calib checked against the manifest (g1_tools.py calib-check)"
  echo "stop: bash $B/pod/stop_server.sh"
  echo "step check: bash $G/pod_g1lite.sh check            (env G1_CALIB=$calib G1_BASE_SELECT=experts G1_DST=$ckpt)"
  echo "step feasibility: bash $G/pod_g1lite.sh feasibility; projected build seconds = $FEAS_SCALE x (sum of the per-layer forward seconds over the $N_LAYERS layers)"
  echo "  REFUSES (G1_SESSION_FAILED, .done written) when the projection exceeds $budget h or the peak GPU memory exceeds $PEAK_GIB_MAX GiB"
  echo "step build: bash $G/pod_g1lite.sh build            (resumable, $ckpt is never deleted)"
  echo "step verify: bash $G/pod_g1lite.sh verify          (olala_quant.verify --calibrated --select experts)"
  echo "step fidelity: bf16 reference $ref when absent (serve_olala.sh bf16, smoke.sh bf16 --timeout 900, fidelity.py collect), apply the loader patch,"
  echo "  serve_olala.sh ckpt $ckpt, smoke.sh $logname --timeout 900 (exit 0 or 2), cd $B/evals && python3 fidelity.py collect --chunks $chunks"
  echo "  --base-url http://127.0.0.1:8000/v1 --model olala --out $raw/g1_wikitext2.npz, fidelity.py compare $ref $raw/g1_wikitext2.npz --json $rdir/fidelity_vs_b0.json"
  echo "  -> $rdir/fidelity.json (mean, median, p99 KL, perplexity ratio, top-1 agreement, with intervals); the server stays up"
  echo "step gate: g1_tools.py gate -> $rdir/gate.json, prints G1_GATE_PASS or G1_GATE_FAIL"
  echo "step accuracy (only on a pass; ALWAYS, whatever the verdict, when G1_ACCURACY_ALWAYS=1, now $always): bash $B/pod/run_accuracy.sh $run_name --effort medium"
  echo "  on a fail with G1_ACCURACY_ALWAYS=0: no accuracy run, G1_SESSION_GATE_FAILED and exit 0; with 1 the session ends G1_SESSION_DONE and its reason says 'gate: fail'"
  echo "the checkpoint stays on the pod; results/g1 gets its sha256 (index, config, quantization_info.json) and size in bytes"
  echo "last: $donefile, written LAST (G1_SESSION_DONE, G1_SESSION_GATE_FAILED or G1_SESSION_FAILED: REASON), then the same marker as the last line"
  exit 0
fi

if ! { [[ $budget =~ ^[0-9]+(\.[0-9]+)?$ ]] && awk "BEGIN{exit !($budget > 0)}"; }; then
  refuse "G1_BUILD_BUDGET_HOURS must be a positive number of hours, got '$budget'"
fi
[[ ${G1_POD_DEADLINE_EPOCH:-} =~ ^[0-9]+$ ]] || refuse "G1_POD_DEADLINE_EPOCH must be the epoch second of the pod's deletion (launch_g1.sh sets it), got '${G1_POD_DEADLINE_EPOCH:-}'"
deadline=$G1_POD_DEADLINE_EPOCH
[[ $stopwait =~ ^[0-9]+$ ]] && [[ $evals_poll =~ ^[0-9]+$ ]] && [[ $evals_tries =~ ^[0-9]+$ ]] || refuse "G1_STOP_WAIT, G1_EVALS_POLL and G1_EVALS_TRIES are numbers"
[[ $noise =~ ^[01]$ ]] || refuse "G1_NOISE must be 0 or 1, got '$noise'"
[[ $always =~ ^[01]$ ]] || refuse "G1_ACCURACY_ALWAYS must be 0 or 1, got '$always'"

if [ -f "$donefile" ]; then
  case $(head -n 1 "$donefile") in
    G1_SESSION_DONE | G1_SESSION_GATE_FAILED)
      echo "refused: $donefile already says $(head -n 1 "$donefile") (nothing was run; remove it to run the session again)"
      echo "$usage" >&2
      exit 2 ;;
  esac
fi
mkdir -p /workspace/run
if [ -f "$pidfile" ]; then
  other=$(cat "$pidfile" 2> /dev/null)
  if [ -n "$other" ] && [ "$other" != "$$" ] && kill -0 "$other" 2> /dev/null \
    && ps -p "$other" -o command= 2> /dev/null | grep -q g1_session.sh; then
    echo "refused: another g1_session.sh is running (pid $other, $pidfile): two sessions would fight over the GPU"
    echo "$usage" >&2
    exit 2
  fi
fi
echo $$ > "$pidfile"
trap 'rm -f "$pidfile"' EXIT

. "$here/job_common.sh"   # apply_patch, EV, QU, OUT (= $raw); its log() is replaced below
stamp=$(date -u +%Y%m%dT%H%M%SZ)-$$
mkdir -p "$rdir" "$sdir" "$raw" || { echo "G1_SESSION_FAILED: cannot create $rdir"; exit 1; }
exec 3>&1
slog=$rdir/session-$stamp.log
log() { local m; m="[$(date -u +%H:%M:%S)] $*"; echo "$m" >&3; echo "$m" >> "$slog"; }
rm -f "$donefile"   # a g1.done of an earlier attempt must not let collect.sh pull an unfinished session

serving=""; STEP_REASON=""; GATE_PASS=""
stop_srv() { bash "$B/pod/stop_server.sh" > /dev/null 2>&1 || log "warning: stop_server.sh exit $?"; serving=""; sleep "$stopwait"; }

# copy_results: small records only, merged, never deleting anything; the checkpoint's weights are never copied
copy_results() {
  local rc=0 f
  mkdir -p "$rdir/g1lite-logs" "$rdir/checkpoint-files" "$rdir/server" || return 1
  if [ -d "$g1logs" ]; then cp -a "$g1logs/." "$rdir/g1lite-logs/" || rc=1; fi
  for f in model.safetensors.index.json config.json quantization_info.json; do
    if [ -f "$ckpt/$f" ]; then cp -a "$ckpt/$f" "$rdir/checkpoint-files/" || rc=1; fi
  done
  if [ -f "$raw/g1_wikitext2.npz" ]; then cp -a "$raw/g1_wikitext2.npz" "$rdir/" || rc=1; fi
  for f in /workspace/logs/"$logname"/server-*.cmd /workspace/logs/bf16/server-*.cmd; do
    if [ -f "$f" ]; then cp -a "$f" "$rdir/server/" || rc=1; fi
  done
  for f in /workspace/logs/"$logname"/server-*.log; do
    if [ -f "$f" ]; then tail -n 300 "$f" > "$rdir/server/$(basename "$f" .log)-tail.log"; fi
  done
  return $rc
}

# finish MARKER [DETAIL]: g1.done is the very last write (the collector pulls the results the moment it exists), then the marker
finish() {
  { printf '%s\n' "$1"; [ -z "${2:-}" ] || printf '%s\n' "$2"; } > "$donefile.tmp" && mv "$donefile.tmp" "$donefile"
  echo "$1"
}
fail() {  # REASON
  local reason
  reason=$(printf '%s' "$*" | tr '\n' ' ')
  log "FAILED: $reason"
  [ -z "$serving" ] || stop_srv
  copy_results || log "warning: the evidence copy into $rdir is incomplete"
  finish "G1_SESSION_FAILED: $reason"
  exit 1
}

log "== g1 session $stamp: results $rdir, budget ${budget} h, deadline epoch $deadline ($(( (deadline - $(date +%s)) / 60 )) minutes from now)"
log "projection formula: projected build seconds = $FEAS_SCALE x (sum of the per-layer forward seconds over the $N_LAYERS layers); gate: KL <= $G1_GATE_KL_MAX, perplexity ratio <= $G1_GATE_PPL_RATIO_MAX"

# ---- preflight: inputs that cost nothing to check; a pod must not be billed for a start that cannot work -----------------------
[ -f "$weights/config.json" ] || fail "preflight: no bf16 weights at $weights"
for f in "$G/pod_g1lite.sh" "$tools" "$B/pod/serve_olala.sh" "$B/pod/smoke.sh" "$B/pod/run_accuracy.sh" "$B/evals/fidelity.py"; do
  [ -f "$f" ] || fail "preflight: the bundle lacks $f"
done
python3 "$tools" calib-check "$calib" --out "$rdir/calibration.json" > "$rdir/calibration-check-$stamp.txt" 2>&1 \
  || fail "preflight: calibration files in $calib do not match their manifest: $(tail -n 1 "$rdir/calibration-check-$stamp.txt" | cut -c1-300)"
log "calibration files checked against the manifest: $rdir/calibration.json"
{ echo "session=$stamp started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2> /dev/null; } >> "$rdir/provenance.txt"

stop_srv   # an earlier attempt's server would hold the GPU the build needs

# ---- the step runner ----------------------------------------------------------------------------------------------------------
# run_step NAME FUNCTION: skip when steps/NAME.ok exists; otherwise the deadline guard, then the function with its output in
# results/g1/NAME-STAMP.log; the function returns 0 or sets STEP_REASON and returns 1. NAME.ok is written on success, unless the
# function returns 0 and sets STEP_NO_OK.
STEP_NO_OK=""
run_step() {
  local name=$1 fn=$2 upper remaining sl rc
  upper=$(printf '%s' "$name" | tr '[:lower:]' '[:upper:]')
  if [ -e "$sdir/$name.ok" ]; then log "G1_STEP_${upper}_SKIPPED (already done: $sdir/$name.ok)"; return 0; fi
  remaining=$((deadline - $(date +%s)))
  if [ "$remaining" -lt "$guard_s" ]; then
    log "G1_STEP_${upper}_FAILED: deadline"
    fail "deadline: $((remaining / 60)) minutes of the pod lifetime remain, fewer than $GUARD_MIN, so step $name is not started"
  fi
  sl=$rdir/$name-$stamp.log
  log "== step $name (log $sl, $((remaining / 60)) minutes of the pod lifetime remain)"
  STEP_REASON=""; STEP_NO_OK=""
  "$fn" >> "$sl" 2>&1
  rc=$?
  tail -n 3 "$sl" | sed 's/^/    | /' >&3
  if [ $rc -ne 0 ]; then
    echo "G1_STEP_${upper}_FAILED: $STEP_REASON" >> "$sl"
    log "G1_STEP_${upper}_FAILED: $STEP_REASON"
    fail "step $name: ${STEP_REASON:-failed}"
  fi
  [ -n "$STEP_NO_OK" ] || : > "$sdir/$name.ok"
  echo "G1_STEP_${upper}_DONE" >> "$sl"
  log "G1_STEP_${upper}_DONE"
}

# g1lite STEP OUTFILE: pod_g1lite.sh for the G1 base; success needs exit status 0 AND its own G1LITE_<STEP>_DONE line
g1lite() {
  local step=$1 out=$2 upper rc
  upper=$(printf '%s' "$step" | tr '[:lower:]' '[:upper:]')
  G1_CALIB=$calib G1_BASE_SELECT=experts G1_DST=$ckpt bash "$G/pod_g1lite.sh" "$step" 2>&1 | tee "$out"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then STEP_REASON="pod_g1lite.sh $step exit $rc"; return 1; fi
  grep -q "^G1LITE_${upper}_DONE" "$out" || { STEP_REASON="pod_g1lite.sh $step printed no G1LITE_${upper}_DONE line"; return 1; }
}

step_check() { g1lite check "$rdir/check-output.txt"; }

step_feasibility() {
  local fo=$rdir/feasibility-output.txt out rc
  if [ -s "$fo" ] && grep -q '^G1LITE_FEASIBILITY_DONE' "$fo"; then
    log "feasibility measurement of an earlier attempt reused: $fo (the decision is taken again)"
  else
    g1lite feasibility "$fo.new" || return 1
    mv "$fo.new" "$fo"
  fi
  out=$(python3 "$tools" feasibility "$fo" --out "$rdir/feasibility.json" --scale "$FEAS_SCALE" --n-layers "$N_LAYERS" \
    --budget-hours "$budget" --peak-gib-max "$PEAK_GIB_MAX" 2>&1)
  rc=$?
  printf '%s\n' "$out"
  case $rc in
    0) ;;
    3) STEP_REASON="feasibility refused: $(printf '%s\n' "$out" | sed -n 's/^feasibility REFUSED: //p' | tr '\n' ';' | cut -c1-400)"; return 1 ;;
    *) STEP_REASON="feasibility output unreadable: $(printf '%s\n' "$out" | tail -n 1 | cut -c1-300)"; return 1 ;;
  esac
}

step_build() {
  g1lite build "$rdir/build-output.txt" || return 1
  python3 "$tools" checkpoint "$ckpt" --out "$rdir/checkpoint.json" || { STEP_REASON="checkpoint record (sha256 and size of $ckpt)"; return 1; }
  { echo "-- checkpoint $ckpt (stays on the pod)"; (cd "$ckpt" && sha256sum model.safetensors.index.json config.json quantization_info.json) \
      && echo "size_bytes=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["size_bytes"])' "$rdir/checkpoint.json")"; } >> "$rdir/provenance.txt" \
    || { STEP_REASON="provenance (sha256 of the checkpoint files)"; return 1; }
}

step_verify() {
  local report
  g1lite verify "$rdir/verify-output.txt" || return 1
  # shellcheck disable=SC2012  # the names are verify-<UTC>.json
  report=$(ls -t "$g1logs"/verify-*.json 2> /dev/null | head -n 1)
  if [ -n "$report" ]; then cp "$report" "$rdir/verify.json" || { STEP_REASON="copy of the verify report"; return 1; }; fi
}

collect_to() {  # NAME: top-20 logprobs of the 64 x 512 wikitext chunks from the server on port 8000, the same call as ablation_arm.sh
  local tmpout
  tmpout=$(mktemp)
  rm -f "$OUT/$1_wikitext2.npz"
  (cd "$EV" && python3 fidelity.py collect --chunks "$chunks" --base-url http://127.0.0.1:8000/v1 --model olala \
    --out "$OUT/$1_wikitext2.npz" > "$tmpout" 2>&1) || { tail -n 4 "$tmpout"; rm -f "$tmpout"; return 1; }
  tail -n 1 "$tmpout"
  rm -f "$tmpout"
  [ -s "$OUT/$1_wikitext2.npz" ] || return 2
}

compare_to() {  # REF VAR JSON MD
  (cd "$EV" && python3 fidelity.py compare "$1" "$2" --json "$3" > "$4" 2>&1) || { tail -n 4 "$4"; return 1; }
}

serve_ckpt() {  # serve the checkpoint as arm_pipeline.sh's ckpt mode does, and smoke-gate it
  local rc
  serving=1
  bash "$B/pod/serve_olala.sh" ckpt "$ckpt" > "$rdir/serve-$stamp.txt" 2>&1 \
    || { tail -n 4 "$rdir/serve-$stamp.txt"; STEP_REASON="serve (serve_olala.sh ckpt $ckpt failed)"; return 1; }
  tail -n 2 "$rdir/serve-$stamp.txt"
  bash "$B/pod/smoke.sh" "$logname" --timeout 900 > "$rdir/smoke-$stamp.txt" 2>&1
  rc=$?
  log "smoke exit $rc: $(head -n 2 "$rdir/smoke-$stamp.txt" | tr '\n' ' ')"
  { [ "$rc" -eq 0 ] || [ "$rc" -eq 2 ]; } || { tail -n 8 "$rdir/smoke-$stamp.txt"; STEP_REASON="smoke gate (exit $rc)"; return 1; }
  # shellcheck disable=SC2012  # the names are server-<UTC>.log: no odd characters
  grep -h -E 'Quantizing [0-9]+ layers|Selected .*Kernel|Using .* MoE backend' \
    "$(ls -t "/workspace/logs/$logname"/server-*.log 2> /dev/null | head -n 1)" 2> /dev/null | cut -c1-300 >> "$rdir/provenance.txt"
  return 0
}

step_fidelity() {
  local rc
  stop_srv
  if [ ! -f "$ref" ]; then
    log "== bf16 reference (once per pod): $ref"
    serving=1
    bash "$B/pod/serve_olala.sh" bf16 > "$rdir/reference-serve-$stamp.txt" 2>&1 \
      || { tail -n 4 "$rdir/reference-serve-$stamp.txt"; STEP_REASON="reference: serve bf16 failed"; return 1; }
    bash "$B/pod/smoke.sh" bf16 --timeout 900 > "$rdir/reference-smoke-$stamp.txt" 2>&1
    rc=$?
    log "reference smoke exit $rc: $(head -n 2 "$rdir/reference-smoke-$stamp.txt" | tr '\n' ' ')"
    { [ "$rc" -eq 0 ] || [ "$rc" -eq 2 ]; } || { STEP_REASON="reference: bf16 smoke gate (exit $rc)"; return 1; }
    collect_to b0 || { STEP_REASON="reference: bf16 collection failed or wrote no logprob file"; return 1; }
    if [ "$noise" -eq 1 ]; then collect_to b0repeat || log "warning: the repeat collection (noise floor) failed"; fi
    stop_srv
    mkdir -p "$rdir/reference"
    cp "$ref" "$rdir/reference/" || { STEP_REASON="reference: copy of $ref"; return 1; }
    if [ "$noise" -eq 1 ] && [ -s "$OUT/b0repeat_wikitext2.npz" ]; then
      compare_to "$ref" "$OUT/b0repeat_wikitext2.npz" "$rdir/reference/noise_floor.json" "$rdir/reference/noise_floor.md" \
        || log "warning: the noise-floor comparison failed"
    fi
  else
    log "reference present: $ref"
  fi
  apply_patch || { STEP_REASON="loader patch"; return 1; }
  serve_ckpt || return 1
  log "== fidelity g1 against $ref"
  collect_to g1
  rc=$?
  [ "$rc" -ne 1 ] || { STEP_REASON="fidelity collect failed (fidelity.py collect exit status not 0)"; return 1; }
  [ "$rc" -eq 0 ] || { STEP_REASON="fidelity collect wrote no logprob file"; return 1; }
  compare_to "$ref" "$OUT/g1_wikitext2.npz" "$rdir/fidelity_vs_b0.json" "$rdir/fidelity_vs_b0.md" \
    || { STEP_REASON="fidelity compare (fidelity.py compare exit status not 0)"; return 1; }
  grep -E 'Perplexity ratio|Mean KL|99th|top-1' "$rdir/fidelity_vs_b0.md" | sed 's/^/g1: /'
  python3 "$tools" fidelity "$rdir/fidelity_vs_b0.json" --reference "$ref" --out "$rdir/fidelity.json" \
    || { STEP_REASON="fidelity record (fidelity_vs_b0.json lacks the headline numbers)"; return 1; }
  # the server stays up for the accuracy run (the gate step stops it on a fail)
}

step_gate() {
  local rc
  python3 "$tools" gate "$rdir/fidelity.json" --out "$rdir/gate.json" --kl-max "$G1_GATE_KL_MAX" \
    --ppl-ratio-max "$G1_GATE_PPL_RATIO_MAX" --n5-kl "$N5_KL" --n5-ppl-ratio "$N5_PPL_RATIO" --n5-from "$N5_FROM" \
    --source "$G1_GATE_SOURCE"
  rc=$?
  case $rc in
    0) GATE_PASS=1; log "G1_GATE_PASS" ;;
    1) GATE_PASS=0; log "G1_GATE_FAIL" ;;
    *) STEP_REASON="gate (g1_tools.py gate exit $rc: fidelity.json unreadable or not a number)"; return 1 ;;
  esac
  STEP_NO_OK=1   # the verdict is taken again on every attempt, from fidelity.json
}

step_accuracy() {
  local rc
  local rundir=$raw/$run_name i
  if [ -d "$rundir" ]; then
    if python3 "$tools" run-status "$rundir/run_info.json"; then
      log "$rundir is already a complete run: taken as done"
      return 0
    fi
    STEP_REASON="$rundir exists and is not complete: move it aside (it is never overwritten or deleted here), then rerun"
    return 1
  fi
  for ((i = 0; i < evals_tries; i++)); do
    grep -q EVALS_ENV_DONE /workspace/logs/setup/evals-env-*.log 2> /dev/null && break
    grep -q EVALS_ENV_FAILED /workspace/logs/setup/evals-env-*.log 2> /dev/null && { STEP_REASON="evals environment setup failed"; return 1; }
    sleep "$evals_poll"
  done
  grep -q EVALS_ENV_DONE /workspace/logs/setup/evals-env-*.log 2> /dev/null || { STEP_REASON="evals environment not ready after the wait"; return 1; }
  if [ -z "$serving" ]; then   # a rerun: the server is not up, start it the same way
    stop_srv
    apply_patch || { STEP_REASON="loader patch"; return 1; }
    serve_ckpt || return 1
  fi
  log "== run_accuracy $run_name"
  bash "$B/pod/run_accuracy.sh" "$run_name" --effort medium
  rc=$?
  log "run_accuracy exit $rc"
  [ "$rc" -eq 0 ] || { STEP_REASON="run_accuracy exit $rc"; return 1; }
  python3 "$tools" run-status "$rundir/run_info.json" || { STEP_REASON="run_info.json of $run_name does not say status complete"; return 1; }
}

run_step check step_check
run_step feasibility step_feasibility
run_step build step_build
run_step verify step_verify
run_step fidelity step_fidelity
run_step gate step_gate

if [ "$GATE_PASS" = 1 ] || [ "$always" = 1 ]; then
  run_step accuracy step_accuracy
  [ -z "$serving" ] || stop_srv
  copy_results || fail "copy (the result could not be copied into $rdir)"
  detail=$(python3 -c 'import json,sys; g=json.load(open(sys.argv[1])); print("gate: " + ("pass" if g["pass"] else "fail") + ", kl", g["kl"], "(max", str(g["thresholds"]["kl_max"]) + "), ppl_ratio", g["ppl_ratio"], "(max", str(g["thresholds"]["ppl_ratio_max"]) + "), accuracy run", sys.argv[2], "complete")' "$rdir/gate.json" "$run_name")
  log "the accuracy run is complete (gate verdict in $rdir/gate.json, G1_ACCURACY_ALWAYS=$always); writing $donefile"
  finish G1_SESSION_DONE "$detail"
else
  [ -z "$serving" ] || stop_srv
  copy_results || fail "copy (the result could not be copied into $rdir)"
  log "G1 gate failed: no accuracy run (a legitimate negative result); writing $donefile"
  finish G1_SESSION_GATE_FAILED "gate: fail, $(python3 -c 'import json,sys; g=json.load(open(sys.argv[1])); print("kl", g["kl"], "max", g["thresholds"]["kl_max"], "ppl_ratio", g["ppl_ratio"], "max", g["thresholds"]["ppl_ratio_max"])' "$rdir/gate.json")"
fi
exit 0
