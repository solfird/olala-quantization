#!/usr/bin/env bash
# Several component ablations one after the other on ONE pod (pod/ablation_arm.sh per pair), so that one pod, one bf16
# reference (made by the first run, reused by the others) and one 4-hour session answer several questions. A failed spec
# does not stop the next; a bf16 reference that cannot be made does (every later run would fail the same way).
# Usage: ablation_batch.sh LABEL SPEC [LABEL SPEC ...]      e.g. abl-experts experts abl-mamba mamba
# Env: ABLATION_DEADLINE_MIN minutes after which no new spec is started (default 195: a run is guessed at 15 minutes, unmeasured, and the
#   watchdog of a 4-hour pod deletes it at 240); the rest goes to ablation_arm.sh unchanged. A spec that is not started
#   gets a /workspace/results/ablation-LABEL.done saying so, so that its collector does not wait for it.
# Exit: 0 every spec done, 1 a spec failed or was skipped, 2 bad arguments, 3 the reference could not be made. The last
#   line is ABLATION_BATCH_DONE ok=N failed=N skipped=N.
set -u
here=$(cd "$(dirname "$0")" && pwd)
usage="usage: ablation_batch.sh LABEL SPEC [LABEL SPEC ...]   (env ABLATION_DEADLINE_MIN)"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
bad() { echo "$usage" >&2; echo "$*" >&2; exit 2; }

pairs=("$@")
n=${#pairs[@]}
{ [ "$n" -ge 2 ] && [ $((n % 2)) -eq 0 ]; } || bad "expected LABEL SPEC pairs, got $n argument(s)"
deadline_min=${ABLATION_DEADLINE_MIN-195}
[[ $deadline_min =~ ^[0-9]+$ ]] || bad "ABLATION_DEADLINE_MIN must be a number of minutes, got '$deadline_min'"
for ((i = 0; i < n; i += 2)); do
  label=${pairs[i]}
  [[ $label =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || bad "bad LABEL '$label'"
  for ((j = 0; j < i; j += 2)); do
    [ "${pairs[j]}" != "$label" ] || bad "LABEL '$label' given twice"
  done
done

results=/workspace/results
mkdir -p "$results"
deadline=$(( $(date +%s) + deadline_min * 60 ))
ok=0; failed=0; skipped=0; stop_reason=""
skip() {  # LABEL WHY
  skipped=$((skipped + 1))
  log "SKIPPED $1: $2"
  printf 'ABLATION_FAILED: skipped, %s\n' "$2" > "$results/ablation-$1.done"
}
for ((i = 0; i < n; i += 2)); do
  label=${pairs[i]}; spec=${pairs[i + 1]}
  if [ -n "$stop_reason" ]; then skip "$label" "$stop_reason"; continue; fi
  if [ "$(date +%s)" -ge "$deadline" ]; then skip "$label" "deadline of $deadline_min minutes reached before it started"; continue; fi
  log "== batch $((i / 2 + 1)) of $((n / 2)): $label $spec"
  bash "$here/ablation_arm.sh" "$label" "$spec"
  rc=$?
  if [ $rc -eq 0 ]; then
    ok=$((ok + 1))
  else
    failed=$((failed + 1))
    [ $rc -ne 3 ] || stop_reason="the bf16 reference could not be made"
  fi
done
echo "ABLATION_BATCH_DONE ok=$ok failed=$failed skipped=$skipped"
[ -z "$stop_reason" ] || exit 3
[ $((failed + skipped)) -eq 0 ] || exit 1
exit 0
