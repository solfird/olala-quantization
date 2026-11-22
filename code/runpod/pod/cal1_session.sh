#!/usr/bin/env bash
# CAL-1 (PROTOCOL amendment 17) generated end to end on ONE fresh pod, unattended: serve the bf16 arm, smoke-gate it, build the
# provenance string, run quant/gptq/pod_cal1.sh plan, run and verify, stop the server, copy the result into /workspace/results/cal1/
# and write /workspace/results/cal1.done LAST. Detach it; the log is the record.
# Usage: cal1_session.sh [--dry-run]
# Env: CAL1_OUT (default /workspace/calib-cal1: the tool's output directory, an absolute path of A-Za-z0-9._/-, not inside
#      /workspace/results), CAL1_STOP_WAIT (seconds to wait after the first stop_server, default 2). Nothing else is read.
# Inputs: the bf16 weights /workspace/models/olala-bf16, the image record /workspace/logs/setup/image.txt (pod_setup.sh writes
#   it), and the bundle's quant/gptq/cal1_revisions.txt (lines SOURCE=40-hex-SHA, # comments allowed): the session refuses, before
#   any server is started, when that file is missing or one source lacks a 40-hex SHA (the tool pins nothing itself).
# Provenance (CAL1_PROVENANCE, one line, the same on every rerun because the tool refuses a changed one): the vLLM image tag and
#   digest, the weights (sha256 of config.json; "identity" = sha256 of the sorted `sha256  name` listing of the *.safetensors),
#   and the flags of the server, read from the `command:` line of its own server-*.cmd file (no timestamp, no log name).
# Files: /workspace/results/cal1/ = calib-cal1/ (a copy of the tool's output directory, with leakage_audit.json), provenance.txt,
#   provenance-details-STAMP.txt, session-STAMP.log, serve/smoke/plan/run/verify-STAMP.txt, server-*.cmd, pod_cal1-logs/ (the tool's
#   own logs). STAMP = UTC time and the pid: no file of an earlier attempt is ever overwritten or removed. And
#   /workspace/results/cal1.done: the file runpod/collect.sh files mode waits for, written LAST, on success (CAL1_SESSION_DONE)
#   and on every failure after the arguments were accepted (CAL1_SESSION_FAILED: REASON), so that the evidence of a failed
#   session (a leakage refusal above all) is pulled before the pod is deleted. Read the first line of the file.
# Resumable: the tool resumes from its output directory, so a rerun continues where the last one stopped; this script never
#   clears that directory or the results. A stale cal1.done is removed when a rerun starts; a session that is already DONE is
#   refused (exit 2); two sessions at the same time are refused (pid file).
# Exit: 0 done, 1 a step failed (a leakage refusal included), 2 bad arguments or inputs. The last line is CAL1_SESSION_DONE or
#   CAL1_SESSION_FAILED: REASON. The pod must have `datasets` for the prompt sources: if python3 cannot import it, an isolated
#   venv (/workspace/venv-cal1, system site packages kept) gets datasets==5.2.0 (quant/gptq/pyproject.toml) before any server starts.
set -u
here=$(cd "$(dirname "$0")" && pwd)
usage="usage: cal1_session.sh [--dry-run]   (env CAL1_OUT)"
refuse() { echo "$usage" >&2; echo "CAL1_SESSION_FAILED: $*"; exit 2; }

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
rdir=$results/cal1
donefile=$results/cal1.done
weights=/workspace/models/olala-bf16
imgrec=/workspace/logs/setup/image.txt
serverlogs=/workspace/logs/bf16
pidfile=/workspace/run/cal1_session.pid
venv=/workspace/venv-cal1
out=${CAL1_OUT:-/workspace/calib-cal1}

[[ $out =~ ^/[A-Za-z0-9._/-]+$ ]] && [[ $out != *..* ]] || refuse "bad CAL1_OUT '$out' (an absolute path of A-Za-z0-9._/-, no '..')"
case $out in
  "$rdir" | "$rdir"/* | "$results" | /workspace) refuse "CAL1_OUT '$out' is the results directory or the workspace itself (the result is copied into $rdir)" ;;
esac

# read_revisions FILE: sets $revisions ("source=SHA source=SHA ...") or returns 1 with the reason in $revfail
revisions=""; revfail=""
read_revisions() {
  local file=$1 line n=0 seen=" "
  [ -f "$file" ] || { revfail="$file not found"; return 1; }
  while IFS= read -r line || [ -n "$line" ]; do
    n=$((n + 1))
    line=${line%%#*}
    line=${line#"${line%%[![:space:]]*}"}; line=${line%"${line##*[![:space:]]}"}
    [ -n "$line" ] || continue
    if [[ $line =~ ^([a-z0-9_]+)=([0-9a-f]{40})$ ]]; then
      case $seen in *" ${BASH_REMATCH[1]} "*) revfail="$file line $n: source ${BASH_REMATCH[1]} given twice"; return 1 ;; esac
      seen="$seen${BASH_REMATCH[1]} "
      revisions="${revisions:+$revisions }$line"
    else
      revfail="$file line $n: '$line' is not SOURCE=SHA with a 40-hex lowercase SHA"; return 1
    fi
  done < "$file"
  [ -n "$revisions" ] || { revfail="$file holds no SOURCE=SHA line"; return 1; }
}
revfile=$G/cal1_revisions.txt
# the Mac has no bundle: the dry run of launch_cal1.sh reads the repo's own file (runpod/pod/ is two levels below the repo root)
if [ $dry -eq 1 ] && [ ! -f "$revfile" ]; then revfile=$here/../../quant/gptq/cal1_revisions.txt; fi
read_revisions "$revfile" || refuse "cal1_revisions: $revfail"

if [ $dry -eq 1 ]; then
  echo "cal1_session.sh dry run (nothing is run, nothing is written)"
  echo "revisions ($revfile): $revisions"
  echo "refuses when $donefile already says CAL1_SESSION_DONE, or another session is running (pid file $pidfile)"
  echo "check: $weights/config.json and *.safetensors, $imgrec (image=), python3 can import datasets (else a venv $venv with datasets==5.2.0)"
  echo "stop: bash $B/pod/stop_server.sh"
  echo "serve: bash $B/pod/serve_olala.sh bf16"
  echo "smoke gate (exit 0 or 2 accepted): bash $B/pod/smoke.sh bf16 --timeout 900"
  echo "provenance: image tag from $imgrec, sha256 of config.json and of the sorted listing of the weights, flags of the newest $serverlogs/server-*.cmd"
  echo "env of the tool: CAL1_REVISIONS (above), CAL1_PROVENANCE, CAL1_OUT=$out"
  echo "plan: bash $G/pod_cal1.sh plan"
  echo "run (resumable, the output directory is never cleared): bash $G/pod_cal1.sh run   (a leakage refusal is a failed session, the audit output is kept)"
  echo "verify: bash $G/pod_cal1.sh verify"
  echo "stop: bash $B/pod/stop_server.sh"
  echo "copy: $out, the logs of the tool, provenance and outputs of every step -> $rdir (nothing there is deleted)"
  echo "last: $donefile, written LAST (CAL1_SESSION_DONE, or CAL1_SESSION_FAILED: REASON after a failure), then the same marker as the last line"
  exit 0
fi

if [ -f "$donefile" ] && [ "$(head -n 1 "$donefile")" = CAL1_SESSION_DONE ]; then
  refuse "$donefile already says CAL1_SESSION_DONE (nothing was run; remove it to run the session again)"
fi
mkdir -p /workspace/run
if [ -f "$pidfile" ]; then
  other=$(cat "$pidfile" 2> /dev/null)
  if [ -n "$other" ] && [ "$other" != "$$" ] && kill -0 "$other" 2> /dev/null \
    && ps -p "$other" -o command= 2> /dev/null | grep -q cal1_session.sh; then
    refuse "another cal1_session.sh is running (pid $other, $pidfile): two sessions would fight over the server"
  fi
fi
echo $$ > "$pidfile"
trap 'rm -f "$pidfile"' EXIT

stamp=$(date -u +%Y%m%dT%H%M%SZ)-$$
mkdir -p "$rdir" || refuse "cannot create $rdir"
slog=$rdir/session-$stamp.log
log() { local m; m="[$(date -u +%H:%M:%S)] $*"; echo "$m"; echo "$m" >> "$slog"; }
rm -f "$donefile"   # a cal1.done of an earlier attempt must not let collect.sh pull an unfinished session

serving=""; cmdfile=""; serverlog=""
stop_srv() { bash "$B/pod/stop_server.sh" > /dev/null 2>&1 || log "warning: stop_server.sh exit $?"; serving=""; }

# copy_results: the tool's output directory, its logs and the server's record into $rdir, merged, never deleting anything
copy_results() {
  local rc=0
  mkdir -p "$rdir/calib-cal1" "$rdir/pod_cal1-logs" || return 1
  if [ -d "$out" ]; then cp -a "$out/." "$rdir/calib-cal1/" || rc=1; else log "no output directory $out (the tool never created it)"; fi
  if [ -d /workspace/logs/cal1 ]; then cp -a /workspace/logs/cal1/. "$rdir/pod_cal1-logs/" || rc=1; fi
  if [ -n "$cmdfile" ]; then cp -a "$cmdfile" "$rdir/" || rc=1; fi
  if [ -n "$serverlog" ] && [ -f "$serverlog" ]; then tail -n 300 "$serverlog" > "$rdir/server-tail-$stamp.log"; fi
  return $rc
}

# finish MARKER: cal1.done is the very last write (the collector pulls the results the moment it exists), then the marker line
finish() {
  printf '%s\n' "$1" > "$donefile.tmp" && mv "$donefile.tmp" "$donefile"
  echo "$1"
}
fail() {  # REASON
  local reason
  reason=$(printf '%s' "$*" | tr '\n' ' ')
  log "FAILED: $reason"
  [ -z "$serving" ] || stop_srv
  copy_results || log "warning: the evidence copy into $rdir is incomplete"
  finish "CAL1_SESSION_FAILED: $reason"
  exit 1
}

log "== cal1 session: out $out, results $rdir, revisions $revisions"

# 1 inputs and provenance parts that need no server: a pod must not be billed for a start that cannot work
[ -f "$weights/config.json" ] || fail "provenance (no $weights/config.json: the bf16 weights are not on this pod)"
sfiles=$(cd "$weights" && ls -1 ./*.safetensors 2> /dev/null | sed 's#^\./##' | LC_ALL=C sort)
[ -n "$sfiles" ] || fail "provenance (no *.safetensors in $weights)"
[ -f "$G/pod_cal1.sh" ] || fail "bundle lacks quant/gptq/pod_cal1.sh"
image=$(sed -n 's/^image=//p' "$imgrec" 2> /dev/null | head -n 1)
digest=$(sed -n 's/^digest=//p' "$imgrec" 2> /dev/null | head -n 1)
{ [ -n "$image" ] && [ "$image" != "(not given)" ]; } || fail "provenance (no image= line in $imgrec: the vLLM image tag cannot be recorded)"
cfg_sha=$(cd "$weights" && sha256sum config.json | cut -d' ' -f1) || fail "provenance (sha256 of config.json)"
# shellcheck disable=SC2086  # $sfiles is a list of file names without spaces
listing=$(cd "$weights" && sha256sum $sfiles) || fail "provenance (sha256 of the safetensors files)"
wt_sha=$(printf '%s\n' "$listing" | sha256sum | cut -d' ' -f1)
nshards=$(printf '%s\n' "$listing" | wc -l | tr -d ' ')
[ -n "$cfg_sha" ] && [ -n "$wt_sha" ] || fail "provenance (empty sha256)"

py=${PYTHON:-python3}
if ! "$py" -c 'import datasets' > /dev/null 2>&1; then
  log "datasets is not importable by $py: isolated venv $venv (system site packages kept), datasets==5.2.0"
  { [ -x "$venv/bin/python3" ] || "$py" -m venv --system-site-packages "$venv"; } > "$rdir/venv-$stamp.txt" 2>&1 \
    && "$venv/bin/python3" -m pip install --quiet datasets==5.2.0 >> "$rdir/venv-$stamp.txt" 2>&1 \
    && "$venv/bin/python3" -c 'import datasets' >> "$rdir/venv-$stamp.txt" 2>&1 \
    || { tail -n 4 "$rdir/venv-$stamp.txt"; fail "datasets package unavailable (python3 cannot import it and the isolated venv install failed)"; }
  py=$venv/bin/python3
fi
export PYTHON=$py

# 2 serve the bf16 arm and gate on the smoke test
bash "$B/pod/stop_server.sh" > /dev/null 2>&1; sleep "${CAL1_STOP_WAIT:-2}"   # idempotent: nothing running is fine
serving=1
bash "$B/pod/serve_olala.sh" bf16 > "$rdir/serve-$stamp.txt" 2>&1 || { tail -n 4 "$rdir/serve-$stamp.txt"; fail "serve (serve_olala.sh bf16 failed)"; }
tail -n 2 "$rdir/serve-$stamp.txt"
bash "$B/pod/smoke.sh" bf16 --timeout 900 > "$rdir/smoke-$stamp.txt" 2>&1
rc=$?
log "smoke exit $rc: $(head -n 2 "$rdir/smoke-$stamp.txt" | tr '\n' ' ')"
{ [ $rc -eq 0 ] || [ $rc -eq 2 ]; } || { tail -n 8 "$rdir/smoke-$stamp.txt"; fail "smoke gate (exit $rc)"; }
smoke_rc=$rc

# 3 provenance, with the flags of the server we just started
cmdfile=$(ls -1 "$serverlogs"/server-*.cmd 2> /dev/null | LC_ALL=C sort | tail -n 1)
serverlog=${cmdfile%.cmd}.log
flags=""
[ -z "$cmdfile" ] || flags=$(sed -n 's/^command: //p' "$cmdfile" | head -n 1 | tr -s ' \t' ' ')
[ -n "$flags" ] || fail "provenance (no server-*.cmd with a command: line in $serverlogs: the server flags cannot be recorded)"
prov="vLLM image $image${digest:+ ($digest)}; weights olala-bf16: config.json sha256 $cfg_sha, $nshards safetensors files, identity sha256 $wt_sha; server flags: $flags"
if [ -f "$rdir/provenance.txt" ] && [ "$(cat "$rdir/provenance.txt")" != "$prov" ]; then
  mv "$rdir/provenance.txt" "$rdir/provenance-before-$stamp.txt"   # the tool will refuse this change; the old one stays as evidence
fi
printf '%s' "$prov" > "$rdir/provenance.txt"
{
  echo "session=$stamp smoke_exit=$smoke_rc"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2> /dev/null
  echo "-- $imgrec"; cat "$imgrec"
  echo "-- sha256 of the weights (identity sha256 = sha256 of this listing: $wt_sha)"; echo "$cfg_sha  config.json"; echo "$listing"
  echo "-- $cmdfile"; cat "$cmdfile"
  echo "-- CAL1_PROVENANCE"; echo "$prov"
} > "$rdir/provenance-details-$stamp.txt"
log "provenance in $rdir/provenance.txt"

export CAL1_REVISIONS=$revisions CAL1_OUT=$out CAL1_PROVENANCE=$prov

# 4 plan, run, verify (the tool's own steps; run resumes from $out)
tool() {  # STEP: the output goes to the log and to results/cal1/STEP-STAMP.txt
  bash "$G/pod_cal1.sh" "$1" 2>&1 | tee "$rdir/$1-$stamp.txt"
  return "${PIPESTATUS[0]}"
}
log "== pod_cal1.sh plan"
tool plan || fail "plan (pod_cal1.sh plan exit $?)"
log "== pod_cal1.sh run"
tool run
rc=$?
if [ $rc -ne 0 ]; then
  if grep -q 'REFUSED (leakage)' "$rdir/run-$stamp.txt"; then
    fail "leakage refusal by cal1_generate.py (the 8-gram audit found an overlap; no manifest; audit output kept in $rdir)"
  fi
  fail "run (pod_cal1.sh run exit $rc)"
fi
log "== pod_cal1.sh verify"
tool verify || fail "verify (pod_cal1.sh verify exit $?)"

# 5 stop the server, copy the result, release the collector
stop_srv
copy_results || fail "copy (the result could not be copied into $rdir)"
[ -f "$rdir/calib-cal1/manifest.json" ] || fail "no manifest.json in $out after a run and a verify that said ok"
log "CAL-1 is in $rdir/calib-cal1; writing $donefile"
finish CAL1_SESSION_DONE
