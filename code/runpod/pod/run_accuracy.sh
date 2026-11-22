#!/usr/bin/env bash
# Runs ON the pod: score one arm with the accuracy harness (evals/run_tasks.py) against the server that is already up.
# Usage: run_accuracy.sh RUNNAME [--effort LEVEL] [--tasks LIST] [--items-from FILE] [--n-items SPEC ...]
#                        [--max-connections N] [--dry-run] [-- EXTRA run_tasks.py ARGS]
#   RUNNAME           the run folder /workspace/results/raw/RUNNAME (refused if it exists: a run is never overwritten)
#   --effort          disabled, low, medium (default), high: sent as chat_template_kwargs.reasoning_effort
#   --tasks           comma or space separated; default the five tasks of the primary metric (mgsm humaneval_plus
#                     mbpp_plus ifeval aime2025: PROTOCOL.md section 4); gsm8k is not one of them, add it by name
#   --items-from      items.json of another run: score exactly those items (then no default --n-items)
#   --n-items SPEC    N or TASK=N, repeatable; default mgsm=550 (50 per language) unless --items-from is given
#   --max-connections N: concurrent requests for every task (default 96; the server runs --max-num-seqs 256). It is
#                     passed to run_tasks.py as a BARE value, which the harness lets lose to its built-in per-task
#                     default aime2025=24 (AIME's long requests would otherwise queue on the server, the KV cache
#                     holds about 23 full-length sequences, and the client timeout would count the queueing:
#                     PROTOCOL.md amendment 10). To change AIME, add `-- --max-connections aime2025=N` (TASK=N beats
#                     the built-in and the bare value)
#   --                everything after it is appended to the run_tasks.py command, last, so it wins (e.g. --epochs aime2025=3)
#   --dry-run         print the exact command and the paths; touch nothing, contact nothing
# Fixed here: seed 0 (item selection), --sandbox local (a pod has no Docker; the pod is disposable and holds no secret,
# see evals/DECISIONS.md), served model name olala, server at OLALA_BASE_URL (default http://127.0.0.1:8000).
# It ASSUMES the server of the arm is running and smoke-tested (serve_olala.sh, then smoke.sh): it checks /health and
# refuses (exit 2) otherwise; a missing smoke.json only warns. It starts nothing and stops nothing.
# Output: /workspace/results/raw/RUNNAME (samples.jsonl, items.json, run_info.json, logs/), plus for provenance the
# server's command file (server.cmd), server_provenance.txt (arm, the name and path of the server log, the served
# directory, or the GGUF file of a llama-server with its llama.cpp commit and pins) and, when the served directory
# has one, its quantization_info.json. The tool's output goes to
# /workspace/logs/RUNNAME/run_tasks.log and to stdout. The server log itself stays in /workspace/logs/ARM (pod_down.sh
# salvages it); only its name is recorded.
# Exit: 0 every task finished (status complete); 1 a task failed or the run is not complete; 2 refused before any
# request. The last line is RUN_ACCURACY_DONE or RUN_ACCURACY_FAILED. Safe to start with `pod_ssh.sh detach`: no stdin,
# no terminal, no relative path.
set -u
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=evals_env.sh
. "$here/evals_env.sh"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN   # no token of any kind lives on a pod

refuse() { echo "error: $*" >&2; exit 2; }

DEFAULT_TASKS="mgsm humaneval_plus mbpp_plus ifeval aime2025"
name=""; effort=medium; tasks=""; items_from=""; n_items=(); max_conn=96; dry=0; extra=()
while [ $# -gt 0 ]; do
  case $1 in
    --effort) effort=${2:-}; shift 2 ;;
    --tasks) tasks=${2:-}; shift 2 ;;
    --items-from) items_from=${2:-}; shift 2 ;;
    --n-items) n_items+=("${2:-}"); shift 2 ;;
    --max-connections) max_conn=${2:-}; shift 2 ;;
    --dry-run) dry=1; shift ;;
    --) shift; extra=("$@"); break ;;
    -*) refuse "unknown option $1 (usage: run_accuracy.sh RUNNAME [--effort L] [--tasks LIST] [--items-from F] [--n-items SPEC ...] [--max-connections N] [--dry-run] [-- EXTRA])" ;;
    *) [ -z "$name" ] || refuse "one run name only (got '$name' and '$1')"; name=$1; shift ;;
  esac
done
[ -n "$name" ] || refuse "usage: run_accuracy.sh RUNNAME [options] [-- EXTRA run_tasks.py ARGS]"
[[ $name =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || refuse "RUNNAME must be letters, digits, dot, dash, underscore (1 to 64 characters), got '$name'"
case $effort in disabled|low|medium|high) ;; *) refuse "--effort must be disabled, low, medium or high, got '$effort'" ;; esac
[[ $max_conn =~ ^[1-9][0-9]*$ ]] || refuse "--max-connections must be a positive integer, got '$max_conn'"
tasks=${tasks:-$DEFAULT_TASKS}; tasks=${tasks//,/ }
if [ -z "$items_from" ] && [ ${#n_items[@]} -eq 0 ]; then n_items=("mgsm=550"); fi

base=${OLALA_BASE_URL:-http://127.0.0.1:8000}
out=$OLALA_WORK/results/raw/$name
logdir=$LOG_ROOT/$name
log=$logdir/run_tasks.log
read -ra uv_args <<< "$UV_RUN_ARGS"
read -ra task_list <<< "$tasks"
cmd=(uv "${uv_args[@]}" python run_tasks.py --tasks "${task_list[@]}" --base-url "$base/v1" --model olala --out "$out"
  --seed 0 --sandbox local --reasoning-effort "$effort" --max-connections "$max_conn")
[ -z "$items_from" ] || cmd+=(--items-from "$items_from")
for spec in ${n_items[@]+"${n_items[@]}"}; do cmd+=(--n-items "$spec"); done
[ ${#extra[@]} -eq 0 ] || cmd+=("${extra[@]}")

if [ $dry -eq 1 ]; then
  echo "run_accuracy.sh $name (dry run: nothing is contacted, nothing is written)"
  echo "server: $base (checked at /health before a real run)"
  echo "out=$out"
  echo "log=$log"
  echo "command: cd $EVALS_DIR && ${cmd[*]}"
  exit 0
fi

[ ! -e "$out" ] || refuse "$out already exists: a run is never overwritten (choose another RUNNAME)"
[ -f "$EVALS_DIR/pyproject.toml" ] || refuse "$EVALS_DIR is not the evals folder of a bundle: run setup_evals_env.sh first"
[ -z "$items_from" ] || [ -f "$items_from" ] || refuse "--items-from file $items_from not found"
python3 -I -c 'import sys, urllib.request
sys.exit(0 if urllib.request.urlopen(sys.argv[1], timeout=10).status == 200 else 1)' "$base/health" 2> /dev/null \
  || refuse "no healthy server at $base/health: serve_olala.sh ARM, then smoke.sh ARM, then this script"

# Provenance of the server that answers: its serve_olala.sh record (arm, log, command, served directory).
serve_log=""; serve_arm=""; serve_cmd=""; served_dir=""
if [ -f "$RUN_DIR/server.log" ]; then
  serve_log=$(cat "$RUN_DIR/server.log"); serve_arm=$(cat "$RUN_DIR/server.arm" 2> /dev/null)
  serve_cmd=${serve_log%.log}.cmd
  [ ! -f "$serve_cmd" ] || served_dir=$(sed -n 's/^command: vllm serve \([^ ]*\).*/\1/p' "$serve_cmd" | head -n 1)
  # a llama-server record (serve_gguf.sh): the served "directory" is the GGUF file given with -m
  [ -n "$served_dir" ] || [ ! -f "$serve_cmd" ] || served_dir=$(sed -n 's/^command: .*llama-server -m \([^ ]*\).*/\1/p' "$serve_cmd" | head -n 1)
  [ -f "$(dirname "$serve_log")/smoke.json" ] || echo "warning: no smoke.json next to $serve_log: was this server smoke-tested?" >&2
else
  echo "warning: no serve_olala.sh record in $RUN_DIR (a server started by hand?): the server log is not recorded" >&2
fi
mkdir -p "$out" "$logdir" || refuse "cannot create $out and $logdir"
quant_info=none
if [ -n "$served_dir" ] && [ -f "$served_dir/quantization_info.json" ]; then
  cp "$served_dir/quantization_info.json" "$out/quantization_info.json" && quant_info=quantization_info.json
fi
[ -z "$serve_cmd" ] || [ ! -f "$serve_cmd" ] || cp "$serve_cmd" "$out/server.cmd"
{
  echo "copied_utc=$(utc_iso)"
  echo "arm=${serve_arm:-unknown}"
  echo "server_log_name=$([ -n "$serve_log" ] && basename "$serve_log" || echo none)"
  echo "server_log=${serve_log:-none}"
  echo "server_cmd=$([ -f "$out/server.cmd" ] && echo server.cmd || echo none)"
  echo "served_dir=${served_dir:-unknown}"
  echo "quantization_info=$quant_info"
  # a llama-server record also carries what the GGUF comparison needs: the fork's commit and the file's pins
  if [ -f "$out/server.cmd" ] && grep -q '^llama_cpp_commit=' "$out/server.cmd"; then
    for key in llama_cpp_repo llama_cpp_commit gguf_file gguf_size gguf_sha256_pinned gguf_sha256_verified_by_setup hf_repo hf_revision np ctx slot_ctx; do
      echo "$key=$(sed -n "s/^$key=//p" "$out/server.cmd" | head -n 1)"
    done
  fi
  echo "server_url=$base"
  echo "command=${cmd[*]}"
} > "$out/server_provenance.txt"

run() {
  echo "== run_accuracy $(utc_iso): $name against $base (arm ${serve_arm:-unknown}), out $out"
  echo "command: cd $EVALS_DIR && ${cmd[*]}"
  (cd "$EVALS_DIR" && "${cmd[@]}")
  rc=$?
  status=$(python3 -I -c 'import json, sys
try:
    info = json.load(open(sys.argv[1]))
except (OSError, ValueError):
    print("missing")
else:
    print(info.get("status", "unknown"), info.get("n_samples"), info.get("request_errors"), info.get("truncations"))' \
    "$out/run_info.json")
  read -r st n_samples n_errors n_trunc <<< "$status"
  echo "run_tasks exit $rc; run_info status=${st:-?} n_samples=${n_samples:-?} request_errors=${n_errors:-?} truncations=${n_trunc:-?}"
  if [ "$rc" -ne 0 ]; then echo "RUN_ACCURACY_FAILED: run_tasks.py exited with $rc"; return 1; fi
  if [ "${st:-?}" != complete ]; then echo "RUN_ACCURACY_FAILED: run_info.json status is '${st:-?}', not 'complete'"; return 1; fi
  echo "RUN_ACCURACY_DONE"
}
run 2>&1 < /dev/null | tee -a "$log"
exit "${PIPESTATUS[0]}"
