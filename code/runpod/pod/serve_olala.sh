#!/usr/bin/env bash
# Start ONE vLLM server for an arm of the study (arms are data: arms.json, turned into a command by arms.py).
# Usage: serve_olala.sh [--dry-run] [--cc X.Y] [--port N] [--no-snapshot] ARM [DIR] [-- EXTRA_VLLM_FLAGS...]
#   ARM    bf16, fp8pc, fp8pt, fp8pb, mxfp4, nvfp4pt, mxfp8, int8wo (each also as ARM_noignore), or ckpt DIR;
#          `arms.py list` prints them with their descriptions
#   --cc   the GPU's compute capability when nvidia-smi is not asked (dry runs); otherwise read from nvidia-smi
# One server per pod: port 8000, pid file /workspace/run/server.pid, log /workspace/logs/ARM/server-<UTC>.log
# (every start keeps its own log, failed ones included), the command in server-<UTC>.cmd. The server runs in
# its own process group so that stop_server.sh ends it and its engine processes by pid, never by text.
# A GPU snapshot log (gpu_snapshot.sh) starts with it. It returns at once: smoke.sh ARM waits for /health.
set -u
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=common.sh
. "$here/common.sh"

dry=0; cc=""; port=""; snapshot=1; positional=(); extra=()
while [ $# -gt 0 ]; do
  case $1 in
    --dry-run) dry=1; shift ;;
    --cc) cc=${2:-}; shift 2 ;;
    --port) port=${2:-}; shift 2 ;;
    --no-snapshot) snapshot=0; shift ;;
    --) shift; extra=("$@"); break ;;
    -*) die "unknown option $1" ;;
    *) positional+=("$1"); shift ;;
  esac
done
arm=${positional[0]:-}
[ -n "$arm" ] || die "usage: serve_olala.sh [--dry-run] ARM [DIR] [-- EXTRA...]; arms: $(python3 -I "$here/arms.py" list 2>&1 | awk '{print $1}' | tr '\n' ' ')"
dir=${positional[1]:-}
args=("$arm")
[ -z "$dir" ] || args+=(--dir "$dir")
[ -z "$port" ] || args+=(--port "$port")
shell_cmd=$(python3 -I "$here/arms.py" shell "${args[@]}" -- ${extra[@]+"${extra[@]}"}) \
  || die "$(python3 -I "$here/arms.py" shell "${args[@]}" 2>&1 >/dev/null | sed 's/^error: //'); arms: $(python3 -I "$here/arms.py" list | awk '{print $1}' | tr '\n' ' ')"
logname=$(python3 -I "$here/arms.py" logname "${args[@]}")
need=$(python3 -I "$here/arms.py" requires "$arm")
jobs=$(max_jobs)
stamp=$(utc_stamp)
log=$LOG_ROOT/$logname/server-$stamp.log
pidfile=$RUN_DIR/server.pid
port=${port:-8000}

# Arms that need a given GPU generation are refused before the 8 minutes of a start that cannot work.
if [ -n "$need" ]; then
  if [ -z "$cc" ] && [ $dry -eq 0 ]; then
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | head -n 1)
  fi
  if [ -n "$cc" ] && [ "${cc%%.*}" != "$need" ]; then
    die "arm $arm needs compute capability ${need}.x (SM${need}0, a B200); this GPU is $cc. vllm/model_executor/layers/quantization/online/nvfp4.py raises after the model is loaded."
  fi
fi

if [ $dry -eq 1 ]; then
  echo "arm=$arm"
  echo "logname=$logname"
  echo "log=$log"
  echo "pidfile=$pidfile"
  echo "port=$port"
  echo "MAX_JOBS=$jobs"
  [ -z "$need" ] || echo "requires_cc_major=$need (cc given: ${cc:-not asked})"
  echo "command: $shell_cmd"
  exit 0
fi

mkdir -p "$LOG_ROOT/$logname" "$RUN_DIR"
if [ -f "$pidfile" ] && alive "$(cat "$pidfile")"; then
  die "a server is already running (pid $(cat "$pidfile"), arm $(cat "$RUN_DIR/server.arm" 2> /dev/null)): stop_server.sh first"
fi
[ -z "$dir" ] || [ -f "$dir/config.json" ] || die "$dir has no config.json: not a checkpoint directory"
[ "$arm" = ckpt ] || [ -f "$MODEL_ROOT/olala-bf16/config.json" ] || die "the BF16 weights are not at $MODEL_ROOT/olala-bf16: run pod_setup.sh first"

mapfile_args() { ARGV=(); while IFS= read -r line; do ARGV+=("$line"); done < <(python3 -I "$here/arms.py" argv "${args[@]}" -- ${extra[@]+"${extra[@]}"}); }
mapfile_args
{
  echo "utc=$(utc_iso)"; echo "arm=$arm"; echo "MAX_JOBS=$jobs"; echo "container_cpus=$(container_cpus)"
  echo "container_mem_gb=$(container_mem_gb)"; echo "command: $shell_cmd"
  env | grep -E '^(VLLM_|OLALA_|MAX_JOBS|HF_HUB_|TORCH_|CUDA_)' | grep -viE 'KEY|TOKEN|SECRET' | sort
} > "${log%.log}.cmd"

cd "$OLALA_WORK" || die "no $OLALA_WORK"
if command -v setsid > /dev/null 2>&1; then
  MAX_JOBS=$jobs setsid nohup "${ARGV[@]}" > "$log" 2>&1 < /dev/null &
else
  MAX_JOBS=$jobs nohup "${ARGV[@]}" > "$log" 2>&1 < /dev/null &
fi
pid=$!
echo "$pid" > "$pidfile"; echo "$arm" > "$RUN_DIR/server.arm"; echo "$log" > "$RUN_DIR/server.log"
pgid=$(ps -o pgid= -p "$pid" 2> /dev/null | tr -d ' ')
[ "$pgid" = "$pid" ] || echo "warning: the server (pid $pid) is not its own process group leader (pgid ${pgid:-?}); stop_server.sh will kill the pid only" >&2
if [ $snapshot -eq 1 ]; then
  if command -v setsid > /dev/null 2>&1; then
    setsid nohup bash "$here/gpu_snapshot.sh" "$logname" > /dev/null 2>&1 < /dev/null &
  else
    nohup bash "$here/gpu_snapshot.sh" "$logname" > /dev/null 2>&1 < /dev/null &
  fi
  echo $! > "$RUN_DIR/gpu_snapshot.pid"
fi
echo "started $arm: pid $pid, port $port"
echo "log: $log"
echo "next: bash $here/smoke.sh $logname"
