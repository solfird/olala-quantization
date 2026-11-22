#!/usr/bin/env bash
# Log the GPU memory and utilization every 30 seconds while the server runs.
# Usage: gpu_snapshot.sh LOGNAME [--interval SECONDS] [--once]
#   LOGNAME  the arm's log folder name (serve_olala.sh passes it); output: /workspace/logs/LOGNAME/gpu-<UTC>.log
# One CSV line per sample: a UTC timestamp then name, memory.used, memory.total, utilization.gpu,
# utilization.memory, power.draw, temperature.gpu. The loop ends when the pid in run/server.pid is gone.
set -u
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=common.sh
. "$here/common.sh"
name=${1:?log name}; shift
interval=30; once=0
while [ $# -gt 0 ]; do
  case $1 in
    --interval) interval=${2:?seconds}; shift 2 ;;
    --once) once=1; shift ;;
    *) die "unknown option $1" ;;
  esac
done
mkdir -p "$LOG_ROOT/$name"
log=$LOG_ROOT/$name/gpu-$(utc_stamp).log
echo "timestamp_utc, name, memory.used, memory.total, utilization.gpu, utilization.memory, power.draw, temperature.gpu" > "$log"
sample() {
  echo "$(utc_iso), $(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu --format=csv,noheader 2>&1 | head -n 1)" >> "$log"
}
if [ $once -eq 1 ]; then sample; exit 0; fi
while alive "$(cat "$RUN_DIR/server.pid" 2> /dev/null)"; do
  sample
  sleep "$interval"
done
