#!/usr/bin/env bash
# Stop the vLLM server by its pid file: the whole process group (the API server and its EngineCore children),
# then the GPU snapshot loop. Never by text (`pkill -f` matches the SSH shell that runs it: it killed ours twice).
# Usage: stop_server.sh [--timeout SECONDS]      (default 60 before SIGKILL)
# The pid is checked to be a vllm or llama-server process before anything is signalled (STOP_SERVER_EXPECT, when set,
# replaces both with one text: the tests use it).
set -u
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=common.sh
. "$here/common.sh"
timeout=60
[ "${1:-}" != "--timeout" ] || timeout=${2:?seconds}
pidfile=$RUN_DIR/server.pid
expect=${STOP_SERVER_EXPECT:-}

stop_snapshot() {
  local f=$RUN_DIR/gpu_snapshot.pid p cmd
  [ -f "$f" ] || return 0
  p=$(cat "$f"); cmd=$(ps -p "$p" -o command= 2> /dev/null)
  case $cmd in *gpu_snapshot.sh*) kill "$p" 2> /dev/null ;; esac
  rm -f "$f"
}

if [ ! -f "$pidfile" ]; then
  echo "no server (no pid file $pidfile)"
  stop_snapshot
  exit 0
fi
pid=$(cat "$pidfile")
if ! alive "$pid"; then
  echo "stale pid file (process $pid is gone): removed"
  rm -f "$pidfile" "$RUN_DIR/server.arm" "$RUN_DIR/server.log"
  stop_snapshot
  exit 0
fi
cmd=$(ps -p "$pid" -o command= 2> /dev/null)
if [ -n "$expect" ]; then
  case $cmd in
    *"$expect"*) ;;
    *) die "pid $pid is not the server ('$cmd' does not contain '$expect'): nothing was signalled" ;;
  esac
else
  case $cmd in
    *vllm*|*llama-server*) ;;
    *) die "pid $pid is not the server ('$cmd' contains neither 'vllm' nor 'llama-server'): nothing was signalled" ;;
  esac
fi
arm=$(cat "$RUN_DIR/server.arm" 2> /dev/null)
pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
if [ "$pgid" = "$pid" ]; then target="-$pid"; else target=$pid; fi
kill -TERM -- "$target" 2> /dev/null
for _ in $(seq 1 "$timeout"); do alive "$pid" || break; sleep 1; done
if alive "$pid"; then
  echo "still alive after $timeout s: SIGKILL"
  kill -KILL -- "$target" 2> /dev/null
  sleep 2
fi
alive "$pid" && die "pid $pid survived SIGKILL"
rm -f "$pidfile" "$RUN_DIR/server.arm" "$RUN_DIR/server.log"
stop_snapshot
echo "stopped server pid $pid (arm ${arm:-unknown})"
# Anything of the group still around, and anything still holding GPU memory, by exact pid.
if [ "$pgid" = "$pid" ]; then
  left=$(ps -eo pid,pgid,args | awk -v g="$pgid" '$2 == g')
  [ -z "$left" ] || echo "WARNING: processes of the server's group remain (kill each by exact pid):
$left"
fi
if command -v nvidia-smi > /dev/null 2>&1; then
  apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2> /dev/null)
  [ -z "$apps" ] || echo "WARNING: GPU memory still in use: $apps"
fi
