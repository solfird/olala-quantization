#!/usr/bin/env bash
# Smoke test of the llama-server serve_gguf.sh started: /health with a timeout while the log is watched for fatal lines,
# one French chat completion at reasoning effort medium sent the way the harness sends it, a check that the final
# answer in `content` does not carry the reasoning trace, a second request at effort `disabled`, the response shape,
# the generation speed from the server's own timings and the memory lines of the log. Result: /workspace/logs/gguf/smoke.json.
# Usage: smoke_gguf.sh [--timeout SECONDS] [--log FILE] [--parallel N]
#   --parallel N  also send N concurrent bounded requests (max_tokens 256) and print the aggregate tokens per second
# Env: OLALA_BASE_URL (default http://127.0.0.1:8000), SMOKE_POLL_SECONDS (2), SMOKE_HEALTH_TIMEOUT (900: loading a
# 6 GB file onto the GPU takes about a minute; a first CUDA start may also JIT-compile PTX).
# Exit: 0 pass, 2 pass with warnings (read them before any measurement), 1 fail.
set -u
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=common.sh
. "$here/common.sh"
timeout=${SMOKE_HEALTH_TIMEOUT:-900}; log=""; parallel=0
while [ $# -gt 0 ]; do
  case $1 in
    --timeout) timeout=${2:?seconds}; shift 2 ;;
    --log) log=${2:?file}; shift 2 ;;
    --parallel) parallel=${2:?N}; shift 2 ;;
    *) die "unknown option $1 (usage: smoke_gguf.sh [--timeout S] [--log FILE] [--parallel N])" ;;
  esac
done
args=(--url "${OLALA_BASE_URL:-http://127.0.0.1:8000}" --timeout "$timeout" --pidfile "$RUN_DIR/server.pid"
  --out-dir "$LOG_ROOT/gguf" --parallel "$parallel")
[ -z "$log" ] || args+=(--log "$log")
exec python3 -I "$here/smoke_gguf.py" "${args[@]}"
