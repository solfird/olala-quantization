#!/usr/bin/env bash
# Smoke test of the server serve_olala.sh started: /health with a timeout while the log is watched for
# errors, one French chat completion with reasoning_effort medium, the log read for slow paths, JIT failures
# and NaN, the 'Quantizing N layers' summary printed; the result goes to /workspace/logs/LOGNAME/smoke.json.
# Usage: smoke.sh LOGNAME [--timeout SECONDS] [--log FILE]
# Env: OLALA_BASE_URL (default http://127.0.0.1:8000), SMOKE_POLL_SECONDS (2), SMOKE_HEALTH_TIMEOUT (1800: the
# first start compiles kernels, about 8 minutes on the model card's hardware).
# Exit: 0 pass, 2 pass with warnings (read them before any measurement), 1 fail.
set -u
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=common.sh
. "$here/common.sh"
name=${1:?usage: smoke.sh LOGNAME [--timeout S] [--log FILE]}; shift
timeout=${SMOKE_HEALTH_TIMEOUT:-1800}; log=""
while [ $# -gt 0 ]; do
  case $1 in
    --timeout) timeout=${2:?seconds}; shift 2 ;;
    --log) log=${2:?file}; shift 2 ;;
    *) die "unknown option $1" ;;
  esac
done
args=(--arm "$name" --url "${OLALA_BASE_URL:-http://127.0.0.1:8000}" --timeout "$timeout"
  --pidfile "$RUN_DIR/server.pid" --out-dir "$LOG_ROOT/$name")
[ -z "$log" ] || args+=(--log "$log")
exec python3 -I "$here/smoke.py" "${args[@]}"
