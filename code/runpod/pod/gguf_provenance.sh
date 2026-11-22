#!/usr/bin/env bash
# Print the provenance of the llama-server that serve_gguf.sh started, one key=value per line, for the provenance
# file of a run (arm_pipeline.sh appends it): the llama.cpp commit and tree, the build (CUDA architecture, nvcc,
# flags), the GGUF file (path, size, sha256 as verified by gguf_setup.sh, the pinned Hub revision), the exact server
# command line and the smoke summary (health time, generation speed, the parallel probe when it ran).
# Usage: gguf_provenance.sh          (reads /workspace/run/server.log, the .cmd next to it, the build record, smoke.json)
# Env: OLALA_WORK replaces /workspace. Exit 1 when no gguf server record exists.
set -u
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=common.sh
. "$here/common.sh"
serve_log=$(cat "$RUN_DIR/server.log" 2> /dev/null) || die "no server record in $RUN_DIR: start serve_gguf.sh first"
cmdfile=${serve_log%.log}.cmd
[ -f "$cmdfile" ] || die "no command file $cmdfile next to the server log"
grep -q '^llama_cpp_commit=' "$cmdfile" || die "$cmdfile is not a gguf server record (no llama_cpp_commit line)"
get() { sed -n "s/^$1=//p" "$cmdfile" | head -n 1; }
build_get() { sed -n "s/^$1=//p" "$OLALA_WORK/llama.cpp-olala/build/olala_build_record.txt" 2> /dev/null | head -n 1; }

echo "== gguf provenance ($(utc_iso))"
echo "server_log=$serve_log"
echo "llama_cpp_repo=$(get llama_cpp_repo)"
echo "llama_cpp_commit=$(get llama_cpp_commit)"
echo "llama_cpp_tree=$(build_get tree)"
echo "cuda_arch=$(build_get cuda_arch)"
echo "nvcc=$(build_get nvcc)"
echo "cmake_flags=$(build_get cmake_flags)"
echo "gguf_variant=$(get variant)"
echo "gguf_file=$(get gguf_file)"
echo "gguf_size=$(get gguf_size)"
verified=$(get gguf_sha256_verified_by_setup)
if [ -n "$verified" ] && [ "$verified" != unverified ]; then
  echo "gguf_sha256=$verified"
else
  echo "gguf_sha256=unverified (pinned: $(get gguf_sha256_pinned))"
fi
echo "hf_repo=$(get hf_repo)"
echo "hf_revision=$(get hf_revision)"
echo "np=$(get np) ctx=$(get ctx) slot_ctx=$(get slot_ctx)"
echo "server_command=$(sed -n 's/^command: //p' "$cmdfile" | head -n 1)"
smoke=$(dirname "$serve_log")/smoke.json
if [ -f "$smoke" ]; then
  python3 -I - "$smoke" << 'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
speed = report.get("tokens_per_second") or {}
print(f"smoke_verdict={report.get('verdict')}")
print(f"smoke_health_seconds={report.get('health_seconds')}")
print(f"smoke_generation_tokens_per_second={speed.get('generation')}")
parallel = report.get("parallel")
if parallel:
    print(f"smoke_parallel_requests={parallel.get('requests')}")
    print(f"smoke_parallel_aggregate_tokens_per_second={parallel.get('aggregate_tokens_per_second')}")
for fact in report.get("log_facts", []):
    print(f"smoke_log_fact={fact}")
PY
else
  echo "smoke_verdict=no smoke.json"
fi
