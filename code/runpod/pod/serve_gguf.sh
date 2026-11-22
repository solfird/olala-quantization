#!/usr/bin/env bash
# Start ONE llama-server (OVH's llama.cpp fork) on a published Olala GGUF file, for the same-hardware baseline.
# Usage: serve_gguf.sh [--dry-run] [--np N] [--ctx C] [--variant V] [--port N] [--no-snapshot]
#                      [--allow-small-slots] [--skip-memory-check] [--gpu-mem-mib M] [--llama-default-samplers]
#                      [-- EXTRA_LLAMA_SERVER_FLAGS...]
#   --np N          parallel slots (default 12); --ctx C total context -c (default N x 32768: every slot gets the
#                   model's 32K, the card's maximum, so the harness's longest request, 32,688 tokens, fits)
#   --variant V     q5_k_m (default), q6_k, q8_0-safe2 or bf16 (the same-engine control): see gguf_files.json
# One server per pod, like serve_olala.sh: port 8000 on 127.0.0.1, pid file /workspace/run/server.pid (stop_server.sh
# reads it), log /workspace/logs/gguf/server-<UTC>.log, the command and the provenance in server-<UTC>.cmd, a GPU
# snapshot log next to them. The server runs in its own process group. It returns at once: smoke_gguf.sh waits for
# /health. Every flag has a reason (recon/llamacpp_olala_notes.md, section 2); the ones that matter:
#   -np N -c N*32768   non-unified KV: each slot owns 32,768 tokens, so no request can find the pool full (in a
#                      unified pool a full cache fails EVERY running request: tools/server/server-context.cpp:3864-3897)
#   -fa on -fit off    flash attention as the card says; no silent shrinking of -c or -ngl to fit memory
#   --jinja            the GGUF's chat template (default on); the template reads `reasoning_effort`, which the harness
#                      sends in chat_template_kwargs on every request (server-common.cpp:1356-1360). No --reasoning-*
#                      flag: reasoning_format defaults to DEEPSEEK, so the analysis channel goes to
#                      message.reasoning_content and message.content is the final channel only (common/chat.cpp:233).
#   --min-p 0 --samplers 'temperature;top_k;top_p'  vLLM's semantics (temperature, then top-k, then top-p, no
#                      min-p); llama.cpp's own chain ends with the temperature and adds min-p 0.05
#   --ctx-checkpoints 2 --cache-ram 0  bounded host memory: a checkpoint is one 62 MB recurrent state, default 32 per
#                      slot (and an 8 GiB prompt cache); the harness never reuses a prefix
#   --timeout 14400    server read/write timeout as long as the harness's client timeout
#   --threads-http 128 one HTTP thread per harness connection (96 in flight): the default pool is max(np + 4, cores - 1),
#                      so the surplus would wait in the kernel's accept queue instead of the server's task queue
# Env: OLALA_WORK replaces /workspace (the tests use it).
set -u
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=common.sh
. "$here/common.sh"
tool=(python3 -I "$here/gguf_tool.py")

dry=0; np=12; ctx=""; variant=""; port=8000; snapshot=1; small=0; skip_mem=0; gpu_mem=""; llama_samplers=0; extra=()
while [ $# -gt 0 ]; do
  case $1 in
    --dry-run) dry=1; shift ;;
    --np) np=${2:-}; shift 2 ;;
    --ctx) ctx=${2:-}; shift 2 ;;
    --variant) variant=${2:-}; shift 2 ;;
    --port) port=${2:-}; shift 2 ;;
    --no-snapshot) snapshot=0; shift ;;
    --allow-small-slots) small=1; shift ;;
    --skip-memory-check) skip_mem=1; shift ;;
    --gpu-mem-mib) gpu_mem=${2:-}; shift 2 ;;
    --llama-default-samplers) llama_samplers=1; shift ;;
    --) shift; extra=("$@"); break ;;
    *) die "unknown option $1 (usage: serve_gguf.sh [--dry-run] [--np N] [--ctx C] [--variant V] [-- EXTRA])" ;;
  esac
done
[[ $np =~ ^[1-9][0-9]*$ ]] || die "--np must be a positive integer, got '$np'"
[[ $port =~ ^[1-9][0-9]*$ ]] || die "--port must be a positive integer, got '$port'"
[ -z "$ctx" ] || [[ $ctx =~ ^[1-9][0-9]*$ ]] || die "--ctx must be a positive integer, got '$ctx'"
variant=${variant:-$("${tool[@]}" get gguf.default)} || die "cannot read the manifest"

name=$("${tool[@]}" get "gguf.files.$variant.name" 2> /dev/null) \
  || die "unknown variant '$variant': $("${tool[@]}" get gguf.files 2> /dev/null | python3 -I -c 'import json,sys; print(" ".join(json.load(sys.stdin)))')"
pinned_size=$("${tool[@]}" get "gguf.files.$variant.size")
pinned_sha=$("${tool[@]}" get "gguf.files.$variant.sha256")
hf_revision=$("${tool[@]}" get gguf.revision)
harness_max=$("${tool[@]}" get model.harness_max_tokens)
[ -n "$ctx" ] || ctx=$(( np * $("${tool[@]}" get model.slot_ctx) ))

plan_args=(plan --variant "$variant" --np "$np" --ctx "$ctx")
plan=$("${tool[@]}" "${plan_args[@]}") || die "memory plan failed"
plan_get() { printf '%s' "$plan" | python3 -I -c 'import json,sys; v=json.load(sys.stdin)[sys.argv[1]]; print(round(v, 1) if isinstance(v, float) else v)' "$1"; }
slot_ctx=$(plan_get slot_ctx)
if [ "$slot_ctx" -lt "$harness_max" ] && [ $small -eq 0 ]; then
  die "each slot would hold $slot_ctx tokens (-c $ctx / -np $np), below the harness's longest request of $harness_max tokens: a long answer would be cut. Raise --ctx or lower --np (--allow-small-slots for a smoke test)"
fi

llama_dir=$OLALA_WORK/llama.cpp-olala
server=$llama_dir/build/bin/llama-server
model_dir=$MODEL_ROOT/olala-gguf
model=$model_dir/$name
glog=$LOG_ROOT/gguf
logname=gguf
stamp=$(utc_stamp)
log=$glog/server-$stamp.log
cmdfile=${log%.log}.cmd
pidfile=$RUN_DIR/server.pid

argv=("$server" -m "$model" --alias olala -ngl 99 -fa on -np "$np" -c "$ctx" -fit off --jinja)
[ $llama_samplers -eq 1 ] || argv+=(--min-p 0 --samplers 'temperature;top_k;top_p')
argv+=(--ctx-checkpoints 2 --cache-ram 0 --timeout 14400 --threads-http 128 --no-webui --host 127.0.0.1 --port "$port")
[ ${#extra[@]} -eq 0 ] || argv+=("${extra[@]}")
shell_cmd=$(printf '%q ' "${argv[@]}"); shell_cmd=${shell_cmd% }

if [ $skip_mem -eq 0 ]; then
  if [ -z "$gpu_mem" ] && [ $dry -eq 0 ]; then
    gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2> /dev/null | head -n 1 | tr -d ' ')
  fi
  if [[ ${gpu_mem:-} =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    checked=$("${tool[@]}" plan --variant "$variant" --np "$np" --ctx "$ctx" --gpu-mem-mib "$gpu_mem")
    fits=$(printf '%s' "$checked" | python3 -I -c 'import json,sys; print(json.load(sys.stdin)["fits_certain"])')
    [ "$fits" = True ] || die "memory: weights + KV cache + recurrent state ($(plan_get certain_mib) MiB, without the unmeasured overhead) do not fit $gpu_mem MiB with 1 GiB spare for -np $np -c $ctx; lower --np or --ctx (--skip-memory-check to try anyway)"
  fi
fi

if [ $dry -eq 1 ]; then
  echo "variant=$variant"
  echo "model=$model"
  echo "log=$log"
  echo "cmd=$cmdfile"
  echo "pidfile=$pidfile"
  echo "port=$port"
  echo "np=$np"
  echo "ctx=$ctx"
  echo "slot_ctx=$slot_ctx (harness maximum $harness_max)"
  echo "kv_mib=$(plan_get kv_mib) recurrent_mib=$(plan_get recurrent_mib) weights_mib=$(plan_get weights_mib)"
  echo "estimate_mib=$(plan_get estimate_mib) (includes $(plan_get assumed_overhead_mib) MiB of assumed, unmeasured overhead: read the real numbers in the server log)"
  echo "command: $shell_cmd"
  exit 0
fi

mkdir -p "$glog" "$RUN_DIR"
if [ -f "$pidfile" ] && alive "$(cat "$pidfile")"; then
  die "a server is already running (pid $(cat "$pidfile"), arm $(cat "$RUN_DIR/server.arm" 2> /dev/null)): stop_server.sh first"
fi
[ -x "$server" ] || die "no llama-server at $server: run gguf_setup.sh first"
[ -f "$model" ] || die "the GGUF file $model is missing: run gguf_setup.sh first"
commit_file=$glog/llama_cpp_commit.txt
[ -s "$commit_file" ] || die "no record of the llama.cpp commit at $commit_file: gguf_setup.sh writes it when it builds"
commit=$(head -n 1 "$commit_file")
vout=$("${tool[@]}" verify --variant "$variant" --file "$model" --size-only) || die "$vout (the file does not match the pinned size: run gguf_setup.sh again)"
verified_sha=unverified
if [ -f "$glog/gguf_verified.txt" ]; then
  verified_sha=$(awk -v f="$name" '$1 == "file=" f { for (i = 2; i <= NF; i++) if ($i ~ /^sha256=/) { sub(/^sha256=/, "", $i); print $i } }' "$glog/gguf_verified.txt" | tail -n 1)
  verified_sha=${verified_sha:-unverified}
fi

{
  echo "utc=$(utc_iso)"
  echo "arm=gguf-$variant"
  echo "variant=$variant"
  echo "llama_cpp_commit=$commit"
  echo "llama_cpp_repo=$("${tool[@]}" get llama_cpp.repo)"
  echo "gguf_file=$model"
  echo "gguf_size=$pinned_size"
  echo "gguf_sha256_pinned=$pinned_sha"
  echo "gguf_sha256_verified_by_setup=$verified_sha"
  echo "hf_repo=$("${tool[@]}" get gguf.repo)"
  echo "hf_revision=$hf_revision"
  echo "np=$np"; echo "ctx=$ctx"; echo "slot_ctx=$slot_ctx"
  echo "container_cpus=$(container_cpus)"; echo "container_mem_gb=$(container_mem_gb)"
  echo "command: $shell_cmd"
  env | grep -E '^(OLALA_|GGML_|LLAMA_|CUDA_)' | grep -viE 'KEY|TOKEN|SECRET' | sort
} > "$cmdfile"

cd "$OLALA_WORK" || die "no $OLALA_WORK"
if command -v setsid > /dev/null 2>&1; then
  setsid nohup "${argv[@]}" > "$log" 2>&1 < /dev/null &
else
  nohup "${argv[@]}" > "$log" 2>&1 < /dev/null &
fi
pid=$!
echo "$pid" > "$pidfile"; echo "gguf-$variant" > "$RUN_DIR/server.arm"; echo "$log" > "$RUN_DIR/server.log"
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
echo "started gguf-$variant: pid $pid, port $port, -np $np -c $ctx (slot $slot_ctx tokens)"
echo "log: $log"
echo "next: bash $here/smoke_gguf.sh"
