#!/usr/bin/env bash
# Throughput sweep against the server that is already up on port 8000 (run it on the pod right after an arm's accuracy
# run, while the server is still serving that arm). One JSON per point in /workspace/results/bench-LABEL/, a summary table
# on stdout. The workloads are synthetic (random tokens) with fixed input and output lengths, ignore-eos, so every point
# produces the same number of tokens per request; decode-heavy workloads are what the model's reasoning answers look like.
# Usage: bench_sweep.sh LABEL [--repeats N] [--quick]
#   LABEL     the arm, for example bf16, F1, F2, N2 (names the output directory; refuses to overwrite)
#   --repeats N   repeats per point (default 3; the medians and the spread are reported)
#   --quick       concurrency 1, 16 and 64 on the short workload only, one repeat
# Workloads: short = 512 in / 512 out, long = 2,048 in / 8,192 out. Concurrency: 1, 4, 16, 32, 64, 128 (long: 1, 16, 32).
set -u
label=${1:?LABEL}; shift
repeats=3; quick=0
while [ $# -gt 0 ]; do
  case $1 in
    --repeats) repeats=${2:?N}; shift 2 ;;
    --quick) quick=1; repeats=1; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
out=/workspace/results/bench-$label
[ ! -e "$out" ] || { echo "error: $out exists" >&2; exit 2; }
curl -sf -o /dev/null http://127.0.0.1:8000/health || { echo "error: no healthy server on port 8000" >&2; exit 2; }
mkdir -p "$out"
{ nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader; date -u +%Y-%m-%dT%H:%M:%SZ; } > "$out/machine.txt"

point() {  # NAME IN OUT CONC NPROMPTS REP
  local name=$1 inl=$2 outl=$3 conc=$4 n=$5 rep=$6
  timeout 3000 vllm bench serve --backend openai-chat --base-url http://127.0.0.1:8000 --endpoint /v1/chat/completions \
    --model olala --tokenizer /workspace/models/olala-bf16 --trust-remote-code --dataset-name random \
    --random-input-len "$inl" --random-output-len "$outl" --num-prompts "$n" --max-concurrency "$conc" --ignore-eos \
    --save-result --result-dir "$out" --result-filename "${name}_c${conc}_r${rep}.json" > "$out/${name}_c${conc}_r${rep}.log" 2>&1 \
    || echo "POINT FAILED: $name c=$conc rep=$rep (see $out/${name}_c${conc}_r${rep}.log)"
}

if [ $quick -eq 1 ]; then
  for c in 1 16 64; do point short 512 512 "$c" $((c < 8 ? 8 : c * 3)) 1; done
else
  for rep in $(seq 1 "$repeats"); do
    for c in 1 4 16 32 64 128; do point short 512 512 "$c" $((c < 8 ? 8 : c * 3)) "$rep"; done
    for c in 1 16 32; do point long 2048 8192 "$c" $((c < 4 ? 4 : c * 2)) "$rep"; done
  done
fi

python3 - "$out" "$label" <<'PY'
import glob, json, statistics as st, sys, collections
out, label = sys.argv[1:3]
pts = collections.defaultdict(list)
for f in sorted(glob.glob(f"{out}/*.json")):
    d = json.load(open(f))
    name, conc, _ = f.split("/")[-1][:-5].split("_")
    pts[(name, int(conc[1:]))].append(d)
print(f"bench {label}: output tok/s (median of repeats; min-max), TTFT ms, TPOT ms, requests ok/failed")
for (name, conc), ds in sorted(pts.items()):
    tps = [d["output_throughput"] for d in ds]
    ttft = st.median(d["median_ttft_ms"] for d in ds); tpot = st.median(d["median_tpot_ms"] for d in ds)
    ok = sum(d["completed"] for d in ds); bad = sum(d.get("failed", 0) for d in ds)
    print(f"{name:5} c={conc:<4} {st.median(tps):8.1f} ({min(tps):.0f}-{max(tps):.0f})  ttft {ttft:8.0f}  tpot {tpot:6.1f}  n={len(ds)} ok={ok} failed={bad}")
PY
echo BENCH_DONE
