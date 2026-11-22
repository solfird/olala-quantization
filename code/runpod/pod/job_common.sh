#!/usr/bin/env bash
# Shared helpers of the exploratory fidelity jobs (m1_gate.sh, h2_bisect.sh). Source it; do not run it.
# Assumes the bundle is extracted at /workspace/bundle and the weights at /workspace/models/olala-bf16.
B=/workspace/bundle
EV=$B/evals
QU=$B/quant
OUT=/workspace/results/raw
mkdir -p "$OUT" /workspace/ckpt /workspace/logs
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

# apply_patch: our vLLM loader patch on the installed fork, once (checks whether it is already applied)
apply_patch() {
  local site=/usr/local/lib/python3.12/dist-packages
  if (cd "$site" && patch -p1 --dry-run -R < "$QU/fork-patch/olala-quant-loader.patch" > /dev/null 2>&1); then
    log "loader patch already applied"
  else
    (cd "$site" && patch -p1 < "$QU/fork-patch/olala-quant-loader.patch") || { log "PATCH FAILED"; return 1; }
    python3 -m py_compile "$site/vllm/model_executor/models/olala.py" "$site/vllm/model_executor/models/olala_quant_loader.py" \
      && log "loader patch applied"
  fi
}

# ckptsel_precheck DIR SPEC: the CPU-only checks of the mode ckptsel:DIR:RECIPE:SPEC, run before any GPU work (arm_pipeline.sh,
# bench_arm.sh). 1. `olala_quant.ablation expected` must accept SPEC for the BF16 header (a bad name, a layer outside the model, a
# selection that quantizes nothing are refused here); its output gives the canonical spelling of SPEC and the expected counts.
# 2. A DIR that already holds a checkpoint (index or quantization_info.json) must record exactly that canonical selection: a stale
# checkpoint of another selection must never be served under this label. Sets CKPTSEL_CANON, CKPTSEL_EXPECTED (compact JSON) and,
# on a refusal (return 1), CKPTSEL_ERR.
ckptsel_precheck() {
  local dir=$1 spec=$2 out err parsed rec
  err=$(mktemp)
  out=$(cd "$QU" && PYTHONPATH="$QU" python3 -m olala_quant.ablation expected --src /workspace/models/olala-bf16 \
    --select "$spec" 2> "$err") \
    || { CKPTSEL_ERR="selection '$spec' refused by olala_quant.ablation expected: $(tail -n 1 "$err" | cut -c1-200)"; rm -f "$err"; return 1; }
  rm -f "$err"
  parsed=$(printf '%s' "$out" | python3 -c 'import json, sys
d = json.load(sys.stdin)
print(d["spec"])
print(json.dumps(d, sort_keys=True, separators=(",", ":")))' 2> /dev/null) \
    || { CKPTSEL_ERR="olala_quant.ablation expected printed no spec for '$spec'"; return 1; }
  CKPTSEL_CANON=$(printf '%s\n' "$parsed" | sed -n 1p)
  CKPTSEL_EXPECTED=$(printf '%s\n' "$parsed" | sed -n 2p)
  [ -n "$CKPTSEL_CANON" ] && [ -n "$CKPTSEL_EXPECTED" ] \
    || { CKPTSEL_ERR="olala_quant.ablation expected printed no spec for '$spec'"; return 1; }
  if [ -e "$dir/model.safetensors.index.json" ] || [ -e "$dir/quantization_info.json" ]; then
    rec=$(python3 -c 'import json, sys
s = json.load(open(sys.argv[1])).get("selection")
print(s.get("spec", "") if isinstance(s, dict) else "")' "$dir/quantization_info.json" 2> /dev/null)
    [ "$rec" = "$CKPTSEL_CANON" ] \
      || { CKPTSEL_ERR="stale checkpoint $dir: it records the selection '${rec:-none}', not '$CKPTSEL_CANON' (remove it or use another DIR)"; return 1; }
  fi
}

# ckptsel_provenance SPEC: the provenance lines of a ckptsel run (after a successful ckptsel_precheck)
ckptsel_provenance() {
  echo "selection=$CKPTSEL_CANON"
  echo "selection_requested=$1"
  echo "expected_counts=$CKPTSEL_EXPECTED"
}

# collect NAME: top-20 logprobs of the 64 x 512 wikitext chunks from the server on port 8000
collect() {
  (cd "$EV" && python3 fidelity.py collect --chunks data/wikitext2_test_64x512.json \
    --base-url http://127.0.0.1:8000/v1 --model olala --out "$OUT/$1_wikitext2.npz" 2>&1 | tail -1)
}

# gate LOGNAME NAME: wait for health and smoke, print the verdict lines, collect on pass
gate() {
  local logname=$1 name=$2
  bash "$B/pod/smoke.sh" "$logname" --timeout 900 > "/workspace/logs/$name.smoke.txt" 2>&1
  local rc=$?
  log "smoke $name exit $rc: $(head -n 2 "/workspace/logs/$name.smoke.txt" | tr '\n' ' ')"
  if [ $rc -eq 0 ] || [ $rc -eq 2 ]; then collect "$name"; else log "GATE FAILED for $name"; tail -n 6 "/workspace/logs/$name.smoke.txt"; fi
  return $rc
}

stop() { bash "$B/pod/stop_server.sh" 2>&1 | tail -n 1; sleep 3; }

# compare REF VAR: fidelity of VAR against REF to a JSON and the headline lines
compare() {
  (cd "$EV" && python3 fidelity.py compare "$OUT/$1_wikitext2.npz" "$OUT/$2_wikitext2.npz" \
    --json "/workspace/results/${2}_vs_${1}_fidelity.json" 2>&1 | grep -E 'Perplexity ratio|Mean KL|99th|top-1' | sed "s/^/$2 vs $1: /")
}
