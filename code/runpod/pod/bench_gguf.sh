#!/usr/bin/env bash
# Runs ON the pod: the short speed sweep of arm Q4 (amendment 24), `vllm bench serve` against llama-server, for each variant in turn.
# Usage: bench_gguf.sh [--dry-run] LABEL VARIANT[,VARIANT...]        e.g. bench_gguf.sh Q4a q4_k_m-ovhrecipe,q5_k_m
#   VARIANT  q4_k_m-ovhrecipe (made by q4_prepare.sh on this pod) or a variant of gguf_files.json such as q5_k_m
# Per variant: the file is made present (gguf_setup.sh --variant V --skip-build downloads the published Q5_K_M), any server is
# stopped, serve_gguf.sh --variant V starts the server with its default 12 slots, smoke_gguf.sh gates it (exit 0 or 2), the GPU
# must be free of other jobs, then the points: random prompts of 512 tokens in and 512 out, --ignore-eos, concurrency 1, 4 and 12
# with N = max(8, 3 x concurrency) prompts, two repeats (env BENCH_REPEATS), every result a JSON under
# /workspace/results/bench-LABEL-VARIANT/ (c<C>_r<REPEAT>.json, with the logs, the provenance of the server and summary.json).
# The tokenizer is the one of OVHaiLLM/Olala-v1.0-mini-7A1B at the revision pinned in setup_remote.sh, only its tokenizer and
# config files (no weights) in /workspace/models/olala-tokenizer; a missing file refuses the sweep.
# A point with failed requests is reported (BENCH_GGUF_POINT_FAILED:...), its data kept, and the exit status is 3; a variant that
# cannot be served or fails its gate is reported (BENCH_GGUF_FAILED:VARIANT:STAGE) and the next variant still runs (exit 1).
# Markers on stdout: BENCH_GGUF_DONE:<variant> after each variant, BENCH_GGUF_ALL_DONE after the last one. The LAST file written is
# /workspace/results/bench-LABEL.done (first line BENCH_GGUF_ALL_DONE or BENCH_GGUF_FAILED:...; a line points_with_failed_requests=N).
# The client uses the plain completions endpoint (--backend openai, /v1/completions), not the chat endpoint: llama-server's chat stream
# puts a reasoning model's first tokens in reasoning_content, and a 512-token random request can end before any content chunk, which
# vllm bench serve then marks as failed ("Never received a valid chunk to calculate TTFT"): the first sweep of 2026-11-21 had 10 of 12
# points with failed requests for that reason, and the tokens of failed requests are not counted in the throughput.
# This speed is not comparable in absolute terms with the vLLM sweeps (another engine, another slot count): amendment 24.
# Env: OLALA_WORK, BENCH_REPEATS (default 2), BENCH_STOP_WAIT (seconds after stop_server.sh, default 3).
set -u
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=common.sh
. "$here/common.sh"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN GITHUB_TOKEN   # no token of any kind lives on a pod: everything fetched is public
tool=(python3 -I "$here/q4_tool.py")
gtool=(python3 -I "$here/gguf_tool.py")
q4_variant=q4_k_m-ovhrecipe
usage="usage: bench_gguf.sh [--dry-run] LABEL VARIANT[,VARIANT...]   (env BENCH_REPEATS)"
refuse() { echo "BENCH_GGUF_FAILED:BAD_ARGUMENTS:$*"; echo "$usage" >&2; exit 2; }

dry=0; positional=()
for a in "$@"; do
  case $a in
    --dry-run) dry=1 ;;
    -*) refuse "unknown option $a" ;;
    *) positional+=("$a") ;;
  esac
done
[ ${#positional[@]} -eq 2 ] || refuse "expected LABEL and VARIANTS, got ${#positional[@]} argument(s)"
label=${positional[0]}; variant_list=${positional[1]}
[[ $label =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || refuse "bad LABEL '$label' (letters, digits, . _ -; it names directories and a file)"
repeats=${BENCH_REPEATS-2}
[[ $repeats =~ ^[1-9][0-9]*$ ]] || refuse "BENCH_REPEATS must be a positive integer, got '$repeats'"
[ -n "$variant_list" ] || refuse "no variant given"
case $variant_list in ,* | *, | *,,*) refuse "empty variant in '$variant_list'" ;; esac
IFS=, read -r -a variants <<< "$variant_list"
seen=" "
for v in "${variants[@]}"; do
  [[ $v =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || refuse "bad variant '$v' in '$variant_list'"
  case $seen in *" $v "*) refuse "variant $v is listed twice" ;; esac
  seen="$seen$v "
  if [ "$v" != "$q4_variant" ] && ! "${gtool[@]}" get "gguf.files.$v.name" > /dev/null 2>&1; then
    refuse "unknown variant '$v': $("${gtool[@]}" get gguf.files 2> /dev/null | python3 -I -c 'import json,sys; print(" ".join(json.load(sys.stdin)))') or $q4_variant"
  fi
done

results=$OLALA_WORK/results
logs=$LOG_ROOT
tokdir=$MODEL_ROOT/olala-tokenizer
done_file=$results/bench-$label.done
concurrencies=(1 4 12)
build_point() {  # build_point CONC N DIR FILE: sets point_cmd, the command of one point
  point_cmd=(vllm bench serve --backend openai --base-url http://127.0.0.1:8000 --endpoint /v1/completions
    --model olala --tokenizer "$tokdir" --trust-remote-code --dataset-name random --random-input-len 512
    --random-output-len 512 --num-prompts "$2" --max-concurrency "$1" --ignore-eos --save-result
    --result-dir "$3" --result-filename "$4")
}
prompts_for() { local n=$(( 3 * $1 )); [ $n -ge 8 ] || n=8; echo $n; }

if [ $dry -eq 1 ]; then
  pin_model_repo=$("${tool[@]}" pins | sed -n 's/^model_repo=//p')
  pin_model_revision=$("${tool[@]}" pins | sed -n 's/^model_revision=//p')
  echo "bench_gguf.sh dry run: $label, variants ${variants[*]} (nothing is run, nothing is written)"
  echo "== tokenizer: only its tokenizer and config files, no weights, no token"
  echo "+ ${tool[*]} tokenizer --dest $tokdir   ($pin_model_repo @ $pin_model_revision)"
  for v in "${variants[@]}"; do
    echo "== variant $v -> $results/bench-$label-$v/"
    echo "+ bash $here/gguf_setup.sh --variant $v --skip-build"
    echo "+ bash $here/stop_server.sh; bash $here/serve_gguf.sh --variant $v   (12 slots by default)"
    echo "+ bash $here/smoke_gguf.sh --timeout 900   (exit 0 or 2)"
    for rep in $(seq 1 "$repeats"); do
      for c in "${concurrencies[@]}"; do
        build_point "$c" "$(prompts_for "$c")" "$results/bench-$label-$v" "c${c}_r${rep}.json"
        echo "+ timeout 3000 ${point_cmd[*]}"
      done
    done
    echo "+ BENCH_GGUF_DONE:$v"
  done
  echo "== last: $done_file"
  echo "BENCH_GGUF_DRY_RUN (nothing was run)"
  exit 0
fi

for v in "${variants[@]}"; do
  [ ! -e "$results/bench-$label-$v" ] || refuse "$results/bench-$label-$v exists (a sweep never overwrites; use a new LABEL)"
done
mkdir -p "$results" "$logs"
rm -f "$done_file"   # a done file of an earlier run must not tell the collector this sweep has finished
failures=()
points_failed=0
log=$logs/bench-gguf-$label.log

stop_srv() { bash "$here/stop_server.sh" > /dev/null 2>&1 || echo "warning: stop_server.sh exit $?"; sleep "${BENCH_STOP_WAIT:-3}"; }

write_done() {  # write_done FIRST_LINE: the last file of the sweep
  {
    echo "$1"
    echo "label=$label"; echo "variants=${variants[*]}"
    echo "points_with_failed_requests=$points_failed"
    echo "finished_utc=$(utc_iso)"
  } > "$done_file.tmp"
  mv "$done_file.tmp" "$done_file"
}

vfail() {  # vfail VARIANT STAGE REASON: this variant is lost, the sweep goes on
  echo "BENCH_GGUF_FAILED:$1:$2:$3"
  failures+=("$1:$2")
  stop_srv
  return 1
}

run_variant() {
  local v=$1 outdir=$results/bench-$label-$1 smokeout rc busy c n rep file point_rc line
  smokeout=$logs/bench-$label-$v.smoke.txt
  echo "== variant $v"
  if ! "${gtool[@]}" get "gguf.files.$v.name" > /dev/null 2>&1; then
    # the edit of q4_prepare.sh is lost when pod_setup.sh --bundle-only replaced the bundle: put it back from its record
    if [ "$v" = "$q4_variant" ] && [ -f "$OLALA_WORK/q4/variant.json" ]; then
      "${tool[@]}" manifest-restore --manifest "$here/gguf_files.json" --record "$OLALA_WORK/q4/variant.json" > /dev/null \
        || { vfail "$v" MANIFEST "cannot restore the variant from $OLALA_WORK/q4/variant.json"; return 1; }
      echo "the manifest lacked $v: restored from $OLALA_WORK/q4/variant.json"
    else
      vfail "$v" MANIFEST "variant $v is not in $here/gguf_files.json and has no record to restore it from"; return 1
    fi
  fi
  bash "$here/gguf_setup.sh" --variant "$v" --skip-build > "$logs/bench-$label-$v.setup.log" 2>&1 \
    || { tail -n 3 "$logs/bench-$label-$v.setup.log"; vfail "$v" SETUP "gguf_setup.sh --variant $v --skip-build failed"; return 1; }
  stop_srv
  bash "$here/serve_gguf.sh" --variant "$v" > "$logs/bench-$label-$v.serve.txt" 2>&1 \
    || { tail -n 4 "$logs/bench-$label-$v.serve.txt"; vfail "$v" SERVE "serve_gguf.sh --variant $v failed"; return 1; }
  bash "$here/smoke_gguf.sh" --timeout 900 > "$smokeout" 2>&1
  rc=$?
  echo "smoke exit $rc: $(head -n 2 "$smokeout" | tr '\n' ' ')"
  { [ $rc -eq 0 ] || [ $rc -eq 2 ]; } || { tail -n 8 "$smokeout"; vfail "$v" SMOKE "smoke gate exit $rc"; return 1; }
  if command -v pgrep > /dev/null 2>&1; then   # nothing else may use the GPU while the sweep measures it
    busy=$(pgrep -af 'run_tasks\.py|run_accuracy\.sh|arm_pipeline\.sh|vllm bench|bench_sweep\.sh|fidelity\.py')
    [ -z "$busy" ] || { vfail "$v" GPU_NOT_EXCLUSIVE "other jobs running: $(echo "$busy" | head -n 3 | tr '\n' ';' | cut -c1-300)"; return 1; }
  fi
  mkdir -p "$outdir"
  { nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader 2> /dev/null; utc_iso; } > "$outdir/machine.txt"
  cp "$smokeout" "$outdir/smoke.txt"
  for rep in $(seq 1 "$repeats"); do
    for c in "${concurrencies[@]}"; do
      n=$(prompts_for "$c")
      file=c${c}_r${rep}.json
      echo "point $v c=$c n=$n r=$rep"
      build_point "$c" "$n" "$outdir" "$file"
      timeout 3000 "${point_cmd[@]}" > "$outdir/c${c}_r${rep}.log" 2>&1
      rc=$?
      if [ ! -f "$outdir/$file" ]; then
        echo "BENCH_GGUF_POINT_FAILED:$v:c=$c:r=$rep:no result (exit $rc, see $outdir/c${c}_r${rep}.log)"
        points_failed=$((points_failed + 1))
        continue
      fi
      line=$("${tool[@]}" bench-point "$outdir/$file" --expect-prompts "$n")
      point_rc=$?
      echo "  $line"
      if [ $point_rc -ne 0 ]; then
        echo "BENCH_GGUF_POINT_FAILED:$v:c=$c:r=$rep:$line"
        points_failed=$((points_failed + 1))
      fi
    done
  done
  bash "$here/gguf_provenance.sh" > "$outdir/provenance.txt" 2>&1 || echo "warning: no server provenance for $v" >> "$outdir/provenance.txt"
  "${tool[@]}" bench-summary --dir "$outdir" --variant "$v" > /dev/null || echo "warning: no summary for $v"
  stop_srv
  echo "BENCH_GGUF_DONE:$v"
}

main() {
  echo "== bench_gguf $(utc_iso): $label, variants ${variants[*]}, $repeats repeat(s)"
  tok=$("${tool[@]}" tokenizer --dest "$tokdir" 2>&1) || {
    echo "BENCH_GGUF_FAILED:TOKENIZER:$tok"
    write_done "BENCH_GGUF_FAILED:TOKENIZER:$(printf '%s' "$tok" | tr '\n' ' ' | cut -c1-200)"
    exit 1
  }
  echo "tokenizer folder: $tok"
  for v in "${variants[@]}"; do
    run_variant "$v" || true
  done
  if [ ${#failures[@]} -gt 0 ]; then
    first=$(IFS=';'; echo "${failures[*]}")
    write_done "BENCH_GGUF_FAILED:$first"
    exit 1
  fi
  echo BENCH_GGUF_ALL_DONE
  write_done BENCH_GGUF_ALL_DONE
  [ "$points_failed" -eq 0 ] || { echo "$points_failed point(s) with failed requests: see the BENCH_GGUF_POINT_FAILED lines above"; exit 3; }
}

main 2>&1 | tee -a "$log"
exit "${PIPESTATUS[0]}"
