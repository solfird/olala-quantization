#!/usr/bin/env bash
# Runs ON the pod, detached by launch_q4.sh: arm Q4 of amendment 24 from the bare pod to the results.
# Usage: q4_driver.sh [--dry-run] RUN TASKS (--imatrix make | --imatrix FILE --imatrix-sha SHA256) [--expect-sha SHA256] [--bench]
#   RUN        the run name of the accuracy pipeline (and the label of the speed sweep with --bench)
#   TASKS      one string, the tasks of this pod (e.g. 'mgsm aime2025'), passed to run_accuracy.sh as one --tasks argument
#   --expect-sha  the sha256 of the quantized file made on the first pod: the three pods' files must be IDENTICAL
# Order: 1 q4_prepare.sh (imatrix, quantization, the variant in this pod's manifest); 2 the quantized file's sha256 is compared with
# --expect-sha and a difference fails the pod with Q4_SHA_MISMATCH BEFORE anything is served (the comparison is recorded in
# /workspace/results/q4/sha_check.json either way); 3 arm_pipeline.sh RUN gguf:q4_k_m-ovhrecipe --tasks 'TASKS'; 4 with --bench,
# bench_gguf.sh RUN q4_k_m-ovhrecipe,q5_k_m (only when the pipeline succeeded).
# Markers on stdout: Q4_SHA_MISMATCH: expected X got Y, Q4_DRIVER_FAILED:<stage>:<reason>, Q4_DRIVER_DONE. The LAST file written is
# /workspace/results/q4-driver.done, whose first line is the marker. Exit 0 done, 1 failed, 2 bad arguments.
# Env: OLALA_WORK (default /workspace).
set -u
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=common.sh
. "$here/common.sh"
tool=(python3 -I "$here/q4_tool.py")
variant=q4_k_m-ovhrecipe
q4_file=$MODEL_ROOT/olala-gguf/Olala-v1.0-mini-7A1B-q4_k_m-ovhrecipe.gguf
results=$OLALA_WORK/results
done_file=$results/q4-driver.done
usage="usage: q4_driver.sh [--dry-run] RUN TASKS (--imatrix make | --imatrix FILE --imatrix-sha SHA256) [--expect-sha SHA256] [--bench]"
refuse() { echo "Q4_DRIVER_FAILED:BAD_ARGUMENTS:$*"; echo "$usage" >&2; exit 2; }

dry=0; bench=0; imatrix_arg=""; imatrix_sha=""; expect=""; positional=()
while [ $# -gt 0 ]; do
  case $1 in
    --dry-run) dry=1; shift ;;
    --bench) bench=1; shift ;;
    --imatrix) [ $# -ge 2 ] || refuse "--imatrix needs a value"; imatrix_arg=$2; shift 2 ;;
    --imatrix-sha) [ $# -ge 2 ] || refuse "--imatrix-sha needs a value"; imatrix_sha=$2; shift 2 ;;
    --expect-sha) [ $# -ge 2 ] || refuse "--expect-sha needs a value"; expect=$(printf '%s' "$2" | tr 'A-F' 'a-f'); shift 2 ;;
    -*) refuse "unknown option $1" ;;
    *) positional+=("$1"); shift ;;
  esac
done
[ ${#positional[@]} -eq 2 ] || refuse "expected RUN and TASKS, got ${#positional[@]} argument(s)"
run=${positional[0]}; tasks=${positional[1]}
[[ $run =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || refuse "bad RUN '$run' (letters, digits, . _ -)"
[ -n "$tasks" ] || refuse "TASKS is empty"
[ -n "$imatrix_arg" ] || refuse "--imatrix make or --imatrix FILE is required"
if [ "$imatrix_arg" = make ]; then
  [ -z "$imatrix_sha" ] || refuse "--imatrix-sha goes with --imatrix FILE, not with make"
  prep_args=(--imatrix make)
else
  [[ $imatrix_sha =~ ^[0-9a-fA-F]{64}$ ]] || refuse "--imatrix FILE needs --imatrix-sha with 64 hexadecimal digits"
  prep_args=(--imatrix "$imatrix_arg" --imatrix-sha "$imatrix_sha")
fi
[ -z "$expect" ] || [[ $expect =~ ^[0-9a-f]{64}$ ]] || refuse "--expect-sha must be 64 hexadecimal digits"

if [ $dry -eq 1 ]; then
  echo "q4_driver.sh dry run: $run, tasks '$tasks' (nothing is run, nothing is written)"
  echo "== 1 prepare"
  echo "+ bash $here/q4_prepare.sh ${prep_args[*]}   (its q4.done must start with Q4_PREPARE_DONE)"
  echo "== 2 compare the sha256 of $q4_file with ${expect:-the file of the first pod (not given: recorded only)}"
  echo "+ a difference fails the pod with Q4_SHA_MISMATCH before anything is served; $results/q4/sha_check.json"
  echo "== 3 accuracy"
  echo "+ bash $here/arm_pipeline.sh $run gguf:$variant --tasks '$tasks'"
  if [ $bench -eq 1 ]; then
    echo "== 4 speed sweep (only when the accuracy run succeeded)"
    echo "+ bash $here/bench_gguf.sh $run $variant,q5_k_m"
  fi
  echo "== last: $done_file"
  echo "Q4_DRIVER_DRY_RUN (nothing was run)"
  exit 0
fi

mkdir -p "$results/q4"
rm -f "$done_file"   # a done file of an earlier run must not tell the collector this driver has finished
finish() {   # finish FIRST_LINE [EXTRA_LINE...]: the LAST file of the driver
  local first=$1; shift
  {
    echo "$first"; echo "run=$run"
    [ $# -eq 0 ] || printf '%s\n' "$@"
    echo "finished_utc=$(utc_iso)"
  } > "$done_file.tmp"
  mv "$done_file.tmp" "$done_file"
  echo "$first"
}
fail() { finish "Q4_DRIVER_FAILED:$1:$2" | tail -n 1; exit 1; }

echo "== q4_driver $(utc_iso): $run, tasks '$tasks'${expect:+, expecting sha256 $expect}"

echo "== 1 prepare"
bash "$here/q4_prepare.sh" "${prep_args[@]}"
rc=$?
first=$(head -n 1 "$results/q4.done" 2> /dev/null)
{ [ $rc -eq 0 ] && [ "$first" = Q4_PREPARE_DONE ]; } || fail PREPARE "${first:-no q4.done} (q4_prepare.sh exit $rc)"

echo "== 2 compare the sha256 of the quantized file"
got=$("${tool[@]}" sha256 "$q4_file") || fail SHA "cannot hash $q4_file"
if [ -n "$expect" ]; then
  if [ "$got" = "$expect" ]; then identical=true; else identical=false; fi
  printf '{"expected": "%s", "got": "%s", "identical": %s}\n' "$expect" "$got" "$identical" > "$results/q4/sha_check.json"
  if [ "$identical" = false ]; then
    echo "Q4_SHA_MISMATCH: expected $expect got $got"
    finish "Q4_SHA_MISMATCH: expected $expect got $got" > /dev/null
    exit 1
  fi
  echo "the quantized file is identical to the first pod's: $got"
else
  printf '{"expected": null, "got": "%s", "identical": null}\n' "$got" > "$results/q4/sha_check.json"
  echo "no --expect-sha: the file's sha256 $got is recorded for the other pods"
fi

echo "== 3 accuracy: arm_pipeline.sh $run gguf:$variant"
bash "$here/arm_pipeline.sh" "$run" "gguf:$variant" --tasks "$tasks"
rc=$?
[ $rc -eq 0 ] || fail PIPELINE "arm_pipeline.sh exit $rc"

bench_rc=""
if [ $bench -eq 1 ]; then
  echo "== 4 speed sweep: bench_gguf.sh $run $variant,q5_k_m"
  bash "$here/bench_gguf.sh" "$run" "$variant,q5_k_m"
  bench_rc=$?
  { [ "$bench_rc" -eq 0 ] || [ "$bench_rc" -eq 3 ]; } || fail BENCH "bench_gguf.sh exit $bench_rc"
fi
finish Q4_DRIVER_DONE "quantized_sha256=$got" ${bench_rc:+"bench_exit=$bench_rc"}
