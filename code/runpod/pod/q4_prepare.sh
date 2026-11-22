#!/usr/bin/env bash
# Runs ON the pod: arm Q4 of amendment 24. OVH's published bf16 GGUF quantized to Q4_K_M with OVH's recipe: an importance
# matrix from wikitext-2 raw train (200 chunks) and two tensor overrides, ffn_latent_up kept in bf16 and ffn_up_exps in q8_0.
# Usage: q4_prepare.sh [--dry-run] --imatrix make
#        q4_prepare.sh [--dry-run] --imatrix FILE --imatrix-sha SHA256
#   --imatrix make   build the importance matrix here: llama-imatrix on the bf16 GGUF and the wikitext-2 text, 200 chunks
#   --imatrix FILE   use an importance matrix made on another pod; its sha256 must equal --imatrix-sha (the same bytes on
#                    every pod: amendment 24 fixes ONE imatrix, shipped by hash)
# The wikitext-2 text: the parquet file wikitext-2-raw-v1/train-00000-of-00001.parquet of the dataset Salesforce/wikitext at
# revision b08601e04326c79dfdd32d625aee71d232d685c3 (public, no token), the `text` rows concatenated in order with no separator
# added (the rows already end with newlines; this is how llama.cpp's wiki.train.raw is made); its sha256 is recorded.
# Steps, one marker file each under /workspace/q4/steps (a finished step is not redone; a step that runs again invalidates the
# steps after it): setup_bf16 (gguf_setup.sh --variant bf16: the pinned fork and the verified bf16 GGUF), build_extra
# (llama-quantize and llama-imatrix in the SAME build directory and configuration as gguf_setup.sh), help (the --help output of both
# tools is saved and the run is REFUSED when --tensor-type, --imatrix or --chunks is not accepted), wikitext, imatrix, quantize,
# manifest (this POD's copy of gguf_files.json gets the variant q4_k_m-ovhrecipe so that serve_gguf.sh, smoke_gguf.sh and
# arm_pipeline.sh accept it; the repository's file on the Mac is never touched).
# Results: /workspace/results/q4/ (prepare.json, logs, tensor_types.json, manifest_edit.json, olala.imatrix) and, LAST,
# /workspace/results/q4.done whose first line is Q4_PREPARE_DONE or Q4_PREPARE_FAILED:<reason>. The same marker is the last line
# of stdout. Logs: /workspace/logs/q4/. Needs the network for the Hub and the GitHub fetch done by gguf_setup.sh.
# Env: OLALA_WORK (default /workspace), CUDA_HOME (default /usr/local/cuda).
set -u
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=common.sh
. "$here/common.sh"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN GITHUB_TOKEN   # no token of any kind lives on a pod: everything fetched is public
tool=(python3 -I "$here/q4_tool.py")
gtool=(python3 -I "$here/gguf_tool.py")
usage="usage: q4_prepare.sh [--dry-run] --imatrix make | --imatrix FILE --imatrix-sha SHA256"

refuse() { echo "Q4_PREPARE_FAILED:BAD_ARGUMENTS:$*"; echo "$usage" >&2; exit 2; }
dry=0; imatrix_arg=""; imatrix_sha=""
while [ $# -gt 0 ]; do
  case $1 in
    --dry-run) dry=1; shift ;;
    --imatrix) [ $# -ge 2 ] || refuse "--imatrix needs a value"; imatrix_arg=$2; shift 2 ;;
    --imatrix-sha) [ $# -ge 2 ] || refuse "--imatrix-sha needs a value"; imatrix_sha=$(printf '%s' "$2" | tr 'A-F' 'a-f'); shift 2 ;;
    *) refuse "unknown argument $1" ;;
  esac
done
[ -n "$imatrix_arg" ] || refuse "--imatrix make or --imatrix FILE is required"
if [ "$imatrix_arg" = make ]; then
  mode="make"
  [ -z "$imatrix_sha" ] || refuse "--imatrix-sha goes with --imatrix FILE, not with make"
else
  mode="file"
  [[ $imatrix_sha =~ ^[0-9a-f]{64}$ ]] || refuse "--imatrix FILE needs --imatrix-sha with 64 hexadecimal digits"
fi

q4=$OLALA_WORK/q4
logs=$LOG_ROOT/q4
results=$OLALA_WORK/results
resq4=$results/q4
done_file=$results/q4.done
model_dir=$MODEL_ROOT/olala-gguf
llama_dir=$OLALA_WORK/llama.cpp-olala
build=$llama_dir/build
qbin=$build/bin/llama-quantize
ibin=$build/bin/llama-imatrix
bf16_name=$("${gtool[@]}" get gguf.files.bf16.name) || die "cannot read the manifest $here/gguf_files.json"
bf16=$model_dir/$bf16_name
variant=q4_k_m-ovhrecipe
out_name=Olala-v1.0-mini-7A1B-q4_k_m-ovhrecipe.gguf
out=$model_dir/$out_name
text=$q4/wiki.train.raw
imatrix=$q4/olala.imatrix
cuda_home=${CUDA_HOME:-/usr/local/cuda}
jobs=$(max_jobs)
threads=$(container_cpus)
stamp=$(utc_stamp)

imatrix_cmd=("$ibin" -m "$bf16" -f "$text" -o "$imatrix" --chunks 200 -ngl 99)
tt_two=(--tensor-type ffn_latent_up=bf16 --tensor-type ffn_up_exps=q8_0)
tt_equals=(--tensor-type=ffn_latent_up=bf16 --tensor-type=ffn_up_exps=q8_0)
build_cmd=(cmake --build "$build" --target llama-quantize llama-imatrix -j "$jobs")
manifest=$here/gguf_files.json
note="produced on this pod from the bf16 GGUF ($bf16_name) with llama-quantize to Q4_K_M, importance matrix from wikitext-2 raw train (200 chunks), --tensor-type ffn_latent_up=bf16 --tensor-type ffn_up_exps=q8_0 (OVH's recipe; amendment 24); not an OVH file"

if [ $dry -eq 1 ]; then
  echo "q4_prepare.sh dry run: nothing is run, fetched or written"
  echo "== 1 bf16 GGUF and the pinned fork (idempotent)"
  echo "+ bash $here/gguf_setup.sh --variant bf16"
  echo "== 2 build the extra targets in the same build directory and configuration"
  echo "+ PATH=$cuda_home/bin:\$PATH ${build_cmd[*]}   (configuration, commit and flags recorded from $build/olala_build_record.txt)"
  echo "== 3 help output of the tools, saved to $logs; the run is refused when a flag of the recipe is not accepted"
  echo "+ $qbin --help; $ibin --help   (--tensor-type and --imatrix must be listed by the first, --chunks by the second)"
  if [ "$mode" = make ]; then
    echo "== 4 wikitext-2 text: Salesforce/wikitext @ b08601e04326c79dfdd32d625aee71d232d685c3, wikitext-2-raw-v1/train-00000-of-00001.parquet (no token)"
    echo "+ HF_HUB_DISABLE_XET=1 ${tool[*]} wikitext --out $text --cache $q4/hub   (the text rows concatenated in order, no separator added)"
    echo "== 5 importance matrix (imatrix) made here"
    echo "+ ${imatrix_cmd[*]}"
  else
    echo "== 4 wikitext-2 text: not needed, an importance matrix is given"
    echo "== 5 importance matrix (imatrix) given: its sha256 must be $imatrix_sha"
    echo "+ sha256 of $imatrix_arg must equal $imatrix_sha; it is used as $imatrix"
  fi
  echo "== 6 quantize to Q4_K_M with OVH's recipe (the --tensor-type syntax is read from the help; the two-argument form is shown)"
  echo "+ $qbin --imatrix $imatrix ${tt_two[*]} $bf16 $out Q4_K_M $threads"
  echo "+ the log is parsed into a table of tensor types; ffn_latent_up must end bf16 and ffn_up_exps q8_0"
  echo "== 7 manifest of this pod: variant $variant added to $manifest (the Mac's copy is never edited)"
  echo "+ ${tool[*]} manifest-add --manifest $manifest --variant $variant --name $out_name --size <measured> --sha256 <measured>"
  echo "== 8 prepare.json and q4.done (last)"
  echo "+ $resq4/prepare.json, then $done_file (first line Q4_PREPARE_DONE or Q4_PREPARE_FAILED:<reason>)"
  echo "Q4_PREPARE_DRY_RUN (nothing was run)"
  exit 0
fi

mkdir -p "$q4/steps" "$logs" "$results"
rm -f "$done_file"   # a done file of an earlier run must not tell the collector this run has finished
log=$logs/prepare-$stamp.log

order=(setup_bf16 build_extra help wikitext imatrix quantize manifest)
marker() { echo "$q4/steps/$1.done"; }
marker_get() { sed -n "s/^$2=//p" "$(marker "$1")" 2> /dev/null | head -n 1; }
have() { [ -f "$(marker "$1")" ]; }
# invalidate_after STEP: the steps after STEP must run again when STEP ran again
invalidate_after() {
  local seen=0 s
  for s in "${order[@]}"; do
    [ $seen -eq 1 ] && rm -f "$(marker "$s")"
    [ "$s" = "$1" ] && seen=1
  done
  return 0
}
sha_of() { "${tool[@]}" sha256 "$1"; }
size_of() { wc -c < "$1" | tr -d ' '; }
json_get() { python3 -I -c 'import json, sys; print(json.loads(sys.stdin.read())[sys.argv[1]])' "$1"; }
has_flag() { grep -Eq -- "(^|[^A-Za-z0-9_-])$1([^A-Za-z0-9_-]|\$)" "$2"; }
skipped() { echo "== $1: skipped (already done)"; }

finish() {   # finish STATUS REASON: copy the records, write prepare.json, then q4.done LAST
  local status=$1 reason=$2 first
  mkdir -p "$resq4/logs"
  cp -R "$logs/." "$resq4/logs/" 2> /dev/null
  for f in tensor_types.json manifest_edit.json wikitext.json; do [ ! -f "$q4/$f" ] || cp "$q4/$f" "$resq4/$f"; done
  [ ! -f "$imatrix" ] || cp "$imatrix" "$resq4/olala.imatrix"
  "${tool[@]}" report --q4dir "$q4" --out "$resq4/prepare.json" --status "$status" --reason "$reason" \
    || echo "warning: prepare.json could not be written"
  if [ "$status" = "done" ]; then
    first=Q4_PREPARE_DONE
    {
      echo "$first"
      echo "quantized_sha256=$(marker_get quantize sha256)"
      echo "quantized_size=$(marker_get quantize size)"
      echo "imatrix_sha256=$(marker_get quantize imatrix_sha256)"
      echo "finished_utc=$(utc_iso)"
    } > "$done_file.tmp"
  else
    first="Q4_PREPARE_FAILED:$reason"
    { echo "$first"; echo "finished_utc=$(utc_iso)"; } > "$done_file.tmp"
  fi
  mv "$done_file.tmp" "$done_file"
  echo "$first"
}
fail() { finish failed "$(printf '%s' "$*" | tr '\n' ' ' | cut -c1-300)"; exit 1; }

main() {
  echo "== q4_prepare $(utc_iso): imatrix $mode, $bf16_name -> $out_name"
  PATH="$cuda_home/bin:$PATH"

  # 1 -------------------------------------------------------------------------------------------------------
  if have setup_bf16 && [ -f "$bf16" ]; then
    skipped "1 bf16 GGUF and the pinned fork"
  else
    echo "== 1 bf16 GGUF and the pinned fork (gguf_setup.sh --variant bf16)"
    invalidate_after setup_bf16
    bash "$here/gguf_setup.sh" --variant bf16 > "$logs/setup-bf16.log" 2>&1
    rc=$?
    tail -n 3 "$logs/setup-bf16.log"
    last=$(tail -n 1 "$logs/setup-bf16.log")
    { [ $rc -eq 0 ] && [ "$last" = GGUF_SETUP_DONE ]; } || fail "SETUP_BF16:$last"
    [ -f "$bf16" ] || fail "SETUP_BF16:no file $bf16 after the setup"
    {
      echo "file=$bf16"; echo "size=$(size_of "$bf16")"
      echo "sha256=$("${gtool[@]}" get gguf.files.bf16.sha256)"   # verified against this pin by gguf_setup.sh
      echo "utc=$(utc_iso)"
    } > "$(marker setup_bf16)"
  fi

  # 2 -------------------------------------------------------------------------------------------------------
  record=$build/olala_build_record.txt
  [ -f "$record" ] || fail "BUILD_RECORD_MISSING:$record (gguf_setup.sh leaves it)"
  rec_get() { sed -n "s/^$1=//p" "$record" | head -n 1; }
  if have build_extra && [ -x "$qbin" ] && [ -x "$ibin" ] && [ "$(marker_get build_extra commit)" = "$(rec_get commit)" ]; then
    skipped "2 extra targets"
  else
    echo "== 2 build the extra targets in $build with the configuration of gguf_setup.sh (jobs $jobs)"
    invalidate_after build_extra
    echo "+ ${build_cmd[*]}"
    "${build_cmd[@]}" > "$logs/build-extra.log" 2>&1 || { tail -n 5 "$logs/build-extra.log"; fail "BUILD_EXTRA:cmake --build failed (see build-extra.log)"; }
    { [ -x "$qbin" ] && [ -x "$ibin" ]; } || fail "BUILD_EXTRA:no llama-quantize or llama-imatrix under $build/bin after the build"
    {
      echo "commit=$(rec_get commit)"; echo "tree=$(rec_get tree)"; echo "cmake_flags=$(rec_get cmake_flags)"
      echo "cuda_arch=$(rec_get cuda_arch)"; echo "build_command=${build_cmd[*]}"; echo "jobs=$jobs"; echo "utc=$(utc_iso)"
    } > "$(marker build_extra)"
  fi

  # 3 -------------------------------------------------------------------------------------------------------
  qhelp=$logs/llama-quantize.help.txt
  ihelp=$logs/llama-imatrix.help.txt
  if have help && [ -f "$qhelp" ] && [ -f "$ihelp" ]; then
    skipped "3 help output"
  else
    echo "== 3 help output of the tools (before anything is run)"
    invalidate_after help
    "$qbin" --help > "$qhelp" 2>&1 < /dev/null || true   # the usage exits with a status: the text is what counts
    "$ibin" --help > "$ihelp" 2>&1 < /dev/null || true
    for want in "--tensor-type:$qhelp:llama-quantize" "--imatrix:$qhelp:llama-quantize" "--chunks:$ihelp:llama-imatrix"; do
      flag=${want%%:*}; rest=${want#*:}; file=${rest%%:*}; tool_name=${rest##*:}
      has_flag "$flag" "$file" || fail "FLAG_NOT_ACCEPTED:$flag ($tool_name --help does not list it)"
    done
    echo "help: --tensor-type and --imatrix (llama-quantize) and --chunks (llama-imatrix) are accepted"
    {
      echo "quantize_help=$qhelp"; echo "imatrix_help=$ihelp"; echo "flags_checked=--tensor-type --imatrix --chunks"
      echo "utc=$(utc_iso)"
    } > "$(marker help)"
  fi
  if grep -Eq -- '--tensor-type=' "$qhelp"; then tt=("${tt_equals[@]}"); tt_syntax=equals; else tt=("${tt_two[@]}"); tt_syntax=two-argument; fi
  echo "--tensor-type syntax from the help: $tt_syntax"

  # 4 and 5 -------------------------------------------------------------------------------------------------
  if [ "$mode" = make ]; then
    if have wikitext && [ -f "$text" ]; then
      skipped "4 wikitext-2 text"
    else
      echo "== 4 wikitext-2 raw train: Salesforce/wikitext @ b08601e04326c79dfdd32d625aee71d232d685c3 (no token)"
      invalidate_after wikitext
      for attempt in 1 2; do
        HF_HUB_DISABLE_XET=1 "${tool[@]}" wikitext --out "$text" --cache "$q4/hub" > "$q4/wikitext.json" 2> "$logs/wikitext.err" && break
        if [ "$attempt" -eq 1 ] && grep -q 'pyarrow is not installed' "$logs/wikitext.err"; then
          echo "pyarrow is missing: installing it once"
          { python3 -m pip install --quiet pyarrow || uv pip install --python "$(command -v python3)" --quiet pyarrow; } > "$logs/pip-pyarrow.log" 2>&1
          continue
        fi
        fail "WIKITEXT:$(tail -n 1 "$logs/wikitext.err")"
      done
      echo "wikitext text: $(json_get text_bytes < "$q4/wikitext.json") bytes, $(json_get rows < "$q4/wikitext.json") rows, sha256 $(json_get text_sha256 < "$q4/wikitext.json")"
      { echo "file=$text"; echo "sha256=$(json_get text_sha256 < "$q4/wikitext.json")"; echo "utc=$(utc_iso)"; } > "$(marker wikitext)"
    fi
    if have imatrix && [ "$(marker_get imatrix mode)" = make ] && [ -f "$imatrix" ] && [ "$(sha_of "$imatrix")" = "$(marker_get imatrix sha256)" ]; then
      skipped "5 importance matrix"
    else
      echo "== 5 importance matrix (imatrix), 200 chunks of wikitext-2"
      invalidate_after imatrix
      rm -f "$imatrix"
      echo "+ ${imatrix_cmd[*]}"
      "${imatrix_cmd[@]}" > "$logs/imatrix.log" 2>&1 < /dev/null
      rc=$?
      tail -n 3 "$logs/imatrix.log"
      [ $rc -eq 0 ] || fail "IMATRIX_FAILED:exit status $rc (see imatrix.log)"
      [ -s "$imatrix" ] || fail "IMATRIX_FAILED:no output file $imatrix"
      chunks_line=$(grep -m 1 'computing over [0-9]* chunks' "$logs/imatrix.log")
      printf '%s' "$chunks_line" | grep -q 'computing over 200 chunks' || fail "IMATRIX_CHUNKS:the run did not use 200 chunks ('${chunks_line:-no chunk line}')"
      {
        echo "mode=make"; echo "command=${imatrix_cmd[*]}"; echo "sha256=$(sha_of "$imatrix")"; echo "size=$(size_of "$imatrix")"
        echo "log=$logs/imatrix.log"; echo "chunks_line=$chunks_line"; echo "utc=$(utc_iso)"
      } > "$(marker imatrix)"
    fi
  else
    echo "== 4 wikitext-2 text: not needed, an importance matrix is given"
    if have imatrix && [ "$(marker_get imatrix mode)" = file ] && [ "$(marker_get imatrix sha256)" = "$imatrix_sha" ] \
      && [ -f "$imatrix" ] && [ "$(sha_of "$imatrix")" = "$imatrix_sha" ]; then
      skipped "5 importance matrix"
    else
      echo "== 5 importance matrix (imatrix) given: sha256 must be $imatrix_sha"
      invalidate_after imatrix
      [ -f "$imatrix_arg" ] || fail "IMATRIX_MISSING:$imatrix_arg"
      got=$(sha_of "$imatrix_arg")
      [ "$got" = "$imatrix_sha" ] || fail "IMATRIX_SHA_MISMATCH:expected $imatrix_sha, $imatrix_arg has $got"
      if [ ! "$imatrix_arg" -ef "$imatrix" ]; then
        { cp "$imatrix_arg" "$imatrix.tmp" && mv "$imatrix.tmp" "$imatrix"; } || fail "IMATRIX_MISSING:cannot copy $imatrix_arg to $imatrix"
        [ "$(sha_of "$imatrix")" = "$imatrix_sha" ] || fail "IMATRIX_SHA_MISMATCH:the copy at $imatrix differs from $imatrix_arg"
      fi
      echo "imatrix verified: $imatrix_sha"
      {
        echo "mode=file"; echo "source=$imatrix_arg"; echo "sha256=$imatrix_sha"; echo "size=$(size_of "$imatrix")"; echo "utc=$(utc_iso)"
      } > "$(marker imatrix)"
    fi
  fi
  imatrix_now=$(marker_get imatrix sha256)

  # 6 -------------------------------------------------------------------------------------------------------
  quant_cmd=("$qbin" --imatrix "$imatrix" "${tt[@]}" "$bf16" "$out" Q4_K_M "$threads")
  if have quantize && [ -f "$out" ] && [ "$(size_of "$out")" = "$(marker_get quantize size)" ] && [ "$(marker_get quantize imatrix_sha256)" = "$imatrix_now" ]; then
    skipped "6 quantize"
  else
    echo "== 6 quantize to Q4_K_M with OVH's recipe ($threads threads)"
    invalidate_after quantize
    rm -f "$out"   # a half-written file of an earlier attempt must never be taken for the result
    echo "+ ${quant_cmd[*]}"
    "${quant_cmd[@]}" > "$logs/quantize.log" 2>&1 < /dev/null
    rc=$?
    tail -n 4 "$logs/quantize.log"
    [ $rc -eq 0 ] || { rm -f "$out"; fail "QUANTIZE_FAILED:exit status $rc (see quantize.log)"; }
    [ -s "$out" ] || fail "QUANTIZE_FAILED:no output file $out"
    table=$("${tool[@]}" quantize-table "$logs/quantize.log" --json "$q4/tensor_types.json") \
      || fail "RECIPE_NOT_APPLIED:$(printf '%s' "$table" | python3 -I -c 'import json, sys; print("; ".join(json.loads(sys.stdin.read() or "{}").get("problems", ["no table"])))')"
    qsize=$(size_of "$out")
    {
      echo "name=$out_name"; echo "file=$out"; echo "size=$qsize"; echo "sha256=$(sha_of "$out")"
      echo "imatrix_sha256=$imatrix_now"; echo "command=${quant_cmd[*]}"; echo "log=$logs/quantize.log"; echo "threads=$threads"
      "${tool[@]}" bits-per-weight --size "$qsize" --gguf "$bf16"
      echo "utc=$(utc_iso)"
    } > "$(marker quantize)"
  fi
  qsha=$(marker_get quantize sha256); qsize=$(marker_get quantize size)
  echo "quantized file: $out, $qsize bytes, sha256 $qsha, $(marker_get quantize bits_per_weight) bits per weight"

  # 7 -------------------------------------------------------------------------------------------------------
  echo "== 7 manifest of this pod: variant $variant in $manifest (the Mac's copy is not touched)"
  edit=$("${tool[@]}" manifest-add --manifest "$manifest" --variant "$variant" --name "$out_name" --size "$qsize" --sha256 "$qsha" \
    --note "$note" --record "$q4/variant.json") || fail "MANIFEST_EDIT:$edit"
  if [ "$(printf '%s' "$edit" | json_get changed)" = True ] || [ ! -f "$q4/manifest_edit.json" ]; then
    printf '%s\n' "$edit" > "$q4/manifest_edit.json"
    echo "manifest edited: $variant added"
  else
    echo "manifest already has $variant (no change)"
  fi
  { echo "manifest=$manifest"; echo "variant=$variant"; echo "utc=$(utc_iso)"; } > "$(marker manifest)"

  # 8 -------------------------------------------------------------------------------------------------------
  echo "== 8 prepare.json and q4.done (last)"
  finish "done" ""
}

main 2>&1 | tee -a "$log"
exit "${PIPESTATUS[0]}"
