#!/usr/bin/env bash
# Stage 1 of PROTOCOL.md amendment 25 on ONE pod: is there an NVFP4 build that is smaller than N5 and keeps its fidelity?
# The control N5 (`experts`, the 32 routed-expert layers outside list I, round-to-nearest) and the candidates (default C1 =
# `experts-all`, C2 = `experts-all` with the MSE-optimal block scales, C3 = `experts` with them) are each converted, verified,
# served, smoke-gated, checked against the selection, scored against the bf16 reference on the wikitext chunk set of
# amendments 19 and 22 (evals/fidelity.py), and gated. The steps of one arm are those of pod/ablation_arm.sh (copied, not
# called: that script cannot pass --scale-search to the converter, converts nothing it keeps, and is not to be edited).
# Usage: fidelity_batch.sh [--dry-run] LABEL [CANDIDATE ...]
#   CANDIDATE = [NAME=]SPEC[:scale-search]   SPEC as in olala_quant/ablation.py (experts, experts-all, experts:layers=A-B ...);
#   `:scale-search` converts with --scale-search. Unnamed candidates are C1, C2, ... in the order given. No candidate given:
#   the amendment's three, C1=experts-all C2=experts-all:scale-search C3=experts:scale-search. The control N5 always runs first.
# Env: FIDELITY_REF = path of the bf16 reference logprob file (default /workspace/results/raw/b0_wikitext2.npz; when absent
#      the bf16 model is served ONCE, here, and the reference and a noise-floor repeat are collected from it);
#      FIDELITY_NOISE 1 (default) or 0: that repeat; FIDELITY_STOP_WAIT seconds after a server stop (3);
#      FIDELITY_DEADLINE_MIN minutes after which no new arm is started (default 195; the pod lives 4 hours).
# Files: /workspace/results/fidelity-LABEL/ (summary.json, selection.json, arms/NAME/ with arm.json, expected.json, verify.json
#   and .txt, check.txt, smoke.txt, provenance.txt, the fidelity json and md, the logprob file, server.log and the checkpoint's
#   index, config and quantization_info.json; reference/ when this run produced the reference) and
#   /workspace/results/fidelity-LABEL.done (written LAST, on success and on failure: the file collect.sh files-mode waits for).
# Checkpoints: /workspace/ckpt/fid-LABEL-NAME, each kept until its arm's fidelity is measured. The control's is removed at once
#   (it is never selected). When all arms are done, the SELECTED candidate's checkpoint (the smallest file among those that pass
#   the gate, python3 fidelity_gate.py) is MOVED to /workspace/ckpt/selected and stays on the pod, never copied home; the other
#   candidates' are removed; with no passing candidate none is kept. A failed arm's checkpoint is left alone, as ablation_arm.sh does.
# Exit: 0 every arm done, 1 an arm failed or was skipped, 2 bad arguments, 3 the bf16 reference could not be made.
#   Markers: FIDELITY_ARM_DONE NAME / FIDELITY_ARM_FAILED NAME: reason per arm, FIDELITY_BATCH_DONE ok=N failed=N skipped=N
#   selected=NAME|none, then FIDELITY_DONE (or FIDELITY_FAILED, the reason on the line before it and in the .done file) last.
set -u
here=$(cd "$(dirname "$0")" && pwd)
usage="usage: fidelity_batch.sh [--dry-run] LABEL [[NAME=]SPEC[:scale-search] ...]   (env FIDELITY_REF, FIDELITY_NOISE, FIDELITY_STOP_WAIT, FIDELITY_DEADLINE_MIN)"
refuse() { echo "$usage" >&2; echo "refused: $*" >&2; echo FIDELITY_FAILED; exit 2; }

dry=0; positional=()
for a in "$@"; do
  case $a in
    --dry-run) dry=1 ;;
    -*) refuse "unknown option $a" ;;
    *) positional+=("$a") ;;
  esac
done
[ ${#positional[@]} -ge 1 ] || refuse "expected LABEL"
label=${positional[0]}
[[ $label =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || refuse "bad LABEL '$label' (letters, digits, . _ -; it names directories, checkpoints and a log)"
cands=("${positional[@]:1}")
[ ${#cands[@]} -gt 0 ] || cands=("C1=experts-all" "C2=experts-all:scale-search" "C3=experts:scale-search")
deadline_min=${FIDELITY_DEADLINE_MIN-195}
[[ $deadline_min =~ ^[0-9]+$ ]] || refuse "FIDELITY_DEADLINE_MIN must be a number of minutes, got '$deadline_min'"
noise=${FIDELITY_NOISE-1}
[[ $noise =~ ^[01]$ ]] || refuse "FIDELITY_NOISE must be 0 or 1, got '$noise'"
stopwait=${FIDELITY_STOP_WAIT-3}
[[ $stopwait =~ ^[0-9]+$ ]] || refuse "FIDELITY_STOP_WAIT must be a number of seconds, got '$stopwait'"

# arms: the control first, then the candidates
spec_re='^[a-z][a-z-]*(:layers=[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*)?$'
names=(N5); specs=(experts); sss=(0); roles=(control)
n=0
for tok in "${cands[@]}"; do
  n=$((n + 1)); name="C$n"; ss=0; spec=$tok
  if [[ $tok =~ ^([A-Za-z0-9][A-Za-z0-9._-]*)=(.+)$ ]]; then name=${BASH_REMATCH[1]}; spec=${BASH_REMATCH[2]}; fi
  if [[ $spec == *:scale-search ]]; then ss=1; spec=${spec%:scale-search}; fi
  [[ $spec =~ $spec_re ]] || refuse "bad candidate '$tok' (SPEC[:scale-search], SPEC = NAME or NAME:layers=A-B[,N...], see olala_quant/ablation.py)"
  for seen in "${names[@]}"; do [ "$seen" != "$name" ] || refuse "name '$name' given twice (N5 is the control)"; done
  names+=("$name"); specs+=("$spec"); sss+=("$ss"); roles+=(candidate)
done
order=$(IFS=,; echo "${names[*]}")

B=/workspace/bundle
bf16=/workspace/models/olala-bf16
results=/workspace/results
raw=$results/raw
ckptroot=/workspace/ckpt
outdir=$results/fidelity-$label
donefile=$results/fidelity-$label.done
selected=$ckptroot/selected
chunks=data/wikitext2_test_64x512.json
ref=${FIDELITY_REF:-$raw/b0_wikitext2.npz}
gate=$here/fidelity_gate.py
ckdir() { echo "$ckptroot/fid-$label-$1"; }

if [ $dry -eq 1 ]; then
  echo "fidelity_batch.sh dry run: $label, arms $order (nothing is run, nothing is written)"
  echo "refuses when $outdir or $selected exists; removes a stale $donefile first"
  for i in "${!names[@]}"; do
    echo "arm ${names[i]} (${roles[i]}): --select ${specs[i]}$([ "${sss[i]}" = 1 ] && echo ' --scale-search')"
  done
  echo "stop: bash $B/pod/stop_server.sh"
  echo "select, every arm first: cd $B/quant && python3 -m olala_quant.ablation expected --src $bf16 --select SPEC > $outdir/arms/NAME/expected.json"
  if [ -n "${FIDELITY_REF:-}" ]; then
    echo "reference: given by FIDELITY_REF, must exist: $ref"
  elif [ -f "$ref" ]; then
    echo "reference present: $ref (bf16 is not served)"
  else
    echo "reference absent: bash $B/pod/serve_olala.sh bf16, smoke.sh bf16 --timeout 900, python3 fidelity.py collect --out $raw/b0_wikitext2.npz"
    [ "$noise" -eq 0 ] || echo "noise floor: a second collect --out $raw/b0repeat_wikitext2.npz and fidelity.py compare of the two"
    echo "then: bash $B/pod/stop_server.sh"
  fi
  echo "patch: apply_patch (olala-quant-loader.patch)"
  echo "per arm, convert: cd $B/quant && python3 -m olala_quant.convert --src $bf16 --dst $ckptroot/fid-$label-NAME --recipe nvfp4 --device cuda --select SPEC [--scale-search]"
  echo "per arm, verify: python3 -m olala_quant.verify --src $bf16 --dst DIR --report $outdir/arms/NAME/verify.json --select SPEC --scale-search|--no-scale-search"
  echo "per arm: provenance (sha256 of the checkpoint's index, config, quantization_info.json), serve: bash $B/pod/serve_olala.sh ckpt DIR, smoke gate (exit 0 or 2 accepted): bash $B/pod/smoke.sh ckpt-fid-$label-NAME --timeout 900"
  echo "per arm, check: python3 -m olala_quant.ablation check --src $bf16 --dst DIR --select SPEC --server-log /workspace/logs/ckpt-fid-$label-NAME/server-*.log"
  echo "per arm, fidelity: cd $B/evals && python3 fidelity.py collect --chunks $chunks --out $raw/fid-$label-NAME_wikitext2.npz, stop, python3 fidelity.py compare $ref ... --json $outdir/arms/NAME/fidelity_vs_b0.json"
  echo "per arm: python3 $gate record ... (fidelity, sizes of the checkpoint); the control's checkpoint is removed (kept until measured, never selected)"
  echo "then: python3 $gate pick --results $outdir --order $order; the winner is moved to $selected (stays on the pod, never copied home), the others are removed"
  echo "then: python3 $gate finalize ... writes $outdir/summary.json and selection.json (thresholds and their source are in fidelity_gate.py only)"
  echo "last: $donefile, then FIDELITY_BATCH_DONE and FIDELITY_DONE (or FIDELITY_FAILED with the reason before it)"
  exit 0
fi

. "$here/job_common.sh"   # log, apply_patch, EV, QU, OUT (= $raw)
[ ! -e "$outdir" ] || refuse "$outdir exists (a batch never overwrites; use a new LABEL)"
[ ! -e "$selected" ] || refuse "$selected exists (the selected checkpoint of an earlier batch; move it away first)"
rm -f "$donefile"   # a .done of an earlier attempt must not let collect.sh pull an unfinished run
mkdir -p "$outdir/arms" "$results" "$ckptroot"
tmpout=$(mktemp)
trap 'rm -f "$tmpout"' EXIT
started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "batch=$label arms=$order started=$started" > "$outdir/provenance.txt"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >> "$outdir/provenance.txt" 2> /dev/null
serving=""; ok=0; failed=0; skipped=0

stop_srv() { bash "$B/pod/stop_server.sh" > /dev/null 2>&1 || log "warning: stop_server.sh exit $?"; serving=""; sleep "$stopwait"; }

collect_to() {  # NAME: top-20 logprobs of the 64 x 512 wikitext chunks from the server on port 8000; the status is checked
  rm -f "$OUT/$1_wikitext2.npz"
  (cd "$EV" && python3 fidelity.py collect --chunks "$chunks" --base-url http://127.0.0.1:8000/v1 --model olala \
    --out "$OUT/$1_wikitext2.npz" > "$tmpout" 2>&1) || { tail -n 4 "$tmpout"; return 1; }
  tail -n 1 "$tmpout"
  [ -s "$OUT/$1_wikitext2.npz" ] || return 2
}

compare_to() {  # REF VAR JSON MD
  (cd "$EV" && python3 fidelity.py compare "$1" "$2" --json "$3" > "$4" 2>&1) || { tail -n 4 "$4"; return 1; }
}

finish() {  # STATUS(done|failed) REASON EXIT: the summary, then the marker that lets collect.sh pull, LAST
  if [ "$1" = failed ]; then
    log "FAILED: $2"
    python3 "$gate" finalize --results "$outdir" --label "$label" --order "$order" > /dev/null 2>&1 || log "warning: summary.json not written"
    printf 'FIDELITY_FAILED: %s\n' "$2" > "$donefile"
    echo FIDELITY_FAILED
  else
    echo "FIDELITY_DONE selected=$2" > "$donefile"
    echo FIDELITY_DONE
  fi
  exit "$3"
}

arm_record() {  # STATUS REASON: arm.json of the arm in progress (sizes of its checkpoint when it exists)
  local ckpt_args=()
  [ ! -d "$dir" ] || ckpt_args=(--ckpt "$dir")
  python3 "$gate" record --arm-dir "$adir" --name "$name" --role "$role" --spec "$spec" --scale-search "$ss" \
    --status "$1" --reason "$2" ${ckpt_args[@]+"${ckpt_args[@]}"} || log "warning: arm.json of $name not written"
}

arm_fail() {  # REASON
  log "FAILED $name: $1"
  [ -z "$serving" ] || stop_srv
  arm_record failed "$1"
  echo "FIDELITY_ARM_FAILED $name: $1"
  failed=$((failed + 1))
}

arm_skip() {  # NAME WHY
  mkdir -p "$outdir/arms/$1"
  python3 "$gate" record --arm-dir "$outdir/arms/$1" --name "$1" --role "${role:-candidate}" --spec "${spec:-}" \
    --scale-search "${ss:-0}" --status skipped --reason "$2" || log "warning: arm.json of $1 not written"
  echo "FIDELITY_ARM_FAILED $1: skipped, $2"
  skipped=$((skipped + 1))
}

run_arm() {  # INDEX: one arm, from the conversion to the recorded fidelity; returns 0 done, 1 failed
  local i=$1 rc serverlog banners
  name=${names[i]}; spec=${specs[i]}; ss=${sss[i]}; role=${roles[i]}
  adir=$outdir/arms/$name; dir=$(ckdir "$name")
  local logname=ckpt-fid-$label-$name npz=fid-$label-$name aprov=$outdir/arms/$name/provenance.txt
  local convert_extra=() verify_flag=--no-scale-search
  [ "$ss" -eq 0 ] || { convert_extra=(--scale-search); verify_flag=--scale-search; }
  log "== arm $name ($role): $spec$([ "$ss" -eq 1 ] && echo ' with --scale-search')"
  echo "arm=$name role=$role spec=$spec scale_search=$ss started=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$aprov"
  apply_patch || { arm_fail "loader patch"; return 1; }
  [ ! -e "$dir" ] || { arm_fail "checkpoint directory $dir exists (left alone: remove it or use a new LABEL)"; return 1; }
  log "convert $dir (nvfp4, --select $spec${convert_extra[*]:+ ${convert_extra[*]}})"
  (cd "$QU" && PYTHONPATH="$QU" python3 -m olala_quant.convert --src "$bf16" --dst "$dir" --recipe nvfp4 --device cuda \
    --select "$spec" ${convert_extra[@]+"${convert_extra[@]}"} > "$tmpout" 2>&1) \
    || { tail -n 4 "$tmpout"; arm_fail "convert (olala_quant.convert nvfp4 --select $spec${convert_extra[*]:+ ${convert_extra[*]}})"; return 1; }
  tail -n 2 "$tmpout"
  log "verify $dir"
  (cd "$QU" && PYTHONPATH="$QU" python3 -m olala_quant.verify --src "$bf16" --dst "$dir" --report "$adir/verify.json" \
    --select "$spec" "$verify_flag" > "$tmpout" 2>&1) \
    || { tail -n 8 "$tmpout"; arm_fail "verify (olala_quant.verify --select $spec $verify_flag exit status not 0)"; return 1; }
  tail -n 1 "$tmpout" > "$adir/verify.txt"
  echo "verify: $(cut -c1-300 "$adir/verify.txt")" >> "$aprov"
  (cd "$dir" && sha256sum model.safetensors.index.json config.json quantization_info.json) >> "$aprov" \
    || { arm_fail "provenance (sha256 of the checkpoint files)"; return 1; }

  serving=1
  bash "$B/pod/serve_olala.sh" ckpt "$dir" > "$tmpout" 2>&1 || { tail -n 4 "$tmpout"; arm_fail "serve (serve_olala.sh ckpt $dir)"; return 1; }
  tail -n 2 "$tmpout"
  bash "$B/pod/smoke.sh" "$logname" --timeout 900 > "$adir/smoke.txt" 2>&1
  rc=$?
  log "smoke exit $rc: $(head -n 2 "$adir/smoke.txt" | tr '\n' ' ')"
  { [ $rc -eq 0 ] || [ $rc -eq 2 ]; } || { tail -n 8 "$adir/smoke.txt"; arm_fail "smoke gate (exit $rc)"; return 1; }
  # shellcheck disable=SC2012  # the names are server-<UTC>.log: no odd characters
  serverlog=$(ls -t "/workspace/logs/$logname"/server-*.log 2> /dev/null | head -n 1)
  banners=$(grep -h -E 'Quantizing [0-9]+ layers|Selected .*Kernel|Using .* MoE backend' "$serverlog" 2> /dev/null | cut -c1-300)
  if [ -n "$banners" ]; then echo "$banners" >> "$aprov"; else echo "warning: no kernel banner found in ${serverlog:-any server log}" >> "$aprov"; fi
  (cd "$QU" && PYTHONPATH="$QU" python3 -m olala_quant.ablation check --src "$bf16" --dst "$dir" --select "$spec" \
    --server-log "$serverlog" > "$adir/check.txt" 2>&1) \
    || { tail -n 6 "$adir/check.txt"; arm_fail "check (olala_quant.ablation check: the checkpoint or the server does not match the selection)"; return 1; }
  tail -n 1 "$adir/check.txt"

  log "== fidelity $name against $ref"
  collect_to "$npz"
  rc=$?
  [ $rc -ne 1 ] || { arm_fail "fidelity collect failed (exit status not 0)"; return 1; }
  [ $rc -eq 0 ] || { arm_fail "fidelity collect wrote no logprob file"; return 1; }
  stop_srv
  compare_to "$ref" "$OUT/${npz}_wikitext2.npz" "$adir/fidelity_vs_b0.json" "$adir/fidelity_vs_b0.md" \
    || { arm_fail "fidelity compare (fidelity.py compare exit status not 0)"; return 1; }
  grep -E 'Perplexity ratio|Mean KL|99th|top-1' "$adir/fidelity_vs_b0.md" | sed "s/^/$name: /"
  cp "$OUT/${npz}_wikitext2.npz" "$adir/" || log "warning: logprob file not copied"
  [ -z "$serverlog" ] || cp "$serverlog" "$adir/server.log"
  cp "$dir/quantization_info.json" "$dir/config.json" "$dir/model.safetensors.index.json" "$adir/" || log "warning: checkpoint metadata not copied"
  arm_record "done" ""   # the size is read now, while the checkpoint is still there
  if [ "$role" = control ]; then rm -rf -- "$dir"; log "checkpoint of the control $name removed (it is never selected)"; fi
  echo "FIDELITY_ARM_DONE $name"
  ok=$((ok + 1))
}

log "== fidelity batch $label: arms $order"
bash "$B/pod/stop_server.sh" > /dev/null 2>&1; sleep "$stopwait"   # idempotent: nothing running is fine

# every selection against the real header first: a typo or an empty selection costs nothing here
selection_refused() {  # INDEX: the code refused the selection of this arm; the reason is the last line of $tmpout
  tail -n 3 "$tmpout"
  name=${names[$1]}; spec=${specs[$1]}; role=${roles[$1]}; ss=${sss[$1]}; adir=$outdir/arms/$name; dir=""
  arm_record failed "selection refused"
  finish failed "selection refused by olala_quant.ablation expected for $name: $(tail -n 1 "$tmpout" | cut -c1-200)" 1
}
for i in "${!names[@]}"; do
  mkdir -p "$outdir/arms/${names[i]}"
  (cd "$QU" && PYTHONPATH="$QU" python3 -m olala_quant.ablation expected --src "$bf16" --select "${specs[i]}" \
    > "$outdir/arms/${names[i]}/expected.json" 2> "$tmpout") || selection_refused "$i"
  log "expected counts ${names[i]}: $(tr -d '\n ' < "$outdir/arms/${names[i]}/expected.json" | cut -c1-300)"
done

# --- the bf16 reference: given, present, or produced once on this pod ----------------------------------------------------
refail() { log "FAILED: $1"; [ -z "$serving" ] || stop_srv; finish failed "$1" 3; }
if [ -n "${FIDELITY_REF:-}" ]; then
  [ -f "$ref" ] || refail "reference: FIDELITY_REF $ref does not exist"
elif [ ! -f "$ref" ]; then
  log "== bf16 reference (once per pod): $ref"
  serving=1
  bash "$B/pod/serve_olala.sh" bf16 > "$tmpout" 2>&1 || { tail -n 4 "$tmpout"; refail "reference: serve bf16 failed"; }
  bash "$B/pod/smoke.sh" bf16 --timeout 900 > "$outdir/reference-smoke.txt" 2>&1
  rc=$?
  log "reference smoke exit $rc: $(head -n 2 "$outdir/reference-smoke.txt" | tr '\n' ' ')"
  { [ $rc -eq 0 ] || [ $rc -eq 2 ]; } || refail "reference: bf16 smoke gate (exit $rc)"
  collect_to b0 || refail "reference: bf16 collection failed or wrote no logprob file"
  if [ "$noise" -eq 1 ]; then collect_to b0repeat || log "warning: the repeat collection (noise floor) failed"; fi
  stop_srv
  mkdir -p "$outdir/reference"
  cp "$ref" "$outdir/reference/" || refail "reference: copy of $ref"
  if [ "$noise" -eq 1 ] && [ -s "$OUT/b0repeat_wikitext2.npz" ]; then
    compare_to "$ref" "$OUT/b0repeat_wikitext2.npz" "$outdir/reference/noise_floor.json" "$outdir/reference/noise_floor.md" \
      || log "warning: the noise-floor comparison failed"
  fi
else
  log "reference present: $ref"
fi

# --- the arms ---------------------------------------------------------------------------------------------------------------
deadline=$(( $(date +%s) + deadline_min * 60 ))
for i in "${!names[@]}"; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    name=${names[i]}; spec=${specs[i]}; role=${roles[i]}; ss=${sss[i]}
    arm_skip "$name" "deadline of $deadline_min minutes reached before it started"
    continue
  fi
  run_arm "$i"
done

# --- the choice, the kept checkpoint, the summary ----------------------------------------------------------------------------
chosen=$(python3 "$gate" pick --results "$outdir" --order "$order") || finish failed "the choice of the candidate failed (fidelity_gate.py pick)" 1
log "gate: selected $chosen"
if [ "$chosen" != none ]; then
  mv -- "$(ckdir "$chosen")" "$selected" || finish failed "the selected checkpoint could not be moved to $selected" 1
  log "checkpoint of $chosen kept on the pod: $selected"
fi
for i in "${!names[@]}"; do
  [ "${roles[i]}" = candidate ] && [ "${names[i]}" != "$chosen" ] || continue
  [ ! -f "$outdir/arms/${names[i]}/fidelity_vs_b0.json" ] || rm -rf -- "$(ckdir "${names[i]}")"   # measured, not selected: removed
done
if [ "$chosen" != none ]; then
  python3 "$gate" finalize --results "$outdir" --label "$label" --order "$order" --selected "$chosen" --checkpoint "$selected" \
    || finish failed "summary.json and selection.json not written (fidelity_gate.py finalize)" 1
else
  python3 "$gate" finalize --results "$outdir" --label "$label" --order "$order" --selected none \
    || finish failed "summary.json and selection.json not written (fidelity_gate.py finalize)" 1
fi
[ ! -f "$outdir/reference/noise_floor.json" ] || cp "$outdir/reference/noise_floor.json" "$outdir/noise_floor.json"
echo "FIDELITY_BATCH_DONE ok=$ok failed=$failed skipped=$skipped selected=$chosen"
if [ $((failed + skipped)) -eq 0 ]; then rc_batch=0; else rc_batch=1; fi
finish "done" "$chosen" "$rc_batch"
