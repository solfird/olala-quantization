#!/usr/bin/env bash
# One component ablation end to end on one pod: which part of the model causes the NVFP4 loss? Convert the bf16 model with
# only the part SPEC quantized (everything else bf16 and listed in quantization_config.ignore, list I always bf16), verify
# the checkpoint, serve it like arm_pipeline.sh's ckpt: mode, smoke-gate, check the checkpoint (and the server's
# "Quantizing N layers" line when it prints one) against the prediction, then run evals/fidelity.py against the bf16
# reference and write KL, perplexity ratio and top-1 agreement. Detach it; the log is the record.
# Usage: ablation_arm.sh [--dry-run] LABEL SPEC       e.g. abl-experts-L12-23  experts:layers=12-23
#   SPEC: all | experts | non-experts | shared-experts | latent | attention | mamba, optionally :layers=A-B[,N...]
#   (olala_quant/ablation.py; layer ranges include both ends). LABEL names a directory, a checkpoint and a log.
# Env: ABLATION_RECIPE nvfp4 (default, the N2 format) or fp8; ABLATION_REF = path of the bf16 reference logprob file
#      (default /workspace/results/raw/b0_wikitext2.npz; when that file is absent the bf16 model is served ONCE, here,
#      and the reference is collected from it, plus a second collection of the same server for the noise floor);
#      ABLATION_NOISE 1 (default) or 0: that second collection; ABLATION_KEEP_CKPT 0 (default: the checkpoint is
#      removed after a good run, 100 GB of disk do not hold ten of them) or 1; ABLATION_STOP_WAIT seconds after a stop (3).
# Files: /workspace/results/ablation-LABEL/ (summary.json, provenance.txt, expected.json, verify.json, check.txt, smoke.txt,
#   fidelity_vs_b0.json and .md, the variant's logprob file, server.log, the checkpoint's index, config and
#   quantization_info.json, and reference/ when this run produced the reference) and
#   /workspace/results/ablation-LABEL.done (the file runpod/collect.sh files-mode waits for: written LAST, on success and on
#   failure, so that a failed run's logs are pulled too).
# Exit: 0 done, 1 a step failed, 2 bad arguments, 3 the bf16 reference could not be made. The last line is ABLATION_DONE or
#   ABLATION_FAILED (the reason is on the line before it and in the .done file).
set -u
here=$(cd "$(dirname "$0")" && pwd)
usage="usage: ablation_arm.sh [--dry-run] LABEL SPEC   (env ABLATION_RECIPE nvfp4|fp8, ABLATION_REF, ABLATION_NOISE 0|1, ABLATION_KEEP_CKPT 0|1)"
refuse() { echo "$usage" >&2; echo "refused: $*" >&2; echo ABLATION_FAILED; exit 2; }

dry=0; positional=()
for a in "$@"; do
  case $a in
    --dry-run) dry=1 ;;
    -*) refuse "unknown option $a" ;;
    *) positional+=("$a") ;;
  esac
done
[ ${#positional[@]} -eq 2 ] || refuse "expected LABEL and SPEC, got ${#positional[@]} argument(s)"
label=${positional[0]}; spec=${positional[1]}
[[ $label =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || refuse "bad LABEL '$label' (letters, digits, . _ -; it names a directory, a checkpoint and a log)"
[[ $spec =~ ^[a-z][a-z-]*(:layers=[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*)?$ ]] || refuse "bad SPEC '$spec' (NAME or NAME:layers=A-B[,N...], see olala_quant/ablation.py)"
recipe=${ABLATION_RECIPE-nvfp4}
[[ $recipe =~ ^(nvfp4|fp8)$ ]] || refuse "ABLATION_RECIPE must be nvfp4 or fp8, got '$recipe'"
noise=${ABLATION_NOISE-1}
[[ $noise =~ ^[01]$ ]] || refuse "ABLATION_NOISE must be 0 or 1, got '$noise'"
keep=${ABLATION_KEEP_CKPT-0}
[[ $keep =~ ^[01]$ ]] || refuse "ABLATION_KEEP_CKPT must be 0 or 1, got '$keep'"
stopwait=${ABLATION_STOP_WAIT-3}
[[ $stopwait =~ ^[0-9]+$ ]] || refuse "ABLATION_STOP_WAIT must be a number of seconds, got '$stopwait'"

B=/workspace/bundle
bf16=/workspace/models/olala-bf16
results=/workspace/results
raw=$results/raw
outdir=$results/ablation-$label
donefile=$results/ablation-$label.done
dir=/workspace/ckpt/abl-$label
logname=ckpt-abl-$label
chunks=data/wikitext2_test_64x512.json
ref=${ABLATION_REF:-$raw/b0_wikitext2.npz}

if [ $dry -eq 1 ]; then
  echo "ablation_arm.sh dry run: $label $spec (recipe $recipe; nothing is run, nothing is written)"
  echo "refuses when $outdir exists; removes a stale $donefile first"
  echo "stop: bash $B/pod/stop_server.sh"
  echo "select: cd $B/quant && python3 -m olala_quant.ablation expected --src $bf16 --select $spec > $outdir/expected.json   (refuses a bad or empty selection)"
  if [ -n "${ABLATION_REF:-}" ]; then
    echo "reference: given by ABLATION_REF, must exist: $ref"
  elif [ -f "$ref" ]; then
    echo "reference present: $ref (bf16 is not served)"
  else
    echo "reference absent: bash $B/pod/serve_olala.sh bf16, smoke.sh bf16 --timeout 900, python3 fidelity.py collect --out $raw/b0_wikitext2.npz"
    [ "$noise" -eq 0 ] || echo "noise floor: a second collect --out $raw/b0repeat_wikitext2.npz and fidelity.py compare of the two"
    echo "then: bash $B/pod/stop_server.sh"
  fi
  echo "patch: apply_patch (olala-quant-loader.patch)"
  echo "convert: cd $B/quant && python3 -m olala_quant.convert --src $bf16 --dst $dir --recipe $recipe --device cuda --select $spec"
  echo "verify: python3 -m olala_quant.verify --src $bf16 --dst $dir --report $outdir/verify.json --select $spec"
  echo "provenance: sha256sum of model.safetensors.index.json config.json quantization_info.json in $outdir/provenance.txt"
  echo "serve: bash $B/pod/serve_olala.sh ckpt $dir"
  echo "smoke gate (exit 0 or 2 accepted): bash $B/pod/smoke.sh $logname --timeout 900"
  echo "check: python3 -m olala_quant.ablation check --src $bf16 --dst $dir --select $spec --server-log /workspace/logs/$logname/server-*.log"
  echo "fidelity: cd $B/evals && python3 fidelity.py collect --chunks $chunks --out $raw/${label}_wikitext2.npz"
  echo "stop: bash $B/pod/stop_server.sh"
  echo "fidelity: python3 fidelity.py compare $ref $raw/${label}_wikitext2.npz --json $outdir/fidelity_vs_b0.json"
  [ "$keep" -eq 1 ] || echo "cleanup: rm -rf $dir after a good run (ABLATION_KEEP_CKPT=1 keeps it)"
  echo "last: $donefile, then ABLATION_DONE (or ABLATION_FAILED with the reason before it)"
  exit 0
fi

. "$here/job_common.sh"   # log, apply_patch, EV, QU, OUT (= $raw)
[ ! -e "$outdir" ] || refuse "$outdir exists (an ablation never overwrites; use a new LABEL)"
rm -f "$donefile"   # a .done of an earlier attempt must not let collect.sh pull an unfinished run
mkdir -p "$outdir" "$results"
prov=$outdir/provenance.txt
started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
{ echo "run=$label spec=$spec recipe=$recipe started=$started"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; } > "$prov"
tmpout=$(mktemp)
trap 'rm -f "$tmpout"' EXIT
serving=""; produced=false; created=""

stop_srv() { bash "$B/pod/stop_server.sh" > /dev/null 2>&1 || log "warning: stop_server.sh exit $?"; serving=""; sleep "$stopwait"; }

write_summary() {  # STATUS REASON
  python3 - "$outdir" "$1" "$2" "$label" "$spec" "$recipe" "$ref" "$produced" "$started" <<'PY' || log "warning: summary.json not written"
import datetime
import hashlib
import json
import pathlib
import sys

outdir, status, reason, label, requested, recipe, ref, produced, started = sys.argv[1:10]
o = pathlib.Path(outdir)


def load(name):
    try:
        return json.loads((o / name).read_text())
    except (OSError, ValueError):
        return None


def first_word(name):
    try:
        return (o / name).read_text().split()[0].rstrip(":")
    except (OSError, IndexError):
        return None


expected, fid, noise = load("expected.json"), load("fidelity_vs_b0.json"), load("reference/noise_floor.json")
lines = (o / "provenance.txt").read_text().splitlines() if (o / "provenance.txt").is_file() else []
summary = {
    "label": label,
    "status": status,
    "spec_requested": requested,
    "spec": (expected or {}).get("spec"),
    "recipe": recipe,
    "started": started,
    "finished": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "gpu": lines[1] if len(lines) > 1 else None,
    "expected": expected,
    "verify": first_word("verify.txt"),
    "check": first_word("check.txt"),
}
if status != "done":
    summary["reason"] = reason
ref_path = pathlib.Path(ref)
summary["reference"] = {
    "path": ref,
    "produced_here": produced == "true",
    "sha256": hashlib.sha256(ref_path.read_bytes()).hexdigest() if ref_path.is_file() else None,
}
if fid is not None:
    keys = ("ppl_ref", "ppl_var", "ppl_ratio", "kld_mean", "kld_median", "kld_p99", "top1_agreement")
    summary["fidelity"] = {k: fid[k] for k in keys}
    summary.update(fidelity={**summary["fidelity"], **{k: fid[k] for k in ("n_chunks", "n_positions", "k")}})
if noise is not None:
    summary["noise_floor"] = {k: noise[k]["value"] for k in ("kld_mean", "top1_agreement", "ppl_ratio")}
(o / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
PY
}

fail() {  # REASON [EXIT]
  log "FAILED: $1"
  [ -z "$serving" ] || stop_srv
  write_summary failed "$1"
  printf 'ABLATION_FAILED: %s\n' "$1" > "$donefile"
  echo ABLATION_FAILED
  exit "${2:-1}"
}

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

log "== ablation $label: $spec ($recipe)"
bash "$B/pod/stop_server.sh" > /dev/null 2>&1; sleep "$stopwait"   # idempotent: nothing running is fine

(cd "$QU" && PYTHONPATH="$QU" python3 -m olala_quant.ablation expected --src "$bf16" --select "$spec" > "$outdir/expected.json" 2> "$tmpout") \
  || { tail -n 3 "$tmpout"; fail "selection refused by olala_quant.ablation expected: $(tail -n 1 "$tmpout" | cut -c1-200)"; }
log "expected counts: $(tr -d '\n ' < "$outdir/expected.json" | cut -c1-300)"

# --- the bf16 reference: given, present, or produced once on this pod ----------------------------------------------------
if [ -n "${ABLATION_REF:-}" ]; then
  [ -f "$ref" ] || fail "reference: ABLATION_REF $ref does not exist" 3
elif [ ! -f "$ref" ]; then
  log "== bf16 reference (once per pod): $ref"
  serving=1
  bash "$B/pod/serve_olala.sh" bf16 > "$tmpout" 2>&1 || { tail -n 4 "$tmpout"; fail "reference: serve bf16 failed" 3; }
  bash "$B/pod/smoke.sh" bf16 --timeout 900 > "$outdir/reference-smoke.txt" 2>&1
  rc=$?
  log "reference smoke exit $rc: $(head -n 2 "$outdir/reference-smoke.txt" | tr '\n' ' ')"
  { [ $rc -eq 0 ] || [ $rc -eq 2 ]; } || fail "reference: bf16 smoke gate (exit $rc)" 3
  collect_to b0 || fail "reference: bf16 collection failed or wrote no logprob file" 3
  if [ "$noise" -eq 1 ]; then collect_to b0repeat || log "warning: the repeat collection (noise floor) failed"; fi
  stop_srv
  mkdir -p "$outdir/reference"
  cp "$ref" "$outdir/reference/" || fail "reference: copy of $ref" 3
  if [ "$noise" -eq 1 ] && [ -s "$OUT/b0repeat_wikitext2.npz" ]; then
    compare_to "$ref" "$OUT/b0repeat_wikitext2.npz" "$outdir/reference/noise_floor.json" "$outdir/reference/noise_floor.md" \
      || log "warning: the noise-floor comparison failed"
  fi
  produced=true
else
  log "reference present: $ref"
fi

# --- the checkpoint --------------------------------------------------------------------------------------------------
apply_patch || fail "loader patch"
[ ! -e "$dir" ] || fail "checkpoint directory $dir exists (left alone: remove it or use a new LABEL)"
log "convert $dir ($recipe, --select $spec)"
created=1
(cd "$QU" && PYTHONPATH="$QU" python3 -m olala_quant.convert --src "$bf16" --dst "$dir" --recipe "$recipe" --device cuda \
  --select "$spec" > "$tmpout" 2>&1) || { tail -n 4 "$tmpout"; fail "convert (olala_quant.convert $recipe --select $spec)"; }
tail -n 2 "$tmpout"
log "verify $dir"
(cd "$QU" && PYTHONPATH="$QU" python3 -m olala_quant.verify --src "$bf16" --dst "$dir" --report "$outdir/verify.json" \
  --select "$spec" > "$tmpout" 2>&1) || { tail -n 8 "$tmpout"; fail "verify (olala_quant.verify exit status not 0)"; }
tail -n 1 "$tmpout" > "$outdir/verify.txt"
echo "verify: $(cut -c1-300 "$outdir/verify.txt")" >> "$prov"
(cd "$dir" && sha256sum model.safetensors.index.json config.json quantization_info.json) >> "$prov" \
  || fail "provenance (sha256 of the checkpoint files)"

serving=1
bash "$B/pod/serve_olala.sh" ckpt "$dir" > "$tmpout" 2>&1 || { tail -n 4 "$tmpout"; fail "serve (serve_olala.sh ckpt $dir)"; }
tail -n 2 "$tmpout"
bash "$B/pod/smoke.sh" "$logname" --timeout 900 > "$outdir/smoke.txt" 2>&1
rc=$?
log "smoke exit $rc: $(head -n 2 "$outdir/smoke.txt" | tr '\n' ' ')"
{ [ $rc -eq 0 ] || [ $rc -eq 2 ]; } || { tail -n 8 "$outdir/smoke.txt"; fail "smoke gate (exit $rc)"; }
# shellcheck disable=SC2012  # the names are server-<UTC>.log: no odd characters
serverlog=$(ls -t "/workspace/logs/$logname"/server-*.log 2> /dev/null | head -n 1)
banners=$(grep -h -E 'Quantizing [0-9]+ layers|Selected .*Kernel|Using .* MoE backend' "$serverlog" 2> /dev/null | cut -c1-300)
if [ -n "$banners" ]; then echo "$banners" >> "$prov"; else echo "warning: no kernel banner found in ${serverlog:-any server log}" >> "$prov"; fi
(cd "$QU" && PYTHONPATH="$QU" python3 -m olala_quant.ablation check --src "$bf16" --dst "$dir" --select "$spec" \
  --server-log "$serverlog" > "$outdir/check.txt" 2>&1) \
  || { tail -n 6 "$outdir/check.txt"; fail "check (olala_quant.ablation check: the checkpoint or the server does not match the selection)"; }
tail -n 1 "$outdir/check.txt"

# --- the fidelity run against the bf16 reference ------------------------------------------------------------------------
log "== fidelity $label against $ref"
collect_to "$label"
rc=$?
[ $rc -ne 1 ] || fail "fidelity collect failed (exit status not 0)"
[ $rc -eq 0 ] || fail "fidelity collect wrote no logprob file"
stop_srv
compare_to "$ref" "$OUT/${label}_wikitext2.npz" "$outdir/fidelity_vs_b0.json" "$outdir/fidelity_vs_b0.md" \
  || fail "fidelity compare (fidelity.py compare exit status not 0)"
grep -E 'Perplexity ratio|Mean KL|99th|top-1' "$outdir/fidelity_vs_b0.md" | sed "s/^/$label: /"

cp "$OUT/${label}_wikitext2.npz" "$outdir/" || log "warning: logprob file not copied"
[ -z "$serverlog" ] || cp "$serverlog" "$outdir/server.log"
cp "$dir/quantization_info.json" "$dir/config.json" "$dir/model.safetensors.index.json" "$outdir/" || log "warning: checkpoint metadata not copied"
if [ "$keep" -eq 0 ] && [ -n "$created" ]; then rm -rf -- "$dir"; log "checkpoint $dir removed (ABLATION_KEEP_CKPT=1 keeps it)"; fi
write_summary "done" ""
echo ABLATION_DONE > "$donefile"   # the data is in place; collect.sh may pull it from now on
echo ABLATION_DONE
