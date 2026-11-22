#!/usr/bin/env bash
# Rebuild olala-quant-loader.patch from a throw-away clone of the OVH vLLM fork.
# Usage: fork-patch/build_patch.sh            (from anywhere; paths are relative to this file)
# It never touches recon/src/vllm: it clones it into quant/work/ and diffs there.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
quant="$(dirname "$here")"
fork="${OLALA_FORK_DIR:-$quant/../recon/src/vllm}"
work="$quant/work/vllm-olala"
rm -rf "$work"
git clone --no-hardlinks -q "$fork" "$work"
cp "$here/olala_quant_loader.py" "$work/vllm/model_executor/models/olala_quant_loader.py"
# olala_py.diff is the hand-edited part (a plain diff of models/olala.py); olala_quant_loader.py is the
# new file. The patch is their sum, regenerated here so the two sources cannot drift apart.
git -C "$work" apply "$here/olala_py.diff"
git -C "$work" add -A
git -C "$work" diff --cached --no-color --src-prefix=a/ --dst-prefix=b/ > "$here/olala-quant-loader.patch"
git -C "$work" reset -q --hard HEAD && git -C "$work" clean -fdq
git -C "$work" apply --check "$here/olala-quant-loader.patch" && echo "git apply --check: OK"
