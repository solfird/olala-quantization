#!/usr/bin/env bash
# Runs ON the pod (started by pod_setup.sh from the uploaded bundle): record the machine, check the toolchain,
# download the BF16 weights, verify them against the Hub. Every record goes to /workspace/logs/setup/.
# Usage: setup_remote.sh [--dry-run] [--skip-download] [--repo R] [--revision SHA] [--dest DIR]
#                        [--image-ref REF] [--image-digest DIGEST] [--expect-cc X.Y]
# Ends with one marker line for pod_setup.sh: SETUP_DONE, SETUP_DONE_WITH_WARNINGS or SETUP_FAILED.
# Records: nvidia-smi (driver, CUDA, GPU, compute capability), nproc and the cgroup limits (cpu.max,
# memory.max), df -h, the image reference and digest, pip list of vllm, torch, transformers, compressed-tensors
# and the kernel libraries, nvcc --version (the fork JIT-compiles its kernels with the nvcc of torch's CUDA major).
# Download: public repository, no token (HF_TOKEN is unset), one pinned revision, HF_HUB_DISABLE_XET=1 and 16
# workers (2026-11-06: xet on one connection ran at 27 MB/s and filled the disk). Verification: sizes and sha256
# against the Hub's LFS metadata (verify_model.py).
set -u
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=common.sh
. "$here/common.sh"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN

dry=0; skip_download=0
repo=OVHaiLLM/Olala-v1.0-mini-7A1B; revision=b40f8165ee2254c18601220e19842600c6d91794
dest=$MODEL_ROOT/olala-bf16; image_ref="(not given)"; image_digest="(not given)"; expect_cc=""
while [ $# -gt 0 ]; do
  case $1 in
    --dry-run) dry=1; shift ;;
    --skip-download) skip_download=1; shift ;;
    --repo) repo=${2:?}; shift 2 ;;
    --revision) revision=${2:?}; shift 2 ;;
    --dest) dest=${2:?}; shift 2 ;;
    --image-ref) image_ref=${2:?}; shift 2 ;;
    --image-digest) image_digest=${2:?}; shift 2 ;;
    --expect-cc) expect_cc=${2:?}; shift 2 ;;
    *) die "unknown argument $1" ;;
  esac
done
rec=$LOG_ROOT/setup
warnings=()
warn() { warnings+=("$*"); echo "WARN $*"; }
fail() { echo "SETUP_FAILED: $*"; exit 1; }

# record FILE COMMAND: run the command text, keep its output in the record folder (a failing record is a finding, not a stop)
record() {
  if [ $dry -eq 1 ]; then echo "+ [$rec/$1] $2"; return 0; fi
  bash -c "$2" > "$rec/$1" 2>&1 || echo "(exit status $?)" >> "$rec/$1"
}

DOWNLOAD_PY='
import sys
from huggingface_hub import snapshot_download
repo, revision, dest = sys.argv[1:4]
path = snapshot_download(repo_id=repo, revision=revision, local_dir=dest, max_workers=16, token=False,
                         ignore_patterns=["assets/*", "*.pdf"])
print("downloaded to", path)
'

[ $dry -eq 1 ] || mkdir -p "$rec" "$MODEL_ROOT"
echo "== setup_remote $(utc_iso): repo $repo @ $revision -> $dest"

echo "== 1 record the machine"
record nvidia-smi.txt 'nvidia-smi'
record gpu.csv 'nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,pci.bus_id --format=csv'
# shellcheck disable=SC2016  # single quotes on purpose: the text is run later by bash -c on the pod
record cpu-memory.txt '. '"$here"'/common.sh; echo "nproc (host cores visible): $(nproc)"; echo "container_cpus: $(container_cpus)"; echo "container_mem_gb: $(container_mem_gb)"; echo "max_jobs: $(max_jobs)"; echo "cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a)"; echo "memory.max: $(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo n/a)"; free -g'
record df.txt 'df -h'
record image.txt "printf 'image=%s\ndigest=%s\n' '$image_ref' '$image_digest'; env | grep -E '^(CUDA_VERSION|NVIDIA_|VLLM_IMAGE_TAG|TORCH_CUDA_ARCH_LIST)'"
# shellcheck disable=SC2016
record pip.txt 'python3 -m pip list 2>/dev/null || uv pip list --python "$(command -v python3)"'
record pip-key.txt 'python3 -m pip list 2>/dev/null | grep -iE "^(vllm|torch|transformers|compressed-tensors|tilelang|flashinfer|nvidia-cutlass|quack|triton|huggingface|safetensors|ninja) " || true'
record torch.txt 'python3 -c "import torch; print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"available\", torch.cuda.is_available(), \"capability\", torch.cuda.get_device_capability() if torch.cuda.is_available() else None)"'
record nvcc.txt 'command -v nvcc; nvcc --version; ls -d /usr/local/cuda*/bin/nvcc 2>&1'

echo "== 2 check the toolchain (the fork needs a driver of 580 or later, CUDA 13.0, and nvcc to JIT-compile its kernels)"
if [ $dry -eq 1 ]; then
  echo "+ fail when torch.cuda.is_available() is False; warn when driver < 580, compute capability != ${expect_cc:-expected}, nvcc missing or its major != torch.version.cuda major"
else
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2> /dev/null | head -n 1)
  cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | head -n 1)
  [ -n "$driver" ] || fail "nvidia-smi answers nothing: no GPU in the container"
  grep -q "available True" "$rec/torch.txt" || fail "torch cannot use the GPU: $(cat "$rec/torch.txt")"
  [ "${driver%%.*}" -ge 580 ] 2> /dev/null || warn "driver $driver is below 580: the CUDA 13.0 image may fail"
  [ -z "$expect_cc" ] || [ "$cc" = "$expect_cc" ] || warn "compute capability is $cc, the profile expected $expect_cc"
  if command -v nvcc > /dev/null 2>&1; then
    nvcc_major=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p' | head -n 1)
    torch_major=$(sed -n 's/.* cuda \([0-9]*\)\..*/\1/p' "$rec/torch.txt" | head -n 1)
    [ "$nvcc_major" = "$torch_major" ] || warn "nvcc major ($nvcc_major) differs from torch's CUDA major ($torch_major): the JIT build of the fork's kernels will fail (set OLALA_JIT_CUDA_HOME)"
  else
    warn "nvcc is not installed: the fork JIT-compiles its kernels, every Olala fast path would fall back to a slow path"
  fi
  echo "driver $driver, compute capability $cc, GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
fi

if [ $skip_download -eq 1 ]; then
  echo "== 3 download skipped (--skip-download)"
else
  echo "== 3 download $repo @ $revision to $dest (public repository, no token)"
  if [ $dry -eq 1 ]; then
    echo "+ require 30 GB free on $OLALA_WORK; HF_HUB_DISABLE_XET=1 python3 -c '$DOWNLOAD_PY' $repo $revision $dest"
  else
    free_gb=$(df -Pk "$OLALA_WORK" | awk 'NR==2{print int($4/1048576)}')
    [ "${free_gb:-0}" -ge 30 ] || fail "only ${free_gb:-0} GB free on $OLALA_WORK: the weights need 14 GB and the caches more"
    HF_HUB_DISABLE_XET=1 python3 -c "$DOWNLOAD_PY" "$repo" "$revision" "$dest" 2>&1 | tail -n 5 || fail "download failed"
  fi
fi

echo "== 4 verify sizes and sha256 against the Hub's metadata"
if [ $dry -eq 1 ]; then
  echo "+ python3 $here/verify_model.py --dir $dest --repo $repo --revision $revision --expect-revision $revision --ignore 'assets/*' --ignore '*.pdf' > $rec/model-verify.txt"
else
  if python3 -I "$here/verify_model.py" --dir "$dest" --repo "$repo" --revision "$revision" \
      --expect-revision "$revision" --ignore 'assets/*' --ignore '*.pdf' > "$rec/model-verify.txt" 2>&1; then
    tail -n 3 "$rec/model-verify.txt"
  else
    cat "$rec/model-verify.txt"; fail "the downloaded weights do not match the Hub's metadata"
  fi
fi
[ $dry -eq 1 ] || echo "$revision" > "$rec/model-revision.txt"

if [ $dry -eq 1 ]; then echo "SETUP_DRY_RUN (nothing was run)"
elif [ ${#warnings[@]} -gt 0 ]; then echo "SETUP_DONE_WITH_WARNINGS: ${#warnings[@]} warning(s) above"; else echo "SETUP_DONE"; fi
