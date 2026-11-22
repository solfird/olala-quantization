#!/usr/bin/env bash
# Runs ON the pod: build OVH's llama.cpp fork (llama-server only, CUDA, one GPU architecture) at a pinned commit and
# download one of OVH's published Olala GGUF files with a pinned revision, size and sha256. Idempotent.
# Usage: gguf_setup.sh [--dry-run] [--variant V] [--cc X.Y | --cuda-arch ARCH] [--skip-build] [--skip-download]
#   --variant V     q5_k_m (default), q6_k, q8_0-safe2 or bf16 (the same-engine control): gguf_files.json
#   --cc X.Y        the GPU's compute capability when nvidia-smi is not asked (dry runs; default 12.0, Blackwell sm_120)
#   --cuda-arch A   CMAKE_CUDA_ARCHITECTURES, replacing the one derived from the compute capability (12.0 -> 120a-real)
# Steps (every one logged to /workspace/logs/gguf/setup-<UTC>.log; the last line is GGUF_SETUP_DONE or
# GGUF_SETUP_FAILED: <reason>; a dry run prints the plan, reads no GPU and touches nothing):
#   1 prerequisites: cmake, g++/make, git, the CUDA toolkit pieces llama.cpp needs (nvcc >= 12.8 for sm_120, the
#     cuda_runtime.h and cublas_v2.h headers, the libcuda.so stub); missing ones are installed with apt-get (the
#     image ovhaillm/olala-vllm is built from NVIDIA's CUDA 13.0.3 Ubuntu 24.04 image and ships cuda-nvcc, cublas-dev)
#   2 fetch the fork by COMMIT (git fetch --depth 1 origin <sha>), check HEAD and the tree against the pins
#   3 configure: CUDA on, Release, -DLLAMA_USE_PREBUILT_UI=OFF (the card: the prebuilt web UI does not match the
#     fork), one CUDA architecture (a short build), no tests, examples or unified app
#   4 build the llama-server target only, with the compile jobs the container's CPU and memory allow
#   5 check that the binary sees a CUDA device (`llama-server --list-devices`), then record the commit
#   6 download the GGUF file from the pinned Hub revision with huggingface_hub (token=False: no credential on a pod;
#     HF_HUB_DISABLE_XET=1 and the resumable plain download, the lesson of 2026-11-06)
#   7 verify the file's size and sha256 against the pins (a mismatch is a failure; nothing is deleted)
# Records: /workspace/logs/gguf/llama_cpp_commit.txt, <build>/olala_build_record.txt (commit, tree, flags, nvcc, arch),
# /workspace/logs/gguf/gguf_verified.txt. Needs the network: GitHub, the Hub, NVIDIA's and Ubuntu's apt mirrors.
# Env: OLALA_WORK (default /workspace), CUDA_HOME (default /usr/local/cuda), GGUF_CUDA_APT_VERSION (default 13-0).
set -u
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=common.sh
. "$here/common.sh"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN GITHUB_TOKEN   # no token of any kind lives on a pod: everything fetched is public
tool=(python3 -I "$here/gguf_tool.py")

dry=0; variant=""; cc=""; arch=""; skip_build=0; skip_download=0
while [ $# -gt 0 ]; do
  case $1 in
    --dry-run) dry=1; shift ;;
    --variant) variant=${2:-}; shift 2 ;;
    --cc) cc=${2:-}; shift 2 ;;
    --cuda-arch) arch=${2:-}; shift 2 ;;
    --skip-build) skip_build=1; shift ;;
    --skip-download) skip_download=1; shift ;;
    *) die "unknown argument $1 (usage: gguf_setup.sh [--dry-run] [--variant V] [--cc X.Y|--cuda-arch A] [--skip-build] [--skip-download])" ;;
  esac
done
variant=${variant:-$("${tool[@]}" get gguf.default)}
name=$("${tool[@]}" get "gguf.files.$variant.name" 2> /dev/null) \
  || die "unknown variant '$variant': $("${tool[@]}" get gguf.files 2> /dev/null | python3 -I -c 'import json,sys; print(" ".join(json.load(sys.stdin)))')"
size=$("${tool[@]}" get "gguf.files.$variant.size")
sha=$("${tool[@]}" get "gguf.files.$variant.sha256")
hf_repo=$("${tool[@]}" get gguf.repo)
hf_rev=$("${tool[@]}" get gguf.revision)
repo=$("${tool[@]}" get llama_cpp.repo)
commit=$("${tool[@]}" get llama_cpp.commit)
tree=$("${tool[@]}" get llama_cpp.tree)
min_cuda=$("${tool[@]}" get llama_cpp.min_cuda_toolkit)
mapfile -t flags < <("${tool[@]}" get llama_cpp.cmake_flags | python3 -I -c 'import json,sys; print("\n".join(json.load(sys.stdin)))')
mapfile -t targets < <("${tool[@]}" get llama_cpp.targets | python3 -I -c 'import json,sys; print("\n".join(json.load(sys.stdin)))')

llama_dir=$OLALA_WORK/llama.cpp-olala
build=$llama_dir/build
glog=$LOG_ROOT/gguf
model_dir=$MODEL_ROOT/olala-gguf
stamp=$(utc_stamp)
log=$glog/setup-$stamp.log
cuda_home=${CUDA_HOME:-/usr/local/cuda}
cuda_dash=${GGUF_CUDA_APT_VERSION:-13-0}
jobs=$(max_jobs)

if [ -z "$arch" ]; then
  if [ -z "$cc" ] && [ $dry -eq 0 ]; then
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | head -n 1 | tr -d ' ')
    [ -n "$cc" ] || die "nvidia-smi gives no compute capability: no GPU in this container (pass --cuda-arch to build anyway)"
  fi
  arch=$("${tool[@]}" arch "${cc:-12.0}") || die "cannot derive the CUDA architecture from '${cc:-12.0}'"
fi
cmake_cfg=(cmake -S "$llama_dir" -B "$build" "${flags[@]}" "-DCMAKE_CUDA_ARCHITECTURES=$arch")
cmake_build=(cmake --build "$build" --target "${targets[@]}" -j "$jobs")

# have_cuda_piece KIND: is the header, library or compiler there?
find_in_cuda() {  # find_in_cuda FILE: the first path under CUDA_HOME (include, targets/x86_64-linux/include, stubs)
  local f
  for f in "$cuda_home/include/$1" "$cuda_home/targets/x86_64-linux/include/$1" "$cuda_home/lib64/stubs/$1" \
    "$cuda_home/targets/x86_64-linux/lib/stubs/$1" "$cuda_home/lib/stubs/$1"; do
    [ -e "$f" ] && { echo "$f"; return 0; }
  done
  return 1
}
missing_packages() {
  local want=()
  command -v cmake > /dev/null 2>&1 || want+=(cmake)
  { command -v g++ > /dev/null 2>&1 && command -v make > /dev/null 2>&1; } || want+=(build-essential)
  command -v git > /dev/null 2>&1 || want+=(git ca-certificates)
  command -v nvcc > /dev/null 2>&1 || want+=("cuda-nvcc-$cuda_dash")
  find_in_cuda cuda_runtime.h > /dev/null || want+=("cuda-cudart-dev-$cuda_dash")
  find_in_cuda cublas_v2.h > /dev/null || want+=("libcublas-dev-$cuda_dash")
  find_in_cuda libcuda.so > /dev/null || want+=("cuda-driver-dev-$cuda_dash")
  echo "${want[*]:-}"
}

if [ $dry -eq 1 ]; then
  echo "gguf_setup.sh dry run: nothing is read from the GPU, nothing is installed, fetched or written"
  echo "== 1 prerequisites"
  echo "+ check: cmake, g++ and make (build-essential), git, nvcc >= $min_cuda (sm_120 needs 12.8), $cuda_home/include/cuda_runtime.h, cublas_v2.h, a libcuda.so stub; install what is missing with apt-get install -y --no-install-recommends (cuda-nvcc-$cuda_dash cuda-cudart-dev-$cuda_dash libcublas-dev-$cuda_dash cuda-driver-dev-$cuda_dash)"
  echo "== 2 fetch the fork by commit and compare it with the pins"
  echo "+ git -C $llama_dir init -q; git -C $llama_dir remote add origin $repo"
  echo "+ git -C $llama_dir fetch --depth 1 origin $commit; git -C $llama_dir checkout -q --detach FETCH_HEAD"
  echo "+ require HEAD $commit and tree $tree"
  echo "== 3 configure (CUDA architecture $arch${cc:+, compute capability $cc}; the dry run assumes 12.0 when none is given)"
  echo "+ ${cmake_cfg[*]} -DCMAKE_CUDA_COMPILER=\$(command -v nvcc)"
  echo "== 4 build"
  if [ $skip_build -eq 1 ]; then echo "(skipped: --skip-build)"; else echo "+ ${cmake_build[*]}"; fi
  echo "== 5 check the binary sees a CUDA device and record the commit"
  echo "+ $build/bin/llama-server --list-devices must list CUDA0; write $glog/llama_cpp_commit.txt and $build/olala_build_record.txt"
  echo "== 6 download $variant: $hf_repo @ $hf_rev ($name, $size bytes; public repository, no token)"
  if [ $skip_download -eq 1 ]; then
    echo "(skipped: --skip-download)"
  else
    echo "+ HF_HUB_DISABLE_XET=1 ${tool[*]} fetch --variant $variant --dest $model_dir"
  fi
  echo "== 7 verify size and sha256 against the pins"
  echo "+ size $size, sha256 $sha (written to $glog/gguf_verified.txt)"
  echo "log: $log"
  echo "GGUF_SETUP_DRY_RUN (nothing was run)"
  exit 0
fi

mkdir -p "$glog" "$MODEL_ROOT"
fail() { echo "GGUF_SETUP_FAILED: $*"; exit 1; }

main() {
  echo "== gguf_setup $(utc_iso): variant $variant, fork $commit, CUDA architecture $arch"
  PATH="$cuda_home/bin:$PATH"

  echo "== 1 prerequisites"
  need_pkgs=$(missing_packages)
  if [ -n "$need_pkgs" ]; then
    echo "missing: $need_pkgs: installing with apt-get (the container runs as root)"
    # shellcheck disable=SC2086  # a list of package names
    { apt-get update -qq && apt-get install -y -qq --no-install-recommends $need_pkgs; } || fail "apt-get could not install: $need_pkgs (install the CUDA toolkit by hand)"
    need_pkgs=$(missing_packages)
    [ -z "$need_pkgs" ] || fail "still missing after apt-get: $need_pkgs"
  else
    echo "nothing to install"
  fi
  nvcc_version=$(nvcc --version 2> /dev/null | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)
  [ -n "$nvcc_version" ] || fail "cannot read the nvcc version"
  [ "$(printf '%s\n%s\n' "$min_cuda" "$nvcc_version" | sort -V | head -n 1)" = "$min_cuda" ] \
    || fail "nvcc $nvcc_version is older than $min_cuda: sm_120 needs CUDA 12.8 or later (set CUDA_HOME to a newer toolkit)"
  nvcc_path=$(command -v nvcc)
  echo "nvcc $nvcc_version at $nvcc_path"

  record=$build/olala_build_record.txt
  flags_text="${flags[*]} -DCMAKE_CUDA_ARCHITECTURES=$arch"
  built=0
  if [ -x "$build/bin/llama-server" ] && [ -f "$record" ] \
    && grep -qx "commit=$commit" "$record" && grep -qx "cuda_arch=$arch" "$record" \
    && grep -qxF "cmake_flags=$flags_text" "$record"; then
    built=1
  fi
  if [ $skip_build -eq 1 ]; then
    echo "== 2 to 5 skipped (--skip-build)"
  elif [ $built -eq 1 ]; then
    echo "== 2 to 4 already built at commit $commit for $arch: nothing to do"
  else
    echo "== 2 fetch the fork by commit and compare it with the pins"
    mkdir -p "$llama_dir"
    git -C "$llama_dir" init -q || fail "git init"
    git -C "$llama_dir" remote add origin "$repo" 2> /dev/null || git -C "$llama_dir" remote set-url origin "$repo"
    git -C "$llama_dir" fetch --depth 1 origin "$commit" || fail "git fetch of commit $commit from $repo failed"
    git -C "$llama_dir" checkout -q --detach FETCH_HEAD || fail "git checkout"
    head=$(git -C "$llama_dir" rev-parse HEAD)
    [ "$head" = "$commit" ] || fail "HEAD is $head, the pinned commit is $commit"
    got_tree=$(git -C "$llama_dir" rev-parse 'HEAD^{tree}')
    [ "$got_tree" = "$tree" ] || fail "the tree of $commit is $got_tree, the pinned tree is $tree"
    echo "commit $head tree $got_tree"

    echo "== 3 configure"
    echo "+ ${cmake_cfg[*]} -DCMAKE_CUDA_COMPILER=$nvcc_path"
    "${cmake_cfg[@]}" "-DCMAKE_CUDA_COMPILER=$nvcc_path" || fail "cmake configure failed (the log above says which CUDA piece is missing)"

    echo "== 4 build (jobs $jobs: one per CPU of the container, at most one per 4 GB)"
    echo "+ ${cmake_build[*]}"
    "${cmake_build[@]}" || fail "build failed"

    echo "== 5 check the binary sees a CUDA device and record the commit"
    devices=$("$build/bin/llama-server" --list-devices 2>&1) || true
    echo "$devices"
    echo "$devices" | grep -Eq '^ *CUDA[0-9]+:' || fail "llama-server lists no CUDA device (built without CUDA, or the container has no GPU)"
    {
      echo "commit=$commit"; echo "tree=$tree"; echo "repo=$repo"
      echo "cmake_flags=$flags_text"; echo "cuda_arch=$arch"
      echo "nvcc=$nvcc_version ($nvcc_path)"; echo "jobs=$jobs"; echo "built_utc=$(utc_iso)"
    } > "$record"
    echo "$commit" > "$glog/llama_cpp_commit.txt"
  fi
  # an existing build keeps its record: write the commit file for the scripts that read it
  [ ! -f "$record" ] || grep -x "commit=.*" "$record" | head -n 1 | sed 's/^commit=//' > "$glog/llama_cpp_commit.txt"

  if [ $skip_download -eq 1 ]; then
    echo "== 6 download skipped (--skip-download)"
  else
    echo "== 6 download $variant: $hf_repo @ $hf_rev -> $model_dir (public repository, no token)"
    free_gb=$(df -Pk "$OLALA_WORK" | awk 'NR==2{print int($4/1048576)}')
    need_gb=$(( size / 1073741824 + 4 ))
    [ "${free_gb:-0}" -ge "$need_gb" ] || fail "only ${free_gb:-0} GB free on $OLALA_WORK: the file needs about $need_gb GB"
    HF_HUB_DISABLE_XET=1 "${tool[@]}" fetch --variant "$variant" --dest "$model_dir" || fail "download or verification failed (see above)"
    echo "== 7 verify size and sha256 against the pins"
    "${tool[@]}" verify --variant "$variant" --file "$model_dir/$name" || fail "the file does not match the pins"
    echo "file=$name size=$size sha256=$sha revision=$hf_rev verified_utc=$(utc_iso)" >> "$glog/gguf_verified.txt"
  fi
  echo "GGUF_SETUP_DONE"
}

main 2>&1 | tee -a "$log"
exit "${PIPESTATUS[0]}"
