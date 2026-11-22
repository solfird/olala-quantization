"""pod/gguf_setup.sh: build OVH's llama.cpp fork for CUDA and fetch the pinned GGUF. No network, no GPU.

A dry run prints the plan and touches nothing (the stubs of conftest fail any call to curl, nvidia-smi and friends).
A real run is tested against stub tools (git, cmake, nvcc, apt-get, nvidia-smi) in a fake /workspace: it proves the
order of the steps, the exact flags, the pins that are checked and the records that are left.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from pathlib import Path

import pytest
from conftest import RUNPOD, run_script, stub_calls, stub_env

POD = RUNPOD / "pod"
MANIFEST = json.loads((POD / "gguf_files.json").read_text())
COMMIT = MANIFEST["llama_cpp"]["commit"]
TREE = MANIFEST["llama_cpp"]["tree"]
Q5 = MANIFEST["gguf"]["files"]["q5_k_m"]


def dry(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return run_script("pod/gguf_setup.sh", "--dry-run", *args, env=env)


# --- dry run ---------------------------------------------------------------------------------------------


def test_the_dry_run_prints_the_plan_in_order_and_calls_nothing(tmp_path, stub_bin):
    out = dry("--cc", "12.0", env=stub_env(stub_bin, tmp_path))
    assert out.returncode == 0, out.stderr
    assert stub_calls(stub_bin) == ""
    text = out.stdout
    steps = [ln for ln in text.splitlines() if ln.startswith("== ")]
    names = " | ".join(steps).lower()
    order = ["prerequisites", "fetch", "configure", "build", "check", "download", "verify"]
    positions = [names.index(word) for word in order]
    assert positions == sorted(positions), names
    assert "GGUF_SETUP_DRY_RUN" in text


def test_the_build_is_cuda_only_for_blackwell_with_the_pinned_flags_and_one_target():
    text = dry("--cc", "12.0").stdout
    (configure,) = [ln for ln in text.splitlines() if ln.startswith("+ cmake -S")]
    assert "-B /workspace/llama.cpp-olala/build" in configure
    for flag in MANIFEST["llama_cpp"]["cmake_flags"]:
        assert flag in configure
    assert "-DCMAKE_CUDA_ARCHITECTURES=120a-real" in configure  # sm_120: a short build
    (build,) = [ln for ln in text.splitlines() if ln.startswith("+ cmake --build")]
    assert "--target llama-server" in build and re.search(r"-j \d+", build)


def test_the_cuda_architecture_follows_the_gpu_or_the_option():
    assert "-DCMAKE_CUDA_ARCHITECTURES=89-real" in dry("--cc", "8.9").stdout
    assert "-DCMAKE_CUDA_ARCHITECTURES=100a-real" in dry("--cuda-arch", "100a-real").stdout


def test_the_fork_is_fetched_by_commit_and_the_tree_is_checked():
    text = dry("--cc", "12.0").stdout
    assert f"fetch --depth 1 origin {COMMIT}" in text
    assert COMMIT in text and TREE in text
    assert MANIFEST["llama_cpp"]["repo"] in text
    assert "/workspace/llama.cpp-olala" in text


def test_the_toolchain_check_names_nvcc_the_headers_and_the_packages_to_install():
    text = dry("--cc", "12.0").stdout
    for needle in (
        "nvcc",
        "cuda_runtime.h",
        "cublas_v2.h",
        "libcuda.so",
        "cmake",
        "build-essential",
        "git",
    ):
        assert needle in text, needle
    assert (
        "12.8" in text
    )  # the minimum toolkit for sm_120 (ggml/src/ggml-cuda/CMakeLists.txt:28-33)


def test_the_download_is_the_pinned_revision_with_no_token_to_the_gguf_folder_and_is_verified():
    text = dry().stdout
    assert "HF_HUB_DISABLE_XET=1" in text
    assert (
        "gguf_tool.py" in text
        and "fetch --variant q5_k_m --dest /workspace/models/olala-gguf" in text
    )
    assert Q5["sha256"] in text and str(Q5["size"]) in text
    assert MANIFEST["gguf"]["revision"] in text
    assert "token" not in text.lower().replace("no token", "")


def test_the_log_goes_under_workspace_logs_gguf():
    text = dry().stdout
    assert re.search(r"/workspace/logs/gguf/setup-\d{8}T\d{6}Z\.log", text)
    assert "/workspace/logs/gguf/llama_cpp_commit.txt" in text


def test_skip_options_remove_their_steps():
    no_build = dry("--cc", "12.0", "--skip-build").stdout
    assert "cmake --build" not in no_build and "download" in no_build.lower()
    no_dl = dry("--cc", "12.0", "--skip-download").stdout
    assert "gguf_tool.py fetch" not in no_dl and "cmake --build" in no_dl


def test_the_variant_changes_the_file_and_an_unknown_one_is_refused():
    out = dry("--variant", "bf16")
    assert out.returncode == 0 and "13594188096" in out.stdout
    bad = dry("--variant", "q4_k_m")
    assert bad.returncode != 0 and "q5_k_m" in bad.stderr


def test_the_setup_script_has_no_pipe_to_shell_no_sudo_and_no_token_name():
    code = "\n".join(
        ln
        for ln in (POD / "gguf_setup.sh").read_text().splitlines()
        if not ln.lstrip().startswith("#")
    )
    assert not re.search(r"(curl|wget)[^\n]*\|\s*(ba)?sh", code)
    assert "sudo" not in code
    for token in ("HF_TOKEN", "GITHUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        assert all("unset" in ln for ln in code.splitlines() if token in ln), token


# --- a real run against stub tools -------------------------------------------------------------------------


STUB_GIT = r"""#!/bin/bash
echo "git $*" >> "$STUB_LOG"
case "$1" in
  init) mkdir -p "$2/.git" 2>/dev/null || mkdir -p .git ;;
  -C) shift 2; case "$1" in
        rev-parse) case "$2" in
          HEAD) echo "${STUB_GIT_HEAD:-}" ;;
          "HEAD^{tree}") echo "${STUB_GIT_TREE:-}" ;;
        esac ;;
      esac ;;
  rev-parse) case "$2" in
      HEAD) echo "${STUB_GIT_HEAD:-}" ;;
      "HEAD^{tree}") echo "${STUB_GIT_TREE:-}" ;;
    esac ;;
esac
exit 0
"""

STUB_CMAKE = r"""#!/bin/bash
echo "cmake $*" >> "$STUB_LOG"
case "$1" in
  --build)
    mkdir -p "$STUB_BUILD_DIR/bin"
    printf '#!/bin/bash\necho "Available devices:"\necho "  CUDA0: NVIDIA RTX PRO 4500 Blackwell (32000 MiB, 31500 MiB free)"\n' > "$STUB_BUILD_DIR/bin/llama-server"
    chmod +x "$STUB_BUILD_DIR/bin/llama-server"
    ;;
esac
exit 0
"""


@pytest.fixture
def fake(tmp_path):
    work = tmp_path / "ws"
    cuda = tmp_path / "cuda"
    (cuda / "include").mkdir(parents=True)
    (cuda / "lib64" / "stubs").mkdir(parents=True)
    for header in ("cuda_runtime.h", "cublas_v2.h"):
        (cuda / "include" / header).write_text("")
    (cuda / "lib64" / "stubs" / "libcuda.so").write_text("")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "calls.log"
    log.write_text("")
    tools = {
        "git": STUB_GIT,
        "cmake": STUB_CMAKE,
        "nvcc": '#!/bin/sh\necho "Cuda compilation tools, release 13.0, V13.0.88"\n',
        "nvidia-smi": '#!/bin/sh\ncase "$*" in *compute_cap*) echo 12.0;; *) echo "NVIDIA RTX PRO 4500 Blackwell";; esac\n',
        "apt-get": '#!/bin/sh\necho "apt-get $*" >> "$STUB_LOG"\n',
        "g++": "#!/bin/sh\nexit 0\n",
        "make": "#!/bin/sh\nexit 0\n",
    }
    for name, body in tools.items():
        path = bindir / name
        path.write_text(body)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    env = {
        "OLALA_WORK": str(work),
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "CUDA_HOME": str(cuda),
        "STUB_LOG": str(log),
        "STUB_BUILD_DIR": str(work / "llama.cpp-olala" / "build"),
        "STUB_GIT_HEAD": COMMIT,
        "STUB_GIT_TREE": TREE,
        "HF_TOKEN": "must-not-leak",
    }
    return work, env, log


def calls(log: Path) -> list[str]:
    return [ln for ln in log.read_text().splitlines() if ln]


def test_a_real_run_fetches_the_pinned_commit_configures_builds_and_records_the_commit(fake):
    work, env, log = fake
    out = run_script("pod/gguf_setup.sh", "--skip-download", env=env)
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.strip().splitlines()[-1] == "GGUF_SETUP_DONE"
    seen = calls(log)
    kinds = [
        c.split()[0] + (" " + c.split()[1] if c.startswith(("cmake", "git")) else "") for c in seen
    ]
    assert any(f"fetch --depth 1 origin {COMMIT}" in c for c in seen)
    configure = next(c for c in seen if c.startswith("cmake -S"))
    for flag in MANIFEST["llama_cpp"]["cmake_flags"]:
        assert flag in configure
    assert "-DCMAKE_CUDA_ARCHITECTURES=120a-real" in configure  # read from the (stub) GPU
    assert seen.index(configure) < next(
        i for i, c in enumerate(seen) if c.startswith("cmake --build")
    )
    assert "cmake --build" in " ".join(kinds)
    record = work / "logs" / "gguf" / "llama_cpp_commit.txt"
    assert record.read_text().strip() == COMMIT
    build_record = (work / "llama.cpp-olala" / "build" / "olala_build_record.txt").read_text()
    assert f"commit={COMMIT}" in build_record and f"tree={TREE}" in build_record
    assert (
        "cmake_flags=" in build_record
        and "nvcc=" in build_record
        and "cuda_arch=120a-real" in build_record
    )
    assert list((work / "logs" / "gguf").glob("setup-*.log"))
    assert "must-not-leak" not in "".join(
        p.read_text() for p in (work / "logs" / "gguf").glob("setup-*.log")
    )


def test_a_second_run_does_not_rebuild_when_the_recorded_commit_matches(fake):
    _work, env, log = fake
    assert run_script("pod/gguf_setup.sh", "--skip-download", env=env).returncode == 0
    log.write_text("")
    again = run_script("pod/gguf_setup.sh", "--skip-download", env=env)
    assert again.returncode == 0, again.stdout + again.stderr
    assert not any(c.startswith("cmake --build") for c in calls(log))
    assert "already built" in again.stdout


def test_a_commit_that_is_not_the_pinned_one_stops_the_run_before_the_build(fake):
    _work, env, log = fake
    env["STUB_GIT_HEAD"] = "0" * 40
    out = run_script("pod/gguf_setup.sh", "--skip-download", env=env)
    assert out.returncode == 1
    assert out.stdout.strip().splitlines()[-1].startswith("GGUF_SETUP_FAILED")
    assert "commit" in out.stdout
    assert not any(c.startswith("cmake --build") for c in calls(log))


def test_a_tree_that_differs_from_the_pinned_one_stops_the_run(fake):
    _work, env, log = fake
    env["STUB_GIT_TREE"] = "1" * 40
    out = run_script("pod/gguf_setup.sh", "--skip-download", env=env)
    assert out.returncode == 1 and "tree" in out.stdout
    assert not any(c.startswith("cmake --build") for c in calls(log))


def test_missing_cuda_headers_trigger_the_apt_install_of_the_toolkit_pieces(fake):
    _work, env, log = fake
    (Path(env["CUDA_HOME"]) / "include" / "cublas_v2.h").unlink()
    out = run_script("pod/gguf_setup.sh", "--skip-download", env=env)
    installs = [c for c in calls(log) if c.startswith("apt-get install")]
    assert installs, out.stdout
    assert "libcublas-dev-13-0" in installs[0]
    assert "build-essential" not in installs[0]  # g++ and make exist (stubs): only what is missing


def test_nvcc_older_than_12_8_is_refused_before_any_build(fake):
    _work, env, log = fake
    nvcc = Path(env["PATH"].split(":")[0]) / "nvcc"
    nvcc.write_text('#!/bin/sh\necho "Cuda compilation tools, release 12.4, V12.4.131"\n')
    out = run_script("pod/gguf_setup.sh", "--skip-download", env=env)
    assert out.returncode == 1 and "12.8" in out.stdout
    assert not any(c.startswith("cmake --build") for c in calls(log))


def test_a_binary_that_does_not_see_a_cuda_device_fails_the_check(fake):
    _work, env, _log = fake
    stub_cmake = Path(env["PATH"].split(":")[0]) / "cmake"
    stub_cmake.write_text(
        STUB_CMAKE.replace(
            'echo "  CUDA0: NVIDIA RTX PRO 4500 Blackwell (32000 MiB, 31500 MiB free)"',
            'echo "  (none)"',
        )
    )
    out = run_script("pod/gguf_setup.sh", "--skip-download", env=env)
    assert out.returncode == 1 and "CUDA device" in out.stdout
