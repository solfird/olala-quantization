"""The gguf mode of the accuracy pipeline: gguf_arm.sh, gguf_provenance.sh, run_accuracy.sh's provenance, arm_pipeline.sh.

No GPU and no network: gguf_arm.sh is run from a copy of pod/ in which gguf_setup.sh, serve_gguf.sh and smoke_gguf.sh
are replaced by stubs that record their arguments; gguf_provenance.sh reads a fake workspace; run_accuracy.sh runs
against the stub uv of test_evals_pod and a local /health server.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import RUNPOD, run_script
from test_evals_pod import HealthServer, Pod

POD = RUNPOD / "pod"
MANIFEST = json.loads((POD / "gguf_files.json").read_text())
COMMIT = MANIFEST["llama_cpp"]["commit"]
TREE = MANIFEST["llama_cpp"]["tree"]
Q5 = MANIFEST["gguf"]["files"]["q5_k_m"]
REVISION = MANIFEST["gguf"]["revision"]


def write_records(work: Path, variant: str = "q5_k_m") -> Path:
    """What gguf_setup.sh and serve_gguf.sh leave in a workspace."""
    entry = MANIFEST["gguf"]["files"][variant]
    glog = work / "logs" / "gguf"
    run = work / "run"
    glog.mkdir(parents=True, exist_ok=True)
    run.mkdir(parents=True, exist_ok=True)
    model = work / "models" / "olala-gguf" / entry["name"]
    log = glog / "server-20261121T100000Z.log"
    log.write_text("llama_kv_cache: CUDA0 KV buffer size = 16128.00 MiB\n")
    (glog / "server-20261121T100000Z.cmd").write_text(
        "utc=2026-11-21T10:00:00Z\n"
        f"arm=gguf-{variant}\n"
        f"variant={variant}\n"
        f"llama_cpp_commit={COMMIT}\n"
        "llama_cpp_repo=https://github.com/OVHai-LLM/llama.cpp.git\n"
        f"gguf_file={model}\n"
        f"gguf_size={entry['size']}\n"
        f"gguf_sha256_pinned={entry['sha256']}\n"
        f"gguf_sha256_verified_by_setup={entry['sha256']}\n"
        "hf_repo=OVHaiLLM/Olala-v1.0-mini-7A1B-GGUF\n"
        f"hf_revision={REVISION}\n"
        "np=12\nctx=393216\nslot_ctx=32768\n"
        f"command: {work}/llama.cpp-olala/build/bin/llama-server -m {model} --alias olala -ngl 99 -fa on "
        "-np 12 -c 393216 --host 127.0.0.1 --port 8000\n"
    )
    (run / "server.log").write_text(f"{log}\n")
    (run / "server.arm").write_text(f"gguf-{variant}\n")
    (run / "server.pid").write_text("4242\n")
    (glog / "llama_cpp_commit.txt").write_text(f"{COMMIT}\n")
    build = work / "llama.cpp-olala" / "build"
    build.mkdir(parents=True, exist_ok=True)
    (build / "olala_build_record.txt").write_text(
        f"commit={COMMIT}\ntree={TREE}\nrepo=https://github.com/OVHai-LLM/llama.cpp.git\n"
        "cmake_flags=-DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a-real\n"
        "cuda_arch=120a-real\nnvcc=13.0 (/usr/local/cuda/bin/nvcc)\n"
    )
    (glog / "smoke.json").write_text(
        json.dumps(
            {
                "verdict": "pass",
                "health_seconds": 41.0,
                "tokens_per_second": {
                    "generation": 88.5,
                    "prompt": 1900.0,
                    "source": "server timings",
                },
                "log_facts": ["CUDA0 KV buffer size = 16128.00 MiB"],
                "parallel": {"requests": 12, "aggregate_tokens_per_second": 640.0},
            }
        )
    )
    return model


# --- gguf_provenance.sh ----------------------------------------------------------------------------------------------


def test_the_provenance_records_commit_file_hash_flags_and_the_smoke_summary(tmp_path):
    work = tmp_path / "ws"
    model = write_records(work)
    out = run_script("pod/gguf_provenance.sh", env={"OLALA_WORK": str(work)})
    assert out.returncode == 0, out.stderr
    text = out.stdout
    assert f"llama_cpp_commit={COMMIT}" in text and f"llama_cpp_tree={TREE}" in text
    assert f"gguf_file={model}" in text
    assert f"gguf_sha256={Q5['sha256']}" in text and f"gguf_size={Q5['size']}" in text
    assert f"hf_revision={REVISION}" in text
    assert "server_command=" in text and "-np 12 -c 393216" in text
    assert "cuda_arch=120a-real" in text and "nvcc=13.0" in text
    assert "smoke_verdict=pass" in text and "smoke_generation_tokens_per_second=88.5" in text
    assert "smoke_parallel_aggregate_tokens_per_second=640.0" in text


def test_the_provenance_refuses_a_workspace_without_a_gguf_server_record(tmp_path):
    out = run_script("pod/gguf_provenance.sh", env={"OLALA_WORK": str(tmp_path)})
    assert out.returncode != 0 and "server" in out.stderr


def test_the_provenance_marks_an_unverified_hash(tmp_path):
    work = tmp_path / "ws"
    write_records(work)
    cmd = next((work / "logs" / "gguf").glob("server-*.cmd"))
    cmd.write_text(
        cmd.read_text().replace(
            f"gguf_sha256_verified_by_setup={Q5['sha256']}",
            "gguf_sha256_verified_by_setup=unverified",
        )
    )
    text = run_script("pod/gguf_provenance.sh", env={"OLALA_WORK": str(work)}).stdout
    assert "gguf_sha256=unverified (pinned: " + Q5["sha256"] in text


# --- run_accuracy.sh records the GGUF server ---------------------------------------------------------------------------


@pytest.fixture
def health():
    made: list[HealthServer] = []

    def make(status: int = 200) -> HealthServer:
        made.append(HealthServer(status))
        return made[-1]

    yield make
    for s in made:
        s.close()


def test_run_accuracy_copies_the_gguf_server_provenance_next_to_the_results(tmp_path, health):
    pod = Pod(tmp_path)
    model = write_records(pod.work)
    out = run_script(
        "pod/run_accuracy.sh",
        "gguf-q5",
        env=pod.env(OLALA_BASE_URL=health().url),
        stdin="",
    )
    assert out.returncode == 0, out.stdout + out.stderr
    run = pod.work / "results" / "raw" / "gguf-q5"
    assert (run / "server.cmd").read_text().startswith("utc=")
    prov = (run / "server_provenance.txt").read_text()
    assert "arm=gguf-q5_k_m" in prov
    assert f"served_dir={model}" in prov  # the file llama-server was given with -m
    assert f"llama_cpp_commit={COMMIT}" in prov
    assert f"gguf_sha256_pinned={Q5['sha256']}" in prov
    assert f"hf_revision={REVISION}" in prov
    assert "quantization_info=none" in prov


# --- gguf_arm.sh (setup, serve, smoke gate, provenance) from a copy of pod/ with stub steps -----------------------------


@pytest.fixture
def arm(tmp_path):
    pod_copy = tmp_path / "pod"
    shutil.copytree(POD, pod_copy)
    calls = tmp_path / "calls.log"
    calls.write_text("")
    work = tmp_path / "ws"
    write_records(work)
    for name, rc_var in (
        ("gguf_setup.sh", "SETUP_RC"),
        ("serve_gguf.sh", "SERVE_RC"),
        ("smoke_gguf.sh", "SMOKE_RC"),
    ):
        (pod_copy / name).write_text(
            f'#!/bin/bash\necho "{name} $*" >> "{calls}"\n'
            f'echo "stub {name}"\nexit "${{{rc_var}:-0}}"\n'
        )
    env = {"OLALA_WORK": str(work)}

    def run(*args: str, **extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(pod_copy / "gguf_arm.sh"), *args],
            capture_output=True,
            text=True,
            env={**os.environ, **env, **extra},
            check=False,
        )

    return run, calls, work


def logged(calls: Path) -> list[str]:
    return [ln for ln in calls.read_text().splitlines() if ln]


def test_gguf_arm_runs_setup_serve_smoke_then_writes_the_provenance_in_that_order(arm, tmp_path):
    run, calls, work = arm
    prov = tmp_path / "prov.txt"
    prov.write_text("run=G mode=gguf:q5_k_m\n")
    out = run("G", "q5_k_m", str(prov))
    assert out.returncode == 0, out.stdout + out.stderr
    steps = logged(calls)
    assert [s.split()[0] for s in steps] == ["gguf_setup.sh", "serve_gguf.sh", "smoke_gguf.sh"]
    assert "--variant q5_k_m" in steps[0] and "--variant q5_k_m" in steps[1]
    assert "--timeout 900" in steps[2]
    text = prov.read_text()
    assert text.startswith("run=G mode=gguf:q5_k_m")  # appended, not overwritten
    assert f"llama_cpp_commit={COMMIT}" in text and f"gguf_sha256={Q5['sha256']}" in text
    assert (work / "logs" / "G.smoke.txt").exists()


def test_gguf_arm_passes_serve_options_from_the_environment(arm, tmp_path):
    run, calls, _ = arm
    prov = tmp_path / "prov.txt"
    out = run("G", "q5_k_m", str(prov), GGUF_SERVE_ARGS="--np 16 --ctx 524288")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "--variant q5_k_m --np 16 --ctx 524288" in logged(calls)[1]


def test_a_smoke_with_warnings_goes_on_and_a_failed_smoke_stops(arm, tmp_path):
    run, calls, _ = arm
    prov = tmp_path / "prov.txt"
    assert run("G", "q5_k_m", str(prov), SMOKE_RC="2").returncode == 0
    calls.write_text("")
    out = run("G", "q5_k_m", str(prov), SMOKE_RC="1")
    assert out.returncode == 1 and "smoke gate" in out.stdout + out.stderr


def test_a_failed_setup_or_serve_stops_before_the_next_step(arm, tmp_path):
    run, calls, _ = arm
    prov = tmp_path / "prov.txt"
    assert run("G", "q5_k_m", str(prov), SETUP_RC="1").returncode == 1
    assert [s.split()[0] for s in logged(calls)] == ["gguf_setup.sh"]
    calls.write_text("")
    assert run("G", "q5_k_m", str(prov), SERVE_RC="1").returncode == 1
    assert [s.split()[0] for s in logged(calls)] == ["gguf_setup.sh", "serve_gguf.sh"]


def test_gguf_arm_accepts_a_file_name_and_refuses_an_unknown_variant(arm, tmp_path):
    run, calls, _ = arm
    prov = tmp_path / "prov.txt"
    assert run("G", f"/somewhere/{Q5['name']}", str(prov)).returncode == 0
    assert "--variant q5_k_m" in logged(calls)[0]
    calls.write_text("")
    bad = run("G", "q4_k_m", str(prov))
    assert bad.returncode != 0 and "q5_k_m" in bad.stdout + bad.stderr and logged(calls) == []


# --- arm_pipeline.sh: the new mode is wired, the old modes still run the generic path -------------------------------------


@pytest.fixture
def pipeline(tmp_path):
    """A copy of pod/ whose /workspace is a temp folder and whose steps are stubs that record their arguments."""
    work = tmp_path / "ws"
    pod_copy = work / "bundle" / "pod"  # B=/workspace/bundle in job_common.sh
    shutil.copytree(POD, pod_copy)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "nvidia-smi").write_text("#!/bin/sh\necho GPU, 580.1\n")
    (bindir / "nvidia-smi").chmod(0o755)
    for sub in ("logs/setup", "logs/fp8pc", "results"):
        (work / sub).mkdir(parents=True)
    (work / "logs" / "setup" / "evals-env-1.log").write_text("EVALS_ENV_DONE\n")
    (work / "logs" / "fp8pc" / "server-1.log").write_text("INFO Quantizing 3 layers of types: x\n")
    for name in ("arm_pipeline.sh", "job_common.sh"):  # the scripts hard-code /workspace
        path = pod_copy / name
        path.write_text(path.read_text().replace("/workspace", str(work)))
    calls = tmp_path / "calls.log"
    calls.write_text("")
    stubs = {
        "stop_server.sh": "",
        "serve_olala.sh": "",
        "smoke.sh": "",
        "run_accuracy.sh": 'echo "RUN_ACCURACY_DONE"\n',
        "gguf_arm.sh": 'echo "gguf_arm_ran $1 $2" >> "$3"\n',
    }
    for name, extra in stubs.items():
        rc_var = {"gguf_arm.sh": "ARM_RC", "run_accuracy.sh": "ACC_RC"}.get(name)
        exit_line = f'exit "${{{rc_var}:-0}}"\n' if rc_var else "exit 0\n"
        (pod_copy / name).write_text(
            f'#!/bin/bash\necho "{name} $*" >> "{calls}"\n{extra}{exit_line}'
        )

    def run(*args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(pod_copy / "arm_pipeline.sh"), *args],
            capture_output=True,
            text=True,
            env={**os.environ, **env, "PATH": f"{bindir}:{os.environ['PATH']}"},
            check=False,
            timeout=60,
        )

    return run, calls, work


def test_the_gguf_mode_runs_gguf_arm_then_the_accuracy_run_and_never_the_vllm_smoke(pipeline):
    run, calls, work = pipeline
    out = run("G5", "gguf:q5_k_m", "--tasks", "mgsm")
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.strip().splitlines()[-1] == "ARM_PIPELINE_DONE"
    steps = logged(calls)
    names = [s.split()[0] for s in steps]
    assert names == ["stop_server.sh", "gguf_arm.sh", "run_accuracy.sh"]
    assert steps[1] == f"gguf_arm.sh G5 q5_k_m {work}/results/provenance-G5.txt"
    assert steps[2] == "run_accuracy.sh G5 --effort medium --tasks mgsm"
    assert "gguf_arm_ran G5 q5_k_m" in (work / "results" / "provenance-G5.txt").read_text()


def test_a_failed_gguf_gate_stops_the_pipeline_before_the_accuracy_run(pipeline):
    run, calls, _ = pipeline
    out = run("G5", "gguf:q5_k_m", ARM_RC="1")
    assert out.returncode == 1
    assert out.stdout.strip().splitlines()[-1] == "ARM_PIPELINE_FAILED"
    assert "run_accuracy.sh" not in " ".join(logged(calls))


def test_the_online_mode_still_serves_runs_the_generic_smoke_and_then_the_accuracy_run(pipeline):
    run, calls, work = pipeline
    out = run("F1", "online:fp8pc")
    assert out.returncode == 0, out.stdout + out.stderr
    steps = logged(calls)
    assert [s.split()[0] for s in steps] == [
        "stop_server.sh",
        "serve_olala.sh",
        "smoke.sh",
        "run_accuracy.sh",
    ]
    assert steps[1] == "serve_olala.sh fp8pc" and steps[2].startswith(
        "smoke.sh fp8pc --timeout 900"
    )
    assert "Quantizing 3 layers" in (work / "results" / "provenance-F1.txt").read_text()


def test_an_unknown_mode_is_still_refused(pipeline):
    run, _calls, _ = pipeline
    out = run("X", "magic:1")
    assert out.returncode == 1 and "ARM_PIPELINE_FAILED" in out.stdout
