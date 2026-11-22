"""serve_olala.sh really starts a process (a stub `vllm`), records it, refuses a second one; stop_server.sh ends it."""

from __future__ import annotations

import os
import shlex
import stat
import time

import pytest
from conftest import run_script

BF16_FLAGS = (
    "--trust-remote-code --served-model-name olala --mamba-ssm-cache-dtype float32 "
    "--reasoning-parser olala --tool-call-parser olala --enable-auto-tool-choice "
    "--max-model-len 32688 --max-num-seqs 256"
)


@pytest.fixture
def pod(tmp_path):
    work = tmp_path / "ws"
    (work / "models" / "olala-bf16").mkdir(parents=True)
    (work / "models" / "olala-bf16" / "config.json").write_text("{}")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    args_file = tmp_path / "vllm-args.txt"
    vllm = bindir / "vllm"
    vllm.write_text(
        "#!/bin/bash\n"
        f'printf "%s\\n" "$@" > "{args_file}"\n'
        'echo "INFO 11-20 10:00:00 [stub] vllm stub started with MAX_JOBS=$MAX_JOBS"\n'
        'echo "(EngineCore pid=1) INFO [base_loader.py:152] Quantizing 3 layers of types: stub"\n'
        "exec sleep 300\n"
    )
    smi = bindir / "nvidia-smi"
    smi.write_text('#!/bin/sh\necho "GPU, 1 MiB, 2 MiB, 3 %, 4 %, 5 W, 6"\n')
    for f in (vllm, smi):
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    env = {
        "OLALA_WORK": str(work),
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "STOP_SERVER_EXPECT": "sleep",
    }
    yield work, env, args_file
    run_script("pod/stop_server.sh", env=env)


def test_serve_starts_records_refuses_a_second_server_and_stop_ends_it(pod):
    work, env, args_file = pod
    out = run_script("pod/serve_olala.sh", "--no-snapshot", "fp8pc", env=env)
    assert out.returncode == 0, out.stdout + out.stderr
    pid = int((work / "run" / "server.pid").read_text())
    os.kill(pid, 0)
    for _ in range(50):
        if args_file.exists() and (list((work / "logs" / "fp8pc").glob("server-*.log"))):
            log = next((work / "logs" / "fp8pc").glob("server-*.log"))
            if "Quantizing 3 layers" in log.read_text():
                break
        time.sleep(0.1)
    # the arguments the process really received are the exact arguments of the command
    got = args_file.read_text().splitlines()
    expected = shlex.split(BF16_FLAGS)
    assert got[0] == "serve" and got[1] == "/workspace/models/olala-bf16"
    assert got[2 : 2 + len(expected)] == expected
    assert got[2 + len(expected) : 4 + len(expected)] == ["--quantization", "fp8_per_channel"]
    assert got[4 + len(expected)] == "--quantization-config"
    assert got[5 + len(expected)].startswith('{"ignore":["model.layers.1.mixer_proj"')
    assert got[-4:] == ["--host", "127.0.0.1", "--port", "8000"]
    # records: command file, arm, log pointer
    cmd = next((work / "logs" / "fp8pc").glob("server-*.cmd")).read_text()
    assert "arm=fp8pc" in cmd and "command: vllm serve" in cmd and "MAX_JOBS=" in cmd
    assert (work / "run" / "server.arm").read_text().strip() == "fp8pc"
    # one server at a time
    again = run_script("pod/serve_olala.sh", "--no-snapshot", "bf16", env=env)
    assert again.returncode != 0 and "already running" in again.stderr
    # stop by pid file
    stop = run_script("pod/stop_server.sh", env=env)
    assert stop.returncode == 0, stop.stderr
    assert "stopped server" in stop.stdout
    with pytest.raises(ProcessLookupError):
        for _ in range(50):
            os.kill(pid, 0)
            time.sleep(0.1)
    assert not (work / "run" / "server.pid").exists()


def test_serve_refuses_to_start_without_the_bf16_weights(pod):
    work, env, _ = pod
    (work / "models" / "olala-bf16" / "config.json").unlink()
    out = run_script("pod/serve_olala.sh", "--no-snapshot", "bf16", env=env)
    assert out.returncode != 0 and "pod_setup.sh" in out.stderr
    assert not (work / "run" / "server.pid").exists()


def test_serve_starts_the_gpu_snapshot_with_the_server(pod):
    work, env, _ = pod
    assert run_script("pod/serve_olala.sh", "bf16", env=env).returncode == 0
    assert (work / "run" / "gpu_snapshot.pid").exists()
    time.sleep(1.0)
    assert list((work / "logs" / "bf16").glob("gpu-*.log"))
    assert run_script("pod/stop_server.sh", env=env).returncode == 0
    assert not (work / "run" / "gpu_snapshot.pid").exists()


def test_a_checkpoint_directory_must_hold_a_config(pod, tmp_path):
    _, env, _ = pod
    empty = tmp_path / "ckpt"
    empty.mkdir()
    out = run_script("pod/serve_olala.sh", "--no-snapshot", "ckpt", str(empty), env=env)
    assert out.returncode != 0 and "config.json" in out.stderr
