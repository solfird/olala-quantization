"""pod/serve_gguf.sh and the llama-server case of pod/stop_server.sh: dry runs, a stub llama-server, no GPU.

The stub `llama-server` records the arguments it received and sleeps, so the tests prove what the process really
got, that it runs in its own process group with the pid file stop_server.sh reads, and that the command file
carries what run_accuracy.sh and the provenance need (GGUF path, sha256, llama.cpp commit).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import time

import pytest
from conftest import RUNPOD, run_script, stub_calls, stub_env

POD = RUNPOD / "pod"
MANIFEST = json.loads((POD / "gguf_files.json").read_text())
COMMIT = MANIFEST["llama_cpp"]["commit"]
Q5 = MANIFEST["gguf"]["files"]["q5_k_m"]
MODEL = f"/workspace/models/olala-gguf/{Q5['name']}"
SERVER = "/workspace/llama.cpp-olala/build/bin/llama-server"
STAMP = r"\d{8}T\d{6}Z"

# The command of the default start: 12 slots of 32,768 tokens.
DEFAULT_FLAGS = (
    f"-m {MODEL} --alias olala -ngl 99 -fa on -np 12 -c {12 * 32768} -fit off --jinja "
    "--min-p 0 --samplers 'temperature;top_k;top_p' --ctx-checkpoints 2 --cache-ram 0 --timeout 14400 "
    "--threads-http 128 --no-webui --host 127.0.0.1 --port 8000"
)


def dry(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return run_script("pod/serve_gguf.sh", "--dry-run", *args, env=env)


def command_line(out: subprocess.CompletedProcess[str]) -> str:
    assert out.returncode == 0, out.stderr
    (line,) = [ln for ln in out.stdout.splitlines() if ln.startswith("command: ")]
    return line.removeprefix("command: ")


# --- dry run -----------------------------------------------------------------------------------------


def test_the_default_command_serves_the_q5_k_m_file_as_olala_on_port_8000():
    line = command_line(dry())
    assert line.startswith(f"{SERVER} ")
    assert shlex.split(line)[1:] == shlex.split(DEFAULT_FLAGS)


def test_the_sampler_chain_applies_temperature_first_like_vllm_and_min_p_is_off():
    """vllm/v1/sample/sampler.py:340-360 applies the temperature, then top-k and top-p; llama.cpp's own chain
    ends with the temperature and adds min-p 0.05 (common/common.h:263-271, 234): both would differ from our
    vLLM arms, so the server is told (--samplers, --min-p 0)."""
    flags = shlex.split(command_line(dry()))
    assert flags[flags.index("--samplers") + 1] == "temperature;top_k;top_p"
    assert flags[flags.index("--min-p") + 1] == "0"


def test_llama_default_samplers_restores_the_engines_own_chain():
    flags = shlex.split(command_line(dry("--llama-default-samplers")))
    assert "--samplers" not in flags and "--min-p" not in flags


def test_np_and_ctx_are_options_and_the_context_follows_np():
    flags = shlex.split(command_line(dry("--np", "8")))
    assert flags[flags.index("-np") + 1] == "8"
    assert flags[flags.index("-c") + 1] == str(8 * 32768)
    flags = shlex.split(command_line(dry("--np", "4", "--ctx", "262144")))
    assert flags[flags.index("-c") + 1] == "262144"


def test_a_context_that_leaves_a_slot_below_the_harness_maximum_is_refused():
    out = dry("--np", "12", "--ctx", "65536")
    assert out.returncode != 0
    assert "32688" in out.stderr and "slot" in out.stderr
    assert dry("--np", "12", "--ctx", "65536", "--allow-small-slots").returncode == 0


def test_the_kv_cache_stays_f16_and_no_quantized_kv_flag_is_sent():
    """The study compares weight quantizations: a quantized KV cache would be a second change."""
    line = command_line(dry())
    assert "-ctk" not in line and "-ctv" not in line and "--cache-type" not in line


def test_no_reasoning_flag_is_sent_so_the_extraction_default_holds_and_effort_comes_from_the_request():
    """common/common.h:654: reasoning_format defaults to DEEPSEEK (reasoning_content split from content); the
    harness sends reasoning_effort per request in chat_template_kwargs (tools/server/server-common.cpp:1356-1360)."""
    line = command_line(dry())
    for flag in (
        "--reasoning-format",
        "--reasoning-effort",
        "--reasoning ",
        "--reasoning-budget",
        "-rea ",
    ):
        assert flag not in line + " "


def test_the_dry_run_names_the_log_the_cmd_the_pid_file_the_port_and_the_memory_plan():
    out = dry()
    assert re.search(rf"^log=/workspace/logs/gguf/server-{STAMP}\.log$", out.stdout, re.M)
    assert re.search(rf"^cmd=/workspace/logs/gguf/server-{STAMP}\.cmd$", out.stdout, re.M)
    assert "pidfile=/workspace/run/server.pid" in out.stdout
    assert "port=8000" in out.stdout
    assert "slot_ctx=32768" in out.stdout
    assert "estimate_mib=" in out.stdout and "assumed" in out.stdout


def test_extra_flags_after_double_dash_are_appended_last():
    line = command_line(dry("--", "--no-warmup"))
    assert line.endswith("--port 8000 --no-warmup")


def test_the_variant_selects_the_file_and_an_unknown_one_lists_the_known_ones():
    line = command_line(dry("--variant", "bf16"))
    assert "Olala-v1.0-mini-7A1B-bf16.gguf" in line
    out = dry("--variant", "q4_k_m")
    assert out.returncode != 0 and "q5_k_m" in out.stderr and "bf16" in out.stderr


def test_a_dry_run_touches_no_network_and_no_gpu_tool(tmp_path, stub_bin):
    out = dry(env=stub_env(stub_bin, tmp_path))
    assert out.returncode == 0, out.stderr
    assert stub_calls(stub_bin) == ""


def test_a_memory_plan_that_cannot_fit_the_card_is_refused_before_any_start():
    out = dry("--np", "24", "--gpu-mem-mib", "32000")
    assert out.returncode != 0 and "memory" in out.stderr.lower()
    assert dry("--np", "12", "--gpu-mem-mib", "32000").returncode == 0
    assert dry("--np", "24", "--gpu-mem-mib", "32000", "--skip-memory-check").returncode == 0


# --- a real start with a stub llama-server -------------------------------------------------------------


@pytest.fixture
def pod(tmp_path):
    work = tmp_path / "ws"
    model_dir = work / "models" / "olala-gguf"
    model_dir.mkdir(parents=True)
    with (model_dir / Q5["name"]).open("wb") as handle:
        handle.truncate(
            Q5["size"]
        )  # a sparse file of the pinned size: the start checks the size, not the content
    server = work / "llama.cpp-olala" / "build" / "bin" / "llama-server"
    server.parent.mkdir(parents=True)
    args_file = tmp_path / "llama-args.txt"
    server.write_text(
        "#!/bin/bash\n"
        f'printf "%s\\n" "$@" > "{args_file}"\n'
        'echo "ggml_cuda_init: found 1 CUDA devices"\n'
        "exec sleep 300\n"
    )
    server.chmod(server.stat().st_mode | stat.S_IEXEC)
    record = work / "logs" / "gguf"
    record.mkdir(parents=True)
    (record / "llama_cpp_commit.txt").write_text(f"{COMMIT}\n")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    smi = bindir / "nvidia-smi"
    smi.write_text('#!/bin/sh\necho "GPU, 1 MiB, 2 MiB, 3 %, 4 %, 5 W, 6"\n')
    smi.chmod(smi.stat().st_mode | stat.S_IEXEC)
    env = {
        "OLALA_WORK": str(work),
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "STOP_SERVER_EXPECT": "sleep",  # the stub execs sleep; the llama-server name has its own test below
    }
    yield work, env, args_file
    run_script("pod/stop_server.sh", env=env)


def test_serve_starts_the_stub_in_its_own_group_records_everything_and_stop_server_ends_it(pod):
    work, env, args_file = pod
    out = run_script("pod/serve_gguf.sh", "--no-snapshot", env=env)
    assert out.returncode == 0, out.stdout + out.stderr
    pid = int((work / "run" / "server.pid").read_text())
    if shutil.which("setsid"):  # macOS has none; the pods (Linux) do
        pgid = int(
            subprocess.run(
                ["ps", "-o", "pgid=", "-p", str(pid)], capture_output=True, text=True, check=True
            ).stdout
        )
        assert pgid == pid  # its own process group: stop_server.sh signals the group
    for _ in range(50):
        if args_file.exists():
            break
        time.sleep(0.1)
    got = args_file.read_text().splitlines()
    model = str(work / "models" / "olala-gguf" / Q5["name"])
    assert got[:2] == ["-m", model]
    assert got[got.index("-np") + 1] == "12" and got[got.index("--alias") + 1] == "olala"
    assert got[-4:] == ["--host", "127.0.0.1", "--port", "8000"]
    logs = list((work / "logs" / "gguf").glob("server-*.log"))
    cmds = list((work / "logs" / "gguf").glob("server-*.cmd"))
    assert len(logs) == 1 and len(cmds) == 1
    cmd = cmds[0].read_text()
    # what run_accuracy.sh and the provenance read
    assert "arm=gguf-q5_k_m" in cmd
    assert f"llama_cpp_commit={COMMIT}" in cmd
    assert f"gguf_file={model}" in cmd
    assert f"gguf_sha256_pinned={Q5['sha256']}" in cmd
    assert f"gguf_size={Q5['size']}" in cmd
    assert f"hf_revision={MANIFEST['gguf']['revision']}" in cmd
    assert re.search(r"^command: .*llama-server -m \S+ --alias olala", cmd, re.M)
    assert (work / "run" / "server.arm").read_text().strip() == "gguf-q5_k_m"
    assert (work / "run" / "server.log").read_text().strip() == str(logs[0])
    # one server at a time
    again = run_script("pod/serve_gguf.sh", "--no-snapshot", env=env)
    assert again.returncode != 0 and "already running" in again.stderr
    stop = run_script("pod/stop_server.sh", env=env)
    assert stop.returncode == 0, stop.stdout + stop.stderr
    assert "stopped server" in stop.stdout
    assert not (work / "run" / "server.pid").exists()
    with pytest.raises(ProcessLookupError):
        for _ in range(50):
            os.kill(pid, 0)
            time.sleep(0.1)


def test_serve_refuses_without_the_binary_the_model_or_the_commit_record(pod):
    work, env, _ = pod
    model = work / "models" / "olala-gguf" / Q5["name"]
    model.unlink()
    out = run_script("pod/serve_gguf.sh", "--no-snapshot", env=env)
    assert out.returncode != 0 and "gguf_setup.sh" in out.stderr
    with model.open("wb") as handle:
        handle.truncate(Q5["size"])
    (work / "logs" / "gguf" / "llama_cpp_commit.txt").unlink()
    out = run_script("pod/serve_gguf.sh", "--no-snapshot", env=env)
    assert out.returncode != 0 and "commit" in out.stderr
    assert not (work / "run" / "server.pid").exists()


def test_serve_refuses_a_file_that_does_not_match_the_pinned_size(pod):
    """A truncated download must not be served: the start checks the size (gguf_setup.sh checked the sha256)."""
    work, env, _ = pod
    model = work / "models" / "olala-gguf" / Q5["name"]
    model.write_bytes(b"gguf stub")  # 9 bytes instead of 5,990,551,040
    out = run_script("pod/serve_gguf.sh", "--no-snapshot", env=env)
    assert out.returncode != 0 and "size" in out.stderr
    assert not (work / "run" / "server.pid").exists()


def test_serve_starts_the_gpu_snapshot_in_the_gguf_log_folder(pod):
    work, env, _ = pod
    assert run_script("pod/serve_gguf.sh", env=env).returncode == 0
    assert (work / "run" / "gpu_snapshot.pid").exists()
    time.sleep(1.0)
    assert list((work / "logs" / "gguf").glob("gpu-*.log"))
    assert run_script("pod/stop_server.sh", env=env).returncode == 0
    assert not (work / "run" / "gpu_snapshot.pid").exists()


# --- stop_server.sh knows llama-server -------------------------------------------------------------------


def test_stop_server_accepts_a_llama_server_process_and_still_refuses_a_stranger(tmp_path):
    (tmp_path / "run").mkdir()
    env = {"OLALA_WORK": str(tmp_path)}
    stranger = subprocess.Popen(
        ["sleep", "300"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    server = subprocess.Popen(
        ["bash", "-c", "exec -a llama-server sleep 300"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(0.4)
        (tmp_path / "run" / "server.pid").write_text(f"{stranger.pid}\n")
        refused = run_script("pod/stop_server.sh", env=env)
        assert refused.returncode != 0 and "not the server" in refused.stderr
        os.kill(stranger.pid, 0)  # nothing was signalled
        (tmp_path / "run" / "server.pid").write_text(f"{server.pid}\n")
        ended = run_script("pod/stop_server.sh", env=env)
        assert ended.returncode == 0, ended.stderr
        assert server.wait(timeout=20) is not None
    finally:
        for proc in (stranger, server):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def test_the_pipeline_never_kills_by_text_in_the_new_scripts():
    for name in ("serve_gguf.sh", "gguf_setup.sh", "smoke_gguf.sh", "gguf_provenance.sh"):
        code = "\n".join(
            ln for ln in (POD / name).read_text().splitlines() if not ln.lstrip().startswith("#")
        )
        assert not re.search(r"\b(pkill|pgrep|killall)\b", code), name


def test_serve_gguf_is_data_driven_from_the_manifest_not_from_hard_coded_hashes():
    text = (POD / "serve_gguf.sh").read_text()
    assert Q5["sha256"] not in text and COMMIT not in text
    assert "gguf_files.json" in text or "gguf_tool.py" in text


def test_every_harness_connection_gets_an_http_thread():
    """tools/server/server-http.cpp:150-152: the default pool is max(np + 4, cores - 1) threads and a non-streaming
    request holds one for its whole generation; run_accuracy.sh keeps up to 96 requests in flight, the surplus
    must queue inside the server's task queue, not in the kernel's accept queue."""
    flags = shlex.split(command_line(dry()))
    assert int(flags[flags.index("--threads-http") + 1]) >= 96
