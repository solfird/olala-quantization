"""Pod-side scripts: serve_olala.sh (arms as data), stop_server.sh, gpu_snapshot.sh, setup_remote.sh, common.sh."""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest
from conftest import RUNPOD, run_script

POD = RUNPOD / "pod"
FORK = RUNPOD.parent / "recon" / "src" / "vllm"
IGNORE = [
    "model.layers.1.mixer_proj",
    "model.layers.7.mixer_proj",
    "model.layers.23.mixer_proj",
    "model.layers.0.mlp.*",
    "model.layers.1.mlp.*",
    "model.layers.5.mlp.*",
    "model.layers.19.mlp.*",
]
IGNORE_JSON = json.dumps({"ignore": IGNORE}, separators=(",", ":"))
BASE = (
    "vllm serve /workspace/models/olala-bf16 --trust-remote-code --served-model-name olala "
    "--mamba-ssm-cache-dtype float32 --reasoning-parser olala --tool-call-parser olala "
    "--enable-auto-tool-choice --max-model-len 32688 --max-num-seqs 256"
)
TAIL = "--host 127.0.0.1 --port 8000"
ARMS = json.loads((POD / "arms.json").read_text())


def serve(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return run_script("pod/serve_olala.sh", "--dry-run", *args, env=env)


def command_line(out: subprocess.CompletedProcess[str]) -> str:
    assert out.returncode == 0, out.stderr
    (line,) = [ln for ln in out.stdout.splitlines() if ln.startswith("command: ")]
    return line.removeprefix("command: ")


def with_ignore(quant: str) -> str:
    return f"{BASE} --quantization {quant} --quantization-config '{IGNORE_JSON}' {TAIL}"


def without_ignore(quant: str) -> str:
    return f"{BASE} --quantization {quant} {TAIL}"


EXPECTED = {
    "bf16": f"{BASE} {TAIL}",
    "fp8pc": with_ignore("fp8_per_channel"),
    "fp8pc_noignore": without_ignore("fp8_per_channel"),
    "fp8pt": with_ignore("fp8_per_tensor"),
    "fp8pt_noignore": without_ignore("fp8_per_tensor"),
    "fp8pb": with_ignore("fp8_per_block"),
    "fp8pb_noignore": without_ignore("fp8_per_block"),
    "mxfp4": with_ignore("mxfp4"),
    "mxfp4_noignore": without_ignore("mxfp4"),
    "nvfp4pt": with_ignore("nvfp4_per_token"),
    "nvfp4pt_noignore": without_ignore("nvfp4_per_token"),
}


@pytest.mark.parametrize(("arm", "expected"), EXPECTED.items())
def test_each_arm_prints_the_exact_command(arm, expected):
    assert command_line(serve(arm)) == expected


def test_the_bf16_command_is_the_model_card_command_plus_the_study_flags():
    line = command_line(serve("bf16"))
    assert line.startswith(
        "vllm serve /workspace/models/olala-bf16 --trust-remote-code --served-model-name olala "
        "--mamba-ssm-cache-dtype float32 --reasoning-parser olala --tool-call-parser olala "
        "--enable-auto-tool-choice --max-model-len 32688 --max-num-seqs 256"
    )
    assert line.endswith(TAIL)


def test_the_ignore_list_is_the_section_c2_list_and_is_valid_json_in_the_command():
    line = command_line(serve("fp8pc"))
    quoted = shlex.split(line)[shlex.split(line).index("--quantization-config") + 1]
    assert json.loads(quoted) == {"ignore": IGNORE}
    assert quoted == IGNORE_JSON  # compact, deterministic


def test_a_checkpoint_directory_is_served_with_no_quantization_flag():
    line = command_line(serve("ckpt", "/workspace/models/olala-fp8-llmc"))
    assert (
        line
        == BASE.replace("/workspace/models/olala-bf16", "/workspace/models/olala-fp8-llmc")
        + " "
        + TAIL
    )
    assert "--quantization" not in line


def test_ckpt_needs_a_directory():
    out = serve("ckpt")
    assert out.returncode != 0 and "DIR" in out.stderr


def test_an_unknown_arm_is_refused_and_the_arms_are_listed():
    out = serve("fp4magic")
    assert out.returncode != 0
    for arm in ("bf16", "fp8pc", "nvfp4pt", "ckpt"):
        assert arm in out.stderr


def test_dry_run_names_the_log_the_pid_file_and_the_port():
    out = serve("fp8pc")
    assert re.search(r"^log=/workspace/logs/fp8pc/server-\d{8}T\d{6}Z\.log$", out.stdout, re.M)
    assert "pidfile=/workspace/run/server.pid" in out.stdout
    assert "port=8000" in out.stdout
    assert re.search(r"^MAX_JOBS=\d+$", out.stdout, re.M)


def test_extra_flags_after_double_dash_are_appended_and_recorded():
    line = command_line(serve("bf16", "--", "--gpu-memory-utilization", "0.85"))
    assert line.endswith(f"{TAIL} --gpu-memory-utilization 0.85")


def test_nvfp4_per_token_is_refused_on_a_gpu_that_is_not_sm100_before_any_start():
    """online/nvfp4.py:98 raises unless current_platform.is_device_capability_family(100)."""
    bad = serve("nvfp4pt", "--cc", "12.0")
    assert bad.returncode != 0
    assert "SM100" in bad.stderr or "sm_100" in bad.stderr
    assert serve("nvfp4pt", "--cc", "10.0").returncode == 0
    assert serve("fp8pc", "--cc", "12.0").returncode == 0


def test_every_flag_of_every_arm_is_a_flag_confirmed_in_the_fork_source():
    confirmed = set(ARMS["confirmed_flags"])
    for arm, expected in EXPECTED.items():
        flags = {tok for tok in shlex.split(expected) if tok.startswith("--")}
        assert flags <= confirmed, (arm, flags - confirmed)
    for flag, where in ARMS["confirmed_flags"].items():
        assert flag.startswith("--") and where, flag


@pytest.mark.skipif(not FORK.exists(), reason="the fork source (recon/src/vllm) is not checked out")
def test_the_quantization_names_exist_in_the_fork_shorthand_table():
    source = (FORK / "vllm" / "config" / "quantization.py").read_text()
    table = source.split("_ONLINE_SHORTHANDS: dict", 1)[1].split("\n}\n", 1)[0]
    names = set(re.findall(r'^    "([a-z0-9_]+)": QuantizationConfigArgs', table, re.M))
    used = {a["quantization"] for a in ARMS["arms"].values() if a.get("quantization")}
    assert used <= names, used - names
    assert {
        "fp8_per_channel",
        "fp8_per_tensor",
        "fp8_per_block",
        "mxfp4",
        "nvfp4_per_token",
    } <= used


@pytest.mark.skipif(not FORK.exists(), reason="the fork source (recon/src/vllm) is not checked out")
def test_the_parsers_and_the_layer_names_exist_in_the_fork():
    assert '"olala"' in (FORK / "vllm" / "reasoning" / "__init__.py").read_text()
    assert '"olala"' in (FORK / "vllm" / "tool_parsers" / "__init__.py").read_text()
    model = (FORK / "vllm" / "model_executor" / "models" / "olala.py").read_text()
    assert 'prefix=f"{prefix}.mixer_proj"' in model and 'prefix=f"{prefix}.mlp"' in model


def test_arms_without_a_known_requirement_say_so_instead_of_guessing():
    for name, arm in ARMS["arms"].items():
        assert arm["description"], name
        assert arm.get("source"), name
    assert ARMS["arms"]["nvfp4pt"]["requires_cc_major"] == 10


# --- stop_server.sh ------------------------------------------------------------------------------


@pytest.fixture
def work(tmp_path):
    (tmp_path / "run").mkdir()
    (tmp_path / "logs" / "bf16").mkdir(parents=True)
    return tmp_path


def spawn_session(argv: list[str]) -> subprocess.Popen:
    """A stand-in for the server: its own session and process group, like serve_olala.sh makes."""
    return subprocess.Popen(
        argv, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def wait_gone(proc: subprocess.Popen, seconds: float = 20) -> bool:
    try:
        proc.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        return False
    return True


def test_stop_server_ends_the_whole_process_group_by_pid_file(work):
    child = spawn_session(["bash", "-c", "sleep 300 & sleep 300; wait"])
    time.sleep(0.5)
    (work / "run" / "server.pid").write_text(f"{child.pid}\n")
    out = run_script(
        "pod/stop_server.sh", env={"OLALA_WORK": str(work), "STOP_SERVER_EXPECT": "bash"}
    )
    assert out.returncode == 0, out.stderr
    assert wait_gone(child)
    assert not (work / "run" / "server.pid").exists()
    assert "stopped" in out.stdout


def test_stop_server_refuses_a_pid_that_is_not_the_server(work):
    child = spawn_session(["sleep", "300"])
    try:
        (work / "run" / "server.pid").write_text(f"{child.pid}\n")
        out = run_script("pod/stop_server.sh", env={"OLALA_WORK": str(work)})  # expects 'vllm'
        assert out.returncode != 0
        assert "not the server" in out.stderr
        assert alive(child.pid)
    finally:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def test_stop_server_cleans_a_stale_pid_file_and_says_nothing_runs_when_there_is_none(work):
    (work / "run" / "server.pid").write_text("999999\n")
    out = run_script("pod/stop_server.sh", env={"OLALA_WORK": str(work)})
    assert out.returncode == 0 and "stale" in out.stdout
    assert not (work / "run" / "server.pid").exists()
    again = run_script("pod/stop_server.sh", env={"OLALA_WORK": str(work)})
    assert again.returncode == 0 and "no server" in again.stdout


def test_stop_server_never_matches_processes_by_text():
    text = (POD / "stop_server.sh").read_text()
    code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    assert not re.search(r"\b(pkill|pgrep|killall)\b", code)


# --- gpu_snapshot.sh ----------------------------------------------------------------------------


@pytest.fixture
def fake_nvidia(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    smi = bindir / "nvidia-smi"
    smi.write_text(
        '#!/bin/sh\necho "NVIDIA RTX PRO 4500 Blackwell, 31000 MiB, 32623 MiB, 87 %, 55 %, 180.5 W, 61"\n'
    )
    smi.chmod(smi.stat().st_mode | stat.S_IEXEC)
    return bindir


def test_gpu_snapshot_once_writes_one_utc_stamped_csv_line(work, fake_nvidia):
    env = {"OLALA_WORK": str(work), "PATH": f"{fake_nvidia}:{os.environ['PATH']}"}
    out = run_script("pod/gpu_snapshot.sh", "bf16", "--once", env=env)
    assert out.returncode == 0, out.stderr
    (log,) = list((work / "logs" / "bf16").glob("gpu-*.log"))
    lines = log.read_text().splitlines()
    assert lines[0].startswith("timestamp_utc,")
    assert re.match(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z, NVIDIA RTX PRO 4500 Blackwell, 31000 MiB", lines[1]
    )


def test_gpu_snapshot_loops_while_the_server_lives_and_stops_when_it_ends(work, fake_nvidia):
    server = spawn_session(["sleep", "300"])
    (work / "run" / "server.pid").write_text(f"{server.pid}\n")
    env = {"OLALA_WORK": str(work), "PATH": f"{fake_nvidia}:{os.environ['PATH']}"}
    snap = subprocess.Popen(
        ["bash", str(POD / "gpu_snapshot.sh"), "bf16", "--interval", "1"],
        env={**os.environ, **env},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(3.5)
        os.killpg(server.pid, signal.SIGKILL)
        server.wait()
        assert wait_gone(snap, 10), "the snapshot loop kept running after the server ended"
    finally:
        if snap.poll() is None:
            snap.kill()
    (log,) = list((work / "logs" / "bf16").glob("gpu-*.log"))
    assert len(log.read_text().splitlines()) >= 3  # header and at least two samples


# --- common.sh: limits of the container, not of the host -------------------------------------------


def common(expr: str, env: dict[str, str]) -> str:
    out = subprocess.run(
        ["bash", "-c", f'. "{POD}/common.sh"; {expr}'],
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        check=False,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def fake_cgroup(root: Path, cpu_max: str, mem: str) -> dict[str, str]:
    (root / "cpu.max").write_text(cpu_max)
    (root / "memory.max").write_text(mem)
    return {"CGROUP_DIR": str(root)}


def test_container_limits_come_from_the_cgroup_not_from_nproc(tmp_path):
    env = fake_cgroup(tmp_path, "400000 100000\n", "34359738368\n")
    assert common("container_cpus", env) == "4"
    assert common("container_mem_gb", env) == "32"
    assert common("max_jobs", env) == "4"  # min(4 cpus, 32 GB / 4 GB per job = 8)


def test_max_jobs_is_bounded_by_memory(tmp_path):
    env = fake_cgroup(tmp_path, "max 100000\n", "17179869184\n")  # 16 GB, cpu unlimited
    assert common("max_jobs", {**env, "OLALA_NPROC": "64"}) == "4"  # 16 GB / 4
    assert common("container_cpus", {**env, "OLALA_NPROC": "64"}) == "64"


# --- setup_remote.sh -----------------------------------------------------------------------------


def test_setup_remote_dry_run_lists_every_recorded_fact_and_the_download():
    out = run_script(
        "pod/setup_remote.sh",
        "--dry-run",
        "--expect-cc",
        "12.0",
        "--image-ref",
        "docker.io/ovhaillm/olala-vllm:ebbadef53",
        "--image-digest",
        "sha256:842ca5b0e9ae6f7166a9d6fcfd79d4f6fe33d9a20d14d55220cc95b11f5c3c34",
    )
    assert out.returncode == 0, out.stderr
    text = out.stdout
    for needle in (
        "nvidia-smi",
        "compute_cap",
        "nproc",
        "cpu.max",
        "memory.max",
        "df -h",
        "nvcc --version",
        "pip list",
        "vllm",
        "torch",
        "transformers",
        "compressed-tensors",
        "sha256:842ca5b0e9ae6f7166a9d6fcfd79d4f6fe33d9a20d14d55220cc95b11f5c3c34",
        "OVHaiLLM/Olala-v1.0-mini-7A1B",
        "/workspace/models/olala-bf16",
        "verify_model.py",
        "HF_HUB_DISABLE_XET=1",
    ):
        assert needle in text, needle
    assert "HF_TOKEN" not in text.replace("unset HF_TOKEN", "")  # no token on the pod, ever


def test_setup_remote_unsets_every_hub_token_before_the_download():
    text = (POD / "setup_remote.sh").read_text()
    assert re.search(r"unset HF_TOKEN\b.*HUGGING_FACE_HUB_TOKEN", text)
    assert "token=False" in text


def run_python(
    script: str, *args: str, stdin: str | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", str(POD / script), *args],
        capture_output=True,
        text=True,
        input=stdin,
        check=False,
    )


def test_the_pod_python_helpers_need_nothing_but_the_standard_library():
    for script in POD.glob("*.py"):
        tree = script.read_text()
        for line in tree.splitlines():
            m = re.match(r"^(?:from|import)\s+([a-zA-Z0-9_]+)", line)
            if m:
                assert m.group(1) in sys.stdlib_module_names or m.group(1) == "__future__", (
                    script,
                    line,
                )
