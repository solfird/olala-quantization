"""pod/setup_evals_env.sh and pod/run_accuracy.sh: the pod-side accuracy runner, without a GPU or a network.

`uv` is replaced by a stub that records its working directory, arguments and the environment the scripts
set, and plays the harness (it writes run_info.json for run_tasks.py). The server is a tiny local HTTP
server answering /health. No pod, no Hugging Face, no NLTK download happens here.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from conftest import RUNPOD, run_script

POD = RUNPOD / "pod"
SYSTEM_PATH = "/usr/bin:/bin"

FAKE_UV = r"""#!/bin/bash
# stand-in for uv: record, then play the part of sync / prefetch / pytest / run_tasks.py
echo "cwd=$PWD args=$* UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-} HF_HOME=${HF_HOME:-} HF_TOKEN=${HF_TOKEN:-}" >> "$FAKE_UV_LOG"
case " $* " in
  *" sync "*)
    [ "${FAKE_UV_FAIL:-}" != sync ] || { echo "sync boom" >&2; exit 1; }
    echo "Resolved 120 packages"
    ;;
  *prefetch_evals_data.py*)
    [ "${FAKE_UV_FAIL:-}" != prefetch ] || { echo "prefetch boom" >&2; exit 1; }
    echo "mgsm 2750 ok"
    ;;
  *run_tasks.py*)
    out=""
    prev=""
    for a in "$@"; do [ "$prev" != "--out" ] || out=$a; prev=$a; done
    mkdir -p "$out"
    printf '{"status": "%s", "n_samples": 7, "request_errors": 1, "truncations": 2}\n' "${FAKE_RUN_STATUS:-complete}" > "$out/run_info.json"
    echo "run_tasks stdout line"
    exit "${FAKE_UV_RC:-0}"
    ;;
  *" pytest "*)
    [ "${FAKE_UV_FAIL:-}" != pytest ] || { echo "1 failed, 117 passed"; exit 1; }
    echo "118 passed, 12 deselected in 8.8s"
    ;;
esac
exit 0
"""


class HealthServer:
    """Answers GET /health with the given status."""

    def __init__(self, status: int = 200) -> None:
        self.status = status

        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                self.send_response(outer.status if self.path == "/health" else 404)
                self.end_headers()

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def health():
    made: list[HealthServer] = []

    def make(status: int = 200) -> HealthServer:
        made.append(HealthServer(status))
        return made[-1]

    yield make
    for s in made:
        s.close()


class Pod:
    """A fake /workspace: bundle with an evals folder, a stub uv, a server record."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.work = root / "workspace"
        self.evals = self.work / "bundle" / "evals"
        self.evals.mkdir(parents=True)
        (self.evals / "pyproject.toml").write_text("[project]\nname='x'\n")
        (self.evals / "uv.lock").write_text("version = 1\n")
        self.home = root / "home"
        self.home.mkdir()
        self.bin = root / "bin"
        self.bin.mkdir()
        self.uv_log = root / "uv-calls.log"
        self.uv_log.write_text("")
        self.fake_uv = self.bin / "uv"
        self.fake_uv.write_text(FAKE_UV)
        self.fake_uv.chmod(self.fake_uv.stat().st_mode | stat.S_IEXEC)
        (self.bin / "python3").symlink_to(sys.executable)  # a real python3, without the Mac's uv

    def env(self, **extra: str) -> dict[str, str]:
        return {
            "PATH": f"{self.bin}:{SYSTEM_PATH}",
            "HOME": str(self.home),
            "OLALA_WORK": str(self.work),
            "OLALA_EVALS_DIR": str(self.evals),
            "FAKE_UV_LOG": str(self.uv_log),
            "HF_TOKEN": "must-not-reach-uv",
            **extra,
        }

    def uv_calls(self) -> list[str]:
        """The recorded uv calls, `uv --version` left out."""
        return [
            ln for ln in self.uv_log.read_text().splitlines() if ln and " args=--version " not in ln
        ]

    def serve_record(self, arm: str = "fp8pc", *, info: dict | None = None) -> Path:
        """What serve_olala.sh leaves: pid file dir, the log name, the .cmd file, the served dir."""
        run = self.work / "run"
        logs = self.work / "logs" / arm
        run.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)
        served = self.work / "models" / f"olala-{arm}"
        served.mkdir(parents=True, exist_ok=True)
        log = logs / "server-20261121T100000Z.log"
        log.write_text("INFO server log\n")
        (logs / "server-20261121T100000Z.cmd").write_text(
            f"utc=2026-11-21T10:00:00Z\narm={arm}\ncommand: vllm serve {served} --trust-remote-code "
            "--served-model-name olala --port 8000\n"
        )
        (run / "server.log").write_text(f"{log}\n")
        (run / "server.arm").write_text(f"{arm}\n")
        (run / "server.pid").write_text("4242\n")
        (logs / "smoke.json").write_text("{}")
        if info is not None:
            (served / "quantization_info.json").write_text(json.dumps(info))
        return served


def args_of(call: str) -> str:
    """The argument part of a recorded uv call (the working directory also names the test)."""
    return call.split(" args=", 1)[1].split(" UV_PROJECT_ENVIRONMENT=", 1)[0]


@pytest.fixture
def pod(tmp_path: Path) -> Pod:
    return Pod(tmp_path)


# --- setup_evals_env.sh --------------------------------------------------------------------------


def setup(pod: Pod, *args: str, **extra: str) -> subprocess.CompletedProcess[str]:
    return run_script("pod/setup_evals_env.sh", *args, env=pod.env(**extra), stdin="")


def setup_logs(pod: Pod) -> list[Path]:
    return sorted((pod.work / "logs" / "setup").glob("evals-env-*.log"))


def test_setup_evals_env_syncs_the_locked_environment_prefetches_the_data_then_runs_the_tests(
    pod: Pod,
):
    out = setup(pod)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "EVALS_ENV_DONE" in out.stdout
    calls = pod.uv_calls()
    kinds = [
        "sync"
        if args_of(c).startswith("sync ")
        else "prefetch"
        if "prefetch_evals_data.py" in args_of(c)
        else "pytest"
        if args_of(c).endswith(" pytest -q")
        else "?"
        for c in calls
    ]
    assert kinds == ["sync", "prefetch", "pytest"]
    for call in calls:
        assert f"cwd={pod.evals.resolve()}" in call or f"cwd={pod.evals}" in call
        assert f"UV_PROJECT_ENVIRONMENT={pod.work}/venv-evals" in call
        assert f"HF_HOME={pod.work}/hf_cache" in call
        assert "HF_TOKEN=must" not in call  # no token of any kind on a pod
    assert args_of(calls[0]) == "sync --frozen --python 3.12"  # the committed uv.lock, Python 3.12
    assert all("--frozen --python 3.12" in args_of(c) for c in calls)
    assert "118 passed" in out.stdout  # the test result is printed


def test_setup_evals_env_logs_every_step_to_a_utc_stamped_file_under_logs_setup(pod: Pod):
    out = setup(pod)
    assert out.returncode == 0, out.stderr
    (log,) = setup_logs(pod)
    assert re.fullmatch(r"evals-env-\d{8}T\d{6}Z\.log", log.name)
    text = log.read_text()
    for needle in ("uv", "sync", "prefetch", "pytest", "118 passed", "EVALS_ENV_DONE"):
        assert needle in text, needle
    assert "Resolved 120 packages" in text  # the tools' own output is in the log too


def test_setup_evals_env_is_idempotent(pod: Pod):
    assert setup(pod).returncode == 0
    time.sleep(1.1)  # the log name carries the second
    assert setup(pod).returncode == 0
    assert len(setup_logs(pod)) == 2
    assert len(pod.uv_calls()) == 6  # sync, prefetch, pytest, twice


@pytest.mark.parametrize(
    ("failing", "ran_after"),
    [("sync", False), ("prefetch", False), ("pytest", True)],
)
def test_setup_evals_env_stops_at_the_first_failing_step_and_exits_non_zero(
    pod: Pod, failing: str, ran_after: bool
):
    out = setup(pod, FAKE_UV_FAIL=failing)
    assert out.returncode != 0
    assert "EVALS_ENV_FAILED" in out.stdout and "EVALS_ENV_DONE" not in out.stdout
    calls = " ".join(args_of(c) for c in pod.uv_calls())
    assert (" pytest -q" in calls) is (ran_after or failing == "pytest")
    assert ("prefetch_evals_data.py" in calls) is (failing != "sync")  # nothing after a failed step
    assert "EVALS_ENV_FAILED" in setup_logs(pod)[0].read_text()


def test_setup_evals_env_installs_a_pinned_uv_from_pypi_when_it_is_missing(pod: Pod):
    """No curl | sh: third-party installers are untrusted. A stub python3 plays pip."""
    pip_log = pod.root / "pip-calls.log"
    stub = pod.bin / "python3"
    stub.unlink()
    stub.write_text(
        "#!/bin/bash\n"
        f'echo "$*" >> "{pip_log}"\n'
        f'mkdir -p "{pod.home}/.local/bin" && cp "{pod.root}/uv-src" "{pod.home}/.local/bin/uv"\n'
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    (pod.root / "uv-src").write_text(FAKE_UV)
    (pod.root / "uv-src").chmod(0o755)
    pod.fake_uv.unlink()  # uv is not on the PATH now
    out = setup(pod)
    assert out.returncode == 0, out.stdout + out.stderr
    (call,) = pip_log.read_text().splitlines()
    assert "-m pip install" in call and re.search(r"uv==\d+\.\d+\.\d+", call), call
    assert "EVALS_ENV_DONE" in out.stdout


def test_setup_evals_env_does_not_install_anything_when_uv_is_present(pod: Pod):
    stub = pod.bin / "python3"
    stub.unlink()
    stub.write_text(f'#!/bin/bash\necho "$*" >> "{pod.root}/pip-calls.log"\n')
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    assert setup(pod).returncode == 0
    assert not (pod.root / "pip-calls.log").exists()


def test_setup_evals_env_refuses_a_bundle_without_the_evals_lock(pod: Pod):
    (pod.evals / "uv.lock").unlink()
    out = setup(pod)
    assert out.returncode != 0
    assert "uv.lock" in out.stdout + out.stderr
    assert pod.uv_calls() == []


def test_setup_evals_env_dry_run_prints_every_step_and_touches_nothing(pod: Pod):
    out = setup(pod, "--dry-run")
    assert out.returncode == 0, out.stderr
    text = out.stdout
    for needle in (
        "uv sync --frozen --python 3.12",
        "prefetch_evals_data.py",
        "MGSM",
        "HumanEval+",
        "MBPP+",
        "GSM8K",
        "IFEval",
        "AIME 2025",
        "NLTK",
        "pytest",
        "logs/setup/evals-env-",
    ):
        assert needle in text, needle
    assert pod.uv_calls() == []
    assert not (pod.work / "logs").exists() and not (pod.work / "hf_cache").exists()


def test_setup_evals_env_never_pipes_a_downloaded_script_into_a_shell():
    code = "\n".join(
        ln
        for ln in (POD / "setup_evals_env.sh").read_text().splitlines()
        if not ln.lstrip().startswith("#")
    )
    assert not re.search(r"\|\s*(ba)?sh\b", code)
    assert "curl" not in code and "wget" not in code


# --- prefetch_evals_data.py ----------------------------------------------------------------------


def test_the_prefetch_helper_flags_a_dataset_with_an_unexpected_item_count():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "prefetch_evals_data", POD / "prefetch_evals_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ok = dict(module.EXPECTED_ITEMS)
    assert module.count_problems(ok) == []
    assert set(ok) == {"mgsm", "humaneval_plus", "mbpp_plus", "gsm8k", "ifeval", "aime2025"}
    short = {**ok, "ifeval": 100}
    (problem,) = module.count_problems(short)
    assert "ifeval" in problem and "100" in problem and "541" in problem
    missing = {k: v for k, v in ok.items() if k != "aime2025"}
    assert any("aime2025" in p for p in module.count_problems(missing))


# --- run_accuracy.sh -----------------------------------------------------------------------------


def accuracy(
    pod: Pod, *args: str, url: str = "http://127.0.0.1:9", **extra: str
) -> subprocess.CompletedProcess[str]:
    return run_script(
        "pod/run_accuracy.sh", *args, env=pod.env(OLALA_BASE_URL=url, **extra), stdin=""
    )


def command_of(out: subprocess.CompletedProcess[str]) -> str:
    assert out.returncode == 0, out.stdout + out.stderr
    (line,) = [ln for ln in out.stdout.splitlines() if ln.startswith("command: ")]
    return line.removeprefix("command: ")


def test_run_accuracy_dry_run_prints_the_exact_command_with_the_study_defaults(pod: Pod):
    out = accuracy(pod, "b0-pilot", "--dry-run")  # no server needed: a dry run contacts nothing
    expected = (
        f"cd {pod.evals} && uv run --frozen --python 3.12 python run_tasks.py "
        "--tasks mgsm humaneval_plus mbpp_plus ifeval aime2025 "
        "--base-url http://127.0.0.1:9/v1 --model olala "
        f"--out {pod.work}/results/raw/b0-pilot --seed 0 --sandbox local "
        "--reasoning-effort medium --max-connections 96 --n-items mgsm=550"
    )
    assert command_of(out) == expected
    assert f"log={pod.work}/logs/b0-pilot/run_tasks.log" in out.stdout
    assert not (pod.work / "results").exists() and not (pod.work / "logs").exists()
    assert pod.uv_calls() == []


def test_run_accuracy_options_replace_the_defaults_and_extra_arguments_come_last(pod: Pod):
    out = accuracy(
        pod,
        "arm-fp8",
        "--effort",
        "high",
        "--tasks",
        "mgsm,aime2025",
        "--n-items",
        "mgsm=110",
        "--n-items",
        "aime2025=10",
        "--max-connections",
        "32",
        "--dry-run",
        "--",
        "--epochs",
        "aime2025=2",
        "--seed",
        "3",
    )
    cmd = command_of(out)
    assert "--tasks mgsm aime2025 --base-url" in cmd
    assert "--reasoning-effort high --max-connections 32" in cmd
    assert "--n-items mgsm=110 --n-items aime2025=10" in cmd
    assert "mgsm=550" not in cmd  # a given --n-items replaces the default
    assert cmd.endswith("--epochs aime2025=2 --seed 3")  # last, so it overrides the default seed 0


def test_run_accuracy_passes_max_connections_as_a_bare_value_so_the_aime_default_applies(
    pod: Pod,
):
    # run_tasks.py resolves --max-connections per task: an explicit TASK=N beats the built-in
    # aime2025=24, which beats a bare N. A bare 96 therefore never reaches AIME, as long as the
    # script does not turn it into aime2025=96.
    cmd = command_of(accuracy(pod, "b0", "--dry-run"))
    assert re.findall(r"--max-connections (\S+)", cmd) == ["96"]
    assert "aime2025=" not in cmd
    cmd = command_of(accuracy(pod, "b0", "--max-connections", "48", "--dry-run"))
    assert re.findall(r"--max-connections (\S+)", cmd) == ["48"]


def test_run_accuracy_lets_an_explicit_aime_value_through_the_extra_arguments(pod: Pod):
    cmd = command_of(accuracy(pod, "b0", "--dry-run", "--", "--max-connections", "aime2025=12"))
    assert re.findall(r"--max-connections (\S+)", cmd) == ["96", "aime2025=12"]
    assert cmd.endswith("--max-connections aime2025=12")


def test_run_accuracy_usage_documents_the_bare_value_and_the_aime_default():
    header = "\n".join(
        line for line in (POD / "run_accuracy.sh").read_text().splitlines() if line.startswith("#")
    )
    flat = " ".join(header.split())
    assert "bare" in flat
    assert "aime2025" in flat
    assert "24" in flat
    assert "aime2025=N" in flat  # how to override it


def test_run_accuracy_with_items_from_does_not_add_the_default_subset(pod: Pod):
    cmd = command_of(
        accuracy(
            pod, "arm-fp8", "--items-from", "/workspace/results/raw/b0/items.json", "--dry-run"
        )
    )
    assert "--items-from /workspace/results/raw/b0/items.json" in cmd
    assert "--n-items" not in cmd


@pytest.mark.parametrize(
    "args",
    [
        ["bad name"],
        ["../escape"],
        ["ok", "--effort", "extreme"],
        ["ok", "--max-connections", "many"],
        ["ok", "--max-connections", "0"],
        ["ok", "--unknown"],
        [],
    ],
)
def test_run_accuracy_refuses_bad_arguments_before_doing_anything(pod: Pod, args: list[str]):
    out = accuracy(pod, *args, "--dry-run")
    assert out.returncode != 0
    assert pod.uv_calls() == []


def test_run_accuracy_refuses_to_overwrite_an_existing_run_directory(pod: Pod, health):
    pod.serve_record()
    (pod.work / "results" / "raw" / "b0").mkdir(parents=True)
    out = accuracy(pod, "b0", url=health().url)
    assert out.returncode != 0
    assert "already exists" in out.stderr
    assert pod.uv_calls() == []


@pytest.mark.parametrize("status", [503, None])
def test_run_accuracy_refuses_when_the_server_is_not_healthy(pod: Pod, health, status):
    pod.serve_record()
    url = health(status).url if status else "http://127.0.0.1:9"
    out = accuracy(pod, "b0", url=url)
    assert out.returncode != 0
    assert "/health" in out.stderr and "smoke" in out.stderr
    assert pod.uv_calls() == []
    assert not (pod.work / "results" / "raw" / "b0").exists()


def test_run_accuracy_runs_the_harness_logs_it_and_copies_the_provenance(pod: Pod, health):
    info = {"arm": "fp8pc", "method": "online", "weights_gb": 7.1}
    served = pod.serve_record(info=info)
    out = accuracy(pod, "fp8-main", url=health().url)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "RUN_ACCURACY_DONE" in out.stdout
    (call,) = pod.uv_calls()
    assert f"cwd={pod.evals}" in call and "run_tasks.py" in call and "--sandbox local" in call
    assert "HF_TOKEN=must" not in call
    run = pod.work / "results" / "raw" / "fp8-main"
    log = pod.work / "logs" / "fp8-main" / "run_tasks.log"
    assert "run_tasks stdout line" in log.read_text()
    assert json.loads((run / "quantization_info.json").read_text()) == info
    assert (run / "server.cmd").read_text().startswith("utc=")
    prov = (run / "server_provenance.txt").read_text()
    assert "server_log_name=server-20261121T100000Z.log" in prov
    assert "arm=fp8pc" in prov and f"served_dir={served}" in prov
    assert "quantization_info=quantization_info.json" in prov
    assert (
        "request_errors=1" in out.stdout and "truncations=2" in out.stdout
    )  # the totals are shown


def test_run_accuracy_without_a_quantization_info_file_records_that_and_still_runs(
    pod: Pod, health
):
    pod.serve_record(arm="bf16")  # the bf16 directory has no quantization_info.json
    out = accuracy(pod, "b0", url=health().url)
    assert out.returncode == 0, out.stderr
    run = pod.work / "results" / "raw" / "b0"
    assert not (run / "quantization_info.json").exists()
    assert "quantization_info=none" in (run / "server_provenance.txt").read_text()


def test_run_accuracy_still_runs_when_no_serve_record_exists_but_says_so(pod: Pod, health):
    out = accuracy(pod, "b0", url=health().url)  # server started by hand: no /workspace/run files
    assert out.returncode == 0, out.stdout + out.stderr
    assert "no serve_olala.sh record" in (out.stdout + out.stderr)
    assert "server_log_name=none" in (pod.work / "results/raw/b0/server_provenance.txt").read_text()


def test_run_accuracy_exits_non_zero_when_the_harness_fails(pod: Pod, health):
    pod.serve_record()
    out = accuracy(pod, "b0", url=health().url, FAKE_UV_RC="1", FAKE_RUN_STATUS="incomplete")
    assert out.returncode != 0
    assert "RUN_ACCURACY_FAILED" in out.stdout and "RUN_ACCURACY_DONE" not in out.stdout


def test_run_accuracy_exits_non_zero_on_a_non_zero_harness_exit_even_if_run_info_says_complete(
    pod: Pod, health
):
    pod.serve_record()
    out = accuracy(pod, "b0", url=health().url, FAKE_UV_RC="3")
    assert out.returncode != 0
    assert "exited with 3" in out.stdout


def test_run_accuracy_exits_non_zero_when_a_task_ended_in_error_even_if_the_exit_code_was_zero(
    pod: Pod, health
):
    pod.serve_record()
    out = accuracy(pod, "b0", url=health().url, FAKE_RUN_STATUS="incomplete")
    assert out.returncode != 0
    assert "incomplete" in out.stdout + out.stderr and "RUN_ACCURACY_FAILED" in out.stdout


def test_run_accuracy_survives_the_detach_of_pod_ssh_sh(pod: Pod, health):
    """pod_ssh.sh detach: cd /workspace && (nohup bash -c CMD > LOG 2>&1 < /dev/null &)."""
    pod.serve_record()
    log = pod.root / "detached.log"
    cmd = f"bash {POD}/run_accuracy.sh b0-detached"
    full = f"cd {pod.work} && (nohup bash -c {cmd!r} > {log} 2>&1 < /dev/null &)"
    env = {**os.environ, **pod.env(OLALA_BASE_URL=health().url)}
    subprocess.run(["bash", "-c", full], env=env, check=True, stdin=subprocess.DEVNULL)
    for _ in range(100):
        if log.exists() and "RUN_ACCURACY_DONE" in log.read_text():
            break
        time.sleep(0.2)
    else:
        pytest.fail(f"no end marker in the detached log: {log.read_text() if log.exists() else ''}")
    assert (pod.work / "results" / "raw" / "b0-detached" / "run_info.json").exists()


def test_run_accuracy_unsets_every_hub_token_and_never_calls_pod_scripts_by_text():
    text = (POD / "run_accuracy.sh").read_text() + (POD / "common.sh").read_text()
    assert re.search(r"unset HF_TOKEN\b.*HUGGING_FACE_HUB_TOKEN", text)
    code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    assert not re.search(r"\b(pkill|pgrep|killall)\b", code)
