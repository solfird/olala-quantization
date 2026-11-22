"""smoke.sh / smoke.py: fail fast on a server log with an error, check one French chat completion, flag fallbacks."""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from conftest import run_script

GOOD_LOG = """\
INFO 11-20 10:00:01 [utils.py:300] vLLM API server version 0.33.0
(EngineCore pid=88) INFO 11-20 10:01:10 [base_loader.py:151] Quantizing 112 layers of types: mixer_proj: 36 (from linear: fp8_per_channel_static); mlp.experts: 32 (from moe: fp8_per_channel_static)
(EngineCore pid=88) INFO 11-20 10:01:40 [base_loader.py:160] Online quantization as part of model loading took 21.30 seconds
INFO 11-20 10:03:05 [api_server.py:2000] Starting vLLM API server on http://127.0.0.1:8000
"""

TRACEBACK_LOG = """\
INFO 11-20 10:00:01 [utils.py:300] vLLM API server version 0.33.0
(EngineCore pid=88) ERROR 11-20 10:01:12 [core.py:100] EngineCore failed to start.
(EngineCore pid=88) Traceback (most recent call last):
(EngineCore pid=88)   File "/vllm-workspace/vllm/engine/core.py", line 90, in run
(EngineCore pid=88)     raise ValueError("nvfp4_per_token online quantization requires a Blackwell (SM100) GPU.")
(EngineCore pid=88) ValueError: nvfp4_per_token online quantization requires a Blackwell (SM100) GPU.
"""

SLOW_LOG = (
    GOOD_LOG
    + """\
WARNING 11-20 10:02:00 [fallback.py:20]
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! OLALA SLOW PATH: TPA factor KV cache disabled
!!   reason : JIT build of tpa_factor_decode.cu failed: nvcc fatal : Unsupported gpu architecture 'compute_120'
!!   impact : dense paged KV cache: 2.5x more KV memory
"""
)

OK_ANSWER = {
    "choices": [
        {
            "finish_reason": "stop",
            "message": {
                "content": "17 multiplié par 23 font 391.",
                "reasoning": "Je calcule 17 x 23 = 17 x 20 + 17 x 3 = 340 + 51 = 391.",
            },
        }
    ],
    "usage": {"prompt_tokens": 60, "completion_tokens": 90},
}


class FakeServer:
    """A tiny OpenAI-style server: /health after `healthy_after` seconds, one canned completion."""

    def __init__(self, answer: dict | None, healthy_after: float = 0.0) -> None:
        self.answer = answer
        self.start = time.time()
        self.healthy_after = healthy_after
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:  # silence
                pass

            def do_GET(self) -> None:
                ok = self.path == "/health" and time.time() - outer.start >= outer.healthy_after
                self.send_response(200 if ok else 503)
                self.end_headers()

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "body": body})
                data = json.dumps(outer.answer).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()


@pytest.fixture
def server():
    made: list[FakeServer] = []

    def make(answer: dict | None = OK_ANSWER, healthy_after: float = 0.0) -> FakeServer:
        made.append(FakeServer(answer, healthy_after))
        return made[-1]

    yield make
    for s in made:
        s.close()


@pytest.fixture
def work(tmp_path):
    (tmp_path / "run").mkdir()
    (tmp_path / "logs" / "fp8pc").mkdir(parents=True)
    return tmp_path


def smoke(work: Path, url: str, log_text: str, *args: str, arm: str = "fp8pc", timeout: str = "20"):
    (work / "logs" / arm).mkdir(parents=True, exist_ok=True)
    (work / "logs" / arm / "server-20261120T100000Z.log").write_text(log_text)
    env = {"OLALA_WORK": str(work), "OLALA_BASE_URL": url, "SMOKE_POLL_SECONDS": "0.2"}
    return run_script("pod/smoke.sh", arm, "--timeout", timeout, *args, env=env)


def report(work: Path, arm: str = "fp8pc") -> dict:
    return json.loads((work / "logs" / arm / "smoke.json").read_text())


def test_a_good_server_passes_and_prints_the_quantized_layers_summary(work, server):
    s = server()
    out = smoke(work, s.url, GOOD_LOG)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "Quantizing 112 layers of types" in out.stdout
    r = report(work)
    assert r["verdict"] == "pass" and r["failures"] == [] and r["warnings"] == []
    assert r["answer_nonempty"] and r["reasoning_nonempty"]
    assert r["expected_in_answer"] is True
    assert r["quantized_layers"].startswith("(EngineCore pid=88) INFO") is True
    assert "Quantizing 112 layers" in r["quantized_layers"]
    assert r["arm"] == "fp8pc" and r["health_seconds"] >= 0
    assert list(
        (work / "logs" / "fp8pc").glob("smoke-*.json")
    )  # a dated copy: failed runs are kept


def test_the_request_is_a_french_maths_question_with_medium_reasoning_effort(work, server):
    s = server()
    assert smoke(work, s.url, GOOD_LOG).returncode == 0
    (req,) = s.requests
    body = req["body"]
    assert req["path"] == "/v1/chat/completions"
    assert body["model"] == "olala"
    assert body["reasoning_effort"] == "medium"
    assert body["chat_template_kwargs"] == {"reasoning_effort": "medium"}
    prompt = body["messages"][-1]["content"]
    assert "17" in prompt and "23" in prompt and "Combien" in prompt
    assert body["temperature"] == 0.8 and body["top_p"] == 0.9 and body["top_k"] == 40


def test_a_traceback_in_the_log_fails_fast_and_prints_the_last_error_line(work, server):
    s = server(healthy_after=999)  # never healthy: only the log can end the wait
    started = time.time()
    out = smoke(work, s.url, TRACEBACK_LOG, timeout="120")
    assert out.returncode == 1
    assert time.time() - started < 20  # not the 120 s timeout
    assert (
        "ValueError: nvfp4_per_token online quantization requires a Blackwell (SM100) GPU."
        in out.stdout + out.stderr
    )
    r = report(work)
    assert r["verdict"] == "fail" and "SM100" in r["error_line"]
    assert (
        r["failures"] and s.requests == []
    )  # no request was sent to a server that failed to start


def test_an_error_level_line_alone_is_enough_to_fail(work, server):
    s = server(healthy_after=999)
    out = smoke(
        work,
        s.url,
        "INFO ok\n(Worker) ERROR 11-20 10:00:00 [x.py:1] CUDA error: an illegal memory access\n",
    )
    assert out.returncode == 1
    assert "illegal memory access" in out.stdout + out.stderr


def test_olala_slow_path_banners_and_jit_failures_are_reported_as_a_warning(work, server):
    s = server()
    out = smoke(work, s.url, SLOW_LOG)
    assert out.returncode == 2, out.stdout + out.stderr  # works, but not on the fast path
    r = report(work)
    assert r["verdict"] == "warn"
    assert any("SLOW PATH" in line for line in r["slow_path"])
    assert any("JIT build of tpa_factor_decode.cu failed" in line for line in r["jit_failures"])
    assert "SLOW PATH" in out.stdout


def test_a_fallback_and_a_nan_in_the_log_are_flagged(work, server):
    s = server()
    out = smoke(
        work, s.url, GOOD_LOG + "WARNING falling back to the fallback kernel\nloss was NaN here\n"
    )
    assert out.returncode == 2
    r = report(work)
    assert r["slow_path"] and r["nan"]


def test_an_empty_reasoning_field_fails(work, server):
    answer = json.loads(json.dumps(OK_ANSWER))
    answer["choices"][0]["message"]["reasoning"] = ""
    out = smoke(work, server(answer).url, GOOD_LOG)
    assert out.returncode == 1
    assert report(work)["reasoning_nonempty"] is False


def test_the_older_reasoning_content_field_name_is_accepted(work, server):
    answer = json.loads(json.dumps(OK_ANSWER))
    message = answer["choices"][0]["message"]
    message["reasoning_content"] = message.pop("reasoning")
    assert smoke(work, server(answer).url, GOOD_LOG).returncode == 0


def test_an_empty_final_answer_fails_and_a_length_stop_is_named(work, server):
    answer = json.loads(json.dumps(OK_ANSWER))
    answer["choices"][0]["message"]["content"] = None
    answer["choices"][0]["finish_reason"] = "length"
    out = smoke(work, server(answer).url, GOOD_LOG)
    assert out.returncode == 1
    r = report(work)
    assert r["answer_nonempty"] is False
    assert "length" in " ".join(r["failures"])


def test_a_wrong_answer_is_a_warning_not_a_failure(work, server):
    answer = json.loads(json.dumps(OK_ANSWER))
    answer["choices"][0]["message"]["content"] = "Je ne sais pas."
    out = smoke(work, server(answer).url, GOOD_LOG)
    assert out.returncode == 2
    assert report(work)["expected_in_answer"] is False


def test_a_server_that_never_answers_health_times_out_and_says_so(work, server):
    s = server(healthy_after=999)
    out = smoke(work, s.url, GOOD_LOG, timeout="2")
    assert out.returncode == 1
    assert "timeout" in (out.stdout + out.stderr).lower()
    assert report(work)["verdict"] == "fail"


def test_a_dead_server_process_ends_the_wait_at_once(work, server):
    s = server(healthy_after=999)
    (work / "run" / "server.pid").write_text("999999\n")
    started = time.time()
    out = smoke(work, s.url, GOOD_LOG, timeout="120")
    assert out.returncode == 1 and time.time() - started < 20
    assert "exited" in (out.stdout + out.stderr)


def test_a_running_server_pid_does_not_end_the_wait(work, server):
    s = server(healthy_after=1)
    (work / "run" / "server.pid").write_text(f"{os.getpid()}\n")
    assert smoke(work, s.url, GOOD_LOG).returncode == 0


def test_the_script_reads_the_newest_log_of_the_arm(work, server):
    s = server()
    (work / "logs" / "fp8pc").mkdir(parents=True, exist_ok=True)
    old = work / "logs" / "fp8pc" / "server-20261120T090000Z.log"
    old.write_text(TRACEBACK_LOG)
    os.utime(old, (1, 1))
    assert smoke(work, s.url, GOOD_LOG).returncode == 0  # the newer, good log wins
    assert report(work)["log"].endswith("server-20261120T100000Z.log")
