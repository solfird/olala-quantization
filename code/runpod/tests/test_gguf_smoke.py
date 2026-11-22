"""pod/smoke_gguf.sh and smoke_gguf.py against a fake llama-server (health, chat completion, /v1/models, log).

The fake answers like OVH's fork documents: `message.reasoning_content` (the analysis channel), `message.content` (the
final channel), `timings` with `predicted_per_second`. The smoke test must send exactly what the harness sends
(temperature, top_p, top_k and `chat_template_kwargs.reasoning_effort`, no top-level reasoning_effort, no max_tokens),
refuse a reply whose `content` carries the reasoning trace, and report the shape and the speed.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from conftest import run_script

GOOD_LOG = """\
llama_model_loader: loaded meta data with 52 key-value pairs and 989 tensors from /workspace/models/olala-gguf/x.gguf
load_tensors: offloaded 37/37 layers to GPU
load_tensors:        CUDA0 model buffer size =  5713.00 MiB
llama_kv_cache:      CUDA0 KV buffer size = 16128.00 MiB
llama_memory_recurrent:      CUDA0 RS buffer size =   716.00 MiB
sched_reserve:      CUDA0 compute buffer size =  1500.25 MiB
sched_reserve: graph (pp bs=512, tg bs=12): nodes = 1530 / 1402, splits = 2 / 2, input objects = 9, input tensors = 14
srv    load_model: initializing slots, n_slots = 12
srv          init: initializing, n_slots = 12, n_ctx_slot = 32768, kv_unified = 'false'
main: server is listening on http://127.0.0.1:8000
"""


def good_reply(body: dict) -> dict:
    """What llama-server answers for the harness's request."""
    kwargs = body.get("chat_template_kwargs") or {}
    thinking = kwargs.get("reasoning_effort") not in ("disabled", "none")
    message = {
        "role": "assistant",
        "content": "17 multiplié par 23 font 391.",
        "reasoning_content": "Je calcule 17 x 23 = 17 x 20 + 17 x 3 = 340 + 51 = 391."
        if thinking
        else "",
    }
    if not thinking:
        del message["reasoning_content"]
    return {
        "choices": [{"finish_reason": "stop", "index": 0, "message": message}],
        "created": 1,
        "model": "olala",
        "system_fingerprint": "b1-55c22dc",
        "object": "chat.completion",
        "usage": {"completion_tokens": 90, "prompt_tokens": 60, "total_tokens": 150},
        "id": "chatcmpl-1",
        "timings": {
            "cache_n": 0,
            "prompt_n": 60,
            "prompt_ms": 30.0,
            "prompt_per_second": 2000.0,
            "predicted_n": 90,
            "predicted_ms": 1000.0,
            "predicted_per_second": 90.0,
        },
    }


class FakeLlama:
    """/health after `healthy_after` seconds, /v1/models, and a chat completion computed by `reply(body)`."""

    def __init__(self, reply: Callable[[dict], dict], healthy_after: float = 0.0) -> None:
        self.reply = reply
        self.start = time.time()
        self.healthy_after = healthy_after
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                if self.path == "/health":
                    ok = time.time() - outer.start >= outer.healthy_after
                    self.send_response(200 if ok else 503)
                    self.end_headers()
                elif self.path == "/v1/models":
                    data = json.dumps({"object": "list", "data": [{"id": "olala"}]}).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "body": body})
                data = json.dumps(outer.reply(body)).encode()
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
        self.httpd.server_close()


@pytest.fixture
def server():
    made: list[FakeLlama] = []

    def make(reply: Callable[[dict], dict] = good_reply, healthy_after: float = 0.0) -> FakeLlama:
        made.append(FakeLlama(reply, healthy_after))
        return made[-1]

    yield make
    for s in made:
        s.close()


@pytest.fixture
def work(tmp_path):
    (tmp_path / "run").mkdir()
    (tmp_path / "logs" / "gguf").mkdir(parents=True)
    return tmp_path


def smoke(work: Path, url: str, log_text: str, *args: str, timeout: str = "20"):
    (work / "logs" / "gguf" / "server-20261120T100000Z.log").write_text(log_text)
    env = {"OLALA_WORK": str(work), "OLALA_BASE_URL": url, "SMOKE_POLL_SECONDS": "0.2"}
    return run_script("pod/smoke_gguf.sh", "--timeout", timeout, *args, env=env)


def report(work: Path) -> dict:
    return json.loads((work / "logs" / "gguf" / "smoke.json").read_text())


def variant_reply(**changes) -> Callable[[dict], dict]:
    """The good reply with the message of the medium-effort request changed."""

    def reply(body: dict) -> dict:
        answer = good_reply(body)
        if (body.get("chat_template_kwargs") or {}).get("reasoning_effort") == "medium":
            answer["choices"][0]["message"].update(changes)
        return answer

    return reply


# --- the good path ---------------------------------------------------------------------------------------------


def test_a_good_server_passes_and_prints_shape_speed_and_the_log_facts(work, server):
    out = smoke(work, server().url, GOOD_LOG)
    assert out.returncode == 0, out.stdout + out.stderr
    r = report(work)
    assert r["verdict"] == "pass" and r["failures"] == [] and r["warnings"] == []
    assert r["answer_nonempty"] and r["reasoning_nonempty"] and r["expected_in_answer"] is True
    assert r["reasoning_in_content"] is False
    assert r["finish_reason"] == "stop"
    shape = r["response_shape"]
    assert "reasoning_content" in shape["message_keys"] and "content" in shape["message_keys"]
    assert "timings" in shape["top_level_keys"] and "predicted_per_second" in shape["timings_keys"]
    assert r["tokens_per_second"] == {
        "generation": 90.0,
        "prompt": 2000.0,
        "source": "server timings",
    }
    assert r["served_model_ids"] == ["olala"]
    facts = " ".join(r["log_facts"])
    assert "KV buffer size = 16128.00 MiB" in facts and "compute buffer size = 1500.25 MiB" in facts
    assert "RS buffer size" in facts and "n_ctx_slot = 32768" in facts
    assert "offloaded 37/37 layers" in facts and "splits = 2 / 2" in facts
    assert r["graph_splits"] == [2, 2] and r["offloaded_layers"] == [37, 37]
    for needle in ("reasoning_content", "90.0", "tokens/s", "n_ctx_slot = 32768"):
        assert needle in out.stdout, needle
    assert list((work / "logs" / "gguf").glob("smoke-*.json"))  # a dated copy: failed runs are kept


def test_the_request_is_exactly_the_harness_request(work, server):
    s = server()
    assert smoke(work, s.url, GOOD_LOG).returncode == 0
    medium = s.requests[0]
    assert medium["path"] == "/v1/chat/completions"
    body = medium["body"]
    assert body["model"] == "olala"
    assert body["chat_template_kwargs"] == {"reasoning_effort": "medium"}
    assert (body["temperature"], body["top_p"], body["top_k"]) == (0.8, 0.9, 40)
    # nothing the harness does not send: a top-level reasoning_effort would hide a failure of the kwargs,
    # and the harness leaves max_tokens unset
    assert set(body) == {
        "model",
        "messages",
        "temperature",
        "top_p",
        "top_k",
        "chat_template_kwargs",
    }
    prompt = body["messages"][-1]["content"]
    assert "17" in prompt and "23" in prompt and "Combien" in prompt


def test_a_second_request_with_effort_disabled_proves_the_kwargs_reach_the_template(work, server):
    s = server()
    assert smoke(work, s.url, GOOD_LOG).returncode == 0
    assert [r["body"]["chat_template_kwargs"]["reasoning_effort"] for r in s.requests] == [
        "medium",
        "disabled",
    ]
    assert report(work)["disabled_check"] == {"answer_nonempty": True, "reasoning_empty": True}


def test_effort_ignored_by_the_server_fails(work, server):
    """If `disabled` still produces a reasoning trace, `high` (the parity run) would be silently `medium`."""

    def ignores(body: dict) -> dict:
        body = {**body, "chat_template_kwargs": {"reasoning_effort": "medium"}}
        return good_reply(body)

    out = smoke(work, server(ignores).url, GOOD_LOG)
    assert out.returncode == 1
    assert "disabled" in " ".join(report(work)["failures"])


# --- reasoning must not pollute the content ----------------------------------------------------------------------


def test_channel_markers_in_the_content_fail_and_name_the_server_flag(work, server):
    leaked = variant_reply(
        content="<|channel_start|>analysis<|content|>Je calcule 17 x 23<|channel_end|><|channel_start|>final<|content|>391",
        reasoning_content="",
    )
    out = smoke(work, server(leaked).url, GOOD_LOG)
    assert out.returncode == 1
    r = report(work)
    assert r["reasoning_in_content"] is True
    text = " ".join(r["failures"])
    assert "reasoning" in text and "--reasoning-format" in text


def test_the_reasoning_text_repeated_in_the_content_fails_even_without_markers(work, server):
    trace = "Je calcule 17 x 23 = 17 x 20 + 17 x 3 = 340 + 51 = 391."
    leaked = variant_reply(content=trace + " Donc 391.")
    out = smoke(work, server(leaked).url, GOOD_LOG)
    assert out.returncode == 1 and report(work)["reasoning_in_content"] is True


def test_an_empty_reasoning_field_at_medium_effort_fails(work, server):
    out = smoke(work, server(variant_reply(reasoning_content="")).url, GOOD_LOG)
    assert out.returncode == 1
    assert report(work)["reasoning_nonempty"] is False


def test_an_empty_final_answer_fails_and_a_length_stop_is_named(work, server):
    def reply(body: dict) -> dict:
        answer = good_reply(body)
        if (body.get("chat_template_kwargs") or {}).get("reasoning_effort") == "medium":
            answer["choices"][0]["message"]["content"] = ""
            answer["choices"][0]["finish_reason"] = "length"
        return answer

    out = smoke(work, server(reply).url, GOOD_LOG)
    assert out.returncode == 1
    r = report(work)
    assert r["answer_nonempty"] is False and "length" in " ".join(r["failures"])


def test_a_wall_of_question_marks_is_the_nan_signature_and_fails(work, server):
    out = smoke(work, server(variant_reply(content="?" * 40)).url, GOOD_LOG)
    assert out.returncode == 1
    assert "NaN" in " ".join(report(work)["failures"]) or "?" in " ".join(report(work)["failures"])


def test_a_wrong_answer_is_a_warning_not_a_failure(work, server):
    out = smoke(work, server(variant_reply(content="Je ne sais pas.")).url, GOOD_LOG)
    assert out.returncode == 2
    assert report(work)["expected_in_answer"] is False


# --- speed ---------------------------------------------------------------------------------------------------------


def test_without_server_timings_the_speed_is_computed_from_the_wall_clock_and_warned(work, server):
    def no_timings(body: dict) -> dict:
        answer = good_reply(body)
        del answer["timings"]
        return answer

    out = smoke(work, server(no_timings).url, GOOD_LOG)
    assert out.returncode == 2
    r = report(work)
    assert r["tokens_per_second"]["source"] == "wall clock"
    assert any("timings" in w for w in r["warnings"])


def test_parallel_requests_report_the_aggregate_speed(work, server):
    s = server()
    out = smoke(work, s.url, GOOD_LOG, "--parallel", "4")
    assert out.returncode == 0, out.stdout + out.stderr
    p = report(work)["parallel"]
    assert p["requests"] == 4 and p["ok"] == 4 and p["errors"] == 0
    assert p["generated_tokens"] == 4 * 90 and p["aggregate_tokens_per_second"] > 0
    assert "aggregate" in out.stdout
    parallel_bodies = [r["body"] for r in s.requests[2:]]
    assert len(parallel_bodies) == 4
    assert all(
        b["max_tokens"] == 256 for b in parallel_bodies
    )  # the probe is bounded; the real request is not


# --- the server log ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "/workspace/llama.cpp-olala/ggml/src/ggml-cuda/mmq.cu:1234: GGML_ASSERT(prec == GGML_PREC_Q8) failed",
        "CUDA error: an illegal memory access was encountered",
        "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 20000.00 MiB on device 0: cudaMalloc failed: out of memory",
        "llama_model_load: error loading model: unknown model architecture: 'olala'",
        "srv  update_slots: Context size has been exceeded. off = 0, n_batch = 1, ret = 1",
        "Segmentation fault (core dumped)",
    ],
)
def test_a_fatal_line_in_the_log_fails_fast_without_waiting_for_health(work, server, line):
    s = server(healthy_after=999)
    started = time.time()
    out = smoke(work, s.url, GOOD_LOG + line + "\n", timeout="120")
    assert out.returncode == 1 and time.time() - started < 20
    assert s.requests == []  # no request is sent to a server that failed to start
    assert report(work)["verdict"] == "fail"


def test_nan_fallback_and_kv_pressure_lines_are_warnings(work, server):
    log = (
        GOOD_LOG + "warning: falling back to the generic kernel\nloss is nan here\n"
        "srv  update_slots: failed to find free space in the KV cache, retrying with smaller batch size\n"
    )
    out = smoke(work, server().url, log)
    assert out.returncode == 2
    r = report(work)
    assert r["nan"] and r["slow_path"] and r["kv_pressure"]


def test_a_dead_server_process_ends_the_wait_at_once(work, server):
    s = server(healthy_after=999)
    (work / "run" / "server.pid").write_text("999999\n")
    started = time.time()
    out = smoke(work, s.url, GOOD_LOG, timeout="120")
    assert out.returncode == 1 and time.time() - started < 20
    assert "exited" in out.stdout + out.stderr


def test_a_running_pid_does_not_end_the_wait_and_a_slow_health_is_waited_for(work, server):
    s = server(healthy_after=1)
    (work / "run" / "server.pid").write_text(f"{os.getpid()}\n")
    assert smoke(work, s.url, GOOD_LOG).returncode == 0


def test_a_server_that_never_answers_health_times_out_and_says_so(work, server):
    out = smoke(work, server(healthy_after=999).url, GOOD_LOG, timeout="2")
    assert out.returncode == 1 and "timeout" in (out.stdout + out.stderr).lower()


def test_the_default_log_is_the_newest_gguf_server_log(work, server):
    old = work / "logs" / "gguf" / "server-20261120T090000Z.log"
    old.write_text("GGML_ASSERT(false) failed\n")
    os.utime(old, (1, 1))
    assert smoke(work, server().url, GOOD_LOG).returncode == 0
    assert report(work)["log"].endswith("server-20261120T100000Z.log")


def test_many_graph_splits_mean_layers_run_on_the_cpu_and_are_a_warning(work, server):
    """The 612-splits, 7.5 tokens/s failure of the upstream v1.0.0 merge (commit e7515f6 of the fork)."""
    log = GOOD_LOG.replace("splits = 2 / 2", "splits = 612 / 612")
    out = smoke(work, server().url, log)
    assert out.returncode == 2
    r = report(work)
    assert r["graph_splits"] == [612, 612]
    assert any("splits" in w and "CPU" in w for w in r["warnings"])


def test_layers_left_on_the_cpu_are_a_warning(work, server):
    out = smoke(work, server().url, GOOD_LOG.replace("37/37", "30/37"))
    assert out.returncode == 2
    assert any("offloaded" in w for w in report(work)["warnings"])
