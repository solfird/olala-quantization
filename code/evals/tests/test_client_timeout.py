"""The client timeout reaches inspect_ai's HTTP client, proved against a local stub (no model).

inspect_ai 0.3.291's OpenAI-compatible provider has its own HTTP client timeout, the model argument
``client_timeout`` (600 s unless given). ``GenerateConfig(timeout=...)`` does not change it, so a
request that needs more than 600 s was cancelled by the client and silently redrawn. These tests
build the model and the generate configuration with the functions ``run_tasks.main`` uses, point
them at a stdlib HTTP server that sleeps a few seconds before it answers, and check what the client
does when its timeout is shorter and longer than the sleep.
"""

import threading
import time
from collections.abc import Callable
from pathlib import Path

import inspect_ai
from fake_server import FakeServer
from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.event._model import ModelEvent
from inspect_ai.log import EvalSample, read_eval_log
from inspect_ai.scorer import includes
from inspect_ai.solver import generate

import run_tasks

SLEEP = 2.0
"""Seconds the stub holds a slow request before it answers."""


def one_question() -> Task:
    sample = Sample(id="q1", input="Question 1: what is 1+1?", target="ok")
    return Task(dataset=MemoryDataset([sample]), solver=[generate()], scorer=includes())


def slow_requests(count: int) -> tuple[list[int], Callable[[dict], tuple[str, str]]]:
    """A chat reply that sleeps ``SLEEP`` s on its first ``count`` requests, then answers fast."""
    seen: list[int] = []
    lock = threading.Lock()

    def reply(body: dict) -> tuple[str, str]:
        with lock:
            seen.append(1)
            nth = len(seen)
        if nth <= count:
            time.sleep(SLEEP)
        return "ok", "stop"

    return seen, reply


def evaluate(tmp_path: Path, base_url: str, *, client_timeout: int, request_timeout: int = 14400):
    """One question through the harness's model, generate config and ``inspect_ai.eval`` call."""
    args = run_tasks._build_parser().parse_args(
        ["--out", "unused", "--timeout", str(request_timeout), "--max-retries", "5"]
    )
    model = run_tasks.build_model(base_url, "olala", client_timeout)
    config = run_tasks.generate_config(args, 1)
    log = inspect_ai.eval(
        one_question(),
        model=model,
        log_dir=str(tmp_path / "logs"),
        display="none",
        fail_on_error=False,
        log_samples=True,
        max_samples=1,
        **config.model_dump(exclude_none=True),
    )[0]
    sample = read_eval_log(log.location).samples[0]
    assert isinstance(sample, EvalSample)
    return sample


def model_events(sample: EvalSample) -> list[ModelEvent]:
    return [e for e in sample.events if isinstance(e, ModelEvent)]


def test_a_request_longer_than_the_client_timeout_is_cancelled_and_sent_again(tmp_path):
    _, reply = slow_requests(1)
    with FakeServer(chat_reply=reply) as srv:
        # the harness-level timeout is 14,400 s and does not help: only client_timeout=1 is shorter
        sample = evaluate(tmp_path, srv.base_url, client_timeout=1)
        requests = len(srv.requests)
    assert requests == 2  # the first request was cancelled at about 1 s, the second one answered
    assert sample.error is None
    row = run_tasks.row_from_sample(sample, "toy", None, None)
    assert row["model_attempts"] == 2  # even though inspect_ai logged one single model event
    assert len(model_events(sample)) == 1


def test_a_request_shorter_than_the_client_timeout_is_answered_by_one_attempt(tmp_path):
    _, reply = slow_requests(1)
    with FakeServer(chat_reply=reply) as srv:
        started = time.monotonic()
        sample = evaluate(tmp_path, srv.base_url, client_timeout=30)
        elapsed = time.monotonic() - started
        requests = len(srv.requests)
    assert requests == 1
    assert elapsed >= SLEEP  # the sleep was waited for, not cancelled
    assert sample.error is None
    assert len(model_events(sample)) == 1
    row = run_tasks.row_from_sample(sample, "toy", None, None)
    assert row["model_attempts"] == 1
    assert row["working_time"] >= SLEEP * 0.9


def test_the_default_client_timeout_of_inspect_ai_is_what_the_harness_replaces():
    # inspect_ai builds its HTTP client from client_timeout; None means the library's own 600 s
    model = run_tasks.build_model("http://127.0.0.1:9/v1", "olala", 14400)
    assert model.api.client_timeout == 14400


def test_the_provider_argument_is_named_client_timeout(monkeypatch):
    import inspect_ai.model as inspect_model

    captured: dict = {}
    real = inspect_model.get_model

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(inspect_model, "get_model", spy)
    run_tasks.build_model("http://127.0.0.1:9/v1", "olala", 7200)
    assert captured["client_timeout"] == 7200
    assert captured["base_url"] == "http://127.0.0.1:9/v1"
    assert captured["memoize"] is False
