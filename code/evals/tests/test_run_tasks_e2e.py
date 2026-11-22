"""End-to-end tests of run_tasks.py: real inspect_ai against the fake OpenAI-compatible server."""

import json
import re
import time

import pytest
from fake_server import FakeHTTPError, FakeServer
from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import generate

import run_tasks
from run_tasks import TaskSpec


def toy_task() -> Task:
    """Twelve tiny questions; even-numbered ones are French, odd-numbered English."""
    samples = [
        Sample(
            id=f"q{i}",
            input=f"Question {i}: what is {i}+{i}?",
            target=str(2 * i),
            metadata={"language": "en" if i % 2 else "fr"},
        )
        for i in range(1, 13)
    ]
    return Task(dataset=MemoryDataset(samples), solver=[generate()], scorer=includes())


def _question_number(body: dict) -> int:
    text = body["messages"][-1]["content"]
    text = text if isinstance(text, str) else text[0]["text"]
    return int(re.search(r"Question (\d+):", text).group(1))


def reply(body: dict) -> tuple[str, str]:
    """Answer correctly for odd questions, wrongly for even ones; cut question 3 short."""
    n = _question_number(body)
    if n == 3:
        return "6 but cut off", "length"
    if n == 5:
        raise FakeHTTPError(400, "bad request for question 5")
    return (str(2 * n) if n % 2 else "no idea"), "stop"


@pytest.fixture(autouse=True)
def toy_registered(monkeypatch):
    spec = TaskSpec(name="toy", factory=toy_task, epochs=2, stratum_field="language")
    monkeypatch.setitem(run_tasks.TASKS, "toy", spec)


def run(tmp_path, server, *extra, name="out", tasks=("toy",)):
    out = tmp_path / name
    argv = [
        "--tasks", *tasks,
        "--base-url", server.base_url,
        "--model", "olala",
        "--out", str(out),
        "--max-retries", "0",
        "--max-connections", "4",
        *extra,
    ]  # fmt: skip
    assert run_tasks.main(argv) == 0
    return out


def rows_of(out):
    return [json.loads(line) for line in (out / "samples.jsonl").read_text().splitlines()]


def test_a_run_writes_one_row_per_item_and_epoch_with_the_expected_fields(tmp_path):
    with FakeServer(chat_reply=reply) as srv:
        out = run(tmp_path, srv)
    rows = rows_of(out)
    assert len(rows) == 12 * 2
    assert {r["epoch"] for r in rows} == {1, 2}
    by_id = {(r["sample_id"], r["epoch"]): r for r in rows}
    assert by_id[("q1", 1)]["score"] == 1.0  # odd question, answered correctly
    assert by_id[("q2", 1)]["score"] == 0.0  # even question, wrong answer
    assert by_id[("q1", 1)]["stratum"] == "en"
    assert by_id[("q2", 1)]["stratum"] == "fr"


def test_truncations_and_request_errors_are_flagged_in_rows_and_counted_in_run_info(tmp_path):
    with FakeServer(chat_reply=reply) as srv:
        out = run(tmp_path, srv)
    rows = {(r["sample_id"], r["epoch"]): r for r in rows_of(out)}
    assert rows[("q3", 1)]["truncated"] is True
    assert rows[("q3", 1)]["stop_reason"] == "max_tokens"
    assert rows[("q5", 1)]["error"] is not None
    assert rows[("q5", 1)]["score"] is None
    info = json.loads((out / "run_info.json").read_text())
    assert info["truncations"] == 2  # q3, two epochs
    assert info["request_errors"] == 2  # q5, two epochs
    assert info["tasks"]["toy"]["truncations"] == 2
    assert info["tasks"]["toy"]["request_errors"] == 2
    assert info["n_samples"] == 24


def test_the_protocol_sampling_parameters_reach_the_server(tmp_path):
    with FakeServer(chat_reply=reply) as srv:
        run(tmp_path, srv)
    body = next(r["body"] for r in srv.requests if r["path"] == "/v1/chat/completions")
    assert body["model"] == "olala"
    assert body["temperature"] == 0.8
    assert body["top_p"] == 0.9
    assert body["top_k"] == 40
    assert body["chat_template_kwargs"] == {"reasoning_effort": "high"}
    assert "seed" not in body  # independent samples across epochs


def test_reasoning_effort_and_max_tokens_are_selectable(tmp_path):
    with FakeServer(chat_reply=reply) as srv:
        out = run(tmp_path, srv, "--reasoning-effort", "disabled", "--max-tokens", "777")
    body = next(r["body"] for r in srv.requests if r["path"] == "/v1/chat/completions")
    assert body["chat_template_kwargs"] == {"reasoning_effort": "disabled"}
    assert 777 in (body.get("max_tokens"), body.get("max_completion_tokens"))
    info = json.loads((out / "run_info.json").read_text())
    assert info["reasoning_effort"] == "disabled"
    assert info["sampling"]["max_tokens"] == 777


def test_run_info_records_the_exact_settings(tmp_path):
    with FakeServer(chat_reply=reply) as srv:
        out = run(tmp_path, srv, "--seed", "7", "--n-items", "6")
        endpoint = srv.base_url
    info = json.loads((out / "run_info.json").read_text())
    assert info["endpoint"] == endpoint
    assert info["model"] == "olala"
    assert info["seed"] == 7
    assert info["sampling"] == {
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": None,
        "seed": None,
        "max_retries": 0,
        "timeout": 14400,
    }
    assert info["reasoning_effort"] == "high"
    assert info["n_items_requested"] == {"default": 6, "per_task": {}}
    assert info["tasks"]["toy"]["epochs"] == 2
    assert info["tasks"]["toy"]["n_items"] == 6
    assert len(info["tasks"]["toy"]["items_sha256"]) == 64
    assert info["started_utc"] <= info["ended_utc"]
    assert info["status"] == "complete"
    assert "inspect_ai" in info["versions"]
    assert "inspect_evals" in info["versions"]
    assert "commit" in info["git"]
    assert len(info["code_sha256"]) == 64
    assert info["argv"][0] == "--tasks"
    assert info["skipped_tasks"] == {}


def test_two_runs_with_the_same_seed_and_n_items_score_identical_items(tmp_path):
    with FakeServer(chat_reply=reply) as srv:
        a = run(tmp_path, srv, "--seed", "1", "--n-items", "6", name="a")
        b = run(tmp_path, srv, "--seed", "1", "--n-items", "6", name="b")
        c = run(tmp_path, srv, "--seed", "2", "--n-items", "6", name="c")
    items = {
        n: json.loads((d / "items.json").read_text()) for n, d in (("a", a), ("b", b), ("c", c))
    }
    assert items["a"] == items["b"]
    assert items["a"] != items["c"]
    assert len(items["a"]["toy"]) == 6
    # stratified by language: 3 English (odd) + 3 French (even) questions
    langs = [int(i[1:]) % 2 for i in items["a"]["toy"]]
    assert sorted(langs) == [0, 0, 0, 1, 1, 1]
    assert {r["sample_id"] for r in rows_of(a)} == set(items["a"]["toy"])


def test_items_from_reuses_the_exact_items_of_another_run(tmp_path):
    with FakeServer(chat_reply=reply) as srv:
        a = run(tmp_path, srv, "--seed", "1", "--n-items", "5", name="a")
        b = run(tmp_path, srv, "--seed", "99", "--items-from", str(a / "items.json"), name="b")
    assert json.loads((a / "items.json").read_text()) == json.loads((b / "items.json").read_text())
    assert {r["sample_id"] for r in rows_of(a)} == {r["sample_id"] for r in rows_of(b)}
    info = json.loads((b / "run_info.json").read_text())
    assert info["items_from"] == str(a / "items.json")


def test_items_from_with_an_unknown_id_is_refused(tmp_path):
    bad = tmp_path / "items.json"
    bad.write_text(json.dumps({"toy": ["q1", "nope"]}))
    with FakeServer(chat_reply=reply) as srv, pytest.raises(SystemExit, match="nope"):
        run(tmp_path, srv, "--items-from", str(bad))


def test_epochs_can_be_overridden(tmp_path):
    with FakeServer(chat_reply=reply) as srv:
        out = run(tmp_path, srv, "--epochs", "1")
    assert len(rows_of(out)) == 12
    assert json.loads((out / "run_info.json").read_text())["tasks"]["toy"]["epochs"] == 1


def test_epochs_can_be_overridden_for_one_task_and_the_others_keep_their_default(
    tmp_path, monkeypatch
):
    other = TaskSpec(name="other", factory=toy_task, epochs=3)
    monkeypatch.setitem(run_tasks.TASKS, "other", other)
    with FakeServer(chat_reply=reply) as srv:
        out = run(tmp_path, srv, "--epochs", "other=1", "--n-items", "4", tasks=("toy", "other"))
    info = json.loads((out / "run_info.json").read_text())
    assert info["tasks"]["toy"]["epochs"] == 2  # the registry default, untouched
    assert info["tasks"]["other"]["epochs"] == 1
    assert info["tasks"]["other"]["n_samples"] == 4
    assert info["epochs_requested"] == {"default": None, "per_task": {"other": 1}}


def test_a_bare_epochs_value_applies_to_every_task_and_a_task_value_beats_it(tmp_path, monkeypatch):
    monkeypatch.setitem(
        run_tasks.TASKS, "other", TaskSpec(name="other", factory=toy_task, epochs=3)
    )
    with FakeServer(chat_reply=reply) as srv:
        out = run(
            tmp_path, srv, "--epochs", "1", "--epochs", "toy=2", "--n-items", "4",
            tasks=("toy", "other"),
        )  # fmt: skip
    info = json.loads((out / "run_info.json").read_text())
    assert (info["tasks"]["toy"]["epochs"], info["tasks"]["other"]["epochs"]) == (2, 1)
    assert info["epochs_requested"] == {"default": 1, "per_task": {"toy": 2}}


def test_an_epochs_override_for_an_unknown_task_is_an_error_and_writes_nothing(tmp_path):
    with FakeServer(chat_reply=reply) as srv, pytest.raises(SystemExit, match="nonsense"):
        run(tmp_path, srv, "--epochs", "nonsense=2")
    assert not (tmp_path / "out").exists()


def test_the_sandbox_mode_reaches_the_tasks_that_run_code_and_is_recorded(tmp_path, monkeypatch):
    seen: list[str] = []

    def code_task(sandbox: str) -> Task:
        seen.append(sandbox)
        return toy_task()

    monkeypatch.setitem(
        run_tasks.TASKS, "coder", TaskSpec(name="coder", factory=code_task, takes_sandbox=True)
    )
    with FakeServer(chat_reply=reply) as srv:
        out = run(tmp_path, srv, "--sandbox", "local", "--n-items", "2", tasks=("toy", "coder"))
    assert seen == ["local"]
    info = json.loads((out / "run_info.json").read_text())
    assert info["sandbox"] == {"mode": "local", "tasks": ["coder"]}


def test_the_sandbox_defaults_to_docker_and_an_unknown_mode_is_refused(tmp_path, monkeypatch):
    seen: list[str] = []

    def code_task(sandbox: str) -> Task:
        seen.append(sandbox)
        return toy_task()

    monkeypatch.setitem(
        run_tasks.TASKS, "coder", TaskSpec(name="coder", factory=code_task, takes_sandbox=True)
    )
    with FakeServer(chat_reply=reply) as srv:
        out = run(tmp_path, srv, "--n-items", "2", tasks=("coder",))
        with pytest.raises(SystemExit):
            run(tmp_path, srv, "--sandbox", "chroot", name="bad", tasks=("coder",))
    assert seen == ["docker"]
    assert json.loads((out / "run_info.json").read_text())["sandbox"]["mode"] == "docker"


def test_the_real_code_tasks_are_the_ones_that_take_a_sandbox():
    assert {n for n, spec in run_tasks.TASKS.items() if spec.takes_sandbox} == {
        "humaneval_plus",
        "mbpp_plus",
    }


def test_a_sampling_seed_is_refused_with_several_epochs_because_epochs_would_be_identical(
    tmp_path,
):
    with FakeServer(chat_reply=reply) as srv, pytest.raises(SystemExit, match="epochs"):
        run(tmp_path, srv, "--sampling-seed", "5")


def test_a_sampling_seed_is_sent_when_there_is_one_epoch(tmp_path):
    with FakeServer(chat_reply=reply) as srv:
        out = run(tmp_path, srv, "--sampling-seed", "5", "--epochs", "1")
    body = next(r["body"] for r in srv.requests if r["path"] == "/v1/chat/completions")
    assert body["seed"] == 5
    assert json.loads((out / "run_info.json").read_text())["sampling"]["seed"] == 5


def test_unavailable_tasks_are_skipped_with_a_note_not_silently(tmp_path):
    with FakeServer(chat_reply=reply) as srv:
        out = run(tmp_path, srv, tasks=("toy", "ifbench"))
    info = json.loads((out / "run_info.json").read_text())
    assert "ifbench" in info["skipped_tasks"]
    assert "inspect_evals" in info["skipped_tasks"]["ifbench"]
    assert "ifbench" not in info["tasks"]


def test_an_unknown_task_name_is_an_error(tmp_path):
    with FakeServer(chat_reply=reply) as srv, pytest.raises(SystemExit, match="nonsense"):
        run(tmp_path, srv, tasks=("nonsense",))


def test_an_existing_output_directory_is_not_overwritten(tmp_path):
    with FakeServer(chat_reply=reply) as srv:
        run(tmp_path, srv, "--epochs", "1")
        with pytest.raises(SystemExit, match="already exists"):
            run(tmp_path, srv, "--epochs", "1")


def test_per_task_n_items_overrides_the_default(tmp_path):
    with FakeServer(chat_reply=reply) as srv:
        out = run(tmp_path, srv, "--n-items", "10", "--n-items", "toy=4", "--epochs", "1")
    assert len(rows_of(out)) == 4


def test_inspect_ai_does_not_forward_generate_config_top_k_so_run_tasks_sends_it_itself():
    """Characterisation test: delete the extra_body workaround if this ever fails."""
    import asyncio

    from inspect_ai.model import GenerateConfig, get_model

    with FakeServer(chat_reply=lambda body: ("hi", "stop")) as srv:
        model = get_model(
            "openai-api/vllm/olala", base_url=srv.base_url, api_key="x", memoize=False
        )
        asyncio.run(model.generate("hello", config=GenerateConfig(top_k=40, top_p=0.9)))
    body = srv.requests[0]["body"]
    assert body["top_p"] == 0.9
    assert "top_k" not in body


def test_a_crash_marks_the_run_as_failed_in_run_info_and_propagates(tmp_path, monkeypatch):
    def broken() -> Task:
        raise RuntimeError("dataset host unreachable")

    monkeypatch.setitem(run_tasks.TASKS, "toy", TaskSpec(name="toy", factory=broken))
    out = tmp_path / "out"
    with (
        FakeServer(chat_reply=reply) as srv,
        pytest.raises(RuntimeError, match="dataset host unreachable"),
    ):
        run_tasks.main(["--tasks", "toy", "--base-url", srv.base_url, "--out", str(out)])
    info = json.loads((out / "run_info.json").read_text())
    assert info["status"] == "failed"
    assert "dataset host unreachable" in info["error"]
    assert info["ended_utc"] is not None


# ------------------------------------------------------------- per-task concurrency and timeout


@pytest.fixture
def eval_calls(monkeypatch):
    """Record the keyword arguments of every ``inspect_ai.eval`` call, then run it for real."""
    import inspect_ai

    calls: list[dict] = []
    real = inspect_ai.eval

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(inspect_ai, "eval", spy)
    return calls


def run_with(tmp_path, server, *extra, tasks=("toy",), name="out"):
    """Like ``run`` but without the fixed ``--max-connections 4``."""
    out = tmp_path / name
    argv = [
        "--tasks", *tasks,
        "--base-url", server.base_url,
        "--out", str(out),
        "--max-retries", "0",
        "--n-items", "2",
        *extra,
    ]  # fmt: skip
    assert run_tasks.main(argv) == 0
    return out


def test_each_task_uses_its_own_resolved_concurrency_for_connections_and_samples(
    tmp_path, monkeypatch, eval_calls
):
    for name in ("other", "third"):
        monkeypatch.setitem(run_tasks.TASKS, name, TaskSpec(name=name, factory=toy_task, epochs=1))
    monkeypatch.setitem(run_tasks.TASK_MAX_CONNECTIONS, "other", 3)  # built-in beats the bare 8
    with FakeServer(chat_reply=reply) as srv:
        out = run_with(
            tmp_path, srv, "--max-connections", "8", "--max-connections", "toy=2",
            tasks=("toy", "other", "third"),
        )  # fmt: skip
    assert [(c["max_connections"], c["max_samples"]) for c in eval_calls] == [
        (2, 2),
        (3, 3),
        (8, 8),
    ]
    info = json.loads((out / "run_info.json").read_text())
    assert info["max_connections"] == 8  # the bare value, as before
    assert info["max_connections_per_task"] == {"toy": 2, "other": 3, "third": 8}


def test_the_global_default_is_32_and_the_timeout_reaches_the_client(tmp_path, eval_calls):
    with FakeServer(chat_reply=reply) as srv:
        out = run_with(tmp_path, srv)
    (call,) = eval_calls
    assert (call["max_connections"], call["max_samples"], call["timeout"]) == (32, 32, 14400)
    info = json.loads((out / "run_info.json").read_text())
    assert info["max_connections"] == 32
    assert info["max_connections_per_task"] == {"toy": 32}
    assert info["sampling"]["timeout"] == 14400


def test_a_max_connections_override_for_an_unknown_task_is_an_error_and_writes_nothing(tmp_path):
    with FakeServer(chat_reply=reply) as srv, pytest.raises(SystemExit, match="nonsense"):
        run_with(tmp_path, srv, "--max-connections", "nonsense=2")
    assert not (tmp_path / "out").exists()


def test_only_the_tasks_that_run_appear_in_max_connections_per_task(tmp_path, monkeypatch):
    monkeypatch.setitem(
        run_tasks.TASKS, "other", TaskSpec(name="other", factory=toy_task, epochs=1)
    )
    with FakeServer(chat_reply=reply) as srv:
        out = run_with(tmp_path, srv, "--max-connections", "other=5")  # other is not run
    info = json.loads((out / "run_info.json").read_text())
    assert info["max_connections_per_task"] == {"toy": 32}


# --------------------------------------------------------------------- client timeout and retries


@pytest.mark.parametrize(("flag", "expected"), [([], 14400), (["--timeout", "7200"], 7200)])
def test_the_timeout_option_is_also_the_providers_client_timeout(
    tmp_path, eval_calls, flag, expected
):
    with FakeServer(chat_reply=reply) as srv:
        out = run_with(tmp_path, srv, *flag)
    (call,) = eval_calls
    assert call["timeout"] == expected
    assert call["model"].api.client_timeout == expected  # inspect_ai's own default is 600 s
    info = json.loads((out / "run_info.json").read_text())
    assert info["client_timeout"] == expected
    assert info["tasks"]["toy"]["client_timeout"] == expected
    assert info["sampling"]["timeout"] == expected


def test_rows_carry_the_attempt_count_and_working_time_and_run_info_summarises_them(tmp_path):
    with FakeServer(chat_reply=reply) as srv:
        out = run(tmp_path, srv)
    rows = rows_of(out)
    assert {r["model_attempts"] for r in rows} == {1}
    assert all(isinstance(r["working_time"], float) and r["working_time"] >= 0 for r in rows)
    info = json.loads((out / "run_info.json").read_text())
    assert info["samples_with_retries"] == 0
    assert info["tasks"]["toy"]["samples_with_retries"] == 0
    longest = max(r["working_time"] for r in rows)
    assert info["max_working_time"] == longest
    assert info["tasks"]["toy"]["max_working_time"] == longest


def test_a_request_cancelled_by_the_client_and_redrawn_is_counted_as_a_retried_sample(tmp_path):
    state = {"n": 0}

    def slow_once(body: dict) -> tuple[str, str]:
        state["n"] += 1
        if state["n"] == 1:
            time.sleep(2.0)  # longer than --timeout 1: the client gives up and sends it again
        return "4", "stop"

    with FakeServer(chat_reply=slow_once) as srv:
        out = run(tmp_path, srv, "--timeout", "1", "--epochs", "1", "--n-items", "4")
    rows = rows_of(out)
    assert sorted(r["model_attempts"] for r in rows) == [1, 1, 1, 2]
    info = json.loads((out / "run_info.json").read_text())
    assert info["samples_with_retries"] == 1
    assert info["tasks"]["toy"]["samples_with_retries"] == 1
    assert info["request_errors"] == 0  # the redraw succeeded: only the counter shows the loss


def test_the_run_wide_counters_add_up_over_every_task_of_the_run(tmp_path, monkeypatch):
    monkeypatch.setitem(
        run_tasks.TASKS, "other", TaskSpec(name="other", factory=toy_task, epochs=1)
    )
    state = {"n": 0}

    def slow_once(body: dict) -> tuple[str, str]:
        state["n"] += 1
        if state["n"] == 1:
            time.sleep(2.0)  # the first request, which belongs to the first task
        return "4", "stop"

    with FakeServer(chat_reply=slow_once) as srv:
        out = run(
            tmp_path, srv, "--timeout", "1", "--epochs", "1", "--n-items", "4",
            tasks=("toy", "other"),
        )  # fmt: skip
    info = json.loads((out / "run_info.json").read_text())
    assert info["tasks"]["toy"]["samples_with_retries"] == 1
    assert info["tasks"]["other"]["samples_with_retries"] == 0
    assert info["samples_with_retries"] == 1
    assert info["max_working_time"] == max(r["working_time"] for r in rows_of(out))
    assert info["max_working_time"] == max(
        info["tasks"][t]["max_working_time"] for t in ("toy", "other")
    )
    assert info["client_timeout"] == 1
