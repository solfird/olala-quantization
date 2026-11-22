"""Unit tests for the pure helpers of run_tasks.py (selection, row mapping, run info)."""

import random
import subprocess

import pytest
from inspect_ai.log import EvalError, EvalSample
from inspect_ai.model import ModelOutput, ModelUsage
from inspect_ai.scorer import Score

from run_tasks import (
    DEFAULT_MAX_CONNECTIONS,
    DEFAULT_TIMEOUT,
    TASK_MAX_CONNECTIONS,
    TASKS,
    _build_parser,
    git_info,
    parse_epochs,
    parse_max_connections,
    parse_n_items,
    resolve_max_connections,
    row_from_sample,
    select_items,
    sha256_of_ids,
)

# --------------------------------------------------------------------------- selection


def test_selection_is_deterministic_and_independent_of_input_order():
    ids = [f"q{i}" for i in range(100)]
    a = select_items(ids, None, 20, seed=1)
    shuffled = ids[:]
    random.Random(0).shuffle(shuffled)
    b = select_items(shuffled, None, 20, seed=1)
    assert sorted(a) == sorted(b)
    assert len(set(a)) == 20


def test_selection_returns_items_in_dataset_order():
    ids = [f"q{i}" for i in range(50)]
    out = select_items(ids, None, 10, seed=3)
    assert out == [i for i in ids if i in set(out)]


def test_different_seeds_give_different_subsets():
    ids = [f"q{i}" for i in range(100)]
    assert select_items(ids, None, 20, seed=1) != select_items(ids, None, 20, seed=2)


def test_asking_for_more_items_than_exist_returns_everything():
    ids = ["a", "b", "c"]
    assert select_items(ids, None, 10, seed=0) == ids
    assert select_items(ids, None, None, seed=0) == ids


def test_stratified_selection_takes_equal_shares_from_every_stratum():
    ids = [f"{lang}-{i}" for lang in ("en", "fr", "de") for i in range(40)]
    strata = [i.split("-")[0] for i in ids]
    out = select_items(ids, strata, 30, seed=0)
    counts = {lang: sum(1 for i in out if i.startswith(lang)) for lang in ("en", "fr", "de")}
    assert counts == {"en": 10, "fr": 10, "de": 10}


def test_stratified_remainder_goes_to_the_first_strata_in_sorted_order():
    ids = [f"{lang}-{i}" for lang in ("b", "a", "c") for i in range(10)]
    strata = [i.split("-")[0] for i in ids]
    out = select_items(ids, strata, 7, seed=0)  # 7 = 3 + 2 + 2 over a, b, c
    counts = {s: sum(1 for i in out if i.startswith(s)) for s in "abc"}
    assert counts == {"a": 3, "b": 2, "c": 2}


def test_a_small_stratum_is_taken_whole_and_the_rest_is_redistributed():
    ids = ["x-1", "y-1", "y-2", "y-3", "y-4", "y-5"]
    strata = [i.split("-")[0] for i in ids]
    out = select_items(ids, strata, 4, seed=0)
    assert "x-1" in out
    assert len(out) == 4


def test_selection_rejects_a_non_positive_count():
    with pytest.raises(ValueError, match="positive"):
        select_items(["a"], None, 0, seed=0)


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        select_items(["a", "a"], None, 1, seed=0)


def test_parse_n_items_accepts_a_default_and_per_task_overrides():
    assert parse_n_items(["200", "mgsm=550", "aime2025=30"]) == (200, {"mgsm": 550, "aime2025": 30})
    assert parse_n_items(None) == (None, {})
    with pytest.raises(ValueError, match="n-items"):
        parse_n_items(["mgsm="])


def test_the_default_epochs_are_the_frozen_protocols_five_for_aime_and_one_elsewhere():
    """PROTOCOL.md section 6: AIME 2025 = 5 samples per problem; every other item 1."""
    assert {name: spec.epochs for name, spec in TASKS.items()} == {
        "mgsm": 1,
        "humaneval_plus": 1,
        "mbpp_plus": 1,
        "gsm8k": 1,
        "ifeval": 1,
        "aime2025": 5,
    }


def test_parse_epochs_accepts_a_bare_number_for_every_task_and_task_overrides():
    known = list(TASKS)
    assert parse_epochs(["3", "aime2025=5", "mgsm=2"], known) == (3, {"aime2025": 5, "mgsm": 2})
    assert parse_epochs(None, known) == (None, {})
    assert parse_epochs(["mbpp_plus=4"], known) == (None, {"mbpp_plus": 4})


def test_parse_epochs_rejects_unknown_tasks_and_malformed_values():
    known = list(TASKS)
    with pytest.raises(ValueError, match="nonsense"):
        parse_epochs(["nonsense=2"], known)
    for bad in ("0", "-1", "x", "mgsm=", "mgsm=0", "=3", "mgsm=1.5"):
        with pytest.raises(ValueError, match="epochs"):
            parse_epochs([bad], known)


def test_parse_max_connections_accepts_a_bare_number_and_task_overrides():
    known = list(TASKS)
    assert parse_max_connections(["16", "aime2025=8", "mgsm=4"], known) == (
        16,
        {"aime2025": 8, "mgsm": 4},
    )
    assert parse_max_connections(None, known) == (None, {})
    assert parse_max_connections(["mbpp_plus=4"], known) == (None, {"mbpp_plus": 4})


def test_parse_max_connections_rejects_unknown_tasks_and_malformed_values():
    known = list(TASKS)
    with pytest.raises(ValueError, match="nonsense"):
        parse_max_connections(["nonsense=2"], known)
    for bad in ("0", "-1", "x", "mgsm=", "mgsm=0", "=3", "mgsm=1.5"):
        with pytest.raises(ValueError, match="max-connections"):
            parse_max_connections([bad], known)


def test_the_builtin_per_task_concurrency_and_global_default_are_pinned():
    # 24 < the ~23 full-length sequences the server's KV cache holds, plus one: see DECISIONS.md
    assert TASK_MAX_CONNECTIONS == {"aime2025": 24}
    assert DEFAULT_MAX_CONNECTIONS == 32


@pytest.mark.parametrize(
    ("bare", "per_task", "expected"),
    [
        (None, {}, {"mgsm": 32, "aime2025": 24}),  # global default; built-in beats the default
        (8, {}, {"mgsm": 8, "aime2025": 24}),  # a bare value does not override the built-in
        (96, {}, {"mgsm": 96, "aime2025": 24}),  # the run_accuracy.sh case
        (8, {"aime2025": 96}, {"mgsm": 8, "aime2025": 96}),  # explicit TASK=N beats the built-in
        (8, {"mgsm": 5}, {"mgsm": 5, "aime2025": 24}),  # explicit beats bare
        (None, {"mgsm": 5, "aime2025": 7}, {"mgsm": 5, "aime2025": 7}),
    ],
)
def test_resolve_max_connections_precedence(bare, per_task, expected):
    assert resolve_max_connections(["mgsm", "aime2025"], bare, per_task) == expected


def test_resolve_max_connections_only_covers_the_tasks_asked_for():
    assert resolve_max_connections(["ifeval"], None, {"mgsm": 5}) == {"ifeval": 32}


def test_the_default_timeout_is_four_hours_and_the_help_says_why():
    parser = _build_parser()
    assert DEFAULT_TIMEOUT == 14400
    assert parser.parse_args(["--out", "x"]).timeout == 14400
    text = " ".join(parser.format_help().split())
    assert "queue" in text  # why: it must exceed worst-case generation incl. server queueing
    assert "aime2025=24" in text  # the built-in per-task concurrency is documented


def test_sha256_of_ids_depends_on_the_ids_not_their_order():
    assert sha256_of_ids(["a", "b"]) == sha256_of_ids(["b", "a"])
    assert sha256_of_ids(["a", "b"]) != sha256_of_ids(["a", "c"])


# --------------------------------------------------------------------------- rows


def _sample(**kw):
    base = {
        "id": "q1",
        "epoch": 2,
        "input": "2+2?",
        "target": "4",
        "metadata": {"language": "fr"},
        "output": ModelOutput.from_content("olala", "4"),
        "scores": {"match": Score(value="C", answer="4")},
    }
    base.update(kw)
    return EvalSample(**base)


def test_a_correct_sample_becomes_a_row_with_numeric_score():
    row = row_from_sample(_sample(), task="toy", primary_metric=None, stratum_field="language")
    assert row["task"] == "toy"
    assert row["sample_id"] == "q1"
    assert row["epoch"] == 2
    assert row["stratum"] == "fr"
    assert row["score"] == 1.0
    assert row["error"] is None
    assert row["truncated"] is False
    assert row["completion"] == "4"
    assert row["target"] == "4"


def test_an_incorrect_sample_scores_zero():
    s = _sample(scores={"match": Score(value="I")})
    assert row_from_sample(s, "toy", None, None)["score"] == 0.0


def test_dict_valued_scores_use_the_named_primary_metric():
    s = _sample(scores={"ifeval": Score(value={"prompt_level_strict": 1, "prompt_level_loose": 1})})
    row = row_from_sample(s, "ifeval", "prompt_level_strict", None)
    assert row["score"] == 1.0
    assert row["scores"]["ifeval"] == {"prompt_level_strict": 1, "prompt_level_loose": 1}
    s2 = _sample(
        scores={"ifeval": Score(value={"prompt_level_strict": 0, "prompt_level_loose": 1})}
    )
    assert row_from_sample(s2, "ifeval", "prompt_level_strict", None)["score"] == 0.0


def test_dict_valued_scores_without_a_primary_metric_are_an_error():
    s = _sample(scores={"x": Score(value={"a": 1})})
    with pytest.raises(ValueError, match="primary_metric"):
        row_from_sample(s, "x", None, None)


def test_a_request_error_is_recorded_and_has_no_score():
    err = EvalError(message="ConnectError: boom", traceback="tb", traceback_ansi="tb")
    row = row_from_sample(_sample(scores=None, error=err), "toy", None, None)
    assert row["error"] == "ConnectError: boom"
    assert row["score"] is None


@pytest.mark.parametrize("reason", ["max_tokens", "model_length"])
def test_length_stops_are_flagged_as_truncations(reason):
    out = ModelOutput.from_content("olala", "partial")
    out.choices[0].stop_reason = reason
    row = row_from_sample(_sample(output=out), "toy", None, None)
    assert row["truncated"] is True
    assert row["stop_reason"] == reason


def test_a_normal_stop_is_not_a_truncation():
    out = ModelOutput.from_content("olala", "ok")
    out.choices[0].stop_reason = "stop"
    assert row_from_sample(_sample(output=out), "toy", None, None)["truncated"] is False


def test_token_usage_is_copied_into_the_row():
    out = ModelOutput.from_content("olala", "ok")
    out.usage = ModelUsage(input_tokens=11, output_tokens=7, total_tokens=18)
    row = row_from_sample(_sample(output=out), "toy", None, None)
    assert (row["input_tokens"], row["output_tokens"]) == (11, 7)


# --------------------------------------------------------------------------- git


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_git_info_of_a_repository_without_commits(tmp_path):
    _git(tmp_path, "init", "-q")
    info = git_info(tmp_path)
    assert info["commit"] is None
    assert "no commit" in info["note"]


def test_git_info_reports_the_commit_and_dirtiness(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "f.txt").write_text("x")
    _git(tmp_path, "add", "f.txt")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    info = git_info(tmp_path)
    assert len(info["commit"]) == 40
    assert info["dirty"] is False
    (tmp_path / "f.txt").write_text("changed")
    assert git_info(tmp_path)["dirty"] is True


def test_git_info_outside_a_repository_does_not_crash(tmp_path):
    info = git_info(tmp_path)
    assert info["commit"] is None


# --------------------------------------------------------------------------- attempts


def _model_event(retries=None, error=None):
    from inspect_ai.event import ModelEvent
    from inspect_ai.model import GenerateConfig

    return ModelEvent(
        model="olala",
        input=[],
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=ModelOutput.from_content("olala", "4"),
        retries=retries,
        error=error,
    )


def test_a_sample_answered_at_the_first_request_has_one_attempt():
    row = row_from_sample(_sample(events=[_model_event()]), "toy", None, None)
    assert row["model_attempts"] == 1


def test_attempts_count_the_requests_including_the_retries_inside_one_model_event():
    # inspect_ai logs the openai client's retries as ModelEvent.retries, not as more events
    row = row_from_sample(_sample(events=[_model_event(retries=2)]), "toy", None, None)
    assert row["model_attempts"] == 3


def test_attempts_add_up_over_the_model_events_of_a_sample():
    events = [_model_event(retries=2, error="Request timed out."), _model_event(retries=1)]
    row = row_from_sample(_sample(events=events), "toy", None, None)
    assert row["model_attempts"] == 5  # 1 + 2 requests, then 1 + 1


def test_events_that_are_not_model_calls_do_not_count_as_attempts():
    from inspect_ai.event import InfoEvent

    events = [InfoEvent(data="x"), _model_event(), InfoEvent(data="y")]
    row = row_from_sample(_sample(events=events), "toy", None, None)
    assert row["model_attempts"] == 1


def test_a_sample_with_no_model_event_has_zero_attempts():
    assert row_from_sample(_sample(events=[]), "toy", None, None)["model_attempts"] == 0


def test_the_working_time_of_the_sample_is_copied_and_may_be_null():
    row = row_from_sample(_sample(working_time=123.5), "toy", None, None)
    assert row["working_time"] == 123.5
    assert row_from_sample(_sample(working_time=None), "toy", None, None)["working_time"] is None
