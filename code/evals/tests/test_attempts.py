"""Tests for attempts.py: what a set of rows says about retried requests and working time."""

from attempts import apply_run_counters, retry_stats, run_client_timeout


def row(attempts=..., working=...):
    out = {"task": "t", "sample_id": "a", "epoch": 1}
    if attempts is not ...:
        out["model_attempts"] = attempts
    if working is not ...:
        out["working_time"] = working
    return out


def test_legacy_rows_say_nothing_about_retries_or_working_time():
    assert retry_stats([row(), row()]) == {}
    assert retry_stats([]) == {}


def test_a_sample_is_retried_when_it_needed_more_than_one_attempt():
    stats = retry_stats([row(1, 1.0), row(2, 5.5), row(3, None), row(1, 2.0)])
    assert stats == {"samples_with_retries": 2, "max_working_time": 5.5}


def test_exactly_one_attempt_is_not_a_retry_and_zero_attempts_is_not_either():
    assert retry_stats([row(1, 1.0), row(0, 1.0)])["samples_with_retries"] == 0


def test_the_max_working_time_is_null_when_no_row_has_one():
    assert retry_stats([row(1, None), row(1, None)]) == {
        "samples_with_retries": 0,
        "max_working_time": None,
    }


def test_only_rows_that_carry_a_field_are_counted_for_it():
    stats = retry_stats([row(), row(2, 9.0), row(1, 3.0)])
    assert stats == {"samples_with_retries": 1, "max_working_time": 9.0}
    only_time = retry_stats([row(working=4.0), row(working=7.0)])
    assert only_time == {"max_working_time": 7.0}  # no row says how many attempts it needed


def test_the_run_wide_client_timeout_is_the_smallest_one_if_every_task_states_it():
    tasks = {"a": {"client_timeout": 14400}, "b": {"client_timeout": 3600}}
    assert run_client_timeout(tasks) == 3600


def test_no_run_wide_client_timeout_if_a_task_does_not_state_it():
    assert run_client_timeout({"a": {"client_timeout": 14400}, "b": {}}) is None
    assert run_client_timeout({"a": {"client_timeout": None}}) is None
    assert run_client_timeout({}) is None


def test_applying_run_counters_replaces_stale_values_and_removes_what_rows_do_not_say():
    info = {
        "samples_with_retries": 50,
        "max_working_time": 1e9,
        "client_timeout": 1,
        "tasks": {"a": {"client_timeout": 7200}},
    }
    apply_run_counters(info, [row(2, 3.0), row(1, 1.0)])
    assert info["samples_with_retries"] == 1
    assert info["max_working_time"] == 3.0
    assert info["client_timeout"] == 7200
    apply_run_counters(info, [row(), row()])
    assert "samples_with_retries" not in info
    assert "max_working_time" not in info
    info["tasks"]["b"] = {}
    apply_run_counters(info, [])
    assert "client_timeout" not in info
