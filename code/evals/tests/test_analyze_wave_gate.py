"""Tests for analyze_wave.py, part 1: argument syntax, loading a run directory and the gate."""

import json
from pathlib import Path

import pytest
from fake_runs import SMALL_EXPECTED, build_run, sha256_of_ids, task_item_ids

import analyze_wave as aw
import run_tasks


def run(tmp_path: Path, name: str, **kw):
    return aw.load_run_dir(build_run(tmp_path / name, **kw))


def has(problems: list[str], text: str) -> bool:
    return any(text in p for p in problems)


# ---------------------------------------------------------------------------------- syntax


def test_a_run_spec_names_an_arm_and_one_or_several_directories():
    assert aw.parse_run_spec("B0=a,b/c") == ("B0", [Path("a"), Path("b/c")])
    assert aw.parse_run_spec("F1=x") == ("F1", [Path("x")])


@pytest.mark.parametrize("bad", ["B0", "=a", "B0=", "B0=a,,b", ""])
def test_a_malformed_run_spec_is_refused(bad):
    with pytest.raises(ValueError, match="run"):
        aw.parse_run_spec(bad)


def test_the_default_expected_counts_are_the_protocols_1783_samples():
    exp = aw.DEFAULT_EXPECTED
    assert {t: v["items"] for t, v in exp.items()} == {
        "mgsm": 550,
        "humaneval_plus": 164,
        "mbpp_plus": 378,
        "ifeval": 541,
        "aime2025": 30,
    }
    assert exp["aime2025"]["epochs"] == 5
    assert all(v["epochs"] == 1 for t, v in exp.items() if t != "aime2025")
    assert aw.total_samples(exp) == 1783


def test_expected_counts_can_be_read_from_a_json_file(tmp_path):
    f = tmp_path / "exp.json"
    f.write_text(json.dumps(SMALL_EXPECTED))
    assert aw.load_expected(f) == SMALL_EXPECTED
    assert aw.load_expected(None) == aw.DEFAULT_EXPECTED


def test_a_malformed_expected_file_is_refused(tmp_path):
    f = tmp_path / "exp.json"
    f.write_text(json.dumps({"mgsm": {"items": 5}}))
    with pytest.raises(ValueError, match="expected"):
        aw.load_expected(f)


def test_the_items_hash_formula_is_the_harnesss():
    ids = ["b", "a", "c"]
    assert aw.sha256_of_ids(ids) == run_tasks.sha256_of_ids(ids) == sha256_of_ids(ids)


# ---------------------------------------------------------------------------------- loading


def test_loading_a_run_records_the_sha256_of_its_files(tmp_path):
    r = run(tmp_path, "r", quant_recipe="fp8_per_channel")
    for key, name in (("run_info", "run_info.json"), ("samples", "samples.jsonl")):
        import hashlib

        assert r.sha256[key] == hashlib.sha256((tmp_path / "r" / name).read_bytes()).hexdigest()
    assert r.quantization["recipe"]["name"] == "fp8_per_channel"
    assert r.problems == []


def test_a_run_without_quantization_info_has_none(tmp_path):
    assert run(tmp_path, "r").quantization is None


@pytest.mark.parametrize("name", ["samples.jsonl", "items.json", "run_info.json"])
def test_a_missing_file_is_a_problem_not_a_crash(tmp_path, name):
    r = run(tmp_path, "r", skip_files=(name,))
    assert has(r.problems, name)
    assert has(aw.check_run(r, SMALL_EXPECTED), name)


def test_a_missing_directory_is_a_problem(tmp_path):
    r = aw.load_run_dir(tmp_path / "nowhere")
    assert has(r.problems, "not a directory")


# ---------------------------------------------------------------------------------- one run


def test_a_complete_run_passes_the_gate(tmp_path):
    assert aw.check_run(run(tmp_path, "r"), SMALL_EXPECTED) == []


@pytest.mark.parametrize("status", ["incomplete", "failed", "running"])
def test_a_run_whose_status_is_not_complete_is_refused(tmp_path, status):
    assert has(aw.check_run(run(tmp_path, "r", status=status), SMALL_EXPECTED), "status")


def test_a_missing_request_is_refused_and_named(tmp_path):
    r = run(tmp_path, "r", drop_rows={("mbpp_plus", "mbpp_plus-003", 1)})
    problems = aw.check_run(r, SMALL_EXPECTED)
    assert has(problems, "mbpp_plus")
    assert has(problems, "mbpp_plus-003")
    assert has(problems, "missing")


def test_a_missing_epoch_of_an_aime_problem_is_a_missing_request(tmp_path):
    r = run(tmp_path, "r", drop_rows={("aime2025", "aime2025-002", 4)})
    assert has(aw.check_run(r, SMALL_EXPECTED), "aime2025-002")


def test_a_duplicated_row_is_refused(tmp_path):
    r = run(tmp_path, "r", duplicate_rows={("ifeval", "ifeval-001", 1)})
    assert has(aw.check_run(r, SMALL_EXPECTED), "duplicate")


def test_a_task_with_the_wrong_number_of_items_is_refused(tmp_path):
    other = {**SMALL_EXPECTED, "mgsm": {"items": 24, "epochs": 1}}
    r = run(tmp_path, "r", expected=other)
    assert has(aw.check_run(r, SMALL_EXPECTED), "mgsm")


def test_a_task_that_is_absent_is_refused(tmp_path):
    other = {k: v for k, v in SMALL_EXPECTED.items() if k != "aime2025"}
    r = run(tmp_path, "r", expected=other)
    assert has(aw.check_run(r, SMALL_EXPECTED), "aime2025")


def test_tasks_outside_the_expected_set_are_ignored(tmp_path):
    other = {**SMALL_EXPECTED, "gsm8k": {"items": 5, "epochs": 1}}
    r = run(tmp_path, "r", expected=other)
    assert aw.check_run(r, SMALL_EXPECTED) == []


def test_epochs_other_than_expected_are_refused(tmp_path):
    other = {**SMALL_EXPECTED, "aime2025": {"items": 6, "epochs": 3}}
    r = run(tmp_path, "r", expected=other)
    assert has(aw.check_run(r, SMALL_EXPECTED), "aime2025")


def test_a_row_without_error_and_without_score_is_refused(tmp_path):
    r = run(tmp_path, "r", scores={("mgsm", "mgsm-en-00", 1): None})
    assert has(aw.check_run(r, SMALL_EXPECTED), "unscored")


def test_run_info_counts_that_disagree_with_the_rows_are_refused(tmp_path):
    r = run(tmp_path, "r", task_info_overrides={"mgsm": {"truncations": 3}})
    assert has(aw.check_run(r, SMALL_EXPECTED), "truncations")
    r2 = run(tmp_path, "r2", task_info_overrides={"mgsm": {"request_errors": 2}})
    assert has(aw.check_run(r2, SMALL_EXPECTED), "request_errors")


def test_a_recorded_items_hash_that_does_not_match_items_json_is_refused(tmp_path):
    r = run(tmp_path, "r", task_info_overrides={"ifeval": {"items_sha256": "0" * 64}})
    assert has(aw.check_run(r, SMALL_EXPECTED), "items_sha256")


ONE_THOUSAND = {
    "mgsm": {"items": 400, "epochs": 1},
    "humaneval_plus": {"items": 150, "epochs": 1},
    "mbpp_plus": {"items": 150, "epochs": 1},
    "ifeval": {"items": 150, "epochs": 1},
    "aime2025": {"items": 150, "epochs": 1},
}


def errors_in(n: int) -> set[tuple[str, str, int]]:
    return {
        ("mgsm", aw_id, 1)
        for aw_id in [
            f"mgsm-{lang}-{k:02d}" for k in range(10) for lang in ("en", "fr", "de", "ja")
        ][:n]
    }


def test_request_errors_at_exactly_half_a_percent_pass_and_above_it_fail(tmp_path):
    ok = run(tmp_path, "ok", expected=ONE_THOUSAND, error_rows=errors_in(5))
    assert aw.check_run(ok, ONE_THOUSAND) == []
    bad = run(tmp_path, "bad", expected=ONE_THOUSAND, error_rows=errors_in(6))
    problems = aw.check_run(bad, ONE_THOUSAND)
    assert has(problems, "request errors")
    assert has(problems, "0.5%")


# ---------------------------------------------------------------------------------- a pair


def pair(tmp_path, a_kw=None, b_kw=None, exp=SMALL_EXPECTED):
    a = [run(tmp_path, "a", label="B0", **(a_kw or {}))]
    b = [run(tmp_path, "b", label="F1", run_seed=1, **(b_kw or {}))]
    return aw.gate_pair(a, b, exp)


def test_two_complete_runs_on_the_same_items_pass_the_gate(tmp_path):
    gate = pair(tmp_path)
    assert gate.problems == []
    assert gate.ok


def test_model_endpoint_and_arm_label_may_differ(tmp_path):
    gate = pair(tmp_path, b_kw={"info_overrides": {"model": "other", "endpoint": "http://z/v1"}})
    assert gate.problems == []


@pytest.mark.parametrize(
    ("override", "needle"),
    [
        ({"sampling.top_k": 50}, "top_k"),
        ({"sampling.temperature": 0.0}, "temperature"),
        ({"sampling.top_p": 1.0}, "top_p"),
        ({"sampling.max_tokens": 4096}, "max_tokens"),
        ({"sampling.seed": 7}, "sampling.seed"),
        ({"reasoning_effort": "high"}, "reasoning_effort"),
        ({"chat_template_kwargs": {"reasoning_effort": "high"}}, "chat_template_kwargs"),
        ({"sandbox.mode": "docker"}, "sandbox"),
        ({"seed": 3}, "seed"),
    ],
)
def test_settings_that_must_not_differ_make_the_pair_invalid(tmp_path, override, needle):
    gate = pair(tmp_path, b_kw={"info_overrides": override})
    assert not gate.ok
    assert has(gate.problems, needle)


def test_different_item_sets_make_the_pair_invalid_even_with_the_right_count(tmp_path):
    swapped = {"humaneval_plus": [f"humaneval_plus-{i:03d}" for i in range(1, 11)]}
    gate = pair(tmp_path, b_kw={"items_override": swapped})
    assert not gate.ok
    assert has(gate.problems, "humaneval_plus")
    assert has(gate.problems, "item sets differ")


def test_different_epochs_make_the_pair_invalid(tmp_path):
    a = [run(tmp_path, "a")]
    b = [run(tmp_path, "b", expected={**SMALL_EXPECTED, "aime2025": {"items": 6, "epochs": 3}})]
    gate = aw.gate_pair(a, b, SMALL_EXPECTED)
    assert not gate.ok
    assert has(gate.problems, "epochs")


def test_a_failing_run_inside_a_pair_is_named_by_directory(tmp_path):
    gate = pair(tmp_path, b_kw={"status": "incomplete"})
    assert not gate.ok
    assert has(gate.problems, str(tmp_path / "b"))


def test_a_code_hash_or_version_difference_is_a_warning_not_a_refusal(tmp_path):
    gate = pair(
        tmp_path,
        b_kw={"info_overrides": {"code_sha256": "d" * 64, "versions": {"python": "3.13"}}},
    )
    assert gate.ok
    assert has(gate.warnings, "code_sha256")
    assert has(gate.warnings, "versions")


def test_concurrency_and_timeout_are_not_part_of_the_must_match_list():
    assert not any("max_connections" in k or "timeout" in k for k in aw.MUST_MATCH)


def test_a_difference_in_max_connections_per_task_is_a_warning_naming_the_task(tmp_path):
    gate = pair(
        tmp_path,
        a_kw={"info_overrides": {"max_connections_per_task": {"mgsm": 32, "aime2025": 24}}},
        b_kw={"info_overrides": {"max_connections_per_task": {"mgsm": 32, "aime2025": 96}}},
    )
    assert gate.ok
    (warning,) = [w for w in gate.warnings if "max_connections_per_task" in w]
    assert "aime2025" in warning
    assert "24" in warning
    assert "96" in warning
    assert not has(gate.warnings, "mgsm")


def test_equal_concurrency_and_timeout_give_no_warning(tmp_path):
    same = {"info_overrides": {"max_connections_per_task": {"aime2025": 24}}}
    gate = pair(tmp_path, a_kw=same, b_kw=same)
    assert gate.ok
    assert gate.warnings == []


def test_a_difference_in_the_timeout_is_a_warning_not_a_refusal(tmp_path):
    gate = pair(tmp_path, b_kw={"info_overrides": {"sampling.timeout": 14400}})
    assert gate.ok
    (warning,) = gate.warnings
    assert "timeout" in warning
    assert "3600" in warning
    assert "14400" in warning


def test_a_run_without_max_connections_per_task_falls_back_to_its_global_value(tmp_path):
    # a legacy run (max_connections 96, no per-task field) against a run that used 24 on AIME
    gate = pair(
        tmp_path,
        b_kw={"info_overrides": {"max_connections_per_task": {"aime2025": 24, "mgsm": 96}}},
    )
    assert gate.ok
    warnings = [w for w in gate.warnings if "max_connections_per_task" in w]
    assert len(warnings) == 1
    assert "aime2025" in warnings[0]
    assert "mgsm" not in warnings[0]


def test_a_concurrency_difference_on_a_task_a_run_did_not_do_is_not_reported(tmp_path):
    gate = pair(
        tmp_path,
        a_kw={"info_overrides": {"max_connections_per_task": {"aime2025": 24}}},
        b_kw={"info_overrides": {"max_connections_per_task": {"aime2025": 24, "other": 5}}},
    )
    assert gate.warnings == []


def test_pooled_runs_must_agree_with_each_other_too(tmp_path):
    a = [
        run(tmp_path, "a1", run_seed=1),
        run(tmp_path, "a2", run_seed=2, info_overrides={"sampling.top_k": 20}),
    ]
    b = [run(tmp_path, "b")]
    gate = aw.gate_pair(a, b, SMALL_EXPECTED)
    assert not gate.ok
    assert has(gate.problems, "top_k")


def test_the_same_directory_listed_twice_in_an_arm_is_refused(tmp_path):
    r = run(tmp_path, "a1")
    gate = aw.gate_pair([r, aw.load_run_dir(tmp_path / "a1")], [run(tmp_path, "b")], SMALL_EXPECTED)
    assert not gate.ok
    assert has(gate.problems, "same run")


def test_the_other_arm_is_compared_with_every_run_of_a_pooled_arm(tmp_path):
    a = [run(tmp_path, "a1", run_seed=1), run(tmp_path, "a2", run_seed=2)]
    b = [run(tmp_path, "b", run_seed=3, info_overrides={"sampling.top_k": 50})]
    gate = aw.gate_pair(a, b, SMALL_EXPECTED)
    assert not gate.ok
    assert has(gate.problems, "top_k")
    b2 = [
        run(tmp_path, "b1", run_seed=4),
        run(tmp_path, "b2", run_seed=5, info_overrides={"reasoning_effort": "high"}),
    ]
    assert has(aw.gate_pair(a, b2, SMALL_EXPECTED).problems, "reasoning_effort")


# ------------------------------------------------------------ retried requests and client timeout

LEGACY_NOTE = "retry count unknown (client timeout 600 s legacy default)"
ALL_TASKS = tuple(SMALL_EXPECTED)


def check(r, accept=False):
    """``(problems, warnings)`` of one run."""
    warnings: list[str] = []
    problems = aw.check_run(r, SMALL_EXPECTED, warnings, accept_legacy_aime_retries=accept)
    return problems, warnings


def retried(task: str, n: int, attempts: int = 2) -> dict:
    """The first ``n`` samples of ``task`` needed ``attempts`` requests."""
    return {(task, sid, 1): attempts for sid in task_item_ids(task, n)}


def test_a_run_whose_samples_were_all_answered_at_the_first_request_has_no_retry_problem(tmp_path):
    assert check(run(tmp_path, "r")) == ([], [])


def test_the_share_of_retried_samples_is_checked_in_each_task_and_names_the_task_and_count(
    tmp_path,
):
    r = run(tmp_path, "r", attempts=retried("mgsm", 1))  # 1 of 20 samples: 5%
    problems, warnings = check(r)
    (problem,) = problems
    assert "mgsm" in problem
    assert "1 of 20" in problem
    assert "0.5%" in problem
    assert warnings == []


def test_a_retry_in_one_task_does_not_name_the_other_tasks(tmp_path):
    problems, _ = check(run(tmp_path, "r", attempts=retried("ifeval", 2)))
    assert [p.split(":")[0] for p in problems] == ["ifeval"]
    assert "2 of 14" in problems[0]


@pytest.mark.parametrize("attempts", [2, 3, 7])
def test_any_attempt_count_above_one_makes_a_retried_sample(tmp_path, attempts):
    problems, _ = check(run(tmp_path, "r", attempts=retried("mbpp_plus", 1, attempts)))
    assert has(problems, "mbpp_plus")


def test_retries_at_exactly_half_a_percent_of_a_task_pass_and_above_it_fail(tmp_path):
    ok = run(tmp_path, "ok", expected=ONE_THOUSAND, attempts=retried("mgsm", 2))  # 2 of 400
    assert check_with(ok, ONE_THOUSAND) == []
    bad = run(tmp_path, "bad", expected=ONE_THOUSAND, attempts=retried("mgsm", 3))  # 3 of 400
    problems = check_with(bad, ONE_THOUSAND)
    assert has(problems, "3 of 400")


def check_with(r, expected):
    return aw.check_run(r, expected, [])


def test_a_single_retry_among_a_hundred_and_fifty_samples_is_above_half_a_percent(tmp_path):
    r = run(tmp_path, "r", expected=ONE_THOUSAND, attempts=retried("aime2025", 1))
    assert has(check_with(r, ONE_THOUSAND), "aime2025: 1 of 150")


def test_rows_that_carry_the_field_in_only_some_rows_of_a_task_are_refused(tmp_path):
    r = run(tmp_path, "r")
    first = next(x for x in r.rows if x["task"] == "mgsm")
    del first["model_attempts"]
    problems, _ = check(r)
    assert has(problems, "mgsm")
    assert has(problems, "some rows")


def test_legacy_rows_of_a_task_give_a_warning_and_no_problem(tmp_path):
    r = run(tmp_path, "r", legacy_tasks=("mgsm", "ifeval"))
    problems, warnings = check(r)
    assert problems == []
    assert sorted(w.split(":")[0] for w in warnings) == ["ifeval", "mgsm"]
    assert all(LEGACY_NOTE in w for w in warnings)


def test_legacy_rows_of_aime_are_a_problem_unless_the_option_accepts_them(tmp_path):
    r = run(tmp_path, "r", legacy_tasks=("aime2025",))
    problems, warnings = check(r)
    (problem,) = problems
    assert "aime2025" in problem
    assert LEGACY_NOTE in problem
    assert "--accept-legacy-aime-retries" in problem
    assert warnings == []


def test_the_option_turns_the_legacy_aime_problem_into_a_warning_that_states_the_measure(tmp_path):
    r = run(tmp_path, "r", legacy_tasks=("aime2025",))
    problems, warnings = check(r, accept=True)
    assert problems == []
    (warning,) = warnings
    assert warning.startswith("aime2025")
    assert LEGACY_NOTE in warning
    assert "3 to 9 percent" in warning
    assert "--accept-legacy-aime-retries" in warning


def test_the_option_does_not_excuse_retries_that_a_new_aime_run_records(tmp_path):
    r = run(tmp_path, "r", attempts=retried("aime2025", 1))
    problems, _ = check(r, accept=True)
    assert has(problems, "aime2025: 1 of 30")


def test_the_option_changes_nothing_for_the_other_tasks(tmp_path):
    r = run(tmp_path, "r", legacy_tasks=("mgsm",), attempts=retried("ifeval", 1))
    assert check(r, accept=False) == check(r, accept=True)


def test_a_fully_legacy_run_is_refused_only_because_of_aime_and_warns_for_the_other_tasks(tmp_path):
    r = run(tmp_path, "r", legacy=True)
    problems, warnings = check(r)
    assert [p.split(":")[0] for p in problems] == ["aime2025"]
    assert sorted(w.split(":")[0] for w in warnings) == [
        "humaneval_plus",
        "ifeval",
        "mbpp_plus",
        "mgsm",
    ]
    problems, warnings = check(r, accept=True)
    assert problems == []
    assert sorted(w.split(":")[0] for w in warnings) == sorted(ALL_TASKS)


def test_legacy_rows_of_a_task_that_is_not_expected_are_not_reported(tmp_path):
    only_mgsm = {"mgsm": SMALL_EXPECTED["mgsm"]}
    r = run(tmp_path, "r", expected=only_mgsm, legacy=True)
    warnings: list[str] = []
    assert aw.check_run(r, only_mgsm, warnings) == []
    assert [w.split(":")[0] for w in warnings] == ["mgsm"]


def test_a_gate_pair_reports_retry_warnings_with_the_run_directory_and_the_option(tmp_path):
    a = [run(tmp_path, "a", legacy=True)]
    b = [run(tmp_path, "b", run_seed=1)]
    refused = aw.gate_pair(a, b, SMALL_EXPECTED)
    assert not refused.ok
    assert has(refused.problems, str(tmp_path / "a"))
    assert has(refused.problems, "aime2025")
    accepted = aw.gate_pair(a, b, SMALL_EXPECTED, accept_legacy_aime_retries=True)
    assert accepted.ok
    assert sum(LEGACY_NOTE in w and str(tmp_path / "a") in w for w in accepted.warnings) == 5
    assert not has(accepted.warnings, str(tmp_path / "b"))


@pytest.mark.parametrize(
    ("client", "timeout", "bad"),
    [(3599, 3600, True), (3600, 3600, False), (14400, 3600, False), (600, 14400, True)],
)
def test_a_client_timeout_below_the_runs_timeout_is_a_problem(tmp_path, client, timeout, bad):
    r = run(
        tmp_path,
        "r",
        info_overrides={"sampling.timeout": timeout},
        client_timeout=client,
    )
    problems, _ = check(r)
    assert has(problems, "client_timeout") is bad
    if bad:
        assert has(problems, str(client))
        assert has(problems, str(timeout))


def test_a_task_client_timeout_below_the_runs_timeout_is_a_problem_naming_the_task(tmp_path):
    r = run(tmp_path, "r", task_info_overrides={"aime2025": {"client_timeout": 600}})
    problems, _ = check(r)
    (problem,) = problems
    assert "aime2025" in problem
    assert "600" in problem


def test_a_run_without_a_client_timeout_is_not_refused_for_it(tmp_path):
    r = run(tmp_path, "r", legacy_tasks=("mgsm",), info_overrides={"sampling.timeout": 14400})
    r.info.pop("client_timeout")
    problems, _ = check(r)
    assert not has(problems, "client_timeout")


def test_run_info_retry_counters_that_disagree_with_the_rows_are_refused(tmp_path):
    r = run(tmp_path, "r", task_info_overrides={"mgsm": {"samples_with_retries": 1}})
    assert has(check(r)[0], "samples_with_retries")
    r2 = run(tmp_path, "r2", task_info_overrides={"mgsm": {"max_working_time": 5.0}})
    assert has(check(r2)[0], "max_working_time")


def test_a_legacy_task_with_run_info_counters_is_not_cross_checked(tmp_path):
    r = run(
        tmp_path,
        "r",
        legacy_tasks=("mgsm",),
        task_info_overrides={"mgsm": {"samples_with_retries": 4}},
    )
    problems, _ = check(r)
    assert not has(problems, "samples_with_retries")


def test_a_run_wide_client_timeout_below_the_runs_timeout_is_a_problem_on_its_own(tmp_path):
    r = run(tmp_path, "r", info_overrides={"client_timeout": 600})  # the task entries stay at 3600
    problems, _ = check(r)
    (problem,) = problems
    assert problem.startswith("run_info:")
    assert "600" in problem
    assert "3600" in problem
