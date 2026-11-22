"""Tests for analyze_wave.py, part 3: flags, secondary analyses and the analysis of a whole wave."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from fake_runs import (
    SMALL_EXPECTED,
    build_run,
    det,
    edge_fn,
    make,
    ref_fn,
    same_fn,
    task_item_ids,
    worse_fn,
)

import analyze_wave as aw


def cfg(tmp_path, **over):
    base = {
        "runs": {
            "B0": [make(tmp_path, "b0a", run_seed=1), make(tmp_path, "b0b", run_seed=2)],
            "F1": [make(tmp_path, "f1", same_fn, run_seed=3)],
            "F2": [make(tmp_path, "f2", worse_fn, run_seed=4)],
            "N2": [make(tmp_path, "n2", edge_fn, run_seed=5)],
        },
        "reference": "B0",
        "candidates": ["F1", "F2", "N2"],
        "expected": SMALL_EXPECTED,
        "n_boot": 400,
        "seed": 0,
    }
    base.update(over)
    return aw.Config(**base)


def has(problems, text):
    return any(text in p for p in problems)


# ---------------------------------------------------------------------------------- truncations


ONE_THOUSAND = {
    "mgsm": {"items": 400, "epochs": 1},
    "humaneval_plus": {"items": 150, "epochs": 1},
    "mbpp_plus": {"items": 150, "epochs": 1},
    "ifeval": {"items": 150, "epochs": 1},
    "aime2025": {"items": 150, "epochs": 1},
}
EVEN = {t: {"items": 200, "epochs": 1} for t in ONE_THOUSAND}


def trunc_rows(task, n, expected=ONE_THOUSAND):
    return {(task, i, 1) for i in task_item_ids(task, expected[task]["items"])[:n]}


def tarm(tmp_path, name, truncated, expected=ONE_THOUSAND):
    r = aw.load_run_dir(
        build_run(tmp_path / name, expected=expected, label=name, truncated_rows=truncated)
    )
    return aw.build_arm(name, [r], list(expected))


def test_rates_differing_by_exactly_one_point_overall_and_per_task_are_not_flagged(tmp_path):
    two_each = set().union(*(trunc_rows(t, 2, EVEN) for t in EVEN))
    x = tarm(tmp_path, "x", two_each, EVEN)
    y = tarm(tmp_path, "y", set(), EVEN)
    t = aw.truncation_summary(x, y, list(EVEN))
    assert t["x"]["truncations"] == 10
    assert t["x"]["rate_pp"] == pytest.approx(1.0)
    assert t["diff_pp"] == pytest.approx(1.0)
    assert t["flagged_scopes"] == []
    assert not t["flag"]


def test_a_rate_difference_above_one_point_is_flagged_in_either_direction(tmp_path):
    x = tarm(tmp_path, "x", trunc_rows("mgsm", 11) | trunc_rows("ifeval", 10))
    y = tarm(tmp_path, "y", set())
    fwd = aw.truncation_summary(x, y, list(ONE_THOUSAND))
    assert fwd["flag"]
    assert fwd["diff_pp"] == pytest.approx(2.1)
    assert "overall" in fwd["flagged_scopes"]
    assert aw.truncation_summary(y, x, list(ONE_THOUSAND))["flag"]


def test_a_task_level_truncation_difference_is_flagged_even_if_the_overall_one_is_small(tmp_path):
    x = tarm(tmp_path, "x", trunc_rows("aime2025", 3))  # 2 points of 150 samples, 0.3 overall
    y = tarm(tmp_path, "y", set())
    t = aw.truncation_summary(x, y, list(ONE_THOUSAND))
    assert t["diff_pp"] == pytest.approx(0.3)
    assert t["flagged_scopes"] == ["aime2025"]
    assert t["flag"]


def test_truncations_are_counted_over_every_run_of_a_pooled_arm(tmp_path):
    r1 = aw.load_run_dir(
        build_run(tmp_path / "r1", expected=ONE_THOUSAND, truncated_rows=trunc_rows("mgsm", 4))
    )
    r2 = aw.load_run_dir(
        build_run(
            tmp_path / "r2", expected=ONE_THOUSAND, run_seed=1, truncated_rows=trunc_rows("mgsm", 8)
        )
    )
    x = aw.build_arm("x", [r1, r2], list(ONE_THOUSAND))
    y = tarm(tmp_path, "y", set())
    t = aw.truncation_summary(x, y, list(ONE_THOUSAND))
    assert t["x"]["samples"] == 2000
    assert t["x"]["truncations"] == 12
    assert t["x"]["rate_pp"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------------- AIME secondary


def aime_runs(tmp_path):
    ids = task_item_ids("aime2025", 2)
    exp = {"aime2025": {"items": 2, "epochs": 5}}
    xs = {("aime2025", ids[0], e): 1.0 for e in range(1, 6)}
    xs |= {("aime2025", ids[1], e): 0.0 for e in range(1, 6)}
    x = aw.load_run_dir(
        build_run(
            tmp_path / "x",
            expected=exp,
            scores=xs,
            # two of the five correct answers of problem 0 were cut off
            truncated_rows={("aime2025", ids[0], 1), ("aime2025", ids[0], 2)},
        )
    )
    ys = {("aime2025", ids[0], e): 1.0 for e in range(1, 6)}
    ys |= {("aime2025", ids[1], e): 0.0 for e in range(1, 6)}
    y = aw.load_run_dir(build_run(tmp_path / "y", expected=exp, scores=ys))
    return aw.build_arm("x", [x], ["aime2025"]), aw.build_arm("y", [y], ["aime2025"])


def test_aime_accuracy_among_non_truncated_samples_sits_next_to_the_headline(tmp_path):
    x, y = aime_runs(tmp_path)
    headline, _ = aw.compare_arms(x, y, ["aime2025"], n_boot=50, seed=0)
    sec = aw.aime_non_truncated(x, y, n_boot=200, seed=0)
    # headline: truncations are failures, (3/5 + 0) / 2 = 30%
    assert headline["tasks"]["aime2025"]["x"]["value"] == pytest.approx(30.0)
    # secondary: 3 correct of 8 samples that were not cut off
    assert sec["x"]["value"] == pytest.approx(100 * 3 / 8)
    assert sec["y"]["value"] == pytest.approx(50.0)
    assert sec["diff"]["value"] == pytest.approx(100 * (3 / 8 - 1 / 2))
    assert sec["x"]["truncated"] == 2
    assert sec["x"]["non_truncated"] == 8
    assert sec["y"]["truncated"] == 0
    assert sec["diff"]["lo"] <= sec["diff"]["value"] <= sec["diff"]["hi"]


def test_a_request_error_stays_a_failure_in_the_secondary_aime_accuracy(tmp_path):
    ids = task_item_ids("aime2025", 1)
    exp = {"aime2025": {"items": 1, "epochs": 5}}
    scores = {("aime2025", ids[0], e): 1.0 for e in range(1, 6)}
    r = aw.load_run_dir(
        build_run(tmp_path / "x", expected=exp, scores=scores, error_rows={("aime2025", ids[0], 3)})
    )
    arm = aw.build_arm("x", [r], ["aime2025"])
    sec = aw.aime_non_truncated(arm, arm, n_boot=20, seed=0)
    assert sec["x"]["value"] == pytest.approx(80.0)


def test_the_secondary_aime_analysis_is_absent_without_aime(tmp_path):
    exp = {"mgsm": {"items": 4, "epochs": 1}}
    r = aw.load_run_dir(build_run(tmp_path / "x", expected=exp))
    arm = aw.build_arm("x", [r], ["mgsm"])
    assert aw.aime_non_truncated(arm, arm, n_boot=20, seed=0) is None


# ---------------------------------------------------------------------------------- MGSM languages


def test_the_mgsm_table_has_one_row_per_language_with_its_own_interval(tmp_path):
    exp = {"mgsm": {"items": 20, "epochs": 1}}
    ids = task_item_ids("mgsm", 20)
    xs = {("mgsm", i, 1): float(not ("-fr-" in i and int(i[-2:]) < 3)) for i in ids}
    x = aw.load_run_dir(build_run(tmp_path / "x", expected=exp, scores=xs))
    y = aw.load_run_dir(build_run(tmp_path / "y", expected=exp, scores=dict.fromkeys(xs, 1.0)))
    rows = aw.mgsm_languages(
        aw.build_arm("x", [x], ["mgsm"]), aw.build_arm("y", [y], ["mgsm"]), n_boot=300, seed=0
    )
    assert list(rows) == ["de", "en", "fr", "ja"]
    assert rows["fr"]["n_items"] == 5
    assert rows["fr"]["diff"]["value"] == pytest.approx(-60.0)
    assert rows["en"]["diff"]["value"] == pytest.approx(0.0)
    assert rows["fr"]["x"]["value"] == pytest.approx(40.0)
    assert rows["fr"]["diff"]["lo"] <= -60.0 <= rows["fr"]["diff"]["hi"]


# ---------------------------------------------------------------------------------- task flag


def summary_with(diffs: dict[str, tuple[float, float, float]]) -> dict:
    return {
        "tasks": {t: {"diff": {"value": v, "lo": lo, "hi": hi}} for t, (v, lo, hi) in diffs.items()}
    }


def test_a_task_whose_interval_upper_bound_is_below_minus_five_is_a_regression():
    s = summary_with({"a": (-9.0, -12.0, -5.5), "b": (-4.0, -9.0, 1.0), "c": (-5.0, -7.0, -5.0)})
    assert aw.task_level_regressions(s) == ["a"]


def test_a_large_average_loss_with_a_wide_interval_is_not_a_task_regression():
    assert aw.task_level_regressions(summary_with({"a": (-12.0, -30.0, 4.0)})) == []


# ---------------------------------------------------------------------------------- the wave


def test_a_whole_wave_gets_each_outcome_label(tmp_path):
    res = aw.analyze(cfg(tmp_path))
    out = {n: c["outcome"] for n, c in res["candidates"].items()}
    assert out == {"F1": "NON-INFERIOR", "F2": "WORSE", "N2": "INCONCLUSIVE"}
    f1 = res["candidates"]["F1"]["summary"]["macro"]["diff"]
    assert f1["value"] == pytest.approx(0.0)


def test_the_primary_margin_and_the_sensitivity_margins_are_reported(tmp_path):
    res = aw.analyze(cfg(tmp_path))
    assert res["confirmatory"]["primary_margin"] == -3.0
    assert res["confirmatory"]["margins"] == [-3.0, -2.0, -5.0]
    assert set(res["confirmatory"]["tests"]) == {"-3", "-2", "-5"}
    n2 = {m: t["N2"]["outcome"] for m, t in res["confirmatory"]["tests"].items()}
    assert n2["-3"] == "INCONCLUSIVE"
    f2 = res["confirmatory"]["tests"]["-3"]["F2"]
    assert set(f2) >= {"p_raw", "p_holm", "lower_bound_one_sided", "interval", "outcome"}


def test_the_margin_option_changes_the_primary_test(tmp_path):
    res = aw.analyze(cfg(tmp_path, margin=-60.0))
    assert res["confirmatory"]["primary_margin"] == -60.0
    assert res["candidates"]["F2"]["outcome"] != "WORSE"


def test_the_holm_family_is_the_candidates_and_not_the_exploratory_arms(tmp_path):
    base = aw.analyze(cfg(tmp_path / "w1"))
    c = cfg(tmp_path / "w2")
    c.runs["F1n"] = [make(tmp_path / "w2", "f1n", worse_fn, run_seed=9)]
    c.exploratory = ["F1n"]
    res = aw.analyze(c)
    for name in ("F1", "F2", "N2"):
        assert res["confirmatory"]["tests"]["-3"][name] == base["confirmatory"]["tests"]["-3"][name]
    assert "F1n" not in res["confirmatory"]["tests"]["-3"]
    assert "outcome" not in res["exploratory"]["F1n"]
    assert "tests" not in res["exploratory"]["F1n"]
    assert res["exploratory"]["F1n"]["summary"]["macro"]["diff"]["value"] < -40


def test_a_failing_gate_on_a_candidate_refuses_and_names_the_reason(tmp_path):
    c = cfg(tmp_path)
    c.runs["F2"] = [
        build_run(
            tmp_path / "f2bad",
            expected=SMALL_EXPECTED,
            scores=det(SMALL_EXPECTED, worse_fn),
            drop_rows={("ifeval", "ifeval-004", 1)},
        )
    ]
    with pytest.raises(aw.GateRefused) as exc:
        aw.analyze(c)
    assert has(exc.value.problems, "ifeval-004")
    assert has(exc.value.problems, "F2")


def test_request_errors_above_half_a_percent_invalidate_the_pair(tmp_path):
    c = cfg(tmp_path)
    c.runs["F1"] = [
        build_run(
            tmp_path / "f1bad",
            expected=SMALL_EXPECTED,
            scores=det(SMALL_EXPECTED, same_fn),
            error_rows={("mgsm", "mgsm-en-00", 1)},
        )
    ]
    with pytest.raises(aw.GateRefused) as exc:
        aw.analyze(c)
    assert has(exc.value.problems, "request errors")


def test_a_failing_gate_on_an_exploratory_arm_only_warns_and_computes_on_common_items(tmp_path):
    c = cfg(tmp_path)
    c.runs["F1n"] = [
        build_run(
            tmp_path / "f1n",
            expected=SMALL_EXPECTED,
            scores=det(SMALL_EXPECTED, worse_fn),
            drop_rows={("mbpp_plus", "mbpp_plus-001", 1)},
        )
    ]
    c.exploratory = ["F1n"]
    res = aw.analyze(c)
    gate = res["gate"]["pairs"]["F1n vs B0"]
    assert gate["status"] == "warn-exploratory"
    assert has(gate["problems"], "mbpp_plus-001")
    assert res["exploratory"]["F1n"]["summary"]["tasks"]["mbpp_plus"]["n_items"] == 11
    assert res["gate"]["pairs"]["F1 vs B0"]["status"] == "pass"


def test_pairwise_contrasts_between_candidates_are_descriptive(tmp_path):
    res = aw.analyze(cfg(tmp_path, contrasts=[("F2", "F1")]))
    con = res["contrasts"]["F2 vs F1"]
    assert con["summary"]["macro"]["diff"]["value"] == pytest.approx(
        res["candidates"]["F2"]["summary"]["macro"]["diff"]["value"]
        - res["candidates"]["F1"]["summary"]["macro"]["diff"]["value"]
    )
    assert "outcome" not in con
    assert "tests" not in con


def test_the_calibration_pair_reports_its_interval_and_width_without_touching_the_tests(tmp_path):
    base = aw.analyze(cfg(tmp_path / "w1"))
    c = cfg(tmp_path / "w2")
    a, b = c.runs["B0"]
    c.calibration = (("B0a", a), ("B0b", b))
    res = aw.analyze(c)
    cal = res["calibration"]
    assert cal["contains_zero"] is True
    assert cal["summary"]["macro"]["diff"]["value"] == pytest.approx(0.0)
    lo, hi = cal["summary"]["macro"]["diff"]["lo"], cal["summary"]["macro"]["diff"]["hi"]
    assert cal["width_pp"] == pytest.approx(hi - lo)
    assert res["confirmatory"] == base["confirmatory"]


def test_a_calibration_interval_that_excludes_zero_is_reported_as_such(tmp_path):
    c = cfg(tmp_path)
    c.calibration = (
        ("B0a", make(tmp_path, "cala", ref_fn)),
        ("B0b", make(tmp_path, "calb", worse_fn)),
    )
    res = aw.analyze(c)
    assert res["calibration"]["contains_zero"] is False


def test_the_calibration_runs_must_pass_the_gate_too(tmp_path):
    c = cfg(tmp_path)
    c.calibration = (
        ("B0a", make(tmp_path, "cala", ref_fn)),
        ("B0b", make(tmp_path, "calb", ref_fn, status="incomplete")),
    )
    with pytest.raises(aw.GateRefused):
        aw.analyze(c)


def test_task_level_regression_and_truncation_flags_are_attached_to_the_candidates(tmp_path):
    c = cfg(tmp_path)
    trunc = {("aime2025", i, e) for i in task_item_ids("aime2025", 6) for e in (1, 2, 3, 4, 5)}
    c.runs["N2"] = [
        build_run(
            tmp_path / "n2t",
            expected=SMALL_EXPECTED,
            scores=det(SMALL_EXPECTED, ref_fn),
            truncated_rows=trunc,
        )
    ]
    res = aw.analyze(c)
    n2 = res["candidates"]["N2"]
    assert n2["flags"]["truncation_rate_difference"] is True
    assert "aime2025" in n2["flags"]["task_level_regression"]
    assert res["candidates"]["F1"]["flags"]["task_level_regression"] == []
    assert res["candidates"]["F1"]["flags"]["truncation_rate_difference"] is False
    # truncations stay failures in the headline
    assert n2["summary"]["tasks"]["aime2025"]["x"]["value"] == pytest.approx(0.0)


def test_the_reference_is_pooled_over_its_runs(tmp_path):
    c = cfg(tmp_path)
    one = make(tmp_path, "b0c", lambda t, k, e: k % 4 == 0 or k % 4 == 1, run_seed=8)
    c.runs["B0"] = [c.runs["B0"][0], one]
    res = aw.analyze(c)
    # reference item scores are averaged over the two runs: 0.75 and 0.5 -> 0.625 on average
    ref = res["candidates"]["F1"]["summary"]["tasks"]["mgsm"]["y"]["value"]
    assert ref == pytest.approx(62.5)
    assert res["arms"]["B0"]["n_runs"] == 2


def test_provenance_hashes_every_run_file_and_the_script(tmp_path):
    c = cfg(tmp_path)
    res = aw.analyze(c)
    first = res["arms"]["B0"]["runs"][0]
    d = Path(first["dir"])
    assert (
        first["run_info_sha256"] == hashlib.sha256((d / "run_info.json").read_bytes()).hexdigest()
    )
    assert first["samples_sha256"] == hashlib.sha256((d / "samples.jsonl").read_bytes()).hexdigest()
    assert res["script"]["sha256"] == hashlib.sha256(Path(aw.__file__).read_bytes()).hexdigest()
    assert res["bootstrap"] == {"seed": 0, "resamples": 400, "confidence": 0.95}


def test_every_run_has_its_settings_and_its_failure_counts(tmp_path):
    c = cfg(tmp_path)
    c.runs["N2"] = [
        build_run(
            tmp_path / "n2q",
            expected=SMALL_EXPECTED,
            scores=det(SMALL_EXPECTED, ref_fn),
            quant_recipe="nvfp4_rtn",
            truncated_rows={("mgsm", "mgsm-en-00", 1)},
        )
    ]
    res = aw.analyze(c)
    run = res["arms"]["N2"]["runs"][0]
    assert run["settings"]["reasoning_effort"] == "medium"
    assert run["settings"]["sampling"]["temperature"] == 0.8
    assert run["settings"]["sandbox"] == "local"
    assert run["settings"]["epochs"]["aime2025"] == 5
    assert run["settings"]["git_commit"] == "abc123"
    assert run["settings"]["code_sha256"] == "c" * 64
    assert run["quantization_recipe"] == "nvfp4_rtn"
    assert run["counts"]["truncations"] == 1
    assert run["counts"]["request_errors"] == 0
    assert run["counts"]["samples"] == aw.total_samples(SMALL_EXPECTED)
    assert res["arms"]["B0"]["runs"][0]["quantization_recipe"] is None


def test_unknown_arm_names_are_refused_before_any_computation(tmp_path):
    with pytest.raises(ValueError, match="F9"):
        aw.analyze(cfg(tmp_path / "a", candidates=["F1", "F9"]))
    with pytest.raises(ValueError, match="reference"):
        aw.analyze(cfg(tmp_path / "b", reference="B9"))
    with pytest.raises(ValueError, match="both"):
        aw.analyze(cfg(tmp_path / "c", exploratory=["F1"]))


def test_the_result_is_json_serialisable(tmp_path):
    res = aw.analyze(cfg(tmp_path))
    json.dumps(res)
    assert not np.isnan(res["candidates"]["F2"]["summary"]["macro"]["diff"]["value"])


def test_the_seed_and_the_resamples_reach_every_bootstrap_of_a_pair(tmp_path, monkeypatch):
    calls = {}

    def spy(name):
        real = getattr(aw, name)

        def wrapper(*args, **kw):
            calls.setdefault(name, []).append((args, kw))
            return real(*args, **kw)

        return wrapper

    for name in ("compare_arms", "aime_non_truncated", "mgsm_languages"):
        monkeypatch.setattr(aw, name, spy(name))
    aw.analyze(cfg(tmp_path, seed=7, n_boot=123))
    for name in ("compare_arms", "aime_non_truncated", "mgsm_languages"):
        assert calls[name], name
        for args, kw in calls[name]:
            assert 123 in args or kw.get("n_boot") == 123, (name, args, kw)
            assert 7 in args or kw.get("seed") == 7, (name, args, kw)


def test_a_wave_without_truncation_has_no_truncation_scope_flagged_per_task(tmp_path):
    res = aw.analyze(cfg(tmp_path))
    t = res["candidates"]["F1"]["truncation"]
    assert t["flagged_scopes"] == []
    assert t["by_task_diff_pp"]["aime2025"] == pytest.approx(0.0)


def test_a_per_task_truncation_scope_is_named_in_the_candidates_flags(tmp_path):
    c = cfg(tmp_path)
    trunc = {("mbpp_plus", i, 1) for i in task_item_ids("mbpp_plus", 12)[:2]}
    c.runs["N2"] = [
        build_run(
            tmp_path / "n2t",
            expected=SMALL_EXPECTED,
            scores=det(SMALL_EXPECTED, ref_fn),
            truncated_rows=trunc,
        )
    ]
    res = aw.analyze(c)
    assert "mbpp_plus" in res["candidates"]["N2"]["flags"]["truncation_scopes"]


def test_the_calibration_width_is_the_interval_width_and_the_names_are_kept(tmp_path):
    c = cfg(tmp_path)
    a, b = c.runs["B0"]
    c.calibration = (("B0a", a), ("B0b", b))
    cal = aw.analyze(c)["calibration"]
    assert (cal["x"], cal["y"]) == ("B0a", "B0b")
    d = cal["summary"]["macro"]["diff"]
    assert cal["width_pp"] == pytest.approx(d["hi"] - d["lo"])
    assert cal["width_pp"] >= 0.0


def test_the_calibration_interval_of_two_noisy_runs_is_wide_and_contains_zero(tmp_path):
    # two random runs of the same model: item scores differ by sampling noise only
    big = {
        t: {"items": 60 if t != "aime2025" else 10, "epochs": e}
        for t, e in (
            ("mgsm", 1),
            ("humaneval_plus", 1),
            ("mbpp_plus", 1),
            ("ifeval", 1),
            ("aime2025", 5),
        )
    }
    a = build_run(tmp_path / "a", expected=big, label="a", run_seed=1)
    b = build_run(tmp_path / "b", expected=big, label="b", run_seed=2)
    c = aw.Config(
        runs={"B0": [a, b], "F1": [build_run(tmp_path / "f", expected=big, run_seed=3)]},
        reference="B0",
        candidates=["F1"],
        expected=big,
        n_boot=400,
        calibration=(("B0a", a), ("B0b", b)),
    )
    cal = aw.analyze(c)["calibration"]
    assert cal["width_pp"] > 1.0
    assert cal["contains_zero"] is True
