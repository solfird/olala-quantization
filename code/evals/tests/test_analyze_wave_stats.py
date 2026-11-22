"""Tests for analyze_wave.py, part 2: item scores, pooling, the bootstrap, Holm and the labels."""

import numpy as np
import pytest
from fake_runs import SMALL_EXPECTED, build_run

import analyze_wave as aw
import stats


def arm_from(name: str, scores: dict[str, list[float]]) -> aw.Arm:
    """An arm built from per-task item-score lists (ids ``i0``, ``i1``, ...)."""
    return aw.Arm(
        name=name,
        runs=[],
        scores={t: {f"i{k}": float(v) for k, v in enumerate(vals)} for t, vals in scores.items()},
        strata={t: {f"i{k}": None for k in range(len(vals))} for t, vals in scores.items()},
    )


def load(tmp_path, name, **kw):
    return aw.load_run_dir(build_run(tmp_path / name, **kw))


# ---------------------------------------------------------------------------------- item scores


def test_an_item_score_is_the_mean_over_its_epochs_with_failures_counted_as_zero(tmp_path):
    ids = [f"aime2025-{i:03d}" for i in range(2)]
    scores = {("aime2025", ids[0], e): s for e, s in zip(range(1, 6), [1, 1, 1, 0, 0], strict=True)}
    scores |= {("aime2025", ids[1], e): 1.0 for e in range(1, 6)}
    r = load(
        tmp_path,
        "r",
        expected={"aime2025": {"items": 2, "epochs": 5}},
        scores=scores,
        # the scorer said 1 but the answer was cut off, so it is a failure: 2 of 5 left for item 1
        truncated_rows={("aime2025", ids[0], 2), ("aime2025", ids[1], 3)},
        error_rows={("aime2025", ids[1], 5)},
    )
    item, _ = aw.run_item_scores(r, "aime2025")
    assert item[ids[0]] == pytest.approx(2 / 5)
    assert item[ids[1]] == pytest.approx(3 / 5)


def test_strata_come_from_the_rows(tmp_path):
    r = load(tmp_path, "r")
    _, strata = aw.run_item_scores(r, "mgsm")
    assert strata["mgsm-fr-02"] == "fr"


def test_pooling_averages_the_item_scores_of_the_runs_of_an_arm(tmp_path):
    ids = [f"aime2025-{i:03d}" for i in range(2)]
    exp = {"aime2025": {"items": 2, "epochs": 5}}
    s1 = {("aime2025", ids[0], e): float(e <= 2) for e in range(1, 6)}  # 0.4
    s1 |= {("aime2025", ids[1], e): 0.0 for e in range(1, 6)}  # 0.0
    s2 = {("aime2025", ids[0], e): 1.0 for e in range(1, 6)}  # 1.0
    s2 |= {("aime2025", ids[1], e): float(e == 1) for e in range(1, 6)}  # 0.2
    r1 = load(tmp_path, "r1", expected=exp, scores=s1)
    r2 = load(tmp_path, "r2", expected=exp, scores=s2)
    arm = aw.build_arm("B0", [r1, r2], ["aime2025"])
    assert arm.scores["aime2025"][ids[0]] == pytest.approx(0.7)
    assert arm.scores["aime2025"][ids[1]] == pytest.approx(0.1)
    single = aw.build_arm("B0", [r1], ["aime2025"])
    assert single.scores["aime2025"][ids[0]] == pytest.approx(0.4)


# ---------------------------------------------------------------------------------- Holm


def test_holm_adjusts_p_values_step_down_and_keeps_them_monotone():
    adj = aw.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


def test_holm_caps_at_one_and_handles_one_test_and_none():
    assert aw.holm_adjust({"a": 0.6, "b": 0.7}) == pytest.approx({"a": 1.0, "b": 1.0})
    assert aw.holm_adjust({"a": 0.02}) == {"a": 0.02}
    assert aw.holm_adjust({}) == {}


def test_holm_does_not_depend_on_the_order_of_the_tests():
    p = {"a": 0.001, "b": 0.02, "c": 0.5}
    assert aw.holm_adjust(p) == aw.holm_adjust(dict(reversed(list(p.items()))))


# ---------------------------------------------------------------------------------- labels


@pytest.mark.parametrize(
    ("p_holm", "hi", "expected"),
    [
        (0.02, -1.0, "NON-INFERIOR"),
        (0.025, 0.0, "NON-INFERIOR"),  # at alpha the null is rejected
        (0.03, 0.0, "INCONCLUSIVE"),
        (0.9, -3.5, "WORSE"),
        (0.9, -3.0, "INCONCLUSIVE"),  # upper bound equal to the margin is not below it
        (1.0, 0.4, "INCONCLUSIVE"),
    ],
)
def test_outcome_labels(p_holm, hi, expected):
    assert aw.outcome_label(p_holm, hi, margin=-3.0, alpha=0.025) == expected


# ---------------------------------------------------------------------------------- resampling


def test_resample_indices_are_reproducible_in_range_and_depend_on_seed_and_task():
    a = aw.resample_indices(0, "mgsm", 50, 100)
    assert a.shape == (100, 50)
    assert a.min() >= 0
    assert a.max() < 50
    assert np.array_equal(a, aw.resample_indices(0, "mgsm", 50, 100))
    assert not np.array_equal(a, aw.resample_indices(1, "mgsm", 50, 100))
    assert not np.array_equal(a, aw.resample_indices(0, "ifeval", 50, 100))


def test_interval_is_the_percentile_interval_of_the_draws():
    d = np.random.default_rng(0).normal(size=5000)
    lo, hi = aw.interval(d)
    assert (lo, hi) == pytest.approx(tuple(np.quantile(d, [0.025, 0.975])))


# ---------------------------------------------------------------------------------- compare_arms


def test_the_macro_average_gives_each_task_the_same_weight():
    big = [1.0] * 100
    small = [1.0] * 4
    x = arm_from("x", {"big": big, "small": [0.0] * 4})
    y = arm_from("y", {"big": big, "small": small})
    res, _ = aw.compare_arms(x, y, ["big", "small"], n_boot=50, seed=0)
    # item-weighted this would be about +3.8 points; macro is (0 + 100) / 2
    assert res["macro"]["diff"]["value"] == pytest.approx(-50.0)
    assert res["tasks"]["small"]["diff"]["value"] == pytest.approx(-100.0)
    assert res["tasks"]["big"]["diff"]["value"] == pytest.approx(0.0)


def test_accuracies_are_percent_and_differences_are_x_minus_y_in_points():
    x = arm_from("x", {"t": [1, 1, 0, 0]})
    y = arm_from("y", {"t": [1, 0, 0, 0]})
    res, _ = aw.compare_arms(x, y, ["t"], n_boot=50, seed=0)
    assert res["tasks"]["t"]["x"]["value"] == pytest.approx(50.0)
    assert res["tasks"]["t"]["y"]["value"] == pytest.approx(25.0)
    assert res["tasks"]["t"]["diff"]["value"] == pytest.approx(25.0)


def test_a_constant_shift_has_a_degenerate_interval_because_the_pairing_is_kept():
    base = np.random.default_rng(1).uniform(0.3, 0.9, size=40)
    x = arm_from("x", {"a": base - 0.1, "b": base[:25] - 0.1})
    y = arm_from("y", {"a": base, "b": base[:25]})
    res, draws = aw.compare_arms(x, y, ["a", "b"], n_boot=500, seed=0)
    d = res["macro"]["diff"]
    assert (d["value"], d["lo"], d["hi"]) == pytest.approx((-10.0, -10.0, -10.0))
    assert np.allclose(draws["macro"], -10.0)


def test_the_task_interval_matches_an_independent_percentile_bootstrap():
    rng = np.random.default_rng(2)
    xv = rng.integers(0, 2, size=60).astype(float)
    yv = np.clip(xv + rng.choice([0, 0, 0, -1, 1], size=60), 0, 1)
    x, y = arm_from("x", {"t": xv}), arm_from("y", {"t": yv})
    res, _ = aw.compare_arms(x, y, ["t"], n_boot=20_000, seed=0)
    _, lo, hi = stats.paired_bootstrap_diff_ci(yv, xv, n_boot=20_000, seed=99)
    d = res["tasks"]["t"]["diff"]
    assert d["lo"] == pytest.approx(100 * lo, abs=0.6)
    assert d["hi"] == pytest.approx(100 * hi, abs=0.6)
    assert d["lo"] < d["value"] < d["hi"]


def test_the_macro_draws_are_the_mean_of_the_task_draws_of_the_same_resamples():
    rng = np.random.default_rng(3)
    x = arm_from("x", {"a": rng.random(30), "b": rng.random(10), "c": rng.random(17)})
    y = arm_from("y", {"a": rng.random(30), "b": rng.random(10), "c": rng.random(17)})
    _, draws = aw.compare_arms(x, y, ["a", "b", "c"], n_boot=300, seed=5)
    assert draws["macro"].shape == (300,)
    assert np.allclose(draws["macro"], (draws["a"] + draws["b"] + draws["c"]) / 3)


def test_each_task_is_resampled_independently_of_the_others():
    rng = np.random.default_rng(4)
    same = rng.random(40)
    x = arm_from("x", {"a": same, "b": same})
    y = arm_from("y", {"a": rng.random(40), "b": rng.random(40)})
    _, draws = aw.compare_arms(x, y, ["a", "b"], n_boot=4000, seed=0)
    corr = np.corrcoef(draws["a"], draws["b"])[0, 1]
    assert abs(corr) < 0.08


def test_the_comparison_is_reproducible_for_a_seed_and_records_it():
    rng = np.random.default_rng(6)
    x = arm_from("x", {"t": rng.random(20)})
    y = arm_from("y", {"t": rng.random(20)})
    first, _ = aw.compare_arms(x, y, ["t"], n_boot=200, seed=3)
    again, _ = aw.compare_arms(x, y, ["t"], n_boot=200, seed=3)
    other, _ = aw.compare_arms(x, y, ["t"], n_boot=200, seed=4)
    assert first == again
    assert first != other
    assert first["n_boot"] == 200
    assert first["seed"] == 3


def test_items_present_in_one_arm_only_are_reported_and_not_paired():
    x = arm_from("x", {"t": [1.0, 0.0, 1.0]})
    y = arm_from("y", {"t": [1.0, 1.0]})
    res, _ = aw.compare_arms(x, y, ["t"], n_boot=20, seed=0)
    assert res["tasks"]["t"]["n_items"] == 2
    assert res["tasks"]["t"]["only_x"] == 1
    assert res["tasks"]["t"]["only_y"] == 0


def test_a_missing_task_leaves_no_macro_average():
    x = arm_from("x", {"a": [1.0, 0.0]})
    y = arm_from("y", {"a": [1.0, 0.0]})
    res, _ = aw.compare_arms(x, y, ["a", "b"], n_boot=20, seed=0)
    assert res["macro"] is None
    assert "b" not in res["tasks"]


# ---------------------------------------------------------------------------------- tests


def draws_with(n_low: int, n: int = 1000, low: float = -4.0, high: float = 0.0) -> np.ndarray:
    return np.array([low] * n_low + [high] * (n - n_low))


def test_the_raw_p_value_is_the_fraction_of_resamples_at_or_below_the_margin():
    assert aw.p_value(np.array([-4.0, -3.0, -2.0, 0.0]), -3.0) == pytest.approx(0.5)


def test_holm_changes_the_decision_when_two_candidates_are_tested():
    # each has raw p = 0.02 <= 0.025, but Holm over two tests gives 0.04 > 0.025
    both = aw.confirmatory_tests(
        {"F1": draws_with(20), "N2": draws_with(20)}, margins=[-3.0], alpha=0.025
    )[-3.0]
    assert both["F1"]["p_raw"] == pytest.approx(0.02)
    assert both["F1"]["p_holm"] == pytest.approx(0.04)
    assert both["F1"]["outcome"] == "INCONCLUSIVE"
    alone = aw.confirmatory_tests({"F1": draws_with(20)}, margins=[-3.0], alpha=0.025)[-3.0]
    assert alone["F1"]["outcome"] == "NON-INFERIOR"


def test_the_one_sided_alpha_is_0_025_not_0_05():
    out = aw.confirmatory_tests({"F1": draws_with(40)}, margins=[-3.0], alpha=aw.ALPHA)[-3.0]
    assert aw.ALPHA == 0.025
    assert out["F1"]["p_raw"] == pytest.approx(0.04)
    assert out["F1"]["outcome"] == "INCONCLUSIVE"


def test_the_margin_is_a_lower_limit_so_a_positive_margin_is_not_accepted():
    with pytest.raises(ValueError, match="margin"):
        aw.confirmatory_tests({"F1": draws_with(0)}, margins=[3.0], alpha=0.025)


def test_labels_follow_the_interval_for_a_clear_pass_a_clear_fail_and_a_borderline_arm():
    rng = np.random.default_rng(0)
    cands = {
        "good": rng.normal(0.0, 0.8, 5000),
        "edge": rng.normal(-3.0, 0.4, 5000),
        "bad": rng.normal(-8.0, 0.8, 5000),
    }
    out = aw.confirmatory_tests(cands, margins=[-3.0, -2.0, -5.0], alpha=0.025)
    assert {k: v["outcome"] for k, v in out[-3.0].items()} == {
        "good": "NON-INFERIOR",
        "edge": "INCONCLUSIVE",
        "bad": "WORSE",
    }
    # the same arms at the other margins: the edge arm passes at -5 and fails at -2
    assert out[-5.0]["edge"]["outcome"] == "NON-INFERIOR"
    assert out[-2.0]["edge"]["outcome"] == "WORSE"


def test_the_one_sided_lower_bound_is_the_2_5th_percentile_of_the_macro_difference():
    d = np.random.default_rng(1).normal(-1.0, 1.0, 4000)
    out = aw.confirmatory_tests({"F1": d}, margins=[-3.0], alpha=0.025)[-3.0]["F1"]
    assert out["lower_bound_one_sided"] == pytest.approx(np.quantile(d, 0.025))
    assert out["interval"] == pytest.approx(list(np.quantile(d, [0.025, 0.975])))


def test_the_expected_small_protocol_is_five_tasks():
    assert list(SMALL_EXPECTED) == ["mgsm", "humaneval_plus", "mbpp_plus", "ifeval", "aime2025"]
