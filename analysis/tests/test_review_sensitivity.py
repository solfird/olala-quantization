"""Tests of review_sensitivity.py on small toy arms with fixed numbers (never the real results)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

import review_sensitivity as rs

aw = rs.aw


def arm(name: str, scores: dict[str, list[float]]) -> aw.Arm:
    """An arm from per-task item-score lists; MGSM ids carry a language prefix, the others are ``t0``, ``t1``..."""
    return aw.Arm(
        name=name,
        runs=[],
        scores={t: {f"{t}-{i:03d}": float(v) for i, v in enumerate(vals)} for t, vals in scores.items()},
        strata={t: {f"{t}-{i:03d}": None for i in range(len(vals))} for t, vals in scores.items()},
    )


def toy(seed: int, shift: float = 0.0, sizes: dict[str, int] | None = None) -> aw.Arm:
    """Random 0/1 scores; AIME items are means over five epochs."""
    sizes = sizes or {"mgsm": 40, "humaneval_plus": 20, "mbpp_plus": 24, "ifeval": 30, "aime2025": 6}
    g = np.random.default_rng(seed)
    scores = {}
    for t, n in sizes.items():
        p = min(0.95, max(0.05, 0.7 + shift))
        scores[t] = (g.random(n) < p).astype(float) if t != "aime2025" else (g.random((n, 5)) < p).mean(axis=1)
    return arm(f"toy{seed}", scores)


X, Y = toy(1), toy(2, shift=-0.1)
SIZES = {"mgsm": 40, "humaneval_plus": 20, "mbpp_plus": 24, "ifeval": 30, "aime2025": 6}


# ----------------------------------------------------------------------------- the report's bootstrap is the one used


def test_contrast_is_the_report_function_unchanged():
    ref, _ = aw.compare_arms(X, Y, rs.TASKS, rs.N_BOOT, rs.SEED)
    got = rs.contrast(X, Y)
    assert got["macro"] == ref["macro"]["diff"]
    assert got["tasks"]["aime2025"] == ref["tasks"]["aime2025"]["diff"]
    assert (rs.N_BOOT, rs.SEED) == (10_000, 0)


def test_leave_aime_out_is_the_macro_of_four_tasks():
    four = tuple(t for t in rs.TASKS if t != rs.AIME)
    c = rs.contrast(X, Y, four)
    assert c["macro"]["value"] == pytest.approx(np.mean([c["tasks"][t]["value"] for t in four]))
    assert rs.AIME not in c["tasks"]


def test_a_gap_that_sits_in_aime_alone_disappears_without_it():
    scores = {t: [1.0] * n for t, n in SIZES.items()}
    better = arm("better", scores)
    worse = arm("worse", {**scores, "aime2025": [0.0] * SIZES["aime2025"]})
    four = tuple(t for t in rs.TASKS if t != rs.AIME)
    with_aime = rs.contrast(better, worse)["macro"]
    without = rs.contrast(better, worse, four)["macro"]
    assert with_aime["value"] == pytest.approx(20.0)  # 100 points on one task of five
    assert (without["value"], without["lo"], without["hi"]) == (0.0, 0.0, 0.0)


# ----------------------------------------------------------------------------- generous rescoring


def row(task: str, sid: str, epoch: int, score: float, truncated: bool = False) -> dict:
    return {"task": task, "sample_id": sid, "epoch": epoch, "score": score, "error": None, "truncated": truncated}


def test_late_counts_reads_the_audit_task_names_and_counts_per_item():
    audit = [
        {"task": "inspect_evals/aime2025", "sample_id": "3", "target_late": True},
        {"task": "inspect_evals/aime2025", "sample_id": "3", "target_late": True},
        {"task": "inspect_evals/aime2025", "sample_id": "3", "target_late": False},
        {"task": "mbpp-plus", "sample_id": 7, "target_late": True},
        {"task": "inspect_evals/mgsm", "sample_id": "de_x", "target_late": False},
    ]
    assert rs.late_counts(audit) == {("aime2025", "3"): 2, ("mbpp_plus", "7"): 1}


def test_credit_turns_only_the_named_number_of_truncated_rows_into_right_answers_and_leaves_the_input_alone():
    # one AIME problem, five samples: two truncated (scored 0), one right and two wrong that were finished
    rows = [row("aime2025", "3", e, s, t) for e, s, t in [(1, 1, False), (2, 0, False), (3, 0, False), (4, 1, True), (5, 0, True)]]
    run = aw.Run(path=Path("."), rows=list(rows))
    before = aw.build_arm("a", [run], ["aime2025"]).scores["aime2025"]["3"]
    credited = rs.credit_runs([run], {("aime2025", "3"): 1})
    after = aw.build_arm("a", credited, ["aime2025"]).scores["aime2025"]["3"]
    assert before == pytest.approx(0.2)  # the truncated 1 was already a failure
    assert after == pytest.approx(0.4)
    assert run.rows == rows  # the original run is untouched
    both = aw.build_arm("a", rs.credit_runs([run], {("aime2025", "3"): 2}), ["aime2025"]).scores["aime2025"]["3"]
    assert both == pytest.approx(0.6)


def test_crediting_more_rows_than_the_run_has_truncated_is_refused():
    run = aw.Run(path=Path("."), rows=[row("mgsm", "de_1", 1, 0.0, True), row("mgsm", "de_2", 1, 0.0, False)])
    with pytest.raises(ValueError, match="not truncated"):
        rs.credit_runs([run], {("mgsm", "de_2"): 1})
    with pytest.raises(ValueError, match="not truncated"):
        rs.credit_runs([run], {("mgsm", "de_1"): 2})


def test_generous_macro_gain_is_credited_over_items_over_five_tasks():
    # the formula of build_data.generous: sum over tasks of credited / items, times 100, over the five tasks
    tasks = {"mgsm": 4, "humaneval_plus": 2, "mbpp_plus": 2, "ifeval": 2, "aime2025": 2}
    rows = [row(t, f"{t}{i}", 1, 0.0, truncated=(t == "mgsm" and i < 2)) for t, n in tasks.items() for i in range(n)]
    rows = [r for r in rows if r["task"] != "aime2025"] + [row("aime2025", f"p{i}", e, 0.0, truncated=(i == 0 and e <= 2)) for i in range(2) for e in range(1, 6)]
    run = aw.Run(path=Path("."), rows=rows)
    strict = aw.build_arm("s", [run], rs.TASKS)
    credit = {("mgsm", "mgsm0"): 1, ("mgsm", "mgsm1"): 1, ("aime2025", "p0"): 2}
    gen = aw.build_arm("g", rs.credit_runs([run], credit), rs.TASKS)
    macro = lambda a: np.mean([np.mean(list(a.scores[t].values())) for t in rs.TASKS]) * 100  # noqa: E731
    assert macro(strict) == 0.0
    assert macro(gen) == pytest.approx((2 / 4 + 2 / 10) / 5 * 100)


# ----------------------------------------------------------------------------- MGSM clusters


def test_the_same_problem_in_two_languages_is_one_cluster_and_numbers_count_with_repeats():
    de = rs.cluster_key("Löse dies. Dan pflanzt 3 Rosenbüsche, jeder trägt 25 Rosen, 25 Euro.", "600")
    fr = rs.cluster_key("Résous. Dan plante 25 rosiers, chacun a 3 roses et 25 euros.", " 600 ")
    assert de == fr
    assert rs.cluster_key("A 3 und 25 und 25", "600") != rs.cluster_key("A 3 und 25", "600")
    assert rs.cluster_key("A 3 und 25 und 25", "601") != de
    assert de == "600|25|25|3"


def test_sample_key_accepts_a_string_or_a_list_of_messages_and_a_list_target():
    as_text = rs.sample_cluster_key("Dan hat 3 Rosen", ["7"])
    as_messages = rs.sample_cluster_key([{"content": "Dan hat"}, {"content": "3 Rosen"}], "7")
    assert as_text == as_messages == "7|3"


def mgsm_arm(diffs: list[float]) -> tuple[aw.Arm, aw.Arm]:
    """Two arms of five tasks where only MGSM differs, by ``diffs`` per item; every other task is equal."""
    sizes = {"mgsm": len(diffs), "humaneval_plus": 10, "mbpp_plus": 10, "ifeval": 10, "aime2025": 3}
    base = {t: [0.5] * n for t, n in sizes.items()}
    x = arm("x", {**base, "mgsm": [0.5 + d / 2 for d in diffs]})
    y = arm("y", {**base, "mgsm": [0.5 - d / 2 for d in diffs]})
    return x, y


def test_clusters_of_one_item_give_exactly_the_item_bootstrap():
    x, y = toy(3), toy(4, shift=-0.2)
    alone = {i: i for i in x.scores["mgsm"]}
    got = rs.clustered_macro_ci(x, y, rs.TASKS, alone)
    ref = rs.contrast(x, y)["macro"]
    assert got == pytest.approx(ref, abs=1e-12)


def test_items_that_come_in_identical_pairs_widen_the_interval_when_resampled_together():
    pattern = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0] * 5
    diffs = [d for d in pattern for _ in (0, 1)]  # every problem translated twice with the same outcome
    x, y = mgsm_arm(diffs)
    ids = sorted(x.scores["mgsm"])
    clusters = {i: f"problem{j // 2}" for j, i in enumerate(ids)}
    item = rs.contrast(x, y)["macro"]
    clu = rs.clustered_macro_ci(x, y, rs.TASKS, clusters)
    assert clu["value"] == pytest.approx(item["value"])
    assert (clu["hi"] - clu["lo"]) > 1.2 * (item["hi"] - item["lo"])  # about sqrt(2) wider


# ----------------------------------------------------------------------------- margins


def test_margin_states_pass_open_and_worse():
    ci = {"value": -2.5, "lo": -4.0, "hi": -1.0}
    assert rs.margin_states(ci, -5.0)["state"] == "pass"  # lower end above -5
    assert rs.margin_states(ci, -3.0)["state"] == "open"
    assert rs.margin_states(ci, -0.5)["state"] == "worse"  # whole interval below the margin
    assert rs.margin_states({"value": 0, "lo": -2.0, "hi": 1}, -2.0)["lo_above"] is False  # a lower end on the margin does not pass


def test_margin_row_flags_a_task_wholly_below_minus_five():
    good = arm("good", {t: [1.0] * n for t, n in SIZES.items()})
    bad = arm("bad", {**{t: [1.0] * n for t, n in SIZES.items()}, "aime2025": [0.0] * SIZES["aime2025"]})
    summary, _ = aw.compare_arms(bad, good, rs.TASKS, 200, 0)
    r = rs.margin_row(summary)
    assert [f["task"] for f in r["task_flag"]] == ["aime2025"]
    assert r["task_flag"][0]["value"] == pytest.approx(-100.0)
    assert r["margins"]["-5"]["state"] == "worse"  # the macro is -20 and its whole interval is below -5
    assert r["margins"]["-2"]["lo_above"] is False
    assert rs.margin_row(aw.compare_arms(good, good, rs.TASKS, 200, 0)[0])["task_flag"] == []


# ----------------------------------------------------------------------------- KL


def topk(probs: list[list[float]], ids: list[list[int]], n_chunks: int = 2, n_pos: int = 3, vocab: int = 8) -> tuple:
    """A top-k container where every position holds the same distribution."""
    tid = np.tile(np.array(ids, dtype=np.int32)[None], (n_chunks, n_pos, 1)) if len(ids) == 1 else np.array(ids, dtype=np.int32)
    tlp = np.log(np.tile(np.array(probs, dtype=np.float32)[None], (n_chunks, n_pos, 1)))
    return np.array([f"token_id:{i}" for i in range(vocab)]), tid, tlp


def test_per_chunk_kl_matches_the_hand_computed_value_and_is_zero_for_the_same_model():
    ref = topk([[0.6, 0.3]], [[0, 1]])
    var = topk([[0.5, 0.4]], [[0, 1]])
    kl = rs.per_chunk_kl(ref, var, 20)
    assert kl.shape == (2,)
    assert kl == pytest.approx([0.6 * math.log(0.6 / 0.5) + 0.3 * math.log(0.3 / 0.4)] * 2, abs=1e-6)  # the 0.1 left over matches
    assert rs.per_chunk_kl(ref, ref, 20) == pytest.approx([0.0, 0.0], abs=1e-9)


def test_per_chunk_kl_with_a_smaller_k_uses_only_that_many_tokens_and_a_bucket_for_the_rest():
    ref = topk([[0.6, 0.3]], [[0, 1]])
    var = topk([[0.5, 0.4]], [[0, 1]])
    kl1 = rs.per_chunk_kl(ref, var, 1)
    assert kl1 == pytest.approx([0.6 * math.log(0.6 / 0.5) + 0.4 * math.log(0.4 / 0.5)] * 2, abs=1e-6)


def test_per_chunk_kl_maps_two_vocabularies_to_the_same_tokens():
    # the same distribution written with different local vocabularies and index orders has KL zero
    a = (np.array(["x", "y", "z"]), np.tile(np.array([[0, 1]], dtype=np.int32)[None], (1, 2, 1)), np.log(np.tile(np.array([[0.6, 0.3]], dtype=np.float32)[None], (1, 2, 1))))
    b = (np.array(["z", "y", "x"]), np.tile(np.array([[2, 1]], dtype=np.int32)[None], (1, 2, 1)), np.log(np.tile(np.array([[0.6, 0.3]], dtype=np.float32)[None], (1, 2, 1))))
    assert rs.per_chunk_kl(a, b, 20) == pytest.approx([0.0], abs=1e-9)


def test_kl_summary_gives_the_paired_difference_its_percent_and_its_interval():
    g = np.random.default_rng(0)
    base = 0.02 + 0.004 * g.random(64)
    per_chunk = {"N5": base, "C1": base + 0.002, "C2": base - 0.001}
    s = rs.kl_summary(per_chunk, n_boot=2000, seed=0)
    d = s["diffs"]["plain_minus_exp"]
    assert (d["x"], d["y"]) == ("C1", "N5")
    assert d["value"] == pytest.approx(0.002) and d["lo"] == pytest.approx(0.002) and d["hi"] == pytest.approx(0.002)  # a constant shift has no spread
    assert d["pct"] == pytest.approx(100 * 0.002 / base.mean())
    assert s["diffs"]["search_minus_exp"]["value"] == pytest.approx(-0.001)
    assert s["diffs"]["search_minus_plain"]["value"] == pytest.approx(-0.003)
    ref = stats_ci(per_chunk["N5"], per_chunk["C1"], 2000)
    assert (d["value"], d["lo"], d["hi"]) == ref
    assert s["means"]["N5"]["lo"] < s["means"]["N5"]["value"] < s["means"]["N5"]["hi"]


def stats_ci(a: np.ndarray, b: np.ndarray, n_boot: int) -> tuple:
    return rs.stats.paired_bootstrap_diff_ci(a, b, n_boot, 0)


def test_order_is_lowest_first():
    assert rs.order_of({"N5": 0.0244, "C1": 0.0263, "C2": 0.0234}) == ["C2", "N5", "C1"]


# ----------------------------------------------------------------------------- reproduction check


def write_analysis(results: Path, summary: dict) -> None:
    (results / "wave2-test-analysis.json").write_text(
        json.dumps({"reference": "B0", "candidates": {"SEL": {"summary": summary}}, "exploratory": {}, "contrasts": {}, "calibration": None})
    )


def test_the_check_accepts_the_stored_interval_and_refuses_a_different_one(tmp_path):
    arms = {"B0": Y, "N6": X}
    summary, _ = aw.compare_arms(X, Y, rs.TASKS, rs.N_BOOT, rs.SEED)
    write_analysis(tmp_path, summary)
    assert rs.check_reproduction(arms, tmp_path) == 1
    summary["macro"]["diff"]["lo"] += 0.01
    write_analysis(tmp_path, summary)
    with pytest.raises(SystemExit, match="cannot reproduce"):
        rs.check_reproduction(arms, tmp_path)
