"""Tests of b0g_control.py on small toy arms with fixed numbers (never the real results).

Mutation checks made by hand (2026-11-22): each of these was introduced in b0g_control.py, the tests run, the failing test noted, the change undone
(14 mutations, all killed; the first run of the leave-AIME-out mutation survived and the macro assertion in the last test was added for it).

 1. ``absolute``: the B0 row read from the first arm of its comparison (B0g) instead of the second     -> test_absolute_gives_each_arm_its_own_score
 2. ``double_difference``: ``(a - b) - (c - d)`` written as ``(a - b) + (c - d)``                       -> the three test_double_difference_* tests
 3. ``decomposition``: the engine part with its sign flipped                                            -> test_decomposition_*
 4. ``check_b0g``: the hash comparison dropped                                                         -> test_the_b0g_check_refuses_a_wrong_hash_arm_context_or_commit
 5. ``check_b0g``: the request-error test dropped                                                      -> test_the_b0g_check_refuses_a_request_error_in_the_info_or_in_a_row
 6. ``ENGINE`` first pair written as ``("B0", "B0g")`` (reference minus control)                         -> test_engine_effect_is_the_control_minus_the_reference
 7. ``analyse``: the leave-AIME-out macro taken over ``TASKS`` instead of ``FOUR``                      -> test_analyse_returns_every_section_in_json_form_and_renders_without_dashes
 8. ``margin_set``: every build read against B0 instead of the given reference                         -> test_margin_set_is_keyed_by_the_build_and_uses_the_given_reference
 9. ``generous_arms``: the credits of B0g applied to the B0a runs                                      -> test_generous_arms_credit_only_the_arms_own_rows_and_pool_the_bf16_runs
10. ``truncation_block``: per-task counts without the task-name normalisation                          -> test_truncation_block_counts_per_task_with_the_names_of_samples_jsonl
11. ``check_reproduction``: the minimum of three stored intervals lowered to one                       -> test_the_reproduction_check_needs_three_stored_intervals_and_refuses_a_different_one
12. ``check_reproduction``: the recomputation skipped (a fixed count returned)                         -> the same test
13. ``check_b0g``: the context-length check dropped                                                    -> test_the_b0g_check_refuses_a_wrong_hash_arm_context_or_commit
14. ``decomposition``: the format-plus-engine consistency check dropped                                -> test_decomposition_with_a_zero_total_has_no_share_and_an_inconsistent_input_is_refused
"""

import ast
import json
from pathlib import Path

import numpy as np
import pytest

import b0g_control as bc

aw, rs = bc.aw, bc.rs
SIZES = {"mgsm": 40, "humaneval_plus": 20, "mbpp_plus": 24, "ifeval": 30, "aime2025": 6}
NB = 300  # resamples in the tests that do not need the report's 10,000


def arm(name: str, scores: dict[str, list[float]]) -> aw.Arm:
    """An arm from per-task item-score lists."""
    return aw.Arm(
        name=name,
        runs=[],
        scores={t: {f"{t}-{i:03d}": float(v) for i, v in enumerate(vals)} for t, vals in scores.items()},
        strata={t: {f"{t}-{i:03d}": None for i in range(len(vals))} for t, vals in scores.items()},
    )


def const(name: str, p: float, sizes: dict[str, int] = SIZES) -> aw.Arm:
    """An arm that scores ``p`` on every item of every task."""
    return arm(name, {t: [p] * n for t, n in sizes.items()})


def toy(seed: int, shift: float = 0.0) -> aw.Arm:
    """Random 0/1 scores; AIME items are means over five epochs."""
    g = np.random.default_rng(seed)
    p = min(0.95, max(0.05, 0.7 + shift))
    return arm(f"toy{seed}", {t: (g.random(n) < p).astype(float) if t != "aime2025" else (g.random((n, 5)) < p).mean(axis=1) for t, n in SIZES.items()})


def toy_arms() -> dict[str, aw.Arm]:
    """The nine arms the analysis needs, random but fixed."""
    shifts = {"B0": 0.0, "B0a": 0.01, "B0b": -0.01, "B0g": -0.03, "Q5": -0.04, "Q4": -0.12, "N6": -0.05, "F1": 0.0, "F2": -0.01}
    return {a: toy(i, s) for i, (a, s) in enumerate(shifts.items())}


# ----------------------------------------------------------------------------- the contrasts


def test_the_bootstrap_is_the_reports_and_the_pairs_are_in_the_stated_direction():
    assert (bc.N_BOOT, bc.SEED) == (10_000, 0)
    arms = toy_arms()
    ref, _ = aw.compare_arms(arms["Q5"], arms["B0g"], bc.TASKS, bc.N_BOOT, bc.SEED)
    got = bc.contrast(arms, "Q5", "B0g")
    assert got["macro"] == ref["macro"]["diff"]
    assert got["tasks"]["mgsm"] == ref["tasks"]["mgsm"]["diff"]
    assert bc.ENGINE == (("B0g", "B0"), ("B0g", "B0a"), ("B0g", "B0b"))
    assert bc.FORMAT == (("Q5", "B0g"), ("Q4", "B0g"), ("Q4", "Q5"))
    assert bc.CROSS == (("N6", "B0g"), ("F1", "B0g"), ("F2", "B0g"))
    assert bc.VLLM == (("Q5", "B0"), ("Q4", "B0"), ("N6", "B0"))


def test_engine_effect_is_the_control_minus_the_reference():
    # run a perfect, run b all wrong: the pooled reference scores 0.5; the control scores 0.75
    arms = {"B0a": const("B0a", 1.0), "B0b": const("B0b", 0.0), "B0": const("B0", 0.5), "B0g": const("B0g", 0.75)}
    eng = bc.contrast_set(arms, bc.ENGINE, n_boot=NB)
    assert set(eng) == {"B0g-B0", "B0g-B0a", "B0g-B0b"}
    assert eng["B0g-B0"]["macro"] == {"value": 25.0, "lo": 25.0, "hi": 25.0}
    assert eng["B0g-B0a"]["macro"]["value"] == -25.0
    assert eng["B0g-B0b"]["macro"]["value"] == 75.0
    assert all(v["value"] == 25.0 for v in eng["B0g-B0"]["tasks"].values())


def test_the_format_set_holds_the_three_pairs_with_the_hand_computed_values():
    arms = {"B0g": const("B0g", 0.75), "Q5": const("Q5", 0.70), "Q4": const("Q4", 0.50)}
    fmt = bc.contrast_set(arms, bc.FORMAT, n_boot=NB)
    assert fmt["Q5-B0g"]["macro"]["value"] == pytest.approx(-5.0)
    assert fmt["Q4-B0g"]["macro"]["value"] == pytest.approx(-25.0)
    assert fmt["Q4-Q5"]["macro"]["value"] == pytest.approx(-20.0)


def test_absolute_gives_each_arm_its_own_score():
    arms = {"B0a": const("B0a", 1.0), "B0b": const("B0b", 0.0), "B0": const("B0", 0.5), "B0g": const("B0g", 0.75), "Q5": const("Q5", 0.7)}
    ab = bc.absolute(arms, ("B0", "B0a", "B0b", "B0g", "Q5"), NB)
    assert [ab[a]["macro"]["value"] for a in ("B0", "B0a", "B0b", "B0g", "Q5")] == pytest.approx([50.0, 100.0, 0.0, 75.0, 70.0])
    assert all(v["value"] == pytest.approx(50.0) for v in ab["B0"]["tasks"].values())
    assert ab["B0g"]["tasks"]["aime2025"]["lo"] <= 75.0 <= ab["B0g"]["tasks"]["aime2025"]["hi"]


def test_cross_engine_pairs_are_the_builds_minus_the_control():
    arms = {"B0g": const("B0g", 0.5), "N6": const("N6", 0.5), "F1": const("F1", 0.6), "F2": const("F2", 0.4)}
    c = bc.contrast_set(arms, bc.CROSS, n_boot=NB)
    assert [round(c[k]["macro"]["value"], 6) for k in ("N6-B0g", "F1-B0g", "F2-B0g")] == [0.0, 10.0, -10.0]


# ----------------------------------------------------------------------------- the double difference


def test_double_difference_has_the_hand_computed_value():
    arms = {"Q5": const("Q5", 0.6), "B0g": const("B0g", 0.5), "B0": const("B0", 0.8)}
    d = bc.double_difference(arms, "Q5", "B0g", "Q5", "B0", n_boot=NB)
    # (0.6 - 0.5) - (0.6 - 0.8) = 0.1 + 0.2 = 0.3
    assert d["macro"] == pytest.approx({"value": 30.0, "lo": 30.0, "hi": 30.0})
    assert all(v["value"] == pytest.approx(30.0) for v in d["tasks"].values())


def test_double_difference_is_minus_the_engine_effect_whatever_x_is_and_matches_compare_arms():
    arms = toy_arms()
    d5 = bc.double_difference(arms, "Q5", "B0g", "Q5", "B0", n_boot=NB)
    d4 = bc.double_difference(arms, "Q4", "B0g", "Q4", "B0", n_boot=NB)
    assert d5["macro"] == pytest.approx(d4["macro"], abs=1e-9)
    ref, _ = aw.compare_arms(arms["B0"], arms["B0g"], bc.TASKS, NB, bc.SEED)  # B0 minus B0g
    assert d5["macro"]["value"] == pytest.approx(ref["macro"]["diff"]["value"], abs=1e-9)
    assert d5["macro"]["lo"] == pytest.approx(ref["macro"]["diff"]["lo"], abs=1e-9)
    assert d5["macro"]["hi"] == pytest.approx(ref["macro"]["diff"]["hi"], abs=1e-9)
    assert d5["tasks"]["aime2025"]["value"] == pytest.approx(ref["tasks"]["aime2025"]["diff"]["value"], abs=1e-9)


def test_double_difference_uses_only_items_in_all_four_arms_and_the_five_task_macro():
    arms = toy_arms()
    d = bc.double_difference(arms, "Q5", "B0g", "Q5", "B0", n_boot=NB)
    assert d["macro"]["value"] == pytest.approx(np.mean([d["tasks"][t]["value"] for t in bc.TASKS]))
    short = {k: v for k, v in arms.items()}
    short["B0"] = aw.Arm(name="B0", runs=[], scores={t: dict(list(s.items())[:5]) for t, s in arms["B0"].scores.items()}, strata={})
    d2 = bc.double_difference(short, "Q5", "B0g", "Q5", "B0", n_boot=NB)
    x = sorted(short["B0"].scores["mgsm"])
    v = [100 * (arms["B0"].scores["mgsm"][i] - arms["B0g"].scores["mgsm"][i]) for i in x]
    assert d2["tasks"]["mgsm"]["value"] == pytest.approx(np.mean(v))


# ----------------------------------------------------------------------------- the split of the loss


def test_decomposition_adds_the_format_part_to_the_engine_part_and_gives_the_engine_share():
    arms = {"B0": const("B0", 0.80), "B0g": const("B0g", 0.76), "Q5": const("Q5", 0.72)}
    vllm = bc.contrast_set(arms, (("Q5", "B0"),), n_boot=NB)
    fmt = bc.contrast_set(arms, (("Q5", "B0g"),), n_boot=NB)
    eng = bc.contrast_set(arms, (("B0g", "B0"),), n_boot=NB)
    d = bc.decomposition(vllm, fmt, eng, "Q5")
    assert (round(d["total"], 6), round(d["format"], 6), round(d["engine"], 6)) == (-8.0, -4.0, -4.0)
    assert d["engine_share"] == pytest.approx(50.0)
    assert d["tasks"]["mgsm"] == pytest.approx({"total": -8.0, "format": -4.0, "engine": -4.0})


def test_decomposition_with_a_zero_total_has_no_share_and_an_inconsistent_input_is_refused():
    arms = {"B0": const("B0", 0.8), "B0g": const("B0g", 0.7), "Q5": const("Q5", 0.8)}
    vllm = bc.contrast_set(arms, (("Q5", "B0"),), n_boot=NB)
    fmt = bc.contrast_set(arms, (("Q5", "B0g"),), n_boot=NB)
    eng = bc.contrast_set(arms, (("B0g", "B0"),), n_boot=NB)
    assert bc.decomposition(vllm, fmt, eng, "Q5")["engine_share"] is None
    fmt["Q5-B0g"]["macro"]["value"] += 1.0
    with pytest.raises(SystemExit, match="not the format part plus the engine part"):
        bc.decomposition(vllm, fmt, eng, "Q5")


# ----------------------------------------------------------------------------- leave AIME out, margins


def test_leave_aime_out_is_the_macro_of_four_tasks_and_a_gap_in_aime_alone_disappears():
    good = const("good", 1.0)
    bad = arm("bad", {**{t: [1.0] * n for t, n in SIZES.items()}, "aime2025": [0.0] * SIZES["aime2025"]})
    arms = {"B0g": bad, "B0": good}
    four = bc.contrast(arms, "B0g", "B0", bc.FOUR, NB)
    five = bc.contrast(arms, "B0g", "B0", bc.TASKS, NB)
    assert bc.AIME not in four["tasks"] and len(four["tasks"]) == 4
    assert five["macro"]["value"] == pytest.approx(-20.0)
    assert (four["macro"]["value"], four["macro"]["lo"], four["macro"]["hi"]) == (0.0, 0.0, 0.0)


def test_margin_set_is_keyed_by_the_build_and_uses_the_given_reference():
    arms = {"B0": const("B0", 0.9), "B0g": const("B0g", 0.8), "Q5": const("Q5", 0.75), "Q4": const("Q4", 0.5), "N6": const("N6", 0.8)}
    vs_g = bc.margin_set(arms, bc.MARGIN_VS_B0G, NB)
    vs_b = bc.margin_set(arms, bc.MARGIN_VS_B0, NB)
    assert list(vs_g) == ["B0g", "Q5", "Q4", "N6"]
    assert vs_g["B0g"]["macro"]["value"] == pytest.approx(-10.0)  # B0g itself is read against B0
    assert vs_g["Q5"]["macro"]["value"] == pytest.approx(-5.0)    # against B0g, not B0 (-15)
    assert vs_b["Q5"]["macro"]["value"] == pytest.approx(-15.0)
    assert vs_g["Q5"]["margins"]["-5"]["state"] == "open"   # the interval is the point -5, its lower end does not pass -5
    assert vs_g["Q5"]["margins"]["-3"]["state"] == "worse"
    assert vs_g["N6"]["macro"]["value"] == pytest.approx(0.0) and vs_g["N6"]["margins"]["-2"]["state"] == "pass"
    assert [f["task"] for f in vs_b["Q4"]["task_flag"]] == list(bc.TASKS)  # 40 points below B0 on every task


# ----------------------------------------------------------------------------- generous rescoring and truncation audit


def row(task: str, sid: str, epoch: int, score: float, truncated: bool = False) -> dict:
    return {"task": task, "sample_id": sid, "epoch": epoch, "score": score, "error": None, "truncated": truncated}


def small_run(truncated: set[tuple[str, str]], tasks: tuple[str, ...] = bc.TASKS) -> aw.Run:
    """Four items per task, each scoring 1 except the ``truncated`` ones (score 0, truncated)."""
    rows = []
    for t in tasks:
        for i in range(4):
            sid = f"{t}-{i}"
            rows.append(row(t, sid, 1, 0.0 if (t, sid) in truncated else 1.0, truncated=(t, sid) in truncated))
    return aw.Run(path=Path("."), rows=rows)


def audit_entry(items: list[tuple[str, str, bool]]) -> dict:
    """An audit arm (summary row and samples) from ``(task name in the audit, sample id, target_late)``."""
    return {"summary": {"truncated": len(items)}, "samples": [{"task": t, "sample_id": s, "target_late": late} for t, s, late in items]}


def test_generous_arms_credit_only_the_arms_own_rows_and_pool_the_bf16_runs():
    cut_g = {("mgsm", "mgsm-0"), ("ifeval", "ifeval-1")}
    runs = {"B0a": [small_run({("mgsm", "mgsm-0")})], "B0b": [small_run(set())], "B0g": [small_run(cut_g)]}
    audit = {"summary": {}, "samples": {}}
    for a, items in (("B0a", [("inspect_evals/mgsm", "mgsm-0", True)]), ("B0b", []), ("B0g", [("inspect_evals/mgsm", "mgsm-0", True), ("inspect_evals/ifeval", "ifeval-1", False)])):
        e = audit_entry(items)
        audit["summary"][bc.AUDIT_NAME[a]], audit["samples"][bc.AUDIT_NAME[a]] = e["summary"], e["samples"]
    garms, credited = bc.generous_arms(runs, audit, ("B0a", "B0b", "B0g"))
    assert credited == {"B0a": 1, "B0b": 0, "B0g": 1}
    assert garms["B0g"].scores["mgsm"]["mgsm-0"] == 1.0   # credited
    assert garms["B0g"].scores["ifeval"]["ifeval-1"] == 0.0   # the right number was not in the last quarter
    assert garms["B0a"].scores["mgsm"]["mgsm-0"] == 1.0
    assert garms["B0"].scores["mgsm"]["mgsm-0"] == 1.0  # (1 + 1) / 2 of the two credited bf16 runs
    # crediting B0a must not touch B0g, and vice versa
    runs2 = {**runs, "B0g": [small_run(set())]}
    audit2 = json.loads(json.dumps(audit))
    audit2["summary"][bc.AUDIT_NAME["B0g"]], audit2["samples"][bc.AUDIT_NAME["B0g"]] = audit_entry([])["summary"], []
    g2, c2 = bc.generous_arms(runs2, audit2, ("B0a", "B0b", "B0g"))
    assert c2["B0g"] == 0 and g2["B0g"].scores["mgsm"]["mgsm-0"] == 1.0  # that item was never cut off in this B0g


def test_generous_arms_refuse_an_audit_that_does_not_match_the_samples():
    runs = {"B0g": [small_run({("mgsm", "mgsm-0")})]}
    e = audit_entry([("inspect_evals/mgsm", "mgsm-0", True), ("inspect_evals/mgsm", "mgsm-1", True)])
    audit = {"summary": {bc.B0G_AUDIT: e["summary"]}, "samples": {bc.B0G_AUDIT: e["samples"]}}
    with pytest.raises(SystemExit, match="do not match the audit"):
        bc.generous_arms(runs, audit, ("B0g",))


def test_generous_block_reports_strict_and_generous_macro_and_paired_contrasts():
    strict = {"B0g": const("B0g", 0.5), "B0": const("B0", 0.6), "Q5": const("Q5", 0.4), "Q4": const("Q4", 0.3)}
    gen = {"B0g": const("B0g", 0.6), "B0": const("B0", 0.6), "Q5": const("Q5", 0.45), "Q4": const("Q4", 0.3)}
    b = bc.generous_block(strict, gen, {"B0g": 7, "Q5": 2}, (("B0g", "B0"), ("Q5", "B0g")), NB)
    assert b["macro"]["B0g"] == pytest.approx({"strict": 50.0, "generous": 60.0, "credited": 7})
    assert b["macro"]["Q4"]["credited"] is None
    assert b["contrasts"]["B0g-B0"]["strict"]["value"] == pytest.approx(-10.0)
    assert b["contrasts"]["B0g-B0"]["generous"]["value"] == pytest.approx(0.0)
    assert b["contrasts"]["Q5-B0g"]["generous"]["value"] == pytest.approx(-15.0)


def test_truncation_block_counts_per_task_with_the_names_of_samples_jsonl():
    items = [("inspect_evals/aime2025", "1", False), ("inspect_evals/aime2025", "2", True), ("mbpp-plus", "3", False), ("inspect_evals/mgsm", "bn_1", False)]
    e = audit_entry(items)
    audit = {"summary": {bc.B0G_AUDIT: e["summary"]}, "samples": {bc.B0G_AUDIT: e["samples"]}}
    t = bc.truncation_block(audit)
    assert t["cut_by_task"] == {"mgsm": 1, "humaneval_plus": 0, "mbpp_plus": 1, "ifeval": 0, "aime2025": 2}
    assert t["cut_total"] == 4
    audit["summary"][bc.B0G_AUDIT]["truncated"] = 5
    with pytest.raises(SystemExit, match="do not add up"):
        bc.truncation_block(audit)


def test_the_audit_carries_the_b0g_row_with_the_label_and_folders_used_here():
    src = (Path(bc.__file__).parent / "truncation_audit.py").read_text()
    runs = next(ast.literal_eval(n.value) for n in ast.parse(src).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "RUNS")
    assert runs["bf16 GGUF (llama.cpp)"] == bc.B0G_DIRS == ["B0g-mgsm-he", "B0g-ifeval-mbpp", "B0g-aime"]
    assert bc.AUDIT_NAME["B0g"] == bc.B0G_AUDIT == "bf16 GGUF (llama.cpp)"
    assert len(runs) == 11 and list(runs)[-1] == bc.B0G_AUDIT  # appended: the earlier rows keep their places


# ----------------------------------------------------------------------------- the checks of the B0g data


PIN, COMMIT = "b266da8e" * 8, "55c22dca" * 5


def pod(tasks: tuple[str, ...], errors: int = 0, n: int = 4) -> aw.Run:
    rows = [row(t, f"{t}-{i}", e, 1.0) for t in tasks for i in range(n) for e in range(1, (3 if t == "aime2025" else 1) + 1)]
    return aw.Run(path=Path("pod"), rows=rows, info={"status": "complete", "request_errors": errors})


def good_pods() -> list[aw.Run]:
    return [pod(("mgsm", "humaneval_plus")), pod(("ifeval", "mbpp_plus")), pod(("aime2025",))]


def good_prov(n: int = 3) -> list[dict]:
    return [{"arm": "gguf-bf16", "gguf_sha256_pinned": PIN, "gguf_sha256_verified_by_setup": PIN, "llama_cpp_commit": COMMIT, "slot_ctx": "32768"} for _ in range(n)]


def ref_arm() -> aw.Arm:
    run = aw.Run(path=Path("ref"), rows=[r for p in good_pods() for r in p.rows])
    return aw.build_arm("B0", [run], bc.TASKS)


def check(pods=None, prov=None, pin=PIN, commit=COMMIT) -> dict:
    return bc.check_b0g(pods or good_pods(), ref_arm(), pin, commit, prov or good_prov())


def test_the_b0g_check_accepts_the_real_shape_and_reports_what_it_saw():
    out = check()
    assert out["sha256"] == PIN and out["slot_ctx"] == 32768 and out["truncations"] == 0
    assert out["samples"] == {"mgsm": 4, "humaneval_plus": 4, "mbpp_plus": 4, "ifeval": 4, "aime2025": 12}


def test_the_b0g_check_refuses_a_request_error_in_the_info_or_in_a_row():
    pods = good_pods()
    pods[1].info["request_errors"] = 2
    with pytest.raises(SystemExit, match="request errors"):
        check(pods)
    pods = good_pods()
    pods[0].rows[0]["error"] = "boom"
    with pytest.raises(SystemExit, match="request errors"):
        check(pods)


def test_the_b0g_check_refuses_a_run_that_is_not_complete_or_misses_a_task_or_repeats_one():
    pods = good_pods()
    pods[2].info["status"] = "running"
    with pytest.raises(SystemExit, match="not complete"):
        check(pods)
    with pytest.raises(SystemExit, match="five tasks exactly once"):
        check(good_pods()[:2])
    with pytest.raises(SystemExit, match="five tasks exactly once"):
        check(good_pods() + [pod(("aime2025",))])


def test_the_b0g_check_refuses_other_items_or_a_different_number_of_rows_than_the_reference():
    pods = good_pods()
    pods[0].rows = [r for r in pods[0].rows if r["sample_id"] != "mgsm-3"]
    with pytest.raises(SystemExit, match="same mgsm items"):
        check(pods)
    pods = good_pods()
    pods[2].rows = pods[2].rows[:-1]
    with pytest.raises(SystemExit):
        check(pods)  # a missing epoch: the items are the same but the rows are not


def test_the_b0g_check_refuses_a_wrong_hash_arm_context_or_commit():
    bad = good_prov()
    bad[1]["gguf_sha256_verified_by_setup"] = "0" * 64
    with pytest.raises(SystemExit, match="hash"):
        check(prov=bad)
    with pytest.raises(SystemExit, match="hash"):
        check(pin="f" * 64)  # the provenance agrees with itself but not with gguf_files.json
    bad = good_prov()
    bad[0]["arm"] = "gguf-q5_k_m"
    with pytest.raises(SystemExit, match="gguf-bf16"):
        check(prov=bad)
    bad = good_prov()
    bad[2]["slot_ctx"] = "16384"
    with pytest.raises(SystemExit, match="context"):
        check(prov=bad)
    with pytest.raises(SystemExit, match="commit"):
        check(commit="1" * 40)


def test_read_provenance_reads_key_value_lines(tmp_path):
    f = tmp_path / "server_provenance.txt"
    f.write_text("arm=gguf-bf16\nslot_ctx=32768\ncommand=uv run x --a=b\n\n")
    assert bc.read_provenance(f) == {"arm": "gguf-bf16", "slot_ctx": "32768", "command": "uv run x --a=b"}


def write_analysis(results: Path, summaries: dict[str, dict]) -> None:
    (results / "wave2-test-analysis.json").write_text(
        json.dumps({"reference": "B0", "candidates": {a: {"summary": s} for a, s in summaries.items()}, "exploratory": {}, "contrasts": {}, "calibration": None})
    )


def test_the_reproduction_check_needs_three_stored_intervals_and_refuses_a_different_one(tmp_path):
    arms = {"B0": toy(1), "F1": toy(2), "F2": toy(3), "Q5": toy(4)}
    sums = {a: aw.compare_arms(arms[a], arms["B0"], bc.TASKS, bc.N_BOOT, bc.SEED)[0] for a in ("F1", "F2", "Q5")}
    write_analysis(tmp_path, sums)
    assert bc.check_reproduction(arms, tmp_path) == 3
    write_analysis(tmp_path, {"F1": sums["F1"], "F2": sums["F2"]})
    with pytest.raises(SystemExit, match="at least 3"):
        bc.check_reproduction(arms, tmp_path)
    changed = json.loads(json.dumps(sums))
    changed["Q5"]["macro"]["diff"]["hi"] += 0.01
    write_analysis(tmp_path, changed)
    with pytest.raises(SystemExit, match="cannot reproduce"):
        bc.check_reproduction(arms, tmp_path)


# ----------------------------------------------------------------------------- the whole analysis on toy data


def test_analyse_returns_every_section_in_json_form_and_renders_without_dashes():
    arms = toy_arms()
    garms = {a: toy_arms()[a] for a in (*bc.GEN_ARMS, "B0")}
    e = audit_entry([("inspect_evals/aime2025", "1", True), ("mbpp-plus", "2", False)])
    audit = {"summary": {bc.B0G_AUDIT: {**e["summary"], "samples": 1783, "long": 1, "loops": 1, "target_late": 1, "with_text": 0}}, "samples": {bc.B0G_AUDIT: e["samples"]}}
    res = bc.analyse(arms, garms, {a: 1 for a in bc.GEN_ARMS}, audit, n_boot=NB)
    assert set(res) == {"absolute", "engine", "format", "cross", "vllm", "double", "double_identity_checked", "decomposition", "noaime", "margin", "generous", "truncation"}
    assert list(res["engine"]) == ["B0g-B0", "B0g-B0a", "B0g-B0b"]
    assert list(res["noaime"]) == ["B0g-B0", "B0g-B0a", "B0g-B0b", "Q5-B0g", "Q4-B0g", "Q4-Q5"]
    assert all("aime2025" not in v["tasks_without"] for v in res["noaime"].values())
    for v in res["noaime"].values():  # the macro of the four tasks is the mean of the four task values, and is not the five-task macro
        assert v["without"]["value"] == pytest.approx(np.mean([d["value"] for d in v["tasks_without"].values()]))
        assert len(v["tasks_without"]) == 4 and v["without"]["value"] != v["with"]["value"]
    assert set(res["margin"]) == {"vs_B0g", "vs_B0"} and set(res["decomposition"]) == {"Q5", "Q4", "N6"}
    assert res["double_identity_checked"] is True
    full = {"meta": {"n_boot": NB, "seed": 0, "reproduced": 25, "b0g": {"sha256": PIN, "truncations": 2}}, **res}
    text = json.dumps(full)
    assert json.loads(text)["engine"]["B0g-B0"]["macro"]["value"] == res["engine"]["B0g-B0"]["macro"]["value"]
    md = bc.render_md(full)
    assert "Engine effect" in md and "Double difference" in md and chr(0x2014) not in md and chr(0x2013) not in md
