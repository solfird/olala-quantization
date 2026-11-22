"""Tests for compare.py: paired bootstrap between two run_tasks result directories."""

import json

import pytest

import compare


def row(task, sid, score, *, epoch=1, error=None, truncated=False, stratum=None):
    return {
        "task": task,
        "sample_id": sid,
        "epoch": epoch,
        "stratum": stratum,
        "score": score,
        "error": error,
        "truncated": truncated,
    }


def write_run(path, rows, tasks=None, **info):
    path.mkdir(parents=True)
    (path / "samples.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    base = {
        "model": "olala",
        "endpoint": "http://x/v1",
        "sampling": {"temperature": 0.8, "top_p": 0.9, "top_k": 40},
        "reasoning_effort": "high",
        "seed": 0,
        "git": {"commit": None},
        "versions": {"inspect_ai": "1"},
        "tasks": tasks or {},
    }
    base.update(info)
    (path / "run_info.json").write_text(json.dumps(base))


def two_runs(tmp_path, a_correct=80, b_correct=70, n=100, **kw):
    a = [row("t", f"i{i}", 1.0 if i < a_correct else 0.0) for i in range(n)]
    b = [row("t", f"i{i}", 1.0 if i < b_correct else 0.0) for i in range(n)]
    write_run(tmp_path / "A", a, **kw)
    write_run(tmp_path / "B", b, **kw)
    return tmp_path / "A", tmp_path / "B"


def test_known_accuracies_and_a_paired_interval_that_excludes_zero(tmp_path):
    a, b = two_runs(tmp_path)
    res = compare.compare_runs(a, b, n_boot=2000, seed=0)
    t = res["tasks"]["t"]
    assert t["n_items"] == 100
    assert t["acc_a"]["value"] == pytest.approx(0.80)
    assert t["acc_b"]["value"] == pytest.approx(0.70)
    assert t["diff"]["value"] == pytest.approx(-0.10)
    # ten items flip from right to left: SE = sqrt(.1*.9/100) = .03 -> about +-.059
    assert -0.17 < t["diff"]["lo"] < -0.13
    assert -0.07 < t["diff"]["hi"] < -0.03
    assert t["acc_a"]["lo"] < 0.80 < t["acc_a"]["hi"]


def test_identical_runs_have_a_zero_difference_with_a_degenerate_interval(tmp_path):
    a, b = two_runs(tmp_path, a_correct=60, b_correct=60)
    t = compare.compare_runs(a, b, n_boot=500, seed=0)["tasks"]["t"]
    assert (t["diff"]["value"], t["diff"]["lo"], t["diff"]["hi"]) == (0.0, 0.0, 0.0)


def test_epochs_are_averaged_per_item_before_comparing(tmp_path):
    a = [row("t", "x", s, epoch=e + 1) for e, s in enumerate([1, 1, 0, 0, 1])]
    b = [row("t", "x", s, epoch=e + 1) for e, s in enumerate([1, 0, 0, 0, 0])]
    write_run(tmp_path / "A", a)
    write_run(tmp_path / "B", b)
    t = compare.compare_runs(tmp_path / "A", tmp_path / "B", n_boot=50, seed=0)["tasks"]["t"]
    assert t["acc_a"]["value"] == pytest.approx(0.6)
    assert t["acc_b"]["value"] == pytest.approx(0.2)
    assert t["epochs_a"] == 5
    assert t["n_items"] == 1


def test_request_errors_and_truncations_count_as_failures_and_are_listed_separately(tmp_path):
    a = [row("t", "a", 1.0), row("t", "b", 1.0), row("t", "c", 1.0), row("t", "d", 1.0)]
    b = [
        row("t", "a", 1.0),
        row("t", "b", None, error="ConnectError"),  # request error: no score at all
        row("t", "c", 1.0, truncated=True),  # scorer said correct but the answer was cut off
        row("t", "d", 1.0),
    ]
    write_run(tmp_path / "A", a)
    write_run(tmp_path / "B", b)
    res = compare.compare_runs(tmp_path / "A", tmp_path / "B", n_boot=50, seed=0)
    t = res["tasks"]["t"]
    assert t["acc_a"]["value"] == 1.0
    assert t["acc_b"]["value"] == pytest.approx(0.5)  # only a and d count
    assert t["failures_a"] == {"request_errors": 0, "truncations": 0, "rows": 4}
    assert t["failures_b"] == {"request_errors": 1, "truncations": 1, "rows": 4}
    assert t["failed_ids_b"] == {"request_errors": ["b"], "truncations": ["c"]}
    md = compare.render_markdown(res)
    assert "Request errors" in md
    assert "Truncations" in md


def test_items_present_in_only_one_run_are_not_paired_but_are_reported(tmp_path):
    write_run(tmp_path / "A", [row("t", "a", 1.0), row("t", "only_a", 1.0)])
    write_run(tmp_path / "B", [row("t", "a", 0.0), row("t", "only_b", 1.0)])
    t = compare.compare_runs(tmp_path / "A", tmp_path / "B", n_boot=50, seed=0)["tasks"]["t"]
    assert t["n_items"] == 1
    assert t["only_a"] == 1
    assert t["only_b"] == 1


def test_a_task_present_in_one_run_only_is_listed_not_dropped(tmp_path):
    write_run(tmp_path / "A", [row("t", "a", 1.0), row("u", "a", 1.0)])
    write_run(tmp_path / "B", [row("t", "a", 1.0)])
    res = compare.compare_runs(tmp_path / "A", tmp_path / "B", n_boot=50, seed=0)
    assert set(res["tasks"]) == {"t"}
    assert res["only_in_a"] == ["u"]
    assert "u" in compare.render_markdown(res)


def test_settings_that_differ_between_the_runs_are_called_out(tmp_path):
    a, b = two_runs(tmp_path)
    info = json.loads((b / "run_info.json").read_text())
    info["sampling"]["temperature"] = 0.0
    info["model"] = "olala-fp8"
    (b / "run_info.json").write_text(json.dumps(info))
    res = compare.compare_runs(a, b, n_boot=50, seed=0)
    keys = {d["key"] for d in res["setting_differences"]}
    assert "sampling.temperature" in keys
    assert "model" in keys
    md = compare.render_markdown(res)
    assert "sampling.temperature" in md


def test_different_item_sets_are_flagged_through_the_items_hash(tmp_path):
    a, b = two_runs(tmp_path)
    for d, sha in ((a, "1" * 64), (b, "2" * 64)):
        info = json.loads((d / "run_info.json").read_text())
        info["tasks"] = {"t": {"items_sha256": sha, "epochs": 1}}
        (d / "run_info.json").write_text(json.dumps(info))
    res = compare.compare_runs(a, b, n_boot=50, seed=0)
    assert res["tasks"]["t"]["same_items_hash"] is False
    assert "different item sets" in compare.render_markdown(res).lower()


def test_by_stratum_breakdown_reports_each_language(tmp_path):
    a = [row("m", f"{lang}{i}", 1.0, stratum=lang) for lang in ("en", "fr") for i in range(10)]
    b = [
        row("m", f"{lang}{i}", 1.0 if (lang == "en" or i < 5) else 0.0, stratum=lang)
        for lang in ("en", "fr")
        for i in range(10)
    ]
    write_run(tmp_path / "A", a)
    write_run(tmp_path / "B", b)
    res = compare.compare_runs(tmp_path / "A", tmp_path / "B", n_boot=50, seed=0)
    strata = res["tasks"]["m"]["strata"]
    assert strata["en"]["diff"]["value"] == 0.0
    assert strata["fr"]["diff"]["value"] == pytest.approx(-0.5)
    md = compare.render_markdown(res, by_stratum=True)
    assert "fr" in md and "en" in md


def test_the_bootstrap_is_reproducible_for_a_seed(tmp_path):
    a, b = two_runs(tmp_path)
    first = compare.render_markdown(compare.compare_runs(a, b, n_boot=300, seed=4))
    second = compare.render_markdown(compare.compare_runs(a, b, n_boot=300, seed=4))
    assert first == second


def test_markdown_numbers_come_from_the_result_dict(tmp_path):
    a, b = two_runs(tmp_path)
    res = compare.compare_runs(a, b, n_boot=300, seed=0)
    md = compare.render_markdown(res)
    t = res["tasks"]["t"]
    assert f"{100 * t['acc_a']['value']:.1f}" in md
    assert f"{100 * t['acc_b']['value']:.1f}" in md
    assert f"{100 * t['diff']['value']:+.1f}" in md
    assert "| t |" in md


def test_the_cli_prints_the_table_and_can_write_markdown_and_json(tmp_path, capsys):
    a, b = two_runs(tmp_path)
    code = compare.main(
        [str(a), str(b), "--n-boot", "200", "--out", str(tmp_path / "r.md"),
         "--json", str(tmp_path / "r.json")]
    )  # fmt: skip
    assert code == 0
    printed = capsys.readouterr().out
    assert printed == (tmp_path / "r.md").read_text()
    assert json.loads((tmp_path / "r.json").read_text())["tasks"]["t"]["n_items"] == 100
