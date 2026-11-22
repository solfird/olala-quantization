"""Tests for pilot_summary.py on a fake run directory (no server, no model)."""

import json
from pathlib import Path

import pytest

import pilot_summary
import run_tasks


def row(task, sid, epoch, score, out_tokens, *, error=None, truncated=False):
    return {
        "task": task,
        "sample_id": sid,
        "epoch": epoch,
        "stratum": None,
        "score": score,
        "scores": {},
        "error": error,
        "truncated": truncated,
        "stop_reason": "max_tokens" if truncated else "stop",
        "input_tokens": 10,
        "output_tokens": out_tokens,
        "total_time": 1.0,
        "target": "",
        "completion": "",
    }


def make_run(tmp_path: Path) -> Path:
    rows = [
        # aime2025: 2 items x 2 epochs, tokens 1000 / 3000 / 2000 / 4000, one wrong answer
        row("aime2025", "a1", 1, 1.0, 1000),
        row("aime2025", "a1", 2, 0.0, 3000),
        row("aime2025", "a2", 1, 1.0, 2000),
        row("aime2025", "a2", 2, 1.0, 4000),
        # mgsm: 3 items x 1 epoch; one truncated (counts as failure), one request error (no tokens)
        row("mgsm", "m1", 1, 1.0, 400),
        row("mgsm", "m2", 1, 1.0, 600, truncated=True),
        row("mgsm", "m3", 1, None, None, error="ConnectError: boom"),
        # gsm8k is not part of the protocol's primary set
        row("gsm8k", "g1", 1, 1.0, 100),
    ]
    run = tmp_path / "pilot"
    run.mkdir()
    (run / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    info = {
        "status": "complete",
        "started_utc": "2026-11-21T10:00:00Z",
        "ended_utc": "2026-11-21T10:30:00Z",
        "tasks": {
            "aime2025": {"epochs": 2, "n_items": 2},
            "mgsm": {"epochs": 1, "n_items": 3},
            "gsm8k": {"epochs": 1, "n_items": 1},
        },
    }
    (run / "run_info.json").write_text(json.dumps(info))
    return run


def test_the_protocol_item_counts_and_epochs_match_the_frozen_protocol_and_the_harness():
    assert pilot_summary.PROTOCOL_ITEMS == {
        "mgsm": 550,
        "humaneval_plus": 164,
        "mbpp_plus": 378,
        "ifeval": 541,
        "aime2025": 30,
    }
    for task, epochs in pilot_summary.PROTOCOL_EPOCHS.items():
        assert run_tasks.TASKS[task].epochs == epochs
    assert pilot_summary.PROTOCOL_EPOCHS["aime2025"] == 5


def test_per_task_statistics_of_a_fake_run(tmp_path):
    summary = pilot_summary.summarise(make_run(tmp_path))
    aime = summary["tasks"]["aime2025"]
    assert (aime["items"], aime["epochs"], aime["samples"]) == (2, 2, 4)
    assert aime["mean_output_tokens"] == pytest.approx(2500)
    assert aime["total_output_tokens"] == 10000
    assert (aime["truncations"], aime["request_errors"]) == (0, 0)
    assert aime["accuracy"] == pytest.approx((0.5 + 1.0) / 2)  # item means, then the mean
    mgsm = summary["tasks"]["mgsm"]
    assert mgsm["mean_output_tokens"] == pytest.approx(500)  # the errored row has no tokens
    assert mgsm["total_output_tokens"] == 1000
    assert (mgsm["truncations"], mgsm["request_errors"]) == (1, 1)
    # the truncated answer and the request error count as failures: only m1 scores
    assert mgsm["accuracy"] == pytest.approx(1 / 3)


def test_projection_scales_mean_tokens_to_the_protocol_item_set_and_epochs(tmp_path):
    summary = pilot_summary.summarise(make_run(tmp_path), tok_per_s=1000.0)
    aime = summary["tasks"]["aime2025"]
    assert aime["projected_samples"] == 30 * 5
    assert aime["projected_output_tokens"] == pytest.approx(2500 * 150)
    mgsm = summary["tasks"]["mgsm"]
    assert mgsm["projected_samples"] == 550
    assert mgsm["projected_output_tokens"] == pytest.approx(500 * 550)
    total = 2500 * 150 + 500 * 550
    assert summary["projection"]["output_tokens"] == pytest.approx(total)
    assert summary["projection"]["hours"] == pytest.approx(total / 1000.0 / 3600)
    assert summary["projection"]["tok_per_s"] == 1000.0


def test_tasks_outside_the_protocol_set_are_reported_but_not_projected(tmp_path):
    summary = pilot_summary.summarise(make_run(tmp_path), tok_per_s=1000.0)
    gsm = summary["tasks"]["gsm8k"]
    assert gsm["projected_output_tokens"] is None
    assert "gsm8k" in summary["projection"]["not_projected"]


def test_without_a_throughput_the_hours_are_not_invented(tmp_path):
    summary = pilot_summary.summarise(make_run(tmp_path))
    assert summary["projection"]["hours"] is None
    assert summary["projection"]["output_tokens"] > 0


def test_a_task_of_the_protocol_that_was_not_piloted_is_listed_as_missing(tmp_path):
    summary = pilot_summary.summarise(make_run(tmp_path), tok_per_s=1000.0)
    assert summary["projection"]["missing"] == ["humaneval_plus", "mbpp_plus", "ifeval"]


def test_a_non_positive_throughput_is_refused(tmp_path):
    with pytest.raises(ValueError, match="tok-per-s"):
        pilot_summary.summarise(make_run(tmp_path), tok_per_s=0)


def test_the_command_line_prints_a_markdown_table_and_the_projection(tmp_path, capsys):
    run = make_run(tmp_path)
    assert pilot_summary.main([str(run), "--tok-per-s", "1000"]) == 0
    out = capsys.readouterr().out
    assert "| aime2025 |" in out and "| mgsm |" in out
    assert "tokens/s" in out
    assert "hours" in out.lower()
    assert "humaneval_plus" in out  # named as missing from the pilot


def test_the_command_line_can_also_write_json(tmp_path):
    run = make_run(tmp_path)
    target = tmp_path / "summary.json"
    assert pilot_summary.main([str(run), "--tok-per-s", "500", "--json", str(target)]) == 0
    data = json.loads(target.read_text())
    assert data["projection"]["tok_per_s"] == 500
    assert data["tasks"]["aime2025"]["items"] == 2


def test_a_directory_that_is_not_a_run_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match=r"samples\.jsonl"):
        pilot_summary.main([str(tmp_path)])
