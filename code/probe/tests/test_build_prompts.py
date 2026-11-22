"""Tests of build_prompts.py with a fake task registry (no inspect_ai, no dataset, no model)."""

from types import SimpleNamespace

import pytest

import build_prompts as bp
import probe_prompts as pp


def _extracted(sizes):
    return {
        "tasks": {
            t: {"items": [{"id": str(i), "fields": {"input": f"{t} question {i}"}} for i in range(n)]}
            for t, n in sizes.items()
        }
    }


SIZES = {"mgsm": 250, "humaneval_plus": 164, "mbpp_plus": 378, "ifeval": 541, "aime2025": 30}


def test_default_build_is_64_task_prompts_plus_16_plain():
    rows = bp.build_default_prompts(_extracted(SIZES))
    assert len(rows) == 80
    chat = [r for r in rows if r["kind"] == "chat"]
    plain = [r for r in rows if r["kind"] == "plain"]
    assert len(chat) == 64 and len(plain) == 16
    per_task = {t: sum(r["id"].startswith(t + ":") for r in chat) for t in SIZES}
    assert per_task == {"mgsm": 13, "humaneval_plus": 13, "mbpp_plus": 13, "ifeval": 13, "aime2025": 12}
    assert len({r["id"] for r in rows}) == 80


def test_build_is_deterministic_and_seed_dependent():
    a = bp.build_default_prompts(_extracted(SIZES), seed=0)
    b = bp.build_default_prompts(_extracted(SIZES), seed=0)
    c = bp.build_default_prompts(_extracted(SIZES), seed=1)
    assert a == b and a != c


def test_chat_prompt_is_one_user_message_with_the_item_input():
    rows = bp.task_prompts(_extracted(SIZES), ["mgsm", "aime2025"], 4, seed=0)
    for r in rows:
        task, idx = r["id"].split(":")
        assert r["messages"] == [{"role": "user", "content": f"{task} question {idx}"}]


def test_a_short_task_gives_all_its_items_and_the_shortfall_goes_elsewhere():
    rows = bp.task_prompts(_extracted({"a": 2, "b": 50, "c": 50}), ["a", "b", "c"], 30, seed=0)
    assert len(rows) == 30
    assert sum(r["id"].startswith("a:") for r in rows) == 2


def test_too_few_items_is_an_error():
    with pytest.raises(ValueError, match="fewer than"):
        bp.task_prompts(_extracted({"a": 2, "b": 3}), ["a", "b"], 10, seed=0)


def test_items_without_input_are_skipped():
    ex = _extracted({"a": 10})
    ex["tasks"]["a"]["items"][0]["fields"] = {}
    rows = bp.task_prompts(ex, ["a"], 9, seed=0)
    assert "a:0" not in {r["id"] for r in rows}


def test_plain_prompts_cover_three_languages_and_vary_in_length():
    rows = bp.plain_prompts()
    assert len(rows) == 16 and rows[0]["id"].startswith("plain:")
    lens = sorted(len(r["text"]) for r in rows)
    assert lens[0] > 200 and lens[-1] > 6 * lens[0]
    text = " ".join(r["text"] for r in rows)
    assert "def " in text and "La Loire" in text and "The tides" in text
    with pytest.raises(ValueError):
        bp.plain_prompts(17)


def test_prompt_file_round_trip(tmp_path):
    rows = bp.build_default_prompts(_extracted(SIZES))
    pp.write_prompts(tmp_path / "p.jsonl", rows)
    assert pp.read_prompts(tmp_path / "p.jsonl") == rows


def test_main_uses_the_registry_through_extract_eval_items(tmp_path, monkeypatch):
    import sys

    fake = SimpleNamespace(
        TASKS={
            t: SimpleNamespace(
                factory=lambda t=t, **kw: SimpleNamespace(
                    dataset=[
                        SimpleNamespace(id=str(i), input=f"{t} q{i}", target="x", choices=None, metadata=None)
                        for i in range(40)
                    ]
                ),
                takes_sandbox=False,
            )
            for t in bp.DEFAULT_TASKS
        }
    )
    monkeypatch.setitem(sys.modules, "run_tasks", fake)
    out = tmp_path / "p.jsonl"
    assert bp.main(["--out", str(out)]) == 0
    assert len(pp.read_prompts(out)) == 80
