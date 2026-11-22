"""Opt-in smoke test: every real task of the protocol through run_tasks.py and compare.py.

The model is the fake server (canned replies), so nothing here measures a model. It needs the
dataset hosts, the NLTK data download used by the IFEval scorer, and a running Docker daemon
for the code tasks: run it with ``uv run pytest -m "network and docker"``.
"""

import json

import pytest
from fake_server import FakeServer

import compare
import run_tasks

pytestmark = [pytest.mark.network, pytest.mark.docker]


def canned(body: dict) -> tuple[str, str]:
    return "```python\ndef f():\n    return 42\n```\nAnswer: 42", "stop"


def test_all_protocol_tasks_run_end_to_end_and_compare(tmp_path):
    common = ["--base-url", None, "--n-items", "3", "--epochs", "1", "--max-retries", "0"]
    with FakeServer(chat_reply=canned) as srv:
        common[1] = srv.base_url
        a, b = tmp_path / "a", tmp_path / "b"
        assert run_tasks.main([*common, "--out", str(a), "--seed", "5"]) == 0
        assert (
            run_tasks.main([*common, "--out", str(b), "--items-from", str(a / "items.json")]) == 0
        )

    info = json.loads((a / "run_info.json").read_text())
    assert set(info["tasks"]) == set(run_tasks.TASKS)
    assert info["request_errors"] == 0
    assert info["n_samples"] == 3 * len(run_tasks.TASKS)
    rows = [json.loads(x) for x in (a / "samples.jsonl").read_text().splitlines()]
    assert all(r["score"] in (0.0, 1.0) for r in rows)
    assert {r["task"] for r in rows} == set(run_tasks.TASKS)

    result = compare.compare_runs(a, b, n_boot=50, seed=0)
    assert set(result["tasks"]) == set(run_tasks.TASKS)
    assert all(t["same_items_hash"] for t in result["tasks"].values())
    assert all(t["n_items"] == 3 for t in result["tasks"].values())
