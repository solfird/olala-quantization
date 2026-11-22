"""Opt-in: the real HumanEval+ and MBPP+ datasets through run_tasks.py, local sandbox (no Docker).

The model is the fake server (canned replies); nothing here measures a model. Needs the Hugging
Face hub: run with ``uv run pytest -m network tests/test_local_sandbox_real.py``.
"""

import json

import pytest
from fake_server import FakeServer

import run_tasks

pytestmark = pytest.mark.network


def canned(body: dict) -> tuple[str, str]:
    return "```python\ndef f():\n    return 42\n```", "stop"


def test_code_tasks_run_with_the_local_sandbox_and_default_to_one_epoch(tmp_path):
    out = tmp_path / "local"
    args = [
        "--tasks", "humaneval_plus", "mbpp_plus",
        "--n-items", "3",
        "--max-retries", "0",
        "--sandbox", "local",
        "--out", str(out),
    ]  # fmt: skip
    with FakeServer(chat_reply=canned) as srv:
        assert run_tasks.main(["--base-url", srv.base_url, *args]) == 0
    info = json.loads((out / "run_info.json").read_text())
    assert info["sandbox"] == {"mode": "local", "tasks": ["humaneval_plus", "mbpp_plus"]}
    assert {t["epochs"] for t in info["tasks"].values()} == {1}  # the protocol's default
    rows = [json.loads(x) for x in (out / "samples.jsonl").read_text().splitlines()]
    assert len(rows) == 6
    assert all(r["error"] is None and r["score"] in (0.0, 1.0) for r in rows)
