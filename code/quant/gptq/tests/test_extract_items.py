"""Tests of the eval-item extraction helper: it uses the harness task factories and never runs a model."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import extract_eval_items as ex


def _sample(i, inp, target, **meta):
    return SimpleNamespace(id=i, input=inp, target=target, choices=None, metadata=meta or None)


class _Task:
    def __init__(self, samples) -> None:
        self.dataset = samples


def _spec(samples, takes_sandbox=False, calls=None):
    def factory(**kw):
        if calls is not None:
            calls.append(kw)
        return _Task(samples)

    return SimpleNamespace(factory=factory, takes_sandbox=takes_sandbox)


def test_items_carry_input_target_and_all_string_metadata() -> None:
    samples = [_sample("a", "What is 2+2?", ["4"], canonical_solution="def f(): return 4", tests=["assert f()==4"])]
    items = ex.items_from_samples(samples)
    assert items == [
        {
            "id": "a",
            "fields": {
                "input": "What is 2+2?",
                "target": "4",
                "metadata.canonical_solution": "def f(): return 4",
                "metadata.tests.0": "assert f()==4",
            },
        }
    ]


def test_chat_message_inputs_are_flattened() -> None:
    msgs = [SimpleNamespace(content="first part"), SimpleNamespace(content=[SimpleNamespace(text="second part")])]
    items = ex.items_from_samples([_sample(1, msgs, "x")])
    assert items[0]["fields"]["input"] == "first part\nsecond part"
    assert items[0]["id"] == "1"


def test_samples_without_an_id_get_their_position() -> None:
    items = ex.items_from_samples([_sample(None, "p", "t"), _sample(None, "q", "u")])
    assert [i["id"] for i in items] == ["0", "1"]


def test_extract_calls_each_factory_without_a_model_and_with_the_local_sandbox_where_asked() -> None:
    calls_a, calls_b = [], []
    tasks = {
        "mgsm": _spec([_sample("1", "q", "1")], calls=calls_a),
        "humaneval_plus": _spec([_sample("2", "q2", "t2")], takes_sandbox=True, calls=calls_b),
    }
    out = ex.extract(tasks, names=["mgsm", "humaneval_plus"])
    assert calls_a == [{}]
    assert calls_b == [{"sandbox": "local"}]
    assert out["schema"] == "eval-items/1"
    assert set(out["tasks"]) == {"mgsm", "humaneval_plus"}
    assert out["tasks"]["mgsm"]["n_items"] == 1


def test_extract_fails_loudly_for_an_unknown_task_and_for_an_empty_dataset() -> None:
    with pytest.raises(KeyError):
        ex.extract({}, names=["mgsm"])
    with pytest.raises(ValueError):
        ex.extract({"mgsm": _spec([])}, names=["mgsm"])


def test_the_output_is_what_the_audit_reads(tmp_path: Path) -> None:
    import leakage_audit as la

    tasks = {
        n: _spec([_sample("1", "alpha beta gamma delta epsilon zeta eta theta iota", "k")]) for n in la.REQUIRED_TASKS
    }
    path = tmp_path / "items.json"
    path.write_text(json.dumps(ex.extract(tasks, names=list(la.REQUIRED_TASKS))))
    items = la.load_eval_items(path)
    assert {i.task for i in items} == set(la.REQUIRED_TASKS)
    assert la.REQUIRED_TASKS == ("mgsm", "humaneval_plus", "mbpp_plus", "ifeval", "aime2025")
