"""Tests of the 8-gram leakage audit: planted positives are caught, negatives are not, any hit refuses."""

import json
from pathlib import Path

import pytest

import leakage_audit as la

HUMANEVAL = (
    "Check if in given list of numbers, are any two numbers closer to each other than given threshold. "
    "Then return the sorted list of the numbers that were found."
)
AIME = "Find the sum of all integer bases b greater than 9 for which 17 base b is a divisor of 97 base b."


def _index(n: int = 8):
    items = [
        la.EvalItem("humaneval_plus", "HumanEval/0", "input", HUMANEVAL),
        la.EvalItem("aime2025", "2025-I-1", "input", AIME),
        la.EvalItem("mgsm", "bn-3", "target", "42"),  # too short to audit
    ]
    return la.build_index(items, n=n, min_words=8)


def test_normalize_lowercases_strips_punctuation_and_splits_on_whitespace() -> None:
    assert la.normalize("Hello,   WORLD!\nIt's -- fine.") == ["hello", "world", "it", "s", "fine"]


def test_a_verbatim_copy_of_a_test_item_is_caught_with_its_task_and_id() -> None:
    doc = "Some preface text here. " + HUMANEVAL + " And a trailing sentence."
    rep = la.audit_documents([("doc0", doc)], _index())
    assert not rep.ok
    assert {(h.task, h.item_id) for h in rep.hits} == {("humaneval_plus", "HumanEval/0")}
    assert rep.by_task == {"humaneval_plus": len(rep.hits)}
    assert rep.hits[0].doc == "doc0"


def test_case_and_punctuation_changes_do_not_hide_a_copy() -> None:
    doc = "CHECK IF, in given list of numbers -- are any two numbers closer to each other than GIVEN threshold!"
    assert not la.audit_documents([("d", doc)], _index()).ok


def test_seven_shared_words_are_not_a_hit_but_eight_are() -> None:
    seven = "check if in given list of numbers purple elephants"
    eight = "check if in given list of numbers are"
    assert la.audit_documents([("d", seven)], _index()).ok
    assert not la.audit_documents([("d", eight)], _index()).ok


def test_unrelated_text_passes() -> None:
    doc = "The quick brown fox jumps over the lazy dog while the committee adjourned for lunch and nobody noticed."
    rep = la.audit_documents([("d", doc)], _index())
    assert rep.ok and rep.hits == []


def test_short_item_fields_are_skipped_and_counted() -> None:
    idx = _index()
    assert idx.n_skipped_short == 1
    assert idx.n_items_indexed == 2
    assert la.audit_documents([("d", "the answer is 42 and that is all there is to it today")], idx).ok


def test_n_is_configurable() -> None:
    doc = "closer to each other than given"
    assert la.audit_documents([("d", doc)], _index(n=8)).ok
    assert not la.audit_documents([("d", doc)], _index(n=5)).ok


def test_two_documents_are_reported_separately() -> None:
    rep = la.audit_documents([("clean", "nothing to see here at all in this short text"), ("dirty", AIME)], _index())
    assert {h.doc for h in rep.hits} == {"dirty"}
    assert rep.n_docs == 2


def test_load_eval_items_reads_the_extraction_format(tmp_path: Path) -> None:
    path = tmp_path / "items.json"
    path.write_text(
        json.dumps(
            {"schema": "eval-items/1", "tasks": {"aime2025": {"items": [{"id": "1", "fields": {"input": AIME}}]}}}
        )
    )
    items = la.load_eval_items(path)
    assert [(i.task, i.item_id, i.field) for i in items] == [("aime2025", "1", "input")]
    assert la.sha256_file(path) == la.sha256_file(path)


def test_load_eval_items_rejects_an_unknown_schema_and_an_empty_set(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "other/9", "tasks": {}}))
    with pytest.raises(ValueError):
        la.load_eval_items(bad)
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"schema": "eval-items/1", "tasks": {"aime2025": {"items": []}}}))
    with pytest.raises(ValueError):
        la.load_eval_items(empty)


def test_the_five_task_names_are_required_by_the_cli(tmp_path: Path) -> None:
    items = tmp_path / "items.json"
    items.write_text(
        json.dumps(
            {"schema": "eval-items/1", "tasks": {"aime2025": {"items": [{"id": "1", "fields": {"input": AIME}}]}}}
        )
    )
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "calib_text.jsonl").write_text(json.dumps({"source": "s", "doc": "d", "text": "hello world"}) + "\n")
    # only one of the five tasks present: refuse unless the caller says so
    assert la.main(["--items", str(items), "--bundle", str(bundle)]) == 2
    assert la.main(["--items", str(items), "--bundle", str(bundle), "--allow-missing-tasks"]) == 0


def test_cli_exit_code_is_one_on_overlap_and_writes_the_report(tmp_path: Path) -> None:
    items = tmp_path / "items.json"
    items.write_text(
        json.dumps(
            {
                "schema": "eval-items/1",
                "tasks": {t: {"items": [{"id": "1", "fields": {"input": AIME}}]} for t in la.REQUIRED_TASKS},
            }
        )
    )
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "calib_text.jsonl").write_text(json.dumps({"source": "s", "doc": "d", "text": AIME}) + "\n")
    report = tmp_path / "report.json"
    code = la.main(["--items", str(items), "--bundle", str(bundle), "--report", str(report)])
    assert code == 1
    data = json.loads(report.read_text())
    assert data["ok"] is False and data["n_hits"] > 0 and data["n"] == 8
    (bundle / "calib_text.jsonl").write_text(json.dumps({"source": "s", "doc": "d", "text": "clean text only"}) + "\n")
    assert la.main(["--items", str(items), "--bundle", str(bundle), "--report", str(report)]) == 0
    assert json.loads(report.read_text())["ok"] is True
