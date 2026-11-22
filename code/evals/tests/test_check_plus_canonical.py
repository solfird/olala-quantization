"""Tests for the pure part of check_plus_canonical.py."""

import json

import pytest

from check_plus_canonical import build_parser, canonical_completion, main, summarise


def test_humaneval_canonical_answer_is_prompt_plus_solution_in_a_code_block():
    out = canonical_completion("humaneval_plus", prompt="def f():\n", target="    return 1\n")
    assert out == "```python\ndef f():\n    return 1\n\n```"


def test_mbpp_canonical_answer_is_the_reference_code_alone():
    out = canonical_completion("mbpp_plus", prompt="", target="def f():\n    return 1")
    assert out == "```python\ndef f():\n    return 1\n```"


def test_summary_counts_correct_errors_and_lists_the_failing_ids():
    out = summarise([("a", "C", False), ("b", "I", False), ("c", None, True)])
    assert out == {"n": 3, "correct": 1, "errors": 1, "failing_ids": ["b", "c"]}


def test_the_sandbox_option_defaults_to_docker_and_accepts_local():
    assert build_parser().parse_args(["--out", "x"]).sandbox == "docker"
    assert build_parser().parse_args(["--out", "x", "--sandbox", "local"]).sandbox == "local"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--out", "x", "--sandbox", "chroot"])


@pytest.mark.network
def test_every_canonical_solution_passes_in_the_local_sandbox_except_the_two_known_dataset_faults(
    tmp_path,
):
    """Needs the evalplus datasets on the Hub. Same counts as the docker run in DECISIONS.md."""
    out = tmp_path / "canonical-local.json"
    assert main(["--sandbox", "local", "--out", str(out)]) == 1  # the two faults make it non-zero
    report = json.loads(out.read_text())
    assert report["sandbox"] == "local"
    assert report["tasks"]["humaneval_plus"]["n"] == 164
    assert report["tasks"]["humaneval_plus"]["correct"] == 163
    assert report["tasks"]["humaneval_plus"]["errors"] == 0
    assert report["tasks"]["humaneval_plus"]["failing_ids"] == ["HumanEval/32"]
    assert report["tasks"]["mbpp_plus"]["n"] == 378
    assert report["tasks"]["mbpp_plus"]["correct"] == 377
    assert report["tasks"]["mbpp_plus"]["errors"] == 0
    assert report["tasks"]["mbpp_plus"]["failing_ids"] == ["590"]
