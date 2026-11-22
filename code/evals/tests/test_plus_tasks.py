"""Tests for plus_tasks.py: HumanEval+ / MBPP+ prompt and program assembly (no sandbox needed).

The docker-marked test at the bottom runs the real sandbox and needs network + Docker.
"""

import subprocess
import sys

import pytest

import plus_tasks
from plus_tasks import (
    assemble_humaneval_program,
    assemble_mbpp_program,
    extract_code,
    mbpp_prompt,
)

PROMPT = '''from typing import List


def add_all(xs: List[int]) -> int:
    """ Sum the numbers. """
'''
TEST = """
def check(candidate):
    assert candidate([1, 2, 3]) == 6
    assert candidate([]) == 0
"""


def run_program(program: str) -> bool:
    proc = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=30, check=False
    )
    return proc.returncode == 0


def test_extract_code_prefers_the_python_fenced_block():
    text = "Here you go:\n```python\ndef f():\n    return 1\n```\nbye"
    assert extract_code(text) == "def f():\n    return 1\n"


def test_extract_code_falls_back_to_the_whole_completion_without_a_fence():
    assert extract_code("def f():\n    return 1\n") == "def f():\n    return 1\n"


def test_a_full_function_answer_is_run_with_the_prompt_stub_and_helpers_in_scope():
    answer = "def add_all(xs: List[int]) -> int:\n    return sum(xs)\n"
    program = assemble_humaneval_program(PROMPT, answer, TEST, "add_all")
    assert run_program(program)  # `List` resolves thanks to the prompt's import


def test_a_wrong_function_answer_fails_the_check():
    answer = "def add_all(xs):\n    return len(xs)\n"
    assert not run_program(assemble_humaneval_program(PROMPT, answer, TEST, "add_all"))


def test_a_body_only_answer_is_appended_to_the_prompt():
    body = "    return sum(xs)\n"
    assert run_program(assemble_humaneval_program(PROMPT, body, TEST, "add_all"))


def test_helper_functions_before_the_main_function_are_kept():
    answer = "def _total(xs):\n    return sum(xs)\n\n\ndef add_all(xs):\n    return _total(xs)\n"
    assert run_program(assemble_humaneval_program(PROMPT, answer, TEST, "add_all"))


def test_mbpp_program_puts_test_imports_first_then_the_answer_then_the_tests():
    answer = "def sq(x):\n    return x * x\n"
    test = "assert sq(3) == 9\nassert sq(0) == 0\n"
    program = assemble_mbpp_program(["import math"], answer, test)
    assert program.index("import math") < program.index("def sq") < program.index("assert sq(3)")
    assert run_program(program)
    assert not run_program(assemble_mbpp_program([], "def sq(x):\n    return x\n", test))


def test_mbpp_prompt_shows_the_task_and_the_first_test_only():
    prompt = mbpp_prompt(
        "Write a function to square a number.", ["assert sq(3) == 9", "assert sq(2) == 4"]
    )
    assert "Write a function to square a number." in prompt
    assert "assert sq(3) == 9" in prompt
    assert "assert sq(2) == 4" not in prompt
    assert "```python" in prompt


def test_a_very_large_test_script_is_delivered_as_a_file_not_a_command_line_argument():
    """Regression: some plus test scripts exceed the OS limit on one argv entry."""
    from inspect_ai import Task
    from inspect_ai import eval as inspect_eval
    from inspect_ai.dataset import Sample
    from inspect_ai.scorer import CORRECT, INCORRECT, Score, scorer
    from inspect_ai.solver import generate

    import plus_tasks

    big = "# " + "x" * 1_500_000 + "\nassert 1 + 1 == 2\n"

    @scorer(metrics=[])
    def big_program():
        async def score(state, target):
            result = await plus_tasks._run(big)
            return Score(value=CORRECT if result.success else INCORRECT, explanation=result.stderr)

        return score

    task = Task(
        dataset=[Sample(input="hi", target="x")],
        solver=generate(),
        scorer=big_program(),
        sandbox="local",
    )
    log = inspect_eval(
        task, model="mockllm/model", display="none", log_dir=str(__import__("tempfile").mkdtemp())
    )[0]
    score = next(iter(log.samples[0].scores.values()))
    assert score.value == "C", score.explanation


# --------------------------------------------------------------------------- sandbox modes


def test_the_docker_sandbox_is_the_default_and_local_is_inspect_ais_local_sandbox():
    assert plus_tasks.sandbox_spec("docker") == plus_tasks.SANDBOX
    assert plus_tasks.SANDBOX[0] == "docker"
    assert plus_tasks.sandbox_spec("local") == "local"
    with pytest.raises(ValueError, match="sandbox"):
        plus_tasks.sandbox_spec("chroot")


def test_the_local_mode_runs_the_interpreter_of_the_harness_and_docker_mode_the_images_python():
    import sys

    assert plus_tasks.python_for("docker") == "python"
    assert plus_tasks.python_for("local") == sys.executable


class _RecordingBox:
    """A sandbox double that records what _run asks of it."""

    def __init__(self) -> None:
        self.files: dict[str, str] = {}
        self.exec_calls: list[dict] = []

    async def write_file(self, file: str, contents: str) -> None:
        self.files[file] = contents

    async def exec(self, cmd: list[str], timeout: int | None = None):
        from inspect_ai.util import ExecResult

        self.exec_calls.append({"cmd": cmd, "timeout": timeout})
        return ExecResult(True, 0, "", "")


@pytest.mark.parametrize("mode", ["docker", "local"])
def test_the_same_wall_clock_limit_and_a_program_file_are_used_in_both_modes(monkeypatch, mode):
    import asyncio

    box = _RecordingBox()
    monkeypatch.setattr(plus_tasks, "sandbox", lambda: box)
    program = "print('hi')\n" * 10
    asyncio.run(plus_tasks._run(program, plus_tasks.python_for(mode)))
    (call,) = box.exec_calls
    assert call["timeout"] == plus_tasks.VERIFY_TIMEOUT
    assert call["cmd"] == [plus_tasks.python_for(mode), "verify_program.py"]
    assert box.files == {"verify_program.py": program}  # a file, never python -c


def _local_scoring_task(name: str, completion: str, sample: "Sample"):  # noqa: F821
    """Run a plus scorer in the real local sandbox on a canned completion; return the Score."""
    import tempfile

    from inspect_ai import Task
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import ModelOutput
    from inspect_ai.solver import Generate, TaskState, solver

    @solver
    def canned():
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            state.output = ModelOutput.from_content("canned", completion)
            return state

        return solve

    scorer = (
        plus_tasks.verify_humaneval_plus(python=plus_tasks.python_for("local"))
        if name == "humaneval_plus"
        else plus_tasks.verify_mbpp_plus(python=plus_tasks.python_for("local"))
    )
    task = Task(
        dataset=[sample], solver=canned(), scorer=scorer, sandbox=plus_tasks.sandbox_spec("local")
    )
    log = inspect_eval(
        task, model="mockllm/model", display="none", log_dir=tempfile.mkdtemp(), fail_on_error=False
    )[0]
    assert log.samples[0].error is None
    return next(iter(log.samples[0].scores.values()))


def _humaneval_toy_sample():
    from inspect_ai.dataset import Sample

    return Sample(
        id="t/0",
        input="x",
        target="    return sum(xs)\n",
        metadata={"prompt": PROMPT, "test": TEST, "entry_point": "add_all"},
    )


def test_the_humaneval_scorer_passes_a_good_answer_and_fails_a_bad_one_in_the_local_sandbox():
    good = _local_scoring_task(
        "humaneval_plus",
        "```python\ndef add_all(xs):\n    return sum(xs)\n```",
        _humaneval_toy_sample(),
    )
    bad = _local_scoring_task(
        "humaneval_plus", "```python\ndef add_all(xs):\n    return 0\n```", _humaneval_toy_sample()
    )
    assert (good.value, bad.value) == ("C", "I")


def test_the_mbpp_scorer_runs_numpy_tests_in_the_local_sandbox():
    from inspect_ai.dataset import Sample

    sample = Sample(
        id="1",
        input="x",
        target="",
        metadata={
            "test": "assert sq(3) == 9\nimport numpy\nassert numpy.sqrt(sq(2)) == 2\n",
            "test_imports": [],
        },
    )
    good = _local_scoring_task("mbpp_plus", "```python\ndef sq(x):\n    return x * x\n```", sample)
    bad = _local_scoring_task("mbpp_plus", "```python\ndef sq(x):\n    return x\n```", sample)
    assert (good.value, bad.value) == ("C", "I")


def test_a_program_that_outlives_the_wall_clock_limit_fails_in_the_local_sandbox(monkeypatch):
    import time

    monkeypatch.setattr(plus_tasks, "VERIFY_TIMEOUT", 2)
    sample = _humaneval_toy_sample()
    slow = "```python\nimport time\n\ndef add_all(xs):\n    time.sleep(60)\n    return 0\n```"
    start = time.monotonic()
    score = _local_scoring_task("humaneval_plus", slow, sample)
    assert score.value == "I"
    assert time.monotonic() - start < 30
    assert "timed out" in score.explanation


@pytest.mark.network
@pytest.mark.parametrize(
    "mode",
    ["local", pytest.param("docker", marks=pytest.mark.docker)],
)
@pytest.mark.parametrize("name", ["humaneval_plus", "mbpp_plus"])
def test_canonical_solutions_pass_and_broken_ones_fail_in_the_real_sandbox(name, mode, tmp_path):
    """Needs the evalplus datasets on the Hugging Face hub (and Docker for the docker mode)."""
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import ModelOutput, get_model

    import plus_tasks

    task = getattr(plus_tasks, name)(sandbox_mode=mode)
    samples = list(task.dataset)[:3]
    if name == "humaneval_plus":
        good = [s.metadata["prompt"] + s.target for s in samples]
        bad = [s.metadata["prompt"] + "    return None\n" for s in samples]
    else:
        good = [s.target for s in samples]
        bad = ["pass" for _ in samples]

    def values(codes: list[str]) -> list[str]:
        outputs = [ModelOutput.from_content("mockllm/model", f"```python\n{c}\n```") for c in codes]
        model = get_model("mockllm/model", custom_outputs=outputs)
        log = inspect_eval(
            getattr(plus_tasks, name)(sandbox_mode=mode),
            model=model,
            limit=3,
            max_samples=1,  # mockllm hands out its outputs in call order
            display="none",
            log_dir=str(tmp_path),
        )[0]
        return [next(iter(s.scores.values())).value for s in log.samples]

    assert values(good) == ["C", "C", "C"]
    assert values(bad) == ["I", "I", "I"]
