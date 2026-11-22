"""HumanEval+ and MBPP+ as inspect_ai tasks.

inspect_evals 0.29.0 ships only the *base* HumanEval and MBPP (the latter few-shot, 0.5
temperature, no plus tests). EvalPlus publishes the extended test suites as Hugging Face
datasets (``evalplus/humanevalplus``, ``evalplus/mbppplus``) whose ``test`` field is an
executable Python script, so a faithful minimal task only needs a prompt, an answer extractor
and a sandboxed executor. By default model code runs in the Docker sandbox of ``plus_sandbox/``
(Python 3.12 + numpy, no network, 1 CPU, 2 GB), never on the host. ``sandbox_mode="local"`` runs
it instead in a subprocess of the harness through inspect_ai's local sandbox, for a machine that
has no Docker (a RunPod pod is itself a container); that mode has NO isolation beyond a fresh
temporary working directory per problem, so use it only on a disposable machine that holds no
secret (see DECISIONS.md).

What is *not* identical to the EvalPlus command-line tool: EvalPlus times each test input
separately (a multiple of the reference runtime); here the whole script has one wall-clock
limit (``VERIFY_TIMEOUT``). The prompts are the zero-shot chat prompts written below; OVH's
exact prompts are not published (see DECISIONS.md).
"""

import sys
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import Solver, TaskState, generate
from inspect_ai.util import ExecResult, sandbox
from inspect_evals.utils.code import extract_code_block
from inspect_evals.utils.huggingface import hf_dataset

HUMANEVAL_PLUS_PATH = "evalplus/humanevalplus"
HUMANEVAL_PLUS_REVISION = "d32357cf319e50e9c8d8dab5ea876c72b0fd321b"
MBPP_PLUS_PATH = "evalplus/mbppplus"
MBPP_PLUS_REVISION = "b2d74c91837c3f2a20c1299ae98133cbe7cfa077"

VERIFY_TIMEOUT = 60
"""Wall-clock seconds for one problem's whole plus test script."""

SANDBOX = ("docker", str(Path(__file__).parent / "plus_sandbox" / "compose.yaml"))
SANDBOX_MODES = ("docker", "local")
"""Where model code runs: the Docker sandbox of ``plus_sandbox/`` or inspect_ai's local sandbox."""

HUMANEVAL_INSTRUCTION = (
    "Complete the following Python function. Reply with the complete function in one "
    "```python code block.\n\n"
)

MBPP_INSTRUCTION = (
    "You are an expert Python programmer. Solve the task below. Reply with a self-contained "
    "Python implementation in one ```python code block.\n\n"
)


def extract_code(completion: str) -> str:
    """Return the first Python code block of a completion, or the whole text if it has none.

    Args:
        completion: The model output.

    Returns:
        The code to execute.
    """
    block = extract_code_block(completion, "python")
    return completion if block is None else block.replace("\r\n", "\n")


def assemble_humaneval_program(prompt: str, answer: str, test: str, entry_point: str) -> str:
    """Build the script that checks one HumanEval+ answer.

    If the answer defines ``entry_point`` it is run after the prompt (so imports and helper
    functions of the prompt stay in scope and the answer overrides the stub); otherwise the
    answer is taken to be the function body and appended to the prompt.

    Args:
        prompt: Signature and docstring (plus any helper code) from the dataset.
        answer: Extracted model code.
        test: The dataset's ``test`` script (defines ``check``).
        entry_point: Name of the function under test.

    Returns:
        Source code to execute in the sandbox.
    """
    check = f"\ncheck({entry_point})\n"
    if f"def {entry_point}" in answer:
        return f"{prompt}    pass\n\n{answer}\n\n{test}{check}"
    return f"{prompt}{answer}\n\n{test}{check}"


def assemble_mbpp_program(test_imports: list[str], answer: str, test: str) -> str:
    """Build the script that checks one MBPP+ answer: imports, answer, then the plus tests.

    Args:
        test_imports: Import statements required by the tests.
        answer: Extracted model code.
        test: The dataset's ``test`` script (assertions calling the target function).

    Returns:
        Source code to execute in the sandbox.
    """
    return "\n".join([*test_imports, answer, "", test])


def mbpp_prompt(task_text: str, test_list: list[str]) -> str:
    """Zero-shot MBPP+ prompt: the task and the first (public) assertion only.

    Args:
        task_text: The natural-language task.
        test_list: The base assertions; only the first is shown.

    Returns:
        The user prompt.
    """
    return (
        f"{MBPP_INSTRUCTION}{task_text}\n\nYour code must pass this test:\n"
        f"```python\n{test_list[0]}\n```\n"
    )


def sandbox_spec(mode: str) -> tuple[str, str] | str:
    """The inspect_ai sandbox specification of a mode.

    Args:
        mode: ``docker`` or ``local``.

    Returns:
        ``SANDBOX`` (docker compose) or ``"local"``.

    Raises:
        ValueError: For any other mode.
    """
    if mode == "docker":
        return SANDBOX
    if mode == "local":
        return "local"
    raise ValueError(f"unknown sandbox mode {mode!r} (use one of {', '.join(SANDBOX_MODES)})")


def python_for(mode: str) -> str:
    """The interpreter that runs a program in a sandbox mode.

    Args:
        mode: ``docker`` (the image's ``python``) or ``local`` (the interpreter running the
            harness, which has the same numpy as the image: the plus tests import it).

    Returns:
        The executable name or path.
    """
    sandbox_spec(mode)
    return sys.executable if mode == "local" else "python"


async def _run(program: str, python: str = "python") -> ExecResult[str]:
    """Write ``program`` into the sandbox and run it.

    The script goes through a file, not ``python -c``: some plus test scripts are larger than
    the operating system allows for a single command-line argument. The wall-clock limit is the
    same in every sandbox mode.
    """
    box = sandbox()
    await box.write_file("verify_program.py", program)
    try:
        return await box.exec(cmd=[python, "verify_program.py"], timeout=VERIFY_TIMEOUT)
    except TimeoutError:
        return ExecResult(False, 1, "", "Verification timed out.")


def _score(result: ExecResult[str], answer: str, program: str) -> Score:
    explanation = "" if result.success else f"Program:\n{program[-2000:]}\n\n{result.stderr}"
    return Score(
        value=CORRECT if result.success else INCORRECT, answer=answer, explanation=explanation
    )


@scorer(metrics=[accuracy(), stderr()])
def verify_humaneval_plus(python: str = "python") -> Scorer:
    """Execute the answer against the HumanEval+ tests in the sandbox.

    Args:
        python: Interpreter to run the program with (see ``python_for``).
    """

    async def score(state: TaskState, target: Target) -> Score:
        answer = extract_code(state.output.completion)
        program = assemble_humaneval_program(
            state.metadata["prompt"], answer, state.metadata["test"], state.metadata["entry_point"]
        )
        return _score(await _run(program, python), answer, program)

    return score


@scorer(metrics=[accuracy(), stderr()])
def verify_mbpp_plus(python: str = "python") -> Scorer:
    """Execute the answer against the MBPP+ tests in the sandbox.

    Args:
        python: Interpreter to run the program with (see ``python_for``).
    """

    async def score(state: TaskState, target: Target) -> Score:
        answer = extract_code(state.output.completion)
        program = assemble_mbpp_program(
            state.metadata["test_imports"], answer, state.metadata["test"]
        )
        return _score(await _run(program, python), answer, program)

    return score


def _humaneval_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        id=record["task_id"],
        input=HUMANEVAL_INSTRUCTION + record["prompt"],
        target=record["canonical_solution"],
        metadata={
            "prompt": record["prompt"],
            "test": record["test"],
            "entry_point": record["entry_point"],
        },
    )


def _mbpp_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        id=str(record["task_id"]),
        input=mbpp_prompt(record["prompt"], record["test_list"]),
        target=record["code"],
        metadata={"test": record["test"], "test_imports": list(record["test_imports"])},
    )


@task
def humaneval_plus(solver: Solver | None = None, sandbox_mode: str = "docker") -> Task:
    """HumanEval+ (164 problems, EvalPlus extended tests), zero-shot, sandboxed scoring.

    Args:
        solver: Replaces ``generate()`` (the canonical-solution check uses this).
        sandbox_mode: ``docker`` (default) or ``local``; see the module docstring.
    """
    return Task(
        dataset=hf_dataset(
            path=HUMANEVAL_PLUS_PATH,
            split="test",
            sample_fields=_humaneval_sample,
            revision=HUMANEVAL_PLUS_REVISION,
        ),
        solver=solver or generate(),
        scorer=verify_humaneval_plus(python_for(sandbox_mode)),
        sandbox=sandbox_spec(sandbox_mode),
    )


@task
def mbpp_plus(solver: Solver | None = None, sandbox_mode: str = "docker") -> Task:
    """MBPP+ (378 problems, EvalPlus extended tests), zero-shot, sandboxed scoring.

    Args:
        solver: Replaces ``generate()`` (the canonical-solution check uses this).
        sandbox_mode: ``docker`` (default) or ``local``; see the module docstring.
    """
    return Task(
        dataset=hf_dataset(
            path=MBPP_PLUS_PATH,
            split="test",
            sample_fields=_mbpp_sample,
            revision=MBPP_PLUS_REVISION,
        ),
        solver=solver or generate(),
        scorer=verify_mbpp_plus(python_for(sandbox_mode)),
        sandbox=sandbox_spec(sandbox_mode),
    )
