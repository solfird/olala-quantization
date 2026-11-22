"""Shared fixtures: one tiny random Olala checkpoint per test session (built with the vendored OVH code)."""

from pathlib import Path

import pytest

import tiny_model


@pytest.fixture(scope="session")
def tiny_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tiny_model.write_tiny_checkpoint(tmp_path_factory.mktemp("tiny-olala-bf16"), seed=0)


@pytest.fixture(scope="session")
def tiny_modules():
    from olala_model import load_olala_modules

    return load_olala_modules()
