"""Shared fixtures: a tiny random Olala (4 layers MMVM, 8 Mamba heads of 16 channels, 16 experts)."""

import copy
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
GPTQ = HERE.parent / "quant" / "gptq"
for p in (str(HERE), str(GPTQ)):
    if p not in sys.path:
        sys.path.insert(0, p)

import activation_probe as ap  # noqa: E402

TINY_SPECIAL = [0, 1, 2, ap.TINY_IM_START, ap.TINY_IM_END]


@pytest.fixture(scope="session")
def tiny_base():
    return ap.build_tiny_model(seed=0)


@pytest.fixture()
def tiny(tiny_base):
    """A fresh deep copy of the tiny model, safe to plant spikes in."""
    return copy.deepcopy(tiny_base)


@pytest.fixture()
def tokens():
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(4, 200, (1, 40), generator=g)
    ids[0, 0] = 1  # the BOS-like first token
    ids[0, 9] = ap.TINY_IM_START
    return ids
