"""Load the vendored OVH modeling code with the environment that G1-lite needs, and build single blocks.

The environment variables must be set BEFORE ``vendor.modeling_olala`` is imported, because the file reads them at
import time: ``OLALA_TORCH_MOE=1`` selects the pure-PyTorch expert loop (``TorchMoEMLP``), the only one that exposes
the rows each expert sees; ``OLALA_TORCH_MAMBA3=1`` selects the pure-PyTorch Mamba-3 chunk loop (no TileLang).
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parent
VENDOR_DIR = HERE / "vendor"
VENDORED_FILES = ("modeling_olala.py", "configuration_olala.py")
_ENV = {"OLALA_TORCH_MOE": "1", "OLALA_TORCH_MAMBA3": "1", "HF_HUB_OFFLINE": "1"}


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_vendored_matches(model_dir: Path) -> dict[str, bool]:
    """Compare the vendored OVH files with the copies that sit next to the real weights.

    Returns:
        ``{file name: identical}`` for each vendored file that exists in ``model_dir``. A missing file is reported as
        ``False`` so that the caller refuses to run on an unknown modeling file.
    """
    return {
        name: (model_dir / name).is_file() and sha256_file(model_dir / name) == sha256_file(VENDOR_DIR / name)
        for name in VENDORED_FILES
    }


def load_olala_modules() -> tuple[ModuleType, ModuleType]:
    """Import ``(modeling_olala, configuration_olala)`` from ``vendor/`` with the G1-lite environment.

    The environment is forced (not ``setdefault``): a stray ``OLALA_TORCH_MOE=0`` would silently hide the expert rows.
    """
    for key, value in _ENV.items():
        os.environ[key] = value
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    modeling = importlib.import_module("vendor.modeling_olala")
    config = importlib.import_module("vendor.configuration_olala")
    if modeling.MOE_IMPL != "torch":
        raise RuntimeError(f"vendored model uses MOE_IMPL={modeling.MOE_IMPL!r}, expected 'torch'")
    return modeling, config
