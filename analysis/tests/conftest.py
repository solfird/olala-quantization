"""Paths for the tests of the analysis scripts: ``analysis/`` itself and the evaluation code of ``code/evals``."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
for p in (str(HERE), str(HERE.parent / "code" / "evals")):
    if p not in sys.path:
        sys.path.insert(0, p)
