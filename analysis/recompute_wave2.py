r"""Recompute the accuracy analyses of the report (``results/wave2-*.json`` and ``results/wave1-analysis.json``) and compare them with the shipped files.

    uv run --python 3.12 --with numpy==2.5.5 python analysis/recompute_wave2.py [--only wave2-sel-analysis] [--keep DIR]

The command line of every analysis is rebuilt from the shipped JSON itself (arms and their run folders, candidates, exploratory arms, contrasts,
calibration pair, margin, seed, resamples), and ``code/evals/analyze_wave.py`` is run on the per-item records of ``results/wave1``. The run folders are
recorded in the JSON as ``../results/wave1/<run>``, relative to ``evals/`` in the private working repository, so the script runs the analysis from a
scratch folder that has the same shape (``scratch/evals`` next to a link ``scratch/results``).

What is compared: every leaf of the recomputed JSON against the shipped one. Two kinds of leaf are expected to differ and are listed apart:

* ``samples_sha256``: the hash of each run's ``samples.jsonl``. The shipped ``samples.jsonl`` files hold no benchmark text in their ``target`` and
  ``completion`` fields (``analysis/make_item_records.py``), so their hash is not the one of the private files that the analysis first read. The shipped
  JSON keeps the hash of the private file.
* ``run_info_sha256``: the same, for the ``run_info.json`` of the combined runs (``Q4v2``, ``Q5v2`` and the folders under ``merged/``). The 28 absolute
  paths of the private working machine in them were replaced by paths relative to the repository root; nothing else in those files changed.
* for ``wave1-analysis`` only: the ``/gate`` and ``/script`` records. That analysis was made by an earlier version of ``analyze_wave.py``, whose
  gate had no ``accept_legacy_aime_retries`` field, so its gate records and the script hash differ; its numbers are compared.

Any other difference is a failed reproduction: the script prints it and exits 1.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "evals"))

import analyze_wave as aw  # noqa: E402

LEGACY_SCRIPT = {"wave1-analysis"}
"""``wave1-analysis`` (the first, legacy wave) was made by an earlier version of ``analyze_wave.py`` (its recorded sha256 is not the shipped file's):
the ``/gate`` and ``/script`` records are not compared for it, every other leaf is."""
NAMES = ["wave1-analysis", "wave2-analysis", "wave2-n5-analysis", "wave2-g1-analysis", "wave2-q5-analysis", "wave2-sel-analysis", "wave2-q4-analysis",
         "wave2-fp8-contrast"]


def argv_of(shipped: dict, out_md: Path, out_json: Path) -> list[str]:
    """The ``analyze_wave.py`` command line that made ``shipped``."""
    a: list[str] = []
    cal = shipped.get("calibration")
    for name, arm in shipped["arms"].items():
        if arm["role"] == "calibration":
            continue
        a += ["--run", f"{name}=" + ",".join(r["dir"] for r in arm["runs"])]
    a += ["--reference", shipped["reference"], "--candidates", ",".join(shipped["candidates"])]
    if shipped["exploratory"]:
        a += ["--exploratory", ",".join(shipped["exploratory"])]
    for c in shipped["contrasts"].values():
        a += ["--contrast", f"{c['x']},{c['y']}"]
    if cal:
        x, y = cal["x"], cal["y"]
        a += ["--calibration", f"{x}={shipped['arms'][x]['runs'][0]['dir']},{y}={shipped['arms'][y]['runs'][0]['dir']}"]
    a += ["--margin", str(shipped["confirmatory"].get("margin", -3.0)), "--seed", str(shipped["bootstrap"]["seed"]), "--resamples", str(shipped["bootstrap"]["resamples"]),
          "--out", str(out_md), "--json", str(out_json)]
    return a


def leaves(x: object, path: str = ""):
    """Yield ``(path, value)`` of every leaf of a JSON value."""
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, x


def main() -> int:
    """Run every analysis and compare."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", default=[], help="a name of NAMES (default: all)")
    ap.add_argument("--keep", type=Path, default=None, help="folder to keep the recomputed files in")
    args = ap.parse_args()
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp)
        (scratch / "evals").mkdir()
        (scratch / "results").symlink_to(ROOT / "results")
        outdir = args.keep or scratch / "out"
        outdir.mkdir(parents=True, exist_ok=True)
        os.chdir(scratch / "evals")
        for name in args.only or NAMES:
            shipped = json.loads((ROOT / "results" / f"{name}.json").read_text())
            md, js = outdir / f"{name}.md", outdir / f"{name}.json"
            argv = argv_of(shipped, md, js)
            if name == "wave1-analysis":
                argv.append("--accept-legacy-aime-retries")
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                code = aw.main(argv)
            if code != 0:
                print(f"{name}: analyze_wave exited {code}: {err.getvalue()[:300]}")
                bad += 1
                continue
            new = json.loads(js.read_text())
            a, b = dict(leaves(shipped)), dict(leaves(new))
            diff = [p for p in sorted(set(a) | set(b)) if a.get(p) != b.get(p)]
            hashes = [p for p in diff if p.endswith(("samples_sha256", "run_info_sha256"))]
            legacy = [p for p in diff if p not in hashes and name in LEGACY_SCRIPT and p.startswith(("/gate", "/script"))]
            other = [p for p in diff if p not in hashes and p not in legacy]
            mask = lambda t: re.sub(r"\b[0-9a-f]{64}\b", "<sha256>", t)  # noqa: E731
            shipped_md, new_md = mask((ROOT / "results" / f"{name}.md").read_text()), mask(md.read_text())
            n_md = sum(1 for x, y in zip(shipped_md.splitlines(), new_md.splitlines(), strict=False) if x != y)
            if name in LEGACY_SCRIPT:
                same_md = f"not compared ({n_md} lines differ: the older script wrote another layout of the gate section)"
            else:
                same_md = "yes, apart from file hashes" if shipped_md == new_md else f"NO, {n_md} lines differ"
            print(f"{name}: {len(a)} leaves, {len(diff)} differ ({len(hashes)} file hashes, {len(legacy)} gate/script record of an older script, {len(other)} other); "
                  f"markdown identical: {same_md}")
            for p in other[:10]:
                print(f"   DIFFERS {p}: shipped {a.get(p)!r} recomputed {b.get(p)!r}")
            bad += bool(other) or same_md.startswith("NO")
    print("RESULT:", "every number reproduced" if not bad else f"{bad} analyses do not reproduce")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
