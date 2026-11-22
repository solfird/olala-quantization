"""A stand-in for ``olala_quant.ablation``'s command line, for the pod-script tests (no checkpoint, no GPU).

It mirrors only what the scripts depend on: ``expected`` and ``check`` with ``--select SPEC``, the exit status (0 ok, 1
mismatch, 2 refused), the canonical spelling of a spec (merged, sorted layer ranges) in the ``spec`` field of the JSON
that ``expected`` prints, and the ``ABLATION_CHECK_OK`` line. Every call is recorded in a calls log, so the tests read the
order of the steps. ``NAMES`` is checked against the real ``SELECT_NAMES`` by a test of the scripts' own spec grammar.
"""

from __future__ import annotations

NAMES = ("all", "experts", "non-experts", "shared-experts", "latent", "attention", "mamba")
LAST_LAYER = 55


def source(calls: object) -> str:
    """The module text of the stub ``olala_quant/ablation.py``; ``calls`` is the path of the calls log."""
    return f"""import json, os, sys
NAMES = {NAMES!r}
a = sys.argv[1:]
open({str(calls)!r}, "a").write("ablation " + " ".join(a) + "\\n")


def refuse(msg):
    print("error: " + msg, file=sys.stderr)
    sys.exit(2)


def canon(spec):
    name, *opts = spec.split(":")
    if name not in NAMES:
        refuse(f"unknown selection {{name!r}} in {{spec!r}}")
    if not opts:
        return name
    ranges = []
    for part in opts[0].split("=", 1)[1].split(","):
        lo, _, hi = part.partition("-")
        lo, hi = int(lo), int(hi or lo)
        if hi < lo:
            refuse(f"{{spec!r}}: the range {{part!r}} runs backwards")
        if hi > {LAST_LAYER}:
            refuse(f"{{spec!r}}: the model has layers 0 to {LAST_LAYER}, not {{lo}}-{{hi}}")
        ranges.append((lo, hi))
    merged = []
    for lo, hi in sorted(ranges):
        if merged and lo <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return name + ":layers=" + ",".join(str(x) if x == y else f"{{x}}-{{y}}" for x, y in merged)


spec = canon(a[a.index("--select") + 1])
rc = os.environ.get("EXPECTED_RC" if a[0] == "expected" else "CHECK_RC", "0")
if rc != "0":
    print("MISMATCH: stub told to fail", file=sys.stderr)
    sys.exit(int(rc))
if a[0] == "expected":
    print(json.dumps({{"spec": spec, "quantized_tensors": 40, "ignored_tensors": 219}}, indent=2))
else:
    print("ABLATION_CHECK_OK " + a[a.index("--select") + 1])
"""
