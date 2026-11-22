"""Check that every number the README states about the results comes from ``report/data.json`` or a shipped result file.

    python tools/check_readme_numbers.py [README.md]

The sections "The result", "Main results" and "Limits" are checked; code blocks and link targets are left out. ``CLAIMS`` names, for every number
of those sections, the key of ``report/data.json`` (``k``: the pre-formatted strings the report text itself uses) or the result file that holds it.
The script (a) finds every number of the README in ``CLAIMS``, (b) checks that the named value really contains the number, with its sign when the
README gives one (a README number without a sign is matched by its magnitude), and (c) prints the table. A number without an entry, or an entry
whose value does not contain the number, stops the check (exit 1). There is no free search: a coincidence cannot pass.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = ("The result", "Main results", "Limits")
NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?![\w])")

W1 = "results/wave1"
CLAIMS: dict[str, list] = {
    # The result
    "5.77": ["N6.gb"], "1.6": ["N6.loss"], "0.4": ["c.N6.Q5.abs"], "5.99": ["Q5.gb"], "-2.5": ["c.N6.Q5.lo"], "+1.7": ["c.N6.Q5.hi"],
    "-0.1": ["F1.diff"], "-0.9": ["F2.diff"], "2.5": ["c.N6.Q4.abs"], "2.0": ["N6.aime_share_of_gap"], "0.8": ["g.eng.abs"],
    "-2.4": ["g.eng.lo", "G1.diff"], "+0.9": ["g.eng.hi"], "0.0290": ["G1.kl"], "0.0243": ["N5.kl", "ab.experts.kl"],
    "4": [("text", "4-bit: the name of the format")], "36": [("file", f"{W1}/analysis/probe/activation_probe.json", r'"n_layers": 36')],
    "3": [("text", "Nemotron 3 Super: a product name"), "probe.mixers_gt448"],
    "4500": [("text", "RTX PRO 4500 Blackwell: the name of the card")],
    "95": [("file", "results/wave2-analysis.json", r'"confidence": 0.95')],
    "13.58": ["B0.gb"], "5.69": ["Q4.gb"],
    # Main results
    "-1.5": ["F1.lo"], "+1.3": ["F1.hi"], "-2.7": ["F2.lo"], "+0.8": ["F2.hi"], "8.38": ["F2.gb"],
    "6.04": ["N2.gb"], "-3.3": ["N2.diff", "N6.lo"], "-5.3": ["N2.lo"], "-1.4": ["N2.hi"],
    "32": [("file", f"{W1}/ablation-abl-experts/summary.json", r'"quantized_layers": 32')], "6.63": ["N5.gb"],
    "-1.2": ["N5.diff", "s.noaime.N6.Q4.ci"], "-3.0": ["N5.lo"], "+0.6": ["N5.hi", "s.noaime.N6.Q4.diff"],
    "-4.2": ["G1.lo"], "-0.7": ["G1.hi"], "6.8": ["N6.bpw"], "-1.6": ["N6.diff"], "0.0": ["N6.hi"],
    "-0.4": ["c.N6.Q5.diff"], "0.22": ["N6.gb_under_Q5"], "-4.1": ["Q4.diff"], "-6.2": ["Q4.lo"], "-2.1": ["Q4.hi"],
    "+2.5": ["c.N6.Q4.diff", "s.noaime.N6.Q4.ci"], "+0.2": ["c.N6.Q4.lo", "g.noaime.eng.diff"], "+4.8": ["c.N6.Q4.hi"],
    "90": ["tr.Q4.n"], "35": ["tr.N6.n"], "72.7": ["g.macro.B0g"], "73.5": ["g.macro.B0"], "-0.8": ["g.eng.diff"],
    "-1.3": ["g.noaime.eng.ci"], "+1.8": ["g.noaime.eng.ci"], "6": ["ab.nonexperts_share_total"], "0.0498": ["ab.non_experts.kl"],
    "71": ["ab.experts_share_total"],
    "16": [("file", "results/bench-summary.md", r"short c=16 "), "probe.max_val"], "128": [("file", "results/bench-summary.md", r"short c=128 ")],
    "1.21": ["bench.N6.load_min"], "1.42": ["bench.N6.load_max"], "1.15": ["bench.F1.load_min"], "1.24": ["bench.F1.load_max"],
    "1.13": ["bench.F2.load_min"], "1.23": ["bench.F2.load_max"],
    "1": ["probe.top3_layers"], "7": ["probe.top3_layers"], "23": ["probe.top3_layers"], "0": ["probe.mixers_gt448"], "2": ["probe.mixers_gt448"],
    "3.6": ["probe.max_val"], "10": ["probe.max_val"], "80": ["probe.n_prompts"],
    # the closing line of Main results
    "-0.6": ["cal.diff"], "-2.8": ["cal.ci"], "+1.6": ["cal.ci"],
}


def sections(text: str) -> str:
    """The text of the checked sections, without fenced code and without link targets."""
    out, keep, fenced = [], False, False
    for line in text.splitlines():
        if line.startswith("## "):
            keep = line[3:].strip() in SECTIONS
            continue
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if keep and not fenced:
            out.append(re.sub(r"\]\([^)]*\)", "]", line))
    return "\n".join(out)


def plain(s: str) -> str:
    """A ``data.json`` string in plain text: ``$-$`` is a minus, ``{,}`` a comma, other TeX markup is dropped."""
    return s.replace("$-$", "-").replace("{,}", ",").replace("\\times", " x ").replace("$", "").replace("{", "").replace("}", "").replace("^", " ")


def contains(value: str, token: str) -> bool:
    """True if ``value`` holds the number ``token`` (with its sign when it has one, else by magnitude)."""
    toks = {t.replace(",", "") for t in NUM.findall(plain(value))}
    return token in toks if token[0] in "+-" else token in {t.lstrip("+-") for t in toks}


def main(argv: list[str]) -> int:
    """Check the README and print the table."""
    readme = ROOT / (argv[0] if argv else "README.md")
    k = json.loads((ROOT / "report" / "data.json").read_text())["k"]
    bad, seen = 0, []
    print(f"{'number':>9}  source  value")
    for raw in NUM.findall(sections(readme.read_text())):
        tok = raw.replace(",", "")
        if tok in seen:
            continue
        seen.append(tok)
        entries = CLAIMS.get(tok)
        if not entries:
            print(f"{raw:>9}  NO ENTRY IN CLAIMS")
            bad += 1
            continue
        for e in entries:
            if isinstance(e, str):
                ok = e in k and contains(k[e], tok)
                print(f"{raw:>9}  k[{e}] = {k.get(e)!r}  {'ok' if ok else 'DOES NOT CONTAIN THE NUMBER'}")
            elif e[0] == "file":
                ok = bool(re.search(e[2], (ROOT / e[1]).read_text()))
                print(f"{raw:>9}  file {e[1]} has /{e[2]}/  {'ok' if ok else 'NOT FOUND'}")
            else:
                ok = True
                print(f"{raw:>9}  {e[1]}  (not a result)")
            bad += not ok
    print(f"{len(seen)} distinct numbers checked, {bad} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
