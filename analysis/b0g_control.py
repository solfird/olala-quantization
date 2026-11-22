r"""The same-engine control: arm B0g, OVH's bf16 GGUF on llama.cpp (PROTOCOL.md amendment 2026-11-22).

Run from the repository root (it uses the analysis code of ``code/evals``; no GPU, no network, numpy only):

    uv run --python 3.12 --with numpy==2.5.5 python analysis/b0g_control.py

Writes ``results/b0g-control.json`` and ``results/b0g-control.md``; ``report/build_data.py`` reads the JSON (keys ``g.*``).
Its input ``results/truncation-audit.json`` carries the B0g row that this script reads (``analysis/truncation_audit.py`` makes it from the held-back raw logs).

Why: every llama.cpp row of the report (OVH's Q5_K_M, the Q4_K_M stand-in) is compared with vLLM rows, so the file format and the serving engine are
confounded. B0g is OVH's own bf16 GGUF on the same llama.cpp commit, harness, tasks and settings as the Q5 arm. Against vLLM bf16 (B0) it measures the
engine; against Q5_K_M and Q4 it measures the file format on one engine. Descriptive, outside the confirmatory family; a single run, so provisional.

Every interval uses the report's bootstrap through ``analysis/review_sensitivity.py`` (itself a thin layer on ``evals/analyze_wave.py``: ``build_arm``,
``compare_arms``, ``resample_indices``, ``interval``): items resampled within each task on a per-task random stream, the five samples of an AIME problem
moving together (an item is the mean over its epochs), the equally weighted macro of the five tasks, 10,000 resamples, seed 0. The run first recomputes
every interval already stored in ``results/wave2-*.json`` for the arms it loads and refuses to go on if one differs; it refuses too if B0g is not the control
that the pre-flight note of the run described (hash, commit, context, item sets, request errors).

* ``absolute``: accuracy of each arm, macro and per task, with its own interval.
* ``engine``: B0g minus the pooled vLLM bf16 reference B0, and minus run a and run b separately.
* ``format``: Q5_K_M minus B0g, Q4_K_M stand-in minus B0g, and Q4 minus Q5 (already in the report; recomputed).
* ``cross``: the new NVFP4 build N6 and the two FP8 builds F1, F2 minus B0g. These cross engines (llama.cpp reference, vLLM builds).
* ``vllm``: Q5, Q4 and N6 minus B0, the report's own contrasts. ``double``: (X minus B0g) minus (X minus B0) for X = Q5 and Q4, bootstrapped on the
  item-level difference. It is B0 minus B0g by algebra (X cancels item by item), so it is the engine effect with its sign reversed and is the same for
  Q5 and Q4; it is computed in full rather than substituted, and the identity is checked. ``decomposition``: the loss of X against B0 as the sum of its
  format part (X minus B0g) and the engine part (B0g minus B0), point values.
* ``noaime``: the contrasts of ``engine`` and ``format`` with the macro of the four tasks without AIME.
* ``margin``: the macro interval against margins of -2, -3 and -5 by the unadjusted lower end, as ``review-sensitivity.json``: B0g, Q5, Q4 and N6 against
  B0g (``vs_B0g``; B0g itself against B0), and B0g, Q5, Q4 and N6 against B0 (``vs_B0``).
* ``generous`` and ``truncation``: the audit row of B0g, the cut-off counts per task, and the generous rescoring (a cut-off answer whose correct number
  appears in the last quarter of the reasoning counts as right) of B0g, B0, Q5, Q4 and N6 with the paired intervals.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for p in (str(ROOT / "code" / "evals"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import review_sensitivity as rs  # noqa: E402

aw = rs.aw
TASKS, AIME = rs.TASKS, rs.AIME
N_BOOT, SEED = rs.N_BOOT, rs.SEED
B0G_DIRS = ["B0g-mgsm-he", "B0g-ifeval-mbpp", "B0g-aime"]
"""Result folders (under ``results/wave1``) of the three B0g pods, one task group each."""
FOUR = tuple(t for t in TASKS if t != AIME)
B0G_AUDIT = "bf16 GGUF (llama.cpp)"
AUDIT_NAME = {**rs.AUDIT_NAME, "B0a": "bf16 (run a)", "B0b": "bf16 (run b)", "B0g": B0G_AUDIT}
"""Names of the arms in ``results/truncation-audit.json``."""
ABS_ARMS = ("B0", "B0a", "B0b", "B0g", "Q5", "Q4", "N6", "F1", "F2")
ENGINE = (("B0g", "B0"), ("B0g", "B0a"), ("B0g", "B0b"))
FORMAT = (("Q5", "B0g"), ("Q4", "B0g"), ("Q4", "Q5"))
CROSS = (("N6", "B0g"), ("F1", "B0g"), ("F2", "B0g"))
VLLM = (("Q5", "B0"), ("Q4", "B0"), ("N6", "B0"))
DOUBLE = ("Q5", "Q4")
MARGIN_VS_B0G = (("B0g", "B0"), ("Q5", "B0g"), ("Q4", "B0g"), ("N6", "B0g"))
MARGIN_VS_B0 = (("B0g", "B0"), ("Q5", "B0"), ("Q4", "B0"), ("N6", "B0"))
GEN_ARMS = ("B0a", "B0b", "B0g", "Q5", "Q4", "N6")
GEN_PAIRS = (("B0g", "B0"), ("Q5", "B0g"), ("Q4", "B0g"), ("N6", "B0g"), ("Q5", "B0"), ("Q4", "B0"))
MIN_REPRODUCED = 3
SLOT_CTX = "32768"


def key(x: str, y: str) -> str:
    """``X-Y``: the name of the contrast ``x - y``."""
    return f"{x}-{y}"


# ------------------------------------------------------------------------------------------ loading and checks


def read_provenance(path: Path) -> dict[str, str]:
    """``key=value`` lines of a ``server_provenance.txt``."""
    out = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def load_b0g(wave: Path) -> list[aw.Run]:
    """The three B0g runs, one per pod; raise if one cannot be read."""
    runs = [aw.load_run_dir(wave / d) for d in B0G_DIRS]
    for r in runs:
        if r.problems:
            raise SystemExit(f"{r.path}: {r.problems}")
    return runs


def check_b0g(runs: list[aw.Run], ref: aw.Arm, pinned_sha: str, q5_commit: str, provenance: list[dict[str, str]]) -> dict:
    """Refuse (SystemExit) unless the runs are the control that was planned (hash, commit, context, item sets); return what was checked.

    * every run complete, no request error (in ``run_info.json`` and in the rows), and the tasks of the three pods are the five tasks, each exactly once;
    * the item ids and the number of rows (epochs) of every task equal those of the vLLM bf16 reference ``ref``;
    * every provenance file: the bf16 GGUF arm, the hash the setup verified equals the pinned one and ``pinned_sha`` (gguf_files.json), the llama.cpp
      commit equals the one of the Q5 arm, a context of 32,768 tokens per slot.
    """
    seen: Counter = Counter()
    for r in runs:
        if r.info.get("status") != "complete":
            raise SystemExit(f"{r.path}: status {r.info.get('status')!r}, not complete")
        if r.info.get("request_errors") != 0 or any(row.get("error") for row in r.rows):
            raise SystemExit(f"{r.path}: request errors; the run is not valid (PROTOCOL.md, amendment B0g)")
        for t in {row["task"] for row in r.rows}:
            seen[t] += 1
    if sorted(seen) != sorted(TASKS) or set(seen.values()) != {1}:
        raise SystemExit(f"the B0g pods do not hold the five tasks exactly once: {dict(seen)}")
    arm = aw.build_arm("B0g", runs, TASKS)
    ref_rows = Counter(row["task"] for run in ref.runs[:1] for row in run.rows)
    got_rows = Counter(row["task"] for run in runs for row in run.rows)
    for t in TASKS:
        if set(arm.scores[t]) != set(ref.scores[t]):
            raise SystemExit(f"B0g and the bf16 reference do not hold the same {t} items")
        if got_rows[t] != ref_rows[t]:
            raise SystemExit(f"B0g has {got_rows[t]} {t} rows, the bf16 reference {ref_rows[t]}")
    commits = set()
    for prov in provenance:
        if prov.get("arm") != "gguf-bf16":
            raise SystemExit(f"a B0g pod served {prov.get('arm')!r}, not gguf-bf16")
        if not (prov.get("gguf_sha256_verified_by_setup") == prov.get("gguf_sha256_pinned") == pinned_sha):
            raise SystemExit("the GGUF hash of a B0g pod differs from gguf_files.json")
        if prov.get("slot_ctx") != SLOT_CTX:
            raise SystemExit(f"a B0g pod ran with a context of {prov.get('slot_ctx')} tokens per slot, not {SLOT_CTX}")
        commits.add(prov.get("llama_cpp_commit"))
    if commits != {q5_commit}:
        raise SystemExit(f"the llama.cpp commit of B0g {commits} is not the one of the Q5 arm {q5_commit}")
    return {"sha256": pinned_sha, "llama_cpp_commit": q5_commit, "slot_ctx": int(SLOT_CTX), "samples": {t: int(got_rows[t]) for t in TASKS},
            "truncations": int(rs.truncated_rows(runs))}


def build_arms_with_b0g(wave: Path) -> tuple[dict[str, list[aw.Run]], dict[str, aw.Arm], dict]:
    """The runs and arms of ``review_sensitivity`` plus B0g (the three pods pooled per task); the checks of ``check_b0g`` are made."""
    runs = rs.load_runs(wave)
    arms = rs.build_arms(runs)
    pinned = json.loads((ROOT / "code" / "runpod" / "pod" / "gguf_files.json").read_text())["gguf"]["files"]["bf16"]["sha256"]
    q5 = read_provenance(wave / rs.RUN_DIRS["Q5"][0] / "server_provenance.txt")["llama_cpp_commit"]
    provenance = [read_provenance(wave / d / "server_provenance.txt") for d in B0G_DIRS]
    runs["B0g"] = load_b0g(wave)
    checked = check_b0g(runs["B0g"], arms["B0"], pinned, q5, provenance)
    arms["B0g"] = aw.build_arm("B0g", runs["B0g"], TASKS)
    return runs, arms, checked


# ------------------------------------------------------------------------------------------ the contrasts


def contrast(arms: dict[str, aw.Arm], x: str, y: str, tasks: tuple[str, ...] = TASKS, n_boot: int = N_BOOT) -> dict:
    """``x - y``: the report's ``compare_arms``, trimmed to ``{macro, tasks}`` of ``{value, lo, hi}`` entries."""
    c = rs.contrast(arms[x], arms[y], tasks, n_boot, SEED)
    return {"macro": c["macro"], "tasks": c["tasks"]}


def contrast_set(arms: dict[str, aw.Arm], pairs: tuple, tasks: tuple[str, ...] = TASKS, n_boot: int = N_BOOT) -> dict[str, dict]:
    """``{"X-Y": contrast}`` for each ``(x, y)`` of ``pairs``."""
    return {key(x, y): contrast(arms, x, y, tasks, n_boot) for x, y in pairs}


def double_difference(arms: dict[str, aw.Arm], a: str, b: str, c: str, d: str, tasks: tuple[str, ...] = TASKS, n_boot: int = N_BOOT) -> dict:
    """``(a - b) - (c - d)`` bootstrapped on the item-level difference, the resamples being those of ``compare_arms``.

    Per task the unit is an item present in all four arms; the vector is ``100 * ((a - b) - (c - d))`` item by item, the resample indices are
    ``aw.resample_indices(seed, task, n, n_boot)`` as in ``compare_arms``, and the macro is the equally weighted mean of the task draws.
    """
    draws, point, out_tasks = [], [], {}
    for t in tasks:
        ids = sorted(set.intersection(*(set(arms[n].scores[t]) for n in (a, b, c, d))))
        s = {n: np.array([arms[n].scores[t][i] for i in ids]) for n in (a, b, c, d)}
        v = 100.0 * ((s[a] - s[b]) - (s[c] - s[d]))
        dr = v[aw.resample_indices(SEED, t, len(ids), n_boot)].mean(axis=1)
        lo, hi = aw.interval(dr)
        out_tasks[t] = {"value": float(v.mean()), "lo": lo, "hi": hi}
        draws.append(dr)
        point.append(v.mean())
    lo, hi = aw.interval(np.mean(draws, axis=0))
    return {"macro": {"value": float(np.mean(point)), "lo": lo, "hi": hi}, "tasks": out_tasks}


def absolute(arms: dict[str, aw.Arm], names: tuple[str, ...] = ABS_ARMS, n_boot: int = N_BOOT) -> dict[str, dict]:
    """Accuracy in percent of each arm, macro and per task, each with its own bootstrap interval (from ``compare_arms`` against B0)."""
    out = {}
    for a in names:
        x, y = (arms["B0g"], arms["B0"]) if a == "B0" else (arms[a], arms["B0"])
        s, _ = aw.compare_arms(x, y, TASKS, n_boot, SEED)
        side = "y" if a == "B0" else "x"   # B0 is the second arm of its own comparison, every other arm the first
        out[a] = {"macro": s["macro"][side], "tasks": {t: s["tasks"][t][side] for t in TASKS}}
    return out


def decomposition(vllm: dict, fmt: dict, engine: dict, x: str) -> dict:
    """The loss of ``x`` against vLLM bf16 as format part plus engine part (point values; the identity is exact for item means).

    ``total`` = x minus B0, ``format`` = x minus B0g, ``engine`` = B0g minus B0; ``engine_share`` is engine over total in percent (None when the total is zero).
    """
    t, f, e = vllm[key(x, "B0")], fmt[key(x, "B0g")], engine[key("B0g", "B0")]
    total, fp, ep = t["macro"]["value"], f["macro"]["value"], e["macro"]["value"]
    if abs(total - (fp + ep)) > 1e-9:
        raise SystemExit(f"{x}: the loss against B0 is not the format part plus the engine part ({total} against {fp} + {ep})")
    return {"total": total, "format": fp, "engine": ep, "engine_share": 100 * ep / total if total != 0 else None,
            "tasks": {tk: {"total": t["tasks"][tk]["value"], "format": f["tasks"][tk]["value"], "engine": e["tasks"][tk]["value"]} for tk in TASKS}}


def margin_set(arms: dict[str, aw.Arm], pairs: tuple, n_boot: int = N_BOOT) -> dict[str, dict]:
    """The margin table of ``review_sensitivity`` (``margin_row``) for each ``(x, y)``, keyed by ``x``."""
    return {x: rs.margin_row(rs.contrast(arms[x], arms[y], TASKS, n_boot, SEED)["summary"]) for x, y in pairs}


# ------------------------------------------------------------------------------------------ generous rescoring and truncation audit


def generous_arms(runs: dict[str, list[aw.Run]], audit: dict, names: tuple[str, ...] = GEN_ARMS) -> tuple[dict[str, aw.Arm], dict[str, int]]:
    """Arms rescored by the generous rule, with the number of samples credited; B0 is the two credited bf16 runs pooled.

    Refuses if the truncated rows of an arm's runs do not match the audit (the audit and the samples must be the same data).
    """
    gruns, credited = {}, {}
    for a in names:
        nm = AUDIT_NAME[a]
        if rs.truncated_rows(runs[a]) != audit["summary"][nm]["truncated"]:
            raise SystemExit(f"{a}: the truncated rows of samples.jsonl do not match the audit ({nm})")
        credit = rs.late_counts(audit["samples"][nm])
        gruns[a] = rs.credit_runs(runs[a], credit)
        credited[a] = int(sum(credit.values()))
    arms = {a: aw.build_arm(a + "-generous", r, TASKS) for a, r in gruns.items()}
    if "B0a" in gruns and "B0b" in gruns:
        arms["B0"] = aw.build_arm("B0-generous", gruns["B0a"] + gruns["B0b"], TASKS)
    return arms, credited


def macro_value(arm: aw.Arm, tasks: tuple[str, ...] = TASKS) -> float:
    """Equally weighted macro of the task accuracies of an arm, in percent (the point value of ``compare_arms``)."""
    return float(np.mean([100.0 * np.mean(list(arm.scores[t].values())) for t in tasks]))


def generous_block(arms: dict[str, aw.Arm], garms: dict[str, aw.Arm], credited: dict[str, int], pairs: tuple = GEN_PAIRS, n_boot: int = N_BOOT) -> dict:
    """Strict and generous macro of each arm, and the strict and generous contrast of each pair."""
    macro = {a: {"strict": macro_value(arms[a]), "generous": macro_value(garms[a]), "credited": credited.get(a)} for a in garms}
    return {"macro": macro, "contrasts": {key(x, y): {"strict": contrast(arms, x, y, TASKS, n_boot)["macro"], "generous": contrast(garms, x, y, TASKS, n_boot)["macro"]}
                                          for x, y in pairs}}


def truncation_block(audit: dict, name: str = B0G_AUDIT) -> dict:
    """The audit row of an arm and its cut-off answers per task (task names as in ``samples.jsonl``)."""
    row = audit["summary"][name]
    per = Counter(rs.norm_task(m["task"]) for m in audit["samples"][name])
    if sum(per.values()) != row["truncated"]:
        raise SystemExit(f"{name}: the per-task cut-off counts do not add up to the audit row")
    return {"audit_name": name, "row": row, "cut_by_task": {t: int(per.get(t, 0)) for t in TASKS}, "cut_total": int(row["truncated"])}


# ------------------------------------------------------------------------------------------ rendering


def fmt(c: dict, nd: int = 1) -> str:
    """``+0.6 (-1.2 to +2.5)``."""
    return rs.fmt(c, nd)


def contrast_table(title: str, cs: dict[str, dict], note: str = "") -> list[str]:
    """A Markdown table of contrasts: one row per contrast, macro first and then the five tasks."""
    tasks = [t for t in TASKS if t in next(iter(cs.values()))["tasks"]]
    L = [f"## {title}", ""] + ([note, ""] if note else [])
    L += ["| Contrast | Macro | " + " | ".join(tasks) + " |", "|---|---|" + "---|" * len(tasks)]
    for k, c in cs.items():
        L.append(f"| {k} | {fmt(c['macro'])} | " + " | ".join(fmt(c["tasks"][t]) for t in tasks) + " |")
    return L + [""]


def render_md(res: dict) -> str:
    """The Markdown report of the results."""
    m = res["meta"]
    L = ["# B0g: OVH's bf16 GGUF on llama.cpp, the same-engine control", "",
         f"Generated by `analysis/b0g_control.py`. Item-level paired bootstrap of `evals/analyze_wave.py` ({m['n_boot']:,} resamples, seed {m['seed']}); "
         f"{m['reproduced']} stored intervals of `results/wave2-*.json` reproduced exactly. Differences are in points of the five-task macro unless a row says otherwise. "
         "B0 is the pooled vLLM bf16 reference (two runs); B0g is one run of OVH's bf16 GGUF on the llama.cpp commit of the Q5 arm "
         f"(sha256 {m['b0g']['sha256'][:8]}..., {m['b0g']['truncations']} truncated answers, 0 request errors).", ""]
    L += ["## Accuracy (percent)", "", "| Arm | Macro | " + " | ".join(TASKS) + " |", "|---|---|" + "---|" * len(TASKS)]
    for a, v in res["absolute"].items():
        L.append(f"| {a} | {v['macro']['value']:.1f} ({v['macro']['lo']:.1f} to {v['macro']['hi']:.1f}) | " + " | ".join(f"{v['tasks'][t]['value']:.1f}" for t in TASKS) + " |")
    L.append("")
    L += contrast_table("Engine effect (llama.cpp bf16 minus vLLM bf16)", res["engine"])
    L += contrast_table("Format effect on one engine (llama.cpp)", res["format"])
    L += contrast_table("Cross-engine: the NVFP4 and FP8 builds (vLLM) minus B0g (llama.cpp)", res["cross"], "These contrasts cross serving engines.")
    L += contrast_table("The report's contrasts against the vLLM reference", res["vllm"])
    L += contrast_table("Double difference (X minus B0g) minus (X minus B0)", res["double"],
                        "Item by item X cancels, so this is B0 minus B0g, the engine effect with its sign reversed, whatever X is. "
                        f"Identity checked: {res['double_identity_checked']}.")
    L += ["## The loss against vLLM bf16, split into format and engine (macro, points)", "", "| Build | Against B0 | Format part (X minus B0g) | Engine part (B0g minus B0) | Engine share of the loss (%) |", "|---|---|---|---|---|"]
    for x, d in res["decomposition"].items():
        share = "n/a" if d["engine_share"] is None else f"{d['engine_share']:.0f}"
        L.append(f"| {x} | {d['total']:+.2f} | {d['format']:+.2f} | {d['engine']:+.2f} | {share} |")
    L += ["", "## Leave AIME out (macro of four tasks)", "", "| Contrast | Without AIME | With all five tasks |", "|---|---|---|"]
    for k, v in res["noaime"].items():
        L.append(f"| {k} | {fmt(v['without'])} | {fmt(v['with'])} |")
    L += ["", "## Margins (lower end of the macro interval; unadjusted)", ""]
    for which, title in (("vs_B0g", "Against B0g (B0g itself against B0)"), ("vs_B0", "Against vLLM bf16 (B0)")):
        L += [f"### {title}", "", "| Build | Macro | Lower end passes -2 | -3 | -5 | Task wholly below -5 |", "|---|---|---|---|---|---|"]
        for a, v in res["margin"][which].items():
            flag = "; ".join(f"{f['task']} {fmt(f)}" for f in v["task_flag"]) or "none"
            L.append(f"| {a} | {fmt(v['macro'])} (lower end {v['macro']['lo']:.3f}) | " + " | ".join(v["margins"][g]["state"] for g in ("-2", "-3", "-5")) + f" | {flag} |")
        L.append("")
    tr = res["truncation"]
    r = tr["row"]
    L += ["## Truncated answers of B0g", "",
          f"{r['truncated']} of {r['samples']} answers were cut off at the context limit: {r['long']} not flagged as a loop, {r['loops']} loops, "
          f"{r['target_late']} with the right number in the last quarter of the reasoning, {r['with_text']} with visible answer text. "
          "Per task: " + ", ".join(f"{t} {n}" for t, n in tr["cut_by_task"].items()) + ".", "",
          "## Generous rule (cut-off answers credited when the right number is in the last quarter of the reasoning)", "",
          "| Arm | Strict macro | Generous macro | Samples credited |", "|---|---|---|---|"]
    for a, v in res["generous"]["macro"].items():
        L.append(f"| {a} | {v['strict']:.1f} | {v['generous']:.1f} | {'-' if v['credited'] is None else v['credited']} |")
    L += ["", "| Contrast | Strict | Generous |", "|---|---|---|"]
    for k, v in res["generous"]["contrasts"].items():
        L.append(f"| {k} | {fmt(v['strict'])} | {fmt(v['generous'])} |")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------ main


def analyse(arms: dict[str, aw.Arm], garms: dict[str, aw.Arm], credited: dict[str, int], audit: dict, n_boot: int = N_BOOT) -> dict:
    """Every table of the module from the arms (``B0``, ``B0a``, ``B0b``, ``B0g``, ``Q5``, ``Q4``, ``N6``, ``F1``, ``F2``) and the generous arms."""
    res: dict = {"absolute": absolute(arms, ABS_ARMS, n_boot)}
    res["engine"] = contrast_set(arms, ENGINE, TASKS, n_boot)
    res["format"] = contrast_set(arms, FORMAT, TASKS, n_boot)
    res["cross"] = contrast_set(arms, CROSS, TASKS, n_boot)
    res["vllm"] = contrast_set(arms, VLLM, TASKS, n_boot)
    res["double"] = {x: double_difference(arms, x, "B0g", x, "B0", TASKS, n_boot) for x in DOUBLE}
    neg_engine = res["engine"][key("B0g", "B0")]
    res["double_identity_checked"] = True
    for x in DOUBLE:
        for tk in ("macro", *TASKS):
            d = res["double"][x]["macro" if tk == "macro" else "tasks"]
            d = d if tk == "macro" else d[tk]
            e = neg_engine["macro"] if tk == "macro" else neg_engine["tasks"][tk]
            # the item-level vector is -(engine vector): same value, the interval ends swapped and negated
            if not (abs(d["value"] + e["value"]) < 1e-9 and abs(d["lo"] + e["hi"]) < 1e-9 and abs(d["hi"] + e["lo"]) < 1e-9):
                raise SystemExit(f"the double difference of {x} is not minus the engine effect ({tk}): {d} against {e}")
    res["decomposition"] = {}
    for x in ("Q5", "Q4", "N6"):
        res["decomposition"][x] = decomposition(res["vllm"], {**res["format"], **res["cross"]}, res["engine"], x)
    res["noaime"] = {}
    for group, pairs in (("engine", ENGINE), ("format", FORMAT)):
        for x, y in pairs:
            res["noaime"][key(x, y)] = {"group": group, "without": contrast(arms, x, y, FOUR, n_boot)["macro"], "with": contrast(arms, x, y, TASKS, n_boot)["macro"],
                                        "tasks_without": contrast(arms, x, y, FOUR, n_boot)["tasks"]}
    res["margin"] = {"vs_B0g": margin_set(arms, MARGIN_VS_B0G, n_boot), "vs_B0": margin_set(arms, MARGIN_VS_B0, n_boot)}
    res["generous"] = generous_block(arms, garms, credited, GEN_PAIRS, n_boot)
    res["truncation"] = truncation_block(audit)
    return res


def check_reproduction(arms: dict[str, aw.Arm], results: Path) -> int:
    """``review_sensitivity.check_reproduction`` (every stored interval whose two arms are loaded), and at least ``MIN_REPRODUCED`` of them."""
    n = rs.check_reproduction(arms, results)
    if n < MIN_REPRODUCED:
        raise SystemExit(f"only {n} stored intervals could be recomputed, at least {MIN_REPRODUCED} are required")
    return n


def main() -> int:
    """Check the data, run every analysis and write the JSON and Markdown files."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=Path, default=ROOT / "results" / "wave1")
    ap.add_argument("--audit", type=Path, default=ROOT / "results" / "truncation-audit.json")
    ap.add_argument("--json", type=Path, default=ROOT / "results" / "b0g-control.json")
    ap.add_argument("--md", type=Path, default=ROOT / "results" / "b0g-control.md")
    args = ap.parse_args()

    runs, arms, checked = build_arms_with_b0g(args.wave)
    reproduced = check_reproduction(arms, args.wave.parent)   # abort before writing anything if the bootstrap is not the report's
    audit = json.loads(args.audit.read_text())
    if B0G_AUDIT not in audit["summary"]:
        raise SystemExit("results/truncation-audit.json has no B0g row: run analysis/truncation_audit.py first")
    garms, credited = generous_arms(runs, audit)
    res = {"meta": {"n_boot": N_BOOT, "seed": SEED, "tasks": list(TASKS), "run_dirs": {**rs.RUN_DIRS, "B0g": B0G_DIRS}, "reproduced": reproduced,
                    "b0g": checked, "script": "analysis/b0g_control.py"}}
    res.update(analyse(arms, garms, credited, audit))
    args.json.write_text(json.dumps(res, indent=1) + "\n")
    args.md.write_text(render_md(res))
    print(render_md(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
