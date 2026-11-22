r"""Sensitivity analyses added after an outside review of the report (the review itself is not part of this repository).

Run from the repository root (it uses the analysis code of ``code/evals``; no GPU, no network, numpy only):

    uv run --python 3.12 --with numpy==2.5.5 python analysis/review_sensitivity.py

Writes ``results/review-sensitivity.json`` and ``results/review-sensitivity.md``; ``report/build_data.py`` reads the JSON.

Every interval uses the report's own bootstrap, imported from ``evals/analyze_wave.py`` (``build_arm``, ``compare_arms``, ``resample_indices``,
``interval``): items resampled within each task on a per-task random stream, the five samples of an AIME problem moving together (an item is the
mean over its epochs), the equally weighted macro of the tasks, 10,000 resamples, seed 0. Each table below says which function made it. The run
checks first that this code reproduces every interval already stored in ``results/wave2-*.json`` and refuses to go on if one differs.

* ``noaime``: macro of the four tasks without AIME, for the new build (N6), the Q4_K_M stand-in (Q4), OVH's Q5_K_M (Q5) and bf16 (B0).
* ``generous``: the rescoring of ``analysis/truncation_audit.py`` (an answer cut off whose correct number appears in the last quarter of the
  reasoning counts as right), with the paired intervals of N6, Q4 and Q5.
* ``kl``: per-chunk top-k KL of the three candidates of the smaller-build step, from the stored ``.npz`` files of ``results/wave1/fidelity-fid``;
  paired bootstrap over the 64 chunks (``stats.paired_bootstrap_diff_ci``); the three means at k = 5, 10 and 20.
* ``bf16``: the two bf16 runs against each other, per task; ``ref``: each build against run a alone, run b alone and the pooled reference.
* ``margin``: the macro-against-bf16 interval of each build against margins of -2, -3 and -5 points, and the task-level flag of PROTOCOL.md 7.
* ``cluster``: MGSM resampled by source problem (key: the answer and the numbers in the question; the same problem in several languages is one
  cluster) against the item bootstrap, for every build against bf16 and for the pairs of the report.

The cluster key needs the question text, which only the raw inspect logs hold (they are not part of this repository). So the repository ships
``results/mgsm-clusters.json``: the cluster id of every MGSM item id, nothing else (the ids are the ranks of the sorted keys, so the order of the
clusters, which the bootstrap uses, is the original one). This script reads that file when no log is found next to ``--wave``; with the logs it
derives the clusters itself, and ``--write-clusters FILE`` writes the file.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "evals"))

import analyze_wave as aw  # noqa: E402
import fidelity_math as fm  # noqa: E402
import stats  # noqa: E402

TASKS = ("mgsm", "humaneval_plus", "mbpp_plus", "ifeval", "aime2025")
AIME, MGSM = aw.AIME, aw.MGSM
N_BOOT, SEED = 10_000, 0
RUN_DIRS = {
    "B0a": ["B0a-v2"], "B0b": ["B0b-v2"], "F1": ["F1-v2"], "F2": ["F2-v2"], "N2": ["N2-v2"], "N5": ["N5-v2"],
    "G1": ["G1-v2"], "N6": ["SEL-v2"], "Q5": ["Q5v2"], "Q4": ["Q4v2"],
}
"""Result folders (under ``results/wave1``) of each run; the names of the report. B0, the pooled bf16 reference, is B0a and B0b together."""
BUILDS = ("F1", "F2", "N2", "N5", "G1", "N6", "Q5", "Q4")
AUDIT_NAME = {"N6": "NVFP4 C2 (this build)", "Q4": "Q4_K_M stand-in", "Q5": "OVH Q5_K_M"}
"""Names of the arms in ``results/truncation-audit.json``."""
NOAIME_PAIRS = (("N6", "Q4"), ("Q4", "Q5"), ("N6", "Q5"), ("N6", "B0"))
GENEROUS_PAIRS = (("N6", "Q4"), ("N6", "Q5"), ("Q4", "Q5"))
CLUSTER_PAIRS = (*((b, "B0") for b in BUILDS), ("N6", "Q5"), ("N6", "Q4"), ("N6", "N5"), ("Q4", "Q5"), ("N5", "N2"), ("G1", "N5"), ("N5", "Q5"))
MARGINS = (-2.0, -3.0, -5.0)
FID_ARMS = ("N5", "C1", "C2")
"""Experts only (control), 36 layers with plain scales, 36 layers with the scale search (the new build)."""
FID_CONTRASTS = (("C1", "N5", "plain_minus_exp"), ("C2", "C1", "search_minus_plain"), ("C2", "N5", "search_minus_exp"))
"""``(x, y, name)``: the difference is the mean over chunks of KL(x) - KL(y)."""
KS = (5, 10, 20)
K_MAIN = 20


def norm_task(name: str) -> str:
    """Task name of an inspect log (``inspect_evals/aime2025``, ``mbpp-plus``) in the form of ``samples.jsonl``."""
    return name.split("/")[-1].replace("-", "_")


# ------------------------------------------------------------------------------------------ accuracy: loading


def load_runs(wave: Path) -> dict[str, list[aw.Run]]:
    """Read every run folder; raise if one cannot be read."""
    out = {}
    for name, dirs in RUN_DIRS.items():
        runs = [aw.load_run_dir(wave / d) for d in dirs]
        for r in runs:
            if r.problems:
                raise SystemExit(f"{r.path}: {r.problems}")
        out[name] = runs
    return out


def build_arms(runs: dict[str, list[aw.Run]]) -> dict[str, aw.Arm]:
    """One arm per run, plus ``B0``: the two bf16 runs pooled by averaging item scores (PROTOCOL.md amendment 9)."""
    arms = {n: aw.build_arm(n, r, TASKS) for n, r in runs.items()}
    arms["B0"] = aw.build_arm("B0", runs["B0a"] + runs["B0b"], TASKS)
    return arms


def contrast(x: aw.Arm, y: aw.Arm, tasks: tuple[str, ...] = TASKS, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """``x - y`` per task and for the macro of ``tasks``: the report's ``compare_arms``, trimmed to ``{value, lo, hi}`` entries."""
    summary, _ = aw.compare_arms(x, y, tasks, n_boot, seed)
    return {"macro": summary["macro"]["diff"], "tasks": {t: summary["tasks"][t]["diff"] for t in tasks},
            "x_macro": summary["macro"]["x"], "y_macro": summary["macro"]["y"], "summary": summary}


def strip(c: dict) -> dict:
    """A contrast without the bulky ``summary`` entry."""
    return {k: v for k, v in c.items() if k != "summary"}


# ------------------------------------------------------------------------------------------ generous rescoring


def late_counts(audit_samples: list[dict]) -> Counter:
    """``{(task, item id): n}`` truncated samples whose correct number appears in the last quarter of the reasoning."""
    c: Counter = Counter()
    for m in audit_samples:
        if m["target_late"]:
            c[(norm_task(m["task"]), str(m["sample_id"]))] += 1
    return c


def credit_runs(runs: list[aw.Run], credit: Counter) -> list[aw.Run]:
    """Copies of ``runs`` in which, per item, ``credit[(task, id)]`` truncated rows score 1 instead of 0.

    A truncated row always scores 0 (``compare.effective_score``), so crediting any of an item's truncated rows gives the
    same item mean: the audit does not need to say which epoch of an AIME problem it was.
    """
    need = Counter(credit)
    new_runs = []
    for run in runs:
        rows = []
        for r in run.rows:
            key = (r["task"], str(r["sample_id"]))
            if r.get("truncated") and need.get(key, 0) > 0:
                need[key] -= 1
                r = {**r, "truncated": False, "error": None, "score": 1.0}
            rows.append(r)
        new_runs.append(dataclasses.replace(run, rows=rows))
    short = {k: v for k, v in need.items() if v > 0}
    if short:
        raise ValueError(f"the audit credits samples that are not truncated rows of these runs: {short}")
    return new_runs


def truncated_rows(runs: list[aw.Run]) -> int:
    """Number of truncated rows of ``runs``."""
    return sum(1 for run in runs for r in run.rows if r.get("truncated"))


# ------------------------------------------------------------------------------------------ MGSM clusters


def cluster_key(question: str, answer: str) -> str:
    """Source-problem key of an MGSM item: the answer and the numbers of the question (with repeats), sorted.

    The same problem translated into several languages has the same answer and the same numbers.
    """
    return "|".join([str(answer).strip(), *sorted(re.findall(r"\d+", question))])


def sample_cluster_key(sample_input: object, target: object) -> str:
    """``cluster_key`` of an inspect sample: ``input`` is a string or a list of chat messages, ``target`` a string or a list."""
    if not isinstance(sample_input, str):
        sample_input = " ".join(c for m in sample_input if isinstance(c := (m.get("content") if isinstance(m, dict) else getattr(m, "content", "")), str))
    return cluster_key(sample_input, target[0] if isinstance(target, list) else target)


def mgsm_clusters(eval_log: Path) -> dict[str, str]:
    """``{item id: source-problem key}`` from an inspect ``.eval`` log of the MGSM task (the question is only in the log).

    Needs ``inspect_ai`` (the log is zstd-compressed) and the raw log, which this repository does not hold.
    """
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(eval_log))
    return {str(s.id): sample_cluster_key(s.input, s.target) for s in log.samples or []}


def cluster_ids(clusters: dict[str, str]) -> dict:
    """``{"cluster_of_item": {item id: "cNNN"}}``: the keys replaced by their rank in sorted order, so that the order the bootstrap uses is kept.

    The file holds item ids and cluster ids only, no question text, no answer.
    """
    ranks = {k: f"c{j:03d}" for j, k in enumerate(sorted(set(clusters.values())))}
    return {"note": "MGSM source-problem clusters: the same problem in several languages shares a cluster id. Made by analysis/review_sensitivity.py "
                    "--write-clusters from the raw log (held back); ids are ranks of the sorted keys.",
            "n_clusters": len(ranks), "cluster_of_item": {i: ranks[k] for i, k in sorted(clusters.items())}}


def clustered_macro_ci(
    x: aw.Arm, y: aw.Arm, tasks: tuple[str, ...], clusters: dict[str, str], n_boot: int = N_BOOT, seed: int = SEED
) -> dict[str, float]:
    """Macro difference ``x - y`` as in ``compare_arms``, but MGSM is resampled by source problem.

    The other tasks use the very same indices as ``compare_arms``. For MGSM the unit is a cluster and the task mean of a resample is the
    sum over the drawn clusters of their summed differences, over the number of items drawn.
    """
    draws, point = [], []
    for t in tasks:
        ids = sorted(set(x.scores[t]) & set(y.scores[t]))
        d = 100.0 * (np.array([x.scores[t][i] for i in ids]) - np.array([y.scores[t][i] for i in ids]))
        point.append(d.mean())
        if t == MGSM:
            keys = sorted({clusters[i] for i in ids})
            pos = {k: j for j, k in enumerate(keys)}
            sums, sizes = np.zeros(len(keys)), np.zeros(len(keys))
            for i, v in zip(ids, d, strict=True):
                sums[pos[clusters[i]]] += v
                sizes[pos[clusters[i]]] += 1
            idx = aw.resample_indices(seed, t, len(keys), n_boot)
            draws.append(sums[idx].sum(axis=1) / sizes[idx].sum(axis=1))
        else:
            idx = aw.resample_indices(seed, t, len(ids), n_boot)
            draws.append(d[idx].mean(axis=1))
    lo, hi = aw.interval(np.mean(draws, axis=0))
    return {"value": float(np.mean(point)), "lo": lo, "hi": hi}


# ------------------------------------------------------------------------------------------ margins


def margin_states(c: dict, margin: float) -> dict:
    """Where a 95% interval lies against a margin: ``pass`` (lower end above it), ``worse`` (whole interval below it) or ``open``."""
    state = "pass" if c["lo"] > margin else "worse" if c["hi"] < margin else "open"
    return {"margin": margin, "lo_above": bool(c["lo"] > margin), "state": state}


def margin_row(summary: dict) -> dict:
    """The margin table row of one build from its ``compare_arms`` summary against bf16."""
    m = summary["macro"]["diff"]
    flagged = aw.task_level_regressions(summary)
    return {
        "macro": m,
        "margins": {f"{int(g)}": margin_states(m, g) for g in MARGINS},
        "task_flag": [{"task": t, **summary["tasks"][t]["diff"]} for t in flagged],
    }


# ------------------------------------------------------------------------------------------ fidelity (KL)


def load_topk(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(vocab, top_ids, top_lp)`` of a ``fidelity.py`` ``.npz`` file."""
    with np.load(path, allow_pickle=False) as z:
        return z["vocab"], z["top_ids"], z["top_lp"]


def per_chunk_kl(ref: tuple, var: tuple, k: int) -> np.ndarray:
    """Mean over the positions of each chunk of KL(ref || var) on the union of the two top-``k`` lists (the measure of ``fidelity.py``).

    Mirrors ``fidelity.compare_fidelity(max_k=k)``: token ids are mapped to one common vocabulary, both lists are cut to ``k``, then
    ``fidelity_math.kl_topk_union`` is applied. (``fidelity.py`` itself imports an HTTP client; this file does not need one.)
    """
    common = np.unique(np.concatenate([ref[0], var[0]]))

    def conv(d: tuple) -> tuple[np.ndarray, np.ndarray]:
        vocab, top_ids, top_lp = d
        mapping = np.searchsorted(common, vocab)
        ids = np.where(top_ids >= 0, mapping[np.maximum(top_ids, 0)], -1)[..., :k]
        lp = top_lp[..., :k]
        return ids.reshape(-1, ids.shape[-1]), lp.reshape(-1, ids.shape[-1]).astype(np.float64)

    r_ids, r_lp = conv(ref)
    v_ids, v_lp = conv(var)
    n_chunks, n_pos = ref[1].shape[:2]
    return fm.kl_topk_union(r_ids, r_lp, v_ids, v_lp).reshape(n_chunks, n_pos).mean(axis=1)


def kl_summary(per_chunk: dict[str, np.ndarray], n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Mean and chunk interval of each arm, and the paired differences of ``FID_CONTRASTS`` (all on the same chunk resamples).

    The difference is ``mean(KL(x) - KL(y))`` over chunks (``stats.paired_bootstrap_diff_ci(y, x)``); ``pct`` is that difference over the mean KL of ``y``.
    """
    means = {a: dict(zip(("value", "lo", "hi"), stats.bootstrap_mean_ci(v, n_boot, seed), strict=True)) for a, v in per_chunk.items()}
    diffs = {}
    for x, y, name in FID_CONTRASTS:
        d, lo, hi = stats.paired_bootstrap_diff_ci(per_chunk[y], per_chunk[x], n_boot, seed)
        base = float(np.mean(per_chunk[y]))
        diffs[name] = {"x": x, "y": y, "value": d, "lo": lo, "hi": hi, "pct": 100 * d / base, "pct_lo": 100 * lo / base, "pct_hi": 100 * hi / base}
    return {"means": means, "diffs": diffs}


def order_of(means: dict[str, float]) -> list[str]:
    """Arm names from the lowest to the highest KL."""
    return sorted(means, key=lambda a: means[a])


# ------------------------------------------------------------------------------------------ rendering


def fmt(c: dict, nd: int = 1) -> str:
    """``+0.6 (-1.2 to +2.5)``."""
    return f"{c['value']:+.{nd}f} ({c['lo']:+.{nd}f} to {c['hi']:+.{nd}f})"


def render_md(res: dict) -> str:
    """The Markdown report of the results."""
    L: list[str] = ["# Sensitivity analyses after the outside review", "",
                    f"Generated by `analysis/review_sensitivity.py`. Item-level paired bootstrap of `evals/analyze_wave.py` ({res['meta']['n_boot']:,} resamples, seed {res['meta']['seed']}); "
                    f"{res['meta']['reproduced']} stored intervals of `results/wave2-*.json` reproduced exactly. Differences are in points of the five-task macro "
                    "unless a row says otherwise.", ""]
    L += ["## Leave AIME out (macro of four tasks)", "", "| Contrast | Without AIME | With all five tasks |", "|---|---|---|"]
    for k_, v in res["noaime"].items():
        L.append(f"| {k_} | {fmt(v['without'])} | {fmt(v['with'])} |")
    g = res["generous"]
    L += ["", "## Generous rule (cut-off answers credited when the right number is in the last quarter of the reasoning)", "",
          "| Build | Strict macro | Generous macro | Samples credited |", "|---|---|---|---|"]
    for a, v in g["macro"].items():
        L.append(f"| {a} | {v['strict']:.1f} | {v['generous']:.1f} | {v['credited']} |")
    L += ["", "| Contrast | Strict | Generous |", "|---|---|---|"]
    for k_, v in g["contrasts"].items():
        L.append(f"| {k_} | {fmt(v['strict'])} | {fmt(v['generous'])} |")
    kl = res["kl"]
    L += ["", f"## Plain-text KL of the smaller-build candidates (top-{K_MAIN}, 64 chunks, chunk bootstrap)", "",
          "| Candidate | Mean KL | 95% interval |", "|---|---|---|"]
    for a, v in kl["by_k"][str(K_MAIN)]["means"].items():
        L.append(f"| {a} | {v['value']:.4f} | {v['lo']:.4f} to {v['hi']:.4f} |")
    L += ["", "| Paired difference | Mean | 95% interval | Percent of the second term |", "|---|---|---|---|"]
    for name, v in kl["by_k"][str(K_MAIN)]["diffs"].items():
        L.append(f"| {name} ({v['x']} - {v['y']}) | {v['value']:+.4f} | {v['lo']:+.4f} to {v['hi']:+.4f} | {v['pct']:+.1f} ({v['pct_lo']:+.1f} to {v['pct_hi']:+.1f}) |")
    L += ["", "| k | " + " | ".join(FID_ARMS) + " | Order, lowest first | Paired differences (" + ", ".join(n for _, _, n in FID_CONTRASTS) + ") |", "|---|---|---|---|---|"]
    for k_ in KS:
        e = kl["by_k"][str(k_)]
        L.append(f"| {k_} | " + " | ".join(f"{e['means'][a]['value']:.4f}" for a in FID_ARMS) + " | " + " < ".join(e["order"]) + " | "
                 + ", ".join(f"{e['diffs'][n]['value']:+.4f}" for _, _, n in FID_CONTRASTS) + " |")
    L += ["", f"The order of the candidates {'holds' if kl['order_holds'] else 'does not hold'} at k = 5, 10 and 20. "
          f"Mean KL of the experts-only build grows {kl['growth']['N5']['5_to_10']:.1f} percent from k = 5 to 10 and {kl['growth']['N5']['10_to_20']:.1f} percent from 10 to 20.", ""]
    b = res["bf16"]
    L += ["## The two bf16 runs (run a minus run b)", "", "| Task | Difference |", "|---|---|"]
    for t in TASKS:
        L.append(f"| {t} | {fmt(b['tasks'][t])} |")
    L.append(f"| macro | {fmt(b['macro'])} |")
    L += ["", "## Each build against bf16 run a alone, run b alone and both pooled (macro)", "", "| Build | Run a | Run b | Pooled |", "|---|---|---|---|"]
    for a, v in res["ref"].items():
        L.append(f"| {a} | {fmt(v['a'])} | {fmt(v['b'])} | {fmt(v['pooled'])} |")
    L += ["", "## Margins (lower end of the macro interval against bf16; unadjusted)", "",
          "| Build | Macro against bf16 | Lower end passes -2 | -3 | -5 | Task wholly below -5 |", "|---|---|---|---|---|---|"]
    for a, v in res["margin"].items():
        flag = "; ".join(f"{f['task']} {fmt(f)}" for f in v["task_flag"]) or "none"
        L.append(f"| {a} | {fmt(v['macro'])} (lower end {v['macro']['lo']:.3f}) | " + " | ".join(v["margins"][m]["state"] for m in ("-2", "-3", "-5")) + f" | {flag} |")
    c = res["cluster"]
    L += ["", "## MGSM resampled by source problem", "",
          f"{c['n_clusters']} clusters for {c['n_items']} items. Largest shift of any interval end, over {len(c['rows'])} contrasts: {c['max_shift']:.2f} points "
          f"({c['max_shift_where']}).", "", "| Contrast | Items resampled | By source problem |", "|---|---|---|"]
    for k_, v in c["rows"].items():
        L.append(f"| {k_} | {fmt(v['item'])} | {fmt(v['cluster'])} |")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------ main


def stored_macro_diffs(results: Path) -> dict[tuple[str, str], dict]:
    """Every macro difference stored in ``results/wave2-*.json``, by ``(x, y)`` with the report's names (SEL is N6)."""
    name = {"SEL": "N6"}
    out: dict[tuple[str, str], dict] = {}
    for f in sorted(results.glob("wave2-*.json")):
        d = json.loads(f.read_text())
        ref = d["reference"]
        for sect in ("candidates", "exploratory"):
            for a, v in d.get(sect, {}).items():
                out[(name.get(a, a), name.get(ref, ref))] = v["summary"]["macro"]["diff"]
        for c in d.get("contrasts", {}).values():
            out[(name.get(c["x"], c["x"]), name.get(c["y"], c["y"]))] = c["summary"]["macro"]["diff"]
        if d.get("calibration"):
            out[(d["calibration"]["x"], d["calibration"]["y"])] = d["calibration"]["summary"]["macro"]["diff"]
    return out


def check_reproduction(arms: dict[str, aw.Arm], results: Path) -> int:
    """Recompute every stored pair with ``compare_arms`` and require the same value and interval; return how many pairs matched."""
    n = 0
    for (x, y), stored in stored_macro_diffs(results).items():
        if x not in arms or y not in arms:
            continue
        mine = contrast(arms[x], arms[y])["macro"]
        for key in ("value", "lo", "hi"):
            if abs(mine[key] - stored[key]) > 1e-9:
                raise SystemExit(f"cannot reproduce the stored {x} - {y} ({key}): {mine[key]} against {stored[key]}")
        n += 1
    if n == 0:
        raise SystemExit("no stored interval could be checked")
    return n


def main() -> int:
    """Run every analysis and write the JSON and Markdown files."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=Path, default=ROOT / "results" / "wave1")
    ap.add_argument("--audit", type=Path, default=ROOT / "results" / "truncation-audit.json")
    ap.add_argument("--json", type=Path, default=ROOT / "results" / "review-sensitivity.json")
    ap.add_argument("--md", type=Path, default=ROOT / "results" / "review-sensitivity.md")
    ap.add_argument("--clusters", type=Path, default=ROOT / "results" / "mgsm-clusters.json",
                    help="cluster id of every MGSM item (used when the raw MGSM log is not next to --wave)")
    ap.add_argument("--write-clusters", type=Path, default=None, help="write the cluster-id file from the raw MGSM log (needs the log)")
    args = ap.parse_args()
    results = args.wave.parent

    runs = load_runs(args.wave)
    arms = build_arms(runs)
    reproduced = check_reproduction(arms, results)
    res: dict = {"meta": {"n_boot": N_BOOT, "seed": SEED, "tasks": list(TASKS), "run_dirs": RUN_DIRS, "reproduced": reproduced,
                          "script": "analysis/review_sensitivity.py"}}

    # (a) leave AIME out
    four = tuple(t for t in TASKS if t != AIME)
    res["noaime"] = {}
    for x, y in NOAIME_PAIRS:
        res["noaime"][f"{x}-{y}"] = {"without": contrast(arms[x], arms[y], four)["macro"], "with": contrast(arms[x], arms[y])["macro"]}

    # (b) generous rescoring
    audit = json.loads(args.audit.read_text())
    gruns, gmacro = {}, {}
    for a, nm in AUDIT_NAME.items():
        credit = late_counts(audit["samples"][nm])
        if truncated_rows(runs[a]) != audit["summary"][nm]["truncated"]:
            raise SystemExit(f"{a}: the truncated rows of samples.jsonl do not match the audit ({nm})")
        gr = credit_runs(runs[a], credit)
        gruns[a] = aw.build_arm(a + "-generous", gr, TASKS)
        strict = contrast(arms[a], arms["B0"])["x_macro"]["value"]
        gmacro[a] = {"strict": strict, "generous": contrast(gruns[a], arms["B0"])["x_macro"]["value"], "credited": int(sum(credit.values()))}
    res["generous"] = {"macro": gmacro, "contrasts": {f"{x}-{y}": {"strict": contrast(arms[x], arms[y])["macro"], "generous": contrast(gruns[x], gruns[y])["macro"]}
                                                   for x, y in GENEROUS_PAIRS}}

    # (c) KL
    fid = args.wave / "fidelity-fid"
    ref = load_topk(fid / "reference" / "b0_wikitext2.npz")
    var = {a: load_topk(fid / "arms" / a / f"fid-fid-{a}_wikitext2.npz") for a in FID_ARMS}
    by_k: dict[str, dict] = {}
    for k_ in KS:
        pc = {a: per_chunk_kl(ref, var[a], k_) for a in FID_ARMS}
        s = kl_summary(pc)
        s["order"] = order_of({a: v["value"] for a, v in s["means"].items()})
        by_k[str(k_)] = s
    stored = {a["name"]: a["fidelity"]["kld_mean"]["value"] for a in json.loads((fid / "summary.json").read_text())["arms"] if a.get("fidelity")}
    for a in FID_ARMS:
        if abs(by_k[str(K_MAIN)]["means"][a]["value"] - stored[a]) > 1e-9:
            raise SystemExit(f"cannot reproduce the stored mean KL of {a}: {by_k[str(K_MAIN)]['means'][a]['value']} against {stored[a]}")
    growth = {a: {"5_to_10": 100 * (by_k["10"]["means"][a]["value"] / by_k["5"]["means"][a]["value"] - 1),
                  "10_to_20": 100 * (by_k["20"]["means"][a]["value"] / by_k["10"]["means"][a]["value"] - 1)} for a in FID_ARMS}
    res["kl"] = {"by_k": by_k, "order_holds": len({tuple(by_k[str(k_)]["order"]) for k_ in KS}) == 1, "growth": growth,
                 "chunks": int(ref[1].shape[0]), "k_main": K_MAIN}

    # (d) bf16 against itself and builds against each run
    cal = contrast(arms["B0a"], arms["B0b"])
    res["bf16"] = {"tasks": cal["tasks"], "macro": cal["macro"], "max_abs_task": max(abs(v["value"]) for v in cal["tasks"].values())}
    res["ref"] = {a: {"a": contrast(arms[a], arms["B0a"])["macro"], "b": contrast(arms[a], arms["B0b"])["macro"], "pooled": contrast(arms[a], arms["B0"])["macro"]} for a in BUILDS}

    # (e) margins
    res["margin"] = {a: margin_row(contrast(arms[a], arms["B0"])["summary"]) for a in BUILDS}

    # (f) MGSM clusters
    logs = sorted((args.wave / RUN_DIRS["B0a"][0] / "logs").glob("*mgsm*.eval"))
    if len(logs) == 1:
        clusters = mgsm_clusters(logs[0])
        if args.write_clusters:
            args.write_clusters.write_text(json.dumps(cluster_ids(clusters), indent=1, sort_keys=True) + "\n")
    elif not logs and args.clusters.is_file() and not args.write_clusters:
        clusters = json.loads(args.clusters.read_text())["cluster_of_item"]
    else:
        raise SystemExit(f"expected one MGSM log in {RUN_DIRS['B0a'][0]} (found {len(logs)}) or the cluster file {args.clusters}")
    ids = set(arms["B0"].scores[MGSM])
    if set(clusters) != ids:
        raise SystemExit("the MGSM log and the runs do not hold the same items")
    rows, worst = {}, (0.0, "")
    for x, y in CLUSTER_PAIRS:
        item = contrast(arms[x], arms[y])["macro"]
        clu = clustered_macro_ci(arms[x], arms[y], TASKS, clusters)
        rows[f"{x}-{y}"] = {"item": item, "cluster": clu}
        for end in ("lo", "hi"):
            shift = abs(clu[end] - item[end])
            if shift > worst[0]:
                worst = (shift, f"{x}-{y}, {end}")
    res["cluster"] = {"n_clusters": len(set(clusters.values())), "n_items": len(clusters), "rows": rows, "max_shift": worst[0], "max_shift_where": worst[1]}

    args.json.write_text(json.dumps(res, indent=1) + "\n")
    args.md.write_text(render_md(res))
    print(render_md(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
