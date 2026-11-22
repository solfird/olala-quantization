"""Collect every number of the report from the result files into report/data.json.

    python3 report/build_data.py [--out DIR]

Reads only files under results/ and analysis/bench_summary.py (sizes come from the converter's own quantization_info.json).
Writes data.json (default report/data.json, or DIR/data.json) with two parts: `raw` (structured values used by the figures and
tables) and `k` (flat, pre-formatted strings used by the text templates, keyed like `N5.diff`). Nothing is typed here
except labels. results/retry-counts.json comes from analysis/retry_counts.py, which needs the held-back raw logs.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_ap = argparse.ArgumentParser()
_ap.add_argument("--out", type=Path, default=ROOT / "report", help="folder to write data.json to")
OUT = _ap.parse_args().out
R = ROOT / "results"
W1 = R / "wave1"


def load(p: Path):
    return json.loads(p.read_text())


def sgn(x: float, nd: int = 1) -> str:
    """Signed number with a typographic minus (LaTeX math minus)."""
    s = f"{abs(x):.{nd}f}"
    return f"$-${s}" if x < 0 and float(s) != 0 else f"+{s}" if x > 0 or float(s) == 0 and x >= 0 else s


def plain(x: float, nd: int = 1) -> str:
    return f"{x:.{nd}f}"


def ci(lo: float, hi: float, nd: int = 1) -> str:
    return f"{sgn(lo, nd)} to {sgn(hi, nd)}"


def gb(b: float, nd: int = 2) -> str:
    return f"{b / 1e9:.{nd}f}"


def tex_pow(x: float, nd: int = 1) -> str:
    """A number as a TeX power of ten, mantissa with nd decimals, in text mode."""
    import math

    e = math.floor(math.log10(abs(x)))
    m = x / 10**e
    if round(m, nd) >= 10:
        m, e = m / 10, e + 1
    return f"${m:.{nd}f}\\times 10^{{{e}}}$"


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


raw: dict = {}
k: dict[str, str] = {}

# ------------------------------------------------------------------ accuracy (wave 2)
A2 = load(R / "wave2-analysis.json")
A5 = load(R / "wave2-n5-analysis.json")
AG = load(R / "wave2-g1-analysis.json")
AQ = load(R / "wave2-q5-analysis.json")
A1 = load(R / "wave1-analysis.json")
AS = load(R / "wave2-sel-analysis.json")      # the new build (SEL in the files, N6 in the report), with N5, N2 and Q5
AQ4 = load(R / "wave2-q4-analysis.json")      # the Q4_K_M stand-in
AFP = load(R / "wave2-fp8-contrast.json")     # the new build against the two FP8 builds
AUD = load(R / "truncation-audit.json")
SENS = load(R / "review-sensitivity.json")     # analysis/review_sensitivity.py: sensitivity analyses added after the outside review (keys s.*)
FID = load(W1 / "fidelity-fid" / "summary.json")
GG = load(R / "bench-gguf-summary.json")
NAME = {"SEL": "N6"}                          # names in the analysis files -> names in the report
TASKS = ["mgsm", "humaneval_plus", "mbpp_plus", "ifeval", "aime2025"]
TASK_LABEL = {"mgsm": "MGSM", "humaneval_plus": "HumanEval+", "mbpp_plus": "MBPP+", "ifeval": "IFEval", "aime2025": "AIME 2025"}


def arm_entry(src: dict) -> dict:
    s = src["summary"]
    return {
        "macro": s["macro"]["x"],
        "diff": s["macro"]["diff"],
        "ref": s["macro"]["y"],
        "tasks": {t: s["tasks"][t] for t in TASKS},
    }


acc = {
    "F1": arm_entry(A2["candidates"]["F1"]),
    "F2": arm_entry(A2["candidates"]["F2"]),
    "N2": arm_entry(A2["candidates"]["N2"]),
    "N5": arm_entry(A5["candidates"]["N5"]),
    "G1": arm_entry(AG["candidates"]["G1"]),
    "Q5": arm_entry(AQ["exploratory"]["Q5"]),
    "N6": arm_entry(AQ4["candidates"]["SEL"]),
    "Q4": arm_entry(AQ4["exploratory"]["Q4"]),
}
raw["acc"] = acc
outcome = {a: A2["confirmatory"]["tests"]["-3"][a]["outcome"] for a in ("F1", "F2", "N2")}
for a, src in (("N5", A5), ("G1", AG)):
    outcome[a] = src["confirmatory"]["tests"]["-3"][a]["outcome"]
outcome["Q5"] = "descriptive"
outcome["N6"] = AQ4["confirmatory"]["tests"]["-3"]["SEL"]["outcome"]
outcome["Q4"] = "descriptive"
raw["outcome"] = outcome
b0 = acc["F1"]["ref"]
k["B0.macro"] = plain(b0["value"])
k["B0.ci"] = f"{plain(b0['lo'])} to {plain(b0['hi'])}"
for a, e in acc.items():
    k[f"{a}.loss"] = f"{abs(e['diff']['value']):.1f}"
    k[f"{a}.macro"] = plain(e["macro"]["value"])
    k[f"{a}.diff"] = sgn(e["diff"]["value"])
    k[f"{a}.ci"] = ci(e["diff"]["lo"], e["diff"]["hi"])
    k[f"{a}.lo"] = sgn(e["diff"]["lo"])
    k[f"{a}.hi"] = sgn(e["diff"]["hi"])
    k[f"{a}.outcome"] = outcome[a].lower().replace("-", " ") if outcome[a] != "descriptive" else "descriptive"
    for t in TASKS:
        d = e["tasks"][t]["diff"]
        k[f"{a}.{t}.diff"] = sgn(d["value"])
        k[f"{a}.{t}.ci"] = ci(d["lo"], d["hi"])
for t in TASKS:
    y = acc["F1"]["tasks"][t]["y"]
    k[f"B0.{t}"] = plain(y["value"])
    k[f"B0.{t}.n"] = str(acc["F1"]["tasks"][t]["n_items"])

# contrasts (descriptive)
def contrast(src: dict, name: str) -> dict:
    s = src["contrasts"][name]["summary"]
    return {"macro": s["macro"]["diff"], "tasks": {t: s["tasks"][t]["diff"] for t in TASKS}}


raw["contrast"] = {
    "N5-N2": contrast(A5, "N5 vs N2") if "N5 vs N2" in A5["contrasts"] else None,
}
for key, src, names in (("n5", A5, ["N5", "N2", "Q5", "F2"]), ("g1", AG, ["G1", "N5", "N2", "Q5"]), ("q5", AQ, ["N2", "Q5", "F2"])):
    for cname, c in src["contrasts"].items():
        s = c["summary"]["macro"]["diff"]
        label = re.sub(r"\W+", "", cname.replace(" vs ", "_")).replace("_", "-")
        raw.setdefault("contrasts_all", {})[f"{key}:{cname}"] = s
        x, y = c["x"], c["y"]
        k[f"c.{x}.{y}.diff"] = sgn(s["value"])
        k[f"c.{x}.{y}.ci"] = ci(s["lo"], s["hi"])
        k[f"c.{x}.{y}.lo"] = sgn(s["lo"])
        k[f"c.{x}.{y}.hi"] = sgn(s["hi"])

raw["contr"] = {}
for src in (AQ4, AS, AFP):
    for cname, c in src["contrasts"].items():
        x, y = NAME.get(c["x"], c["x"]), NAME.get(c["y"], c["y"])
        d = c["summary"]["macro"]["diff"]
        raw["contr"][f"{x}-{y}"] = {"macro": d, "tasks": {t: c["summary"]["tasks"][t]["diff"] for t in TASKS}}
        k[f"c.{x}.{y}.diff"] = sgn(d["value"])
        k[f"c.{x}.{y}.abs"] = f"{abs(d['value']):.1f}"
        k[f"c.{x}.{y}.ci"] = ci(d["lo"], d["hi"])
        k[f"c.{x}.{y}.lo"] = sgn(d["lo"])
        k[f"c.{x}.{y}.hi"] = sgn(d["hi"])
        for t in TASKS:
            dt = c["summary"]["tasks"][t]["diff"]
            k[f"c.{x}.{y}.{t}.diff"] = sgn(dt["value"])
            k[f"c.{x}.{y}.{t}.ci"] = ci(dt["lo"], dt["hi"])
k["N6.hi_exact"] = f"{acc['N6']['diff']['hi']:.3f}"
k["N6.hi_abs"] = f"{abs(acc['N6']['diff']['hi']):.3f}"
k["N6.lo_abs"] = f"{abs(acc['N6']['diff']['lo']):.1f}"

# truncation rates and run-to-run calibration
cal = A2["calibration"]["summary"]["macro"]["diff"]
k["cal.diff"] = sgn(cal["value"])
k["cal.ci"] = ci(cal["lo"], cal["hi"])
k["cal.width"] = plain(cal["hi"] - cal["lo"])
for a in ("F1", "F2", "N2"):
    c = A2["confirmatory"]["tests"]["-3"][a]
    k[f"{a}.p_holm"] = f"{c['p_holm']:.4f}"
k["N2.p_raw"] = f"{A2['confirmatory']['tests']['-3']['N2']['p_raw']:.2f}"
k["N2.p_holm"] = f"{A2['confirmatory']['tests']['-3']['N2']['p_holm']:.2f}"


def trunc_rate(run: str) -> tuple[int, int]:
    info = load(W1 / run / "run_info.json")
    return info["truncations"], info["n_samples"]


for a, run in (("B0a", "B0a-v2"), ("B0b", "B0b-v2"), ("F1", "F1-v2"), ("F2", "F2-v2"), ("N2", "N2-v2"), ("N5", "N5-v2"), ("G1", "G1-v2"), ("Q5", "Q5v2"), ("N6", "SEL-v2"), ("Q4", "Q4v2")):
    t, n = trunc_rate(run)
    k[f"{a}.trunc"] = f"{100 * t / n:.1f}"
    k[f"{a}.trunc_n"] = str(t)
k["n_samples"] = str(trunc_rate("B0a-v2")[1])

# wave 1 (legacy) versus wave 2
for a in ("F1", "F2", "N2"):
    w1 = A1["candidates"][a]["summary"]["macro"]["diff"]
    k[f"w1.{a}.diff"] = sgn(w1["value"])
    k[f"w1.{a}.ci"] = ci(w1["lo"], w1["hi"])
    k[f"w1.{a}.outcome"] = A1["confirmatory"]["tests"]["-3"][a]["outcome"].lower().replace("-", " ")
k["w1.B0.macro"] = plain(A1["candidates"]["F1"]["summary"]["macro"]["y"]["value"])

# ------------------------------------------------------------------ sizes
def tensor_bytes(run: str) -> int:
    return load(W1 / run / "quantization_info.json")["output"]["tensor_bytes"]


q5 = (W1 / "Q5v2-aime" / "server_provenance.txt").read_text()
q5_size = int(re.search(r"gguf_size=(\d+)", q5).group(1))
n5i = load(W1 / "N5-v2" / "quantization_info.json")
bf16_bytes = sum(v["bytes_in"] for v in n5i["totals_by_category"].values())
tot_params_all = sum(v["params"] for v in n5i["totals_by_category"].values())
g1info = W1 / "g1" / "checkpoint-files" / "quantization_info.json"
g1_bytes = load(g1info)["output"]["tensor_bytes"] if g1info.exists() else load(W1 / "g1" / "checkpoint.json")["size_bytes"]
q4_size = load(W1 / "q4" / "prepare.json")["quantized"]["size_bytes"]
sizes = {"B0": bf16_bytes, "F1": bf16_bytes, "F2": tensor_bytes("F2-v2"), "N2": tensor_bytes("N2-v2"), "N5": tensor_bytes("N5-v2"), "G1": g1_bytes, "Q5": q5_size,
         "N6": tensor_bytes("SEL-v2"), "Q4": q4_size}
raw["sizes"] = sizes
for a, v in sizes.items():
    k[f"{a}.gb"] = gb(v)
for a, v in sizes.items():
    k[f"{a}.bpw"] = f"{8 * v / tot_params_all:.1f}"
k["N5.gb_over_N2"] = gb(sizes["N5"] - sizes["N2"])
k["N5.gb_over_Q5"] = gb(sizes["N5"] - sizes["Q5"])
k["N5.pct_over_Q5"] = f"{100 * (sizes['N5'] - sizes['Q5']) / sizes['Q5']:.0f}"
k["N5.pct_of_F2"] = f"{100 * (1 - sizes['N5'] / sizes['F2']):.0f}"
k["N6.gb_under_Q5"] = gb(sizes["Q5"] - sizes["N6"])
k["N6.pct_under_Q5"] = f"{100 * (sizes['Q5'] - sizes['N6']) / sizes['Q5']:.1f}"
k["N6.pct_under_F2"] = f"{100 * (sizes['F2'] - sizes['N6']) / sizes['F2']:.0f}"
k["N6.pct_larger_F2"] = f"{100 * (sizes['F2'] - sizes['N6']) / sizes['N6']:.0f}"   # the offline FP8 file over the new build, minus one
k["N6.gb_under_N5"] = gb(sizes["N5"] - sizes["N6"])
k["N6.pct_under_N5"] = f"{100 * (sizes['N5'] - sizes['N6']) / sizes['N5']:.0f}"
k["N6.pct_under_B0"] = f"{100 * (sizes['B0'] - sizes['N6']) / sizes['B0']:.0f}"
k["Q4.gb_under_N6"] = gb(sizes["N6"] - sizes["Q4"])
k["bytes_ratio"] = f"{16 / n5i['output']['bits_per_quantized_weight']:.1f}"
k["N5.bits"] = f"{n5i['output']['bits_per_weight']:.1f}"
k["N2.bits"] = f"{load(W1 / 'N2-v2' / 'quantization_info.json')['output']['bits_per_weight']:.1f}"

# ------------------------------------------------------------------ throughput
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "analysis"))
import bench_summary as bs  # noqa: E402

arms_bench = {}
for label in ["bf16", "bf16-pod2", "F1", "F2", "N2", "N5", "SEL-bench"]:
    folder = R / "bench" / f"bench-{label}"
    arms_bench[NAME.get(label, "N6" if label == "SEL-bench" else label)] = {f"{w}:{c}": bs.point_stats(v) for (w, c), v in bs.load_arm(folder).items()}
raw["bench"] = arms_bench
ref = {p: (arms_bench["bf16"][p]["median"] + arms_bench["bf16-pod2"][p]["median"]) / 2 for p in arms_bench["bf16"]}
raw["bench_ref"] = ref
for label in ["F1", "F2", "N2", "N5", "N6"]:
    for p, s in arms_bench[label].items():
        w, c = p.split(":")
        k[f"bench.{label}.{w}.{c}"] = f"{s['median'] / ref[p]:.2f}"
for p in ref:
    w, c = p.split(":")
    k[f"bench.bf16.{w}.{c}"] = f"{ref[p]:.0f}"
    spread = abs(arms_bench["bf16"][p]["median"] - arms_bench["bf16-pod2"][p]["median"]) / ref[p]
    k[f"bench.spread.{w}.{c}"] = f"{100 * spread:.1f}"
sp = [abs(arms_bench["bf16"][p]["median"] - arms_bench["bf16-pod2"][p]["median"]) / ref[p] for p in ref]
k["bench.spread.max"] = f"{100 * max(sp):.1f}"
load_ratios = {a: [arms_bench[a][p]["median"] / ref[p] for p in ref if p.split(":")[1] in ("16", "32", "64", "128")] for a in ["F1", "F2", "N2", "N5", "N6"]}
for a, v in load_ratios.items():
    k[f"bench.{a}.load_min"] = f"{min(v):.2f}"
    k[f"bench.{a}.load_max"] = f"{max(v):.2f}"
fp8r = [r for a in ("F1", "F2") for r in load_ratios[a]]
k["bench.fp8.load_min"] = f"{min(fp8r):.2f}"
k["bench.fp8.load_max"] = f"{max(fp8r):.2f}"
k["bench.all.load_min"] = f"{min(r for v in load_ratios.values() for r in v):.2f}"
k["bench.all.load_max"] = f"{max(r for v in load_ratios.values() for r in v):.2f}"
fp8_pts = [p for p in ref if p.split(":")[1] in ("16", "32", "64", "128")]
ahead = sum(1 for p in fp8_pts if arms_bench["N6"][p]["median"] > max(arms_bench["F1"][p]["median"], arms_bench["F2"][p]["median"]))
assert ahead == len(fp8_pts), "the new build is not faster than both FP8 builds at every load point: rewrite the sentence in Step 9"
k["bench.N6.over_fp8_points"], k["bench.N6.load_points"] = str(ahead), str(len(fp8_pts))
raw["bench_failed"] = sum(s["failed"] for a in arms_bench.values() for s in a.values())

# ------------------------------------------------------------------ ablation
ab = {}
for f in glob.glob(str(W1 / "ablation-abl-*" / "**" / "summary.json"), recursive=True) + glob.glob(str(W1 / "ablation-abl-*" / "summary.json")):
    d = load(Path(f))
    fi = d["fidelity"]
    ab[d["spec"]] = {
        "kl": fi["kld_mean"]["value"], "kl_lo": fi["kld_mean"]["lo"], "kl_hi": fi["kld_mean"]["hi"],
        "median": fi["kld_median"]["value"], "p99": fi["kld_p99"]["value"],
        "ppl": fi["ppl_ratio"]["value"], "top1": fi["top1_agreement"]["value"],
        "quantized_layers": d["expected"]["quantized_layers"], "quantized_tensors": d["expected"]["quantized_tensors"],
    }
    if d.get("noise_floor"):
        noise = d["noise_floor"]
raw["ablation"] = ab
raw["ab_noise"] = noise["kld_mean"]
for spec, v in ab.items():
    key = re.sub(r"[^a-z0-9]+", "_", spec.lower()).strip("_")
    k[f"ab.{key}.kl"] = f"{v['kl']:.4f}"
    k[f"ab.{key}.ppl"] = f"{v['ppl']:.3f}"
    k[f"ab.{key}.top1"] = f"{v['top1']:.3f}"
k["ab.noise"] = f"{noise['kld_mean']:.4f}"
ex, nex = ab["experts"], ab["non-experts"]
k["ab.share_ratio"] = f"{nex['kl'] / ex['kl']:.1f}"
exp_params, tot_params = 4831838208, None
tot_params = sum(v["params"] for v in n5i["totals_by_category"].values())
nonexp_params = n5i["totals_by_category"]["kept"]["params"] + n5i["totals_by_category"]["ignored"]["params"]
# parameter shares of the quantized sets (quantization_info of N2 gives the full N2 set)
n2i = load(W1 / "N2-v2" / "quantization_info.json")
n2_q_params = n2i["totals_by_category"]["quantized"]["params"]
n5_q_params = n5i["totals_by_category"]["quantized"]["params"]
non_exp_q_params = n2_q_params - n5_q_params
k["ab.experts_params_share"] = f"{100 * n5_q_params / n2_q_params:.0f}"
k["ab.nonexperts_params_share"] = f"{100 * non_exp_q_params / n2_q_params:.0f}"
per_param = (nex["kl"] / non_exp_q_params) / (ex["kl"] / n5_q_params)
k["ab.per_param"] = f"{per_param:.0f}"
k["ab.experts_share_total"] = f"{100 * n5_q_params / tot_params:.0f}"
k["ab.nonexperts_share_total"] = f"{100 * non_exp_q_params / tot_params:.0f}"
raw["ab_params"] = {"experts": n5_q_params, "non_experts": non_exp_q_params, "all": n2_q_params}
k["n2.quantized_params_b"] = f"{n2_q_params / 1e9:.2f}"
k["n5.quantized_params_b"] = f"{n5_q_params / 1e9:.2f}"
k["total_params_b"] = f"{tot_params / 1e9:.2f}"

# ------------------------------------------------------------------ G1 and CAL-1
fg = load(W1 / "g1" / "fidelity.json")
gate = load(W1 / "g1" / "gate.json")
k["G1.kl"] = f"{fg['mean_kl']['value']:.4f}"
k["G1.ppl"] = f"{fg['ppl_ratio']['value']:.3f}"
k["G1.top1"] = f"{fg['top1_agreement']['value']:.3f}"
k["G1.kl_vs_N5_pct"] = f"{100 * (fg['mean_kl']['value'] / ab['experts']['kl'] - 1):.0f}"
k["gate.kl_max"] = f"{gate['thresholds']['kl_max']:.4f}"
k["gate.ppl_max"] = f"{gate['thresholds']['ppl_ratio_max']:.4f}"
k["N5.kl"] = k["ab.experts.kl"]
k["N5.ppl"] = k["ab.experts.ppl"]
k["N5.top1"] = k["ab.experts.top1"]
cm = load(W1 / "cal1-final" / "cal1" / "calib-cal1" / "manifest.json")
c = cm["counts"]
k["cal1.traces"] = str(c["n_traces"])
k["cal1.windows"] = str(cm["n_windows"])
k["cal1.tokens"] = thousands(c["n_tokens_rendered"])
k["cal1.dropped"] = str(c["n_dropped_leak"])
k["cal1.prompts"] = str(c["n_prompts"])
k["cal1.truncated"] = str(c["n_truncated"])
for dom, v in cm["domains"].items():
    k[f"cal1.{dom}"] = str(v["n_traces"])
    k[f"cal1.{dom}.dropped"] = str(v["n_dropped_leak"])
bo = (W1 / "g1" / "build-output.txt").read_text()
lr = [float(x) for x in re.findall(r"loss ratio up ([0-9.]+)", bo)]
k["g1.lossratio_min"] = f"{min(lr):.2f}"
k["g1.lossratio_max"] = f"{max(lr):.2f}"
k["g1.build_min"] = f"{float(re.search(r'calibrated in ([0-9.]+) s', bo).group(1)) / 60:.0f}"
rtn = [int(x) for x in re.findall(r"/ (\d+) RTN experts", bo)]
k["g1.rtn_share"] = f"{100 * sum(rtn) / (256 * len(rtn)):.1f}"
rep = glob.glob(str(W1 / "g1" / "g1lite-logs" / "**" / "g1lite_report.json"), recursive=True)
raw["g1_report_found"] = bool(rep)

# ------------------------------------------------------------------ probe
P = load(W1 / "analysis" / "probe" / "activation_probe.json")
layers = P["layers"]
raw["probe_layers"] = [{"layer": l["layer"], "type": l["type"], "mixer": l["sites"]["mixer_out"]["max_abs"], "moe": l["sites"]["moe_out"]["max_abs"],
                        "moe_post": l["sites"]["moe_post"]["max_abs"], "mixer_gt448": l["sites"]["mixer_out"]["frac_tokens_gt_448"]} for l in layers]
mam = [x for x in raw["probe_layers"] if x["type"] == "M"]
top3 = sorted(mam, key=lambda x: -x["mixer"])[:3]
k["probe.top3_layers"] = ", ".join(str(x["layer"]) for x in sorted(top3, key=lambda x: x["layer"]))
k["probe.n_mamba"] = str(len(mam))
k["probe.n_attn"] = str(len(layers) - len(mam))
big = {x["layer"]: x for x in mam if x["mixer"] > 448}
k["probe.mixers_gt448"] = ", ".join(str(i) for i in sorted(big))
k["probe.n_prompts"] = str(P["meta"].get("n_prompts", 80))
k["probe.max_prompt_tokens"] = str(P["meta"]["max_tokens_in_a_prompt"])
for lay in (0, 1, 2, 3, 7, 23):
    x = next(y for y in raw["probe_layers"] if y["layer"] == lay)
    k[f"probe.L{lay}.mixer"] = tex_pow(x["mixer"])
mx = max(raw["probe_layers"], key=lambda x: x["mixer"])
k["probe.max_val"] = tex_pow(mx["mixer"])
k["probe.max_layer"] = str(mx["layer"])
k["probe.max_orders"] = f"{__import__('math').log10(mx['mixer'] / 65504):.1f}"
moe_top = sorted(raw["probe_layers"], key=lambda x: -x["moe_post"])[:4]
k["probe.moe_top4"] = ", ".join(str(x["layer"]) for x in sorted(moe_top, key=lambda x: x["layer"]))
# claim verdicts from the markdown table
md = (W1 / "analysis" / "probe" / "activation_probe.md").read_text()
verd = {}
for m in re.finditer(r"^\| ([A-E][0-9A-Za-z-]*) \| (.*?) \| (.*?) \| (.*?) \| (.*?) \| (consistent|partly consistent|not consistent|not judged) \|$", md, re.M):
    verd[m.group(1)] = {"claim": m.group(2), "claimed": m.group(3), "measured": m.group(4), "verdict": m.group(6)}
raw["claims"] = verd
from collections import Counter  # noqa: E402

cnt = Counter(v["verdict"] for v in verd.values())
k["probe.n_claims"] = str(len(verd))
for kind in ("consistent", "partly consistent", "not consistent", "not judged"):
    k[f"probe.n.{kind.replace(' ', '_')}"] = str(cnt.get(kind, 0))
for cid in ("A2", "A3", "A4", "A5", "B1", "C1", "C2"):
    if cid in verd:
        k[f"probe.{cid}"] = verd[cid]["verdict"]
for cid, name in (("D-mixer-L1", "Dmix1"), ("D-head-L1", "Dh1"), ("D-head-L23", "Dh23")):
    m = verd[cid]["measured"]
    k[f"probe.{name}.kl"] = re.search(r"; KL ([0-9.]+)", m).group(1)
    k[f"probe.{name}.top1"] = re.search(r"top-1 agreement ([0-9.]+)", m).group(1)
    k[f"probe.{name}.verdict"] = verd[cid]["verdict"]
    k[f"probe.{name}.dnll"] = re.search(r"(-?[0-9.]+) nats", m).group(1).replace("-", "$-$")
# ablation of the spiking branches: zeroing layer-1 mixer
abl = {a.get("name", ""): a for a in P["ablations"]} if isinstance(P["ablations"], list) else {}
raw["probe_ablation_names"] = list(abl)[:12]

# ------------------------------------------------------------------ weight statistics
WS = load(W1 / "analysis" / "weight_stats" / "weight_stats.json")
ws = {}
for r in WS["summary"]:
    ws.setdefault(r["class"], {"params": r["params"]})[r["format"]] = r["wmean_rel_fro"]
for r in WS["distribution_summary"]:
    ws.setdefault(r["class"], {"params": r["params"]})["kurt"] = r["wmean_excess_kurtosis"]
    ws[r["class"]]["max_rms"] = r["worst_max_over_rms"]
    ws[r["class"]]["chan_range"] = r["worst_channel_rms_range"]
raw["ws"] = ws
nv = [v["nvfp4"] for v in ws.values() if "nvfp4" in v]
fp = [v["fp8"] for v in ws.values() if "fp8" in v]
k["ws.nvfp4_min"] = f"{min(nv):.3f}"
k["ws.nvfp4_max"] = f"{max(nv):.3f}"
k["ws.fp8_min"] = f"{min(fp):.3f}"
k["ws.fp8_max"] = f"{max(fp):.3f}"
gr = load(R / "gaussian-reference.json")["rel_fro"]
k["ws.gauss_nvfp4"] = f"{gr['nvfp4']:.3f}"
k["ws.gauss_fp8"] = f"{gr['fp8']:.3f}"
k["ws.n_tensors"] = str(sum(1 for r in WS["rows"]))
routed_all = ws["routed_up"]["params"] + ws["routed_down"]["params"]   # the routed experts of all 36 expert layers
k["ab.routed_all_share"] = f"{100 * routed_all / tot_params:.1f}"
k["ab.routed_all_m"] = thousands(round(routed_all / 1e6))
k["ws.kurt.mamba_in_proj"] = f"{ws['mamba_in_proj']['kurt']:.0f}"
k["ws.kurt.routed_up"] = f"{ws['routed_up']['kurt']:.0f}"
k["ws.kurt.routed_down"] = f"{ws['routed_down']['kurt']:.0f}"
k["ws.kurt.latent_proj"] = f"{ws['latent_proj']['kurt']:.1f}"
k["ws.kurt.attn_gate"] = f"{ws['attn_gate']['kurt']:.1f}"
k["ws.chan.mamba_in_proj"] = f"{ws['mamba_in_proj']['chan_range']:.1e}"

# ------------------------------------------------------------------ retries (harness artifact)
RC = load(R / "retry-counts.json")
raw["retry"] = RC


def share(group: str, run: str, task: str) -> tuple[int, int]:
    t = RC[group][run][task]
    return t["more_than_one_request"], t["samples"]


first = {a: share("wave1_full", a, "aime2025") for a in ("B0a", "B0b", "F1", "F2", "N2", "F1n")}
rerun = {a: share("wave1_aime_rerun", r, "aime2025") for a, r in (("B0a", "B0a-aime2"), ("B0b", "B0b-aime3"), ("F1", "F1-aime2"), ("F2", "F2-aime2"), ("N2", "N2-aime2"), ("F1n", "F1n-aime2"))}
pct = lambda t: 100 * t[0] / t[1]  # noqa: E731
k["rt.aime_first_min"] = f"{min(pct(v) for v in first.values()):.0f}"
k["rt.aime_first_max"] = f"{max(pct(v) for v in first.values()):.0f}"
k["rt.aime_rerun_min"] = f"{min(pct(v) for v in rerun.values()):.0f}"
k["rt.aime_rerun_max"] = f"{max(pct(v) for v in rerun.values()):.0f}"
mg = {a: share("wave1_full", a, "mgsm") for a in ("B0a", "B0b", "F1", "F2", "N2", "F1n")}
k["rt.mgsm_min"] = f"{min(pct(v) for v in mg.values()):.1f}"
k["rt.mgsm_max"] = f"{max(pct(v) for v in mg.values()):.1f}"
q5w1 = share("q5_wave1", "Q5-mgsm-he", "mgsm")
k["rt.q5_mgsm"] = f"{pct(q5w1):.0f}"
k["rt.ifeval_max"] = f"{max(pct(share('wave1_full', a, 'ifeval')) for a in ('B0a', 'B0b', 'F1', 'F2', 'N2', 'F1n')):.1f}"
k["rt.wave2_total"] = str(sum(v["more_than_one_request"] for r in RC["wave2"].values() for v in r.values()))
k["rt.wave2_max_wt"] = f"{max(v['max_working_time_s'] for r in RC['wave2'].values() for v in r.values()):.0f}"
k["rt.wave1_max_wt"] = f"{max(v['max_working_time_s'] for g in ('wave1_full', 'wave1_aime_rerun') for r in RC[g].values() for v in r.values()):.0f}"
k["rt.q5_wave2_max_wt"] = f"{max(v['max_working_time_s'] for r in RC['q5_wave2'].values() for v in r.values()):.0f}"
raw["rt_first"] = first
raw["rt_rerun"] = rerun

# ------------------------------------------------------------------ the smaller-build candidates (stage 1)
fa = {a["name"]: a for a in FID["arms"]}
raw["fid"] = {n: {"kl": fa[n]["fidelity"]["kld_mean"]["value"], "ppl": fa[n]["fidelity"]["ppl_ratio"]["value"], "top1": fa[n]["fidelity"]["top1_agreement"]["value"],
                  "p99": fa[n]["fidelity"]["kld_p99"]["value"], "bytes": fa[n]["file_bytes"]} for n in ("N5", "C1", "C2")}
for n, v in raw["fid"].items():
    k[f"fid.{n}.kl"], k[f"fid.{n}.ppl"], k[f"fid.{n}.top1"], k[f"fid.{n}.p99"] = f"{v['kl']:.4f}", f"{v['ppl']:.3f}", f"{v['top1']:.3f}", f"{v['p99']:.3f}"
    k[f"fid.{n}.gb"] = gb(v["bytes"])
k["fid.kl_max"] = FID["thresholds"]["mean_kl_max"]
k["fid.ppl_max"] = FID["thresholds"]["ppl_ratio_max"]
k["fid.C2.kl_vs_C1_pct"] = f"{100 * (1 - raw['fid']['C2']['kl'] / raw['fid']['C1']['kl']):.0f}"
k["fid.C1.kl_vs_N5_pct"] = f"{100 * (raw['fid']['C1']['kl'] / raw['fid']['N5']['kl'] - 1):.0f}"
k["fid.C2.kl_vs_N5_pct"] = f"{100 * (1 - raw['fid']['C2']['kl'] / raw['fid']['N5']['kl']):.0f}"
NF = load(W1 / "fidelity-fid" / "noise_floor.json")["kld_mean"]   # two bf16 collections on the server of the smaller-build step
k["fid.noise"], k["fid.noise_ci"] = f"{NF['value']:.4f}", f"{NF['lo']:.4f} to {NF['hi']:.4f}"

# ------------------------------------------------------------------ truncated answers (audit of the raw logs)
N_ITEMS = dict(zip(TASKS, (550, 164, 378, 541, 150)))
asum = AUD["summary"]
AUDK = {"N6": "NVFP4 C2 (this build)", "Q4": "Q4_K_M stand-in", "Q5": "OVH Q5_K_M", "B0a": "bf16 (run a)", "B0b": "bf16 (run b)", "N5": "NVFP4 experts only (N5)",
        "N2": "NVFP4 all linear (N2)", "F1": "FP8 online", "F2": "FP8 offline", "G1": "NVFP4 experts, calibrated (G1)"}
for a, nm in AUDK.items():
    v = asum[nm]
    k[f"tr.{a}.n"], k[f"tr.{a}.loops"], k[f"tr.{a}.long"] = str(v["truncated"]), str(v["loops"]), str(v["long"])
    k[f"tr.{a}.target_late"], k[f"tr.{a}.with_text"] = str(v["target_late"]), str(v["with_text"])
allv = list(asum.values())
k["tr.frac_long"] = f"{round(100 * sum(v['long'] for v in allv) / sum(v['truncated'] for v in allv))}"
shares = [v["long"] / v["truncated"] for v in allv if v["truncated"]]
k["tr.long_min"], k["tr.long_max"] = f"{round(100 * min(shares))}", f"{round(100 * max(shares))}"
_all = [m_ for rows_ in AUD["samples"].values() for m_ in rows_]
_isloop = lambda m_, thr=0.12: m_["compress"] < thr or m_["unique_lines"] < 0.25  # noqa: E731
k["aud.loop_max"] = f"{max(m_['compress'] for m_ in _all if _isloop(m_) and m_['compress'] < 0.12):.3f}"
k["aud.long_min"] = f"{min(m_['compress'] for m_ in _all if not _isloop(m_)):.3f}"
k["tr.cut_lo"], k["tr.cut_hi"] = "0.05", "0.15"
for nm, thr in (("lo", 0.05), ("hi", 0.15)):
    k[f"tr.frac_long_{nm}"] = f"{round(100 * sum(1 for m_ in _all if not _isloop(m_, thr)) / len(_all))}"
k["tr.loops_bf16"] = f"{asum[AUDK['B0a']]['loops']} and {asum[AUDK['B0b']]['loops']}"
k["tr.bf16_n"] = f"{asum[AUDK['B0a']]['truncated']} and {asum[AUDK['B0b']]['truncated']}"
k["tr.n_builds"], k["tr.long_total"], k["tr.trunc_total"] = str(len(allv)), str(sum(v["long"] for v in allv)), str(sum(v["truncated"] for v in allv))
for nm, thr in (("20", 0.20), ("25", 0.25)):
    k[f"tr.frac_long_{nm}"] = f"{round(100 * sum(1 for m_ in _all if not _isloop(m_, thr)) / len(_all))}"
for a, nm in AUDK.items():   # cut-off answers of every audited build, per task
    cnt_ = Counter(m_["task"].split("/")[-1].replace("-", "_") for m_ in AUD["samples"][nm])
    assert sum(cnt_.values()) == asum[nm]["truncated"]
    for t in TASKS:
        k[f"tr.{a}.cut.{t}"] = str(cnt_.get(t, 0))


def generous(arm_name: str, base: float) -> float:
    by_task: dict[str, int] = {}
    for m_ in AUD["samples"][arm_name]:
        if m_["target_late"]:
            t = m_["task"].split("/")[-1].replace("-", "_")
            by_task[t] = by_task.get(t, 0) + 1
    return base + sum(by_task.get(t, 0) / N_ITEMS[t] * 100 for t in TASKS) / len(TASKS)


gen = {a: generous(AUDK[a], acc[a]["macro"]["value"]) for a in ("N6", "Q4", "Q5")}
raw["generous"] = gen
for a in ("N6", "Q4", "Q5"):
    k[f"gen.{a}"] = f"{gen[a]:.1f}"
k["gen.gap_N6_Q4"] = f"{gen['N6'] - gen['Q4']:.1f}"
k["gen.gap_Q4_Q5"] = f"{gen['Q5'] - gen['Q4']:.1f}"
k["N6.aime_share_of_gap"] = f"{raw['contr']['N6-Q4']['tasks']['aime2025']['value'] / len(TASKS):.1f}"
q4t = AQ4["exploratory"]["Q4"]["truncation"]["x"]["by_task"]
q5t = AQ4["exploratory"]["Q5"]["truncation"]["x"]["by_task"]
k["tr.Q4.aime"], k["tr.Q4.aime_n"], k["tr.Q5.aime"] = str(q4t["aime2025"]["truncations"]), str(q4t["aime2025"]["samples"]), str(q5t["aime2025"]["truncations"])
k["tr.Q4.mgsm"], k["tr.Q5.mgsm"] = str(q4t["mgsm"]["truncations"]), str(q5t["mgsm"]["truncations"])
k["tr.N6.aime"] = str(AQ4["candidates"]["SEL"]["truncation"]["x"]["by_task"]["aime2025"]["truncations"])
assert k["tr.N6.aime"] == k["tr.N6.cut.aime2025"] and k["tr.Q4.aime"] == k["tr.Q4.cut.aime2025"] and k["tr.Q5.aime"] == k["tr.Q5.cut.aime2025"]
k["Q4.aime_nontrunc"] = sgn(AQ4["exploratory"]["Q4"]["aime_non_truncated"]["diff"]["value"])
k["Q4.aime_nontrunc_ci"] = ci(AQ4["exploratory"]["Q4"]["aime_non_truncated"]["diff"]["lo"], AQ4["exploratory"]["Q4"]["aime_non_truncated"]["diff"]["hi"])


def out_tokens(dirs: list[str], task: str = "aime2025") -> list[int]:
    v = []
    for d in dirs:
        for line in (W1 / d / "samples.jsonl").read_text().splitlines():
            try:
                r_ = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r_["task"].split("/")[-1].replace("-", "_") == task:
                v.append(r_["output_tokens"])
    return v


import statistics as _st  # noqa: E402

for a, dirs in (("Q4", ["Q4-c"]), ("Q5", ["Q5v2-aime"]), ("B0", ["B0a-v2", "B0b-v2"]), ("N6", ["SEL-v2"])):
    k[f"aime.tokens.{a}"] = f"{round(_st.median(out_tokens(dirs)) / 1000, 1):.1f}"
k["ctx.max_len"] = "32,688"

# ------------------------------------------------------------------ llama.cpp sweeps of the Q4 stand-in and Q5_K_M (two pods)
GGr = GG["result"]
for c_ in (1, 4, 12):
    k[f"gg.q4.{c_}"], k[f"gg.q5.{c_}"] = f"{GGr[f'q4_k_m-ovhrecipe:{c_}']['median']:.0f}", f"{GGr[f'q5_k_m:{c_}']['median']:.0f}"
    k[f"gg.q4.{c_}.n"], k[f"gg.q5.{c_}.n"] = str(GGr[f"q4_k_m-ovhrecipe:{c_}"]["valid"]), str(GGr[f"q5_k_m:{c_}"]["valid"])
_v = GGr["q5_k_m:1"]["values"]   # the first two repeats come from one pod, the last two from the other
_a, _b = _st.median(_v[:2]), _st.median(_v[2:])
k["gg.q5.pod_spread"] = f"{100 * abs(_a - _b) / ((_a + _b) / 2):.0f}"
k["gg.excluded"] = str(len(GG["excluded"]))
k["gg.failed_min"], k["gg.failed_max"] = str(min(e["failed"] for e in GG["excluded"])), str(max(e["failed"] for e in GG["excluded"]))


# ------------------------------------------------------------------ sensitivity analyses after the outside review (analysis/review_sensitivity.py)
# Every key is s.*. Differences and intervals are formatted like c.X.Y.diff and c.X.Y.ci (one decimal, "lo to hi"); the KL ones have four decimals.
assert SENS["meta"]["n_boot"] == 10000 and SENS["meta"]["seed"] == 0, "the sensitivity analysis must use the report's bootstrap"
SAME = lambda a, b: all(abs(a[q] - b[q]) < 1e-9 for q in ("value", "lo", "hi"))  # noqa: E731


def put(prefix: str, c: dict, nd: int = 1) -> None:
    k[f"{prefix}.diff"], k[f"{prefix}.ci"] = sgn(c["value"], nd), ci(c["lo"], c["hi"], nd)


# (a) macro of four tasks, without AIME
for pair, v in SENS["noaime"].items():
    x_, y_ = pair.split("-")
    put(f"s.noaime.{x_}.{y_}", v["without"])
    ref_ = raw["contr"][pair]["macro"] if pair in raw["contr"] else acc[x_]["diff"]   # the five-task value is already a report number
    assert SAME(v["with"], ref_), f"the five-task value of {pair} differs from the report"

# (b) generous rescoring
for a_ in ("N6", "Q4", "Q5"):
    assert abs(SENS["generous"]["macro"][a_]["generous"] - gen[a_]) < 1e-9, f"the generous macro of {a_} differs from Table 9"
for pair, v in SENS["generous"]["contrasts"].items():
    x_, y_ = pair.split("-")
    put(f"s.gen.{x_}.{y_}", v["generous"])
    assert SAME(v["strict"], raw["contr"][pair]["macro"])
    k[f"s.gen.{x_}.{y_}.strict"] = sgn(v["strict"]["value"])
k["s.gen.credited.N6"], k["s.gen.credited.Q4"], k["s.gen.credited.Q5"] = (str(SENS["generous"]["macro"][a_]["credited"]) for a_ in ("N6", "Q4", "Q5"))

# (c) plain-text KL of the three smaller-build candidates: paired chunk bootstrap
SK = SENS["kl"]["by_k"][str(SENS["kl"]["k_main"])]
for n_ in ("N5", "C1", "C2"):
    m_ = SK["means"][n_]
    assert abs(m_["value"] - raw["fid"][n_]["kl"]) < 1e-9
    k[f"s.kl.{n_}.mean"], k[f"s.kl.{n_}.ci"] = f"{m_['value']:.4f}", f"{m_['lo']:.4f} to {m_['hi']:.4f}"
for name_, v in SK["diffs"].items():
    put(f"s.kl.{name_}", v, 4)
    k[f"s.kl.{name_}.pct"], k[f"s.kl.{name_}.pct_ci"] = sgn(v["pct"]), ci(v["pct_lo"], v["pct_hi"])
    k[f"s.kl.{name_}.pct_abs"], k[f"s.kl.{name_}.pct0"] = f"{abs(v['pct']):.1f}", f"{abs(v['pct']):.0f}"
k["s.kl.n_chunks"] = str(SENS["kl"]["chunks"])
# k = 5, 10 and 20
for kk_, v in SENS["kl"]["by_k"].items():
    for n_, m_ in v["means"].items():
        k[f"s.ksens.{n_}.k{kk_}"] = f"{m_['value']:.4f}"
    for name_, d_ in v["diffs"].items():
        k[f"s.ksens.{name_}.k{kk_}"] = sgn(d_["value"], 4)
    k[f"s.ksens.order.k{kk_}"] = " below ".join(v["order"])
k["s.ksens.order_holds"] = "yes" if SENS["kl"]["order_holds"] else "no"
k["s.ksens.signs_hold"] = "yes" if len({(name_, d_["value"] > 0) for v in SENS["kl"]["by_k"].values() for name_, d_ in v["diffs"].items()}) == 3 else "no"
for n_, g_ in SENS["kl"]["growth"].items():
    k[f"s.ksens.{n_}.grow_5_10"], k[f"s.ksens.{n_}.grow_10_20"] = f"{g_['5_to_10']:.0f}", f"{g_['10_to_20']:.0f}"

# (d) the two bf16 runs (run a minus run b), and each build against each run alone
for t in TASKS:
    put(f"s.bf16.{t}", SENS["bf16"]["tasks"][t])
put("s.bf16.macro", SENS["bf16"]["macro"])
assert k["s.bf16.macro.diff"] == k["cal.diff"] and k["s.bf16.macro.ci"] == k["cal.ci"]
k["s.bf16.maxabs"] = f"{SENS['bf16']['max_abs_task']:.1f}"
for a_, v in SENS["ref"].items():
    put(f"s.ref.{a_}.a", v["a"])
    put(f"s.ref.{a_}.b", v["b"])
    assert SAME(v["pooled"], acc[a_]["diff"]), f"the pooled-reference value of {a_} differs from Table 1"

# (e) margins: does the lower end of the macro interval against bf16 pass -2, -3 and -5, and the task-level flag
flagged = []
for a_, v in SENS["margin"].items():
    assert SAME(v["macro"], acc[a_]["diff"])
    k[f"s.margin.{a_}.lo3"] = sgn(v["macro"]["lo"], 3)
    for m_, st_ in v["margins"].items():
        key_ = abs(int(m_))
        k[f"s.margin.{a_}.pass{key_}"] = "yes" if st_["lo_above"] else "no"
        k[f"s.margin.{a_}.state{key_}"] = st_["state"]
    k[f"s.margin.{a_}.flag_n"] = str(len(v["task_flag"]))
    k[f"s.margin.{a_}.flag"] = "; ".join(f"{TASK_LABEL[f_['task']]} {sgn(f_['value'])} ({ci(f_['lo'], f_['hi'])})" for f_ in v["task_flag"]) or "none"
    flagged += [a_] if v["task_flag"] else []
k["s.margin.flag_builds"] = ", ".join(flagged) or "none"
k["s.margin.flag_n"] = str(len(flagged))
raw["margin"] = SENS["margin"]

# (f) MGSM resampled by source problem
CL = SENS["cluster"]
k["s.clust.n_clusters"], k["s.clust.n_items"] = str(CL["n_clusters"]), str(CL["n_items"])
k["s.clust.maxshift"], k["s.clust.maxshift2"], k["s.clust.n_pairs"] = f"{CL['max_shift']:.1f}", f"{CL['max_shift']:.2f}", str(len(CL["rows"]))
assert CL["max_shift"] < 0.15, "the intervals moved by more than 0.1 points when MGSM is resampled by source problem: rewrite the sentence"

# ------------------------------------------------------------------ B0g, the same-engine control (analysis/b0g_control.py)
# Every key is g.*. Differences and intervals are formatted like c.X.Y.diff and c.X.Y.ci (one decimal, "lo to hi", $-$ for minus).
BG = load(R / "b0g-control.json")
assert BG["meta"]["n_boot"] == 10000 and BG["meta"]["seed"] == 0, "the B0g analysis must use the report's bootstrap"
k["g.n.reproduced"] = str(BG["meta"]["reproduced"])
raw["g"] = {name_: BG[name_] for name_ in ("absolute", "engine", "format", "cross", "vllm", "double", "decomposition", "noaime", "margin", "truncation")}   # for report/make_tables.py


def put_full(prefix: str, c: dict) -> None:
    put(prefix, c)
    k[f"{prefix}.lo"], k[f"{prefix}.hi"], k[f"{prefix}.abs"] = sgn(c["lo"]), sgn(c["hi"]), f"{abs(c['value']):.1f}"


def put_set(prefix: str, cd: dict) -> None:
    """Macro (diff, ci, lo, hi, abs) and, per task, diff and ci of a contrast ``{macro, tasks}``."""
    put_full(prefix, cd["macro"])
    for t_ in TASKS:
        put(f"{prefix}.{t_}", cd["tasks"][t_])


# absolute accuracy: the arms that are already in the report must agree with it
for a_, v in BG["absolute"].items():
    k[f"g.macro.{a_}"], k[f"g.macro.{a_}.ci"] = plain(v["macro"]["value"]), f"{plain(v['macro']['lo'])} to {plain(v['macro']['hi'])}"
    for t_ in TASKS:
        k[f"g.abs.{a_}.{t_}"] = plain(v["tasks"][t_]["value"])
    if a_ in acc or a_ == "B0":
        ref_ = acc["F1"]["ref"] if a_ == "B0" else acc[a_]["macro"]
        assert SAME(v["macro"], ref_), f"the macro of {a_} differs from the report"
        for t_ in TASKS:
            assert SAME(v["tasks"][t_], acc["F1"]["tasks"][t_]["y"] if a_ == "B0" else acc[a_]["tasks"][t_]["x"])
assert k["g.macro.B0"] == k["B0.macro"]

# (a) engine effect: B0g minus vLLM bf16 (pooled), minus run a, minus run b
for pair_, pre_ in (("B0g-B0", "g.eng"), ("B0g-B0a", "g.eng.a"), ("B0g-B0b", "g.eng.b")):
    put_set(pre_, BG["engine"][pair_])
# (b) format effect on llama.cpp: Q5 and Q4 minus B0g, and Q4 minus Q5 (the last is already in the report)
for pair_, pre_ in (("Q5-B0g", "g.fmt.Q5"), ("Q4-B0g", "g.fmt.Q4"), ("Q4-Q5", "g.fmt.Q4Q5")):
    put_set(pre_, BG["format"][pair_])
assert SAME(BG["format"]["Q4-Q5"]["macro"], raw["contr"]["Q4-Q5"]["macro"]), "Q4 minus Q5 differs from the report"
# (c) cross-engine: the NVFP4 new build and the two FP8 builds (vLLM) minus B0g (llama.cpp)
for x_ in ("N6", "F1", "F2"):
    put_set(f"g.cross.{x_}", BG["cross"][f"{x_}-B0g"])
# (d) the same builds against the vLLM reference of the report, and the double difference
for x_ in ("Q5", "Q4", "N6"):
    put_set(f"g.vllm.{x_}", BG["vllm"][f"{x_}-B0"])
    assert SAME(BG["vllm"][f"{x_}-B0"]["macro"], acc[x_]["diff"]), f"{x_} minus B0 differs from the report"
    d_ = BG["decomposition"][x_]
    k[f"g.dec.{x_}.total"], k[f"g.dec.{x_}.fmt"], k[f"g.dec.{x_}.eng"] = sgn(d_["total"]), sgn(d_["format"]), sgn(d_["engine"])
    k[f"g.dec.{x_}.eng_share"] = "n/a" if d_["engine_share"] is None else f"{d_['engine_share']:.0f}"
    k[f"g.dec.{x_}.fmt_share"] = "n/a" if d_["engine_share"] is None else f"{100 - d_['engine_share']:.0f}"
for x_ in ("Q5", "Q4"):
    put_set(f"g.dbl.{x_}", BG["double"][x_])
    e_ = BG["engine"]["B0g-B0"]["macro"]   # the double difference is B0 minus B0g: minus the engine effect, interval ends exchanged
    assert abs(BG["double"][x_]["macro"]["value"] + e_["value"]) < 1e-9 and abs(BG["double"][x_]["macro"]["lo"] + e_["hi"]) < 1e-9
# (e) leave AIME out of (a) and (b)
NOAIME_KEY = {"B0g-B0": "eng", "B0g-B0a": "eng.a", "B0g-B0b": "eng.b", "Q5-B0g": "fmt.Q5", "Q4-B0g": "fmt.Q4", "Q4-Q5": "fmt.Q4Q5"}
for pair_, pre_ in NOAIME_KEY.items():
    put(f"g.noaime.{pre_}", BG["noaime"][pair_]["without"])
    assert SAME(BG["noaime"][pair_]["with"], BG["engine" if pair_.startswith("B0g") else "format"][pair_]["macro"])
# (f) margins: the lower end of the macro interval against -2, -3 and -5 (unadjusted); vs_B0g reads B0g itself against B0
for which_, pre_ in (("vs_B0g", "g.margin"), ("vs_B0", "g.margin.vB0")):
    for a_, v in BG["margin"][which_].items():
        k[f"{pre_}.{a_}.lo3"] = sgn(v["macro"]["lo"], 3)
        for m_, st_ in v["margins"].items():
            k[f"{pre_}.{a_}.pass{abs(int(m_))}"] = "yes" if st_["lo_above"] else "no"
            k[f"{pre_}.{a_}.state{abs(int(m_))}"] = st_["state"]
        k[f"{pre_}.{a_}.flag_n"] = str(len(v["task_flag"]))
        k[f"{pre_}.{a_}.flag"] = "; ".join(f"{TASK_LABEL[f_['task']]} {sgn(f_['value'])} ({ci(f_['lo'], f_['hi'])})" for f_ in v["task_flag"]) or "none"
for a_ in ("Q5", "Q4", "N6"):   # the vs_B0 rows are the ones of the report's margin table
    assert SAME(BG["margin"]["vs_B0"][a_]["macro"], SENS["margin"][a_]["macro"])
assert BG["margin"]["vs_B0"]["B0g"]["macro"] == BG["engine"]["B0g-B0"]["macro"]

# truncated answers of B0g: the audit row, per task, and the generous rule
TB = BG["truncation"]
assert TB["row"] == asum[TB["audit_name"]], "the B0g row of the audit changed since b0g-control.json was written: rerun analysis/b0g_control.py"
for key_, nm_ in (("n", "truncated"), ("long", "long"), ("loops", "loops"), ("target_late", "target_late"), ("with_text", "with_text")):
    k[f"g.tr.{key_}"] = str(TB["row"][nm_])
for t_ in TASKS:
    k[f"g.tr.cut.{t_}"] = str(TB["cut_by_task"][t_])
k["g.n.cut.total"] = str(TB["cut_total"])
assert sum(TB["cut_by_task"].values()) == TB["cut_total"] == TB["row"]["truncated"]
b0g_t, b0g_n = (sum(x_) for x_ in zip(*(trunc_rate(d_) for d_ in BG["meta"]["run_dirs"]["B0g"]), strict=True))
assert b0g_t == TB["cut_total"], "the truncations of run_info.json and of the audit differ for B0g"
k["g.trunc"], k["g.trunc_n"], k["g.n.samples"] = f"{100 * b0g_t / b0g_n:.1f}", str(b0g_t), str(b0g_n)
raw["g_generous"] = {a_: v["generous"] for a_, v in BG["generous"]["macro"].items()}
for a_, v in BG["generous"]["macro"].items():
    k[f"g.gen.macro.{a_}"] = plain(v["generous"])
    if v["credited"] is not None:
        k[f"g.gen.credited.{a_}"] = str(v["credited"])
for a_ in ("Q5", "Q4", "N6"):
    assert abs(BG["generous"]["macro"][a_]["generous"] - gen[a_]) < 1e-9, f"the generous macro of {a_} differs from Table 9"
assert abs(BG["generous"]["macro"]["B0g"]["generous"] - generous(TB["audit_name"], BG["absolute"]["B0g"]["macro"]["value"])) < 1e-9
for pair_, pre_ in (("B0g-B0", "g.gen.eng"), ("Q5-B0g", "g.gen.fmt.Q5"), ("Q4-B0g", "g.gen.fmt.Q4"), ("N6-B0g", "g.gen.cross.N6"), ("Q5-B0", "g.gen.vllm.Q5"), ("Q4-B0", "g.gen.vllm.Q4")):
    v = BG["generous"]["contrasts"][pair_]
    put_full(pre_, v["generous"])
    k[f"{pre_}.strict"] = sgn(v["strict"]["value"])
    assert SAME(v["strict"], {"B0g-B0": BG["engine"], "N6-B0g": BG["cross"]}.get(pair_, BG["format"] if "B0g" in pair_ else BG["vllm"])[pair_]["macro"])

# Table 8: how many comparisons it makes, counted from the table data (columns are listed in raw["minus_cols"] and used by make_tables.py)
MINUS_COLS = ["N6-N5", "N6-Q5", "N6-Q4", "N6-F1", "N6-F2"]
raw["minus_cols"] = MINUS_COLS
k["tab8.n_cols"] = str(len(MINUS_COLS))
k["tab8.n_task"] = str(sum(len(raw["contr"][c_]["tasks"]) for c_ in MINUS_COLS))
k["tab8.n_all"] = str(sum(len(raw["contr"][c_]["tasks"]) + 1 for c_ in MINUS_COLS))

# ------------------------------------------------------------------ spend
led = load(R / "spend-cap.json")   # the cap alone; the ledger of the pods stays in the private working repository
k["spend.cap"] = str(led["cap_usd"])
out = {"raw": raw, "k": k}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "data.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
print(f"{len(k)} keys written to {OUT / 'data.json'}")
for key in ("B0.macro", "N5.diff", "N5.ci", "G1.diff", "N5.gb", "N2.gb", "Q5.gb", "ab.per_param", "ab.share_ratio", "probe.top3_layers", "probe.mixers_gt448", "ws.nvfp4_min", "ws.nvfp4_max", "rt.aime_first_min", "bench.N5.short.1", "bench.N2.short.32"):
    print(key, "=", k.get(key))
