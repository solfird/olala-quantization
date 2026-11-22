"""Write the LaTeX tables of the report from report/data.json (run build_data.py first).

    python3 report/make_tables.py [--data FILE] [--out DIR]    (defaults: report/data.json, report/)

Result tables use the lab's three-state rule, computed here: green fill and an up triangle when the whole 95% interval
of a difference is above zero, red fill and a down triangle when it is below, no mark when undecided. No best score is
bolded. Output: report/tab_*.tex.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_ap = argparse.ArgumentParser()
_ap.add_argument("--data", type=Path, default=HERE / "data.json")
_ap.add_argument("--out", type=Path, default=HERE)
_args = _ap.parse_args()
D = json.loads(_args.data.read_text())
raw, k = D["raw"], D["k"]
TASKS = ["mgsm", "humaneval_plus", "mbpp_plus", "ifeval", "aime2025"]
TL = {"mgsm": "MGSM", "humaneval_plus": "HumanEval+", "mbpp_plus": "MBPP+", "ifeval": "IFEval", "aime2025": "AIME 2025"}
GREEN, RED = "CDEBDD", "F6D3C9"
UP, DOWN = r"\textcolor[HTML]{1B7F5A}{$\blacktriangle$}", r"\textcolor[HTML]{B4452A}{$\blacktriangledown$}"
NAMES = {"F1": "FP8 online", "F2": "FP8 offline", "N2": "NVFP4, all quantizable layers", "N5": "NVFP4, routed experts only",
         "G1": "NVFP4, experts, calibrated", "Q5": "OVH Q5\\_K\\_M",
         "N6": "NVFP4, experts of all layers, scale search", "Q4": "Q4\\_K\\_M stand-in"}


def sgn(x: float, nd: int = 1) -> str:
    s = f"{abs(x):.{nd}f}"
    if float(s) == 0:
        return s
    return f"$-${s}" if x < 0 else f"+{s}"


def cell(d: dict) -> str:
    """A difference cell: value, interval, three-state shading."""
    txt = f"{sgn(d['value'])} ({sgn(d['lo'])} to {sgn(d['hi'])})"
    if d["lo"] > 0:
        return f"\\cellcolor[HTML]{{{GREEN}}}{txt} {UP}"
    if d["hi"] < 0:
        return f"\\cellcolor[HTML]{{{RED}}}{txt} {DOWN}"
    return txt


def write(name: str, text: str) -> None:
    _args.out.mkdir(parents=True, exist_ok=True)
    (_args.out / f"tab_{name}.tex").write_text(text)


acc, sizes = raw["acc"], raw["sizes"]

# --------------------------------------------------------------- summary table (arms)
rows = []
rows.append(f"bf16 (reference) & {k['B0.gb']} & {k['B0.bpw']} & 0 & reference & 1.00 \\\\")
ordr = ["F1", "F2", "N5", "G1", "N2", "Q5", "N6", "Q4"]
speed = {"F1": f"{k['bench.F1.load_min']} to {k['bench.F1.load_max']}", "F2": f"{k['bench.F2.load_min']} to {k['bench.F2.load_max']}",
         "N2": f"{k['bench.N2.load_min']} to {k['bench.N2.load_max']}", "N5": f"{k['bench.N5.load_min']} to {k['bench.N5.load_max']}",
         "G1": "not measured", "Q5": "other engine", "N6": f"{k['bench.N6.load_min']} to {k['bench.N6.load_max']}", "Q4": "other engine"}
label = {"F1": "non-inferior", "F2": "non-inferior", "N2": "inconclusive", "N5": "inconclusive", "G1": "inconclusive", "Q5": "not tested", "N6": "inconclusive", "Q4": "not tested"}
for a in ordr:
    d = acc[a]["diff"]
    rows.append(f"{NAMES[a]} & {k[a + '.gb']} & {k[a + '.bpw']} & {cell(d)} & {label[a]} & {speed[a]} \\\\")
write("summary", "\\begin{tabular}{@{}lrrllc@{}}\n\\toprule\nBuild & Size (GB) & Bits per weight & Accuracy change (points) & Test at $-3$ points & Speed under load \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# --------------------------------------------------------------- per-task differences (two tables of three builds)
def tasks_table(arms: list[str]) -> str:
    hdr = " & ".join(NAMES[a].replace("NVFP4, ", "NVFP4 ") for a in arms)
    body = [f"{TL[t]} & " + " & ".join(cell(acc[a]["tasks"][t]["diff"]) for a in arms) + " \\\\" for t in TASKS]
    body.append("\\midrule")
    body.append("Macro-average & " + " & ".join(cell(acc[a]["diff"]) for a in arms) + " \\\\")
    return "\\begin{tabular}{@{}l" + "c" * len(arms) + "@{}}\n\\toprule\nTask & " + hdr + " \\\\\n\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n"


write("tasks_a", tasks_table(["F1", "F2", "Q5"]))
write("tasks_b", tasks_table(["N2", "N5", "G1"]))

# --------------------------------------------------------------- claims (selected rows, verdicts from the probe report)
cl = raw["claims"]
sel = [
    ("A2", "A single head drives each of layers 1, 7 and 23 (heads 23, 26, 40)", "heads 23, 26 and 40"),
    ("A3", "Branch outputs reach about $10^{16}$ in layers 1 and 7, $10^{11}$ in layer 23", "{L1}, {L7}, {L23}"),
    ("A1", "Exactly three Mamba layers carry runaway values", "layers {top3} lead; {gt448} exceed 448"),
    ("A5", "All other Mamba mixer outputs are of order 0.1 to 10", "largest of the rest: {A5max}"),
    ("B1", "Four mixture-of-experts layers spike through the squared ReLU (0, 1, 5, 19)", "layers {moe}"),
    ("C1", "The residual stream stays below 0.69", "largest 0.68"),
    ("C2", "Activation scales exceed the fp16 range by up to ten orders", "up to {orders} orders"),
]
rows = []
for cid, claim, meas in sel:
    v = cl[cid]["verdict"]
    mm = meas.format(L1=k["probe.L1.mixer"], L7=k["probe.L7.mixer"], L23=k["probe.L23.mixer"],
                     top3=k["probe.top3_layers"], gt448=k["probe.mixers_gt448"], moe=k["probe.moe_top4"], orders=k["probe.max_orders"],
                     A5max=k["probe.L2.mixer"])
    rows.append(f"{claim} & {mm} & {v} \\\\")
write("claims", "\\begin{tabularx}{\\textwidth}{@{}Xp{3.6cm}l@{}}\n\\toprule\nOVH statement & Measured here & Verdict \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabularx}\n")

# --------------------------------------------------------------- weight statistics by class
ws = raw["ws"]
cls = [("routed_up", "Routed experts, up projection"), ("routed_down", "Routed experts, down projection"), ("shared_expert", "Shared experts"),
       ("latent_proj", "Latent projections"), ("attn_q", "Attention query"), ("attn_o", "Attention output"), ("mamba_in_proj", "Mamba input projection"),
       ("mamba_out_proj", "Mamba output projection"), ("router", "Router"), ("lm_head", "Output head")]
rows = []
for c, name in cls:
    v = ws[c]
    rows.append(f"{name} & {v['params'] / 1e6:.0f} & {v['nvfp4']:.3f} & {v['fp8']:.3f} & {v['int4_g128']:.3f} & {v['kurt']:.1f} \\\\")
write("weights", "\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\nTensor class & Parameters (M) & NVFP4 & FP8 & INT4 & Excess kurtosis \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# --------------------------------------------------------------- ablation numbers
ab = raw["ablation"]
order = [("all", "All quantizable layers (the N2 build)"), ("experts", "Routed experts only"), ("non-experts", "Everything but the routed experts"),
         ("experts:layers=0-11", "Routed experts, layers 0 to 11"), ("experts:layers=12-23", "Routed experts, layers 12 to 23"),
         ("experts:layers=24-35", "Routed experts, layers 24 to 35"), ("shared-experts", "Shared experts"), ("latent", "Latent projections"),
         ("mamba", "Mamba projections"), ("attention", "Attention projections")]
rows = [f"{lab} & {ab[s]['kl']:.4f} & {ab[s]['ppl']:.3f} & {ab[s]['top1']:.3f} \\\\" for s, lab in order]
write("ablation", "\\begin{tabular}{@{}lrrr@{}}\n\\toprule\nWhat is in NVFP4 & Mean KL & Perplexity ratio & Top-1 agreement \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# --------------------------------------------------------------- harness artifact
first, rerun = raw["rt_first"], raw["rt_rerun"]
rows = []
for a, nm in (("B0a", "bf16, run a"), ("B0b", "bf16, run b"), ("F1", "FP8 online"), ("F2", "FP8 offline"), ("N2", "NVFP4, all quantizable layers")):
    f, r = first[a], rerun[a]
    rows.append(f"{nm} & {f[0]} of {f[1]} & {r[0]} of {r[1]} & 0 of {raw['retry']['wave2'][{'B0a': 'B0a-v2', 'B0b': 'B0b-v2', 'F1': 'F1-v2', 'F2': 'F2-v2', 'N2': 'N2-v2'}[a]]['aime2025']['samples']} \\\\")
write("retry", "\\begin{tabular}{@{}lrrr@{}}\n\\toprule\nAIME samples that needed a second request & First pass & Rerun & Corrected run \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# --------------------------------------------------------------- appendix: absolute accuracy per task
cols = ["F1", "F2", "N2", "N5", "G1", "Q5", "N6", "Q4"]
rows = []
for t in TASKS:
    rows.append(f"{TL[t]} ({k['B0.' + t + '.n']} items) & {k['B0.' + t]} & {k['g.abs.B0g.' + t]} & " + " & ".join(f"{acc[a]['tasks'][t]['x']['value']:.1f}" for a in cols) + " \\\\")
rows.append("\\midrule")
rows.append(f"Macro-average & {k['B0.macro']} & {k['g.macro.B0g']} & " + " & ".join(k[a + ".macro"] for a in cols) + " \\\\")
rows.append("Truncated answers (\\%) & " + f"{k['B0a.trunc']} and {k['B0b.trunc']} & {k['g.trunc']} & " + " & ".join(k[a + ".trunc"] for a in cols) + " \\\\")
write("absolute", "\\begin{tabular}{@{}l" + "r" * (len(cols) + 2) + "@{}}\n\\toprule\nTask & bf16 & B0g & " + " & ".join({"F1": "FP8 on", "F2": "FP8 off", "N2": "N2", "N5": "N5", "G1": "G1", "Q5": "Q5\\_K\\_M", "N6": "N6", "Q4": "Q4"}[a] for a in cols) + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
# --------------------------------------------------------------- this build minus each alternative, per task (shaded by the three-state rule)
C = raw["contr"]
MINUS_LABEL = {"N5": "experts only", "Q5": "OVH Q5\\_K\\_M", "Q4": "Q4\\_K\\_M stand-in", "F1": "FP8 online", "F2": "FP8 offline"}
alts = [(c, MINUS_LABEL[c.split("-")[1]]) for c in raw["minus_cols"]]   # the columns (and so the comparison count of the caption) come from build_data.py
body = [f"{TL[t]} & " + " & ".join(cell(C[a]["tasks"][t]) for a, _ in alts) + " \\\\" for t in TASKS]
body.append("\\midrule")
body.append("Macro-average & " + " & ".join(cell(C[a]["macro"]) for a, _ in alts) + " \\\\")
write("minus", "\\begin{tabular}{@{}l" + "c" * len(alts) + "@{}}\n\\toprule\nTask & " + " & ".join(f"minus {n}" for _, n in alts) + " \\\\\n\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n")

# --------------------------------------------------------------- the three candidates of the smaller-build step (plain-text fidelity)
fid = raw["fid"]
fnames = {"N5": "Experts only, 32 layers (control)", "C1": "Experts of all 36 layers, plain scales", "C2": "Experts of all 36 layers, scale search"}
rows = [f"{fnames[n]} & {fid[n]['bytes'] / 1e9:.2f} & {fid[n]['kl']:.4f} & {fid[n]['ppl']:.3f} & {fid[n]['p99']:.3f} \\\\" for n in ("N5", "C1", "C2")]
write("fidelity", "\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\nBuild & Size (GB) & Mean KL & Perplexity ratio & KL, 99th percentile \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# --------------------------------------------------------------- the Q4 stand-in under two ways of scoring answers that ran out of context
gen = raw["generous"]
trn = lambda a: k[f"tr.{a}.n"]  # noqa: E731
rows = [
    f"Counted as wrong, as for every build & {acc['Q4']['macro']['value']:.1f} & {acc['N6']['macro']['value']:.1f} & {acc['Q5']['macro']['value']:.1f} & {sgn(acc['N6']['macro']['value'] - acc['Q4']['macro']['value'])} & {k['c.N6.Q4.ci']} \\\\",
    f"Credited if the right number had appeared in the last quarter of the reasoning & {gen['Q4']:.1f} & {gen['N6']:.1f} & {gen['Q5']:.1f} & {sgn(gen['N6'] - gen['Q4'])} & {k['s.gen.N6.Q4.ci']} \\\\",
    "\\midrule",
    f"Answers that ran out of context, all five tasks & {trn('Q4')} & {trn('N6')} & {trn('Q5')} & & \\\\",
]
write("q4", "\\begin{tabularx}{\\textwidth}{@{}Xrrrrr@{}}\n\\toprule\nHow answers that ran out of context are scored & Q4 stand-in & New build & Q5\\_K\\_M & New build minus Q4 & 95\\% interval \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabularx}\n")

# --------------------------------------------------------------- appendix: the nature of every build's truncated answers
arms_t = [("B0a", "bf16, run a"), ("B0b", "bf16, run b"), ("F1", "FP8 online"), ("F2", "FP8 offline"), ("N2", "NVFP4, all quantizable layers"), ("N5", "NVFP4, routed experts only"),
          ("G1", "NVFP4, routed experts, calibrated"), ("N6", "New build"), ("Q5", "OVH Q5\\_K\\_M"), ("Q4", "Q4\\_K\\_M stand-in")]
rows = [f"{nm} & {k[f'tr.{a}.n']} & {k[f'tr.{a}.long']} & {k[f'tr.{a}.loops']} & {k[f'tr.{a}.target_late']} \\\\" for a, nm in arms_t]
rows.insert(2, f"bf16 GGUF on llama.cpp (B0g) & {k['g.tr.n']} & {k['g.tr.long']} & {k['g.tr.loops']} & {k['g.tr.target_late']} \\\\")   # after the two vLLM bf16 runs
write("trunc", "\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\nBuild & Cut off & Not flagged as a loop & Loops & Right number in last quarter \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# --------------------------------------------------------------- appendix: each build against bf16 at three margins (unadjusted intervals)
YES = f"\\cellcolor[HTML]{{{GREEN}}}yes"
NO, NO_ALL_BELOW = "no", f"\\cellcolor[HTML]{{{RED}}}no"
mrows = []
for a in ["F1", "F2", "N5", "G1", "N2", "Q5", "N6", "Q4"]:
    mg = raw["margin"][a]
    st = [mg["margins"][m]["state"] for m in ("-2", "-3", "-5")]
    flag = k[f"s.margin.{a}.flag"]
    mrows.append(f"{NAMES[a]} & {cell(mg['macro'])} & " + " & ".join({"pass": YES, "open": NO, "worse": NO_ALL_BELOW}[x] for x in st) + f" & {flag} \\\\")
write("margin", "\\begin{tabular}{@{}lccccl@{}}\n\\toprule\nBuild & Against bf16 (points) & Above $-$2 & Above $-$3 & Above $-$5 & Task below $-$5 \\\\\n\\midrule\n"
      + "\n".join(mrows) + "\n\\bottomrule\n\\end{tabular}\n")

# --------------------------------------------------------------- appendix: B0g, the same-engine control (bf16 GGUF on llama.cpp)
G = raw["g"]
hdr = ["Macro", "MGSM", "HE+", "MBPP+", "IFEval", "AIME"]
ncol = len(hdr) + 1


def acc_row(label: str, arm: str) -> str:
    v = G["absolute"][arm]
    return f"{label} & {v['macro']['value']:.1f} & " + " & ".join(f"{v['tasks'][t]['value']:.1f}" for t in TASKS) + " \\\\"


def diff_row(label: str, c: dict) -> str:
    return f"{label} & {cell(c['macro'])} & " + " & ".join(cell(c["tasks"][t]) for t in TASKS) + " \\\\"


def panel(title: str) -> str:
    return f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\"


body = [panel("Accuracy (\\%)"),
        acc_row("B0g, bf16 GGUF on llama.cpp", "B0g"), acc_row("bf16 on vLLM (two runs pooled)", "B0"), acc_row("OVH Q5\\_K\\_M, llama.cpp", "Q5"), acc_row("Q4\\_K\\_M stand-in, llama.cpp", "Q4"),
        "\\midrule", panel("Engine: llama.cpp bf16 minus vLLM bf16 (points)"),
        diff_row("B0g minus bf16 pooled", G["engine"]["B0g-B0"]), diff_row("B0g minus bf16 run a", G["engine"]["B0g-B0a"]), diff_row("B0g minus bf16 run b", G["engine"]["B0g-B0b"]),
        "\\midrule", panel("File format on llama.cpp (points)"),
        diff_row("Q5\\_K\\_M minus B0g", G["format"]["Q5-B0g"]), diff_row("Q4\\_K\\_M minus B0g", G["format"]["Q4-B0g"]), diff_row("Q4\\_K\\_M minus Q5\\_K\\_M", G["format"]["Q4-Q5"])]
write("b0g", "\\begin{tabular}{@{}l" + "c" * len(hdr) + "@{}}\n\\toprule\n & " + " & ".join(hdr) + " \\\\\n\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n")

# --------------------------------------------------------------- appendix: llama.cpp sweeps
rows = [f"{c} & {k[f'gg.q4.{c}']} ({k[f'gg.q4.{c}.n']}) & {k[f'gg.q5.{c}']} ({k[f'gg.q5.{c}.n']}) \\\\" for c in (1, 4, 12)]
write("gguf", "\\begin{tabular}{@{}rrr@{}}\n\\toprule\nConcurrent requests & Q4\\_K\\_M stand-in & OVH Q5\\_K\\_M \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
print("tables written")
