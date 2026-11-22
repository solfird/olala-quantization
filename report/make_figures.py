"""Draw the figures of the report from report/data.json (run build_data.py first).

    uv run --no-project --python 3.12 --with matplotlib==3.11.2 --with numpy==2.5.5 python report/make_figures.py [--data FILE] [--out DIR]

(matplotlib 3.11.2 and numpy 2.5.5 are the versions that drew the figures of the report; the sources of the study did not pin them, this
repository does. Defaults: report/data.json, and the figures go to figures/.)

Palette and font follow the lab's report style: blue #2a78d6, orange #eb6834, aqua #1baf7a, yellow #eda100,
greys #52514e and #8d8c87, grid #d9d8d4, STIX fonts. Output: fig_*.pdf and fig_*.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
_ap = argparse.ArgumentParser()
_ap.add_argument("--data", type=Path, default=HERE / "data.json")
_ap.add_argument("--out", type=Path, default=HERE.parent / "figures")
_args = _ap.parse_args()
OUT = _args.out
OUT.mkdir(parents=True, exist_ok=True)
D = json.loads(_args.data.read_text())
raw, k = D["raw"], D["k"]

BLUE, ORANGE, AQUA, YELLOW, GREY, LGREY, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#52514e", "#8d8c87", "#d9d8d4"
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9, "axes.edgecolor": GREY, "axes.labelcolor": GREY,
    "xtick.color": GREY, "ytick.color": GREY, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight",
})


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------- 1. accuracy against size (teaser)
acc, sizes = raw["acc"], raw["sizes"]
names = {"N6": "NVFP4, experts of all layers, scale search (new)", "Q4": "Q4_K_M stand-in (llama.cpp)", "F1": "FP8 online", "F2": "FP8 offline", "N2": "NVFP4, all quantizable layers", "N5": "NVFP4, routed experts only",
         "G1": "NVFP4 experts, calibrated", "Q5": "OVH Q5_K_M (llama.cpp)"}
colors = {"F1": BLUE, "F2": BLUE, "N2": ORANGE, "N5": AQUA, "G1": YELLOW, "Q5": GREY, "N6": "#0f7a52", "Q4": "#8d8c87"}
order_t = ["F1", "F2", "N5", "G1", "N2", "Q5", "N6", "Q4"]   # largest file first; evenly spaced rows, so nothing overlaps
labels = {"N6": "NVFP4, experts of all layers, new", "Q4": "Q4_K_M stand-in (llama.cpp)", "F1": "FP8 online", "F2": "FP8 offline", "N5": "NVFP4, routed experts only", "G1": "NVFP4, experts, calibrated",
          "N2": "NVFP4, all quantizable layers", "Q5": "OVH Q5_K_M (llama.cpp)"}
fig, ax = plt.subplots(figsize=(6.4, 3.9))
ys = list(range(len(order_t)))[::-1]
ax.axvline(0, color=GREY, lw=0.9)
ax.axvline(-3, color=ORANGE, lw=0.9, ls="--")
ax.text(-3.08, ys[0] + 0.62, "margin: 3 points", color=ORANGE, ha="right", va="center", fontsize=8)
ax.text(0.08, ys[0] + 0.62, "bf16 reference", color=GREY, ha="left", va="center", fontsize=8)
for y, a in zip(ys, order_t):
    d = acc[a]["diff"]
    ax.errorbar(d["value"], y, xerr=[[d["value"] - d["lo"]], [d["hi"] - d["value"]]], fmt="o" if a not in ("Q5", "Q4") else "s", color=colors[a],
                capsize=3, ms=6, lw=1.5)
ax.set_yticks(ys)
ax.set_yticklabels([f"{labels[a]}  ({sizes[a] / 1e9:.2f} GB)" for a in order_t])
ax.set_xlabel("accuracy change against bf16 (points), 95% interval")
ax.set_xlim(-6.2, 2.4)
ax.set_ylim(ys[-1] - 0.6, ys[0] + 1.0)
ax.grid(axis="y", visible=False)
save(fig, "fig_teaser")

# ----------------------------------------------------------------------------- 2. activation magnitude by layer
L = raw["probe_layers"]
fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True)
import math  # noqa: E402

for ax, key, title in ((axes[0], "mixer", "Output of the sequence-mixing branch (Mamba or attention)"), (axes[1], "moe", "Output of the mixture-of-experts branch")):
    for l in L:
        v = max(l[key], 1e-3)
        c = BLUE if l["type"] == "M" else LGREY
        ax.vlines(l["layer"], 1e-2, v, color=c, lw=3.2 if l["type"] == "M" else 2.4)
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 1e18)
    ax.axhline(448, color=ORANGE, lw=0.9, ls="--")
    ax.axhline(65504, color=YELLOW, lw=0.9, ls="--")
    ax.set_title(title, loc="left", fontsize=9, color=GREY)
    ax.set_ylabel("largest absolute value")
axes[0].text(35.6, 600, "FP8 range (448)", color=ORANGE, ha="right", va="bottom", fontsize=8)
axes[0].text(35.6, 8e4, "fp16 range (65,504)", color=YELLOW, ha="right", va="bottom", fontsize=8)
for lay in (1, 7, 23):
    axes[0].text(lay, next(l for l in L if l["layer"] == lay)["mixer"] * 2.5, str(lay), ha="center", fontsize=8, color=GREY)
for lay in (0, 1, 5, 19):
    axes[1].text(lay, next(l for l in L if l["layer"] == lay)["moe"] * 2.5, str(lay), ha="center", fontsize=8, color=GREY)
axes[1].set_xlabel("layer (0 is the first block; gray bars are attention layers)")
axes[1].set_xlim(-1, 36)
save(fig, "fig_probe")

# ----------------------------------------------------------------------------- 3. ablation
ab = raw["ablation"]
order = [("all", "all quantized (N2)"), ("experts", "routed experts"), ("non-experts", "everything but routed experts"),
         ("experts:layers=0-11", "experts, layers 0 to 11"), ("experts:layers=12-23", "experts, layers 12 to 23"),
         ("experts:layers=24-35", "experts, layers 24 to 35"), ("shared-experts", "shared experts"), ("latent", "latent projections"),
         ("mamba", "Mamba projections"), ("attention", "attention projections")]
fig, ax = plt.subplots(figsize=(6.4, 3.3))
ys = list(range(len(order)))[::-1]
cols = [ORANGE, AQUA, BLUE] + [AQUA] * 3 + [BLUE] * 4
for y, (spec, label), c in zip(ys, order, cols):
    v = ab[spec]
    ax.barh(y, v["kl"], color=c, height=0.62)
    ax.text(v["kl"] + 0.0012, y, f"{v['kl']:.4f}", va="center", fontsize=8, color=GREY)
ax.set_yticks(ys)
ax.set_yticklabels([o[1] for o in order])
ax.axvline(raw["ab_noise"], color=GREY, lw=0.8, ls=":")
ax.text(raw["ab_noise"] + 0.0008, ys[-1] - 0.75, "noise floor", fontsize=7.5, color=GREY, va="top")
ax.set_xlabel("mean KL divergence from bf16 on plain text (nats)")
ax.set_xlim(0, 0.085)
ax.grid(axis="y", visible=False)
save(fig, "fig_ablation")

# ----------------------------------------------------------------------------- 4. throughput ratio against bf16
bench, ref = raw["bench"], raw["bench_ref"]
pts = [("short", c) for c in (1, 4, 16, 32, 64, 128)]
fig, ax = plt.subplots(figsize=(6.4, 3.0))
for a, c in (("F1", BLUE), ("F2", "#5a9be6"), ("N2", ORANGE), ("N5", AQUA), ("N6", "#0f7a52")):
    xs, ys = [], []
    for w, cc in pts:
        p = f"{w}:{cc}"
        if p in bench[a]:
            xs.append(cc)
            ys.append(bench[a][p]["median"] / ref[p])
    ax.plot(xs, ys, marker="o", ms=4, color=c, lw=1.4, label={"F1": "FP8 online", "F2": "FP8 offline", "N2": "NVFP4 all", "N5": "NVFP4 experts only", "N6": "NVFP4 new build"}[a])
ax.axhline(1, color=GREY, lw=0.8)
ax.set_xscale("log", base=2)
ax.set_xticks([c for _, c in pts])
ax.set_xticklabels([str(c) for _, c in pts])
ax.set_xlabel("concurrent requests")
ax.set_ylabel("output tokens per second\nrelative to bf16")
ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
ax.set_ylim(0.85, 1.5)
save(fig, "fig_speed")
# ----------------------------------------------------------------------------- 0. the layer map (architecture)
L = raw["probe_layers"]
fig, ax = plt.subplots(figsize=(6.4, 1.7))
for l in L:
    x = l["layer"]
    is_m = l["type"] == "M"
    ax.add_patch(plt.Rectangle((x, 1.05), 0.86, 0.9, color=BLUE if is_m else LGREY, lw=0))
    ax.add_patch(plt.Rectangle((x, 0.1), 0.86, 0.75, color=AQUA, lw=0))
for lay in (1, 7, 23):
    ax.add_patch(plt.Rectangle((lay - 0.06, 1.0), 0.98, 1.0, fill=False, ec=ORANGE, lw=1.6))
for lay in range(0, 36, 5):
    ax.text(lay + 0.43, -0.12, str(lay), ha="center", va="top", fontsize=7.5, color=GREY)
ax.text(-0.9, 1.5, "mixer", ha="right", va="center", fontsize=8, color=GREY)
ax.text(-0.9, 0.48, "experts", ha="right", va="center", fontsize=8, color=GREY)
ax.set_xlim(-6.2, 36.5)
ax.set_ylim(-0.7, 2.2)
ax.axis("off")
handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=LGREY),
           plt.Rectangle((0, 0), 1, 1, color=AQUA), plt.Rectangle((0, 0), 1, 1, fill=False, ec=ORANGE, lw=1.6)]
ax.legend(handles, ["Mamba-3 layer", "attention layer", "mixture of experts", "runaway Mamba layer (Step 2)"], frameon=False, fontsize=7.5,
          ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.38), handlelength=1.2, columnspacing=1.2)
save(fig, "fig_arch")
print("figures written")
