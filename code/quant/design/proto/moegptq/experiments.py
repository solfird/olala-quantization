"""Experiments of the design study. Prints markdown tables and writes JSON under ``results/``.

Nothing here is tuned: the methods, the grids of N and the seeds are fixed in the code below before the
first run, and every number printed comes from this script. Run with ``uv run python -m moegptq.experiments``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch

from moegptq.pipeline import Method, evaluate, quantize_layer
from moegptq.schemes import Int4Scheme, Nvfp4Scheme
from moegptq.toy import ToyConfig, make_layer, sample_tokens

SEEDS = (0, 1, 2, 3, 4)
N_TEST = 8192
SCHEMES: dict[str, Callable[[], Nvfp4Scheme | Int4Scheme]] = {"nvfp4": Nvfp4Scheme, "int4": Int4Scheme}
RESULTS = Path(__file__).resolve().parent.parent / "results"

CONFIGS: dict[str, ToyConfig] = {
    "weak routing (random router, tokens one population)": ToyConfig(),
    "specialised routing (16 token clusters, expert c is home of cluster c)": ToyConfig(
        n_clusters=16, rare_experts=0, bias_sigma=0.0
    ),
    "strong specialisation (as above, and each cluster has its own covariance subspace)": ToyConfig(
        n_clusters=16, rare_experts=0, bias_sigma=0.0, cluster_rotate=True
    ),
}

E1_METHODS = [
    Method("rtn", gptq=False),
    Method("gptq dynamic scales, routed (original GPTQ)", dynamic=True),
    Method("gptq static scales, routed", dynamic=False),
    Method("gptq static + act-order, routed", dynamic=False, act_order=True),
    Method(
        "gptq static + act-order, ALL tokens (llm-compressor style)",
        dynamic=False,
        act_order=True,
        hessian="all",
    ),
    Method(
        "gptq static + act-order, BLEND (routed + 1 x dim pseudo-tokens of the all-token moment)",
        dynamic=False,
        act_order=True,
        hessian="blend",
        blend_pseudo=1.0,
    ),
    Method("gptq dynamic, routed, gate-weighted", dynamic=True, gate_weighted=True),
    Method("gptq dynamic, routed, true-sequential", dynamic=True, true_sequential=True),
]


def _mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def run_e1(cfg: ToyConfig, n_cals: tuple[int, ...]) -> dict:
    """Layer-output error of every method for each calibration size and grid."""
    out: dict = {}
    for scheme_name, factory in SCHEMES.items():
        for n_cal in n_cals:
            cell: dict[str, list[float]] = {m.name: [] for m in E1_METHODS}
            counts_min: list[int] = []
            counts_med: list[float] = []
            for seed in SEEDS:
                layer = make_layer(cfg, seed)
                cal = sample_tokens(layer, n_cal, seed=1000 + seed)
                test = sample_tokens(layer, N_TEST, seed=2000 + seed)
                for m in E1_METHODS:
                    res = quantize_layer(layer, cal, factory, m)
                    cell[m.name].append(evaluate(layer, res.w_up, res.w_down, test)["rel_err"])
                    if m.name == "rtn":
                        c = torch.bincount(cal.idx.flatten(), minlength=cfg.n_experts)
                        counts_min.append(int(c.min()))
                        counts_med.append(float(c.double().median()))
            out[f"{scheme_name}|{n_cal}"] = {
                "rel_err": {k: _mean_sd(v) for k, v in cell.items()},
                "per_seed": cell,
                "min_tokens_per_expert": counts_min,
                "median_tokens_per_expert": counts_med,
            }
    return out


def run_e2(cfg: ToyConfig, n_cals: tuple[int, ...]) -> dict:
    """Per-expert error ratio (GPTQ / RTN) binned by the number of routed calibration tokens."""
    variants = [
        ("gptq damp 0.001", Method("d001", percdamp=0.001)),
        ("gptq damp 0.01", Method("d01", percdamp=0.01)),
        ("gptq damp 0.1", Method("d1", percdamp=0.1)),
        ("gptq damp 1.0", Method("d10", percdamp=1.0)),
        ("D1 routed (static + act-order) damp 0.01", Method("r", dynamic=False, act_order=True)),
        (
            "D1 all-token (static + act-order) damp 0.01",
            Method("a", dynamic=False, act_order=True, hessian="all"),
        ),
        (
            "D1 blend (static + act-order) damp 0.01",
            Method("b", dynamic=False, act_order=True, hessian="blend", blend_pseudo=1.0),
        ),
    ]
    bins = [(0, 16), (16, 64), (64, 256), (256, 1024), (1024, 10**9)]
    out: dict = {}
    for scheme_name, factory in SCHEMES.items():
        ratios: dict[str, dict[tuple[int, int], list[float]]] = {
            label: {b: [] for b in bins} for label, _ in variants
        }
        for n_cal in n_cals:
            for seed in SEEDS:
                layer = make_layer(cfg, seed)
                cal = sample_tokens(layer, n_cal, seed=1000 + seed)
                test = sample_tokens(layer, N_TEST, seed=2000 + seed)
                counts = torch.bincount(cal.idx.flatten(), minlength=cfg.n_experts).tolist()
                r0 = quantize_layer(layer, cal, factory, Method("rtn", gptq=False))
                e0 = evaluate(layer, r0.w_up, r0.w_down, test)["expert_rel_err"]
                for label, m in variants:
                    r = quantize_layer(layer, cal, factory, m)
                    e = evaluate(layer, r.w_up, r.w_down, test)["expert_rel_err"]
                    for ex in range(cfg.n_experts):
                        if e0[ex] != e0[ex] or e0[ex] == 0.0:
                            continue
                        for b in bins:
                            if b[0] <= counts[ex] < b[1]:
                                ratios[label][b].append(e[ex] / e0[ex])
        out[scheme_name] = {
            label: {
                f"{b[0]}-{b[1] if b[1] < 10**9 else 'inf'}": (
                    statistics.fmean(v) if v else None,
                    len(v),
                )
                for b, v in per_bin.items()
            }
            for label, per_bin in ratios.items()
        }
    return out


def run_e3(base: ToyConfig, n_cal: int) -> dict:
    """One huge input channel: persistent (all tokens) and sporadic (5% of tokens)."""
    scenarios = [
        ("none", {}),
        ("persistent x30", {"spike_channel": 0, "spike_factor": 30.0}),
        ("persistent x1000", {"spike_channel": 0, "spike_factor": 1000.0}),
        ("sporadic 5% x1000", {"spike_channel": 0, "spike_factor": 1000.0, "spike_mode": "sporadic"}),
    ]
    methods = [
        Method("rtn", gptq=False),
        Method("gptq mean-damping 0.01"),
        Method("gptq diag-damping 0.01", damp_mode="diag"),
        Method("gptq mean-damping + token clip 3x median", token_clip=3.0),
        Method("gptq diag-damping + token clip 3x median", damp_mode="diag", token_clip=3.0),
    ]
    out: dict = {}
    for scheme_name, factory in SCHEMES.items():
        for label, kw in scenarios:
            cfg = ToyConfig(**{**base.__dict__, **kw})
            cell: dict[str, dict[str, list[float]]] = {m.name: {"all": [], "plain": []} for m in methods}
            for seed in SEEDS:
                layer = make_layer(cfg, seed)
                cal = sample_tokens(layer, n_cal, seed=1000 + seed)
                test = sample_tokens(layer, N_TEST, seed=2000 + seed)
                for m in methods:
                    res = quantize_layer(layer, cal, factory, m)
                    ev = evaluate(layer, res.w_up, res.w_down, test)
                    cell[m.name]["all"].append(ev["rel_err"])
                    cell[m.name]["plain"].append(ev["rel_err_plain"])
            out[f"{scheme_name}|{label}"] = {
                k: {kk: _mean_sd(vv) for kk, vv in v.items()} for k, v in cell.items()
            }
    return out


def run_e4(seeds: tuple[int, ...] = SEEDS, n_cal: int = 8192) -> list[dict]:
    """Per-diagonal damping on INT4: which expert blows up and how many tokens it saw."""
    rows = []
    for seed in seeds:
        layer = make_layer(ToyConfig(), seed)
        cal = sample_tokens(layer, n_cal, seed=1000 + seed)
        test = sample_tokens(layer, N_TEST, seed=2000 + seed)
        res = quantize_layer(layer, cal, Int4Scheme, Method("diag", damp_mode="diag"))
        ev = evaluate(layer, res.w_up, res.w_down, test)
        per = [x if x == x else -1.0 for x in ev["expert_rel_err"]]
        worst = max(range(len(per)), key=per.__getitem__)
        rows.append(
            {
                "seed": seed,
                "layer_rel_err": ev["rel_err"],
                "worst_expert": worst,
                "worst_expert_tokens": int((cal.idx == worst).any(dim=1).sum()),
                "worst_expert_rel_err": per[worst],
            }
        )
    return rows


def requant_changed_fraction(q: torch.Tensor) -> tuple[float, float]:
    """Fraction of values (and the largest relative change) when ``q`` is re-quantized by ``olala_quant``.

    A change above 1e-5 of the tensor amax counts; float32 ulp differences do not.
    """
    from olala_quant.nvfp4 import dequantize_nvfp4, quantize_nvfp4

    r = quantize_nvfp4(q)
    rq = dequantize_nvfp4(r.packed, r.block_scale, r.global_scale)
    rel = (rq - q).abs() / q.abs().amax()
    return float((rel > 1e-5).double().mean()), float(rel.max())


def run_e5() -> dict:
    """Pitfall: pass GPTQ output through the data-free NVFP4 quantizer and see what changes."""
    from moegptq.gptq import gptq_quantize

    out: dict = {}
    for label, dyn, ao in (
        ("dynamic", True, False),
        ("static", False, False),
        ("static+act-order", False, True),
    ):
        fracs, maxes = [], []
        for seed in SEEDS:
            layer = make_layer(ToyConfig(), seed)
            cal = sample_tokens(layer, 8192, seed=1000 + seed)
            rows = torch.nonzero((cal.idx == 3).any(dim=1)).flatten()
            xe = cal.x[rows]
            h = xe.T @ xe
            q, _ = gptq_quantize(layer.w_up[3], h, Nvfp4Scheme(), dynamic=dyn, act_order=ao)
            f, m = requant_changed_fraction(q)
            fracs.append(f)
            maxes.append(m)
        out[label] = {"fraction_changed": _mean_sd(fracs), "max_rel_change": max(maxes)}
    return out


def _fmt(ms: tuple[float, float]) -> str:
    return f"{ms[0]:.4g} ± {ms[1]:.2g}"


def print_e1(res: dict) -> None:
    """Markdown table of E1."""
    print(
        "\n### E1: layer-output relative squared error on held-out routed tokens (mean ± sd over 5 seeds)\n"
    )
    for key, cell in res.items():
        scheme, n_cal = key.split("|")
        lo = min(cell["min_tokens_per_expert"])
        med = statistics.median(cell["median_tokens_per_expert"])
        head = f"{scheme}, N_cal = {n_cal} tokens"
        print(f"\n{head} (min / median routed tokens per expert over seeds: {lo} / {med:.0f})\n")
        print("| method | rel. error | ratio to RTN |")
        print("|---|---|---|")
        rtn = cell["rel_err"]["rtn"][0]
        for name, ms in cell["rel_err"].items():
            print(f"| {name} | {_fmt(ms)} | {ms[0] / rtn:.3f} |")


def print_e2(res: dict) -> None:
    """Markdown table of E2."""
    print(
        "\n### E2: per-expert error ratio GPTQ / RTN (below 1 = GPTQ better), by routed calibration tokens\n"
    )
    for scheme, per_variant in res.items():
        print(f"\n{scheme}\n")
        bins = list(next(iter(per_variant.values())).keys())
        print("| variant | " + " | ".join(f"n_e {b}" for b in bins) + " |")
        print("|---|" + "---|" * len(bins))
        for label, per_bin in per_variant.items():
            cells = []
            for b in bins:
                mean, n = per_bin[b]
                cells.append("-" if mean is None else f"{mean:.3f} (n={n})")
            print(f"| {label} | " + " | ".join(cells) + " |")


def print_e3(res: dict) -> None:
    """Markdown table of E3."""
    print("\n### E3: one huge input channel, N_cal = 8192 (rel. error: all test tokens / non-spike tokens)\n")
    last = None
    for key, cell in res.items():
        scheme, scenario = key.split("|")
        if scheme != last:
            print(f"\n{scheme}\n")
            last = scheme
        print(f"\nscenario: {scenario}\n")
        print("| method | all tokens | non-spike tokens |")
        print("|---|---|---|")
        for name, v in cell.items():
            print(f"| {name} | {_fmt(v['all'])} | {_fmt(v['plain'])} |")


def render(results_dir: Path) -> None:
    """Print every table from the JSON files saved by a previous run."""
    for slug, label in (
        ("weak", "weak routing"),
        ("special", "specialised routing"),
        ("strong", "strong specialisation"),
    ):
        for kind, printer in (("e1", print_e1), ("e2", print_e2)):
            f = results_dir / f"{kind}_{slug}.json"
            if f.exists():
                print(f"\n## Configuration: {label}")
                printer(json.loads(f.read_text()))
    f = results_dir / "e3.json"
    if f.exists():
        print_e3(json.loads(f.read_text()))


def main() -> None:
    """Run the requested experiments."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["e1", "e2", "e3", "e4", "e5", "all"], default="all")
    ap.add_argument("--render", action="store_true", help="print the tables of the saved JSON files and exit")
    args = ap.parse_args()
    if args.render:
        render(RESULTS)
        return
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    for label, cfg in CONFIGS.items():
        slug = {"weak": "weak", "specialised": "special", "strong": "strong"}[label.split()[0]]
        if args.exp in ("e1", "all"):
            r1 = run_e1(cfg, (512, 2048, 8192))
            (RESULTS / f"e1_{slug}.json").write_text(json.dumps(r1, indent=1))
            print(f"\n## Configuration: {label}")
            print_e1(r1)
        if args.exp in ("e2", "all"):
            r2 = run_e2(cfg, (512, 2048, 8192))
            (RESULTS / f"e2_{slug}.json").write_text(json.dumps(r2, indent=1))
            print(f"\n## Configuration: {label}")
            print_e2(r2)
    if args.exp in ("e3", "all"):
        r3 = run_e3(ToyConfig(), 8192)
        (RESULTS / "e3.json").write_text(json.dumps(r3, indent=1))
        print_e3(r3)
    if args.exp in ("e4", "all"):
        r4 = run_e4()
        (RESULTS / "e4.json").write_text(json.dumps(r4, indent=1))
        print("\n### E4: INT4 with per-diagonal damping (weak-routing toy, N_cal = 8192)\n")
        print("| seed | layer rel. error | worst expert | its routed calibration tokens | its rel. error |")
        print("|---|---|---|---|---|")
        for r in r4:
            print(
                f"| {r['seed']} | {r['layer_rel_err']:.4g} | {r['worst_expert']} | "
                f"{r['worst_expert_tokens']} | {r['worst_expert_rel_err']:.4g} |"
            )
    if args.exp in ("e5", "all"):
        r5 = run_e5()
        (RESULTS / "e5.json").write_text(json.dumps(r5, indent=1))
        print("\n### E5: GPTQ output re-quantized by the data-free quantizer (expert 3 up-proj, 5 seeds)\n")
        print("| solver variant | fraction of values changed (> 1e-5 of amax) | largest relative change |")
        print("|---|---|---|")
        for k, v in r5.items():
            print(f"| {k} | {_fmt(v['fraction_changed'])} | {v['max_rel_change']:.3g} |")
    print(f"\n(total {time.time() - t0:.0f} s on CPU)")


if __name__ == "__main__":
    main()
