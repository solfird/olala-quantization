"""OVH's Appendix C claims, checked one by one against a probe result, and the markdown report.

Layer numbers are 0-based (as in the OVH report: its attention layers are L4 ... L34). Each claim row states what OVH
claimed, what the probe measured, the RULE used to compare them and a verdict. The verdict vocabulary is deliberately
weak: ``consistent`` (the measured value satisfies the stated rule), ``partly consistent``, ``not consistent`` and
``not judged`` (the claim has no checkable definition here, the numbers are printed anyway). A verdict only speaks
for this prompt set, this sequence length and the pure-PyTorch forward path (see the caveats section of the report).
"""

from __future__ import annotations

import math
from typing import Any

CONSISTENT, PARTLY, NOT_CONSISTENT, NOT_JUDGED = "consistent", "partly consistent", "not consistent", "not judged"

# what OVH claims (Appendix C, pages 60-65 of the report), 0-based layer numbers
OVH_MAMBA_LAYERS = (1, 7, 23)
OVH_HEADS = {1: 23, 7: 26, 23: 40}
OVH_MIXER_MAX = {1: 5e15, 7: 5e15, 23: 1.5e11}
OVH_MIXER_RANGE = (1e11, 1.8e16)
OVH_SIBLING_SCALE = 0.1
OVH_ERASURE = (0.96, 1.0)
OVH_GROWTH = {1: 1.9, 7: 1.0, 23: 0.36}
OVH_MOE_LAYERS = (0, 1, 5, 19)
OVH_MOE_RANGE = (1e7, 3e8)
OVH_UPPROJ_GAIN = 30.0
OVH_L19_FIRE = (0.04, 0.07)
OVH_RESIDUAL_MAX = 0.69
OVH_FP16_ORDERS = 10.0
OVH_ATTENTION_LAYERS = (4, 9, 14, 19, 24, 29, 34)
OVH_ABLATIONS = {  # (label, claimed delta NLL on wikitext)
    "zero_mixer_L1": ("zeroing the whole layer-1 mixer", -0.009),
    "zero_head_L1": ("zeroing the integrator head of layer 1", 0.040),
    "zero_head_L23": ("zeroing the integrator head of layer 23", 0.0),
}
ABLATION_SMALL = 0.1  # OVH's own control surgery costs up to +0.111; |dNLL| <= 0.1 is called small here
EXPONENT_TOL = 0.5
ORDER_TOL = 1.0  # one decade
BANNED = ("proves", "proven", "confirmed", "verified", "refuted", "disproves", "establishes")


def fmt(x: Any, digits: int = 3) -> str:
    """Compact number formatting (``1.23e+15``, ``0.456``, ``n/a``)."""
    if x is None:
        return "n/a"
    if isinstance(x, str):
        return x
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return f"{x:.{digits}g}"


class View:
    """Convenience accessors over a probe result dict."""

    def __init__(self, result: dict[str, Any]) -> None:
        """Index the layers of ``result`` by their 0-based number."""
        self.r = result
        self.meta = result["meta"]
        self.rows = {row["layer"]: row for row in result["layers"]}

    def site(self, layer: int, kind: str) -> dict[str, Any] | None:
        """The pooled numbers of one site, or ``None``."""
        row = self.rows.get(layer)
        return row["sites"].get(kind) if row else None

    def mx(self, layer: int, kind: str) -> float | None:
        """Maximum ``|x|`` of a site."""
        s = self.site(layer, kind)
        return s["max_abs"] if s else None

    def heads(self, layer: int) -> dict[str, Any] | None:
        """Per-head numbers of a Mamba layer."""
        row = self.rows.get(layer)
        return row["heads"] if row else None

    def mamba_layers(self) -> list[int]:
        """Layers of type M."""
        return sorted(layer for layer, row in self.rows.items() if row["type"] == "M")

    def top_layers(self, kind: str, k: int, among: list[int] | None = None) -> list[int]:
        """The ``k`` layers with the largest maximum of ``kind`` (ties broken by layer number)."""
        pool = among if among is not None else sorted(self.rows)
        ranked = sorted(
            (layer for layer in pool if self.mx(layer, kind) is not None), key=lambda x: (-self.mx(x, kind), x)
        )
        return sorted(ranked[:k])

    def ablation(self, layer: int, branch: str, mode: str, head: int | None = None) -> dict[str, Any] | None:
        """A measured ablation row, or ``None`` when it was not run."""
        for a in self.r.get("ablations", []):
            if (
                a["layer"] == layer
                and a["branch"] == branch
                and a["mode"] == mode
                and (head is None or a["head"] == head)
            ):
                return a
        return None


def _within_decades(measured: float | None, claimed: float, tol: float = ORDER_TOL) -> bool:
    return measured is not None and measured > 0 and abs(math.log10(measured / claimed)) <= tol


def _tally(flags: list[bool]) -> str:
    if flags and all(flags):
        return CONSISTENT
    return PARTLY if any(flags) else NOT_CONSISTENT


def claim_rows(result: dict[str, Any]) -> list[dict[str, str]]:
    """The claim-by-claim comparison: one dict per claim with the strings of the table's columns."""
    v = View(result)
    thr = v.meta["spike_threshold"]
    rows: list[dict[str, str]] = []

    def add(
        cid: str, claim: str, claimed: str, measured: str, rule: str, verdict: str, needs: tuple[int, ...] = ()
    ) -> None:
        absent = [layer for layer in needs if layer not in v.rows]
        if absent:
            measured += f" (layers {absent} do not exist in this model)"
            verdict = NOT_JUDGED
        rows.append(
            {"id": cid, "claim": claim, "claimed": claimed, "measured": measured, "rule": rule, "verdict": verdict}
        )

    mamba = v.mamba_layers()
    top3 = v.top_layers("mixer_out", 3, mamba)
    flagged = sorted(layer for layer in mamba if (v.mx(layer, "mixer_out") or 0) > thr)
    k_ok = top3 == sorted(OVH_MAMBA_LAYERS) and flagged == sorted(OVH_MAMBA_LAYERS)
    ranked = sorted(mamba, key=lambda x: -(v.mx(x, "mixer_out") or 0))
    gap = (
        (v.mx(ranked[2], "mixer_out") / v.mx(ranked[3], "mixer_out"))
        if len(ranked) > 3 and v.mx(ranked[3], "mixer_out")
        else None
    )
    inter = set(top3) & set(OVH_MAMBA_LAYERS)
    add(
        "A1",
        "Three Mamba-3 layers emit runaway branch outputs (App. C.1; 12)",
        f"layers {list(OVH_MAMBA_LAYERS)}",
        f"top-3 Mamba layers by max abs of the mixer branch: {top3}; layers above {fmt(thr)}: {flagged}; "
        f"3rd/4th ratio {fmt(gap)}",
        "the top-3 set AND the set above the spike threshold equal OVH's",
        CONSISTENT if k_ok else (PARTLY if inter else NOT_CONSISTENT),
    )
    one_based = sorted(x - 1 for x in OVH_MAMBA_LAYERS)
    add(
        "A1b",
        "Layer-index convention (not stated in the report text; Table 50 uses L4 ... L34 for attention)",
        "0-based; attention layers 4, 9, ..., 34",
        f"attention layers of this model: {v.meta['attention_layers_0based']}; measured top-3 Mamba layers {top3} "
        f"(a 1-based reading of OVH's numbers would be 0-based layers {one_based}, which {'match' if top3 == one_based else 'do not match'})",
        "the model's attention layers equal OVH's list; the measured top-3 are compared with both readings",
        CONSISTENT
        if tuple(v.meta["attention_layers_0based"]) == OVH_ATTENTION_LAYERS and top3 != one_based
        else (PARTLY if top3 == one_based else NOT_CONSISTENT),
    )
    hd = {layer: (v.heads(layer) or {}).get("top_head") for layer in OVH_MAMBA_LAYERS}
    dom = {layer: (v.heads(layer) or {}).get("dominance_ratio") for layer in OVH_MAMBA_LAYERS}
    add(
        "A2",
        "In each of these layers a single SSM head is responsible (heads 23, 26, 40)",
        f"{OVH_HEADS}",
        f"largest head per layer: {hd}; largest/second-largest head max: "
        + ", ".join(f"L{k} {fmt(x)}" for k, x in dom.items()),
        "the largest head of each claimed layer equals OVH's head index",
        _tally([hd[layer] == OVH_HEADS[layer] for layer in OVH_MAMBA_LAYERS]),
        needs=OVH_MAMBA_LAYERS,
    )
    meas = {layer: v.mx(layer, "mixer_out") for layer in OVH_MAMBA_LAYERS}
    add(
        "A3",
        "Branch output magnitudes: layers 1 and 7 up to ~5e15, layer 23 up to ~1.5e11 (overall 1e11 to 1.8e16)",
        ", ".join(f"L{layer} ~{fmt(OVH_MIXER_MAX[layer])}" for layer in OVH_MAMBA_LAYERS),
        ", ".join(f"L{layer} {fmt(meas[layer])}" for layer in OVH_MAMBA_LAYERS)
        + f" (longest prompt {v.meta['max_tokens_in_a_prompt']} tokens; OVH used 512)",
        "measured max within one decade of the claimed maximum",
        _tally([_within_decades(meas[layer], OVH_MIXER_MAX[layer]) for layer in OVH_MAMBA_LAYERS]),
        needs=OVH_MAMBA_LAYERS,
    )
    sib = {layer: (v.heads(layer) or {}) for layer in OVH_MAMBA_LAYERS}
    add(
        "A4",
        "Sibling heads of the dominant head stay at O(0.1)",
        "O(0.1)",
        ", ".join(
            f"L{k} median sibling RMS {fmt(s.get('median_sibling_rms'))}, max {fmt(s.get('second_head_max'))}"
            for k, s in sib.items()
        ),
        "median sibling-head RMS within [0.01, 1]",
        _tally([0.01 <= (s.get("median_sibling_rms") or 0) <= 1.0 for s in sib.values()]),
        needs=OVH_MAMBA_LAYERS,
    )
    others = [layer for layer in mamba if layer not in OVH_MAMBA_LAYERS]
    omax = max((v.mx(layer, "mixer_out") or 0 for layer in others), default=None)
    add(
        "A5",
        "All other Mamba mixer outputs are O(0.1-10)",
        "O(0.1-10)",
        f"largest mixer-branch max abs among the {len(others)} other Mamba layers: {fmt(omax)}",
        "that maximum is at most 100",
        NOT_JUDGED if omax is None else (CONSISTENT if omax <= 100 else NOT_CONSISTENT),
    )
    er = {layer: v.rows[layer]["mixer_erasure"] for layer in OVH_MAMBA_LAYERS if layer in v.rows}
    add(
        "A6",
        "Removing the other 47 heads leaves 96-100% of the layer output bit-identical (bf16)",
        "96-100%",
        ", ".join(
            f"L{k} {fmt(100 * e['fraction_bit_identical'])}% (positions >= 10: {fmt(100 * e['fraction_bit_identical_pos_ge_10'])}%)"
            for k, e in er.items()
        ),
        "all-position share >= 96% in every claimed layer",
        _tally([e["fraction_bit_identical"] >= OVH_ERASURE[0] for e in er.values()]) if er else NOT_JUDGED,
        needs=OVH_MAMBA_LAYERS,
    )
    ge = {layer: (v.site(layer, "mixer_out") or {}).get("growth_exponent") for layer in OVH_MAMBA_LAYERS}
    add(
        "A7",
        "The magnitudes grow with position, as t^1.9 (L1), t^1.0 (L7), t^0.36 (L23)",
        ", ".join(f"L{k} {x}" for k, x in OVH_GROWTH.items()),
        ", ".join(f"L{k} {fmt(x)}" for k, x in ge.items())
        + " (slope of mean log10 token max against log10 position, positions >= 16)",
        f"each exponent within +-{EXPONENT_TOL} of the claimed one",
        _tally([ge[k] is not None and abs(ge[k] - OVH_GROWTH[k]) <= EXPONENT_TOL for k in OVH_MAMBA_LAYERS]),
        needs=OVH_MAMBA_LAYERS,
    )
    p0 = {layer: (v.site(layer, "mixer_out") or {}).get("position0_is_smallest") for layer in OVH_MAMBA_LAYERS}
    add(
        "A8",
        "Position 0 carries the smallest value, not the largest",
        "position 0 smallest",
        f"position 0 is the smallest mean log10 token max: {p0}",
        "true in every claimed layer",
        _tally([bool(x) for x in p0.values()]),
        needs=OVH_MAMBA_LAYERS,
    )
    sites = {layer: v.site(layer, "mixer_out") or {} for layer in OVH_MAMBA_LAYERS}
    ok = [
        s.get("frac_tokens_gt_448", 0) >= 0.9
        and s.get("frac_prompts_argmax_first", 1) == 0
        and s.get("frac_prompts_argmax_special", 1) <= 0.1
        for s in sites.values()
    ]
    add(
        "A9",
        "The mixer magnitudes appear at every position, with no privileged delimiter or first token",
        "all positions; no BOS dependence",
        ", ".join(
            f"L{k}: tokens > 448 {fmt(100 * s.get('frac_tokens_gt_448', 0))}%, argmax is first token in "
            f"{fmt(100 * s.get('frac_prompts_argmax_first', 0))}% and a special token in "
            f"{fmt(100 * s.get('frac_prompts_argmax_special', 0))}% of prompts"
            for k, s in sites.items()
        ),
        ">= 90% of tokens above 448; the argmax is never the first token and a special token in <= 10% of prompts",
        _tally(ok),
        needs=OVH_MAMBA_LAYERS,
    )
    moe_layers = sorted(layer for layer, row in v.rows.items() if "moe_post" in row["sites"])
    top4 = v.top_layers("moe_post", 4, moe_layers)
    ranked_m = sorted(moe_layers, key=lambda x: -(v.mx(x, "moe_post") or 0))
    gap_m = (
        (v.mx(ranked_m[3], "moe_post") / v.mx(ranked_m[4], "moe_post"))
        if len(ranked_m) > 4 and v.mx(ranked_m[4], "moe_post")
        else None
    )
    top4_out = v.top_layers("moe_out", 4, moe_layers)
    inter_m = set(top4) & set(OVH_MOE_LAYERS)
    add(
        "B1",
        "Four MoE layers spike through the squared ReLU (0, 1, 5, 19)",
        f"layers {list(OVH_MOE_LAYERS)}",
        f"top-4 MoE layers by max post-ReLU^2 activation: {top4} (4th/5th ratio {fmt(gap_m)}); by max MoE branch output: {top4_out}",
        "the top-4 set by post-ReLU^2 maximum equals OVH's",
        CONSISTENT if top4 == sorted(OVH_MOE_LAYERS) else (PARTLY if inter_m else NOT_CONSISTENT),
    )
    mpost = {layer: v.mx(layer, "moe_post") for layer in OVH_MOE_LAYERS}
    add(
        "B2",
        "These layers spike to 1e7-3e8 (the report does not say at which tensor; the post-ReLU^2 hidden is compared here)",
        "1e7 - 3e8",
        ", ".join(
            f"L{layer}: pre {fmt(v.mx(layer, 'moe_pre'))}, post {fmt(mpost[layer])}, latent out {fmt(v.mx(layer, 'routed_latent'))}, "
            f"routed branch {fmt(v.mx(layer, 'routed_out'))}, MoE branch {fmt(v.mx(layer, 'moe_out'))}"
            for layer in OVH_MOE_LAYERS
        ),
        "post-ReLU^2 maximum within [1e6, 3e9] (one decade around OVH's range) in every claimed layer",
        _tally(
            [
                mpost[layer] is not None and OVH_MOE_RANGE[0] / 10 <= mpost[layer] <= OVH_MOE_RANGE[1] * 10
                for layer in OVH_MOE_LAYERS
            ]
        ),
        needs=OVH_MOE_LAYERS,
    )
    gains = {}
    for layer in OVH_MOE_LAYERS:
        lat, out = v.mx(layer, "routed_latent"), v.mx(layer, "routed_out")
        gains[layer] = (out / lat) if lat else None
    add(
        "B3",
        "The up-projection amplifies the spike by a further ~30x",
        "~30x",
        ", ".join(
            f"L{k}: {fmt(g)}{'' if g is None else 'x'} (max routed branch / max latent expert output)"
            for k, g in gains.items()
        ),
        "ratio of the two maxima within [10, 100] in every claimed layer",
        _tally([g is not None and 10 <= g <= 100 for g in gains.values()]),
        needs=OVH_MOE_LAYERS,
    )
    fr = {layer: (v.site(layer, "moe_post") or {}).get("frac_tokens_ge_rel_max", {}) for layer in (19, 1)}
    add(
        "B4",
        "The MoE spikes are token-triggered: layer 19 fires on 4-7% of tokens (function words), layer 1 on rare / named-entity tokens",
        "L19: 4-7% of tokens",
        "; ".join(
            f"L{layer}: tokens at >= 50% / 10% / 1% of the layer max: "
            f"{fmt(100 * fr[layer].get('0.5', 0))}% / {fmt(100 * fr[layer].get('0.1', 0))}% / {fmt(100 * fr[layer].get('0.01', 0))}%"
            for layer in (19, 1)
        ),
        "none: OVH gives no firing threshold; token lists are in the JSON (top_tokens of moe_post)",
        NOT_JUDGED,
    )
    res_max = max(((v.mx(layer, "resid_out") or 0, layer) for layer in v.rows), default=(None, None))
    add(
        "C1",
        "The residual stream is bounded: entries never exceed 0.69 at any layer or position",
        "<= 0.69",
        f"largest residual max abs after any block: {fmt(res_max[0])} (layer {res_max[1]})",
        "maximum at most 0.695 (OVH's two decimals)",
        NOT_JUDGED if res_max[0] is None else (CONSISTENT if res_max[0] <= 0.695 else NOT_CONSISTENT),
    )
    allmax = max(
        ((s["max_abs"], layer, kind) for layer, row in v.rows.items() for kind, s in row["sites"].items()),
        default=(0, 0, ""),
    )
    orders = math.log10(allmax[0] / 65504.0) if allmax[0] > 0 else None
    add(
        "C2",
        "Activation block scales exceed the fp16 range by up to ten orders of magnitude",
        "up to ~10 orders above 65504",
        f"largest value at any site: {fmt(allmax[0])} (layer {allmax[1]}, {allmax[2]}) = {fmt(orders)} orders above 65504",
        "measured orders within +-1.5 of 10",
        NOT_JUDGED if orders is None else (CONSISTENT if abs(orders - OVH_FP16_ORDERS) <= 1.5 else NOT_CONSISTENT),
    )
    spec_rows = [
        ("zero_mixer_L1", v.ablation(1, "mixer_out", "zero")),
        ("zero_head_L1", v.ablation(1, "head", "zero")),
        ("zero_head_L23", v.ablation(23, "head", "zero")),
    ]
    for key, a in spec_rows:
        label, claimed = OVH_ABLATIONS[key]
        m = a["metrics"] if a else None
        d = m["delta_nll_mean"] if m else None
        add(
            "D-" + key.replace("zero_", "").replace("_", "-"),
            f"Ablation: {label} changes wikitext NLL by {claimed:+.3f} (the magnitudes are 'vestigial')",
            f"{claimed:+.3f} nats",
            "not run (the branch / head is not above the threshold or not in this model)"
            if m is None
            else f"{fmt(d)} nats (teacher-forced on the probe prompts); KL {fmt(m['kl_mean_all_positions'])}, "
            f"last-position KL {fmt(m['kl_last_position_mean'])}, top-1 agreement {fmt(m['top1_agreement'])}",
            f"absolute measured change <= {ABLATION_SMALL} (OVH's own control surgery costs up to +0.111); text differs from OVH's wikitext",
            NOT_JUDGED if m is None or d is None else (CONSISTENT if abs(d) <= ABLATION_SMALL else NOT_CONSISTENT),
        )
    add(
        "D-dims",
        "Ablation: zeroing only the 'massive dimensions' changes NLL by +0.010 to +0.018; greedy generations stay coherent",
        "+0.010 to +0.018",
        "this probe ablates whole branches and whole heads, not individual channels; no generation is run",
        "none",
        NOT_JUDGED,
    )
    add(
        "E1",
        "The recurrent state of the three heads holds magnitudes of 2e9 to 1.2e10 after a 512-token prefill",
        "2e9 - 1.2e10",
        "the probe records outputs, not the SSM state (internal to the pure-PyTorch recurrence)",
        "none",
        NOT_JUDGED,
    )
    return rows


def _table(header: list[str], body: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(c.replace("|", "/") for c in row) + " |" for row in body]
    return out


def render_markdown(result: dict[str, Any], title: str = "Olala activation probe") -> str:
    """The markdown report: run description, the claim table, per-layer tables, ablations, quantization consequences."""
    v = View(result)
    m = v.meta
    text = result.get("token_text", {})

    def tok(t: int) -> str:
        return repr(text.get(str(t), f"<{t}>"))

    lines = [f"# {title}", ""]
    lines += [
        f"- **Layer numbering: 0-based** (`block.layer_idx`). The attention layers of this model are "
        f"{m['attention_layers_0based']}; layer 0 is the first block. OVH's Appendix C uses the same convention.",
        f"- Prompts: {m['n_prompts']} ({m['n_tokens_total']} tokens, longest {m['max_tokens_in_a_prompt']}); "
        f"device {m['device']}; weights dtype {m['weights_dtype']}; attention implementation {m['attn_impl']}; torch {m['torch']}.",
        f"- Mamba-3 path: {m['mamba3_path']}.",
        f"- Spike threshold used to flag a branch for ablation: {fmt(m['spike_threshold'])}; ablations on {m['ablation_prompts']} prompts.",
        "- Verdict words: `consistent` = the measured value satisfies the rule in the table; `not judged` = the claim has no "
        "checkable definition here. A verdict speaks only for this prompt set, sequence length and forward path.",
        "",
        "## Claim by claim",
        "",
    ]
    rows = claim_rows(result)
    lines += _table(
        ["ID", "OVH claim", "Claimed", "Measured here", "Rule", "Verdict"],
        [[r["id"], r["claim"], r["claimed"], r["measured"], r["rule"], r["verdict"]] for r in rows],
    )
    lines += ["", "## Mamba-3 layers: mixer branch (output of the projection) and heads", ""]
    body = []
    for layer in v.mamba_layers():
        s = v.site(layer, "mixer_out")
        h = v.heads(layer)
        a = s["argmax"]
        body.append(
            [
                str(layer),
                fmt(s["max_abs"]),
                fmt(s["rms"]),
                fmt(s["max_over_rms"]),
                fmt(s["p999_median_over_prompts"]),
                f"{a['position']} / {tok(a['token_id'])} / {'special' if a['is_special'] else '-'}{' first' if a['is_first'] else ''}",
                str(a["channel"]),
                f"{fmt(100 * s['frac_tokens_gt_448'])} / {fmt(100 * s['frac_tokens_gt_65504'])}",
                f"{s['n_nan']} / {s['n_inf']} / {s['n_ge_3p4e38']}",
                str(h["top_head"]),
                fmt(h["top_head_max"]),
                fmt(h["second_head_max"]),
                fmt(h["dominance_ratio"]),
            ]
        )
    lines += _table(
        [
            "layer",
            "max abs",
            "RMS",
            "max/RMS",
            "p99.9 (median)",
            "max at position / token / flags",
            "channel",
            "% tokens > 448 / > 65504",
            "NaN / inf / >=3.4e38",
            "top head",
            "top head max",
            "2nd head max",
            "top/2nd",
        ],
        body,
    )
    lines += ["", "## MoE layers: activations and branches", ""]
    body = []
    for layer in sorted(v.rows):
        row = v.rows[layer]
        if "moe_post" not in row["sites"]:
            continue
        ex = row["experts"]
        top_e = ex["expert_hist_top_pairs"][0] if ex["expert_hist_top_pairs"] else None
        body.append(
            [
                str(layer),
                fmt(v.mx(layer, "moe_pre")),
                fmt(v.mx(layer, "moe_post")),
                fmt(v.mx(layer, "routed_latent")),
                fmt(v.mx(layer, "routed_out")),
                fmt(v.mx(layer, "shared_out")),
                fmt(v.mx(layer, "moe_out")),
                fmt(row["sites"]["moe_post"]["rms"]),
                fmt(row["sites"]["moe_out"]["max_over_rms"]),
                f"{fmt(100 * row['sites']['moe_out']['frac_tokens_gt_448'])} / {fmt(100 * row['sites']['moe_out']['frac_tokens_gt_65504'])}",
                f"{top_e['expert']} (x{top_e['count']})" if top_e else "n/a",
            ]
        )
    lines += _table(
        [
            "layer",
            "max pre-act",
            "max post-ReLU^2",
            "max latent expert out",
            "max routed branch",
            "max shared branch",
            "max MoE branch",
            "RMS post-ReLU^2",
            "branch max/RMS",
            "% tokens > 448 / > 65504 (branch)",
            "expert most often in the top pairs",
        ],
        body,
    )
    lines += ["", "## Residual stream after each block", ""]
    body = [
        [
            str(layer),
            fmt(v.mx(layer, "resid_out")),
            fmt(v.site(layer, "resid_out")["rms"]),
            f"{v.site(layer, 'resid_out')['n_nan']}",
        ]
        for layer in sorted(v.rows)
    ]
    lines += _table(["layer", "max abs", "RMS", "NaN"], body)
    lines += ["", "## Dependence of the next-token distribution on the spiking branches", ""]
    if result.get("ablations"):
        body = []
        for a in result["ablations"]:
            mm = a["metrics"]
            body.append(
                [
                    a["label"],
                    fmt(mm["kl_mean_all_positions"]),
                    fmt(mm["kl_max_position"]),
                    fmt(mm["kl_last_position_mean"]),
                    fmt(mm["top1_agreement"]),
                    fmt(mm["delta_nll_mean"]),
                    str(mm["n_nonfinite_positions"]),
                ]
            )
        lines += _table(
            [
                "ablation (layer.branch:mode)",
                "KL mean (nats)",
                "KL max",
                "KL last position",
                "top-1 agreement",
                "delta NLL",
                "non-finite positions",
            ],
            body,
        )
        lines += [
            "",
            f"Prompts used: {', '.join(result['ablation_prompt_ids'])}. `noop` is a control that must give 0; `zero` replaces the "
            "branch (or head) by zeros; `clamp448` clamps each element to [-448, 448].",
        ]
    else:
        lines += ["No branch exceeded the spike threshold; no ablation was run."]
    lines += ["", "## What this means for quantization", ""]
    body = []
    for layer in sorted(v.rows):
        for kind, s in v.rows[layer]["sites"].items():
            if s["max_abs"] > 448 or s["n_nan"] or s["n_inf"]:
                body.append(
                    [
                        str(layer),
                        kind,
                        fmt(s["max_abs"]),
                        fmt(math.log10(s["max_abs"] / 448.0) if s["max_abs"] > 0 else None),
                        fmt(math.log10(s["max_abs"] / 65504.0) if s["max_abs"] > 0 else None),
                        fmt(100 * s["frac_tokens_gt_448"]),
                        fmt(100 * s["frac_tokens_gt_65504"]),
                        f"{s['n_nan']} / {s['n_inf']}",
                    ]
                )
    if body:
        lines += _table(
            [
                "layer",
                "site",
                "max abs",
                "orders above e4m3 max (448)",
                "orders above fp16 max (65504)",
                "% tokens > 448",
                "% tokens > 65504",
                "NaN / inf",
            ],
            body,
        )
    else:
        lines += ["No site exceeded 448 and no non-finite value was seen."]
    lines += ["", "## Experts producing the largest post-ReLU^2 values (flagged MoE layers)", ""]
    for layer in top_moe_layers(v, 4):
        ex = v.rows[layer]["experts"]
        hist = ", ".join(
            f"expert {h['expert']}: {h['count']} of the top pairs (max {fmt(h['max_post'])})"
            for h in ex["expert_hist_top_pairs"][:6]
        )
        toks = ", ".join(
            f"{tok(t['token_id'])}@{t['position']} {fmt(t['value'])}"
            for t in v.rows[layer]["sites"]["moe_post"].get("top_tokens", [])[:8]
        )
        lines += [f"- layer {layer}: {hist}. Largest tokens: {toks}."]
    lines += ["", "## Caveats", ""]
    lines += [
        "- The forward path is the repository's pure-PyTorch float32 Mamba-3 recurrence and expert loop, not the TileLang / scattermoe kernels "
        "OVH measured with; magnitudes of 1e15 are representable in both but a kernel-level rounding difference cannot be excluded.",
        "- OVH measured teacher-forced NLL on 512-token wikitext-2 passages; the prompts here are chat-templated task inputs and plain text, "
        "so ablation deltas are not on the same text and the position-dependent magnitudes (A3, A7) depend on the sequence length.",
        "- Verdict rules are mine and printed in the table; OVH's wording (`O(0.1)`, `up to`) is not a numeric threshold.",
        "- The SSM state itself is not recorded (claim E1).",
        "- Every figure is a maximum or a pooled statistic over the listed prompts, not a population estimate.",
    ]
    return "\n".join(lines) + "\n"


def top_moe_layers(v: View, k: int) -> list[int]:
    """The ``k`` MoE layers with the largest post-ReLU^2 maximum, in layer order."""
    pool = sorted(layer for layer, row in v.rows.items() if "moe_post" in row["sites"])
    return v.top_layers("moe_post", k, pool)
