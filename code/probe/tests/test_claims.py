"""Tests of probe_claims.py on synthetic probe results shaped like the real one."""

import copy

import pytest

import probe_claims as pc

ATTN = [4, 9, 14, 19, 24, 29, 34]


def _site(mx=1.0, rms=0.5, **kw):
    s = {
        "max_abs": mx,
        "rms": rms,
        "max_over_rms": mx / rms,
        "p999_median_over_prompts": mx / 2,
        "argmax": {"prompt": 0, "position": 40, "token_id": 7, "channel": 3, "is_special": False, "is_first": False},
        "frac_prompts_argmax_special": 0.0,
        "frac_prompts_argmax_first": 0.0,
        "frac_tokens_gt_448": 0.0,
        "frac_tokens_gt_65504": 0.0,
        "frac_tokens_ge_rel_max": {"0.5": 0.01, "0.1": 0.05, "0.01": 0.1, "0.0001": 0.5},
        "n_nan": 0,
        "n_inf": 0,
        "n_ge_3p4e38": 0,
        "growth_exponent": 0.0,
        "position0_is_smallest": True,
        "profile_log10": {},
        "top_tokens": [{"value": mx, "prompt": 0, "position": 40, "token_id": 7}],
    }
    s.update(kw)
    return s


def make_result(mamba_spikes=None, moe_spikes=None, heads=None, exponents=None, erasure=0.99, resid=0.6, ablate=None):
    mamba_spikes = mamba_spikes or {1: 5e15, 7: 5e15, 23: 1.5e11}
    moe_spikes = moe_spikes or {0: 1e8, 1: 3e8, 5: 1e7, 19: 2e8}
    heads = heads or {1: 23, 7: 26, 23: 40}
    exponents = exponents or {1: 1.9, 7: 1.0, 23: 0.36}
    layers = []
    for i in range(36):
        typ = "V" if i in ATTN else "M"
        sites = {
            "mixer_out": _site(
                mamba_spikes.get(i, 3.0),
                growth_exponent=exponents.get(i, 0.0),
                frac_tokens_gt_448=1.0 if i in mamba_spikes else 0.0,
            ),
            "moe_out": _site(moe_spikes.get(i, 1.0) * 30),
            "shared_out": _site(2.0),
            "routed_out": _site(moe_spikes.get(i, 1.0) * 30),
            "routed_latent": _site(moe_spikes.get(i, 1.0)),
            "moe_pre": _site(moe_spikes.get(i, 1.0) ** 0.5),
            "moe_post": _site(moe_spikes.get(i, 1.0)),
            "resid_out": _site(resid, 0.02),
        }
        row = {
            "layer": i,
            "type": typ,
            "sites": sites,
            "heads": None,
            "mixer_erasure": None,
            "experts": {
                "expert_hist_top_pairs": [{"expert": 3, "count": 5, "max_post": 1.0, "n_pairs_post_gt_448": 0}],
                "top_pairs": [],
                "top_expert_by_post": 3,
            },
        }
        if typ == "M":
            row["heads"] = {
                "top_head": heads.get(i, 0),
                "top_head_max": sites["mixer_out"]["max_abs"],
                "second_head_max": 0.2,
                "median_sibling_max": 0.2,
                "median_sibling_rms": 0.05,
                "dominance_ratio": sites["mixer_out"]["max_abs"] / 0.2,
            }
            row["mixer_erasure"] = {
                "fraction_bit_identical": erasure if i in mamba_spikes else 0.1,
                "fraction_bit_identical_pos_ge_10": 1.0,
                "n_total": 100,
            }
        layers.append(row)
    meta = {
        "spike_threshold": 448.0,
        "attention_layers_0based": ATTN,
        "max_tokens_in_a_prompt": 1500,
        "n_prompts": 80,
        "n_tokens_total": 50000,
        "device": "cuda",
        "weights_dtype": "torch.bfloat16",
        "attn_impl": "fa2",
        "torch": "2.14",
        "mamba3_path": "pure torch",
        "ablation_prompts": 16,
    }
    return {
        "layers": layers,
        "meta": meta,
        "ablations": ablate or [],
        "ablation_prompt_ids": ["a"],
        "token_text": {"7": "the"},
    }


def _by(rows):
    return {r["id"]: r for r in rows}


def test_everything_consistent_when_the_data_equals_the_claims():
    ab = [
        {
            "label": "L1.mixer_out:zero",
            "layer": 1,
            "branch": "mixer_out",
            "mode": "zero",
            "head": None,
            "metrics": {
                "delta_nll_mean": -0.009,
                "kl_mean_all_positions": 0.01,
                "kl_last_position_mean": 0.02,
                "top1_agreement": 0.99,
                "kl_max_position": 0.1,
                "n_nonfinite_positions": 0,
            },
        },
        {
            "label": "L1.head23:zero",
            "layer": 1,
            "branch": "head",
            "mode": "zero",
            "head": 23,
            "metrics": {
                "delta_nll_mean": 0.04,
                "kl_mean_all_positions": 0.01,
                "kl_last_position_mean": 0.02,
                "top1_agreement": 0.99,
                "kl_max_position": 0.1,
                "n_nonfinite_positions": 0,
            },
        },
    ]
    rows = _by(pc.claim_rows(make_result(ablate=ab)))
    for cid in (
        "A1",
        "A1b",
        "A2",
        "A3",
        "A4",
        "A5",
        "A6",
        "A7",
        "A8",
        "A9",
        "B1",
        "B2",
        "B3",
        "C1",
        "C2",
        "D-mixer-L1",
        "D-head-L1",
    ):
        assert rows[cid]["verdict"] == pc.CONSISTENT, (cid, rows[cid])
    assert rows["B4"]["verdict"] == pc.NOT_JUDGED and rows["E1"]["verdict"] == pc.NOT_JUDGED
    assert rows["D-head-L23"]["verdict"] == pc.NOT_JUDGED  # not run


def test_wrong_layers_are_reported_as_such_and_the_one_based_reading_is_flagged():
    res = make_result(mamba_spikes={0: 5e15, 6: 5e15, 22: 1.5e11})
    rows = _by(pc.claim_rows(res))
    assert rows["A1"]["verdict"] == pc.NOT_CONSISTENT
    assert "do match" not in rows["A1b"]["measured"] and "which match" in rows["A1b"]["measured"]
    assert rows["A1b"]["verdict"] == pc.PARTLY


def test_partial_layer_overlap_is_partly_consistent():
    rows = _by(pc.claim_rows(make_result(mamba_spikes={1: 5e15, 7: 5e15, 22: 1.5e11})))
    assert rows["A1"]["verdict"] == pc.PARTLY


def test_head_identity_tally():
    rows = _by(pc.claim_rows(make_result(heads={1: 23, 7: 26, 23: 41})))
    assert rows["A2"]["verdict"] == pc.PARTLY
    rows = _by(pc.claim_rows(make_result(heads={1: 1, 7: 2, 23: 3})))
    assert rows["A2"]["verdict"] == pc.NOT_CONSISTENT


def test_magnitude_rule_is_one_decade():
    rows = _by(
        pc.claim_rows(make_result(mamba_spikes={1: 5e14, 7: 5e15, 23: 1.5e11}))
    )  # L1 is one decade low: still within
    assert rows["A3"]["verdict"] == pc.CONSISTENT
    rows = _by(pc.claim_rows(make_result(mamba_spikes={1: 5e12, 7: 5e15, 23: 1.5e11})))
    assert rows["A3"]["verdict"] == pc.PARTLY


def test_erasure_exponent_and_other_rules():
    assert _by(pc.claim_rows(make_result(erasure=0.9)))["A6"]["verdict"] == pc.NOT_CONSISTENT
    assert _by(pc.claim_rows(make_result(exponents={1: 0.2, 7: 1.0, 23: 0.36})))["A7"]["verdict"] == pc.PARTLY
    assert _by(pc.claim_rows(make_result(resid=0.9)))["C1"]["verdict"] == pc.NOT_CONSISTENT
    assert _by(pc.claim_rows(make_result(resid=0.69)))["C1"]["verdict"] == pc.CONSISTENT


def test_moe_set_and_magnitude_rules():
    rows = _by(pc.claim_rows(make_result(moe_spikes={0: 1e8, 1: 3e8, 5: 1e7, 20: 2e8})))
    assert rows["B1"]["verdict"] == pc.PARTLY
    rows = _by(pc.claim_rows(make_result(moe_spikes={0: 1e3, 1: 1e3, 5: 1e3, 19: 1e3})))
    assert rows["B2"]["verdict"] == pc.NOT_CONSISTENT


def test_special_token_argmax_breaks_the_no_privileged_token_claim():
    res = make_result()
    res["layers"][1]["sites"]["mixer_out"]["frac_prompts_argmax_first"] = 0.5
    assert _by(pc.claim_rows(res))["A9"]["verdict"] == pc.PARTLY


def test_markdown_has_the_tables_the_indexing_note_and_no_overstrong_words():
    res = make_result()
    md = pc.render_markdown(res)
    assert "0-based" in md and "[4, 9, 14, 19, 24, 29, 34]" in md
    for heading in (
        "## Claim by claim",
        "## Mamba-3 layers",
        "## MoE layers",
        "## Residual stream",
        "## What this means for quantization",
        "## Caveats",
    ):
        assert heading in md
    assert md.count("| A1 |") == 1 and "| B2 |" in md
    low = md.lower()
    assert not [w for w in pc.BANNED if w in low]


def test_fmt():
    assert pc.fmt(None) == "n/a" and pc.fmt(float("nan")) == "NaN" and pc.fmt(1.8e16) == "1.8e+16" and pc.fmt(3) == "3"


def test_markdown_does_not_mutate_the_result():
    res = make_result()
    before = copy.deepcopy(res)
    pc.render_markdown(res)
    assert res == before


@pytest.mark.parametrize("missing", ["ablations"])
def test_markdown_without_ablations(missing):
    res = make_result()
    res[missing] = []
    assert "No branch exceeded the spike threshold" in pc.render_markdown(res)


def test_claims_about_layers_that_do_not_exist_are_not_judged():
    res = make_result()
    res["layers"] = res["layers"][:4]  # a 4-layer model has no layer 7, 19 or 23
    rows = _by(pc.claim_rows(res))
    for cid in ("A2", "A3", "A4", "A6", "A7", "A8", "A9", "B2", "B3"):
        assert rows[cid]["verdict"] == pc.NOT_JUDGED, cid
        assert "do not exist in this model" in rows[cid]["measured"]


def test_table_cells_contain_no_pipe_characters():
    md = pc.render_markdown(make_result())
    for line in md.splitlines():
        if line.startswith("| A") or line.startswith("| D"):
            assert line.count("|") == 7, line
