"""Tests of probe_stats.py: per-site summaries and their accumulation across prompts, against explicit python sums."""

import math

import numpy as np
import pytest
import torch

import probe_stats as ps


def _x(rows=6, d=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, d, generator=g)


def test_summarize_basic_values_match_explicit_python():
    x = _x()
    s = ps.summarize(x)
    flat = [abs(v) for row in x.tolist() for v in row]
    assert s.max_abs == pytest.approx(max(flat), rel=1e-6)
    assert s.sumsq == pytest.approx(sum(v * v for row in x.tolist() for v in row), rel=1e-5)
    assert s.n_elems == 30
    assert s.n_nan == 0 and s.n_inf == 0 and s.n_ge_f32max == 0
    assert s.tok_max.shape == (6,)
    assert s.tok_max == pytest.approx([max(abs(v) for v in row) for row in x.tolist()], rel=1e-6)


def test_summarize_uses_absolute_value_not_signed_max():
    x = torch.zeros(4, 3)
    x[2, 1] = -1234.0
    x[0, 0] = 5.0
    s = ps.summarize(x)
    assert s.max_abs == 1234.0
    assert (s.arg_token, s.arg_channel) == (2, 1)


def test_summarize_p999_is_the_order_statistic():
    x = torch.arange(1.0, 2001.0).reshape(-1, 4)  # 2000 values 1..2000
    s = ps.summarize(x)
    assert s.p999 == 1998.0  # ceil(0.999 * 2000) = 1998th smallest


def test_summarize_counts_nan_inf_and_huge_and_keeps_max_finite():
    x = torch.ones(3, 4)
    x[0, 0] = float("nan")
    x[1, 1] = float("inf")
    x[2, 2] = -float("inf")
    x[2, 3] = 7.0
    s = ps.summarize(x)
    assert (s.n_nan, s.n_inf) == (1, 2)
    assert s.n_ge_f32max == 2  # the two infinities are >= 3.4e38
    assert s.max_abs == 7.0  # max over the finite entries


def test_summarize_threshold_counts_per_token():
    x = torch.zeros(5, 2)
    x[0, 0] = 447.0
    x[1, 0] = 449.0
    x[2, 1] = -70000.0
    x[3, 1] = 65504.0  # not strictly above
    s = ps.summarize(x)
    assert s.n_tok_gt_448 == 3  # 449, 70000 and 65504; 447 is below
    assert s.n_tok_gt_65504 == 1


def test_summarize_with_rows_mapped_to_tokens_takes_the_max_over_rows_of_a_token():
    x = torch.tensor([[1.0, 2.0], [9.0, 3.0], [4.0, -5.0], [0.5, 0.5]])
    token_of = torch.tensor([0, 2, 2, 0])
    s = ps.summarize(x, token_of=token_of, num_tokens=4)
    assert s.tok_max.tolist() == [2.0, 0.0, 9.0, 0.0]
    assert s.arg_token == 2 and s.arg_row == 1 and s.arg_channel == 0


def test_summarize_handles_bf16_and_does_not_overflow_the_square_sum():
    x = torch.full((2, 2), 1e16, dtype=torch.bfloat16)
    s = ps.summarize(x)
    assert math.isfinite(s.sumsq) and s.sumsq == pytest.approx(4 * float(x[0, 0]) ** 2, rel=1e-6)


def _acc(layer=0, kind="mixer_out", max_len=16):
    return ps.SiteAcc(layer=layer, kind=kind, max_len=max_len)


def test_accumulator_pools_prompts_and_locates_the_global_max():
    acc = _acc()
    toks_a = np.array([10, 11, 12, 13])
    toks_b = np.array([20, 21, 22])
    special = lambda t: np.isin(t, [10, 22])  # noqa: E731
    xa = torch.ones(4, 2)
    xb = torch.ones(3, 2)
    xb[1, 1] = -900.0
    acc.add(ps.summarize(xa), prompt=0, tokens=toks_a, special=special(toks_a))
    acc.add(ps.summarize(xb), prompt=1, tokens=toks_b, special=special(toks_b))
    r = acc.result()
    assert r["max_abs"] == 900.0
    assert r["argmax"] == {
        "prompt": 1,
        "position": 1,
        "token_id": 21,
        "channel": 1,
        "is_special": False,
        "is_first": False,
    }
    assert r["n_tokens"] == 7 and r["n_prompts"] == 2
    assert r["rms"] == pytest.approx(math.sqrt((4 * 2 + 3 * 2 - 1 + 900.0**2) / 14), rel=1e-6)
    assert r["max_over_rms"] == pytest.approx(900.0 / r["rms"], rel=1e-6)
    assert r["frac_tokens_gt_448"] == pytest.approx(1 / 7)
    assert r["frac_tokens_gt_65504"] == 0.0


def test_accumulator_flags_special_and_first_token_for_the_argmax():
    acc = _acc()
    x = torch.ones(4, 2)
    x[0, 0] = 1000.0
    toks = np.array([7, 8, 9, 10])
    acc.add(ps.summarize(x), prompt=0, tokens=toks, special=np.array([True, False, False, False]))
    r = acc.result()
    assert r["argmax"]["is_first"] is True and r["argmax"]["is_special"] is True and r["argmax"]["position"] == 0
    x2 = torch.ones(4, 2)
    x2[3, 0] = 2000.0
    acc2 = _acc()
    acc2.add(ps.summarize(x2), prompt=0, tokens=toks, special=np.array([True, False, False, True]))
    r2 = acc2.result()
    assert r2["argmax"]["is_first"] is False and r2["argmax"]["is_special"] is True
    assert r2["frac_prompts_argmax_special"] == 1.0 and r2["frac_prompts_argmax_first"] == 0.0


def test_accumulator_token_histogram_gives_fraction_above_a_relative_level():
    acc = _acc()
    x = torch.ones(10, 1)
    x[0, 0] = 1000.0  # 1 token at the max, 9 at 1.0
    acc.add(ps.summarize(x), prompt=0, tokens=np.arange(10), special=np.zeros(10, bool))
    r = acc.result()
    fr = r["frac_tokens_ge_rel_max"]
    assert fr["0.5"] == pytest.approx(0.1)
    assert fr["0.01"] == pytest.approx(0.1)
    assert fr["0.0001"] == pytest.approx(1.0)  # 1.0 >= 1e-4 * 1000 (bin resolution is a quarter decade)


def test_growth_exponent_recovers_a_power_law():
    t = np.arange(1, 600)
    mean_log10 = 1.9 * np.log10(t) + 0.3
    g = ps.growth_exponent(mean_log10, np.ones_like(t), lo=16)
    assert g == pytest.approx(1.9, abs=1e-6)
    flat = ps.growth_exponent(np.full(t.shape, 2.0), np.ones_like(t), lo=16)
    assert flat == pytest.approx(0.0, abs=1e-9)


def test_growth_exponent_none_when_too_few_positions():
    assert ps.growth_exponent(np.array([1.0, 2.0, 3.0]), np.ones(3), lo=16) is None


def test_accumulator_profile_tracks_position_zero_and_exponent():
    acc = _acc(max_len=64)
    n = 64
    pos = np.arange(n)
    x = torch.from_numpy((pos + 1.0) ** 1.0).float().reshape(n, 1)  # token max = t (1-based), exponent 1
    acc.add(ps.summarize(x), prompt=0, tokens=np.arange(n), special=np.zeros(n, bool))
    r = acc.result()
    assert r["growth_exponent"] == pytest.approx(1.0, abs=1e-3)
    assert r["position0_is_smallest"] is True
    assert r["profile_log10"]["0"] == pytest.approx(0.0, abs=1e-6)


def test_head_summary_per_head_values_and_top_head():
    y = torch.randn(1, 12, 3 * 4, generator=torch.Generator().manual_seed(1))
    y[0, 7, 2 * 4 + 1] = -5000.0  # head 2, position 7, channel 1
    h = ps.head_summary(y, n_heads=3)
    assert h.head_max.shape == (3,)
    assert int(h.head_max.argmax()) == 2
    assert float(h.head_max[2]) == 5000.0
    assert int(h.head_arg_pos[2]) == 7 and int(h.head_arg_channel[2]) == 1
    assert h.head_tok_max.shape == (12, 3)
    ref = y.view(12, 3, 4).double().pow(2).sum(dim=(0, 2))
    assert h.head_sumsq.tolist() == pytest.approx(ref.tolist(), rel=1e-6)


def test_head_acc_pools_and_reports_dominance():
    acc = ps.HeadAcc(layer=3, n_heads=4, max_len=32, head_dim=2)
    for prompt in range(2):
        y = torch.full((1, 10, 8), 0.1)
        y[0, 5 + prompt, 2 * 2] = 1.0e6  # head 2
        acc.add(ps.head_summary(y, 4), prompt=prompt, tokens=np.arange(10), special=np.zeros(10, bool))
    r = acc.result()
    assert r["top_head"] == 2
    assert r["head_max"][2] == pytest.approx(1.0e6)
    assert r["second_head_max"] == pytest.approx(0.1, rel=1e-5)
    assert r["median_sibling_max"] == pytest.approx(0.1, rel=1e-5)
    assert r["dominance_ratio"] == pytest.approx(1.0e7, rel=1e-4)
    assert r["top_head_argmax"]["position"] == 5 and r["top_head_argmax"]["prompt"] == 0
    assert r["head_rms"][0] == pytest.approx(0.1, rel=1e-5)


def test_accumulator_keeps_the_top_tokens_across_prompts():
    acc = ps.SiteAcc(layer=0, kind="moe_post", max_len=16, top_n=3)
    for prompt, (a, b) in enumerate([(5.0, 50.0), (500.0, 7.0)]):
        x = torch.ones(6, 1)
        x[1, 0], x[4, 0] = a, b
        acc.add(ps.summarize(x), prompt=prompt, tokens=np.arange(100, 106), special=np.zeros(6, bool))
    top = acc.result()["top_tokens"]
    assert [(t["value"], t["prompt"], t["position"], t["token_id"]) for t in top] == [
        (500.0, 1, 1, 101),
        (50.0, 0, 4, 104),
        (7.0, 1, 4, 104),
    ]
