"""The tiny end-to-end run over several random models: tiny model -> N2 and G1-lite checkpoints -> reload -> forward."""

import pytest

import tiny_e2e

pytestmark = pytest.mark.slow


def test_g1_beats_rtn_on_average_on_fresh_windows_of_the_calibration_distribution(tmp_path) -> None:
    res = tiny_e2e.run_seeds([0, 1, 2], tmp_path)
    fresh = res["mean"]["fresh"]
    assert fresh["n2"]["mean_kl"] > 1e-4
    assert fresh["g1"]["mean_kl"] < fresh["n2"]["mean_kl"], res["mean"]
    assert fresh["g1"]["rel_logit_rmse"] < fresh["n2"]["rel_logit_rmse"], res["mean"]
    assert all(r["n_gptq_experts"] > 0 for r in res["seeds"])
    # the other distribution is reported, not asserted: calibration is specific to the data it saw
    assert set(res["mean"]) == {"fresh", "random_tokens"}


def test_experts_only_base_compares_gptq_with_rtn_on_the_same_bf16_non_experts(tmp_path) -> None:
    """The same comparison with the experts-only base (N5 = RTN, G1 = GPTQ): reported on the tiny model, no claim."""
    res = tiny_e2e.run_seeds([0, 1], tmp_path, base_select="experts")
    assert res["base_select"] == "experts"
    assert set(res["mean"]) == {"fresh", "random_tokens"}
    for kind in ("fresh", "random_tokens"):
        assert set(res["mean"][kind]) == {"n5", "g1"}  # the RTN arm of this base is N5, not N2
        for arm in ("n5", "g1"):
            assert res["mean"][kind][arm]["mean_kl"] > 0 and res["mean"][kind][arm]["rel_logit_rmse"] > 0
    assert res["mean"]["fresh"]["n5"]["mean_kl"] > 1e-5
    assert all(r["n_gptq_experts"] > 0 for r in res["seeds"])
    # both arms quantize exactly the routed experts of layers 1 to 3 (6 tensors) and nothing else
    assert all(r["n_quantized_tensors"] == {"g1": 6, "n5": 6} for r in res["seeds"])
    # the experts-only RTN arm is closer to bf16 than the full-N2 RTN arm: it quantizes a subset of N2's tensors
    base = tiny_e2e.run_seeds([0, 1], tmp_path / "n2", base_select="")
    assert base["base_select"] == ""
    assert res["mean"]["fresh"]["n5"]["mean_kl"] < base["mean"]["fresh"]["n2"]["mean_kl"]
