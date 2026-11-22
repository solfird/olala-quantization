"""End-to-end tests of build_g1lite.py on the TINY random Olala model: checkpoint format, scales, guards, resume."""

import json
import shutil
from pathlib import Path

import pytest
import torch
from olala_quant import convert as conv
from olala_quant import nvfp4, verify
from olala_quant.checkpoint_io import SourceCheckpoint
from safetensors import safe_open

import build_g1lite as bg
import tiny_model
from tests_support import make_calibration

pytestmark = pytest.mark.slow

ROUTED = ("mlp.experts.experts.weight", "mlp.experts.output_experts.weight")


def run_build(src: Path, calib: Path, dst: Path, *extra: str) -> int:
    ignore = tmp_ignore_file(dst.parent)
    return bg.main(
        [
            "--src",
            str(src),
            "--calib",
            str(calib),
            "--dst",
            str(dst),
            "--ignore-file",
            str(ignore),
            "--heldout-every",
            "4",
            "--batch-windows",
            "4",
            *extra,
        ]
    )


def tmp_ignore_file(directory: Path) -> Path:
    path = directory / "ignore.json"
    path.write_text(json.dumps(list(tiny_model.TINY_IGNORE)))
    return path


def read_all(directory: Path) -> dict[str, torch.Tensor]:
    out = {}
    for f in sorted(directory.glob("model-*.safetensors")):
        with safe_open(f, framework="pt") as h:
            out.update({k: h.get_tensor(k) for k in h.keys()})  # noqa: SIM118
    return out


def raw(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().reshape(-1).view(torch.uint8)


@pytest.fixture(scope="module")
def built(tiny_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> dict:
    tmp = tmp_path_factory.mktemp("g1")
    calib = make_calibration(tmp)
    dst = tmp / "g1"
    assert run_build(tiny_dir, calib, dst) == 0
    n2 = tmp / "n2"
    conv.convert(tiny_dir, n2, "nvfp4", tiny_model.TINY_IGNORE, log=lambda _s: None)
    return {"tmp": tmp, "calib": calib, "dst": dst, "n2": n2, "src": tiny_dir}


def test_the_checkpoint_verifies_in_calibrated_mode_and_is_refused_by_the_strict_mode(built) -> None:
    rep = verify.verify(built["src"], built["dst"], calibrated=True)
    assert rep.violations == []
    strict = verify.verify(built["src"], built["dst"])
    assert {v.check for v in strict.violations} == {"calibrated_checkpoint"}


def test_scales_are_exactly_those_of_n2_and_every_other_tensor_is_unchanged(built) -> None:
    g1, n2 = read_all(built["dst"]), read_all(built["n2"])
    assert set(g1) == set(n2)
    n_changed = 0
    for name, t in g1.items():
        assert t.dtype == n2[name].dtype and t.shape == n2[name].shape
        is_routed_codes = name.endswith("weight_packed") and any(r.removesuffix(".weight") in name for r in ROUTED)
        if is_routed_codes and ".layers.0." not in name:
            n_changed += int(not torch.equal(t, n2[name]))
        else:
            assert torch.equal(raw(t), raw(n2[name])), name
    assert n_changed >= 2  # calibration really changed codes in the routed tensors
    # same config, same index layout
    assert json.loads((built["dst"] / "config.json").read_text()) == json.loads(
        (built["n2"] / "config.json").read_text()
    )


def test_scales_equal_what_olala_quant_computes_from_the_same_weights(built) -> None:
    src = SourceCheckpoint(built["src"])
    g1 = read_all(built["dst"])
    for layer in (1, 2, 3):
        for r in ROUTED:
            name = f"model.layers.{layer}.{r}"
            ref = nvfp4.quantize_nvfp4(src.get_tensor(name))
            base = name.removesuffix(".weight")
            assert torch.equal(raw(g1[f"{base}.weight_scale"]), raw(ref.block_scale))
            assert torch.equal(g1[f"{base}.weight_global_scale"], ref.global_scale)


def test_info_carries_the_calibration_recipe(built) -> None:
    info = json.loads((built["dst"] / "quantization_info.json").read_text())
    cal = info["calibration"]
    assert info["recipe"]["name"] == "nvfp4"
    assert cal["scope"] == "routed experts only"
    assert cal["hessian"]["mode"] == "blend" and cal["hessian"]["token_clip"] == 3.0
    assert cal["solver"]["act_order"] is True and cal["solver"]["percdamp"] == 0.01
    assert cal["calibrated_layers"] == [1, 2, 3]
    assert len(cal["calibrated_tensors"]) == 6
    assert cal["partial"] is False
    assert cal["manifest"]["sha256"]
    report_file = built["dst"] / cal["report"]["file"]
    assert report_file.name == "g1lite_report.json"
    assert cal["report"]["sha256"] == __import__("hashlib").sha256(report_file.read_bytes()).hexdigest()


def test_report_has_every_expert_with_tokens_ratios_fallbacks_and_times(built) -> None:
    rep = json.loads((built["dst"] / "g1lite_report.json").read_text())
    assert set(rep["layers"]) == {"1", "2", "3"}
    for layer in rep["layers"].values():
        assert len(layer["experts"]) == 16
        e0 = layer["experts"][0]
        for key in (
            "expert",
            "n_tokens",
            "n_heldout",
            "diag_ratio_up",
            "diag_ratio_down",
            "loss_ratio_up",
            "loss_ratio_down",
            "heldout_err_gptq",
            "heldout_err_rtn",
            "fallback_up",
            "fallback_down",
            "final",
            "reason",
        ):
            assert key in e0
        assert layer["wall_seconds"]["total"] > 0 and layer["wall_seconds"]["forward"] > 0
        assert layer["summary"]["n_gptq"] + layer["summary"]["n_rtn"] == 16
        assert sum(e["n_tokens"] for e in layer["experts"]) > 0
    assert [r["layer"] for r in rep["forward_stats"]] == [0, 1, 2, 3]  # every layer that was run, list-I ones too
    assert rep["totals"]["n_experts"] == 48
    assert rep["totals"]["n_gptq"] >= 1


def test_calibration_lowers_the_hessian_weighted_error_of_the_experts_it_changed(built) -> None:
    rep = json.loads((built["dst"] / "g1lite_report.json").read_text())
    ratios = [
        e["loss_ratio_up"]
        for layer in rep["layers"].values()
        for e in layer["experts"]
        if e["final"] == "gptq" and e["loss_ratio_up"] is not None
    ]
    assert ratios and sum(ratios) / len(ratios) < 1.0


def test_experts_that_fell_back_carry_a_reason_and_the_rtn_codes(built) -> None:
    rep = json.loads((built["dst"] / "g1lite_report.json").read_text())
    for layer in rep["layers"].values():
        for e in layer["experts"]:
            assert (e["final"] == "rtn") == (e["reason"] != "")


def test_partial_build_calibrates_only_the_requested_layers(built, tmp_path: Path) -> None:
    dst = tmp_path / "part"
    assert run_build(built["src"], built["calib"], dst, "--layers", "2") == 0
    info = json.loads((dst / "quantization_info.json").read_text())
    assert info["calibration"]["calibrated_layers"] == [2] and info["calibration"]["partial"] is True
    assert verify.verify(built["src"], dst, calibrated=True).violations == []
    part, n2 = read_all(dst), read_all(built["n2"])
    for r in ROUTED:
        for layer in (1, 3):
            k = f"model.layers.{layer}.{r.removesuffix('.weight')}.weight_packed"
            assert torch.equal(part[k], n2[k])
    full = read_all(built["dst"])
    k2 = f"model.layers.2.{ROUTED[0].removesuffix('.weight')}.weight_packed"
    assert torch.equal(part[k2], full[k2])  # same calibration of layer 2 as in the full run (bf16 propagation)


def test_requesting_a_layer_of_the_ignore_list_skips_it_with_a_note(built, tmp_path: Path) -> None:
    dst = tmp_path / "skip"
    assert run_build(built["src"], built["calib"], dst, "--layers", "0,3") == 0
    info = json.loads((dst / "quantization_info.json").read_text())
    assert info["calibration"]["calibrated_layers"] == [3]
    assert info["calibration"]["skipped_layers"] == {"0": "all routed tensors of the layer are in the ignore list"}


def test_calibrate_only_writes_the_report_and_no_checkpoint(built, tmp_path: Path) -> None:
    dst = tmp_path / "only"
    assert run_build(built["src"], built["calib"], dst, "--layers", "2", "--calibrate-only") == 0
    assert (dst / "g1lite_report.json").is_file()
    assert not list(dst.glob("model-*.safetensors"))


def test_a_run_interrupted_after_a_layer_resumes_and_gives_the_same_checkpoint(
    built, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dst = tmp_path / "resume"
    real = bg.solve_layer

    def boom(info, *a, **kw):
        if info.layer == 3:
            raise RuntimeError("interrupted")
        return real(info, *a, **kw)

    monkeypatch.setattr(bg, "solve_layer", boom)
    with pytest.raises(RuntimeError):
        run_build(built["src"], built["calib"], dst)
    work = dst.with_name(dst.name + ".work")
    assert (work / "layer_01.pt").is_file() and (work / "layer_02.pt").is_file()
    assert not (work / "layer_03.pt").exists()
    monkeypatch.setattr(bg, "solve_layer", real)
    done = []
    monkeypatch.setattr(bg, "solve_layer", lambda info, *a, **kw: (done.append(info.layer), real(info, *a, **kw))[1])
    assert run_build(built["src"], built["calib"], dst) == 0
    assert done == [3]
    a, b = read_all(dst), read_all(built["dst"])
    assert all(torch.equal(raw(a[k]), raw(b[k])) for k in a)


def test_resume_with_other_settings_is_refused(built, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dst = tmp_path / "mix"
    assert run_build(built["src"], built["calib"], dst, "--layers", "1", "--calibrate-only") == 0
    assert run_build(built["src"], built["calib"], dst, "--layers", "2", "--calibrate-only", "--hessian", "routed") == 2


def test_quantized_propagation_also_gives_a_valid_checkpoint(built, tmp_path: Path) -> None:
    dst = tmp_path / "prop"
    assert run_build(built["src"], built["calib"], dst, "--propagate", "quantized") == 0
    assert verify.verify(built["src"], dst, calibrated=True).violations == []
    a, b = read_all(dst), read_all(built["dst"])
    k1 = f"model.layers.1.{ROUTED[0].removesuffix('.weight')}.weight_packed"
    assert torch.equal(a[k1], b[k1])  # layer 1 sees the same bf16 inputs in both modes
    k3 = f"model.layers.3.{ROUTED[0].removesuffix('.weight')}.weight_packed"
    assert not torch.equal(a[k3], b[k3])  # later layers see the quantized layers below them


def test_refuses_a_manifest_that_failed_its_audit(built, tmp_path: Path) -> None:
    calib = tmp_path / "calib"
    shutil.copytree(built["calib"].parent, calib)
    m = json.loads((calib / "manifest.json").read_text())
    m["audit"]["ok"] = False
    (calib / "manifest.json").write_text(json.dumps(m))
    assert run_build(built["src"], calib / "manifest.json", tmp_path / "dst") == 2
    assert not (tmp_path / "dst" / "model-00001-of-00001.safetensors").exists()


def test_refuses_a_modeling_file_that_differs_from_the_vendored_one(built, tmp_path: Path) -> None:
    src = tmp_path / "src"
    shutil.copytree(built["src"], src)
    (src / "modeling_olala.py").write_text((src / "modeling_olala.py").read_text() + "\n# changed\n")
    assert run_build(src, built["calib"], tmp_path / "dst") == 2
    assert run_build(src, built["calib"], tmp_path / "dst2", "--allow-modeling-mismatch") == 0


def test_refuses_a_non_empty_destination(built, tmp_path: Path) -> None:
    dst = tmp_path / "dst"
    dst.mkdir()
    (dst / "keep.txt").write_text("x")
    assert run_build(built["src"], built["calib"], dst) == 2


def test_layer_spec_parsing() -> None:
    assert bg.parse_layers("all", 6) is None
    assert bg.parse_layers("2,4-5", 6) == {2, 4, 5}
    with pytest.raises(ValueError):
        bg.parse_layers("9", 6)
    with pytest.raises(ValueError):
        bg.parse_layers("x", 6)


def test_forward_error_on_fresh_windows_of_the_calibration_distribution_is_lower_than_rtn(built) -> None:
    import tiny_e2e

    fresh = tiny_e2e.fresh_windows(seed=7)
    res = tiny_e2e.compare(built["src"], built["dst"], built["n2"], windows=fresh)
    assert res["n2"]["mean_kl"] > 1e-4  # the model is sensitive enough for the comparison to mean something
    assert res["g1"]["mean_kl"] < res["n2"]["mean_kl"], res
    assert res["g1"]["rel_logit_rmse"] < res["n2"]["rel_logit_rmse"], res


def test_a_heldout_guard_that_cannot_judge_sends_every_expert_to_rtn_and_the_checkpoint_equals_n2(
    built, tmp_path: Path
) -> None:
    dst = tmp_path / "guarded"
    assert run_build(built["src"], built["calib"], dst, "--min-heldout-rows", "1000000") == 0
    rep = json.loads((dst / "g1lite_report.json").read_text())
    assert rep["totals"]["n_gptq"] == 0
    reasons = set(rep["totals"]["reasons"])
    assert "heldout_too_few" in reasons and reasons <= {"heldout_too_few", "inactive"}
    a, b = read_all(dst), read_all(built["n2"])
    assert set(a) == set(b) and all(torch.equal(raw(a[k]), raw(b[k])) for k in a)
    assert verify.verify(built["src"], dst, calibrated=True).violations == []


def test_scales_that_differ_from_the_data_free_ones_are_refused_when_the_checkpoint_is_written(
    built, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_load = bg.G1Handler.load

    def tampered(self, layer):
        data = real_load(self, layer)
        raw_scale = data["tensors"]["up"]["block_scale"].view(torch.uint8).clone()
        raw_scale[0, 0, 0] = raw_scale[0, 0, 0] ^ 1
        data["tensors"]["up"]["block_scale"] = raw_scale.view(torch.float8_e4m3fn)
        return data

    monkeypatch.setattr(bg.G1Handler, "load", tampered)
    with pytest.raises(RuntimeError, match="data-free"):
        run_build(built["src"], built["calib"], tmp_path / "bad")


def test_a_crash_between_saving_a_layer_and_saving_the_state_does_not_solve_that_layer_twice(
    built, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hessians as hs_mod

    dst = tmp_path / "midlayer"
    real_save = hs_mod._atomic_save

    def crash_on_layer_3_state(obj, path):
        if path.name == "hidden_3.pt":  # the state after layer 2: layer 2's result is already on disk
            raise RuntimeError("crash after the layer result was saved")
        return real_save(obj, path)

    monkeypatch.setattr(hs_mod, "_atomic_save", crash_on_layer_3_state)
    with pytest.raises(RuntimeError, match="crash after"):
        run_build(built["src"], built["calib"], dst)
    work = dst.with_name(dst.name + ".work")
    assert (work / "layer_02.pt").is_file()
    assert json.loads((work / "state" / "progress.json").read_text())["next_layer"] == 2
    monkeypatch.setattr(hs_mod, "_atomic_save", real_save)
    real_solve, done = bg.solve_layer, []
    monkeypatch.setattr(
        bg, "solve_layer", lambda info, *a, **kw: (done.append(info.layer), real_solve(info, *a, **kw))[1]
    )
    assert run_build(built["src"], built["calib"], dst) == 0
    assert done == [3]  # layer 2 was run again forward-only, its saved result reused
    a, b = read_all(dst), read_all(built["dst"])
    assert all(torch.equal(raw(a[k]), raw(b[k])) for k in a)
