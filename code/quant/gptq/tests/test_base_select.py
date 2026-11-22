"""``build_g1lite.py --base-select``: calibrated routed experts on a bf16 base (spec ``experts``), CAL-1 directories."""

import json
import shutil
from pathlib import Path

import pytest
import torch
from olala_quant import convert as conv
from olala_quant import verify
from olala_quant.checkpoint_io import SourceCheckpoint
from safetensors import safe_open

import build_g1lite as bg
import tiny_model
from tests_support import make_cal1_layout, make_calibration

pytestmark = pytest.mark.slow

ROUTED_SUFFIXES = ("mlp.experts.experts", "mlp.experts.output_experts")


def build(src: Path, calib: Path, dst: Path, *extra: str) -> int:
    ignore = dst.parent / "ignore.json"
    ignore.write_text(json.dumps(list(tiny_model.TINY_IGNORE)))
    return bg.main(
        [
            "--src", str(src), "--calib", str(calib), "--dst", str(dst), "--ignore-file", str(ignore),
            "--heldout-every", "4", "--batch-windows", "4", *extra,
        ]
    )  # fmt: skip


def read_all(directory: Path) -> dict[str, torch.Tensor]:
    out = {}
    for f in sorted(directory.glob("model-*.safetensors")):
        with safe_open(f, framework="pt") as h:
            out.update({k: h.get_tensor(k) for k in h.keys()})  # noqa: SIM118
    return out


def raw(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().reshape(-1).view(torch.uint8)


def is_routed(name: str) -> bool:
    return any(s in name for s in ROUTED_SUFFIXES)


@pytest.fixture(scope="module")
def made(tiny_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> dict:
    tmp = tmp_path_factory.mktemp("bs")
    calib = make_calibration(tmp)
    dst = tmp / "g1x"
    assert build(tiny_dir, calib, dst, "--base-select", "experts") == 0
    n5 = tmp / "n5"
    conv.convert(tiny_dir, n5, "nvfp4", tiny_model.TINY_IGNORE, select="experts", log=lambda _s: None)
    return {"tmp": tmp, "calib": calib, "dst": dst, "n5": n5, "src": tiny_dir}


def test_the_experts_base_verifies_in_calibrated_mode_with_the_selection_required(made) -> None:
    rep = verify.verify(made["src"], made["dst"], calibrated=True, select="experts")
    assert rep.violations == []
    assert rep.stats["n_quantized"] == 6  # the routed tensors of layers 1 to 3, nothing else


def test_verify_refuses_the_checkpoint_when_another_selection_is_required(made) -> None:
    rep = verify.verify(made["src"], made["dst"], calibrated=True, select="experts:layers=1-2")
    assert {v.check for v in rep.violations} == {"info_selection"}
    rep = verify.verify(made["src"], made["dst"], calibrated=True, select="non-experts")
    assert {v.check for v in rep.violations} == {"info_selection"}


def test_every_tensor_outside_the_routed_experts_is_the_bf16_source_bit_for_bit(made) -> None:
    src = SourceCheckpoint(made["src"])
    got = read_all(made["dst"])
    for name in src.names():
        if is_routed(name) and not name.startswith("model.layers.0."):
            continue
        assert name in got, name
        assert got[name].dtype == src.get_tensor(name).dtype
        assert torch.equal(raw(got[name]), raw(src.get_tensor(name))), name
    quantized = {n for n in got if n.endswith("weight_packed")}
    assert len(quantized) == 6 and all(is_routed(n) for n in quantized)
    assert not any(n.endswith(("weight_scale", "weight_global_scale")) and not is_routed(n) for n in got)


def test_the_ignore_list_config_and_layout_are_exactly_those_of_convert_select_experts(made) -> None:
    a, b = made["dst"], made["n5"]
    assert (a / "config.json").read_bytes() == (b / "config.json").read_bytes()
    ia, ib = (json.loads((d / "quantization_info.json").read_text()) for d in (a, b))
    assert ia["quantization_config"]["ignore"] == ib["quantization_config"]["ignore"]
    assert len(ia["quantization_config"]["ignore"]) > 1
    assert ia["ignore"] == ib["ignore"]
    assert ia["selection"] == ib["selection"]
    assert ia["selection"]["spec"] == "experts"
    for key in ("categories", "totals_by_category", "recipe", "unclassified"):
        assert ia[key] == ib[key], key
    ta, tb = read_all(a), read_all(b)
    assert set(ta) == set(tb)
    assert (
        json.loads((a / "model.safetensors.index.json").read_text())["weight_map"]
        == json.loads((b / "model.safetensors.index.json").read_text())["weight_map"]
    )


def test_scales_and_non_routed_tensors_equal_n5_and_the_routed_codes_changed(made) -> None:
    ta, tb = read_all(made["dst"]), read_all(made["n5"])
    changed = 0
    for name, t in ta.items():
        if name.endswith("weight_packed"):
            changed += int(not torch.equal(t, tb[name]))
        else:
            assert torch.equal(raw(t), raw(tb[name])), name
    assert changed >= 2


def test_info_records_the_base_selection_in_the_calibration_block(made) -> None:
    cal = json.loads((made["dst"] / "quantization_info.json").read_text())["calibration"]
    assert cal["base_select"] == "experts"
    assert cal["non_expert_tensors"] == "bf16, listed in quantization_config.ignore"
    assert cal["scope"] == "routed experts only"
    assert cal["calibrated_layers"] == [1, 2, 3] and cal["partial"] is False


def test_the_report_names_the_base_selection(made) -> None:
    rep = json.loads((made["dst"] / "g1lite_report.json").read_text())
    assert rep["base_select"] == "experts"


def test_a_layer_range_quantizes_and_calibrates_only_those_layers(made, tmp_path: Path) -> None:
    dst = tmp_path / "g1x23"
    assert build(made["src"], made["calib"], dst, "--base-select", "experts:layers=2-3") == 0
    info = json.loads((dst / "quantization_info.json").read_text())
    assert info["selection"]["spec"] == "experts:layers=2-3"
    assert info["calibration"]["calibrated_layers"] == [2, 3] and info["calibration"]["partial"] is False
    assert info["calibration"]["base_select"] == "experts:layers=2-3"
    got = read_all(dst)
    assert {n.split(".")[2] for n in got if n.endswith("weight_packed")} == {"2", "3"}
    assert "model.layers.1.mlp.experts.experts.weight" in got  # layer 1 routed experts stay bf16
    assert verify.verify(made["src"], dst, calibrated=True, select="experts:layers=2-3").violations == []


def test_the_spec_is_recorded_in_its_canonical_spelling(made, tmp_path: Path) -> None:
    dst = tmp_path / "canon"
    assert build(made["src"], made["calib"], dst, "--base-select", "experts:layers=3,2") == 0
    info = json.loads((dst / "quantization_info.json").read_text())
    assert info["selection"]["spec"] == "experts:layers=2-3" == info["calibration"]["base_select"]


@pytest.mark.parametrize("spec", ["all", "non-experts", "shared-experts", "latent", "attention", "mamba"])
def test_a_spec_that_would_quantize_a_non_expert_tensor_is_refused_before_any_work(
    made, tmp_path: Path, spec: str, capsys: pytest.CaptureFixture[str]
) -> None:
    dst = tmp_path / "refused"
    assert build(made["src"], made["calib"], dst, "--base-select", spec) == 2
    err = capsys.readouterr().err
    assert "only routed experts are calibrated" in err and spec in err
    assert not dst.exists() and not dst.with_name("refused.work").exists()


@pytest.mark.parametrize("spec", ["bogus", "experts:layers=9", "experts:layers=3-2", "experts:foo=1"])
def test_a_malformed_or_out_of_range_spec_is_refused(made, tmp_path: Path, spec: str) -> None:
    assert build(made["src"], made["calib"], tmp_path / "bad", "--base-select", spec) == 2
    assert not (tmp_path / "bad").exists()


def test_base_select_cannot_be_combined_with_no_ignore(
    made, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = bg.main(
        ["--src", str(made["src"]), "--calib", str(made["calib"]), "--dst", str(tmp_path / "x"),
         "--no-ignore", "--base-select", "experts"]
    )  # fmt: skip
    assert code == 2 and "--no-ignore" in capsys.readouterr().err


def test_the_default_base_still_quantizes_the_non_experts_data_free(made, tmp_path: Path) -> None:
    full = tmp_path / "default-full"
    assert build(made["src"], made["calib"], full, "--layers", "2") == 0
    got = read_all(full)
    assert "model.layers.2.mixer_proj.weight_packed" in got  # a non-expert tensor, quantized in the N2 base
    info = json.loads((full / "quantization_info.json").read_text())
    assert "selection" not in info and "base_select" not in info["calibration"]


def test_require_routed_only_names_the_offending_tensors(made) -> None:
    from olala_quant import ablation
    from olala_quant.checkpoint_io import SourceCheckpoint as Src
    from olala_quant.recipes import get_recipe

    header = Src(made["src"]).header()
    ok = ablation.resolve_selection(header, "experts", tiny_model.TINY_IGNORE, get_recipe("nvfp4"))
    bg.require_routed_only(ok.plan, "experts")
    bad = ablation.resolve_selection(header, "all", tiny_model.TINY_IGNORE, get_recipe("nvfp4"))
    with pytest.raises(bg.BaseSelectError, match="only routed experts are calibrated") as e:
        bg.require_routed_only(bad.plan, "all")
    assert "mixer_proj" in str(e.value) or "down_proj" in str(e.value) or "fc_" in str(e.value)


# ------------------------------------------------------------------------------------------------ CAL-1 windows


@pytest.fixture(scope="module")
def cal1(tmp_path_factory: pytest.TempPathFactory):
    return make_cal1_layout(tmp_path_factory.mktemp("cal1"))


def test_a_cal1_output_directory_is_accepted_as_calib_and_gives_the_same_checkpoint_as_its_manifest(
    made, cal1, tmp_path: Path
) -> None:
    a, b = tmp_path / "by-dir", tmp_path / "by-file"
    assert build(made["src"], cal1.directory, a, "--base-select", "experts") == 0
    assert build(made["src"], cal1.manifest, b, "--base-select", "experts") == 0
    ta, tb = read_all(a), read_all(b)
    assert all(torch.equal(raw(ta[k]), raw(tb[k])) for k in ta) and set(ta) == set(tb)
    assert verify.verify(made["src"], a, calibrated=True, select="experts").violations == []
    n = json.loads((a / "quantization_info.json").read_text())["calibration"]["manifest"]
    assert n["n_windows"] == json.loads(cal1.manifest.read_text())["n_windows"] and n["window"] == 64
    assert n["path"].endswith("manifest.json")


def test_the_report_gives_the_windows_used_per_domain_for_a_cal1_manifest(made, cal1, tmp_path: Path) -> None:
    dst = tmp_path / "dom"
    assert build(made["src"], cal1.directory, dst, "--base-select", "experts") == 0
    rep = json.loads((dst / "g1lite_report.json").read_text())
    pd = rep["calibration"]["per_domain"]
    assert pd["method"] == "exact" and pd["n_windows_used"] == rep["calibration"]["n_windows"]
    assert set(pd["domains"]) == set(cal1.n_traces)
    for d, r in pd["domains"].items():
        assert r["tokens"] == sum(1 for x in cal1.labels if x == d)
        assert r["windows_touched"] >= 1 and r["window_equivalents"] > 0
    info = json.loads((dst / "quantization_info.json").read_text())
    assert info["calibration"]["manifest"]["per_domain"] == pd


def test_with_max_windows_the_per_domain_counts_cover_the_windows_actually_used(made, cal1, tmp_path: Path) -> None:
    dst = tmp_path / "dom4"
    assert build(made["src"], cal1.directory, dst, "--base-select", "experts", "--max-windows", "8") == 0
    pd = json.loads((dst / "g1lite_report.json").read_text())["calibration"]["per_domain"]
    assert pd["n_windows_used"] == 8
    assert sum(r["tokens"] for r in pd["domains"].values()) == 8 * 64


def test_a_manifest_that_is_not_cal1_has_no_per_domain_record_and_the_default_info_is_unchanged(
    made, tmp_path: Path
) -> None:
    dst = tmp_path / "lite"
    assert build(made["src"], made["calib"], dst, "--base-select", "experts") == 0
    rep = json.loads((dst / "g1lite_report.json").read_text())
    assert rep["calibration"]["per_domain"]["method"] == "unavailable"
    assert "per_domain" not in json.loads((dst / "quantization_info.json").read_text())["calibration"]["manifest"]


def test_a_cal1_directory_without_a_passing_audit_is_refused(made, tmp_path: Path) -> None:
    lay = make_cal1_layout(tmp_path)
    m = json.loads(lay.manifest.read_text())
    m["audit"]["ok"] = False
    lay.manifest.write_text(json.dumps(m))
    assert build(made["src"], lay.directory, tmp_path / "dst", "--base-select", "experts") == 2
    assert not (tmp_path / "dst").exists()


def test_a_cal1_directory_whose_audit_has_hits_or_is_missing_is_refused(made, tmp_path: Path) -> None:
    for i, edit in enumerate((lambda a: a.update(n_hits=3), lambda a: a.clear())):
        lay = make_cal1_layout(tmp_path / f"c{i}")
        m = json.loads(lay.manifest.read_text())
        edit(m["audit"])
        lay.manifest.write_text(json.dumps(m))
        assert build(made["src"], lay.directory, tmp_path / f"dst{i}", "--base-select", "experts") == 2


def test_a_cal1_directory_whose_windows_changed_is_refused(made, tmp_path: Path) -> None:
    lay = make_cal1_layout(tmp_path)
    w = json.loads((lay.directory / "windows.json").read_text())
    w[0][5] = (w[0][5] + 1) % 200
    (lay.directory / "windows.json").write_text(json.dumps(w))
    assert build(made["src"], lay.directory, tmp_path / "dst", "--base-select", "experts") == 2


def test_a_directory_without_a_manifest_is_refused(made, tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert build(made["src"], empty, tmp_path / "dst") == 2


def test_a_copy_of_the_cal1_directory_elsewhere_works_the_same(made, cal1, tmp_path: Path) -> None:
    copy = tmp_path / "copy"
    shutil.copytree(cal1.directory, copy)
    assert build(made["src"], copy, tmp_path / "dst", "--base-select", "experts") == 0


def test_the_note_for_a_requested_layer_that_is_not_calibrated_depends_on_the_base(made, tmp_path: Path) -> None:
    notes = {}
    for key, spec in (("default", None), ("selected", "experts")):
        extra = ["--layers", "0,3", *(["--base-select", spec] if spec else [])]
        assert build(made["src"], made["calib"], tmp_path / key, *extra) == 0
        info = json.loads((tmp_path / key / "quantization_info.json").read_text())
        notes[key] = info["calibration"]["skipped_layers"]
    assert notes["default"] == {"0": "all routed tensors of the layer are in the ignore list"}
    assert notes["selected"] == {
        "0": "its routed tensors are not quantized under the base selection or are in the ignore list"
    }
