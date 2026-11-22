"""``--scale-search`` in the converter and the verifier, and ``experts-all`` end to end (tiny model).

The fake model has 4 layers ``MMVM``; its list I analogue (``IGNORE``: the MoE of layer 0 and the mixer_proj
of layer 1) has one pattern that covers routed experts (lifted by ``experts-all``) and one that does not
(kept).
"""

import json
from pathlib import Path

import fakes
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from olala_quant import ablation, nvfp4, verify
from olala_quant import convert as conv
from olala_quant.checkpoint_io import SourceCheckpoint

IGNORE = ("model.layers.0.mlp.*", "model.layers.1.mixer_proj.weight")
EXPECTED_BLOCK = {"factors": [1.0, 0.95, 0.9, 0.85, 0.8, 0.75], "version": 1}


@pytest.fixture
def src(tmp_path: Path) -> Path:
    d = tmp_path / "src"
    fakes.write_fake_checkpoint(d)
    return d


def _convert(src: Path, dst: Path, recipe: str = "nvfp4", select: str | None = None, **kw) -> dict:
    return conv.convert(
        src, dst, recipe, IGNORE, max_shard_bytes=60_000, log=lambda *_: None, select=select, **kw
    )


def _read_all(directory: Path) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for f in sorted(directory.glob("model-*.safetensors")):
        with safe_open(f, framework="pt") as h:
            for k in h.keys():  # noqa: SIM118
                out[k] = h.get_tensor(k)
    return out


def _quantized_names(dst: Path) -> set[str]:
    return {
        n.removesuffix(".weight_scale") + ".weight" for n in _read_all(dst) if n.endswith(".weight_scale")
    }


def _raw(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().reshape(-1).view(torch.uint8)


def _rewrite(dst: Path, edit) -> None:
    """Load every output tensor, let ``edit`` change them, write the shards back in the same layout."""
    index = json.loads((dst / "model.safetensors.index.json").read_text())
    for f in sorted(set(index["weight_map"].values())):
        with safe_open(dst / f, framework="pt") as h:
            part = {k: h.get_tensor(k) for k in h.keys()}  # noqa: SIM118
        edit(part)
        save_file(part, str(dst / f), metadata={"format": "pt"})


# ------------------------------------------------------------------------------------------------ the
# converter


def test_the_search_is_recorded_in_quantization_info_and_only_when_it_ran(src: Path, tmp_path: Path) -> None:
    info = _convert(src, tmp_path / "ss", scale_search=True)
    assert info["scale_search"] == EXPECTED_BLOCK
    assert (
        json.loads((tmp_path / "ss" / "quantization_info.json").read_text())["scale_search"] == EXPECTED_BLOCK
    )
    assert list(nvfp4.SCALE_SEARCH_FACTORS) == EXPECTED_BLOCK["factors"]
    assert EXPECTED_BLOCK["version"] == nvfp4.SCALE_SEARCH_VERSION
    plain = _convert(src, tmp_path / "plain")
    assert "scale_search" not in plain
    assert "scale_search" not in json.loads((tmp_path / "plain" / "quantization_info.json").read_text())
    assert _convert(src, tmp_path / "off", scale_search=False).get("scale_search") is None
    assert "scale_search" not in json.loads((tmp_path / "off" / "quantization_info.json").read_text())


def test_a_dry_run_records_it_too(src: Path, tmp_path: Path) -> None:
    info = _convert(src, tmp_path / "dst", scale_search=True, dry_run=True)
    assert info["scale_search"] == EXPECTED_BLOCK and not (tmp_path / "dst").exists()


def test_the_layout_and_the_size_are_those_of_the_plain_conversion_and_only_the_scales_and_codes_differ(
    src: Path, tmp_path: Path
) -> None:
    plain_info = _convert(src, tmp_path / "plain")
    ss_info = _convert(src, tmp_path / "ss", scale_search=True)
    plain, ss = _read_all(tmp_path / "plain"), _read_all(tmp_path / "ss")
    assert set(plain) == set(ss)
    differing = set()
    for name, a in plain.items():
        b = ss[name]
        assert (a.dtype, a.shape) == (b.dtype, b.shape), name
        if not torch.equal(_raw(a), _raw(b)):
            differing.add(name)
    assert differing, "the search changed nothing on random data: it is not running"
    for (
        name
    ) in differing:  # only quantized block scales and packed codes may differ; the global scale never does
        assert name.endswith((".weight_scale", ".weight_packed")), name
    for name in plain:
        if name.endswith(".weight_global_scale"):
            assert torch.equal(plain[name], ss[name]), name
        if not name.endswith((".weight_scale", ".weight_packed", ".weight_global_scale")):
            assert torch.equal(_raw(plain[name]), _raw(ss[name])), (
                name
            )  # bf16 and fp32 tensors: bit identical
    assert ss_info["output"]["tensor_bytes"] == plain_info["output"]["tensor_bytes"]
    assert ss_info["output"]["file_bytes"] == plain_info["output"]["file_bytes"]
    assert ss_info["output"]["n_shards"] == plain_info["output"]["n_shards"]
    assert (
        ss_info["quantization_config"] == plain_info["quantization_config"]
    )  # the loader sees the same config
    assert json.loads((tmp_path / "ss" / "config.json").read_text()) == json.loads(
        (tmp_path / "plain" / "config.json").read_text()
    )


def test_the_search_lowers_the_recorded_error_of_every_tensor_or_leaves_it(src: Path, tmp_path: Path) -> None:
    plain = _convert(src, tmp_path / "plain")["tensors"]
    ss = _convert(src, tmp_path / "ss", scale_search=True)["tensors"]
    assert set(plain) == set(ss)
    for name in plain:
        assert ss[name]["rel_fro_error"] <= plain[name]["rel_fro_error"] * (1 + 1e-6), name
        assert set(ss[name]) == set(plain[name]), name  # same record keys: verify reads them
    assert any(ss[n]["rel_fro_error"] < plain[n]["rel_fro_error"] for n in plain)


def test_fp8_and_a_calibration_override_cannot_use_the_search(src: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nvfp4"):
        _convert(src, tmp_path / "a", "fp8", scale_search=True)
    with pytest.raises(ValueError, match="override"):
        _convert(src, tmp_path / "b", scale_search=True, override=lambda *_: None)
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


def test_the_command_line_flag(src: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ignore = tmp_path / "ign.json"
    ignore.write_text(json.dumps(list(IGNORE)))
    base = ["--src", str(src), "--ignore-file", str(ignore), "--max-shard-gb", "0.00006"]
    assert conv.main([*base, "--recipe", "nvfp4", "--dst", str(tmp_path / "a"), "--scale-search"]) == 0
    assert (
        json.loads((tmp_path / "a" / "quantization_info.json").read_text())["scale_search"] == EXPECTED_BLOCK
    )
    assert conv.main([*base, "--recipe", "nvfp4", "--dst", str(tmp_path / "b")]) == 0
    assert "scale_search" not in json.loads((tmp_path / "b" / "quantization_info.json").read_text())
    capsys.readouterr()
    assert conv.main([*base, "--recipe", "fp8", "--dst", str(tmp_path / "c"), "--scale-search"]) == 2
    assert "nvfp4" in capsys.readouterr().err and not (tmp_path / "c").exists()
    with pytest.raises(SystemExit):
        conv.main(["--help"])
    help_text = capsys.readouterr().out
    assert "--scale-search" in help_text and "experts-all" in help_text


def test_the_search_combines_with_a_selection(src: Path, tmp_path: Path) -> None:
    info = _convert(src, tmp_path / "dst", select="experts", scale_search=True)
    assert info["scale_search"] == EXPECTED_BLOCK and info["selection"]["spec"] == "experts"
    assert verify.verify(src, tmp_path / "dst", select="experts").violations == []


# ------------------------------------------------------------------------------------------------ experts-all


def test_experts_all_quantizes_every_layers_routed_experts_and_keeps_the_rest_of_list_i(
    src: Path, tmp_path: Path
) -> None:
    dst = tmp_path / "dst"
    info = _convert(src, dst, select="experts-all")
    assert _quantized_names(dst) == {
        f"model.layers.{i}.mlp.experts.{t}.weight" for i in range(4) for t in ("experts", "output_experts")
    }
    out, source = _read_all(dst), SourceCheckpoint(src)
    for name, original in source.iter_tensors():
        if name in _quantized_names(dst):
            continue
        assert torch.equal(_raw(out[name]), _raw(original)), name  # everything else bf16, bit identical
    sel = info["selection"]
    assert sel["spec"] == "experts-all" and sel["list_i_lifted"] is True
    assert sel["lifted_ignore_patterns"] == ["model.layers.0.mlp.*"]
    assert sel["kept_ignore_patterns"] == ["model.layers.1.mixer_proj.weight"]
    assert sel["base_ignore_patterns"] == list(IGNORE)
    assert sel["expected"]["quantized_layers"] == 4 and sel["expected"]["quantized_tensors"] == 8
    cfg = json.loads((dst / "config.json").read_text())["quantization_config"]["ignore"]
    assert "model.layers.0.mlp.experts" not in cfg and "model.layers.1.mixer_proj" in cfg
    assert "model.layers.0.mlp.shared_experts.up_proj" in cfg  # layer 0's other MLP tensors stay bf16
    assert info["ignore"]["patterns"][0] == "model.layers.1.mixer_proj.weight"
    assert "model.layers.0.mlp.*" not in info["ignore"]["patterns"]
    want = ablation.expected_counts(source.header(), "experts-all", IGNORE)
    assert sel["expected"] == want


def test_experts_all_verifies_and_is_checked_by_ablation_check(src: Path, tmp_path: Path) -> None:
    dst = tmp_path / "dst"
    _convert(src, dst, select="experts-all", scale_search=True)
    assert verify.verify(src, dst).violations == []
    assert verify.verify(src, dst, select="experts-all").violations == []
    wrong = verify.verify(src, dst, select="experts")
    assert {v.check for v in wrong.violations} == {"info_selection"}
    assert ablation.check_checkpoint(src, dst, "experts-all", IGNORE) == []
    assert ablation.check_checkpoint(src, dst, "experts", IGNORE) != []


def test_verify_rederives_the_lifting_and_catches_a_record_that_hides_it(src: Path, tmp_path: Path) -> None:
    dst = tmp_path / "dst"
    _convert(src, dst, select="experts-all")
    path = dst / "quantization_info.json"
    good = json.loads(path.read_text())

    def tampered(edit) -> set[str]:
        info = json.loads(json.dumps(good))
        edit(info)
        path.write_text(json.dumps(info))
        checks = {v.check for v in verify.verify(src, dst).violations}
        path.write_text(json.dumps(good))
        return checks

    assert verify.verify(src, dst).violations == []
    assert "info_selection" in tampered(lambda i: i["selection"].update(lifted_ignore_patterns=[]))
    assert "info_selection" in tampered(lambda i: i["selection"].pop("list_i_lifted"))
    assert "info_selection" in tampered(lambda i: i["selection"].update(kept_ignore_patterns=list(IGNORE)))
    assert "info_selection" in tampered(lambda i: i["ignore"]["patterns"].insert(0, "model.layers.0.mlp.*"))


def test_the_prediction_is_what_the_converter_wrote_for_experts_all(src: Path, tmp_path: Path) -> None:
    dst = tmp_path / "dst"
    info = _convert(src, dst, select="experts-all:layers=0-1")
    names = _quantized_names(dst)
    assert names == {
        f"model.layers.{i}.mlp.experts.{t}.weight" for i in (0, 1) for t in ("experts", "output_experts")
    }
    assert info["selection"]["expected"]["quantized_tensors"] == len(names)
    assert info["selection"]["spec"] == "experts-all:layers=0-1"
    assert verify.verify(src, dst, select="experts-all:layers=1,0").violations == []


def test_the_command_line_takes_experts_all(
    src: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ignore = tmp_path / "ign.json"
    ignore.write_text(json.dumps(list(IGNORE)))
    base = ["--src", str(src), "--recipe", "nvfp4", "--ignore-file", str(ignore), "--max-shard-gb", "0.00006"]
    assert conv.main([*base, "--dst", str(tmp_path / "a"), "--select", "experts-all", "--scale-search"]) == 0
    info = json.loads((tmp_path / "a" / "quantization_info.json").read_text())
    assert info["selection"]["spec"] == "experts-all" and info["scale_search"] == EXPECTED_BLOCK
    assert verify.main(["--src", str(src), "--dst", str(tmp_path / "a"), "--select", "experts-all"]) == 0
    capsys.readouterr()
    # --no-ignore would drop list I altogether: still refused with a selection
    assert conv.main([*base[:4], "--no-ignore", "--dst", str(tmp_path / "b"), "--select", "experts-all"]) == 2


# ------------------------------------------------------------------------------------------------ the
# verifier


def _searched(src: Path, tmp_path: Path, **kw) -> Path:
    dst = tmp_path / "dst"
    _convert(src, dst, scale_search=True, **kw)
    return dst


def _checks(report: verify.VerifyReport) -> set[str]:
    return {v.check for v in report.violations}


def test_verify_does_not_flag_a_checkpoint_made_with_the_search(src: Path, tmp_path: Path) -> None:
    dst = _searched(src, tmp_path)
    report = verify.verify(src, dst)
    assert report.violations == []
    assert report.stats["n_quantized"] == 4 * 7 + 1 - 7
    assert verify.main(["--src", str(src), "--dst", str(dst)]) == 0


def test_the_stored_block_scales_of_a_searched_checkpoint_are_not_all_448_at_the_top(
    src: Path, tmp_path: Path
) -> None:
    """Why the strict plain-path checks (top block scale 448; half-step bound) cannot apply to it."""
    dst = _searched(src, tmp_path)
    out = _read_all(dst)
    tops = [
        float(t.to(torch.float32).amax(dim=tuple(range(1, t.ndim)))[i])
        if t.ndim == 3
        else float(t.to(torch.float32).max())
        for n, t in out.items()
        if n.endswith(".weight_scale")
        for i in range(t.shape[0] if t.ndim == 3 else 1)
    ]
    assert tops and max(tops) <= 448.0
    assert any(t < 448.0 for t in tops), (
        "on random data some tensor's top block must have chosen a smaller factor"
    )


def test_verify_requires_or_forbids_the_search_when_asked(src: Path, tmp_path: Path) -> None:
    dst = _searched(src, tmp_path)
    assert verify.verify(src, dst, scale_search=True).violations == []
    assert _checks(verify.verify(src, dst, scale_search=False)) == {"info_scale_search"}
    plain = tmp_path / "plain"
    _convert(src, plain)
    assert verify.verify(src, plain, scale_search=False).violations == []
    assert _checks(verify.verify(src, plain, scale_search=True)) == {"info_scale_search"}
    argv = ["--src", str(src), "--dst"]
    assert verify.main([*argv, str(dst), "--scale-search"]) == 0
    assert verify.main([*argv, str(dst), "--no-scale-search"]) == 1
    assert verify.main([*argv, str(plain), "--scale-search"]) == 1
    assert verify.main([*argv, str(plain), "--no-scale-search"]) == 0
    with pytest.raises(SystemExit):
        verify.main([*argv, str(dst), "--scale-search", "--no-scale-search"])


def test_searched_scales_in_a_checkpoint_that_does_not_say_so_are_still_caught(
    src: Path, tmp_path: Path
) -> None:
    dst = _searched(src, tmp_path)
    path = dst / "quantization_info.json"
    info = json.loads(path.read_text())
    del info["scale_search"]
    path.write_text(json.dumps(info))
    checks = _checks(verify.verify(src, dst))
    assert checks & {"scale_range", "reconstruction_error"}


@pytest.mark.parametrize(
    "edit",
    [
        lambda i: i["scale_search"].update(version=2),
        lambda i: i["scale_search"].update(factors=[1.0, 0.9]),
        lambda i: i.update(scale_search={}),
        lambda i: i.update(scale_search="yes"),
    ],
    ids=["version", "factors", "empty", "not-a-dict"],
)
def test_a_recorded_search_this_tool_does_not_know_is_refused(src: Path, tmp_path: Path, edit) -> None:
    dst = _searched(src, tmp_path)
    path = dst / "quantization_info.json"
    info = json.loads(path.read_text())
    edit(info)
    path.write_text(json.dumps(info))
    assert "info_scale_search" in _checks(verify.verify(src, dst))


def test_a_block_scale_that_is_none_of_the_six_candidates_is_caught(src: Path, tmp_path: Path) -> None:
    dst = _searched(src, tmp_path)
    victim = "model.layers.2.mlp.shared_experts.fc_1.weight_scale"

    def damage(part: dict[str, torch.Tensor]) -> None:
        if victim in part:
            scale = part[victim].to(torch.float32)
            scale[0, 0] = scale[0, 0] * 0.4  # below the smallest factor (0.75): no candidate of that group
            part[victim] = scale.to(torch.float8_e4m3fn)

    _rewrite(dst, damage)
    report = verify.verify(src, dst)
    assert "scale_search_scale" in _checks(report)


def test_a_candidate_scale_that_is_worse_than_the_plain_one_is_caught(src: Path, tmp_path: Path) -> None:
    """Every block scale replaced by a REAL candidate (factor 0.75), the codes kept: legal, but worse."""
    dst = _searched(src, tmp_path)
    victim = "model.layers.2.mlp.shared_experts.fc_1"
    original = SourceCheckpoint(src).get_tensor(f"{victim}.weight").to(torch.float32)
    plain = nvfp4.quantize_nvfp4(original)
    rows, cols = original.shape
    g = plain.global_scale.reshape(1, 1, 1, 1)
    blocks = original.reshape(1, rows, cols // 16, 16)
    base = g * (blocks.abs().amax(dim=-1, keepdim=True) * (1.0 / 6.0))
    low = nvfp4._candidate_scale(base, 0.75).reshape(rows, cols // 16)

    def damage(part: dict[str, torch.Tensor]) -> None:
        key = f"{victim}.weight_scale"
        if key in part:
            part[key] = low

    _rewrite(dst, damage)
    checks = _checks(verify.verify(src, dst))
    assert "scale_search_error" in checks
    assert (
        "scale_search_scale" not in checks
    )  # every scale is a legal candidate: only the error check can see it


def test_a_searched_checkpoint_cannot_be_verified_as_calibrated(src: Path, tmp_path: Path) -> None:
    dst = _searched(src, tmp_path)
    assert "info_scale_search" in _checks(verify.verify(src, dst, calibrated=True))


def test_the_plain_checks_are_unchanged_for_a_plain_checkpoint(src: Path, tmp_path: Path) -> None:
    """The strict half-step bound and the 448 top scale still hold without the search."""
    plain = tmp_path / "plain"
    _convert(src, plain)
    assert verify.verify(src, plain).violations == []
    victim = "model.layers.2.mlp.shared_experts.fc_1.weight_scale"

    def damage(part: dict[str, torch.Tensor]) -> None:
        if victim in part:
            part[victim] = (part[victim].to(torch.float32) * 0.5).to(torch.float8_e4m3fn)

    _rewrite(plain, damage)
    assert _checks(verify.verify(src, plain)) & {"scale_range", "reconstruction_error"}
