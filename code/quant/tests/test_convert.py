"""Tests for olala_quant.convert on tiny fake checkpoints with the real tensor names."""

import json
import re
from pathlib import Path

import fakes
import pytest
import torch
from safetensors import safe_open

from olala_quant import convert as conv
from olala_quant import fp8, nvfp4, recipes
from olala_quant.checkpoint_io import SourceCheckpoint
from olala_quant.plan import IgnoreListError, build_plan

IGNORE = ("model.layers.0.mlp.*", "model.layers.1.mixer_proj.weight")


def _read_all(directory: Path) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for f in sorted(directory.glob("model-*.safetensors")):
        with safe_open(f, framework="pt") as h:
            for k in h.keys():  # noqa: SIM118 - safetensors handle, not a dict
                out[k] = h.get_tensor(k)
    return out


@pytest.fixture
def src(tmp_path: Path) -> Path:
    d = tmp_path / "src"
    fakes.write_fake_checkpoint(d)
    return d


def test_plan_shards_is_greedy_ordered_and_never_splits_a_tensor() -> None:
    sizes = [("a", 40), ("b", 40), ("c", 40), ("d", 200), ("e", 10)]
    shards = conv.plan_shards(sizes, max_bytes=100)
    assert shards == [["a", "b"], ["c"], ["d"], ["e"]]
    assert conv.plan_shards([], max_bytes=100) == []
    assert conv.plan_shards([("x", 5), ("y", 5)], max_bytes=10) == [["x", "y"]]


def test_fp8_conversion_end_to_end(src: Path, tmp_path: Path) -> None:
    dst = tmp_path / "dst"
    info = conv.convert(src, dst, "fp8", IGNORE, max_shard_bytes=60_000)
    source = SourceCheckpoint(src)
    out = _read_all(dst)
    orig = dict(source.iter_tensors())

    # output tensor set == plan, with the documented dtypes and shapes
    planned = build_plan(SourceCheckpoint(src).header(), recipes.RECIPES["fp8"], IGNORE)
    assert set(out) == {o.name for o in planned.output_tensors()}
    assert {n: (str(o.shape)) for n, o in ((o.name, o) for o in planned.output_tensors())} == {
        n: str(tuple(t.shape)) for n, t in out.items()
    }
    w = out["model.layers.2.mlp.experts.experts.weight"]
    s = out["model.layers.2.mlp.experts.experts.weight_scale"]
    assert w.dtype == torch.float8_e4m3fn and w.shape == (
        fakes.N_EXPERTS,
        fakes.EXPERT_FF,
        fakes.LATENT,
    )
    assert s.dtype == torch.float32 and s.shape == (fakes.N_EXPERTS, fakes.EXPERT_FF, 1)
    # quantizer output matches the library function exactly
    ref = fp8.quantize_fp8(orig["model.layers.2.mlp.experts.experts.weight"])
    assert torch.equal(w.view(torch.uint8), ref.weight.view(torch.uint8))
    assert torch.equal(s, ref.scale)

    # ignored and never-quantized tensors are bit-identical
    for name in info["ignore"]["quantizable_ignored"]:
        assert out[name].dtype == torch.bfloat16
        assert torch.equal(out[name].view(torch.uint8), orig[name].view(torch.uint8))
    assert "model.layers.0.mlp.experts.experts.weight_scale" not in out
    for name in ("model.embedding.weight", "lm_head.weight", "model.layers.0.mixer.in_proj.weight"):
        assert torch.equal(out[name].view(torch.uint8), orig[name].view(torch.uint8))
    assert out["model.layers.0.mlp.moe_gate.weight"].dtype == torch.float32
    assert torch.equal(out["model.layers.0.mlp.moe_gate.weight"], orig["model.layers.0.mlp.moe_gate.weight"])

    # sharding: several files, each within the cap, an index that covers every tensor
    shard_files = sorted(dst.glob("model-*-of-*.safetensors"))
    assert len(shard_files) > 1
    index = json.loads((dst / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(out)
    assert index["metadata"]["total_size"] == info["output"]["tensor_bytes"]
    for f in shard_files:
        assert re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", f.name)
        n_tensors = len([k for k, v in index["weight_map"].items() if v == f.name])
        assert n_tensors >= 1
        assert f.stat().st_size <= 60_000 + 4096 or n_tensors == 1


def test_config_and_auxiliary_files(src: Path, tmp_path: Path) -> None:
    dst = tmp_path / "dst"
    info = conv.convert(src, dst, "nvfp4", IGNORE)
    cfg = json.loads((dst / "config.json").read_text())
    src_cfg = json.loads((src / "config.json").read_text())
    qc = cfg.pop("quantization_config")
    assert cfg == src_cfg
    assert qc["format"] == "nvfp4-pack-quantized"
    assert qc["ignore"] == info["ignore"]["vllm_ignore"]
    assert "lm_head" in qc["ignore"]
    for fname in fakes.AUX_FILES:
        assert (dst / fname).read_bytes() == (src / fname).read_bytes(), fname
    assert not (dst / "model.safetensors").exists()


def test_nvfp4_conversion_layout_and_error(src: Path, tmp_path: Path) -> None:
    dst = tmp_path / "dst"
    info = conv.convert(src, dst, "nvfp4", ())
    out = _read_all(dst)
    base = "model.layers.3.mlp.experts.output_experts"
    assert out[f"{base}.weight_packed"].dtype == torch.uint8
    assert out[f"{base}.weight_packed"].shape == (
        fakes.N_EXPERTS,
        fakes.LATENT,
        fakes.EXPERT_FF // 2,
    )
    assert out[f"{base}.weight_scale"].dtype == torch.float8_e4m3fn
    assert out[f"{base}.weight_global_scale"].shape == (fakes.N_EXPERTS,)
    assert f"{base}.weight" not in out
    assert out["model.layers.3.mixer_proj.weight_global_scale"].shape == (1,)
    orig = SourceCheckpoint(src).get_tensor(f"{base}.weight")
    deq = nvfp4.dequantize_nvfp4(
        out[f"{base}.weight_packed"],
        out[f"{base}.weight_scale"],
        out[f"{base}.weight_global_scale"],
    )
    err = (deq - orig.float()).norm() / orig.float().norm()
    rec = info["tensors"][f"{base}.weight"]
    assert rec["rel_fro_error"] == pytest.approx(float(err), rel=1e-4)
    assert 0.05 < rec["rel_fro_error"] < 0.2
    assert info["scale_summary"]["n_overflow"] == 0
    assert info["scale_summary"]["max_block_scale"] <= 448.0


def test_quantization_info_content(src: Path, tmp_path: Path) -> None:
    dst = tmp_path / "dst"
    conv.convert(src, dst, "fp8", IGNORE)
    info = json.loads((dst / "quantization_info.json").read_text())
    assert info["recipe"]["name"] == "fp8"
    assert info["ignore"]["patterns"] == list(IGNORE)
    assert "model.layers.0.mlp.moe_gate.weight" in info["ignore"]["resolved"]
    assert "model.layers.1.mixer_proj.weight" in info["ignore"]["quantizable_ignored"]
    assert "model.layers.0.mlp.moe_gate.weight" not in info["ignore"]["quantizable_ignored"]
    source = SourceCheckpoint(src)
    kind, digest = source.identity_hash()
    assert info["source"]["identity_kind"] == kind
    assert info["source"]["identity_sha256"] == digest
    assert info["source"]["n_tensors"] == len(source.names())
    for key in ("python", "torch", "safetensors", "olala_quant"):
        assert info["tool"][key]
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?\+00:00", info["created_utc"])
    cats = info["categories"]
    assert cats["routed_up"]["count"] == 3  # layers 1..3 (layer 0 ignored)
    assert cats["ignored:routed_up"]["count"] == 1
    assert sum(v["count"] for v in cats.values()) == info["source"]["n_tensors"]
    total_params = info["source"]["total_params"]
    assert info["output"]["bits_per_weight"] == pytest.approx(
        info["output"]["tensor_bytes"] * 8 / total_params
    )
    on_disk = sum(f.stat().st_size for f in dst.glob("model-*.safetensors"))
    assert info["output"]["file_bytes"] == on_disk
    assert on_disk >= info["output"]["tensor_bytes"]
    rec = info["tensors"]["model.layers.2.mixer_proj.weight"]
    assert rec["max_abs_error"] > 0 and 0 < rec["rel_fro_error"] < 0.05
    assert rec["role"] == "mixer_proj"
    assert "model.layers.1.mixer_proj.weight" not in info["tensors"]  # ignored: no error recorded


def test_refuses_ignore_list_that_matches_nothing_and_writes_nothing(src: Path, tmp_path: Path) -> None:
    dst = tmp_path / "dst"
    with pytest.raises(IgnoreListError):
        conv.convert(src, dst, "fp8", ("model.layers.77.*",))
    assert not dst.exists() or not any(dst.iterdir())


def test_refuses_non_empty_destination_and_unimplemented_recipe(src: Path, tmp_path: Path) -> None:
    dst = tmp_path / "dst"
    dst.mkdir()
    (dst / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError):
        conv.convert(src, dst, "fp8", ())
    with pytest.raises(NotImplementedError, match="input_global_scale"):
        conv.convert(src, tmp_path / "other", "nvfp4_w4a4", ())


def test_non_finite_source_aborts_and_cleans_up(tmp_path: Path) -> None:
    src = tmp_path / "src"
    tensors = fakes.write_fake_checkpoint(src)
    from safetensors.torch import save_file

    bad = dict(tensors)
    w = bad["model.layers.1.mixer_proj.weight"].clone()
    w[0, 0] = float("nan")
    bad["model.layers.1.mixer_proj.weight"] = w
    save_file(bad, src / "model.safetensors", metadata={"format": "pt"})
    dst = tmp_path / "dst"
    with pytest.raises(ValueError, match="non-finite"):
        conv.convert(src, dst, "fp8", ())
    assert not list(dst.glob("*.safetensors")) and not (dst / "model.safetensors.index.json").exists()


def test_dry_run_writes_nothing_and_reports_projection(src: Path, tmp_path: Path) -> None:
    dst = tmp_path / "dst"
    info = conv.convert(src, dst, "fp8", IGNORE, dry_run=True)
    assert not dst.exists()
    assert info["output"]["tensor_bytes"] > 0
    assert info["dry_run"] is True


def test_sharded_source_is_read_through_its_index(tmp_path: Path) -> None:
    src = tmp_path / "src"
    fakes.write_fake_checkpoint(src, shards=3)
    dst = tmp_path / "dst"
    info = conv.convert(src, dst, "fp8", ())
    assert info["source"]["identity_kind"] == "index_json"
    assert info["source"]["n_tensors"] == len(fakes.make_tensors())
    assert (dst / "model.safetensors.index.json").is_file()


def test_cli_main_default_ignore_no_ignore_and_ignore_file(
    src: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # --no-ignore
    assert conv.main(["--src", str(src), "--dst", str(tmp_path / "a"), "--recipe", "fp8", "--no-ignore"]) == 0
    info = json.loads((tmp_path / "a" / "quantization_info.json").read_text())
    assert info["ignore"]["patterns"] == []
    # --ignore-file
    f = tmp_path / "ign.json"
    f.write_text(json.dumps(list(IGNORE)))
    assert (
        conv.main(
            [
                "--src",
                str(src),
                "--dst",
                str(tmp_path / "b"),
                "--recipe",
                "nvfp4",
                "--ignore-file",
                str(f),
            ]
        )
        == 0
    )
    info = json.loads((tmp_path / "b" / "quantization_info.json").read_text())
    assert info["ignore"]["patterns"] == list(IGNORE)
    # default list I does not match the 4-layer fake model's layers 5/19/23/7 but does match 0 and 1:
    # patterns for layers that do not exist are an error, which is the point of the check
    assert conv.main(["--src", str(src), "--dst", str(tmp_path / "c"), "--recipe", "fp8"]) == 2
    err = capsys.readouterr().err
    assert "matches no tensor" in err
    # mutually exclusive options and bad recipe
    with pytest.raises(SystemExit):
        conv.main(
            [
                "--src",
                str(src),
                "--dst",
                str(tmp_path / "d"),
                "--recipe",
                "fp8",
                "--no-ignore",
                "--ignore-file",
                str(f),
            ]
        )
    with pytest.raises(SystemExit):
        conv.main(["--src", str(src), "--dst", str(tmp_path / "e"), "--recipe", "int4"])
