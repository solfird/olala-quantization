"""Component ablation through the converter and the verifier, on the tiny fake checkpoint.

The fake model has 4 layers ``MMVM`` (Mamba at 0, 1, 3, attention at 2), so the list I analogue used by the
other converter tests (``IGNORE``: the MoE of layer 0 and the mixer_proj of layer 1) leaves 22 of its 29
quantizable tensors to be quantized.
"""

import hashlib
import json
import re
from pathlib import Path

import fakes
import pytest
import torch
from safetensors import safe_open

from olala_quant import ablation, verify
from olala_quant import convert as conv
from olala_quant.checkpoint_io import SourceCheckpoint

IGNORE = ("model.layers.0.mlp.*", "model.layers.1.mixer_proj.weight")
N_QUANTIZABLE = 4 * 7 + 1

# Characterization of the converter BEFORE the ablation code existed (git 75a917a, torch 2.14.0,
# safetensors 0.8.0): sha256 of the JSON map {file name: sha256 of its bytes} of the whole output folder,
# quantization_info.json with its volatile keys (created_utc, tool, source.directory) removed and the
# rest sorted. A default conversion (no --select) must keep producing exactly this.
GOLDEN_DEFAULT = {
    "nvfp4": "7f00f592e1e09d85b88b5c48a5c92a5efcdbc494d8c4a90467616b710bf2ef9e",
    "fp8": "2fe8255499ff03d0eab807d5d2d83f96bb4eb77afbe12bc1cfba6d532873c4b4",
}
_VOLATILE = ("created_utc", "tool")


def _digests(dst: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for f in sorted(dst.iterdir()):
        if f.name == "quantization_info.json":
            info = json.loads(f.read_text())
            for key in _VOLATILE:
                info.pop(key)
            info["source"].pop("directory")
            out[f.name] = hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()
        else:
            out[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


def _overall(digests: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()


def _read_all(directory: Path) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for f in sorted(directory.glob("model-*.safetensors")):
        with safe_open(f, framework="pt") as h:
            for k in h.keys():  # noqa: SIM118 - safetensors handle, not a dict
                out[k] = h.get_tensor(k)
    return out


def _raw(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().reshape(-1).view(torch.uint8)


def _quantized_names(dst: Path) -> set[str]:
    """Source names of the tensors that were quantized, read from the output tensors only."""
    return {
        n.removesuffix(".weight_scale") + ".weight" for n in _read_all(dst) if n.endswith(".weight_scale")
    }


def _convert(src: Path, dst: Path, recipe: str = "nvfp4", select: str | None = None, **kw) -> dict:
    return conv.convert(
        src, dst, recipe, IGNORE, max_shard_bytes=60_000, log=lambda *_: None, select=select, **kw
    )


@pytest.fixture
def src(tmp_path: Path) -> Path:
    d = tmp_path / "src"
    fakes.write_fake_checkpoint(d)
    return d


# ------------------------------------------------------------------------- the default is untouched


@pytest.mark.parametrize("recipe", ["nvfp4", "fp8"])
def test_a_default_conversion_is_byte_for_byte_what_it_was_before_the_ablation_code(
    src: Path, tmp_path: Path, recipe: str
) -> None:
    info = _convert(src, tmp_path / "dst", recipe)
    digests = _digests(tmp_path / "dst")
    assert _overall(digests) == GOLDEN_DEFAULT[recipe], digests
    assert "selection" not in info
    assert "selection" not in json.loads((tmp_path / "dst" / "quantization_info.json").read_text())


@pytest.mark.parametrize("recipe", ["nvfp4", "fp8"])
def test_the_command_line_without_select_is_the_same_conversion(
    src: Path, tmp_path: Path, recipe: str
) -> None:
    ignore = tmp_path / "ign.json"
    ignore.write_text(json.dumps(list(IGNORE)))
    argv = ["--src", str(src), "--ignore-file", str(ignore), "--recipe", recipe, "--max-shard-gb", "0.00006"]
    assert conv.main([*argv, "--dst", str(tmp_path / "cli")]) == 0
    assert _overall(_digests(tmp_path / "cli")) == GOLDEN_DEFAULT[recipe]


def test_select_all_changes_no_tensor_no_config_and_no_index(src: Path, tmp_path: Path) -> None:
    _convert(src, tmp_path / "plain")
    info = _convert(src, tmp_path / "all", select="all")
    plain, allsel = _digests(tmp_path / "plain"), _digests(tmp_path / "all")
    differing = {k for k in plain if plain[k] != allsel[k]}
    assert differing == {"quantization_info.json"}
    assert info["selection"]["spec"] == "all"
    assert info["selection"]["deselected_tensors"] == []
    assert info["ignore"]["patterns"] == list(IGNORE)


# ------------------------------------------------------------------------- what a selection writes


@pytest.mark.parametrize("recipe", ["nvfp4", "fp8"])
@pytest.mark.parametrize(
    ("spec", "quantized"),
    [
        (
            "experts",
            {
                f"model.layers.{i}.mlp.experts.{t}.weight"
                for i in (1, 2, 3)
                for t in ("experts", "output_experts")
            },
        ),
        (
            "experts:layers=2-3",
            {
                f"model.layers.{i}.mlp.experts.{t}.weight"
                for i in (2, 3)
                for t in ("experts", "output_experts")
            },
        ),
        ("attention", {"model.layers.2.mixer_proj.weight", "model.layers.2.gate_proj.weight"}),
        ("mamba", {"model.layers.0.mixer_proj.weight", "model.layers.3.mixer_proj.weight"}),  # 1 is list I
        (
            "shared-experts",
            {f"model.layers.{i}.mlp.shared_experts.fc_{k}.weight" for i in (1, 2, 3) for k in (1, 2)},
        ),
        (
            "latent",
            {f"model.layers.{i}.mlp.{t}.weight" for i in (1, 2, 3) for t in ("down_proj", "up_proj")},
        ),
    ],
)
def test_only_the_selected_tensors_are_quantized_and_everything_else_is_bit_identical(
    src: Path, tmp_path: Path, recipe: str, spec: str, quantized: set[str]
) -> None:
    dst = tmp_path / "dst"
    info = _convert(src, dst, recipe, spec)
    assert _quantized_names(dst) == quantized
    source, out = SourceCheckpoint(src), _read_all(dst)
    for name, original in source.iter_tensors():
        if name in quantized:
            continue
        assert out[name].dtype == original.dtype, name
        assert torch.equal(_raw(out[name]), _raw(original)), name
    cfg = json.loads((dst / "config.json").read_text())
    ignore = set(cfg["quantization_config"]["ignore"])
    assert ignore == set(info["ignore"]["vllm_ignore"]) == set(info["quantization_config"]["ignore"])
    assert info["selection"]["spec"] == spec
    # every quantizable module that is not in the selection is listed in ignore
    not_quantized = set(info["ignore"]["quantizable_ignored"])
    assert len(not_quantized) + len(quantized) == N_QUANTIZABLE
    assert not_quantized.isdisjoint(quantized)
    assert verify.verify(src, dst).violations == []


def test_list_i_stays_bf16_even_when_the_selection_covers_it(src: Path, tmp_path: Path) -> None:
    dst = tmp_path / "dst"
    info = _convert(src, dst, select="all")
    out = _read_all(dst)
    for name in ("model.layers.1.mixer_proj.weight", "model.layers.0.mlp.experts.experts.weight"):
        assert out[name].dtype == torch.bfloat16
        assert name in info["ignore"]["quantizable_ignored"]
    info = _convert(src, tmp_path / "dst2", select="experts:layers=0-1")
    assert _quantized_names(tmp_path / "dst2") == {
        f"model.layers.1.mlp.experts.{t}.weight" for t in ("experts", "output_experts")
    }  # layer 0 is in the range but is list I
    cfg = json.loads((tmp_path / "dst2" / "config.json").read_text())
    assert "model.layers.0.mlp.experts" in cfg["quantization_config"]["ignore"]
    assert "model.layers.1.mixer_proj" in cfg["quantization_config"]["ignore"]


def test_the_selection_is_recorded_in_quantization_info(src: Path, tmp_path: Path) -> None:
    info = _convert(src, tmp_path / "dst", select="experts:layers=3,1-2")
    on_disk = json.loads((tmp_path / "dst" / "quantization_info.json").read_text())
    assert on_disk["selection"] == info["selection"]
    sel = info["selection"]
    assert sel["spec"] == "experts:layers=1-3"
    assert sel["name"] == "experts"
    assert sel["components"] == ["experts"]
    assert sel["layers"] == [[1, 3]]
    assert sel["base_ignore_patterns"] == list(IGNORE)
    assert sel["expected"]["quantized_layers"] == 3
    assert sel["expected"]["quantized_tensors"] == 6
    assert sel["expected"]["ignored_tensors"] == N_QUANTIZABLE - 6
    # the patterns given to the plan are the base list followed by the deselected tensors, and are recorded
    assert info["ignore"]["patterns"][: len(IGNORE)] == list(IGNORE)
    assert info["ignore"]["patterns"][len(IGNORE) :] == sel["deselected_tensors"]
    assert "model.layers.2.mixer_proj.weight" in sel["deselected_tensors"]
    assert "model.layers.2.mlp.experts.experts.weight" not in sel["deselected_tensors"]


@pytest.mark.parametrize("recipe", ["nvfp4", "fp8"])
@pytest.mark.parametrize(
    "spec",
    ["all", "experts", "non-experts", "experts:layers=2", "shared-experts", "latent", "attention", "mamba"],
)
def test_the_prediction_is_what_the_converter_wrote(
    src: Path, tmp_path: Path, recipe: str, spec: str
) -> None:
    dst = tmp_path / "dst"
    info = _convert(src, dst, recipe, spec)
    header = SourceCheckpoint(src).header()
    want = ablation.expected_counts(header, spec, IGNORE)
    assert info["selection"]["expected"] == want
    on_disk = _quantized_names(dst)
    assert len(on_disk) == want["quantized_tensors"]
    # one module per counted layer: the up and down routed tensors of a layer are the same module
    modules = {
        re.sub(r"\.mlp\.experts\.(experts|output_experts)\.weight$", ".mlp.experts", n) for n in on_disk
    }
    assert len(modules) == want["quantized_layers"]
    ignored = sum(v["count"] for k, v in info["categories"].items() if k.startswith("ignored:"))
    assert ignored == want["ignored_tensors"]


def test_a_dry_run_with_a_selection_writes_nothing_and_records_it(src: Path, tmp_path: Path) -> None:
    info = _convert(src, tmp_path / "dst", select="mamba", dry_run=True)
    assert not (tmp_path / "dst").exists()
    assert info["selection"]["spec"] == "mamba"
    assert info["selection"]["expected"]["quantized_tensors"] == 2


def test_a_selection_that_quantizes_nothing_is_refused_and_writes_nothing(src: Path, tmp_path: Path) -> None:
    with pytest.raises(ablation.SelectionError, match="quantizes no tensor"):
        _convert(src, tmp_path / "dst", select="experts:layers=0")
    assert not (tmp_path / "dst").exists()
    with pytest.raises(ablation.SelectionError, match="layers 0 to 3"):
        _convert(src, tmp_path / "dst", select="experts:layers=0-4")


# ------------------------------------------------------------------------- the verifier


def test_verify_checks_the_recorded_selection_against_the_recorded_ignore_list(
    src: Path, tmp_path: Path
) -> None:
    dst = tmp_path / "dst"
    _convert(src, dst, select="experts")
    info_path = dst / "quantization_info.json"
    good = json.loads(info_path.read_text())

    def tampered(edit) -> set[str]:
        info = json.loads(json.dumps(good))
        edit(info)
        info_path.write_text(json.dumps(info))
        checks = {v.check for v in verify.verify(src, dst).violations}
        info_path.write_text(json.dumps(good))
        return checks

    assert verify.verify(src, dst).violations == []
    assert "info_selection" in tampered(lambda i: i["selection"].update(spec="attention"))
    assert "info_selection" in tampered(lambda i: i["selection"].update(deselected_tensors=[]))
    assert "info_selection" in tampered(lambda i: i["selection"]["expected"].update(quantized_layers=99))
    assert "info_selection" in tampered(lambda i: i["selection"].update(base_ignore_patterns=[]))
    assert "info_selection" in tampered(lambda i: i["ignore"]["patterns"].pop())


def test_verify_can_require_a_selection(src: Path, tmp_path: Path) -> None:
    dst = tmp_path / "dst"
    _convert(src, dst, select="experts:layers=2-3")
    assert (
        verify.verify(src, dst, select="experts:layers=3,2").violations == []
    )  # same selection, other spelling
    wrong = verify.verify(src, dst, select="experts")
    assert {v.check for v in wrong.violations} == {"info_selection"}
    plain = tmp_path / "plain"
    _convert(src, plain)
    assert {v.check for v in verify.verify(src, plain, select="experts").violations} == {"info_selection"}
    assert verify.main(["--src", str(src), "--dst", str(dst), "--select", "experts:layers=2-3"]) == 0
    assert verify.main(["--src", str(src), "--dst", str(dst), "--select", "mamba"]) == 1


def test_a_checkpoint_of_a_selection_that_quantized_a_deselected_tensor_is_caught(
    src: Path, tmp_path: Path
) -> None:
    """The converter is fine; the checkpoint is not: the info claims `experts` but a shared expert is fp8."""
    dst = tmp_path / "dst"
    _convert(src, dst, "fp8", "experts")
    other = tmp_path / "other"
    _convert(src, other, "fp8", "shared-experts")
    # info and config of the wrong ablation over the shards of the right one
    for f in ("quantization_info.json", "config.json"):
        (dst / f).write_bytes((other / f).read_bytes())
    checks = {v.check for v in verify.verify(src, dst).violations}
    assert checks & {"missing_tensor", "unexpected_tensor", "dtype", "shape"}


# ------------------------------------------------------------------------- the command line


def test_the_command_line_takes_select_and_refuses_what_would_drop_list_i(
    src: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ignore = tmp_path / "ign.json"
    ignore.write_text(json.dumps(list(IGNORE)))
    base = ["--src", str(src), "--recipe", "nvfp4", "--ignore-file", str(ignore)]
    assert conv.main([*base, "--dst", str(tmp_path / "a"), "--select", "attention"]) == 0
    info = json.loads((tmp_path / "a" / "quantization_info.json").read_text())
    assert info["selection"]["spec"] == "attention"
    assert info["selection"]["base_ignore_patterns"] == list(IGNORE)
    assert "wrote:" in capsys.readouterr().out
    # --no-ignore would drop list I: a selection never does
    assert (
        conv.main(
            [
                "--src",
                str(src),
                "--recipe",
                "nvfp4",
                "--no-ignore",
                "--dst",
                str(tmp_path / "b"),
                "--select",
                "attention",
            ]
        )
        == 2
    )
    assert "list I" in capsys.readouterr().err
    assert not (tmp_path / "b").exists()
    # a bad spec and an empty selection are exit 2, with the reason, and write nothing
    assert conv.main([*base, "--dst", str(tmp_path / "c"), "--select", "expert"]) == 2
    assert "unknown selection" in capsys.readouterr().err
    assert conv.main([*base, "--dst", str(tmp_path / "d"), "--select", "experts:layers=0"]) == 2
    assert "quantizes no tensor" in capsys.readouterr().err
    assert not (tmp_path / "c").exists()
    assert not (tmp_path / "d").exists()


def test_the_command_line_default_ignore_is_list_i_for_a_selection(
    src: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # list I of the real model names layers the 4-layer fake does not have: refused, as without --select
    code = conv.main(
        ["--src", str(src), "--recipe", "nvfp4", "--dst", str(tmp_path / "e"), "--select", "experts"]
    )
    assert code == 2
    assert "matches no tensor" in capsys.readouterr().err


# ------------------------------------------------------------------------- python -m olala_quant.ablation


def test_the_expected_subcommand_prints_the_prediction_as_json(
    src: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ignore = tmp_path / "ign.json"
    ignore.write_text(json.dumps(list(IGNORE)))
    assert (
        ablation.main(["expected", "--src", str(src), "--select", "experts", "--ignore-file", str(ignore)])
        == 0
    )
    counts = json.loads(capsys.readouterr().out)
    assert counts["quantized_layers"] == 3 and counts["quantized_tensors"] == 6
    assert (
        ablation.main(["expected", "--src", str(src), "--select", "bogus", "--ignore-file", str(ignore)]) == 2
    )


def test_the_check_subcommand_compares_a_checkpoint_and_a_server_log_with_the_prediction(
    src: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ignore = tmp_path / "ign.json"
    ignore.write_text(json.dumps(list(IGNORE)))
    dst = tmp_path / "dst"
    _convert(src, dst, select="experts")
    base = ["check", "--src", str(src), "--dst", str(dst), "--ignore-file", str(ignore)]
    assert ablation.main([*base, "--select", "experts"]) == 0
    assert "ABLATION_CHECK_OK experts" in capsys.readouterr().out
    # another selection than the checkpoint's
    assert ablation.main([*base, "--select", "attention"]) == 1
    assert "MISMATCH" in capsys.readouterr().err
    # a server log with the Quantizing line: counts compared; without the line (offline checkpoint): disk only
    log = tmp_path / "server.log"
    log.write_text("INFO Quantizing 3 layers of types: mlp.experts: 3 (from moe: x)\n")
    assert ablation.main([*base, "--select", "experts", "--server-log", str(log)]) == 0
    log.write_text("INFO Quantizing 4 layers of types: mlp.experts: 4 (from moe: x)\n")
    assert ablation.main([*base, "--select", "experts", "--server-log", str(log)]) == 1
    assert "server quantizes 4 layers" in capsys.readouterr().err
    log.write_text("INFO Loading weights took 2.8 seconds\n")
    assert ablation.main([*base, "--select", "experts", "--server-log", str(log)]) == 0
    # a checkpoint missing its selection block, a tensor count that is off, an ignore list that was edited
    plain = tmp_path / "plain"
    _convert(src, plain)
    assert (
        ablation.main(
            ["check", "--src", str(src), "--dst", str(plain), "--ignore-file", str(ignore), "--select", "all"]
        )
        == 1
    )
    cfg_path = dst / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["quantization_config"]["ignore"].remove("model.layers.2.mixer_proj")
    cfg_path.write_text(json.dumps(cfg))
    assert ablation.main([*base, "--select", "experts"]) == 1
    assert "ignore" in capsys.readouterr().err


def test_the_check_subcommand_counts_the_quantized_tensors_on_disk(
    src: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ignore = tmp_path / "ign.json"
    ignore.write_text(json.dumps(list(IGNORE)))
    dst = tmp_path / "dst"
    _convert(src, dst, select="experts")
    index_path = dst / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    victim = next(n for n in index["weight_map"] if n.endswith(".weight_scale"))
    del index["weight_map"][victim]  # info and config still say the right thing; the disk does not
    index_path.write_text(json.dumps(index))
    args = [
        "check",
        "--src",
        str(src),
        "--dst",
        str(dst),
        "--ignore-file",
        str(ignore),
        "--select",
        "experts",
    ]
    assert ablation.main(args) == 1
    assert "quantized tensors on disk" in capsys.readouterr().err


def test_expected_can_read_a_bare_header_json_so_a_spec_is_checkable_without_the_weights(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    header = Path(__file__).parent / "data" / "olala_header.json"
    assert ablation.main(["expected", "--header", str(header), "--select", "experts:layers=12-23"]) == 0
    counts = json.loads(capsys.readouterr().out)
    assert (counts["quantized_layers"], counts["quantized_tensors"]) == (11, 22)
    assert ablation.main(["expected", "--header", str(header), "--select", "experts:layers=0-1"]) == 2
    assert "quantizes no tensor" in capsys.readouterr().err
    with pytest.raises(SystemExit):  # one of --src and --header is required, not both
        ablation.main(["expected", "--select", "experts"])
    with pytest.raises(SystemExit):
        ablation.main(["expected", "--header", str(header), "--src", str(tmp_path), "--select", "experts"])
