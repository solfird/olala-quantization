"""Tests for olala_quant.checkpoint_io on tiny synthetic safetensors checkpoints."""

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from olala_quant.checkpoint_io import (
    SourceCheckpoint,
    TensorMeta,
    header_from_json,
    read_safetensors_header,
)


def test_tensor_meta_sizes() -> None:
    m = TensorMeta("BF16", (3, 4))
    assert (m.numel, m.nbytes) == (12, 24)
    assert (TensorMeta("F32", ()).numel, TensorMeta("F32", ()).nbytes) == (1, 4)
    assert TensorMeta("F8_E4M3", (5,)).nbytes == 5


def test_header_from_json_drops_metadata() -> None:
    h = header_from_json(
        {
            "__metadata__": {"format": "pt"},
            "a": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]},
        }
    )
    assert h == {"a": TensorMeta("BF16", (2, 3))}


def test_single_file_checkpoint(tmp_path: Path) -> None:
    tensors = {
        "b": torch.ones(2, 2, dtype=torch.bfloat16),
        "a": torch.arange(3, dtype=torch.float32),
    }
    save_file(tensors, tmp_path / "model.safetensors", metadata={"format": "pt"})
    ck = SourceCheckpoint(tmp_path)
    assert ck.names() == ["a", "b"]
    assert ck.header() == {"a": TensorMeta("F32", (3,)), "b": TensorMeta("BF16", (2, 2))}
    assert torch.equal(ck.get_tensor("b"), tensors["b"])
    kind, digest = ck.identity_hash()
    assert kind == "safetensors_header"
    _, raw = read_safetensors_header(tmp_path / "model.safetensors")
    assert digest == hashlib.sha256(raw).hexdigest()


def test_sharded_checkpoint_uses_index_hash(tmp_path: Path) -> None:
    save_file({"x": torch.zeros(4)}, tmp_path / "s1.safetensors", metadata={"format": "pt"})
    save_file(
        {"y": torch.ones(4, dtype=torch.bfloat16)},
        tmp_path / "s2.safetensors",
        metadata={"format": "pt"},
    )
    index = {"metadata": {}, "weight_map": {"x": "s1.safetensors", "y": "s2.safetensors"}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    ck = SourceCheckpoint(tmp_path)
    assert ck.names() == ["x", "y"]
    assert ck.shard_files() == ["s1.safetensors", "s2.safetensors"]
    assert ck.header()["y"] == TensorMeta("BF16", (4,))
    kind, digest = ck.identity_hash()
    assert kind == "index_json"
    assert digest == hashlib.sha256((tmp_path / "model.safetensors.index.json").read_bytes()).hexdigest()
    assert [n for n, _ in ck.iter_tensors()] == ["x", "y"]


def test_missing_weights_raise(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SourceCheckpoint(tmp_path)
