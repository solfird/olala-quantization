"""Header-level access to safetensors checkpoints (single file or sharded with an index)."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "BOOL": 1,
}
ST_TO_TORCH = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F8_E4M3": torch.float8_e4m3fn,
    "U8": torch.uint8,
    "I8": torch.int8,
}
TORCH_TO_ST = {v: k for k, v in ST_TO_TORCH.items()}
INDEX_NAME = "model.safetensors.index.json"
SINGLE_NAME = "model.safetensors"


@dataclass(frozen=True)
class TensorMeta:
    """Dtype (safetensors spelling, e.g. ``BF16``) and shape of one tensor."""

    dtype: str
    shape: tuple[int, ...]

    @property
    def numel(self) -> int:
        """Number of elements (1 for a 0-dim tensor)."""
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        """Size in bytes."""
        return self.numel * DTYPE_BYTES[self.dtype]


def header_from_json(obj: Mapping[str, Any]) -> dict[str, TensorMeta]:
    """Turn a parsed safetensors header into ``{name: TensorMeta}``, dropping ``__metadata__``."""
    return {
        name: TensorMeta(str(entry["dtype"]), tuple(int(d) for d in entry["shape"]))
        for name, entry in obj.items()
        if name != "__metadata__"
    }


def read_safetensors_header(path: Path) -> tuple[dict[str, Any], bytes]:
    """Read the JSON header of a safetensors file without touching the data.

    Returns:
        The parsed header and the raw header bytes (8-byte length prefix included).
    """
    with path.open("rb") as f:
        raw_len = f.read(8)
        (length,) = struct.unpack("<Q", raw_len)
        body = f.read(length)
    return json.loads(body), raw_len + body


def sha256_bytes(data: bytes) -> str:
    """Hex sha256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


class SourceCheckpoint:
    """A safetensors checkpoint directory: one ``model.safetensors`` or an indexed set of shards."""

    def __init__(self, directory: Path) -> None:
        """Open ``directory``; raises FileNotFoundError when no weights are found."""
        self.directory = directory
        index = directory / INDEX_NAME
        single = directory / SINGLE_NAME
        self._weight_map: dict[str, str]
        if index.is_file():
            self.index_path: Path | None = index
            self._weight_map = json.loads(index.read_text())["weight_map"]
        elif single.is_file():
            self.index_path = None
            header, _ = read_safetensors_header(single)
            self._weight_map = {k: SINGLE_NAME for k in header if k != "__metadata__"}
        else:
            raise FileNotFoundError(f"no {INDEX_NAME} or {SINGLE_NAME} in {directory}")
        self._handles: dict[str, Any] = {}

    @classmethod
    def from_shard_headers(cls, directory: Path, pattern: str = "model-*.safetensors") -> SourceCheckpoint:
        """Open ``directory`` trusting the shard headers instead of any index file.

        Used by the verifier so that a damaged or incomplete index cannot hide a tensor.
        """
        self = cls.__new__(cls)
        self.directory = directory
        self.index_path = None
        self._weight_map = {}
        for f in sorted(directory.glob(pattern)):
            raw, _ = read_safetensors_header(f)
            self._weight_map.update({k: f.name for k in raw if k != "__metadata__"})
        self._handles = {}
        return self

    def names(self) -> list[str]:
        """Tensor names in sorted order."""
        return sorted(self._weight_map)

    def shard_files(self) -> list[str]:
        """Distinct shard file names."""
        return sorted(set(self._weight_map.values()))

    def header(self) -> dict[str, TensorMeta]:
        """Dtype and shape of every tensor, from the shard headers only."""
        out: dict[str, TensorMeta] = {}
        for shard in self.shard_files():
            raw, _ = read_safetensors_header(self.directory / shard)
            out.update(header_from_json(raw))
        return {name: out[name] for name in self.names()}

    def identity_hash(self) -> tuple[str, str]:
        """Return ``(kind, sha256)``: of the index file if there is one, else of the header bytes.

        A single-file checkpoint has no index, so the hash of its safetensors header (length
        prefix plus JSON, which fixes every name, dtype, shape and offset) stands in for it.
        """
        if self.index_path is not None:
            return "index_json", sha256_bytes(self.index_path.read_bytes())
        _, raw = read_safetensors_header(self.directory / SINGLE_NAME)
        return "safetensors_header", sha256_bytes(raw)

    def get_tensor(self, name: str) -> torch.Tensor:
        """Load one tensor (memory-mapped read, copied to a regular tensor)."""
        shard = self._weight_map[name]
        if shard not in self._handles:
            self._handles[shard] = safe_open(self.directory / shard, framework="pt", device="cpu")
        return self._handles[shard].get_tensor(name)

    def iter_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield every tensor in name order."""
        for name in self.names():
            yield name, self.get_tensor(name)

    def close(self) -> None:
        """Drop the open shard handles."""
        self._handles.clear()
