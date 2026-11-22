"""pod/verify_model.py: the downloaded weights are checked against the Hub's file sizes and hashes."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from conftest import FIXTURES, RUNPOD

SCRIPT = RUNPOD / "pod" / "verify_model.py"
HUB = json.loads((FIXTURES / "hub_model_info.json").read_text())


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()  # noqa: S324 - git's object id


def meta_for(files: dict[str, bytes], lfs: set[str]) -> dict:
    siblings = []
    for name, data in files.items():
        entry = {"rfilename": name, "size": len(data), "blobId": git_blob_sha1(data)}
        if name in lfs:
            entry["lfs"] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        siblings.append(entry)
    return {"sha": "abc123", "siblings": siblings}


def write_model(root: Path, files: dict[str, bytes]) -> None:
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def verify(root: Path, meta: dict, tmp: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    meta_file = tmp / "meta.json"
    meta_file.write_text(json.dumps(meta))
    return subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--dir", str(root), "--meta", str(meta_file), *extra],
        capture_output=True,
        text=True,
        check=False,
    )


FILES = {
    "config.json": b'{"a": 1}\n',
    "model.safetensors": b"weights" * 1000,
    "sub/tokenizer.json": b"{}",
}
LFS = {"model.safetensors", "sub/tokenizer.json"}


def test_a_matching_download_is_verified_with_sizes_and_hashes(tmp_path):
    write_model(tmp_path / "m", FILES)
    out = verify(tmp_path / "m", meta_for(FILES, LFS), tmp_path)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "VERIFIED 3 files" in out.stdout
    assert "model.safetensors" in out.stdout and "sha256" in out.stdout


def test_a_wrong_size_fails_and_names_the_file(tmp_path):
    write_model(tmp_path / "m", {**FILES, "model.safetensors": b"short"})
    out = verify(tmp_path / "m", meta_for(FILES, LFS), tmp_path)
    assert out.returncode == 1
    assert "FAIL model.safetensors" in out.stdout and "size" in out.stdout


def test_a_corrupt_lfs_file_of_the_right_size_fails_on_the_sha256(tmp_path):
    bad = bytearray(FILES["model.safetensors"])
    bad[10] ^= 0xFF
    write_model(tmp_path / "m", {**FILES, "model.safetensors": bytes(bad)})
    out = verify(tmp_path / "m", meta_for(FILES, LFS), tmp_path)
    assert out.returncode == 1
    assert "FAIL model.safetensors" in out.stdout and "sha256" in out.stdout


def test_a_corrupt_small_file_of_the_right_size_fails_on_the_git_blob_id(tmp_path):
    write_model(tmp_path / "m", {**FILES, "config.json": b'{"a": 2}\n'})
    out = verify(tmp_path / "m", meta_for(FILES, LFS), tmp_path)
    assert out.returncode == 1 and "FAIL config.json" in out.stdout


def test_a_missing_file_fails(tmp_path):
    files = {k: v for k, v in FILES.items() if k != "sub/tokenizer.json"}
    write_model(tmp_path / "m", files)
    out = verify(tmp_path / "m", meta_for(FILES, LFS), tmp_path)
    assert out.returncode == 1 and "MISSING sub/tokenizer.json" in out.stdout


def test_ignored_patterns_are_neither_required_nor_checked(tmp_path):
    files = {**FILES, "assets/logo.png": b"png", "paper.pdf": b"%PDF"}
    write_model(tmp_path / "m", FILES)
    out = verify(
        tmp_path / "m",
        meta_for(files, LFS | {"assets/logo.png", "paper.pdf"}),
        tmp_path,
        "--ignore",
        "assets/*",
        "--ignore",
        "*.pdf",
    )
    assert out.returncode == 0, out.stdout


def test_the_download_cache_folder_of_the_hub_client_is_not_a_model_file(tmp_path):
    write_model(tmp_path / "m", {**FILES, ".cache/huggingface/download/x.metadata": b"m"})
    assert verify(tmp_path / "m", meta_for(FILES, LFS), tmp_path).returncode == 0


def test_the_real_hub_metadata_of_the_study_model_checks_a_real_small_file(tmp_path):
    """tests/fixtures/olala_config.json is the public config.json of the pinned revision."""
    write_model(tmp_path / "m", {"config.json": (FIXTURES / "olala_config.json").read_bytes()})
    out = verify(tmp_path / "m", HUB, tmp_path, "--only", "config.json")
    assert out.returncode == 0, out.stdout + out.stderr
    assert HUB["sha"] == "b40f8165ee2254c18601220e19842600c6d91794"


def test_the_real_metadata_lists_one_weights_file_with_a_sha256():
    weights = next(s for s in HUB["siblings"] if s["rfilename"] == "model.safetensors")
    assert weights["size"] == 13579677592
    assert len(weights["lfs"]["sha256"]) == 64


def test_a_wrong_revision_in_the_metadata_is_reported(tmp_path):
    write_model(tmp_path / "m", FILES)
    out = verify(tmp_path / "m", meta_for(FILES, LFS), tmp_path, "--expect-revision", "deadbeef")
    assert out.returncode == 1 and "revision" in out.stdout
