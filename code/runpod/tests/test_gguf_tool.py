"""pod/gguf_tool.py and pod/gguf_files.json: the pinned facts about OVH's GGUF files and llama.cpp fork.

No network, no GPU: the Hub download is replaced by a callable that writes a small file, and the pinned manifest is
checked against the numbers read from the Hub's public API on 2026-11-20 (recon/llamacpp_olala_notes.md).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import RUNPOD

POD = RUNPOD / "pod"
TOOL = POD / "gguf_tool.py"
MANIFEST = POD / "gguf_files.json"

Q5_NAME = "Olala-v1.0-mini-7A1B-q5_k_m.gguf"
Q5_SIZE = 5990551040
Q5_SHA = "ed353d2632c406be9ca2022b35e4c13049d6f8f7390b1c1bb61d6a7d6b971860"
FORK_COMMIT = "55c22dca0c987f247c21b427c3b3a09e1a28187f"


def load_tool():
    spec = importlib.util.spec_from_file_location("gguf_tool", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules["gguf_tool"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    return load_tool()


def cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", str(TOOL), *args], capture_output=True, text=True, check=False
    )


def small_manifest(tmp_path: Path, payload: bytes = b"hello gguf") -> Path:
    manifest = json.loads(MANIFEST.read_text())
    manifest["gguf"]["files"] = {
        "tiny": {
            "name": "tiny.gguf",
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    }
    manifest["gguf"]["default"] = "tiny"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


# --- the pinned manifest -----------------------------------------------------------------------


def test_the_manifest_pins_the_q5_k_m_file_with_the_name_size_and_sha256_read_from_the_hub():
    manifest = json.loads(MANIFEST.read_text())
    gguf = manifest["gguf"]
    assert gguf["repo"] == "OVHaiLLM/Olala-v1.0-mini-7A1B-GGUF"
    assert gguf["default"] == "q5_k_m"
    entry = gguf["files"]["q5_k_m"]
    assert entry == {"name": Q5_NAME, "size": Q5_SIZE, "sha256": Q5_SHA}
    assert len(gguf["revision"]) == 40  # a commit, never a branch name


def test_the_manifest_lists_the_four_ovh_files_for_a_same_engine_control():
    files = json.loads(MANIFEST.read_text())["gguf"]["files"]
    assert set(files) == {"q5_k_m", "q6_k", "q8_0-safe2", "bf16"}
    assert files["bf16"]["size"] == 13594188096
    for entry in files.values():
        assert len(entry["sha256"]) == 64 and entry["name"].endswith(".gguf")


def test_the_manifest_pins_the_fork_commit_and_the_cuda_only_build():
    fork = json.loads(MANIFEST.read_text())["llama_cpp"]
    assert fork["repo"] == "https://github.com/OVHai-LLM/llama.cpp.git"
    assert fork["commit"] == FORK_COMMIT
    assert len(fork["tree"]) == 40
    assert fork["targets"] == ["llama-server"]
    flags = fork["cmake_flags"]
    assert "-DGGML_CUDA=ON" in flags and "-DCMAKE_BUILD_TYPE=Release" in flags
    assert (
        "-DLLAMA_USE_PREBUILT_UI=OFF" in flags
    )  # the model card: the prebuilt UI does not match the fork


# --- verify ------------------------------------------------------------------------------------


def test_verify_accepts_the_file_with_the_pinned_size_and_hash(tool, tmp_path):
    payload = b"hello gguf"
    path = tmp_path / "tiny.gguf"
    path.write_bytes(payload)
    entry = json.loads(small_manifest(tmp_path, payload).read_text())["gguf"]["files"]["tiny"]
    ok, message = tool.verify(path, entry)
    assert ok, message
    assert "sha256" in message


def test_verify_refuses_a_wrong_size_before_hashing_and_a_wrong_hash(tool, tmp_path):
    path = tmp_path / "tiny.gguf"
    path.write_bytes(b"hello gguf")
    entry = json.loads(small_manifest(tmp_path).read_text())["gguf"]["files"]["tiny"]
    ok, message = tool.verify(path, {**entry, "size": entry["size"] + 1})
    assert not ok and "size" in message
    ok, message = tool.verify(path, {**entry, "sha256": "0" * 64})
    assert not ok and "sha256" in message
    ok, message = tool.verify(tmp_path / "missing.gguf", entry)
    assert not ok and "missing" in message.lower()


# --- fetch -------------------------------------------------------------------------------------


def test_fetch_downloads_the_pinned_revision_without_a_token_then_verifies(tool, tmp_path):
    payload = b"hello gguf"
    manifest = json.loads(small_manifest(tmp_path, payload).read_text())
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.write_bytes(payload)
        return str(target)

    path = tool.fetch(manifest, "tiny", tmp_path / "models", download=fake_download)
    assert path == tmp_path / "models" / "tiny.gguf"
    (call,) = calls
    assert call["repo_id"] == "OVHaiLLM/Olala-v1.0-mini-7A1B-GGUF"
    assert call["revision"] == manifest["gguf"]["revision"]
    assert call["token"] is False  # a public repository: no credential on a pod
    assert call["filename"] == "tiny.gguf"


def test_fetch_raises_when_the_download_does_not_match_the_pins(tool, tmp_path):
    manifest = json.loads(small_manifest(tmp_path).read_text())

    def corrupt(**kwargs):
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.write_bytes(b"not the file")
        return str(target)

    with pytest.raises(tool.GgufError, match="size"):
        tool.fetch(manifest, "tiny", tmp_path / "models", download=corrupt)


def test_fetch_skips_the_download_when_a_verified_file_is_already_there(tool, tmp_path):
    payload = b"hello gguf"
    manifest = json.loads(small_manifest(tmp_path, payload).read_text())
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "tiny.gguf").write_bytes(payload)

    def must_not_run(**kwargs):
        raise AssertionError("downloaded again")

    assert tool.fetch(manifest, "tiny", tmp_path / "models", download=must_not_run).exists()


def test_an_unknown_variant_lists_the_known_ones(tool):
    manifest = json.loads(MANIFEST.read_text())
    with pytest.raises(tool.GgufError) as raised:
        tool.variant_entry(manifest, "q4_k_m")
    assert "q5_k_m" in str(raised.value) and "bf16" in str(raised.value)


# --- CUDA architecture --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cc", "arch"),
    [
        (
            "12.0",
            "120a-real",
        ),  # RTX PRO 4500 / 6000 Blackwell: the Blackwell FP4 instructions need 12Xa
        ("12.1", "121a-real"),
        ("10.0", "100a-real"),  # B200
        ("9.0", "90a-real"),  # H100
        ("8.9", "89-real"),  # L40S
        ("8.6", "86-real"),
    ],
)
def test_the_cuda_architecture_of_the_build_follows_the_gpu(tool, cc, arch):
    assert tool.cuda_arch(cc) == arch


def test_a_malformed_compute_capability_is_refused(tool):
    with pytest.raises(tool.GgufError):
        tool.cuda_arch("blackwell")


# --- slots and memory ----------------------------------------------------------------------------


def test_the_default_context_gives_every_slot_the_full_32768_tokens(tool):
    assert tool.default_ctx(12) == 12 * 32768
    assert tool.slot_ctx(12 * 32768, 12) == 32768


def test_llama_cpp_pads_the_slot_context_to_a_multiple_of_256(tool):
    # src/llama-context.cpp:297-298: n_ctx_seq = GGML_PAD(n_ctx / n_seq_max, 256)
    assert tool.slot_ctx(100000, 3) == 33536  # 33333 padded up
    assert tool.slot_ctx(32768, 1) == 32768


def test_the_memory_plan_counts_seven_attention_layers_and_a_state_per_slot(tool):
    manifest = json.loads(MANIFEST.read_text())
    plan = tool.memory_plan(manifest, "q5_k_m", np=12, ctx=12 * 32768)
    # 7 layers x (K + V) x 12 kv heads x 128 dims x 2 bytes (f16) = 43,008 bytes per token
    assert plan["kv_bytes_per_token"] == 43008
    assert plan["kv_mib"] == 12 * 32768 * 43008 / 2**20 == 16128
    # (3,072 conv-like floats + 431,616 SSM floats) x 4 bytes x 36 layers per sequence
    assert plan["recurrent_bytes_per_slot"] == 62595072
    assert plan["recurrent_mib"] == pytest.approx(12 * 62595072 / 2**20)
    assert plan["weights_mib"] == pytest.approx(Q5_SIZE / 2**20)
    assert plan["slot_ctx"] == 32768
    assert plan["certain_mib"] == pytest.approx(
        plan["weights_mib"] + plan["kv_mib"] + plan["recurrent_mib"]
    )
    assert plan["estimate_mib"] == plan["certain_mib"] + plan["assumed_overhead_mib"]
    assert plan["assumed_overhead_mib"] > 0  # an assumption, printed as one


def test_twelve_full_slots_fit_a_32_gb_card_and_sixteen_leave_too_little_margin(tool):
    manifest = json.loads(MANIFEST.read_text())
    card_mib = 32000
    ok = tool.memory_plan(manifest, "q5_k_m", np=12, ctx=12 * 32768)
    tight = tool.memory_plan(manifest, "q5_k_m", np=16, ctx=16 * 32768)
    assert ok["estimate_mib"] < card_mib - 4000
    assert tight["estimate_mib"] > card_mib - 2500


def test_the_bf16_file_needs_more_memory_than_the_q5_k_m_file(tool):
    manifest = json.loads(MANIFEST.read_text())
    q5 = tool.memory_plan(manifest, "q5_k_m", np=4, ctx=4 * 32768)
    bf16 = tool.memory_plan(manifest, "bf16", np=4, ctx=4 * 32768)
    assert bf16["weights_mib"] - q5["weights_mib"] == pytest.approx((13594188096 - Q5_SIZE) / 2**20)


# --- command line --------------------------------------------------------------------------------


def test_cli_get_prints_a_manifest_value(tmp_path):
    out = cli("get", "llama_cpp.commit")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == FORK_COMMIT
    assert cli("get", "gguf.files.q5_k_m.name").stdout.strip() == Q5_NAME
    assert cli("get", "no.such.key").returncode != 0


def test_cli_arch_and_plan(tmp_path):
    assert cli("arch", "12.0").stdout.strip() == "120a-real"
    out = cli("plan", "--variant", "q5_k_m", "--np", "12")
    assert out.returncode == 0, out.stderr
    plan = json.loads(out.stdout)
    assert plan["np"] == 12 and plan["ctx"] == 12 * 32768 and plan["slot_ctx"] == 32768


def test_cli_verify_exit_codes(tmp_path):
    payload = b"hello gguf"
    manifest = small_manifest(tmp_path, payload)
    good = tmp_path / "tiny.gguf"
    good.write_bytes(payload)
    ok = cli("--manifest", str(manifest), "verify", "--variant", "tiny", "--file", str(good))
    assert ok.returncode == 0 and "OK" in ok.stdout
    good.write_bytes(b"hello gguF")
    bad = cli("--manifest", str(manifest), "verify", "--variant", "tiny", "--file", str(good))
    assert bad.returncode == 1 and "sha256" in bad.stdout


def test_cli_verify_size_only_skips_the_hash(tmp_path):
    payload = b"hello gguf"
    manifest = small_manifest(tmp_path, payload)
    other = tmp_path / "tiny.gguf"
    other.write_bytes(b"hello gguF")  # same size, other content
    quick = cli(
        "--manifest",
        str(manifest),
        "verify",
        "--variant",
        "tiny",
        "--file",
        str(other),
        "--size-only",
    )
    assert quick.returncode == 0 and "size" in quick.stdout
    other.write_bytes(b"short")
    bad = cli(
        "--manifest",
        str(manifest),
        "verify",
        "--variant",
        "tiny",
        "--file",
        str(other),
        "--size-only",
    )
    assert bad.returncode == 1 and "size" in bad.stdout


def test_resolve_accepts_a_variant_a_file_name_or_a_path(tool):
    manifest = json.loads(MANIFEST.read_text())
    assert tool.resolve_variant(manifest, "q5_k_m") == "q5_k_m"
    assert tool.resolve_variant(manifest, Q5_NAME) == "q5_k_m"
    assert tool.resolve_variant(manifest, f"/workspace/models/olala-gguf/{Q5_NAME}") == "q5_k_m"
    with pytest.raises(tool.GgufError, match="q5_k_m"):
        tool.resolve_variant(manifest, "model.gguf")


def test_cli_resolve_prints_the_variant():
    assert cli("resolve", f"/x/{Q5_NAME}").stdout.strip() == "q5_k_m"
    assert cli("resolve", "nope.gguf").returncode == 2
