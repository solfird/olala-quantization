"""pod/q4_tool.py: the pure helpers of arm Q4 (amendment 24). No network, no GPU, no pod.

The Hub downloads are replaced by callables that write small files; the parquet reader is replaced by a callable that
returns rows; the GGUF header is a tiny file written here with struct.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import RUNPOD

POD = RUNPOD / "pod"
TOOL = POD / "q4_tool.py"
MANIFEST = POD / "gguf_files.json"

# The pins of amendment 24 and of setup_remote.sh, typed here on purpose: a drift in the tool fails a test.
WIKITEXT_REPO = "Salesforce/wikitext"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_FILE = "wikitext-2-raw-v1/train-00000-of-00001.parquet"
MODEL_REPO = "OVHaiLLM/Olala-v1.0-mini-7A1B"
MODEL_REVISION = "b40f8165ee2254c18601220e19842600c6d91794"


def load_tool():
    spec = importlib.util.spec_from_file_location("q4_tool", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules["q4_tool"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    return load_tool()


def cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", str(TOOL), *args], capture_output=True, text=True, check=False
    )


# --- wikitext ----------------------------------------------------------------------------------------------


def test_the_wikitext_text_is_the_rows_concatenated_in_order_with_no_separator(tool, tmp_path):
    rows = [" \n", " = Valkyria Chronicles III = \n", "\n", "Senjo no Valkyria 3 : café\n"]
    seen = {}

    def download(**kwargs):
        seen.update(kwargs)
        path = tmp_path / "cache" / "x.parquet"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"not really parquet")
        return str(path)

    out = tmp_path / "wiki.train.raw"
    record = tool.fetch_wikitext(
        out, tmp_path / "cache", download=download, read_rows=lambda _p: rows
    )
    assert out.read_bytes() == "".join(rows).encode("utf-8")  # no "\n".join, no stripping
    assert record["text_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert record["rows"] == 4 and record["text_bytes"] == out.stat().st_size
    assert record["source"] == "hub"
    # the pinned dataset file, at the pinned revision, with no token
    assert seen["repo_id"] == WIKITEXT_REPO and seen["repo_type"] == "dataset"
    assert seen["revision"] == WIKITEXT_REVISION and seen["filename"] == WIKITEXT_FILE
    assert seen["token"] is False


def test_a_local_parquet_is_read_without_any_download_and_said_so(tool, tmp_path):
    local = tmp_path / "local.parquet"
    local.write_bytes(b"x")

    def download(**_kwargs):
        raise AssertionError("no download when a local parquet is given")

    out = tmp_path / "wiki.train.raw"
    record = tool.fetch_wikitext(
        out, tmp_path, download=download, read_rows=lambda _p: ["a\n", "b\n"], parquet=local
    )
    assert out.read_text() == "a\nb\n" and str(local) in record["source"]


def test_a_row_that_is_not_text_stops_the_text_from_being_written(tool, tmp_path):
    out = tmp_path / "wiki.train.raw"
    with pytest.raises(tool.Q4Error, match="row 1"):
        tool.fetch_wikitext(
            out,
            tmp_path,
            download=lambda **_k: str(tmp_path / "p"),
            read_rows=lambda _p: ["a\n", None],
        )
    assert not out.exists()


# --- the GGUF header ---------------------------------------------------------------------------------------


def gguf_string(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def tiny_gguf(path: Path, dims: list[list[int]]) -> Path:
    kv = b""
    kv += gguf_string("general.name") + struct.pack("<I", 8) + gguf_string("tiny")
    kv += gguf_string("olala.block_count") + struct.pack("<II", 4, 36)
    kv += gguf_string("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 3)
    kv += gguf_string("a") + gguf_string("bb") + gguf_string("ccc")
    kv += (
        gguf_string("tokenizer.ggml.token_type")
        + struct.pack("<IIQ", 9, 5, 3)
        + struct.pack("<3i", 1, 1, 3)
    )
    kv += gguf_string("some.flag") + struct.pack("<I?", 7, True)
    kv += gguf_string("some.float") + struct.pack("<If", 6, 0.5)
    kv += gguf_string("some.big") + struct.pack("<IQ", 10, 2**40)
    infos = b""
    for i, shape in enumerate(dims):
        infos += gguf_string(f"blk.{i}.w") + struct.pack("<I", len(shape))
        infos += struct.pack(f"<{len(shape)}Q", *shape) + struct.pack("<IQ", 30, 0)
    head = b"GGUF" + struct.pack("<IQQ", 3, len(dims), 7)
    path.write_bytes(head + kv + infos + b"\x00" * 64)
    return path


def test_the_parameter_count_is_the_sum_of_the_tensor_elements_of_the_header(tool, tmp_path):
    path = tiny_gguf(tmp_path / "tiny.gguf", [[4, 3], [5], [2, 2, 2]])
    assert tool.gguf_parameter_count(path) == 12 + 5 + 8


def test_a_file_that_is_not_gguf_is_refused(tool, tmp_path):
    path = tmp_path / "x.gguf"
    path.write_bytes(b"GGML" + b"\x00" * 64)
    with pytest.raises(tool.Q4Error, match="GGUF"):
        tool.gguf_parameter_count(path)


def test_the_bits_per_weight_use_the_header_count_and_say_where_the_count_came_from(tool, tmp_path):
    path = tiny_gguf(tmp_path / "tiny.gguf", [[1000, 1000]])
    bpw = tool.bits_per_weight(2_000_000, path)
    assert bpw["parameters"] == 1_000_000 and bpw["bits_per_weight"] == 16.0
    assert "header" in bpw["parameters_source"]


def test_an_unreadable_header_falls_back_to_the_stated_approximation(tool, tmp_path):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"nope")
    bpw = tool.bits_per_weight(6_766_000_000 // 8, bad)
    assert bpw["parameters"] == 6_766_000_000 and bpw["bits_per_weight"] == 1.0
    assert "approximate" in bpw["parameters_source"]


# --- the quantize log --------------------------------------------------------------------------------------

LOG_HEAD = "main: build = 1 (55c22dc)\nmain: quantizing 'bf16.gguf' to 'q4.gguf' as Q4_K_M using 8 threads\n"


def quantize_log(rows: list[tuple[str, str, str]], declared: int | None = None) -> str:
    """rows: (name, source type, final type); final == source means copied."""
    total = declared if declared is not None else len(rows)
    lines = [LOG_HEAD, "llama_model_quantize_impl: have importance matrix data with 144 entries"]
    for i, (name, src, final) in enumerate(rows, 1):
        head = f"[{i:4d}/{total:4d}] {name:<36} - [ 2048,  2048,     1,     1], type = {src:>6}, "
        if final == src:
            lines.append(head + "size =     8.000 MiB")
        else:
            lines.append(head + f"converting to {final} .. size =   8.00 MiB ->   2.25 MiB")
    lines.append("llama_model_quantize_impl: model size  = 12964.00 MiB (16.00 BPW)")
    lines.append("llama_model_quantize_impl: quant size  =  4300.00 MiB (5.31 BPW)")
    return "\n".join(lines) + "\n"


GOOD_ROWS = [
    ("token_embd.weight", "bf16", "q4_K"),
    ("blk.0.attn_norm.weight", "f32", "f32"),
    ("blk.0.ffn_latent_up.weight", "bf16", "bf16"),
    ("blk.0.ffn_up_exps.weight", "bf16", "q8_0"),
    ("blk.0.ffn_down_exps.weight", "bf16", "q6_K"),
    ("blk.1.ffn_latent_up.weight", "bf16", "bf16"),
    ("blk.1.ffn_up_exps.weight", "bf16", "q8_0"),
    ("blk.1.ffn_up_shexp.weight", "bf16", "q4_K"),
    ("output.weight", "bf16", "q6_K"),
]


def test_the_log_is_parsed_into_a_final_type_count_table(tool):
    table = tool.parse_quantize_log(quantize_log(GOOD_ROWS))
    assert table["n_tensors"] == 9 and table["declared_tensors"] == 9
    assert table["by_type"] == {"bf16": 2, "f32": 1, "q4_K": 2, "q6_K": 2, "q8_0": 2}
    assert table["by_group"]["ffn_latent_up"] == {"bf16": 2}
    assert table["by_group"]["ffn_up_exps"] == {"q8_0": 2}
    assert table["problems"] == [] and table["recipe_applied"] is True
    assert "5.31 BPW" in table["quant_size_line"]


def test_a_tensor_that_was_not_overridden_is_reported_as_a_problem(tool):
    rows = [(n, s, "q4_K" if "ffn_up_exps" in n else f) for n, s, f in GOOD_ROWS]
    table = tool.parse_quantize_log(quantize_log(rows))
    assert table["recipe_applied"] is False
    assert any("ffn_up_exps" in p and "q8_0" in p for p in table["problems"])


def test_an_override_that_matched_nothing_is_a_problem(tool):
    rows = [r for r in GOOD_ROWS if "ffn_latent_up" not in r[0]]
    table = tool.parse_quantize_log(quantize_log(rows))
    assert table["recipe_applied"] is False
    assert any("ffn_latent_up" in p and "no tensor" in p for p in table["problems"])


def test_a_log_that_stops_before_the_last_tensor_is_a_problem(tool):
    table = tool.parse_quantize_log(quantize_log(GOOD_ROWS, declared=989))
    assert table["recipe_applied"] is False
    assert any("989" in p for p in table["problems"])


def real_log_with_warning_lines() -> str:
    """The shape of the real log: a tensor with no imatrix data has a warning line between its header
    and its `converting to` text (seen on the first pod, output.weight and token_embd.weight)."""
    lines = [LOG_HEAD]
    rows = [
        ("output.weight", "bf16", "q6_K"),
        ("token_embd.weight", "bf16", "q4_K"),
        *GOOD_ROWS[1:-1],
    ]
    for i, (name, src, final) in enumerate(rows, 1):
        head = (
            f"[{i:4d}/{len(rows):4d}] {name:<36} - [ 1536, 151936,     1,     1], type = {src:>6}, "
        )
        if name in ("output.weight", "token_embd.weight"):
            lines.append(head)
            lines.append(f"====== llama_model_quantize_impl: did not find weights for {name}")
            lines.append(f"converting to {final} .. size =   445.12 MiB ->   125.19 MiB")
        elif final == src:
            lines.append(head + "size =     8.000 MiB")
        else:
            lines.append(head + f"converting to {final} .. size =   8.00 MiB ->   2.25 MiB")
    return "\n".join(lines) + "\n"


def test_a_warning_line_between_a_tensor_header_and_its_conversion_is_read(tool):
    table = tool.parse_quantize_log(real_log_with_warning_lines())
    assert table["n_tensors"] == table["declared_tensors"] == 9
    assert table["by_type"]["q6_K"] == 2 and table["by_type"]["q4_K"] == 2
    assert table["problems"] == [] and table["recipe_applied"] is True


def test_a_tensor_header_without_any_conversion_text_is_a_problem(tool):
    text = quantize_log(GOOD_ROWS).replace(
        "converting to q6_K .. size =   8.00 MiB ->   2.25 MiB", ""
    )
    table = tool.parse_quantize_log(text)
    assert table["recipe_applied"] is False and any("no conversion" in p for p in table["problems"])


def test_a_log_without_tensor_lines_is_a_problem(tool):
    table = tool.parse_quantize_log("llama_model_quantize: failed\n")
    assert table["n_tensors"] == 0 and table["problems"]


# --- the pod's copy of the manifest ------------------------------------------------------------------------

VARIANT = "q4_k_m-ovhrecipe"
NOTE = "produced on this pod from the bf16 GGUF with imatrix and tensor-type overrides"


def manifest_copy(tmp_path: Path) -> Path:
    path = tmp_path / "gguf_files.json"
    path.write_text(MANIFEST.read_text())
    return path


def add(tool, path: Path, sha: str = "a" * 64, size: int = 4_300_000_000):
    return tool.add_variant(
        path, VARIANT, "Olala-v1.0-mini-7A1B-q4_k_m-ovhrecipe.gguf", size, sha, NOTE
    )


def test_the_variant_is_added_with_the_measured_name_size_sha_and_a_note(tool, tmp_path):
    path = manifest_copy(tmp_path)
    before = json.loads(path.read_text())
    edit = add(tool, path)
    after = json.loads(path.read_text())
    entry = after["gguf"]["files"][VARIANT]
    assert entry["name"] == "Olala-v1.0-mini-7A1B-q4_k_m-ovhrecipe.gguf"
    assert entry["size"] == 4_300_000_000 and entry["sha256"] == "a" * 64
    assert "this pod" in entry["note"] and entry["derived_from"] == "bf16"
    assert {k: v for k, v in after["gguf"]["files"].items() if k != VARIANT} == before["gguf"][
        "files"
    ]
    assert after["llama_cpp"] == before["llama_cpp"] and after["model"] == before["model"]
    assert edit["changed"] is True and edit["sha256_before"] != edit["sha256_after"]
    assert edit["sha256_after"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_existing_gguf_tool_accepts_the_added_variant(tool, tmp_path):
    path = manifest_copy(tmp_path)
    payload = tmp_path / "Olala-v1.0-mini-7A1B-q4_k_m-ovhrecipe.gguf"
    payload.write_bytes(b"quantized bytes")
    tool.add_variant(
        path,
        VARIANT,
        payload.name,
        payload.stat().st_size,
        hashlib.sha256(payload.read_bytes()).hexdigest(),
        NOTE,
    )
    gguf_tool = POD / "gguf_tool.py"
    resolved = subprocess.run(
        [sys.executable, "-I", str(gguf_tool), "--manifest", str(path), "resolve", VARIANT],
        capture_output=True,
        text=True,
        check=False,
    )
    assert resolved.stdout.strip() == VARIANT
    verified = subprocess.run(
        [
            sys.executable,
            "-I",
            str(gguf_tool),
            "--manifest",
            str(path),
            "verify",
            "--variant",
            VARIANT,
            "--file",
            str(payload),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert verified.returncode == 0, verified.stdout + verified.stderr


def test_adding_the_same_entry_twice_changes_nothing(tool, tmp_path):
    path = manifest_copy(tmp_path)
    add(tool, path)
    text = path.read_text()
    edit = add(tool, path)
    assert path.read_text() == text and edit["changed"] is False


@pytest.mark.parametrize("variant", ["q5_k_m", "bf16", "q6_k", "q8_0-safe2"])
def test_a_pinned_ovh_variant_is_never_overwritten(tool, tmp_path, variant):
    path = manifest_copy(tmp_path)
    text = path.read_text()
    with pytest.raises(tool.Q4Error, match="pinned"):
        tool.add_variant(path, variant, "x.gguf", 1, "b" * 64, NOTE)
    assert path.read_text() == text


def test_a_sha256_that_is_not_a_sha256_is_refused(tool, tmp_path):
    path = manifest_copy(tmp_path)
    with pytest.raises(tool.Q4Error, match="sha256"):
        add(tool, path, sha="xyz")


def test_a_new_file_of_another_hash_replaces_the_previous_entry_of_the_variant_and_says_so(
    tool, tmp_path
):
    path = manifest_copy(tmp_path)
    add(tool, path, sha="a" * 64)
    edit = add(tool, path, sha="c" * 64)
    assert edit["changed"] is True and edit["replaced_sha256"] == "a" * 64
    assert json.loads(path.read_text())["gguf"]["files"][VARIANT]["sha256"] == "c" * 64


# --- the tokenizer files for vllm bench --------------------------------------------------------------------

REPO_FILES = {
    "config.json": json.dumps(
        {
            "auto_map": {
                "AutoConfig": "configuration_olala.OlalaConfig",
                "AutoModelForCausalLM": "modeling_olala.OlalaForCausalLM",
            }
        }
    ),
    "configuration_olala.py": "class OlalaConfig: pass\n",
    "modeling_olala.py": "class OlalaForCausalLM: pass\n",
    "tokenizer.json": "{}",
    "tokenizer_config.json": json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"}),
    "chat_template.jinja": "{{ messages }}",
    "generation_config.json": "{}",
    "special_tokens_map.json": "{}",
    "model-00001-of-00004.safetensors": "weights",
    "model.safetensors.index.json": "{}",
    "README.md": "readme",
}


def fake_snapshot(files: dict[str, str]):
    """A snapshot_download that honours allow_patterns like the real one and records its arguments."""
    import fnmatch

    seen: dict = {}

    def snapshot(**kwargs):
        seen.update(kwargs)
        dest = Path(kwargs["local_dir"])
        dest.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            if any(fnmatch.fnmatch(name, pat) for pat in kwargs["allow_patterns"]):
                (dest / name).write_text(content)
        return str(dest)

    return snapshot, seen


def test_only_tokenizer_and_config_files_are_downloaded_at_the_pinned_revision_with_no_token(
    tool, tmp_path
):
    snapshot, seen = fake_snapshot(REPO_FILES)
    dest = tmp_path / "olala-tokenizer"
    record = tool.fetch_tokenizer(dest, snapshot=snapshot)
    assert seen["repo_id"] == MODEL_REPO and seen["revision"] == MODEL_REVISION
    assert seen["token"] is False and seen["local_dir"] == str(dest)
    for needed in (
        "tokenizer.json",
        "tokenizer_config.json",
        "config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
    ):
        assert (dest / needed).is_file(), needed
    assert not list(dest.glob("*.safetensors")) and not (dest / "README.md").exists()
    assert not any(
        p in ("*", "*.safetensors", "*.bin", "*.gguf", "*.pt") for p in seen["allow_patterns"]
    )
    assert record["revision"] == MODEL_REVISION and "tokenizer.json" in record["files"]


def test_the_pin_of_the_tokenizer_repository_is_the_one_of_setup_remote():
    assert f"revision={MODEL_REVISION}" in (POD / "setup_remote.sh").read_text()
    text = TOOL.read_text()
    assert MODEL_REVISION in text and MODEL_REPO in text


@pytest.mark.parametrize(
    ("missing", "needle"),
    [
        ("tokenizer.json", "tokenizer"),
        ("tokenizer_config.json", "tokenizer_config.json"),
        ("config.json", "config.json"),
        ("chat_template.jinja", "chat template"),
        ("configuration_olala.py", "configuration_olala.py"),
    ],
)
def test_a_missing_needed_file_is_refused(tool, tmp_path, missing, needle):
    files = {k: v for k, v in REPO_FILES.items() if k != missing}
    snapshot, _ = fake_snapshot(files)
    with pytest.raises(tool.Q4Error, match=needle):
        tool.fetch_tokenizer(tmp_path / "t", snapshot=snapshot)


def test_a_chat_template_inside_tokenizer_config_is_accepted(tool, tmp_path):
    files = {k: v for k, v in REPO_FILES.items() if k != "chat_template.jinja"}
    files["tokenizer_config.json"] = json.dumps({"chat_template": "{{ x }}"})
    snapshot, _ = fake_snapshot(files)
    assert tool.fetch_tokenizer(tmp_path / "t", snapshot=snapshot)["files"]


def test_a_weight_file_in_the_folder_is_refused(tool, tmp_path):
    snapshot, _ = fake_snapshot(REPO_FILES)

    def greedy(**kwargs):
        kwargs["allow_patterns"] = [*kwargs["allow_patterns"], "*.safetensors"]
        return snapshot(**kwargs)

    with pytest.raises(tool.Q4Error, match="weight"):
        tool.fetch_tokenizer(tmp_path / "t", snapshot=greedy)


def test_a_complete_folder_is_not_downloaded_again(tool, tmp_path):
    snapshot, _ = fake_snapshot(REPO_FILES)
    dest = tmp_path / "t"
    tool.fetch_tokenizer(dest, snapshot=snapshot)

    def never(**_kwargs):
        raise AssertionError("complete folder: no download")

    assert tool.fetch_tokenizer(dest, snapshot=never)["downloaded"] is False


# --- the bench points --------------------------------------------------------------------------------------


def write_point(path: Path, completed: int, failed: int | None, prompts: int, tput: float = 100.0):
    data = {
        "completed": completed,
        "num_prompts": prompts,
        "output_throughput": tput,
        "median_ttft_ms": 50.0,
        "median_tpot_ms": 10.0,
    }
    if failed is not None:
        data["failed"] = failed
    path.write_text(json.dumps(data))
    return path


def test_a_point_with_every_request_completed_is_clean(tool, tmp_path):
    point = tool.check_point(write_point(tmp_path / "c1_r1.json", 8, 0, 8), 8)
    assert (
        point["failed_requests"] == 0
        and point["clean"] is True
        and point["output_throughput"] == 100.0
    )


@pytest.mark.parametrize(("completed", "failed"), [(7, 1), (6, None), (8, 2)])
def test_a_point_with_failed_requests_is_not_clean(tool, tmp_path, completed, failed):
    point = tool.check_point(write_point(tmp_path / "p.json", completed, failed, 8), 8)
    assert point["clean"] is False and point["failed_requests"] >= 1


def test_the_cli_exits_3_on_failed_requests_and_2_on_an_unreadable_point(tmp_path):
    ok = write_point(tmp_path / "ok.json", 8, 0, 8)
    bad = write_point(tmp_path / "bad.json", 5, 3, 8)
    assert cli("bench-point", str(ok), "--expect-prompts", "8").returncode == 0
    out = cli("bench-point", str(bad), "--expect-prompts", "8")
    assert out.returncode == 3 and "failed=3" in out.stdout
    assert (
        cli("bench-point", str(tmp_path / "missing.json"), "--expect-prompts", "8").returncode == 2
    )


def test_the_summary_has_the_median_of_the_repeats_and_the_failed_requests_per_point(
    tool, tmp_path
):
    write_point(tmp_path / "c1_r1.json", 8, 0, 8, 90.0)
    write_point(tmp_path / "c1_r2.json", 8, 0, 8, 110.0)
    write_point(tmp_path / "c4_r1.json", 11, 1, 12, 300.0)
    summary = tool.summarize_bench(tmp_path, "q4_k_m-ovhrecipe")
    by = {p["concurrency"]: p for p in summary["points"]}
    assert by[1]["output_throughput_median"] == 100.0 and by[1]["repeats"] == 2
    assert by[4]["failed_requests"] == 1
    assert summary["points_with_failed_requests"] == 1
    assert json.loads((tmp_path / "summary.json").read_text()) == summary


# --- small commands used by the shell scripts --------------------------------------------------------------


def test_the_sha256_command_prints_the_digest_of_a_file(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    out = cli("sha256", str(path))
    assert out.stdout.strip() == hashlib.sha256(b"abc").hexdigest() and out.returncode == 0


def test_the_bits_per_weight_command_prints_key_value_lines(tmp_path):
    gguf = tiny_gguf(tmp_path / "tiny.gguf", [[1000, 1000]])
    out = cli("bits-per-weight", "--size", "2000000", "--gguf", str(gguf))
    lines = dict(ln.split("=", 1) for ln in out.stdout.splitlines())
    assert lines["parameters"] == "1000000" and lines["bits_per_weight"] == "16.0"
    assert "header" in lines["parameters_source"]


def test_the_report_is_built_from_the_markers_and_the_json_records(tool, tmp_path):
    steps = tmp_path / "steps"
    steps.mkdir()
    (steps / "build_extra.done").write_text("commit=abc\ncmake_flags=-DX=1 -DY=2\njobs=3\n")
    (steps / "imatrix.done").write_text(
        "mode=make\ncommand=llama-imatrix -m a\nsha256=" + "1" * 64 + "\n"
    )
    (steps / "quantize.done").write_text(
        "name=q.gguf\nsize=100\nsha256="
        + "2" * 64
        + "\nparameters=50\nbits_per_weight=16.0\ncommand=llama-quantize x\n"
    )
    (tmp_path / "wikitext.json").write_text(json.dumps({"text_sha256": "3" * 64}))
    (tmp_path / "manifest_edit.json").write_text(json.dumps({"changed": True}))
    (tmp_path / "tensor_types.json").write_text(
        json.dumps(
            {
                "n_tensors": 3,
                "by_type": {"bf16": 3},
                "by_group": {},
                "recipe_applied": True,
                "quant_size_line": "x",
            }
        )
    )
    report = tool.build_report(tmp_path, "done", "")
    assert report["status"] == "done" and report["amendment"] == 24
    assert report["llama_cpp"]["commit"] == "abc" and report["llama_cpp"]["jobs"] == "3"
    assert (
        report["imatrix"]["sha256"] == "1" * 64
        and report["quantize"]["command"] == "llama-quantize x"
    )
    assert (
        report["quantized"]["sha256"] == "2" * 64 and report["quantized"]["bits_per_weight"] == 16.0
    )
    assert report["wikitext"]["text_sha256"] == "3" * 64 and report["manifest_edit"] == {
        "changed": True
    }
    assert report["tensor_types"]["by_type"] == {"bf16": 3}
    assert report["recipe"]["tensor_types"] == {"ffn_latent_up": "bf16", "ffn_up_exps": "q8_0"}
    assert report["recipe"]["imatrix_chunks"] == 200


def test_a_failed_report_carries_the_reason_and_no_quantized_file(tool, tmp_path):
    report = tool.build_report(tmp_path, "failed", "SOMETHING")
    assert (
        report["status"] == "failed"
        and report["reason"] == "SOMETHING"
        and "quantized" not in report
    )


def test_the_pins_command_prints_the_model_repository_and_revision():
    out = cli("pins")
    lines = dict(ln.split("=", 1) for ln in out.stdout.splitlines())
    assert lines == {"model_repo": MODEL_REPO, "model_revision": MODEL_REVISION}
