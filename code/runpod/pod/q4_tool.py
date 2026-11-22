r"""Helpers of arm Q4 (amendment 24): OVH's quantization recipe at Q4_K_M, then a speed sweep.

The shell scripts of the pod (q4_prepare.sh, q4_driver.sh, bench_gguf.sh) call this module for
what is worth a unit test: the wikitext-2 text file, the parameter count of a GGUF header, the
parse of llama-quantize's log, the edit of the POD's copy of gguf_files.json, the tokenizer files
for `vllm bench serve`, the check of a bench point and the assembly of results/q4/prepare.json.

Commands (run with `python3 -I`; huggingface_hub and pyarrow are imported only when needed):
  wikitext --out TEXT --cache DIR [--parquet FILE]  fetch wikitext-2 raw train (pinned), write text
  params FILE                                       parameter count of a GGUF header (JSON)
  sha256 FILE                                       sha256 of a file
  pins                                              the pinned model repository and revision
  bits-per-weight --size N --gguf BF16              key=value lines: parameters, bits per weight
  quantize-table LOG [--json OUT]                   final tensor types of a llama-quantize log
                                                    (exit 1: the recipe was not applied)
  manifest-add --manifest M --variant V --name N --size S --sha256 H --note T [--record F]
  manifest-restore --manifest M --record F          add the variant again (a fresh bundle lost it)
  tokenizer --dest DIR                              tokenizer and config files of the pinned model
                                                    revision, no weights
  bench-point FILE --expect-prompts N               completed and failed requests of one result
  bench-summary --dir DIR --variant V               summary.json of a sweep directory
  report --q4dir DIR --out FILE --status S [--reason R]   prepare.json from the step markers
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import struct
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path

WIKITEXT_REPO = "Salesforce/wikitext"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_FILE = "wikitext-2-raw-v1/train-00000-of-00001.parquet"

# setup_remote.sh and lib.sh pin the same revision of the safetensors repository
# (a test compares them)
MODEL_REPO = "OVHaiLLM/Olala-v1.0-mini-7A1B"
MODEL_REVISION = "b40f8165ee2254c18601220e19842600c6d91794"
TOKENIZER_PATTERNS = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "added_tokens.json",
    "special_tokens_map.json",
    "config.json",
    "generation_config.json",
    "chat_template.jinja",
    "chat_template.json",
    "configuration_*.py",
    "tokenization_*.py",
]
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf", ".pt", ".pth", ".onnx", ".ckpt")
TOKENIZER_MAX_BYTES = (
    64 * 1024 * 1024
)  # a tokenizer.json is about 10 MB; a weight shard is gigabytes

APPROX_PARAMETERS = (
    6_766_000_000  # amendment 24: "6.77 billion", used only when the header cannot be read
)
TENSOR_TYPE_OVERRIDES = {
    "ffn_latent_up": "bf16",
    "ffn_up_exps": "q8_0",
}  # OVH's recipe (model card)
GGUF_SCALARS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
TENSOR_LINE = re.compile(r"\[\s*(\d+)/\s*(\d+)\]\s+(\S+)\s+-\s+\[[^\]]*\],\s+type\s*=\s*(\S+),")
CONVERTING = re.compile(r"converting to\s+(\S+)")
COPIED = re.compile(r"\bsize\s*=")
CONVERSION_LOOKAHEAD = 6  # lines after a tensor header that may hold its conversion text


class Q4Error(Exception):
    """A pin does not hold, a file is missing or an argument is wrong."""


def sha256_of(path: Path) -> str:
    """sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


# --- wikitext-2 ----------------------------------------------------------------------------------


def _read_text_rows(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq  # only on the pod, only when a parquet is read
    except ImportError as exc:
        raise Q4Error("pyarrow is not installed (pip install pyarrow)") from exc
    return pq.read_table(path, columns=["text"]).column("text").to_pylist()


def fetch_wikitext(
    out: Path,
    cache: Path,
    download: Callable[..., str] | None = None,
    read_rows: Callable[[Path], list[str]] | None = None,
    parquet: Path | None = None,
) -> dict:
    """Write wikitext-2 raw train as one text file: the `text` rows in order, concatenated.

    No separator is added: the rows already end with their newlines (this is how llama.cpp's
    wiki.train.raw is made). The parquet is the pinned file of the pinned dataset revision (no
    token). `parquet` replaces the download (the tests use it).
    """
    if parquet is not None:
        source = f"local parquet {parquet}"
        path = parquet
    else:
        if download is None:
            from huggingface_hub import hf_hub_download  # only on the pod

            download = hf_hub_download
        cache.mkdir(parents=True, exist_ok=True)
        path = Path(
            download(
                repo_id=WIKITEXT_REPO,
                repo_type="dataset",
                revision=WIKITEXT_REVISION,
                filename=WIKITEXT_FILE,
                local_dir=str(cache),
                token=False,
            )
        )
        source = "hub"
    rows = (read_rows or _read_text_rows)(path)
    for i, row in enumerate(rows):
        if not isinstance(row, str):
            raise Q4Error(f"row {i} of {path.name} is not text ({type(row).__name__})")
    data = "".join(rows).encode("utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return {
        "repo": WIKITEXT_REPO,
        "revision": WIKITEXT_REVISION,
        "file": WIKITEXT_FILE,
        "source": source,
        "parquet_sha256": sha256_of(path),
        "rows": len(rows),
        "text_bytes": len(data),
        "text_sha256": hashlib.sha256(data).hexdigest(),
        "construction": (
            "the text rows in order, concatenated with no separator added "
            "(rows already end with newlines)"
        ),
    }


# --- the GGUF header --------------------------------------------------------------------------


def _read_exact(handle, n: int) -> bytes:  # noqa: ANN001 - a binary file object
    if n > 1 << 30:
        raise Q4Error("the GGUF header is corrupt (a field of more than 1 GiB)")
    data = handle.read(n)
    if len(data) != n:
        raise Q4Error("the GGUF header ends early")
    return data


def _skip_value(handle, vtype: int) -> None:  # noqa: ANN001
    if vtype in GGUF_SCALARS:
        _read_exact(handle, struct.calcsize("<" + GGUF_SCALARS[vtype]))
    elif vtype == 8:
        (length,) = struct.unpack("<Q", _read_exact(handle, 8))
        _read_exact(handle, length)
    elif vtype == 9:
        etype, count = struct.unpack("<IQ", _read_exact(handle, 12))
        if etype in GGUF_SCALARS:
            handle.seek(count * struct.calcsize("<" + GGUF_SCALARS[etype]), os.SEEK_CUR)
        else:
            for _ in range(count):
                _skip_value(handle, etype)
    else:
        raise Q4Error(f"unknown GGUF value type {vtype}")


def gguf_parameter_count(path: Path) -> int:
    """The number of tensor elements of a GGUF file, from its header only."""
    with path.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise Q4Error(f"{path.name} is not a GGUF file")
        version, n_tensors, n_kv = struct.unpack("<IQQ", _read_exact(handle, 20))
        if version < 2:
            raise Q4Error(f"GGUF version {version} is not supported")
        for _ in range(n_kv):
            (length,) = struct.unpack("<Q", _read_exact(handle, 8))
            _read_exact(handle, length)
            (vtype,) = struct.unpack("<I", _read_exact(handle, 4))
            _skip_value(handle, vtype)
        total = 0
        for _ in range(n_tensors):
            (length,) = struct.unpack("<Q", _read_exact(handle, 8))
            _read_exact(handle, length)
            (n_dims,) = struct.unpack("<I", _read_exact(handle, 4))
            dims = struct.unpack(f"<{n_dims}Q", _read_exact(handle, 8 * n_dims))
            _read_exact(handle, 12)  # type and data offset
            elements = 1
            for dim in dims:
                elements *= dim
            total += elements
    return total


def bits_per_weight(size_bytes: int, bf16_gguf: Path) -> dict:
    """size*8 / parameters; the parameters come from the bf16 header, else the approximation."""
    try:
        parameters = gguf_parameter_count(bf16_gguf)
        source = f"sum of the tensor elements in the header of {bf16_gguf.name}"
    except (Q4Error, OSError, struct.error) as exc:
        parameters = APPROX_PARAMETERS
        source = (
            "approximate: 6,766,000,000 stated by amendment 24 "
            f"(the header could not be read: {exc})"
        )
    return {
        "size_bytes": size_bytes,
        "parameters": parameters,
        "parameters_source": source,
        "bits_per_weight": size_bytes * 8 / parameters,
    }


# --- the log of llama-quantize -------------------------------------------------------------------


def parse_quantize_log(text: str) -> dict:
    """The final type of every tensor in llama-quantize's log, counted, and did the overrides work?

    A tensor line names the source type and then either `converting to T` (the final type) or `size`
    (copied as it is).
    """
    tensors = []
    declared = 0
    unresolved = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        match = TENSOR_LINE.search(line)
        if not match:
            continue
        declared = int(match.group(2))
        source = match.group(4)
        # A tensor with no imatrix data has a warning line between its header and the conversion
        # text, so the text is searched on the header line and the few lines after it, up to the
        # next tensor header.
        final = None
        tail = line[match.end() :]
        window = [tail, *lines[i + 1 : i + 1 + CONVERSION_LOOKAHEAD]]
        for k, candidate in enumerate(window):
            if k > 0 and TENSOR_LINE.search(candidate):
                break
            converting = CONVERTING.search(candidate)
            if converting:
                final = converting.group(1)
                break
            if COPIED.search(candidate):
                final = source
                break
        if final is None:
            unresolved.append(match.group(3))
            final = source
        tensors.append({"name": match.group(3), "source": source, "final": final})
    by_type = Counter(t["final"] for t in tensors)
    by_group: dict[str, dict[str, int]] = {}
    problems = []
    if not tensors:
        problems.append("no tensor line in the log")
    if unresolved:
        problems.append(
            f"no conversion text for {len(unresolved)} tensor(s), first: {unresolved[0]}"
        )
    if tensors and len(tensors) != declared:
        problems.append(
            f"the log has {len(tensors)} tensor lines, the last line declares {declared} tensors"
        )
    for group, wanted in TENSOR_TYPE_OVERRIDES.items():
        members = [t for t in tensors if group in t["name"]]
        by_group[group] = dict(Counter(t["final"] for t in members))
        if tensors and not members:
            problems.append(f"override {group}={wanted}: no tensor of that name in the log")
        wrong = [t["name"] for t in members if t["final"].lower() != wanted]
        if wrong:
            problems.append(
                f"override {group}={wanted} not applied to {len(wrong)} tensor(s), "
                f"first: {wrong[0]}"
            )

    def last(prefix: str) -> str:
        return next(
            (ln.strip() for ln in reversed(text.splitlines()) if prefix in ln and "BPW" in ln), ""
        )

    return {
        "n_tensors": len(tensors),
        "declared_tensors": declared,
        "by_type": dict(sorted(by_type.items())),
        "by_group": by_group,
        "problems": problems,
        "recipe_applied": not problems,
        "model_size_line": last("model size"),
        "quant_size_line": last("quant size"),
        "tensors": tensors,
    }


# --- the pod's copy of the manifest -----------------------------------------------------------


def _write_json(path: Path, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(data, indent=2) + "\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def add_variant(manifest: Path, variant: str, name: str, size: int, sha256: str, note: str) -> dict:
    """Add the measured Q4 file to the manifest at `manifest` (the pod's copy) for the gguf scripts.

    A pinned OVH entry (one without `derived_from`) is never touched. Adding the same entry again
    changes nothing.
    """
    if not SHA_RE.match(sha256):
        raise Q4Error(f"not a sha256: '{sha256}'")
    data = json.loads(manifest.read_text())
    files = data["gguf"]["files"]
    existing = files.get(variant)
    if existing is not None and "derived_from" not in existing:
        raise Q4Error(f"'{variant}' is a pinned OVH file: not overwritten")
    entry = {
        "name": name,
        "size": size,
        "sha256": sha256,
        "derived_from": "bf16",
        "note": note,
    }
    before = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if existing == entry:
        return {
            "changed": False,
            "sha256_before": before,
            "sha256_after": before,
            "manifest": str(manifest),
        }
    files[variant] = entry
    _write_json(manifest, data)
    return {
        "changed": True,
        "sha256_before": before,
        "sha256_after": sha256_of(manifest),
        "replaced_sha256": existing["sha256"] if existing else None,
        "manifest": str(manifest),
        "entry": entry,
    }


# --- the tokenizer for `vllm bench serve` -----------------------------------------------------


def _tokenizer_problems(dest: Path) -> list[str]:
    problems = []
    names = {p.name for p in dest.iterdir()} if dest.is_dir() else set()
    for need in ("config.json", "tokenizer_config.json"):
        if need not in names:
            problems.append(f"{need} is missing")
    if "tokenizer.json" not in names and not {"vocab.json", "merges.txt"} <= names:
        problems.append("no tokenizer: neither tokenizer.json nor vocab.json with merges.txt")
    config = {}
    if "tokenizer_config.json" in names:
        config = json.loads((dest / "tokenizer_config.json").read_text())
    if not ({"chat_template.jinja", "chat_template.json"} & names or config.get("chat_template")):
        problems.append(
            "no chat template (chat_template.jinja, chat_template.json or one in "
            "tokenizer_config.json)"
        )
    # the remote code that transformers needs to read the config and the tokenizer
    for source in ("config.json", "tokenizer_config.json"):
        if source not in names:
            continue
        auto_map = json.loads((dest / source).read_text()).get("auto_map") or {}
        for key in ("AutoConfig", "AutoTokenizer"):
            targets = auto_map.get(key)
            for target in [targets] if isinstance(targets, str) else (targets or []):
                module = str(target).split(".")[0] + ".py"
                if target and "--" not in str(target) and module not in names:
                    problems.append(f"{module} is missing ({source} auto_map {key})")
    weights = sorted(
        p.name for p in dest.rglob("*") if p.is_file() and p.name.endswith(WEIGHT_SUFFIXES)
    )
    if weights:
        problems.append(f"weight file in the tokenizer folder: {weights[0]}")
    total = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file()) if dest.is_dir() else 0
    if total > TOKENIZER_MAX_BYTES:
        problems.append(f"the folder holds {total} bytes: more than a tokenizer (weights?)")
    return problems


def fetch_tokenizer(dest: Path, snapshot: Callable[..., str] | None = None) -> dict:
    """The tokenizer and config files of the pinned model revision, never the weights.

    Refuses on a missing file.
    """
    downloaded = False
    if (
        dest.is_dir()
        and not _tokenizer_problems(dest)
        and (dest / ".olala_revision").is_file()
        and ((dest / ".olala_revision").read_text().strip() == MODEL_REVISION)
    ):
        pass
    else:
        if snapshot is None:
            from huggingface_hub import snapshot_download  # only on the pod

            snapshot = snapshot_download
        dest.mkdir(parents=True, exist_ok=True)
        snapshot(
            repo_id=MODEL_REPO,
            revision=MODEL_REVISION,
            local_dir=str(dest),
            allow_patterns=list(TOKENIZER_PATTERNS),
            token=False,
        )
        downloaded = True
        problems = _tokenizer_problems(dest)
        if problems:
            raise Q4Error("tokenizer folder refused: " + "; ".join(problems))
        (dest / ".olala_revision").write_text(MODEL_REVISION + "\n")
    files = sorted(p.name for p in dest.iterdir() if p.is_file() and not p.name.startswith("."))
    return {
        "repo": MODEL_REPO,
        "revision": MODEL_REVISION,
        "dest": str(dest),
        "files": files,
        "downloaded": downloaded,
    }


# --- the bench points -------------------------------------------------------------------------


def check_point(path: Path, expect_prompts: int) -> dict:
    """Completed and failed requests of one `vllm bench serve` result (missing ones are failed)."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise Q4Error(f"unreadable bench result {path}: {exc}") from exc
    completed = int(data.get("completed", 0))
    failed = max(int(data.get("failed", 0) or 0), expect_prompts - completed)
    return {
        "file": path.name,
        "completed": completed,
        "failed_requests": failed,
        "expected_prompts": expect_prompts,
        "clean": failed == 0 and completed > 0,
        "output_throughput": data.get("output_throughput"),
        "median_ttft_ms": data.get("median_ttft_ms"),
        "median_tpot_ms": data.get("median_tpot_ms"),
    }


def summarize_bench(directory: Path, variant: str) -> dict:
    """summary.json of a sweep directory: per concurrency, the median and the failed requests."""
    by_conc: dict[int, list[dict]] = {}
    for path in sorted(directory.glob("c*_r*.json")):
        match = re.fullmatch(r"c(\d+)_r(\d+)\.json", path.name)
        if not match:
            continue
        conc = int(match.group(1))
        n = max(8, 3 * conc)
        by_conc.setdefault(conc, []).append(check_point(path, n))
    points = []
    for conc, items in sorted(by_conc.items()):
        tputs = [i["output_throughput"] for i in items if i["output_throughput"] is not None]
        points.append(
            {
                "concurrency": conc,
                "repeats": len(items),
                "num_prompts": max(8, 3 * conc),
                "output_throughput_median": statistics.median(tputs) if tputs else None,
                "output_throughput_each": tputs,
                "completed": sum(i["completed"] for i in items),
                "failed_requests": sum(i["failed_requests"] for i in items),
            }
        )
    summary = {
        "variant": variant,
        "points": points,
        "points_with_failed_requests": sum(1 for p in points if p["failed_requests"] > 0),
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


# --- prepare.json -----------------------------------------------------------------------------


def read_marker(path: Path) -> dict[str, str]:
    """A step marker: one key=value per line."""
    values: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep:
                values[key] = value
    return values


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def build_report(q4dir: Path, status: str, reason: str) -> dict:
    """results/q4/prepare.json from the step markers under q4dir/steps."""
    steps = {p.stem: read_marker(p) for p in sorted((q4dir / "steps").glob("*.done"))}
    build, helps = steps.get("build_extra", {}), steps.get("help", {})
    imatrix, quant = steps.get("imatrix", {}), steps.get("quantize", {})
    setup = steps.get("setup_bf16", {})
    wikitext = _read_json(q4dir / "wikitext.json")
    report: dict = {
        "schema": 1,
        "amendment": 24,
        "status": status,
        "reason": reason,
        "recipe": {
            "quant_type": "Q4_K_M",
            "tensor_types": TENSOR_TYPE_OVERRIDES,
            "imatrix_chunks": 200,
            "imatrix_data": f"{WIKITEXT_REPO} @ {WIKITEXT_REVISION}, {WIKITEXT_FILE}",
            "note": "OVH's own imatrix (olala-rl.imatrix) is not published: this one is built here",
        },
        "llama_cpp": {
            k: build.get(k)
            for k in ("commit", "tree", "cmake_flags", "cuda_arch", "build_command", "jobs")
        },
        "help_check": helps,
        "bf16": {k: setup.get(k) for k in ("file", "size", "sha256")},
        "wikitext": wikitext,
        "imatrix": {
            k: imatrix.get(k) for k in ("mode", "command", "sha256", "size", "log", "chunks_line")
        },
        "quantize": {k: quant.get(k) for k in ("command", "log", "threads")},
        "steps": steps,
    }
    if quant.get("sha256"):
        report["quantized"] = {
            "name": quant.get("name"),
            "path": quant.get("file"),
            "size_bytes": int(quant["size"]),
            "sha256": quant["sha256"],
            "parameters": int(quant["parameters"]) if quant.get("parameters") else None,
            "parameters_source": quant.get("parameters_source"),
            "bits_per_weight": float(quant["bits_per_weight"])
            if quant.get("bits_per_weight")
            else None,
        }
    table = q4dir / "tensor_types.json"
    if table.is_file():
        data = json.loads(table.read_text())
        report["tensor_types"] = {
            k: data[k]
            for k in ("n_tensors", "by_type", "by_group", "recipe_applied", "quant_size_line")
        }
    manifest = _read_json(q4dir / "manifest_edit.json")
    if manifest:
        report["manifest_edit"] = manifest
    return report


# --- command line -----------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    wiki = sub.add_parser("wikitext")
    wiki.add_argument("--out", type=Path, required=True)
    wiki.add_argument("--cache", type=Path, required=True)
    wiki.add_argument("--parquet", type=Path)
    sub.add_parser("params").add_argument("file", type=Path)
    sub.add_parser("sha256").add_argument("file", type=Path)
    sub.add_parser("pins")
    bpw = sub.add_parser("bits-per-weight")
    bpw.add_argument("--size", type=int, required=True)
    bpw.add_argument("--gguf", type=Path, required=True)
    table = sub.add_parser("quantize-table")
    table.add_argument("log", type=Path)
    table.add_argument("--json", type=Path)
    for name in ("manifest-add", "manifest-restore"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--record", type=Path)
        if name == "manifest-add":
            p.add_argument("--variant", required=True)
            p.add_argument("--name", required=True)
            p.add_argument("--size", type=int, required=True)
            p.add_argument("--sha256", required=True)
            p.add_argument("--note", required=True)
    sub.add_parser("tokenizer").add_argument("--dest", type=Path, required=True)
    point = sub.add_parser("bench-point")
    point.add_argument("file", type=Path)
    point.add_argument("--expect-prompts", type=int, required=True)
    summ = sub.add_parser("bench-summary")
    summ.add_argument("--dir", type=Path, required=True)
    summ.add_argument("--variant", required=True)
    rep = sub.add_parser("report")
    rep.add_argument("--q4dir", type=Path, required=True)
    rep.add_argument("--out", type=Path, required=True)
    rep.add_argument("--status", required=True)
    rep.add_argument("--reason", default="")
    args = parser.parse_args(argv)

    try:
        if args.command == "wikitext":
            print(json.dumps(fetch_wikitext(args.out, args.cache, parquet=args.parquet)))
        elif args.command == "params":
            print(json.dumps({"parameters": gguf_parameter_count(args.file)}))
        elif args.command == "pins":
            print(f"model_repo={MODEL_REPO}")
            print(f"model_revision={MODEL_REVISION}")
        elif args.command == "sha256":
            print(sha256_of(args.file))
        elif args.command == "bits-per-weight":
            for key, value in bits_per_weight(args.size, args.gguf).items():
                print(f"{key}={value}")
        elif args.command == "quantize-table":
            result = parse_quantize_log(args.log.read_text(errors="replace"))
            if args.json:
                args.json.write_text(json.dumps(result, indent=1) + "\n")
            print(json.dumps({k: v for k, v in result.items() if k != "tensors"}))
            return 0 if result["recipe_applied"] else 1
        elif args.command == "manifest-add":
            edit = add_variant(
                args.manifest, args.variant, args.name, args.size, args.sha256, args.note
            )
            if args.record:
                entry = json.loads(args.manifest.read_text())["gguf"]["files"][args.variant]
                args.record.write_text(
                    json.dumps({"variant": args.variant, "entry": entry}, indent=2) + "\n"
                )
            print(json.dumps(edit))
        elif args.command == "manifest-restore":
            record = json.loads(args.record.read_text())
            entry = record["entry"]
            edit = add_variant(
                args.manifest,
                record["variant"],
                entry["name"],
                entry["size"],
                entry["sha256"],
                entry["note"],
            )
            print(json.dumps(edit))
        elif args.command == "tokenizer":
            print(json.dumps(fetch_tokenizer(args.dest)))
        elif args.command == "bench-point":
            result = check_point(args.file, args.expect_prompts)
            print(
                f"completed={result['completed']} failed={result['failed_requests']} "
                f"output_throughput={result['output_throughput']}"
            )
            return 0 if result["clean"] else 3
        elif args.command == "bench-summary":
            print(json.dumps(summarize_bench(args.dir, args.variant)))
        elif args.command == "report":
            _write_json(args.out, build_report(args.q4dir, args.status, args.reason))
    except (Q4Error, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
