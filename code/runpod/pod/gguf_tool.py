r"""Pinned facts and helpers for OVH's published Olala GGUF files and OVH's llama.cpp fork.

The facts live in gguf_files.json (read from public pages on 2026-11-20): the fork's commit and
tree, the CMake flags, the Hub repository, its revision and, for each GGUF file, the name, size
and sha256 of the Hub's LFS metadata. This module reads them for the shell scripts of the pod
(gguf_setup.sh, serve_gguf.sh) and holds the small pure functions worth a unit test: file
verification, the Hub download (no token), the CUDA architecture of the build and the memory plan
of `-np N -c C`.

Commands (run with `python3 -I`, standard library only; the Hub download imports huggingface_hub
only when asked):
  get KEY            print a manifest value, KEY like llama_cpp.commit or gguf.files.q5_k_m.name
  arch CC            the CMAKE_CUDA_ARCHITECTURES value of a compute capability (12.0 -> 120a-real)
  resolve TEXT       a variant, a pinned file name or a path ending in one -> the variant key
  plan --variant V --np N [--ctx C] [--gpu-mem-mib M]    memory plan as JSON
  verify --variant V --file F [--size-only]    size and sha256 against the pins (0 ok, 1 mismatch)
  fetch --variant V --dest DIR    download the pinned revision (no token), then verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

MANIFEST = Path(__file__).with_name("gguf_files.json")
CHUNK = 8 * 1024 * 1024
MIB = 2**20
PAD = 256  # src/llama-context.cpp:298: n_ctx_seq = GGML_PAD(n_ctx / n_seq_max, 256)


class GgufError(Exception):
    """A pin does not hold or an argument is wrong; the message is for the operator."""


def load_manifest(path: Path | None = None) -> dict:
    """The pinned facts."""
    return json.loads((path or MANIFEST).read_text())


def variant_entry(manifest: dict, variant: str) -> dict:
    """The name, size and sha256 of one GGUF file."""
    files = manifest["gguf"]["files"]
    if variant not in files:
        raise GgufError(f"unknown variant '{variant}'; known: {', '.join(sorted(files))}")
    return files[variant]


def resolve_variant(manifest: dict, text: str) -> str:
    """A variant key (q5_k_m), a pinned file name or a path ending in one -> the variant key."""
    files = manifest["gguf"]["files"]
    if text in files:
        return text
    base = Path(text).name
    for key, entry in files.items():
        if entry["name"] == base:
            return key
    known = ", ".join(sorted(files))
    raise GgufError(f"'{text}' is neither a variant nor a pinned file name; variants: {known}")


def sha256_of(path: Path) -> str:
    """sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, entry: dict, size_only: bool = False) -> tuple[bool, str]:
    """Compare a file with its pins: the size first (cheap), then the sha256 unless `size_only`."""
    if not path.is_file():
        return False, f"MISSING {path}"
    size = path.stat().st_size
    if size != entry["size"]:
        return False, f"FAIL {path.name} size {size} != pinned {entry['size']}"
    if size_only:
        return True, f"OK {path.name} size {size} bytes (sha256 not computed)"
    got = sha256_of(path)
    if got != entry["sha256"]:
        return False, f"FAIL {path.name} sha256 {got} != pinned {entry['sha256']}"
    return True, f"OK {path.name} {size} bytes sha256 {got}"


def fetch(
    manifest: dict,
    variant: str,
    dest: Path,
    download: Callable[..., str] | None = None,
) -> Path:
    """The pinned file in `dest`: already there and verified, else fetched at the pinned revision.

    `token=False`: the repository is public and no credential may reach a pod. A file that does
    not match the pins after the download is an error (nothing is deleted: the evidence stays).
    """
    entry = variant_entry(manifest, variant)
    path = dest / entry["name"]
    if path.is_file() and verify(path, entry)[0]:
        return path
    dest.mkdir(parents=True, exist_ok=True)
    if download is None:
        from huggingface_hub import hf_hub_download  # only on the pod, only when downloading

        download = hf_hub_download
    download(
        repo_id=manifest["gguf"]["repo"],
        filename=entry["name"],
        revision=manifest["gguf"]["revision"],
        local_dir=str(dest),
        token=False,
    )
    ok, message = verify(path, entry)
    if not ok:
        raise GgufError(f"the download does not match the pins: {message}")
    return path


def cuda_arch(cc: str) -> str:
    """CMAKE_CUDA_ARCHITECTURES for a compute capability such as 12.0.

    Blackwell (12.x, 10.x) and Hopper need the architecture-specific 'a' variant for their newest
    instructions (ggml/src/ggml-cuda/CMakeLists.txt:92-106 rewrites a plain 12X to 12Xa too).
    """
    match = re.fullmatch(r"(\d+)\.(\d+)", cc.strip())
    if not match:
        raise GgufError(f"not a compute capability: '{cc}' (expected like 12.0)")
    major, minor = int(match.group(1)), int(match.group(2))
    suffix = "a" if major >= 9 else ""
    return f"{major}{minor}{suffix}-real"


def default_ctx(np: int, slot_ctx: int = 32768) -> int:
    """The -c that gives each of np slots slot_ctx tokens (the model's 32K, the card's maximum)."""
    return np * slot_ctx


def slot_ctx(ctx: int, np: int) -> int:
    """Tokens one slot holds: -c divided by -np, padded to 256 (non-unified KV cache)."""
    per_slot = ctx // np
    return -(-per_slot // PAD) * PAD


def memory_plan(manifest: dict, variant: str, np: int, ctx: int) -> dict:
    """Device memory of `-np np -c ctx` with an f16 KV cache. Only the overhead is a guess.

    The attention KV cache is one tensor of n_ctx_seq x n_seq_max cells for the 7 attention
    layers (src/llama-kv-cache.cpp:233-234), so its total is -c whatever -np is; the recurrent
    state is one f32 block per sequence for all 36 layers.
    """
    model = manifest["model"]
    entry = variant_entry(manifest, variant)
    kv_bytes_per_token = (
        model["attention_layers"]
        * 2
        * model["kv_heads"]
        * model["head_dim"]
        * model["kv_bytes_per_element"]
    )
    recurrent_bytes = (
        model["layers"] * model["recurrent_floats_per_layer"] * model["recurrent_bytes_per_element"]
    )
    plan = {
        "variant": variant,
        "np": np,
        "ctx": ctx,
        "slot_ctx": slot_ctx(ctx, np),
        "kv_bytes_per_token": kv_bytes_per_token,
        "recurrent_bytes_per_slot": recurrent_bytes,
        "weights_mib": entry["size"] / MIB,
        "kv_mib": ctx * kv_bytes_per_token / MIB,
        "recurrent_mib": np * recurrent_bytes / MIB,
        "assumed_overhead_mib": model["assumed_overhead_mib"],
    }
    plan["certain_mib"] = plan["weights_mib"] + plan["kv_mib"] + plan["recurrent_mib"]
    plan["estimate_mib"] = plan["certain_mib"] + plan["assumed_overhead_mib"]
    return plan


def _get(manifest: dict, dotted: str) -> object:
    node: object = manifest
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise GgufError(f"no such key: {dotted}")
        node = node[part]
    return node


def main(argv: list[str]) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("get").add_argument("key")
    sub.add_parser("arch").add_argument("cc")
    sub.add_parser("resolve").add_argument("text")
    plan_p = sub.add_parser("plan")
    plan_p.add_argument("--variant")
    plan_p.add_argument("--np", type=int, required=True)
    plan_p.add_argument("--ctx", type=int)
    plan_p.add_argument("--gpu-mem-mib", type=float)
    verify_p = sub.add_parser("verify")
    verify_p.add_argument("--variant")
    verify_p.add_argument("--file", type=Path, required=True)
    verify_p.add_argument(
        "--size-only", action="store_true", help="skip the sha256 (a start, not a download)"
    )
    fetch_p = sub.add_parser("fetch")
    fetch_p.add_argument("--variant")
    fetch_p.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
        variant = getattr(args, "variant", None) or manifest["gguf"]["default"]
        if args.command == "get":
            value = _get(manifest, args.key)
            print(value if isinstance(value, str) else json.dumps(value))
        elif args.command == "arch":
            print(cuda_arch(args.cc))
        elif args.command == "resolve":
            print(resolve_variant(manifest, args.text))
        elif args.command == "plan":
            ctx = args.ctx or default_ctx(args.np, manifest["model"]["slot_ctx"])
            plan = memory_plan(manifest, variant, args.np, ctx)
            if args.gpu_mem_mib is not None:
                plan["gpu_mem_mib"] = args.gpu_mem_mib
                plan["fits_certain"] = plan["certain_mib"] <= args.gpu_mem_mib - 1024
                plan["fits_estimate"] = plan["estimate_mib"] <= args.gpu_mem_mib
            print(json.dumps(plan, indent=2))
        elif args.command == "verify":
            ok, message = verify(args.file, variant_entry(manifest, variant), args.size_only)
            print(message)
            return 0 if ok else 1
        elif args.command == "fetch":
            print(fetch(manifest, variant, args.dest))
    except GgufError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
