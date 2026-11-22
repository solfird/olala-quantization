"""Turn an arm of arms.json into the `vllm serve` command of the study.

Commands: `list`, `argv ARM [--dir DIR] [--port N] [-- EXTRA...]` (one argument per line, for
exec), `shell ARM ...` (the same command quoted for a shell), `requires ARM` (the compute-capability
major the arm needs, or nothing), `logname ARM [--dir DIR]` (the folder name for logs). An arm named
X has an X_noignore variant without the ignore list when arms.json says `ignore_variants`. `ckpt`
serves an offline checkpoint directory. Standard library only (it runs on the pod and on the Mac).
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

ARMS_FILE = Path(__file__).resolve().parent / "arms.json"


def load() -> dict:
    """Read arms.json."""
    return json.loads(ARMS_FILE.read_text())


def arm_names(data: dict) -> list[str]:
    """Every arm name, with the _noignore variants and `ckpt`."""
    names: list[str] = []
    for name, arm in data["arms"].items():
        names.append(name)
        if arm["ignore_variants"]:
            names.append(f"{name}_noignore")
    return [*names, "ckpt"]


def resolve(data: dict, arm: str) -> tuple[dict, bool]:
    """The arm entry and whether the ignore list applies."""
    if arm == "ckpt":
        return data["ckpt"], False
    base, noignore = (arm[: -len("_noignore")], True) if arm.endswith("_noignore") else (arm, False)
    entry = data["arms"].get(base)
    if entry is None or (noignore and not entry["ignore_variants"]):
        names = ", ".join(arm_names(data))
        msg = f"unknown arm '{arm}': choose one of {names}"
        raise ValueError(msg)
    return entry, bool(entry["ignore_variants"]) and not noignore


def build_argv(
    arm: str,
    *,
    model_dir: str | None = None,
    port: int | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    """The full `vllm serve` argument vector of an arm."""
    data = load()
    entry, use_ignore = resolve(data, arm)
    if arm == "ckpt":
        if not model_dir:
            msg = "arm 'ckpt' needs a checkpoint DIR"
            raise ValueError(msg)
    elif model_dir:
        msg = f"--dir only applies to the ckpt arm, not '{arm}'"
        raise ValueError(msg)
    argv = ["vllm", "serve", model_dir or data["model_dir"], *data["common_flags"]]
    if entry.get("quantization"):
        argv += ["--quantization", entry["quantization"]]
        if use_ignore:
            config = json.dumps({"ignore": data["ignore"]}, separators=(",", ":"))
            argv += ["--quantization-config", config]
    argv += ["--host", data["host"], "--port", str(port or data["port"])]
    return argv + list(extra or [])


def requires_cc_major(arm: str) -> int | None:
    """The compute-capability major the arm needs (e.g. 10 for NVFP4), or None."""
    entry, _ = resolve(load(), arm)
    return entry.get("requires_cc_major")


def log_name(arm: str, model_dir: str | None) -> str:
    """Folder name for the arm's logs: the arm, plus the directory name for a checkpoint."""
    if arm == "ckpt" and model_dir:
        return "ckpt-" + Path(model_dir.rstrip("/")).name
    return arm


def _split(args: list[str]) -> tuple[list[str], list[str]]:
    if "--" in args:
        i = args.index("--")
        return args[:i], args[i + 1 :]
    return args, []


def main(argv: list[str]) -> int:
    """Command line entry point."""
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    command, rest = argv[0], argv[1:]
    try:
        if command == "list":
            data = load()
            for name in arm_names(data):
                entry, _ = resolve(data, name)
                print(f"{name:18} {entry['description']}")
            return 0
        head, extra = _split(rest)
        arm = head[0]
        opts = {head[i]: head[i + 1] for i in range(1, len(head) - 1, 2)}
        model_dir = opts.get("--dir")
        port = int(opts["--port"]) if "--port" in opts else None
        if command == "argv":
            print("\n".join(build_argv(arm, model_dir=model_dir, port=port, extra=extra)))
        elif command == "shell":
            print(shlex.join(build_argv(arm, model_dir=model_dir, port=port, extra=extra)))
        elif command == "requires":
            need = requires_cc_major(arm)
            print("" if need is None else need)
        elif command == "logname":
            print(log_name(arm, model_dir))
        else:
            print(f"unknown command {command}", file=sys.stderr)
            return 2
    except (ValueError, IndexError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
