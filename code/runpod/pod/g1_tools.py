"""Helpers of pod/g1_session.sh: feasibility projection, gate, fidelity, checkpoint records.

Standard library only (it runs on the pod before anything else). Every number that decides
something (the thresholds of the gate, the build budget, the memory limit, the scale of the
projection) is an ARGUMENT: the session script holds them in one place and this file holds none.
Exit status: 0 ok or pass, 1 gate fail (a legitimate negative result) or run not complete,
2 unreadable or missing input, 3 feasibility refusal.

Subcommands::

    feasibility LOG --out F --scale S --n-layers N --budget-hours H --peak-gib-max G
    gate FIDELITY_JSON --out F --kl-max X --ppl-ratio-max Y --n5-kl A --n5-ppl-ratio B
         --n5-from TEXT --source TEXT
    fidelity COMPARE_JSON --reference NPZ --out F
    checkpoint DIR --out F
    run-status RUN_INFO_JSON
    calib-check DIR [--out F] [--list]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

GIB = 2**30
CHUNK = 8 * 1024 * 1024


def die(message: str, code: int = 2) -> None:
    """Print an error and exit with ``code``."""
    print(f"error: {message}", file=sys.stderr)
    sys.exit(code)


def write_json(path: str, data: dict) -> None:
    """Write ``data`` as indented JSON, atomically."""
    target = Path(path)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(target)


def read_json(path: str) -> dict:
    """Read a JSON object or exit 2."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError) as err:
        die(f"cannot read {path}: {err}")
    if not isinstance(data, dict):
        die(f"{path} is not a JSON object")
    return data


# --- feasibility -------------------------------------------------------------------------------


def parse_tables(text: str) -> tuple[list[dict], list[float]]:
    """The rows of the two tables that `pod_g1lite.sh feasibility` prints.

    The first table (header `layer captured forward_s ...`) gives, per layer, the forward
    seconds and the peak CUDA GB; the second (header `layer hessian_s solve_s ...`) gives the
    peak after the solve, GB. Raises ValueError on a row that is not numbers.
    """
    forward: list[dict] = []
    solve_peaks: list[float] = []
    mode = ""
    for raw in text.splitlines():
        line = raw.strip()
        tok = line.split()
        if line.startswith("layer") and "forward_s" in tok and "captured" in tok:
            mode = "forward"
            continue
        if line.startswith("layer") and "hessian_s" in tok and "solve_s" in tok:
            mode = "solve"
            continue
        if not tok or not tok[0].isdigit():
            if mode and not line.startswith("layer"):
                mode = ""
            continue
        if mode == "forward":
            if len(tok) < 6:
                raise ValueError(f"short forward row: {line!r}")
            row = {"layer": int(tok[0]), "forward_s": float(tok[2]), "peak_gb": float(tok[5])}
            forward.append(row)
        elif mode == "solve":
            if len(tok) < 5:
                raise ValueError(f"short calibrated row: {line!r}")
            solve_peaks.append(float(tok[4]))
    return forward, solve_peaks


def cmd_feasibility(a: argparse.Namespace) -> int:
    """Project the full build time and compare it with the budget and the memory limit."""
    try:
        rows, solve_peaks = parse_tables(Path(a.log).read_text(errors="replace"))
    except (OSError, ValueError) as err:
        die(f"feasibility output unreadable: {err}")
    if not rows:
        die(
            "no forward_s rows in the feasibility output (table `layer captured forward_s` missing)"
        )
    seconds = [r["forward_s"] for r in rows]
    if not all(math.isfinite(s) and s >= 0 for s in seconds):
        die("a forward_s value is not a finite non-negative number")
    mean = sum(seconds) / len(seconds)
    full_sum = mean * a.n_layers
    projected = a.scale * full_sum
    hours = projected / 3600
    peak_gb = max([r["peak_gb"] for r in rows] + solve_peaks)
    peak_gib = peak_gb * 1e9 / GIB
    refusals = []
    if hours > a.budget_hours:
        refusals.append(
            f"projected build {hours:.2f} h exceeds the budget of {a.budget_hours:g} h "
            f"({a.scale:g} x {full_sum:.1f} s)"
        )
    if peak_gib > a.peak_gib_max:
        refusals.append(f"peak GPU memory {peak_gib:.2f} GiB exceeds {a.peak_gib_max:g} GiB")
    formula = (
        f"projected build seconds = {a.scale:g} x (mean forward seconds of the {len(rows)} "
        f"measured layers x {a.n_layers} layers)"
    )
    data = {
        "formula": formula,
        "measured_layers": len(rows),
        "layers": rows,
        "mean_forward_s": mean,
        "sum_forward_s_36_layers": full_sum,
        "n_layers": a.n_layers,
        "scale": a.scale,
        "projected_build_seconds": projected,
        "projected_build_hours": hours,
        "budget_hours": a.budget_hours,
        "peak_gb": peak_gb,
        "peak_gib": peak_gib,
        "peak_gib_max": a.peak_gib_max,
        "refusals": refusals,
        "ok": not refusals,
    }
    write_json(a.out, data)
    print(f"feasibility: {formula}")
    print(
        f"feasibility: mean forward {mean:.1f} s, {a.n_layers}-layer sum {full_sum:.1f} s, "
        f"projected build {projected:.0f} s = {hours:.2f} h (budget {a.budget_hours:g} h); "
        f"peak GPU memory {peak_gib:.2f} GiB (limit {a.peak_gib_max:g})"
    )
    for refusal in refusals:
        print(f"feasibility REFUSED: {refusal}")
    return 3 if refusals else 0


# --- gate --------------------------------------------------------------------------------------


def exact(value: object, what: str) -> Decimal:
    """A finite non-negative number as an exact Decimal (the shortest repr of the float)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        die(f"{what} is not a number: {value!r}")
    if not math.isfinite(value) or value < 0:
        die(f"{what} is not a finite non-negative number: {value!r}")
    return Decimal(repr(value))


def threshold(text: str, what: str) -> Decimal:
    """A threshold argument as an exact Decimal."""
    try:
        value = Decimal(text)
    except InvalidOperation:
        die(f"{what} is not a number: {text!r}")
    if not value.is_finite() or value < 0:
        die(f"{what} is not a finite non-negative number: {text!r}")
    return value


def cmd_gate(a: argparse.Namespace) -> int:
    """Stage 1 gate: both numbers at or below their threshold pass; unreadable input is exit 2."""
    fid = read_json(a.fidelity)
    kl_raw = (fid.get("mean_kl") or {}).get("value")
    ppl_raw = (fid.get("ppl_ratio") or {}).get("value")
    kl, ppl = exact(kl_raw, "mean_kl.value"), exact(ppl_raw, "ppl_ratio.value")
    kl_max = threshold(a.kl_max, "--kl-max")
    ppl_max = threshold(a.ppl_ratio_max, "--ppl-ratio-max")
    kl_ok, ppl_ok = kl <= kl_max, ppl <= ppl_max
    passed = kl_ok and ppl_ok
    data = {
        "pass": passed,
        "kl": kl_raw,
        "ppl_ratio": ppl_raw,
        "kl_ok": kl_ok,
        "ppl_ratio_ok": ppl_ok,
        "thresholds": {
            "kl_max": float(kl_max),
            "ppl_ratio_max": float(ppl_max),
            "source": a.source,
        },
        "n5_reference": {
            "kl": float(a.n5_kl),
            "ppl_ratio": float(a.n5_ppl_ratio),
            "from": a.n5_from,
        },
    }
    write_json(a.out, data)
    print(
        f"gate: mean KL {kl_raw} (max {kl_max}) {'ok' if kl_ok else 'TOO HIGH'}; "
        f"perplexity ratio {ppl_raw} (max {ppl_max}) {'ok' if ppl_ok else 'TOO HIGH'}"
    )
    return 0 if passed else 1


# --- fidelity.json -----------------------------------------------------------------------------

HEADLINE = {
    "mean_kl": "kld_mean",
    "median_kl": "kld_median",
    "p99_kl": "kld_p99",
    "ppl_ratio": "ppl_ratio",
    "top1_agreement": "top1_agreement",
    "ppl_ref": "ppl_ref",
    "ppl_var": "ppl_var",
}


def sha256_file(path: Path) -> str:
    """sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def cmd_fidelity(a: argparse.Namespace) -> int:
    """results/g1/fidelity.json from the output of `fidelity.py compare --json`."""
    src = read_json(a.compare)
    out: dict = {}
    for ours, theirs in HEADLINE.items():
        entry = src.get(theirs)
        if not isinstance(entry, dict) or not all(k in entry for k in ("value", "lo", "hi")):
            die(f"{a.compare} has no {theirs} with value, lo and hi")
        out[ours] = entry
    for key in ("n_chunks", "n_positions", "k", "n_boot", "seed", "confidence"):
        if key in src:
            out[key] = src[key]
    ref = Path(a.reference)
    out["reference"] = {"path": a.reference, "sha256": sha256_file(ref) if ref.is_file() else None}
    out["source"] = Path(a.compare).name
    write_json(a.out, out)
    print(
        f"fidelity: mean KL {out['mean_kl']['value']:.5f}, "
        f"perplexity ratio {out['ppl_ratio']['value']:.5f}"
    )
    return 0


# --- checkpoint record -------------------------------------------------------------------------

META = ("model.safetensors.index.json", "config.json", "quantization_info.json")


def cmd_checkpoint(a: argparse.Namespace) -> int:
    """sha256 of the index, config and quantization_info.json, and the size in bytes."""
    root = Path(a.dir)
    missing = [n for n in META if not (root / n).is_file()]
    if missing:
        die(f"{root} lacks {', '.join(missing)}")
    files = sorted(p for p in root.iterdir() if p.is_file())
    data = {
        "path": str(root),
        "size_bytes": sum(p.stat().st_size for p in files),
        "n_files": len(files),
        "n_shards": sum(1 for p in files if p.name.endswith(".safetensors")),
        "sha256": {n: sha256_file(root / n) for n in META},
    }
    write_json(a.out, data)
    print(f"checkpoint: {data['size_bytes']} bytes in {data['n_files']} files")
    return 0


def cmd_run_status(a: argparse.Namespace) -> int:
    """0 only when run_info.json says status complete."""
    try:
        info = json.loads(Path(a.run_info).read_text())
    except (OSError, ValueError):
        return 1
    return 0 if isinstance(info, dict) and info.get("status") == "complete" else 1


# --- calibration files -------------------------------------------------------------------------

CALIB_FILES = ("manifest.json", "windows.json", "calib_text.jsonl")


def cmd_calib_check(a: argparse.Namespace) -> int:
    """The three CAL-1 files the build needs, against the `files` block of the manifest.

    manifest.json holds no hash of itself: its own sha256 is recorded (and listed) so that a copy
    on a pod can be compared with the one that was shipped.
    """
    root = Path(a.dir)
    manifest = read_json(str(root / "manifest.json"))
    files = manifest.get("files") or {}
    problems = []
    if manifest.get("schema") != "cal1/1":
        problems.append(f"schema is {manifest.get('schema')!r}, not 'cal1/1'")
    audit = manifest.get("audit") or {}
    if audit.get("ok") is not True or audit.get("n_hits", 1) != 0:
        problems.append("the leakage audit of the manifest is not a pass")
    for name in CALIB_FILES[1:]:
        want, path = files.get(name), root / name
        if not want:
            problems.append(f"the manifest files block has no {name}")
        elif not path.is_file():
            problems.append(f"{name} is missing")
        elif sha256_file(path) != want:
            problems.append(f"{name} differs from the sha256 of the manifest")
    windows = manifest.get("windows") or {}
    if windows.get("file") != "windows.json" or windows.get("sha256") != files.get("windows.json"):
        problems.append("the windows entry of the manifest does not agree with its files block")
    if problems:
        die(f"{root}: " + "; ".join(problems))
    record = {name: sha256_file(root / name) for name in CALIB_FILES}
    if a.out:
        write_json(a.out, {"path": str(root), "sha256": record})
    if a.list:
        for name, digest in record.items():
            print(f"{digest}  {name}")
    else:
        print(f"calib-check: {root} ok ({', '.join(CALIB_FILES)})")
    return 0


def parser() -> argparse.ArgumentParser:
    """The command line."""
    p = argparse.ArgumentParser(prog="g1_tools.py", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("feasibility")
    f.add_argument("log")
    f.add_argument("--out", required=True)
    f.add_argument("--scale", type=float, required=True)
    f.add_argument("--n-layers", type=int, required=True)
    f.add_argument("--budget-hours", type=float, required=True)
    f.add_argument("--peak-gib-max", type=float, required=True)
    f.set_defaults(fn=cmd_feasibility)
    g = sub.add_parser("gate")
    g.add_argument("fidelity")
    g.add_argument("--out", required=True)
    g.add_argument("--kl-max", required=True)
    g.add_argument("--ppl-ratio-max", required=True)
    g.add_argument("--n5-kl", required=True)
    g.add_argument("--n5-ppl-ratio", required=True)
    g.add_argument("--n5-from", required=True)
    g.add_argument("--source", required=True)
    g.set_defaults(fn=cmd_gate)
    i = sub.add_parser("fidelity")
    i.add_argument("compare")
    i.add_argument("--reference", required=True)
    i.add_argument("--out", required=True)
    i.set_defaults(fn=cmd_fidelity)
    c = sub.add_parser("checkpoint")
    c.add_argument("dir")
    c.add_argument("--out", required=True)
    c.set_defaults(fn=cmd_checkpoint)
    r = sub.add_parser("run-status")
    r.add_argument("run_info")
    r.set_defaults(fn=cmd_run_status)
    k = sub.add_parser("calib-check")
    k.add_argument("dir")
    k.add_argument("--out")
    k.add_argument("--list", action="store_true")
    k.set_defaults(fn=cmd_calib_check)
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
