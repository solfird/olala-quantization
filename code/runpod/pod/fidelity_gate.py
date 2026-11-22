"""The stage 1 gate of PROTOCOL.md amendment 25, the choice of a candidate, and the result files.

Used by ``pod/fidelity_batch.sh`` (pod side, standard library only) and read by its tests. The
gate's numbers live HERE and nowhere else in the scripts: :data:`MAX_MEAN_KL`,
:data:`MAX_PPL_RATIO`, :data:`THRESHOLD_SOURCE`.

Commands (``python3 fidelity_gate.py COMMAND ...``)::

    thresholds
        print the thresholds and their source as JSON
    record --arm-dir D --name N --role control|candidate --spec S --scale-search 0|1
           --status done|failed|skipped [--reason TEXT] [--ckpt DIR]
        write D/arm.json from the arm's files (fidelity_vs_b0.json, expected.json, verify.txt,
        check.txt) and the size of the checkpoint directory DIR
    pick --results ROOT --order A,B,C
        print the name of the smallest passing candidate, or ``none``
    finalize --results ROOT --label L --order A,B,C [--selected NAME|none] [--checkpoint PATH]
        write ROOT/summary.json and ROOT/selection.json; with --selected, refuse (exit 1, nothing
        written) when the rule picks another candidate

The rule: a candidate PASSES if its mean KL is at most 0.0304 and its perplexity ratio at most
1.030 (both against bf16, on the text set of amendments 19 and 22), compared as exact decimals.
Among the candidates that pass, the one with the SMALLEST file is selected; ties go to the lowest
mean KL, then to the first in the order given. The control (N5) is measured, gated and reported but
is never eligible. A candidate that failed, was skipped or has no file size is not eligible.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

MAX_MEAN_KL = Decimal("0.0304")
MAX_PPL_RATIO = Decimal("1.030")
THRESHOLD_SOURCE = (
    "PROTOCOL.md amendment 25, stage 1: a candidate passes if its mean KL against bf16 is at most "
    "0.0304 (N5's 0.0243 plus 25%) and its perplexity ratio at most 1.030, on the wikitext chunk "
    "set of amendments 19 and 22"
)
RULE = (
    "among the candidates that pass the gate, the smallest file; ties: lowest mean KL, then the "
    "first in the order run; the control is never selected; a failed or skipped candidate is not "
    "eligible"
)
FIDELITY_KEYS = ("kld_mean", "kld_median", "kld_p99", "ppl_ratio", "top1_agreement")


def _decimal(x: Any) -> Decimal | None:
    """An exact Decimal for a JSON number (a float goes through its shortest repr), or None."""
    if isinstance(x, bool) or x is None:
        return None
    try:
        if isinstance(x, Decimal):
            d = x
        elif isinstance(x, float):
            d = Decimal(repr(x))
        else:
            d = Decimal(str(x))
    except (InvalidOperation, ValueError):
        return None
    return d if d.is_finite() else None


def _value(fid: dict[str, Any] | None, key: str) -> Decimal | None:
    entry = (fid or {}).get(key)
    return _decimal(entry.get("value")) if isinstance(entry, dict) else None


def verdict(fid: dict[str, Any] | None) -> dict[str, Any]:
    """The gate verdict of a fidelity block (``kld_mean`` and ``ppl_ratio`` values)."""
    kl, ppl = _value(fid, "kld_mean"), _value(fid, "ppl_ratio")
    out: dict[str, Any] = {
        "passes": False,
        "mean_kl": {"value": None if kl is None else str(kl), "max": str(MAX_MEAN_KL), "ok": False},
        "ppl_ratio": {
            "value": None if ppl is None else str(ppl),
            "max": str(MAX_PPL_RATIO),
            "ok": False,
        },
        "reason": "",
    }
    missing = [n for n, v in (("kld_mean", kl), ("ppl_ratio", ppl)) if v is None]
    if missing:
        out["reason"] = "no usable number for " + " and ".join(missing)
        return out
    out["mean_kl"]["ok"] = kl <= MAX_MEAN_KL
    out["ppl_ratio"]["ok"] = ppl <= MAX_PPL_RATIO
    out["passes"] = bool(out["mean_kl"]["ok"] and out["ppl_ratio"]["ok"])
    if not out["passes"]:
        why = [
            f"{n} {v} above {m}"
            for n, v, m, ok in (
                ("mean KL", kl, MAX_MEAN_KL, out["mean_kl"]["ok"]),
                ("perplexity ratio", ppl, MAX_PPL_RATIO, out["ppl_ratio"]["ok"]),
            )
            if not ok
        ]
        out["reason"] = "; ".join(why)
    return out


def pick(arms: list[dict[str, Any]]) -> str | None:
    """The name of the smallest passing candidate (ties: lowest mean KL, then first), or None."""
    eligible = []
    for index, arm in enumerate(arms):
        v = arm.get("verdict") or verdict(arm.get("fidelity"))
        size = arm.get("file_bytes")
        if arm.get("role") != "candidate" or arm.get("status") != "done" or not v["passes"]:
            continue
        if not isinstance(size, int) or size <= 0:
            continue
        kl = _value(arm["fidelity"], "kld_mean")
        if kl is not None:  # a passing verdict implies it; kept explicit rather than asserted
            eligible.append((size, kl, index, arm["name"]))
    return min(eligible)[3] if eligible else None


def _first_word(path: Path) -> str | None:
    try:
        return path.read_text().split()[0].rstrip(":")
    except (OSError, IndexError):
        return None


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def checkpoint_sizes(ckpt: Path | None) -> tuple[int | None, int | None]:
    """``(bytes of the *.safetensors shards, bytes of every file)`` of a checkpoint directory."""
    if ckpt is None or not ckpt.is_dir():
        return None, None
    shards = sum(f.stat().st_size for f in ckpt.glob("*.safetensors"))
    everything = sum(f.stat().st_size for f in ckpt.rglob("*") if f.is_file())
    return shards, everything


def record_arm(
    arm_dir: Path,
    *,
    name: str,
    role: str,
    spec: str,
    scale_search: bool,
    status: str,
    reason: str,
    ckpt: Path | None,
) -> dict[str, Any]:
    """Write ``arm_dir/arm.json`` from the arm's files and the size of the checkpoint ``ckpt``."""
    fidelity = _load(arm_dir / "fidelity_vs_b0.json")
    if status == "done" and not isinstance(fidelity, dict):
        raise SystemExit(
            f"record: arm {name} is done but {arm_dir}/fidelity_vs_b0.json is missing or unreadable"
        )
    if isinstance(fidelity, dict):
        fidelity = {
            k: fidelity[k]
            for k in (*FIDELITY_KEYS, "ppl_ref", "ppl_var", "n_chunks", "n_positions", "k")
            if k in fidelity
        }
    expected = _load(arm_dir / "expected.json")
    file_bytes, dir_bytes = checkpoint_sizes(ckpt)
    record = {
        "name": name,
        "role": role,
        "spec": spec,
        "spec_canonical": (expected or {}).get("spec"),
        "scale_search": scale_search,
        "status": status,
        "reason": reason,
        "expected": expected,
        "verify": _first_word(arm_dir / "verify.txt"),
        "check": _first_word(arm_dir / "check.txt"),
        "file_bytes": file_bytes,
        "dir_bytes": dir_bytes,
        "fidelity": fidelity if isinstance(fidelity, dict) else None,
    }
    arm_dir.mkdir(parents=True, exist_ok=True)
    (arm_dir / "arm.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def _read_arms(root: Path, order: list[str]) -> list[dict[str, Any]]:
    arms = []
    for name in order:
        record = _load(root / "arms" / name / "arm.json")
        if not isinstance(record, dict):  # the batch never got to write it
            record = {
                "name": name,
                "role": "candidate",
                "spec": None,
                "scale_search": None,
                "status": "skipped",
                "reason": "no record was written for this arm",
                "fidelity": None,
                "file_bytes": None,
            }
        if record.get("status") == "done":
            record["verdict"] = verdict(record.get("fidelity"))
        else:  # failed or skipped: no number was measured, whatever the files say
            record["verdict"] = verdict(None)
            record["verdict"]["reason"] = (
                f"{record.get('status')}: {record.get('reason') or 'no reason recorded'}"
            )
        arms.append(record)
    return arms


def finalize(root: Path, *, label: str, order: list[str], checkpoint: str | None) -> str | None:
    """Write ``summary.json`` and ``selection.json`` under ``root``; return the chosen name."""
    arms = _read_arms(root, order)
    chosen = pick(arms)
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    summary = {
        "label": label,
        "status": "done",
        "finished": now,
        "thresholds": {
            "mean_kl_max": str(MAX_MEAN_KL),
            "ppl_ratio_max": str(MAX_PPL_RATIO),
            "source": THRESHOLD_SOURCE,
        },
        "rule": RULE,
        "selected": chosen or "none",
        "arms": arms,
    }
    by = {a["name"]: a for a in arms}
    selection: dict[str, Any] = {
        "selected": chosen or "none",
        "checkpoint": checkpoint if chosen else None,
        "spec": by[chosen]["spec"] if chosen else None,
        "scale_search": by[chosen]["scale_search"] if chosen else None,
        "file_bytes": by[chosen]["file_bytes"] if chosen else None,
        "mean_kl": by[chosen]["verdict"]["mean_kl"]["value"] if chosen else None,
        "ppl_ratio": by[chosen]["verdict"]["ppl_ratio"]["value"] if chosen else None,
        "passing": [a["name"] for a in arms if a["role"] == "candidate" and a["verdict"]["passes"]],
        "rule": RULE,
        "thresholds_source": THRESHOLD_SOURCE,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (root / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    return chosen


def _names(text: str) -> list[str]:
    return [n for n in text.split(",") if n]


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: 0 ok, 1 a refusal (finalize), 2 bad arguments."""
    p = argparse.ArgumentParser(prog="fidelity_gate.py", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("thresholds")
    rec = sub.add_parser("record")
    rec.add_argument("--arm-dir", type=Path, required=True)
    rec.add_argument("--name", required=True)
    rec.add_argument("--role", choices=("control", "candidate"), required=True)
    rec.add_argument("--spec", required=True)
    rec.add_argument("--scale-search", choices=("0", "1"), required=True)
    rec.add_argument("--status", choices=("done", "failed", "skipped"), required=True)
    rec.add_argument("--reason", default="")
    rec.add_argument("--ckpt", type=Path)
    pk = sub.add_parser("pick")
    pk.add_argument("--results", type=Path, required=True)
    pk.add_argument("--order", required=True)
    fin = sub.add_parser("finalize")
    fin.add_argument("--results", type=Path, required=True)
    fin.add_argument("--label", required=True)
    fin.add_argument("--order", required=True)
    fin.add_argument("--selected")
    fin.add_argument("--checkpoint")
    try:
        args = p.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    if args.command == "thresholds":
        print(
            json.dumps(
                {
                    "mean_kl_max": str(MAX_MEAN_KL),
                    "ppl_ratio_max": str(MAX_PPL_RATIO),
                    "source": THRESHOLD_SOURCE,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "record":
        try:
            record_arm(
                args.arm_dir,
                name=args.name,
                role=args.role,
                spec=args.spec,
                scale_search=args.scale_search == "1",
                status=args.status,
                reason=args.reason,
                ckpt=args.ckpt,
            )
        except SystemExit as e:
            print(e.code, file=sys.stderr)
            return 1
        return 0
    order = _names(args.order)
    arms = _read_arms(args.results, order)
    chosen = pick(arms)
    if args.command == "pick":
        print(chosen or "none")
        return 0
    if args.selected is not None and args.selected != (chosen or "none"):
        print(
            f"finalize: the rule selects {chosen or 'none'!r}, not {args.selected!r}",
            file=sys.stderr,
        )
        return 1
    finalize(args.results, label=args.label, order=order, checkpoint=args.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
