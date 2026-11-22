"""pod/fidelity_gate.py: the stage 1 gate of PROTOCOL.md amendment 25, the choice of the smallest passing candidate, and the
summary and selection files. Pure Python, no pod, no GPU: the per-arm inputs are the JSON files the batch writes.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest
from conftest import RUNPOD

TOOL = RUNPOD / "pod" / "fidelity_gate.py"


def load():
    spec = importlib.util.spec_from_file_location("fidelity_gate", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules["fidelity_gate"] = module
    spec.loader.exec_module(module)
    return module


gate = load()


def fid(kl="0.0243", ppl="1.0197", median="0.0097", p99="0.257", top1="0.9122") -> dict:
    """A fidelity block like the one ablation_arm.sh summarises (every metric with its interval)."""

    def m(v: str) -> dict:
        d = Decimal(v)
        return {"value": d, "lo": d * Decimal("0.95"), "hi": d * Decimal("1.05")}

    return {
        "kld_mean": m(kl),
        "kld_median": m(median),
        "kld_p99": m(p99),
        "ppl_ratio": m(ppl),
        "top1_agreement": m(top1),
        "n_chunks": 64,
        "n_positions": 32704,
        "k": 20,
    }


# ------------------------------------------------------------------------------------------------ the thresholds


def test_the_thresholds_are_stated_once_with_their_source() -> None:
    assert Decimal("0.0304") == gate.MAX_MEAN_KL
    assert Decimal("1.030") == gate.MAX_PPL_RATIO
    assert isinstance(gate.MAX_MEAN_KL, Decimal) and isinstance(gate.MAX_PPL_RATIO, Decimal)
    for needle in ("amendment 25", "0.0243", "25%", "1.030"):
        assert needle in gate.THRESHOLD_SOURCE


def test_no_other_script_of_the_batch_repeats_the_numbers() -> None:
    for name in ("pod/fidelity_batch.sh", "launch_fidelity.sh"):
        path = RUNPOD / name
        if path.exists():
            text = path.read_text()
            assert "0.0304" not in text and "1.030" not in text, name


# ------------------------------------------------------------------------------------------------ the gate arithmetic


@pytest.mark.parametrize(
    ("kl", "ppl", "passes"),
    [
        ("0.0243", "1.0197", True),  # N5 itself
        ("0.0304", "1.0197", True),  # exactly at the KL limit: "at most"
        ("0.03040", "1.030", True),  # both exactly at their limits
        ("0.0304", "1.03", True),
        ("0.03040000000000001", "1.0197", False),  # one unit in the last place above
        ("0.0305", "1.0197", False),
        ("0.0303", "1.0197", True),
        ("0.0243", "1.0300000000000002", False),
        ("0.0243", "1.031", False),
        ("0.0243", "1.0299", True),
        ("0.0305", "1.031", False),
        ("0.0001", "0.99", True),  # a variant better than bf16 on this text still passes
    ],
)
def test_the_gate_is_at_most_on_both_numbers_with_exact_decimal_arithmetic(
    kl: str, ppl: str, passes: bool
) -> None:
    verdict = gate.verdict(fid(kl, ppl))
    assert verdict["passes"] is passes, verdict
    assert verdict["mean_kl"]["ok"] is (Decimal(kl) <= Decimal("0.0304"))
    assert verdict["ppl_ratio"]["ok"] is (Decimal(ppl) <= Decimal("1.030"))
    assert verdict["mean_kl"]["max"] == "0.0304" and verdict["ppl_ratio"]["max"] == "1.030"
    assert verdict["mean_kl"]["value"] == str(Decimal(kl))


def test_a_float_in_a_json_file_is_compared_through_its_shortest_repr_not_its_binary_expansion() -> (
    None
):
    """0.0304 written by json.dumps is the text 0.0304; as a binary float it is 0.03039999999999999869... : must pass."""
    text = json.dumps({"kld_mean": {"value": 0.0304}, "ppl_ratio": {"value": 1.03}})
    parsed = json.loads(text, parse_float=Decimal)
    assert gate.verdict(parsed)["passes"] is True
    assert (
        gate.verdict({"kld_mean": {"value": 0.0304}, "ppl_ratio": {"value": 1.03}})["passes"]
        is True
    )
    assert (
        gate.verdict({"kld_mean": {"value": 0.030400000000000003}, "ppl_ratio": {"value": 1.0}})[
            "passes"
        ]
        is False
    )


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {},
        {"kld_mean": {"value": 0.01}},
        {"ppl_ratio": {"value": 1.0}},
        {"kld_mean": {"value": float("nan")}, "ppl_ratio": {"value": 1.0}},
        {"kld_mean": {"value": float("inf")}, "ppl_ratio": {"value": 1.0}},
        {"kld_mean": {"value": "abc"}, "ppl_ratio": {"value": 1.0}},
        {"kld_mean": {"value": None}, "ppl_ratio": {"value": 1.0}},
    ],
)
def test_a_missing_or_non_finite_number_fails_the_gate_with_a_reason(bad) -> None:
    verdict = gate.verdict(bad)
    assert verdict["passes"] is False and verdict["reason"]


# ------------------------------------------------------------------------------------------------ the choice


def arm(name, size, kl="0.0243", ppl="1.0197", role="candidate", status="done", **extra) -> dict:
    return {
        "name": name,
        "role": role,
        "status": status,
        "spec": extra.pop("spec", "experts-all"),
        "scale_search": extra.pop("scale_search", False),
        "file_bytes": size,
        "fidelity": fid(kl, ppl) if status == "done" else None,
        **extra,
    }


def with_verdicts(arms):
    return [{**a, "verdict": gate.verdict(a["fidelity"])} for a in arms]


def test_the_smallest_passing_candidate_is_chosen() -> None:
    arms = with_verdicts(
        [
            arm("N5", 6_645_856_945, role="control"),
            arm("C1", 5_777_000_000, kl="0.0301"),
            arm("C2", 5_777_000_000, kl="0.0270"),
            arm("C3", 6_645_856_945, kl="0.0230"),
        ]
    )
    assert gate.pick(arms) == "C2"  # C1 and C2 are the same size; C2 has the lower KL; C3 is bigger


def test_it_is_the_smallest_file_not_the_lowest_kl_and_never_the_largest() -> None:
    arms = with_verdicts(
        [
            arm("C1", 5_000_000_001, kl="0.0300"),
            arm("C2", 5_000_000_000, kl="0.0303"),
            arm("C3", 9_000_000_000, kl="0.0100"),
        ]
    )
    assert gate.pick(arms) == "C2"


def test_ties_on_size_go_to_the_lowest_kl_then_to_the_first() -> None:
    arms = with_verdicts(
        [arm("C1", 100, kl="0.0290"), arm("C2", 100, kl="0.0280"), arm("C3", 100, kl="0.0280")]
    )
    assert gate.pick(arms) == "C2"
    assert gate.pick(with_verdicts([arm("A", 100), arm("B", 100)])) == "A"


def test_a_candidate_that_fails_the_gate_is_never_chosen_even_if_it_is_the_smallest() -> None:
    arms = with_verdicts(
        [
            arm("C1", 5_000_000_000, kl="0.0305"),  # one step over
            arm("C2", 5_500_000_000, ppl="1.031"),
            arm("C3", 6_600_000_000, kl="0.0304", ppl="1.030"),  # exactly at both limits: passes
        ]
    )
    assert gate.pick(arms) == "C3"


def test_the_control_is_never_chosen() -> None:
    arms = with_verdicts([arm("N5", 1, role="control"), arm("C1", 9_000, kl="0.0300")])
    assert gate.pick(arms) == "C1"
    assert gate.pick(with_verdicts([arm("N5", 1, role="control")])) is None


def test_none_is_returned_when_no_candidate_passes_and_a_failed_arm_does_not_count() -> None:
    arms = with_verdicts([arm("C1", 5, kl="0.04"), arm("C2", 6, ppl="1.2")])
    failed = {**arm("C3", 4, status="failed", reason="convert"), "verdict": gate.verdict(None)}
    assert gate.pick([*arms, failed]) is None
    assert gate.pick([]) is None


def test_a_passing_arm_without_a_file_size_is_not_eligible() -> None:
    arms = with_verdicts([arm("C1", None), arm("C2", 7)])
    assert gate.pick(arms) == "C2"


# ------------------------------------------------------------------------------------------------ the record of one arm


def make_arm_dir(
    root: Path, name: str, *, kl="0.0243", ppl="1.0197", verify="OK", check="ABLATION_CHECK_OK"
) -> Path:
    d = root / "arms" / name
    d.mkdir(parents=True)
    (d / "fidelity_vs_b0.json").write_text(
        json.dumps(
            {
                k: {"value": float(v), "lo": float(v) * 0.9, "hi": float(v) * 1.1}
                for k, v in {
                    "ppl_ref": "10.0",
                    "ppl_var": "10.2",
                    "ppl_ratio": ppl,
                    "kld_mean": kl,
                    "kld_median": "0.0097",
                    "kld_p99": "0.257",
                    "top1_agreement": "0.9122",
                }.items()
            }
            | {"n_chunks": 64, "n_positions": 32704, "k": 20}
        )
    )
    (d / "expected.json").write_text(
        json.dumps({"spec": "experts-all", "quantized_layers": 36, "quantized_tensors": 72})
    )
    (d / "verify.txt").write_text(f"{verify}: 945 tensors checked\n")
    (d / "check.txt").write_text(f"{check} experts-all\n")
    return d


def make_ckpt(path: Path, shard_bytes: tuple[int, ...] = (1000, 500), other: int = 77) -> Path:
    path.mkdir(parents=True)
    for i, n in enumerate(shard_bytes):
        (path / f"model-{i + 1:05d}-of-{len(shard_bytes):05d}.safetensors").write_bytes(b"x" * n)
    (path / "config.json").write_bytes(b"c" * other)
    (path / "model.safetensors.index.json").write_text("{}")
    return path


def test_record_reads_the_fidelity_the_sizes_and_the_verdicts_of_a_finished_arm(
    tmp_path: Path,
) -> None:
    d = make_arm_dir(tmp_path, "C2", kl="0.0287", ppl="1.0211")
    ckpt = make_ckpt(tmp_path / "ckpt-C2", (1000, 500), other=77)
    rec = gate.record_arm(
        d,
        name="C2",
        role="candidate",
        spec="experts-all",
        scale_search=True,
        status="done",
        reason="",
        ckpt=ckpt,
    )
    assert rec["name"] == "C2" and rec["role"] == "candidate" and rec["scale_search"] is True
    assert rec["spec"] == "experts-all" and rec["status"] == "done"
    assert (
        rec["file_bytes"] == 1500
    )  # the .safetensors shards: what the study calls the size of the file
    assert rec["dir_bytes"] == 1500 + 77 + len("{}")
    assert rec["fidelity"]["kld_mean"]["value"] == 0.0287
    assert rec["verify"] == "OK" and rec["check"] == "ABLATION_CHECK_OK"
    assert rec["expected"]["quantized_tensors"] == 72
    assert json.loads((d / "arm.json").read_text())["name"] == "C2"


def test_record_of_a_failed_arm_keeps_the_reason_and_no_numbers(tmp_path: Path) -> None:
    d = tmp_path / "arms" / "C1"
    d.mkdir(parents=True)
    rec = gate.record_arm(
        d,
        name="C1",
        role="candidate",
        spec="experts-all",
        scale_search=False,
        status="failed",
        reason="convert",
        ckpt=None,
    )
    assert rec["status"] == "failed" and rec["reason"] == "convert"
    assert rec["fidelity"] is None and rec["file_bytes"] is None


def test_record_refuses_a_done_arm_without_a_comparison_file(tmp_path: Path) -> None:
    d = tmp_path / "arms" / "C1"
    d.mkdir(parents=True)
    with pytest.raises(SystemExit):
        gate.record_arm(
            d,
            name="C1",
            role="candidate",
            spec="experts",
            scale_search=False,
            status="done",
            reason="",
            ckpt=None,
        )


# ------------------------------------------------------------------------------------------------ summary and selection


def build_run(tmp_path: Path, rows: dict[str, tuple[str, str, int]], control="N5", **kw) -> Path:
    """results/fidelity-LABEL with an arm.json per row (name -> (kl, ppl, bytes))."""
    root = tmp_path / "fidelity-x"
    for name, (kl, ppl, size) in rows.items():
        d = make_arm_dir(root, name, kl=kl, ppl=ppl)
        ckpt = make_ckpt(tmp_path / "ckpt" / name, (size,), other=0)
        gate.record_arm(
            d,
            name=name,
            role="control" if name == control else "candidate",
            spec="experts" if name in ("N5", "C3") else "experts-all",
            scale_search=name in ("C2", "C3"),
            status="done",
            reason="",
            ckpt=ckpt,
        )
    return root


ROWS = {
    "N5": ("0.0243", "1.0197", 6000),
    "C1": ("0.0310", "1.0250", 5000),
    "C2": ("0.0290", "1.0240", 5000),
    "C3": ("0.0230", "1.0190", 6000),
}


def test_finalize_writes_the_summary_and_the_selection(tmp_path: Path) -> None:
    root = build_run(tmp_path, ROWS)
    out = gate.finalize(
        root, label="x", order=["N5", "C1", "C2", "C3"], checkpoint="/workspace/ckpt/selected"
    )
    summary = json.loads((root / "summary.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    assert out == "C2" == summary["selected"] == selection["selected"]
    assert [a["name"] for a in summary["arms"]] == ["N5", "C1", "C2", "C3"]
    by = {a["name"]: a for a in summary["arms"]}
    assert by["C1"]["verdict"]["passes"] is False and by["C1"]["verdict"]["mean_kl"]["ok"] is False
    assert by["C2"]["verdict"]["passes"] is True and by["C3"]["verdict"]["passes"] is True
    assert (
        by["N5"]["role"] == "control" and by["N5"]["verdict"]["passes"] is True
    )  # reported, never eligible
    for a in summary["arms"]:  # the numbers of every arm, each with its interval, and the bytes
        for key in ("kld_mean", "kld_median", "kld_p99", "ppl_ratio", "top1_agreement"):
            assert set(a["fidelity"][key]) == {"value", "lo", "hi"}, (a["name"], key)
        assert isinstance(a["file_bytes"], int) and a["file_bytes"] > 0
    assert summary["thresholds"] == {
        "mean_kl_max": "0.0304",
        "ppl_ratio_max": "1.030",
        "source": gate.THRESHOLD_SOURCE,
    }
    assert summary["label"] == "x" and summary["status"] == "done"
    assert "smallest file" in summary["rule"] and "lowest mean KL" in summary["rule"]
    assert selection["checkpoint"] == "/workspace/ckpt/selected"
    assert selection["spec"] == "experts-all" and selection["scale_search"] is True
    assert selection["file_bytes"] == 5000 and selection["passing"] == ["C2", "C3"]


def test_finalize_without_a_passing_candidate_selects_none_and_names_no_checkpoint(
    tmp_path: Path,
) -> None:
    rows = {"N5": ROWS["N5"], "C1": ("0.05", "1.2", 5000)}
    root = build_run(tmp_path, rows)
    assert (
        gate.finalize(root, label="x", order=["N5", "C1"], checkpoint="/workspace/ckpt/selected")
        is None
    )
    selection = json.loads((root / "selection.json").read_text())
    assert (
        selection["selected"] == "none"
        and selection["checkpoint"] is None
        and selection["passing"] == []
    )
    assert json.loads((root / "summary.json").read_text())["selected"] == "none"


def test_finalize_lists_failed_and_skipped_arms_with_their_reason(tmp_path: Path) -> None:
    root = build_run(tmp_path, {"N5": ROWS["N5"], "C2": ROWS["C2"]})
    for name, status, reason in (("C1", "failed", "convert"), ("C3", "skipped", "deadline")):
        d = root / "arms" / name
        d.mkdir(parents=True)
        gate.record_arm(
            d,
            name=name,
            role="candidate",
            spec="experts",
            scale_search=False,
            status=status,
            reason=reason,
            ckpt=None,
        )
    assert gate.finalize(root, label="x", order=["N5", "C1", "C2", "C3"], checkpoint="/c") == "C2"
    by = {a["name"]: a for a in json.loads((root / "summary.json").read_text())["arms"]}
    assert (
        by["C1"]["status"] == "failed"
        and by["C1"]["verdict"]["passes"] is False
        and "convert" in by["C1"]["reason"]
    )
    assert by["C3"]["status"] == "skipped" and by["C3"]["verdict"]["passes"] is False


def test_the_summary_holds_numbers_as_json_numbers_with_no_checkpoint_bytes(tmp_path: Path) -> None:
    root = build_run(tmp_path, ROWS)
    gate.finalize(root, label="x", order=["N5", "C1", "C2", "C3"], checkpoint="/c")
    text = (root / "summary.json").read_text()
    summary = json.loads(text)
    assert isinstance(summary["arms"][0]["fidelity"]["kld_mean"]["value"], float)
    assert not list(
        root.rglob("*.safetensors")
    )  # the checkpoints are in tmp_path/ckpt, never under the results


# ------------------------------------------------------------------------------------------------ the command line


def run_tool(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL), *args], capture_output=True, text=True, check=False, timeout=60
    )


def test_the_command_line_record_pick_and_finalize(tmp_path: Path) -> None:
    root = tmp_path / "fidelity-x"
    for name, (kl, ppl, size) in ROWS.items():
        d = make_arm_dir(root, name, kl=kl, ppl=ppl)
        ckpt = make_ckpt(tmp_path / "ckpt" / name, (size,), other=0)
        out = run_tool(
            "record",
            "--arm-dir",
            str(d),
            "--name",
            name,
            "--role",
            "control" if name == "N5" else "candidate",
            "--spec",
            "experts",
            "--scale-search",
            "1" if name in ("C2", "C3") else "0",
            "--status",
            "done",
            "--ckpt",
            str(ckpt),
        )
        assert out.returncode == 0, out.stderr
    pick = run_tool("pick", "--results", str(root), "--order", "N5,C1,C2,C3")
    assert pick.returncode == 0 and pick.stdout.strip() == "C2"
    fin = run_tool(
        "finalize", "--results", str(root), "--label", "x", "--order", "N5,C1,C2,C3", "--selected", "C2",
        "--checkpoint", "/workspace/ckpt/selected",
    )  # fmt: skip
    assert fin.returncode == 0, fin.stderr
    assert json.loads((root / "selection.json").read_text())["selected"] == "C2"
    # a --selected that disagrees with the rule is refused and writes nothing new
    (root / "selection.json").unlink()
    bad = run_tool(
        "finalize", "--results", str(root), "--label", "x", "--order", "N5,C1,C2,C3", "--selected", "C3",
        "--checkpoint", "/c",
    )  # fmt: skip
    assert bad.returncode == 1 and not (root / "selection.json").exists()
    none = run_tool("pick", "--results", str(tmp_path / "empty"), "--order", "C1")
    assert none.returncode == 0 and none.stdout.strip() == "none"


def test_the_thresholds_subcommand_prints_them(tmp_path: Path) -> None:
    out = run_tool("thresholds")
    assert out.returncode == 0
    data = json.loads(out.stdout)
    assert (
        data["mean_kl_max"] == "0.0304"
        and data["ppl_ratio_max"] == "1.030"
        and "amendment 25" in data["source"]
    )


def test_a_bad_command_line_is_exit_2() -> None:
    assert run_tool().returncode == 2
    assert run_tool("record", "--arm-dir").returncode == 2
    assert run_tool("frobnicate").returncode == 2
