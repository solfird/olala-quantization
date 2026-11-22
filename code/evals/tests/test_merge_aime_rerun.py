"""Tests for merge_aime_rerun.py: a full run and a one-task rerun become one analysis directory."""

import hashlib
import json
from pathlib import Path

import pytest
from fake_runs import SMALL_EXPECTED, build_run, det, ref_fn, task_item_ids

import analyze_wave as aw
import merge_aime_rerun as mar

TASK = "aime2025"
AIME_ONLY = {TASK: SMALL_EXPECTED[TASK]}
RERUN_SETTINGS = {
    "sampling.timeout": 14400,
    "max_connections": 24,
    "max_connections_per_task": {TASK: 24},
    "sandbox.tasks": [],  # the rerun ran no code task
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jload(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows_of(path: Path) -> list[dict]:
    return [json.loads(line) for line in (path / "samples.jsonl").read_text().splitlines()]


def lines_of(path: Path, task: str | None = None, *, other: bool = False) -> list[str]:
    out = []
    for line in (path / "samples.jsonl").read_text().splitlines():
        is_task = json.loads(line)["task"] == task
        if task is None or is_task != other:
            out.append(line)
    return out


def make_original(tmp_path: Path, name: str = "orig", **kw) -> Path:
    kw.setdefault("scores", det(SMALL_EXPECTED, ref_fn))
    kw.setdefault("label", "B0")
    return build_run(tmp_path / name, **kw)


def make_rerun(tmp_path: Path, name: str = "rerun", **kw) -> Path:
    kw.setdefault("expected", AIME_ONLY)
    kw.setdefault("scores", det(kw["expected"], ref_fn))
    kw.setdefault("label", "B0r")
    kw.setdefault("run_seed", 3)
    overrides = {**RERUN_SETTINGS, **kw.pop("info_overrides", {})}
    return build_run(tmp_path / name, info_overrides=overrides, **kw)


def timeout_errors(*epochs_by_item: tuple[int, int]) -> set[tuple[str, str, int]]:
    ids = task_item_ids(TASK, SMALL_EXPECTED[TASK]["items"])
    return {(TASK, ids[i], e) for i, e in epochs_by_item}


def merge(tmp_path: Path, orig: Path, rerun: Path, *extra: str, out: str = "merged") -> int:
    argv = ["--original", str(orig), "--rerun", str(rerun), "--out", str(tmp_path / out), *extra]
    return mar.main(argv)


def refused(tmp_path: Path, capsys, orig: Path, rerun: Path, *extra: str, text: str) -> None:
    before = sorted(p.name for p in tmp_path.iterdir())
    assert merge(tmp_path, orig, rerun, *extra) != 0
    assert text in capsys.readouterr().err
    assert sorted(p.name for p in tmp_path.iterdir()) == before  # nothing written


# ------------------------------------------------------------------------------ what is merged


def test_other_tasks_are_copied_unchanged_and_the_task_comes_from_the_rerun(tmp_path):
    orig = make_original(tmp_path, error_rows=timeout_errors((0, 1), (2, 3)))
    rerun = make_rerun(tmp_path)
    assert merge(tmp_path, orig, rerun) == 0
    merged = tmp_path / "merged"
    assert lines_of(merged, TASK, other=True) == lines_of(orig, TASK, other=True)  # byte for byte
    assert lines_of(merged, TASK) == lines_of(rerun, TASK)
    assert len(rows_of(merged)) == 86  # 20 + 10 + 12 + 14 + 6 * 5


def test_the_original_rows_of_the_task_are_not_in_the_merged_file(tmp_path):
    orig = make_original(tmp_path)
    rerun = make_rerun(tmp_path)
    assert merge(tmp_path, orig, rerun) == 0
    completions = {r["completion"] for r in rows_of(tmp_path / "merged") if r["task"] == TASK}
    assert completions == {"B0r-3"}  # the rerun's, none of the original's "B0-0"


def test_items_json_takes_the_task_from_the_rerun_and_the_rest_from_the_original(tmp_path):
    orig = make_original(tmp_path)
    ids = task_item_ids(TASK, SMALL_EXPECTED[TASK]["items"])[::-1]  # same set, other order
    rerun = make_rerun(tmp_path, items_override={TASK: ids})
    assert merge(tmp_path, orig, rerun) == 0
    items = jload(tmp_path / "merged" / "items.json")
    expected = jload(orig / "items.json")
    expected[TASK] = ids
    assert items == expected


def test_run_info_is_copied_from_the_original_with_counts_recomputed_from_the_merged_rows(tmp_path):
    orig = make_original(
        tmp_path,
        error_rows=timeout_errors((0, 1), (2, 3), (4, 5)) | {("mgsm", "mgsm-en-00", 1)},
        truncated_rows=timeout_errors((1, 1)) | {("ifeval", "ifeval-000", 1)},
    )
    rerun = make_rerun(tmp_path, truncated_rows=timeout_errors((5, 2), (5, 3)))
    assert merge(tmp_path, orig, rerun) == 0
    info = jload(tmp_path / "merged" / "run_info.json")
    old = jload(orig / "run_info.json")
    assert info["n_samples"] == 86
    assert info["request_errors"] == 1  # the mgsm one; the three AIME timeouts are gone
    assert info["truncations"] == 3  # ifeval 1, rerun AIME 2
    assert info["status"] == "complete"
    for key in ("endpoint", "model", "git", "code_sha256", "versions", "reasoning_effort", "seed"):
        assert info[key] == old[key]
    assert info["sampling"] == old["sampling"]
    ids = task_item_ids(TASK, SMALL_EXPECTED[TASK]["items"])
    assert info["tasks"][TASK]["items_sha256"] == aw.sha256_of_ids(ids)
    assert info["tasks"][TASK]["n_samples"] == 30
    assert info["tasks"][TASK]["request_errors"] == 0
    assert info["tasks"][TASK]["truncations"] == 2
    for other in ("mgsm", "humaneval_plus", "mbpp_plus", "ifeval"):
        assert info["tasks"][other] == old["tasks"][other]


def test_the_counts_and_hash_of_the_replaced_task_are_recomputed_from_its_merged_rows(tmp_path):
    orig = make_original(tmp_path)
    stale = {
        TASK: {"n_samples": 99, "request_errors": 9, "truncations": 9, "items_sha256": "0" * 64}
    }
    rerun = make_rerun(tmp_path, task_info_overrides=stale, truncated_rows=timeout_errors((5, 2)))
    assert merge(tmp_path, orig, rerun) == 0
    task_info = jload(tmp_path / "merged" / "run_info.json")["tasks"][TASK]
    ids = task_item_ids(TASK, SMALL_EXPECTED[TASK]["items"])
    assert task_info["n_samples"] == 30
    assert task_info["request_errors"] == 0
    assert task_info["truncations"] == 1
    assert task_info["n_items"] == 6
    assert task_info["items_sha256"] == aw.sha256_of_ids(ids)
    assert task_info["epochs"] == 5
    assert task_info["eval_status"] == "success"


def test_the_merged_block_records_provenance_and_the_counts_of_the_replaced_rows(tmp_path):
    orig = make_original(
        tmp_path,
        error_rows=timeout_errors((0, 1), (2, 3), (4, 5)),
        truncated_rows=timeout_errors((1, 1), (3, 2)),
    )
    rerun = make_rerun(tmp_path, truncated_rows=timeout_errors((5, 2)))
    assert merge(tmp_path, orig, rerun) == 0
    block = jload(tmp_path / "merged" / "run_info.json")["merged"]
    assert block["reason"] == (
        "PROTOCOL amendment 10/11: AIME client-timeout artifact, rerun with non-binding settings"
    )
    assert block["task"] == TASK
    assert block["original_dir"] == str(orig.resolve())
    assert block["rerun_dir"] == str(rerun.resolve())
    assert block["sha256"] == {
        "original_samples": sha(orig / "samples.jsonl"),
        "original_run_info": sha(orig / "run_info.json"),
        "rerun_samples": sha(rerun / "samples.jsonl"),
        "rerun_run_info": sha(rerun / "run_info.json"),
    }
    assert block["task_rows"] == {"original": 30, "rerun": 30}
    assert block["original_task_request_errors"] == 3
    assert block["original_task_truncations"] == 2
    assert block["rerun_task_request_errors"] == 0
    assert block["rerun_task_truncations"] == 1


def test_the_settings_that_differ_are_recorded_with_both_values(tmp_path):
    orig = make_original(tmp_path)  # legacy run: global max_connections 96, timeout 3600
    rerun = make_rerun(tmp_path)
    assert merge(tmp_path, orig, rerun) == 0
    block = jload(tmp_path / "merged" / "run_info.json")["merged"]
    assert block["settings_differing"] == {
        "max_connections_per_task": {"original": 96, "rerun": 24},
        "timeout": {"original": 3600, "rerun": 14400},
    }


def test_settings_that_are_equal_are_not_listed_as_differing(tmp_path):
    same = {"max_connections_per_task": {TASK: 24, "mgsm": 32}, "sampling.timeout": 14400}
    orig = make_original(tmp_path, info_overrides=same)
    rerun = make_rerun(tmp_path)
    assert merge(tmp_path, orig, rerun) == 0
    assert jload(tmp_path / "merged" / "run_info.json")["merged"]["settings_differing"] == {}


def test_the_original_directory_is_left_untouched(tmp_path):
    orig = make_original(tmp_path)
    rerun = make_rerun(tmp_path)
    names = ("samples.jsonl", "items.json", "run_info.json")
    before = {n: sha(orig / n) for n in names} | {n: sha(rerun / n) for n in names}
    assert merge(tmp_path, orig, rerun) == 0
    after = {n: sha(orig / n) for n in names} | {n: sha(rerun / n) for n in names}
    assert before == after
    assert sorted(p.name for p in (tmp_path / "merged").iterdir()) == [
        "items.json",
        "run_info.json",
        "samples.jsonl",
    ]


def test_another_task_can_be_replaced_with_task(tmp_path):
    orig = make_original(tmp_path)
    rerun = make_rerun(tmp_path, expected={"mgsm": SMALL_EXPECTED["mgsm"]})
    assert merge(tmp_path, orig, rerun, "--task", "mgsm") == 0
    merged = tmp_path / "merged"
    assert lines_of(merged, "mgsm") == lines_of(rerun, "mgsm")
    assert lines_of(merged, "mgsm", other=True) == lines_of(orig, "mgsm", other=True)


def test_the_summary_is_printed_and_the_exit_code_is_zero(tmp_path, capsys):
    orig = make_original(tmp_path, error_rows=timeout_errors((0, 1)))
    rerun = make_rerun(tmp_path)
    assert merge(tmp_path, orig, rerun) == 0
    out = capsys.readouterr().out
    assert "merged" in out
    assert TASK in out


# ------------------------------------------------------------------------------ analyze_wave


def test_the_merged_directory_passes_the_gate_where_the_original_fails_it(tmp_path):
    orig = make_original(tmp_path, error_rows=timeout_errors((0, 1), (2, 3), (4, 5)))
    rerun = make_rerun(tmp_path)
    ref = make_original(tmp_path, "ref", label="REF", run_seed=1)
    assert merge(tmp_path, orig, rerun) == 0
    expected = SMALL_EXPECTED
    bad = aw.gate_pair([aw.load_run_dir(ref)], [aw.load_run_dir(orig)], expected)
    assert not bad.ok  # 3 request errors in 86 samples: above 0.5%
    gate = aw.gate_pair([aw.load_run_dir(ref)], [aw.load_run_dir(tmp_path / "merged")], expected)
    assert gate.problems == []
    assert gate.ok


def test_a_rerun_with_request_errors_still_fails_the_gate_of_the_merged_run(tmp_path):
    orig = make_original(tmp_path)
    rerun = make_rerun(tmp_path, error_rows=timeout_errors((0, 1)))
    ref = make_original(tmp_path, "ref", label="REF", run_seed=1)
    assert merge(tmp_path, orig, rerun) == 0  # the merge does not hide errors of the rerun
    gate = aw.gate_pair(
        [aw.load_run_dir(ref)], [aw.load_run_dir(tmp_path / "merged")], SMALL_EXPECTED
    )
    assert not gate.ok
    assert any("request errors" in p for p in gate.problems)


def test_the_quantization_info_of_the_original_is_copied(tmp_path):
    orig = make_original(tmp_path, quant_recipe="fp8-pc")
    rerun = make_rerun(tmp_path, quant_recipe="fp8-pc")
    assert merge(tmp_path, orig, rerun) == 0
    merged = aw.load_run_dir(tmp_path / "merged")
    assert merged.quantization == aw.load_run_dir(orig).quantization
    assert (tmp_path / "merged" / "quantization_info.json").read_bytes() == (
        orig / "quantization_info.json"
    ).read_bytes()


def test_no_quantization_info_is_written_when_the_original_has_none(tmp_path):
    orig = make_original(tmp_path)
    rerun = make_rerun(tmp_path, quant_recipe="fp8-pc")
    assert merge(tmp_path, orig, rerun) == 0
    assert not (tmp_path / "merged" / "quantization_info.json").exists()


def test_a_quantization_info_that_differs_between_the_two_runs_is_a_warning(tmp_path, capsys):
    orig = make_original(tmp_path, quant_recipe="fp8-pc")
    rerun = make_rerun(tmp_path, quant_recipe="nvfp4")
    assert merge(tmp_path, orig, rerun) == 0
    assert "WARNING" in capsys.readouterr().err
    sums = jload(tmp_path / "merged" / "run_info.json")["merged"]["quantization_info_sha256"]
    assert sums == {
        "original": sha(orig / "quantization_info.json"),
        "rerun": sha(rerun / "quantization_info.json"),
    }


# ------------------------------------------------------------------------------ refusals


def test_a_rerun_that_is_not_complete_is_refused(tmp_path, capsys):
    orig = make_original(tmp_path)
    rerun = make_rerun(tmp_path, status="incomplete")
    refused(tmp_path, capsys, orig, rerun, text="not complete")


def test_a_rerun_with_a_missing_request_is_refused(tmp_path, capsys):
    orig = make_original(tmp_path)
    rerun = make_rerun(tmp_path, drop_rows=timeout_errors((0, 1)))
    refused(tmp_path, capsys, orig, rerun, text="rerun")


def test_a_rerun_with_a_duplicated_row_is_refused(tmp_path, capsys):
    orig = make_original(tmp_path)
    rerun = make_rerun(tmp_path, duplicate_rows=timeout_errors((0, 1)))
    refused(tmp_path, capsys, orig, rerun, text="rerun")


def test_a_rerun_on_other_item_ids_is_refused(tmp_path, capsys):
    orig = make_original(tmp_path)
    other_ids = [f"{TASK}-x{i}" for i in range(6)]
    rerun = make_rerun(tmp_path, items_override={TASK: other_ids})
    refused(tmp_path, capsys, orig, rerun, text="item ids")


@pytest.mark.parametrize(
    "override",
    [
        {"sampling.temperature": 0.7},
        {"sampling.top_p": 0.8},
        {"sampling.top_k": 20},
        {"sampling.max_tokens": 1000},
        {"sampling.seed": 5},
        {"reasoning_effort": "high"},
        {"chat_template_kwargs": {"reasoning_effort": "high"}},
        {"sandbox.mode": "docker"},
    ],
)
def test_a_rerun_with_other_sampling_effort_or_sandbox_settings_is_refused(
    tmp_path, capsys, override
):
    orig = make_original(tmp_path)
    rerun = make_rerun(tmp_path, info_overrides=override)
    refused(tmp_path, capsys, orig, rerun, text=next(iter(override)))


def test_a_rerun_that_sandboxes_the_task_differently_is_refused(tmp_path, capsys):
    orig = make_original(tmp_path)
    expected = {"humaneval_plus": SMALL_EXPECTED["humaneval_plus"]}
    rerun = make_rerun(tmp_path, expected=expected, info_overrides={"sandbox.tasks": []})
    refused(tmp_path, capsys, orig, rerun, "--task", "humaneval_plus", text="sandbox")


def test_a_rerun_with_other_epochs_for_the_task_is_refused(tmp_path, capsys):
    orig = make_original(tmp_path)
    rerun = make_rerun(tmp_path, expected={TASK: {"items": 6, "epochs": 3}})
    refused(tmp_path, capsys, orig, rerun, text="epochs")


def test_an_existing_out_directory_is_refused_and_left_alone(tmp_path, capsys):
    orig = make_original(tmp_path)
    rerun = make_rerun(tmp_path)
    (tmp_path / "merged").mkdir()
    (tmp_path / "merged" / "keep.txt").write_text("mine")
    assert merge(tmp_path, orig, rerun) != 0
    assert "exists" in capsys.readouterr().err
    assert [p.name for p in (tmp_path / "merged").iterdir()] == ["keep.txt"]


def test_a_task_absent_from_the_rerun_is_refused(tmp_path, capsys):
    orig = make_original(tmp_path)
    rerun = make_rerun(tmp_path, expected={"mgsm": SMALL_EXPECTED["mgsm"]})
    refused(tmp_path, capsys, orig, rerun, text=f"{TASK} is absent from the rerun")


def test_a_task_absent_from_the_original_is_refused(tmp_path, capsys):
    short = {k: v for k, v in SMALL_EXPECTED.items() if k != TASK}
    orig = make_original(tmp_path, expected=short, scores=det(short, ref_fn))
    rerun = make_rerun(tmp_path)
    refused(tmp_path, capsys, orig, rerun, text=f"{TASK} is absent from the original")


def test_an_unreadable_directory_is_refused(tmp_path, capsys):
    orig = make_original(tmp_path, skip_files=("run_info.json",))
    rerun = make_rerun(tmp_path)
    refused(tmp_path, capsys, orig, rerun, text="run_info.json")


def test_the_same_directory_given_twice_is_refused(tmp_path, capsys):
    orig = make_original(tmp_path)
    refused(tmp_path, capsys, orig, orig, text="same")


# ------------------------------------------------------------------------------ incomplete original


def test_an_incomplete_original_is_refused_by_default(tmp_path, capsys):
    orig = make_original(tmp_path, status="incomplete")
    rerun = make_rerun(tmp_path)
    refused(tmp_path, capsys, orig, rerun, text="--allow-incomplete-original")


def test_an_incomplete_original_is_accepted_when_every_other_task_is_full(tmp_path):
    # the original died during AIME: some of its AIME rows are missing, the other tasks are full
    orig = make_original(
        tmp_path, status="incomplete", drop_rows=timeout_errors((3, 1), (3, 2), (4, 1))
    )
    rerun = make_rerun(tmp_path)
    assert merge(tmp_path, orig, rerun, "--allow-incomplete-original") == 0
    info = jload(tmp_path / "merged" / "run_info.json")
    assert info["status"] == "complete"
    assert info["merged"]["task_rows"] == {"original": 27, "rerun": 30}
    ref = make_original(tmp_path, "ref", label="REF", run_seed=1)
    gate = aw.gate_pair(
        [aw.load_run_dir(ref)], [aw.load_run_dir(tmp_path / "merged")], SMALL_EXPECTED
    )
    assert gate.ok


def test_an_incomplete_original_with_a_short_other_task_is_refused_even_with_the_flag(
    tmp_path, capsys
):
    orig = make_original(tmp_path, status="incomplete", drop_rows={("mgsm", "mgsm-en-00", 1)})
    rerun = make_rerun(tmp_path)
    refused(
        tmp_path,
        capsys,
        orig,
        rerun,
        "--allow-incomplete-original",
        text="mgsm: 19 rows, expected 20",
    )


def test_an_incomplete_original_whose_other_task_has_no_rows_is_refused_even_with_the_flag(
    tmp_path, capsys
):
    drop = {("ifeval", i, 1) for i in task_item_ids("ifeval", 14)}
    orig = make_original(tmp_path, status="incomplete", drop_rows=drop)
    rerun = make_rerun(tmp_path)
    refused(tmp_path, capsys, orig, rerun, "--allow-incomplete-original", text="ifeval: 0 rows")


def test_the_flag_changes_nothing_for_a_complete_original(tmp_path):
    orig = make_original(tmp_path)
    rerun = make_rerun(tmp_path)
    assert merge(tmp_path, orig, rerun, "--allow-incomplete-original") == 0
    assert jload(tmp_path / "merged" / "run_info.json")["status"] == "complete"


# ------------------------------------------------------------------------------ retries, timeouts


def aime_key(i: int, epoch: int = 1) -> tuple[str, str, int]:
    return (TASK, task_item_ids(TASK, SMALL_EXPECTED[TASK]["items"])[i], epoch)


def test_retry_counters_are_recomputed_from_the_merged_rows_not_copied_from_the_original(tmp_path):
    orig = make_original(
        tmp_path,
        attempts={("mgsm", "mgsm-en-00", 1): 2, aime_key(0): 3, aime_key(1): 2},  # 2 AIME: dropped
        working_times={("mgsm", "mgsm-fr-00", 1): 100.0, aime_key(0): 500.0},
    )
    rerun = make_rerun(
        tmp_path, attempts={aime_key(2): 2, aime_key(3): 4}, working_times={aime_key(4): 900.0}
    )
    assert merge(tmp_path, orig, rerun) == 0
    info = jload(tmp_path / "merged" / "run_info.json")
    assert info["samples_with_retries"] == 3  # one MGSM row of the original, two rows of the rerun
    assert info["max_working_time"] == 900.0  # the original's 500 s AIME row is not in the merge
    assert info["tasks"][TASK]["samples_with_retries"] == 2
    assert info["tasks"][TASK]["max_working_time"] == 900.0
    assert info["tasks"]["mgsm"]["samples_with_retries"] == 1  # unchanged task: its own entry
    assert info["tasks"]["mgsm"]["max_working_time"] == 100.0
    assert info["merged"]["original_task_samples_with_retries"] == 2
    assert info["merged"]["rerun_task_samples_with_retries"] == 2


def test_the_client_timeout_of_the_replaced_task_is_the_reruns_and_the_total_is_the_smallest(
    tmp_path,
):
    orig = make_original(tmp_path)  # sampling.timeout 3600 and client_timeout 3600
    rerun = make_rerun(tmp_path)  # 14400 for both
    assert merge(tmp_path, orig, rerun) == 0
    info = jload(tmp_path / "merged" / "run_info.json")
    assert info["tasks"][TASK]["client_timeout"] == 14400
    assert info["tasks"]["mgsm"]["client_timeout"] == 3600
    assert info["client_timeout"] == 3600
    assert info["merged"]["client_timeout"] == {"original": 3600, "rerun": 14400}


def test_a_legacy_original_keeps_its_legacy_rows_and_gets_no_invented_counters(tmp_path):
    orig = make_original(tmp_path, legacy=True)
    rerun = make_rerun(tmp_path, attempts={aime_key(2): 2})
    assert merge(tmp_path, orig, rerun) == 0
    merged = tmp_path / "merged"
    assert lines_of(merged, TASK, other=True) == lines_of(orig, TASK, other=True)  # byte for byte
    assert all("model_attempts" not in r for r in rows_of(merged) if r["task"] != TASK)
    info = jload(merged / "run_info.json")
    assert "client_timeout" not in info  # four tasks have none: no run-wide value to state
    assert "client_timeout" not in info["tasks"]["mgsm"]
    assert "samples_with_retries" not in info["tasks"]["mgsm"]
    assert info["tasks"][TASK]["client_timeout"] == 14400
    assert info["tasks"][TASK]["samples_with_retries"] == 1
    assert info["samples_with_retries"] == 1  # counted over the rows that say; the rest is unknown
    assert info["merged"]["client_timeout"] == {"original": None, "rerun": 14400}
    assert info["merged"]["original_task_samples_with_retries"] is None


def test_two_legacy_runs_merge_without_any_new_field(tmp_path):
    orig = make_original(tmp_path, legacy=True)
    rerun = make_rerun(tmp_path, legacy=True)
    assert merge(tmp_path, orig, rerun) == 0
    info = jload(tmp_path / "merged" / "run_info.json")
    for key in ("client_timeout", "samples_with_retries", "max_working_time"):
        assert key not in info
        assert all(key not in t for t in info["tasks"].values())
    assert info["merged"]["rerun_task_samples_with_retries"] is None


def test_a_legacy_rerun_does_not_inherit_the_originals_counters_for_the_replaced_task(tmp_path):
    orig = make_original(tmp_path, attempts={aime_key(0): 3, ("mgsm", "mgsm-en-00", 1): 2})
    rerun = make_rerun(tmp_path, legacy=True)
    assert merge(tmp_path, orig, rerun) == 0
    info = jload(tmp_path / "merged" / "run_info.json")
    assert info["samples_with_retries"] == 1  # the MGSM row; the dropped AIME row is not counted
    assert "samples_with_retries" not in info["tasks"][TASK]
    assert "client_timeout" not in info["tasks"][TASK]
    assert "client_timeout" not in info


def test_the_gate_reads_the_retries_of_the_merged_rows(tmp_path):
    ref = make_original(tmp_path, "ref", label="REF", run_seed=1)
    orig = make_original(tmp_path, attempts={aime_key(0): 3})  # dropped by the merge
    clean = make_rerun(tmp_path)
    retried = make_rerun(tmp_path, "retried", attempts={aime_key(2): 2})
    assert merge(tmp_path, orig, clean, out="m1") == 0
    assert merge(tmp_path, orig, retried, out="m2") == 0
    for out, ok in (("m1", True), ("m2", False)):
        gate = aw.gate_pair(
            [aw.load_run_dir(ref)], [aw.load_run_dir(tmp_path / out)], SMALL_EXPECTED
        )
        assert gate.ok is ok
        assert gate.warnings == []
    assert any(
        "aime2025" in p and "1 of 30" in p for p in gate.problems
    )  # the one retried rerun row
