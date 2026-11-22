"""Tests for combine_task_runs.py: per-task run directories of one arm become one directory."""

import hashlib
import json
from pathlib import Path

import pytest
from fake_runs import SMALL_EXPECTED, build_run, det, ref_fn

import analyze_wave as aw
import combine_task_runs as ctr

GROUPS = {
    "A": ("mgsm", "humaneval_plus"),
    "B": ("ifeval", "mbpp_plus"),
    "C": ("aime2025",),
}
POD_SETTINGS = {
    "A": {
        "sandbox.tasks": ["humaneval_plus"],
        "max_connections_per_task": {"mgsm": 32, "humaneval_plus": 16},
        "endpoint": "http://pod-a:8000/v1",
    },
    "B": {
        "sandbox.tasks": ["mbpp_plus"],
        "max_connections_per_task": {"ifeval": 32, "mbpp_plus": 16},
        "endpoint": "http://pod-b:8000/v1",
    },
    "C": {
        "sandbox.tasks": [],
        "max_connections_per_task": {"aime2025": 24},
        "sampling.timeout": 14400,
        "endpoint": "http://pod-c:8000/v1",
    },
}
PROV = {
    "arm": "gguf-q5_k_m",
    "variant": "q5_k_m",
    "llama_cpp_commit": "c0ffee" * 6 + "abcd",
    "llama_cpp_repo": "https://example.org/llama.cpp",
    "gguf_file": "/workspace/models/olala-Q5_K_M.gguf",
    "gguf_size": "27000000000",
    "gguf_sha256_pinned": "ab" * 32,
    "gguf_sha256_verified_by_setup": "ab" * 32,
    "hf_repo": "ovh/olala-gguf",
    "hf_revision": "r" * 40,
    "np": "4",
    "ctx": "32768",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jload(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows_of(path: Path) -> list[dict]:
    return [json.loads(line) for line in (path / "samples.jsonl").read_text().splitlines()]


def lines_of(path: Path, *tasks: str) -> list[str]:
    return [
        line
        for line in (path / "samples.jsonl").read_text().splitlines()
        if json.loads(line)["task"] in tasks
    ]


def write_provenance(run: Path, name: str, **override: str | None) -> None:
    """Write ``server_provenance.txt`` and ``server.cmd`` the way ``run_accuracy.sh`` does."""
    fields = {**PROV, **override}
    lines = [f"copied_utc=2026-11-21T10:0{len(name)}:00Z", f"server_log=/w/{name}/server.log"]
    lines += [f"{k}={v}" for k, v in fields.items() if v is not None]
    lines += [f"command=uv run python run_tasks.py --out /w/{name}"]
    (run / "server_provenance.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run / "server.cmd").write_text(
        f"utc=x\narm={fields['arm']}\ncmd of {name}\n", encoding="utf-8"
    )


def pod(tmp_path: Path, group: str, *, provenance: bool = True, **kw) -> Path:
    """The run directory of one pod: its task group only, ``complete``."""
    expected = {t: SMALL_EXPECTED[t] for t in GROUPS[group]}
    kw.setdefault("scores", det(expected, ref_fn))
    kw.setdefault("label", "B0g")
    kw.setdefault("run_seed", ord(group))
    overrides = {**POD_SETTINGS[group], **kw.pop("info_overrides", {})}
    run = build_run(tmp_path / f"run{group}", expected=expected, info_overrides=overrides, **kw)
    if provenance:
        write_provenance(run, group)
    return run


def three(tmp_path: Path, **kw) -> tuple[Path, Path, Path]:
    return pod(tmp_path, "A", **kw), pod(tmp_path, "B", **kw), pod(tmp_path, "C", **kw)


def args_for(runs: tuple[Path, ...], groups: str = "ABC") -> list[str]:
    argv: list[str] = []
    for run, group in zip(runs, groups, strict=True):
        argv += ["--run", f"{run}:{','.join(GROUPS[group])}"]
    return argv


def combine(tmp_path: Path, argv: list[str], *extra: str, out: str = "combined") -> int:
    return ctr.main([*argv, "--out", str(tmp_path / out), *extra])


def refused(tmp_path: Path, capsys, argv: list[str], *extra: str, text: str) -> None:
    before = sorted(p.name for p in tmp_path.iterdir())
    assert combine(tmp_path, argv, *extra) == 2
    err = capsys.readouterr().err
    assert text in err, err
    assert sorted(p.name for p in tmp_path.iterdir()) == before  # nothing written, no .partial


# ------------------------------------------------------------------------------ what is combined


def test_the_rows_of_the_listed_tasks_are_copied_byte_for_byte_with_one_source_per_task(tmp_path):
    a, b, c = three(tmp_path)
    assert combine(tmp_path, args_for((a, b, c))) == 0
    out = tmp_path / "combined"
    for run, group in ((a, "A"), (b, "B"), (c, "C")):
        for task in GROUPS[group]:
            assert lines_of(out, task) == lines_of(run, task)  # the source's exact text lines
    assert len(rows_of(out)) == 20 + 10 + 12 + 14 + 6 * 5 == 86
    assert {r["task"] for r in rows_of(out)} == set(SMALL_EXPECTED)
    completions = {r["task"]: r["completion"] for r in rows_of(out)}
    assert completions["mgsm"] == f"B0g-{ord('A')}"
    assert completions["aime2025"] == f"B0g-{ord('C')}"


def test_only_the_listed_tasks_of_a_source_are_taken(tmp_path):
    full = build_run(tmp_path / "full", label="B0g", scores=det(SMALL_EXPECTED, ref_fn))
    write_provenance(full, "F")
    rest = pod(tmp_path, "B")
    c = pod(tmp_path, "C")
    argv = ["--run", f"{full}:mgsm,humaneval_plus", "--run", f"{rest}:ifeval,mbpp_plus"]
    argv += ["--run", f"{c}:aime2025"]
    assert combine(tmp_path, argv) == 0
    out = tmp_path / "combined"
    assert lines_of(out, "mgsm", "humaneval_plus") == lines_of(full, "mgsm", "humaneval_plus")
    assert len(rows_of(out)) == 86  # the full run's own ifeval, mbpp_plus and aime rows are left


def test_items_json_is_the_union_one_source_per_task(tmp_path):
    a, b, c = three(tmp_path)
    assert combine(tmp_path, args_for((a, b, c))) == 0
    items = jload(tmp_path / "combined" / "items.json")
    assert list(items) == ["mgsm", "humaneval_plus", "mbpp_plus", "ifeval", "aime2025"]
    for run in (a, b, c):
        for task, ids in jload(run / "items.json").items():
            assert items[task] == ids


def test_the_order_of_the_run_arguments_does_not_change_the_directory_content(tmp_path):
    a, b, c = three(tmp_path)
    assert combine(tmp_path, args_for((a, b, c))) == 0
    argv = ["--run", f"{c}:aime2025", "--run", f"{b}:ifeval,mbpp_plus"]
    argv += ["--run", f"{a}:mgsm,humaneval_plus"]
    assert combine(tmp_path, argv, out="other") == 0
    for name in ("samples.jsonl", "items.json"):
        assert (tmp_path / "combined" / name).read_bytes() == (
            tmp_path / "other" / name
        ).read_bytes()


def test_run_info_is_copied_from_the_first_input_with_counts_recomputed_from_the_rows(tmp_path):
    stale = {"n_samples": 999, "request_errors": 9, "truncations": 9}
    stale_task = {
        t: {"n_samples": 77, "request_errors": 7, "truncations": 7, "items_sha256": "0" * 64}
        for t in SMALL_EXPECTED
    }
    a = pod(
        tmp_path,
        "A",
        info_overrides=stale,
        task_info_overrides=stale_task,
        error_rows={("mgsm", "mgsm-en-00", 1)},
    )
    b = pod(tmp_path, "B", truncated_rows={("ifeval", "ifeval-000", 1)})
    c = pod(tmp_path, "C", truncated_rows={("aime2025", "aime2025-001", e) for e in (1, 2)})
    assert combine(tmp_path, args_for((a, b, c))) == 0
    info = jload(tmp_path / "combined" / "run_info.json")
    first = jload(a / "run_info.json")
    assert info["status"] == "complete"
    assert info["n_samples"] == 86
    assert info["request_errors"] == 1
    assert info["truncations"] == 3
    for key in ("model", "git", "code_sha256", "versions", "reasoning_effort", "seed", "sampling"):
        assert info[key] == first[key]
    assert info["chat_template_kwargs"] == first["chat_template_kwargs"]
    counts = {
        t: (v["n_items"], v["epochs"], v["n_samples"], v["request_errors"], v["truncations"])
        for t, v in info["tasks"].items()
    }
    assert counts == {
        "mgsm": (20, 1, 20, 1, 0),
        "humaneval_plus": (10, 1, 10, 0, 0),
        "mbpp_plus": (12, 1, 12, 0, 0),
        "ifeval": (14, 1, 14, 0, 1),
        "aime2025": (6, 5, 30, 0, 2),
    }
    items = jload(tmp_path / "combined" / "items.json")
    for task, entry in info["tasks"].items():
        assert entry["items_sha256"] == aw.sha256_of_ids(items[task])
        assert entry["eval_status"] == "success"


def test_a_first_input_that_is_still_running_is_refused(tmp_path, capsys):
    a, b, c = three(tmp_path)
    info = jload(a / "run_info.json")
    info["status"] = "running"
    (a / "run_info.json").write_text(json.dumps(info))
    refused(tmp_path, capsys, args_for((a, b, c)), text="not complete")


def test_sandbox_tasks_and_per_task_connections_follow_the_source_of_each_task(tmp_path):
    a, b, c = three(tmp_path)
    assert combine(tmp_path, args_for((a, b, c))) == 0
    info = jload(tmp_path / "combined" / "run_info.json")
    assert info["sandbox"] == {"mode": "local", "tasks": ["humaneval_plus", "mbpp_plus"]}
    assert info["max_connections_per_task"] == {
        "mgsm": 32,
        "humaneval_plus": 16,
        "mbpp_plus": 16,
        "ifeval": 32,
        "aime2025": 24,
    }


def test_the_combined_block_records_every_source(tmp_path):
    a, b, c = three(tmp_path)
    assert combine(tmp_path, args_for((a, b, c))) == 0
    block = jload(tmp_path / "combined" / "run_info.json")["combined"]
    assert block["allow_partial_tasks"] is False
    assert block["missing_tasks"] == []
    assert block["task_sources"] == {
        "mgsm": str(a.resolve()),
        "humaneval_plus": str(a.resolve()),
        "mbpp_plus": str(b.resolve()),
        "ifeval": str(b.resolve()),
        "aime2025": str(c.resolve()),
    }
    assert [s["dir"] for s in block["sources"]] == [str(r.resolve()) for r in (a, b, c)]
    for src, run, group in zip(block["sources"], (a, b, c), "ABC", strict=True):
        assert src["tasks"] == list(GROUPS[group])
        assert src["sha256"]["samples"] == sha(run / "samples.jsonl")
        assert src["sha256"]["run_info"] == sha(run / "run_info.json")
        assert src["sha256"]["items"] == sha(run / "items.json")
        assert src["timeout"] == jload(run / "run_info.json")["sampling"]["timeout"]
        assert src["max_connections_per_task"] == {
            t: POD_SETTINGS[group]["max_connections_per_task"][t] for t in GROUPS[group]
        }
        assert src["model"] == "olala-B0g"
        assert src["endpoint"] == POD_SETTINGS[group]["endpoint"]
        assert src["arm"] == "gguf-q5_k_m"
        assert src["status"] == "complete"
        assert src["task_rows"] == {
            t: SMALL_EXPECTED[t]["items"] * SMALL_EXPECTED[t]["epochs"] for t in GROUPS[group]
        }
        assert src["provenance_sha256"] == {
            "server_provenance.txt": sha(run / "server_provenance.txt"),
            "server.cmd": sha(run / "server.cmd"),
            "quantization_info.json": None,
        }
    assert [s["timeout"] for s in block["sources"]] == [3600, 3600, 14400]


def test_provenance_files_are_copied_from_the_first_source_and_all_are_hashed(tmp_path):
    a, b, c = three(tmp_path, quant_recipe="gguf-none")
    assert combine(tmp_path, args_for((a, b, c))) == 0
    out = tmp_path / "combined"
    for name in ("server_provenance.txt", "server.cmd", "quantization_info.json"):
        assert (out / name).read_bytes() == (a / name).read_bytes()
    assert sorted(p.name for p in out.iterdir()) == [
        "items.json",
        "quantization_info.json",
        "run_info.json",
        "samples.jsonl",
        "server.cmd",
        "server_provenance.txt",
    ]
    block = jload(out / "run_info.json")["combined"]
    for src, run in zip(block["sources"], (a, b, c), strict=True):
        assert src["provenance_sha256"]["server_provenance.txt"] == sha(
            run / "server_provenance.txt"
        )
        assert src["provenance_sha256"]["quantization_info.json"] == sha(
            run / "quantization_info.json"
        )
    assert aw.load_run_dir(out).quantization == aw.load_run_dir(a).quantization


def test_the_source_directories_are_left_untouched(tmp_path):
    runs = three(tmp_path)
    names = ("samples.jsonl", "items.json", "run_info.json", "server_provenance.txt", "server.cmd")
    before = {(r, n): sha(r / n) for r in runs for n in names}
    assert combine(tmp_path, args_for(runs)) == 0
    assert before == {(r, n): sha(r / n) for r in runs for n in names}


def test_the_summary_is_printed(tmp_path, capsys):
    assert combine(tmp_path, args_for(three(tmp_path))) == 0
    out = capsys.readouterr().out
    assert "combined" in out
    assert "86 samples" in out
    assert "3 source" in out


def test_settings_that_the_pods_do_not_share_are_recorded_not_refused(tmp_path):
    a, b, c = three(tmp_path)  # timeout 3600/3600/14400, other concurrency, other endpoints
    assert combine(tmp_path, args_for((a, b, c))) == 0


# ------------------------------------------------------------------------------ analyze_wave


def test_the_combined_directory_passes_the_gate_against_a_full_reference_run(tmp_path):
    ref = build_run(tmp_path / "ref", label="REF", run_seed=1, scores=det(SMALL_EXPECTED, ref_fn))
    assert combine(tmp_path, args_for(three(tmp_path))) == 0
    gate = aw.gate_pair(
        [aw.load_run_dir(ref)], [aw.load_run_dir(tmp_path / "combined")], SMALL_EXPECTED
    )
    assert gate.problems == []
    assert gate.ok


def test_a_directory_with_one_task_short_fails_the_gate_when_partial_tasks_are_allowed(tmp_path):
    ref = build_run(tmp_path / "ref", label="REF", run_seed=1, scores=det(SMALL_EXPECTED, ref_fn))
    a, b, _ = three(tmp_path)
    assert combine(tmp_path, args_for((a, b), "AB"), "--allow-partial-tasks") == 0
    gate = aw.gate_pair(
        [aw.load_run_dir(ref)], [aw.load_run_dir(tmp_path / "combined")], SMALL_EXPECTED
    )
    assert not gate.ok
    assert any("aime2025: absent from items.json" in p for p in gate.problems)


def test_request_errors_of_a_source_still_fail_the_gate_of_the_combined_run(tmp_path):
    ref = build_run(tmp_path / "ref", label="REF", run_seed=1, scores=det(SMALL_EXPECTED, ref_fn))
    errors = {("aime2025", "aime2025-000", 1), ("aime2025", "aime2025-001", 1)}
    a, b, c = pod(tmp_path, "A"), pod(tmp_path, "B"), pod(tmp_path, "C", error_rows=errors)
    assert combine(tmp_path, args_for((a, b, c))) == 0  # the tool does not hide request errors
    gate = aw.gate_pair(
        [aw.load_run_dir(ref)], [aw.load_run_dir(tmp_path / "combined")], SMALL_EXPECTED
    )
    assert any("request errors" in p for p in gate.problems)


def test_the_gate_still_sees_the_item_sets_of_the_sources(tmp_path):
    ref = build_run(tmp_path / "ref", label="REF", run_seed=1, scores=det(SMALL_EXPECTED, ref_fn))
    ids = [f"aime2025-x{i}" for i in range(6)]
    c = pod(tmp_path, "C", items_override={"aime2025": ids}, scores={})
    assert combine(tmp_path, args_for((pod(tmp_path, "A"), pod(tmp_path, "B"), c))) == 0
    gate = aw.gate_pair(
        [aw.load_run_dir(ref)], [aw.load_run_dir(tmp_path / "combined")], SMALL_EXPECTED
    )
    assert any("aime2025: item sets differ" in p for p in gate.problems)


# ------------------------------------------------------------------------------ refusals: tasks


def test_a_task_listed_twice_in_two_runs_is_refused(tmp_path, capsys):
    a, b, c = three(tmp_path)
    argv = ["--run", f"{a}:mgsm,humaneval_plus", "--run", f"{b}:ifeval,mbpp_plus,mgsm"]
    argv += ["--run", f"{c}:aime2025"]
    refused(tmp_path, capsys, argv, text="mgsm is listed twice")


def test_a_task_listed_twice_in_one_run_is_refused(tmp_path, capsys):
    a, b, c = three(tmp_path)
    argv = ["--run", f"{a}:mgsm,mgsm,humaneval_plus", "--run", f"{b}:ifeval,mbpp_plus"]
    argv += ["--run", f"{c}:aime2025"]
    refused(tmp_path, capsys, argv, text="mgsm is listed twice")


def test_the_same_directory_given_twice_is_refused(tmp_path, capsys):
    a, b, c = three(tmp_path)
    argv = ["--run", f"{a}:mgsm", "--run", f"{a}:humaneval_plus", "--run", f"{b}:ifeval,mbpp_plus"]
    argv += ["--run", f"{c}:aime2025"]
    refused(tmp_path, capsys, argv, text="same directory")


def test_a_task_absent_from_its_items_json_is_refused(tmp_path, capsys):
    a, b, _ = three(tmp_path)
    argv = ["--run", f"{a}:mgsm,humaneval_plus,aime2025", "--run", f"{b}:ifeval,mbpp_plus"]
    refused(tmp_path, capsys, argv, text="aime2025 is absent from")


def test_an_unknown_task_is_refused_even_with_allow_partial_tasks(tmp_path, capsys):
    a, b, c = three(tmp_path)
    info = jload(a / "run_info.json")
    items = jload(a / "items.json")
    items["gsm8k"] = items["mgsm"]
    info["tasks"]["gsm8k"] = info["tasks"]["mgsm"]
    (a / "items.json").write_text(json.dumps(items))
    (a / "run_info.json").write_text(json.dumps(info))
    argv = ["--run", f"{a}:mgsm,humaneval_plus,gsm8k", "--run", f"{b}:ifeval,mbpp_plus"]
    argv += ["--run", f"{c}:aime2025"]
    refused(tmp_path, capsys, argv, "--allow-partial-tasks", text="gsm8k is not a protocol task")


def test_a_union_short_of_the_five_protocol_tasks_is_refused_by_default(tmp_path, capsys):
    a, b, _ = three(tmp_path)
    refused(tmp_path, capsys, args_for((a, b), "AB"), text="missing aime2025")


def test_the_message_for_a_short_union_names_the_flag(tmp_path, capsys):
    a, b, _ = three(tmp_path)
    assert combine(tmp_path, args_for((a, b), "AB")) == 2
    assert "--allow-partial-tasks" in capsys.readouterr().err


def test_allow_partial_tasks_accepts_a_short_union_and_says_so(tmp_path, capsys):
    a, b, _ = three(tmp_path)
    assert combine(tmp_path, args_for((a, b), "AB"), "--allow-partial-tasks") == 0
    out = capsys.readouterr().out
    assert "PARTIAL" in out
    assert "aime2025" in out
    info = jload(tmp_path / "combined" / "run_info.json")
    assert info["combined"]["allow_partial_tasks"] is True
    assert info["combined"]["missing_tasks"] == ["aime2025"]
    assert sorted(info["tasks"]) == ["humaneval_plus", "ifeval", "mbpp_plus", "mgsm"]
    assert info["n_samples"] == 56


def test_allow_partial_tasks_does_not_note_a_partial_union_for_a_full_one(tmp_path, capsys):
    assert combine(tmp_path, args_for(three(tmp_path)), "--allow-partial-tasks") == 0
    assert "PARTIAL" not in capsys.readouterr().out
    assert jload(tmp_path / "combined" / "run_info.json")["combined"]["missing_tasks"] == []


# ------------------------------------------------------------------------------ refusals: rows


def test_a_source_with_an_unfinished_extra_task_is_refused(tmp_path, capsys):
    """Run A also had aime2025, which never finished: its status is not complete."""
    full = build_run(
        tmp_path / "full",
        label="B0g",
        status="incomplete",
        scores=det(SMALL_EXPECTED, ref_fn),
        drop_rows={("aime2025", f"aime2025-{i:03d}", e) for i in range(6) for e in range(1, 6)},
    )
    b, c = pod(tmp_path, "B"), pod(tmp_path, "C")
    argv = ["--run", f"{full}:mgsm,humaneval_plus", "--run", f"{b}:ifeval,mbpp_plus"]
    argv += ["--run", f"{c}:aime2025"]
    refused(tmp_path, capsys, argv, text="not complete (run_info status is 'incomplete')")


@pytest.mark.parametrize("status", ["running", "incomplete", "failed", None])
def test_every_status_but_complete_is_refused(tmp_path, capsys, status):
    a, b, c = three(tmp_path)
    info = jload(b / "run_info.json")
    if status is None:
        del info["status"]
    else:
        info["status"] = status
    (b / "run_info.json").write_text(json.dumps(info))
    refused(tmp_path, capsys, args_for((a, b, c)), text="not complete")


def test_a_missing_row_is_refused(tmp_path, capsys):
    a, b, c = three(tmp_path, drop_rows={("mgsm", "mgsm-en-00", 1)})
    refused(tmp_path, capsys, args_for((a, b, c)), text="mgsm: 19 rows, expected 20 (1 missing")


def test_a_missing_epoch_of_aime_is_refused(tmp_path, capsys):
    c = pod(tmp_path, "C", drop_rows={("aime2025", "aime2025-002", 4)})
    refused(
        tmp_path,
        capsys,
        args_for((pod(tmp_path, "A"), pod(tmp_path, "B"), c)),
        text="aime2025: 29 rows, expected 30 (1 missing",
    )


def test_a_duplicated_row_is_refused(tmp_path, capsys):
    a, b, c = three(tmp_path, duplicate_rows={("ifeval", "ifeval-003", 1)})
    refused(tmp_path, capsys, args_for((a, b, c)), text="1 duplicated")


def test_an_unexpected_row_is_refused(tmp_path, capsys):
    a, b, c = three(tmp_path)
    extra = json.loads(lines_of(a, "mgsm")[0])
    extra["sample_id"] = "mgsm-en-99"
    with (a / "samples.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(extra) + "\n")
    refused(tmp_path, capsys, args_for((a, b, c)), text="1 unexpected")


def test_a_row_of_an_unexpected_epoch_is_refused(tmp_path, capsys):
    a, b, c = three(tmp_path, task_info_overrides={"aime2025": {"epochs": 4}})
    refused(tmp_path, capsys, args_for((a, b, c)), text="aime2025: 30 rows, expected 24")


def test_run_info_without_epochs_for_a_listed_task_is_refused(tmp_path, capsys):
    a, b, c = three(tmp_path)
    info = jload(c / "run_info.json")
    del info["tasks"]["aime2025"]["epochs"]
    (c / "run_info.json").write_text(json.dumps(info))
    refused(tmp_path, capsys, args_for((a, b, c)), text="no epochs for aime2025")


def test_a_malformed_row_is_refused_not_a_traceback(tmp_path, capsys):
    a, b, c = three(tmp_path)
    lines = (a / "samples.jsonl").read_text().splitlines()
    row = json.loads(lines[0])
    del row["epoch"]
    lines[0] = json.dumps(row)
    (a / "samples.jsonl").write_text("\n".join(lines) + "\n")
    refused(tmp_path, capsys, args_for((a, b, c)), text="malformed row")


def test_a_duplicated_item_id_in_items_json_is_refused(tmp_path, capsys):
    a, b, c = three(tmp_path)
    items = jload(c / "items.json")
    items["aime2025"][1] = items["aime2025"][0]
    (c / "items.json").write_text(json.dumps(items))
    refused(tmp_path, capsys, args_for((a, b, c)), text="aime2025: items.json lists")


def test_an_unreadable_source_is_refused(tmp_path, capsys):
    a, b, c = three(tmp_path)
    (b / "items.json").unlink()
    refused(tmp_path, capsys, args_for((a, b, c)), text="missing file items.json")


def test_a_missing_source_directory_is_refused(tmp_path, capsys):
    a, _, c = three(tmp_path)
    refused(tmp_path, capsys, args_for((a, tmp_path / "nope", c)), text="is not a directory")


# ------------------------------------------------------------------------------ refusals: settings

DIFFERENT = {
    "sampling.temperature": 0.7,
    "sampling.top_p": 0.5,
    "sampling.top_k": 10,
    "sampling.max_tokens": 1000,
    "sampling.seed": 5,
    "reasoning_effort": "high",
    "chat_template_kwargs": {"reasoning_effort": "high"},
    "sandbox.mode": "docker",
    "seed": 1,
}


def test_the_settings_compared_are_those_of_merge_aime_rerun_plus_the_selection_seed():
    assert set(ctr.MUST_MATCH) == set(ctr.mar.MUST_MATCH) | {"seed"}
    assert set(ctr.MUST_MATCH) == set(DIFFERENT)


@pytest.mark.parametrize("key", sorted(DIFFERENT))
@pytest.mark.parametrize("which", ["B", "C"])
def test_a_source_that_differs_in_a_setting_that_must_match_is_refused(
    tmp_path, capsys, key, which
):
    over = {which: {key: DIFFERENT[key]}}
    a = pod(tmp_path, "A")
    b = pod(tmp_path, "B", info_overrides=over.get("B", {}))
    c = pod(tmp_path, "C", info_overrides=over.get("C", {}))
    refused(tmp_path, capsys, args_for((a, b, c)), text=f"differs from the first input in {key}")


def test_the_timeout_the_concurrency_and_the_endpoint_may_differ(tmp_path):
    a = pod(tmp_path, "A", info_overrides={"sampling.timeout": 100, "max_connections": 8})
    b = pod(tmp_path, "B", info_overrides={"sampling.timeout": 200, "endpoint": "http://z/v1"})
    c = pod(tmp_path, "C")
    assert combine(tmp_path, args_for((a, b, c))) == 0


def test_a_different_model_label_is_recorded_and_warned_about(tmp_path, capsys):
    a, b = pod(tmp_path, "A"), pod(tmp_path, "B", label="other")
    assert combine(tmp_path, args_for((a, b, pod(tmp_path, "C")))) == 0
    assert "model label differs" in capsys.readouterr().err
    block = jload(tmp_path / "combined" / "run_info.json")["combined"]
    assert [s["model"] for s in block["sources"]] == ["olala-B0g", "olala-other", "olala-B0g"]


def test_differing_code_or_versions_are_a_warning_not_a_refusal(tmp_path, capsys):
    a = pod(tmp_path, "A")
    b = pod(tmp_path, "B", info_overrides={"code_sha256": "d" * 64})
    assert combine(tmp_path, args_for((a, b, pod(tmp_path, "C")))) == 0
    assert "code_sha256 differs" in capsys.readouterr().err


# ---------------------------------------------------------------------------- refusals: provenance


@pytest.mark.parametrize(
    "key",
    [
        "arm",
        "gguf_sha256_pinned",
        "gguf_sha256_verified_by_setup",
        "llama_cpp_commit",
        "hf_revision",
        "gguf_size",
    ],
)
def test_a_source_served_by_another_arm_or_file_is_refused(tmp_path, capsys, key):
    a, b = pod(tmp_path, "A"), pod(tmp_path, "B")
    c = pod(tmp_path, "C")
    write_provenance(c, "C", **{key: "different"})
    refused(tmp_path, capsys, args_for((a, b, c)), text=f"server_provenance.txt differs in {key}")


def test_a_provenance_key_present_in_one_source_only_is_refused(tmp_path, capsys):
    a, b, c = three(tmp_path)
    write_provenance(b, "B", gguf_sha256_pinned=None)
    refused(tmp_path, capsys, args_for((a, b, c)), text="gguf_sha256_pinned")


def test_provenance_in_some_sources_only_is_refused(tmp_path, capsys):
    a, b = pod(tmp_path, "A"), pod(tmp_path, "B", provenance=False)
    refused(
        tmp_path,
        capsys,
        args_for((a, b, pod(tmp_path, "C"))),
        text="server_provenance.txt is missing",
    )


def test_no_provenance_anywhere_is_accepted_with_a_warning_and_nothing_is_copied(tmp_path, capsys):
    runs = three(tmp_path, provenance=False)
    assert combine(tmp_path, args_for(runs)) == 0
    assert "no server_provenance.txt" in capsys.readouterr().err
    out = tmp_path / "combined"
    assert not (out / "server_provenance.txt").exists()
    assert not (out / "server.cmd").exists()
    block = jload(out / "run_info.json")["combined"]
    assert all(s["arm"] is None for s in block["sources"])
    assert block["provenance_compared"] is None


def test_provenance_lines_that_may_differ_do_not_matter(tmp_path):
    a, b, c = three(tmp_path)  # copied_utc, server_log and command differ between the pods
    assert combine(tmp_path, args_for((a, b, c))) == 0
    block = jload(tmp_path / "combined" / "run_info.json")["combined"]
    assert block["provenance_compared"]["arm"] == "gguf-q5_k_m"
    assert block["provenance_compared"]["gguf_sha256_pinned"] == "ab" * 32


def test_an_unverified_gguf_checksum_is_warned_about(tmp_path, capsys):
    runs = three(tmp_path)
    for run, group in zip(runs, "ABC", strict=True):
        write_provenance(run, group, gguf_sha256_verified_by_setup="unverified")
    assert combine(tmp_path, args_for(runs)) == 0
    assert "unverified" in capsys.readouterr().err


def test_quantization_info_that_differs_is_refused(tmp_path, capsys):
    a = pod(tmp_path, "A", quant_recipe="fp8")
    b = pod(tmp_path, "B", quant_recipe="fp8")
    c = pod(tmp_path, "C", quant_recipe="nvfp4")
    refused(tmp_path, capsys, args_for((a, b, c)), text="quantization_info.json differs")


def test_quantization_info_in_some_sources_only_is_refused(tmp_path, capsys):
    a = pod(tmp_path, "A", quant_recipe="fp8")
    refused(
        tmp_path,
        capsys,
        args_for((a, pod(tmp_path, "B"), pod(tmp_path, "C"))),
        text="quantization_info.json is missing",
    )


# ------------------------------------------------------------------------------ refusals: usage


def test_an_existing_out_is_refused(tmp_path, capsys):
    runs = three(tmp_path)
    (tmp_path / "combined").mkdir()
    refused(tmp_path, capsys, args_for(runs), text="exists")


@pytest.mark.parametrize("spec", ["nocolon", "dir:", ":mgsm", "dir:mgsm,,ifeval"])
def test_a_malformed_run_argument_is_refused(tmp_path, capsys, spec):
    refused(tmp_path, capsys, ["--run", spec], text="invalid --run")


def test_no_run_argument_is_a_usage_error(tmp_path):
    with pytest.raises(SystemExit) as exc:
        ctr.main(["--out", str(tmp_path / "x")])
    assert exc.value.code == 2


def test_a_stale_partial_directory_is_replaced_and_gone_afterwards(tmp_path):
    runs = three(tmp_path)
    stale = tmp_path / ".combined.partial"
    stale.mkdir()
    (stale / "junk").write_text("x")
    assert combine(tmp_path, args_for(runs)) == 0
    assert not stale.exists()
    assert (tmp_path / "combined" / "samples.jsonl").is_file()


def test_a_failure_while_writing_leaves_neither_out_nor_partial(tmp_path, monkeypatch):
    runs = three(tmp_path)
    before = sorted(p.name for p in tmp_path.iterdir())

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(ctr.shutil, "copyfile", boom)
    with pytest.raises(OSError, match="disk full"):
        combine(tmp_path, args_for(runs))
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_the_output_appears_by_renaming_a_finished_partial_directory(tmp_path, monkeypatch):
    runs = three(tmp_path)
    seen: list[tuple[Path, Path, list[str]]] = []
    real = Path.rename

    def spy(self: Path, target):
        seen.append((self, Path(target), sorted(p.name for p in self.iterdir())))
        return real(self, target)

    monkeypatch.setattr(Path, "rename", spy)
    assert combine(tmp_path, args_for(runs)) == 0
    ((src, dst, names),) = seen
    assert src.name == ".combined.partial"
    assert dst == tmp_path / "combined"
    assert {"samples.jsonl", "items.json", "run_info.json"} <= set(names)


# ------------------------------------------------------------------------------ retries, timeouts


def test_retry_counters_are_recomputed_from_the_rows_of_the_combined_run(tmp_path):
    a = pod(
        tmp_path,
        "A",
        attempts={("mgsm", "mgsm-en-00", 1): 2, ("mgsm", "mgsm-fr-00", 1): 3},
        working_times={("humaneval_plus", "humaneval_plus-001", 1): 80.0},
        task_info_overrides={"mgsm": {"samples_with_retries": 99, "max_working_time": 1e9}},
    )
    b = pod(tmp_path, "B", attempts={("ifeval", "ifeval-000", 1): 2})
    c = pod(
        tmp_path,
        "C",
        attempts={("aime2025", "aime2025-001", 2): 2},
        working_times={("aime2025", "aime2025-002", 1): 900.0},
    )
    assert combine(tmp_path, args_for((a, b, c))) == 0
    info = jload(tmp_path / "combined" / "run_info.json")
    assert info["samples_with_retries"] == 4
    assert info["max_working_time"] == 900.0
    assert {t: v["samples_with_retries"] for t, v in info["tasks"].items()} == {
        "mgsm": 2,
        "humaneval_plus": 0,
        "mbpp_plus": 0,
        "ifeval": 1,
        "aime2025": 1,
    }
    assert info["tasks"]["mgsm"]["max_working_time"] == 1.0  # the stale 1e9 is replaced
    assert info["tasks"]["humaneval_plus"]["max_working_time"] == 80.0
    assert info["tasks"]["aime2025"]["max_working_time"] == 900.0


def test_the_client_timeout_is_kept_per_task_and_the_total_is_the_smallest(tmp_path):
    a, b, c = three(tmp_path)  # pods A and B: 3600; pod C: sampling.timeout 14400, so the same
    assert combine(tmp_path, args_for((a, b, c))) == 0
    info = jload(tmp_path / "combined" / "run_info.json")
    assert {t: v["client_timeout"] for t, v in info["tasks"].items()} == {
        "mgsm": 3600,
        "humaneval_plus": 3600,
        "mbpp_plus": 3600,
        "ifeval": 3600,
        "aime2025": 14400,
    }
    assert info["client_timeout"] == 3600
    blocks = {tuple(s["tasks"]): s["client_timeout"] for s in info["combined"]["sources"]}
    assert blocks[("aime2025",)] == {"aime2025": 14400}
    assert blocks[("mgsm", "humaneval_plus")] == {"mgsm": 3600, "humaneval_plus": 3600}


def test_a_legacy_pod_leaves_its_tasks_without_counters_and_the_run_without_a_client_timeout(
    tmp_path,
):
    a = pod(tmp_path, "A")
    b = pod(tmp_path, "B", legacy=True)
    c = pod(tmp_path, "C", attempts={("aime2025", "aime2025-001", 2): 2})
    assert combine(tmp_path, args_for((a, b, c))) == 0
    out = tmp_path / "combined"
    for task in GROUPS["B"]:
        assert lines_of(out, task) == lines_of(b, task)  # byte for byte, still without the fields
    assert all("model_attempts" not in r for r in rows_of(out) if r["task"] in GROUPS["B"])
    info = jload(out / "run_info.json")
    for task in GROUPS["B"]:
        assert "client_timeout" not in info["tasks"][task]
        assert "samples_with_retries" not in info["tasks"][task]
    assert "client_timeout" not in info
    assert info["samples_with_retries"] == 1  # over the 56 rows that say; the other 30 are unknown


def test_legacy_pods_alone_produce_no_new_field_at_all(tmp_path):
    a, b, c = three(tmp_path, legacy=True)
    assert combine(tmp_path, args_for((a, b, c))) == 0
    info = jload(tmp_path / "combined" / "run_info.json")
    for key in ("client_timeout", "samples_with_retries", "max_working_time"):
        assert key not in info
        assert all(key not in t for t in info["tasks"].values())


def test_the_combined_directory_of_new_pods_is_read_by_the_gate(tmp_path):
    a = pod(tmp_path, "A")
    b = pod(tmp_path, "B")
    c = pod(tmp_path, "C", attempts={("aime2025", "aime2025-001", 2): 2})
    assert combine(tmp_path, args_for((a, b, c))) == 0
    ref = build_run(tmp_path / "ref", label="REF", scores=det(SMALL_EXPECTED, ref_fn))
    gate = aw.gate_pair(
        [aw.load_run_dir(ref)], [aw.load_run_dir(tmp_path / "combined")], SMALL_EXPECTED
    )
    assert any("aime2025" in p and "1 of 30" in p for p in gate.problems)
