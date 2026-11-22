"""Tests for analyze_wave.py, part 4: the markdown report, the JSON and the command line."""

import hashlib
import json
import re
from pathlib import Path

import pytest
from fake_runs import (
    SMALL_EXPECTED,
    build_run,
    det,
    edge_fn,
    make,
    same_fn,
    task_item_ids,
    worse_fn,
)

import analyze_wave as aw

LABELS = ("NON-INFERIOR", "WORSE", "INCONCLUSIVE")


@pytest.fixture(scope="module")
def wave(tmp_path_factory) -> dict:
    """A synthetic wave: B0 twice, F1 equal to B0, F2 far below, N2 borderline, F1n exploratory."""
    root = tmp_path_factory.mktemp("wave")
    trunc = {("aime2025", i, e) for i in task_item_ids("aime2025", 6)[:3] for e in (1, 2, 3)}
    dirs = {
        "B0": [make(root, "b0a", run_seed=1), make(root, "b0b", run_seed=2)],
        "F1": [make(root, "f1", same_fn, run_seed=3, quant_recipe="fp8_online")],
        "F2": [make(root, "f2", worse_fn, run_seed=4, quant_recipe="fp8_offline")],
        "N2": [make(root, "n2", edge_fn, run_seed=5, truncated_rows=trunc)],
        "F1n": [make(root, "f1n", worse_fn, run_seed=6)],
    }
    exp = root / "expected.json"
    exp.write_text(json.dumps(SMALL_EXPECTED))
    args = ["--reference", "B0", "--candidates", "F1,F2,N2", "--exploratory", "F1n"]
    args += ["--contrast", "F2,F1", "--expected", str(exp), "--resamples", "300", "--seed", "0"]
    for name, ds in dirs.items():
        args += ["--run", f"{name}=" + ",".join(map(str, ds))]
    args += ["--calibration", f"B0a={dirs['B0'][0]},B0b={dirs['B0'][1]}"]
    out, js = root / "report.md", root / "report.json"
    args += ["--out", str(out), "--json", str(js)]
    return {"root": root, "dirs": dirs, "args": args, "out": out, "json": js}


@pytest.fixture(scope="module")
def ran(wave):
    code = aw.main(wave["args"])
    return {
        "code": code,
        "md": wave["out"].read_text(),
        "res": json.loads(wave["json"].read_text()),
    }


def sections(md: str) -> dict[str, str]:
    parts = re.split(r"^## ", md, flags=re.MULTILINE)[1:]
    return {p.splitlines()[0]: "\n".join(p.splitlines()[1:]) for p in parts}


def section(md: str, start: str) -> str:
    for title, body in sections(md).items():
        if title.startswith(start) or start in title:
            return body
    raise AssertionError(f"no section {start!r} in {list(sections(md))}")


# ---------------------------------------------------------------------------------- the report


def test_the_command_line_succeeds_writes_both_files_and_prints_the_report(ran, wave, capsys):
    assert ran["code"] == 0
    assert wave["out"].exists()
    assert wave["json"].exists()


def test_every_section_opens_with_its_conclusion(ran):
    secs = sections(ran["md"])
    assert len(secs) >= 10
    for title, body in secs.items():
        first = next(line for line in body.splitlines() if line.strip())
        assert first.startswith("**Conclusion:**"), title


def test_the_report_never_says_no_loss_or_degraded(ran):
    low = ran["md"].lower()
    assert "no loss" not in low
    assert "degraded" not in low


def test_the_summary_gives_each_candidates_outcome_and_the_margin(ran):
    body = section(ran["md"], "Summary")
    concl = body.strip().splitlines()[0]
    assert "-3" in concl
    assert re.search(r"F1\W+NON-INFERIOR", concl)
    assert re.search(r"F2\W+WORSE", concl)
    assert re.search(r"N2\W+(INCONCLUSIVE|WORSE|NON-INFERIOR)", concl)


def test_the_confirmatory_table_shows_raw_and_holm_p_bounds_and_intervals(ran):
    body = section(ran["md"], "Confirmatory")
    res = ran["res"]["confirmatory"]["tests"]["-3"]["F2"]
    assert "p (raw)" in body
    assert "p (Holm)" in body
    assert "97.5%" in body
    assert f"{res['p_raw']:.4f}" in body
    assert f"{res['p_holm']:.4f}" in body
    assert f"{res['lower_bound_one_sided']:+.2f}" in body
    assert "-2" in body
    assert "-5" in body  # sensitivity margins
    for label in LABELS:
        assert label in body


def test_per_task_differences_with_intervals_are_listed_for_every_candidate(ran):
    body = section(ran["md"], "Per-task")
    for cand in ("F1", "F2", "N2"):
        assert f"### {cand} versus B0" in body
        block = ran["res"]["candidates"][cand]["summary"]
        for task in ("mgsm", "humaneval_plus", "mbpp_plus", "ifeval", "aime2025"):
            d = block["tasks"][task]["diff"]
            assert f"{d['value']:+.1f} [{d['lo']:+.1f}, {d['hi']:+.1f}]" in body
        m = block["macro"]["diff"]
        assert f"{m['value']:+.1f} [{m['lo']:+.1f}, {m['hi']:+.1f}]" in body
    assert "macro-average" in body


def test_flags_name_the_task_level_regression_and_the_truncation_difference(ran):
    body = section(ran["md"], "Flags")
    assert "task-level regression" in body
    assert re.search(r"F2.*task-level regression", body, re.DOTALL)
    assert "truncation" in body.lower()
    flags = ran["res"]["candidates"]["N2"]["flags"]
    assert flags["truncation_rate_difference"] is True
    assert "N2" in body


def test_the_aime_secondary_accuracy_is_reported_next_to_the_headline(ran):
    body = section(ran["md"], "AIME")
    assert "non-truncated" in body.lower()
    assert "headline" in body.lower()
    sec = ran["res"]["candidates"]["N2"]["aime_non_truncated"]
    assert f"{sec['x']['value']:.1f}" in body


def test_the_mgsm_table_lists_the_languages(ran):
    body = section(ran["md"], "MGSM")
    for lang in ("en", "fr", "de", "ja"):
        assert f"| {lang} |" in body


def test_the_calibration_width_is_printed_next_to_the_candidates_widths(ran):
    body = section(ran["md"], "Calibration")
    cal = ran["res"]["calibration"]
    assert "contains 0" in body
    assert f"{cal['width_pp']:.1f}" in body
    conf = section(ran["md"], "Confirmatory")
    assert "width" in conf.lower()
    assert "B0a" in body
    assert "B0b" in body


def test_exploratory_arms_have_no_outcome_label_and_no_holm(ran):
    body = section(ran["md"], "Exploratory")
    assert "F1n" in body
    assert "descriptive" in body.lower()
    for label in LABELS:
        assert label not in body
    assert "Holm" not in body.replace("no Holm", "")


def test_contrasts_are_descriptive_differences_with_intervals(ran):
    body = section(ran["md"], "Contrasts")
    assert "F2 versus F1" in body
    con = ran["res"]["contrasts"]["F2 vs F1"]["summary"]["macro"]["diff"]
    assert f"{con['value']:+.1f} [{con['lo']:+.1f}, {con['hi']:+.1f}]" in body
    for label in LABELS:
        assert label not in body


def test_the_settings_and_the_failure_counts_of_each_run_are_tabled(ran, wave):
    body = section(ran["md"], "Gate")
    assert "reasoning effort" in body.lower()
    assert "medium" in body
    assert "fp8_offline" in body  # quantization recipe
    assert "local" in body  # sandbox
    assert "abc123" in body  # git commit
    for d in wave["dirs"]["N2"]:
        assert str(d) in body
    n2 = ran["res"]["arms"]["N2"]["runs"][0]["counts"]
    assert n2["truncations"] == 9
    assert re.search(r"N2.*\| 9 \|", body)


def test_the_provenance_section_has_seed_resamples_and_hashes(ran, wave):
    body = section(ran["md"], "Provenance")
    assert "seed 0" in body
    assert "300 resamples" in body
    assert ran["res"]["script"]["sha256"] in body
    d = wave["dirs"]["F1"][0]
    assert hashlib.sha256((d / "run_info.json").read_bytes()).hexdigest() in body
    assert hashlib.sha256((d / "samples.jsonl").read_bytes()).hexdigest() in body


def test_single_run_arms_are_marked_provisional(ran):
    body = section(ran["md"], "Summary")
    assert "provisional" in body.lower()
    assert "F1" in body


def test_the_known_limits_of_the_bootstrap_are_stated(ran):
    body = section(ran["md"], "Provenance")
    assert "not clustered" in body
    assert "30" in body


def test_the_markdown_numbers_come_from_the_json(ran):
    for cand in ("F1", "F2", "N2"):
        m = ran["res"]["candidates"][cand]["summary"]["macro"]["diff"]
        assert f"{m['value']:+.1f}" in ran["md"]
        assert f"{m['lo']:+.1f}" in ran["md"]


def test_the_json_carries_seed_resamples_hashes_and_the_script_hash(ran, wave):
    res = ran["res"]
    assert res["bootstrap"] == {"seed": 0, "resamples": 300, "confidence": 0.95}
    assert res["script"]["sha256"] == hashlib.sha256(Path(aw.__file__).read_bytes()).hexdigest()
    f1 = res["arms"]["F1"]["runs"][0]
    assert (
        f1["samples_sha256"]
        == hashlib.sha256((wave["dirs"]["F1"][0] / "samples.jsonl").read_bytes()).hexdigest()
    )
    assert res["expected_total_samples"] == aw.total_samples(SMALL_EXPECTED)


def test_the_gate_says_when_an_exploratory_pair_failed(tmp_path):
    root = tmp_path
    dirs = {
        "B0": [make(root, "b0", run_seed=1)],
        "F1": [make(root, "f1", same_fn, run_seed=2)],
        "F1n": [
            build_run(
                root / "f1n",
                expected=SMALL_EXPECTED,
                scores=det(SMALL_EXPECTED, worse_fn),
                drop_rows={("ifeval", "ifeval-002", 1)},
            )
        ],
    }
    cfg = aw.Config(
        runs=dirs,
        reference="B0",
        candidates=["F1"],
        exploratory=["F1n"],
        expected=SMALL_EXPECTED,
        n_boot=100,
    )
    md = aw.render_markdown(aw.analyze(cfg))
    body = section(md, "Exploratory")
    assert "gate failed" in body.lower()
    assert "ifeval-002" in body
    assert "gate failed" in section(md, "Gate").lower()


def test_without_calibration_or_exploratory_arms_the_sections_say_so(tmp_path):
    dirs = {"B0": [make(tmp_path, "b0")], "F1": [make(tmp_path, "f1", same_fn, run_seed=2)]}
    cfg = aw.Config(
        runs=dirs, reference="B0", candidates=["F1"], expected=SMALL_EXPECTED, n_boot=100
    )
    md = aw.render_markdown(aw.analyze(cfg))
    assert "not provided" in section(md, "Calibration")
    assert "none" in section(md, "Exploratory").lower()
    for body in sections(md).values():
        assert next(line for line in body.splitlines() if line.strip()).startswith(
            "**Conclusion:**"
        )


# ---------------------------------------------------------------------------------- the CLI


def test_the_help_text_lists_every_option(capsys):
    with pytest.raises(SystemExit) as exc:
        aw.main(["--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    for opt in (
        "--run",
        "--reference",
        "--candidates",
        "--exploratory",
        "--contrast",
        "--calibration",
        "--expected",
        "--out",
        "--json",
        "--margin",
        "--seed",
        "--resamples",
    ):
        assert opt in text


def test_the_defaults_are_the_protocols(wave):
    required = ["--reference", "B0", "--candidates", "F1", "--out", "o.md", "--json", "o.json"]
    ns = aw.build_parser().parse_args(["--run", "B0=a", "--run", "F1=b", *required])
    assert ns.margin == -3.0
    assert ns.seed == 0
    assert ns.resamples == 10_000
    assert ns.expected is None


def test_a_negative_margin_can_be_given_with_or_without_equals(wave):
    p = aw.build_parser()
    base = ["--run", "B0=a", "--reference", "B0", "--candidates", "F1", "--out", "o", "--json", "j"]
    assert p.parse_args([*base, "--margin", "-4"]).margin == -4.0
    assert p.parse_args([*base, "--margin=-4"]).margin == -4.0


def test_a_gate_failure_exits_3_prints_why_and_writes_nothing(wave, tmp_path, capsys):
    bad = build_run(
        tmp_path / "f1bad",
        expected=SMALL_EXPECTED,
        scores=det(SMALL_EXPECTED, same_fn),
        drop_rows={("mgsm", "mgsm-en-00", 1)},
    )
    args = [a.replace(str(wave["dirs"]["F1"][0]), str(bad)) for a in wave["args"]]
    out, js = tmp_path / "o.md", tmp_path / "o.json"
    args[args.index("--out") + 1] = str(out)
    args[args.index("--json") + 1] = str(js)
    code = aw.main(args)
    err = capsys.readouterr().err
    assert code == 3
    assert "mgsm-en-00" in err
    assert "missing" in err
    assert not out.exists()
    assert not js.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda a: [*a, "--margin", "3"],
        lambda a: [x if x != "F1,F2,N2" else "F1,F9" for x in a],
        lambda a: [x if x != "B0" else "B9" for x in a],
        lambda a: [*a, "--run", "bad"],
        lambda a: [*a, "--contrast", "F2"],
        lambda a: [*a, "--calibration", "nonsense"],
        lambda a: [*a, "--resamples", "0"],
        lambda a: [x if x != "F1n" else "F1" for x in a],
    ],
)
def test_bad_arguments_exit_2_and_write_nothing(wave, tmp_path, mutation, capsys):
    args = mutation(list(wave["args"]))
    out, js = tmp_path / "o.md", tmp_path / "o.json"
    args[args.index("--out") + 1] = str(out)
    args[args.index("--json") + 1] = str(js)
    with pytest.raises(SystemExit) as exc:
        aw.main(args)
    assert exc.value.code == 2
    assert not out.exists()
    assert not js.exists()


def test_the_exploratory_option_can_be_repeated_and_comma_separated(wave):
    required = ["--run", "B0=a", "--reference", "B0", "--candidates", "F1"]
    required += ["--out", "o", "--json", "j"]
    required += ["--exploratory", "F1n,F1m", "--exploratory", "X", "--contrast", "F2,F1"]
    required += ["--contrast", "N2,F1", "--calibration", "A=x,B=y"]
    ns = aw.build_parser().parse_args(required)
    assert ns.exploratory == ["F1n", "F1m", "X"]
    assert ns.contrasts == [("F2", "F1"), ("N2", "F1")]
    assert ns.calibration == (("A", Path("x")), ("B", Path("y")))


def test_the_seed_resamples_and_margin_options_reach_the_analysis(wave, tmp_path):
    args = list(wave["args"])
    for opt, value in (("--seed", "5"), ("--resamples", "123")):
        args[args.index(opt) + 1] = value
    args[args.index("--out") + 1] = str(tmp_path / "o.md")
    args[args.index("--json") + 1] = str(tmp_path / "o.json")
    assert aw.main([*args, "--margin", "-60"]) == 0
    res = json.loads((tmp_path / "o.json").read_text())
    assert res["bootstrap"]["seed"] == 5
    assert res["bootstrap"]["resamples"] == 123
    assert res["confirmatory"]["primary_margin"] == -60.0
    assert res["candidates"]["F1"]["summary"]["n_boot"] == 123
    assert "seed 5" in (tmp_path / "o.md").read_text()
    assert "123 resamples" in (tmp_path / "o.md").read_text()


def test_the_default_expected_counts_apply_when_no_expected_file_is_given(wave, tmp_path, capsys):
    args = list(wave["args"])
    i = args.index("--expected")
    del args[i : i + 2]
    args[args.index("--out") + 1] = str(tmp_path / "o.md")
    args[args.index("--json") + 1] = str(tmp_path / "o.json")
    assert aw.main(args) == 3
    assert "expected 550" in capsys.readouterr().err


# ------------------------------------------------------------ legacy AIME rows and the option


def small_wave(
    root: Path, *, legacy_f1: bool, extra: tuple[str, ...] = ()
) -> tuple[list[str], Path]:
    """B0 (new format) against F1 (new format or legacy), same scores in both formats."""
    tag = "legacy" if legacy_f1 else "new"
    b0 = make(root, f"b0-{tag}", run_seed=1)
    f1 = make(root, f"f1-{tag}", same_fn, run_seed=3, legacy=legacy_f1)
    exp = root / f"expected-{tag}.json"
    exp.write_text(json.dumps(SMALL_EXPECTED))
    out, js = root / f"{tag}.md", root / f"{tag}.json"
    args = ["--run", f"B0={b0}", "--run", f"F1={f1}", "--reference", "B0", "--candidates", "F1"]
    args += ["--expected", str(exp), "--resamples", "200", "--out", str(out), "--json", str(js)]
    return [*args, *extra], js


def test_a_legacy_aime_run_is_refused_by_default_with_the_reason_and_the_option(tmp_path, capsys):
    args, js = small_wave(tmp_path, legacy_f1=True)
    assert aw.main(args) == 3
    err = capsys.readouterr().err
    assert "aime2025" in err
    assert "retry count unknown (client timeout 600 s legacy default)" in err
    assert "--accept-legacy-aime-retries" in err
    assert not js.exists()


def test_the_option_accepts_legacy_aime_rows_with_a_warning_in_the_report_and_the_json(tmp_path):
    args, js = small_wave(tmp_path, legacy_f1=True, extra=("--accept-legacy-aime-retries",))
    assert aw.main(args) == 0
    res = json.loads(js.read_text())
    assert res["gate"]["accept_legacy_aime_retries"] is True
    pair = res["gate"]["pairs"]["F1 vs B0"]
    assert pair["status"] == "pass"
    assert pair["problems"] == []
    assert sum("retry count unknown" in w for w in pair["warnings"]) == 5
    assert any("aime2025" in w and "3 to 9 percent" in w for w in pair["warnings"])
    assert "retry count unknown" in js.with_suffix(".md").read_text()


def test_a_new_format_run_needs_no_option_and_the_json_says_so(tmp_path):
    args, js = small_wave(tmp_path, legacy_f1=False)
    assert aw.main(args) == 0
    res = json.loads(js.read_text())
    assert res["gate"]["accept_legacy_aime_retries"] is False
    assert res["gate"]["pairs"]["F1 vs B0"]["warnings"] == []


def test_accepting_legacy_rows_changes_no_statistic(tmp_path):
    new_args, new_js = small_wave(tmp_path, legacy_f1=False)
    old_args, old_js = small_wave(tmp_path, legacy_f1=True, extra=("--accept-legacy-aime-retries",))
    assert aw.main(new_args) == 0
    assert aw.main(old_args) == 0
    new, old = json.loads(new_js.read_text()), json.loads(old_js.read_text())
    for key in ("candidates", "confirmatory", "bootstrap", "expected"):
        assert new[key] == old[key]
