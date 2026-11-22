"""Windows per domain of a CAL-1 manifest: which domain every token of ``windows.json`` comes from."""

import json
import shutil
from pathlib import Path

import pytest
import torch

import calib_data
import window_domains as wd
from tests_support import make_cal1_layout, make_calibration

REAL_DIR = Path(__file__).resolve().parents[3] / "results" / "wave1" / "cal1-final" / "cal1" / "calib-cal1"


def windows_of(layout) -> torch.Tensor:
    return calib_data.read_windows(layout.manifest)


def oracle(layout, n_windows: int | None = None) -> dict[str, dict[str, int]]:
    """Ground truth from the labels the generator wrote: tokens per domain and windows touched, brute force."""
    w = layout.window
    labels = layout.labels if n_windows is None else layout.labels[: n_windows * w]
    out: dict[str, dict] = {}
    for i, d in enumerate(labels):
        rec = out.setdefault(d, {"tokens": 0, "windows": set()})
        rec["tokens"] += 1
        rec["windows"].add(i // w)
    return {d: {"tokens": r["tokens"], "windows_touched": len(r["windows"])} for d, r in out.items()}


def test_exact_attribution_matches_the_ground_truth_of_the_generator(tmp_path: Path) -> None:
    lay = make_cal1_layout(tmp_path)
    res = wd.windows_per_domain(lay.manifest, windows_of(lay))
    assert res["method"] == "exact"
    truth = oracle(lay)
    assert set(res["domains"]) == set(truth)
    for d, t in truth.items():
        assert res["domains"][d]["tokens"] == t["tokens"], d
        assert res["domains"][d]["windows_touched"] == t["windows_touched"], d
        assert res["domains"][d]["window_equivalents"] == pytest.approx(t["tokens"] / lay.window)
    assert sum(r["tokens"] for r in res["domains"].values()) == windows_of(lay).numel()
    assert res["n_windows_used"] == windows_of(lay).shape[0]


def test_a_prefix_of_the_windows_is_attributed_exactly_too(tmp_path: Path) -> None:
    lay = make_cal1_layout(tmp_path)
    res = wd.windows_per_domain(lay.manifest, windows_of(lay)[:4])
    truth = oracle(lay, 4)
    assert res["method"] == "exact" and res["n_windows_used"] == 4
    for d, t in truth.items():
        assert res["domains"][d]["tokens"] == t["tokens"], d
        assert res["domains"][d]["windows_touched"] == t["windows_touched"], d
    assert sum(r["tokens"] for r in res["domains"].values()) == 4 * lay.window


def test_every_domain_of_the_manifest_is_listed_even_when_the_used_windows_hold_none_of_it(tmp_path: Path) -> None:
    lay = make_cal1_layout(tmp_path)
    res = wd.windows_per_domain(lay.manifest, windows_of(lay)[:1])
    assert set(res["domains"]) == set(lay.n_traces)
    assert any(r["tokens"] == 0 and r["windows_touched"] == 0 for r in res["domains"].values())


def test_a_changed_text_file_is_refused(tmp_path: Path) -> None:
    lay = make_cal1_layout(tmp_path)
    with (lay.directory / "calib_text.jsonl").open("a") as f:
        f.write(json.dumps({"source": "prompt", "doc": "math_train/default-99", "text": "x"}) + "\n")
    with pytest.raises(calib_data.ManifestError, match=r"calib_text\.jsonl"):
        wd.windows_per_domain(lay.manifest, windows_of(lay))


def test_without_the_text_file_the_attribution_is_an_estimate_by_token_share(tmp_path: Path) -> None:
    lay = make_cal1_layout(tmp_path)
    (lay.directory / "calib_text.jsonl").unlink()
    w = windows_of(lay)
    res = wd.windows_per_domain(lay.manifest, w)
    assert res["method"] == "token_share_estimate" and "calib_text.jsonl" in res["reason"]
    total = sum(lay.tokens[d] + lay.n_traces[d] for d in lay.tokens)
    for d in lay.tokens:
        share = (lay.tokens[d] + lay.n_traces[d]) / total
        assert res["domains"][d]["window_equivalents"] == pytest.approx(share * w.shape[0])
        assert res["domains"][d]["windows_touched"] is None
    assert sum(r["window_equivalents"] for r in res["domains"].values()) == pytest.approx(w.shape[0])


def test_a_trace_count_that_disagrees_with_the_manifest_falls_back_to_the_estimate(tmp_path: Path) -> None:
    lay = make_cal1_layout(tmp_path)
    m = json.loads(lay.manifest.read_text())
    m["counts"]["n_traces"] += 1
    lay.manifest.write_text(json.dumps(m))
    res = wd.windows_per_domain(lay.manifest, windows_of(lay))
    assert res["method"] == "token_share_estimate" and "n_traces" in res["reason"]


def test_a_marker_that_occurs_more_often_than_there_are_traces_falls_back_to_the_estimate(tmp_path: Path) -> None:
    lay = make_cal1_layout(tmp_path)
    w = windows_of(lay).clone()
    n_traces = sum(lay.n_traces.values())
    assert w.shape[0] > n_traces  # enough windows to plant one extra marker per window
    for k in range(1, w.shape[0]):
        w[k, : wd.MARKER_TOKENS] = w[0, : wd.MARKER_TOKENS]
    res = wd.windows_per_domain(lay.manifest, w)
    assert res["method"] == "token_share_estimate" and "marker" in res["reason"]


def test_attribution_that_cannot_add_up_to_the_manifest_totals_falls_back_to_the_estimate(tmp_path: Path) -> None:
    lay = make_cal1_layout(tmp_path)
    m = json.loads(lay.manifest.read_text())
    m["domains"]["code"]["n_tokens"] -= 500  # the windows hold more code than the manifest says it rendered
    lay.manifest.write_text(json.dumps(m))
    res = wd.windows_per_domain(lay.manifest, windows_of(lay))
    assert res["method"] == "token_share_estimate" and "code" in res["reason"]


def test_a_manifest_without_domains_has_no_attribution(tmp_path: Path) -> None:
    manifest = make_calibration(tmp_path)  # the lite bundle: schema g1lite-calib/1, no domains
    res = wd.windows_per_domain(manifest, calib_data.read_windows(manifest))
    assert res["method"] == "unavailable" and "domains" in res["reason"]


@pytest.mark.skipif(not REAL_DIR.is_dir(), reason="the real CAL-1 output is not in this checkout")
def test_on_the_real_cal1_windows_the_attribution_is_exact_and_adds_up_to_the_manifest(tmp_path: Path) -> None:
    d = tmp_path / "real"
    d.mkdir()
    for name in ("manifest.json", "windows.json", "calib_text.jsonl"):
        shutil.copy2(REAL_DIR / name, d / name)
    m = json.loads((d / "manifest.json").read_text())
    res = wd.windows_per_domain(d / "manifest.json", calib_data.read_windows(d / "manifest.json"))
    assert res["method"] == "exact" and res["n_windows_used"] == 291
    got = {k: v["tokens"] for k, v in res["domains"].items()}
    assert sum(got.values()) == 291 * 2048
    for k, v in m["domains"].items():
        slots = v["n_tokens"] + v["n_traces"]
        assert got[k] <= slots and slots - got[k] < 2048  # only the dropped tail is missing, from one domain
    assert sum(m["domains"][k]["n_tokens"] + m["domains"][k]["n_traces"] - got[k] for k in got) < 2048


def test_a_manifest_that_rendered_more_than_the_windows_hold_falls_back_to_the_estimate(tmp_path: Path) -> None:
    lay = make_cal1_layout(tmp_path)
    m = json.loads(lay.manifest.read_text())
    m["domains"]["code"]["n_tokens"] += 3 * lay.window  # windows from other traces: more than a tail is missing
    lay.manifest.write_text(json.dumps(m))
    res = wd.windows_per_domain(lay.manifest, windows_of(lay))
    assert res["method"] == "token_share_estimate" and "tail" in res["reason"]


def test_a_prompt_of_an_unknown_source_falls_back_to_the_estimate(tmp_path: Path) -> None:
    lay = make_cal1_layout(tmp_path)
    m = json.loads(lay.manifest.read_text())
    m["prompt_sources"] = [s for s in m["prompt_sources"] if s["name"] != "code_oss_python"]
    lay.manifest.write_text(json.dumps(m))
    res = wd.windows_per_domain(lay.manifest, windows_of(lay))
    assert res["method"] == "token_share_estimate" and "unknown source" in res["reason"]


def test_a_manifest_of_another_schema_has_no_attribution_even_with_domains(tmp_path: Path) -> None:
    lay = make_cal1_layout(tmp_path)
    m = json.loads(lay.manifest.read_text())
    m["schema"] = "other/1"
    lay.manifest.write_text(json.dumps(m))
    res = wd.windows_per_domain(lay.manifest, windows_of(lay))
    assert res["method"] == "unavailable" and "other/1" in res["reason"]
