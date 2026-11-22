"""Tests of the calibration-set builder: sources, manifest, windows, and the refusal on leakage."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

import calib_data as cd
import leakage_audit as la

AIME = "Find the sum of all integer bases b greater than 9 for which 17 base b is a divisor of 97 base b."


def _items(tmp_path: Path, text: str = AIME) -> Path:
    path = tmp_path / "items.json"
    path.write_text(
        json.dumps(
            {
                "schema": "eval-items/1",
                "tasks": {t: {"items": [{"id": "1", "fields": {"input": text}}]} for t in la.REQUIRED_TASKS},
            }
        )
    )
    return path


def _docs(n: int, words: int = 120, tag: str = "a") -> list[cd.Document]:
    return [
        cd.Document(tag, f"{tag}{i}", " ".join(f"{tag}w{(i * 7 + j) % 97}" for j in range(words))) for i in range(n)
    ]


def test_byte_tokenizer_is_deterministic_and_has_an_eos() -> None:
    tok = cd.ByteTokenizer()
    assert tok.encode("ab") == [97, 98]
    assert tok.eos_id == 256


def test_pack_windows_concatenates_documents_with_eos_and_drops_the_tail() -> None:
    w = cd.pack_windows([[1, 2, 3], [4, 5], [6, 7, 8]], eos_id=0, window=4)
    assert w.tolist() == [[1, 2, 3, 0], [4, 5, 0, 6]]  # 11 tokens with the separators -> 2 full windows of 4
    assert w.dtype == torch.int64


def test_pack_windows_with_too_few_tokens_raises() -> None:
    with pytest.raises(ValueError):
        cd.pack_windows([[1, 2]], eos_id=0, window=8)


def test_split_wikitext_documents_on_level_one_headings() -> None:
    rows = [" = Alpha = \n", " Some text. \n", " = = Sub = = \n", " More. \n", " = Beta = \n", " Other. \n"]
    docs = cd.split_wikitext(rows)
    assert len(docs) == 2
    assert "Some text." in docs[0] and "More." in docs[0] and "Other." not in docs[0]
    assert docs[1].lstrip().startswith("= Beta =")


def test_long_documents_are_split_on_line_boundaries_without_losing_text() -> None:
    text = "\n".join(f"line {i} " + "x" * 40 for i in range(100))
    docs = cd.split_long([cd.Document("s", "big", text), cd.Document("s", "small", "short")], max_chars=500)
    assert [d.doc_id for d in docs[:2]] == ["big#0", "big#1"]
    assert docs[-1].doc_id == "small"
    assert all(len(d.text) <= 500 for d in docs)
    assert "\n".join(d.text for d in docs if d.doc_id.startswith("big#")) == text


def test_a_single_line_longer_than_the_limit_is_cut_hard() -> None:
    docs = cd.split_long([cd.Document("s", "d", "y" * 1200)], max_chars=500)
    assert [len(d.text) for d in docs] == [500, 500, 200]


def test_text_dir_source_reads_matching_files_and_skips_short_ones(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n" * 200)
    (tmp_path / "b.py").write_text("y")
    (tmp_path / "c.txt").write_text("z" * 5000)
    docs = cd.load_text_dir("python", tmp_path, suffixes=(".py",), min_chars=100)
    assert [d.doc_id for d in docs] == ["a.py"]


def test_selection_is_reproducible_and_respects_the_token_budget_per_source() -> None:
    tok = cd.ByteTokenizer()
    sources = [
        cd.SourceSpec("a", _docs(40, tag="a"), share=0.75),
        cd.SourceSpec("b", _docs(40, tag="b"), share=0.25),
    ]
    sel1 = cd.select_documents(sources, tok, n_tokens=4000, seed=7)
    sel2 = cd.select_documents(sources, tok, n_tokens=4000, seed=7)
    assert [d.doc_id for d in sel1.docs] == [d.doc_id for d in sel2.docs]
    by = {s["name"]: s["n_tokens"] for s in sel1.sources}
    assert by["a"] >= 3000 and by["b"] >= 1000
    assert by["a"] < 3000 + 1500  # at most one document over the share
    other = cd.select_documents(sources, tok, n_tokens=4000, seed=8)
    assert [d.doc_id for d in other.docs] != [d.doc_id for d in sel1.docs]


def test_a_source_with_too_little_text_is_an_error_not_a_silent_shortfall() -> None:
    with pytest.raises(ValueError, match="not enough text"):
        cd.select_documents(
            [cd.SourceSpec("a", _docs(2, tag="a"), share=1.0)], cd.ByteTokenizer(), n_tokens=10**6, seed=0
        )


def test_shares_must_be_positive_and_are_normalised() -> None:
    sel = cd.select_documents(
        [cd.SourceSpec("a", _docs(30, tag="a"), share=3.0), cd.SourceSpec("b", _docs(30, tag="b"), share=1.0)],
        cd.ByteTokenizer(),
        n_tokens=2000,
        seed=0,
    )
    assert sel.sources[0]["share"] == pytest.approx(0.75)
    with pytest.raises(ValueError):
        cd.select_documents([cd.SourceSpec("a", _docs(5), share=0.0)], cd.ByteTokenizer(), n_tokens=10, seed=0)


def test_build_writes_text_bundle_windows_and_a_manifest_with_hashes(tmp_path: Path) -> None:
    out = tmp_path / "calib"
    manifest = cd.build_calibration(
        [cd.SourceSpec("a", _docs(60, tag="a"), share=0.5), cd.SourceSpec("b", _docs(60, tag="b"), share=0.5)],
        tokenizer=cd.ByteTokenizer(),
        out_dir=out,
        n_tokens=4096,
        window=512,
        seed=1,
        eval_items=_items(tmp_path),
    )
    m = json.loads((out / "manifest.json").read_text())
    assert m == json.loads(json.dumps(manifest))
    assert m["window"] == 512 and m["n_windows"] == 8 and m["n_tokens"] == 4096
    assert m["audit"]["ok"] is True and m["audit"]["n"] == 8 and m["audit"]["n_hits"] == 0
    assert m["audit"]["eval_items_sha256"] == hashlib.sha256((tmp_path / "items.json").read_bytes()).hexdigest()
    assert m["windows"]["sha256"] == hashlib.sha256((out / m["windows"]["file"]).read_bytes()).hexdigest()
    assert m["text"]["sha256"] == hashlib.sha256((out / m["text"]["file"]).read_bytes()).hexdigest()
    assert {s["name"] for s in m["sources"]} == {"a", "b"}
    assert all(len(s["sha256"]) == 64 and s["n_docs"] > 0 for s in m["sources"])
    windows = cd.read_windows(out / "manifest.json")
    assert windows.shape == (8, 512)


def test_read_windows_refuses_a_modified_window_file(tmp_path: Path) -> None:
    out = tmp_path / "calib"
    cd.build_calibration(
        [cd.SourceSpec("a", _docs(60), share=1.0)],
        tokenizer=cd.ByteTokenizer(),
        out_dir=out,
        n_tokens=2048,
        window=256,
        seed=0,
        eval_items=_items(tmp_path),
    )
    (out / "windows.json").write_text(json.dumps([[1] * 256] * 8))
    with pytest.raises(cd.ManifestError, match="sha256"):
        cd.read_windows(out / "manifest.json")


def test_read_windows_refuses_a_manifest_without_a_passing_audit(tmp_path: Path) -> None:
    out = tmp_path / "calib"
    cd.build_calibration(
        [cd.SourceSpec("a", _docs(60), share=1.0)],
        tokenizer=cd.ByteTokenizer(),
        out_dir=out,
        n_tokens=2048,
        window=256,
        seed=0,
        eval_items=_items(tmp_path),
    )
    m = json.loads((out / "manifest.json").read_text())
    m["audit"]["ok"] = False
    (out / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(cd.ManifestError, match="audit"):
        cd.read_windows(out / "manifest.json")
    del m["audit"]
    (out / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(cd.ManifestError, match="audit"):
        cd.read_windows(out / "manifest.json")


def _leaky_pool() -> list[cd.Document]:
    pool = _docs(5, words=200, tag="a")
    filler = " ".join(f"fill{j}" for j in range(100))
    pool[3] = cd.Document("a", "leak", filler + " " + AIME + " " + filler)
    return pool


def test_build_refuses_on_any_overlap_with_an_evaluation_item(tmp_path: Path) -> None:
    out = tmp_path / "calib"
    with pytest.raises(la.LeakageError) as err:
        cd.build_calibration(
            [cd.SourceSpec("a", _leaky_pool(), share=1.0)],
            tokenizer=cd.ByteTokenizer(),
            out_dir=out,
            n_tokens=5000,
            window=256,
            seed=0,
            eval_items=_items(tmp_path),
        )
    assert "a/leak" in str(err.value)
    assert not (out / "manifest.json").exists()


def test_drop_flagged_removes_the_documents_and_still_passes_the_audit(tmp_path: Path) -> None:
    pool = _leaky_pool() + _docs(5, words=200, tag="b")
    out = tmp_path / "calib"
    manifest = cd.build_calibration(
        [cd.SourceSpec("a", pool, share=1.0)],
        tokenizer=cd.ByteTokenizer(),
        out_dir=out,
        n_tokens=5000,
        window=256,
        seed=0,
        eval_items=_items(tmp_path),
        drop_flagged=True,
    )
    assert manifest["audit"]["ok"] is True
    assert manifest["audit"]["dropped_docs"] == ["a/leak"]
    assert "base b is a divisor" not in (out / "calib_text.jsonl").read_text()


def test_cli_build_from_text_files_then_audit_standalone(tmp_path: Path) -> None:
    src = tmp_path / "txt"
    src.mkdir()
    for i in range(30):
        (src / f"f{i}.txt").write_text(" ".join(f"tok{(i * 13 + j) % 311}" for j in range(300)))
    out = tmp_path / "calib"
    code = cd.main(
        [
            "build",
            "--out",
            str(out),
            "--tokenizer",
            "byte",
            "--n-tokens",
            "2048",
            "--window",
            "256",
            "--eval-items",
            str(_items(tmp_path)),
            "--source",
            f"textdir:txt:{src}:.txt:1.0",
        ]
    )
    assert code == 0
    assert (out / "manifest.json").is_file()
    assert la.main(["--items", str(tmp_path / "items.json"), "--bundle", str(out)]) == 0


def test_stage_two_tokenize_rebuilds_windows_from_a_text_bundle_and_reaudits(tmp_path: Path) -> None:
    out = tmp_path / "calib"
    cd.build_calibration(
        [cd.SourceSpec("a", _docs(60), share=1.0)],
        tokenizer=cd.ByteTokenizer(),
        out_dir=out,
        n_tokens=2048,
        window=256,
        seed=0,
        eval_items=_items(tmp_path),
        text_only=True,
    )
    m = json.loads((out / "manifest.json").read_text())
    assert m["windows"] is None and m["audit"]["ok"] is True
    with pytest.raises(cd.ManifestError):
        cd.read_windows(out / "manifest.json")
    cd.tokenize_bundle(out, cd.ByteTokenizer(), n_tokens=2048, window=256)
    assert cd.read_windows(out / "manifest.json").shape == (8, 256)
