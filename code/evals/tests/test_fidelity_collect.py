"""Pipeline tests for fidelity.py against the fake OpenAI-compatible server (no GPU)."""

import json

import numpy as np
import pytest
from fake_server import FakeLM, FakeServer, exact_stats
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

import fidelity
from fidelity import (
    RequestFailedError,
    chunk_tokens,
    collect,
    compare_fidelity,
    load_chunks,
    load_fidelity,
    prepare_chunks,
)

# --------------------------------------------------------------------------- chunking


def test_chunk_tokens_cuts_consecutive_non_overlapping_chunks_from_the_start():
    out = chunk_tokens(list(range(10)), n_chunks=3, chunk_len=3)
    assert out.tolist() == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]


def test_chunk_tokens_refuses_a_text_that_is_too_short():
    with pytest.raises(ValueError, match="needs 9 tokens"):
        chunk_tokens(list(range(8)), n_chunks=3, chunk_len=3)


def _tiny_tokenizer(tmp_path):
    words = ["a", "b", "c", "d", "e", "f", "g", "h"]
    vocab = {"[UNK]": 0, **{w: i + 1 for i, w in enumerate(words)}}
    tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))
    return path


def test_prepare_chunks_writes_ids_and_provenance(tmp_path):
    tok = _tiny_tokenizer(tmp_path)
    text = tmp_path / "t.txt"
    text.write_text("a b c d e f g h a b c d", encoding="utf-8")
    out = tmp_path / "chunks.json"
    prepare_chunks(text, tok, out, n_chunks=2, chunk_len=4, name="toy")
    chunks = load_chunks(out)
    assert chunks.token_ids.tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert chunks.meta["name"] == "toy"
    assert chunks.meta["chunk_len"] == 4
    assert chunks.meta["n_chunks"] == 2
    assert len(chunks.meta["text_sha256"]) == 64
    assert len(chunks.meta["tokenizer_sha256"]) == 64
    assert chunks.meta["tokens_available"] == 12
    assert chunks.sha256 == chunks.meta["chunks_sha256"]


def test_chunks_sha256_changes_when_a_token_changes(tmp_path):
    tok = _tiny_tokenizer(tmp_path)
    for name, body in (("x", "a b c d e f g h"), ("y", "a b c d e f g a")):
        (tmp_path / f"{name}.txt").write_text(body, encoding="utf-8")
        prepare_chunks(tmp_path / f"{name}.txt", tok, tmp_path / f"{name}.json", 2, 4, name=name)
    assert load_chunks(tmp_path / "x.json").sha256 != load_chunks(tmp_path / "y.json").sha256


# --------------------------------------------------------------------------- collection


def _write_chunks(tmp_path, vocab_size=16, n_chunks=3, chunk_len=24, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, vocab_size, size=(n_chunks, chunk_len))
    path = tmp_path / "chunks.json"
    path.write_text(
        json.dumps(
            {
                "meta": {"name": "synthetic", "n_chunks": n_chunks, "chunk_len": chunk_len},
                "token_ids": ids.tolist(),
            }
        ),
        encoding="utf-8",
    )
    return path, ids


def _collect(server, chunks_path, out, **kw):
    kw.setdefault("k", 20)
    kw.setdefault("backoff_s", 0.0)
    return collect(chunks_path, server.base_url, "olala", out, **kw)


def test_full_vocabulary_top_k_reproduces_the_exact_metrics(tmp_path):
    chunks_path, ids = _write_chunks(tmp_path, vocab_size=16)
    lm_ref, lm_var = FakeLM(16, seed=0), FakeLM(16, seed=0, noise=0.4)
    with FakeServer(lm_ref) as ref_srv, FakeServer(lm_var) as var_srv:
        ref = _collect(ref_srv, chunks_path, tmp_path / "ref.npz")
        var = _collect(var_srv, chunks_path, tmp_path / "var.npz")
    res = compare_fidelity(ref, var, n_boot=20, seed=0)
    truth = exact_stats(lm_ref, lm_var, ids)
    assert res["k"] == 16  # the 17th entry vLLM adds for the actual token is dropped
    assert res["kld_mean"]["value"] == pytest.approx(truth["kld_mean"], rel=1e-4)
    assert res["top1_agreement"]["value"] == pytest.approx(truth["top1_agreement"])
    assert res["ppl_ref"]["value"] == pytest.approx(truth["ppl_ref"], rel=1e-5)
    assert res["ppl_var"]["value"] == pytest.approx(truth["ppl_var"], rel=1e-5)
    assert res["kld_mean"]["value"] > 0.01  # the variant really differs


def test_truncated_top_k_is_close_to_the_exact_divergence(tmp_path):
    chunks_path, ids = _write_chunks(tmp_path, vocab_size=64, n_chunks=4, chunk_len=64)
    lm_ref, lm_var = FakeLM(64, seed=3), FakeLM(64, seed=3, noise=0.3)
    with FakeServer(lm_ref) as ref_srv, FakeServer(lm_var) as var_srv:
        ref = _collect(ref_srv, chunks_path, tmp_path / "ref.npz", k=20)
        var = _collect(var_srv, chunks_path, tmp_path / "var.npz", k=20)
    res = compare_fidelity(ref, var, n_boot=20, seed=0)
    truth = exact_stats(lm_ref, lm_var, ids)
    assert res["k"] == 20
    # The approximation error is bounded here (64-token vocabulary, k = 20); the real model
    # has a 150k vocabulary, so DECISIONS.md tells users to check sensitivity with --max-k.
    assert res["kld_mean"]["value"] == pytest.approx(truth["kld_mean"], rel=0.15)
    assert res["ppl_ref"]["value"] == pytest.approx(truth["ppl_ref"], rel=1e-5)  # PPL is exact


def test_prompt_logprobs_mode_gives_the_same_arrays_as_echo_mode(tmp_path):
    chunks_path, _ = _write_chunks(tmp_path, vocab_size=32)
    with FakeServer(FakeLM(32, seed=1)) as srv:
        echo = _collect(srv, chunks_path, tmp_path / "e.npz", k=8, mode="echo")
        plp = _collect(srv, chunks_path, tmp_path / "p.npz", k=8, mode="prompt_logprobs")
    assert echo.meta["mode"] == "echo"
    assert plp.meta["mode"] == "prompt_logprobs"
    res = compare_fidelity(echo, plp, n_boot=5, seed=0)
    assert res["kld_mean"]["value"] == pytest.approx(0.0, abs=1e-9)
    assert res["top1_agreement"]["value"] == 1.0
    np.testing.assert_allclose(echo.target_lp, plp.target_lp)


def test_the_request_has_the_documented_shape_and_the_key_is_not_stored(tmp_path):
    chunks_path, ids = _write_chunks(tmp_path)
    with FakeServer(FakeLM(16)) as srv:
        data = _collect(srv, chunks_path, tmp_path / "x.npz", api_key="sekret-value")
    body = srv.requests[0]["body"]
    assert srv.requests[0]["path"] == "/v1/completions"
    assert body["echo"] is True
    assert body["max_tokens"] == 1
    assert body["logprobs"] == 20
    assert body["temperature"] == 0
    assert body["return_tokens_as_token_ids"] is True
    assert body["model"] == "olala"
    assert sorted(map(tuple, (r["body"]["prompt"] for r in srv.requests))) == sorted(
        map(tuple, ids.tolist())
    )
    assert srv.headers[0]["authorization"] == "Bearer sekret-value"
    assert "sekret-value" not in json.dumps(data.meta)
    assert "sekret-value" not in json.dumps(
        {k: str(v) for k, v in load_fidelity(tmp_path / "x.npz").meta.items()}
    )


def test_run_settings_are_recorded_in_the_output_file(tmp_path):
    chunks_path, _ = _write_chunks(tmp_path)
    with FakeServer(FakeLM(16), max_model_len=1234) as srv:
        _collect(srv, chunks_path, tmp_path / "x.npz", k=7)
        endpoint = srv.base_url
    meta = load_fidelity(tmp_path / "x.npz").meta
    assert meta["endpoint"] == endpoint
    assert meta["model"] == "olala"
    assert meta["k"] == 7
    assert meta["mode"] == "echo"
    assert meta["n_chunks"] == 3
    assert meta["chunk_len"] == 24
    assert meta["chunks_name"] == "synthetic"
    assert len(meta["chunks_sha256"]) == 64
    assert meta["request"]["max_tokens"] == 1
    assert meta["server_model_info"]["max_model_len"] == 1234
    assert meta["request_retries"] == 0
    assert meta["started_utc"] <= meta["collected_utc"]
    assert "httpx" in meta["versions"]


def test_transient_server_errors_are_retried_and_counted(tmp_path):
    chunks_path, _ = _write_chunks(tmp_path)
    with FakeServer(FakeLM(16), fail_first=2) as srv:
        data = _collect(srv, chunks_path, tmp_path / "x.npz", retries=3, concurrency=1)
    assert data.meta["request_retries"] == 2


def test_persistent_server_errors_raise_after_the_retry_budget(tmp_path):
    chunks_path, _ = _write_chunks(tmp_path)
    with FakeServer(FakeLM(16), fail_first=100) as srv, pytest.raises(RequestFailedError):
        _collect(srv, chunks_path, tmp_path / "x.npz", retries=1, concurrency=1)


def test_client_errors_are_not_retried_and_carry_the_server_message(tmp_path):
    chunks_path, _ = _write_chunks(tmp_path)
    msg = "logprobs must be <= max_logprobs 5"
    with FakeServer(FakeLM(16), reject_with=(400, msg)) as srv:
        with pytest.raises(RequestFailedError, match="max_logprobs"):
            _collect(srv, chunks_path, tmp_path / "x.npz", retries=5, concurrency=1)
        assert len(srv.requests) == 1


def test_a_server_that_changes_the_prompt_tokens_is_detected(tmp_path):
    chunks_path, _ = _write_chunks(tmp_path)
    with (
        FakeServer(FakeLM(16), wrong_prompt_ids=True) as srv,
        pytest.raises(ValueError, match="prompt tokens"),
    ):
        _collect(srv, chunks_path, tmp_path / "x.npz", retries=0, concurrency=1)


def test_a_server_that_counts_more_prompt_tokens_than_were_sent_is_detected(tmp_path):
    chunks_path, _ = _write_chunks(tmp_path)
    with (
        FakeServer(FakeLM(16), usage_offset=1) as srv,
        pytest.raises(ValueError, match="prompt tokens"),
    ):
        _collect(srv, chunks_path, tmp_path / "x.npz", retries=0, concurrency=1)


def test_the_cli_runs_collect_then_compare_end_to_end(tmp_path, capsys):
    chunks_path, _ = _write_chunks(tmp_path)
    with FakeServer(FakeLM(16)) as ref_srv, FakeServer(FakeLM(16, noise=0.5)) as var_srv:
        for srv, name in ((ref_srv, "ref"), (var_srv, "var")):
            code = fidelity.main(
                [
                    "collect",
                    "--chunks", str(chunks_path),
                    "--base-url", srv.base_url,
                    "--model", "olala",
                    "--out", str(tmp_path / f"{name}.npz"),
                    "--backoff", "0",
                ]
            )  # fmt: skip
            assert code == 0
    code = fidelity.main(
        [
            "compare",
            str(tmp_path / "ref.npz"),
            str(tmp_path / "var.npz"),
            "--n-boot", "20",
            "--json", str(tmp_path / "cmp.json"),
        ]
    )  # fmt: skip
    assert code == 0
    out = capsys.readouterr().out
    assert "Logit fidelity" in out
    saved = json.loads((tmp_path / "cmp.json").read_text())
    assert f"{saved['kld_mean']['value']:.5f}" in out


# --------------------------------------------------------------------------- wikitext


class _FakeRows:
    def __init__(self, texts):
        self._texts = texts

    def __getitem__(self, key):
        assert key == "text"
        return self._texts

    def __len__(self):
        return len(self._texts)


def test_fetch_wikitext_joins_rows_unchanged_and_records_the_dataset_revision(
    tmp_path, monkeypatch
):
    import datasets
    import huggingface_hub

    seen = {}

    def fake_load(repo, config, split, revision):
        seen.update(repo=repo, config=config, split=split, revision=revision)
        return _FakeRows([" = Title = \n", "", " Body text .\n"])

    class FakeApi:
        def dataset_info(self, repo):
            return type("Info", (), {"sha": "deadbeef"})()

    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    out = tmp_path / "data" / "wt.txt"
    record = fidelity.fetch_wikitext(out, "test")
    assert out.read_text(encoding="utf-8") == " = Title = \n Body text .\n"
    assert seen == {
        "repo": "Salesforce/wikitext",
        "config": "wikitext-2-raw-v1",
        "split": "test",
        "revision": "deadbeef",
    }
    assert record["revision"] == "deadbeef"
    assert record["n_rows"] == 3
    saved = json.loads((tmp_path / "data" / "wt.txt.meta.json").read_text())
    assert saved["text_sha256"] == record["text_sha256"]
    assert len(saved["text_sha256"]) == 64


def test_the_prepare_cli_writes_chunks_that_collect_can_read(tmp_path, capsys):
    tok = _tiny_tokenizer(tmp_path)
    (tmp_path / "t.txt").write_text("a b c d e f g h a b c d", encoding="utf-8")
    code = fidelity.main(
        [
            "prepare",
            "--text", str(tmp_path / "t.txt"),
            "--tokenizer", str(tok.parent),
            "--out", str(tmp_path / "c.json"),
            "--n-chunks", "2",
            "--chunk-len", "4",
        ]
    )  # fmt: skip
    assert code == 0
    assert load_chunks(tmp_path / "c.json").token_ids.shape == (2, 4)
    assert "sha256" in capsys.readouterr().out
