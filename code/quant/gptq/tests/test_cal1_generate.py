"""Tests of the CAL-1 generator: prompt draw, resumable generation against a fake local server, windows, audit.

No network beyond 127.0.0.1 fakes and no real model: the server is a stdlib HTTP server, the tokenizer is the byte
tokenizer (and, for one test, a word-level fast tokenizer carrying the REAL Olala chat template of ``recon/``).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import threading
import time
import types
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import cal1_generate as c
import calib_data
import leakage_audit as la

AIME = "Find the sum of all integer bases b greater than 9 for which 17 base b is a divisor of 97 base b."
TEMPLATE = Path(
    os.environ.get(
        "OLALA_CHAT_TEMPLATE", Path(__file__).resolve().parents[3] / "recon" / "src" / "hf" / "chat_template.jinja"
    )
)
SHA = "0123456789abcdef0123456789abcdef01234567"
NAMES = [s.name for s in c.DEFAULT_SOURCES]


def salad(seed: int | str, n: int) -> str:
    """Pseudo-text over a closed lexicon that shares no 8-gram with the item texts."""
    rng = random.Random(str(seed))
    return " ".join(f"lex{rng.randrange(150)}" for _ in range(n))


def write_items(tmp: Path, extra: str = "") -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    path = tmp / "items.json"
    tasks = {
        t: {"items": [{"id": "1", "fields": {"input": AIME if t == "aime2025" else f"{t} {salad(t, 12)}"}}]}
        for t in la.REQUIRED_TASKS
    }
    if extra:
        tasks["mgsm"]["items"].append({"id": "2", "fields": {"input": extra}})
    path.write_text(json.dumps({"schema": "eval-items/1", "tasks": tasks}))
    return path


def write_pool(root: Path, per_source: int = 40, leak_in: str | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for s in c.DEFAULT_SOURCES:
        rows = [
            {
                "id": f"{s.name}-{i}",
                "text": f"Problem {i} of {s.name}: {salad(f'{s.name}{i}', 25)}",
                "lang": "fr" if i % 2 else "de",
            }
            for i in range(per_source)
        ]
        if leak_in == s.name:
            rows.insert(0, {"id": f"{s.name}-leak", "text": f"Please solve this one. {AIME}", "lang": "en"})
        (root / f"{s.name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root


def default_respond(body: dict) -> tuple[int, dict]:
    prompt = body["messages"][-1]["content"]
    return 200, completion(f"I will think about this carefully. {salad(prompt, 40)}", salad(prompt + "a", 30))


def completion(reasoning: str, answer: str, finish: str = "stop") -> dict:
    return {
        "choices": [
            {"finish_reason": finish, "message": {"role": "assistant", "content": answer, "reasoning": reasoning}}
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 70},
    }


class FakeServer:
    """OpenAI-style chat endpoint on 127.0.0.1; records bodies and the maximum number of requests in flight."""

    def __init__(self, respond: Callable[[dict], tuple[int, dict]] = default_respond, delay: float = 0.0) -> None:
        self.respond, self.delay = respond, delay
        self.requests: list[dict] = []
        self.inflight = self.max_inflight = 0
        self.lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def _send(self, status: int, payload: dict) -> None:
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                self._send(200, {"data": [{"id": "fake-model", "root": "/models/olala-bf16"}]})

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with outer.lock:
                    outer.requests.append({"path": self.path, "body": body})
                    outer.inflight += 1
                    outer.max_inflight = max(outer.max_inflight, outer.inflight)
                time.sleep(outer.delay)
                try:
                    status, payload = outer.respond(body)
                finally:
                    with outer.lock:
                        outer.inflight -= 1
                self._send(status, payload)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    made: list[FakeServer] = []

    def make(*args: object, **kwargs: object) -> FakeServer:
        s = FakeServer(*args, **kwargs)  # type: ignore[arg-type]
        made.append(s)
        return s

    yield make
    for s in made:
        s.close()


def argv(
    tmp: Path, srv: FakeServer | None, *extra: str, total: int = 8, pool: Path | None = None, items: Path | None = None
) -> list[str]:
    a = [
        "--out",
        str(tmp / "cal1"),
        "--eval-items",
        str(items or write_items(tmp)),
        "--pool-dir",
        str(pool or write_pool(tmp / "pool")),
        "--total",
        str(total),
        "--seed",
        "3",
        "--tokenizer",
        "byte",
        "--window",
        "256",
        "--concurrency",
        "2",
        "--model",
        "fake-model",
    ]
    for n in NAMES:
        a += ["--revision", f"{n}={SHA}"]
    if srv is not None:
        a += ["--base-url", srv.url, "--server-provenance", "fake vllm 0.0 olala-bf16 sha-test"]
    return a + list(extra)


def index_of(items: Path) -> la.NgramIndex:
    return c.load_index(items)


REVS = {n: SHA for n in NAMES}


def draw(tmp: Path, total: int = 8, seed: int = 3, pool: Path | None = None, items: Path | None = None) -> c.PromptSet:
    quotas = c.compute_quotas(total, c.DEFAULT_WEIGHTS)
    return c.draw_prompts(
        c.DEFAULT_SOURCES,
        c.dir_pool_loader(pool or write_pool(tmp / "pool")),
        quotas,
        seed,
        index_of(items or write_items(tmp)),
        REVS,
    )


# ------------------------------------------------------------------------------------------------ quotas, sources


def test_default_quotas_for_240_follow_the_macro_average_weights() -> None:
    assert c.compute_quotas(240, c.DEFAULT_WEIGHTS) == {
        "math": 48,
        "multilingual_math": 48,
        "code": 96,
        "instruction": 48,
    }


@pytest.mark.parametrize("total", [1, 5, 7, 8, 100, 239, 240, 241])
def test_quotas_always_sum_to_the_total_and_stay_within_one_of_the_exact_share(total: int) -> None:
    q = c.compute_quotas(total, c.DEFAULT_WEIGHTS)
    assert sum(q.values()) == total
    w = sum(c.DEFAULT_WEIGHTS.values())
    assert all(abs(q[d] - total * v / w) < 1 for d, v in c.DEFAULT_WEIGHTS.items())


def test_the_default_sources_cover_every_domain_and_name_their_licence_and_dataset() -> None:
    assert {s.domain for s in c.DEFAULT_SOURCES} == set(c.DEFAULT_WEIGHTS)
    assert len(NAMES) == len(set(NAMES))
    for s in c.DEFAULT_SOURCES:
        assert "/" in s.repo and s.license and s.split == "train"
        assert s.revision == ""  # pinned at run time, never guessed here


# ------------------------------------------------------------------------------------------------ the draw


def test_draw_meets_every_quota_exactly_with_unique_prompts(tmp_path: Path) -> None:
    ps = draw(tmp_path, total=11)
    quotas = c.compute_quotas(11, c.DEFAULT_WEIGHTS)
    for d, q in quotas.items():
        assert sum(1 for p in ps.prompts if p.domain == d) == q
    assert len({p.id for p in ps.prompts}) == len(ps.prompts) == 11
    assert len({p.text for p in ps.prompts}) == 11


def test_draw_is_seeded_and_reproducible(tmp_path: Path) -> None:
    a, b, other = draw(tmp_path), draw(tmp_path), draw(tmp_path, seed=4)
    assert [p.id for p in a.prompts] == [p.id for p in b.prompts]
    # the SELECTION depends on the seed, not only the order
    assert {p.id for p in a.prompts} != {p.id for p in other.prompts}


def test_draw_skips_and_records_a_prompt_that_overlaps_an_evaluation_item(tmp_path: Path) -> None:
    pool = write_pool(tmp_path / "pool", per_source=5, leak_in="math_train")  # 5 clean rows + 1 copy of an item
    quotas = {"math": 5, "multilingual_math": 1, "code": 1, "instruction": 1}  # all 5 clean math rows are needed
    ps = c.draw_prompts(c.DEFAULT_SOURCES, c.dir_pool_loader(pool), quotas, 3, index_of(write_items(tmp_path)), REVS)
    assert not any("17 base b" in p.text for p in ps.prompts)
    assert [d["doc_id"] for d in ps.dropped] == ["math_train-leak"]
    assert ps.dropped[0]["task"] == "aime2025"
    assert sum(1 for p in ps.prompts if p.domain == "math") == 5


def test_draw_with_a_pool_too_small_for_its_quota_raises(tmp_path: Path) -> None:
    pool = write_pool(tmp_path / "pool", per_source=3)
    with pytest.raises(c.Cal1Error, match="math_train"):
        draw(tmp_path, total=40, pool=pool)


def test_draw_takes_a_duplicated_prompt_once(tmp_path: Path) -> None:
    pool = write_pool(tmp_path / "pool", per_source=6)
    rows = [json.loads(line) for line in (pool / "math_train.jsonl").read_text().splitlines()]
    (pool / "math_train.jsonl").write_text(
        "".join(json.dumps(r | {"id": f"dup{i}"}) + "\n" for i in range(5) for r in rows[:1])
        + (pool / "math_train.jsonl").read_text()
    )
    ps = draw(tmp_path, total=10, pool=pool)
    assert len({p.text for p in ps.prompts}) == len(ps.prompts)


def test_draw_requires_a_pinned_revision_for_every_source(tmp_path: Path) -> None:
    quotas = c.compute_quotas(8, c.DEFAULT_WEIGHTS)
    loader = c.dir_pool_loader(write_pool(tmp_path / "pool"))
    index = index_of(write_items(tmp_path))
    for bad in ("", "main", "abc123", SHA[:-1]):
        with pytest.raises(c.Cal1Error, match="math_train"):
            c.draw_prompts(c.DEFAULT_SOURCES, loader, quotas, 0, index, REVS | {"math_train": bad})
    with pytest.raises(c.Cal1Error, match="math_train"):
        c.draw_prompts(c.DEFAULT_SOURCES, loader, quotas, 0, index, {n: SHA for n in NAMES if n != "math_train"})


def test_the_draw_records_sources_with_revision_licence_and_language_counts(tmp_path: Path) -> None:
    ps = draw(tmp_path, total=20)
    by_name = {s["name"]: s for s in ps.sources}
    s = by_name["mgsm_train_parallel"]
    assert s["revision"] == SHA and s["license"] == "apache-2.0" and s["repo"] == "Mathoctopus/GSM8KInstruct_Parallel"
    assert s["n_selected"] == c.compute_quotas(20, c.DEFAULT_WEIGHTS)["multilingual_math"]
    assert sum(s["languages"].values()) == s["n_selected"]
    assert s["pool_size"] == 40


def test_load_index_refuses_items_without_one_of_the_five_tasks(tmp_path: Path) -> None:
    items = write_items(tmp_path)
    data = json.loads(items.read_text())
    del data["tasks"]["ifeval"]
    items.write_text(json.dumps(data))
    with pytest.raises(c.Cal1Error, match="ifeval"):
        c.load_index(items)


ALPACA = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request in "
    "{lang}. Please answer in {lang}.\n\n### Instruction:\n{q}\n\n### Response:"
)


def test_the_multilingual_math_source_reads_the_prompt_column_and_extracts_the_question() -> None:
    """2026-11-20: the dataset has columns chosen, prompt, reject (no `query`); `prompt` is an Alpaca wrapper."""
    src = next(s for s in c.DEFAULT_SOURCES if s.name == "mgsm_train_parallel")
    assert src.column == "prompt" and src.extract == "alpaca_instruction"
    recs = [
        {
            "prompt": ALPACA.format(lang="German", q=" Natalia hat im April Clips verkauft. Wie viele? "),
            "chosen": "x",
            "reject": "y",
        },
        {"prompt": ALPACA.format(lang="English", q="Natalia sold clips. How many?"), "chosen": "x", "reject": "y"},
        {"prompt": "no template markers in this row", "chosen": "x", "reject": "y"},
        {"prompt": "### Instruction:\n   \n\n### Response:", "chosen": "x", "reject": "y"},
    ]
    rows = c.rows_from_records(src, recs, "default")
    assert [r.text for r in rows] == [
        "Natalia hat im April Clips verkauft. Wie viele?",
        "Natalia sold clips. How many?",
    ]
    assert [r.doc_id for r in rows] == ["default-0", "default-1"]  # the original row numbers are kept


def test_a_source_without_extract_keeps_its_text_unchanged() -> None:
    src = c.SourceDef("s", "math", "x/y", "mit", column="problem")
    rows = c.rows_from_records(src, [{"problem": "  keep me as is\n### Instruction:\nnot a template here  "}], "cfg")
    assert rows[0].text == "  keep me as is\n### Instruction:\nnot a template here  "


def test_an_unknown_extract_name_is_refused() -> None:
    src = c.SourceDef("s", "math", "x/y", "mit", column="problem", extract="nope")
    with pytest.raises(c.Cal1Error, match="unknown extract"):
        c.rows_from_records(src, [{"problem": "t"}], "cfg")


def test_rows_from_records_applies_filters_and_names_a_missing_column() -> None:
    math = next(s for s in c.DEFAULT_SOURCES if s.domain == "math")
    rows = c.rows_from_records(
        math, [{"problem": "p1 " * 20, "level": "Level 5"}, {"problem": "p2 " * 20, "level": "Level 1"}], "algebra"
    )
    assert [r.text.split()[0] for r in rows] == ["p1"]
    with pytest.raises(c.Cal1Error, match="columns"):
        c.rows_from_records(math, [{"question": "x"}], "algebra")
    code = next(s for s in c.DEFAULT_SOURCES if s.domain == "code")
    rows = c.rows_from_records(
        code, [{"problem": "a " * 30, "lang": "python"}, {"problem": "b " * 30, "lang": "rust"}], "default"
    )
    assert len(rows) == 1


def test_the_hf_loader_pins_the_revision_and_reads_every_config(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple] = []

    def load_dataset(repo: str, config: str | None = None, **kw: object) -> list[dict]:
        calls.append((repo, config, kw))
        return [{"problem": f"{config} " + "w " * 30, "level": "Level 4"}]

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
    math = next(s for s in c.DEFAULT_SOURCES if s.domain == "math")
    rows = c.hf_pool_loader(REVS)(math)
    assert len(calls) == len(math.configs) == len(rows) > 1
    assert all(kw["revision"] == SHA and kw["split"] == "train" and repo == math.repo for repo, _, kw in calls)
    assert len({r.doc_id for r in rows}) == len(rows)


# ------------------------------------------------------------------------------------------------ prompt manifest


def test_prompt_manifest_roundtrip_and_declared_content(tmp_path: Path) -> None:
    ps = draw(tmp_path)
    out = tmp_path / "out"
    quotas = c.compute_quotas(8, c.DEFAULT_WEIGHTS)
    c.write_prompt_set(out, ps, seed=3, weights=c.DEFAULT_WEIGHTS, quotas=quotas, eval_items_sha256="x" * 64)
    prompts, m = c.load_prompt_set(out)
    assert [p.id for p in prompts] == [p.id for p in ps.prompts]
    assert m["quotas"] == quotas and m["weights"] == c.DEFAULT_WEIGHTS and m["seed"] == 3
    assert m["prompts"]["sha256"] == hashlib.sha256((out / c.PROMPTS_FILE).read_bytes()).hexdigest()
    assert {s["name"] for s in m["sources"]} == set(NAMES)


def test_a_modified_prompt_file_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "out"
    c.write_prompt_set(
        out,
        draw(tmp_path),
        seed=3,
        weights=c.DEFAULT_WEIGHTS,
        quotas=c.compute_quotas(8, c.DEFAULT_WEIGHTS),
        eval_items_sha256="x",
    )
    path = out / c.PROMPTS_FILE
    path.write_text(path.read_text().replace("Problem", "Probleme", 1))
    with pytest.raises(c.Cal1Error, match="sha256"):
        c.load_prompt_set(out)


def test_a_prompt_file_that_no_longer_meets_the_declared_quotas_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "out"
    c.write_prompt_set(
        out,
        draw(tmp_path),
        seed=3,
        weights=c.DEFAULT_WEIGHTS,
        quotas=c.compute_quotas(8, c.DEFAULT_WEIGHTS),
        eval_items_sha256="x",
    )
    path = out / c.PROMPTS_FILE
    path.write_text(
        "".join(path.read_text().splitlines(keepends=True)[1:])
    )  # drop a row, keep the manifest hash honest
    m = json.loads((out / c.PROMPTS_MANIFEST).read_text())
    m["prompts"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / c.PROMPTS_MANIFEST).write_text(json.dumps(m))
    with pytest.raises(c.Cal1Error, match="quota"):
        c.load_prompt_set(out)


# ------------------------------------------------------------------------------------------------ requests


def test_the_request_carries_the_harness_sampling_and_chat_template_kwargs() -> None:
    body = c.request_body("fake-model", "Compute 2+2.", c.Sampling())
    assert body["model"] == "fake-model"
    assert body["messages"] == [{"role": "user", "content": "Compute 2+2."}]
    assert (body["temperature"], body["top_p"], body["top_k"], body["max_tokens"]) == (0.8, 0.9, 40, 8192)
    assert body["chat_template_kwargs"] == {"reasoning_effort": "medium"}
    assert not body.get("stream")


def test_parse_completion_reads_both_reasoning_field_names_and_flags_truncation() -> None:
    ok = c.parse_completion(completion("think", "ans"))
    assert (ok["reasoning"], ok["answer"], ok["truncated"], ok["finish_reason"]) == ("think", "ans", False, "stop")
    old = completion("", "ans")
    old["choices"][0]["message"].pop("reasoning")
    old["choices"][0]["message"]["reasoning_content"] = "older field"
    assert c.parse_completion(old)["reasoning"] == "older field"
    cut = c.parse_completion(completion("long thought", "", finish="length"))
    assert cut["truncated"] is True and cut["answer"] == ""
    assert c.parse_completion(completion("t", "a", finish="stop"))["truncated"] is False


@pytest.mark.parametrize("bad", [{}, {"choices": []}, completion("", ""), completion("", "", finish="length")])
def test_parse_completion_refuses_a_response_without_any_text(bad: dict) -> None:
    with pytest.raises(c.Cal1Error):
        c.parse_completion(bad)


# ------------------------------------------------------------------------------------------------ generation store


def prompts_of(n: int) -> list[c.Prompt]:
    return [c.Prompt(f"p{i}", "math", "s", f"d{i}", f"question {i} {salad(i, 10)}", "") for i in range(n)]


def good_call(p: c.Prompt) -> dict:
    return completion(f"thinking about {p.id}", f"answer to {p.id}")


def test_store_refuses_a_second_ok_row_for_the_same_prompt(tmp_path: Path) -> None:
    store = c.GenStore(tmp_path / "g.jsonl")
    row = {"id": "p0", "status": "ok", "attempt": 1}
    store.append(row)
    with pytest.raises(c.Cal1Error, match="p0"):
        store.append(row)
    assert len((tmp_path / "g.jsonl").read_text().splitlines()) == 1


def test_store_reloads_rows_and_repairs_a_line_cut_by_an_interruption(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    path.write_text(json.dumps({"id": "p0", "status": "ok", "attempt": 1}) + "\n" + '{"id": "p1", "status": "o')
    store = c.GenStore(path)
    assert store.ok_ids == {"p0"}
    store.append({"id": "p1", "status": "ok", "attempt": 1})
    assert [json.loads(line)["id"] for line in path.read_text().splitlines()] == ["p0", "p1"]


def test_store_refuses_a_corrupt_line_in_the_middle(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    path.write_text('{"id": "p0", "status": "ok"}\nnot json\n{"id": "p2", "status": "ok"}\n')
    with pytest.raises(c.Cal1Error, match="line 2"):
        c.GenStore(path)


def run_generate(tmp: Path, prompts: list[c.Prompt], call: Callable[[c.Prompt], dict], **kw: object) -> dict:
    store = c.GenStore(tmp / "gen.jsonl")
    kwargs = {"concurrency": 2, "max_attempts": 3, "config_sha": "cfg"} | kw
    return c.generate(prompts, store, call, **kwargs)  # type: ignore[arg-type]


def rows_of(tmp: Path) -> list[dict]:
    return [json.loads(line) for line in (tmp / "gen.jsonl").read_text().splitlines()]


def test_generate_writes_one_ok_row_per_prompt_with_its_provenance(tmp_path: Path) -> None:
    ps = prompts_of(5)
    stats = run_generate(tmp_path, ps, good_call)
    rows = rows_of(tmp_path)
    assert stats["ok"] == 5 and stats["failed"] == 0
    assert sorted(r["id"] for r in rows) == sorted(p.id for p in ps)
    r = next(r for r in rows if r["id"] == "p3")
    assert r["status"] == "ok" and r["reasoning"] == "thinking about p3" and r["answer"] == "answer to p3"
    assert (
        r["truncated"] is False
        and r["config_sha256"] == "cfg"
        and r["prompt_sha256"] == hashlib.sha256(ps[3].text.encode()).hexdigest()
    )


def test_a_truncated_trace_is_kept_and_flagged(tmp_path: Path) -> None:
    def call(p: c.Prompt) -> dict:
        return completion("a very long thought that never ends", "", finish="length") if p.id == "p1" else good_call(p)

    run_generate(tmp_path, prompts_of(3), call)
    r = next(r for r in rows_of(tmp_path) if r["id"] == "p1")
    assert r["status"] == "ok" and r["truncated"] is True and r["reasoning"].startswith("a very long")
    assert sum(1 for r in rows_of(tmp_path) if r["truncated"]) == 1


def test_a_failure_is_recorded_not_dropped_and_retried_on_resume_up_to_max_attempts(tmp_path: Path) -> None:
    attempts: list[str] = []

    def flaky(p: c.Prompt) -> dict:
        attempts.append(p.id)
        if p.id == "p2":
            raise c.Cal1Error("HTTP 500")
        return good_call(p)

    ps = prompts_of(4)
    s1 = run_generate(tmp_path, ps, flaky, max_attempts=2)
    assert s1["ok"] == 3 and s1["failed"] == 1
    failed = [r for r in rows_of(tmp_path) if r["status"] == "failed"]
    assert (
        len(failed) == 1 and failed[0]["id"] == "p2" and "HTTP 500" in failed[0]["error"] and failed[0]["attempt"] == 1
    )
    s2 = run_generate(tmp_path, ps, flaky, max_attempts=2)  # resume: only p2 is tried again
    assert attempts.count("p2") == 2 and attempts.count("p0") == 1
    assert s2["failed"] == 1
    s3 = run_generate(tmp_path, ps, flaky, max_attempts=2)  # attempts exhausted: not asked again, still on record
    assert attempts.count("p2") == 2 and s3["gave_up"] == 1
    assert sum(1 for r in rows_of(tmp_path) if r["id"] == "p2") == 2


def test_resume_never_asks_for_a_prompt_that_is_done_and_never_duplicates_it(tmp_path: Path) -> None:
    ps = prompts_of(6)
    run_generate(tmp_path, ps[:3], good_call)  # the interrupted run finished three
    asked: list[str] = []

    def call(p: c.Prompt) -> dict:
        asked.append(p.id)
        return good_call(p)

    stats = run_generate(tmp_path, ps, call)
    assert sorted(asked) == ["p3", "p4", "p5"] and stats["skipped_done"] == 3
    ids = [r["id"] for r in rows_of(tmp_path) if r["status"] == "ok"]
    assert sorted(ids) == sorted(p.id for p in ps) and len(ids) == len(set(ids))


def test_resume_refuses_rows_of_a_different_sampling_configuration(tmp_path: Path) -> None:
    run_generate(tmp_path, prompts_of(2), good_call, config_sha="cfg-A")
    with pytest.raises(c.Cal1Error, match="configuration"):
        run_generate(tmp_path, prompts_of(4), good_call, config_sha="cfg-B")


def test_resume_refuses_a_prompt_whose_text_changed_and_rows_of_unknown_prompts(tmp_path: Path) -> None:
    ps = prompts_of(3)
    run_generate(tmp_path, ps, good_call)
    changed = [c.Prompt(p.id, p.domain, p.source, p.doc_id, p.text + " edited", p.lang) for p in ps]
    with pytest.raises(c.Cal1Error, match="p0"):
        run_generate(tmp_path, changed, good_call)
    with pytest.raises(c.Cal1Error, match="not in the prompt manifest"):
        run_generate(tmp_path, ps[:2], good_call)


def test_concurrency_is_bounded_and_validated(tmp_path: Path) -> None:
    live = {"now": 0, "max": 0}
    lock = threading.Lock()

    def call(p: c.Prompt) -> dict:
        with lock:
            live["now"] += 1
            live["max"] = max(live["max"], live["now"])
        time.sleep(0.03)
        with lock:
            live["now"] -= 1
        return good_call(p)

    run_generate(tmp_path, prompts_of(12), call, concurrency=3)
    assert 2 <= live["max"] <= 3
    for bad in (0, -1, c.MAX_CONCURRENCY + 1):
        with pytest.raises(c.Cal1Error, match="concurrency"):
            run_generate(tmp_path / f"x{bad}", prompts_of(1), good_call, concurrency=bad)


# ------------------------------------------------------------------------------------------------ HTTP client


def test_http_call_posts_the_harness_request_to_the_chat_endpoint(server: Callable[..., FakeServer]) -> None:
    srv = server()
    call = c.http_call(srv.url, "fake-model", c.Sampling(), api_key="EMPTY", timeout=10)
    resp = call(c.Prompt("p0", "math", "s", "d", "What is 6 times 7?", ""))
    assert resp["choices"][0]["finish_reason"] == "stop"
    (req,) = srv.requests
    assert req["path"] == "/v1/chat/completions"
    assert req["body"]["top_k"] == 40 and req["body"]["chat_template_kwargs"] == {"reasoning_effort": "medium"}
    assert req["body"]["messages"][0]["content"] == "What is 6 times 7?"


def test_http_call_turns_an_http_error_and_a_dead_server_into_cal1_error(server: Callable[..., FakeServer]) -> None:
    srv = server(lambda body: (500, {"error": "boom"}))
    call = c.http_call(srv.url, "m", c.Sampling(), api_key="EMPTY", timeout=5)
    with pytest.raises(c.Cal1Error, match="500"):
        call(prompts_of(1)[0])
    dead = c.http_call("http://127.0.0.1:9/v1", "m", c.Sampling(), api_key="EMPTY", timeout=2)
    with pytest.raises(c.Cal1Error):
        dead(prompts_of(1)[0])


# ------------------------------------------------------------------------------------------------ renderers


def test_byte_renderer_follows_the_olala_channel_layout() -> None:
    r = c.make_renderer("byte")
    text = r.render_text("Q?", "think", "A.", "medium")
    order = [
        text.index(s)
        for s in (
            "Reasoning effort: medium",
            "<|im_start|>user",
            "Q?",
            "<|channel_start|>analysis<|content|>think",
            "<|channel_start|>final<|content|>A.",
        )
    ]
    assert order == sorted(order)
    assert r.render_text("Q?", "t", "A", "high").count("Reasoning effort: high") == 1


@pytest.mark.skipif(not TEMPLATE.is_file(), reason="recon/ (the real chat template) is not in this checkout")
def test_hf_renderer_uses_the_real_olala_template_with_reasoning_and_effort() -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tk = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]")
    tok.chat_template = TEMPLATE.read_text()
    r = c.HFChatRenderer(tok, eos_id=0, name="hf:test")
    text = r.render_text("What is 2+2?", "add them", "4", "medium")
    assert "Reasoning effort: medium" in text
    assert text.endswith(
        "<|im_start|>assistant<|channel_start|>analysis<|content|>add them<|channel_end|>"
        "<|channel_start|>final<|content|>4<|channel_end|><|im_end|>"
    )
    gen_prompt = tok.apply_chat_template(
        [{"role": "user", "content": "What is 2+2?"}],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort="medium",
    )
    assert text.startswith(gen_prompt)  # what the server saw is exactly the prefix of what is calibrated on
    assert "Reasoning effort: high" in r.render_text("q", "t", "a", "high")
    assert isinstance(r.encode("q", "t", "a", "medium"), list)


# ------------------------------------------------------------------------------------------------ end to end (CLI)


def run_ok(tmp: Path, srv: FakeServer, *extra: str, **kw: object) -> dict:
    assert c.main(argv(tmp, srv, *extra, **kw)) == 0  # type: ignore[arg-type]
    return json.loads((tmp / "cal1" / "manifest.json").read_text())


def test_end_to_end_manifest_is_accepted_by_calib_data_read_windows(
    tmp_path: Path, server: Callable[..., FakeServer]
) -> None:
    srv = server()
    m = run_ok(tmp_path, srv)
    windows = calib_data.read_windows(tmp_path / "cal1" / "manifest.json")
    assert windows.shape == (m["n_windows"], 256) and m["n_windows"] > 3
    assert m["schema"] == "cal1/1" and m["audit"]["ok"] is True and m["audit"]["n_hits"] == 0 and m["audit"]["n"] == 8
    assert m["audit"]["n_scanned_prompts"] == 8 and m["audit"]["n_scanned_generations"] == 8
    assert len(srv.requests) == 8


def test_manifest_declares_sampling_server_sources_counts_and_a_hash_of_every_file(
    tmp_path: Path, server: Callable[..., FakeServer]
) -> None:
    m = run_ok(tmp_path, server())
    out = tmp_path / "cal1"
    assert m["sampling"] == {
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 40,
        "reasoning_effort": "medium",
        "max_tokens": 8192,
    }
    assert m["server"]["provenance"] == "fake vllm 0.0 olala-bf16 sha-test" and m["server"]["models"] == ["fake-model"]
    assert (
        m["domains"]["code"]["quota"] == 3
        and m["domains"]["math"]["quota"] == 2
        and m["domains"]["instruction"]["quota"] == 1
    )
    assert sum(d["n_ok"] for d in m["domains"].values()) == 8 and all(d["weight"] > 0 for d in m["domains"].values())
    assert {s["name"]: s["revision"] for s in m["prompt_sources"]} == REVS
    for name in (
        "prompts.jsonl",
        "prompts_manifest.json",
        "generations.jsonl",
        "calib_text.jsonl",
        "eval_items.json",
        "windows.json",
    ):
        assert m["files"][name] == hashlib.sha256((out / name).read_bytes()).hexdigest()
    assert m["tokenizer"]["name"] == "byte" and m["chat_template"]["reasoning_effort"] == "medium"


def test_truncated_traces_are_counted_in_the_manifest_and_stay_in_the_windows(
    tmp_path: Path, server: Callable[..., FakeServer]
) -> None:
    n = {"i": 0}

    def respond(body: dict) -> tuple[int, dict]:
        n["i"] += 1
        status, payload = default_respond(body)
        if n["i"] % 4 == 0:
            payload["choices"][0]["finish_reason"] = "length"
            payload["choices"][0]["message"]["content"] = ""
        return status, payload

    m = run_ok(tmp_path, server(respond))
    assert m["counts"]["n_truncated"] == 2 and m["counts"]["n_traces"] == 8
    assert sum(d["n_truncated"] for d in m["domains"].values()) == 2


def test_a_failed_prompt_is_in_the_manifest_not_hidden(tmp_path: Path, server: Callable[..., FakeServer]) -> None:
    n = {"i": 0}

    def respond(body: dict) -> tuple[int, dict]:
        n["i"] += 1
        return (500, {"error": "boom"}) if n["i"] == 2 else default_respond(body)

    m = run_ok(tmp_path, server(respond), "--max-attempts", "1", total=12)
    assert m["counts"]["n_failed"] == 1 and m["counts"]["n_ok"] == 11 and m["counts"]["n_traces"] == 11
    assert sum(d["n_failed"] for d in m["domains"].values()) == 1
    assert len(m["failures"]) == 1 and "500" in m["failures"][0]["error"] and m["failures"][0]["attempts"] == 1


def test_a_domain_with_no_successful_trace_is_refused(tmp_path: Path, server: Callable[..., FakeServer]) -> None:
    srv = server(lambda body: (500, {}) if "if_personas" in body["messages"][0]["content"] else default_respond(body))
    assert c.main(argv(tmp_path, srv, "--max-attempts", "1")) == 2
    assert not (tmp_path / "cal1" / "manifest.json").exists()


def test_rerunning_a_finished_run_asks_the_server_for_nothing(
    tmp_path: Path, server: Callable[..., FakeServer]
) -> None:
    srv = server()
    run_ok(tmp_path, srv)
    first = len(srv.requests)
    run_ok(tmp_path, srv)
    assert len(srv.requests) == first
    ids = [json.loads(line)["id"] for line in (tmp_path / "cal1" / "generations.jsonl").read_text().splitlines()]
    assert len(ids) == len(set(ids)) == 8


def test_resume_after_an_interruption_generates_only_the_missing_prompts(
    tmp_path: Path, server: Callable[..., FakeServer]
) -> None:
    srv = server()
    run_ok(tmp_path, srv)
    gen = tmp_path / "cal1" / "generations.jsonl"
    lines = gen.read_text().splitlines()
    gen.write_text("\n".join(lines[:5]) + "\n" + lines[5][:20])  # killed in the middle of a write
    (tmp_path / "cal1" / "manifest.json").unlink()
    before = len(srv.requests)
    run_ok(tmp_path, srv)
    assert len(srv.requests) - before == 3
    ids = [json.loads(line)["id"] for line in gen.read_text().splitlines()]
    assert len(ids) == len(set(ids)) == 8


def test_requests_use_the_declared_sampling_and_the_prompts_of_the_manifest(
    tmp_path: Path, server: Callable[..., FakeServer]
) -> None:
    srv = server()
    run_ok(tmp_path, srv)
    prompts, _ = c.load_prompt_set(tmp_path / "cal1")
    assert sorted(r["body"]["messages"][0]["content"] for r in srv.requests) == sorted(p.text for p in prompts)
    for r in srv.requests:
        b = r["body"]
        assert (b["temperature"], b["top_p"], b["top_k"], b["max_tokens"]) == (0.8, 0.9, 40, 8192)
        assert b["chat_template_kwargs"] == {"reasoning_effort": "medium"} and b["model"] == "fake-model"


def test_concurrency_of_the_cli_is_bounded(tmp_path: Path, server: Callable[..., FakeServer]) -> None:
    srv = server(delay=0.05)
    run_ok(tmp_path, srv, total=10)
    assert 1 <= srv.max_inflight <= 2


def test_changing_the_sampling_server_or_model_on_resume_is_refused(
    tmp_path: Path, server: Callable[..., FakeServer]
) -> None:
    srv = server()
    run_ok(tmp_path, srv)
    assert c.main(argv(tmp_path, srv, "--max-tokens", "4096")) == 2
    assert c.main(argv(tmp_path, srv, "--temperature", "0.5")) == 2
    assert c.main(argv(tmp_path, srv, "--reasoning-effort", "high")) == 2
    other = argv(tmp_path, srv)
    other[other.index("--server-provenance") + 1] = "another server"
    assert c.main(other) == 2
    other = argv(tmp_path, srv)
    other[other.index("--model") + 1] = "another-model"
    assert c.main(other) == 2


def test_a_real_run_needs_a_provenance_string_and_pinned_revisions_and_asks_the_server_nothing_first(
    tmp_path: Path, server: Callable[..., FakeServer]
) -> None:
    srv = server()
    a = argv(tmp_path / "a", srv)
    i = a.index("--server-provenance")
    assert c.main(a[:i] + a[i + 2 :]) == 2
    b = argv(tmp_path / "b", srv)
    j = b.index("--revision")
    assert c.main(b[:j] + b[j + 2 :]) == 2
    assert srv.requests == []


# ------------------------------------------------------------------------------------------------ the audit


def leaking_server(server: Callable[..., FakeServer], text: str, once: bool = False) -> FakeServer:
    seen = {"n": 0}

    def respond(body: dict) -> tuple[int, dict]:
        seen["n"] += 1
        if not once or seen["n"] == 3:
            return 200, completion(f"Hmm, I recall {text} and so on.", "done")
        return default_respond(body)

    return server(respond)


def test_a_generated_trace_that_repeats_an_evaluation_item_is_refused_and_nothing_is_marked_passing(
    tmp_path: Path, server: Callable[..., FakeServer]
) -> None:
    srv = leaking_server(server, AIME, once=True)
    assert c.main(argv(tmp_path, srv)) == 1
    out = tmp_path / "cal1"
    assert not (out / "manifest.json").exists() and not (out / "windows.json").exists()


def test_a_single_overlapping_8_gram_in_the_reasoning_is_enough_to_refuse(
    tmp_path: Path, server: Callable[..., FakeServer]
) -> None:
    eight = " ".join(AIME.split()[:8])
    srv = leaking_server(server, eight, once=True)
    assert c.main(argv(tmp_path, srv)) == 1


def test_overlap_in_the_final_answer_alone_is_refused(tmp_path: Path, server: Callable[..., FakeServer]) -> None:
    srv = server(lambda body: (200, completion(salad(body["messages"][0]["content"], 20), f"So: {AIME}")))
    assert c.main(argv(tmp_path, srv)) == 1


def test_seven_words_of_an_item_are_not_an_8_gram_overlap(tmp_path: Path, server: Callable[..., FakeServer]) -> None:
    srv = leaking_server(server, " ".join(AIME.split()[:7]))
    assert c.main(argv(tmp_path, srv)) == 0


def test_the_audit_covers_the_prompts_of_the_manifest_too(tmp_path: Path, server: Callable[..., FakeServer]) -> None:
    items = write_items(tmp_path)
    ps = draw(
        tmp_path
    )  # a clean draw, then one prompt is replaced by a copy of an item: only the final audit can see it
    leaky = c.Prompt("math-leak", "math", "math_train", "leak", f"Please solve: {AIME}", "")
    ps.prompts[0] = c.Prompt(leaky.id, ps.prompts[0].domain, ps.prompts[0].source, "leak", leaky.text, "")
    out = tmp_path / "cal1"
    c.write_prompt_set(
        out, ps, seed=3, weights=c.DEFAULT_WEIGHTS, quotas=c.compute_quotas(8, c.DEFAULT_WEIGHTS), eval_items_sha256="x"
    )
    assert c.main(argv(tmp_path, server(), items=items)) == 1
    assert not (out / "manifest.json").exists()


def test_a_refusal_removes_the_manifest_and_windows_of_an_earlier_passing_run(
    tmp_path: Path, server: Callable[..., FakeServer]
) -> None:
    srv = server()
    run_ok(tmp_path, srv)
    out = tmp_path / "cal1"
    assert (out / "manifest.json").exists()
    gen = out / "generations.jsonl"
    rows = [json.loads(line) for line in gen.read_text().splitlines()]
    rows[0]["reasoning"] += f" {AIME}"
    gen.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert c.main(argv(tmp_path, srv)) == 1
    assert not (out / "manifest.json").exists() and not (out / "windows.json").exists()


def test_drop_flagged_drops_the_leaking_trace_records_it_and_audits_the_rest(
    tmp_path: Path, server: Callable[..., FakeServer]
) -> None:
    srv = leaking_server(server, AIME, once=True)
    assert c.main(argv(tmp_path, srv)) == 1
    m = run_ok(tmp_path, srv, "--drop-flagged")
    assert m["audit"]["ok"] is True and m["audit"]["n_hits"] == 0 and len(m["audit"]["dropped_traces"]) == 1
    assert m["counts"]["n_traces"] == 7 and m["counts"]["n_dropped_leak"] == 1
    assert sum(d["n_dropped_leak"] for d in m["domains"].values()) == 1
    assert calib_data.read_windows(tmp_path / "cal1" / "manifest.json").shape[0] == m["n_windows"]


def test_the_audit_bundle_is_readable_by_the_stock_audit_cli(tmp_path: Path, server: Callable[..., FakeServer]) -> None:
    run_ok(tmp_path, server())
    out = tmp_path / "cal1"
    assert la.main(["--items", str(out / "eval_items.json"), "--bundle", str(out)]) == 0
    sources = {json.loads(line)["source"] for line in (out / "calib_text.jsonl").read_text().splitlines()}
    assert sources == {"prompt", "generation"}


# ------------------------------------------------------------------------------------------------ verification


@pytest.mark.parametrize(
    "name",
    [
        "prompts.jsonl",
        "prompts_manifest.json",
        "generations.jsonl",
        "calib_text.jsonl",
        "eval_items.json",
        "windows.json",
    ],
)
def test_verify_refuses_any_modified_file(tmp_path: Path, server: Callable[..., FakeServer], name: str) -> None:
    run_ok(tmp_path, server())
    out = tmp_path / "cal1"
    c.verify_bundle(out)  # intact: passes
    path = out / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(c.Cal1Error, match=name):
        c.verify_bundle(out)


def test_verify_reruns_the_audit_and_refuses_a_bundle_that_now_overlaps(
    tmp_path: Path, server: Callable[..., FakeServer]
) -> None:
    run_ok(tmp_path, server())
    out = tmp_path / "cal1"
    items = json.loads((out / "eval_items.json").read_text())
    items["tasks"]["mgsm"]["items"].append(
        {
            "id": "x",
            "fields": {
                "input": " ".join(
                    json.loads((out / "calib_text.jsonl").read_text().splitlines()[0])["text"].split()[:12]
                )
            },
        }
    )
    (out / "eval_items.json").write_text(json.dumps(items))
    m = json.loads((out / "manifest.json").read_text())
    m["files"]["eval_items.json"] = hashlib.sha256((out / "eval_items.json").read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(la.LeakageError):
        c.verify_bundle(out)


def test_cli_verify_flag(tmp_path: Path, server: Callable[..., FakeServer]) -> None:
    run_ok(tmp_path, server())
    out = tmp_path / "cal1"
    assert c.main(["--out", str(out), "--verify"]) == 0
    (out / "generations.jsonl").write_bytes((out / "generations.jsonl").read_bytes() + b"\n")
    assert c.main(["--out", str(out), "--verify"]) == 2


# ------------------------------------------------------------------------------------------------ dry run


def test_dry_run_prints_the_plan_without_a_server_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = argv(tmp_path, None, "--dry-run", total=240, pool=write_pool(tmp_path / "pool", per_source=120))
    assert c.main(a) == 0
    out = capsys.readouterr().out
    assert not (tmp_path / "cal1").exists()
    for needle in (
        "math: 48",
        "multilingual_math: 48",
        "code: 96",
        "instruction: 48",
        "240 prompts",
        "temperature 0.8",
        "top_k 40",
        "max_tokens 8192",
        "reasoning_effort medium",
        SHA,
        "Mathoctopus/GSM8KInstruct_Parallel",
        "window 256",
    ):
        assert needle in out, needle


def test_dry_run_reports_progress_of_an_existing_run(
    tmp_path: Path, server: Callable[..., FakeServer], capsys: pytest.CaptureFixture[str]
) -> None:
    run_ok(tmp_path, server())
    gen = tmp_path / "cal1" / "generations.jsonl"
    gen.write_text("".join(gen.read_text().splitlines(keepends=True)[:3]))
    capsys.readouterr()
    assert c.main(argv(tmp_path, None, "--dry-run")) == 0
    out = capsys.readouterr().out
    assert "3 done" in out and "5 to generate" in out


def test_dry_run_refuses_unpinned_revisions_like_a_real_run(tmp_path: Path) -> None:
    a = argv(tmp_path, None, "--dry-run")
    j = a.index("--revision")
    assert c.main(a[:j] + a[j + 2 :]) == 2


def test_dry_run_of_a_draw_that_cannot_meet_its_quota_fails(tmp_path: Path) -> None:
    assert c.main(argv(tmp_path, None, "--dry-run", total=240, pool=write_pool(tmp_path / "pool", per_source=5))) == 2


# ------------------------------------------------------------------------------------------------ options


def test_resuming_into_prompts_drawn_with_other_quotas_seed_or_revisions_is_refused(
    tmp_path: Path, server: Callable[..., FakeServer]
) -> None:
    srv = server()
    run_ok(tmp_path, srv)
    assert c.main(argv(tmp_path, srv, total=12)) == 2
    assert c.main(argv(tmp_path, srv, "--seed", "9")) == 2
    other = argv(tmp_path, srv)
    other[other.index(f"math_train={SHA}")] = "math_train=" + "f" * 40
    assert c.main(other) == 2
    assert len(srv.requests) == 8  # nothing was generated for any of them


def test_a_domain_quota_can_be_overridden_and_an_unknown_domain_is_refused(
    tmp_path: Path, server: Callable[..., FakeServer]
) -> None:
    m = run_ok(tmp_path, server(), "--quota", "code=1", "--quota", "math=4")
    assert (m["quotas"]["code"], m["quotas"]["math"]) == (1, 4) and m["counts"]["n_prompts"] == 8 - 3 + 1 + 2
    assert c.main(argv(tmp_path / "x", None, "--dry-run", "--quota", "poetry=3")) == 2


def test_a_source_column_can_be_overridden_and_is_declared(tmp_path: Path, server: Callable[..., FakeServer]) -> None:
    m = run_ok(tmp_path, server(), "--column", "mgsm_train_parallel=question")
    assert {s["name"]: s["column"] for s in m["prompt_sources"]}["mgsm_train_parallel"] == "question"
    assert c.main(argv(tmp_path / "y", None, "--dry-run", "--column", "nosuch=a")) == 2


def test_a_build_with_prompts_still_to_generate_is_refused(tmp_path: Path, server: Callable[..., FakeServer]) -> None:
    n = {"i": 0}

    def respond(body: dict) -> tuple[int, dict]:
        n["i"] += 1
        return (500, {}) if n["i"] == 2 else default_respond(body)

    srv = server(respond)
    assert c.main(argv(tmp_path, srv, total=12)) == 2  # one prompt failed once and has attempts left: not finished
    assert not (tmp_path / "cal1" / "manifest.json").exists()
    assert c.main(argv(tmp_path, srv, total=12)) == 0  # the resume retries it and finishes
