"""Smoke test of the llama-server that serve_gguf.sh started (OVH's fork, an Olala GGUF file).

1. Wait for GET /health, polling the server log: a fatal line (GGML_ASSERT, CUDA error, out of
   memory, an unknown architecture, 'Context size has been exceeded') fails at once; a dead
   server process ends the wait.
2. One French chat completion with reasoning effort medium, sent exactly the way the harness
   sends it: temperature, top_p and top_k, and `chat_template_kwargs.reasoning_effort`
   (inspect_ai's extra_body); no top-level reasoning_effort (it would hide a failure of the
   kwargs) and no max_tokens (the harness leaves it unset). Required: a non-empty final answer
   in `message.content` that does NOT contain the reasoning trace (the analysis channel must be
   in `message.reasoning_content`), and no wall of '?' (the NaN signature of a quantized file
   on a GPU).
3. A second request with effort `disabled`: the template must then close the analysis channel,
   so the reasoning field is empty. It proves the per-request kwargs reach the template (the
   parity run uses `high`).
4. The response shape, the speed from the server's own `timings` fields, the memory lines of
   the log (model, KV, recurrent-state and compute buffers, slots) and, with --parallel N, the
   aggregate speed of N bounded requests.

Writes smoke.json (and a dated copy). Exit 0 pass, 2 pass with warnings (read them), 1 fail.
Standard library only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

_spec = importlib.util.spec_from_file_location("smoke_base", Path(__file__).with_name("smoke.py"))
base = importlib.util.module_from_spec(_spec)
sys.modules["smoke_base"] = base
_spec.loader.exec_module(base)

FATAL = [
    re.compile(r"GGML_ASSERT"),
    re.compile(r"CUDA error"),
    re.compile(r"out of memory|OutOfMemory|cudaMalloc failed|failed to allocate", re.IGNORECASE),
    re.compile(
        r"error loading model|failed to load model|unknown model architecture", re.IGNORECASE
    ),
    re.compile(r"Context size has been exceeded|Compute error|Invalid input batch"),
    re.compile(r"Segmentation fault|terminate called|Aborted"),
]
SLOW = [re.compile(r"slow path|fallback|falling back|fall back", re.IGNORECASE)]
NAN = re.compile(r"\bnan\b", re.IGNORECASE)
KV_PRESSURE = re.compile(r"failed to find free space in the KV cache")
SPLITS = re.compile(r"graph.*\bsplits\s*=\s*(\d+(?:\s*/\s*\d+)?)")
OFFLOADED = re.compile(r"offloaded (\d+)/(\d+) layers")
MAX_SPLITS = (
    8  # the GPU path has 2; the upstream-v1.0.0 regression of the fork had 612 (layers on the CPU)
)
FACTS = [
    OFFLOADED,
    SPLITS,
    re.compile(r"(model|KV|RS|compute) buffer size\s*="),
    re.compile(r"n_slots\s*="),
    re.compile(r"n_ctx_slot\s*="),
    re.compile(r"kv_unified\s*="),
]
MAX_LINES = 20
LEAK_MARKERS = [
    "<|channel_start|>",
    "<|channel_end|>",
    "<|content|>",
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
]
PROMPT = base.PROMPT
EXPECTED = base.EXPECTED
QUESTION_MARKS = re.compile(r"\?{8,}")
REASONING_FLAG_HINT = (
    "the server must extract the reasoning into message.reasoning_content: start it WITHOUT "
    "`--reasoning-format none` (default is deepseek) and with `--jinja`"
)


def scan(text: str) -> dict:
    """Classify the log lines: fatal, slow-path, NaN, KV-cache pressure, and the memory facts."""
    out: dict = {
        "errors": [],
        "slow_path": [],
        "nan": [],
        "kv_pressure": [],
        "log_facts": [],
        "graph_splits": [],
        "offloaded_layers": [],
    }
    for line in text.splitlines():
        stripped = line.strip()
        if match := SPLITS.search(line):
            out["graph_splits"] = [int(n) for n in re.findall(r"\d+", match.group(1))]
        if match := OFFLOADED.search(line):
            out["offloaded_layers"] = [int(match.group(1)), int(match.group(2))]
        if any(p.search(line) for p in FATAL):
            out["errors"].append(stripped)
        if any(p.search(line) for p in SLOW):
            out["slow_path"].append(stripped)
        if NAN.search(line):
            out["nan"].append(stripped)
        if KV_PRESSURE.search(line):
            out["kv_pressure"].append(stripped)
        if any(p.search(line) for p in FACTS):
            out["log_facts"].append(re.sub(r"\s+", " ", stripped))
    out["error_line"] = out["errors"][-1] if out["errors"] else None
    keep = ("graph_splits", "offloaded_layers")
    return {
        k: (v[:MAX_LINES] if isinstance(v, list) and k not in keep else v) for k, v in out.items()
    }


def post_chat(base_url: str, body: dict, timeout: float) -> dict:
    """One chat completion; returns the parsed JSON."""
    request = urllib.request.Request(  # noqa: S310 - local server
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.load(response)


def harness_body(effort: str, prompt: str = PROMPT, max_tokens: int | None = None) -> dict:
    """The harness request (evals/run_tasks.py config_for) plus an optional token bound."""
    body: dict = {
        "model": "olala",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 40,
        "chat_template_kwargs": {"reasoning_effort": effort},
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    return body


def split(answer: dict) -> tuple[dict, str, str, str | None]:
    """The first choice's message, content, reasoning (either field name) and finish reason."""
    choice = (answer.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = (message.get("content") or "").strip()
    reasoning = (message.get("reasoning_content") or message.get("reasoning") or "").strip()
    return message, content, reasoning, choice.get("finish_reason")


def reasoning_leaked(content: str, reasoning: str) -> bool:
    """The reasoning trace is in the content: channel or think markers, or its start repeated."""
    if any(marker in content for marker in LEAK_MARKERS):
        return True
    return len(reasoning) >= 40 and reasoning[:40] in content


def generated_tokens(answer: dict) -> int:
    """Tokens generated, from the server's timings, else from the usage."""
    timings = answer.get("timings") or {}
    if "predicted_n" in timings:
        return int(timings["predicted_n"])
    return int((answer.get("usage") or {}).get("completion_tokens") or 0)


def parallel_probe(base_url: str, n: int, timeout: float) -> dict:
    """N concurrent bounded requests (effort medium, max_tokens 256): decode speed under load."""
    results: list[dict | None] = [None] * n
    errors: list[str] = []

    def one(i: int) -> None:
        prompt = f"Combien font {17 + i} multiplié par {23 + i} ? Réponds en français."
        try:
            results[i] = post_chat(
                base_url, harness_body("medium", prompt, max_tokens=256), timeout
            )
        except (urllib.error.URLError, OSError, ValueError) as exc:
            errors.append(str(exc))

    started = time.time()
    threads = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - started
    done = [r for r in results if r is not None]
    tokens = sum(generated_tokens(r) for r in done)
    return {
        "requests": n,
        "ok": sum(1 for r in done if generated_tokens(r) > 0),
        "errors": len(errors) + (len(done) - sum(1 for r in done if generated_tokens(r) > 0)),
        "error_messages": errors[:5],
        "generated_tokens": tokens,
        "wall_seconds": round(wall, 2),
        "aggregate_tokens_per_second": round(tokens / wall, 1) if wall > 0 else 0.0,
    }


def judge(report: dict) -> None:
    """The verdict from the facts already in the report."""
    failures: list[str] = report["failures"]
    warnings: list[str] = report["warnings"]
    if report["slow_path"]:
        warnings.append(f"{len(report['slow_path'])} fallback or slow-path line(s) in the log")
    if report["nan"]:
        warnings.append(f"{len(report['nan'])} NaN line(s) in the log")
    if report["kv_pressure"]:
        warnings.append(
            "the KV cache ran short of space at least once (retry with a smaller batch)"
        )
    if report["graph_splits"] and max(report["graph_splits"]) > MAX_SPLITS:
        warnings.append(
            f"graph splits {report['graph_splits']}: many splits mean layers run on the CPU "
            "(the GPU path has 2); expect a very slow server"
        )
    loaded = report["offloaded_layers"]
    if loaded and loaded[0] < loaded[1]:
        warnings.append(f"only {loaded[0]}/{loaded[1]} layers offloaded to the GPU")
    if report.get("expected_in_answer") is False:
        warnings.append(f"the final answer does not contain {EXPECTED}")
    parallel = report.get("parallel")
    if parallel and parallel["errors"]:
        warnings.append(
            f"{parallel['errors']} of {parallel['requests']} parallel request(s) failed"
        )
    report["verdict"] = "fail" if failures else ("warn" if warnings else "pass")


def check_main(report: dict, answer: dict, wall: float) -> None:
    """Fill the report from the medium-effort answer."""
    message, content, reasoning, finish = split(answer)
    top = sorted(answer)
    timings = answer.get("timings") or {}
    leaked = reasoning_leaked(content, reasoning)
    report.update(
        answer=content[:500],
        answer_nonempty=bool(content),
        reasoning_nonempty=bool(reasoning),
        reasoning_chars=len(reasoning),
        reasoning_in_content=leaked,
        finish_reason=finish,
        usage=answer.get("usage"),
        expected_in_answer=EXPECTED in content,
        response_shape={
            "top_level_keys": top,
            "choice_keys": sorted((answer.get("choices") or [{}])[0]),
            "message_keys": sorted(message),
            "timings_keys": sorted(timings),
            "content_chars": len(content),
            "reasoning_chars": len(reasoning),
        },
    )
    if "predicted_per_second" in timings:
        report["tokens_per_second"] = {
            "generation": round(float(timings["predicted_per_second"]), 2),
            "prompt": round(float(timings.get("prompt_per_second", 0.0)), 2),
            "source": "server timings",
        }
    else:
        tokens = generated_tokens(answer)
        report["tokens_per_second"] = {
            "generation": round(tokens / wall, 2) if wall > 0 else 0.0,
            "prompt": None,
            "source": "wall clock",
        }
        report["warnings"].append(
            "the reply has no `timings` object: speed computed from the wall clock"
        )
    failures = report["failures"]
    if not content:
        why = " (finish_reason length: the answer was cut)" if finish == "length" else ""
        failures.append("the final answer (message.content) is empty" + why)
    if leaked:
        failures.append("the reasoning trace is in message.content: " + REASONING_FLAG_HINT)
    elif not reasoning:
        failures.append(
            "message.reasoning_content is empty at effort medium: the model did not think, or "
            "the trace is somewhere else; " + REASONING_FLAG_HINT
        )
    if QUESTION_MARKS.search(content) or QUESTION_MARKS.search(reasoning):
        failures.append(
            "the output is a run of '?': the NaN signature of a quantized file on a GPU"
        )


def check_disabled(report: dict, answer: dict) -> None:
    """Effort `disabled` must close the analysis channel: an answer and no reasoning."""
    _, content, reasoning, _ = split(answer)
    report["disabled_check"] = {"answer_nonempty": bool(content), "reasoning_empty": not reasoning}
    if reasoning:
        report["failures"].append(
            "effort `disabled` still produced a reasoning trace: "
            "chat_template_kwargs.reasoning_effort does not reach the chat template, "
            "so `high` would silently be `medium`"
        )
    if not content:
        report["failures"].append("effort `disabled` produced no final answer")


def models(base_url: str) -> list[str]:
    """The ids /v1/models lists (empty when the call fails)."""
    try:
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=10) as response:  # noqa: S310
            data = json.load(response)
    except (urllib.error.URLError, OSError, ValueError):
        return []
    return [m.get("id", "") for m in data.get("data", [])]


def run(
    log: Path | None,
    url: str,
    timeout: float,
    pidfile: Path,
    out_dir: Path,
    parallel: int,
    request_timeout: float,
) -> int:
    """The whole smoke test; returns the exit code."""
    poll = float(os.environ.get("SMOKE_POLL_SECONDS", "2"))
    started = time.time()
    report: dict = {
        "arm": "gguf",
        "utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "url": url,
        "log": str(log) if log else None,
        "failures": [],
        "warnings": [],
        "health_seconds": None,
    }
    state = "waiting"
    while state == "waiting":
        found = scan(base.read(log))
        if found["errors"]:
            time.sleep(1.0)  # let the message finish
            found = scan(base.read(log))
            report["failures"].append("the server log has an error: " + (found["error_line"] or ""))
            state = "error"
        elif base.healthy(url):
            report["health_seconds"] = round(time.time() - started, 1)
            state = "healthy"
        elif base.process_alive(pidfile) is False:
            report["failures"].append("the server process has exited before /health answered")
            state = "exited"
        elif time.time() - started > timeout:
            report["failures"].append(f"timeout: /health did not answer in {timeout:g} s")
            state = "timeout"
        else:
            time.sleep(poll)

    if state == "healthy":
        report["served_model_ids"] = models(url)
        if "olala" not in report["served_model_ids"]:
            report["warnings"].append(
                f"/v1/models does not list `olala`: {report['served_model_ids']}"
            )
        try:
            body = harness_body("medium")
            report["request"] = body
            t0 = time.time()
            answer = post_chat(url, body, request_timeout)
            check_main(report, answer, time.time() - t0)
            check_disabled(
                report, post_chat(url, harness_body("disabled", max_tokens=512), request_timeout)
            )
            if parallel > 0:
                report["parallel"] = parallel_probe(url, parallel, request_timeout)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            report["failures"].append(f"the chat request failed: {exc}")
        found = scan(base.read(log))
        if found["errors"]:
            report["failures"].append("the server log has an error: " + (found["error_line"] or ""))

    keys = (
        "errors",
        "slow_path",
        "nan",
        "kv_pressure",
        "log_facts",
        "graph_splits",
        "offloaded_layers",
    )
    report.update({k: found[k] for k in keys})
    report["error_line"] = found["error_line"]
    judge(report)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    (out_dir / "smoke.json").write_text(text)
    (out_dir / f"smoke-{report['utc'].replace('-', '').replace(':', '')}.json").write_text(text)
    print(f"smoke gguf: {report['verdict'].upper()}")
    if report["health_seconds"] is not None:
        print(f"health after {report['health_seconds']} s")
    print(f"log: {report['log']}")
    for line in report.get("log_facts", []):
        print(f"  log: {line}")
    if "response_shape" in report:
        shape = report["response_shape"]
        print(
            f"response: top-level {shape['top_level_keys']}, message {shape['message_keys']}, "
            f"finish {report['finish_reason']}"
        )
        print(
            f"content {shape['content_chars']} chars, reasoning_content "
            f"{shape['reasoning_chars']} chars; answer: {report['answer'][:200]!r}"
        )
    speed = report.get("tokens_per_second")
    if speed:
        print(
            f"speed: {speed['generation']} tokens/s generation, {speed['prompt']} prompt "
            f"({speed['source']})"
        )
    if report.get("parallel"):
        p = report["parallel"]
        print(
            f"parallel {p['requests']}: aggregate {p['aggregate_tokens_per_second']} tokens/s "
            f"({p['generated_tokens']} tokens in {p['wall_seconds']} s, {p['errors']} error(s))"
        )
    for line in report["failures"]:
        print(f"FAIL {line}")
    for line in report["warnings"]:
        print(f"WARN {line}")
    for key in ("slow_path", "nan", "kv_pressure"):
        for line in report[key][:5]:
            print(f"  {key}: {line}")
    print(f"written: {out_dir / 'smoke.json'}")
    return {"pass": 0, "warn": 2, "fail": 1}[report["verdict"]]


def main(argv: list[str]) -> int:
    """Command line entry point (smoke_gguf.sh fills in the paths)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--log", help="the server log (default: the newest server-*.log of --out-dir)"
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument("--pidfile", default="/workspace/run/server.pid")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--parallel", type=int, default=0, help="also send N concurrent bounded requests"
    )
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    log = Path(args.log) if args.log else None
    if log is None:
        logs = sorted(out_dir.glob("server-*.log"), key=lambda p: p.stat().st_mtime)
        log = logs[-1] if logs else None
    return run(
        log,
        args.url.rstrip("/"),
        args.timeout,
        Path(args.pidfile),
        out_dir,
        args.parallel,
        args.request_timeout,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
