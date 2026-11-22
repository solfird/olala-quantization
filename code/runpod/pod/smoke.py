"""Smoke test of a started vLLM server: health, one question, the log.

1. Wait for GET /health, polling every SMOKE_POLL_SECONDS (default 2) up to --timeout. On every poll
   the server log is scanned: an ERROR-level line, a Traceback, a CUDA error or an out-of-memory
   fails at once (after one second, so that the traceback is complete) and the last error line is
   printed; a server process that has exited ends the wait too.
2. Send one chat completion: French, a small maths question, reasoning_effort medium, the sampling
   of PROTOCOL.md section 6. It must return a non-empty final answer and a non-empty reasoning
   field (`reasoning`; the older `reasoning_content` is accepted).
3. Read the log for what must not pass unseen (PROTOCOL.md section 8, condition 2): the fork's
   'OLALA SLOW PATH' banners and any 'fallback', kernel JIT failures, NaN; and print the line vLLM
   logs when it quantizes online ('Quantizing N layers of types: ...'), the only proof that the
   ignore list took effect.

Writes smoke.json (and a dated copy: failed runs are kept). Exit 0 pass, 2 pass with warnings (read
them before any measurement), 1 fail. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

FATAL = [
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"(?:^|[\s)])ERROR(?:\s|:)"),
    re.compile(r"CUDA error"),
    re.compile(r"OutOfMemoryError|out of memory", re.IGNORECASE),
    re.compile(r"Segmentation fault"),
]
EXCEPTION = re.compile(
    r"^(?:\([^)]*\)\s+)?(?:[A-Za-z_][\w.]*\.)?[A-Za-z_]\w*(?:Error|Exception)\b.*:"
)
SLOW = [re.compile(r"slow path", re.IGNORECASE), re.compile(r"fallback", re.IGNORECASE)]
JIT = [
    re.compile(r"JIT build of .* failed"),
    re.compile(r"ninja: build stopped"),
    re.compile(r"nvcc fatal"),
    re.compile(r"Error building extension", re.IGNORECASE),
    re.compile(r"nvcc.*(?:not found|No such file)", re.IGNORECASE),
]
NAN = re.compile(r"\bnan\b", re.IGNORECASE)
QUANT = [
    re.compile(r"Quantizing \d+ layers of types"),
    re.compile(r"quantized[_ ]layers", re.IGNORECASE),
]
ONLINE_ARM = re.compile(r"^(fp8|mx|nvfp4|int8)")
PROMPT = (
    "Combien font 17 multiplié par 23 ? Explique brièvement ton calcul, puis donne le résultat "
    "final en une phrase, en français."
)
EXPECTED = "391"
MAX_LINES = 20


def scan(text: str) -> dict:
    """Classify the log lines: errors, slow-path banners, JIT failures, NaN, quantization line."""
    lines = text.splitlines()
    out: dict = {
        "errors": [],
        "slow_path": [],
        "jit_failures": [],
        "nan": [],
        "quantized_layers": None,
    }
    for line in lines:
        if any(p.search(line) for p in FATAL):
            out["errors"].append(line.strip())
        if any(p.search(line) for p in SLOW):
            out["slow_path"].append(line.strip())
        if any(p.search(line) for p in JIT):
            out["jit_failures"].append(line.strip())
        if NAN.search(line):
            out["nan"].append(line.strip())
        if out["quantized_layers"] is None and any(p.search(line) for p in QUANT):
            out["quantized_layers"] = line.strip()
    out["error_line"] = _error_line(lines, out["errors"])
    return {k: (v[:MAX_LINES] if isinstance(v, list) else v) for k, v in out.items()}


def _error_line(lines: list[str], errors: list[str]) -> str | None:
    """The most telling error line: the exception ending a traceback, else the last ERROR line."""
    if not errors:
        return None
    if any("Traceback (most recent call last)" in e for e in errors):
        for line in reversed(lines):
            if EXCEPTION.match(line.strip()):
                return line.strip()
    level = [e for e in errors if re.search(r"(?:^|[\s)])ERROR(?:\s|:)", e)]
    return (level or errors)[-1]


def read(path: Path | None) -> str:
    """The log so far (it may not exist yet)."""
    try:
        return path.read_text(errors="replace") if path else ""
    except OSError:
        return ""


def process_alive(pidfile: Path) -> bool | None:
    """True or False for the pid in the file (zombies are dead); None without a pid file."""
    try:
        pid = int(pidfile.read_text().strip())
    except (OSError, ValueError):
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    state = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False
    )
    return bool(state.stdout.strip()) and not state.stdout.strip().startswith("Z")


def healthy(base_url: str) -> bool:
    """GET /health answers 200."""
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=5) as response:  # noqa: S310 - local server
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def ask(base_url: str, timeout: float = 900) -> tuple[dict, dict]:
    """One chat completion; returns the request body and the parsed response."""
    body = {
        "model": "olala",
        "messages": [{"role": "user", "content": PROMPT}],
        "reasoning_effort": "medium",
        "chat_template_kwargs": {"reasoning_effort": "medium"},
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": 4096,
    }
    request = urllib.request.Request(  # noqa: S310 - local server
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return body, json.load(response)


def judge(report: dict, arm: str) -> None:
    """Fill the verdict from the facts already in the report."""
    failures: list[str] = report["failures"]
    warnings: list[str] = report["warnings"]
    if report["slow_path"]:
        warnings.append(f"{len(report['slow_path'])} slow-path or fallback line(s) in the log")
    if report["jit_failures"]:
        warnings.append(f"{len(report['jit_failures'])} kernel JIT failure line(s) in the log")
    if report["nan"]:
        warnings.append(f"{len(report['nan'])} NaN line(s) in the log")
    if ONLINE_ARM.match(arm) and not report["quantized_layers"] and not failures:
        warnings.append(
            "no 'Quantizing N layers of types' line: online quantization may not have applied"
        )
    if report.get("expected_in_answer") is False:
        warnings.append(f"the final answer does not contain {EXPECTED}")
    report["verdict"] = "fail" if failures else ("warn" if warnings else "pass")


def run(arm: str, log: Path | None, url: str, timeout: float, pidfile: Path, out_dir: Path) -> int:
    """The whole smoke test; returns the exit code."""
    poll = float(os.environ.get("SMOKE_POLL_SECONDS", "2"))
    started = time.time()
    report: dict = {
        "arm": arm,
        "utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "url": url,
        "log": str(log) if log else None,
        "failures": [],
        "warnings": [],
        "health_seconds": None,
    }
    state = "waiting"
    while state == "waiting":
        found = scan(read(log))
        if found["errors"]:
            time.sleep(1.0)  # let the traceback finish
            found = scan(read(log))
            report["failures"].append("the server log has an error: " + (found["error_line"] or ""))
            state = "error"
        elif healthy(url):
            report["health_seconds"] = round(time.time() - started, 1)
            state = "healthy"
        elif process_alive(pidfile) is False:
            report["failures"].append("the server process has exited before /health answered")
            state = "exited"
        elif time.time() - started > timeout:
            report["failures"].append(f"timeout: /health did not answer in {timeout:g} s")
            state = "timeout"
        else:
            time.sleep(poll)

    if state == "healthy":
        try:
            body, answer = ask(url)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            report["failures"].append(f"the chat request failed: {exc}")
        else:
            report["request"] = body
            choice = (answer.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            content = (message.get("content") or "").strip()
            reasoning = (message.get("reasoning") or message.get("reasoning_content") or "").strip()
            report.update(
                answer=content[:500],
                answer_nonempty=bool(content),
                reasoning_nonempty=bool(reasoning),
                reasoning_chars=len(reasoning),
                finish_reason=choice.get("finish_reason"),
                usage=answer.get("usage"),
                expected_in_answer=EXPECTED in content,
            )
            if not content:
                why = (
                    " (finish_reason length: raise max_tokens or look at the reasoning)"
                    if choice.get("finish_reason") == "length"
                    else ""
                )
                report["failures"].append("the final answer is empty" + why)
            if not reasoning:
                report["failures"].append("the reasoning field is empty")
        found = scan(read(log))
        if found["errors"]:
            report["failures"].append("the server log has an error: " + (found["error_line"] or ""))

    report.update(
        {k: found[k] for k in ("errors", "slow_path", "jit_failures", "nan", "quantized_layers")}
    )
    report["error_line"] = found["error_line"]
    judge(report, arm)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    (out_dir / "smoke.json").write_text(text)
    (out_dir / f"smoke-{report['utc'].replace('-', '').replace(':', '')}.json").write_text(text)
    print(f"smoke {arm}: {report['verdict'].upper()}")
    if report["health_seconds"] is not None:
        print(f"health after {report['health_seconds']} s")
    print(f"log: {report['log']}")
    if report.get("quantized_layers"):
        print(f"quantized layers: {report['quantized_layers']}")
    if "answer" in report:
        print(f"answer: {report['answer'][:200]!r}; reasoning: {report['reasoning_chars']} chars")
    for line in report["failures"]:
        print(f"FAIL {line}")
    for line in report["warnings"]:
        print(f"WARN {line}")
    for key in ("slow_path", "jit_failures", "nan"):
        for line in report[key][:5]:
            print(f"  {key}: {line}")
    print(f"written: {out_dir / 'smoke.json'}")
    return {"pass": 0, "warn": 2, "fail": 1}[report["verdict"]]


def main(argv: list[str]) -> int:
    """Command line entry point (smoke.sh fills in the paths)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--arm", required=True)
    parser.add_argument(
        "--log", help="the server log (default: the newest server-*.log of --out-dir)"
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--pidfile", default="/workspace/run/server.pid")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    log = Path(args.log) if args.log else None
    if log is None:
        logs = sorted(out_dir.glob("server-*.log"), key=lambda p: p.stat().st_mtime)
        log = logs[-1] if logs else None
    return run(args.arm, log, args.url.rstrip("/"), args.timeout, Path(args.pidfile), out_dir)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
