"""A tiny fake OpenAI/vLLM-compatible HTTP server for tests (stdlib only, no GPU, no network).

It serves, on a random localhost port:

* ``POST /v1/completions``: a deterministic bigram "language model" with vLLM-style
  ``echo`` + ``logprobs`` and ``prompt_logprobs`` responses;
* ``POST /v1/chat/completions``: replies produced by a caller-supplied function;
* ``GET /v1/models``.

Every request body and header set is recorded so tests can assert on what was sent.
"""

import json
import math
import sys
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

ChatReply = Callable[[dict[str, Any]], tuple[str, str]]


class FakeHTTPError(Exception):
    """Raised by a chat reply function to make the server answer with an HTTP error."""

    def __init__(self, status: int, message: str = "injected error"):
        super().__init__(message)
        self.status = status
        self.message = message


class FakeLM:
    """Bigram model: the logits for predicting a token depend only on the previous token.

    ``noise`` perturbs the logits with a second fixed table, which makes a "quantized" variant.
    The full-vocabulary distributions are exposed through :meth:`logprobs` so tests can compute
    exact KL divergences.
    """

    def __init__(self, vocab_size: int = 16, seed: int = 0, noise: float = 0.0, scale: float = 2.0):
        self.vocab_size = vocab_size
        rng = np.random.default_rng(seed)
        self._table = rng.normal(size=(vocab_size, vocab_size)) * scale
        self._noise_table = np.random.default_rng(seed + 1000).normal(size=(vocab_size, vocab_size))
        self.noise = noise

    def logprobs(self, prev_token: int) -> np.ndarray:
        """Log-probabilities over the whole vocabulary after ``prev_token``."""
        x = self._table[prev_token] + self.noise * self._noise_table[prev_token]
        return x - np.log(np.exp(x).sum())


def _lp(x: float) -> float:
    return float(max(x, -9999.0))


class FakeServer:
    """Context manager running the fake server in a background thread."""

    def __init__(
        self,
        lm: FakeLM | None = None,
        *,
        chat_reply: ChatReply | None = None,
        fail_first: int = 0,
        reject_with: tuple[int, str] | None = None,
        wrong_prompt_ids: bool = False,
        usage_offset: int = 0,
        max_model_len: int = 32688,
    ):
        self.lm = lm
        self.chat_reply = chat_reply
        self.fail_first = fail_first
        self.reject_with = reject_with
        self.wrong_prompt_ids = wrong_prompt_ids
        self.usage_offset = usage_offset
        self.max_model_len = max_model_len
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self._lock = threading.Lock()
        self._failed = 0
        self._httpd: ThreadingHTTPServer | None = None

    # ------------------------------------------------------------------ plumbing
    @property
    def base_url(self) -> str:
        """The ``/v1`` base URL of the running server."""
        assert self._httpd is not None
        return f"http://127.0.0.1:{self._httpd.server_address[1]}/v1"

    def __enter__(self) -> "FakeServer":
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # silence the default stderr logging
                pass

            def _send(self, status: int, body: dict[str, Any]) -> None:
                raw = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:
                if self.path.rstrip("/") == "/v1/models":
                    self._send(
                        200,
                        {
                            "object": "list",
                            "data": [
                                {
                                    "id": "olala",
                                    "object": "model",
                                    "max_model_len": server.max_model_len,
                                }
                            ],
                        },
                    )
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                with server._lock:
                    server.requests.append({"path": self.path, "body": body})
                    server.headers.append({k.lower(): v for k, v in self.headers.items()})
                    if server._failed < server.fail_first:
                        server._failed += 1
                        self._send(500, {"error": "injected failure"})
                        return
                if server.reject_with is not None:
                    status, message = server.reject_with
                    self._send(status, {"error": {"message": message}})
                    return
                if self.path == "/v1/completions" and server.lm is not None:
                    self._send(200, server._completions(body))
                elif self.path == "/v1/chat/completions" and server.chat_reply is not None:
                    try:
                        self._send(200, server._chat(body))
                    except FakeHTTPError as exc:
                        self._send(exc.status, {"error": {"message": exc.message}})
                else:
                    self._send(404, {"error": "not found"})

        class QuietServer(ThreadingHTTPServer):
            def handle_error(self, request: Any, client_address: Any) -> None:
                # a client that gave up (a timeout test) leaves a broken pipe: not worth a traceback
                if not isinstance(sys.exc_info()[1], BrokenPipeError | ConnectionResetError):
                    super().handle_error(request, client_address)

        self._httpd = QuietServer(("127.0.0.1", 0), Handler)
        threading.Thread(
            target=self._httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        ).start()
        return self

    def __exit__(self, *exc: object) -> None:
        assert self._httpd is not None
        self._httpd.shutdown()
        self._httpd.server_close()

    # ------------------------------------------------------------------ chat
    def _chat(self, body: dict[str, Any]) -> dict[str, Any]:
        assert self.chat_reply is not None
        content, finish = self.chat_reply(body)
        return {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "olala"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish,
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    # ------------------------------------------------------------------ completions
    def _completions(self, body: dict[str, Any]) -> dict[str, Any]:
        assert self.lm is not None
        ids = [int(t) for t in body["prompt"]]
        k = int(body.get("logprobs") or body.get("prompt_logprobs") or 0)
        as_ids = bool(body.get("return_tokens_as_token_ids"))
        shown_ids = [t + 1 for t in ids] if self.wrong_prompt_ids else ids  # simulate a BOS shift

        def name(t: int) -> str:
            return f"token_id:{t}" if as_ids else f"tok{t}"

        sampled = int(np.argmax(self.lm.logprobs(ids[-1])))
        # position -> (distribution or None, actual token)
        steps: list[tuple[np.ndarray | None, int]] = [(None, ids[0])]
        steps += [(self.lm.logprobs(ids[i - 1]), ids[i]) for i in range(1, len(ids))]
        steps.append((self.lm.logprobs(ids[-1]), sampled))

        def top_entries(dist: np.ndarray, target: int) -> dict[int, float]:
            order = [int(t) for t in np.argsort(-dist, kind="stable")[:k]]
            if target not in order:  # vLLM adds the actual token as the (k+1)-th entry
                order.append(target)
            return {t: float(dist[t]) for t in order}

        result: dict[str, Any] = {
            "id": "cmpl-fake",
            "object": "text_completion",
            "created": 0,
            "model": body.get("model", "olala"),
            "usage": {
                "prompt_tokens": len(ids) + self.usage_offset,
                "completion_tokens": 1,
                "total_tokens": len(ids) + 1,
            },
        }
        choice: dict[str, Any] = {"index": 0, "text": "", "finish_reason": "length"}
        if body.get("echo") and body.get("logprobs") is not None:
            tokens = [name(t) for t in shown_ids] + [name(sampled)]
            choice["logprobs"] = {
                "tokens": tokens,
                "token_logprobs": [None if d is None else _lp(d[t]) for d, t in steps],
                "top_logprobs": [
                    None
                    if d is None
                    else {name(t): _lp(lp) for t, lp in top_entries(d, tok).items()}
                    for d, tok in steps
                ],
                "text_offset": list(range(len(tokens))),
            }
        if body.get("prompt_logprobs") is not None:
            plp: list[dict[str, Any] | None] = [None]
            for dist, tok in steps[1:-1]:
                assert dist is not None
                entries = top_entries(dist, tok)
                plp.append(
                    {
                        str(t): {"logprob": lp, "rank": r + 1, "decoded_token": f"tok{t}"}
                        for r, (t, lp) in enumerate(entries.items())
                    }
                )
            choice["prompt_logprobs"] = plp
        result["choices"] = [choice]
        return result


def exact_stats(lm_ref: FakeLM, lm_var: FakeLM, chunks: np.ndarray) -> dict[str, float]:
    """Full-vocabulary ground truth for the bigram models over ``chunks`` (positions 1..L-1)."""
    kls, agree, nll_ref, nll_var = [], [], [], []
    for chunk in chunks:
        for i in range(1, len(chunk)):
            lp_r, lp_v = lm_ref.logprobs(int(chunk[i - 1])), lm_var.logprobs(int(chunk[i - 1]))
            p = np.exp(lp_r)
            kls.append(float(np.sum(p * (lp_r - lp_v))))
            agree.append(int(np.argmax(lp_r) == np.argmax(lp_v)))
            nll_ref.append(-lp_r[chunk[i]])
            nll_var.append(-lp_v[chunk[i]])
    return {
        "kld_mean": float(np.mean(kls)),
        "top1_agreement": float(np.mean(agree)),
        "ppl_ref": math.exp(float(np.mean(nll_ref))),
        "ppl_var": math.exp(float(np.mean(nll_var))),
    }
