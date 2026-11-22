"""Prompt files and tokenizers of the activation probe.

A prompt file is JSON lines. Each line is ``{"id": str, "kind": "chat", "messages": [{"role", "content"}, ...]}`` (it is
rendered with the model's chat template, generation prompt included) or ``{"id": str, "kind": "plain", "text": str}``
(tokenized as is, with the sink token prepended when ``bos_id`` is given: Appendix D of the OVH report says the training
data loader prepends id 151643, and the chat template starts with it too).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


@dataclass
class Prompt:
    """One tokenized prompt: ids, special-token mask (same length) and whether it was cut to ``max_tokens``."""

    id: str
    kind: str
    tokens: np.ndarray
    special: np.ndarray
    truncated: bool


class PromptTokenizer(Protocol):
    """What the probe needs from a tokenizer."""

    special_ids: set[int]

    def encode_chat(self, messages: list[dict[str, str]]) -> list[int]:
        """Token ids of the chat-templated conversation (generation prompt included)."""

    def encode_plain(self, text: str) -> list[int]:
        """Token ids of ``text`` without any added special token."""

    def decode(self, token_id: int) -> str:
        """A printable form of one token."""


def read_prompts(path: Path) -> list[dict[str, Any]]:
    """Read and validate a prompt file."""
    rows = []
    seen: set[str] = set()
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        pid = r.get("id")
        if not isinstance(pid, str) or not pid:
            raise ValueError(f"line {n}: missing id")
        if pid in seen:
            raise ValueError(f"line {n}: duplicate id {pid!r}")
        seen.add(pid)
        kind = r.get("kind")
        if kind == "chat":
            msgs = r.get("messages")
            if not isinstance(msgs, list) or not msgs:
                raise ValueError(f"line {n}: a chat prompt needs a non-empty list of messages")
        elif kind == "plain":
            if not isinstance(r.get("text"), str) or not r["text"]:
                raise ValueError(f"line {n}: a plain prompt needs a non-empty text")
        else:
            raise ValueError(f"line {n}: unknown kind {kind!r} (chat or plain)")
        rows.append(r)
    if not rows:
        raise ValueError(f"no prompt in {path}")
    return rows


def write_prompts(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write prompt rows as JSON lines."""
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


class ByteChatTokenizer:
    """The tiny model's tokenizer: one id per UTF-8 byte (``byte + 3``) and two chat markers (tests, ``--tiny``)."""

    BOS, IM_START, IM_END = 1, 300, 301
    special_ids = frozenset({0, 1, 2, 300, 301})

    def __init__(self) -> None:
        """Make the special ids a mutable ``set`` per instance."""
        self.special_ids = set(type(self).special_ids)

    def encode_plain(self, text: str) -> list[int]:
        """The bytes of ``text`` shifted by 3 (ids 0-2 are the model's pad / bos / eos)."""
        return [b + 3 for b in text.encode("utf-8")]

    def encode_chat(self, messages: list[dict[str, str]]) -> list[int]:
        """``<bos>`` then ``<im_start>role\\ncontent<im_end>`` per message, then ``<im_start>assistant\\n``."""
        ids = [self.BOS]
        for m in messages:
            ids += [self.IM_START, *self.encode_plain(f"{m['role']}\n{m['content']}"), self.IM_END]
        return [*ids, self.IM_START, *self.encode_plain("assistant\n")]

    def decode(self, token_id: int) -> str:
        """The character of a byte id, or the marker name."""
        names = {0: "<pad>", 1: "<bos>", 2: "<eos>", self.IM_START: "<|im_start|>", self.IM_END: "<|im_end|>"}
        if token_id in names:
            return names[token_id]
        return chr(token_id - 3) if 3 <= token_id < 259 else f"<{token_id}>"


class HFPromptTokenizer:
    """A ``transformers`` tokenizer used for chat prompts (its chat template) and plain text."""

    def __init__(self, tokenizer: Any, reasoning_effort: str | None = "high") -> None:
        """Wrap ``tokenizer``; ``reasoning_effort`` is passed to the chat template (Olala's template reads it)."""
        self.tok = tokenizer
        self.reasoning_effort = reasoning_effort
        special = set(tokenizer.all_special_ids)
        for tid, added in getattr(tokenizer, "added_tokens_decoder", {}).items():
            if getattr(added, "special", False):
                special.add(int(tid))
        self.special_ids = special

    @classmethod
    def from_dir(cls, path: Path, reasoning_effort: str | None = "high") -> HFPromptTokenizer:
        """Load the tokenizer files of a model directory (no network)."""
        from transformers import AutoTokenizer

        return cls(AutoTokenizer.from_pretrained(str(path), local_files_only=True), reasoning_effort)

    def encode_chat(self, messages: list[dict[str, str]]) -> list[int]:
        """Render with the chat template (generation prompt added), then tokenize without adding special tokens."""
        kwargs = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
        return list(self.tok(text, add_special_tokens=False)["input_ids"])

    def encode_plain(self, text: str) -> list[int]:
        """Tokenize without adding special tokens."""
        return list(self.tok(text, add_special_tokens=False)["input_ids"])

    def decode(self, token_id: int) -> str:
        """The token as text."""
        return self.tok.decode([token_id])


def tokenize_prompts(
    rows: list[dict[str, Any]], tok: PromptTokenizer, max_tokens: int, bos_id: int | None
) -> list[Prompt]:
    """Tokenize prompt rows: plain text gets ``bos_id`` in front; longer than ``max_tokens`` is cut (head kept)."""
    out = []
    for r in rows:
        if r["kind"] == "chat":
            ids = tok.encode_chat(r["messages"])
        else:
            ids = ([bos_id] if bos_id is not None else []) + tok.encode_plain(r["text"])
        truncated = len(ids) > max_tokens
        arr = np.asarray(ids[:max_tokens], dtype=np.int64)
        out.append(Prompt(r["id"], r["kind"], arr, np.isin(arr, sorted(tok.special_ids)), truncated))
    return out
