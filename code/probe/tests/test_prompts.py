"""Tests of probe_prompts.py: the jsonl format, the tokenizers, truncation and the special-token mask."""

import json

import numpy as np
import pytest
from conftest import TINY_SPECIAL

import probe_prompts as pp


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


CHAT = {"id": "c1", "kind": "chat", "messages": [{"role": "user", "content": "hi"}]}
PLAIN = {"id": "p1", "kind": "plain", "text": "hello"}


def test_read_prompts_parses_both_kinds(tmp_path):
    rows = pp.read_prompts(_write(tmp_path / "p.jsonl", [CHAT, PLAIN]))
    assert [r["id"] for r in rows] == ["c1", "p1"]
    assert rows[0]["kind"] == "chat" and rows[1]["kind"] == "plain"


@pytest.mark.parametrize(
    "rows,msg",
    [
        ([], "no prompt"),
        ([CHAT, CHAT], "duplicate"),
        ([{"id": "x", "kind": "chat"}], "messages"),
        ([{"id": "x", "kind": "plain"}], "text"),
        ([{"id": "x", "kind": "weird", "text": "a"}], "kind"),
        ([{"kind": "plain", "text": "a"}], "id"),
        ([{"id": "x", "kind": "chat", "messages": []}], "messages"),
        ([{"id": "x", "kind": "plain", "text": ""}], "text"),
    ],
)
def test_read_prompts_rejects_bad_rows(tmp_path, rows, msg):
    with pytest.raises(ValueError, match=msg):
        pp.read_prompts(_write(tmp_path / "p.jsonl", rows))


def test_write_then_read_round_trips(tmp_path):
    p = tmp_path / "x.jsonl"
    pp.write_prompts(p, [CHAT, PLAIN])
    assert pp.read_prompts(p) == [CHAT, PLAIN]


def test_byte_tokenizer_chat_has_markers_and_special_ids():
    tok = pp.ByteChatTokenizer()
    ids = tok.encode_chat([{"role": "user", "content": "hi"}])
    assert ids[0] == pp.ByteChatTokenizer.BOS and ids[1] == pp.ByteChatTokenizer.IM_START
    assert ids[-1] == ord("\n") + 3  # the generation prompt ends with the newline after 'assistant'
    assert set(TINY_SPECIAL) == tok.special_ids
    assert tok.encode_plain("A") == [ord("A") + 3]
    assert tok.decode(ord("A") + 3) == "A"
    assert tok.decode(pp.ByteChatTokenizer.IM_START) == "<|im_start|>"


def test_tokenize_prompts_flags_special_tokens_and_prepends_bos_to_plain_text():
    tok = pp.ByteChatTokenizer()
    ps = pp.tokenize_prompts([CHAT, PLAIN], tok, max_tokens=4096, bos_id=1)
    chat, plain = ps
    assert chat.special[0] and chat.special[1] and not chat.special.all()
    assert plain.tokens[0] == 1 and plain.special[0] and len(plain.tokens) == len("hello") + 1
    assert chat.kind == "chat" and plain.kind == "plain"
    assert not chat.truncated


def test_tokenize_prompts_without_bos_leaves_plain_text_alone():
    ps = pp.tokenize_prompts([PLAIN], pp.ByteChatTokenizer(), max_tokens=4096, bos_id=None)
    assert len(ps[0].tokens) == len("hello") and ps[0].tokens[0] == ord("h") + 3


def test_tokenize_prompts_truncates_to_max_tokens_and_says_so():
    long = {"id": "l", "kind": "plain", "text": "x" * 100}
    ps = pp.tokenize_prompts([long], pp.ByteChatTokenizer(), max_tokens=32, bos_id=1)
    assert len(ps[0].tokens) == 32 and ps[0].truncated is True and ps[0].tokens[0] == 1
    assert len(ps[0].special) == 32


def test_hf_prompt_tokenizer_applies_the_chat_template_and_passes_reasoning_effort():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {w: i for i, w in enumerate(["<unk>", "hi", "high", "effort", "user", "assistant", "there", "x"])}
    core = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    hf = PreTrainedTokenizerFast(tokenizer_object=core, unk_token="<unk>")
    hf.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})
    hf.chat_template = (
        "{% for m in messages %}<|im_start|> {{ m.role }} {{ m.content }} <|im_end|>{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|> assistant {{ reasoning_effort }}{% endif %}"
    )
    tok = pp.HFPromptTokenizer(hf, reasoning_effort="high")
    ids = tok.encode_chat([{"role": "user", "content": "hi"}])
    text = [tok.decode(i) for i in ids]
    assert text[0] == "<|im_start|>" and text[-1] == "high" and "assistant" in text
    assert ids[0] in tok.special_ids and ids[-1] not in tok.special_ids
    assert tok.encode_plain("hi there") == [vocab["hi"], vocab["there"]]
    assert pp.HFPromptTokenizer(hf, reasoning_effort="low").decode(vocab["x"]) == "x"


def test_prompt_dataclass_holds_numpy_arrays():
    p = pp.Prompt("a", "plain", np.array([1, 2]), np.array([True, False]), False)
    assert p.tokens.dtype.kind == "i" and p.special.dtype == bool
