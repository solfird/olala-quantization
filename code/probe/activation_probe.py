"""Activation probe: reproduce OVH's Appendix C activation-outlier claims on the released Olala weights.

Runs the vendored OVH modeling code (``quant/gptq/vendor``, pure-PyTorch Mamba-3 and MoE paths) over a fixed list of
prompts, hooks every layer (see :mod:`probe_hooks` for the sites), measures what the quantization of those activations
would have to survive, ablates the spiking branches to see how much the next-token distribution depends on them, and
writes one JSON file plus one markdown report that compares every measured number with the number OVH claims.

Layer numbers are 0-based throughout (``block.layer_idx``); the attention layers of the real model are 4, 9, 14, 19,
24, 29 and 34. Appendix C of the OVH report uses the same convention (its attention layers are L4 ... L34).

Usage::

    uv run python activation_probe.py --model-dir /models/Olala-v1.0-mini-7A1B --prompts prompts.jsonl --out out/ \\
        --device cuda --max-tokens 4096
    uv run python activation_probe.py --tiny --out out/        # the tiny random model, CPU, for tests

This module holds the model loading, the prompt handling and the command line; the hooks are in ``probe_hooks.py``,
the statistics in ``probe_stats.py`` and the claim-by-claim table in ``probe_claims.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
GPTQ_DIR = HERE.parent / "quant" / "gptq"
if str(GPTQ_DIR) not in sys.path:
    sys.path.insert(0, str(GPTQ_DIR))

TINY_IM_START = 300  # ids of the two chat markers of the tiny byte tokenizer (the real ones are 151644 / 151645)
TINY_IM_END = 301
TINY_HEADDIM = 16  # the tiny model gets 8 Mamba heads of 16 channels (the real one has 48 heads of 64)


def build_tiny_model(seed: int = 0, headdim: int = TINY_HEADDIM) -> Any:
    """The repo's tiny random Olala (``quant/gptq/tiny_model.py``) with ``headdim``-wide Mamba heads (8 of them)."""
    import tiny_model

    saved = dict(tiny_model.TINY_OVERRIDES)
    tiny_model.TINY_OVERRIDES["mamba_headdim"] = headdim
    try:
        return tiny_model.build_tiny_model(seed)
    finally:
        tiny_model.TINY_OVERRIDES.clear()
        tiny_model.TINY_OVERRIDES.update(saved)


# ------------------------------------------------------------------------------------------------ model loading
def _state_files(model_dir: Path) -> list[Path]:
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        return [model_dir / n for n in names]
    single = model_dir / "model.safetensors"
    if single.is_file():
        return [single]
    raise FileNotFoundError(f"no model.safetensors or model.safetensors.index.json in {model_dir}")


def load_model(model_dir: Path, device: str) -> Any:
    """Load the bf16 checkpoint in ``model_dir`` into the vendored ``OlalaForCausalLM`` on ``device``, dtypes as stored.

    The model is built on the meta device and the tensors are assigned from the safetensors files (so the 13.6 GB of the
    real checkpoint are never held twice and never in float32). Refuses a directory whose ``modeling_olala.py`` /
    ``configuration_olala.py`` differ from the vendored copies, a checkpoint with missing or unexpected tensors (for
    example a quantized one) and any tensor left on the meta device.
    """
    import olala_model
    from safetensors import safe_open

    same = olala_model.check_vendored_matches(model_dir)
    if not all(same.values()):
        raise RuntimeError(f"the modeling files in {model_dir} differ from quant/gptq/vendor: {same}")
    modeling, configuration = olala_model.load_olala_modules()
    cfg = configuration.OlalaConfig(**json.loads((model_dir / "config.json").read_text()))
    cfg.quantization_config = None
    with torch.device("meta"):
        model = modeling.OlalaForCausalLM(cfg)
    state: dict[str, torch.Tensor] = {}
    for f in _state_files(model_dir):
        with safe_open(str(f), framework="pt", device=device) as sf:
            for name in sf.keys():  # noqa: SIM118 (safe_open is not a mapping)
                state[name] = sf.get_tensor(name)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint does not match the model: missing {missing[:5]}, unexpected {unexpected[:5]}")
    for module in model.modules():  # non-persistent buffers (tokens_per_expert) were never in the checkpoint
        for name, buf in list(module._buffers.items()):
            if buf is not None and buf.is_meta:
                module._buffers[name] = torch.zeros(buf.shape, dtype=buf.dtype, device=device)
    left = [n for n, t in [*model.named_parameters(), *model.named_buffers()] if t.is_meta]
    if left:
        raise RuntimeError(f"tensors left on the meta device: {left[:5]}")
    return model.eval()


# ------------------------------------------------------------------------------------------------ memory estimate
GIB = 1024**3


def estimate_memory_gib(
    cfg: Any, tokens: int, weights_gib: float | None = None, attn: str = "flash"
) -> dict[str, float]:
    """Peak GPU memory (GiB) of one probe pass of ``tokens`` tokens at batch 1, from the shapes of ``cfg``.

    A model, not a measurement: weights (bf16 matrices; ``weights_gib`` overrides the parameter count), the largest
    transient of the pure-PyTorch Mamba-3 path (float32, built from the shapes in ``_mamba3_mimo_torch_segment``), the
    eager attention score tensors (``attn == "eager"``: the float32 ``[heads, T, T]`` scores of the signal heads, three
    alive at once; ``"flash"`` and ``"flex"`` keep none), the probe's own float copies, one chunk of logits, and a
    CUDA context plus allocator slack.
    """
    d = cfg.hidden_size
    heads = (2 * d) // cfg.mamba_headdim
    p, n, r = cfg.mamba_headdim, cfg.mamba_d_state, cfg.mamba_mimo_dim
    f32 = 4
    if weights_gib is None:
        per_expert = 2 * cfg.moe_routed_input_dim * cfg.moe_routed_intermediate_size
        moe = cfg.moe_num_routed_experts * per_expert
        moe += 2 * d * cfg.moe_shared_intermediate_size + 2 * d * cfg.moe_routed_input_dim
        mamba = d * (4 * d + 3 * heads) + d * (2 * n * r + n // 4) + 2 * d * d
        n_attn = sum(c == "V" for c in cfg.layers_config)
        n_mamba = len(cfg.layers_config) - n_attn
        signal = cfg.num_signal_heads_diff or cfg.num_attention_heads // 2
        noise = cfg.num_attention_heads - signal
        attn_w = d * cfg.num_attention_heads * cfg.head_dim + 2 * d * noise * cfg.tpa_rank
        attn_w += 2 * d * cfg.tpa_rank * cfg.head_dim + 2 * d * signal * cfg.head_dim  # B_k, B_v; gate and out proj
        total = len(cfg.layers_config) * moe + n_mamba * mamba + n_attn * attn_w
        total += 2 * cfg.vocab_size * d
        weights_gib = total * 2 / GIB
    mamba_transient = tokens * (6 * r * heads * n + 4 * r * heads * p) * f32 / GIB  # q/k copies and v/o tensors
    scores = 0.0
    if attn == "eager":
        signal = cfg.num_signal_heads_diff or cfg.num_attention_heads // 2
        scores = 3 * signal * tokens * tokens * f32 / GIB
    pairs = tokens * cfg.moe_num_active_experts
    probe = (pairs * cfg.moe_routed_intermediate_size * (2 * 2 + 4 * 4) + tokens * 2 * d * 2 * 4 * 4) / GIB
    logits = 256 * cfg.vocab_size * f32 * 6 / GIB
    base = weights_gib + mamba_transient + scores + probe + logits
    parts = {
        "weights": weights_gib,
        "mamba3_torch_transient": mamba_transient,
        "eager_attention_scores": scores,
        "probe_float_copies": probe,
        "logit_chunk": logits,
        "cuda_context": 1.0,
        "allocator_slack_10pct": 0.1 * base,
    }
    parts["total"] = base + parts["cuda_context"] + parts["allocator_slack_10pct"]
    return parts


# ------------------------------------------------------------------------------------------------ the run
def _jsonable(obj: Any) -> Any:
    """Recursively make ``obj`` strict-JSON safe: numpy scalars to Python, non-finite floats to "NaN" / "Infinity"."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return _jsonable(obj.item())
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
    return obj


def _collect_token_ids(obj: Any, out: set[int]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "token_id" and isinstance(v, int):
                out.add(v)
            else:
                _collect_token_ids(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_token_ids(v, out)


def _config_summary(model: Any) -> dict[str, Any]:
    cfg = model.config
    layers = cfg.layers_config
    first = next(b for b in model.model.layers if hasattr(b.mixer, "nheads"))
    return {
        "layers_config": layers,
        "n_layers": len(layers),
        "mamba_layers_0based": [i for i, c in enumerate(layers) if c == "M"],
        "attention_layers_0based": [i for i, c in enumerate(layers) if c == "V"],
        "hidden_size": cfg.hidden_size,
        "mamba_heads_per_layer": first.mixer.nheads,
        "mamba_head_dim": first.mixer.headdim,
        "n_routed_experts": cfg.moe_num_routed_experts,
        "top_k": cfg.moe_num_active_experts,
        "vocab_size": cfg.vocab_size,
    }


def run_probe(
    model: Any,
    prompts: Sequence[Any],
    device: str,
    tok: Any | None = None,
    *,
    spike_threshold: float = 448.0,
    ablation_prompts: int = 16,
    max_ablate_branches: int = 10,
    top_pairs: int = 32,
    chunk: int = 256,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Probe every prompt, ablate the spiking branches on an evenly spaced subset, and return the full result dict."""
    import probe_ablate as pa
    import probe_hooks as ph

    say = log or (lambda _m: None)
    max_len = max(len(p.tokens) for p in prompts)
    t0 = time.time()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    probe = ph.Probe(model, max_len=max_len, top_pairs=top_pairs)
    pass_seconds = []
    try:
        with torch.no_grad():
            for i, p in enumerate(prompts):
                probe.begin_prompt(i, p.tokens, p.special)
                t1 = time.time()
                model.model(input_ids=torch.as_tensor(p.tokens, device=device).unsqueeze(0))
                pass_seconds.append(time.time() - t1)
                say(f"probe {i + 1}/{len(prompts)} {p.id}: {len(p.tokens)} tokens, {pass_seconds[-1]:.1f}s")
        result = probe.result()
    finally:
        probe.close()
    peak = torch.cuda.max_memory_allocated() / GIB if device.startswith("cuda") else None
    specs = pa.select_targets(result, spike_threshold, max_ablate_branches)
    idx = sorted({round(x) for x in np.linspace(0, len(prompts) - 1, min(ablation_prompts, len(prompts)))})
    subset = [prompts[i] for i in idx]
    ablations = pa.run_ablations(model, subset, specs, device, chunk=chunk, log=say) if specs else []
    out: dict[str, Any] = {
        "layers": result["layers"],
        "ablations": ablations,
        "ablation_prompt_ids": [p.id for p in subset],
        "prompts": [
            {"id": p.id, "kind": p.kind, "n_tokens": len(p.tokens), "truncated": bool(p.truncated)} for p in prompts
        ],
        "meta": {
            "layer_indexing": "0-based (block.layer_idx); the attention layers are listed in attention_layers_0based",
            **_config_summary(model),
            "device": device,
            "weights_dtype": str(next(model.parameters()).dtype),
            "n_prompts": len(prompts),
            "n_tokens_total": int(sum(len(p.tokens) for p in prompts)),
            "max_tokens_in_a_prompt": max_len,
            "spike_threshold": spike_threshold,
            "ablation_prompts": len(subset),
            "torch": torch.__version__,
            "mamba3_path": "pure-PyTorch float32 recurrence (OLALA_TORCH_MAMBA3=1), output cast to the activation "
            "dtype; "
            "not the TileLang kernel OVH measured with",
            "moe_path": "pure-PyTorch expert loop (TorchMoEMLP), as the probe needs to see every (token, expert) pair",
            "attn_impl": sys.modules["vendor.modeling_olala"].ATTN_IMPL
            if "vendor.modeling_olala" in sys.modules
            else None,
            "seconds_probe": time.time() - t0,
            "seconds_per_pass_mean": float(np.mean(pass_seconds)),
            "peak_gpu_gib": peak,
            "memory_estimate_gib_4096_tokens": {
                a: estimate_memory_gib(model.config, 4096, attn=a) for a in ("flash", "eager")
            },
        },
    }
    if tok is not None:
        ids: set[int] = set()
        _collect_token_ids(out, ids)
        out["token_text"] = {str(t): tok.decode(t) for t in sorted(ids)}
    return out


# ------------------------------------------------------------------------------------------------ command line
SINK_TOKEN_ID = (
    151643  # <|endoftext|>: prepended by the training data loader (OVH report, App. D) and first in the template
)
TINY_PROMPTS = [
    {"id": "tiny-chat-0", "kind": "chat", "messages": [{"role": "user", "content": "What is 12 times 7?"}]},
    {"id": "tiny-chat-1", "kind": "chat", "messages": [{"role": "user", "content": "Write a haiku about rain."}]},
    {"id": "tiny-plain-0", "kind": "plain", "text": "The quick brown fox jumps over the lazy dog. " * 3},
    {"id": "tiny-plain-1", "kind": "plain", "text": "def add(a, b):\n    return a + b\n" * 4},
]


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="activation_probe.py", description=__doc__.split("\n")[0])
    p.add_argument(
        "--model-dir", type=Path, help="directory with config.json, tokenizer files and the bf16 safetensors"
    )
    p.add_argument("--prompts", type=Path, help="JSON lines (see probe_prompts.py); built by build_prompts.py")
    p.add_argument(
        "--out", type=Path, required=True, help="output directory (activation_probe.json and .md are written)"
    )
    p.add_argument("--device", choices=("cuda", "cpu"), default=None, help="default: cuda when available")
    p.add_argument(
        "--max-tokens", type=int, default=4096, help="longest forward pass; longer prompts are cut (head kept)"
    )
    p.add_argument("--tiny", action="store_true", help="the repository's tiny random Olala on the CPU (for tests)")
    p.add_argument("--seed", type=int, default=0, help="seed of the tiny model")
    p.add_argument(
        "--spike-threshold", type=float, default=448.0, help="branches above this are ablated (default: e4m3 max)"
    )
    p.add_argument("--ablation-prompts", type=int, default=16, help="prompts used for the ablations (evenly spaced)")
    p.add_argument("--max-ablate-branches", type=int, default=10)
    p.add_argument("--reasoning-effort", default="high", help="passed to the chat template")
    p.add_argument("--no-bos", action="store_true", help="do not prepend the sink token to plain-text prompts")
    p.add_argument(
        "--attn-impl", choices=("fa3", "fa2", "flex", "eager"), help="sets OLALA_ATTN_IMPL (eager is memory-hungry)"
    )
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    if not args.tiny and (args.model_dir is None or args.prompts is None):
        p.error("--model-dir and --prompts are required unless --tiny is given")
    if args.max_tokens < 2:
        p.error("--max-tokens must be at least 2")
    return args


def write_outputs(result: dict[str, Any], out: Path) -> tuple[Path, Path]:
    """Write ``activation_probe.json`` (strict JSON) and ``activation_probe.md`` into ``out``."""
    import probe_claims

    out.mkdir(parents=True, exist_ok=True)
    clean = _jsonable(result)
    jpath, mpath = out / "activation_probe.json", out / "activation_probe.md"
    jpath.write_text(json.dumps(clean, indent=1, allow_nan=False) + "\n")
    mpath.write_text(probe_claims.render_markdown(clean))
    return jpath, mpath


def main(argv: Sequence[str] | None = None) -> int:
    """CLI. Returns 0 on success, 2 for a refused input."""
    import probe_prompts as pp

    args = _parse_args(argv)
    if args.attn_impl:
        os.environ["OLALA_ATTN_IMPL"] = args.attn_impl
    device = args.device or ("cuda" if torch.cuda.is_available() and not args.tiny else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("error: --device cuda but no CUDA device is available", file=sys.stderr)
        return 2
    log = (lambda _m: None) if args.quiet else (lambda m: print(m, file=sys.stderr, flush=True))
    try:
        if args.tiny:
            model = build_tiny_model(args.seed).to(device)
            tok: Any = pp.ByteChatTokenizer()
            rows = pp.read_prompts(args.prompts) if args.prompts else TINY_PROMPTS
            bos = None if args.no_bos else pp.ByteChatTokenizer.BOS
        else:
            model = load_model(args.model_dir, device)
            tok = pp.HFPromptTokenizer.from_dir(args.model_dir, args.reasoning_effort)
            rows = pp.read_prompts(args.prompts)
            bos = None if args.no_bos else SINK_TOKEN_ID
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    prompts = pp.tokenize_prompts(rows, tok, args.max_tokens, bos)
    log(f"{len(prompts)} prompts, {sum(len(p.tokens) for p in prompts)} tokens, device {device}")
    result = run_probe(
        model,
        prompts,
        device,
        tok,
        spike_threshold=args.spike_threshold,
        ablation_prompts=args.ablation_prompts,
        max_ablate_branches=args.max_ablate_branches,
        log=log,
    )
    result["meta"]["model_dir"] = str(args.model_dir) if args.model_dir else "tiny random model"
    result["meta"]["tiny"] = bool(args.tiny)
    result["meta"]["modeling_sha256"] = hashlib.sha256(
        (GPTQ_DIR / "vendor" / "modeling_olala.py").read_bytes()
    ).hexdigest()
    jpath, mpath = write_outputs(result, args.out)
    log(f"wrote {jpath} and {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
