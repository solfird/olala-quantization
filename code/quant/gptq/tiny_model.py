"""A TINY random Olala model (4 layers, hidden 64, 16 experts) for tests and for the pod smoke check.

It runs the vendored OVH modeling code on the CPU with the real layer pattern idea (Mamba-3 layers with one attention
layer among them, every layer followed by a latent MoE) and the real ``config.json`` flags; only the sizes change. The
weights are random, with the router and the experts scaled so that routing depends on the data and the experts are not
vanishingly small. The written checkpoint has the real tensor names and dtype policy (bf16 for the matrices, float32
for the scalars, the router and the Mamba parameters), so ``olala_quant`` converts and verifies it like the real one.
"""

from __future__ import annotations

import json
import random
import re
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from olala_model import VENDOR_DIR, VENDORED_FILES, load_olala_modules

REAL_CONFIG = VENDOR_DIR / "config.json"  # the real config.json of the model repository, copied verbatim
TINY_OVERRIDES: dict[str, Any] = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "layers_config": "MMVM",
    "num_hidden_layers": 4,
    "head_dim": 16,
    "num_attention_heads": 4,
    "num_signal_heads_diff": 2,
    "tpa_rank": 2,
    "mamba_d_state": 16,
    "moe_num_routed_experts": 16,
    "num_experts": 16,
    "moe_num_active_experts": 4,
    "num_experts_per_tok": 4,
    "moe_routed_input_dim": 32,
    "moe_routed_intermediate_size": 64,
    "moe_shared_intermediate_size": 64,
    "vocab_size": 512,
    "num_key_value_heads": 0,
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "base_depth": 2,
}
# What the tiny checkpoint keeps in float32 (the real checkpoint does the same, see recon/src/hf/tensors.tsv).
_F32 = re.compile(
    r"(geodesic_.*\.(scale|bias|prosres_scalar)$|mixer\.(B_bias|C_bias|D|dt_bias|in_proj_mimo_x|in_proj_mimo_z"
    r"|out_proj_mimo|softmax_scaler)$|mlp\.expert_bias$|mlp\.moe_gate\.weight$)"
)
#: Ignore list for the tiny model: layer 0 MLP and the layer 1 mixer_proj play the role of list I.
TINY_IGNORE = ("model.layers.0.mlp.*", "model.layers.1.mixer_proj.weight")
TINY_WINDOW = 64


def tiny_config_dict() -> dict[str, Any]:
    """The real ``config.json`` with the tiny sizes substituted (and the remote-code keys removed)."""
    cfg = json.loads(REAL_CONFIG.read_text())
    cfg.update(TINY_OVERRIDES)
    for key in ("auto_map", "architectures", "transformers_version", "dtype"):
        cfg.pop(key, None)
    return cfg


def build_tiny_model(
    seed: int = 0, expert_std: float = 0.5, proj_std: float = 0.5, head_std: float | None = 1.0
) -> Any:
    """Build the tiny ``OlalaForCausalLM`` with random weights (float32 parameters).

    ``expert_std`` and ``proj_std`` scale the routed experts and the latent projections (larger values make the MoE
    branch matter for the output); ``head_std`` re-initialises ``lm_head`` (``None``: the
    model's own init). The defaults make the MoE branch and the logits matter.
    """
    modeling, configuration = load_olala_modules()
    torch.manual_seed(seed)
    model = modeling.OlalaForCausalLM(configuration.OlalaConfig(**tiny_config_dict()))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("moe_gate.weight"):
                p.normal_(0.0, 0.6)
            elif name.endswith("experts.experts.weight") or name.endswith("experts.output_experts.weight"):
                p.normal_(0.0, expert_std)
            elif name.endswith(("mlp.down_proj.weight", "mlp.up_proj.weight")):
                p.normal_(0.0, proj_std)
            elif name == "lm_head.weight" and head_std is not None:
                p.normal_(0.0, head_std)
    return model.eval()


def checkpoint_tensors(model: Any) -> dict[str, torch.Tensor]:
    """The state dict with the real dtype policy: bf16 matrices, float32 for the names in ``_F32``."""
    out = {}
    for name, t in model.state_dict().items():
        out[name] = t.detach().clone().to(torch.float32 if _F32.search(name) else torch.bfloat16).contiguous()
    return out


def write_tiny_checkpoint(directory: Path, seed: int = 0, **model_kwargs: float | None) -> Path:
    """Write the tiny model as a safetensors checkpoint with ``config.json`` and the vendored remote code."""
    directory.mkdir(parents=True, exist_ok=True)
    model = build_tiny_model(seed, **model_kwargs)
    save_file(checkpoint_tensors(model), str(directory / "model.safetensors"), metadata={"format": "pt"})
    cfg = tiny_config_dict()
    cfg["architectures"] = ["OlalaForCausalLM"]
    cfg["dtype"] = "bfloat16"
    (directory / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    for name in VENDORED_FILES:
        shutil.copy2(VENDOR_DIR / name, directory / name)
    return directory


def random_windows(n_windows: int, window: int = TINY_WINDOW, seed: int = 0) -> torch.Tensor:
    """Random token windows ``[n_windows, window]`` (ids 3 and up, below the tiny vocabulary)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, 256, (n_windows, window), generator=g)


def word_docs(n: int, seed: int, tag: str = "toy", lexicon: int = 0, vocab: int = 400, words: int = 200) -> list[Any]:
    """``n`` toy documents of random words from the ``lexicon``-th lexicon of ``vocab`` random words.

    Documents with the same ``lexicon`` follow one distribution (calibration and held-out evaluation text of a run);
    ``seed`` picks the words of each document.
    """
    import calib_data

    lex_rng = random.Random(f"lexicon|{lexicon}")
    lex = [
        "".join(lex_rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(lex_rng.randint(2, 9)))
        for _ in range(vocab)
    ]
    rng = random.Random(f"docs|{seed}")
    return [calib_data.Document(tag, f"d{i}", " ".join(rng.choice(lex) for _ in range(words))) for i in range(n)]
