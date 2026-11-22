"""Tiny random stand-ins for the Olala checkpoint, with the real tensor names.

Dimensions are scaled down (hidden 64, 8 experts, ...) but every name, dtype and tensor rank
follows the safetensors header of Olala-v1.0-mini-7A1B (recon/src/hf/header.json).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

HIDDEN = 64
LATENT = 32
EXPERT_FF = 64
SHARED_FF = 64
N_EXPERTS = 8
MIXER_OUT = 96  # multiple of 16 so NVFP4 applies
VOCAB = 128
LAYERS = "MMVM"

AUX_FILES = {
    "tokenizer_config.json": '{"tokenizer_class": "Fake"}',
    "chat_template.jinja": "{{ messages }}",
    "configuration_olala.py": "# remote code stand-in\n",
    "modeling_olala.py": "# remote code stand-in\n",
    "generation_config.json": '{"do_sample": true}',
    "tokenizer.json": '{"version": "1.0"}',
    "LICENSE": "Apache-2.0 stand-in\n",
}


def make_tensors(seed: int = 0, layers: str = LAYERS) -> dict[str, torch.Tensor]:
    """Return a dict of random tensors named like the real checkpoint."""
    g = torch.Generator().manual_seed(seed)

    def bf(*shape: int, scale: float = 0.05) -> torch.Tensor:
        return (torch.randn(shape, generator=g) * scale).to(torch.bfloat16)

    def f32(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=g)

    t: dict[str, torch.Tensor] = {
        "model.embedding.weight": bf(VOCAB, HIDDEN),
        "lm_head.weight": bf(VOCAB, HIDDEN),
    }
    for i, kind in enumerate(layers):
        p = f"model.layers.{i}"
        for blk in ("geodesic_mixer", "geodesic_mlp"):
            for leaf in ("scale", "bias", "prosres_scalar"):
                t[f"{p}.{blk}.{leaf}"] = f32()
        t[f"{p}.mixer_proj.weight"] = bf(HIDDEN, MIXER_OUT)
        if kind == "M":
            t[f"{p}.mixer.in_proj.weight"] = bf(48, HIDDEN)
            t[f"{p}.mixer.in_proj_dyn.weight"] = bf(24, HIDDEN)
            for leaf in ("B_bias", "C_bias", "in_proj_mimo_x", "in_proj_mimo_z", "out_proj_mimo"):
                t[f"{p}.mixer.{leaf}"] = f32(4, 4, 8)
            t[f"{p}.mixer.D"] = f32(4)
            t[f"{p}.mixer.dt_bias"] = f32(4)
            t[f"{p}.mixer.B_norm.norm.weight"] = bf(16)
            t[f"{p}.mixer.C_norm.norm.weight"] = bf(16)
        else:
            t[f"{p}.mixer.c_q.weight"] = bf(48, HIDDEN)
            for leaf in ("W_B_k", "W_B_v"):
                t[f"{p}.mixer.{leaf}.weight"] = bf(16, HIDDEN)
            for leaf in ("W_A_k", "W_A_v", "shift_proj_k", "shift_proj_v", "lambda_proj"):
                t[f"{p}.mixer.{leaf}.weight"] = bf(4, HIDDEN)
            t[f"{p}.mixer.q_norm.norm.weight"] = bf(16)
            t[f"{p}.mixer.k_norm.norm.weight"] = bf(16)
            t[f"{p}.mixer.softmax_scaler"] = f32(4)
            t[f"{p}.gate_proj.weight"] = bf(MIXER_OUT, HIDDEN)
        # per-expert magnitude spread, so per-expert global scales matter
        spread = torch.rand(N_EXPERTS, 1, 1, generator=g) * 4 + 0.2
        t[f"{p}.mlp.experts.experts.weight"] = (
            torch.randn(N_EXPERTS, EXPERT_FF, LATENT, generator=g) * 0.03 * spread
        ).to(torch.bfloat16)
        t[f"{p}.mlp.experts.output_experts.weight"] = (
            torch.randn(N_EXPERTS, LATENT, EXPERT_FF, generator=g) * 0.03 * spread
        ).to(torch.bfloat16)
        t[f"{p}.mlp.shared_experts.fc_1.weight"] = bf(SHARED_FF, HIDDEN)
        t[f"{p}.mlp.shared_experts.fc_2.weight"] = bf(HIDDEN, SHARED_FF)
        t[f"{p}.mlp.down_proj.weight"] = bf(LATENT, HIDDEN)
        t[f"{p}.mlp.up_proj.weight"] = bf(HIDDEN, LATENT)
        t[f"{p}.mlp.moe_gate.weight"] = f32(N_EXPERTS, HIDDEN)
        t[f"{p}.mlp.expert_bias"] = f32(N_EXPERTS)
    return t


def write_fake_checkpoint(
    directory: Path, seed: int = 0, layers: str = LAYERS, shards: int = 1
) -> dict[str, torch.Tensor]:
    """Write a fake checkpoint (weights, config.json and auxiliary files) and return the tensors."""
    directory.mkdir(parents=True, exist_ok=True)
    tensors = make_tensors(seed, layers)
    if shards == 1:
        save_file(tensors, directory / "model.safetensors", metadata={"format": "pt"})
    else:
        names = sorted(tensors)
        weight_map = {}
        for s in range(shards):
            part = {n: tensors[n] for n in names[s::shards]}
            fname = f"part-{s}.safetensors"
            save_file(part, directory / fname, metadata={"format": "pt"})
            weight_map.update({n: fname for n in part})
        (directory / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map})
        )
    config = {"architectures": ["OlalaForCausalLM"], "dtype": "bfloat16", "layers_config": layers}
    (directory / "config.json").write_text(json.dumps(config, indent=2))
    for fname, text in AUX_FILES.items():
        (directory / fname).write_text(text)
    return tensors
