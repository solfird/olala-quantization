"""POD ONLY (never run on the Mac): check that the fork parses the written quantization_config.

Imports compressed_tensors and the OVH vLLM fork's compressed-tensors module, so it must run where
the fork is installed (a container or pod). It loads NO weights and needs no GPU.

Usage: python tools/pod_check_config.py --ckpt DIR_WITH_CONVERTED_CHECKPOINT

Checks, for the converted checkpoint in DIR:
1. ``QuantizationArgs.model_validate`` accepts the written weights / input_activations blocks;
2. ``CompressedTensorsConfig.from_config`` accepts the whole block and ``_get_scheme_from_parts``
   picks the scheme the DECISIONS.md table predicts;
3. for all 36 layers, the vLLM module names that quantization_config.ignore matches are exactly the
   modules whose tensors the converter left in bf16 (dense modules by exact name, routed experts by
   the ``<layer>.<expert>.<proj>`` names compressed_tensors_moe.py:36-49 builds, including the
   literal ``...0.None`` for the absent up projection).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from olala_quant.recipes import classify_tensor


def main() -> int:
    """Run the checks; exit 1 on any mismatch."""
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    args = p.parse_args()
    from compressed_tensors.quantization import QuantizationArgs  # type: ignore[import-not-found]
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # type: ignore[import-not-found]
        CompressedTensorsConfig,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.utils import (  # type: ignore[import-not-found]
        should_ignore_layer,
    )

    cfg = json.loads((args.ckpt / "config.json").read_text())
    info = json.loads((args.ckpt / "quantization_info.json").read_text())
    qc = cfg["quantization_config"]
    group = qc["config_groups"]["group_0"]
    weight_q = QuantizationArgs.model_validate(group["weights"])
    input_q = (
        QuantizationArgs.model_validate(group["input_activations"]) if group["input_activations"] else None
    )
    ct = CompressedTensorsConfig.from_config(json.loads(json.dumps(qc)))
    scheme = ct._get_scheme_from_parts(
        weight_quant=weight_q, input_quant=input_q, format=qc["format"], layer_name="probe"
    )
    print(f"dense scheme chosen: {type(scheme).__name__}")

    ignored_hf = set(info["ignore"]["quantizable_ignored"])
    problems = []
    n_layers = int(cfg["num_hidden_layers"])
    for layer in range(n_layers):
        probes: dict[str, list[str]] = {}
        for tail in (
            "mlp.experts.experts.weight",
            "mlp.experts.output_experts.weight",
            "mlp.shared_experts.fc_1.weight",
            "mlp.shared_experts.fc_2.weight",
            "mlp.down_proj.weight",
            "mlp.up_proj.weight",
            "mixer_proj.weight",
            "gate_proj.weight",
        ):
            hf = f"model.layers.{layer}.{tail}"
            _, spec, _ = classify_tensor(hf)
            if spec is None or (tail == "gate_proj.weight" and cfg["layers_config"][layer] != "V"):
                continue
            module = spec.vllm_module.format(L=layer)
            names = (
                [f"{module}.0.up_proj", f"{module}.0.None", f"{module}.0.down_proj"]
                if spec.routed
                else [module]
            )
            probes[hf] = names
        for hf, names in probes.items():
            got = [should_ignore_layer(n, ignore=ct.ignore) for n in names]
            want = hf in ignored_hf
            if any(got) != want or len(set(got)) != 1:
                problems.append(f"{hf}: vLLM ignore says {got}, converter left bf16={want}")
    print("ignore-list agreement over all layers:", "OK" if not problems else "FAILED")
    for line in problems:
        print("  ", line)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
