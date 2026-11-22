# analysis/: the activation probe

`activation_probe.py` reproduces OVH's Appendix C activation-outlier claims on the released Olala-v1.0-mini-7A1B weights
(Mamba-3 layers 1, 7, 23 with single SSM heads 23, 26, 40; MoE layers 0, 1, 5, 19 spiking through the squared ReLU) and
measures what they mean for quantization. It runs the vendored OVH modeling code from `../quant/gptq/vendor` (pure-PyTorch
Mamba-3 and expert paths), never imports the OVH vLLM fork, never touches the network.

**Layer numbers are 0-based** everywhere (`block.layer_idx`; the attention layers are 4, 9, 14, 19, 24, 29, 34). The
output states this and tests a 1-based reading of OVH's numbers explicitly (claim A1b).

## Files

| Path | What |
|---|---|
| `activation_probe.py` | CLI, model loading (meta device + safetensors, dtypes as stored), `run_probe`, memory estimate |
| `probe_hooks.py` | `Probe` (the hooks on every layer) and the ablation hooks |
| `probe_stats.py` | per-site statistics and their pooling over prompts |
| `probe_ablate.py` | KL / top-1 / delta-NLL against the unmodified run, ablation target selection |
| `probe_claims.py` | the claim-by-claim table and the markdown report |
| `probe_prompts.py`, `build_prompts.py` | prompt file format, tokenizers; the default 64 + 16 prompt file |
| `tests/`, `tools/mutation_check.py` | 90+ tests on the tiny random Olala (CPU); 36 weakenings of the guards |

## Use

```bash
cd analysis && uv sync
uv run pytest && uv run ruff check . && uv run ruff format --check .
uv run python tools/mutation_check.py                 # about 4 minutes

# default prompt file: 64 task inputs through evals/run_tasks.py's factories (no model) + 16 plain-text prompts
uv run --project ../evals python build_prompts.py --out prompts_default.jsonl

# the tiny random model, CPU
uv run python activation_probe.py --tiny --out out-tiny/

# the real weights (pod, GPU, bf16): writes out/activation_probe.json and out/activation_probe.md
uv run python activation_probe.py --model-dir /workspace/Olala-v1.0-mini-7A1B --prompts prompts_default.jsonl \
    --out out/ --device cuda --max-tokens 4096 [--attn-impl fa2|fa3|flex|eager]
```

Prompt file: JSON lines, `{"id", "kind": "chat", "messages": [...]}` (rendered with the model's chat template, generation
prompt included, `reasoning_effort` passed through) or `{"id", "kind": "plain", "text"}` (the sink token 151643 is
prepended unless `--no-bos`). Prompts longer than `--max-tokens` are cut (head kept) and flagged in the JSON.

## Reading the output

Verdicts are `consistent` (the measured value satisfies the rule printed next to it), `partly consistent`,
`not consistent` or `not judged` (OVH gives no checkable definition, or the layer does not exist in the model). They speak for
this prompt set, these sequence lengths and this forward path only; the caveats section of the report lists what differs
from OVH's own measurement (pure-PyTorch float32 recurrence instead of the TileLang kernel, chat prompts instead of
512-token wikitext passages, no recurrent-state readout).
