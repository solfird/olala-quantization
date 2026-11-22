# quant: data-free FP8 and NVFP4 for Olala-v1.0-mini-7A1B, plus the vLLM loader patch

Two deliverables, both written without a GPU and without the weights:

* **(A) `olala_quant/`**: converts the bf16 checkpoint into FP8 (per-output-channel scales, dynamic per-token
  activations at serve time) or NVFP4 weight-only (W4A16) checkpoints, and verifies them.
* **(B) `fork-patch/`**: a unified diff for the OVH vLLM fork (`olala-v0.33`, `ebbadef`) so that those
  checkpoints load correctly, and fail loudly when they do not.

Read `DECISIONS.md` for every format decision with its source line and the ranked list of guesses. Nothing here
was run against vLLM, a GPU or the real weights: see "What is and is not verified".

## Layout

| Path | What |
|---|---|
| `olala_quant/recipes.py` | Recipes as data (`fp8`, `nvfp4`, declared `nvfp4_w4a4`), quantized-tensor families, default ignore list I |
| `olala_quant/fp8.py`, `nvfp4.py` | Pure-PyTorch quantize / dequantize for 2D `[out,in]` and fused 3D `[E,out,in]` tensors, expert-chunked |
| `olala_quant/plan.py` | Header-only plan: classification of the 945 tensors, ignore-list resolution and refusal, output names and sizes |
| `olala_quant/convert.py` | CLI: shard-by-shard conversion, `config.json`, auxiliary files, `quantization_info.json` |
| `olala_quant/verify.py` | CLI: checks every output tensor against the source, non-zero exit on any violation; `--calibrated` for error-compensated (G1) checkpoints |
| `olala_quant/budget.py` | Projected sizes from the real header (every size quoted in the docs) |
| `olala_quant/weight_stats.py` | CLI (`python -m olala_quant.weight_stats --src DIR --out DIR [--device cuda\|cpu] [--layers A:B] [--max-tensors N]`): per-tensor weight distribution (kurtosis, max/RMS, outliers, channel and group ranges) and round-to-nearest error for NVFP4, FP8, INT4 g128 and MXFP4; writes `weight_stats.json` and `weight_stats.md` headed by the checkpoint index sha256; definitions in the module docstring |
| `fork-patch/olala-quant-loader.patch` | The patch (adds `olala_quant_loader.py`, rewrites `load_weights` in `olala.py`) |
| `fork-patch/olala_quant_loader.py` | The standalone, vLLM-free logic the patch adds (tested from `tests/`) |
| `fork-patch/olala_py.diff`, `build_patch.sh` | Hand-edited part of the patch, and the script that rebuilds the patch in a throw-away clone under `work/` |
| `tools/` | `pod_check_config.py` (pod only), `bench_quantize.py`, `mutation_check.py`, `regenerate_reports.sh` |
| `reports/` | Script-generated numbers: `budget.json`, `bench_quantize_cpu.json`, `mutation_report.json` |
| `gptq/` | G1-lite: calibrated NVFP4 codes for the routed experts (GPTQ with routing-aware Hessians, N2 scales), its own uv project; see `gptq/README.md` |
| `tests/` | 109 tests (122 with the calibrated-mode tests); `tests/data/olala_header.json` is the real safetensors header (945 tensors) |

## Setup and tests (Mac CPU, no network needed after `uv sync`)

```bash
cd quant
uv sync                       # python 3.12, torch 2.14.0 (the fork's pin), safetensors 0.8.0, pinned dev tools
uv run pytest                 # 109 tests, about 2 s
uv run ruff check . && uv run ruff format --check .
./tools/regenerate_reports.sh # budget table, CPU benchmark, mutation check (23 mutations, all caught)
```

On the pod, install without touching the pod's torch: `cd quant && python -m pip install --no-deps -e .`
(needs the pod's `torch`, `safetensors`, `numpy`; the fork's image already has them with Python 3.12).

## Convert and verify

```bash
# FP8 per-channel, ignore list I (default), math on the GPU
python -m olala_quant.convert --src /path/olala-bf16 --dst /path/olala-fp8  --recipe fp8   --device cuda
python -m olala_quant.verify  --src /path/olala-bf16 --dst /path/olala-fp8  --report /path/olala-fp8.verify.json

# NVFP4 weight-only, ignore list I
python -m olala_quant.convert --src /path/olala-bf16 --dst /path/olala-nvfp4 --recipe nvfp4 --device cuda
python -m olala_quant.verify  --src /path/olala-bf16 --dst /path/olala-nvfp4

# without the ignore list (arm N1 weights; add --device cuda as above)
python -m olala_quant.convert --src ... --dst ... --recipe nvfp4 --no-ignore
# a custom list: JSON array of fnmatch patterns over HF tensor names (or {"patterns": [...]})
python -m olala_quant.convert --src ... --dst ... --recipe fp8 --ignore-file my_ignore.json
# projected size only, writes nothing
python -m olala_quant.convert --src ... --dst /tmp/x --recipe fp8 --dry-run
# component ablation: quantize ONLY this part, every other quantizable tensor stays bf16 and is listed in `ignore`
python -m olala_quant.convert --src ... --dst ... --recipe nvfp4 --device cuda --select experts:layers=12-23
```

### Component ablation (`--select SPEC`, `olala_quant/ablation.py`)

Which part of the model causes the NVFP4 loss? `--select` quantizes only the named part and puts every other
quantizable tensor on the ignore list, exactly as list I is (the plan keeps it bf16, `quantization_config.ignore` names
its vLLM modules and the `re:` regex of the routed experts, `verify` re-derives the same plan). List I (or the list of
`--ignore-file`) stays bf16 whatever the selection; `--select` with `--no-ignore` is refused. Without `--select` the
output is byte for byte what it was before (regression test on a golden hash). SPEC is `NAME` or `NAME:layers=A-B[,N...]`
(both ends included): `all` (the N2 recipe, the control), `experts` (routed experts), `non-experts`, `shared-experts`,
`latent` (the latent reduce/expand projections), `attention` (`mixer_proj` and `gate_proj` of the 7 attention layers),
`mamba` (`mixer_proj` of the Mamba-3 layers); the five components partition the 259 quantizable tensors. The selection is
recorded under `selection` in `quantization_info.json` (spec, base list, deselected tensors, expected counts).
`python -m olala_quant.ablation expected --header tests/data/olala_header.json --select SPEC` prints the predicted
counts: `quantized_layers` is the N of the server's `Quantizing N layers of types: ...` line (the up and down routed tensors
of a layer are one module; list I included: `all` is 200, `experts` 32, `experts:layers=0-11` 9, `non-experts` 168,
`attention` 14, `mamba` 26). A server that loads a checkpoint prints no such line, so `ablation check --src ... --dst ...
--select SPEC [--server-log F]` also counts the quantized tensors on disk and compares `config.json`'s ignore list.

Exit codes: `convert` returns 0 on success and 2 when it refuses (ignore pattern that matches nothing, ignore
list that excludes nothing quantizable, non-empty destination, NaN or infinity in the source, unimplemented
recipe). `verify` returns 0 clean, 1 on any violation (listed by check name and tensor), 2 on unusable input.
`--device cpu` works on a 64 GB box: memory is one output shard (at most 5 GB) plus about 1.3 GB of scratch,
measured below.

What the converter writes (`<dst>`): `model-0000i-of-0000N.safetensors` (each at most 5 GB),
`model.safetensors.index.json`, `config.json` (source plus `quantization_config`), the tokenizer / chat template
/ remote-code / licence files copied byte for byte, and `quantization_info.json` (recipe, resolved ignore list,
counts and bytes per category, bits per weight, per-tensor max abs error and relative Frobenius error, scale
statistics, versions, UTC time, sha256 of the source index or, for a single-file source, of its header).

### Tensor layout of the output

| Recipe | Quantized tensor `X.weight` becomes | Fused experts `[256, out, in]` |
|---|---|---|
| `fp8` | `X.weight` float8_e4m3fn (same shape), `X.weight_scale` float32 `[out, 1]` | `[256, out, in]` and `[256, out, 1]` |
| `nvfp4` | `X.weight_packed` uint8 `[out, in/2]`, `X.weight_scale` float8_e4m3fn `[out, in/16]`, `X.weight_global_scale` float32 `[1]` | `[256, out, in/2]`, `[256, out, in/16]`, `[256]` (one per expert) |

Quantized set (what the fork can consume): routed experts (`mlp.experts.experts.weight`,
`mlp.experts.output_experts.weight`), shared expert `fc_1` / `fc_2`, latent `down_proj` / `up_proj`,
`mixer_proj` of all layers, attention `gate_proj`: 259 tensors, 5,884,084,224 parameters, 86.9 % of the model.
Copied unchanged: Mamba `in_proj*`, attention q/k/v-side projections, router, embedding, `lm_head`, norms and
all F32 tensors. Ignore list I keeps 27 of the 259 in bf16 (list: `olala_quant.recipes.DEFAULT_IGNORE_I`).

### Projected sizes (script: `python -m olala_quant.budget`, file `reports/budget.json`)

Source: 6,773,124,360 parameters, 13.580 GB, 16.04 bits per weight.

| Recipe | Ignore list | Quantized tensors | Size (GB) | Bits per weight |
|---|---|---|---|---|
| fp8 | none | 259 | 7.739 | 9.141 |
| fp8 | I | 232 | 8.376 | 9.893 |
| nvfp4 | none | 259 | 5.121 | 6.049 |
| nvfp4 | I | 232 | 6.044 | 7.139 |

Quantized weights alone cost 8.06 bits (FP8, with the fp32 scale per row) and 4.50 bits (NVFP4). List I leaves
four whole MoE layers in bf16, which is why it costs 0.64 to 0.92 GB.

CPU speed on this Mac (random `[256, 768, 384]` bf16 tensor, 12 threads, torch 2.14.0;
`reports/bench_quantize_cpu.json`): FP8 569 Mparams/s, NVFP4 254 Mparams/s, peak resident memory 1.2 to 1.3 GB
including the 0.96 GB input. Extrapolating the NVFP4 rate to the 5.88 B quantized parameters gives about 23 s of
arithmetic; reading 13.6 GB and writing the shards dominates. That is a projection, not a measurement.

## The loader patch

```bash
cd /path/to/olala-vllm-fork               # branch olala-v0.33, commit ebbadef (or a later one: apply --check first)
git apply --check /path/to/quant/fork-patch/olala-quant-loader.patch
git apply /path/to/quant/fork-patch/olala-quant-loader.patch
python -m py_compile vllm/model_executor/models/olala.py vllm/model_executor/models/olala_quant_loader.py
```

It touches two files: new `vllm/model_executor/models/olala_quant_loader.py` (pure logic, no vLLM import) and
`vllm/model_executor/models/olala.py` (`_expert_params` becomes `_routed_experts`; `load_weights` is rebuilt on
`_load_one_weight` and `_finalize_quantized_params`). What it changes:

1. **Loads the quantized tensors.** Dense suffixes follow their module (`mlp.down_proj.weight_scale` goes to
   `mlp.fc1_latent_proj.weight_scale`, and so on); fused expert tensors are scattered expert by expert through the
   existing `RoutedExperts.weight_loader` into `w13_weight`, `w13_weight_scale` (FP8, `[E, N, 1]`) or
   `w13_weight_packed`, `w13_weight_scale`, `w13_weight_global_scale` (NVFP4), and the `w2_*` equivalents.
2. **Fails loudly.** A checkpoint tensor nothing consumed raises `OlalaQuantLoadError` listing every name (the
   `warning_once(... skipping)` is gone); a tensor crossing the quantized-storage boundary (fp8 or packed vs bf16)
   raises; a wrong shape or expert count raises; a parameter of a quantized module that received nothing raises.
   `OLALA_ALLOW_UNCONSUMED=<regex>` tolerates named, known-harmless tensors and logs them.
3. **Selects the quantization method from `config.json`**, with no code change: `fp8` selects
   `CompressedTensorsW8A8Fp8` / `CompressedTensorsW8A8Fp8MoEMethod`; `nvfp4` selects
   `CompressedTensorsW4A4Fp4(use_a16=True)` / `CompressedTensorsW4A4Nvfp4MoEMethod(use_a16=True)`. The table, the
   reason, and how the online `fp8_per_channel` path reconciles with it are in `DECISIONS.md` section 1.

Rebuild the patch after editing `fork-patch/olala_quant_loader.py` or `fork-patch/olala_py.diff`:
`./fork-patch/build_patch.sh` (clones the fork into `work/`, never writes into `recon/src`).

## Serving on the pod

Flags are from PROTOCOL.md section 3 and `recon/quantizability.md`; check them against OVH's README on the pod.

```bash
# the checkpoint carries its quantization_config: no --quantization flag
vllm serve /path/olala-fp8 --trust-remote-code --max-num-seqs 256 --reasoning-parser olala

# reference for FP8, bf16 weights quantized on the fly with the SAME ignore list (fnmatch globs are valid here)
vllm serve /path/olala-bf16 --trust-remote-code --quantization fp8_per_channel \
  --quantization-config '{"ignore":["model.layers.1.mixer_proj","model.layers.7.mixer_proj","model.layers.23.mixer_proj","model.layers.0.mlp.*","model.layers.1.mlp.*","model.layers.5.mlp.*","model.layers.19.mlp.*"]}'
```

### First pod test (order matters; details and pass criteria in `DECISIONS.md` section 6)

1. `python tools/pod_check_config.py --ckpt /path/olala-fp8` (no GPU, no weights): the written
   `quantization_config` parses and `ignore` agrees with the converter's list on all 36 layers.
2. Apply the patch; load the **bf16** checkpoint; greedy logits must equal the unpatched fork (regression, and the
   test of the "raise on any unconsumed tensor" policy).
3. Load FP8; read the MoE backend log line; one request; compare top-20 logprobs with the online reference above.
4. Same for NVFP4 weight-only.

## What is and is not verified

Verified here (CPU, torch 2.14.0, 109 tests, 23 of 23 mutations of the guards caught, ruff clean):

* Quantizers: exact round trip on representable values, bitwise equality with a slow pure-Python reference for
  the NVFP4 codes and e4m3 scale bytes, FP8 against a row-by-row e4m3 table search, chunk invariance of the 3D
  paths, scale-range handling (`n_overflow`, `n_flushed`, `n_subnormal`, max and min block scale are reported).
* Plan and recipes on the REAL header: 945 tensors all classified, 259 quantizable (5,884,084,224 parameters),
  list I resolves to 27, sizes match an independent computation.
* Converter and verifier on tiny fake checkpoints with the real names: sharding, index, config, auxiliary files,
  refusals, cleanup after an error; every kind of damage tried is detected by `verify`.
* Loader logic without importing vLLM: name routing for all 945 names, dtype and shape contract, scatter into
  fakes that mirror `RoutedExperts.weight_loader`, unconsumed-tensor report, and a contract test between what the
  converter writes for the real model and the parameters the fork's quantization methods create.
* The patch applies with `git apply --check` to the pristine fork file and to a clean clone, compiles with
  `py_compile`, and its `load_weights` calls the loud logic (AST test).

NOT verified (needs the fork running, a GPU, or the weights):

* That vLLM really builds the parameters I assume, that `RoutedExperts.weight_loader` accepts the scales as I read
  it, and that a quantized model loads at all (risks R2, R3 in `DECISIONS.md`).
* Which MoE and dense kernels are chosen on the RTX PRO 4500 (sm_120) and whether any of them silently drops the
  per-channel scales (R1, R7).
* That `config.json`'s `quantization_config` is accepted by the real `compressed_tensors` library (R4).
* Quality: error bounds on real weights, perplexity, KL, accuracy (R5). Nothing in this directory claims any.

## W4A4 (declared, not implemented): what it additionally needs

`nvfp4_w4a4` (arms N1 and N2) writes the same weight tensors plus static activation global scales, which cannot be
made from weights:

* `input_global_scale` (float32 `[1]`) for each quantized dense Linear module (168 with list I, 187 without): `448*6/amax` of the
  activation entering the module over the calibration set (`compressed_tensors_w4a4_nvfp4.py:124-135`).
* `w13_input_global_scale` `[E]` and `w2_input_global_scale` `[E]` for every routed-expert layer, per expert, from
  the tokens the router actually sends to each expert (`compressed_tensors_moe_w4a4_nvfp4.py:170-189`): this needs a
  forward pass through the real model with hooks on the latent expert input and the ReLU^2 output.
* The calibration set of PROTOCOL.md section 5 (512 samples up to 2048 tokens, chat template at
  `reasoning_effort=medium`, disjoint from test items) and a GPU run of the OVH modeling code.
* The loader and `quantization_config` already handle it (`input_activations` NVFP4, `use_a16=False`); the patch
  would then expect the input scales instead of filling 1.0.
* Caution from the Appendix C spikes (1e7 to 1e16): an activation amax that large puts every ordinary block scale
  below the e4m3 minimum (about 0.002), so blocks flush to zero; this is the mechanism arm N1 tests.
