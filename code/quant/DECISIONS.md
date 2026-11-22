# DECISIONS: every format decision with its source line, every guess ranked by risk

Everything here was read from text (nothing from the fork was executed or imported). "Fork" is
`recon/src/vllm`, branch `olala-v0.33`, commit `ebbadef`. Paths are under
`vllm/model_executor/` unless they start with a dot. Abbreviations used in citations:

| Abbrev | Path |
|---|---|
| `CT` | `layers/quantization/compressed_tensors/compressed_tensors.py` |
| `CT-utils` | `layers/quantization/compressed_tensors/utils.py` |
| `CT-W8A8` | `layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py` |
| `CT-FP4` | `layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py` |
| `CT-MoE` | `layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe.py` |
| `CT-MoE-FP8` | `.../compressed_tensors_moe/compressed_tensors_moe_w8a8_fp8.py` |
| `CT-MoE-FP4` | `.../compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4.py` |
| `RE` | `layers/fused_moe/routed_experts.py` |
| `ON-FP8` | `layers/quantization/online/fp8.py` |
| `FP4-ref` | `layers/quantization/utils/nvfp4_emulation_utils.py` |
| `OLALA` | `models/olala.py` |

## 1. Which quantization method each recipe selects (and why)

The recipes write `config.json: quantization_config` with `quant_method: "compressed-tensors"`.
vLLM reads it from the HF config (`ModelConfig` quantization detection, standard for every
compressed-tensors model; `OlalaConfig` forwards `**kwargs` to `PretrainedConfig`,
`configuration_olala.py:183,282-287`). `CompressedTensorsConfig.from_config` (`CT:235-270`) builds the
scheme map; the layer-level choice is `CompressedTensorsConfig._get_scheme_from_parts` (`CT:783-949`).

| Recipe | `format` | Weights block | Dense Linear scheme | RoutedExperts method |
|---|---|---|---|---|
| `fp8` (arm F2) | `float-quantized` | 8-bit float, `strategy: channel`, symmetric, static | `CompressedTensorsW8A8Fp8(weight_quant, is_static_input_scheme=False)` (`CT:876-905`, needs `is_activation_quantization_format`, `CT-utils:43-50`; `_is_fp8_w8a8` `CT:581-616`) | `CompressedTensorsW8A8Fp8MoEMethod` (`CT-MoE:161-172`), per-channel and per-token branch `CT-MoE-FP8:66-87` |
| `nvfp4` (arm N3, weight-only) | `nvfp4-pack-quantized` | 4-bit float, `strategy: tensor_group`, group 16, symmetric; `input_activations: null` | `CompressedTensorsW4A4Fp4(use_a16=True)` (`CT:795-797`; `_is_nvfp4_format` `CT:420-437`) | `CompressedTensorsW4A4Nvfp4MoEMethod(use_a16=True)` (`CT-MoE:145-160`) |
| `nvfp4_w4a4` (declared, not implemented) | `nvfp4-pack-quantized` | as above, plus `input_activations` NVFP4 | `CompressedTensorsW4A4Fp4()` (`CT:804`) | same, `use_a16=False` |

Why compressed-tensors and not the other paths:

* **`quant_method: "fp8"` (serialized `Fp8Config`) is per-tensor.** Its weight key is
  `kFp8StaticTensorSym` (`quantization/fp8.py:273`) and the scale parameter is a
  `PerTensorScaleParameter` (`fp8.py:317-326`); block-wise is the only other variant (`:327-339`). It cannot
  express the per-output-channel scales the recipe asks for. Plain `--quantization fp8` also refuses the
  online use (`fp8.py:105-112`).
* **Online `--quantization fp8_per_channel`** (`Fp8PtpcOnlineLinearMethod`, `ON-FP8:377-465`;
  `Fp8PtpcOnlineMoEMethod`, `ON-FP8:767-851`) needs the bf16 weights at every load and quantizes on the
  fly. It is the reference the offline FP8 checkpoint must agree with, not the artifact.
* **ModelOpt NVFP4** (`quantization/modelopt.py`) stores `weight_scale_2` as the DEQUANT multiplier and has
  dedicated branches in the MoE loader (`RE:831-925`). compressed-tensors is the generic path
  (`RE:927-981`) and is what llm-compressor writes, which keeps the artifact interchangeable.

### Reconciling the online path with the serialized one (`fp8_per_channel` vs `float-quantized`)

Both end in the same runtime state, and differ only while loading:

| | Online `fp8_per_channel` | Offline `float-quantized` channel/token |
|---|---|---|
| Weight / activation keys | `kFp8StaticChannelSym` / `kFp8DynamicTokenSym` (`ON-FP8:385-386`, MoE `:787-788`) | same keys (`CT-W8A8:50-53,77-80`; MoE `CT-MoE-FP8:89-104`) |
| MoE backend oracle | `select_fp8_moe_backend(..., allow_vllm_cutlass=True)` (`ON-FP8:784-790`) | the same call (`CT-MoE-FP8:107-112`) |
| Parameters created at `create_weights` | bf16 `w13_weight` / `w2_weight` on the meta device, then quantized after loading (`online/moe_base.py:59-85`, `initialize_online_processing`) | fp8 `w13_weight` / `w2_weight` plus fp32 `w13_weight_scale` `[E, 2*.., 1]` and `w2_weight_scale` `[E, hidden, 1]` with `quant_method=channel` (`CT-MoE-FP8:143-209`) |
| Dense | bf16 meta `weight`, quantized after loading to `weight.t()` + `weight_scale` (`ON-FP8:426-446`) | fp8 `weight` `[out,in]` plus `weight_scale` `[out,1]` (`fp8_utils.py:1376-1411`, `CT-W8A8:117-130`) |
| Scale formula | `max(amax/448, 1/(448*512))`, true division, clamp, cast (`ON-FP8:81-106`); MoE w13 uses `ops.scaled_fp8_quant(..., use_per_token_if_dynamic=True)` (`:833-837`) | the checkpoint's values; the converter uses the same formula (`olala_quant/fp8.py`) |
| Guard on bad backends | rejects MARLIN, CPU, FLASHINFER_* because they fold scales per tensor (`ON-FP8:791-807`) | **no such guard** (see risk R1) |

How the patch reconciles them: the existing scatter code already loads a bf16 3D tensor named
`experts.experts.weight` into a parameter called `w13_weight` (`OLALA:1337-1371`). The serialized FP8
checkpoint keeps that name with dtype float8_e4m3fn, so the same branch works, and the new
`weight_scale` branch scatters `[256, out, 1]` scales into `w13_weight_scale` / `w2_weight_scale`
(`RE:943-981` dispatches on the substring "scale" and the parameter's `quant_method` attribute `channel`,
`CT-MoE-FP8:204-209`). The online path keeps working unchanged: bf16 tensors still go to bf16
(meta) parameters; the patch only refuses crossings of the quantized-storage boundary (section 3).
The best end-to-end test of the quantizer and the loader is therefore: **offline FP8 logits against online
`fp8_per_channel` logits with the same ignore list** (they should agree to float noise; see section 6).

## 2. Checkpoint format decisions

| ID | Decision | Justification |
|---|---|---|
| F1 | Checkpoint keeps the HF tensor names; quantized tensors add suffixes to the module name (`<module>.weight_scale`...). The loader renames to vLLM names. | `OLALA:1319-1325` already renames at load time; renaming in the checkpoint would break the HF names the rest of the study uses. |
| F2 | FP8: `<module>.weight` float8_e4m3fn (same name, new dtype) and `<module>.weight_scale` float32 `[out, 1]`; fused experts `[E, out, in]` and `[E, out, 1]`. | Dense: `create_fp8_scale_parameter` builds `(sum(out), 1)` for channel (`fp8_utils.py:1406-1411`); the v2 loaders assert equal shapes (`parameter.py:150`). MoE: `w13_weight_scale` `[E, N, 1]` and `w2_weight_scale` `[E, hidden, 1]` (`CT-MoE-FP8:189-203`); the channel branch does `expert_data.copy_(loaded)` (`RE:443-451`), which needs the trailing 1. |
| F3 | FP8 scale = `max(amax_row/448, 1/(448*512))`; `q = clamp(w/scale, -448, 448)` cast to e4m3fn with a true division. | Identical to the online path (`ON-FP8:81-106`), so offline and online differ only by the kernel's own scale for w13 (`:833-837`). |
| F4 | NVFP4 dense: `weight_packed` uint8 `[out, in/2]`, `weight_scale` float8_e4m3fn `[out, in/16]`, `weight_global_scale` float32 `[1]`. | `CT-FP4:53-84` (parameter shapes and dtypes; global scale has one entry per output partition, here 1). |
| F5 | NVFP4 fused experts: `weight_packed` `[E, out, in/2]`, `weight_scale` `[E, out, in/16]`, `weight_global_scale` `[E]` (one per expert). | MoE parameters: `CT-MoE-FP4:89-168`: `w13_weight_packed` `[E, I, H/2]`, `w13_weight_scale` `[E, I, H/16]`, `w13_weight_global_scale` `[E, 1]` (non-gated: one shard, `:87`), `w2_weight_global_scale` `[E]`. |
| F6 | The global scale is stored as the ENCODE scale `448*6/amax` (a divisor of the block scale), not as a multiplier. | compressed-tensors convention, inverted by the loader: `CT-FP4:110-112` ("CT stores as divisors"), MoE `CT-MoE-FP4:248,252` (`1.0 / w13_weight_global_scale`), `FP4-ref:633-653` (`invert_global_scales=True` for CT, off for ModelOpt). Dequantization is `fp4 * (block_scale * (1/g))` (`FP4-ref:576-630`, `:621`). |
| F7 | One global scale PER EXPERT for the fused expert tensors, not one per fused tensor. | The vLLM parameters are per expert (`CT-MoE-FP4:149-168`), so a per-expert scale loads without any broadcast, and it is what 256 separate Linear layers would produce under llm-compressor. A single scale for the whole tensor would also load, but would waste e4m3 range on small-magnitude experts; since e4m3 keeps 3 mantissa bits over its whole normal range the precision difference is small unless an expert is below about 1e-4 of the largest. A design choice, not a measured gain. |
| F8 | e2m1 values `0, .5, 1, 1.5, 2, 3, 4, 6`; nibble = `sign<<3 | code`; two per byte, EVEN element in the LOW nibble. Negative zero is never written (sign bit only for non-zero results). | Table `FP4-ref:23-25`; `break_fp4_bytes` stacks `(low, high)` (`:546-563`); Triton decode interleaves low then high (`:120-122`, `:346-351`). Canonical zero is my choice; every decoder maps both zeros to 0. |
| F9 | Round to nearest e2m1 with ties to the even code: thresholds 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, values above 6 saturate. | `cast_to_fp4` (`FP4-ref:695-706`) and the Triton twin `_round_to_fp4` (`:363-376`); the tests pin every tie. |
| F10 | Block scale = `g * (vec_max * (1/6))` clamped to [0, 448] and cast to e4m3; values are divided by the QUANTIZED scale (`1/(s_q * (1/g))`), clamped to +-6. A block whose scale rounds to zero is flushed to zero and counted. | `ref_nvfp4_quant` (`FP4-ref:709-722`), including the order of the float32 operations. Flushing is the mathematically forced outcome (a zero scale cannot encode anything); the count is reported (`n_flushed`). |
| F11 | `n_overflow` counts blocks whose scale exceeds 448 by more than 1e-6 relative. | With the exact tensor amax the largest block lands on 448 up to one float32 ulp; counting that as a clip would make the metric noise. Real overflow cannot occur by construction and is reported if it ever does. |
| F12 | Output shards are <= 5 GB (decimal, 5_000_000_000 bytes), planned greedily from shapes before any data is read; all outputs of one source tensor stay in one shard; a tensor larger than the cap would get its own shard. `model.safetensors.index.json` is written last. | Requirement of the brief; planning from shapes makes `model-0000i-of-0000N` names exact and keeps the memory bound explicit (one shard buffer). A missing index is therefore the signature of an interrupted run. |
| F13 | `config.json` = source config + `quantization_config`; tokenizer, chat template, remote code, LICENSE, NOTICE, generation config are copied byte for byte (any top-level regular file under 100 MB that is not weights, index or config). | Lab need: the output must serve with `--trust-remote-code` and parse the same template. |
| F14 | `quantization_info.json` records recipe, resolved ignore list, counts and bytes per category, bits per weight (all weights and quantized weights), per-tensor error and scale statistics, versions, UTC time and the sha256 of the source index; for a single-file source (no index exists) the sha256 of the safetensors header bytes (length prefix + JSON) stands in and `identity_kind` says so. | Brief. The header fixes every name, dtype, shape and offset. |

## 3. Ignore list and loader decisions

| ID | Decision | Justification |
|---|---|---|
| I1 | Ignore patterns are fnmatch patterns over HF tensor names, default list I from the brief (3 `mixer_proj`, `model.layers.{0,1,5,19}.mlp.*`). They resolve to 27 quantizable tensors of the real header (script: `tests/test_plan.py`). | Brief and `recon/quantizability.md` C. The brief says "every mlp tensor of layers 0, 1, 5, 19"; PROTOCOL.md section 2 says "down projections of the MoE blocks"; the brief is newer and wins, but see D2. |
| I2 | The converter REFUSES to run if a pattern matches no tensor, or if the list excludes no tensor that would otherwise be quantized. | Brief ("refuse if the ignore list matches nothing"), extended to single typos because a silently ineffective ignore list is the failure this study cannot afford. `--no-ignore` is the explicit way to quantize everything. |
| I3 | Ignoring one of the two routed tensors of a layer ignores both, and the converter says so (`forced` in the info file). | One `RoutedExperts` owns `w13` and `w2` under one quant method (`OLALA:296-326`; `CT-MoE:36-57` demands one scheme for every projection). It cannot be split (`recon` C.2 says the same). |
| I4 | `quantization_config.ignore` uses vLLM MODULE names: exact names for dense modules (`model.layers.1.mixer_proj`, `...mlp.fc1_latent_proj`, `...shared_experts.up_proj`), and BOTH the exact `...mlp.experts` and a `re:` regex for the routed experts, plus `lm_head`. | compressed-tensors ignore is exact-or-`re:` only (`.../utils/config_utils.py:443-452`, called without fnmatch by `CT-utils:53-70`), and RoutedExperts is matched through the names `<layer>.<expert>.<proj>` (`CT-MoE:36-49`), including the literal `...experts.0.None` for the absent up projection (Olala passes `ckpt_names=("up_proj","down_proj",None)`, `OLALA:307`). The recon's ready-to-paste ignore list uses `mlp.*` globs, which are fnmatch and only valid for the ONLINE path (`online/base.py:301-306`); pasted into a compressed-tensors config it would silently ignore nothing. |
| L1 | Dense checkpoint names are routed by module with the suffix preserved: `mlp.down_proj.<sfx>` to `mlp.fc1_latent_proj.<sfx>`, `up_proj` to `fc2_latent_proj`, `shared_experts.fc_1/fc_2` to `shared_experts.up_proj/down_proj`, `moe_gate` to `gate`. | Generalizes the exact-name table at `OLALA:1319-1325`; otherwise `down_proj.weight_scale` would hit the silent skip at `OLALA:1423-1426`. |
| L2 | Fused experts: every suffix goes through the existing per-expert `weight_loader`, with a name that contains the real parameter name (`...experts.w13_weight_packed`). | `RE:688-1002` dispatches on substrings of `weight_name`: "scale" then the parameter's `quant_method` attr (`channel`: `RE:434-459`; `group`: `RE:371-432`; `tensor`: `RE:337-353`), "weight" for the data. `OLALA:1343-1349` already documents this. |
| L3 | A dtype contract refuses to copy across the quantized-storage boundary (fp8, uint8, int8 vs bf16/fp32) and demands exact equality inside it. | `Tensor.copy_` casts silently: a bf16 `mixer_proj.weight` loaded into an fp8 parameter (ignore list and config disagree) would load garbage with no error. The default loader's completeness check is OFF for quantized models (`model_loader/default_loader.py:510-516`). |
| L4 | Every checkpoint tensor that nothing consumed is collected and raised together, by name, at the end (`OlalaQuantLoadError`). `OLALA_ALLOW_UNCONSUMED` (a regex that must fully match) tolerates named harmless tensors and logs them. | Brief (ii). The old `logger.warning_once("... skipping")` (`OLALA:1423-1426`) is gone. See R3: this also applies to the bf16 checkpoint. |
| L5 | After loading, every parameter of a quantized `LinearBase` and of every non-online `RoutedExperts` (`w13_*`, `w2_*`) must have been loaded, else raise. | Same reason as L3/L4: the strict check is disabled for quantized models, and `torch.empty` parameters would otherwise serve garbage. |
| L6 | For NVFP4 weight-only, `w13_input_global_scale` / `w2_input_global_scale` are filled with 1.0 and reported loaded. | `CT-MoE-FP4:170-189` creates them for every NVFP4 variant and `process_weights_after_loading` computes `1.0 / layer.w13_input_global_scale` unconditionally (`:248-253`). A W4A16 checkpoint cannot carry them. 1.0 is neutral and never read by weight-only kernels. |
| L7 | The serialized `experts_int8` layout is preserved: when `w13_weight_scale` does not exist but `w13_scale` does, the scale is copied as before. | `OLALA:1372-1392`. |
| L8 | `_expert_params()` (which read `routed.w13_weight`) is replaced by `_routed_experts()`; parameters are fetched by attribute at load time. | NVFP4 layers have no `w13_weight` until `process_weights_after_loading` renames `w13_weight_packed` (`CT-MoE-FP4:207-221`). |
| L9 | The per-expert shape check requires the same rank and no dimension larger than the parameter's (a smaller one is accepted: padded parameters are narrowed by the loader, `RE:494-527`); it is skipped under tensor parallelism, where only TP=1 is the supported target. | The weight loader narrows by rank (`RE:529-578`) and for padding; strict equality would refuse loads vLLM itself performs. |

## 4. Verifier bounds (guesses)

| ID | Bound | Basis |
|---|---|---|
| V1 | FP8: relative Frobenius error <= 0.04 per tensor; per element, error <= 16 x row scale (half the e4m3 spacing at 256..448). | Measured 0.026 on Gaussian weights (`reports/bench_quantize_cpu.json`); the per-element bound is exact theory. Real heavy-tailed weights are not measured: GUESS. |
| V2 | NVFP4: relative Frobenius error <= 0.25 per tensor; per block, max error <= 1.001 x block scale (half the largest e2m1 gap, 2.0, is 1.0 in scaled units); flushed blocks <= `6*2^-10/g`. | Measured 0.095 on Gaussian weights. Real weights with outlier blocks may exceed 0.25: GUESS; the verifier takes `--max-rel-fro-nvfp4` and reports the worst tensors. |
| V3 | FP8 scale must equal `max(amax/448, MIN)` within 1e-5; NVFP4 global scale must equal `448*6/amax` within 1e-5 and the largest block scale of each tensor (expert) must be 448. | Re-derivation from the source; catches a wrong global scale even when the reconstruction looks fine. |
| V4 | Recorded errors in `quantization_info.json` must equal the recomputed ones (rtol 1e-3). | "Every reported number generated by script" is then checked, not trusted. |

## 5. Risks and guesses, ranked (highest risk of being wrong first)

Each line says what is unknown and what the pod smoke test must look at.

1. **R1. MoE kernel backend per quantization (GPU only; could be silently wrong for FP8).**
   * FP8: the oracle picks a backend by `is_supported_config` over a priority list
     (`fused_moe/oracle/fp8.py:79-105`). For the online path OVH-adjacent code REJECTS MARLIN, CPU and
     FLASHINFER_* backends because "they fold scales into a per-tensor alpha" (`ON-FP8:791-807`); the
     compressed-tensors MoE path (`CT-MoE-FP8:107-112`) has no such guard. If one of them is selected for
     per-channel weights the model would run with wrong scales, not fail. recon D.2 also notes CUTLASS FP8 does not support
     non-gated experts (`cutlass_moe.py:282,317`), so only a Triton-class backend can serve ReLU^2 experts.
   * NVFP4: candidates are FlashInfer TRTLLM/CUTLASS/CuTe-DSL, vLLM CUTLASS, Marlin, Humming, emulation
     (`oracle/nvfp4.py:203-212`); which of them accept non-gated `relu2` on sm_120 (RTX PRO 4500) was not traced.
   * Check: the logs `Using '<X>' Fp8 MoE backend` and `Using '<X>' NvFp4 MoE backend`; then FP8 offline vs
     online `fp8_per_channel` top-20 logprobs on the same prompts (R1 is excluded only if they agree).
2. **R2. The scatter semantics were verified against a fake that mirrors my reading of
   `RoutedExperts.weight_loader`** (`tests/test_loader_logic.py::FakeRoutedParam`), not against vLLM.
   Specific unknowns: the `channel` scale `copy_` shape `[N, 1]` (`RE:443-451`), the `group` scale for the
   NVFP4 `[E, N, K/16]` e4m3 parameter (`RE:371-432`), the `tensor` global scales `[E, 1]` and `[E]`
   (`RE:337-353`), and the module path under which `named_parameters()` reports the `RoutedExperts`
   parameters (the coverage check avoids depending on it). Check: a clean load with no `OlalaQuantLoadError`.
3. **R3. The "raise on every unconsumed tensor" policy also changes the bf16 path.** The static audit
   (`tests/test_loader_logic.py::test_every_real_checkpoint_name_lands_on_a_vllm_parameter`) maps all 945
   real names to vLLM parameter names read from `mamba3.py`, `diff_tpa.py`, `olala.py`, but nothing was
   run. If one benign HF tensor has no vLLM counterpart, the bf16 model stops loading. Check FIRST: load the
   unmodified bf16 checkpoint with the patch (this is also the regression test) and, only if it fails on a
   named harmless tensor, allow that exact name with `OLALA_ALLOW_UNCONSUMED`.
4. **R4. The `quantization_config` block was not validated against the real `compressed_tensors` library.**
   The attempt to install `compressed-tensors==0.17.0` into a scratch environment was refused by the
   session's permission classifier, so I did not work around it. Fields (`observer: minmax`,
   `quantization_status: compressed`, `block_structure`, `actorder`, `observer_kwargs`) follow llm-compressor
   conventions from memory and from the keys vLLM reads (`CT:305-385`). Check: `tools/pod_check_config.py`
   (needs no GPU, no weights): it runs `QuantizationArgs.model_validate`, `CompressedTensorsConfig.from_config`,
   `_get_scheme_from_parts` and compares `should_ignore_layer` with the converter's ignore list on all 36 layers.
5. **R5. Numerical behaviour on the real weights is unmeasured.** Error bounds V1 and V2 are guesses
   (Gaussian measurements only). NVFP4 on experts with outlier blocks may flush blocks (counted in
   `scale_summary.n_flushed`) or exceed 0.25. The first conversion on real weights decides; nothing here predicts it.
6. **R6. FP8 dense and shared-expert path under the generic MoE runner.** With any `quant_config` the Olala
   fast decode paths (`_fused_proj`, small/medium MoE paths) are off (`OLALA:288-295,353-357`), so the
   runner owns the latent projections, the shared expert and the routed scale (`OLALA:320-326`). Decode speed
   and exact numerical parity with the bf16 fast path are untested; the `OLALA_PLAIN_MIXER` branch for
   ignored `mixer_proj` layers (`OLALA:894-913`) is assumed to use the plain bf16 weight.
7. **R7. NVFP4 on dense layers on sm_120.** The weight-only dense kernel is Marlin FP4 unless a native
   kernel is selected (`kernels/linear/__init__.py:1237-1248`); alignment (`n % 64`, `k % 128` classes) for
   `K=384`, `N=384` latent projections was not checked. Check the Marlin/`FlashInfer` log lines and that
   all 36 layers produce finite logits.
8. **R8. Names of the study arms.** The brief calls the FP8 recipe "arm F2"; PROTOCOL.md section 2 calls the
   FP8 arm F1 (N1, N2, N3 agree). The recipe is named `fp8`; `arms=("F2",)` is recorded in the info file.

## 6. What can only be validated on a GPU, and what the first pod test must check

Nothing below can be checked on the Mac because it needs the fork, CUDA kernels or the real weights.
In order (cheapest first, stop at the first failure):

0. `tools/pod_check_config.py --ckpt DIR` (no GPU, no weights): R4.
1. Patch applies and the module compiles against the fork actually installed on the pod
   (`git apply --check`), then the **bf16 regression**: load the bf16 checkpoint with the patch and compare
   greedy logits of 2 prompts with the unpatched fork (R3, L4, L5).
2. FP8: load, read the backend log lines (R1), one request (non-empty answer), compare offline top-20
   logprobs with online `fp8_per_channel` + the same ignore list on 64 short chunks (agreement = R1, R2 excluded
   for FP8), then the fidelity script of `evals/`.
3. NVFP4 weight-only: same sequence, then check `n_flushed` in `quantization_info.json` against the logged
   perplexity (R5).
4. Negative test (30 seconds): serve the FP8 checkpoint with an edited config that drops `ignore`: the load
   must fail with `OlalaQuantLoadError` naming the bf16-vs-fp8 tensors (L3), not start and not answer.

## 7. Discrepancies found in the inputs

* D1. Arm label F2 (brief) vs F1 (PROTOCOL.md): R8.
* D2. Ignore list I: brief = every mlp tensor of layers 0, 1, 5, 19; PROTOCOL.md = their down projections
  only; the recon (C.2) says ignore all down paths because the code does not say which spikes. Because
  RoutedExperts cannot be split (I3), the routed `up` of those layers is ignored in any case; FP8 with list I
  keeps 8.38 GB, NVFP4 6.04 GB (`reports/budget.json`), against 7.74 and 5.12 GB without the list.
* D3. recon section A.2 puts the vLLM-reachable maximum at 5.68 bits per weight for W4 g128 with fp16
  scales. NVFP4 (4-bit values plus an e4m3 scale per 16 plus a global scale) comes to 6.05 bits per weight
  without an ignore list and 7.14 with list I; FP8 to 9.14 and 9.89 (script output, not an estimate).
* D4. recon D.4 estimated the loader patch as small; it is: one new 274-line standalone module and a rewrite
  of `_expert_params` / `load_weights` in `olala.py` (about 200 lines added, 120 removed).
