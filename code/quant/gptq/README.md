# G1-lite: calibrated NVFP4 for the routed experts of Olala-v1.0-mini-7A1B

First calibrated build of the study (`../design/GPTQ_DESIGN.md`, milestone M4, "the cheapest experiment for risk 2"):
**routed experts only** (the 72 fused tensors `experts.experts.weight [256,768,384]` and
`experts.output_experts.weight [256,384,768]`; 4.83 B parameters in the 32 layers that list I leaves quantized, 82% of
the quantized set), GPTQ-style error compensation per expert with routing-aware Hessians, **static N2 scales (decision
D1)**: per-group-16 e4m3 block scales and per-expert fp32 global scales are exactly those `olala_quant.nvfp4` computes
from the original weights, the solver only chooses the 4-bit codes. Everything else is the N2 recipe (list I in bf16,
dense linears and shared experts through the data-free path). The output is a checkpoint in EXACTLY the N2 format (same
tensor names, packing, `config.json`; `quantization_info.json` extended with a `calibration` block), so the loader patch
and `olala_quant.verify` (now with a `--calibrated` mode) apply unchanged.

Nothing here was run on the real weights, a GPU, or the real model. Everything below "Verified on the Mac" was measured
on a tiny random Olala (4 layers, hidden 64, 16 experts) with the vendored OVH modeling code on the CPU.

## Layout

| Path | What |
|---|---|
| `solve.py` | Batched GPTQ on the static NVFP4 grid (experts in chunks, float64), RTN fallbacks with reasons, held-out guard |
| `hessians.py` | `MoECapture` (hook on `TorchMoEMLP`), `build_hessians` (float32, clip, held-out windows, minimum tokens), `run_sequential` (layer by layer, resume per layer, optional quantized propagation) |
| `build_g1lite.py` | The CLI `--src --calib --dst [--base-select SPEC] [--device cuda] [--layers ...]`; writes the checkpoint and `g1lite_report.json` |
| `window_domains.py` | Windows used per domain of a CAL-1 manifest (exact from the windows, or a labelled estimate) |
| `calib_data.py` | Calibration set from PUBLIC text: sources, manifest with sha256, windows, refusal on leakage |
| `leakage_audit.py` | 8-gram audit against the item texts of MGSM, HumanEval+, MBPP+, IFEval, AIME 2025; exit 1 on any overlap |
| `extract_eval_items.py` | Writes those item texts from the harness's own task factories (`evals/run_tasks.py`), no model |
| `olala_model.py`, `vendor/` | The vendored OVH modeling code (Apache-2.0, third party, see `vendor/README.md`) and its loader |
| `tiny_model.py`, `tiny_e2e.py`, `tests_support.py` | The tiny random Olala fixture and the end-to-end comparison |
| `pod_g1lite.sh` | Pod-side wrapper (`--dry-run`, steps `check tokenize feasibility build verify all`; env `G1_CALIB`, `G1_BASE_SELECT`) |
| `tools/mutation_check.py` | 40 weakenings of the guards (results in `results/`) |
| `data/` | `eval_items.json` (extracted item texts) and `calib-lite/` (the audited text bundle, no windows yet) |
| `../olala_quant/{convert,verify}.py` | Two additive changes: `convert(override=, info_extra=)` and `verify(calibrated=True)` / `--calibrated` |

## Setup and tests (Mac CPU)

```bash
cd quant/gptq
uv sync --extra data          # python 3.12, torch 2.14.0, transformers 5.8.0, einops 0.8.1, pinned (uv.lock)
uv run pytest                 # 109 tests, about 12 s (27 of them build tiny models and run them)
uv run ruff check . && uv run ruff format --check .
uv run python tools/mutation_check.py       # about 8 minutes
uv run python tiny_e2e.py --seeds 0 1 2 3 4 --out results/tiny_e2e.json
(cd .. && uv run pytest)      # the olala_quant tests (122), including the new calibrated-mode tests
```

## What the build does (and where it differs from the design text)

1. **Windows.** `calib_data.read_windows(manifest)` refuses a manifest without a passing leakage audit, without windows,
   or whose window file hash changed.
2. **Layer loop** (`hessians.run_sequential`). Embeds the windows exactly as `OlalaModel.forward` does, then for each layer
   builds ONE `OlalaMonoBlock` from the checkpoint (checkpoint dtypes kept: bf16 matrices, float32 router and Mamba
   parameters), runs it over the windows, and drops it. Only the layers up to the last calibrated one are run.
   `--propagate bf16` (default) feeds the next layer the bf16 outputs: the design's "parallel" Hessians, one forward per
   layer. `--propagate quantized` installs the dequantized experts and runs the layer again: true sequential, twice the
   forward cost.
3. **Capture.** A forward pre-hook on the expert loop stores its arguments, the 384-dim latent and the top-6 ids. The
   Hessians are computed from the stored rows, not accumulated in the hook: the clip needs the median norm of an expert's
   whole token set, the held-out guard needs the held-out rows, and the three Hessian modes share one forward. One layer's
   rows (50 MB for 65,536 tokens) and Hessians (755 MB in float32) are held at a time.
4. **Hessians** (`hessians.build_hessians`). Default `--hessian blend`: routed second moment plus one pseudo-token per
   dimension of the all-token moment (the design's default; `routed` and `all` are one flag away). Float32. Tokens above
   3x the median norm of their set are down-weighted to that norm (also in the all-token part: a deviation from the
   prototype, which clipped only the routed part). Every 10th window (`--heldout-every`) is held out of every Hessian,
   the all-token moment included, and kept for the guard. The down-projection input is `relu(W_up x)^2` of the ORIGINAL bf16
   up-projection computed in the model dtype, as the expert loop does.
5. **Solve** (`solve.solve_experts`). Static scales and static act-order (columns by descending Hessian diagonal, natural
   groups, no `g_idx`), damping 0.01 x mean diagonal, float64. Matches `moegptq.gptq.gptq_quantize` (tests compare codes
   and loss per expert).
6. **Fallbacks to round-to-nearest, per expert, with a reason** (`reason` in the report): no routed token, fewer than
   `--min-tokens` (16 for `routed`, 1 otherwise), Hessian not finite / negative diagonal / not positive definite, diagonal
   ratio above `--diag-ratio-max` (1e6), and the **held-out guard**: if the expert pair (up and down as chosen) has a larger
   output error on the held-out rows than round-to-nearest, or fewer than `--min-heldout-rows` (4) held-out rows to judge
   it, the expert is rounded to nearest. The last rule is stricter than the design (which only says "worse"): a calibrated
   result has to be shown to help.
7. **Write** (`olala_quant.convert` with the new `override` hook). Same shards, index, `config.json`, auxiliary files;
   the solver's packed codes replace the routed tensors; the writer recomputes the data-free scales from the source and
   refuses (RuntimeError) if the solver's scales differ by one bit.

## Calibration set (lite): public text only

`data/calib-lite/` holds the audited TEXT bundle (stage 1, built on the Mac): 40 documents (wikitext-2 train from the local
Hugging Face cache, 32 pieces; the Python standard library of 3.12, 8 pieces), sized for 65,536 tokens with a conservative
3 characters per token, zero overlap with 1,427,605 distinct 8-grams of the five tasks' item texts
(`data/calib-lite/leakage_audit.json`; 15,572 item fields indexed, 9,898 fields shorter than 8 words are not auditable by
n-grams and are counted, not ignored: gold answers such as `42`). **There is no French slice**: no French text was available
offline, and none was fetched. Add one before the pod session if wanted:

```bash
uv run python calib_data.py build --text-only --out data/calib-lite --tokenizer chars:3.0 --n-tokens 65536 --window 2048 \
  --eval-items data/eval_items.json --source wikitext2_train:0.6 --source pystdlib:python:0.2 \
  --source textfile:french:/path/to/public_french.txt:0.2          # or hfdataset:french:wikimedia/wikipedia:20231101.fr:train:text:0.2
```

Stage 2 needs the model's tokenizer (it lives with the weights) and runs on the pod (`pod_g1lite.sh tokenize`):
`calib_data.py tokenize --out data/calib-lite --tokenizer /workspace/models/olala-bf16`. It re-audits the bundle, tokenizes,
packs windows of 2,048 tokens (documents joined with the EOS id of `config.json`, no document masking) and records the
sha256 of `windows.json`. The item texts are regenerated with
`uv run --project ../../evals python extract_eval_items.py --out data/eval_items.json` (offline from the HF cache: it read
MGSM 2,750 items, HumanEval+ 164, MBPP+ 378, IFEval 541, AIME 2025 30; it never starts a model or a sandbox). This is NOT the
CAL-1 set of the design (no self-generated traces, no chat template): a PROTOCOL amendment fixing CAL-1 must precede any
accuracy claim.

## The pod: exact commands

Prerequisites (outside this directory): a pod created for the run
(RTX PRO 4500, 32 GB, 0.72 USD/h), the bf16 weights at `/workspace/models/olala-bf16`, and the bundle extracted at
`/workspace/bundle` (`runpod/build_bundle.sh` bundles `quant/gptq`, with `data/calib-lite` and without `.venv`). The pod
needs no network for G1-lite itself (the weights are already there); `check` installs `einops==0.8.1 --no-deps` if missing and
nothing else.

```bash
cd /workspace/bundle/quant/gptq
./pod_g1lite.sh --dry-run feasibility           # prints the exact commands, creates nothing
./pod_g1lite.sh check                           # ~1 min   versions, GPU, vendored modeling file == the weights' copy
./pod_g1lite.sh tokenize                        # ~2 min   windows.json + re-audit
./pod_g1lite.sh feasibility                     # ~15-25 min (design: the 30-minute session, 0.4 USD): layers 2, 4, 6 on 4 windows
./pod_g1lite.sh build                           # design M4: ~3 pod hours incl. serving, 2.2 USD; the calibration pass itself 0.5-3 h (a guess)
./pod_g1lite.sh verify                          # ~3-5 min  python -m olala_quant.verify --calibrated
```

`feasibility` is `build_g1lite.py --layers 2,4,6 --max-windows 4 --heldout-every 4 --batch-windows 1 --attn-impl eager
--calibrate-only` (the pure-torch Mamba-3 fallback, `OLALA_TORCH_MOE=1 OLALA_TORCH_MAMBA3=1 HF_HUB_OFFLINE=1`): layers 0 and 1
(list I, forward only), then 2 (Mamba), 3, 4 (attention), 5, 6 (Mamba). It prints, per layer, forward seconds (mixer and MLP
separately), peak CUDA memory, `max |mixer output|` and `max |MLP output|` (the design's monitor of the massive activations of
layers 1, 7 and 23, which need `--layers` up to 7 and 23 to see), and for the calibrated layers Hessian and solve seconds,
the GPTQ/RTN split and the mean loss ratio. To calibrate a few layers only, `./pod_g1lite.sh build --layers 2,3` keeps the
round-to-nearest codes elsewhere (`partial: true` in the info). Every finished layer is saved in `DST.work`; rerunning the
same command resumes after the last finished layer; `--restart` starts over; changed settings are refused.

Then, outside this directory: serve with the loader patch (`runpod/pod/arm_pipeline.sh RUN ckpt:/workspace/ckpt/g1lite:nvfp4`
applies the patch, serves, runs the smoke gate), fidelity against N2 with `evals/fidelity.py` (N2: PPL ratio 1.0798, mean KL
0.0668, top-1 0.8502), go/no-go of the design (mean KL <= 0.045, p99 KL not above 0.621, PPL ratio <= 1.05; thresholds are the
design's guesses).

## Verified on the Mac (tiny random Olala, CPU)

* Solver: known answers (identity and any diagonal Hessian give RTN codes, with and without act-order), codes and loss equal
  the prototype's per expert, chunk invariance, grid membership, dead columns, every fallback with its reason.
* Hessians: an expert's H equals the explicit sum over its routed (token, slot) pairs for the routed, all-token and blend
  modes, with clip and held-out windows; the same on the hooked tiny model (routing re-derived with the prototype's router).
* Runner: boundary states equal the full model's forward, resume gives the identical final state, a state made for other
  windows is refused, quantized propagation changes the next layer only when asked.
* Checkpoint: verifies with `olala_quant.verify --calibrated`, is refused by the strict mode, scales and all non-routed tensors
  are byte-identical to N2's, `config.json` equals N2's.
* Tiny end-to-end (`tiny_e2e.py`, 5 random tiny models, each quantized by N2 and by G1-lite, reloaded through `olala_quant`
  and run on held-out windows; `results/tiny_e2e.json`):

  | held-out windows | arm | rel. logit RMSE | mean KL | top-1 agreement |
  |---|---|---|---|---|
  | fresh documents of the calibration lexicon | N2 (RTN) | 0.1541 | 1.231e-02 | 0.765 |
  | | G1-lite | 0.1236 | 7.903e-03 | 0.840 |
  | uniformly random token ids (never seen) | N2 (RTN) | 0.1522 | 1.159e-02 | 0.773 |
  | | G1-lite | 0.1621 | 1.312e-02 | 0.764 |

  G1-lite is better on the distribution it was calibrated on (KL lower on all 5 seeds, mean -36%) and WORSE on random
  tokens (mean KL +13%): calibration is specific to the data the Hessians saw. The tiny model and its one-lexicon byte-level
  toy corpus say nothing about the size of either effect on the real model; they show that the pipeline does what it should
  and that the calibration domain matters.

## What only a pod can validate (ranked by what can sink the plan)

1. **The HF block forward on the real model**: seconds per layer of the pure-torch Mamba-3 chunk loop and of the 256-iteration
   expert loop, peak memory, whether transformers 5.16-5.20 of the pod image (the file targets 5.8.0) runs the vendored modeling
   code at all, whether the eager attention path matches the served kernels, and whether `max |branch output|` of layers 1, 7,
   23 stays inside float32 range at 2,048 tokens (the design caps the length at 8,192 until it is measured).
2. **Whether calibration moves real fidelity**: real per-expert Hessians may be near-diagonal (no gain), spike-dominated (the
   diagonal-ratio guard sends them to RTN: the report counts them), or rank-poor for rare experts at 57k training tokens
   (the held-out guard decides). The tiny model cannot say anything about this; its 20-35% error reduction on fresh text is a
   property of a random model and a one-lexicon toy corpus.
3. **Serving**: the checkpoint is format-identical to N2, which is the premise; that Marlin FP4 serves it and that fidelity
   against N2 improves is only known from the fidelity run. Also unverified: the calibrated mode of `verify` at real size
   (it recomputes the data-free quantization of 5.9 B parameters on the CPU), the memory of `solve_experts` at `expert_chunk=32`
   on the card, and the solver's wall time on a GPU (about 5 s per layer on this Mac's CPU for the 256 experts).

## Known limits and decisions to confirm

* Default propagation is `bf16` (one forward per layer; the design's M4 uses parallel Hessians). `quantized` is implemented
  and tested but costs a second forward per calibrated layer and was not measured at scale.
* The guard compares an expert's PAIR of matrices with the output error on held-out WINDOWS of the same calibration set: it
  detects overfitting to the Hessian, not distribution shift (see the tiny results: calibration on one distribution made the
  model worse on random tokens). A fidelity set that is not the calibration domain is the real test.
* `--min-heldout-rows 4`, `--heldout-every 10` (3 of 32 windows), `--token-clip 3`, `--percdamp 0.01`, `--diag-ratio-max 1e6`
  are the design's values or my defaults, untuned; the verify bound (`nvfp4_calibrated_factor = 2.0` times the data-free error)
  is a guess, documented in `olala_quant/verify.py`.
* The shared experts, latent projections, `mixer_proj` and `gate_proj` are quantized data-free as in N2 (out of scope).

## CAL-1: self-generated reasoning traces on public prompts (`cal1_generate.py`)

The calibration set of the design (`../design/GPTQ_DESIGN.md` sections 5.3 and 5.4) for the calibrated NVFP4 builds: what the
bf16 model WRITES (analysis channel and final answer) in answer to PUBLIC prompts, rendered with the model's chat template,
because the experts see different activations on their own reasoning than on plain text and the tiny study showed that the
calibration domain decides whether GPTQ helps. Nothing here was run against a real model or a real dataset: the tests use a
fake local HTTP server and fixtures (`tests/test_cal1_generate.py`, `tests/test_pod_cal1.py`).

| Path | What |
|---|---|
| `cal1_generate.py` | Prompt draw, resumable generation, 8-gram audit of prompts AND generated text, chat-template windows, manifest, `--verify` |
| `pod_cal1.sh` | Pod wrapper, steps `plan` (the tool's dry run), `run`, `verify` |
| `tools/mutation_check_cal1.py` | 46 weakenings of the CAL-1 guards (report in `results/mutation_cal1_report.json`) |

**Prompt sources** (public, train splits only, domain mix = weight of the five tasks in the macro-average, 240 prompts:
math 48 = AIME 2025, multilingual math 48 = MGSM, code 96 = HumanEval+ and MBPP+, instruction following 48 = IFEval).
Metadata read from the Hub on 2026-11-20 (licence tags and file lists only, no data); the revision of each is NOT pinned in
the code: `--revision SOURCE=SHA` (40 hex) is mandatory and the tool refuses anything else, so the SHA is read at run time
(`curl -s https://huggingface.co/api/datasets/REPO | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])'`) and
recorded in `prompts_manifest.json` and `manifest.json`.

| source name | domain | Hugging Face dataset | licence | used | why it is disjoint from the test tasks |
|---|---|---|---|---|---|
| `math_train` | math | `EleutherAI/hendrycks_math` | MIT | `train`, 7 subjects, levels 4 and 5, column `problem` | MATH (2021) predates AIME 2025 and has no code, GSM8K or IFEval content; the test split is not read |
| `mgsm_train_parallel` | multilingual_math | `Mathoctopus/GSM8KInstruct_Parallel` | Apache-2.0 | `train`, column `query` (UNVERIFIED, the viewer was down: the tool names the real columns, `--column` overrides) | GSM8K TRAIN questions translated into bn, de, es, fr, ja, ru, sw, th, zh and en; MGSM is the first 250 GSM8K TEST items |
| `code_oss_python` | code | `ise-uiuc/Magicoder-OSS-Instruct-75K` | MIT | `train`, `lang == python`, column `problem` | problems written by an LLM from random open-source snippets, not from HumanEval or MBPP (MBPP+ is built on MBPP); the card asks to respect OpenAI's usage policy |
| `if_personas` | instruction | `allenai/tulu-3-sft-personas-instruction-following` | ODC-BY | `train`, column `prompt` | synthetic persona prompts using the IFEval constraint taxonomy, not IFEval's own 541 prompts |

"Disjoint by construction" is an argument about provenance, not a proof: the proof is the audit. A prompt that shares any
8-gram of normalised words with any text field of any item of the five tasks is skipped at draw time and listed
(`dropped_prompts`), and the final audit covers prompts and generated text again.

```bash
cd quant/gptq
uv run python cal1_generate.py --dry-run --out /tmp/cal1 --eval-items data/eval_items.json --revision math_train=SHA ... \
  --revision mgsm_train_parallel=SHA --revision code_oss_python=SHA --revision if_personas=SHA     # plan and counts, writes nothing
# on the pod, bf16 server up (runpod/pod/arms.json: port 8000, served name "olala"):
CAL1_REVISIONS="math_train=SHA mgsm_train_parallel=SHA code_oss_python=SHA if_personas=SHA" \
CAL1_PROVENANCE="vllm <version>, olala-bf16 <weights sha256>" ./pod_cal1.sh run     # resumable; rerun after an interruption
./pod_cal1.sh verify                                                                # every sha256, windows, audit
```

`manifest.json` has the schema `calib_data.read_windows` accepts (`audit.ok`, `audit.n_hits == 0`, `windows`, `n_windows`,
`window`), so `build_g1lite.py --calib <dir>/manifest.json` takes it unchanged. It also records the sha256 of every file
(`prompts.jsonl`, `prompts_manifest.json`, `generations.jsonl`, `generation_config.json`, `calib_text.jsonl`,
`eval_items.json`, `leakage_audit.json`, `windows.json`), the sources with dataset id, revision, licence and counts, the
quotas and weights, the sampling, the server provenance string and the models the server listed, and per domain the numbers
of prompts, successes, failures, truncated traces and traces dropped by the audit.

Decisions and limits to know:

* **Sampling and request**: `temperature 0.8`, `top_p 0.9`, `top_k 40` and `chat_template_kwargs.reasoning_effort medium` in
  the body of `POST /chat/completions`, as the harness sends them (`top_k` and the template kwargs as `extra_body`), one user
  message, no system message, `max_tokens 8192`, no per-request seed (the harness default).
* **Truncated traces are kept and flagged** (`finish_reason: length`, `truncated: true`, `counts.n_truncated`); they are
  rendered with the template's closing tokens, which the model never wrote.
* **Failures are rows** (`status: failed`, with the error). A resume retries prompts without a successful row, up to
  `--max-attempts` (3) failed attempts each; a prompt that has an `ok` row is never asked again, and the store refuses a second
  `ok` row. A domain without any trace refuses the build; other failures are listed in `manifest.json` (`failures`).
* **Audit**: 8-gram like `calib_data` (the design text says 13-gram for completions; the stricter 8 was asked for). Any overlap
  exits 1 and deletes `manifest.json` and `windows.json` of an earlier run. `--drop-flagged` (opt-in, as in `calib_data`) removes
  the flagged traces instead and lists them under `audit.dropped_traces`. The audit splits on whitespace: for Chinese, Japanese
  and Thai prompts a whole sentence is one "word", so it can hardly see an 8-word overlap there. That is a limit of the existing
  audit, not fixed here.
* **Windows** follow `calib_data.pack_windows` (traces in a seeded order, EOS id after each, no masking, tail dropped). Every
  trace starts with the template's system prompt (about 3.5 kB: it is what the server saw), so short traces are partly
  boilerplate. A real tokenizer is only needed at the last stage (`--tokenizer /workspace/models/olala-bf16`).
* **Resume refuses** a changed prompt file, other quotas, seed or dataset revisions, another sampling, model or provenance
  string, and rows whose prompt text changed.

Tests: `uv run pytest` (109 existing + the CAL-1 tests); mutation check: `uv run python tools/mutation_check_cal1.py`.

## Experts-only base (`--base-select experts`) and CAL-1 windows (amendment 22)

The ablation batch showed that the 20% of weights outside the routed experts carry about twice the distribution shift of
the 80% inside them, so the calibrated arm leaves them bf16: `build_g1lite.py --base-select experts` quantizes (with GPTQ,
routing-aware Hessians, static N2 scales, exactly as above) ONLY the routed experts of the 32 layers outside list I, copies
every other tensor bf16 and lists it in `quantization_config.ignore`, exactly as `olala_quant.convert --select experts` does
(the converter does the writing, so the `selection` block of `quantization_info.json` is the converter's; the `calibration`
block adds `base_select` and `non_expert_tensors`). The same build with round-to-nearest codes is N5 (`convert --select
experts`): G1 and N5 differ only by the codes of the routed experts (scales, ignore list, config, every other tensor are
bit-equal; tested). Other specs of the ablation grammar work if they quantize only routed experts (`experts:layers=12-23`);
anything that would quantize another tensor (`all`, `non-experts`, `shared-experts`, `latent`, `attention`, `mamba`) is
refused before any work with "only routed experts are calibrated". Without the option nothing changes (golden sha256 of the
shards, index, config and info on the tiny fixture: `tests/test_default_unchanged.py`).

`--calib` accepts a `manifest.json` or the directory that holds it, in particular the output directory of `cal1_generate.py`
(`calib_data.read_windows` still refuses a manifest without a passing leakage audit, without windows, or whose windows changed).
`g1lite_report.json` (`calibration.per_domain`, and the same block in the info's `calibration.manifest`) says how much of each
domain is in the windows used: tokens, window equivalents (tokens / 2,048) and windows touched. CAL-1 windows are a stream of
traces, so a window holds several domains; the attribution (`window_domains.py`) finds the trace starts (every rendered trace
begins with the same system prompt) and reproduces the seeded order of `cal1_generate`; on the real CAL-1 windows it equals the
manifest's per-domain token counts (math 116,165, multilingual math 107,333, code 282,786; instruction following 89,459 of
90,006, the rest is the dropped tail). If it cannot be exact it says `token_share_estimate` and why.

```bash
cd /workspace/bundle/quant/gptq
export G1_CALIB=/workspace/cal1/calib-cal1 G1_BASE_SELECT=experts        # the CAL-1 output directory; no tokenize step
./pod_g1lite.sh --dry-run all                                            # prints every command, runs nothing
./pod_g1lite.sh feasibility && ./pod_g1lite.sh build && ./pod_g1lite.sh verify
```

`tiny_e2e.py --base-select experts --out results/tiny_e2e_experts_only.json` repeats the tiny comparison with N5 (RTN) against
G1 on the experts-only base. Tiny random model, 5 seeds, mean (`results/tiny_e2e_experts_only_output.txt`): on fresh documents
of the calibration lexicon the KL against bf16 goes from 6.70e-3 (N5) to 1.54e-3 (G1), on uniformly random tokens it goes from
7.09e-3 to 8.96e-3 (worse), the same specificity to the calibration domain as in the N2 comparison above; nothing about the real
model follows. Mutation check of the new guards: `uv run python tools/mutation_check_base_select.py`.
