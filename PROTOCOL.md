# PROTOCOL: quantizing Olala-v1.0-mini-7A1B for GPU serving (FP8, NVFP4)

> This is the pre-registration of the study with internal process notes removed; every removal is marked
> "[removed from the public copy: internal process note]". No design, comparison, margin, analysis or decision-rule text was changed; budget figures (caps, prices) are among the removals.

Status: v1, 2026-11-20, [removed from the public copy: internal process note] and frozen by the
commit that contains this line, before the first paid job. From the commit on it is only amended by dated entries at
the end, never edited.

## 1. Question

OVHcloud published GGUF and MLX builds of Olala-v1.0-mini-7A1B (hybrid Mamba-3 / differential-TPA / latent-MoE,
6.8B parameters, about 1B active) and withdrew its 4-bit builds because quality fell (Q4_K_M: MBPP+ 67.7 and MGSM
65.1 against 71.1 and 68.7 in BF16, on subsets with a stated standard error of 2 to 3 points). Its technical report
(Appendix C) says every standard post-training quantization recipe collapsed to NaN on realistic inputs because
three Mamba-3 mixer layers (1, 7, 23) emit branch outputs up to 1e16 and four MoE layers (0, 1, 5, 19) spike to 1e8
through the squared-ReLU expert activation. No FP8, NVFP4, GPTQ or AWQ build exists on 2026-11-20 (Hugging Face
search and the OVHaiLLM organisation, both read that day).

Primary question: can Olala-mini be served in NVFP4 weight-only (offline checkpoint, arm N2) on a Blackwell GPU with a
quality loss within noise of BF16, and does keeping the layers named in Appendix C in BF16 (list I, present in N2 and absent in N1) change that result?
Secondary: the same for FP8 (F1, F1n, F2). Exploratory: NVFP4 with FP4 activations (N3, B200 only), the only arm in which
the overflow mechanism of Appendix C can act.

## 2. Arms (every arm differs from its comparison in one factor)

Scope set by the recon (`recon/quantizability.md`, 2026-11-20): the routed experts are fused 3D tensors holding 80.3%
of the parameters; OVH's vLLM fork builds the Mamba in_proj, in_proj_dyn and all attention q/k/v-side projections
without a quantization config, so at most 86.9% of the weights can be quantized and a 4-bit build reaches about 5.7
bits per weight (4.8 GB), not 4. Every table states this. 'FP8' and 'NVFP4' below mean exactly that scope.

Arms come in two groups. Online arms need no new code (OVH's fork quantizes the bf16 checkpoint at load time); offline
arms need our quantizer and our loader patch (`quant/`), and are the models that can be published as checkpoints.
Each arm has a feasibility gate (server starts, smoke test passes, no fallback banner that other arms do not have);
an arm that fails its gate is reported as 'does not serve' with its log, and is not evaluated.

| Arm | Group | Weights | Activations | Layers kept in BF16 | Compared with | Isolates |
|---|---|---|---|---|---|---|
| B0 | reference | BF16 | BF16 | all | (run twice) | noise floor |
| F1 | online | FP8 per-channel | FP8 dynamic per token | list I | B0 | FP8 cost |
| F1n | online | FP8 per-channel | FP8 dynamic per token | none of list I | F1 | role of list I under FP8 |
| F2 | offline | FP8 per-channel, our quantizer | FP8 dynamic per token | list I | F1 | loader and quantizer parity (logits must match F1) |
| N1 | offline | NVFP4 (weights) | BF16 (weight-only) | none of list I | N2 | role of list I under NVFP4 |
| N2 | offline | NVFP4 (weights) | BF16 (weight-only) | list I | B0 | NVFP4 weight cost |
| N3 | online, B200 only | NVFP4 routed experts only (linears stay BF16) | FP4 per token (W4A4) | list I | N2 (not the same scope: partly confounded) | cost of quantizing activations; the only arm that can show the Appendix C overflow |

Amendment 2026-11-20 (before freeze, [removed from the public copy: internal process note]): N1 and N2 are OFFLINE weight-only NVFP4 built
with our quantizer and loader patch (`quant/`), runnable on the RTX PRO 4500 (sm_120) if the fork serves there. The fork's
online `nvfp4_per_token` requires an SM100 GPU (`online/nvfp4.py:100`), so N3 runs only on a B200 and quantizes the routed
experts only; N3 versus N2 therefore differs in scope as well as in activation precision and is reported as an exploratory
contrast, not a clean test. Offline FP8 (F2) is the arm whose checkpoint can be published; F1 and F1n are online. Ignore list
I = `model.layers.{1,7,23}.mixer_proj` plus every tensor of the MLP of layers 0, 1, 5 and 19 (vLLM cannot split w13 from w2).
It costs about 0.9 GB: FP8 8.38 GB with the list against 7.74 GB without; NVFP4 6.04 GB against 5.12 GB (`quant/reports/budget.json`).
The pod API is RunPod v2 (v1 no longer resolves). Smoke wave before any arm: bf16 on RTX PRO 4500, L40S and H100, 1 hour each at most.

Ignore list I (module names from the recon, section C; the fork's names, HF names in brackets): `model.layers.{1,7,23}
.mixer_proj` [same] for the Mamba-3 mixers named in Appendix C; every tensor under `model.layers.{0,1,5,19}.mlp`
(routed down projection, latent expand, shared down) for the squared-ReLU spikes. Always bf16 regardless: router,
embedding, lm_head, Mamba in_proj and in_proj_dyn, attention projections other than gate_proj. Layer indexing is
resolved [removed from the public copy: internal process note]: the report counts from 0 (its Table 50 names the attention layers L4, L9, L14, L19, ...,
as `config.json` does, and its MoE list contains layer 0); every layer has a MoE block, so 'MoE layer 19' is the MoE of an
attention layer. Open and not resolvable before the run: which of the routed and shared experts spike (the list takes
both). List I is frozen at the commit of this protocol and is not changed after any accuracy value is seen.
Arms are added only by dated amendment.

Comparisons, frozen here (one-sided non-inferiority margin of 3 points on the macro-average, see section 7):
| Comparison | Role | Test |
|---|---|---|
| F1 versus B0 | confirmatory | non-inferiority, margin -3 |
| N2 versus B0 | confirmatory | non-inferiority, margin -3 |
| F1n versus F1, N1 versus N2 | mechanism contrasts (does list I matter) | two-sided difference with interval, no margin |
| F2 versus F1 | implementation parity | top-1 agreement and KLD between the two servers' logits, not accuracy |
| B0 repeat versus B0 | calibration of the interval | the interval of the difference must contain 0 |
| N3 versus N2 (and B0 on the same B200) | exploratory | descriptive; B0 and N2 are re-run on the same B200 so that hardware does not confound it, or N3 is labelled hardware-confounded |

## 3. Deployment and measurement path

Deployment: OVH's vLLM fork (branch olala-v0.33, Docker image of OVH or a pinned build), SSM state in float32,
`--max-num-seqs 256` for the hybrid cache, reasoning parser `olala`. Measurement: the same server for all arms,
same pod, one arm at a time, harness in `evals/` (inspect_ai against the OpenAI-compatible endpoint).

## 4. Metrics (why each is valid)

- Fidelity: perplexity, mean and 99th-percentile KL divergence to BF16, top-1 agreement, on 64 chunks of 512
  tokens each of wikitext-2 test, a French slice and a Python slice. Valid because it needs no answer key and moves
  before accuracy does. The KLD uses the top 20 log-probabilities plus an 'other' bucket (an approximation,
  documented in `evals/DECISIONS.md`). Status [removed from the public copy: internal process note]: a secondary diagnostic, labelled a lower bound on
the full-vocabulary KL, never a headline. Required before it is quoted: a sensitivity run on at least 8 chunks at
k = 5, 10, 20, 50, 100 (server `--max-logprobs` raised) showing the KL has converged or by how much it moves; direction
KL(reference || variant), raw (untempered) distributions, mean over all scored positions, p99 over positions; perplexity on
the true next tokens; non-finite logits counted. Headline claims rest on the accuracy table.
- Accuracy (primary): macro-average of MGSM (11 languages), HumanEval+, MBPP+, IFEval and AIME 2025 (30 problems, 5
  samples). Chosen because they are the ones OVH reports and the ones the paper says are most exposed: maths and
  code.
- Efficiency: weights size on disk, GPU memory, tokens per second at concurrency 1 and 32.
- Integrity: request errors, truncations and NaN or empty outputs are counted per arm and reported; they count as
  failures.

## 5. Items, calibration and leakage

No arm consumes calibration data [removed from the public copy: internal process note]: F1 and F1n quantize online with dynamic per-token activation
scales; F2, N1 and N2 are data-free weight quantizers; N3 uses per-token activation scales (to be confirmed at its gate
from the fork's log). Any later arm that needs calibration (static activation scales, GPTQ) is added by a dated amendment
that fixes its calibration set first (GSM8K train prompts, wikitext-2 train, Python and French slices, disjoint from the test
items by exact and 8-gram overlap on prompts and reference solutions, counts in `results/leakage-audit.json`). Test items
are fixed before the first arm runs; what the model saw in its own pre-training is outside our control and is stated as a
limit, in particular for AIME.

## 6. Settings

Generation: temperature 0.8, top-p 0.9, top-k 40 (OVH's), `reasoning_effort=medium` for all arms (OVH used `high`;
the difference is stated in every table). Harness parity check: B0 at `high` on MGSM, HumanEval+ and AIME 2025
against OVH's published numbers (68.7, 88.3, 61.3); the published numbers are context, since prompts, subset and
scorer may differ (our HumanEval+ and MBPP+ scorer is our own, our harness sends top-k explicitly): a miss triggers an
investigation and a written explanation, not an automatic stop, and no arm is read until the explanation exists. A smoke
test of the frozen medium-effort path on known items (reasoning extracted, truncation counted, code scorer) precedes B0.
Scoring, fixed here: AIME 2025 = mean over 5 samples per problem (30 problems, not pass@k); IFEval = prompt-level strict
accuracy (instruction-level reported); MGSM = equal weight per language, 50 items per language (550); HumanEval+ and MBPP+
= pass@1, 1 sample per item; maximum output tokens unset (OVH's setting) with the server's context limit, so a truncated
answer is a failure. Per-request seeds are fixed and recorded but not assumed to produce matched trajectories across arms.
The item subset and sampling seed are written to every output.

## 7. Statistics and decision rule

Estimand: the difference in the equally weighted macro-average of five task scores (MGSM, HumanEval+, MBPP+, IFEval, AIME
2025) between an arm and its comparison, over the fixed benchmark items. Inference: paired bootstrap (10,000 resamples)
that resamples ITEMS within each task, keeping each item's per-sample outcomes and the pairing across arms (the five
AIME samples of a problem move together; the item score is the mean over its samples), then recomputes the five task
means and their macro-average. Known limitation, stated in the report: MGSM translations of one problem are separate
items in the harness and are not clustered by source problem, so the MGSM interval is somewhat too narrow; AIME has 30
clusters, so its interval and the macro interval are wide and the bootstrap's coverage is approximate. Generation noise
at temperature 0.8 is already inside each item's mean; the B0 repeat is the empirical check that the interval has the
right width. Confirmatory test: one-sided non-inferiority against the margin of -3 points for F1 and N2 versus B0, Holm-
adjusted over those two tests at one-sided 0.025. Outcome per arm: NON-INFERIOR (adjusted bound above -3), WORSE
(upper bound of the interval below -3) or INCONCLUSIVE; never 'no loss' and never 'degraded' without the interval. The
margin is a convention (the size of the loss at which the model's authors withdrew a build and about one standard error of
their own subsets), stated as such; results are also shown against margins of -2 and -5. Every task's difference and
interval is reported next to the macro-average, and an arm is flagged 'task-level regression' whenever any single task
has an interval upper bound below -5 points, whatever the macro-average says. A superiority claim between two quantized
arms waits for the test. One seed is one seed: any arm that is published is run twice and its two runs are pooled by
averaging the item scores; both runs and every failed feasibility gate are published.

## 8. Invalid-result conditions (checked before any number is read)

1. Request errors above 0.5%, or truncation rates that differ by more than 1 percentage point between arms (a length
   change caused by quantization is then a result to report, not a reason to discard; the harness counts all of them as
   failures either way).
2. The server logs a fallback kernel or attention path that the expected quantized path (listed per arm before the run from
   the fork's source) does not explain, or that differs between arms for a reason other than the quantization itself.
3. The B0 harness parity check misses OVH's numbers by more than 3 points (MGSM, HumanEval+) or 8 (AIME 2025) and no
   written explanation exists (see section 6).
4. Any NaN output in an arm that is later published.
5. Calibration overlap with test items.

## 9. Budget and stops

[removed from the public copy: internal process note] Pod: RTX PRO 4500 Blackwell, secure cloud, [removed from the public copy: internal process note], watchdog 4 h per session, at most 3 pods live at once
(parallel pods allowed [removed from the public copy: internal process note], same card type for all arms that are compared).
Projected: three to five sessions of 2 to 4 h, [removed from the public copy: internal process note]. Stop rule: [removed from the public copy: internal process note] Before the first create call an authenticated read-only call
(list pods) must succeed; the first create is itself a test (a refusal costs nothing) and the smoke wave tests the
create, status and delete cycle on 1-hour pods. N3 needs a B200 ([removed from the public copy: internal process note]): it runs only
if, after all other arms, [removed from the public copy: internal process note] remain for B0, N2 and N3 together on that card; otherwise it is
dropped by this rule, and the report says so. Throughput and token budgets for the evaluation are measured in the smoke
wave and projected for all arms and repetitions before the arms wave starts.

## 10. Publication

Hugging Face repositories go to the Solfird organisation (huggingface.co/solfird, [removed from the public copy: internal process note]), only after the
smoke gate and the accuracy table for that arm; [removed from the public copy: internal process note]
Which arms are published: those that meet the decision rule of section 7 (within noise of BF16) plus, clearly labelled, the
best 4-bit arm even if it does not, so that readers get a measured loss and not a hidden one. Checkpoints stay private on the
Hub until the technical report is ready ([removed from the public copy: internal process note]), then are made public together with it.
The independent technical report is a second deliverable of this study [removed from the public copy: internal process note]: it covers the model and OVH's claims
[removed from the public copy: internal process note], our re-measured benchmarks (harness parity H6 and the arms), the quantization behaviour and its mechanism
tests, and states only what a run measured. Apache-2.0, NOTICE kept. The model card states what was measured, the
exact recipe, the runtime required, and what was not tested. 'First' is claimed only with the date and the search
repeated immediately before the push.

## 11. Script that generates every reported number

`evals/compare.py` and `evals/fidelity.py compare` over `results/`; tables are rendered from result files only.

## Amendments

(none yet)

### Amendment 2026-11-20 (after the smoke wave; dated entry, original text above unchanged)

1. Study card: RTX PRO 4500 Blackwell (sm_120) serves the fork's bf16 model; the L40S had no stock and the H100 gave no SSH in
   15 minutes. [removed from the public copy: internal process note]
2. H2 (parity F2 versus F1) was NOT MET on the first measurement (top-1 0.931, mean KL 0.0138; noise floor 0.980 and 0.0037).
   Per the rule that a criterion broken by its own first measurement is reported as unmet, F2 is not published until the
   difference is explained; an explanation is an amendment, not an edit of the criterion.
3. Fidelity measurement has a non-trivial run-to-run noise floor (same server, same chunks: mean KL 0.0037, top-1 0.980). The
   B0 repeat of section 7 therefore also repeats the fidelity collection on bf16, and every fidelity comparison is read
   against that floor.
4. Round-to-nearest NVFP4 weight-only (N2 as built, no error compensation) raised perplexity by 8% in the exploratory pass.
   Whether to add a GPTQ-style (calibrated) NVFP4 arm is decided after the accuracy results and needs its own amendment with
   its calibration set (section 5) fixed first.

### Amendment 2026-11-20 (accuracy wave planning; [removed from the public copy: internal process note], before any accuracy value exists)

5. New arm M1 (exploratory, not in the confirmatory family of section 2): the fork's ONLINE `mxfp4` (OCP microscaling FP4,
   block of 32 elements, shared power-of-two scale) on the linears and routed experts the fork can quantize, activations in
   16-bit (`config/quantization.py:214-216`: weight spec only), list I kept in BF16, served with `--quantization mxfp4`.
   Gate as every arm (serves, smoke passes, expected path in the log; the MoE backend for sm_120 is unknown until the
   log is read). Compared with B0 for the non-inferiority margin of section 7 as a descriptive result, and with N2
   descriptively: both are weight-only 4-bit, but M1 and N2 differ in block size (32 vs 16), scale type (e8m0 vs e4m3 with a
   global scale), quantizer (the fork's online quantizer vs ours) and loader, so the contrast is not one-factor and is
   reported as such. An OFFLINE MXFP4 checkpoint would need our quantizer and loader extended; it is added only by a
   further amendment, after M1's gate and results.
6. Investigation of the H2 failure (F2 vs F1), narrowed by the saved logs: both paths log the same linear kernel
   (`CutlassFP8ScaledMMLinearKernel`) and the same MoE backend (TRITON Fp8), so the kernels are not the cause. Remaining
   candidates: the weight-scale convention or clamp of the online quantizer versus ours, and the quantization of the fused
   expert tensors. Planned test (a unit-style comparison on a pod, no accuracy run): quantize the same tensors with the
   fork's online quantization functions and with ours and compare weights and scales element by element.
7. Wave design: three pods of the same card type (RTX PRO 4500), same item set (seed 0, MGSM 550, other tasks complete),
   reasoning_effort medium (parity run at high), samples per item as section 6 (AIME 5, others 1). Pod 1: harness pilot,
   parity run (B0, high), B0 twice. Pod 2: F1, F2, then the H2 weight comparison. Pod 3: N2, M1, F1n (F1n exploratory). The
   projection of tokens and hours comes from the pilot before the full runs; the stop rules of section 9 apply.

### Amendment 2026-11-20 (written before any accuracy value of the wave exists)

8. FP8 publication does not wait for an explanation of the F2-versus-F1 gap. The bisect [removed from the public copy: internal process note]
   shows the two FP8 paths differ in both tensor families and the cause is untested; F2 and F1 are each about as far from bf16
   (mean KL 0.0145). The accuracy comparison against B0 is what validates the artifact that would be published. Therefore:
   (a) F2 joins the confirmatory family of section 7, which becomes {F1, F2, N2} versus B0, one-sided non-inferiority at -3
   points, Holm-adjusted at one-sided 0.025 over three tests; (b) amendment 2's wording that F2 'is not published until
   explained' is replaced by: F2 is published only if it is non-inferior to B0 on accuracy, and its model card states that its
   logits differ from vLLM's online FP8 path (top-1 agreement about 93%, noise floor 98%) with the cause untested; (c) the
   parity result stays reported as unmet. The pilot run (small item subsets, bf16 only) exists to measure tokens per item and
   to test the harness; its accuracy values are not used in any analysis.

### Amendment 2026-11-20 (accuracy wave, before any accuracy value; [removed from the public copy: internal process note])

9. Run structure and analysis rules, fixed now. Stage 1: one run of each of B0 (call it B0a), F1, F2, N2 on the five-task suite, plus
   B0b (a second bf16 run, used only for the noise calibration of the interval and for pooling), F1n (exploratory) and the
   parity run (B0 at high effort). Stage 2, after stage 1: a second run of every arm that is a candidate for publication
   (F2, N2 if it is to be shown, F1). The confirmatory decision of section 7 uses, for every arm and for B0, the average of the
   item scores of all its completed runs at the time of the decision, so B0 enters as the pooled B0a and B0b; stage-1 results
   are reported as provisional. B0b is not chosen after seeing results. The resampling unit is the item (the problem for
   AIME, with its five samples averaged), within each task. Secondary sensitivity for AIME: accuracy among non-truncated samples,
   reported next to the headline (truncations stay failures in the headline). The analysis script refuses to compute a
   comparison unless the two runs have identical item sets and the full sample count and no missing request. A run interrupted
   by a watchdog is discarded and rerun from scratch on a new pod.

### Amendment 2026-11-20 (written before any confirmatory accuracy value was read)

10. AIME client timeouts. Observation (integrity counts only): every request error of the stage-1 runs is a client timeout
    (`APITimeoutError`, 3,606 to 3,620 s) on AIME 2025, concentrated on the hardest problems (for example 8, 13, 17, 19): N2 34
    errors, B0-high 34, B0a 5, F1 0, F2 0, F1n 0; B0b pending. Cause (from the server logs and the settings, not tested by a
    separate experiment): the harness keeps 96 requests in flight while the server's KV cache holds about 23 full-length
    sequences, so long requests queue, and the harness's 3,600-s client timeout counts the queueing. The artifact penalises
    the arms that reason longer. By section 8 condition 1, N2 and B0-high are invalid (errors above 0.5% of samples).
    Decision, equal for all arms: (a) the AIME task is rerun in full (30 problems x 5 samples, same items, sampling and
    reasoning effort) for every arm that had at least one timeout error (B0a, N2, B0-high, and B0b if it has any), with
    non-binding settings: `--max-connections 24` and a client timeout of 14,400 s; (b) for such an arm the rerun replaces its
    AIME task in the analysis, the original AIME rows are kept and published beside it, and the analysis input is a merged run
    directory whose `run_info.json` records the merge; (c) an arm with no timeout keeps its original AIME rows (a timeout that
    never fired cannot have changed its outputs); (d) request errors in a rerun are failures and the 0.5% limit applies to the merged
    run; (e) truncations (context limit) remain failures as before; (f) the parity comparison of AIME uses the B0-high rerun.
    Disclosure of what was read before this amendment: the parity accuracies MGSM 69.1% and HumanEval+ 87.8% (both within the
    protocol's 3-point tolerance of OVH's 68.7 and 88.3) and the AIME parity value 52.0%, which includes the 34 timeouts as
    failures and is therefore not interpreted. No accuracy of any arm was read.

### Amendment 2026-11-20 (decision, still before any confirmatory accuracy value was read)

11. It was decided that the AIME timeout artifact is a harness problem to be fixed, and the runs repeated. This
    supersedes items (a) to (c) of amendment 10 as follows. The harness is fixed (the AIME connection cap is 24, the client
    timeout default is 14,400 s, so that queueing and long reasoning can never be counted as request errors) and the AIME task
    is rerun in full for EVERY arm of stage 1 (B0a, B0b, F1, F2, N2, F1n, and B0-high for the parity check), under the same
    settings, so that all arms' AIME rows are produced identically. For each arm the analysis input is a merged run directory:
    the rows of MGSM, HumanEval+, MBPP+ and IFEval from the original run (which had no request errors in any arm) and the AIME
    rows from the rerun; the original AIME rows stay in the original directories and are published beside the results. Items
    (d) to (f) of amendment 10 stand. The first-pass AIME values of the original runs are not used for any decision.
    Disclosure: before this amendment the integrity counts of the original runs, the parity accuracies of MGSM and HumanEval+
    and the AIME parity value 52.0% (which includes 34 timeouts as failures) had been read; no accuracy of any arm was read.

### Amendment 2026-11-20 (budget)

12. [removed from the public copy: internal process note] [removed from the public copy: internal process note] [removed from the public copy: internal process note]

### Amendment 2026-11-20 (before any accuracy value of the arm exists)

13. Baseline arm Q5 (exploratory, outside the confirmatory family of section 7): OVHcloud's published `Olala-v1.0-mini-7A1B-q5_k_m.gguf`
    (5,990,551,040 bytes, sha256 ed353d26...1860, Hub revision b9f5d20a...) served by OVH's llama.cpp fork pinned at commit
    55c22dca0c98 (tree ff04ce13ba15) built with CUDA for sm_120, on the same RTX PRO 4500 Blackwell card type as every other arm.
    Purpose: a same-hardware reference for the size class of our 4-bit builds (Q5_K_M 5.99 GB against N2 6.04 GB). Differences that
    cannot be removed and are stated with every table: another engine (llama.cpp, not vLLM), another quantization method (K-quants
    with an importance matrix and mixed precision), 12 parallel slots of 32,768 tokens (the harness's longest request is 32,688).
    Sampler semantics are forced to vLLM's (`--min-p 0`, samplers temperature, top_k, top_p) with the same temperature 0.8, top_p 0.9,
    top_k 40 and reasoning effort medium; llama.cpp's own default chain was not run. Because one pod cannot score the five tasks within
    a pod lifetime at this engine's speed (about 350 tokens/s aggregate), the arm is scored on three pods, one task group each
    (aime2025; ifeval and mbpp_plus; mgsm and humaneval_plus), and the three per-task runs are combined by `evals/combine_task_runs.py`
    (refuses on any disagreement of settings, item sets or served file; records every source and its hash) into one run directory that
    passes the same gate as every other arm. The client timeout is 14,400 s and AIME is capped at 24 connections as for the other arms
    (the server has 12 slots, so requests queue on the server, never on a timed-out client). Comparison with N2, G1 and the other arms is
    descriptive with intervals, never a test.

### Amendment 2026-11-20 (correction of a statement in amendment 12)

14. [removed from the public copy: internal process note] [removed from the public copy: internal process note] [removed from the public copy: internal process note] [removed from the public copy: internal process note] [removed from the public copy: internal process note] [removed from the public copy: internal process note]

### Amendment 2026-11-20 (budget)

15. [removed from the public copy: internal process note] [removed from the public copy: internal process note] [removed from the public copy: internal process note] [removed from the public copy: internal process note] [removed from the public copy: internal process note] [removed from the public copy: internal process note]

### Amendment 2026-11-20 (budget)

16. [removed from the public copy: internal process note] [removed from the public copy: internal process note] [removed from the public copy: internal process note] The plan is not enlarged by this amendment, and any item added later is pre-registered before it runs. [removed from the public copy: internal process note]

### Amendment 2026-11-20 (calibration set CAL-1 and the calibrated arm G1, fixed before any trace is generated)

17. **Calibration set CAL-1** (the only calibration data any calibrated arm of this study may use): self-generated reasoning traces of the bf16
    model on public prompts, built by `quant/gptq/cal1_generate.py` (46 mutations of its guards caught). Sources, at the dataset commits read
    from the Hub on 2026-11-20 and recorded in `quant/gptq/cal1_revisions.txt`: `EleutherAI/hendrycks_math` train levels 4 and 5 (math,
    stands for AIME 2025), `Mathoctopus/GSM8KInstruct_Parallel` train (multilingual math, stands for MGSM; GSM8K train questions in ten
    languages, MGSM is made from GSM8K test), `ise-uiuc/Magicoder-OSS-Instruct-75K` python (code, stands for HumanEval+ and MBPP+),
    `allenai/tulu-3-sft-personas-instruction-following` (instruction following, stands for IFEval). Quotas 48, 48, 96, 48 (240 prompts: the
    macro-average gives each task equal weight, so code carries two of five), seed 0, largest-remainder rounding. Generation: the bf16
    server of the study (OVH vLLM fork image of the accuracy runs), the same sampling as the harness (temperature 0.8, top_p 0.9, top_k 40,
    reasoning effort medium), at most 8,192 output tokens, truncated traces kept and flagged. Windows of 2,048 tokens of the chat-template
    rendering of prompt, reasoning and answer. Guards: any prompt with an 8-gram overlap with a test item is skipped at the draw, and the final
    audit of prompts AND generated text against the item texts of the five tasks refuses (exit 1, no manifest) on any overlap.
    **Known limit, stated now:** the 8-gram audit splits on whitespace, so for Chinese, Japanese and Thai prompts it can hardly see an overlap;
    disjointness there rests on provenance (GSM8K train against GSM8K test) and on the fact that MGSM's items are 50 per language from
    translations of 250 GSM8K test problems. The audit is not claimed to certify those three languages.
    No accuracy value of any arm was used to choose these sources, quotas or sampling; the choices follow the design note
    [removed from the public copy: internal process note] and the toy study's lesson that the calibration domain decides whether calibration helps or hurts.
18. **Calibrated arm G1 (G1-lite: routed experts only, static NVFP4 grid, GPTQ error compensation with routing-aware Hessians).** One build
    from CAL-1, no tuning of any kind on accuracy. Stage 1, fidelity go/no-go against the bf16 reference with the study's fidelity tool on the
    same text set as N2: mean KL at most 0.045 (N2: 0.0668) and perplexity ratio at most 1.05. If either fails the arm gets no accuracy run and
    is reported as failed with its numbers. Stage 2, only if stage 1 passes: one accuracy run on the five tasks with the frozen harness, outside
    the confirmatory family of section 7 (its tests were fixed before G1 existed): reported as the macro difference to the pooled bf16
    reference with its interval and the label NON-INFERIOR, WORSE or INCONCLUSIVE at the same -3 margin, without Holm membership, and described
    against N2, F2 and the Q5_K_M baseline with intervals and no test. A second build with different settings is allowed only as a new
    amendment dated before it runs.

### Amendment 2026-11-20 (component ablation of the NVFP4 loss, exploratory, fixed before it runs)

19. **Question:** which part of the model carries the loss of the N2 arm (4.2 points below bf16 on the macro-average)? **Design:** N2-format
    NVFP4 checkpoints in which only a chosen part is quantized and everything else stays bf16 (list I always stays bf16), built by
    `olala_quant.convert --select SPEC`, served on one RTX PRO 4500 pod, scored with the study's fidelity tool against the bf16 reference served
    on the same pod and card (the wikitext chunk file of `job_common.sh`, the same set that gave N2 its KL of 0.0668): mean, median and p99 KL,
    perplexity ratio, top-1 agreement, each with its interval. Specs: `all` (the control, equal to N2), `experts`, `non-experts`,
    `experts:layers=0-11`, `experts:layers=12-23`, `experts:layers=24-35`, `shared-experts`, `latent`, `attention`, `mamba`; expected
    quantized-module counts per spec (200, 32, 168, 9, 11, 12, 64, 64, 14, 26) are checked by `olala_quant.ablation check` and a mismatch fails the spec.
    **Validity rule:** the `all` control must reproduce N2's mean KL (0.0668) within 0.01; if it does not, the whole batch is invalid and is
    reported as such. **Reading rule:** fidelity measures distribution shift on plain text, not task accuracy; KL contributions of parts need not
    add up; the results are descriptive (which part shifts the distribution most, how much of the `all` shift each part and each third of the
    layers reproduces) and are never presented as an accuracy result. No accuracy run follows from this batch without a new amendment.
    Outside the confirmatory family and no outcome labels.

    *Clarification of amendment 19, written before the batch runs ([removed from the public copy: internal process note]):* the question is answered as a question about
    plain-text distribution shift, not about accuracy: the conclusion will say "the part that shifts the next-token distribution most", never
    "the part that costs the accuracy". `experts` and `non-experts` are complements of each other, so both directions of the main split are
    measured (a part alone, and everything but that part). Where the checkpoints' recorded hashes allow it, the `all` checkpoint is also compared
    with the N2 checkpoint of the accuracy wave (index and `quantization_info.json` hashes), in addition to the KL rule. Failed specs are listed
    with their reason and are not interpreted. Routing frequencies come from the activation probe, not from this batch.

### Amendment 2026-11-21 (weight statistics and activation probe on the released weights, descriptive, fixed before they run)

20. **What runs:** `olala_quant.weight_stats` (per tensor and per expert: distribution shape and the round-to-nearest relative error of NVFP4,
    FP8 per channel, INT4 symmetric group 128 and MXFP4 per tensor class) and `analysis/activation_probe.py` (per layer and branch magnitudes of
    the residual stream and of the Mamba, MoE and shared-expert branches on 80 prompts: 64 from the inputs of the five test tasks, 16 plain text;
    spike location by layer, head, expert, token and channel; zero and clamp-at-448 ablation of flagged branches with KL, top-1 and delta NLL
    against the unmodified run) on the bf16 weights, on one RTX PRO 4500 pod. **Purpose:** reproduce or fail to reproduce OVH's Appendix C
    claims (layers 1, 7, 23 with single SSM heads 23, 26, 40 at up to 1e16; MoE layers 0, 1, 5, 19 up to 1e8; fp16-scale overflow) on the
    released weights, and give the weight-level and activation-level evidence for why the model quantizes badly. **Reading rules fixed now:**
    the probe's claim table uses the verdict words consistent, partly consistent, not consistent and not judged, with the thresholds printed in
    the table (one decade for magnitudes, 0.5 on growth exponents, 0.1 on delta NLL, and so on) as they are in the committed tool; they are not
    changed after the real numbers are seen, and a claim whose definition is not checkable stays "not judged". Layer numbers are 0-based
    (attention layers 4, 9, 14, 19, 24, 29, 34); the 1-based reading of OVH's numbers is tested explicitly as a separate row. **Stated limits:**
    the probe forces the pure PyTorch float32 Mamba-3 recurrence, not the TileLang kernel OVH measured with, and magnitudes depend on sequence
    length (OVH measured at 512 tokens, the probe on up to 4,096); the prompts are not OVH's wikitext. Nothing here is an accuracy result
    and no outcome label is attached.

    *Correction to amendment 17, written before any trace was generated (the first session attempt stopped at the plan step, 2026-11-20 23:25 UTC, no model request was made):* the multilingual-math source has the columns `chosen`, `prompt` and `reject`, not `query`; `prompt` is an Alpaca wrapper ("Below is an instruction ... ### Instruction: <question> ### Response:") and the prompt used is the question between those markers (`cal1_generate.py` source field `extract=alpaca_instruction`, tested, 46 mutations still caught). Sources, revisions, quotas, seed, sampling and the audit are unchanged.

    *Second clarification of amendment 17, written after the generation finished and before any window or manifest exists (2026-11-20, 23:50 UTC):* the generation of the 240 traces completed in 7.5 minutes (240 ok, 0 failed, 11 truncated at 8,192 tokens) and the final 8-gram audit REFUSED, as designed: 942 overlapping 8-grams with item texts (HumanEval+ 346, MBPP+ 593, AIME 2025 1, IFEval 2) in 10 traces (6 `math_train`, 3 `code_oss_python`, 1 `if_personas`), the evidence is kept in `results/wave1/cal1/`. Decision, taken from the audit result alone (no score of any arm was consulted): the 10 flagged traces are dropped (`cal1_generate.py --drop-flagged`, the tool's own opt-in) and the remaining 230 traces are audited again; the set is accepted only if that audit finds no overlap, the list of dropped traces and the new counts are written to the manifest, no trace is regenerated to refill a quota (the domain counts become 42 math, 48 multilingual math, 93 code, 47 instruction following), and a remaining overlap is a refusal again.

    *Correction of the count above (read from the finished manifest, `results/wave1/cal1-final/cal1/calib-cal1/manifest.json`):* the refusal listed ten flagged documents because the log line prints only the first ten; the audit flagged 15 traces (math 11, code 3, instruction following 1). They were dropped as decided, the 225 remaining traces were audited again with 0 hits (1,427,605 item 8-grams scanned), and the set is: math 37, multilingual math 48, code 93, instruction following 47 traces (9 truncated at 8,192 tokens), 291 windows of 2,048 tokens, 596,290 rendered tokens; 18 candidate prompts had already been skipped at the draw for overlap. The domain mix therefore differs from the 48/48/96/48 of the design: math is under-represented (37 of 48) because MATH problems are the ones whose generated reasoning most often overlapped item texts; this is a stated limit of CAL-1, not corrected by regeneration. `cal1_generate.py verify` passes on the final directory. The manual steps (`pod_cal1.sh run --drop-flagged`, `verify`, copy) are recorded in `results/wave1/cal1/` and in this amendment because the pod session script has no option to forward `--drop-flagged`.

### Amendment 2026-11-21 (client-timeout artifact; wave 2 supersedes wave 1; written before any wave 2 run starts)

21. **Finding (2026-11-20, 23:20 to 00:10 UTC).** inspect_ai's OpenAI-compatible provider has its own HTTP client timeout (`client_timeout`, default
    600 s) that `GenerateConfig.timeout` does not change. A request that needs more than 600 s is cancelled by the client and retried, i.e. the long
    generation is thrown away and redrawn. Measured from the saved eval logs, counting REQUESTS (the OpenAI client's own retries are logged inside one
    model event, so counting events undercounts; an earlier count of events gave 3 to 9% for AIME and "clean" for the other tasks, which was wrong):
    samples that needed more than one request, vLLM arms: AIME 11 to 21% (16 to 31 of 150 in the six reruns), MGSM 0.9 to 5.5% (5 to 30 of 550) in all
    seven runs, IFEval 2.2% (N2) and 1.5% (F1n) and 0.2% (B0a, B0b), HumanEval+ and MBPP+ 0; the llama.cpp baseline (about 30 tokens/s per slot): MGSM 25%.
    Effect: the longest generations are redrawn until one finishes under 600 s, so the accuracy of the affected tasks is measured on a distribution
    shortened by an amount that depends on the arm's decode speed (slower arms lose more). **Consequence:** every accuracy value of wave 1
    (`results/wave1/{B0a,B0b,F1,F2,N2,F1n,B0-high}`, the merged AIME reruns, and the Q5_K_M runs) carries this artifact. The wave 1 analysis
    (`results/wave1-analysis.md`, commit aff5c88; N2 INCONCLUSIVE, F1 and F2 NON-INFERIOR) is SUPERSEDED, is kept as a record, and will be reported next to wave 2
    with this explanation; it is not used for any confirmatory statement.
    **Fix:** harness commit a9cce36 passes `client_timeout` equal to `--timeout` (14,400 s; an integration test with a slow stub server proves the cancellation
    is gone), records `model_attempts` (requests) and `working_time` per sample and `samples_with_retries` per task and run, and the analysis gate treats
    more than 0.5% of a task's samples with `model_attempts > 1` as a problem (legacy rows only warn, legacy AIME rows are a problem unless explicitly accepted).
    **Wave 2 (the confirmatory data):** full reruns of all five tasks, same items (seed 0, `--n-items mgsm=550`), same sampling and reasoning effort, epochs
    AIME 5, 24 connections for AIME, one pod per run, for B0a, B0b, F1, F2 and N2 (the pooled reference and the confirmatory family of section 7, amendment 9 and
    amendment 8 unchanged: Holm over {F1, F2, N2}, margin -3 points, labels, task-level flag at -5, AIME non-truncated sensitivity, B0 repeat calibration);
    F1n (exploratory) and the B0 high-effort parity run are rerun only if budget remains, otherwise they stay labelled legacy; the Q5_K_M baseline is run
    fresh on three pods (task groups) and combined with `combine_task_runs.py`. New draws of every sample; no analysis choice is changed in light of the wave 1
    numbers, which were seen before wave 2 started (stated here so that the order is on record). A wave 2 run that fails the retry gate is rerun, not analysed.

### Amendment 2026-11-21 (arm N5 and a new base for the calibrated arm; fixed before either is built; the ablation batch of amendment 19 is the evidence for it)

22. **What the ablation batch showed** (descriptive fidelity, `results/wave1/ablation-abl-*`; control `all` equal to N2 within the noise floor): quantizing only the
    routed experts of the 32 layers outside list I gives mean KL 0.0243 and perplexity ratio 1.020 against bf16, quantizing everything else but the routed
    experts gives 0.0498 and 1.057, and N2 (both) 0.0669 and 1.080. The calibrated arm of amendment 18 leaves the non-experts at round-to-nearest, so it could not
    reach its fidelity gate (0.045) whatever the calibration does. **Arm N5:** routed experts of the 32 layers outside list I in NVFP4 by round-to-nearest
    (the N2 recipe), every other tensor bf16 (`olala_quant.convert --select experts`: 32 quantized modules, 64 tensors quantized, 195 kept bf16, counts checked by
    `ablation check`); one build, one accuracy run with the frozen wave 2 harness (a9cce36, all five tasks, same settings as wave 2), one throughput sweep (the
    bench pipeline), file size in bytes reported. Outside the confirmatory family of section 7: reported as the macro difference to the pooled bf16 wave 2
    reference with its interval and the label NON-INFERIOR, WORSE or INCONCLUSIVE at the -3 margin, without Holm membership, and described against N2, F2 and
    the Q5_K_M baseline with intervals and no test. **Arm G1 (amendment 18, base replaced):** the same base (`--select experts`) with GPTQ error compensation of
    the routed experts from CAL-1 (everything else bf16 as in N5), so that G1 differs from N5 only by the calibration. **Stage 1 gate (replaces the 0.045 and
    1.05 of amendment 18, fixed now from N5's measured values 0.0243 and 1.0197 on the same text set and tool):** mean KL at most 0.0194 (a reduction of at least
    20% against N5) and perplexity ratio at most 1.0197; a pass earns one accuracy run (same settings, same label rule as N5), a fail is reported as failed with
    its numbers and no accuracy run. **Stated limit:** the experts-only base is larger than N2 (an estimate of about 7.2 GB against 6.04 GB, to be measured on the
    built file) and than the Q5_K_M file (5.99 GB), so a comparison with Q5_K_M is a comparison along an accuracy-versus-size frontier, never a same-size
    comparison; the size of every arm is printed beside its accuracy. A different base (for example quantizing the Mamba and attention projections too) is a new
    amendment dated before it runs.

    *Correction of the size estimate in amendment 22 (computed from the model header with the plan of `--select experts`, before the build):* the experts-only base is 6,633,872,416 bytes (6.63 GB), not about 7.2 GB; N2 on the same header is 6,043,754,176 bytes (6.04 GB), the Q5_K_M file 5,990,551,040 bytes (5.99 GB). So the experts-only base is 0.59 GB (9.8%) larger than N2 and 0.64 GB (10.7%) larger than Q5_K_M; the statement that a comparison with Q5_K_M is along an accuracy-versus-size line stands, with these sizes. Sizes of the built files are measured and printed beside every accuracy.

### Amendment 2026-11-21 (the accuracy run of G1 no longer depends on the fidelity gate; written before any G1 number exists)

23. [removed from the public copy: internal process note]: a plain-text fidelity gate cannot decide whether the real
    task test happens, because it could hide a model that is good on the tasks. Amendment 22 is changed in one respect: G1 gets its ONE accuracy run (wave 2 settings,
    same label rule, outside the confirmatory family) whether or not the stage 1 gate passes. The gate remains a reported result (mean KL, perplexity ratio, top-1,
    with the thresholds 0.0194 and 1.0197 and N5's values), and its outcome is stated as "passed" or "failed the plain-text fidelity gate" and never as an accuracy
    statement. One build, one run, no retuning on either number; a second build needs a new amendment dated before it runs. The feasibility refusal (projected build time
    above 4.0 h, peak memory above 30 GiB) is unchanged: it protects the pod budget, not the result.

### Amendment 2026-11-21 (arm Q4: OVH's quantization recipe applied at Q4_K_M, a stand-in for the 4-bit file they did not publish; written before it is built)

24. **What is measured and what is not.** OVH's model card documents how its K-quant files were made: an importance matrix (`olala-rl.imatrix`, 200 chunks of
    wikitext-2 train) and two tensor overrides, `ffn_latent_up` kept in bf16 and `ffn_up_exps` kept in q8_0, because those tensors cannot be K-quantized safely.
    The published Q5_K_M file therefore averages about 7.1 bits per weight (5,990,551,040 bytes for 6.77 billion parameters), not 5. OVH did not publish a
    4-bit file, so the file they chose not to ship cannot be measured. **Arm Q4** is the closest honest substitute: their published bf16 GGUF
    (`OVHaiLLM/Olala-v1.0-mini-7A1B-GGUF` at revision b9f5d20a, sha256 pinned in `runpod/pod/gguf_files.json`), quantized with `llama-quantize` from their fork
    (commit 55c22dca) to Q4_K_M with the same two overrides and an importance matrix that I build myself: `llama-imatrix` on wikitext-2 raw train
    (`Salesforce/wikitext` at revision b08601e04326, config `wikitext-2-raw-v1`, text rows concatenated), 200 chunks, the bf16 GGUF, otherwise default settings,
    because OVH's own imatrix file is not published. The differences from OVH's unpublished file are therefore: my imatrix (same data and chunk count, not the same bytes),
    and any setting of theirs that the card does not state; the report says so wherever Q4 appears. **Fixed before the build:** one imatrix, produced once and
    shipped by hash to the other pods; one quantized file, produced by deterministic `llama-quantize` on each pod and accepted only if the three sha256 are identical;
    served with the same flags as the Q5_K_M baseline (12 slots, 32,768 tokens each, the same sampler semantics); the same five tasks, items, sampling, reasoning effort,
    epochs, client timeout and integrity gate as wave 2, run on three pods by task group and combined with `combine_task_runs.py`. Outside the confirmatory family,
    one build and one run, reported as the macro difference to the pooled bf16 wave 2 reference with its interval and descriptive contrasts with Q5_K_M, N5 and N2, with
    no label; the file size in bytes and the average bits per weight are printed beside the accuracy. **Speed:** a short synthetic sweep on the same card with
    `vllm bench serve` against llama-server (512 tokens in, 512 out, concurrency 1, 4 and 12, two repeats) for the Q4 file and for the published Q5_K_M file,
    reported as tokens per second and relative to the Q5_K_M file; it is not comparable in absolute terms with the vLLM sweeps (another engine, another slot count)
    and is never presented as such. **Also stated now:** the report's "5-bit" label for OVH's file becomes "Q5\_K\_M (about 7 bits per weight)" and every build's average
    bits per weight is shown next to its size, because the "4-bit" builds of this study (N2 and N5) also keep many tensors in bf16.

### Amendment 2026-11-21 (candidates for a smaller NVFP4 build that keeps N5's accuracy; written before any of them exists)

25. **Aim:** an NVFP4 build that is no larger than OVH's Q5_K_M file (5,990,551,040 bytes) and no worse in accuracy, with the speed profile of N5. **Ground:** in N5 about 2.1 GB are the routed experts and mixers of the layers that OVH's protected list keeps in bf16 for activation spikes (the MoE of layers 0, 1, 5, 19 and the mixer projections of layers 1, 7, 23). Weight-only rounding does not touch activations, and none of my arms or ablations has quantized the routed experts of those four layers, so their sensitivity is unmeasured. **Candidates (one build each, all other tensors as in N5, i.e. bf16, and the mixer projections of layers 1, 7 and 23 stay bf16 in every candidate):** C1 routed experts of all 36 layers in NVFP4 by round-to-nearest, including layers 0, 1, 5 and 19 (about 5.8 GB); C2 the same set with MSE-optimal block scales; C3 N5's set (32 layers) with MSE-optimal block scales (6.63 GB). **MSE-optimal block scales:** for every group of 16 weights the converter tries the amax-derived scale multiplied by each of six fixed factors (1.00, 0.95, 0.90, 0.85, 0.80, 0.75), rounds the scale to e4m3 and keeps the one with the smallest squared error of the dequantized group; data-free, same on-disk format, same loader. **Stage 1 (fidelity, one pod, the ablation tooling and the same text set as amendments 19 and 22):** mean KL, perplexity ratio, top-1 agreement, each against bf16, with N5 re-measured in the same session as the control; a candidate **passes** if mean KL is at most 0.0304 (N5's 0.0243 plus 25%) and perplexity ratio at most 1.030. **Stage 2:** among the candidates that pass, the SMALLEST file (ties: lowest KL) gets exactly one accuracy run (wave 2 settings, the corrected harness and its gate) and one vLLM throughput sweep (the bench pipeline); the choice uses size and the stage 1 numbers only, never an accuracy result. It is reported against the pooled bf16 reference with its interval and the label NON-INFERIOR, WORSE or INCONCLUSIVE at the -3 margin, outside the confirmatory family, and descriptively against N5, N2, Q5_K_M and, when it exists, arm Q4. If no candidate passes, the stage 1 numbers are reported as the result. Nothing else is tuned; a different candidate set is a new amendment dated before it runs.

    *Clarification of amendment 25, written before the stage 1 batch runs ([removed from the public copy: internal process note]):* (a) only C1 and C2 can meet the size aim (5.77 GB against 5.99 GB); C3 has N5's size, so if it is the only passing candidate it still gets the stage 2 run but the report says it does not meet the size aim and answers only whether the scale search improves N5; (b) C1 and C2 have the same size, so a tie between them is broken by the lower mean KL, as stated; (c) selection uses shard bytes as recorded by the batch; (d) alongside the gate, the p99 KL, the top-1 agreement and the count of non-finite logprobs of every candidate are reported, and a candidate whose p99 KL exceeds N5's by more than 50% is flagged as such in the report (the flag does not change the gate); (e) stage 2 is judged descriptively against Q5_K_M with intervals, not as a proof of "no worse than Q5": one run, one build; (f) if the selected candidate fails the margin test the report says so and no further candidate is tried without a new amendment.

    *Departure from amendment 25 during stage 1, written before any stage 2 number exists (2026-11-21 08:27 UTC):* arms N5 (control), C1 and C2 were measured; C2 (`experts-all` with the scale search, 5,765,781,448 bytes) has mean KL 0.0234 and perplexity ratio 1.0148, C1 0.0263 and 1.0208, the control N5 0.0244 and 1.0199 (the control matches the earlier 0.0243). Both C1 and C2 pass the gate; C2 is selected (same size as C1, lower KL, as the rule says). C3 (`experts` with the scale search, 6.63 GB, the size of N5) was stopped manually while its checkpoint was being verified, to save about 25 minutes of a pod that ends at 11:27 UTC: it cannot be selected whatever it measures, since it is larger than the two passing candidates. The question "does the scale search improve N5" is therefore answered only through the pair C1 to C2 (same layout, same format, 0.0263 to 0.0234). The closing steps of `fidelity_batch.sh` were run by hand with the same arguments (`fidelity_gate.py pick` returned C2, then the move and `finalize --selected C2`); `selection.json` and `summary.json` are the script's own outputs. C3 is reported as not run.

    *Departure from amendment 24 (the Q4 speed sweep), written before any valid sweep number exists (2026-11-21 11:35 CET):* the first run of `bench_gguf.sh` (label Q4sw, pod q4-2, Q4 stand-in then OVH's Q5_K_M) had failed requests at 10 of its 12 points, at every concurrency including one request at a time, in both variants ("Never received a valid chunk to calculate TTFT" in each point's log). Cause: the client used `--backend openai-chat` against llama-server; its chat stream puts a reasoning model's first tokens in `reasoning_content`, and a 512-token random request can finish with no content chunk, which `vllm bench serve` marks failed; the tokens of failed requests are not counted in `output_throughput`, so those throughputs are biased low by an amount that changes from point to point. Under the rule of the throughput pre-flight (a point with failed requests is excluded) the sweep has no valid point. Fix: the client now uses the plain completions endpoint (`--backend openai --endpoint /v1/completions`), every other parameter unchanged (512 in, 512 out, `--ignore-eos`, concurrency 1, 4 and 12, N = max(8, 3 x concurrency), two repeats). The first sweep's raw files stay in `results/wave1/` as a record and are not used. The rerun is label Q4sw2 on pod q4-2; pod q4-1 receives the same script before its own sweep (label Q4-a). The vLLM sweeps of the other builds were not affected: they ran against vLLM, whose chat stream carries the parsed reasoning, and had no failed request.

### Amendment 2026-11-22 (arm B0g: OVH's bf16 GGUF on llama.cpp, the same-engine control; written before it runs)
Why: [removed from the public copy: internal process note] every llama.cpp row (Q5_K_M, the Q4_K_M stand-in) is compared with vLLM rows, so file format and engine are confounded. B0g separates them: OVH's own bf16 GGUF (`gguf_files.json`, variant bf16, sha256 b266da8e...) on the same llama.cpp fork commit, the same harness, the same five tasks and settings, three pods by task group exactly as for Q5v2 (mgsm+humaneval_plus, ifeval+mbpp_plus, aime2025; the AIME pod is given 6 hours because bf16 reads about twice the bytes per token of Q5_K_M).
Analysis (descriptive, outside the confirmatory family, nothing in the frozen confirmatory results changes): the same item-level paired bootstrap; B0g against vLLM bf16 (the pooled B0a+B0b) gives the engine effect; Q5_K_M and the Q4_K_M stand-in against B0g give the format effect on one engine. A single run, so provisional like every arm. Invalid-result conditions: accuracy 0.000 with request errors, a hash that differs from gguf_files.json, a run that did not reach the 32K context setting of the other arms.
