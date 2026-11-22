# Decisions, limitations and what is not done

Written 2026-11-20 together with the code. Tables marked *generated* were produced by a script and
inserted by a script, not typed; their source file is named under them.

## 1. What the protocol is, and where it comes from

OVH's own description (model card `models/Olala-v1.0-mini-7A1B/README.md`, "Evaluation details"):
every model was run "with the same evaluation code (inspect_ai querying a vLLM server)", "zero-shot",
"with the sampling parameters from its model card", "the same context window and no separate limit
on answer length"; Olala uses `temperature=0.8`, `top_p=0.9`, `top_k=40`, `reasoning_effort="high"`
and a float32 Mamba state. HumanEval+, MBPP+ and AIME are pass@1 averaged over 5 samples; the other
benchmarks use one sample per question; MGSM covers 11 languages.
The card documents the reasoning level as a chat-template argument:
`"chat_template_kwargs": {"reasoning_effort": "..."}` in a vLLM request, with levels `disabled`,
`low`, `medium`, `high`. The vLLM options it requires: `--mamba-ssm-cache-dtype float32`,
`--reasoning-parser olala`, `--max-model-len 32688`.

Not published by OVH, so chosen here and recorded in every `run_info.json`: the prompts of
HumanEval+, MBPP+ and AIME, the answer extraction, the answer-length limit (none beyond the context
window, which is what the card says), the code-execution timeout, the number of items used.

## 2. Tool choices

Principle: prefer popular, maintained, open-source tools; write bespoke code only for what none of
them provides. Popularity and maintenance (*generated* by `repo_stats.py`; raw rows in
`repo_stats.json`):

Queried with `gh api` on 2026-11-20 (GitHub REST API).

| Repository | Stars | Last push | Latest release | Release date | License | Archived |
|---|---|---|---|---|---|---|
| UKGovernmentBEIS/inspect_ai | 3294 | 2026-11-20 | 0.3.291 | 2026-11-17 | MIT | False |
| UKGovernmentBEIS/inspect_evals | 750 | 2026-11-19 | v0.29.0 | 2026-11-13 | MIT | False |
| evalplus/evalplus | 1839 | 2025-10-02 | v0.3.1 | 2024-10-20 | Apache-2.0 | False |
| josejg/instruction_following_eval | 18 | 2025-01-16 | none | none | Apache-2.0 | False |
| EleutherAI/lm-evaluation-harness | 14505 | 2026-11-04 | v0.4.13 | 2026-08-31 | MIT | False |
| openai/simple-evals | 4710 | 2026-04-22 | none | none | MIT | False |
| allenai/IFBench | 183 | 2026-11-13 | none | none | Apache-2.0 | False |
| vllm-project/vllm | 97254 | 2026-11-20 | v0.34.0 | 2026-11-16 | Apache-2.0 | False |

| Need | Chosen | Why | Alternatives looked at |
|---|---|---|---|
| Task runner against an OpenAI-compatible endpoint | `inspect_ai` | OVH used it, so the harness, scorers and answer extraction match their protocol; actively released | `lm-evaluation-harness` has more stars but is not what OVH used; `simple-evals` has no releases and no packaging |
| Task definitions | `inspect_evals` for MGSM, GSM8K, IFEval, AIME 2025 | maintained with `inspect_ai`; datasets pinned to revisions | none needed |
| HumanEval+, MBPP+ | **own minimal tasks** (`plus_tasks.py`) | `inspect_evals` 0.29.0 has only base HumanEval and MBPP (checked: no `evalplus` or plus task, MBPP there is few-shot at temperature 0.5) | `evalplus` itself: 1.8k stars but last release and last push are old (table above), and it is a command-line tool that does its own generation; its HF datasets carry the extended tests as executable scripts, so only prompt, extraction and a sandboxed executor had to be written |
| IFEval scoring | `instruction_following_eval` fork pinned to commit `0c495b2` | what `inspect_evals` itself requires; **weakly maintained** (see table), hence pinned and not upgraded | official google-research code is not installable |
| IFBench | **not done** | not in `inspect_evals` 0.29.0; implementing it means vendoring the OOD-constraint verifiers of `allenai/IFBench` (not a small job) | |
| Bootstrap | numpy, `stats.py` | a percentile bootstrap is a few lines and avoids adding scipy for it; the code is tested against known answers (constant data, constant shift, pairing narrows the interval) | `scipy.stats.bootstrap` (could do it with item or chunk indices as the data) |
| Tokenizer for chunking | `tokenizers` (Hugging Face) on the local `tokenizer.json` | tiny dependency, exactly the model's tokenizer | `transformers` (heavy), vLLM `/tokenize` (needs a server) |
| HTTP | `httpx` | already a dependency of the stack | `openai` SDK (extra fields such as `prompt_logprobs` are awkward) |

Versions pinned in `uv.lock` (*generated* from the lock file):

| Package | Version |
|---|---|
| inspect-ai | 0.3.291 |
| inspect-evals | 0.29.0 |
| instruction-following-eval | 0.1.0 |
| openai | 3.57.0 |
| httpx | 0.28.1 |
| datasets | 5.2.0 |
| huggingface-hub | 2.4.0 |
| numpy | 2.5.5 |
| tokenizers | 0.23.3 |
| langdetect | 1.0.9 |
| nltk | 3.10.4 |
| pytest | 9.2.0 |
| ruff | 0.17.6 |
| Python | 3.12.x (uv-managed) |

## 3. Tasks: what runs and how faithful it is

| Task | Source | Epochs | Notes |
|---|---|---|---|
| `mgsm` | `inspect_evals.mgsm`, all 11 languages (simple-evals TSVs, sha256-verified) | 1 | prompt is the native-language chain-of-thought instruction of simple-evals; `shuffle` is never used (it is seeded with the wall clock); subsets are chosen by this harness, stratified by language |
| `gsm8k` | `inspect_evals.gsm8k` with `fewshot=0` | 1 | `inspect_evals` defaults to 10-shot; the protocol is zero-shot |
| `ifeval` | `inspect_evals.ifeval` | 1 | primary metric: prompt-level strict accuracy; loose and instruction-level values stay in `scores` of each row |
| `aime2025` | `inspect_evals.aime2025` (`math-ai/aime25`, 30 problems) | 5 | |
| `humaneval_plus` | `plus_tasks.py`, `evalplus/humanevalplus` | 1 | was 5 until 2026-11-20, see below |
| `mbpp_plus` | `plus_tasks.py`, `evalplus/mbppplus` | 1 | was 5 until 2026-11-20, see below |
| `ifbench` | not available | | recorded as skipped in `run_info.json`, never silently dropped |

**Epoch defaults changed (2026-11-20).** `run_tasks.py` defaulted HumanEval+ and MBPP+ to 5 epochs, copied
from OVH's description ("pass@1 averaged over 5 samples" for HumanEval+, MBPP+ and AIME, section 1). Our
frozen protocol (PROTOCOL.md section 6) is different and binds this study: AIME 2025 = 5 samples per problem,
HumanEval+ = MBPP+ = MGSM = IFEval = 1 sample per item (GSM8K, outside the primary metric, 1 as well). The defaults now are the protocol's (AIME 2025 5,
everything else 1), for two reasons: the code must run what the protocol froze, and token cost (a reasoning model
at `medium` effort with no answer-length limit; five samples of 164 + 378 code items is 2,710 requests instead of
542 for nothing the protocol asks for). `--epochs [TASK=]N` (repeatable; a bare N still means every task; an unknown
task name is an error) overrides one task; `run_info.json` records the epochs used per task (`tasks.<name>.epochs`)
and what was requested (`epochs_requested`). Consequence for comparisons: a run made before this change used 5
epochs for the two code tasks, and `compare.py` pairs items by id and averages epochs, so such a run is still
comparable item by item, but its interval is not the interval of a 1-sample run; runs meant to be compared should
use the same epochs (`compare.py` already warns "different number of epochs" when they differ).

Validation of `plus_tasks.py`: the EvalPlus reference solutions must all pass the sandboxed scorer
(`check_plus_canonical.py`, output below, *generated* from `validation/plus_canonical.json`):

Checked 2026-11-20T09:46:37Z, per-problem time limit 60 s.

| Task | Dataset | Problems | Reference solutions passing | Errors | Not passing |
|---|---|---|---|---|---|
| humaneval_plus | evalplus/humanevalplus@d32357cf | 164 | 163 | 0 | HumanEval/32 |
| mbpp_plus | evalplus/mbppplus@b2d74c91 | 378 | 377 | 0 | 590 |

The remaining failures are not harness faults: `HumanEval/32` fails inside the dataset's own test
(`_poly(*candidate(*inp), inp)` unpacks a float: the test script of that problem does not match its
reference solution) and MBPP `590` fails an exact float comparison that differs in the last digit
(reference output computed on another platform). Both are in the dataset revisions pinned here; they
fail for every model and variant alike, so they do not bias the paired comparison, but the maximum
attainable score is slightly below 100%.

The same check was run with the **local** sandbox (`--sandbox local`, next subsection) on 2026-11-20 at
11:51 UTC, on the Mac (macOS, arm64), `validation/plus_canonical_local.json`:

| Task | Problems | Reference solutions passing | Errors | Not passing |
|---|---|---|---|---|
| humaneval_plus | 164 | 163 | 0 | HumanEval/32 |
| mbpp_plus | 378 | 377 | 0 | 590 |

Identical to the docker run (163/164 and 377/378, the same two ids), so the two dataset faults above do not
come from the container. Not yet measured: the same check on a pod (Linux, x86-64); `590` is a float-comparison
fault that depends on the platform, so a different failing set there would be a platform finding, not a harness fault.

**Sandbox modes and what they isolate (`--sandbox {docker,local}`, default docker).** A RunPod pod is a container
and has no Docker daemon, so the Docker sandbox cannot run there. `--sandbox local` uses inspect_ai's local sandbox: the
program is written to a file in a fresh temporary directory (one per problem) and run as a subprocess of the harness
with the harness's own interpreter (the same numpy 2.5.5 as the image), with the same wall-clock limit
(`VERIFY_TIMEOUT` = 60 s, passed to the same `exec` call in both modes, tested). What the local mode does NOT give,
and the docker mode does:

| | docker | local |
|---|---|---|
| network | none (`network_mode: none`) | the pod's: model code can reach the Internet and the pod's other ports |
| file system | the container's own | the pod's: model code can read and overwrite `/workspace` (weights, results, logs, the run's `samples.jsonl`) |
| memory and CPU | 2 GB, 1 CPU per problem | none per problem (only the container's limits of the pod) |
| user | root in a throw-away container | the user of the harness (root on a pod) |
| processes left behind | removed with the container | a timeout kills the direct child only; grandchildren started by model code can survive until the pod ends |
| host | the Mac or pod never runs model code | the pod runs model code directly |

Why this is acceptable here, and only here: the pod is disposable (a watchdog deletes it), holds no secret (no
token of any kind lives on a pod, `HF_TOKEN` is unset by the pod scripts, the API key never leaves the Mac), and the
results are salvaged to the Mac before deletion, so a model that overwrote its own rows would be seen in the
comparison but could still corrupt a run; the code is written by the model under test, and a bf16 or quantized Olala
is not an adversary but is also not trusted. Do not use `--sandbox local` on the Mac or on any machine with
credentials; the Mac keeps the docker default. Memory limit absent means a runaway program is stopped only by the
60 s wall-clock limit and the pod's own memory limit.

What differs from the EvalPlus command-line tool: one wall-clock limit per problem (60 s) instead
of a per-input limit scaled from the reference runtime; the prompts are zero-shot chat prompts
written here; the scripts run in Docker (Python 3.12, numpy) or, with `--sandbox local`, in a subprocess, instead of EvalPlus's
in-process guard.
The first version passed the program with `python -c` and some reference solutions failed with
"argument list too long" (the plus test scripts are large); it is now written to a file, with a
regression test. Absolute pass rates should not be compared one-to-one with published EvalPlus numbers;
they are comparable between two variants run through this harness, which is the use case.

## 4. Sampling, reasoning effort, seeds, truncation

* **top_k is sent by hand.** inspect_ai 0.3.291 does not forward `GenerateConfig.top_k` to
  OpenAI-compatible providers (it is dropped silently; characterisation test
  `test_inspect_ai_does_not_forward_generate_config_top_k...`). `run_tasks.py` therefore sends
  `top_k` and `chat_template_kwargs` in `extra_body`. If OVH's run used `GenerateConfig.top_k`, their
  top_k may not have been applied; this cannot be checked from here.
* **Reasoning effort** goes in `chat_template_kwargs.reasoning_effort`, as the model card says. In
  the vLLM source tree under `recon/src/vllm` (commit `ebbadef5`) a top-level `reasoning_effort`
  request field is also merged into the template kwargs; only one route is used so they cannot
  disagree. `disabled` is accepted by the Olala template (it closes the analysis channel empty) but
  was not exercised against the real model.
* **Provider.** `openai-api/vllm/<model>` (inspect_ai's generic provider) rather than the dedicated
  `vllm/` provider: the latter keeps a process-wide registry keyed by model name, so a second endpoint
  for the same model name in one process silently reuses the first URL (found by a failing test). A
  real run uses one endpoint per process, so this only matters for tests and scripted comparisons.
* **Seeds.** `--seed` selects items and nothing else. Sampling is not seeded by default because
  every epoch of an item sends the same request, so a fixed per-request seed would be expected to
  make the 5 samples of an item (near) identical (reasoned from how seeded sampling works, not tested
  against a real server). `--sampling-seed` exists for one-epoch runs and is refused otherwise. Sampled
  tokens of two different numerical variants are not comparable even with equal seeds, so the paired
  bootstrap over items, not seed matching, is the noise control.
* **max_tokens** is unset by default (the card: no separate limit). A reasoning model that hits the
  context limit is a failure of the variant, so truncations count as failures in `compare.py` and
  are listed separately. The card itself notes that RL makes the model "hit the token limit more
  often while reasoning", so a quantized variant that reasons longer is expected to show up here.
* **Request errors** are retried by inspect_ai up to `--max-retries`; a sample that still fails is
  recorded (`error`, score null) and counts as a failure; the run does not stop.
* **Concurrency and client timeout are per-task, non-binding settings (2026-11-20, PROTOCOL.md
  amendment 10).** The first accuracy runs kept 96 requests in flight (`run_accuracy.sh`) with a
  3,600-s client timeout. The vLLM server's KV cache holds about 23 full-length (32k-token)
  sequences, so long AIME 2025 requests queued on the server and the client's clock counted the
  queueing: 34 `APITimeoutError` (3,606 to 3,620 s) in some arms, 0 in others, an artifact that
  penalised the arms that reason longer (cause read from the server logs and the settings, not
  tested by a separate experiment). The harness now (a) takes `--max-connections [TASK=]N`,
  repeatable, with the precedence explicit `TASK=N`, then the built-in `aime2025: 24`, then a bare
  `N`, then 32, and uses the resolved value for both `max_connections` and `max_samples` of that
  task's own `inspect_ai.eval` call; a bare 96 therefore cannot reach AIME. 24 is the 23 or so sequences the
  cache holds (an estimate from the server logs, so a queue of one request is possible but it
  clears at the first completion, not after an hour). (b) Raises the default `--timeout` to
  14,400 s: it must exceed the worst-case generation of one request, so that it never turns queueing
  or long reasoning into a request error, and only guards against a hung connection. `run_info.json`
  records `max_connections` (the bare value, or 32), `max_connections_per_task` (resolved, tasks
  that ran) and `sampling.timeout`. Neither is in `analyze_wave.py`'s must-match list (a run made
  with other values is still a valid, comparable run: the settings are non-binding by design), but
  the analysis prints a warning when two arms differ in either. Arms that had the artifact are
  rerun on AIME alone and merged with `merge_aime_rerun.py` (README), which records both sources,
  their hashes, the settings that differ and the replaced rows' error and truncation counts.
* **What the merge does not decide.** `merge_aime_rerun.py` keeps the original's top-level
  `max_connections`, `max_connections_per_task` and `sampling.timeout` (they describe the other
  tasks' rows); the replaced task's actual values are in `merged.settings_differing`. It does not
  compare `max_retries` (a robustness setting, not a sampling one), the harness code hash or the
  package versions of the two runs; the gate's existing warnings cover the latter two between arms.

## 5. Statistics

* Unit of resampling for task accuracy: the item (epochs of one item are correlated, so they are
  averaged first). 95% percentile bootstrap, 10,000 resamples, seed recorded. Difference: paired
  (same resampled items in both runs).
* AIME 2025 has 30 items: even with 5 epochs the interval on the difference is wide (items, not
  samples, are the independent unit). It is reported anyway; do not read a small AIME difference as
  signal. The same holds for MGSM subsets of a few dozen items per language.
* No multiple-comparison correction across tasks or variants.
* Fidelity: the unit is the 512-token chunk (positions inside a chunk are correlated);
  1000 resamples. Consecutive chunks of one text are not independent either, so the intervals are
  somewhat optimistic.
* Perplexity is that of the model's own tokenizer, comparable between variants of the same model only.

## 6. Fidelity tool: the KL approximation and its limits

`/v1/completions` returns only the top-k log-probabilities of each position (k = 20 by default), so
the full-vocabulary divergence cannot be computed. Method (`fidelity_math.py`), per position, for
`KL(P_ref || Q_variant)`:

1. take the union of the two top-k token sets and one extra bucket `other` for all other tokens;
2. a token of the union that a distribution does not list gets
   `min(p_k, unseen / n_missing)`, where `p_k` is the smallest probability that distribution lists
   (an upper bound on any unlisted token) and `unseen = 1 - sum(listed)`;
3. `other` gets what remains of the unseen mass;
4. both vectors are renormalised to sum to one (rounding of printed log-probs), then the usual sum.

Properties and limits:

* **Not the full-vocabulary KL.** If the two top-k sets coincide, lumping the tail is a
  coarse-graining and the result is a mathematical *lower bound* of the true KL. When sets differ,
  the imputation breaks that guarantee: on a toy bigram model with a 64-token vocabulary
  (`tests/test_fidelity_collect.py::test_truncated_top_k_is_close_to_the_exact_divergence`) the
  estimate was below the exact value and the gap grew as k shrank; the test pins it within 15% at
  k = 20. The real vocabulary is far larger and the distributions differ; the direction and size
  of the error on the real model are **unknown**. Use `compare --max-k 5/10` to see how much the
  number moves with k; if it still moves, the k = 20 number is not converged.
* The tail of a real LM is heavy, and the mass lumped into `other` can differ between the two models
  (for example a quantized model being slightly flatter). This part is measured (in the `other`
  bucket) but its internal structure is not.
* **Teacher-forced, on fixed text.** It measures the divergence of next-token distributions given
  the same reference text, not the divergence of generated reasoning chains, and not behaviour at
  temperature 0.8 / top-p / top-k. Small KL does not guarantee equal task accuracy and the reverse.
* **What vLLM returns.** Log-probs are the raw (pre-temperature) values (`logprobs_mode` default
  `raw_logprobs` in the vLLM tree reviewed); `-inf` is clamped to -9999 and treated as probability
  zero; the actual token is returned as an extra entry when outside the top-k and is dropped from
  the top-k list (it is kept as `target_lp`). Echo response format was read from
  `vllm/entrypoints/openai/completion/serving.py` in `recon/src/vllm`; it was **not** run against a
  real server (no GPU here), only against `tests/fake_server.py`, which implements the same
  fields. If a newer vLLM changes the response layout the parser will raise rather than guess.
* Position 0 of each chunk has no context and is not scored; the position produced by `max_tokens=1`
  is discarded. No BOS token is added (the tokenizer has none by default; `--add-special-tokens`
  exists).
* Tokens are identified by vLLM `token_id:N` strings (`return_tokens_as_token_ids`). A non-vLLM server
  would return decoded strings, which can collide for different ids; the tool then compares strings.
* Hybrid Mamba model: nothing in the tool is architecture-specific, but both servers must use the same
  `--mamba-ssm-cache-dtype`, otherwise the comparison measures that flag instead of quantization.
  Whether `prompt_logprobs` works with vLLM's prefix caching for this hybrid model was not checked.

## 7. Data fetched (provenance)

| What | From | Pinned by | Used for |
|---|---|---|---|
| wikitext-2 raw, test split (the whole config was downloaded by `datasets`, only `test` is used) | Hugging Face dataset `Salesforce/wikitext`, config `wikitext-2-raw-v1`, through `datasets` and `huggingface_hub` | commit hash recorded below | `fidelity.py fetch-wikitext` |
| `evalplus/humanevalplus`, `evalplus/mbppplus` | Hugging Face datasets | revisions in `plus_tasks.py` | HumanEval+, MBPP+ |
| GSM8K, IFEval, AIME 2025 | Hugging Face datasets (`openai/gsm8k`, `google/IFEval`, `math-ai/aime25`) | revisions in `inspect_evals` | tasks |
| MGSM TSV files | `openaipublic.blob.core.windows.net/simple-evals` through `inspect_evals` | sha256 checked by `inspect_evals` | MGSM |
| NLTK data, `python:3.12-slim` image, `numpy` in the image | NLTK data host, Docker Hub, PyPI | numpy pinned; image tag not digest-pinned | IFEval scorer, code sandbox |
| Olala `tokenizer.json` | read from `../models/Olala-v1.0-mini-7A1B` (already on disk, not downloaded) | sha256 in `chunks.json` | chunking |

No model API other than the local fake server was called. Nothing was downloaded from sources outside
the above. *Generated* provenance of the wikitext file (`data/wikitext2_test.txt.meta.json`):

```json
{
  "repo": "Salesforce/wikitext",
  "config": "wikitext-2-raw-v1",
  "split": "test",
  "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
  "n_rows": 4358,
  "n_chars": 1285622,
  "text_sha256": "bbf94c53a05abe9ee670d3b6343608095822c85e26de37c70b24fc571964574a",
  "fetched_utc": "2026-11-20T09:28:20Z",
  "datasets_version": "5.2.0"
}
```

Chunk set made from it with the Olala tokenizer (`data/wikitext2_test_64x512.json`):

```json
{
  "name": "wikitext2-test",
  "n_chunks": 64,
  "chunk_len": 512,
  "tokens_available": 298938,
  "tokenizer_sha256": "286ead4e29f85215a29ffc035c2529c635ed2dc390d101c8738ab7bcf8f984b5",
  "chunks_sha256": "ca949342463d9739660d94243bbd156d7e1b0f29fa54d2cdd8b866e0b95e94ff"
}
```

## 8. Not done, not verified

* **Nothing was run against the real model or any real vLLM server** (no GPU in this environment).
  `fidelity.py` and `run_tasks.py` are verified against a fake OpenAI/vLLM-compatible server and, for
  the response formats, against the vLLM sources in `recon/src/vllm`. The first real run may reveal a
  difference (response fields, `--max-logprobs`, reasoning-parser output handling).
* **IFBench**: not in inspect_evals, not implemented. Multi-IF, LIFEBench, GPQA, TruthfulQA, DROP,
  BFCL, AIME 2024/2026 from OVH's table are outside the requested protocol.
* **French and Python fidelity slices** are supported (`prepare --text FILE`) but no such files were
  produced or supplied; only wikitext-2 was prepared.
* **OVH's exact prompts and answer extraction** for HumanEval+, MBPP+ and AIME are unknown, so
  absolute scores will not match the model card; only variant-versus-reference differences are meaningful.
* **No statistical power analysis** (how many MGSM/GSM8K items are needed to detect a 1 point drop).
  The intervals are reported; sizing is left to the first real comparison.
* The `disabled` reasoning level, `reasoning_content` handling by the reasoning parser, and long
  generations near the 32,688-token context were not exercised.
* Docker base image is referenced by tag, not digest.
* **`--sandbox local` on a pod, and the pod-side runner** (`runpod/pod/setup_evals_env.sh`, `run_accuracy.sh`) were
  tested with a stub `uv`, a stub server and the fake RunPod, and the local sandbox with the real EvalPlus datasets
  on the Mac; nothing of this has met a real pod (Linux x86-64, the pod's PATH, PyPI and GitHub access for `uv sync`
  and the pinned `instruction-following-eval`, the NLTK host, the Hub).
* **`pilot_summary.py`'s projection is linear** (pilot mean tokens per sample x protocol samples; hours at one
  aggregate throughput figure): prefill, code-execution time, the tail of slow requests and the pilot's item mix
  versus the full set are not modelled.
* Nothing was committed or pushed (the repository has no commit yet; `run_info.json` records `null`
  for the commit and a hash of the harness sources instead).
