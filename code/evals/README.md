# evals: compare quantized variants of Olala-v1.0-mini-7A1B with the bf16 reference

Two instruments, both driven from the command line, both tested without a GPU:

| Tool | Question it answers |
|---|---|
| `run_tasks.py` + `compare.py` | Do task accuracies change (MGSM, HumanEval+, MBPP+, GSM8K, IFEval, AIME 2025)? |
| `fidelity.py` | Does the next-token distribution change (perplexity, KL divergence, top-1 agreement)? |

Rationale, tool popularity data, known limitations and everything that was not done are in
[DECISIONS.md](DECISIONS.md). Read its last section before trusting a number.

## Setup

```sh
cd evals
uv sync                      # Python 3.12, versions pinned in uv.lock
uv run pytest                # hermetic suite: no GPU, no network, no Docker
uv run ruff check . && uv run ruff format --check .
```

Serve every variant (bf16 and quantized) with the **same** vLLM options, in particular the ones from
the model card: `--mamba-ssm-cache-dtype float32 --reasoning-parser olala --max-model-len 32688`.
`fidelity.py` also needs `--max-logprobs` of at least the `--k` you ask for (vLLM default 20).

## 1. Task accuracy

### Run

```sh
# reference
uv run python run_tasks.py --base-url http://localhost:8000/v1 --model olala \
    --out ../results/raw/bf16 --seed 0 --n-items mgsm=550

# variant: score exactly the same items
uv run python run_tasks.py --base-url http://localhost:8001/v1 --model olala \
    --out ../results/raw/fp8 --items-from ../results/raw/bf16/items.json

# compare (markdown on stdout, optional --out/--json files)
uv run python compare.py ../results/raw/bf16 ../results/raw/fp8 --by-stratum \
    --out ../results/bf16_vs_fp8.md
```

`../results/raw/` is git-ignored by the repository; keep the generated comparison tables outside it.

Protocol defaults (OVH model card): zero-shot, `temperature 0.8`, `top_p 0.9`, `top_k 40`,
`reasoning_effort high`, no answer-length limit beyond the server's context window.

| Option | Default | Meaning |
|---|---|---|
| `--tasks` | all six | `mgsm humaneval_plus mbpp_plus gsm8k ifeval aime2025`; `ifbench` is accepted and recorded as skipped (not in inspect_evals) |
| `--base-url`, `--model` | `http://localhost:8000/v1`, `olala` | endpoint and served model name |
| `--out` | required | new directory; refuses to overwrite an existing run |
| `--seed` | 0 | seed of the **item selection** |
| `--n-items [TASK=]N` | all items | subset size, repeatable. MGSM is stratified by language (equal share per language, remainder to the first languages in sorted order). `--n-items mgsm=550` is 50 per language |
| `--items-from FILE` | none | `items.json` of another run: score exactly those items (overrides `--n-items`) |
| `--epochs [TASK=]N` | per task, the frozen protocol's: 5 for AIME 2025, 1 for every other task | samples per item, repeatable. A bare `N` applies to every task; `TASK=N` overrides one task and beats a bare `N`. An unknown task name is an error. The epochs actually used are in `run_info.json` (`tasks.<name>.epochs`, and `epochs_requested` for what was asked) |
| `--sandbox {docker,local}` | `docker` | where HumanEval+ and MBPP+ run model code; see below |
| `--temperature --top-p --top-k` | 0.8, 0.9, 40 | sampling |
| `--max-tokens` | none | cap on generated tokens; unset means the server context limit |
| `--reasoning-effort` | `high` | `disabled`, `low`, `medium`, `high`; sent as `chat_template_kwargs.reasoning_effort` |
| `--sampling-seed` | none | per-request vLLM seed; refused when a task has more than one epoch (every epoch would be identical) |
| `--max-retries` | 5 | retries of a failed request |
| `--timeout` | 14400 s | seconds per request including retries. It must exceed the worst-case generation time so that it never counts server-side queueing or long reasoning as a request error (it was 3600 s until 2026-11-20, see below). Also passed to inspect_ai's provider as its HTTP client timeout, the model argument `client_timeout` (see "Client timeout and retried requests") |
| `--max-connections [TASK=]N` | per task: `aime2025` 24, every other task 32 | requests in flight, also inspect_ai's `max_samples` (each task runs in its own `inspect_ai.eval` call). Repeatable. Precedence per task: an explicit `TASK=N`, then the built-in per-task default (`aime2025=24`), then a bare `N`, then 32. A bare `--max-connections 96` therefore gives 96 to every task except AIME 2025, which keeps 24 unless `--max-connections aime2025=N` says otherwise. An unknown task name is an error. `run_info.json` records `max_connections` (the bare value, or 32) and `max_connections_per_task` (the resolved value of each task that ran) |

Why AIME 2025 has its own concurrency and why the timeout is 4 hours: AIME requests can run to the
32k-token context and the vLLM server's KV cache holds only about 23 such sequences. On 2026-11-20
the harness kept 96 requests in flight, the surplus queued on the server, and the 3,600-s client
timeout counted the queueing: 34 request errors (`APITimeoutError` at about 3,600 s) in the arms
that reason longer, none in the others (PROTOCOL.md amendment 10). 24 is about what the cache holds (at most a
queue of one), so a request's clock measures its own generation, and the 4-hour timeout is only a guard
against a hung connection. `analyze_wave.py` does not require these two settings to match between
arms (they are non-binding) but prints a warning when two arms differ in `sampling.timeout` or in
`max_connections_per_task` for a task they share (for a run without that field the global
`max_connections` stands in).

#### Client timeout and retried requests (2026-11-21)

inspect_ai 0.3.291's OpenAI-compatible provider has its own HTTP client timeout, the model argument
`client_timeout`, 600 s unless given; `GenerateConfig(timeout=...)` does not change it. Until
2026-11-21 the harness set only the latter, so any request that needed more than 600 s was
cancelled by the client and silently sent again (the generation under way was thrown away and
redrawn, with no request error). `run_tasks.py` now passes `--timeout` to both (`build_model`), and
`tests/test_client_timeout.py` proves it against a local HTTP stub that sleeps a few seconds.

Every row of `samples.jsonl` now has `model_attempts` (HTTP requests the sample needed; 1 if the
first one was answered) and `working_time` (inspect_ai's seconds, may be `null`), and
`run_info.json` has `client_timeout`, `samples_with_retries` and `max_working_time`, per task and in
total. `model_attempts` is `sum(1 + retries)` over the sample's model events: inspect_ai records
the OpenAI client's retries as `ModelEvent.retries` of one event, so counting events misses a
request that was cancelled and answered at its second attempt. `analyze_wave.py` refuses a run in
which more than 0.5% of the samples of a task needed more than one request, or whose
`client_timeout` is below its `timeout`; a run made before this change (no `model_attempts`) gets
the warning "retry count unknown (client timeout 600 s legacy default)" per task, and for AIME is
refused unless `--accept-legacy-aime-retries` is given (the gate then warns and states what was
measured on the six saved AIME reruns).

Epoch defaults changed on 2026-11-20: HumanEval+ and MBPP+ used to default to 5, the frozen protocol
(PROTOCOL.md section 6) says 1 sample per item for them (only AIME 2025 has 5), and five samples
cost five times the output tokens for no protocol reason. To reproduce a run made with the old
defaults pass `--epochs humaneval_plus=5 --epochs mbpp_plus=5`.

HumanEval+ and MBPP+ execute model-written code, by default in a Docker sandbox (`plus_sandbox/`:
Python 3.12 and numpy, no network, 1 CPU, 2 GB). Docker must be running; the first run builds the
image. If a run is killed, remove leftovers with `docker ps -aq --filter name=inspect- | xargs docker rm -f`
and `docker images -q 'inspect-*' | xargs docker rmi`.

`--sandbox local` uses inspect_ai's local sandbox instead: the code runs in a subprocess of the
harness, in a fresh temporary directory, with the harness's own Python (same numpy as the image).
Use it only on a disposable machine that holds no secret and nothing to lose, such as a RunPod pod
(a pod is a container and has no Docker). The per-problem wall-clock limit (`VERIFY_TIMEOUT`, 60 s) and
the way the program is delivered (a file, not `python -c`) are the same in both modes; the
isolation is not (limits and risks in DECISIONS.md section 3). The mode is recorded in `run_info.json`
(`sandbox.mode`, `sandbox.tasks`).

### Output directory

| File | Content |
|---|---|
| `samples.jsonl` | one row per (task, item, epoch): `task, sample_id, epoch, stratum, score, scores, error, truncated, stop_reason, input_tokens, output_tokens, total_time, working_time, model_attempts, target, completion` (the last two timing fields are absent from runs made before 2026-11-21) |
| `items.json` | `{task: [item ids]}` actually run (feed to `--items-from`) |
| `run_info.json` | endpoint, model, git commit and dirtiness of `evals/`, sha256 of the harness sources, package versions, sampling parameters (including `timeout`), reasoning effort, `max_connections` and `max_connections_per_task`, selection seed and subset sizes, per-task epochs and `items_sha256`, `argv`, start and end UTC, request errors, truncations, skipped tasks, `client_timeout`, `samples_with_retries` and `max_working_time` (run-wide and per task), `status` (`running`, `complete`, `incomplete` if an inspect_ai task ended in error, `failed` plus `error` after a crash). A directory made by `merge_aime_rerun.py` also has a `merged` block |
| `logs/` | raw inspect_ai `.eval` logs |

The repository had no commit when this was written, so `git.commit` is `null` until one exists;
`code_sha256` identifies the harness version in the meantime.

### Replacing one task by its rerun: `merge_aime_rerun.py`

```sh
uv run python merge_aime_rerun.py --original ../results/raw/n2 --rerun ../results/raw/n2-aime \
    --out ../results/raw/n2-merged [--task aime2025] [--allow-incomplete-original]
```

Builds a NEW run directory for `analyze_wave.py` (PROTOCOL.md amendments 10 and 11: the AIME rows of
the 2026-11-20 runs contain client timeouts that were an artifact of the harness, so the task is
rerun with non-binding settings). The rows of every other task are copied from the original
unchanged, the rows of the replaced task come from the rerun unchanged, and `items.json` follows
(the original rows of that task are not copied: they stay in the original directory).
`run_info.json` is the original's with `n_samples`, `request_errors`, `truncations` and the
replaced task's counts and `items_sha256` recomputed from the merged rows, `status` `complete`, and
a `merged` block: both source directories, the sha256 of both `samples.jsonl` and `run_info.json`,
the replaced task, the reason, the settings that differ (`max_connections_per_task` of that task,
`timeout`), the row counts of the replaced task in the original and the rerun, and the original
rows' request-error and truncation counts for that task. `quantization_info.json` is copied from
the original if it has one. The merge does not hide request errors of the rerun: the gate of
`analyze_wave.py` (at most 0.5%) still applies to the merged directory.

It refuses (exit code 2, a message, nothing written) if the rerun is not complete (status, or any
row missing, duplicated or unexpected); if its item ids for the task differ from the original's; if
the sampling parameters, reasoning effort, chat-template arguments, sandbox mode, whether the task
runs in the sandbox, or the epochs of the task differ; if `--out` exists; or if the task is absent
from either `items.json`. The original must be `complete`; `--allow-incomplete-original` accepts
one that is not (for example a run that died during AIME) only if every other task in its
`items.json` has exactly `items x epochs` rows (the epochs from its `run_info.json`). A task of the
protocol that is missing from the original's `items.json` altogether is not detected here: the gate
of `analyze_wave.py` catches it.

### Combining per-task runs of one arm: `combine_task_runs.py`

```sh
uv run python combine_task_runs.py \
    --run ../results/raw/b0g-a:mgsm,humaneval_plus \
    --run ../results/raw/b0g-b:ifeval,mbpp_plus \
    --run ../results/raw/b0g-c:aime2025 \
    --out ../results/raw/b0g [--allow-partial-tasks]
```

For an arm scored on several pods at once, each pod running a task group with `run_tasks.py`
(one run directory per pod). Builds ONE new run directory with all five protocol tasks for
`analyze_wave.py`. For each `--run DIR:TASK[,TASK...]` only the listed tasks are taken, their rows
copied as the exact text lines of the source; the output holds the union, one source per task. The
source directories are not changed.

`run_info.json` is the first `--run`'s with `status` `complete`, the run counts and the per-task
entries (`n_items`, `epochs`, `n_samples`, `request_errors`, `truncations`, `items_sha256`)
recomputed from the rows, `sandbox.tasks` and `max_connections_per_task` taken from the source of
each task, and a `combined` block: every source directory with the sha256 of its `samples.jsonl`,
`run_info.json` and `items.json`, its tasks and row counts, its `max_connections_per_task`,
`timeout`, model, endpoint and arm, the sha256 of its `server_provenance.txt`, `server.cmd` and
`quantization_info.json`, and the compared provenance values. Those three files are copied from the
first `--run` if it has them. Timeout, concurrency and endpoint may differ between pods (recorded,
not compared); a differing model label, harness version or code hash is a warning on stderr.

It refuses (exit code 2, a message, nothing written: the directory is built in `.NAME.partial` and
renamed) if a task is listed twice, the same directory is given twice, or a listed task is not a
protocol task or is absent from its input's `items.json`; if an input's `run_info.json` status is
not `complete` (even when the listed tasks are done: a run with another task unfinished is not
accepted); if the rows of a listed task are not exactly the (item, epoch) set implied by its
`items.json` and epochs; if the sampling parameters, reasoning effort, chat-template arguments,
sandbox mode or item-selection seed differ between inputs (the list of `merge_aime_rerun.py` plus
`seed`); if `server_provenance.txt` differs between inputs in the arm, llama.cpp repository and
commit, GGUF size or sha256 (pinned and verified), or Hugging Face repository and revision, or is
present in some inputs only, or `quantization_info.json` differs or is present in some only; if the
union of tasks is not exactly the five protocol tasks (`--allow-partial-tasks` accepts a partial
union and the output says so: `PARTIAL` on stdout, `combined.missing_tasks`; `analyze_wave.py`
still refuses such a directory); or if `--out` exists. Request errors are not hidden: the gate of
`analyze_wave.py` (at most 0.5%) applies to the combined directory. The tests are in
`tests/test_combine_task_runs.py`; `mutation_check_combine_task_runs.py` weakens the tool 25 ways,
one at a time, and requires the tests to catch each.

### How `compare.py` scores

* a row counts as a failure (0) if it has a request error, a truncated answer (`max_tokens` or
  `model_length` stop reason), or no score, whatever the scorer said;
* the score of an item is the mean over its epochs, so a task with several epochs (AIME 2025: 5 by
  default) is pass@1 averaged over its samples;
* IFEval uses prompt-level strict accuracy (`prompt_level_strict`);
* accuracy and difference are computed over the items present in both runs; intervals are 95%
  percentile bootstrap over items (10,000 resamples, seed 0, both recorded in the output), and the
  difference (B minus A, in percentage points) is paired;
* request errors and truncations are listed in their own table, with the item ids;
* settings that differ between the two `run_info.json` files, different item sets and unpaired
  items are listed under *Warnings* and *Settings that differ*.

## 2. Logit fidelity

```sh
# once: wikitext-2 test split (Salesforce/wikitext, wikitext-2-raw-v1, via `datasets`)
uv run python fidelity.py fetch-wikitext --out data/wikitext2_test.txt

# once per text: tokenize with the model's tokenizer, cut into 64 x 512 token chunks
uv run python fidelity.py prepare --text data/wikitext2_test.txt \
    --tokenizer ../models/Olala-v1.0-mini-7A1B --out data/wikitext2_64x512.json --name wikitext2

# optional extra slices: any UTF-8 text file you supply (French prose, Python code, ...)
uv run python fidelity.py prepare --text my_french.txt --tokenizer ../models/Olala-v1.0-mini-7A1B \
    --out data/french_64x512.json --name french

# per variant: send the chunks, store top-k log-probs
uv run python fidelity.py collect --chunks data/wikitext2_64x512.json \
    --base-url http://localhost:8000/v1 --model olala --out ../results/raw/bf16_wikitext2.npz
uv run python fidelity.py collect --chunks data/wikitext2_64x512.json \
    --base-url http://localhost:8001/v1 --model olala --out ../results/raw/fp8_wikitext2.npz

# compare reference vs variant
uv run python fidelity.py compare ../results/raw/bf16_wikitext2.npz ../results/raw/fp8_wikitext2.npz \
    --json ../results/fp8_wikitext2_fidelity.json
```

`collect` sends token ids (not text), so every variant sees identical tokens, with
`POST /v1/completions {"echo": true, "max_tokens": 1, "logprobs": 20, "temperature": 0,
"return_tokens_as_token_ids": true}` (vLLM style). `--mode prompt_logprobs` uses vLLM's
`prompt_logprobs` field instead. It checks that the server echoes back the prompt tokens that were
sent and that the prompt token count matches, so a server that prepends a BOS token is detected
rather than silently shifting every position. Requests are retried on transport errors, 429 and 5xx
(`--retries`, `--backoff`); other 4xx errors (for example `logprobs` above `--max-logprobs`) fail at
once with the server's message.

The `.npz` holds `vocab`, `top_ids`, `top_lp`, `target_lp` and `meta_json` (endpoint, model, mode,
k, exact request template, chunk-set hash, server model info, versions, UTC times, number of
retries; never the API key). `compare` refuses files made from different chunk sets.

Reported by `compare` (each with a 95% bootstrap interval over chunks, 1000 resamples, seed 0):
perplexity of each file and their ratio (paired), mean, median and 99th-percentile
`KL(reference || variant)` in nats, and the fraction of positions with the same top-1 token.
The KL is an approximation (top-k union plus one 'other' bucket, see DECISIONS.md section 6);
`--max-k 5` or `--max-k 10` recomputes it from shorter lists to see how sensitive it is to k.

## 3. Tests and checks

```sh
uv run pytest                                   # default: hermetic, fake server only
uv run pytest -m network                        # real dataset wiring (needs Hugging Face, Azure blob)
uv run pytest -m docker                         # sandbox scorers (needs Docker; network for datasets)
uv run pytest -m "network and docker"           # every real task through run_tasks.py (fake model)
uv run pytest -m "network and not docker" tests/test_check_plus_canonical.py   # canonical solutions, local sandbox
uv run python check_plus_canonical.py --out validation/plus_canonical.json   # harness validation (docker)
uv run python check_plus_canonical.py --sandbox local --out validation/plus_canonical_local.json   # same, no Docker
uv run python repo_stats.py --json repo_stats.json                           # popularity data
```

## 4. Pilot summary

```sh
uv run python pilot_summary.py ../results/raw/pilot --tok-per-s 2500 [--json summary.json]
```

For each task of a run directory: items, epochs, samples, mean output tokens per sample, total
output tokens, truncations, request errors and accuracy (scored as `compare.py` does). Then the
projection to the frozen protocol's item set (MGSM 550, HumanEval+ 164, MBPP+ 378, IFEval 541, AIME 2025
30 x 5 samples): output tokens = pilot mean tokens per sample x protocol samples, and hours at the
aggregate decode throughput given by `--tok-per-s` (tokens per second summed over all concurrent
requests). A linear extrapolation: prefill, code-execution time and the tail of slow requests are not
in it, and protocol tasks missing from the pilot are named as missing, not guessed. GSM8K is reported
but outside the projection (it is not part of the primary metric).

## Files

```
run_tasks.py            inspect_ai wrapper (task registry, item selection, rows, run_info)
plus_tasks.py           HumanEval+ and MBPP+ tasks (not in inspect_evals)
plus_sandbox/           Dockerfile + compose.yaml for the code-execution sandbox
compare.py              paired bootstrap between two run directories -> markdown
merge_aime_rerun.py     replaces one task of a full run by its rerun -> a new run directory for analysis
combine_task_runs.py     takes listed tasks from several per-task run directories of one arm -> one run directory
mutation_check_combine_task_runs.py  weakens combine_task_runs.py one way at a time; the tests must catch each
fidelity.py             fetch-wikitext / prepare / collect / compare
fidelity_math.py        KL, perplexity, top-1 agreement (pure numpy)
stats.py                percentile bootstrap helpers
pilot_summary.py        per-task token and accuracy summary of a run, projection to the full protocol
check_plus_canonical.py validates plus_tasks.py with the EvalPlus reference solutions
repo_stats.py           popularity and maintenance table for DECISIONS.md
tests/                  pytest suite and the fake OpenAI-compatible server (tests/fake_server.py)
```
