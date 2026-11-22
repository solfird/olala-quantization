<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/solfird-ai-on-dark.svg">
  <img src="assets/solfird-ai-on-light.svg" alt="Solfird AI" width="240">
</picture>

# Olala-v1.0-mini-7A1B under Quantization: What Holds, What Breaks, and Where

Solfird AI Technical Report 2026-04 · November 2026 · Julien Simon

[Report (PDF, 22 pages)](olala-quantization.pdf) · DOI [10.5281/zenodo.24443547](https://doi.org/10.5281/zenodo.24443547) (version 1.0) · [NVFP4 checkpoint (public)](https://huggingface.co/solfird/Olala-v1.0-mini-7A1B-NVFP4-experts) · OVH's [model](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B), [technical report](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B/blob/main/Olala-v1.0-mini.pdf) and [GGUF files](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B-GGUF)

## The result

A 4-bit NVFP4 build of Olala-v1.0-mini-7A1B that quantizes only the routed experts, in all 36 layers, with a per-group scale search, is 5.77 GB and scores 1.6 points below bf16 on the average of five tasks. It scores 0.4 points below OVH's 5.99 GB Q5_K_M file, with an interval from -2.5 to +1.7, so the two cannot be told apart. Both FP8 builds pass the three-point test (-0.1 and -0.9 points). A Q4_K_M file made with OVH's recipe, about the size of the new build, is 2.5 points behind it, mostly on AIME (2.0 of the 2.5 points), where it runs out of context more often. A bf16 run on llama.cpp scores 0.8 points below bf16 on vLLM (-2.4 to +0.9), so the file format and the serving engine are mixed in every llama.cpp row. A build calibrated on the model's own reasoning traces did worse on plain text than plain rounding (mean KL 0.0290 against 0.0243). NVIDIA's Nemotron 3 Super checkpoint also chooses its block scales by a squared-error sweep, so the idea is not new.

For a builder: serve FP8 when memory allows. For 4-bit on a GPU, quantize the routed experts of every layer and keep the small tensors in bf16, with scales chosen by the search. Do not assume that calibration helps; test it on text outside the calibration data first.

OVHcloud released Olala-v1.0-mini-7A1B, a small hybrid Mamba and mixture-of-experts model, with a technical report that says why it is hard to quantize. I tested that report's claims about the activations on the released weights and measured what FP8, NVFP4 and GGUF builds cost in accuracy and speed, on RTX PRO 4500 Blackwell cards with OVH's forks of vLLM and llama.cpp. One run per build, five tasks, no larger model.

![Forest plot of the accuracy change of eight builds against bf16, each with a 95 percent interval, ordered by file size from 13.58 GB to 5.69 GB. Both FP8 builds sit just left of zero inside the three-point margin. The new 5.77 GB NVFP4 build sits near the OVH Q5_K_M file, and the Q4_K_M stand-in sits lowest.](figures/fig_teaser.png)

*Accuracy change against bf16, in points, with 95% intervals (paired bootstrap over items), for eight builds ordered by file size. A dashed line marks the three-point margin. The Q5_K_M file and the Q4_K_M stand-in ran on llama.cpp, the others on vLLM.*

## Main results

| Result | Value | Basis |
|---|---|---|
| FP8 online against bf16 | -0.1 (-1.5 to +1.3) points; passes the three-point test | Confirmatory test fixed before the runs |
| FP8 offline against bf16 | -0.9 (-2.7 to +0.8) points; 8.38 GB; passes the three-point test | Confirmatory test fixed before the runs |
| NVFP4, all quantizable layers, 6.04 GB | -3.3 (-5.3 to -1.4) points; inconclusive at the margin | Confirmatory test, one run |
| NVFP4, routed experts of 32 layers, 6.63 GB | -1.2 (-3.0 to +0.6) points | Descriptive, one run |
| NVFP4, same experts, calibrated | -2.4 (-4.2 to -0.7) points; mean KL 0.0290 against 0.0243 for plain rounding | Descriptive, one run |
| New build: routed experts of all 36 layers, scale search | 5.77 GB, 6.8 bits per weight; -1.6 (-3.3 to 0.0) points against bf16 | Descriptive, one run, selected by a gate fixed before it was built |
| New build against OVH's Q5_K_M (5.99 GB, llama.cpp) | -0.4 (-2.5 to +1.7) points; 0.22 GB smaller | Descriptive, crosses engines |
| Q4_K_M stand-in made with OVH's recipe | 5.69 GB; -4.1 (-6.2 to -2.1) points against bf16; the new build is ahead by +2.5 (+0.2 to +4.8) points | Descriptive, one run, my reconstruction of OVH's file |
| Gap to the stand-in without AIME | +0.6 (-1.2 to +2.5) points; the stand-in cut off 90 answers, the new build 35 | Descriptive |
| Engine effect: OVH's bf16 GGUF on llama.cpp against bf16 on vLLM | 72.7 against 73.5; -0.8 (-2.4 to +0.9) points; without AIME +0.2 (-1.3 to +1.8) | Descriptive, one run |
| Where the 4-bit change in predictions sits | Quantized tensors outside the routed experts are 6 percent of the parameters and give a mean KL of 0.0498; the routed experts are 71 percent and give 0.0243 | Measured on plain text, a fidelity measure and not accuracy |
| Speed under load, 16 to 128 concurrent requests, relative to bf16 | New build 1.21 to 1.42; FP8 online 1.15 to 1.24; FP8 offline 1.13 to 1.23 | Measured, synthetic requests, one card type |
| OVH's activation claims | Mamba layers 1, 7 and 23 reach 3.6 × 10^16; three more Mamba layers (0, 2, 3) exceed the FP8 range | Measured, pure-PyTorch path on 80 prompts |

Builds are labeled by the interval against bf16 and a margin of three points. Every build has one run, so every label is provisional; the two bf16 runs differ by -0.6 points (-2.8 to +1.6). The scope is one model, one card type and five tasks.

## Reproduce the numbers and the figures

No GPU, model, key or network access to a model host is needed. You need bash and [`uv`](https://docs.astral.sh/uv/) (it fetches Python 3.12, numpy 2.5.5 and matplotlib 3.11.2 once). The analysis uses numpy only; matplotlib is used for the figures, and the sources of the study did not pin it, so this repository does.

```sh
./reproduce.sh              # rebuilds in a temporary folder and compares with the shipped files
./reproduce.sh --in-place   # writes the rebuilt files over the shipped ones
git diff --stat results/ report/
```

`./reproduce.sh` rebuilds `report/data.json` (every number of the text), the LaTeX tables, the figures, `results/review-sensitivity.*`, `results/b0g-control.*`, the throughput summaries, and recomputes `results/wave2-*.json` and `results/wave1-analysis.json` with `code/evals/analyze_wave.py`, then runs the tests of the analysis scripts and checks the links and numbers of this README. The last command after `--in-place` must print nothing for `results/` and `report/`. The PDF figures in `figures/` carry a build time and show as changed; the PNG files can differ in bytes between matplotlib builds, and the script then compares pixels.

Two kinds of difference are expected and listed by the script. The recomputed `results/wave2-*.json` differ from the shipped ones only in the hashes of the per-item files, because the shipped per-item files hold no benchmark text (see below). `results/wave1-analysis.json` was made by an earlier version of `analyze_wave.py`, so its gate records differ; its numbers are recomputed and equal. The measurements themselves need GPUs; the scripts that take them are in `code/`.

## What is in this repository

| Path | Content |
|---|---|
| [`olala-quantization.pdf`](olala-quantization.pdf) | The report |
| [`PROTOCOL.md`](PROTOCOL.md) | The pre-registration: margin, tests, arms and the dated amendments. Internal process notes are removed and each removal is marked |
| [`results/`](results) | The evidence. Start with [`wave2-analysis.md`](results/wave2-analysis.md). Per-item records in `results/wave1/<run>/` (item ids, scores, flags, token counts, `run_info.json`, `server_provenance.txt`); fidelity, ablation, probe and weight-statistics summaries; throughput sweeps in [`bench/`](results/bench) |
| [`analysis/`](analysis) | Scripts that recompute the numbers: `review_sensitivity.py`, `b0g_control.py`, `recompute_wave2.py`, `bench_summary.py`, `gguf_bench_summary.py`; scripts that document how a shipped file was made from held-back inputs: `truncation_audit.py`, `retry_counts.py`, `make_item_records.py`, `gaussian_reference.py` |
| [`report/`](report) | The files generated for the report: `data.json`, `tab_*.tex`, and `build_data.py`, `make_tables.py`, `make_figures.py` |
| [`figures/`](figures) | The data figures of the report, PNG and PDF |
| [`code/evals/`](code/evals) | The evaluation harness, the scoring, the paired bootstrap (`analyze_wave.py`) and the plain-text fidelity tool, with their tests |
| [`code/quant/`](code/quant) | The data-free FP8 and NVFP4 quantizer, the vLLM loader patch for OVH's fork, the calibrated build, the weight statistics, and their tests |
| [`code/probe/`](code/probe) | The activation probe of Step 2 and its tests |
| [`code/runpod/`](code/runpod) | The scripts that ran on the pods (serving, accuracy runs, fidelity, GGUF, benchmarks) and their tests |
| [`reproduce.sh`](reproduce.sh), [`tools/`](tools) | The rebuild and compare script, and the checks of this README |

The per-item records are not the private ones. Item ids, scores, truncation flags, stop reasons and token counts are unchanged. `target` is kept for MGSM and AIME, where it is a short number, and dropped for HumanEval+, MBPP+ and IFEval. `completion`, the model's visible answer, is kept for MGSM and AIME only, cut to its first 1,000 characters and withheld when 30 consecutive characters also occur in the prompt; it is dropped for HumanEval+, MBPP+ and IFEval, where answers repeat prompt text or invent names and addresses. The counts per task are in [`results/item-records.json`](results/item-records.json), and [`analysis/make_item_records.py`](analysis/make_item_records.py) documents the rule. No number of the report reads these two fields. The absolute paths of the working machine in eight `run_info.json` files (the combined runs) were replaced by paths relative to the repository root; nothing else in them changed.

[`results/wave1/SEL-v2/quantization_info.json`](results/wave1/SEL-v2/quantization_info.json) is the record of the new build and is the `quantization_info.json` of the public checkpoint (sha256 `6a83e3cec274970291adb5889d7323d9708860dfe24fd1e567501f4ac92cf566`). Code comments and documents in `code/` sometimes name files of the private working repository (`recon/`, pre-flight notes); they are not part of this release.

## Where each result comes from

Section numbers are those of the report.

| Result | Section | Evidence |
|---|---|---|
| FP8 passes, 4-bit costs a few points; per-task tables | Step 1, Tables 1 to 3 | [`wave2-analysis.json`](results/wave2-analysis.json), [`wave2-n5-analysis.json`](results/wave2-n5-analysis.json), [`wave2-g1-analysis.json`](results/wave2-g1-analysis.json), [`wave2-q5-analysis.json`](results/wave2-q5-analysis.json), [`wave1-analysis.json`](results/wave1-analysis.json) |
| OVH's activation claims | Step 2, Table 4 | [`activation_probe.json`](results/wave1/analysis/probe/activation_probe.json), [`activation_probe.md`](results/wave1/analysis/probe/activation_probe.md), [`code/probe/`](code/probe) |
| The weights are ordinary where rounding looks | Step 3 | [`weight_stats.json`](results/wave1/analysis/weight_stats/weight_stats.json), [`gaussian-reference.json`](results/gaussian-reference.json) |
| The 4-bit change sits outside the routed experts | Step 4, Table 6 | [`ablation-abl-all/summary.json`](results/wave1/ablation-abl-all/summary.json) and the nine other `ablation-abl-*` folders |
| Keeping the sensitive part in bf16 | Step 5 | [`wave2-n5-analysis.md`](results/wave2-n5-analysis.md), [`wave2-sel-analysis.md`](results/wave2-sel-analysis.md) |
| Calibration made plain-text fidelity worse | Step 6 | [`g1/`](results/wave1/g1), [`manifest.json`](results/wave1/cal1-final/cal1/calib-cal1/manifest.json), [`wave2-g1-analysis.md`](results/wave2-g1-analysis.md) |
| The new build | Step 7, Tables 7 and 8 | [`fidelity-fid/summary.json`](results/wave1/fidelity-fid/summary.json), [`wave2-sel-analysis.md`](results/wave2-sel-analysis.md), [`wave2-fp8-contrast.md`](results/wave2-fp8-contrast.md), [`review-sensitivity.md`](results/review-sensitivity.md) |
| The Q4_K_M stand-in and the answers that ran out of context | Step 8, Tables 9, 11, 14 | [`wave2-q4-analysis.md`](results/wave2-q4-analysis.md), [`truncation-audit.json`](results/truncation-audit.json), [`q4/`](results/wave1/q4), [`bench-gguf-summary.md`](results/bench-gguf-summary.md) |
| The same-engine control | Step 8, Table 13 | [`b0g-control.md`](results/b0g-control.md), [`b0g-control.json`](results/b0g-control.json) |
| Speed under load | Step 9 | [`bench-summary.md`](results/bench-summary.md), [`bench/`](results/bench) |
| Margins at two, three and five points; sensitivity analyses | Table 12, Limits | [`review-sensitivity.json`](results/review-sensitivity.json), [`mgsm-clusters.json`](results/mgsm-clusters.json) |
| Every number and table of the text | all | [`report/data.json`](report/data.json), [`report/tab_summary.tex`](report/tab_summary.tex) |

## What you cannot reproduce from this repository

- **The raw inspect logs and the benchmark item text.** They are held back: they embed the prompts of MGSM, HumanEval+, MBPP+, IFEval and AIME 2025 under their own licenses, and they are gigabytes. The per-item records above stand in for them.
- **The GPU runs.** They need RunPod pods and OVH's forks of vLLM (image tag `ebbadef53`) and llama.cpp (commit `55c22dca0c98`) at the pinned commits, and the bf16 weights; the model weights and the GGUF files are not in this repository. `code/` holds the scripts that ran them.
- **The truncation audit and the retry counts.** `analysis/truncation_audit.py` and `analysis/retry_counts.py` read the held-back logs. Their outputs, `results/truncation-audit.json` and `results/retry-counts.json`, are shipped and every table reads them.
- **The MGSM source-problem clustering from the question text.** It needs the questions in the held-back log. `results/mgsm-clusters.json` holds the cluster id of every MGSM item (ids only), and `analysis/review_sensitivity.py` uses it when the log is absent; with the log it derives the same clusters and the same output.
- **The outside-model reviews.** They are not part of the release. References to them were removed from `PROTOCOL.md` and are marked there.
- **Edited logs.** A vendor warning line that names an unrelated client was removed from 14 server logs; no other log was edited.
- **The check of the text against the numbers.** `check_numbers.py` reads the LaTeX source, which is not part of this repository. `report/data.json` holds the numbers it checked.

## Limits

Every build has one run, so every label is provisional, and the two bf16 runs give the only measure of run-to-run scatter. The fidelity measure is plain text, uses the top twenty tokens, and its values had not converged. OVH's file and the Q4_K_M stand-in ran on another engine with another method, so format and engine are mixed, and the stand-in is my reconstruction of OVH's recipe. I measured no larger model, no recall over long contexts, and throughput only on synthetic requests on one card type. The report's Limits section lists the rest.

## Corrections

None yet. Found an error? [Open an issue](https://github.com/solfird/olala-quantization/issues). A correction is a new release of this repository, and its release notes say what changed.

## Cite

```bibtex
@techreport{simon2026olalaquantization,
  author      = {Simon, Julien},
  title       = {{Olala-v1.0-mini-7A1B under Quantization: What Holds, What Breaks, and Where}},
  institution = {Solfird AI},
  number      = {2026-04},
  year        = {2026},
  month       = nov,
  doi         = {10.5281/zenodo.24443547},
  url         = {https://github.com/solfird/olala-quantization},
}
```

GitHub's "Cite this repository" button gives the same reference from [`CITATION.cff`](CITATION.cff). The DOI above is the one of this version; the DOI of all versions of the report is [10.5281/zenodo.24443546](https://doi.org/10.5281/zenodo.24443546).

## Licenses

- The report, the figures and the data: CC BY 4.0, with the exceptions listed per path in [`LICENSE-DATA.md`](LICENSE-DATA.md). The per-item records and files derived from MGSM items are CC BY-SA 4.0. The 17 `.npz` fidelity files, the WikiText-2 token file in `code/evals/data/` and the importance matrix `olala.imatrix` derive from WikiText-2 and are CC BY-SA 3.0, not CC BY 4.0.
- The code: Apache 2.0, in [`LICENSE`](LICENSE). The files under `code/quant/gptq/vendor/` are OVHcloud's modeling code, unchanged, with their own `LICENSE` and `NOTICE`.
- OVH's model, Olala-v1.0-mini-7A1B, is Apache 2.0. The NVFP4 checkpoint derived from it carries the same license.
- The Solfird AI name and logo in `assets/` are not under these licenses.

Contact: julien@solfird.ai.
