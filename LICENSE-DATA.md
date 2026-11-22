# License for the report, the data and the figures

© 2026 Julien Simon · CC BY 4.0 · [creativecommons.org/licenses/by/4.0](https://creativecommons.org/licenses/by/4.0/)

The report, the text, the figures and the data are licensed under the Creative Commons Attribution 4.0 International license, with the exceptions in
the table below. You may share and adapt them for any purpose, commercial included, if you credit the author and link to the license.

Suggested credit: "Julien Simon, Solfird AI Technical Report 2026-04, CC BY 4.0."

## License by path

| Path pattern | License |
|---|---|
| `olala-quantization.pdf`, `README.md`, `PROTOCOL.md`, `figures/`, `report/data.json`, `report/tab_*.tex`, and every file of `results/` not named below | CC BY 4.0 |
| `results/wave1/*/samples.jsonl` and `results/wave1/*/items.json` (the per-item records), `results/mgsm-clusters.json`, `results/truncation-audit.json` | CC BY-SA 4.0, as a precaution: they are keyed by the items of MGSM, which is CC BY-SA 4.0 |
| The 17 `.npz` fidelity files in `results/wave1/` and `code/evals/data/wikitext2_test_64x512.json` (token ids and log-probabilities of WikiText-2 test chunks) | CC BY-SA 3.0, the license of WikiText-2 (it is also GFDL). **Not** under the CC BY 4.0 license of the repository |
| `results/wave1/q4/olala.imatrix` (importance matrix computed on WikiText-2 train) | CC BY-SA 3.0, as above. **Not** under CC BY 4.0 |
| `analysis/`, `code/`, `report/*.py`, `tools/`, `reproduce.sh` | Apache License 2.0 in `LICENSE` |
| `code/quant/gptq/vendor/` | OVHcloud's modeling code, Apache 2.0, unchanged, with its own `LICENSE` and `NOTICE` |
| `assets/` | The Solfird AI name and logo: not under these licenses |

Aggregate numbers computed from WikiText-2 (means, intervals, ratios in the summary files) are results, not copies of its text, and stay under CC BY 4.0.

## What the benchmark files hold

This repository holds only per-item scores, flags and token counts, the short reference answers (numbers) of MGSM and AIME 2025, and the first 1,000
characters of the model's answers to MGSM and AIME 2025 items. It holds no item: no question, prompt, test or reference solution. The raw logs, which
embed the prompts, are held back. Each benchmark keeps its own license, checked on its Hugging Face dataset card:

| Benchmark (dataset) | License of the items |
|---|---|
| MGSM (`juletxara/mgsm`) | CC BY-SA 4.0 |
| HumanEval+ (`evalplus/humanevalplus`) | Apache 2.0 |
| MBPP+ (`evalplus/mbppplus`) | Apache 2.0 |
| IFEval (`google/IFEval`) | Apache 2.0 |
| AIME 2025 (`math-ai/aime25`) | Apache 2.0, as stated on the dataset card; the problem text is not shipped |
| WikiText-2 (`Salesforce/wikitext`) | CC BY-SA 3.0 and GFDL |

## Other material

- The calibration sets of the calibrated build were drawn from MATH (MIT), GSM8K translations (Apache 2.0), OSS-Instruct (MIT) and a persona
  instruction-following set (ODC-BY). The repository holds their manifests and counts, not their text.
- OVH's model, Olala-v1.0-mini-7A1B, and its llama.cpp and vLLM forks belong to OVHcloud and are Apache 2.0. The NVFP4 checkpoint derived from the
  model carries the same license. This repository holds no model file and no GGUF file; the `quantization_info.json` records and the loader patch
  (`code/quant/fork-patch/`, Apache 2.0) are Solfird AI's.
- Quotations of OVH in the report are short quotations of public statements, with their sources named.
