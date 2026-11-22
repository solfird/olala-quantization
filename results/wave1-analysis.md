# Accuracy wave: pre-registered analysis

Reference `B0`; confirmatory family F1, F2, N2. Outcome labels are NON-INFERIOR, WORSE or INCONCLUSIVE, always read with the interval.

## 1. Summary

**Conclusion:** At the margin of -3 points on the macro-average (one-sided alpha 0.025, Holm over 3 candidate(s)), F1: NON-INFERIOR; F2: NON-INFERIOR; N2: INCONCLUSIVE.

| Candidate | Macro difference, pp [95% interval] | Outcome | B0 macro % | Candidate macro % | Flags |
|---|---|---|---|---|---|
| F1 | -0.4 [-2.1, +1.3] | NON-INFERIOR | 74.0 | 73.6 | truncation-rate difference |
| F2 | -0.3 [-2.1, +1.5] | NON-INFERIOR | 74.0 | 73.7 | none |
| N2 | -4.2 [-5.9, -2.5] | INCONCLUSIVE | 74.0 | 69.8 | truncation-rate difference |

Arms with a single run, so provisional under PROTOCOL.md amendment 9 (stage 1): F1, F2, N2, F1n.
Estimand: difference in the equally weighted macro-average of the five task scores, in percentage points, over the fixed items; request errors and truncations count as 0.
Calibration (B0a versus B0b): the interval contains 0, width 3.5 pp.

## 2. Gate and integrity

**Conclusion:** all 6 comparison pair(s) passed the gate; truncation rates differ by more than 1 point for F1, N2.

### Pairs

| Pair | Kind | Status | Problems | Warnings |
|---|---|---|---|---|
| F1 vs B0 | candidate | pass | 0 | 0 |
| F2 vs B0 | candidate | pass | 0 | 0 |
| N2 vs B0 | candidate | pass | 0 | 0 |
| F1n vs B0 | exploratory | pass | 0 | 0 |
| F2 vs F1 | contrast | pass | 0 | 0 |
| B0a vs B0b | calibration | pass | 0 | 0 |

### Settings per run

| Arm | Run | Status | Reasoning effort | Sampling | Sandbox | Epochs per task | Git commit | Code sha256 | Quantization recipe |
|---|---|---|---|---|---|---|---|---|---|
| B0 | `../results/wave1/merged/B0a` | complete | medium | T 0.8, top-p 0.9, top-k 40, max tokens None | local | mgsm 1, humaneval_plus 1, mbpp_plus 1, ifeval 1, aime2025 5 | None | 0587d7332d5cab245c16584ac1613cfe179d3b6f7affab662558c5382bf5e941 | none |
| B0 | `../results/wave1/merged/B0b` | complete | medium | T 0.8, top-p 0.9, top-k 40, max tokens None | local | mgsm 1, humaneval_plus 1, mbpp_plus 1, ifeval 1, aime2025 5 | None | 0587d7332d5cab245c16584ac1613cfe179d3b6f7affab662558c5382bf5e941 | none |
| F1 | `../results/wave1/merged/F1` | complete | medium | T 0.8, top-p 0.9, top-k 40, max tokens None | local | mgsm 1, humaneval_plus 1, mbpp_plus 1, ifeval 1, aime2025 5 | None | 0587d7332d5cab245c16584ac1613cfe179d3b6f7affab662558c5382bf5e941 | none |
| F2 | `../results/wave1/merged/F2` | complete | medium | T 0.8, top-p 0.9, top-k 40, max tokens None | local | mgsm 1, humaneval_plus 1, mbpp_plus 1, ifeval 1, aime2025 5 | None | 0587d7332d5cab245c16584ac1613cfe179d3b6f7affab662558c5382bf5e941 | fp8 |
| N2 | `../results/wave1/merged/N2` | complete | medium | T 0.8, top-p 0.9, top-k 40, max tokens None | local | mgsm 1, humaneval_plus 1, mbpp_plus 1, ifeval 1, aime2025 5 | None | 0587d7332d5cab245c16584ac1613cfe179d3b6f7affab662558c5382bf5e941 | nvfp4 |
| F1n | `../results/wave1/merged/F1n` | complete | medium | T 0.8, top-p 0.9, top-k 40, max tokens None | local | mgsm 1, humaneval_plus 1, mbpp_plus 1, ifeval 1, aime2025 5 | None | 0587d7332d5cab245c16584ac1613cfe179d3b6f7affab662558c5382bf5e941 | none |
| B0a | `../results/wave1/merged/B0a` | complete | medium | T 0.8, top-p 0.9, top-k 40, max tokens None | local | mgsm 1, humaneval_plus 1, mbpp_plus 1, ifeval 1, aime2025 5 | None | 0587d7332d5cab245c16584ac1613cfe179d3b6f7affab662558c5382bf5e941 | none |
| B0b | `../results/wave1/merged/B0b` | complete | medium | T 0.8, top-p 0.9, top-k 40, max tokens None | local | mgsm 1, humaneval_plus 1, mbpp_plus 1, ifeval 1, aime2025 5 | None | 0587d7332d5cab245c16584ac1613cfe179d3b6f7affab662558c5382bf5e941 | none |

### Request errors and truncations per run (counted as failures)

| Arm | Run | Samples | Request errors | Truncations | Errors % | Truncated % |
|---|---|---|---|---|---|---|
| B0 | `../results/wave1/merged/B0a` | 1783 | 0 | 11 | 0.00% | 0.62% |
| B0 | `../results/wave1/merged/B0b` | 1783 | 0 | 16 | 0.00% | 0.90% |
| F1 | `../results/wave1/merged/F1` | 1783 | 0 | 23 | 0.00% | 1.29% |
| F2 | `../results/wave1/merged/F2` | 1783 | 0 | 15 | 0.00% | 0.84% |
| N2 | `../results/wave1/merged/N2` | 1783 | 0 | 24 | 0.00% | 1.35% |
| F1n | `../results/wave1/merged/F1n` | 1783 | 0 | 14 | 0.00% | 0.79% |
| B0a | `../results/wave1/merged/B0a` | 1783 | 0 | 11 | 0.00% | 0.62% |
| B0b | `../results/wave1/merged/B0b` | 1783 | 0 | 16 | 0.00% | 0.90% |

## 3. Confirmatory tests

**Conclusion:** at the margin of -3 points, 2 of 3 candidate(s) are NON-INFERIOR after Holm adjustment (F1, F2); outcomes: F1 NON-INFERIOR; F2 NON-INFERIOR; N2 INCONCLUSIVE.

One-sided non-inferiority against -3 points; bootstrap p = fraction of 10000 resamples with a macro difference at or below the margin; Holm over 3 test(s) at one-sided alpha 0.025. The one-sided 97.5% lower bound equals the lower end of the two-sided 95% interval.

| Candidate | Macro difference, pp [two-sided 95% interval] | Interval width, pp | One-sided 97.5% lower bound | p (raw) | p (Holm) | Outcome |
|---|---|---|---|---|---|---|
| F1 | -0.4 [-2.1, +1.3] | 3.4 | -2.10 | 0.0009 | 0.0027 | NON-INFERIOR |
| F2 | -0.3 [-2.1, +1.5] | 3.7 | -2.13 | 0.0018 | 0.0036 | NON-INFERIOR |
| N2 | -4.2 [-5.9, -2.5] | 3.4 | -5.92 | 0.9126 | 0.9126 | INCONCLUSIVE |

### Sensitivity to the margin (outcome, Holm-adjusted p)

| Candidate | margin -3 | margin -2 | margin -5 |
|---|---|---|---|
| F1 | NON-INFERIOR (0.0027) | INCONCLUSIVE (0.0990) | NON-INFERIOR (0.0000) |
| F2 | NON-INFERIOR (0.0036) | INCONCLUSIVE (0.0990) | NON-INFERIOR (0.0000) |
| N2 | INCONCLUSIVE (0.9126) | WORSE (0.9946) | INCONCLUSIVE (0.1720) |

Width of pure run-to-run noise (B0a versus B0b, one run each): 3.5 pp, against the candidates' widths above. It is not used in any test.

## 4. Per-task differences

**Conclusion:** no candidate has a task whose interval upper bound is below -5 points.

### F1 versus B0

| Task | Items | B0 % | F1 % | Difference, pp [95% interval] | Flag |
|---|---|---|---|---|---|
| mgsm | 550 | 68.6 [65.0, 72.3] | 68.0 [64.0, 71.8] | -0.6 [-2.7, +1.4] |  |
| humaneval_plus | 164 | 85.1 [80.2, 89.6] | 86.6 [81.1, 91.5] | +1.5 [-3.4, +6.1] |  |
| mbpp_plus | 378 | 71.6 [67.5, 75.5] | 70.4 [65.6, 74.9] | -1.2 [-4.4, +1.9] |  |
| ifeval | 541 | 89.0 [86.7, 91.2] | 87.6 [84.8, 90.2] | -1.4 [-3.6, +0.7] |  |
| aime2025 | 30 | 55.7 [40.7, 70.3] | 55.3 [38.7, 71.3] | -0.3 [-6.0, +5.0] |  |
| macro-average |  | 74.0 [70.6, 77.2] | 73.6 [69.8, 77.2] | -0.4 [-2.1, +1.3] |  |

### F2 versus B0

| Task | Items | B0 % | F2 % | Difference, pp [95% interval] | Flag |
|---|---|---|---|---|---|
| mgsm | 550 | 68.6 [65.0, 72.3] | 67.3 [63.3, 71.1] | -1.4 [-3.4, +0.6] |  |
| humaneval_plus | 164 | 85.1 [80.2, 89.6] | 86.0 [80.5, 90.9] | +0.9 [-2.8, +4.6] |  |
| mbpp_plus | 378 | 71.6 [67.5, 75.5] | 71.2 [66.4, 75.7] | -0.4 [-3.6, +2.6] |  |
| ifeval | 541 | 89.0 [86.7, 91.2] | 88.7 [86.0, 91.3] | -0.3 [-2.7, +2.0] |  |
| aime2025 | 30 | 55.7 [40.7, 70.3] | 55.3 [39.3, 70.7] | -0.3 [-7.7, +7.0] |  |
| macro-average |  | 74.0 [70.6, 77.2] | 73.7 [70.0, 77.3] | -0.3 [-2.1, +1.5] |  |

### N2 versus B0

| Task | Items | B0 % | N2 % | Difference, pp [95% interval] | Flag |
|---|---|---|---|---|---|
| mgsm | 550 | 68.6 [65.0, 72.3] | 66.2 [62.2, 70.2] | -2.5 [-4.8, -0.1] |  |
| humaneval_plus | 164 | 85.1 [80.2, 89.6] | 82.9 [77.4, 88.4] | -2.1 [-7.3, +3.0] |  |
| mbpp_plus | 378 | 71.6 [67.5, 75.5] | 66.9 [62.2, 71.7] | -4.6 [-7.9, -1.2] |  |
| ifeval | 541 | 89.0 [86.7, 91.2] | 84.8 [81.9, 87.8] | -4.2 [-6.7, -1.7] |  |
| aime2025 | 30 | 55.7 [40.7, 70.3] | 48.0 [34.0, 62.0] | -7.7 [-12.3, -2.7] |  |
| macro-average |  | 74.0 [70.6, 77.2] | 69.8 [66.4, 73.1] | -4.2 [-5.9, -2.5] |  |


## 5. Flags

**Conclusion:** task-level regression: none; truncation-rate difference above 1 pp: F1, N2, F1n.

- F1: truncation-rate difference above 1 pp (scopes: aime2025); F1 1.29% (23/1783) against 0.76% (27/3566) for B0. A result to report, not a reason to discard; truncated answers count as failures.
- N2: truncation-rate difference above 1 pp (scopes: humaneval_plus); N2 1.35% (24/1783) against 0.76% (27/3566) for B0. A result to report, not a reason to discard; truncated answers count as failures.
- F1n: truncation-rate difference above 1 pp (scopes: aime2025); F1n 0.79% (14/1783) against 0.76% (27/3566) for B0. A result to report, not a reason to discard; truncated answers count as failures.

## 6. AIME accuracy among non-truncated samples (secondary)

**Conclusion:** AIME difference among non-truncated samples only (points): F1 +0.9 [-5.6, +6.9]; F2 -0.7 [-8.3, +6.8]; N2 -7.5 [-12.6, -2.3]. The headline counts truncated samples as failures.

| Candidate | Headline AIME difference (truncations = failures) | Difference among non-truncated samples | Truncated samples (arm / B0) | Arm accuracy among non-truncated % | B0 accuracy among non-truncated % |
|---|---|---|---|---|---|
| F1 | -0.3 [-6.0, +5.0] | +0.9 [-5.6, +6.9] | 7 / 8 | 58.0 [41.4, 73.8] | 57.2 [41.9, 71.6] |
| F2 | -0.3 [-7.7, +7.0] | -0.7 [-8.3, +6.8] | 3 / 8 | 56.5 [40.4, 71.9] | 57.2 [41.9, 71.6] |
| N2 | -7.7 [-12.3, -2.7] | -7.5 [-12.6, -2.3] | 5 / 8 | 49.7 [35.3, 63.7] | 57.2 [41.9, 71.6] |

## 7. MGSM per language

**Conclusion:** descriptive only; the largest drop in a language is F1: sw -7.0 [-14.0, +1.0]; F2: sw -7.0 [-13.0, -1.0]; N2: th -10.0 [-21.0, +0.0].

### F1 versus B0

| Language | Items | B0 % | F1 % | Difference, pp [95% interval] |
|---|---|---|---|---|
| bn | 50 | 39.0 [28.0, 51.0] | 38.0 [26.0, 52.0] | -1.0 [-9.0, +7.0] |
| de | 50 | 82.0 [70.0, 92.0] | 80.0 [68.0, 90.0] | -2.0 [-6.0, +0.0] |
| en | 50 | 95.0 [89.0, 99.0] | 98.0 [94.0, 100.0] | +3.0 [+0.0, +7.0] |
| es | 50 | 89.0 [80.0, 96.0] | 88.0 [78.0, 96.0] | -1.0 [-9.0, +6.0] |
| fr | 50 | 81.0 [70.0, 91.0] | 88.0 [78.0, 96.0] | +7.0 [-1.0, +16.0] |
| ja | 50 | 89.0 [81.0, 96.0] | 84.0 [74.0, 94.0] | -5.0 [-13.0, +2.0] |
| ru | 50 | 87.0 [78.0, 95.0] | 88.0 [78.0, 96.0] | +1.0 [-3.0, +5.0] |
| sw | 50 | 11.0 [5.0, 17.0] | 4.0 [0.0, 10.0] | -7.0 [-14.0, +1.0] |
| te | 50 | 4.0 [1.0, 8.0] | 8.0 [2.0, 16.0] | +4.0 [-2.0, +11.0] |
| th | 50 | 82.0 [73.0, 91.0] | 78.0 [66.0, 88.0] | -4.0 [-12.0, +4.0] |
| zh | 50 | 96.0 [91.0, 100.0] | 94.0 [86.0, 100.0] | -2.0 [-9.0, +3.0] |

### F2 versus B0

| Language | Items | B0 % | F2 % | Difference, pp [95% interval] |
|---|---|---|---|---|
| bn | 50 | 39.0 [28.0, 51.0] | 36.0 [24.0, 50.0] | -3.0 [-14.0, +8.0] |
| de | 50 | 82.0 [70.0, 92.0] | 82.0 [70.0, 92.0] | +0.0 [+0.0, +0.0] |
| en | 50 | 95.0 [89.0, 99.0] | 94.0 [86.0, 100.0] | -1.0 [-5.0, +2.0] |
| es | 50 | 89.0 [80.0, 96.0] | 92.0 [84.0, 98.0] | +3.0 [-4.0, +10.0] |
| fr | 50 | 81.0 [70.0, 91.0] | 78.0 [66.0, 88.0] | -3.0 [-12.0, +5.0] |
| ja | 50 | 89.0 [81.0, 96.0] | 88.0 [78.0, 96.0] | -1.0 [-7.0, +4.0] |
| ru | 50 | 87.0 [78.0, 95.0] | 88.0 [78.0, 96.0] | +1.0 [-3.0, +5.0] |
| sw | 50 | 11.0 [5.0, 17.0] | 4.0 [0.0, 10.0] | -7.0 [-13.0, -1.0] |
| te | 50 | 4.0 [1.0, 8.0] | 8.0 [2.0, 16.0] | +4.0 [-1.0, +10.0] |
| th | 50 | 82.0 [73.0, 91.0] | 76.0 [64.0, 88.0] | -6.0 [-16.0, +4.0] |
| zh | 50 | 96.0 [91.0, 100.0] | 94.0 [86.0, 100.0] | -2.0 [-7.0, +2.0] |

### N2 versus B0

| Language | Items | B0 % | N2 % | Difference, pp [95% interval] |
|---|---|---|---|---|
| bn | 50 | 39.0 [28.0, 51.0] | 32.0 [20.0, 46.0] | -7.0 [-17.0, +2.0] |
| de | 50 | 82.0 [70.0, 92.0] | 82.0 [70.0, 92.0] | +0.0 [-6.0, +6.0] |
| en | 50 | 95.0 [89.0, 99.0] | 92.0 [84.0, 98.0] | -3.0 [-9.0, +2.0] |
| es | 50 | 89.0 [80.0, 96.0] | 94.0 [86.0, 100.0] | +5.0 [+1.0, +11.0] |
| fr | 50 | 81.0 [70.0, 91.0] | 76.0 [64.0, 88.0] | -5.0 [-16.0, +5.0] |
| ja | 50 | 89.0 [81.0, 96.0] | 84.0 [74.0, 94.0] | -5.0 [-12.0, +1.0] |
| ru | 50 | 87.0 [78.0, 95.0] | 92.0 [84.0, 98.0] | +5.0 [+0.0, +11.0] |
| sw | 50 | 11.0 [5.0, 17.0] | 6.0 [0.0, 14.0] | -5.0 [-12.0, +3.0] |
| te | 50 | 4.0 [1.0, 8.0] | 8.0 [2.0, 16.0] | +4.0 [-2.0, +11.0] |
| th | 50 | 82.0 [73.0, 91.0] | 72.0 [60.0, 84.0] | -10.0 [-21.0, +0.0] |
| zh | 50 | 96.0 [91.0, 100.0] | 90.0 [82.0, 98.0] | -6.0 [-15.0, +1.0] |


## 8. Calibration of the interval

**Conclusion:** the interval of the macro-average difference between the two reference runs (B0a minus B0b) contains 0: +1.2 [-0.5, +2.9], width 3.5 pp.

The check expects the interval to contain 0; it does not alter any test.
Candidate interval widths (pp, each against the pooled reference): F1 3.4, F2 3.7, N2 3.4.

| Task | Difference, pp [95% interval] |
|---|---|
| mgsm | -0.5 [-3.3, +2.2] |
| humaneval_plus | +3.0 [-1.8, +8.5] |
| mbpp_plus | -1.9 [-5.8, +2.1] |
| ifeval | +0.6 [-2.0, +3.1] |
| aime2025 | +4.7 [+0.0, +9.3] |
| macro-average | +1.2 [-0.5, +2.9] |

## 9. Exploratory arms (descriptive only)

**Conclusion:** descriptive only (difference and interval, no outcome label, no Holm membership): F1n +0.5 [-1.0, +2.1].

| Arm | mgsm | humaneval_plus | mbpp_plus | ifeval | aime2025 | Macro-average |
|---|---|---|---|---|---|---|
| F1n | +0.1 [-2.2, +2.5] | +0.3 [-4.0, +4.3] | +0.1 [-2.9, +3.3] | +0.3 [-1.7, +2.3] | +1.7 [-3.0, +6.7] | +0.5 [-1.0, +2.1] |

- F1n: truncation-rate difference above 1 pp (scopes: aime2025); F1n 0.79% (14/1783) against 0.76% (27/3566) for B0. A result to report, not a reason to discard; truncated answers count as failures.

## 10. Contrasts between arms (descriptive)

**Conclusion:** descriptive differences with 95% intervals, no outcome label: F2 versus F1 +0.1 [-1.9, +2.2].

| Contrast (first minus second) | mgsm | humaneval_plus | mbpp_plus | ifeval | aime2025 | Macro-average |
|---|---|---|---|---|---|---|
| F2 versus F1 | -0.7 [-3.1, +1.6] | -0.6 [-5.5, +4.3] | +0.8 [-2.6, +4.2] | +1.1 [-1.7, +3.9] | +0.0 [-7.3, +7.3] | +0.1 [-1.9, +2.2] |

## 11. Provenance and limits

**Conclusion:** bootstrap seed 0, 10000 resamples; every number comes from the run files hashed below and from this script, whose own hash is given.

Bootstrap: seed 0, 10000 resamples, 95% two-sided percentile intervals; items are resampled independently within each task and the five task means are then averaged with equal weight; the reference is pooled over its runs.
Script: `analyze_wave.py`, sha256 `7b1fd8b4d5833ab096bbe0725bb80c686772879361fa25477302c2b195fa68d7`.
Expected full run: 1783 samples (mgsm 550 x 1, humaneval_plus 164 x 1, mbpp_plus 378 x 1, ifeval 541 x 1, aime2025 30 x 5).

Limits (PROTOCOL.md section 7): MGSM translations of one problem are separate items and are not clustered by source problem, so the MGSM interval is somewhat too narrow; AIME has 30 problems, so its interval and the macro interval are wide and the bootstrap coverage is approximate. A bootstrap p-value is a fraction of resamples: its resolution is 1/resamples.

| Arm | Run directory | run_info.json sha256 | samples.jsonl sha256 | items.json sha256 |
|---|---|---|---|---|
| B0 | `../results/wave1/merged/B0a` | 1d8f99a1950143638694dfda5e1af49056d341f178dd2d0498b6df741d23fa9f | fce688bdf29ab9a895679aad23daec892a0f0ffb8f7e86131f1753636bf8fb1a | 81488b253371973770df7d8429b97044db34c1bb32bb01c662baa387a8ac669a |
| B0 | `../results/wave1/merged/B0b` | b47a17af4ddaba2cfd00b41b26b154dc36698d4d84ab154b995799cdd024e49e | 6983823ba570fe319ca78b8e80a7dc1f97ea5641bedda5c92521c8f220b6f851 | 81488b253371973770df7d8429b97044db34c1bb32bb01c662baa387a8ac669a |
| F1 | `../results/wave1/merged/F1` | 79aec4bde8ade60289769f0ae804ea73fa43e4092b317d54af4f7558978a93fa | 7561f18cef845e3bc4cb406f7d87c061fb961fe97a3bbb2218817b54a9956dcf | 81488b253371973770df7d8429b97044db34c1bb32bb01c662baa387a8ac669a |
| F2 | `../results/wave1/merged/F2` | 9835f7fbfcbb481d1ad8c7ae0eee7d4c9457aee1c82ea836935293ec1c6511f1 | 90d62cb22750f03e86481a694319e94b947ea9160997678aa666479d0d8a19f4 | 81488b253371973770df7d8429b97044db34c1bb32bb01c662baa387a8ac669a |
| N2 | `../results/wave1/merged/N2` | 5c77dce29c845e4a03ad007fffc9dab57030f7069d2dfcd072b443a3e065cff1 | c6cadc26afcff8f258bf9ea055a6511d9dff5f4974ccbc1998b9e2b78ef27424 | 81488b253371973770df7d8429b97044db34c1bb32bb01c662baa387a8ac669a |
| F1n | `../results/wave1/merged/F1n` | 1a4de1542fddfddb9d67e8e5a53f1ee65e56139029c1d41d3de74abdaf668968 | 29e8bce911c18a7c7b2e9d05c1265cb01c081bff9afb25ae4ca0036b90fdbba9 | 81488b253371973770df7d8429b97044db34c1bb32bb01c662baa387a8ac669a |
| B0a | `../results/wave1/merged/B0a` | 1d8f99a1950143638694dfda5e1af49056d341f178dd2d0498b6df741d23fa9f | fce688bdf29ab9a895679aad23daec892a0f0ffb8f7e86131f1753636bf8fb1a | 81488b253371973770df7d8429b97044db34c1bb32bb01c662baa387a8ac669a |
| B0b | `../results/wave1/merged/B0b` | b47a17af4ddaba2cfd00b41b26b154dc36698d4d84ab154b995799cdd024e49e | 6983823ba570fe319ca78b8e80a7dc1f97ea5641bedda5c92521c8f220b6f851 | 81488b253371973770df7d8429b97044db34c1bb32bb01c662baa387a8ac669a |
