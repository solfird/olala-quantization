# Accuracy wave: pre-registered analysis

Reference `B0`; confirmatory family F1, F2, N2. Outcome labels are NON-INFERIOR, WORSE or INCONCLUSIVE, always read with the interval.

## 1. Summary

**Conclusion:** At the margin of -3 points on the macro-average (one-sided alpha 0.025, Holm over 3 candidate(s)), F1: NON-INFERIOR; F2: NON-INFERIOR; N2: INCONCLUSIVE.

| Candidate | Macro difference, pp [95% interval] | Outcome | B0 macro % | Candidate macro % | Flags |
|---|---|---|---|---|---|
| F1 | -0.1 [-1.5, +1.3] | NON-INFERIOR | 73.5 | 73.3 | truncation-rate difference |
| F2 | -0.9 [-2.7, +0.8] | NON-INFERIOR | 73.5 | 72.5 | truncation-rate difference |
| N2 | -3.3 [-5.3, -1.4] | INCONCLUSIVE | 73.5 | 70.1 | truncation-rate difference |

Arms with a single run, so provisional under PROTOCOL.md amendment 9 (stage 1): F1, F2, N2, Q5.
Estimand: difference in the equally weighted macro-average of the five task scores, in percentage points, over the fixed items; request errors and truncations count as 0.
Calibration (B0a versus B0b): the interval contains 0, width 4.4 pp.

## 2. Gate and integrity

**Conclusion:** all 7 comparison pair(s) passed the gate; truncation rates differ by more than 1 point for F1, F2, N2.

### Pairs

| Pair | Kind | Status | Problems | Warnings |
|---|---|---|---|---|
| F1 vs B0 | candidate | pass | 0 | 0 |
| F2 vs B0 | candidate | pass | 0 | 0 |
| N2 vs B0 | candidate | pass | 0 | 0 |
| Q5 vs B0 | exploratory | pass | 0 | 0 |
| N2 vs Q5 | exploratory-contrast | pass | 0 | 0 |
| F2 vs Q5 | exploratory-contrast | pass | 0 | 0 |
| B0a vs B0b | calibration | pass | 0 | 0 |

### Settings per run

| Arm | Run | Status | Reasoning effort | Sampling | Sandbox | Epochs per task | Git commit | Code sha256 | Quantization recipe |
|---|---|---|---|---|---|---|---|---|---|
| B0 | `../results/wave1/B0a-v2` | complete | medium | T 0.8, top-p 0.9, top-k 40, max tokens None | local | mgsm 1, humaneval_plus 1, mbpp_plus 1, ifeval 1, aime2025 5 | None | a13aa4cc4491d3af8b73f61a7d2fc7bf568e6ad5fb840e4075e58f9fb5c5f7a8 | none |
| B0 | `../results/wave1/B0b-v2` | complete | medium | T 0.8, top-p 0.9, top-k 40, max tokens None | local | mgsm 1, humaneval_plus 1, mbpp_plus 1, ifeval 1, aime2025 5 | None | a13aa4cc4491d3af8b73f61a7d2fc7bf568e6ad5fb840e4075e58f9fb5c5f7a8 | none |
| F1 | `../results/wave1/F1-v2` | complete | medium | T 0.8, top-p 0.9, top-k 40, max tokens None | local | mgsm 1, humaneval_plus 1, mbpp_plus 1, ifeval 1, aime2025 5 | None | a13aa4cc4491d3af8b73f61a7d2fc7bf568e6ad5fb840e4075e58f9fb5c5f7a8 | none |
| F2 | `../results/wave1/F2-v2` | complete | medium | T 0.8, top-p 0.9, top-k 40, max tokens None | local | mgsm 1, humaneval_plus 1, mbpp_plus 1, ifeval 1, aime2025 5 | None | a13aa4cc4491d3af8b73f61a7d2fc7bf568e6ad5fb840e4075e58f9fb5c5f7a8 | fp8 |
| N2 | `../results/wave1/N2-v2` | complete | medium | T 0.8, top-p 0.9, top-k 40, max tokens None | local | mgsm 1, humaneval_plus 1, mbpp_plus 1, ifeval 1, aime2025 5 | None | a13aa4cc4491d3af8b73f61a7d2fc7bf568e6ad5fb840e4075e58f9fb5c5f7a8 | nvfp4 |
| Q5 | `../results/wave1/Q5v2` | complete | medium | T 0.8, top-p 0.9, top-k 40, max tokens None | local | mgsm 1, humaneval_plus 1, mbpp_plus 1, ifeval 1, aime2025 5 | None | a13aa4cc4491d3af8b73f61a7d2fc7bf568e6ad5fb840e4075e58f9fb5c5f7a8 | none |
| B0a | `../results/wave1/B0a-v2` | complete | medium | T 0.8, top-p 0.9, top-k 40, max tokens None | local | mgsm 1, humaneval_plus 1, mbpp_plus 1, ifeval 1, aime2025 5 | None | a13aa4cc4491d3af8b73f61a7d2fc7bf568e6ad5fb840e4075e58f9fb5c5f7a8 | none |
| B0b | `../results/wave1/B0b-v2` | complete | medium | T 0.8, top-p 0.9, top-k 40, max tokens None | local | mgsm 1, humaneval_plus 1, mbpp_plus 1, ifeval 1, aime2025 5 | None | a13aa4cc4491d3af8b73f61a7d2fc7bf568e6ad5fb840e4075e58f9fb5c5f7a8 | none |

### Request errors and truncations per run (counted as failures)

| Arm | Run | Samples | Request errors | Truncations | Errors % | Truncated % |
|---|---|---|---|---|---|---|
| B0 | `../results/wave1/B0a-v2` | 1783 | 0 | 36 | 0.00% | 2.02% |
| B0 | `../results/wave1/B0b-v2` | 1783 | 0 | 37 | 0.00% | 2.08% |
| F1 | `../results/wave1/F1-v2` | 1783 | 0 | 21 | 0.00% | 1.18% |
| F2 | `../results/wave1/F2-v2` | 1783 | 0 | 37 | 0.00% | 2.08% |
| N2 | `../results/wave1/N2-v2` | 1783 | 0 | 47 | 0.00% | 2.64% |
| Q5 | `../results/wave1/Q5v2` | 1783 | 0 | 35 | 0.00% | 1.96% |
| B0a | `../results/wave1/B0a-v2` | 1783 | 0 | 36 | 0.00% | 2.02% |
| B0b | `../results/wave1/B0b-v2` | 1783 | 0 | 37 | 0.00% | 2.08% |

## 3. Confirmatory tests

**Conclusion:** at the margin of -3 points, 2 of 3 candidate(s) are NON-INFERIOR after Holm adjustment (F1, F2); outcomes: F1 NON-INFERIOR; F2 NON-INFERIOR; N2 INCONCLUSIVE.

One-sided non-inferiority against -3 points; bootstrap p = fraction of 10000 resamples with a macro difference at or below the margin; Holm over 3 test(s) at one-sided alpha 0.025. The one-sided 97.5% lower bound equals the lower end of the two-sided 95% interval.

| Candidate | Macro difference, pp [two-sided 95% interval] | Interval width, pp | One-sided 97.5% lower bound | p (raw) | p (Holm) | Outcome |
|---|---|---|---|---|---|---|
| F1 | -0.1 [-1.5, +1.3] | 2.8 | -1.54 | 0.0002 | 0.0006 | NON-INFERIOR |
| F2 | -0.9 [-2.7, +0.8] | 3.4 | -2.65 | 0.0089 | 0.0178 | NON-INFERIOR |
| N2 | -3.3 [-5.3, -1.4] | 3.8 | -5.25 | 0.6133 | 0.6133 | INCONCLUSIVE |

### Sensitivity to the margin (outcome, Holm-adjusted p)

| Candidate | margin -3 | margin -2 | margin -5 |
|---|---|---|---|
| F1 | NON-INFERIOR (0.0006) | NON-INFERIOR (0.0120) | NON-INFERIOR (0.0000) |
| F2 | NON-INFERIOR (0.0178) | INCONCLUSIVE (0.2310) | NON-INFERIOR (0.0000) |
| N2 | INCONCLUSIVE (0.6133) | INCONCLUSIVE (0.8975) | INCONCLUSIVE (0.0439) |

Width of pure run-to-run noise (B0a versus B0b, one run each): 4.4 pp, against the candidates' widths above. It is not used in any test.

## 4. Per-task differences

**Conclusion:** no candidate has a task whose interval upper bound is below -5 points.

### F1 versus B0

| Task | Items | B0 % | F1 % | Difference, pp [95% interval] | Flag |
|---|---|---|---|---|---|
| mgsm | 550 | 67.2 [63.5, 70.8] | 67.3 [63.3, 71.1] | +0.1 [-2.3, +2.5] |  |
| humaneval_plus | 164 | 87.8 [83.2, 91.8] | 90.2 [85.4, 94.5] | +2.4 [-0.9, +5.8] |  |
| mbpp_plus | 378 | 71.0 [66.8, 75.3] | 70.9 [66.4, 75.4] | -0.1 [-3.0, +2.8] |  |
| ifeval | 541 | 88.3 [85.9, 90.6] | 87.6 [84.8, 90.4] | -0.6 [-2.9, +1.6] |  |
| aime2025 | 30 | 53.0 [38.3, 67.3] | 50.7 [34.7, 66.7] | -2.3 [-7.0, +2.0] |  |
| macro-average |  | 73.5 [70.1, 76.7] | 73.3 [69.7, 76.8] | -0.1 [-1.5, +1.3] |  |

### F2 versus B0

| Task | Items | B0 % | F2 % | Difference, pp [95% interval] | Flag |
|---|---|---|---|---|---|
| mgsm | 550 | 67.2 [63.5, 70.8] | 67.5 [63.5, 71.5] | +0.3 [-2.1, +2.7] |  |
| humaneval_plus | 164 | 87.8 [83.2, 91.8] | 86.6 [81.1, 91.5] | -1.2 [-5.5, +3.0] |  |
| mbpp_plus | 378 | 71.0 [66.8, 75.3] | 70.1 [65.3, 74.6] | -0.9 [-4.1, +2.2] |  |
| ifeval | 541 | 88.3 [85.9, 90.6] | 85.8 [82.8, 88.7] | -2.5 [-4.8, -0.2] |  |
| aime2025 | 30 | 53.0 [38.3, 67.3] | 52.7 [37.3, 68.0] | -0.3 [-6.3, +5.3] |  |
| macro-average |  | 73.5 [70.1, 76.7] | 72.5 [69.0, 76.1] | -0.9 [-2.7, +0.8] |  |

### N2 versus B0

| Task | Items | B0 % | N2 % | Difference, pp [95% interval] | Flag |
|---|---|---|---|---|---|
| mgsm | 550 | 67.2 [63.5, 70.8] | 65.1 [61.3, 69.1] | -2.1 [-4.6, +0.5] |  |
| humaneval_plus | 164 | 87.8 [83.2, 91.8] | 82.9 [77.4, 88.4] | -4.9 [-9.5, -0.6] |  |
| mbpp_plus | 378 | 71.0 [66.8, 75.3] | 68.5 [64.0, 73.0] | -2.5 [-6.0, +0.9] |  |
| ifeval | 541 | 88.3 [85.9, 90.6] | 86.9 [83.9, 89.6] | -1.4 [-3.9, +1.2] |  |
| aime2025 | 30 | 53.0 [38.3, 67.3] | 47.3 [32.7, 62.0] | -5.7 [-13.0, +1.0] |  |
| macro-average |  | 73.5 [70.1, 76.7] | 70.1 [66.7, 73.5] | -3.3 [-5.3, -1.4] |  |


## 5. Flags

**Conclusion:** task-level regression: none; truncation-rate difference above 1 pp: F1, F2, N2, Q5.

- F1: truncation-rate difference above 1 pp (scopes: aime2025); F1 1.18% (21/1783) against 2.05% (73/3566) for B0. A result to report, not a reason to discard; truncated answers count as failures.
- F2: truncation-rate difference above 1 pp (scopes: humaneval_plus, aime2025); F2 2.08% (37/1783) against 2.05% (73/3566) for B0. A result to report, not a reason to discard; truncated answers count as failures.
- N2: truncation-rate difference above 1 pp (scopes: humaneval_plus, aime2025); N2 2.64% (47/1783) against 2.05% (73/3566) for B0. A result to report, not a reason to discard; truncated answers count as failures.
- Q5: truncation-rate difference above 1 pp (scopes: aime2025); Q5 1.96% (35/1783) against 2.05% (73/3566) for B0. A result to report, not a reason to discard; truncated answers count as failures.

## 6. AIME accuracy among non-truncated samples (secondary)

**Conclusion:** AIME difference among non-truncated samples only (points): F1 -4.5 [-10.8, +1.7]; F2 -2.0 [-9.1, +4.6]; N2 -3.4 [-11.1, +3.4]. The headline counts truncated samples as failures.

| Candidate | Headline AIME difference (truncations = failures) | Difference among non-truncated samples | Truncated samples (arm / B0) | Arm accuracy among non-truncated % | B0 accuracy among non-truncated % |
|---|---|---|---|---|---|
| F1 | -2.3 [-7.0, +2.0] | -4.5 [-10.8, +1.7] | 8 / 26 | 53.5 [37.4, 69.4] | 58.0 [43.1, 71.9] |
| F2 | -0.3 [-6.3, +5.3] | -2.0 [-9.1, +4.6] | 9 / 26 | 56.0 [40.3, 71.6] | 58.0 [43.1, 71.9] |
| N2 | -5.7 [-13.0, +1.0] | -3.4 [-11.1, +3.4] | 20 / 26 | 54.6 [38.8, 69.9] | 58.0 [43.1, 71.9] |

## 7. MGSM per language

**Conclusion:** descriptive only; the largest drop in a language is F1: ja -8.0 [-16.0, -2.0]; F2: zh -5.0 [-11.0, -1.0]; N2: bn -7.0 [-20.0, +6.0].

### F1 versus B0

| Language | Items | B0 % | F1 % | Difference, pp [95% interval] |
|---|---|---|---|---|
| bn | 50 | 37.0 [26.0, 49.0] | 40.0 [26.0, 54.0] | +3.0 [-9.0, +15.0] |
| de | 50 | 81.0 [70.0, 90.0] | 80.0 [68.0, 90.0] | -1.0 [-7.0, +4.0] |
| en | 50 | 94.0 [87.0, 99.0] | 94.0 [86.0, 100.0] | +0.0 [-6.0, +6.0] |
| es | 50 | 90.0 [82.0, 97.0] | 90.0 [80.0, 98.0] | +0.0 [-7.0, +6.0] |
| fr | 50 | 73.0 [62.0, 84.0] | 82.0 [70.0, 92.0] | +9.0 [+2.0, +17.0] |
| ja | 50 | 88.0 [79.0, 96.0] | 80.0 [68.0, 90.0] | -8.0 [-16.0, -2.0] |
| ru | 50 | 88.0 [79.0, 95.0] | 90.0 [82.0, 98.0] | +2.0 [-3.0, +8.0] |
| sw | 50 | 8.0 [3.0, 13.0] | 6.0 [0.0, 14.0] | -2.0 [-9.0, +6.0] |
| te | 50 | 8.0 [2.0, 15.0] | 10.0 [2.0, 20.0] | +2.0 [-6.0, +10.0] |
| th | 50 | 77.0 [67.0, 86.0] | 76.0 [64.0, 88.0] | -1.0 [-12.0, +10.0] |
| zh | 50 | 95.0 [89.0, 99.0] | 92.0 [84.0, 98.0] | -3.0 [-10.0, +3.0] |

### F2 versus B0

| Language | Items | B0 % | F2 % | Difference, pp [95% interval] |
|---|---|---|---|---|
| bn | 50 | 37.0 [26.0, 49.0] | 36.0 [24.0, 50.0] | -1.0 [-12.0, +10.0] |
| de | 50 | 81.0 [70.0, 90.0] | 82.0 [70.0, 92.0] | +1.0 [-2.0, +4.0] |
| en | 50 | 94.0 [87.0, 99.0] | 94.0 [86.0, 100.0] | +0.0 [-5.0, +4.0] |
| es | 50 | 90.0 [82.0, 97.0] | 88.0 [78.0, 96.0] | -2.0 [-11.0, +6.0] |
| fr | 50 | 73.0 [62.0, 84.0] | 78.0 [66.0, 90.0] | +5.0 [-3.0, +13.0] |
| ja | 50 | 88.0 [79.0, 96.0] | 84.0 [74.0, 94.0] | -4.0 [-11.0, +1.0] |
| ru | 50 | 88.0 [79.0, 95.0] | 90.0 [82.0, 98.0] | +2.0 [-2.0, +6.0] |
| sw | 50 | 8.0 [3.0, 13.0] | 14.0 [6.0, 24.0] | +6.0 [-4.0, +17.0] |
| te | 50 | 8.0 [2.0, 15.0] | 6.0 [0.0, 14.0] | -2.0 [-12.0, +8.0] |
| th | 50 | 77.0 [67.0, 86.0] | 80.0 [68.0, 90.0] | +3.0 [-7.0, +12.0] |
| zh | 50 | 95.0 [89.0, 99.0] | 90.0 [82.0, 98.0] | -5.0 [-11.0, -1.0] |

### N2 versus B0

| Language | Items | B0 % | N2 % | Difference, pp [95% interval] |
|---|---|---|---|---|
| bn | 50 | 37.0 [26.0, 49.0] | 30.0 [18.0, 44.0] | -7.0 [-20.0, +6.0] |
| de | 50 | 81.0 [70.0, 90.0] | 80.0 [68.0, 90.0] | -1.0 [-7.0, +4.0] |
| en | 50 | 94.0 [87.0, 99.0] | 94.0 [86.0, 100.0] | +0.0 [-5.0, +4.0] |
| es | 50 | 90.0 [82.0, 97.0] | 86.0 [76.0, 94.0] | -4.0 [-13.0, +4.0] |
| fr | 50 | 73.0 [62.0, 84.0] | 72.0 [60.0, 84.0] | -1.0 [-13.0, +10.0] |
| ja | 50 | 88.0 [79.0, 96.0] | 86.0 [76.0, 94.0] | -2.0 [-10.0, +5.0] |
| ru | 50 | 88.0 [79.0, 95.0] | 88.0 [78.0, 96.0] | +0.0 [-6.0, +5.0] |
| sw | 50 | 8.0 [3.0, 13.0] | 6.0 [0.0, 14.0] | -2.0 [-10.0, +7.0] |
| te | 50 | 8.0 [2.0, 15.0] | 2.0 [0.0, 6.0] | -6.0 [-14.0, +1.0] |
| th | 50 | 77.0 [67.0, 86.0] | 78.0 [66.0, 88.0] | +1.0 [-8.0, +10.0] |
| zh | 50 | 95.0 [89.0, 99.0] | 94.0 [86.0, 100.0] | -1.0 [-8.0, +5.0] |


## 8. Calibration of the interval

**Conclusion:** the interval of the macro-average difference between the two reference runs (B0a minus B0b) contains 0: -0.6 [-2.8, +1.6], width 4.4 pp.

The check expects the interval to contain 0; it does not alter any test.
Candidate interval widths (pp, each against the pooled reference): F1 2.8, F2 3.4, N2 3.8.

| Task | Difference, pp [95% interval] |
|---|---|
| mgsm | +2.4 [-0.5, +5.3] |
| humaneval_plus | -3.7 [-8.5, +1.2] |
| mbpp_plus | -2.9 [-6.9, +0.8] |
| ifeval | +0.6 [-2.0, +3.1] |
| aime2025 | +0.7 [-7.3, +8.7] |
| macro-average | -0.6 [-2.8, +1.6] |

## 9. Exploratory arms (descriptive only)

**Conclusion:** descriptive only (difference and interval, no outcome label, no Holm membership): Q5 -1.3 [-3.0, +0.5].

| Arm | mgsm | humaneval_plus | mbpp_plus | ifeval | aime2025 | Macro-average |
|---|---|---|---|---|---|---|
| Q5 | +0.5 [-1.7, +2.6] | -3.7 [-8.2, +0.9] | -0.9 [-4.1, +2.1] | -2.5 [-5.0, +0.0] | +0.3 [-5.0, +6.0] | -1.3 [-3.0, +0.5] |

- Q5: truncation-rate difference above 1 pp (scopes: aime2025); Q5 1.96% (35/1783) against 2.05% (73/3566) for B0. A result to report, not a reason to discard; truncated answers count as failures.

## 10. Contrasts between arms (descriptive)

**Conclusion:** descriptive differences with 95% intervals, no outcome label: N2 versus Q5 -2.0 [-4.3, +0.1]; F2 versus Q5 +0.3 [-1.7, +2.3].

| Contrast (first minus second) | mgsm | humaneval_plus | mbpp_plus | ifeval | aime2025 | Macro-average |
|---|---|---|---|---|---|---|
| N2 versus Q5 | -2.5 [-5.3, +0.2] | -1.2 [-6.1, +3.7] | -1.6 [-5.6, +2.4] | +1.1 [-1.8, +3.9] | -6.0 [-14.7, +2.0] | -2.0 [-4.3, +0.1] |
| F2 versus Q5 | -0.2 [-2.9, +2.5] | +2.4 [-2.4, +7.3] | +0.0 [-3.7, +3.7] | +0.0 [-2.8, +2.8] | -0.7 [-7.3, +5.3] | +0.3 [-1.7, +2.3] |

## 11. Provenance and limits

**Conclusion:** bootstrap seed 0, 10000 resamples; every number comes from the run files hashed below and from this script, whose own hash is given.

Bootstrap: seed 0, 10000 resamples, 95% two-sided percentile intervals; items are resampled independently within each task and the five task means are then averaged with equal weight; the reference is pooled over its runs.
Script: `analyze_wave.py`, sha256 `b04e5c3507a75b486ccbd4fb654003abff46ae105a906203863d13d956967bd9`.
Expected full run: 1783 samples (mgsm 550 x 1, humaneval_plus 164 x 1, mbpp_plus 378 x 1, ifeval 541 x 1, aime2025 30 x 5).

Limits (PROTOCOL.md section 7): MGSM translations of one problem are separate items and are not clustered by source problem, so the MGSM interval is somewhat too narrow; AIME has 30 problems, so its interval and the macro interval are wide and the bootstrap coverage is approximate. A bootstrap p-value is a fraction of resamples: its resolution is 1/resamples.

| Arm | Run directory | run_info.json sha256 | samples.jsonl sha256 | items.json sha256 |
|---|---|---|---|---|
| B0 | `../results/wave1/B0a-v2` | d6672b2e865daafb24ede002889931e3fa6b7351a60ef383224246431dfc4ec7 | 930b136fd4488f9444884ca4a53b552efbecaa354ed3d07baf868552bb4a18ba | 81488b253371973770df7d8429b97044db34c1bb32bb01c662baa387a8ac669a |
| B0 | `../results/wave1/B0b-v2` | acacd3dd83eea7cc3139777ac514514e8155dcfefe64a4510b47870e62073db6 | c622954b996855a37081de96f104398c4fc20bd5b249a28c2091d8dee41924cc | 81488b253371973770df7d8429b97044db34c1bb32bb01c662baa387a8ac669a |
| F1 | `../results/wave1/F1-v2` | 39f4537e5bdb1a834ceef3f68f08b2bb08edad976f1ffc5d7532e2cf088dd5e3 | a6fedc0104e9c18c48f40a06f6dd7504cf6d2f492a65bec91897b09044275194 | 81488b253371973770df7d8429b97044db34c1bb32bb01c662baa387a8ac669a |
| F2 | `../results/wave1/F2-v2` | 45d361243d6e78448a1d1e3d60ca03effeb27c2e66ff5b7f3af01aa34938ba59 | 90de6d9669ccf96acad91ae2d5d81a9e6c001db300c072d8ae8ca4040c8e682b | 81488b253371973770df7d8429b97044db34c1bb32bb01c662baa387a8ac669a |
| N2 | `../results/wave1/N2-v2` | b7887057dd932bf53e772eeffdc06f2ead542be694091e820b89f2e7c25b90a4 | d86b9ca4c5dae6a56e00d6853d50462cb2fcaea093a96e622aad72def2ef5e61 | 81488b253371973770df7d8429b97044db34c1bb32bb01c662baa387a8ac669a |
| Q5 | `../results/wave1/Q5v2` | a8e85677bfccdd487ad0ddb848582cd2aa60922e495f35efdfd69d4975af22c0 | 509d01d8e432120a2e9dd22650b5268d6933d9b315dc493abab41533df2d4988 | 81488b253371973770df7d8429b97044db34c1bb32bb01c662baa387a8ac669a |
| B0a | `../results/wave1/B0a-v2` | d6672b2e865daafb24ede002889931e3fa6b7351a60ef383224246431dfc4ec7 | 930b136fd4488f9444884ca4a53b552efbecaa354ed3d07baf868552bb4a18ba | 81488b253371973770df7d8429b97044db34c1bb32bb01c662baa387a8ac669a |
| B0b | `../results/wave1/B0b-v2` | acacd3dd83eea7cc3139777ac514514e8155dcfefe64a4510b47870e62073db6 | c622954b996855a37081de96f104398c4fc20bd5b249a28c2091d8dee41924cc | 81488b253371973770df7d8429b97044db34c1bb32bb01c662baa387a8ac669a |
