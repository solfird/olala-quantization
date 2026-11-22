# Accuracy wave: pre-registered analysis

Reference `B0`; confirmatory family SEL. Outcome labels are NON-INFERIOR, WORSE or INCONCLUSIVE, always read with the interval.

## 1. Summary

**Conclusion:** At the margin of -3 points on the macro-average (one-sided alpha 0.025, Holm over 1 candidate(s)), SEL: INCONCLUSIVE.

| Candidate | Macro difference, pp [95% interval] | Outcome | B0 macro % | Candidate macro % | Flags |
|---|---|---|---|---|---|
| SEL | -1.6 [-3.3, -0.0] | INCONCLUSIVE | 73.5 | 71.8 | truncation-rate difference |

Arms with a single run, so provisional under PROTOCOL.md amendment 9 (stage 1): SEL, Q4, N5, N2, Q5.
Estimand: difference in the equally weighted macro-average of the five task scores, in percentage points, over the fixed items; request errors and truncations count as 0.

## 2. Gate and integrity

**Conclusion:** all 8 comparison pair(s) passed the gate; truncation rates differ by more than 1 point for SEL.

### Pairs

| Pair | Kind | Status | Problems | Warnings |
|---|---|---|---|---|
| SEL vs B0 | candidate | pass | 0 | 0 |
| Q4 vs B0 | exploratory | pass | 0 | 0 |
| N5 vs B0 | exploratory | pass | 0 | 0 |
| N2 vs B0 | exploratory | pass | 0 | 0 |
| Q5 vs B0 | exploratory | pass | 0 | 0 |
| SEL vs Q4 | exploratory-contrast | pass | 0 | 0 |
| Q4 vs Q5 | exploratory-contrast | pass | 0 | 0 |
| SEL vs Q5 | exploratory-contrast | pass | 0 | 0 |

### Settings per run

| Arm | Run | Status | Reasoning effort | Sampling | Sandbox | Epochs per task | Git commit | Code sha256 | Quantization recipe |
|---|---|---|---|---|---|---|---|---|---|
| B0 | `../results/wave1/B0a-v2` | complete | medium | T 0.8, top-p 0.9, top-k 40, max tokens None | local | mgsm 1, humaneval_plus 1, mbpp_plus 1, ifeval 1, aime2025 5 | None | a13aa4cc4491d3af8b73f61a7d2fc7bf568e6ad5fb840e4075e58f9fb5c5f7a8 | none |
| B0 | `../results/wave1/B0b-v2` | complete | medium | T 0.8, top-p 0.9, top-k 40, max tokens None | local | mgsm 1, humaneval_plus 1, mbpp_plus 1, ifeval 1, aime2025 5 | None | a13aa4cc4491d3af8b73f61a7d2fc7bf568e6ad5fb840e4075e58f9fb5c5f7a8 | none |
| SEL | `../results/wave1/SEL-v2` | complete | medium | T 0.8, top-p 0.9, top-k 40, max tokens None | local | mgsm 1, humaneval_plus 1, mbpp_plus 1, ifeval 1, aime2025 5 | None | a13aa4cc4491d3af8b73f61a7d2fc7bf568e6ad5fb840e4075e58f9fb5c5f7a8 | nvfp4 |
| Q4 | `../results/wave1/Q4v2` | complete | medium | T 0.8, top-p 0.9, top-k 40, max tokens None | local | mgsm 1, humaneval_plus 1, mbpp_plus 1, ifeval 1, aime2025 5 | None | a13aa4cc4491d3af8b73f61a7d2fc7bf568e6ad5fb840e4075e58f9fb5c5f7a8 | none |
| N5 | `../results/wave1/N5-v2` | complete | medium | T 0.8, top-p 0.9, top-k 40, max tokens None | local | mgsm 1, humaneval_plus 1, mbpp_plus 1, ifeval 1, aime2025 5 | None | a13aa4cc4491d3af8b73f61a7d2fc7bf568e6ad5fb840e4075e58f9fb5c5f7a8 | nvfp4 |
| N2 | `../results/wave1/N2-v2` | complete | medium | T 0.8, top-p 0.9, top-k 40, max tokens None | local | mgsm 1, humaneval_plus 1, mbpp_plus 1, ifeval 1, aime2025 5 | None | a13aa4cc4491d3af8b73f61a7d2fc7bf568e6ad5fb840e4075e58f9fb5c5f7a8 | nvfp4 |
| Q5 | `../results/wave1/Q5v2` | complete | medium | T 0.8, top-p 0.9, top-k 40, max tokens None | local | mgsm 1, humaneval_plus 1, mbpp_plus 1, ifeval 1, aime2025 5 | None | a13aa4cc4491d3af8b73f61a7d2fc7bf568e6ad5fb840e4075e58f9fb5c5f7a8 | none |

### Request errors and truncations per run (counted as failures)

| Arm | Run | Samples | Request errors | Truncations | Errors % | Truncated % |
|---|---|---|---|---|---|---|
| B0 | `../results/wave1/B0a-v2` | 1783 | 0 | 36 | 0.00% | 2.02% |
| B0 | `../results/wave1/B0b-v2` | 1783 | 0 | 37 | 0.00% | 2.08% |
| SEL | `../results/wave1/SEL-v2` | 1783 | 0 | 35 | 0.00% | 1.96% |
| Q4 | `../results/wave1/Q4v2` | 1783 | 0 | 90 | 0.00% | 5.05% |
| N5 | `../results/wave1/N5-v2` | 1783 | 0 | 40 | 0.00% | 2.24% |
| N2 | `../results/wave1/N2-v2` | 1783 | 0 | 47 | 0.00% | 2.64% |
| Q5 | `../results/wave1/Q5v2` | 1783 | 0 | 35 | 0.00% | 1.96% |

## 3. Confirmatory tests

**Conclusion:** at the margin of -3 points, 0 of 1 candidate(s) are NON-INFERIOR after Holm adjustment; outcomes: SEL INCONCLUSIVE.

One-sided non-inferiority against -3 points; bootstrap p = fraction of 10000 resamples with a macro difference at or below the margin; Holm over 1 test(s) at one-sided alpha 0.025. The one-sided 97.5% lower bound equals the lower end of the two-sided 95% interval.

| Candidate | Macro difference, pp [two-sided 95% interval] | Interval width, pp | One-sided 97.5% lower bound | p (raw) | p (Holm) | Outcome |
|---|---|---|---|---|---|---|
| SEL | -1.6 [-3.3, -0.0] | 3.3 | -3.33 | 0.0560 | 0.0560 | INCONCLUSIVE |

### Sensitivity to the margin (outcome, Holm-adjusted p)

| Candidate | margin -3 | margin -2 | margin -5 |
|---|---|---|---|
| SEL | INCONCLUSIVE (0.0560) | INCONCLUSIVE (0.3285) | NON-INFERIOR (0.0000) |

## 4. Per-task differences

**Conclusion:** no candidate has a task whose interval upper bound is below -5 points.

### SEL versus B0

| Task | Items | B0 % | SEL % | Difference, pp [95% interval] | Flag |
|---|---|---|---|---|---|
| mgsm | 550 | 67.2 [63.5, 70.8] | 66.5 [62.5, 70.5] | -0.6 [-2.8, +1.6] |  |
| humaneval_plus | 164 | 87.8 [83.2, 91.8] | 85.4 [79.9, 90.2] | -2.4 [-7.0, +1.8] |  |
| mbpp_plus | 378 | 71.0 [66.8, 75.3] | 70.6 [65.9, 75.1] | -0.4 [-3.6, +2.9] |  |
| ifeval | 541 | 88.3 [85.9, 90.6] | 88.5 [85.8, 91.1] | +0.3 [-1.8, +2.4] |  |
| aime2025 | 30 | 53.0 [38.3, 67.3] | 48.0 [33.3, 63.3] | -5.0 [-10.7, +0.0] |  |
| macro-average |  | 73.5 [70.1, 76.7] | 71.8 [68.4, 75.3] | -1.6 [-3.3, -0.0] |  |


## 5. Flags

**Conclusion:** task-level regression: Q4; truncation-rate difference above 1 pp: SEL, Q4, N5, N2, Q5.

- SEL: truncation-rate difference above 1 pp (scopes: mgsm, aime2025); SEL 1.96% (35/1783) against 2.05% (73/3566) for B0. A result to report, not a reason to discard; truncated answers count as failures.
- Q4: task-level regression in aime2025.
- Q4: truncation-rate difference above 1 pp (scopes: overall, mgsm, humaneval_plus, aime2025); Q4 5.05% (90/1783) against 2.05% (73/3566) for B0. A result to report, not a reason to discard; truncated answers count as failures.
- N5: truncation-rate difference above 1 pp (scopes: humaneval_plus, aime2025); N5 2.24% (40/1783) against 2.05% (73/3566) for B0. A result to report, not a reason to discard; truncated answers count as failures.
- N2: truncation-rate difference above 1 pp (scopes: humaneval_plus, aime2025); N2 2.64% (47/1783) against 2.05% (73/3566) for B0. A result to report, not a reason to discard; truncated answers count as failures.
- Q5: truncation-rate difference above 1 pp (scopes: aime2025); Q5 1.96% (35/1783) against 2.05% (73/3566) for B0. A result to report, not a reason to discard; truncated answers count as failures.

## 6. AIME accuracy among non-truncated samples (secondary)

**Conclusion:** AIME difference among non-truncated samples only (points): SEL -4.3 [-12.0, +2.8]. The headline counts truncated samples as failures.

| Candidate | Headline AIME difference (truncations = failures) | Difference among non-truncated samples | Truncated samples (arm / B0) | Arm accuracy among non-truncated % | B0 accuracy among non-truncated % |
|---|---|---|---|---|---|
| SEL | -5.0 [-10.7, +0.0] | -4.3 [-12.0, +2.8] | 16 / 26 | 53.7 [37.9, 69.4] | 58.0 [43.1, 71.9] |

## 7. MGSM per language

**Conclusion:** descriptive only; the largest drop in a language is SEL: ja -6.0 [-14.0, +0.0].

### SEL versus B0

| Language | Items | B0 % | SEL % | Difference, pp [95% interval] |
|---|---|---|---|---|
| bn | 50 | 37.0 [26.0, 49.0] | 38.0 [24.0, 52.0] | +1.0 [-11.0, +13.0] |
| de | 50 | 81.0 [70.0, 90.0] | 82.0 [70.0, 92.0] | +1.0 [-2.0, +4.0] |
| en | 50 | 94.0 [87.0, 99.0] | 94.0 [86.0, 100.0] | +0.0 [-5.0, +4.0] |
| es | 50 | 90.0 [82.0, 97.0] | 90.0 [82.0, 98.0] | +0.0 [-7.0, +6.0] |
| fr | 50 | 73.0 [62.0, 84.0] | 78.0 [66.0, 88.0] | +5.0 [-2.0, +12.0] |
| ja | 50 | 88.0 [79.0, 96.0] | 82.0 [70.0, 92.0] | -6.0 [-14.0, +0.0] |
| ru | 50 | 88.0 [79.0, 95.0] | 82.0 [70.0, 92.0] | -6.0 [-13.0, +0.0] |
| sw | 50 | 8.0 [3.0, 13.0] | 10.0 [2.0, 20.0] | +2.0 [-6.0, +11.0] |
| te | 50 | 8.0 [2.0, 15.0] | 4.0 [0.0, 10.0] | -4.0 [-12.0, +4.0] |
| th | 50 | 77.0 [67.0, 86.0] | 82.0 [70.0, 92.0] | +5.0 [-4.0, +14.0] |
| zh | 50 | 95.0 [89.0, 99.0] | 90.0 [80.0, 98.0] | -5.0 [-11.0, -1.0] |


## 8. Calibration of the interval

**Conclusion:** not provided (no --calibration), so the width of pure run-to-run noise is not shown.

Give the two runs of the reference as separate inputs with --calibration.

## 9. Exploratory arms (descriptive only)

**Conclusion:** descriptive only (difference and interval, no outcome label, no Holm membership): Q4 -4.1 [-6.2, -2.1]; N5 -1.2 [-3.0, +0.6]; N2 -3.3 [-5.3, -1.4]; Q5 -1.3 [-3.0, +0.5].

| Arm | mgsm | humaneval_plus | mbpp_plus | ifeval | aime2025 | Macro-average |
|---|---|---|---|---|---|---|
| Q4 | -1.9 [-4.2, +0.5] | -2.4 [-7.0, +2.1] | -0.9 [-4.4, +2.5] | -0.5 [-2.8, +1.8] | -15.0 [-22.7, -7.3] | -4.1 [-6.2, -2.1] |
| N5 | -0.3 [-2.5, +2.0] | -5.5 [-10.1, -1.2] | +0.4 [-2.8, +3.6] | -0.3 [-2.6, +1.9] | -0.3 [-7.0, +5.7] | -1.2 [-3.0, +0.6] |
| N2 | -2.1 [-4.6, +0.5] | -4.9 [-9.5, -0.6] | -2.5 [-6.0, +0.9] | -1.4 [-3.9, +1.2] | -5.7 [-13.0, +1.0] | -3.3 [-5.3, -1.4] |
| Q5 | +0.5 [-1.7, +2.6] | -3.7 [-8.2, +0.9] | -0.9 [-4.1, +2.1] | -2.5 [-5.0, +0.0] | +0.3 [-5.0, +6.0] | -1.3 [-3.0, +0.5] |

- Q4: task-level regression in aime2025.
- Q4: truncation-rate difference above 1 pp (scopes: overall, mgsm, humaneval_plus, aime2025); Q4 5.05% (90/1783) against 2.05% (73/3566) for B0. A result to report, not a reason to discard; truncated answers count as failures.
- N5: truncation-rate difference above 1 pp (scopes: humaneval_plus, aime2025); N5 2.24% (40/1783) against 2.05% (73/3566) for B0. A result to report, not a reason to discard; truncated answers count as failures.
- N2: truncation-rate difference above 1 pp (scopes: humaneval_plus, aime2025); N2 2.64% (47/1783) against 2.05% (73/3566) for B0. A result to report, not a reason to discard; truncated answers count as failures.
- Q5: truncation-rate difference above 1 pp (scopes: aime2025); Q5 1.96% (35/1783) against 2.05% (73/3566) for B0. A result to report, not a reason to discard; truncated answers count as failures.

## 10. Contrasts between arms (descriptive)

**Conclusion:** descriptive differences with 95% intervals, no outcome label: SEL versus Q4 +2.5 [+0.2, +4.8]; Q4 versus Q5 -2.9 [-5.0, -0.8]; SEL versus Q5 -0.4 [-2.5, +1.7].

| Contrast (first minus second) | mgsm | humaneval_plus | mbpp_plus | ifeval | aime2025 | Macro-average |
|---|---|---|---|---|---|---|
| SEL versus Q4 | +1.3 [-1.5, +3.8] | +0.0 [-4.9, +4.9] | +0.5 [-3.7, +4.5] | +0.7 [-1.8, +3.3] | +10.0 [+1.3, +19.3] | +2.5 [+0.2, +4.8] |
| Q4 versus Q5 | -2.4 [-5.1, +0.4] | +1.2 [-3.7, +6.1] | +0.0 [-3.7, +3.7] | +2.0 [-0.7, +5.0] | -15.3 [-23.3, -8.0] | -2.9 [-5.0, -0.8] |
| SEL versus Q5 | -1.1 [-3.8, +1.6] | +1.2 [-3.7, +6.1] | +0.5 [-3.4, +4.2] | +2.8 [+0.2, +5.4] | -5.3 [-13.3, +2.7] | -0.4 [-2.5, +1.7] |

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
| SEL | `../results/wave1/SEL-v2` | 23dc41f3bfcd18b6fa7f164f335adc508dd17d8635e72d197ceb7bd472f42e06 | 77b3e9f9415f31f74053773b03c7b78fd2f0511e8e7b943f770c2976d1705e99 | 81488b253371973770df7d8429b97044db34c1bb32bb01c662baa387a8ac669a |
| Q4 | `../results/wave1/Q4v2` | f2419ae5f8b38ebab8a0d5941055063277f2797c19d7ce277ff08a97a026fd57 | 12680e0433499384c63f4053d9754ee7bb4146f68746ac7b06c85bd2ed4fa385 | 81488b253371973770df7d8429b97044db34c1bb32bb01c662baa387a8ac669a |
| N5 | `../results/wave1/N5-v2` | 80b49b876f6000250ad42f4d6808cfa592f751cff992c148df0a30cd97a1858b | 35817674793d8487be21bed4df825866d9d2ff7487f159777cbd766cf30c84f6 | 81488b253371973770df7d8429b97044db34c1bb32bb01c662baa387a8ac669a |
| N2 | `../results/wave1/N2-v2` | b7887057dd932bf53e772eeffdc06f2ead542be694091e820b89f2e7c25b90a4 | d86b9ca4c5dae6a56e00d6853d50462cb2fcaea093a96e622aad72def2ef5e61 | 81488b253371973770df7d8429b97044db34c1bb32bb01c662baa387a8ac669a |
| Q5 | `../results/wave1/Q5v2` | a8e85677bfccdd487ad0ddb848582cd2aa60922e495f35efdfd69d4975af22c0 | 509d01d8e432120a2e9dd22650b5268d6933d9b315dc493abab41533df2d4988 | 81488b253371973770df7d8429b97044db34c1bb32bb01c662baa387a8ac669a |
