# Accuracy wave: pre-registered analysis

Reference `B0`; confirmatory family SEL. Outcome labels are NON-INFERIOR, WORSE or INCONCLUSIVE, always read with the interval.

## 1. Summary

**Conclusion:** At the margin of -3 points on the macro-average (one-sided alpha 0.025, Holm over 1 candidate(s)), SEL: INCONCLUSIVE.

| Candidate | Macro difference, pp [95% interval] | Outcome | B0 macro % | Candidate macro % | Flags |
|---|---|---|---|---|---|
| SEL | -1.6 [-3.3, -0.0] | INCONCLUSIVE | 73.5 | 71.8 | truncation-rate difference |

Arms with a single run, so provisional under PROTOCOL.md amendment 9 (stage 1): SEL, F1, F2.
Estimand: difference in the equally weighted macro-average of the five task scores, in percentage points, over the fixed items; request errors and truncations count as 0.

## 2. Gate and integrity

**Conclusion:** all 5 comparison pair(s) passed the gate; truncation rates differ by more than 1 point for SEL.

### Pairs

| Pair | Kind | Status | Problems | Warnings |
|---|---|---|---|---|
| SEL vs B0 | candidate | pass | 0 | 0 |
| F1 vs B0 | exploratory | pass | 0 | 0 |
| F2 vs B0 | exploratory | pass | 0 | 0 |
| SEL vs F1 | exploratory-contrast | pass | 0 | 0 |
| SEL vs F2 | exploratory-contrast | pass | 0 | 0 |

### Settings per run

| Arm | Run | Status | Reasoning effort | Sampling | Sandbox | Epochs per task | Git commit | Code sha256 | Quantization recipe |
|---|---|---|---|---|---|---|---|---|---|
| B0 | `../results/wave1/B0a-v2` | complete | medium | T 0.8, top-p 0.9, top-k 40, max tokens None | local | mgsm 1, humaneval_plus 1, mbpp_plus 1, ifeval 1, aime2025 5 | None | a13aa4cc4491d3af8b73f61a7d2fc7bf568e6ad5fb840e4075e58f9fb5c5f7a8 | none |
| B0 | `../results/wave1/B0b-v2` | complete | medium | T 0.8, top-p 0.9, top-k 40, max tokens None | local | mgsm 1, humaneval_plus 1, mbpp_plus 1, ifeval 1, aime2025 5 | None | a13aa4cc4491d3af8b73f61a7d2fc7bf568e6ad5fb840e4075e58f9fb5c5f7a8 | none |
| SEL | `../results/wave1/SEL-v2` | complete | medium | T 0.8, top-p 0.9, top-k 40, max tokens None | local | mgsm 1, humaneval_plus 1, mbpp_plus 1, ifeval 1, aime2025 5 | None | a13aa4cc4491d3af8b73f61a7d2fc7bf568e6ad5fb840e4075e58f9fb5c5f7a8 | nvfp4 |
| F1 | `../results/wave1/F1-v2` | complete | medium | T 0.8, top-p 0.9, top-k 40, max tokens None | local | mgsm 1, humaneval_plus 1, mbpp_plus 1, ifeval 1, aime2025 5 | None | a13aa4cc4491d3af8b73f61a7d2fc7bf568e6ad5fb840e4075e58f9fb5c5f7a8 | none |
| F2 | `../results/wave1/F2-v2` | complete | medium | T 0.8, top-p 0.9, top-k 40, max tokens None | local | mgsm 1, humaneval_plus 1, mbpp_plus 1, ifeval 1, aime2025 5 | None | a13aa4cc4491d3af8b73f61a7d2fc7bf568e6ad5fb840e4075e58f9fb5c5f7a8 | fp8 |

### Request errors and truncations per run (counted as failures)

| Arm | Run | Samples | Request errors | Truncations | Errors % | Truncated % |
|---|---|---|---|---|---|---|
| B0 | `../results/wave1/B0a-v2` | 1783 | 0 | 36 | 0.00% | 2.02% |
| B0 | `../results/wave1/B0b-v2` | 1783 | 0 | 37 | 0.00% | 2.08% |
| SEL | `../results/wave1/SEL-v2` | 1783 | 0 | 35 | 0.00% | 1.96% |
| F1 | `../results/wave1/F1-v2` | 1783 | 0 | 21 | 0.00% | 1.18% |
| F2 | `../results/wave1/F2-v2` | 1783 | 0 | 37 | 0.00% | 2.08% |

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

**Conclusion:** task-level regression: none; truncation-rate difference above 1 pp: SEL, F1, F2.

- SEL: truncation-rate difference above 1 pp (scopes: mgsm, aime2025); SEL 1.96% (35/1783) against 2.05% (73/3566) for B0. A result to report, not a reason to discard; truncated answers count as failures.
- F1: truncation-rate difference above 1 pp (scopes: aime2025); F1 1.18% (21/1783) against 2.05% (73/3566) for B0. A result to report, not a reason to discard; truncated answers count as failures.
- F2: truncation-rate difference above 1 pp (scopes: humaneval_plus, aime2025); F2 2.08% (37/1783) against 2.05% (73/3566) for B0. A result to report, not a reason to discard; truncated answers count as failures.

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

**Conclusion:** descriptive only (difference and interval, no outcome label, no Holm membership): F1 -0.1 [-1.5, +1.3]; F2 -0.9 [-2.7, +0.8].

| Arm | mgsm | humaneval_plus | mbpp_plus | ifeval | aime2025 | Macro-average |
|---|---|---|---|---|---|---|
| F1 | +0.1 [-2.3, +2.5] | +2.4 [-0.9, +5.8] | -0.1 [-3.0, +2.8] | -0.6 [-2.9, +1.6] | -2.3 [-7.0, +2.0] | -0.1 [-1.5, +1.3] |
| F2 | +0.3 [-2.1, +2.7] | -1.2 [-5.5, +3.0] | -0.9 [-4.1, +2.2] | -2.5 [-4.8, -0.2] | -0.3 [-6.3, +5.3] | -0.9 [-2.7, +0.8] |

- F1: truncation-rate difference above 1 pp (scopes: aime2025); F1 1.18% (21/1783) against 2.05% (73/3566) for B0. A result to report, not a reason to discard; truncated answers count as failures.
- F2: truncation-rate difference above 1 pp (scopes: humaneval_plus, aime2025); F2 2.08% (37/1783) against 2.05% (73/3566) for B0. A result to report, not a reason to discard; truncated answers count as failures.

## 10. Contrasts between arms (descriptive)

**Conclusion:** descriptive differences with 95% intervals, no outcome label: SEL versus F1 -1.5 [-3.5, +0.3]; SEL versus F2 -0.7 [-2.7, +1.3].

| Contrast (first minus second) | mgsm | humaneval_plus | mbpp_plus | ifeval | aime2025 | Macro-average |
|---|---|---|---|---|---|---|
| SEL versus F1 | -0.7 [-3.5, +2.0] | -4.9 [-9.8, +0.0] | -0.3 [-4.2, +3.7] | +0.9 [-1.7, +3.5] | -2.7 [-9.3, +2.7] | -1.5 [-3.5, +0.3] |
| SEL versus F2 | -0.9 [-3.3, +1.5] | -1.2 [-6.1, +3.7] | +0.5 [-3.4, +4.8] | +2.8 [+0.0, +5.5] | -4.7 [-12.0, +2.0] | -0.7 [-2.7, +1.3] |

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
| F1 | `../results/wave1/F1-v2` | 39f4537e5bdb1a834ceef3f68f08b2bb08edad976f1ffc5d7532e2cf088dd5e3 | a6fedc0104e9c18c48f40a06f6dd7504cf6d2f492a65bec91897b09044275194 | 81488b253371973770df7d8429b97044db34c1bb32bb01c662baa387a8ac669a |
| F2 | `../results/wave1/F2-v2` | 45d361243d6e78448a1d1e3d60ca03effeb27c2e66ff5b7f3af01aa34938ba59 | 90de6d9669ccf96acad91ae2d5d81a9e6c001db300c072d8ae8ca4040c8e682b | 81488b253371973770df7d8429b97044db34c1bb32bb01c662baa387a8ac669a |
