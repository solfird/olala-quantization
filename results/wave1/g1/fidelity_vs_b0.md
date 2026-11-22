### Logit fidelity

64 chunks, 32704 scored positions, top-20 log-probs, 1000 bootstrap resamples over chunks (seed 0), 95% percentile intervals.

| metric | value | 95% CI |
|---|---|---|
| Perplexity, reference | 14.649 | [13.356, 15.841] |
| Perplexity, variant | 15.124 | [13.783, 16.378] |
| Perplexity ratio (variant / reference) | 1.0324 | [1.0277, 1.0369] |
| Mean KL(ref || variant), nats | 0.02900 | [0.02669, 0.03128] |
| Median KL(ref || variant), nats | 0.01045 | [0.01002, 0.01085] |
| 99th percentile KL, nats | 0.31886 | [0.28224, 0.37756] |
| Same top-1 token (fraction) | 0.9070 | [0.9035, 0.9105] |

Note: KL is computed on the union of the two top-k lists plus one 'other' bucket; tokens a list does not show are imputed (see fidelity_math.py and DECISIONS.md). It is an approximation, not the full-vocabulary divergence.

| file | model | endpoint | mode | k | collected (UTC) |
|---|---|---|---|---|---|
| reference | olala | http://127.0.0.1:8000/v1 | echo | 20 | 2026-11-21T03:05:05Z |
| variant | olala | http://127.0.0.1:8000/v1 | echo | 20 | 2026-11-21T03:08:01Z |
