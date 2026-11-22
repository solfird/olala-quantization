### Logit fidelity

64 chunks, 32704 scored positions, top-20 log-probs, 1000 bootstrap resamples over chunks (seed 0), 95% percentile intervals.

| metric | value | 95% CI |
|---|---|---|
| Perplexity, reference | 14.649 | [13.356, 15.841] |
| Perplexity, variant | 14.643 | [13.354, 15.834] |
| Perplexity ratio (variant / reference) | 0.9995 | [0.9984, 1.0007] |
| Mean KL(ref || variant), nats | 0.00241 | [0.00185, 0.00294] |
| Median KL(ref || variant), nats | 0.00047 | [0.00000, 0.00093] |
| 99th percentile KL, nats | 0.02592 | [0.02150, 0.03023] |
| Same top-1 token (fraction) | 0.9789 | [0.9746, 0.9835] |

Note: KL is computed on the union of the two top-k lists plus one 'other' bucket; tokens a list does not show are imputed (see fidelity_math.py and DECISIONS.md). It is an approximation, not the full-vocabulary divergence.

| file | model | endpoint | mode | k | collected (UTC) |
|---|---|---|---|---|---|
| reference | olala | http://127.0.0.1:8000/v1 | echo | 20 | 2026-11-21T03:05:05Z |
| variant | olala | http://127.0.0.1:8000/v1 | echo | 20 | 2026-11-21T03:05:15Z |
