### Logit fidelity

64 chunks, 32704 scored positions, top-20 log-probs, 1000 bootstrap resamples over chunks (seed 0), 95% percentile intervals.

| metric | value | 95% CI |
|---|---|---|
| Perplexity, reference | 14.655 | [13.357, 15.844] |
| Perplexity, variant | 14.846 | [13.531, 16.062] |
| Perplexity ratio (variant / reference) | 1.0130 | [1.0099, 1.0162] |
| Mean KL(ref || variant), nats | 0.01553 | [0.01500, 0.01611] |
| Median KL(ref || variant), nats | 0.00857 | [0.00835, 0.00878] |
| 99th percentile KL, nats | 0.12387 | [0.11559, 0.13643] |
| Same top-1 token (fraction) | 0.9235 | [0.9199, 0.9272] |

Note: KL is computed on the union of the two top-k lists plus one 'other' bucket; tokens a list does not show are imputed (see fidelity_math.py and DECISIONS.md). It is an approximation, not the full-vocabulary divergence.

| file | model | endpoint | mode | k | collected (UTC) |
|---|---|---|---|---|---|
| reference | olala | http://127.0.0.1:8000/v1 | echo | 20 | 2026-11-20T23:16:45Z |
| variant | olala | http://127.0.0.1:8000/v1 | echo | 20 | 2026-11-20T23:49:15Z |
