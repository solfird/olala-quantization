### Logit fidelity

64 chunks, 32704 scored positions, top-20 log-probs, 1000 bootstrap resamples over chunks (seed 0), 95% percentile intervals.

| metric | value | 95% CI |
|---|---|---|
| Perplexity, reference | 14.655 | [13.357, 15.844] |
| Perplexity, variant | 14.789 | [13.479, 15.975] |
| Perplexity ratio (variant / reference) | 1.0091 | [1.0065, 1.0117] |
| Mean KL(ref || variant), nats | 0.01466 | [0.01354, 0.01581] |
| Median KL(ref || variant), nats | 0.00497 | [0.00478, 0.00513] |
| 99th percentile KL, nats | 0.17157 | [0.14843, 0.19272] |
| Same top-1 token (fraction) | 0.9348 | [0.9321, 0.9376] |

Note: KL is computed on the union of the two top-k lists plus one 'other' bucket; tokens a list does not show are imputed (see fidelity_math.py and DECISIONS.md). It is an approximation, not the full-vocabulary divergence.

| file | model | endpoint | mode | k | collected (UTC) |
|---|---|---|---|---|---|
| reference | olala | http://127.0.0.1:8000/v1 | echo | 20 | 2026-11-20T23:16:45Z |
| variant | olala | http://127.0.0.1:8000/v1 | echo | 20 | 2026-11-20T23:38:17Z |
