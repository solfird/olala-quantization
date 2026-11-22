### Logit fidelity

64 chunks, 32704 scored positions, top-20 log-probs, 1000 bootstrap resamples over chunks (seed 0), 95% percentile intervals.

| metric | value | 95% CI |
|---|---|---|
| Perplexity, reference | 14.655 | [13.357, 15.844] |
| Perplexity, variant | 14.642 | [13.345, 15.829] |
| Perplexity ratio (variant / reference) | 0.9991 | [0.9980, 1.0001] |
| Mean KL(ref || variant), nats | 0.00204 | [0.00151, 0.00255] |
| Median KL(ref || variant), nats | 0.00000 | [0.00000, 0.00060] |
| 99th percentile KL, nats | 0.02405 | [0.01935, 0.02872] |
| Same top-1 token (fraction) | 0.9815 | [0.9767, 0.9862] |

Note: KL is computed on the union of the two top-k lists plus one 'other' bucket; tokens a list does not show are imputed (see fidelity_math.py and DECISIONS.md). It is an approximation, not the full-vocabulary divergence.

| file | model | endpoint | mode | k | collected (UTC) |
|---|---|---|---|---|---|
| reference | olala | http://127.0.0.1:8000/v1 | echo | 20 | 2026-11-20T23:16:45Z |
| variant | olala | http://127.0.0.1:8000/v1 | echo | 20 | 2026-11-20T23:16:55Z |
