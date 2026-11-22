### Logit fidelity

64 chunks, 32704 scored positions, top-20 log-probs, 1000 bootstrap resamples over chunks (seed 0), 95% percentile intervals.

| metric | value | 95% CI |
|---|---|---|
| Perplexity, reference | 14.655 | [13.357, 15.844] |
| Perplexity, variant | 14.696 | [13.400, 15.887] |
| Perplexity ratio (variant / reference) | 1.0027 | [1.0004, 1.0049] |
| Mean KL(ref || variant), nats | 0.00889 | [0.00825, 0.00951] |
| Median KL(ref || variant), nats | 0.00325 | [0.00311, 0.00340] |
| 99th percentile KL, nats | 0.09043 | [0.08028, 0.10032] |
| Same top-1 token (fraction) | 0.9475 | [0.9451, 0.9499] |

Note: KL is computed on the union of the two top-k lists plus one 'other' bucket; tokens a list does not show are imputed (see fidelity_math.py and DECISIONS.md). It is an approximation, not the full-vocabulary divergence.

| file | model | endpoint | mode | k | collected (UTC) |
|---|---|---|---|---|---|
| reference | olala | http://127.0.0.1:8000/v1 | echo | 20 | 2026-11-20T23:16:45Z |
| variant | olala | http://127.0.0.1:8000/v1 | echo | 20 | 2026-11-20T23:31:42Z |
