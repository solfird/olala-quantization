### Logit fidelity

64 chunks, 32704 scored positions, top-20 log-probs, 1000 bootstrap resamples over chunks (seed 0), 95% percentile intervals.

| metric | value | 95% CI |
|---|---|---|
| Perplexity, reference | 14.655 | [13.357, 15.844] |
| Perplexity, variant | 14.944 | [13.625, 16.139] |
| Perplexity ratio (variant / reference) | 1.0197 | [1.0164, 1.0230] |
| Mean KL(ref || variant), nats | 0.02429 | [0.02265, 0.02601] |
| Median KL(ref || variant), nats | 0.00966 | [0.00933, 0.00997] |
| 99th percentile KL, nats | 0.25704 | [0.23354, 0.29124] |
| Same top-1 token (fraction) | 0.9122 | [0.9095, 0.9151] |

Note: KL is computed on the union of the two top-k lists plus one 'other' bucket; tokens a list does not show are imputed (see fidelity_math.py and DECISIONS.md). It is an approximation, not the full-vocabulary divergence.

| file | model | endpoint | mode | k | collected (UTC) |
|---|---|---|---|---|---|
| reference | olala | http://127.0.0.1:8000/v1 | echo | 20 | 2026-11-20T23:16:45Z |
| variant | olala | http://127.0.0.1:8000/v1 | echo | 20 | 2026-11-20T23:25:41Z |
