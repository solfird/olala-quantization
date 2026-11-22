### Logit fidelity

64 chunks, 32704 scored positions, top-20 log-probs, 1000 bootstrap resamples over chunks (seed 0), 95% percentile intervals.

| metric | value | 95% CI |
|---|---|---|
| Perplexity, reference | 14.655 | [13.357, 15.844] |
| Perplexity, variant | 14.968 | [13.642, 16.205] |
| Perplexity ratio (variant / reference) | 1.0213 | [1.0180, 1.0245] |
| Mean KL(ref || variant), nats | 0.02277 | [0.02185, 0.02363] |
| Median KL(ref || variant), nats | 0.01187 | [0.01153, 0.01222] |
| 99th percentile KL, nats | 0.18060 | [0.16589, 0.19213] |
| Same top-1 token (fraction) | 0.9124 | [0.9092, 0.9155] |

Note: KL is computed on the union of the two top-k lists plus one 'other' bucket; tokens a list does not show are imputed (see fidelity_math.py and DECISIONS.md). It is an approximation, not the full-vocabulary divergence.

| file | model | endpoint | mode | k | collected (UTC) |
|---|---|---|---|---|---|
| reference | olala | http://127.0.0.1:8000/v1 | echo | 20 | 2026-11-20T23:16:45Z |
| variant | olala | http://127.0.0.1:8000/v1 | echo | 20 | 2026-11-20T23:41:03Z |
