### Logit fidelity

64 chunks, 32704 scored positions, top-20 log-probs, 1000 bootstrap resamples over chunks (seed 0), 95% percentile intervals.

| metric | value | 95% CI |
|---|---|---|
| Perplexity, reference | 14.655 | [13.357, 15.844] |
| Perplexity, variant | 14.718 | [13.423, 15.907] |
| Perplexity ratio (variant / reference) | 1.0042 | [1.0020, 1.0065] |
| Mean KL(ref || variant), nats | 0.01106 | [0.01041, 0.01175] |
| Median KL(ref || variant), nats | 0.00487 | [0.00471, 0.00504] |
| 99th percentile KL, nats | 0.10158 | [0.09120, 0.11495] |
| Same top-1 token (fraction) | 0.9386 | [0.9363, 0.9410] |

Note: KL is computed on the union of the two top-k lists plus one 'other' bucket; tokens a list does not show are imputed (see fidelity_math.py and DECISIONS.md). It is an approximation, not the full-vocabulary divergence.

| file | model | endpoint | mode | k | collected (UTC) |
|---|---|---|---|---|---|
| reference | olala | http://127.0.0.1:8000/v1 | echo | 20 | 2026-11-20T23:16:45Z |
| variant | olala | http://127.0.0.1:8000/v1 | echo | 20 | 2026-11-20T23:34:59Z |
