### Logit fidelity

64 chunks, 32704 scored positions, top-20 log-probs, 1000 bootstrap resamples over chunks (seed 0), 95% percentile intervals.

| metric | value | 95% CI |
|---|---|---|
| Perplexity, reference | 14.655 | [13.357, 15.844] |
| Perplexity, variant | 14.734 | [13.425, 15.945] |
| Perplexity ratio (variant / reference) | 1.0054 | [1.0029, 1.0080] |
| Mean KL(ref || variant), nats | 0.01291 | [0.01230, 0.01350] |
| Median KL(ref || variant), nats | 0.00584 | [0.00566, 0.00603] |
| 99th percentile KL, nats | 0.11343 | [0.10346, 0.12390] |
| Same top-1 token (fraction) | 0.9371 | [0.9346, 0.9396] |

Note: KL is computed on the union of the two top-k lists plus one 'other' bucket; tokens a list does not show are imputed (see fidelity_math.py and DECISIONS.md). It is an approximation, not the full-vocabulary divergence.

| file | model | endpoint | mode | k | collected (UTC) |
|---|---|---|---|---|---|
| reference | olala | http://127.0.0.1:8000/v1 | echo | 20 | 2026-11-20T23:16:45Z |
| variant | olala | http://127.0.0.1:8000/v1 | echo | 20 | 2026-11-20T23:46:33Z |
