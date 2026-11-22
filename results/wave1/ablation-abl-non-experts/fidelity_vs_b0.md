### Logit fidelity

64 chunks, 32704 scored positions, top-20 log-probs, 1000 bootstrap resamples over chunks (seed 0), 95% percentile intervals.

| metric | value | 95% CI |
|---|---|---|
| Perplexity, reference | 14.655 | [13.357, 15.844] |
| Perplexity, variant | 15.483 | [14.087, 16.770] |
| Perplexity ratio (variant / reference) | 1.0565 | [1.0511, 1.0617] |
| Mean KL(ref || variant), nats | 0.04976 | [0.04799, 0.05157] |
| Median KL(ref || variant), nats | 0.02652 | [0.02593, 0.02715] |
| 99th percentile KL, nats | 0.45956 | [0.42105, 0.49084] |
| Same top-1 token (fraction) | 0.8690 | [0.8652, 0.8729] |

Note: KL is computed on the union of the two top-k lists plus one 'other' bucket; tokens a list does not show are imputed (see fidelity_math.py and DECISIONS.md). It is an approximation, not the full-vocabulary divergence.

| file | model | endpoint | mode | k | collected (UTC) |
|---|---|---|---|---|---|
| reference | olala | http://127.0.0.1:8000/v1 | echo | 20 | 2026-11-20T23:16:45Z |
| variant | olala | http://127.0.0.1:8000/v1 | echo | 20 | 2026-11-20T23:28:28Z |
