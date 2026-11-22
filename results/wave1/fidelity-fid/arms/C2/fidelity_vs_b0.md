### Logit fidelity

64 chunks, 32704 scored positions, top-20 log-probs, 1000 bootstrap resamples over chunks (seed 0), 95% percentile intervals.

| metric | value | 95% CI |
|---|---|---|
| Perplexity, reference | 14.658 | [13.357, 15.845] |
| Perplexity, variant | 14.875 | [13.556, 16.095] |
| Perplexity ratio (variant / reference) | 1.0148 | [1.0111, 1.0185] |
| Mean KL(ref || variant), nats | 0.02344 | [0.02215, 0.02481] |
| Median KL(ref || variant), nats | 0.00922 | [0.00891, 0.00952] |
| 99th percentile KL, nats | 0.25007 | [0.22307, 0.28116] |
| Same top-1 token (fraction) | 0.9134 | [0.9109, 0.9162] |

Note: KL is computed on the union of the two top-k lists plus one 'other' bucket; tokens a list does not show are imputed (see fidelity_math.py and DECISIONS.md). It is an approximation, not the full-vocabulary divergence.

| file | model | endpoint | mode | k | collected (UTC) |
|---|---|---|---|---|---|
| reference | olala | http://127.0.0.1:8000/v1 | echo | 20 | 2026-11-21T07:40:25Z |
| variant | olala | http://127.0.0.1:8000/v1 | echo | 20 | 2026-11-21T08:19:59Z |
