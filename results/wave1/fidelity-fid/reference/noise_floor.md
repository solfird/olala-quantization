### Logit fidelity

64 chunks, 32704 scored positions, top-20 log-probs, 1000 bootstrap resamples over chunks (seed 0), 95% percentile intervals.

| metric | value | 95% CI |
|---|---|---|
| Perplexity, reference | 14.658 | [13.357, 15.845] |
| Perplexity, variant | 14.647 | [13.353, 15.834] |
| Perplexity ratio (variant / reference) | 0.9993 | [0.9982, 1.0003] |
| Mean KL(ref || variant), nats | 0.00254 | [0.00206, 0.00304] |
| Median KL(ref || variant), nats | 0.00065 | [0.00005, 0.00105] |
| 99th percentile KL, nats | 0.02822 | [0.02374, 0.03231] |
| Same top-1 token (fraction) | 0.9769 | [0.9724, 0.9815] |

Note: KL is computed on the union of the two top-k lists plus one 'other' bucket; tokens a list does not show are imputed (see fidelity_math.py and DECISIONS.md). It is an approximation, not the full-vocabulary divergence.

| file | model | endpoint | mode | k | collected (UTC) |
|---|---|---|---|---|---|
| reference | olala | http://127.0.0.1:8000/v1 | echo | 20 | 2026-11-21T07:40:25Z |
| variant | olala | http://127.0.0.1:8000/v1 | echo | 20 | 2026-11-21T07:40:38Z |
