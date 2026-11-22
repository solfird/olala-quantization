### Logit fidelity

64 chunks, 32704 scored positions, top-20 log-probs, 1000 bootstrap resamples over chunks (seed 0), 95% percentile intervals.

| metric | value | 95% CI |
|---|---|---|
| Perplexity, reference | 14.655 | [13.357, 15.844] |
| Perplexity, variant | 15.823 | [14.415, 17.116] |
| Perplexity ratio (variant / reference) | 1.0797 | [1.0734, 1.0858] |
| Mean KL(ref || variant), nats | 0.06687 | [0.06402, 0.06962] |
| Median KL(ref || variant), nats | 0.03333 | [0.03247, 0.03413] |
| 99th percentile KL, nats | 0.61533 | [0.56312, 0.67503] |
| Same top-1 token (fraction) | 0.8492 | [0.8452, 0.8534] |

Note: KL is computed on the union of the two top-k lists plus one 'other' bucket; tokens a list does not show are imputed (see fidelity_math.py and DECISIONS.md). It is an approximation, not the full-vocabulary divergence.

| file | model | endpoint | mode | k | collected (UTC) |
|---|---|---|---|---|---|
| reference | olala | http://127.0.0.1:8000/v1 | echo | 20 | 2026-11-20T23:16:45Z |
| variant | olala | http://127.0.0.1:8000/v1 | echo | 20 | 2026-11-20T23:21:31Z |
