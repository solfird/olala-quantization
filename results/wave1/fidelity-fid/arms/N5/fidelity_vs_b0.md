### Logit fidelity

64 chunks, 32704 scored positions, top-20 log-probs, 1000 bootstrap resamples over chunks (seed 0), 95% percentile intervals.

| metric | value | 95% CI |
|---|---|---|
| Perplexity, reference | 14.658 | [13.357, 15.845] |
| Perplexity, variant | 14.949 | [13.631, 16.148] |
| Perplexity ratio (variant / reference) | 1.0199 | [1.0162, 1.0234] |
| Mean KL(ref || variant), nats | 0.02443 | [0.02275, 0.02626] |
| Median KL(ref || variant), nats | 0.00959 | [0.00927, 0.00994] |
| 99th percentile KL, nats | 0.25695 | [0.22754, 0.29844] |
| Same top-1 token (fraction) | 0.9133 | [0.9103, 0.9161] |

Note: KL is computed on the union of the two top-k lists plus one 'other' bucket; tokens a list does not show are imputed (see fidelity_math.py and DECISIONS.md). It is an approximation, not the full-vocabulary divergence.

| file | model | endpoint | mode | k | collected (UTC) |
|---|---|---|---|---|---|
| reference | olala | http://127.0.0.1:8000/v1 | echo | 20 | 2026-11-21T07:40:25Z |
| variant | olala | http://127.0.0.1:8000/v1 | echo | 20 | 2026-11-21T07:49:43Z |
