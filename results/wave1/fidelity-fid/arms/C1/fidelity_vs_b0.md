### Logit fidelity

64 chunks, 32704 scored positions, top-20 log-probs, 1000 bootstrap resamples over chunks (seed 0), 95% percentile intervals.

| metric | value | 95% CI |
|---|---|---|
| Perplexity, reference | 14.658 | [13.357, 15.845] |
| Perplexity, variant | 14.962 | [13.631, 16.183] |
| Perplexity ratio (variant / reference) | 1.0208 | [1.0167, 1.0247] |
| Mean KL(ref || variant), nats | 0.02632 | [0.02449, 0.02825] |
| Median KL(ref || variant), nats | 0.01023 | [0.00989, 0.01060] |
| 99th percentile KL, nats | 0.27750 | [0.24922, 0.31672] |
| Same top-1 token (fraction) | 0.9105 | [0.9076, 0.9136] |

Note: KL is computed on the union of the two top-k lists plus one 'other' bucket; tokens a list does not show are imputed (see fidelity_math.py and DECISIONS.md). It is an approximation, not the full-vocabulary divergence.

| file | model | endpoint | mode | k | collected (UTC) |
|---|---|---|---|---|---|
| reference | olala | http://127.0.0.1:8000/v1 | echo | 20 | 2026-11-21T07:40:25Z |
| variant | olala | http://127.0.0.1:8000/v1 | echo | 20 | 2026-11-21T07:58:52Z |
