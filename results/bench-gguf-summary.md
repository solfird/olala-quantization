# llama.cpp throughput of the Q4 stand-in and OVH's Q5_K_M (output tokens per second)

Synthetic random-token requests, 512 in and 512 out, `--ignore-eos`, one RTX PRO 4500 card per sweep, 12 server slots. A repeat with any failed request is excluded. Values are the median of the valid repeats (their number in brackets).

| Concurrent requests | Q4_K_M (stand-in) | Q5_K_M (OVH) | Q4 / Q5 |
|---|---|---|---|
| 1 | 78 (4) | 85 (4) | 0.92 |
| 4 | 162 (4) | 142 (4) | 1.14 |
| 12 | 327 (2) | 320 (2) | 1.02 |

## Excluded repeats (failed requests)

- q4_k_m-ovhrecipe c=12 c12_r2.json (sweep Q4sw2): 1 of 36 requests failed, throughput 336 not used
- q5_k_m c=12 c12_r1.json (sweep Q4sw2): 3 of 36 requests failed, throughput 292 not used
- q4_k_m-ovhrecipe c=12 c12_r1.json (sweep Q4-a): 2 of 36 requests failed, throughput 292 not used
- q5_k_m c=12 c12_r1.json (sweep Q4-a): 2 of 36 requests failed, throughput 296 not used
