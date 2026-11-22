# Nature of the truncated answers (stop_reason max_tokens)

A truncated sample is called a LOOP when the last 8,000 characters of its reasoning compress below 0.12 (zlib) or fewer than 25% of its last 200 lines are distinct; otherwise LONG (still producing new text). Counts at 0.10 and 0.15 are in the JSON.

| Arm | Samples | Truncated | Loops | Long | Visible answer text | Correct answer in the reasoning | Correct answer in its last quarter | Median reasoning (k chars) |
|---|---|---|---|---|---|---|---|---|
| bf16 (run a) | 1783 | 36 | 9 | 27 | 15 | 7 | 6 | 69 |
| bf16 (run b) | 1783 | 37 | 13 | 24 | 15 | 11 | 10 | 57 |
| FP8 online | 1783 | 21 | 3 | 18 | 11 | 5 | 4 | 53 |
| FP8 offline | 1783 | 37 | 15 | 22 | 17 | 7 | 6 | 51 |
| NVFP4 all linear (N2) | 1783 | 47 | 14 | 33 | 13 | 10 | 9 | 75 |
| NVFP4 experts only (N5) | 1783 | 40 | 11 | 29 | 13 | 9 | 8 | 72 |
| NVFP4 C2 (this build) | 1783 | 35 | 9 | 26 | 12 | 2 | 2 | 76 |
| OVH Q5_K_M | 1783 | 35 | 10 | 25 | 18 | 3 | 3 | 54 |
| Q4_K_M stand-in | 1783 | 90 | 23 | 67 | 50 | 23 | 21 | 56 |
| NVFP4 experts, calibrated (G1) | 1783 | 46 | 11 | 35 | 25 | 9 | 8 | 51 |
| bf16 GGUF (llama.cpp) | 1783 | 34 | 12 | 22 | 12 | 5 | 5 | 66 |

## Truncations per task

- bf16 (run a): humaneval_plus 1, inspect_evals/aime2025 15, inspect_evals/ifeval 9, inspect_evals/mgsm 9, mbpp_plus 2
- bf16 (run b): humaneval_plus 1, inspect_evals/aime2025 11, inspect_evals/ifeval 8, inspect_evals/mgsm 14, mbpp_plus 3
- FP8 online: humaneval_plus 2, inspect_evals/aime2025 8, inspect_evals/ifeval 4, inspect_evals/mgsm 6, mbpp_plus 1
- FP8 offline: humaneval_plus 4, inspect_evals/aime2025 9, inspect_evals/ifeval 9, inspect_evals/mgsm 12, mbpp_plus 3
- NVFP4 all linear (N2): humaneval_plus 4, inspect_evals/aime2025 20, inspect_evals/ifeval 12, inspect_evals/mgsm 8, mbpp_plus 3
- NVFP4 experts only (N5): humaneval_plus 3, inspect_evals/aime2025 18, inspect_evals/ifeval 9, inspect_evals/mgsm 10
- NVFP4 C2 (this build): humaneval_plus 2, inspect_evals/aime2025 16, inspect_evals/ifeval 10, inspect_evals/mgsm 3, mbpp_plus 4
- OVH Q5_K_M: inspect_evals/aime2025 11, inspect_evals/ifeval 11, inspect_evals/mgsm 10, mbpp_plus 3
- Q4_K_M stand-in: humaneval_plus 4, inspect_evals/aime2025 43, inspect_evals/ifeval 11, inspect_evals/mgsm 29, mbpp_plus 3
- NVFP4 experts, calibrated (G1): humaneval_plus 2, inspect_evals/aime2025 14, inspect_evals/ifeval 8, inspect_evals/mgsm 19, mbpp_plus 3
- bf16 GGUF (llama.cpp): humaneval_plus 2, inspect_evals/aime2025 12, inspect_evals/ifeval 7, inspect_evals/mgsm 11, mbpp_plus 2
