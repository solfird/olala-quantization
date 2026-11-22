# Olala activation probe

- **Layer numbering: 0-based** (`block.layer_idx`). The attention layers of this model are [4, 9, 14, 19, 24, 29, 34]; layer 0 is the first block. OVH's Appendix C uses the same convention.
- Prompts: 80 (32645 tokens, longest 1741); device cuda; weights dtype torch.bfloat16; attention implementation eager; torch 2.14.0+cu130.
- Mamba-3 path: pure-PyTorch float32 recurrence (OLALA_TORCH_MAMBA3=1), output cast to the activation dtype; not the TileLang kernel OVH measured with.
- Spike threshold used to flag a branch for ablation: 448; ablations on 16 prompts.
- Verdict words: `consistent` = the measured value satisfies the rule in the table; `not judged` = the claim has no checkable definition here. A verdict speaks only for this prompt set, sequence length and forward path.

## Claim by claim

| ID | OVH claim | Claimed | Measured here | Rule | Verdict |
|---|---|---|---|---|---|
| A1 | Three Mamba-3 layers emit runaway branch outputs (App. C.1; 12) | layers [1, 7, 23] | top-3 Mamba layers by max abs of the mixer branch: [1, 7, 23]; layers above 448: [0, 1, 2, 3, 7, 23]; 3rd/4th ratio 4.18e+03 | the top-3 set AND the set above the spike threshold equal OVH's | partly consistent |
| A1b | Layer-index convention (not stated in the report text; Table 50 uses L4 ... L34 for attention) | 0-based; attention layers 4, 9, ..., 34 | attention layers of this model: [4, 9, 14, 19, 24, 29, 34]; measured top-3 Mamba layers [1, 7, 23] (a 1-based reading of OVH's numbers would be 0-based layers [0, 6, 22], which do not match) | the model's attention layers equal OVH's list; the measured top-3 are compared with both readings | consistent |
| A2 | In each of these layers a single SSM head is responsible (heads 23, 26, 40) | {1: 23, 7: 26, 23: 40} | largest head per layer: {1: 23, 7: 26, 23: 40}; largest/second-largest head max: L1 2.11e+08, L7 1.73e+05, L23 1.05e+14 | the largest head of each claimed layer equals OVH's head index | consistent |
| A3 | Branch output magnitudes: layers 1 and 7 up to ~5e15, layer 23 up to ~1.5e11 (overall 1e11 to 1.8e16) | L1 ~5e+15, L7 ~5e+15, L23 ~1.5e+11 | L1 3.63e+16, L7 1.06e+16, L23 4.23e+11 (longest prompt 1741 tokens; OVH used 512) | measured max within one decade of the claimed maximum | consistent |
| A4 | Sibling heads of the dominant head stay at O(0.1) | O(0.1) | L1 median sibling RMS 0.00116, max 6.54e+08, L7 median sibling RMS 0.000293, max 5.37e+10, L23 median sibling RMS 0.00025, max 0.0203 | median sibling-head RMS within [0.01, 1] | not consistent |
| A5 | All other Mamba mixer outputs are O(0.1-10) | O(0.1-10) | largest mixer-branch max abs among the 26 other Mamba layers: 1.01e+08 | that maximum is at most 100 | not consistent |
| A6 | Removing the other 47 heads leaves 96-100% of the layer output bit-identical (bf16) | 96-100% | L1 100% (positions >= 10: 100%), L7 92.9% (positions >= 10: 95.1%), L23 100% (positions >= 10: 100%) | all-position share >= 96% in every claimed layer | partly consistent |
| A7 | The magnitudes grow with position, as t^1.9 (L1), t^1.0 (L7), t^0.36 (L23) | L1 1.9, L7 1.0, L23 0.36 | L1 0.983, L7 0.539, L23 0.296 (slope of mean log10 token max against log10 position, positions >= 16) | each exponent within +-0.5 of the claimed one | partly consistent |
| A8 | Position 0 carries the smallest value, not the largest | position 0 smallest | position 0 is the smallest mean log10 token max: {1: True, 7: True, 23: True} | true in every claimed layer | consistent |
| A9 | The mixer magnitudes appear at every position, with no privileged delimiter or first token | all positions; no BOS dependence | L1: tokens > 448 100%, argmax is first token in 0% and a special token in 0% of prompts, L7: tokens > 448 100%, argmax is first token in 0% and a special token in 0% of prompts, L23: tokens > 448 100%, argmax is first token in 0% and a special token in 0% of prompts | >= 90% of tokens above 448; the argmax is never the first token and a special token in <= 10% of prompts | consistent |
| B1 | Four MoE layers spike through the squared ReLU (0, 1, 5, 19) | layers [0, 1, 5, 19] | top-4 MoE layers by max post-ReLU^2 activation: [0, 1, 5, 19] (4th/5th ratio 10.7); by max MoE branch output: [0, 1, 5, 19] | the top-4 set by post-ReLU^2 maximum equals OVH's | consistent |
| B2 | These layers spike to 1e7-3e8 (the report does not say at which tensor; the post-ReLU^2 hidden is compared here) | 1e7 - 3e8 | L0: pre 3.79e+03, post 1.44e+07, latent out 1.72e+07, routed branch 3.77e+08, MoE branch 3.77e+08, L1: pre 1.95e+03, post 3.82e+06, latent out 3.87e+06, routed branch 1.84e+08, MoE branch 1.84e+08, L5: pre 290, post 8.4e+04, latent out 4.24e+05, routed branch 1.11e+07, MoE branch 1.11e+07, L19: pre 241, post 5.81e+04, latent out 6.55e+05, routed branch 1.99e+07, MoE branch 1.99e+07 | post-ReLU^2 maximum within [1e6, 3e9] (one decade around OVH's range) in every claimed layer | partly consistent |
| B3 | The up-projection amplifies the spike by a further ~30x | ~30x | L0: 22x (max routed branch / max latent expert output), L1: 47.5x (max routed branch / max latent expert output), L5: 26.1x (max routed branch / max latent expert output), L19: 30.4x (max routed branch / max latent expert output) | ratio of the two maxima within [10, 100] in every claimed layer | consistent |
| B4 | The MoE spikes are token-triggered: layer 19 fires on 4-7% of tokens (function words), layer 1 on rare / named-entity tokens | L19: 4-7% of tokens | L19: tokens at >= 50% / 10% / 1% of the layer max: 0.0429% / 1.03% / 1.64%; L1: tokens at >= 50% / 10% / 1% of the layer max: 2.95% / 45.9% / 85.7% | none: OVH gives no firing threshold; token lists are in the JSON (top_tokens of moe_post) | not judged |
| C1 | The residual stream is bounded: entries never exceed 0.69 at any layer or position | <= 0.69 | largest residual max abs after any block: 0.68 (layer 29) | maximum at most 0.695 (OVH's two decimals) | consistent |
| C2 | Activation block scales exceed the fp16 range by up to ten orders of magnitude | up to ~10 orders above 65504 | largest value at any site: 3.63e+16 (layer 1, mixer_out) = 11.7 orders above 65504 | measured orders within +-1.5 of 10 | not consistent |
| D-mixer-L1 | Ablation: zeroing the whole layer-1 mixer changes wikitext NLL by -0.009 (the magnitudes are 'vestigial') | -0.009 nats | -0.0507 nats (teacher-forced on the probe prompts); KL 0.0876, last-position KL 0.00411, top-1 agreement 0.91 | absolute measured change <= 0.1 (OVH's own control surgery costs up to +0.111); text differs from OVH's wikitext | consistent |
| D-head-L1 | Ablation: zeroing the integrator head of layer 1 changes wikitext NLL by +0.040 (the magnitudes are 'vestigial') | +0.040 nats | -0.072 nats (teacher-forced on the probe prompts); KL 0.192, last-position KL 0.0136, top-1 agreement 0.865 | absolute measured change <= 0.1 (OVH's own control surgery costs up to +0.111); text differs from OVH's wikitext | consistent |
| D-head-L23 | Ablation: zeroing the integrator head of layer 23 changes wikitext NLL by +0.000 (the magnitudes are 'vestigial') | +0.000 nats | -0.187 nats (teacher-forced on the probe prompts); KL 0.218, last-position KL 0.0311, top-1 agreement 0.852 | absolute measured change <= 0.1 (OVH's own control surgery costs up to +0.111); text differs from OVH's wikitext | not consistent |
| D-dims | Ablation: zeroing only the 'massive dimensions' changes NLL by +0.010 to +0.018; greedy generations stay coherent | +0.010 to +0.018 | this probe ablates whole branches and whole heads, not individual channels; no generation is run | none | not judged |
| E1 | The recurrent state of the three heads holds magnitudes of 2e9 to 1.2e10 after a 512-token prefill | 2e9 - 1.2e10 | the probe records outputs, not the SSM state (internal to the pure-PyTorch recurrence) | none | not judged |

## Mamba-3 layers: mixer branch (output of the projection) and heads

| layer | max abs | RMS | max/RMS | p99.9 (median) | max at position / token / flags | channel | % tokens > 448 / > 65504 | NaN / inf / >=3.4e38 | top head | top head max | 2nd head max | top/2nd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.15e+05 | 8.49e+03 | 13.6 | 4.61e+04 | 1520 / '4' / - | 1313 | 98.7 / 0.671 | 0 / 0 / 0 | 30 | 3.97e+04 | 2.55e+04 | 1.56 |
| 1 | 3.63e+16 | 1.96e+15 | 18.5 | 5e+15 | 1700 / '-' / - | 288 | 100 / 100 | 0 / 0 / 0 | 23 | 1.38e+17 | 6.54e+08 | 2.11e+08 |
| 2 | 1.01e+08 | 2.57e+06 | 39.4 | 2.67e+07 | 603 / ' the' / - | 192 | 100 / 100 | 0 / 0 / 0 | 10 | 3.98e+08 | 6.55e+07 | 6.08 |
| 3 | 8.85e+06 | 3.43e+05 | 25.8 | 4.36e+06 | 331 / '的' / - | 77 | 100 / 100 | 0 / 0 / 0 | 6 | 1.58e+06 | 1.49e+06 | 1.06 |
| 5 | 50.2 | 1.2 | 41.9 | 9.09 | 136 / 'once' / - | 1529 | 0 / 0 | 0 / 0 / 0 | 32 | 27.1 | 20 | 1.36 |
| 6 | 116 | 3.07 | 37.6 | 27.2 | 714 / '�' / - | 1274 | 0 / 0 | 0 / 0 / 0 | 31 | 63.2 | 61 | 1.04 |
| 7 | 1.06e+16 | 9.97e+13 | 106 | 9.02e+14 | 1609 / ' self' / - | 1274 | 100 / 100 | 0 / 0 / 0 | 26 | 9.29e+15 | 5.37e+10 | 1.73e+05 |
| 8 | 0.875 | 0.0256 | 34.2 | 0.229 | 282 / '</im_end/>' / special | 206 | 0 / 0 | 0 / 0 / 0 | 35 | 1.2 | 1.09 | 1.1 |
| 10 | 0.938 | 0.00899 | 104 | 0.0935 | 1 / '</im_start/>' / special | 206 | 0 / 0 | 0 / 0 / 0 | 45 | 0.346 | 0.084 | 4.12 |
| 11 | 1.39 | 0.0233 | 59.8 | 0.187 | 136 / 'once' / - | 206 | 0 / 0 | 0 / 0 / 0 | 38 | 0.758 | 0.196 | 3.86 |
| 12 | 0.824 | 0.0226 | 36.4 | 0.195 | 337 / '姆' / - | 206 | 0 / 0 | 0 / 0 / 0 | 40 | 0.633 | 0.246 | 2.57 |
| 13 | 0.688 | 0.02 | 34.3 | 0.183 | 0 / '</endoftext/>' / special first | 462 | 0 / 0 | 0 / 0 / 0 | 35 | 0.676 | 0.539 | 1.25 |
| 15 | 12.2 | 0.405 | 30.1 | 2.56 | 534 / ' break' / - | 206 | 0 / 0 | 0 / 0 / 0 | 37 | 9.75 | 6.84 | 1.42 |
| 16 | 1.93 | 0.039 | 49.5 | 0.404 | 0 / '</endoftext/>' / special first | 206 | 0 / 0 | 0 / 0 / 0 | 40 | 1.27 | 0.322 | 3.93 |
| 17 | 1.32 | 0.023 | 57.5 | 0.227 | 0 / '</endoftext/>' / special first | 206 | 0 / 0 | 0 / 0 / 0 | 18 | 0.555 | 0.441 | 1.26 |
| 18 | 0.688 | 0.0172 | 40 | 0.148 | 0 / '</endoftext/>' / special first | 1529 | 0 / 0 | 0 / 0 / 0 | 41 | 0.832 | 0.132 | 6.31 |
| 20 | 0.945 | 0.0192 | 49.3 | 0.185 | 1 / '</im_start/>' / special | 1296 | 0 / 0 | 0 / 0 / 0 | 31 | 0.539 | 0.177 | 3.05 |
| 21 | 12.7 | 0.201 | 63 | 1.7 | 0 / '</endoftext/>' / special first | 462 | 0 / 0 | 0 / 0 / 0 | 44 | 7.75 | 2.83 | 2.74 |
| 22 | 0.777 | 0.0165 | 47.2 | 0.146 | 0 / '</endoftext/>' / special first | 462 | 0 / 0 | 0 / 0 / 0 | 31 | 0.758 | 0.566 | 1.34 |
| 23 | 4.23e+11 | 1.04e+10 | 40.7 | 1.45e+11 | 716 / ' s' / - | 462 | 100 / 100 | 0 / 0 / 0 | 40 | 2.13e+12 | 0.0203 | 1.05e+14 |
| 25 | 0.824 | 0.0153 | 53.9 | 0.151 | 0 / '</endoftext/>' / special first | 1529 | 0 / 0 | 0 / 0 / 0 | 34 | 0.262 | 0.221 | 1.19 |
| 26 | 1.41 | 0.042 | 33.7 | 0.369 | 0 / '</endoftext/>' / special first | 462 | 0 / 0 | 0 / 0 / 0 | 4 | 0.551 | 0.543 | 1.01 |
| 27 | 0.551 | 0.0157 | 35.1 | 0.073 | 336 / '罗' / - | 462 | 0 / 0 | 0 / 0 / 0 | 12 | 0.801 | 0.249 | 3.22 |
| 28 | 0.352 | 0.0163 | 21.6 | 0.0806 | 165 / '8' / - | 685 | 0 / 0 | 0 / 0 / 0 | 43 | 0.598 | 0.377 | 1.59 |
| 30 | 0.98 | 0.0169 | 58.1 | 0.0986 | 458 / 'analysis' / - | 694 | 0 / 0 | 0 / 0 / 0 | 45 | 0.357 | 0.324 | 1.1 |
| 31 | 0.594 | 0.016 | 37.1 | 0.133 | 293 / '解' / - | 1529 | 0 / 0 | 0 / 0 / 0 | 26 | 0.404 | 0.363 | 1.11 |
| 32 | 0.887 | 0.0213 | 41.7 | 0.194 | 1 / 'Ch' / - | 1296 | 0 / 0 | 0 / 0 / 0 | 35 | 0.582 | 0.488 | 1.19 |
| 33 | 0.664 | 0.0156 | 42.5 | 0.0791 | 124 / ' ' / - | 685 | 0 / 0 | 0 / 0 / 0 | 0 | 0.379 | 0.363 | 1.04 |
| 35 | 2.31 | 0.0896 | 25.8 | 0.602 | 96 / ' cheap' / - | 374 | 0 / 0 | 0 / 0 / 0 | 6 | 6 | 5.66 | 1.06 |

## MoE layers: activations and branches

| layer | max pre-act | max post-ReLU^2 | max latent expert out | max routed branch | max shared branch | max MoE branch | RMS post-ReLU^2 | branch max/RMS | % tokens > 448 / > 65504 (branch) | expert most often in the top pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 3.79e+03 | 1.44e+07 | 1.72e+07 | 3.77e+08 | 0.0347 | 3.77e+08 | 4.52e+04 | 18.5 | 99.9 / 94.3 | 30 (x780) |
| 1 | 1.95e+03 | 3.82e+06 | 3.87e+06 | 1.84e+08 | 0.0645 | 1.84e+08 | 1.51e+04 | 33.9 | 87.2 / 86.3 | 132 (x498) |
| 2 | 8.25 | 68 | 60.2 | 2.93e+03 | 0.531 | 2.93e+03 | 0.205 | 169 | 2.37 / 0 | 180 (x1160) |
| 3 | 73.5 | 5.41e+03 | 7.14e+03 | 1.25e+05 | 0.311 | 1.25e+05 | 22.2 | 113 | 6.77 / 2.37 | 110 (x1677) |
| 4 | 0.98 | 0.441 | 0.113 | 0.34 | 0.395 | 0.633 | 0.0031 | 53.7 | 0 / 0 | 88 (x223) |
| 5 | 290 | 8.4e+04 | 4.24e+05 | 1.11e+07 | 0.402 | 1.11e+07 | 76.1 | 318 | 2.28 / 2.26 | 97 (x640) |
| 6 | 0.688 | 0.228 | 0.0518 | 0.214 | 0.35 | 0.361 | 0.00116 | 71.4 | 0 / 0 | 40 (x168) |
| 7 | 3.58e-07 | 1.28e-13 | 1.25e-16 | 8.51e-20 | 2.56 | 2.56 | 4.98e-15 | 84 | 0 / 0 | 75 (x162) |
| 8 | 3.74e-07 | 1.4e-13 | 1.16e-16 | 5.68e-20 | 0.742 | 0.742 | 5.08e-15 | 49.3 | 0 / 0 | 222 (x127) |
| 9 | 3.56e-07 | 1.27e-13 | 1.1e-16 | 5.46e-20 | 0.539 | 0.539 | 5.04e-15 | 48.8 | 0 / 0 | 94 (x87) |
| 10 | 3.73e-07 | 1.39e-13 | 1.2e-16 | 5.51e-20 | 0.252 | 0.252 | 4.96e-15 | 21 | 0 / 0 | 106 (x265) |
| 11 | 0.957 | 0.326 | 0.0728 | 0.22 | 0.453 | 0.508 | 0.00255 | 63.9 | 0 / 0 | 199 (x125) |
| 12 | 0.883 | 0.406 | 0.0757 | 0.209 | 0.879 | 0.879 | 0.00185 | 95.2 | 0 / 0 | 10 (x176) |
| 13 | 0.746 | 0.188 | 0.0508 | 0.139 | 0.467 | 0.467 | 0.00132 | 94.9 | 0 / 0 | 173 (x167) |
| 14 | 3.58e-07 | 1.28e-13 | 1.12e-16 | 5.42e-20 | 0.49 | 0.49 | 4.98e-15 | 41 | 0 / 0 | 83 (x118) |
| 15 | 1.08 | 0.273 | 0.0608 | 0.201 | 0.49 | 0.49 | 0.00145 | 65 | 0 / 0 | 190 (x250) |
| 16 | 0.68 | 0.165 | 0.028 | 0.0938 | 0.206 | 0.217 | 0.000795 | 66.2 | 0 / 0 | 229 (x136) |
| 17 | 0.797 | 0.277 | 0.082 | 0.117 | 0.699 | 0.703 | 0.00186 | 109 | 0 / 0 | 64 (x113) |
| 18 | 0.746 | 0.361 | 0.124 | 0.212 | 0.957 | 0.957 | 0.00289 | 88.7 | 0 / 0 | 49 (x151) |
| 19 | 241 | 5.81e+04 | 6.55e+05 | 1.99e+07 | 5.06 | 1.99e+07 | 39.3 | 835 | 1.69 / 1.66 | 34 (x502) |
| 20 | 1.1 | 0.367 | 0.119 | 0.179 | 1.62 | 1.62 | 0.00266 | 115 | 0 / 0 | 208 (x151) |
| 21 | 1.71 | 0.719 | 0.123 | 0.287 | 1.16 | 1.16 | 0.0035 | 73.3 | 0 / 0 | 45 (x130) |
| 22 | 0.965 | 0.193 | 0.0625 | 0.0913 | 0.777 | 0.777 | 0.00133 | 106 | 0 / 0 | 195 (x80) |
| 23 | 0.82 | 0.154 | 0.0645 | 0.113 | 0.828 | 0.824 | 0.00179 | 130 | 0 / 0 | 43 (x92) |
| 24 | 1.05 | 0.32 | 0.0874 | 0.25 | 0.445 | 0.527 | 0.00213 | 56 | 0 / 0 | 251 (x194) |
| 25 | 0.855 | 0.207 | 0.0645 | 0.112 | 0.266 | 0.332 | 0.00122 | 70.5 | 0 / 0 | 193 (x163) |
| 26 | 0.793 | 0.22 | 0.0742 | 0.117 | 0.42 | 0.449 | 0.00124 | 74.9 | 0 / 0 | 235 (x187) |
| 27 | 0.797 | 0.18 | 0.0479 | 0.155 | 0.291 | 0.35 | 0.00102 | 72.8 | 0 / 0 | 240 (x184) |
| 28 | 0.859 | 0.185 | 0.0476 | 0.178 | 0.461 | 0.465 | 0.00117 | 83.6 | 0 / 0 | 150 (x223) |
| 29 | 0.926 | 0.24 | 0.0435 | 0.122 | 0.209 | 0.237 | 0.00082 | 75.5 | 0 / 0 | 75 (x170) |
| 30 | 0.859 | 0.228 | 0.0342 | 0.207 | 0.249 | 0.256 | 0.000504 | 96.1 | 0 / 0 | 131 (x196) |
| 31 | 0.828 | 0.131 | 0.0184 | 0.0505 | 0.246 | 0.247 | 0.00045 | 101 | 0 / 0 | 214 (x138) |
| 32 | 1.27 | 0.338 | 0.0713 | 0.175 | 0.244 | 0.252 | 0.00106 | 50 | 0 / 0 | 208 (x364) |
| 33 | 0.762 | 0.159 | 0.0393 | 0.0618 | 0.183 | 0.186 | 0.000535 | 75.5 | 0 / 0 | 140 (x324) |
| 34 | 0.902 | 0.299 | 0.0258 | 0.115 | 0.127 | 0.147 | 0.000641 | 53.5 | 0 / 0 | 40 (x129) |
| 35 | 24.5 | 302 | 792 | 1.69e+03 | 0.543 | 1.69e+03 | 0.0247 | 1.58e+03 | 0.00306 / 0 | 49 (x181) |

## Residual stream after each block

| layer | max abs | RMS | NaN |
|---|---|---|---|
| 0 | 0.235 | 0.0255 | 0 |
| 1 | 0.307 | 0.0256 | 0 |
| 2 | 0.418 | 0.0255 | 0 |
| 3 | 0.434 | 0.0256 | 0 |
| 4 | 0.453 | 0.0255 | 0 |
| 5 | 0.479 | 0.0255 | 0 |
| 6 | 0.562 | 0.0255 | 0 |
| 7 | 0.436 | 0.0254 | 0 |
| 8 | 0.451 | 0.0255 | 0 |
| 9 | 0.531 | 0.0255 | 0 |
| 10 | 0.49 | 0.0255 | 0 |
| 11 | 0.531 | 0.0254 | 0 |
| 12 | 0.602 | 0.0254 | 0 |
| 13 | 0.445 | 0.0254 | 0 |
| 14 | 0.539 | 0.0254 | 0 |
| 15 | 0.664 | 0.0253 | 0 |
| 16 | 0.465 | 0.0253 | 0 |
| 17 | 0.52 | 0.0253 | 0 |
| 18 | 0.428 | 0.0253 | 0 |
| 19 | 0.439 | 0.0252 | 0 |
| 20 | 0.543 | 0.0252 | 0 |
| 21 | 0.406 | 0.0253 | 0 |
| 22 | 0.406 | 0.0253 | 0 |
| 23 | 0.418 | 0.0253 | 0 |
| 24 | 0.496 | 0.0253 | 0 |
| 25 | 0.574 | 0.0253 | 0 |
| 26 | 0.578 | 0.0253 | 0 |
| 27 | 0.5 | 0.0253 | 0 |
| 28 | 0.641 | 0.0253 | 0 |
| 29 | 0.68 | 0.0253 | 0 |
| 30 | 0.562 | 0.0253 | 0 |
| 31 | 0.496 | 0.0253 | 0 |
| 32 | 0.629 | 0.0253 | 0 |
| 33 | 0.438 | 0.0253 | 0 |
| 34 | 0.461 | 0.0253 | 0 |
| 35 | 0.174 | 0.0253 | 0 |

## Dependence of the next-token distribution on the spiking branches

| ablation (layer.branch:mode) | KL mean (nats) | KL max | KL last position | top-1 agreement | delta NLL | non-finite positions |
|---|---|---|---|---|---|---|
| L1.mixer_out:noop | 0 | 0 | 0 | 1 | 0 | 0 |
| L1.mixer_out:zero | 0.0876 | 5.62 | 0.00411 | 0.91 | -0.0507 | 0 |
| L1.mixer_out:clamp448 | 0.139 | 7.87 | 0.01 | 0.884 | -0.0902 | 0 |
| L1.head23:zero | 0.192 | 11 | 0.0136 | 0.865 | -0.072 | 0 |
| L7.mixer_out:zero | 0.184 | 12.2 | 0.00886 | 0.866 | 0.0248 | 0 |
| L7.mixer_out:clamp448 | 0.0963 | 6.77 | 0.00511 | 0.905 | 0.00738 | 0 |
| L7.head26:zero | 0.172 | 8.21 | 0.0123 | 0.869 | -0.0625 | 0 |
| L23.mixer_out:zero | 0.149 | 9.06 | 0.0451 | 0.87 | -0.158 | 0 |
| L23.mixer_out:clamp448 | 0.178 | 8.88 | 0.032 | 0.867 | -0.0463 | 0 |
| L23.head40:zero | 0.218 | 11.3 | 0.0311 | 0.852 | -0.187 | 0 |
| L0.routed_out:zero | 0.295 | 12.1 | 0.0224 | 0.829 | -0.044 | 0 |
| L0.routed_out:clamp448 | 0.428 | 12.9 | 0.0517 | 0.791 | -0.04 | 0 |
| L0.moe_out:zero | 0.239 | 9.6 | 0.0374 | 0.839 | -0.132 | 0 |
| L0.moe_out:clamp448 | 0.428 | 12.9 | 0.0517 | 0.791 | -0.04 | 0 |
| L1.routed_out:zero | 0.117 | 6.75 | 0.006 | 0.907 | -0.00567 | 0 |
| L1.routed_out:clamp448 | 0.162 | 9.25 | 0.0149 | 0.88 | -0.0678 | 0 |
| L1.moe_out:zero | 0.252 | 9.59 | 0.0474 | 0.837 | -0.223 | 0 |
| L1.moe_out:clamp448 | 0.162 | 9.25 | 0.0149 | 0.88 | -0.0678 | 0 |
| L2.mixer_out:zero | 0.456 | 13.1 | 0.226 | 0.794 | -0.0748 | 0 |
| L2.mixer_out:clamp448 | 0.304 | 9.53 | 0.0641 | 0.834 | 0.1 | 0 |
| L2.head10:zero | 0.091 | 7.4 | 0.00319 | 0.911 | -0.000581 | 0 |
| L19.routed_out:zero | 0.8 | 16.3 | 0.0627 | 0.739 | -0.44 | 0 |
| L19.routed_out:clamp448 | 0.0411 | 5.89 | 0.0082 | 0.939 | 0.0133 | 0 |
| L19.moe_out:zero | 1.28 | 15.4 | 0.282 | 0.668 | -0.608 | 0 |
| L19.moe_out:clamp448 | 0.0412 | 5.89 | 0.00825 | 0.939 | 0.0136 | 0 |
| L5.mixer_out:zero | 0.429 | 9.79 | 0.0766 | 0.793 | 0.0113 | 0 |
| L4.routed_out:zero | 0.885 | 13.9 | 0.103 | 0.736 | -0.0575 | 0 |
| L4.moe_out:zero | 1.05 | 14.2 | 0.647 | 0.679 | -0.658 | 0 |

Prompts used: mgsm:de_2a65fc42, mgsm:ja_fc0700e5, mgsm:zh_115d95f9, humaneval_plus:HumanEval/53, humaneval_plus:HumanEval/111, mbpp_plus:77, mbpp_plus:419, mbpp_plus:619, ifeval:1800, ifeval:2531, aime2025:1, aime2025:14, aime2025:28, plain:fr-loire, plain:en-long-1, plain:mixed-long. `noop` is a control that must give 0; `zero` replaces the branch (or head) by zeros; `clamp448` clamps each element to [-448, 448].

## What this means for quantization

| layer | site | max abs | orders above e4m3 max (448) | orders above fp16 max (65504) | % tokens > 448 | % tokens > 65504 | NaN / inf |
|---|---|---|---|---|---|---|---|
| 0 | mixer_out | 1.15e+05 | 2.41 | 0.245 | 98.7 | 0.671 | 0 / 0 |
| 0 | moe_out | 3.77e+08 | 5.93 | 3.76 | 99.9 | 94.3 | 0 / 0 |
| 0 | routed_out | 3.77e+08 | 5.93 | 3.76 | 99.9 | 94.3 | 0 / 0 |
| 0 | routed_latent | 1.72e+07 | 4.58 | 2.42 | 96.7 | 88.9 | 0 / 0 |
| 0 | moe_pre | 3.79e+03 | 0.928 | -1.24 | 97.7 | 0 | 0 / 0 |
| 0 | moe_post | 1.44e+07 | 4.51 | 2.34 | 100 | 92.6 | 0 / 0 |
| 1 | mixer_out | 3.63e+16 | 13.9 | 11.7 | 100 | 100 | 0 / 0 |
| 1 | moe_out | 1.84e+08 | 5.61 | 3.45 | 87.2 | 86.3 | 0 / 0 |
| 1 | routed_out | 1.84e+08 | 5.61 | 3.45 | 87.2 | 86.3 | 0 / 0 |
| 1 | routed_latent | 3.87e+06 | 3.94 | 1.77 | 86.8 | 68.7 | 0 / 0 |
| 1 | moe_pre | 1.95e+03 | 0.639 | -1.53 | 79.2 | 0 | 0 / 0 |
| 1 | moe_post | 3.82e+06 | 3.93 | 1.77 | 87.2 | 83.1 | 0 / 0 |
| 2 | mixer_out | 1.01e+08 | 5.35 | 3.19 | 100 | 100 | 0 / 0 |
| 2 | moe_out | 2.93e+03 | 0.815 | -1.35 | 2.37 | 0 | 0 / 0 |
| 2 | routed_out | 2.93e+03 | 0.815 | -1.35 | 2.37 | 0 | 0 / 0 |
| 3 | mixer_out | 8.85e+06 | 4.3 | 2.13 | 100 | 100 | 0 / 0 |
| 3 | moe_out | 1.25e+05 | 2.45 | 0.282 | 6.77 | 2.37 | 0 / 0 |
| 3 | routed_out | 1.25e+05 | 2.45 | 0.282 | 6.77 | 2.37 | 0 / 0 |
| 3 | routed_latent | 7.14e+03 | 1.2 | -0.963 | 6.77 | 0 | 0 / 0 |
| 3 | moe_post | 5.41e+03 | 1.08 | -1.08 | 6.77 | 0 | 0 / 0 |
| 5 | moe_out | 1.11e+07 | 4.39 | 2.23 | 2.28 | 2.26 | 0 / 0 |
| 5 | routed_out | 1.11e+07 | 4.39 | 2.23 | 2.28 | 2.26 | 0 / 0 |
| 5 | routed_latent | 4.24e+05 | 2.98 | 0.811 | 2.26 | 1.34 | 0 / 0 |
| 5 | moe_post | 8.4e+04 | 2.27 | 0.108 | 2.26 | 0.0735 | 0 / 0 |
| 7 | mixer_out | 1.06e+16 | 13.4 | 11.2 | 100 | 100 | 0 / 0 |
| 19 | moe_out | 1.99e+07 | 4.65 | 2.48 | 1.69 | 1.66 | 0 / 0 |
| 19 | routed_out | 1.99e+07 | 4.65 | 2.48 | 1.69 | 1.66 | 0 / 0 |
| 19 | routed_latent | 6.55e+05 | 3.17 | 1 | 1.69 | 0.91 | 0 / 0 |
| 19 | moe_post | 5.81e+04 | 2.11 | -0.052 | 1.67 | 0 | 0 / 0 |
| 23 | mixer_out | 4.23e+11 | 8.98 | 6.81 | 100 | 100 | 0 / 0 |
| 35 | moe_out | 1.69e+03 | 0.576 | -1.59 | 0.00306 | 0 | 0 / 0 |
| 35 | routed_out | 1.69e+03 | 0.576 | -1.59 | 0.00306 | 0 | 0 / 0 |
| 35 | routed_latent | 792 | 0.247 | -1.92 | 0.00306 | 0 | 0 / 0 |

## Experts producing the largest post-ReLU^2 values (flagged MoE layers)

- layer 0: expert 30: 780 of the top pairs (max 1.44e+07), expert 206: 744 of the top pairs (max 7.77e+06), expert 217: 368 of the top pairs (max 6.49e+06), expert 122: 200 of the top pairs (max 6.39e+06), expert 42: 170 of the top pairs (max 6.82e+06), expert 156: 81 of the top pairs (max 5.31e+06). Largest tokens: ' self'@303 1.44e+07, ' self'@303 1.44e+07, ' self'@303 1.44e+07, ' self'@303 1.44e+07, ' self'@303 1.44e+07, ' self'@303 1.44e+07, ' self'@303 1.44e+07, ' self'@303 1.44e+07.
- layer 1: expert 132: 498 of the top pairs (max 3.82e+06), expert 13: 478 of the top pairs (max 1.96e+06), expert 4: 430 of the top pairs (max 2.72e+06), expert 69: 372 of the top pairs (max 1.96e+06), expert 117: 217 of the top pairs (max 1.92e+06), expert 159: 183 of the top pairs (max 2.03e+06). Largest tokens: ' corresponding'@337 3.82e+06, ' кроме'@343 3.65e+06, ' manner'@345 3.57e+06, 'file'@339 3.57e+06, '茨'@337 3.54e+06, 'ponse'@327 3.51e+06, ',a'@455 3.51e+06, 'ponse'@346 3.47e+06.
- layer 5: expert 97: 640 of the top pairs (max 8.4e+04), expert 190: 144 of the top pairs (max 11.3), expert 112: 135 of the top pairs (max 0.348), expert 59: 85 of the top pairs (max 0.746), expert 41: 79 of the top pairs (max 0.155), expert 138: 74 of the top pairs (max 310). Largest tokens: "'"@918 8.4e+04, "'"@293 8.19e+04, "'"@315 8.09e+04, 'の'@335 7.94e+04, ' les'@199 7.94e+04, "'"@940 7.73e+04, ' les'@102 7.53e+04, ' les'@102 7.37e+04.
- layer 19: expert 34: 502 of the top pairs (max 5.81e+04), expert 26: 118 of the top pairs (max 0.266), expert 127: 106 of the top pairs (max 0.0952), expert 225: 100 of the top pairs (max 0.262), expert 77: 72 of the top pairs (max 0.11), expert 213: 66 of the top pairs (max 0.168). Largest tokens: 'ภา'@395 5.81e+04, 'analysis'@321 5.3e+04, 'analysis'@383 5.15e+04, 'analysis'@412 5.12e+04, 'analysis'@373 5.07e+04, 'analysis'@368 4.89e+04, 'analysis'@329 4.89e+04, 'analysis'@402 4.51e+04.

## Caveats

- The forward path is the repository's pure-PyTorch float32 Mamba-3 recurrence and expert loop, not the TileLang / scattermoe kernels OVH measured with; magnitudes of 1e15 are representable in both but a kernel-level rounding difference cannot be excluded.
- OVH measured teacher-forced NLL on 512-token wikitext-2 passages; the prompts here are chat-templated task inputs and plain text, so ablation deltas are not on the same text and the position-dependent magnitudes (A3, A7) depend on the sequence length.
- Verdict rules are mine and printed in the table; OVH's wording (`O(0.1)`, `up to`) is not a numeric threshold.
- The SSM state itself is not recorded (claim E1).
- Every figure is a maximum or a pooled statistic over the listed prompts, not a population estimate.
