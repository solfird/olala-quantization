# Weight-level quantizability statistics

Checkpoint index sha256 (`safetensors_header`): `81df2fdc92cc4faa6c66b9786d2103045c6b466a575b31f97151d64f653882f5`

Source: `/workspace/models/olala-bf16`. Device: cuda. torch 2.14.0+cu130. Tensors measured: 411; not weight matrices (skipped): 534.

## Formats

* **NVFP4 (g16, e4m3 scale, fp32 global)** (`nvfp4`): olala_quant.nvfp4.quantize_nvfp4: e2m1, group 16, e4m3 block scale, fp32 global scale 448*6/amax (per tensor, per expert for 3D); underflow = nonzero block with scale zero or below 2^-6, flushed = scale zero, saturate = scale above 448 before the clamp
* **FP8 e4m3 per-output-channel** (`fp8`): olala_quant.fp8.quantize_fp8: e4m3, one fp32 scale per output row (per expert row for 3D), scale floor 1/(448*512); underflow = nonzero row clamped to the floor
* **INT4 sym g128 (fp16 scale)** (`int4_g128`): symmetric int4, group 128, scale = amax/7.5 stored fp16, q = clamp(round(w/scale), -8, 7), ties to even; underflow = fp16 scale below 2^-14, flushed = zero, saturate = amax/7.5 > 65504
* **MXFP4 (g32, e8m0 scale)** (`mxfp4`): OCP MX v1.0: e2m1, group 32, e8m0 scale 2^(floor(log2 amax) - 2) clamped to [-127, 127], elements clamped to +-6 then rounded to nearest e2m1 (ties to even code); underflow = exponent < -127

## Round-to-nearest error per class and format

Mean error is the parameter-weighted mean of the per-tensor relative Frobenius error `||W-Q(W)||/||W||`; the worst layer is the layer of the worst tensor of the class; ratio to best is the mean error over that of the best format of the class (1.00x is the best). Scale underflow and saturation are the parameter-weighted fractions of weights in blocks whose scale underflows or saturates.

| class | format | tensors | params | mean rel. error | worst layer (error) | ratio to best | scale underflow | scale saturate |
|---|---|---:|---:|---:|---|---:|---:|---:|
| `routed_up` | nvfp4 | 36 | 2,717,908,992 | 0.09502 | L0 (0.09716) | 3.63x | 0% | 0% |
| `routed_up` | fp8 | 36 | 2,717,908,992 | 0.02618 | L14 (0.02656) | 1.00x | 18.5% | 0% |
| `routed_up` | int4_g128 | 36 | 2,717,908,992 | 0.1073 | L2 (0.1191) | 4.10x | 13.8% | 0% |
| `routed_up` | mxfp4 | 36 | 2,717,908,992 | 0.116 | L4 (0.1174) | 4.43x | 0% | 0% |
| `routed_down` | nvfp4 | 36 | 2,717,908,992 | 0.09148 | L14 (0.0951) | 3.58x | 0% | 0% |
| `routed_down` | fp8 | 36 | 2,717,908,992 | 0.02553 | L14 (0.02657) | 1.00x | 13.9% | 0% |
| `routed_down` | int4_g128 | 36 | 2,717,908,992 | 0.1122 | L3 (0.1264) | 4.40x | 11.6% | 0% |
| `routed_down` | mxfp4 | 36 | 2,717,908,992 | 0.1181 | L0 (0.135) | 4.63x | 0% | 0% |
| `shared_expert` | nvfp4 | 72 | 169,869,312 | 0.09343 | L0 (0.09694) | 3.55x | 2.83e-05% | 0% |
| `shared_expert` | fp8 | 72 | 169,869,312 | 0.02632 | L0 (0.02679) | 1.00x | 5.78% | 0% |
| `shared_expert` | int4_g128 | 72 | 169,869,312 | 0.1185 | L0 (0.2285) | 4.50x | 5.16% | 0% |
| `shared_expert` | mxfp4 | 72 | 169,869,312 | 0.1182 | L0 (0.1529) | 4.49x | 0% | 0% |
| `latent_proj` | nvfp4 | 72 | 42,467,328 | 0.09537 | L0 (0.1035) | 3.62x | 0% | 0% |
| `latent_proj` | fp8 | 72 | 42,467,328 | 0.02637 | L0 (0.02725) | 1.00x | 13.9% | 0% |
| `latent_proj` | int4_g128 | 72 | 42,467,328 | 0.1103 | L21 (0.1198) | 4.18x | 13.9% | 0% |
| `latent_proj` | mxfp4 | 72 | 42,467,328 | 0.1148 | L21 (0.1179) | 4.35x | 0% | 0% |
| `attn_q` | nvfp4 | 7 | 66,060,288 | 0.09476 | L4 (0.09489) | 3.59x | 0% | 0% |
| `attn_q` | fp8 | 7 | 66,060,288 | 0.02642 | L29 (0.02643) | 1.00x | 0% | 0% |
| `attn_q` | int4_g128 | 7 | 66,060,288 | 0.1152 | L24 (0.116) | 4.36x | 0% | 0% |
| `attn_q` | mxfp4 | 7 | 66,060,288 | 0.1168 | L34 (0.1176) | 4.42x | 0% | 0% |
| `attn_kv` | nvfp4 | 14 | 11,010,048 | 0.09475 | L34 (0.09508) | 3.59x | 0% | 0% |
| `attn_kv` | fp8 | 14 | 11,010,048 | 0.02639 | L4 (0.02644) | 1.00x | 0% | 0% |
| `attn_kv` | int4_g128 | 14 | 11,010,048 | 0.1173 | L9 (0.1229) | 4.45x | 0% | 0% |
| `attn_kv` | mxfp4 | 14 | 11,010,048 | 0.117 | L14 (0.1182) | 4.44x | 0% | 0% |
| `attn_small` | nvfp4 | 35 | 1,419,264 | 0.09492 | L19 (0.09674) | 3.59x | 0% | 0% |
| `attn_small` | fp8 | 35 | 1,419,264 | 0.02646 | L24 (0.0268) | 1.00x | 0% | 0% |
| `attn_small` | int4_g128 | 35 | 1,419,264 | 0.1148 | L29 (0.1367) | 4.34x | 0% | 0% |
| `attn_small` | mxfp4 | 35 | 1,419,264 | 0.1161 | L29 (0.1225) | 4.39x | 0% | 0% |
| `attn_gate` | nvfp4 | 7 | 49,545,216 | 0.09551 | L24 (0.09569) | 3.61x | 0% | 0% |
| `attn_gate` | fp8 | 7 | 49,545,216 | 0.02645 | L9 (0.02647) | 1.00x | 0% | 0% |
| `attn_gate` | int4_g128 | 7 | 49,545,216 | 0.1082 | L34 (0.1105) | 4.09x | 0% | 0% |
| `attn_gate` | mxfp4 | 7 | 49,545,216 | 0.1156 | L19 (0.1165) | 4.37x | 0% | 0% |
| `attn_o` | nvfp4 | 7 | 49,545,216 | 0.09502 | L4 (0.09522) | 3.59x | 0% | 0% |
| `attn_o` | fp8 | 7 | 49,545,216 | 0.02645 | L24 (0.02646) | 1.00x | 0% | 0% |
| `attn_o` | int4_g128 | 7 | 49,545,216 | 0.1127 | L29 (0.1141) | 4.26x | 0% | 0% |
| `attn_o` | mxfp4 | 7 | 49,545,216 | 0.1164 | L29 (0.1173) | 4.40x | 0% | 0% |
| `mamba_in_proj` | nvfp4 | 29 | 280,092,672 | 0.09502 | L1 (0.09968) | 3.60x | 11.1% | 0% |
| `mamba_in_proj` | fp8 | 29 | 280,092,672 | 0.02637 | L1 (0.0265) | 1.00x | 12.6% | 0% |
| `mamba_in_proj` | int4_g128 | 29 | 280,092,672 | 0.1222 | L7 (0.2096) | 4.63x | 12.3% | 0% |
| `mamba_in_proj` | mxfp4 | 29 | 280,092,672 | 0.1179 | L7 (0.1362) | 4.47x | 0% | 0% |
| `mamba_other` | nvfp4 | 29 | 47,038,464 | 0.09505 | L0 (0.0957) | 3.60x | 0% | 0% |
| `mamba_other` | fp8 | 29 | 47,038,464 | 0.02639 | L1 (0.02656) | 1.00x | 1.36% | 0% |
| `mamba_other` | int4_g128 | 29 | 47,038,464 | 0.1153 | L7 (0.2144) | 4.37x | 0.00381% | 0% |
| `mamba_other` | mxfp4 | 29 | 47,038,464 | 0.1172 | L7 (0.1422) | 4.44x | 0% | 0% |
| `mamba_out_proj` | nvfp4 | 29 | 136,839,168 | 0.09277 | L20 (0.09501) | 3.55x | 1.25% | 0% |
| `mamba_out_proj` | fp8 | 29 | 136,839,168 | 0.0261 | L0 (0.02661) | 1.00x | 0% | 0% |
| `mamba_out_proj` | int4_g128 | 29 | 136,839,168 | 0.1134 | L3 (0.1318) | 4.35x | 9.63% | 0% |
| `mamba_out_proj` | mxfp4 | 29 | 136,839,168 | 0.1178 | L3 (0.1216) | 4.51x | 0% | 0% |
| `router` | nvfp4 | 36 | 14,155,776 | 0.09414 | L1 (0.09759) | 3.58x | 0% | 0% |
| `router` | fp8 | 36 | 14,155,776 | 0.02633 | L10 (0.02698) | 1.00x | 0% | 0% |
| `router` | int4_g128 | 36 | 14,155,776 | 0.133 | L7 (0.2279) | 5.05x | 0% | 0% |
| `router` | mxfp4 | 36 | 14,155,776 | 0.1197 | L7 (0.1431) | 4.55x | 0% | 0% |
| `embedding` | nvfp4 | 1 | 233,373,696 | 0.09514 | n/a (0.09514) | 3.60x | 0% | 0% |
| `embedding` | fp8 | 1 | 233,373,696 | 0.02645 | n/a (0.02645) | 1.00x | 0% | 0% |
| `embedding` | int4_g128 | 1 | 233,373,696 | 0.1109 | n/a (0.1109) | 4.19x | 0% | 0% |
| `embedding` | mxfp4 | 1 | 233,373,696 | 0.1144 | n/a (0.1144) | 4.33x | 0% | 0% |
| `lm_head` | nvfp4 | 1 | 233,373,696 | 0.09701 | n/a (0.09701) | 3.64x | 0% | 0% |
| `lm_head` | fp8 | 1 | 233,373,696 | 0.02662 | n/a (0.02662) | 1.00x | 0% | 0% |
| `lm_head` | int4_g128 | 1 | 233,373,696 | 0.09642 | n/a (0.09642) | 3.62x | 0% | 0% |
| `lm_head` | mxfp4 | 1 | 233,373,696 | 0.113 | n/a (0.113) | 4.24x | 0% | 0% |

## Distribution shape per class

| class | tensors | params | mean excess kurtosis | mean frac > 6 RMS | worst max/RMS (layer) | worst channel RMS range | worst g16 amax max/median | worst g32 amax max/median |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| `routed_up` | 36 | 2,717,908,992 | 103.8 | 0.0001106 | 138.1 (L5) | 1.367e+04 | 4364 | 3879 |
| `routed_down` | 36 | 2,717,908,992 | 175.8 | 9.07e-05 | 168.4 (L0) | 399.9 | 1516 | 735.7 |
| `shared_expert` | 72 | 169,869,312 | 13.8 | 0.0003282 | 108.1 (L0) | 5540 | 4270 | 3871 |
| `latent_proj` | 72 | 42,467,328 | 0.2888 | 3.775e-05 | 18.25 (L12) | 30.73 | 9.094 | 8.051 |
| `attn_q` | 7 | 66,060,288 | 3.219 | 0.0005168 | 23.41 (L4) | 30.17 | 12.22 | 10.65 |
| `attn_kv` | 14 | 11,010,048 | 1.502 | 5.531e-05 | 37.69 (L19) | 9.058 | 18.97 | 16.56 |
| `attn_small` | 35 | 1,419,264 | 2.421 | 0.0002255 | 12.31 (L14) | 29.89 | 7.365 | 6.415 |
| `attn_gate` | 7 | 49,545,216 | 0.1291 | 3.976e-06 | 43.52 (L34) | 5.341 | 22.22 | 19.74 |
| `attn_o` | 7 | 49,545,216 | 0.6097 | 3.371e-05 | 38.8 (L29) | 5.701 | 19.23 | 16.88 |
| `mamba_in_proj` | 29 | 280,092,672 | 1763 | 0.001439 | 646.1 (L7) | 7.037e+04 | 1.934e+05 | 1.71e+05 |
| `mamba_other` | 29 | 47,038,464 | 84.53 | 0.0002516 | 183.2 (L7) | 592.9 | 316.4 | 277.2 |
| `mamba_out_proj` | 29 | 136,839,168 | 355.9 | 0.0005265 | 282 (L7) | 48.76 | 2.72e+04 | 2.407e+04 |
| `router` | 36 | 14,155,776 | 2.496 | 0.0004648 | 25.41 (L3) | 16.57 | 14.04 | 11.81 |
| `embedding` | 1 | 233,373,696 | 0.4111 | 2.442e-07 | 14.52 (n/a) | 20.43 | 7.195 | 6.306 |
| `lm_head` | 1 | 233,373,696 | 0.809 | 8.476e-06 | 13.64 (n/a) | 19.99 | 7.942 | 7.211 |
