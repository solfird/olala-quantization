# vendor/: OVHcloud's HF modeling code, vendored verbatim

`modeling_olala.py` and `configuration_olala.py` are copied byte for byte from `recon/src/hf/` (the files of the
model repository `OVHcloud/Olala-v1.0-mini-7A1B`, the model's official `trust_remote_code` files), with the
`LICENSE` and `NOTICE` of that repository. They are **third-party code, Apache-2.0, Copyright 2026 OVH LLM SAS
(trading as OVHai LLM)**. Nothing here is ours and nothing was edited.

| File | sha256 |
|---|---|
| `modeling_olala.py` | `357c01b34490ee8650a42a8acdd38bc186bd35863edbbeb4e2bf0746e9b31bcf` |
| `config.json` (the model's real configuration; `tiny_model.py` shrinks its sizes) | `f2668c0a369eeb2285e21edb46dd6307c32d55aaaefa985a1abf0eeb73dbd8e3` |
| `configuration_olala.py` | `0fd6af3cbd2d08031919da5bea529743ba0aca814ec7a71514cb2870b4edeaeb` |

Why vendored: G1-lite has to run the model's own blocks (`OlalaMonoBlock`) to see the inputs of the routed experts, and
that code must not drift from the one that produced the bf16 checkpoint. `olala_model.py` refuses to run when the
`modeling_olala.py` next to the real weights differs from this copy (`check_vendored_matches`).

How it is allowed to run (the hard rules of the G1-lite task): only with a TINY random configuration on the Mac (CPU,
`OLALA_TORCH_MOE=1 OLALA_TORCH_MAMBA3=1 HF_HUB_OFFLINE=1`), and on the pod with the real configuration under the same
environment variables. It is read in `recon/` (no network or subprocess calls except optional CUDA libraries:
`flash_attn`, `mamba_ssm`, `scattermoe`, `cut_cross_entropy` are optional imports that fall back to PyTorch) and is
never given weights from the network. The OVH vLLM fork is never imported.
