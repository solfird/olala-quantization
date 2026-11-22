# coding=utf-8
"""PyTorch Olala model."""

from typing import Optional, Tuple, Union
from dataclasses import dataclass
import inspect
from contextlib import nullcontext

import math
import os
from einops import rearrange
import torch
import torch.nn.functional as F
import torch.nn as nn

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin
from transformers.utils import ModelOutput, logging

# LOAD ORDER MATTERS, keep this above the mamba_ssm import below.
# mamba_ssm pulls in tilelang, which ships libcudart_stub.so exporting the cudart
# symbols itself. Any CUDA extension dlopened AFTER it binds its cudart calls to
# that stub, whose own fallback resolution then fails -> "TileLang Error:
# libcudart symbols not found globally" + abort() (kills the process at the first
# flash-attn call, no Python traceback). Loading flash-attn's .so first makes it
# bind to the real libcudart.
try:
    import flash_attn_interface  # noqa: F401  (FA3, imports flash_attn_3._C)
except ImportError:
    try:
        import flash_attn  # noqa: F401  (FA2)
    except ImportError:
        pass

# Mamba-3 MIMO chunk kernel, from state-spaces/mamba (TileLang). It is the
# same kernel Megatron-LM trains with. Two properties matter at the call sites:
#  - it takes RAW angles and applies the angle cumsum itself, so nothing must
#    pre-cumsum them;
#  - it returns a bare tensor when return_state=False, so the wrapper below
#    normalizes to the (y, state) tuple this file unpacks.
# Without it (no mamba_ssm, no GPU), the mixer runs _mamba3_mimo_torch below:
# the same recurrence in plain PyTorch, much slower, backward via autograd.
# OLALA_TORCH_MAMBA3=1 forces that fallback (testing, CPU debugging).
_FORCE_TORCH_MAMBA3 = os.environ.get("OLALA_TORCH_MAMBA3", "0") == "1"
try:
    if _FORCE_TORCH_MAMBA3:
        raise ImportError("OLALA_TORCH_MAMBA3=1")
    from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo as _mamba3_mimo
except ImportError as _e:
    _mamba3_mimo = None
    print(f"mamba3: no TileLang chunk kernel ({_e}) -> pure-PyTorch fallback (slow)")


def mamba3_tilelang(*, return_state=False, **kwargs):
    impl = _mamba3_mimo if _mamba3_mimo is not None else _mamba3_mimo_torch
    out = impl(return_state=return_state, **kwargs)
    if not return_state:
        return out, None
    # mamba3_mimo returns a FLAT 5-tuple
    # (Out, Final_Angle, Final_SSM_State, Final_K, Final_V), but this file
    # unpacks `y, kernel_state` and then splits kernel_state into four states.
    # Returning `out` unchanged raised "too many values to unpack (expected 2)"
    # on every prefill that passed a cache -- i.e. the whole cached path was
    # dead.
    y, *state = out
    return y, tuple(state)


# Decode kernels. Kept in their own try so a missing CuteDSL only switches
# cached decode to the PyTorch step (_mamba3_step_torch) instead of failing.
try:
    if _FORCE_TORCH_MAMBA3:
        raise ImportError("OLALA_TORCH_MAMBA3=1")
    from mamba_ssm.ops.cute.mamba3.mamba3_step_fn import (
        mamba3_step_fn as _mamba3_step_fn_impl,
    )
    from mamba_ssm.ops.triton.mamba3.mamba3_mimo_rotary_step import (
        apply_rotary_qk_inference_fwd,
    )

    def mamba3_step_fn(*a, **kw):
        # The CuteDSL step kernel imports cleanly but can still fail at CALL
        # time on a cutlass-dsl / tvm-ffi version mismatch (observed:
        # nvidia-cutlass-dsl 4.6.0 with apache-tvm-ffi 0.1.9 ->
        # "make_kwargs_wrapper() got an unexpected keyword argument
        # 'map_dataclass_to_tuple'"). Turn that into something actionable
        # rather than an opaque cutlass traceback. Deliberately NOT a silent
        # fallback to recompute: that costs ~5000x per token and would look
        # like the model is merely slow.
        try:
            return _mamba3_step_fn_impl(*a, **kw)
        except TypeError as e:
            raise RuntimeError(
                "mamba3 cached decode unavailable: the CuteDSL step kernel "
                f"failed to launch ({e}). Align the nvidia-cutlass-dsl and "
                "apache-tvm-ffi versions, or set OLALA_TORCH_MAMBA3=1 to use "
                "the (slower) PyTorch implementation."
            ) from e

    print("mamba3: chunk + decode kernels (cached generation available)")
except ImportError as _e:
    mamba3_step_fn = None
    apply_rotary_qk_inference_fwd = None
    print(f"mamba3: no CuteDSL decode kernel ({_e}) -> PyTorch decode step")

# Routed experts. scattermoe (Triton) is the fast path. Without it the MoE runs
# TorchMoEMLP below: same parameters (experts.weight / output_experts.weight,
# both (E, out, in)), so a checkpoint loads into either; slower, one matmul pair
# per active expert. OLALA_TORCH_MOE=1 forces the fallback.
try:
    if os.environ.get("OLALA_TORCH_MOE", "0") == "1":
        raise ImportError("OLALA_TORCH_MOE=1")
    import scattermoe
    from scattermoe.mlp import MLP as ScatterMoE
    scattermoe.kernels.ops.ALLOW_TF32 = False
    MOE_IMPL = "scattermoe"
except ImportError as _e:
    ScatterMoE = None
    MOE_IMPL = "torch"
    print(f"moe: no scattermoe ({_e}) -> pure-PyTorch experts (slow)")

from .configuration_olala import OlalaConfig

logger = logging.get_logger(__name__)

try:
    from cut_cross_entropy import linear_cross_entropy
except ImportError:
    linear_cross_entropy = None

# attention backend selection: FA3 > FA2 > flex > eager, or force one with
# OLALA_ATTN_IMPL=fa3|fa2|flex|eager.
ATTN_IMPL = "eager"
_ATTN_FORCE = os.environ.get("OLALA_ATTN_IMPL", "")
assert _ATTN_FORCE in ("", "fa3", "fa2", "flex", "eager"), f"bad OLALA_ATTN_IMPL={_ATTN_FORCE!r}"
try:
    if _ATTN_FORCE not in ("", "fa3"):
        raise ImportError(f"OLALA_ATTN_IMPL={_ATTN_FORCE}")
    import flash_attn_interface # FA3
    flash_attn_func = flash_attn_interface.flash_attn_func
    flash_attn_varlen_func = flash_attn_interface.flash_attn_varlen_func
    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
    if not _flash_supports_window_size:
        raise ImportError("flash_attn_func does not support window_size parameter. Please update to more recent flash_attn version")
    ATTN_IMPL = "fa3"
except ImportError:
    try:
        if _ATTN_FORCE not in ("", "fa2"):
            raise ImportError(f"OLALA_ATTN_IMPL={_ATTN_FORCE}")
        from flash_attn import flash_attn_func, flash_attn_varlen_func # FA2
        ATTN_IMPL = "fa2"
    except ImportError:
        try:
            if _ATTN_FORCE not in ("", "flex"):
                raise ImportError(f"OLALA_ATTN_IMPL={_ATTN_FORCE}")
            from torch.nn.attention.flex_attention import flex_attention, create_block_mask
            flex_attention = torch.compile(flex_attention)
            ATTN_IMPL = "flex"
        except Exception:
            if _ATTN_FORCE != "eager":
                logger.warning_once(
                "Neither Flash Attention nor Flex Attention is installed, using eager attention implementation. "
                "For better performance, consider installing flash-attention (https://github.com/Dao-AILab/flash-attention)."
                )

print(f"Using attention implementation: {ATTN_IMPL}")

logger.info(f"Using attention implementation: {ATTN_IMPL}")


# Layer types this file implements. Older revisions also carried Gated
# DeltaNet ('g'), Mamba-2 ('2') and three other attention variants
# ('v', 'w', 't'); no released checkpoint uses them, so they were removed.
SUPPORTED_LAYER_TYPES = "MV"  # M: Mamba-3 MIMO, V: differential TPA attention


def _check_supported_config(config):
    """Fail at construction on anything this file no longer implements, rather
    than silently building a different model."""
    bad = sorted(set(config.layers_config) - set(SUPPORTED_LAYER_TYPES))
    if bad:
        raise ValueError(
            f"layers_config={config.layers_config!r} uses layer type(s) {bad}; this "
            f"modeling file only implements {list(SUPPORTED_LAYER_TYPES)} "
            "(M = Mamba-3 MIMO, V = differential TPA attention). Gated DeltaNet, "
            "Mamba-2 and the other attention variants live in older olala-v1 revisions.")
    bad_mlp = sorted(set(config.layers_mlp_config) - set("md"))
    if bad_mlp:
        raise ValueError(f"layers_mlp_config uses unknown MLP type(s) {bad_mlp}; expected 'm' or 'd'.")
    dense_used = ("d" in config.layers_mlp_config) or (not config.layers_mlp_config and not config.moe)
    if dense_used and config.mlp_type != "simple":
        raise ValueError(f"mlp_type={config.mlp_type!r} is no longer implemented; only 'simple' is.")
    if getattr(config, "token_conv1d_attn", False):
        raise ValueError("token_conv1d_attn=True is no longer implemented.")
    if config.use_value_embedding:
        bad_ve = sorted({t for t, ve in zip(config.layers_config, config.layers_ve_config)
                         if ve == "1" and t != "M"})
        if bad_ve:
            raise ValueError(f"value embeddings are only implemented for 'M' layers, got {bad_ve}.")


# Parameter-free placeholders used for pruned/skipped MoE blocks.  When
# config.skip_moe_layers includes a layer's index, OlalaMonoBlock installs
# these so the MLP phase becomes a no-op (residual passes through unchanged).
# The analogous skip_mixer_layers config skips the *mixer* phase (Mamba/Attn)
# entirely — the residual passes through unchanged on that phase.
class _NullMlp(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


class _NullGeodesic(nn.Module):
    def forward(self, residual, g):
        return residual


class _NullMixer(nn.Module):
    """Stand-in for a OlalaMonoBlock mixer when the layer is in
    ``config.skip_mixer_layers``. Returns (zeros_like_input, None, None)."""
    def forward(self, hidden_states, **kwargs):
        return torch.zeros_like(hidden_states), None, None


class OlalaHeadWiseRMSNorm(nn.Module):
    def __init__(self, n_heads, d_head, eps=1e-6, zero_centered_gamma=False):
        super().__init__()
        self.rms = nn.RMSNorm(d_head, eps=eps, elementwise_affine=False)
        self.weight = nn.Parameter(torch.zeros(n_heads, d_head)) if zero_centered_gamma else nn.Parameter(torch.ones(n_heads, d_head))
        self.zero_centered_gamma = zero_centered_gamma

    def forward(self, hidden_states):
        B, L, H, D = hidden_states.shape
        y = self.rms(hidden_states) * (1.0 + self.weight.view(1, 1, H, D)) if self.zero_centered_gamma else self.rms(hidden_states) * self.weight.view(1, 1, H, D)
        return y.view(B, L, H, D)

class OlalaNorm(nn.Module):
    def __init__(self, config: OlalaConfig, hidden_size: int):
        super().__init__()
        if config.normalization_type == "rmsnorm":
            self.norm = OlalaRMSNorm(hidden_size, eps=config.norm_epsilon, zero_centered_gamma=config.zero_centered_gamma)
        else:
            raise ValueError(f"Unknown normalization_type: {config.normalization_type}")

    def forward(self, hidden_states):
        return self.norm(hidden_states)

class OlalaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, zero_centered_gamma=False):
        super().__init__()
        self.rms = nn.RMSNorm(hidden_size, eps=eps, elementwise_affine=False)
        self.weight = nn.Parameter(torch.zeros(hidden_size)) if zero_centered_gamma else nn.Parameter(torch.ones(hidden_size))
        self.zero_centered_gamma = zero_centered_gamma

    def forward(self, hidden_states):
        y = self.rms(hidden_states) * (1.0 + self.weight) if self.zero_centered_gamma else self.rms(hidden_states) * self.weight
        return y

class OlalaLinear(nn.Linear):
    def __init__(self, config: OlalaConfig, in_features, out_features, bias=False, cosnet=True, **kwargs):
        super().__init__(in_features, out_features, bias, **kwargs)
        self.config = config

        if self.config.cosnet and cosnet:
            self.cosnet_branch = OlalaCosNetBranch(
                in_features=in_features,
                out_features=out_features,
                rank=config.cosnet_rank,
            )

    def forward(self, x):
        out = super().forward(x)
        if self.config.cosnet:
            out = out + self.cosnet_branch(x)
        return out

class OlalaCosNetBranch(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 128,
        wmin: float = 0.8,
        wmax: float = 1.2,
        phase_std: float = 0.1,
        up_scale: float = 0.01,
    ):
        super().__init__()

        self.rank = rank

        self.down = nn.Linear(in_features, rank, bias=False)
        self.mix = nn.Linear(rank, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        self.up.weight.dim_factor = min(in_features, out_features)
        self.mix.weight.dim_factor = min(in_features, out_features)

        self.omega1 = nn.Parameter(torch.ones(rank))
        self.phi1 = nn.Parameter(torch.ones(rank))
        self.omega2 = nn.Parameter(torch.ones(rank))
        self.phi2 = nn.Parameter(torch.ones(rank))

    def forward(self, x):
        h = self.down(x)
        h = torch.cos(h * self.omega1 + self.phi1)
        h = self.mix(h)
        h = torch.cos(h * self.omega2 + self.phi2)
        return self.up(h)

class HybridOlalaDynamicCache(DynamicCache):
    """
    Cache for the hybrid model: a growing KV cache (plus the token-shift
    last K/V) for the attention layers, and the four constant-size Mamba-3
    states (angle, ssm, k, v) for the Mamba-3 layers.
    KV tensors are (batch, seq_len, num_kv_heads, head_dim).
    """
    def __init__(self, config: OlalaConfig):
        super().__init__()
        self.config = config
        # If layer_idx_map is set (pruned-block variant), blocks may index
        # into these caches at their ORIGINAL layer_idx, which can exceed
        # len(layers_config). Size the per-layer lists by the largest
        # original index + 1 so writes don't go out of bounds.
        _map = getattr(config, "layer_idx_map", None)
        cache_n = max(_map) + 1 if _map else len(config.layers_config)
        # attention
        self._key_cache = {}
        self._value_cache = {}
        # attention - kv shift
        self._kv_shift_last_k = [None for _ in range(cache_n)] # (B, H_kv, D)
        self._kv_shift_last_v = [None for _ in range(cache_n)] # (B, H_kv, D)
        # mamba3
        self.mamba3_angle_states = [None for _ in range(cache_n)]
        self.mamba3_ssm_states = [None for _ in range(cache_n)]
        self.mamba3_k_states = [None for _ in range(cache_n)]
        self.mamba3_v_states = [None for _ in range(cache_n)]

        # Build per-position auxiliary caches keyed by ORIGINAL layer_idx
        # when a layer_idx_map is present. Otherwise use the position
        # sequence as before.
        _origs = _map if _map else list(range(len(config.layers_config)))
        for orig in _origs:
            self._key_cache[orig] = None
            self._value_cache[orig] = None

        self.past_length = [0 for _ in range(cache_n)]

    def update(
        self,
        k: torch.Tensor, # (B, L, h, D)
        v: torch.Tensor, # (B, L, h, D)
        layer_idx: int,
    ):
        added_len = k.size(1)
        # grab cache
        k_cache = self._key_cache[layer_idx]
        v_cache = self._value_cache[layer_idx]
        if k_cache is None:
            k_cache = k
            v_cache = v
        else:
            k_cache = torch.cat([k_cache, k], dim=1)
            v_cache = torch.cat([v_cache, v], dim=1)
        # save cache
        self._key_cache[layer_idx] = k_cache
        self._value_cache[layer_idx] = v_cache
        # update cache length
        self.past_length[layer_idx] += added_len
        return k_cache, v_cache

    # kv shift
    def get_last_kv(self, layer_idx):
        return self._kv_shift_last_k[layer_idx], self._kv_shift_last_v[layer_idx]

    def set_last_kv(self, layer_idx, k_last, v_last):
        self._kv_shift_last_k[layer_idx] = k_last
        self._kv_shift_last_v[layer_idx] = v_last

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        raise NotImplementedError("HybridOlalaDynamicCache does not have a legacy cache equivalent.")

    @classmethod
    def from_legacy_cache(cls, cache_params: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
        raise NotImplementedError("HybridOlalaDynamicCache does not have a legacy cache equivalent.")

class OlalaRotaryEmbedding(torch.nn.Module):
    def __init__(self, config: OlalaConfig, head_dim: int, theta: float):
        super().__init__()
        self.config = config

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, position_ids):
        max_pos = self.config.max_position_embeddings
        if max_pos > self.seq_len_cached:
            self.seq_len_cached = max(2 * max_pos, 16)
            t = torch.arange(self.seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            self.cos_cached = freqs.cos().to(torch.bfloat16)
            self.sin_cached = freqs.sin().to(torch.bfloat16)

        cos = self.cos_cached[position_ids] # (B, T, head_dim/2)
        sin = self.sin_cached[position_ids]
        cos = cos[..., None, :]       # (B, T, 1, head_dim/2), broadcasts over heads
        sin = sin[..., None, :]

        return cos, sin

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2 # head dim
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

def apply_p_rotary_emb(x, cos, sin, p: float = 0.75):
    """Partial RoPE: rotate only the top p fraction of (half) dims; rest are identity."""
    assert x.ndim == 4 and 0.0 <= p <= 1.0  # x: (B, L, H, D)
    d = x.shape[3] // 2                     # half-dim per your layout
    rope_d = int(d * p)

    x1, x2 = x[..., :d], x[..., d:]

    if rope_d > 0:
        y1_head = x1[..., :rope_d] * cos[..., :rope_d] + x2[..., :rope_d] * sin[..., :rope_d]
        y2_head = x1[..., :rope_d] * (-sin[..., :rope_d]) + x2[..., :rope_d] * cos[..., :rope_d]
        y1 = torch.cat([y1_head, x1[..., rope_d:]], dim=-1)
        y2 = torch.cat([y2_head, x2[..., rope_d:]], dim=-1)
    else:
        y1, y2 = x1, x2

    return torch.cat([y1, y2], dim=-1).type_as(x)

# heavily adapated from Gemma3
def eager_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool = True,
    window_size: Optional[Tuple[int, int]] = None,
    softcap: Optional[float] = None,
    softmax_scale: Optional[float] = None,    
    **kwargs,
) -> torch.Tensor:
    if softmax_scale is None:
        softmax_scale = query.size(3)**-0.5
    if window_size == (-1, 0):
        window_size = None

    out_dtype = query.dtype
    # fp32 throughout, like flash-attn's fp32 accumulation
    query = query.transpose(1, 2).float() # (B, H, L, D)
    key = key.transpose(1, 2).float() # (B, h, L, D)
    value = value.transpose(1, 2).float() # (B, h, L, D)

    key = key.repeat_interleave(query.size(1) // key.size(1), dim=1)
    value = value.repeat_interleave(query.size(1) // value.size(1), dim=1)

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * softmax_scale

    if softcap is not None and softcap > 0.:
        attn_weights = torch.tanh(attn_weights / softcap) * softcap

    if causal or (window_size is not None):
        Lq = query.size(2)
        Lk = key.size(2)
        past = max(Lk - Lq, 0)
        i = torch.arange(Lq, device=attn_weights.device).unsqueeze(1) + past # [Lq,1]
        j = torch.arange(Lk, device=attn_weights.device).unsqueeze(0) # [1,Lk]

        allowed = torch.ones((Lq, Lk), dtype=torch.bool, device=attn_weights.device)
        if causal:
            allowed &= (j <= i) # prevent attending to future positions
        if window_size is not None:
            w_left, w_right = window_size
            # treat None as "no limit" on that side
            if w_left is None:
                w_left = Lk
            if w_right is None:
                w_right = Lk
            allowed &= (j >= i - w_left) & (j <= i + w_right)
        # broadcast [Lq,Lk] -> [B, H, Lq, Lk]
        attn_weights = attn_weights.masked_fill(~allowed, float("-inf"))

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous().to(out_dtype)
    return attn_output

class OlalaDifferentialTensorProductAttentionV2(nn.Module):
    """
    differential attention V2 + TPA
    """

    def __init__(self, config: OlalaConfig, layer_idx: Optional[int], use_ve: bool = False, **kwargs):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.num_attention_heads = config.num_attention_heads
        self.num_signal_heads = config.num_signal_heads_diff if config.num_signal_heads_diff else self.num_attention_heads//2
        self.num_noise_heads = self.num_attention_heads - self.num_signal_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.rank = config.tpa_rank
        self.qk_norm = config.qk_norm
        self.softcap = config.softcap_attn
        self.scalable_softmax = config.scalable_softmax

        assert self.num_signal_heads % self.num_noise_heads == 0, "number of signal heads must be a multiple of number of noise heads."
        self.snr = self.num_signal_heads // self.num_noise_heads
        self.num_key_value_heads = self.num_noise_heads

        self.c_q = OlalaLinear(config, self.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.W_A_k = OlalaLinear(config, self.hidden_size, self.num_key_value_heads * self.rank, bias=False)
        self.W_A_v = OlalaLinear(config, self.hidden_size, self.num_key_value_heads * self.rank, bias=False)
        self.W_B_k = OlalaLinear(config, self.hidden_size, self.rank * self.head_dim, bias=False)
        self.W_B_v = OlalaLinear(config, self.hidden_size, self.rank * self.head_dim, bias=False)

        if use_ve:
            self.ve_scalars = nn.Parameter(torch.zeros(self.num_noise_heads, self.head_dim, dtype=torch.float32))

        if self.config.token_shift_attn:
            self.shift_proj_k = OlalaLinear(config, self.hidden_size, self.num_key_value_heads, bias=False)
            self.shift_proj_v = OlalaLinear(config, self.hidden_size, self.num_key_value_heads, bias=False)
            if self.config.scalar_proj_as_hidden_matrix:    
                self.shift_proj_k.is_scalar_weight = True
                self.shift_proj_v.is_scalar_weight = True

        if self.qk_norm:
            self.q_norm = OlalaNorm(config, self.head_dim)
            self.k_norm = OlalaNorm(config, self.head_dim)

        if self.scalable_softmax:
            self.softmax_scaler = nn.Parameter(torch.ones(self.num_attention_heads, dtype=torch.float32))

        self.lambda_proj = OlalaLinear(config, config.hidden_size, self.num_noise_heads, bias=False)

        if ATTN_IMPL == "flex":
            # score mod (for softcap)
            def score_mod(score, batch_idx, head_idx, q_idx, kv_idx):
                if self.config.softcap_attn > 0.:
                    score = self.config.softcap_attn * torch.tanh(score / self.config.softcap_attn)
                return score
            self.score_mod = score_mod

        if self.config.rope_theta > 0.0 and self.config.rope_type != "":
            self.rotary_emb = OlalaRotaryEmbedding(config, head_dim=self.head_dim, theta=config.rope_theta)
        else:
            self.rotary_emb = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[HybridOlalaDynamicCache] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        ve=None,
        **kwargs,
    ):
        b, q_len, _ = hidden_states.shape

        # Q, K, V projections.
        query_states = self.c_q(hidden_states).view(b, q_len, self.num_attention_heads, self.head_dim)
        A_k = self.W_A_k(hidden_states).view(b, q_len, self.num_key_value_heads, self.rank)
        A_v = self.W_A_v(hidden_states).view(b, q_len, self.num_key_value_heads, self.rank)
        B_k = self.W_B_k(hidden_states).view(b, q_len, self.rank, self.head_dim)
        B_v = self.W_B_v(hidden_states).view(b, q_len, self.rank, self.head_dim)
        # (rope done on query_states and B_k)
        A_k = A_k.view(b * q_len, self.num_key_value_heads, self.rank)
        A_v = A_v.view(b * q_len, self.num_key_value_heads, self.rank)
        B_k = B_k.view(b * q_len, self.rank, self.head_dim)
        B_v = B_v.view(b * q_len, self.rank, self.head_dim)
        key_states = torch.bmm(A_k, B_k).div_(self.rank).view(b, q_len, self.num_key_value_heads, self.head_dim)
        value_states = torch.bmm(A_v, B_v).div_(self.rank).view(b, q_len, self.num_key_value_heads, self.head_dim)

        # value embeddings
        if ve is not None:
            value_states = value_states + self.ve_scalars * ve.view_as(value_states)

        # token-shift.
        if self.config.token_shift_attn:
            alpha_k = torch.sigmoid(self.shift_proj_k(hidden_states).float()).float().to(key_states.dtype).unsqueeze(-1) # (B, L, Hkv, 1)
            alpha_v = torch.sigmoid(self.shift_proj_v(hidden_states).float()).float().to(value_states.dtype).unsqueeze(-1) # (B, L, Hkv//2, 1)

            if cache_params is not None:
                k_last, v_last = cache_params.get_last_kv(self.layer_idx)
                B, L = key_states.shape[:2]

                if L == 1:
                    # decode step
                    if k_last is None:
                        k_prev = torch.zeros_like(key_states)      # (B, 1, H, D)
                        v_prev = torch.zeros_like(value_states)    # (B, 1, H, D)
                    else:
                        k_prev, v_prev = k_last, v_last            # (B, 1, H, D)
                else:
                    # prefill step: first token uses cached last, rest shift within the chunk
                    first_k = k_last if k_last is not None else torch.zeros_like(key_states[:, :1])
                    first_v = v_last if v_last is not None else torch.zeros_like(value_states[:, :1])
                    k_prev = torch.cat([first_k, key_states[:, :-1]], dim=1)   # (B, L, H, D)
                    v_prev = torch.cat([first_v, value_states[:, :-1]], dim=1) # (B, L, H, D)

                # keep caching the *raw* last KV from this chunk (matches the no-cache path)
                cache_params.set_last_kv(self.layer_idx, key_states[:, -1:], value_states[:, -1:])
            else:
                k_prev = F.pad(key_states, (0, 0, 0, 0, 1, 0))[:, :-1] # (B, L, H, D)
                v_prev = F.pad(value_states, (0, 0, 0, 0, 1, 0))[:, :-1] # (B, L, H, D)

            if position_ids is not None:
                # first token of each doc has pos==0
                doc_start = (position_ids == 0) # (B, L) bool
            else:
                B, L = hidden_states.shape[:2]
                doc_start = torch.zeros(B, L, dtype=torch.bool, device=hidden_states.device)
                doc_start[:, 0] = True
            m = doc_start.unsqueeze(-1).unsqueeze(-1) # (B, L, 1, 1) bool

            # zero the previous contribution at boundaries
            k_prev  = k_prev.masked_fill(m, 0)
            v_prev  = v_prev.masked_fill(m, 0)
            alpha_k = alpha_k.masked_fill(m, 0)
            alpha_v = alpha_v.masked_fill(m, 0)

            key_states = alpha_k * k_prev + (1 - alpha_k) * key_states
            value_states = alpha_v * v_prev + (1 - alpha_v) * value_states

        # QK-norm.
        if self.qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        wsize = self.config.slw_wsize

        #rope
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(hidden_states, position_ids)
            if self.config.rope_type == "rope":
                query_states = apply_rotary_emb(query_states, cos, sin)
                key_states = apply_rotary_emb(key_states, cos, sin)
            elif self.config.rope_type == "p-rope":
                query_states = apply_p_rotary_emb(query_states, cos, sin)
                key_states = apply_p_rotary_emb(key_states, cos, sin)
            else:
                raise ValueError(f"Unknow rope type : {self.config.rope_type}")
        # scalable softmax.
        if self.scalable_softmax:
            # scalable-softmax (https://arxiv.org/abs/2501.19399): multiply q by s*log(n)
            pos = position_ids.to(torch.float32)
            if pos.dim() == 1:
                pos = pos.unsqueeze(0)
            pos = pos.unsqueeze(-1).unsqueeze(-1) + 1.0
            log_pos = pos.log() if wsize <= 0 else torch.clamp_max(pos, wsize).log()
            query_states = (self.softmax_scaler.view(1, 1, -1, 1) * log_pos) * query_states
            # TODO: caching mechanism for log_pos

        # KV-cache.
        if cache_params is not None:
            key_states, value_states = cache_params.update(key_states, value_states, self.layer_idx)

        # attention computation.
        # Row boundaries from a packed batch need the varlen path just as
        # document boundaries do; eager/flex have none, which is why
        # OlalaModel.forward only packs under fa2/fa3.
        _varlen_attn = self.config.intra_doc_masking or kwargs.get("unpadded", False)
        # flash-attn and flex kernels are CUDA-only: eager everywhere else
        attn_impl = ATTN_IMPL if query_states.is_cuda else "eager"
        if attn_impl == "eager":
            assert not _varlen_attn
            attention_interface = lambda q, k, v, wsize, **kw: eager_attention_forward(q, k, v, window_size=(wsize, 0), **kw)
        elif attn_impl == "flex":
            _w = self.config.max_position_embeddings if wsize == -1 else wsize
            def attention_interface(q, k, v, softmax_scale, **kw):
                # with a KV cache the queries are the LAST q_len key positions
                off = k.size(1) - q.size(1)
                def mask_mod(b, h, q_idx, kv_idx):
                    qi = q_idx + off
                    return (qi >= kv_idx) & (qi - kv_idx <= _w)
                block_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=q.size(1), KV_LEN=k.size(1), device=q.device)
                return flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask,
                                      score_mod=self.score_mod, scale=softmax_scale,
                                      enable_gqa=self.num_attention_heads > self.num_key_value_heads).transpose(1, 2)
        elif attn_impl == "fa2":
            if not _varlen_attn:
                attention_interface = lambda q, k, v, wsize, **kw: flash_attn_func(q, k, v, window_size=(wsize, 0), **kw)
            else:
                attention_interface = lambda q, k, v, wsize, **kw: flash_attn_varlen_func(q[0], k[0], v[0], cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens, max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen, window_size=(wsize, 0), **kw).unsqueeze(0)
        elif attn_impl == "fa3":
            if not _varlen_attn:
                attention_interface = lambda q, k, v, wsize, **kw: flash_attn_func(q, k, v, window_size=(wsize, 0), **kw)
            else:
                attention_interface = lambda q, k, v, wsize, **kw: flash_attn_varlen_func(q[0], k[0], v[0], cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens, max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen, window_size=(wsize, 0), **kw).unsqueeze(0)
        else:
            raise ValueError(f"Unknown ATTN_IMPL: {ATTN_IMPL}")

        # num_heads = num_signal_heads + num_noise_heads
        # num_kv_heads = num_signal_heads // (snr * gqa)
        # where snr = num_signal_heads // num_noise_heads (typically 3)
        #       gqa = num_heads // num_kv_heads (typically 1)
        # identity : snr+1 = num_heads/num_noise_heads

        # query_states: (B, L, num_heads, D)
        # key_states: (B, L, num_kv_heads, D)
        # value_states: (B, L, num_kv_heads, D)

        attn_output = attention_interface(
            query_states.bfloat16(),
            key_states.bfloat16(),
            value_states.bfloat16(),
            causal=True,
            wsize=wsize,
            softcap=self.config.softcap_attn,
            softmax_scale=None if not self.config.use_completed_p else 1/self.head_dim,
        ) # (B, L, H, D)
        if self.config.xsa:
            v_self = value_states.repeat_interleave(self.num_attention_heads // value_states.size(2), dim=2)
            v_self = F.normalize(v_self.float(), dim=-1, eps=1e-6).to(attn_output.dtype)
            attn_output = attn_output - (attn_output * v_self).sum(dim=-1, keepdim=True) * v_self

        attn_output = attn_output.reshape(attn_output.size(0), attn_output.size(1), -1, self.num_attention_heads//self.num_noise_heads, self.head_dim) # (B, L, num_noise_heads, snr+1, D)
        attn_sig = attn_output[:, :, :, :self.snr, :] # (B, L, num_noise_heads, snr, D)
        attn_noi = attn_output[:, :, :, self.snr:self.snr+1, :] # (B, L, num_noise_heads, 1, D)

        lambda_val = self.lambda_proj(hidden_states).unsqueeze(-1).unsqueeze(-1) # (B, L, H, 1, 1)
        attn_output = attn_sig - torch.sigmoid(lambda_val) * attn_noi # (B, L, num_noise_heads, snr, D) (each noise head is broadcasted/repeated SNR times)
        attn_output = attn_output.view(attn_output.size(0), attn_output.size(1), -1, self.head_dim) # (B, L, num_signal_heads, D)

        #if cache_params is not None:
        #    cache_params.trim(self.layer_idx)

        return attn_output, None, None

# ---------------------------------------------------------------------------
# Pure-PyTorch Mamba-3 MIMO, used when the TileLang / CuteDSL kernels from
# mamba_ssm are unavailable. Same contracts as mamba3_mimo (chunked prefill)
# and mamba3_step_fn (+ apply_rotary_qk_inference_fwd) for decode, forward
# only (autograd provides the backward). Recurrence, per head, with
# u_t = sum_r k_{t,r} v_{t,r}^T and v_{t,r} = MIMO_V[r] * x_t:
#
#   h_t = a_t * h_{t-1} + b_t * u_{t-1} + g_t * u_t
#   a_t = exp(A_t dt_t),  b_t = (1 - trap_t) dt_t a_t,  g_t = trap_t dt_t
#   o_{t,r} = h_t q_{t,r} + D v_{t,r},   y_t = sum_r MIMO_O[r] * o_{t,r} * silu(MIMO_Z[r] * z_t)
#
# q/k carry a per-head bias and a data-dependent rotary: the angle is
# cumsum(tanh(angle_proj) * dt * pi), applied to dims (i, i + N/2), i < n_angles.
# Reference: mamba3_MIMO_step_ref / mamba3_MIMO_chunk_ref in the mamba repo's
# tests/ops/tilelang/test_mamba3_mimo.py.
# ---------------------------------------------------------------------------

def _mamba3_rotate(x, cos, sin):
    """Rotary in the kernels' convention: pairs (i, i + N/2) for i < n_angles."""
    n = cos.shape[-1]
    half = x.shape[-1] // 2
    x0, x1 = x[..., :n], x[..., half:half + n]
    return torch.cat([x0 * cos - x1 * sin, x[..., n:half],
                      x0 * sin + x1 * cos, x[..., half + n:]], dim=-1)


def _mamba3_mimo_torch_segment(Q, K, V, ADT, DT, Trap, Q_bias, K_bias, MIMO_V, MIMO_Z,
                               MIMO_Out, Angles, D, Z, block=64):
    """One sequence (leading batch dim 1), zero initial state, fp32 throughout.
    Returns y (1, S, H, P) and the final (angle, ssm, k, v) states."""
    f32 = torch.float32
    _, S, R, G, N = Q.shape
    H, P = V.shape[2], V.shape[3]
    q, k = Q[0].to(f32), K[0].to(f32)                       # (S, R, G, N)
    if G != H:
        q, k = q.repeat_interleave(H // G, dim=2), k.repeat_interleave(H // G, dim=2)
    x, z = V[0].to(f32), Z[0].to(f32)                       # (S, H, P)
    adt, dt = ADT[0].to(f32).t(), DT[0].to(f32).t()         # (S, H)
    trap = torch.sigmoid(Trap[0].to(f32)).t()               # (S, H)

    # bias, then rotary with the cumulative angle
    angle = torch.cumsum(torch.tanh(Angles[0].to(f32)) * dt[..., None] * math.pi, dim=0)  # (S, H, A)
    cos, sin = torch.cos(angle)[:, None], torch.sin(angle)[:, None]                        # (S, 1, H, A)
    q = _mamba3_rotate(q + Q_bias.to(f32).transpose(0, 1)[None], cos, sin)                 # (S, R, H, N)
    k = _mamba3_rotate(k + K_bias.to(f32).transpose(0, 1)[None], cos, sin)

    v = x[:, None] * MIMO_V.to(f32).transpose(0, 1)[None]   # (S, R, H, P)
    gam = trap * dt                                         # weight of u_t in h_t
    dlt = (1 - trap) * dt                                   # weight of u_{t-1} in h_t (before a_t)

    h = x.new_zeros(H, P, N)                                # SSM state
    k_prev = v_prev = None                                  # u_{t-1} factors
    ys = []
    for c0 in range(0, S, block):
        c1 = min(c0 + block, S)
        L = c1 - c0
        qc, kc, vc = q[c0:c1], k[c0:c1], v[c0:c1]
        cs = torch.cumsum(adt[c0:c1], dim=0)                # (L, H) log-decay from block start
        # fold the previous block's last u into the carried state: h_{c0} uses
        # a_{c0} * (h_{c0-1} + dlt_{c0} * u_{c0-1})
        if k_prev is not None:
            h = h + dlt[c0][:, None, None] * torch.einsum("rhn,rhp->hpn", k_prev, v_prev)
        # intra-block weights w[t, s] = exp(cs_t - cs_s) * (gam_s + dlt_{s+1} [s < t])
        dlt_next = torch.cat([dlt[c0 + 1:c1], dlt.new_zeros(1, H)], dim=0)        # (L, H)
        causal = torch.ones(L, L, dtype=torch.bool, device=x.device).tril()
        strict = torch.ones(L, L, dtype=torch.bool, device=x.device).tril(-1)
        decay = torch.exp((cs[:, None] - cs[None, :]).masked_fill(~causal[..., None], float("-inf")))  # (t, s, H)
        w = decay * (gam[c0:c1][None] + dlt_next[None] * strict[..., None])
        scores = torch.einsum("trhn,sjhn->htsrj", qc, kc)        # (H, t, s, r, j)
        o = torch.einsum("hts,htsrj,sjhp->trhp", w.permute(2, 0, 1), scores, vc)
        o = o + torch.exp(cs)[:, None, :, None] * torch.einsum("hpn,trhn->trhp", h, qc)
        ys.append(o)
        # state at the end of the block (its own last u enters with gam only)
        w_end = torch.exp(cs[-1][None] - cs) * (gam[c0:c1] + dlt_next * (torch.arange(L, device=x.device) < L - 1)[:, None])
        h = torch.exp(cs[-1])[:, None, None] * h + torch.einsum("sh,sjhn,sjhp->hpn", w_end, kc, vc)
        k_prev, v_prev = kc[-1], vc[-1]

    o = torch.cat(ys, dim=0)                                # (S, R, H, P)
    o = o + D.to(f32)[None, None, :, None] * v
    o = o * F.silu(z[:, None] * MIMO_Z.to(f32).transpose(0, 1)[None])
    y = torch.einsum("srhp,rhp->shp", o, MIMO_Out.to(f32).transpose(0, 1))
    states = (torch.remainder(angle[-1], 2 * math.pi), h, k[-1], x[-1])
    return y[None], states


def _mamba3_mimo_torch(Q, K, V, ADT, DT, Trap, Q_bias, K_bias, MIMO_V, MIMO_Z, MIMO_Out,
                       Angles, D, Z, chunk_size, rotary_dim_divisor, dtype,
                       return_state=False, cu_seqlens=None):
    """Drop-in for mamba3_mimo (same arguments, MIMO_Out and Z required, no
    input states). Returns y, or (y, (angle, ssm, k, v)) with one state row per
    sequence: per batch row, or per cu_seqlens segment."""
    assert MIMO_Out is not None and Z is not None
    if cu_seqlens is not None:
        assert Q.shape[0] == 1, "varlen expects one packed row"
        bounds = cu_seqlens.tolist()
        segs = [(0, bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    else:
        segs = [(b, 0, Q.shape[1]) for b in range(Q.shape[0])]
    ys, states = [], []
    for b, s0, s1 in segs:
        sl = slice(s0, s1)
        y, st = _mamba3_mimo_torch_segment(
            Q[b:b + 1, sl], K[b:b + 1, sl], V[b:b + 1, sl],
            ADT[b:b + 1, :, sl], DT[b:b + 1, :, sl], Trap[b:b + 1, :, sl],
            Q_bias, K_bias, MIMO_V, MIMO_Z, MIMO_Out, Angles[b:b + 1, sl], D, Z[b:b + 1, sl])
        ys.append(y)
        states.append(st)
    y = torch.cat(ys, dim=1 if cu_seqlens is not None else 0).to(dtype)
    if not return_state:
        return y
    angle, ssm, k, v = (torch.stack(t) for t in zip(*states))
    return y, angle, ssm, k.to(dtype), v.to(dtype)


def _mamba3_rotary_step_torch(q, k, angle_state, angle_proj, dt, bias_q, bias_k):
    """apply_rotary_qk_inference_fwd(conjugate=False, rotate_pairwise=False).
    q, k: (B, R, H, N); angle_state, angle_proj: (B, H, A); dt: (B, H); biases (R, H, N)."""
    angle = angle_state.float() + torch.tanh(angle_proj.float()) * dt.float()[..., None] * math.pi
    cos, sin = torch.cos(angle)[:, None], torch.sin(angle)[:, None]
    q = _mamba3_rotate(q.float() + bias_q.float()[None], cos, sin)
    k = _mamba3_rotate(k.float() + bias_k.float()[None], cos, sin)
    return q, k, angle


def _mamba3_step_torch(state, k_state, v_state, A, B, C, D, x, dt, trap, xproj, outproj, z, zproj):
    """mamba3_step_fn (see selective_state_update_fused_ref_v2 in mamba_ssm).
    state (B, H, P, N) fp32; k_state/B/C (B, R, H, N); v_state/x/z (B, H, P);
    A, dt, trap (B, H) with trap post-sigmoid; xproj/outproj/zproj (R, H, P).
    Returns y (B, H, P) fp32 and the new state."""
    f32 = torch.float32
    A, dt, trap = A.to(f32), dt.to(f32), trap.to(f32)
    alpha = torch.exp(A * dt)
    beta = (1 - trap) * dt * alpha
    gamma = trap * dt
    xv = x.to(f32)[:, None] * xproj.to(f32)[None]           # (B, R, H, P)
    xs = v_state.to(f32)[:, None] * xproj.to(f32)[None]
    new = (state.to(f32) * alpha[..., None, None]
           + torch.einsum("brhp,brhn->bhpn", xv * gamma[:, None, :, None], B.to(f32))
           + torch.einsum("brhp,brhn->bhpn", xs * beta[:, None, :, None], k_state.to(f32)))
    o = torch.einsum("bhpn,brhn->brhp", new, C.to(f32)) + xv * D.to(f32)[None, None, :, None]
    o = o * F.silu(z.to(f32)[:, None] * zproj.to(f32)[None])
    y = torch.einsum("brhp,rhp->bhp", o, outproj.to(f32))
    return y, new


def _packed_doc_cu_seqlens(position_ids, device):
    """cu_seqlens for ONE packed row of documents, from position_ids resets.

    Frameworks with use_remove_padding (verl) feed a (1, T) row holding many
    documents, position_ids restarting at 0 at each document start, and no
    attention_mask worth the name. Returns (cu_seqlens int32, max_seqlen), or
    (None, None) when the row holds a single document -- dense attention and a
    single SSM segment are already exact there.
    """
    starts = (position_ids[0] == 0).nonzero(as_tuple=False).flatten()
    if starts.numel() <= 1:
        return None, None
    cu_seqlens = torch.cat([
        starts.to(torch.int32),
        torch.tensor([position_ids.shape[-1]], dtype=torch.int32, device=starts.device),
    ]).to(device=device, dtype=torch.int32)
    max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
    return cu_seqlens, max_seqlen


def _pack_cu_seqlens(cu_seqlens, batch, seqlen, device):
    """cu_seqlens for a (batch, seqlen) input packed into ONE varlen row.

    Row i occupies [i*seqlen, (i+1)*seqlen), so rows stay independent exactly as
    they are in the dense path. Inside a row the caller's document boundaries
    (if any) are passed straight through; nothing else is added.
    """
    if cu_seqlens is None:
        inner = torch.tensor([0, seqlen], dtype=torch.int32, device=device)
    else:
        inner = cu_seqlens.to(device=device, dtype=torch.int32).flatten()

    if batch == 1:
        return inner.contiguous()
    offsets = torch.arange(batch, device=device, dtype=torch.int32) * seqlen
    return torch.cat([
        (inner[:-1].unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1),
        torch.full((1,), batch * seqlen, dtype=torch.int32, device=device),
    ])


class OlalaMamba3MimoFast(nn.Module):
    def __init__(self, config: OlalaConfig, layer_idx: int, use_ve: bool = False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.d_model = config.hidden_size
        self.d_inner = 2*self.d_model
        self.rope_fraction = 0.5
        self.rotary_dim_divisor = 4
        self.A_floor = 1e-4
        self.mimo_dim = config.mamba_mimo_dim
        self.d_state = config.mamba_d_state
        self.headdim = config.mamba_headdim
        self.ngroups = config.mamba_ngroups
        self.mimo_proj_block_order = 1
        
        self.nheads = self.d_inner // self.headdim
        self.dr_out_dim = self.d_inner // self.mimo_proj_block_order
        self.chunk_size = 64 // self.mimo_dim

        self.split_tensor_size = int(self.d_state * self.rope_fraction)
        if self.split_tensor_size % 2 != 0:
            self.split_tensor_size -= 1
        self.num_rope_angles = self.split_tensor_size // 2

        tp_size = 1

        # Ensure that each TP rank gets at least one head:
        assert self.nheads % tp_size == 0, "nheads must be evenly divisble by tp_size"
        self.nheads_per_group = self.nheads // self.ngroups
        self.nheads_local_tp = self.nheads // tp_size
        self.n_repeat = self.nheads_local_tp // self.ngroups

        # Note that we do not need to confirm that `d_inner % tp_size == 0` because
        # `d_inner % headdim == 0`, `nheads = d_inner // headdim`, and `nheads % tp_size == 0`
        self.d_inner_per_group = self.d_inner // self.ngroups
        self.d_inner_local_tp = self.d_inner // tp_size
        self.dr_out_dim_local_tp = self.dr_out_dim // tp_size

        # Ensure that each TP rank gets at least one group:
        assert self.ngroups % tp_size == 0, "ngroups must be evenly divisible by tp_size"
        self.ngroups_local_tp = self.ngroups // tp_size

        # Ensure that each group has a positive integer number of heads:
        assert self.nheads % self.ngroups == 0, "nheads must be evenly divisible by ngroups"

        # Assume sequence parallelism: input is already partitioned along the sequence dimension
        self.in_proj = OlalaLinear(
            config,
            self.d_model,
            self.d_inner * 2 + 3 * self.nheads,
            bias=False,
        )
        self.in_proj_dyn = OlalaLinear(
            config,
            self.d_model,
            2 * self.ngroups * self.d_state * self.mimo_dim + self.num_rope_angles,
            bias=False,
        )

        self.B_bias = nn.Parameter(torch.ones((self.nheads_local_tp, self.mimo_dim, self.d_state), dtype=torch.float32), requires_grad=True)
        self.C_bias = nn.Parameter(torch.ones((self.nheads_local_tp, self.mimo_dim, self.d_state), dtype=torch.float32), requires_grad=True)
        self.B_norm = OlalaNorm(config, self.d_state)
        self.C_norm = OlalaNorm(config, self.d_state)

        # Initialize up/down MIMO projection (for x and z)
        in_proj_mimo_x_init_weights = torch.ones(self.nheads_local_tp, self.mimo_dim, self.headdim, dtype=torch.float32)/self.mimo_dim
        in_proj_mimo_z_init_weights = torch.ones(self.nheads_local_tp, self.mimo_dim, self.headdim, dtype=torch.float32)
        out_proj_mimo_init_weights = torch.ones(self.nheads_local_tp, self.mimo_dim, self.headdim, dtype=torch.float32)/self.mimo_dim
        self.in_proj_mimo_x = nn.Parameter(in_proj_mimo_x_init_weights, requires_grad=True)
        self.in_proj_mimo_z = nn.Parameter(in_proj_mimo_z_init_weights, requires_grad=True)
        self.out_proj_mimo = nn.Parameter(out_proj_mimo_init_weights, requires_grad=True)

        with nullcontext():
            dt_min = 0.001
            dt_max = 0.1
            dt_init_floor = 1e-4
            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(
                    self.nheads_local_tp,
                    # `.current_device()` raises without a driver; the CPU-only
                    # checkpoint conversion builds this model with no GPU. On a
                    # GPU this is unchanged, RNG stream included.
                    device=torch.cuda.current_device() if torch.cuda.is_available() else "cpu",
                )
                * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias = nn.Parameter(inv_dt)
            # Our initialization would set all Linear.bias to zero,
            # need to mark this one as _no_reinit
            self.dt_bias._no_reinit = True
            # Just to be explicit. Without this we already don't
            # put wd on dt_bias because of the check
            # name.endswith("bias") in param_grouping.py
            self.dt_bias._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads_local_tp,
                                         device=torch.cuda.current_device() if torch.cuda.is_available() else "cpu")) # Keep in fp32
        self.D._no_weight_decay = True # useless flag

        if self.config.mamba3_postgate_norm:
            self.output_norm = OlalaNorm(config, self.d_inner_local_tp)

        self.previous_window_size = 0

    def forward(self, hidden_states,
                cache_params: Optional[HybridOlalaDynamicCache] = None,
                cu_seqlens: Optional[torch.Tensor] = None,
                unpadded: bool = False,
                **kwargs):
        """
        hidden_states: (B L D)
        Returns: same shape as hidden_states
        """

        layer_idx = self.layer_idx
        batch, q_len, _ = hidden_states.shape

        use_precomputed_states = cache_params is not None and q_len == 1
        is_prefill = cache_params is not None

        if use_precomputed_states:
            state = (
                cache_params.mamba3_angle_states[layer_idx],
                cache_params.mamba3_ssm_states[layer_idx],
                cache_params.mamba3_k_states[layer_idx],
                cache_params.mamba3_v_states[layer_idx],
            )

            y_t, *state = self.step(hidden_states[:, 0, :], *state)

            (
                cache_params.mamba3_angle_states[layer_idx],
                cache_params.mamba3_ssm_states[layer_idx],
                cache_params.mamba3_k_states[layer_idx],
                cache_params.mamba3_v_states[layer_idx],
            ) = state
            cache_params.past_length[layer_idx] += 1

            y = y_t.unsqueeze(1)          # (B, 1, D)
            if self.config.mamba3_postgate_norm:
                y = self.output_norm(y)
            return y, None, None

        if self.config.complete_slw and self.config.slw_wsize > 128:
            assert not is_prefill
            if self.previous_window_size != self.config.slw_wsize:
                logger.info(f"Mamba3Mimo complete_slw: updating previous_window_size from {self.previous_window_size} to {self.config.slw_wsize}")
                self.previous_window_size = self.config.slw_wsize
            batch, l, dim = hidden_states.shape
            assert batch == 1, "complete_slw only supports batch size of 1"
            hidden_states = hidden_states.reshape(-1, self.config.slw_wsize, dim).contiguous()
            # The rows ARE the windows now, so caller offsets (expressed in the
            # original L coordinates) no longer apply. One segment per window,
            # which is what the dense path did here.
            cu_seqlens = None

        # Input projection
        zxdtAtrap = self.in_proj(hidden_states)
        per_head = zxdtAtrap.view(*zxdtAtrap.shape[:-1], self.nheads_local_tp, 2*self.headdim+3)
        off = 0
        z    = per_head[..., off : off + self.headdim];   off += self.headdim # (L, B, H, p)
        x    = per_head[..., off : off + self.headdim];   off += self.headdim # (L, B, H, p)
        dt   = per_head[..., off];                        off += 1 # (L, B, H)
        A    = per_head[..., off];                        off += 1 # (L, B, H)
        trap = per_head[..., off];                        off += 1 # (L, B, H)
        z = rearrange(z, "b l H p -> b l H p") # (B, L, H_local, p)
        x = rearrange(x, "b l H p -> b l H p") # (B, L, H_local, p)
        dt   = rearrange(dt,   "b l n -> b l n").to(torch.float32) # (B, L, H_local)
        A    = rearrange(A,    "b l n -> b l n") # (B, L, H_local)
        trap = rearrange(trap, "b l n -> b n l") # (B, H_local, L)

        BCangle = self.in_proj_dyn(hidden_states)
        B = BCangle[..., 0:self.ngroups*self.mimo_dim*self.d_state]
        C = BCangle[..., self.ngroups*self.mimo_dim*self.d_state:2*self.ngroups*self.mimo_dim*self.d_state]
        angle = BCangle[..., 2*self.ngroups*self.mimo_dim*self.d_state:] # (L, B, S)
        B = rearrange(B, "b l (G r n) -> b l r G n", G=self.ngroups, r=self.mimo_dim)
        C = rearrange(C, "b l (G r n) -> b l r G n", G=self.ngroups, r=self.mimo_dim)

        _A = -F.softplus(A.to(torch.float32)) # (B, L, N)
        _A = torch.clamp(_A, max=-self.A_floor)
        dt = F.softplus(dt + self.dt_bias) # (B, L, N)
        ADT = _A * dt

        B = self.B_norm(B)
        C = self.C_norm(C)

        angle = angle.unsqueeze(-2).expand(-1, -1, self.nheads_local_tp, -1) # (B, L, G, S)
        # No angle cumsum here: mamba3_mimo takes RAW angles and applies it
        # internally (see the kernel imports at the top of this file).

        ADT = rearrange(ADT, "b l n -> b n l")
        dt = rearrange(dt, "b l n -> b n l")

        # Route the chunk kernel through its VARLEN path by describing the
        # prompt as a single segment [0, q_len]. Two wins, no kernel change:
        #  * varlen declares S = T.dynamic, so it is compiled ONCE rather than
        #    once per prompt length -- the dense path costs 12-18 s of tilelang
        #    compilation for every new length (measured);
        #  * varlen carries the two-level scan, which removes the
        #    longest-segment bottleneck: one thread-block per (head, SEGMENT)
        #    becomes one per (head, CHUNK-BLOCK). A 65k prefill at B=1 is
        #    otherwise 48 blocks on 114 SMs with 4096 sequential chunk steps.
        # Outputs match the dense path to bf16 noise (rel <= 2.1e-04, measured
        # at S = 1024 / 1152 / 2048).
        #
        # The varlen helpers (angle_dt_fwd, dacs/segsum) assert B == 1,
        # so a real batch is PACKED into one row: (b, l, ...) -> (1, b*l, ...)
        # with cu_seqlens = [0, l, 2l, ..., b*l]. That IS the dense semantics --
        # every row already starts from a zero state and never sees its
        # neighbours -- and the per-row final states fall out for free, because
        # varlen returns one state per SEGMENT ([NS, H, ...]), NS == b here.
        # `cu_seqlens` from the caller is passed straight through, matching
        # training: Megatron's Mamba-3 layer hands packed_seq_params.cu_seqlens_q
        # to this same kernel UNCONDITIONALLY, so the SSM state was reset at every
        # document boundary during pretraining. (config.intra_doc_masking is an
        # ATTENTION-only switch there -- causal vs padding_causal -- so it must NOT
        # gate this.) Row boundaries from a batch are appended the same way.
        # Set OLALA_MAMBA3_VARLEN=0 to force the dense path back.
        #
        # NB: read the batch off x, not `batch` -- complete_slw reshapes the
        # rows into windows above and leaves `batch` stale.
        kb, kl = x.shape[0], x.shape[1]
        _use_varlen = os.environ.get("OLALA_MAMBA3_VARLEN", "1") != "0"
        _varlen_kw = {}
        if _use_varlen:
            _varlen_kw["cu_seqlens"] = _pack_cu_seqlens(cu_seqlens, kb, kl, x.device)
            if kb > 1:
                C     = rearrange(C,     "b l r g n -> 1 (b l) r g n")
                B     = rearrange(B,     "b l r g n -> 1 (b l) r g n")
                x     = rearrange(x,     "b l h p -> 1 (b l) h p")
                z     = rearrange(z,     "b l h p -> 1 (b l) h p")
                angle = rearrange(angle, "b l h n -> 1 (b l) h n")
                ADT   = rearrange(ADT,   "b h l -> 1 h (b l)")
                dt    = rearrange(dt,    "b h l -> 1 h (b l)")
                trap  = rearrange(trap,  "b h l -> 1 h (b l)")

        y, kernel_state = mamba3_tilelang(
            **_varlen_kw,
            Q=C.contiguous(),#.bfloat16(),
            K=B.contiguous(),#.bfloat16(),
            V=x.contiguous(),#.bfloat16(),
            ADT=ADT.to(torch.float32).contiguous(),
            DT=dt.to(torch.float32).contiguous(),
            Trap=trap.contiguous(),
            Q_bias=self.C_bias.to(torch.float32),
            K_bias=self.B_bias.to(torch.float32),
            MIMO_V=self.in_proj_mimo_x.to(torch.float32),
            MIMO_Z=self.in_proj_mimo_z.to(torch.float32),
            MIMO_Out=self.out_proj_mimo.to(torch.float32),
            Angles=angle.to(torch.float32).contiguous(),
            D=self.D.to(torch.float32).contiguous(),
            Z=z.contiguous(),
            chunk_size=self.chunk_size,
            rotary_dim_divisor=self.rotary_dim_divisor,
            dtype=x.dtype,
            return_state=is_prefill,
        )

        if _use_varlen and kb > 1:
            y = y.reshape(kb, kl, *y.shape[2:])

        if is_prefill:
            # varlen returns one state per SEGMENT, the cache holds one per ROW.
            # Segments are row-major (row r owns r*w .. r*w+w-1), so the state
            # generation must continue from is the LAST segment of each row.
            # w > 1 whenever the caller packed several documents into a row.
            n_seg = (_varlen_kw["cu_seqlens"].numel() - 1) if _varlen_kw else kb
            if n_seg % kb:
                raise ValueError(
                    f"cache prefill got {n_seg} varlen segments over {kb} rows, "
                    "which do not divide; the state cache is per-row.")
            angle_state_out, ssm_state_out, k_state_out, v_state_out = kernel_state
            if _use_varlen:
                # The chunk kernel sets Final_V = V[:, -1] of the WHOLE packed
                # row (mamba3_mimo.py:127) instead of per segment, so it is only
                # right for the last one. Recompute it: last token of each
                # segment. Without this, batched prefill-with-cache would give
                # every row the final row's V.
                _ends = _varlen_kw["cu_seqlens"][1:].long() - 1
                v_state_out = x[0, _ends].contiguous()          # (NS, H, P)
            if n_seg != kb:
                w = n_seg // kb
                angle_state_out, ssm_state_out, k_state_out, v_state_out = (
                    t.reshape(kb, w, *t.shape[1:])[:, -1].contiguous()
                    for t in (angle_state_out, ssm_state_out, k_state_out, v_state_out)
                )
            cache_params.mamba3_angle_states[layer_idx] = angle_state_out
            cache_params.mamba3_ssm_states[layer_idx]   = ssm_state_out
            cache_params.mamba3_k_states[layer_idx]     = k_state_out
            cache_params.mamba3_v_states[layer_idx]     = v_state_out
            cache_params.past_length[layer_idx]         += q_len

        y = rearrange(y, "b l h p -> b l (h p)")
        if self.config.mamba3_postgate_norm:
            y = self.output_norm(y)
        #y = rearrange(y, "b l (h p) -> b l h p", h=self.nheads_local_tp)

        if self.config.complete_slw and self.config.slw_wsize > 128:
            y = y.reshape(batch, l, -1).contiguous()

        return y, None, None
    
    def _preprocess(self, A_proj, dd_dt, B, C, x, z, trap_proj, angle_proj):
        _A = -F.softplus(A_proj.to(torch.float32))
        _A = torch.clamp(_A, max=-self.A_floor)
        DT = F.softplus(dd_dt + self.dt_bias)
        trap = torch.sigmoid(trap_proj)

        B = rearrange(B, "b (r g s) -> b r g s", g=self.ngroups, r=self.mimo_dim)
        C = rearrange(C, "b (r g s) -> b r g s", g=self.ngroups, r=self.mimo_dim)
        B = self.B_norm(B)
        C = self.C_norm(C)
        B = B.expand(-1, -1, self.nheads, -1) # (B, R, N, S)
        C = C.expand(-1, -1, self.nheads, -1) # (B, R, N, S)

        angles = angle_proj.unsqueeze(-2).expand(-1, self.nheads, -1)

        return DT, B, C, x, z, trap, _A, angles

    def step(self, u, angle_state, ssm_state, k_state, v_state):
        """
        Decode function using CuteDSL kernel from mamba3_step_fn.py, or the
        PyTorch step when that kernel is unavailable.

        Args:
            u: (batch, d_model)
            angle_state: (batch, nheads, num_rope_angles)
            ssm_state: (batch, nheads, headdim, d_state)
            k_state: (batch, R, nheads, d_state), where R = mimo_rank
            v_state: (batch, nheads, headdim)
        Returns:
            out: (batch, d_model)
            nxt_angle_state: (batch, nheads, num_rope_angles)
            state_out: (batch, nheads, headdim, d_state)
            nxt_k_state: (batch, R, nheads, d_state), where R = mimo_rank
            nxt_v_state: (batch, nheads, headdim)
        """

        zxdtAtrap = self.in_proj(u)
        per_head = zxdtAtrap.view(*zxdtAtrap.shape[:-1], self.nheads_local_tp, 2*self.headdim+3)
        off = 0
        z    = per_head[..., off : off + self.headdim];   off += self.headdim # (L, B, H, p)
        x    = per_head[..., off : off + self.headdim];   off += self.headdim # (L, B, H, p)
        dt   = per_head[..., off];                        off += 1 # (L, B, H)
        A    = per_head[..., off];                        off += 1 # (L, B, H)
        trap = per_head[..., off];                        off += 1 # (L, B, H)
        z = z.flatten(-2) # TODO: better efficient way to do it ? (considering the op we do just after)
        x = x.flatten(-2)

        x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
        z = rearrange(z, "b (h p) -> b h p", p=self.headdim)

        BCangle = self.in_proj_dyn(u)
        B = BCangle[..., 0:self.ngroups*self.mimo_dim*self.d_state]
        C = BCangle[..., self.ngroups*self.mimo_dim*self.d_state:2*self.ngroups*self.mimo_dim*self.d_state]
        angles = BCangle[..., 2*self.ngroups*self.mimo_dim*self.d_state:] # (L, B, S)

        DT, B, C, x, z, trap, A, angles = self._preprocess(A, dt, B, C, x, z, trap, angles)

        bias_q = rearrange(self.C_bias, "h r n -> r h n")
        bias_k = rearrange(self.B_bias, "h r n -> r h n")

        if mamba3_step_fn is None:
            return self._step_torch(x, z, DT, B, C, trap, A, angles, bias_q, bias_k,
                                    angle_state, ssm_state, k_state, v_state)

        C, B, nxt_angle_state = apply_rotary_qk_inference_fwd(
            q=C,
            k=B,
            angle_state=angle_state, 
            angle_proj=angles,
            dt=DT,
            bias_q=bias_q,
            bias_k=bias_k, 
            conjugate=False,
            inplace=False,
            rotate_pairwise=False
        )

        nxt_v_state = x
        nxt_k_state = B

        xpj = rearrange(self.in_proj_mimo_x, "h r p -> r h p", p=self.headdim).contiguous()
        zpj = rearrange(self.in_proj_mimo_z, "h r p -> r h p", p=self.headdim).contiguous()
        outpj = rearrange(self.out_proj_mimo, "h r p -> r h p", p=self.headdim).contiguous()

        state_out = torch.empty_like(ssm_state)
        y = torch.empty_like(x)
        mamba3_step_fn(
            ssm_state.to(torch.float32),
            k_state.to(torch.bfloat16),
            v_state.to(torch.bfloat16),
            A,
            B.to(torch.bfloat16),
            C.to(torch.bfloat16),
            self.D,
            x,
            DT,
            trap,
            xpj,
            outpj,
            state_out,
            y,
            z=z,
            zproj=zpj,
            tile_D=64,
            num_warps=4,
        )

        y = rearrange(y, "b h p -> b (h p)")

        return y, nxt_angle_state, state_out, nxt_k_state, nxt_v_state

    def _step_torch(self, x, z, DT, B, C, trap, A, angles, bias_q, bias_k,
                    angle_state, ssm_state, k_state, v_state):
        """step() without the Triton rotary / CuteDSL step kernels."""
        C, B, nxt_angle_state = _mamba3_rotary_step_torch(
            C, B, angle_state, angles, DT, bias_q, bias_k)
        y, state_out = _mamba3_step_torch(
            ssm_state, k_state, v_state, A, B, C, self.D, x, DT, trap,
            rearrange(self.in_proj_mimo_x, "h r p -> r h p"),
            rearrange(self.out_proj_mimo, "h r p -> r h p"),
            z, rearrange(self.in_proj_mimo_z, "h r p -> r h p"))
        y = rearrange(y.to(x.dtype), "b h p -> b (h p)")
        return y, nxt_angle_state, state_out, B.to(x.dtype), x

    def allocate_inference_cache(self, batch_size, max_seqlen, device=None, dtype=None, inplace_state=None, **kwargs):
        device = self.in_proj.weight.device if device is None else device
        dtype = self.in_proj.weight.dtype if dtype is None else dtype

        # RoPE State
        angle_dt_state = torch.zeros(
            (batch_size, self.nheads, self.num_rope_angles),
            device=device,
            dtype=torch.float32,
        )

        # SSM State
        ssm_state = torch.zeros(
            (batch_size, self.nheads, self.headdim, self.d_state),
            device=device,
            dtype=torch.float32,
        )

        # K (=B) State
        k_state = torch.zeros(
            (batch_size, self.mimo_dim, self.nheads, self.d_state),
            device=device,
            dtype=dtype,
        )

        # V (=x) State
        v_state = torch.zeros(
            (batch_size, self.nheads, self.headdim),
            device=device,
            dtype=dtype,
        )

        return (angle_dt_state, ssm_state, k_state, v_state)

class OlalaMLP(nn.Module):
    def __init__(self, config: OlalaConfig, intermediate_size: Optional[int] = None):
        super().__init__()
        self.config = config
        intermediate_size = intermediate_size or config.intermediate_size
        self.fc_1 = OlalaLinear(config, config.hidden_size, intermediate_size, bias=False)
        self.fc_2 = OlalaLinear(config, intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states):
        hidden_states = self.fc_1(hidden_states)
        hidden_states = F.relu(hidden_states).square()
        hidden_states = self.fc_2(hidden_states)
        return hidden_states

class _TorchParallelExperts(nn.Module):
    """Parameter holder laid out like scattermoe.ParallelExperts: weight (E, out, in)."""
    def __init__(self, num_experts, input_size, output_size):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, output_size, input_size))
        nn.init.normal_(self.weight, std=0.02)


class TorchMoEMLP(nn.Module):
    """Pure-PyTorch stand-in for scattermoe.mlp.MLP, used when scattermoe is not
    installed. Same parameters, same forward signature, same math:

        y[t] = sum_k p[t, k] * W_out[e] @ act(W_in[e] @ x[t]),   e = idx[t, k]

    Tokens are grouped by expert (one gather + two matmuls per active expert)
    and the weighted outputs are accumulated in fp32.
    """
    def __init__(self, input_size, hidden_size, num_experts, top_k, bias=False, activation=None):
        super().__init__()
        assert not bias, "TorchMoEMLP: bias is not supported (Olala uses none)"
        self.num_experts = num_experts
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.top_k = min(top_k, num_experts)
        self.activation = activation
        self.experts = _TorchParallelExperts(num_experts, input_size, hidden_size)
        self.output_experts = _TorchParallelExperts(num_experts, hidden_size, input_size)

    def forward(self, x: torch.Tensor, expert_p: torch.Tensor, expert_idxs: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-1])
        top_k = expert_idxs.shape[-1]
        flat_idx = expert_idxs.reshape(-1)
        order = torch.argsort(flat_idx, stable=True)
        token_of = order // top_k
        gate_of = expert_p.reshape(-1)[order].float()
        counts = torch.bincount(flat_idx, minlength=self.num_experts).tolist()
        w_in, w_out = self.experts.weight, self.output_experts.weight
        out = torch.zeros(x.shape[0], self.input_size, device=x.device, dtype=torch.float32)
        start = 0
        for e, n in enumerate(counts):
            if n == 0:
                continue
            tok = token_of[start:start + n]
            # weights follow the activation dtype, as in the scattermoe kernels
            h = self.activation(F.linear(x[tok], w_in[e].to(x.dtype)))
            y = F.linear(h, w_out[e].to(x.dtype))
            out.index_add_(0, tok, y.float() * gate_of[start:start + n, None])
            start += n
        return out.to(x.dtype).view(*x_shape[:-1], self.input_size)


class OlalaMoE(nn.Module):
    def __init__(self, config: OlalaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.moe_gate = OlalaLinear(config, config.hidden_size, config.moe_num_routed_experts, bias=False, cosnet=False)
        if self.config.moe_routed_input_dim:
            self.down_proj = OlalaLinear(config, config.hidden_size, config.moe_routed_input_dim, bias=False)
            self.up_proj = OlalaLinear(config, config.moe_routed_input_dim, config.hidden_size, bias=False)
        self.experts = (ScatterMoE if ScatterMoE is not None else TorchMoEMLP)(
            input_size=config.moe_routed_input_dim or config.hidden_size,
            hidden_size=config.moe_routed_intermediate_size,
            num_experts=config.moe_num_routed_experts,
            top_k=config.moe_num_active_experts,
            activation=lambda x: F.relu(x).square(),
        )
        if self.config.moe_shared_expert_gate:
            self.shared_gate = OlalaLinear(config, config.hidden_size, 1, bias=False)
        self.shared_experts = (
            OlalaMLP(config, config.moe_shared_intermediate_size)
            if config.moe_shared_intermediate_size and config.moe_shared_intermediate_size > 0
            else None
        )

        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(config.moe_num_routed_experts, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "expert_bias",
            torch.zeros(config.moe_num_routed_experts, dtype=torch.float32),
            persistent=True,
        )

        with torch.no_grad():
            self.experts.experts.weight.normal_(mean=0.0, std=self.config.initializer_range)
            self.experts.output_experts.weight.normal_(mean=0.0, std=self.config.initializer_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, slen, dim = x.shape
        input_dtype = x.dtype
        x = x.view(-1, dim)

        # router.
        logits = torch.matmul(x.float(), self.moe_gate.weight.float().t())

        scores = torch.sigmoid(logits.float()).type_as(logits)
        scores_for_routing = scores + self.expert_bias.float()
        #scores_orig = scores_for_routing.clone()
        _, top_indices = torch.topk(scores_for_routing, k=self.config.moe_num_active_experts, dim=1)
        scores = torch.gather(scores, dim=1, index=top_indices).type_as(logits)
        probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if self.config.moe_num_active_experts > 1 else scores
        probs = probs * self.config.moe_routed_scaling_factor
        probs = probs.to(input_dtype)

        if self.training:
            with torch.no_grad():
                idx = top_indices.reshape(-1) # (N*K,)
                tpe = torch.bincount(idx, minlength=self.config.moe_num_routed_experts).to(self.tokens_per_expert.dtype).to(x.device)
                self.tokens_per_expert.add_(tpe)

        # experts.
        x0 = x
        if self.config.moe_routed_input_dim:
            x0 = self.down_proj(x).to(x.dtype)
        out_experts = self.experts(x0, probs, top_indices)
        if self.config.moe_routed_input_dim:
            out_experts = self.up_proj(out_experts).to(out_experts.dtype)
        # shared experts.
        out = None
        if self.shared_experts is not None:
            out = self.shared_experts(x)
            if self.config.moe_shared_expert_gate:
                logits = self.shared_gate(x)
                scores = torch.sigmoid(logits.float()).type_as(logits)
                out = out * scores
        if out is None:
            return out_experts.reshape(bs, slen, dim)
        return (out + out_experts).reshape(bs, slen, dim)#, top_indices, scores_orig

class OlalaGeodesicNorm(nn.Module):
    def __init__(self, config: OlalaConfig, layer_idx: int):
        super().__init__()

        # Declared 0-dim to match what every export on disk stores. FSDP
        # rejects 0-dim parameters, so they are widened to [1] AFTER loading
        # by OlalaPreTrainedModel.widen_geodesic_scalars(), which
        # from_pretrained calls for you.
        #
        # Declaring [1] here instead does NOT work: transformers records a
        # shape mismatch on the way in, and _initialize_missing_keys()
        # REINITIALIZES every mismatched key (loading_info
        # .missing_and_mismatched()) before any model hook can intervene, so
        # the trained values are silently replaced with garbage. Widening
        # after the load is the only point where both constraints hold.
        self.scale = nn.Parameter(torch.tensor(1.))
        self.bias = nn.Parameter(torch.tensor(0.))
        self.register_buffer("prosres_scalar", torch.tensor(1.0))
        self.clamp = torch.pi/4
        self.layer_idx = layer_idx

    _SCALARS = ("scale", "bias")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # torch's load_state_dict (FSDP / verl checkpoint restore) copies INTO
        # the existing tensor and validates shapes first, so it raises on any
        # 0-dim vs [1] difference. Reshape the incoming entry to whatever this
        # module currently holds -- the model may be 0-dim (fresh) or [1]
        # (already widened), and a checkpoint may be either.
        for _name in OlalaGeodesicNorm._SCALARS:
            _key = prefix + _name
            _incoming = state_dict.get(_key)
            _current = getattr(self, _name, None)
            if _incoming is None or _current is None:
                continue
            if _incoming.shape != _current.shape and _incoming.numel() == 1:
                state_dict[_key] = _incoming.reshape(_current.shape)
        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x, g):
        """
        x: residual;
        g: ffn(x) or attn(x);
        """

        x_norm_sq = x.square().sum(dim=-1, keepdim=True).clamp_min(1e-12)
        proj_coeff = (x * g).sum(dim=-1, keepdim=True) / x_norm_sq
        gradient = g - proj_coeff * x
        tangent_norm = torch.norm(gradient, p=2, dim=-1, keepdim=True)
        safe_tangent_norm = torch.clamp(tangent_norm, min=1e-8)
        unit_tangent = gradient / safe_tangent_norm
        R = torch.norm(x, p=2, dim=-1, keepdim=True)
        safe_R = torch.clamp(R, min=1e-6)
        theta = torch.clamp(safe_tangent_norm / safe_R, max=self.clamp)
        theta = torch.clamp((theta * self.scale + self.bias) / (self.layer_idx + 1), max=self.clamp)
        output = x * torch.cos(theta) + unit_tangent * safe_R * torch.sin(theta)
        # Dtype-stability: `theta * self.scale + self.bias` promotes theta --
        # and through cos/sin the OUTPUT -- to the params' dtype, which can
        # differ from the residual dtype under mixed precision. Pin the
        # output to the residual dtype; no-op when dtypes already agree.
        return output.to(x.dtype)

class OlalaMonoBlock(GradientCheckpointingLayer):
    def __init__(self, config: OlalaConfig, layer_idx: int, layer_type: str, use_ve: bool = False, mlp_type: str = 'd'):
        super().__init__()
        self.config = config
        # When blocks have been pruned out (config.layer_idx_map present),
        # the surviving block at *position* `layer_idx` originally had a
        # different index. We preserve that original index so that
        # OlalaGeodesicNorm.theta scaling by 1/(layer_idx+1) and any other
        # layer-index-dependent logic stays consistent with how the model
        # was trained. Pruning configs (skip_moe_layers, skip_mixer_layers)
        # are also expressed in ORIGINAL indices.
        _map = getattr(config, "layer_idx_map", None)
        if _map is not None and layer_idx < len(_map):
            layer_idx = int(_map[layer_idx])
        self.layer_idx = layer_idx

        if use_ve:
            assert layer_type == 'M', "VE is only supported for 'M' layers."

        if layer_type == 'V':
            self.mixer = OlalaDifferentialTensorProductAttentionV2(config, layer_idx=layer_idx, use_ve=use_ve)
            head_dim = self.mixer.head_dim
            num_attention_heads = self.mixer.num_signal_heads
            use_gate = config.gate_attn
        elif layer_type == 'M':
            self.mixer = OlalaMamba3MimoFast(config, layer_idx=layer_idx, use_ve=use_ve)
            head_dim = self.mixer.headdim
            num_attention_heads = self.mixer.nheads
            use_gate = False # inside Mamba3MimoFast
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

        if use_gate:
            if self.config.gate_type == "elementwise":
                self.gate_proj = OlalaLinear(self.config, config.hidden_size, num_attention_heads*head_dim, bias=False)
            elif self.config.gate_type == "kimi":
                self.gate_proj = nn.Sequential(
                    OlalaLinear(config, config.hidden_size, head_dim, bias=False),
                    OlalaLinear(config, head_dim, num_attention_heads*head_dim, bias=True),
                )
            elif self.config.gate_type == "headwise":
                self.gate_proj = OlalaLinear(self.config, config.hidden_size, num_attention_heads, bias=False)
                if self.config.scalar_proj_as_hidden_matrix:
                    self.gate_proj.is_scalar_weight = True
            else:
                raise ValueError(f"Unknown gate_type: {self.config.gate_type}")
            val = 0.
            if self.config.zero_centered_gate:
                val = 1.15
            self.gate_bias = val
            if self.config.gate_act == "silu":
                self.gate_act = F.silu
            elif self.config.gate_act == "sigmoid":
                self.gate_act = F.sigmoid
            else:
                raise ValueError(f"Unknown gate_act: {self.config.gate_act}")
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.use_gate = use_gate

        self.mixer_proj = OlalaLinear(config, head_dim*num_attention_heads, config.hidden_size, bias=False)
        if config.mixer_gn:
            self.mixer_group_norm = OlalaHeadWiseRMSNorm(n_heads=num_attention_heads, d_head=head_dim, eps=config.norm_epsilon, zero_centered_gamma=config.zero_centered_gamma)

        if not config.geodesic_update:
            self.input_norm = OlalaNorm(config, config.hidden_size)
            self.postmixer_norm = OlalaNorm(config, config.hidden_size)
        else:
            self.input_norm = torch.nn.Identity()
            self.postmixer_norm = torch.nn.Identity()
            self.geodesic_mixer = OlalaGeodesicNorm(config, self.layer_idx)
            self.geodesic_mlp = OlalaGeodesicNorm(config, self.layer_idx)

        # ---- pruning / skip-MoE support ----
        # If `config.skip_moe_layers` contains this layer_idx, replace mlp with a
        # parameter-free no-op and also disable the geodesic_mlp combine so the
        # block becomes "mixer-only" at this layer. Saved checkpoints with these
        # layers don't carry MoE params for them at all.
        skip_moe = getattr(config, "skip_moe_layers", None) or []
        if layer_idx in skip_moe:
            self.mlp = _NullMlp()
            if config.geodesic_update:
                self.geodesic_mlp = _NullGeodesic()
        elif not config.moe or mlp_type == 'd':
            self.mlp = OlalaMLP(config)
        elif mlp_type == 'm':
            self.mlp = OlalaMoE(config, layer_idx=layer_idx)
        else:
            raise ValueError(f"Unknown mlp_type: {mlp_type}")

        # Skip-mixer support: if this layer is in config.skip_mixer_layers,
        # replace the mixer-side submodules with parameter-free placeholders.
        # The forward also short-circuits the mixer phase via _skip_mixer.
        skip_mixer = getattr(config, "skip_mixer_layers", None) or []
        if layer_idx in skip_mixer:
            self.mixer = _NullMixer()
            self.mixer_proj = _NullMlp()                # returns zeros_like(input)
            self.input_norm = nn.Identity()
            if config.geodesic_update:
                self.geodesic_mixer = _NullGeodesic()
            if self.use_gate:
                self.gate_proj = _NullMlp()
            if config.mixer_gn:
                self.mixer_group_norm = nn.Identity()

        lns = 1.
        if config.layer_norm_scaling:
            lns = 1. / math.sqrt(layer_idx + 1)
        self.lns = float(lns)

        a, b = 1., 1.
        if self.config.use_completed_p:
            a = (len(self.config.layers_config)/self.config.base_depth) ** (-self.config.completed_p_alpha)
        self.a = float(a)
        self.b = float(b)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[HybridOlalaDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        unpadded: bool = False,
        ve=None,
        **kwargs,
    ):
        # Phase-stable compute dtype under FSDP2 mixed precision + HF
        # gradient checkpointing. FSDP2's cast_forward_inputs runs inside
        # this (checkpointed) __call__ during the original forward, casting
        # the incoming fp32 residual to bf16 -- but the checkpoint saves the
        # PRE-CAST input and the hook does not re-run at the same point in
        # the recompute, so every derived activation flips dtype and trips
        # torch.utils.checkpoint's check_recomputed_tensors_match. Re-apply
        # the autocast dtype explicitly: autocast state is replayed during
        # recompute, so this is identical in both phases and a no-op when
        # the input is already cast.
        if (
            torch.is_autocast_enabled()
            and hidden_states.is_floating_point()
            and hidden_states.dtype != torch.get_autocast_dtype("cuda")
        ):
            hidden_states = hidden_states.to(torch.get_autocast_dtype("cuda"))
        # Skip-mixer support: if this layer is in config.skip_mixer_layers,
        # the mixer phase is bypassed entirely — residual passes through and
        # no KV is produced (downstream layers receive None from this slot).
        _skip_mixer = self.layer_idx in (getattr(self.config, "skip_mixer_layers", None) or [])
        if not _skip_mixer:
            # MIXER.
            residual = hidden_states
            x_in = hidden_states
            hidden_states = self.lns * self.input_norm(x_in) # (B, L, D)
            y_mixer, _, _ = self.mixer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                cache_params=cache_params,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                unpadded=unpadded,
                ve=ve,
            ) # (B, L, E*D)
            if self.config.mixer_gn and not self.config.gate_before_norm:
                y_mixer = self.mixer_group_norm(y_mixer)
            if self.use_gate:
                if self.config.gate_type == "elementwise" or self.config.gate_type == "kimi":
                    g_proj = self.gate_proj(hidden_states).view(hidden_states.size(0), hidden_states.size(1), self.num_attention_heads, self.head_dim).to(y_mixer.dtype)
                elif self.config.gate_type == "headwise":
                    g_proj = self.gate_proj(hidden_states).unsqueeze(-1).to(y_mixer.dtype)
                else:
                    raise ValueError(f"Unknown gate_type: {self.config.gate_type}")
                y_mixer = y_mixer * self.gate_act(g_proj + self.gate_bias)
            if self.config.mixer_gn and self.config.gate_before_norm:
                y_mixer = self.mixer_group_norm(y_mixer)
            y_mixer = y_mixer.view(y_mixer.size(0), y_mixer.size(1), -1)
            y_mixer = self.mixer_proj(y_mixer)

            if self.config.geodesic_update:
                hidden_states = self.geodesic_mixer(residual, y_mixer)
            else:
                hidden_states = self.b * residual + self.a * y_mixer

        # MLP.
        residual = hidden_states
        x_in = residual
        hidden_states = self.lns * self.postmixer_norm(x_in)
        y_mlp = self.mlp(hidden_states) # (B, L, D)

        if self.config.geodesic_update:
            hidden_states = self.geodesic_mlp(residual, y_mlp)
        else:
            hidden_states = self.b * residual + self.a * y_mlp

        return hidden_states

class OlalaPreTrainedModel(PreTrainedModel):
    config: OlalaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OlalaMonoBlock"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": OlalaMonoBlock,
        "attentions": OlalaMonoBlock,
    }

    def widen_geodesic_scalars(self) -> int:
        """Reshape every 0-dim GeodesicNorm scale/bias to [1]; returns the count.

        FSDP1 and FSDP2 both refuse 0-dim parameters ("fully_shard doesn't
        support scalar parameters") -- sharding needs a dim 0 to shard along --
        but every Olala export stores these as 0-dim scalars. Widening here,
        after the weights are in, keeps the load itself mismatch-free: declaring
        [1] in __init__ makes transformers record a shape mismatch and then
        REINITIALIZE those keys, destroying the trained values.

        Numerically identical: the only use is ``theta * scale + bias`` with
        theta of shape (..., 1), which broadcasts the same either way.

        from_pretrained() calls this automatically. Call it yourself if you
        build the model some other way (from_config, or a manual state_dict
        restore) and intend to wrap it in FSDP.
        """
        widened = 0
        for module in self.modules():
            if not isinstance(module, OlalaGeodesicNorm):
                continue
            for name in OlalaGeodesicNorm._SCALARS:
                param = getattr(module, name, None)
                if isinstance(param, nn.Parameter) and param.dim() == 0:
                    setattr(module, name,
                            nn.Parameter(param.data.reshape(1),
                                         requires_grad=param.requires_grad))
                    widened += 1
        return widened

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        # Only meaningful on a real module tree (not a meta/quantized shell).
        if isinstance(model, nn.Module):
            model.widen_geodesic_scalars()
        return model

    def save_pretrained(self, *args, state_dict=None, **kwargs):
        """Write GeodesicNorm scale/bias back out 0-dim.

        The widening above is an in-MEMORY concern (FSDP), and 0-dim is the
        canonical on-disk shape -- every export and the vLLM loader assume it.
        Leaving [1] in the file would re-create the corruption this whole dance
        avoids, just mirrored: a [1] checkpoint against the 0-dim declaration in
        __init__ is a recorded mismatch, and transformers REINITIALIZES
        mismatched keys before any hook can intervene.

        It is also the only shape vLLM can take. Its OlalaGeodesicNorm declares
        [1] and loads with copy_, which broadcasts a 0-dim tensor up to [1] but
        NOT a [1] tensor down to 0-dim -- so 0-dim on disk loads everywhere,
        while [1] on disk loads only where the parameter is already [1].
        """
        if state_dict is None:
            state_dict = self.state_dict()
        squeezed = dict(state_dict)
        for key, value in state_dict.items():
            if (".geodesic_" in key
                    and key.rsplit(".", 1)[-1] in OlalaGeodesicNorm._SCALARS
                    and getattr(value, "shape", None) == (1,)):
                squeezed[key] = value.reshape(())
        return super().save_pretrained(*args, state_dict=squeezed, **kwargs)

@dataclass
class OlalaOutput(ModelOutput):
    """
    Class for the Olala model outputs.
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        cache_params (`HybridOlalaDynamicCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
            Includes both the RNN-like state matrices after the selective scan, and the conv states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[HybridOlalaDynamicCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class OlalaCausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cache_params (`HybridOlalaDynamicCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    loss: Optional[torch.FloatTensor] = None
    geo_loss: Optional[torch.FloatTensor] = None
    ce_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[HybridOlalaDynamicCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

class OlalaModel(OlalaPreTrainedModel):
    def __init__(self, config: OlalaConfig):
        super().__init__(config)
        self.config: OlalaConfig = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        _check_supported_config(config)

        if config.use_value_embedding:
            layers_ve_flags = [c == "1" for c in config.layers_ve_config]
            assert len(layers_ve_flags) == len(config.layers_config)
            self.value_embedding = nn.ModuleList()
            self.value_embedding_map = []
            for use_ve, layer_type in zip(layers_ve_flags, config.layers_config):
                if not use_ve:
                    self.value_embedding_map.append(-1)
                    continue
                out_dim = 2 * config.hidden_size # d_inner of an 'M' layer
                self.value_embedding_map.append(len(self.value_embedding))
                self.value_embedding.append(nn.Embedding(config.vocab_size, out_dim, self.padding_idx))

        layers_mlp_config = config.layers_mlp_config
        if self.config.layers_mlp_config == '':
            if self.config.moe:
                layers_mlp_config = 'm' * len(config.layers_config)
            else:
                layers_mlp_config = 'd' * len(config.layers_config)
        assert len(layers_mlp_config) == len(config.layers_config)

        layers_ve_config = config.layers_ve_config if config.use_value_embedding else "0" * len(config.layers_config)
        assert len(layers_ve_config) == len(config.layers_config)
        self.layers = nn.ModuleList([
            OlalaMonoBlock(config, layer_idx=i, layer_type=layer, use_ve=ve == "1", mlp_type=mlp_type)
            for i, (layer, ve, mlp_type) in enumerate(zip(config.layers_config, layers_ve_config, layers_mlp_config))
        ])

        # See the padding note in forward(); the check runs on the first batch
        # only, padding side being a property of the collator.
        self._padding_checked = False

        if self.config.final_norm:
            self.final_norm = OlalaNorm(config, config.hidden_size)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, new_embeddings):
        self.embedding = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[HybridOlalaDynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        **kwargs
    ) -> OlalaOutput:
        B, L = input_ids.shape if input_ids is not None else inputs_embeds.shape[:2]
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        if self.config.normalize_embeddings:
            inputs_embeds = F.normalize(inputs_embeds, dim=-1) * math.sqrt(self.config.hidden_size)

        if self.config.normalize_embeddings_ngpt:
            inputs_embeds = F.normalize(inputs_embeds, dim=-1)

        if self.config.patch_level_training:
            # (B, KL, D) => (B, L, D) OR (B, L, D) ==> (B, L//K, D)
            inputs_embeds = inputs_embeds.reshape(B, L//self.config.patch_level_training_size, self.config.patch_level_training_size, inputs_embeds.size(2)).mean(dim=2)

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if use_cache:
            if past_key_values is None:
                past_key_values = HybridOlalaDynamicCache(self.config)
            elif not isinstance(past_key_values, HybridOlalaDynamicCache):
                if type(past_key_values) is DynamicCache:
                    del past_key_values
                    past_key_values = HybridOlalaDynamicCache(self.config)
                else:
                    raise TypeError(f"Unsupported cache type: {type(past_key_values)}")

        hidden_states = inputs_embeds

        if cache_position is None:
            cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

            if self.config.patch_level_training:
                position_ids = position_ids[:, 0:L//self.config.patch_level_training_size]

        # ---- padding-free packing ------------------------------------
        # Fold a padded batch into ONE varlen row holding only the real tokens,
        # with cu_seqlens marking the rows. Every mixer here honours those
        # boundaries (mamba3 takes cu_seqlens, attention takes the flash
        # varlen path), so the result is exactly a per-row forward -- padding
        # side becomes irrelevant, ragged included, and no compute is spent on
        # pads. This is the only way this model handles LEFT padding at all:
        # it has no attention-mask plumbing, so a pad left in place pollutes
        # both the attention softmax and the SSM recurrence.
        #
        # Skipped when anything would break, and then the right-padding check
        # below still applies:
        #   use_cache      decode appends one token per ROW into a dense
        #                  (B, L, h, D) KV cache, which a packed row cannot
        #                  extend -- generation needs its own masking instead
        #   cu_seqlens     caller already packed; its offsets are in padded
        #                  coordinates and would not survive the gather
        #   ATTN_IMPL      eager/flex have no varlen path (see the mixers)
        unpadded = False
        if (attention_mask is not None and attention_mask.dim() == 2
                and cu_seqlens is None and not use_cache
                and not output_hidden_states
                and not self.config.patch_level_training
                and ATTN_IMPL in ("fa2", "fa3") and hidden_states.is_cuda
                and not bool(attention_mask.all())):
            _mask = attention_mask.bool()
            _keep = _mask.reshape(-1).nonzero().flatten()
            _lengths = _mask.sum(-1)
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])[
                _keep
            ].unsqueeze(0)
            # Position of each kept token WITHIN its row, so left padding does
            # not shift anything.
            position_ids = (_mask.cumsum(-1) - 1).reshape(-1)[_keep].unsqueeze(0)
            cache_position = position_ids[0]
            cu_seqlens = F.pad(_lengths.cumsum(0), (1, 0)).to(torch.int32)
            max_seqlen = int(_lengths.max())
            if input_ids is not None:
                input_ids = input_ids.reshape(-1)[_keep].unsqueeze(0)
            unpadded = True

        if (not unpadded and attention_mask is not None
                and attention_mask.dim() == 2 and not self._padding_checked):
            # Could not pack, so the pads stay in place: only RIGHT padding is
            # then harmless (causality keeps real tokens from seeing it, and
            # the pads' own garbage output is dropped by the loss).
            self._padding_checked = True
            lengths = attention_mask.sum(-1)
            right = (torch.arange(attention_mask.shape[1], device=attention_mask.device)[None, :]
                     < lengths[:, None])
            if not torch.equal(attention_mask.bool(), right):
                raise ValueError(
                    "attention_mask is not right-padded, and this batch could not be "
                    "packed into a varlen row (see the conditions in "
                    "OlalaModel.forward). Olala has no attention-mask plumbing: it "
                    "relies on causality, which only holds when the padding is at the "
                    "end of each row. Use padding_side='right', pack the batch "
                    "yourself and pass cu_seqlens, or generate one sequence at a time."
                )

        # ---- packed-row document boundaries ---------------------------
        # Training frameworks with use_remove_padding (verl dynamic batching)
        # feed one packed (1, T) row holding many documents, position_ids
        # resetting to 0 at each document start. Without cu_seqlens the
        # attention layers run DENSE flash attention over the whole row and the
        # mamba3 mixers carry SSM state across documents: a training/inference
        # logprob mismatch (rollout_probs_diff blowup) whenever packing is on.
        # Derive cu_seqlens once here and hand it down; every mixer honours it.
        # An externally provided cu_seqlens is normalised instead: verl passes
        # its packed offsets as int64 and never max_seqlen, flash_attn wants
        # int32 and both. (Restored from the pre-rename modeling, where verl
        # training was validated; the rebase had dropped it.)
        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.to(device=hidden_states.device, dtype=torch.int32)
            if max_seqlen is None:
                max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
        elif past_key_values is None and B == 1 and position_ids is not None:
            cu_seqlens, max_seqlen = _packed_doc_cu_seqlens(position_ids, hidden_states.device)

        all_hidden_states = () if output_hidden_states else None

        for i, block in enumerate(self.layers):
            ve_i = None
            if self.config.use_value_embedding:
                j = self.value_embedding_map[i]
                if j != -1:
                    ve_i = self.value_embedding[j](input_ids)

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states = block(
                hidden_states,
                position_ids=position_ids,
                cache_params=past_key_values,
                cache_position=cache_position,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                unpadded=unpadded,
                ve=ve_i,
                **kwargs,
            )

        if unpadded:
            # Scatter the packed row back into (B, L, D); pad positions stay
            # zero and are dropped by the loss / ignored by the caller.
            _flat = hidden_states.new_zeros(B * L, hidden_states.shape[-1])
            _flat[_keep] = hidden_states[0]
            hidden_states = _flat.view(B, L, -1)

        if self.config.final_norm:
            hidden_states = self.final_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if past_key_values and not past_key_values.has_previous_state:
            past_key_values.has_previous_state = True

        return OlalaOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
        )
OlalaModel.register_for_auto_class("AutoModel")

# HuggingFace/TRL convention for "no loss here". Megatron-side data marks the
# same positions with pad_token_id; forward() normalises both to this.
IGNORE_INDEX = -100


class OlalaForCausalLM(OlalaPreTrainedModel, GenerationMixin):
    def __init__(self, config: OlalaConfig):
        super().__init__(config)
        self.config = config
        self.model = OlalaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = OlalaLinear(config, config.hidden_size, config.vocab_size, bias=False)
        if config.normalize_lm_head:
            self.temperature = nn.Parameter(torch.tensor(math.log(math.sqrt(config.hidden_size))))
        if config.logits_scaling_ngpt:
            self.temperature_ngpt = nn.Parameter(torch.ones(config.vocab_size))
        self.post_init()
        if config.tie_lm_head:
            self.lm_head.weight = self.model.embedding.weight

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[HybridOlalaDynamicCache] = None,
        cache_position: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
        just_loss: Optional[bool] = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        token_type_ids=None,
        num_items_in_batch: Optional[Union[torch.Tensor, int]] = None,
        **kwargs,
    ) -> OlalaCausalLMOutput:
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)

        outputs: OlalaOutput = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
            cache_position=cache_position,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        logits = None
        loss = None
        geo_loss = None
        ce_loss = None
        cosine_sims = None
        if labels is not None:
            labels = labels.to(hidden_states.device)
            # Two label conventions reach this model: pretraining data marks
            # dead positions with pad_token_id, HF/TRL marks them with -100.
            # Normalise to -100 so every branch below agrees -- and so that
            # cross_entropy does not trip over a negative class index.
            # NOTE: this also makes the fused (cut_cross_entropy) path ignore
            # pad_token_id, which the unfused path already did; a config switch
            # should not change what the loss counts.
            labels = labels.masked_fill(labels == self.model.padding_idx, IGNORE_INDEX)

            # --- Step 1: compute logits (3 paths) ---
            if self.config.normalize_lm_head:
                h = hidden_states.to(self.lm_head.weight.dtype)[:, slice_indices, :]
                w_norm = F.normalize(self.lm_head.weight, dim=-1)
                x_norm = F.normalize(h, dim=-1)
                cosine_sims = x_norm @ w_norm.T  # (B, T, V)
                #print(f"temperature: {self.temperature.exp().item():.4f}, log_temp: {self.temperature.item():.4f}")
                logits = (self.temperature.clamp(min=-2.0, max=5.0).exp() * cosine_sims).float()
            elif linear_cross_entropy is None or not self.config.fused_loss_computation:
                logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)[:, slice_indices, :]).float()
                if self.config.logits_scaling_ngpt:
                    logits = logits * self.temperature_ngpt

            # --- Step 2: CE loss ---
            if logits is not None:
                if not self.config.patch_level_training:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=IGNORE_INDEX)
                else:
                    shift_logits = logits[..., :-1, :].reshape(-1, self.config.vocab_size)
                    shift_labels = labels[..., self.config.patch_level_training_size:].reshape(-1, self.config.patch_level_training_size)
                    loss = 0
                    log_probs = F.log_softmax(shift_logits, dim=-1)
                    for i in range(self.config.patch_level_training_size):
                        loss = loss + F.nll_loss(log_probs, shift_labels[:, i], ignore_index=IGNORE_INDEX)
                    loss = loss / self.config.patch_level_training_size
            else:
                assert not self.config.patch_level_training, "Fused loss computation is not supported with patch-level training."
                loss = linear_cross_entropy(
                    hidden_states[:, slice_indices, :].view(-1, hidden_states.size(-1)),
                    self.lm_head.weight,
                    labels.view(-1),
                    impl="cce_exact",
                    shift=1,
                    ignore_index=IGNORE_INDEX,
                )

            # --- Step 2b: gradient-accumulation-safe normalisation ---
            # transformers' Trainer sees **kwargs on this forward, concludes the
            # model handles loss kwargs, and therefore passes the GLOBAL token
            # count and does NOT divide by gradient_accumulation_steps: it
            # expects sum/num_items_in_batch, not a per-microbatch mean. Convert
            # the mean back (mean * n_valid == sum). Without this, accumulated
            # steps are scaled by ~gradient_accumulation_steps and short
            # microbatches are over-weighted.
            # n_valid is counted exactly as the Trainer counts num_items_in_batch:
            # over labels[..., 1:], since the loss shifts.
            if (num_items_in_batch is not None and loss is not None
                    and not self.config.patch_level_training):
                n_valid = (labels[..., 1:] != IGNORE_INDEX).sum()
                loss = loss * (n_valid / num_items_in_batch)

            # --- Step 3: geodesic loss ---
            if self.config.geo_loss_coeff > 0:
                shift_targets = labels[..., 1:].contiguous()
                mask = shift_targets != IGNORE_INDEX
                if cosine_sims is not None:
                    # reuse cosine similarities from normalized path (no temperature)
                    flat_cos = cosine_sims[..., :-1, :].contiguous().view(-1, self.config.vocab_size)
                    idx = torch.arange(shift_targets.numel(), device=shift_targets.device)
                    cos_sim_target = flat_cos[idx, shift_targets.view(-1)].view_as(shift_targets)
                else:
                    shift_hidden = hidden_states[:, :-1, :].contiguous()
                    target_embeds = self.model.embedding.weight[shift_targets]
                    dot = (shift_hidden * target_embeds).sum(dim=-1)
                    cos_sim_target = dot / (shift_hidden.norm(dim=-1).clamp(min=1e-8) * target_embeds.norm(dim=-1).clamp(min=1e-8))
                geo_loss = (1 - cos_sim_target).masked_fill(~mask, 0.0).sum() / mask.sum().clamp(min=1)
                ce_loss = loss.detach()
                loss = loss + self.config.geo_loss_coeff * geo_loss
        else:
            if self.config.normalize_lm_head:
                h = hidden_states.to(self.lm_head.weight.dtype)[:, slice_indices, :]
                w_norm = F.normalize(self.lm_head.weight, dim=-1)
                x_norm = F.normalize(h, dim=-1)
                logits = (self.temperature.exp() * (x_norm @ w_norm.T)).float()
            else:
                logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)[:, slice_indices, :]).float()

        return OlalaCausalLMOutput(
            loss=loss,
            geo_loss=geo_loss,
            ce_loss=ce_loss,
            logits=logits if not just_loss else None,
            past_key_values=outputs.past_key_values if not just_loss else None,
            hidden_states=outputs.hidden_states if not just_loss else None,
        )

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

OlalaForCausalLM.register_for_auto_class("AutoModelForCausalLM")

__all__ = ["OlalaModel", "OlalaForCausalLM", "OlalaPreTrainedModel"]
