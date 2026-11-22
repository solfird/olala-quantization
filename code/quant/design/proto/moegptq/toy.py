"""A random fused non-gated MoE layer with the real structure at toy size, and its token generator.

Structure copied from ``modeling_olala.py`` (fused ``[E, out, in]`` experts, ``relu(x)^2`` activation with
no gate matrix, sigmoid router with a selection-only bias, renormalised top-k scaled by 2.45). Sizes are
scaled down (real: 256 experts, latent 384, intermediate 768, top-6). Activations are heavy tailed
(Student-t) and correlated (a decaying spectrum), because GPTQ can only help when the Hessian is not
close to a multiple of the identity; the choice of that correlation is the toy's main assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from moegptq.routing import route


@dataclass(frozen=True)
class ToyConfig:
    """Shape and statistics of the toy layer."""

    n_experts: int = 16
    d: int = 256
    h: int = 512
    top_k: int = 2
    scale: float = 2.45
    t_df: int = 4
    spectrum_alpha: float = 0.5
    ch_sigma: float = 0.5
    bias_sigma: float = 0.15
    rare_experts: int = 2
    rare_bias: float = -0.3
    spike_channel: int | None = None
    spike_mode: str = "persistent"
    spike_factor: float = 1.0
    spike_frac: float = 0.05
    n_clusters: int = 0  # 0: tokens are one population; else tokens come from clusters (must equal n_experts)
    cluster_strength: float = 1.0  # cluster mean norm = strength * sqrt(d)
    cluster_cov_sigma: float = 0.7  # per-cluster log-normal channel scaling (different Hessian per cluster)
    router_selectivity: float = 8.0  # added logit of expert c for tokens of cluster c
    cluster_prior_sigma: float = 1.0  # log-normal spread of cluster frequencies (rare clusters)
    cluster_rotate: bool = False  # each cluster has its own mixing matrix (own covariance subspace)


@dataclass
class ToyLayer:
    """Weights of the fused layer and the fixed statistics of its inputs."""

    cfg: ToyConfig
    w_up: torch.Tensor  # [E, h, d]
    w_down: torch.Tensor  # [E, d, h]
    w_gate: torch.Tensor  # [E, d]
    bias: torch.Tensor  # [E]
    mix: torch.Tensor  # [d, d] token correlation structure
    ch_scale: torch.Tensor  # [d]
    extras: dict = field(default_factory=dict)


@dataclass
class Tokens:
    """A batch of latent expert inputs and their routing."""

    x: torch.Tensor  # [N, d] float64, after the optional spike
    idx: torch.Tensor  # [N, k]
    p: torch.Tensor  # [N, k] float64
    spike_mask: torch.Tensor  # [N] bool
    cluster: torch.Tensor  # [N] cluster id, -1 when the toy has no clusters


def make_layer(cfg: ToyConfig, seed: int) -> ToyLayer:
    """Draw the layer weights, the router and the input statistics."""
    g = torch.Generator().manual_seed(seed)
    e, d, h = cfg.n_experts, cfg.d, cfg.h
    row_up = torch.exp(0.3 * torch.randn(e, h, 1, generator=g))
    row_dn = torch.exp(0.3 * torch.randn(e, d, 1, generator=g))
    exp_scale = torch.exp(0.4 * torch.randn(e, 1, 1, generator=g))
    w_up = torch.randn(e, h, d, generator=g) / d**0.5 * row_up * exp_scale
    w_down = torch.randn(e, d, h, generator=g) / h**0.5 * row_dn * exp_scale
    w_gate = torch.randn(e, d, generator=g) / d**0.5 * 2.0
    bias = cfg.bias_sigma * torch.randn(e, generator=g)
    if cfg.rare_experts:
        bias[: cfg.rare_experts] = cfg.rare_bias
    u, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    lam = torch.arange(1, d + 1, dtype=torch.float32) ** (-cfg.spectrum_alpha)
    lam = lam * (d / (lam**2).sum()).sqrt()
    mix = u * lam
    ch = torch.exp(cfg.ch_sigma * torch.randn(d, generator=g))
    ch = ch / (ch**2).mean().sqrt()
    extras: dict = {}
    if cfg.n_clusters:
        if cfg.n_clusters != e:
            raise ValueError("n_clusters must equal n_experts (cluster c is the home of expert c)")
        mean_dir = torch.randn(e, d, generator=g)
        mean_dir = mean_dir / mean_dir.norm(dim=1, keepdim=True)
        ch_c = torch.exp(cfg.cluster_cov_sigma * torch.randn(e, d, generator=g))
        ch_c = ch_c / (ch_c**2).mean(dim=1, keepdim=True).sqrt()
        prior = torch.softmax(cfg.cluster_prior_sigma * torch.randn(e, generator=g), dim=0)
        extras = {"mean_dir": mean_dir, "ch_c": ch_c, "prior": prior}
        if cfg.cluster_rotate:
            mixes = []
            for _ in range(e):
                uc, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
                mixes.append(uc * lam)
            extras["mix_c"] = torch.stack(mixes)
    return ToyLayer(cfg, w_up, w_down, w_gate, bias, mix, ch, extras)


def sample_tokens(layer: ToyLayer, n: int, seed: int) -> Tokens:
    """Draw ``n`` tokens; route them on the un-spiked input, then apply the spike."""
    cfg = layer.cfg
    g = torch.Generator().manual_seed(seed)
    normal = torch.randn(n, cfg.d, generator=g)
    chi2 = (torch.randn(n, 1, cfg.t_df, generator=g) ** 2).sum(-1)  # one chi-square per token
    z = normal / (chi2 / cfg.t_df).sqrt() / (cfg.t_df / (cfg.t_df - 2.0)) ** 0.5  # Student-t, unit variance
    cluster = torch.full((n,), -1, dtype=torch.long)
    if cfg.n_clusters:
        cluster = torch.multinomial(layer.extras["prior"], n, replacement=True, generator=g)
        z = z * layer.extras["ch_c"][cluster]
    z = z * layer.ch_scale
    if cfg.n_clusters and cfg.cluster_rotate:
        x = torch.empty_like(z)
        for c in range(cfg.n_clusters):
            sel = cluster == c
            x[sel] = z[sel] @ layer.extras["mix_c"][c].T
    else:
        x = z @ layer.mix.T
    logits = x @ layer.w_gate.T
    if cfg.n_clusters:
        mean = layer.extras["mean_dir"]
        x = x + cfg.cluster_strength * cfg.d**0.5 * mean[cluster]
        home = cfg.router_selectivity * (x @ mean.T) / cfg.d**0.5 / (cfg.cluster_strength + 1e-9)
        logits = 0.3 * (x @ layer.w_gate.T) + home
    idx, p = route(logits, layer.bias, cfg.top_k, cfg.scale)
    mask = torch.zeros(n, dtype=torch.bool)
    x = x.double()
    if cfg.spike_channel is not None:
        if cfg.spike_mode == "persistent":
            x[:, cfg.spike_channel] *= cfg.spike_factor
            mask[:] = True
        else:
            mask = torch.rand(n, generator=g) < cfg.spike_frac
            x[mask, cfg.spike_channel] *= cfg.spike_factor
    return Tokens(x=x, idx=idx, p=p.double(), spike_mask=mask, cluster=cluster)


def expert_forward(w_up: torch.Tensor, w_down: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """One expert: ``W_down relu(W_up x)^2`` for rows of ``x`` (float64)."""
    hid = torch.relu(x @ w_up.double().T) ** 2
    return hid @ w_down.double().T


def layer_forward(
    layer: ToyLayer,
    x: torch.Tensor,
    idx: torch.Tensor,
    p: torch.Tensor,
    w_up: torch.Tensor | None = None,
    w_down: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gate-weighted sum of the selected experts, with optional replacement (quantized) weights."""
    wu = layer.w_up if w_up is None else w_up
    wd = layer.w_down if w_down is None else w_down
    y = torch.zeros(x.shape[0], layer.cfg.d, dtype=torch.float64)
    for e in range(layer.cfg.n_experts):
        tok, slot = torch.nonzero(idx == e, as_tuple=True)
        if tok.numel() == 0:
            continue
        out = expert_forward(wu[e], wd[e], x[tok])
        y.index_add_(0, tok, out * p[tok, slot].unsqueeze(1))
    return y
