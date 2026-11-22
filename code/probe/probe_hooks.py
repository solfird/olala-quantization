"""Forward hooks of the activation probe and of the ablations, on the vendored OVH modeling code.

Layer numbers are **0-based** everywhere (``block.layer_idx``; the attention layers of the real model are 4, 9, ...,
34). The sites recorded per layer ``l`` are:

=================  ==========================================================================================
``mixer_out``      output of ``block.mixer_proj``: the mixer branch, i.e. the ``g`` that enters the geodesic
                   combine (this is what Appendix C calls the "branch output")
``moe_out``        output of ``block.mlp`` (routed + shared experts): the MoE branch ``g``
``shared_out``     output of ``block.mlp.shared_experts`` (the always-on expert)
``routed_out``     output of ``block.mlp.up_proj``: the routed-experts branch after the latent up-projection
``routed_latent``  output of the routed experts before ``up_proj`` (384 channels in the real model)
``moe_pre``        pre-activation ``W_in x`` of every (token, expert) pair, before the squared ReLU
``moe_post``       ``relu(W_in x) ** 2`` of every (token, expert) pair: the input of ``W_out``
``resid_out``      output of the block: the residual stream after the layer
=================  ==========================================================================================

plus, for Mamba-3 layers, the per-head outputs (the input of ``mixer_proj``) and the sibling-erasure test: the share of
``mixer_proj`` output elements that are bit-identical when every head but the largest one is zeroed.

The hooks never change what they observe. They must be registered AFTER any hook that edits an activation (a planted
spike in a test), because PyTorch runs hooks in registration order and passes the edited output on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

import probe_stats as ps

TOKEN_LIST_KINDS = frozenset({"mixer_out", "moe_out", "moe_post"})
SITE_KINDS = ("mixer_out", "moe_out", "shared_out", "routed_out", "routed_latent", "moe_pre", "moe_post", "resid_out")
FIRST_POSITIONS_EXCLUDED = 10  # OVH: the non-identical residue "is confined to the first ~10 positions"
_INT_VIEW = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}


@dataclass
class _Ctx:
    prompt: int
    tokens: np.ndarray
    special: np.ndarray


class ExpertAcc:
    """Which experts produce the largest post-activation values (pairs of one token and one expert)."""

    def __init__(self, layer: int, n_experts: int, top_pairs: int) -> None:
        """Create an empty accumulator for a layer with ``n_experts`` routed experts."""
        self.layer, self.n_experts, self.top_k = layer, n_experts, top_pairs
        self.max_post = np.zeros(n_experts)
        self.max_pre = np.zeros(n_experts)
        self.n_pairs = np.zeros(n_experts, dtype=np.int64)
        self.n_gt_448 = np.zeros(n_experts, dtype=np.int64)
        self.hist_top = np.zeros(n_experts, dtype=np.int64)
        self.top: list[tuple[float, int, int, int, int]] = []  # value, expert, prompt, position, token id

    def add(self, pre: torch.Tensor, post: torch.Tensor, tok: torch.Tensor, exp: torch.Tensor, ctx: _Ctx) -> None:
        """Fold in the pairs of one forward pass (``pre`` / ``post`` are ``[pairs, hidden]``)."""

        def pair_max(t: torch.Tensor) -> torch.Tensor:
            a = t.detach().float().abs()
            return torch.where(torch.isfinite(a), a, torch.zeros_like(a)).max(dim=1).values

        pm_post, pm_pre = pair_max(post), pair_max(pre)
        e = self.n_experts
        zeros = torch.zeros(e, dtype=pm_post.dtype, device=pm_post.device)
        self.max_post = np.maximum(self.max_post, zeros.scatter_reduce(0, exp, pm_post, "amax").double().cpu().numpy())
        self.max_pre = np.maximum(self.max_pre, zeros.scatter_reduce(0, exp, pm_pre, "amax").double().cpu().numpy())
        self.n_pairs += torch.bincount(exp, minlength=e).cpu().numpy()
        self.n_gt_448 += torch.bincount(exp[pm_post > ps.E4M3_MAX], minlength=e).cpu().numpy()
        k = min(self.top_k, pm_post.numel())
        vals, rows = pm_post.topk(k)
        for v, r in zip(vals.tolist(), rows.tolist(), strict=True):
            ex, pos = int(exp[r]), int(tok[r])
            self.hist_top[ex] += 1
            self.top.append((v, ex, ctx.prompt, pos, int(ctx.tokens[pos])))
        self.top = sorted(self.top, key=lambda r: -r[0])[: self.top_k]

    def result(self) -> dict[str, Any]:
        """Per-expert maxima and the expert histogram of the largest pairs, JSON-ready."""
        order = np.argsort(-self.hist_top, kind="stable")
        hist = [
            {
                "expert": int(j),
                "count": int(self.hist_top[j]),
                "max_post": float(self.max_post[j]),
                "n_pairs_post_gt_448": int(self.n_gt_448[j]),
            }
            for j in order
            if self.hist_top[j] > 0
        ][:16]
        total_gt = int(self.n_gt_448.sum())
        return {
            "layer": self.layer,
            "top_expert_by_post": int(self.max_post.argmax()),
            "max_post_per_expert": self.max_post.tolist(),
            "max_pre_per_expert": self.max_pre.tolist(),
            "n_pairs_per_expert": self.n_pairs.tolist(),
            "n_pairs_post_gt_448": total_gt,
            "n_experts_with_post_gt_448": int((self.n_gt_448 > 0).sum()),
            "expert_hist_top_pairs": hist,
            "top_pairs": [
                {"value": v, "expert": ex, "prompt": pr, "position": pos, "token_id": tid}
                for v, ex, pr, pos, tid in self.top
            ],
        }


class ErasureAcc:
    """Share of ``mixer_proj`` outputs that are bit-identical when all heads but the largest are removed."""

    def __init__(self, layer: int) -> None:
        """Create an empty accumulator."""
        self.layer = layer
        self.n_equal = self.n_total = self.n_equal_late = self.n_total_late = 0
        self.heads: list[int] = []

    def add(self, full: torch.Tensor, keep: torch.Tensor, head: int) -> None:
        """Compare the two projections bit for bit (``[1, L, D]`` each)."""
        view = _INT_VIEW.get(full.dtype)
        a, b = (full.contiguous().view(view), keep.contiguous().view(view)) if view else (full, keep)
        same = a == b
        self.n_equal += int(same.sum())
        self.n_total += same.numel()
        late = same[:, FIRST_POSITIONS_EXCLUDED:]
        self.n_equal_late += int(late.sum())
        self.n_total_late += late.numel()
        self.heads.append(head)

    def result(self) -> dict[str, Any]:
        """The shares, JSON-ready."""
        top = max(set(self.heads), key=self.heads.count) if self.heads else None
        return {
            "layer": self.layer,
            "fraction_bit_identical": self.n_equal / max(self.n_total, 1),
            "fraction_bit_identical_pos_ge_10": self.n_equal_late / max(self.n_total_late, 1),
            "n_total": self.n_total,
            "kept_head_most_common": top,
            "kept_head_share_of_prompts": (self.heads.count(top) / len(self.heads)) if self.heads else None,
        }


@dataclass
class _Moe:
    """Scratch state of one expert-loop call (mirrors ``TorchMoEMLP.forward`` in modeling_olala.py)."""

    seq: list[int] = field(default_factory=list)
    token_of: torch.Tensor | None = None
    ptr: int = 0
    start: int = 0
    pre: list[torch.Tensor] = field(default_factory=list)
    post: list[torch.Tensor] = field(default_factory=list)
    tok: list[torch.Tensor] = field(default_factory=list)
    exp: list[torch.Tensor] = field(default_factory=list)


@dataclass
class _LayerRec:
    layer: int
    kind: str  # "M" or "V"
    sites: dict[str, ps.SiteAcc]
    heads: ps.HeadAcc | None = None
    experts: ExpertAcc | None = None
    erasure: ErasureAcc | None = None
    moe: _Moe = field(default_factory=_Moe)


class Probe:
    """Attach recording hooks to every layer of an ``OlalaForCausalLM``; call :meth:`begin_prompt` before each pass."""

    def __init__(self, model: Any, max_len: int, top_pairs: int = 32, top_tokens: int = 64) -> None:
        """Attach the hooks.

        Raises:
            RuntimeError: If a layer's expert module is not the pure-PyTorch ``TorchMoEMLP`` (the only one whose
                (token, expert) rows can be observed; ``olala_model.load_olala_modules`` forces it).
        """
        self.model = model
        self.max_len = max_len
        self._ctx: _Ctx | None = None
        self._busy = False
        self._handles: list[Any] = []
        self._restore: list[Any] = []
        self._layers: list[_LayerRec] = []
        for blk in model.model.layers:
            self._attach(blk, top_pairs, top_tokens)

    # ------------------------------------------------------------------ public
    def begin_prompt(self, prompt: int, tokens: np.ndarray, special: np.ndarray) -> None:
        """Declare the prompt of the next forward pass (its index, token ids and special-token flags)."""
        self._ctx = _Ctx(prompt, np.asarray(tokens), np.asarray(special, dtype=bool))

    def close(self) -> None:
        """Remove every hook and restore the experts' activation function."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        for fn in self._restore:
            fn()
        self._restore.clear()
        self._ctx = None

    def result(self) -> dict[str, Any]:
        """Everything recorded so far, one entry per layer (0-based ``layer``)."""
        rows = []
        for rec in sorted(self._layers, key=lambda r: r.layer):
            rows.append(
                {
                    "layer": rec.layer,
                    "type": rec.kind,
                    "sites": {k: acc.result() for k, acc in rec.sites.items()},
                    "heads": rec.heads.result() if rec.heads else None,
                    "mixer_erasure": rec.erasure.result() if rec.erasure else None,
                    "experts": rec.experts.result() if rec.experts else None,
                }
            )
        return {"layers": rows}

    # ------------------------------------------------------------------ attach
    def _attach(self, blk: Any, top_pairs: int, top_tokens: int) -> None:
        layer = int(blk.layer_idx)
        is_mamba = hasattr(blk.mixer, "nheads")
        mlp = blk.mlp
        has_experts = hasattr(mlp, "experts")
        if has_experts and not hasattr(mlp.experts, "activation"):
            raise RuntimeError(
                f"layer {layer}: the routed experts are {type(mlp.experts).__name__}, not the pure-PyTorch TorchMoEMLP "
                "(set OLALA_TORCH_MOE=1 before importing the modeling file)"
            )
        kinds = [k for k in SITE_KINDS if has_experts or k in ("mixer_out", "moe_out", "resid_out")]
        sites = {k: ps.SiteAcc(layer, k, self.max_len, top_n=top_tokens if k in TOKEN_LIST_KINDS else 0) for k in kinds}
        rec = _LayerRec(layer, "M" if is_mamba else "V", sites)
        if is_mamba:
            rec.heads = ps.HeadAcc(layer, blk.mixer.nheads, self.max_len, blk.mixer.headdim)
            rec.erasure = ErasureAcc(layer)
        if has_experts:
            rec.experts = ExpertAcc(layer, mlp.experts.num_experts, top_pairs)
        self._layers.append(rec)
        add = self._handles.append
        add(blk.register_forward_hook(lambda _m, _a, out: self._site(rec, "resid_out", out)))
        add(blk.mixer_proj.register_forward_hook(lambda _m, _a, out: self._site(rec, "mixer_out", out)))
        if is_mamba:
            add(blk.mixer_proj.register_forward_pre_hook(lambda m, args: self._heads(rec, m, args[0])))
        add(mlp.register_forward_hook(lambda _m, _a, out: self._site(rec, "moe_out", out)))
        if not has_experts:
            return
        if getattr(mlp, "shared_experts", None) is not None:
            add(mlp.shared_experts.register_forward_hook(lambda _m, _a, out: self._site(rec, "shared_out", out)))
        if hasattr(mlp, "up_proj"):
            add(mlp.up_proj.register_forward_hook(lambda _m, _a, out: self._site(rec, "routed_out", out)))
            add(mlp.experts.register_forward_hook(lambda _m, _a, out: self._site(rec, "routed_latent", out)))
        self._wrap_experts(rec, mlp.experts)

    # ------------------------------------------------------------------ hook bodies
    def _live(self) -> bool:
        return self._ctx is not None and not self._busy

    def _site(self, rec: _LayerRec, kind: str, out: torch.Tensor) -> None:
        if not self._live() or kind not in rec.sites:
            return
        ctx = self._ctx
        rec.sites[kind].add(ps.summarize(out), ctx.prompt, ctx.tokens, ctx.special)

    def _heads(self, rec: _LayerRec, proj: Any, y: torch.Tensor) -> None:
        if not self._live():
            return
        ctx = self._ctx
        n_heads = rec.heads.n_heads
        hs = ps.head_summary(y, n_heads)
        rec.heads.add(hs, ctx.prompt, ctx.tokens, ctx.special)
        top = int(hs.head_max.argmax())
        width = y.shape[-1] // n_heads
        keep = torch.zeros_like(y)
        keep[..., top * width : (top + 1) * width] = y[..., top * width : (top + 1) * width]
        self._busy = True
        try:
            full_out, keep_out = proj(y), proj(keep)
        finally:
            self._busy = False
        rec.erasure.add(full_out, keep_out, top)

    def _wrap_experts(self, rec: _LayerRec, experts: Any) -> None:
        orig = experts.activation
        state = rec.moe

        def activation(x: torch.Tensor) -> torch.Tensor:
            out = orig(x)
            if state.token_of is not None:
                n = x.shape[0]
                state.pre.append(x.detach())
                state.post.append(out.detach())
                state.tok.append(state.token_of[state.start : state.start + n])
                state.exp.append(torch.full((n,), state.seq[state.ptr], dtype=torch.long, device=x.device))
                state.ptr += 1
                state.start += n
            return out

        def pre_hook(_m: Any, args: tuple[Any, ...]) -> None:
            if not self._live():
                return
            idx = args[2]
            flat = idx.reshape(-1)
            order = torch.argsort(flat, stable=True)  # the same grouping as TorchMoEMLP.forward
            counts = torch.bincount(flat, minlength=experts.num_experts).tolist()
            state.token_of = order // idx.shape[-1]
            state.seq = [e for e, n in enumerate(counts) if n > 0]
            state.ptr = state.start = 0
            state.pre, state.post, state.tok, state.exp = [], [], [], []

        def post_hook(_m: Any, args: tuple[Any, ...], _out: Any) -> None:
            if state.token_of is None:
                return
            ctx = self._ctx
            num = int(args[0].shape[0])
            pre, post = torch.cat(state.pre), torch.cat(state.post)
            tok, exp = torch.cat(state.tok), torch.cat(state.exp)
            state.token_of = None
            rec.sites["moe_pre"].add(
                ps.summarize(pre, token_of=tok, num_tokens=num), ctx.prompt, ctx.tokens, ctx.special
            )
            rec.sites["moe_post"].add(
                ps.summarize(post, token_of=tok, num_tokens=num), ctx.prompt, ctx.tokens, ctx.special
            )
            rec.experts.add(pre, post, tok, exp, ctx)

        experts.activation = activation
        self._restore.append(lambda: setattr(experts, "activation", orig))
        self._handles.append(experts.register_forward_pre_hook(pre_hook))
        self._handles.append(experts.register_forward_hook(post_hook))


# ---------------------------------------------------------------------------------------------------- ablations
BRANCHES = ("mixer_out", "moe_out", "routed_out", "shared_out", "head")
MODES = ("zero", "clamp448", "noop")


@dataclass(frozen=True)
class AblationSpec:
    """One edit of one branch: ``layer`` is 0-based; ``head`` is only used with ``branch == "head"``."""

    layer: int
    branch: str
    mode: str
    head: int | None = None

    def label(self) -> str:
        """A short readable name, e.g. ``L1.mixer_out:zero`` or ``L1.head23:zero``."""
        b = f"head{self.head}" if self.branch == "head" else self.branch
        return f"L{self.layer}.{b}:{self.mode}"


def _edit(out: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "zero":
        return torch.zeros_like(out)
    if mode == "clamp448":
        return out.clamp(-ps.E4M3_MAX, ps.E4M3_MAX)
    if mode == "noop":
        return out
    raise ValueError(f"unknown ablation mode {mode!r}")


def install_ablation(model: Any, spec: AblationSpec) -> list[Any]:
    """Register the hook that applies ``spec``; returns the handles (call ``.remove()`` on each).

    ``zero`` replaces the branch by zeros, ``clamp448`` clamps every element to [-448, 448], ``noop`` returns it
    unchanged (a control that must reproduce the baseline exactly). For ``branch == "head"`` only the channels of one
    Mamba-3 head are edited, in the mixer output before ``mixer_proj``.
    """
    if spec.branch not in BRANCHES:
        raise ValueError(f"unknown branch {spec.branch!r}")
    if spec.mode not in MODES:
        raise ValueError(f"unknown ablation mode {spec.mode!r}")
    blk = next((b for b in model.model.layers if int(b.layer_idx) == spec.layer), None)
    if blk is None:
        raise ValueError(f"no layer {spec.layer}")
    if spec.branch == "mixer_out":
        return [blk.mixer_proj.register_forward_hook(lambda _m, _a, out: _edit(out, spec.mode))]
    if spec.branch == "moe_out":
        return [blk.mlp.register_forward_hook(lambda _m, _a, out: _edit(out, spec.mode))]
    if spec.branch == "routed_out":
        return [blk.mlp.up_proj.register_forward_hook(lambda _m, _a, out: _edit(out, spec.mode))]
    if spec.branch == "shared_out":
        return [blk.mlp.shared_experts.register_forward_hook(lambda _m, _a, out: _edit(out, spec.mode))]
    if spec.head is None or not hasattr(blk.mixer, "nheads"):
        raise ValueError("a head ablation needs a Mamba-3 layer and a head index")
    width = blk.mixer.headdim
    sl = slice(spec.head * width, (spec.head + 1) * width)

    def head_hook(_m: Any, _a: Any, out: Any) -> Any:
        y = out[0].clone()
        y[..., sl] = _edit(y[..., sl], spec.mode)
        return (y, *out[1:])

    return [blk.mixer.register_forward_hook(head_hook)]
