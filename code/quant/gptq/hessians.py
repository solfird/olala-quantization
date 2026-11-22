"""Layer-by-layer calibration: run the HF blocks in order, capture what each expert sees, build per-expert Hessians.

Three parts.

1. :class:`MoECapture` is a forward pre-hook on ``TorchMoEMLP`` (the expert loop of ``modeling_olala.py:1448-1469``). It
   stores the exact arguments of the loop: the 384-dim latent input ``x`` and the selected expert ids. Storing the rows
   instead of accumulating ``x^T x`` inside the hook is deliberate: the token-norm clip needs the median norm over an
   expert's whole token set, the held-out guard needs the held-out rows, and the three Hessian modes
   (routed / all-token / blend) all derive from the same forward. One layer holds ``N x 384`` values (50 MB for 65,536
   tokens in bf16), never more than one layer is held, and the result is the exact sum of the design.
2. :func:`build_hessians` turns a :class:`LayerCapture` into float32 Hessians of the up-projection input (latent,
   384-dim) and the down-projection input (``relu(W_up x)^2``, 768-dim), with the guards of the design: float32
   accumulation, token-norm clipping at ``token_clip`` times the median norm, held-out WINDOWS kept out of every Hessian
   (the all-token moment included), and the minimum-token fallback. The diagonal-ratio test is applied by the solver
   (``solve.solve_experts``), which reports the ratio.
3. :func:`run_sequential` runs the blocks in order over the calibration windows, one block in memory at a time, hands
   each requested layer's capture to a handler, optionally propagates the QUANTIZED experts to the next layer, and
   saves the boundary hidden state after every layer so that a run resumes at the next layer.
"""

from __future__ import annotations

import hashlib
import json
import math
import resource
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn.functional as F
from olala_quant.checkpoint_io import SourceCheckpoint


class ResumeError(RuntimeError):
    """The saved state belongs to another run (other windows, model or settings)."""


# ------------------------------------------------------------------------------------------------ data containers


@dataclass
class LayerCapture:
    """The inputs of one layer's expert loop over all calibration windows.

    Attributes:
        x: ``[N, d]`` latent inputs, in the dtype the model produced (bf16 in the real model).
        idx: ``[N, k]`` selected expert ids.
        window: ``[N]`` index of the calibration window each token belongs to.
        n_windows: Number of calibration windows (default: ``max(window) + 1``).
    """

    x: torch.Tensor
    idx: torch.Tensor
    window: torch.Tensor
    n_windows: int | None = None

    def total_windows(self) -> int:
        """Number of windows of the calibration set."""
        return self.n_windows if self.n_windows is not None else int(self.window.max()) + 1


@dataclass(frozen=True)
class HessianConfig:
    """Settings of :func:`build_hessians`.

    Attributes:
        mode: ``routed`` (tokens routed to the expert), ``all`` (every token, llm-compressor style) or ``blend``
            (routed plus ``blend_pseudo * dim`` pseudo-tokens of the all-token mean second moment; the design default).
        blend_pseudo: Strength of the all-token prior in units of the Hessian dimension.
        token_clip: Tokens whose norm exceeds ``token_clip`` times the median norm of their set are down-weighted to
            that norm (weight ``(cap / norm)^2``); ``None`` disables the clip.
        heldout_every: Every ``heldout_every``-th window (the last of each group) is held out of every Hessian and
            kept for the held-out guard; 0 disables the hold-out.
        min_tokens: Routed calibration tokens an expert needs to be solved (default 16 for ``routed``, else 1).
    """

    mode: str = "blend"
    blend_pseudo: float = 1.0
    token_clip: float | None = 3.0
    heldout_every: int = 10
    min_tokens: int | None = None

    def effective_min_tokens(self) -> int:
        """The minimum number of routed tokens for this mode."""
        if self.min_tokens is not None:
            return self.min_tokens
        return 16 if self.mode == "routed" else 1


@dataclass
class ExpertHessians:
    """Per-expert Hessians of one layer and their bookkeeping.

    Attributes:
        h_up: ``float32 [E, d, d]`` Hessian of the up-projection input.
        h_down: ``float32 [E, m, m]`` Hessian of the down-projection input ``relu(W_up x)^2``.
        n_tokens: ``[E]`` routed tokens used (held-out windows excluded).
        n_heldout: ``[E]`` routed tokens in the held-out windows.
        held_x: Per expert, the held-out latent rows ``[n_heldout, d]``.
        clip_frac_up, clip_frac_down: ``[E]`` fraction of the expert's routed tokens that were clipped.
        active: ``[E]`` False for experts that must be rounded to nearest (too few tokens).
        inactive_reason: Reason string per expert (empty when active).
    """

    h_up: torch.Tensor
    h_down: torch.Tensor
    n_tokens: torch.Tensor
    n_heldout: torch.Tensor
    held_x: list[torch.Tensor]
    clip_frac_up: torch.Tensor
    clip_frac_down: torch.Tensor
    active: torch.Tensor
    inactive_reason: list[str] = field(default_factory=list)


# ------------------------------------------------------------------------------------------------ Hessians


def heldout_windows(n_windows: int, every: int) -> torch.Tensor:
    """``bool [n_windows]``: windows ``every-1, 2*every-1, ...`` are held out (none if ``every <= 0`` or too few)."""
    mask = torch.zeros(n_windows, dtype=torch.bool)
    if every > 0:
        mask[every - 1 :: every] = True
    return mask


def clip_weights(norms: torch.Tensor, clip: float | None) -> tuple[torch.Tensor, int]:
    """Token weights ``(cap / norm)^2`` (at most 1), ``cap = clip * median(norms)``; returns them and #clipped."""
    ones = torch.ones_like(norms, dtype=torch.float32)
    if clip is None or norms.numel() == 0:
        return ones, 0
    cap = clip * norms.median()
    w = torch.clamp(cap / norms.clamp_min(1e-30), max=1.0).square().to(torch.float32)
    return w, int((norms > cap).sum())


def _activation(x: torch.Tensor, w_up: torch.Tensor) -> torch.Tensor:
    """``relu(x W_up^T)^2`` in the dtype of ``x``, as the expert loop computes it (``modeling_olala.py:1465``)."""
    return F.relu(F.linear(x, w_up.to(x.dtype))).square()


def _gram(rows: torch.Tensor, clip: float | None) -> tuple[torch.Tensor, int, int]:
    """``sum_t w_t r_t r_t^T`` in float32 with clipped weights; returns the matrix, #clipped and #rows."""
    r = rows.to(torch.float32)
    w, n_clipped = clip_weights(r.norm(dim=1), clip)
    return (r * w[:, None]).T @ r, n_clipped, r.shape[0]


def build_hessians(
    cap: LayerCapture,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    cfg: HessianConfig,
    device: torch.device | str | None = None,
) -> ExpertHessians:
    """Build the float32 Hessians of every expert of one layer.

    Args:
        cap: The captured expert-loop inputs of the layer.
        w_up: ``[E, m, d]`` up-projection weights (the ORIGINAL bf16 weights: the down-projection input is what the
            bf16 model produces, not what a quantized up-projection would).
        w_down: ``[E, d, m]`` down-projection weights (only its shape is used; kept for symmetry and checks).
        cfg: Hessian settings.
        device: Device for the arithmetic (default: the device of ``w_up``).

    Raises:
        ValueError: For an unknown mode or inconsistent shapes.
    """
    if cfg.mode not in ("routed", "all", "blend"):
        raise ValueError(f"mode must be 'routed', 'all' or 'blend', got {cfg.mode!r}")
    n_exp, m, d = w_up.shape
    if w_down.shape != (n_exp, d, m) or cap.x.shape[1] != d:
        raise ValueError(
            f"shape mismatch: w_up {tuple(w_up.shape)}, w_down {tuple(w_down.shape)}, x {tuple(cap.x.shape)}"
        )
    dev = torch.device(device) if device is not None else w_up.device
    x = cap.x.to(dev)
    idx = cap.idx.to(dev)
    wu = w_up.to(dev)
    held_win = heldout_windows(cap.total_windows(), cfg.heldout_every).to(dev)
    held_tok = held_win[cap.window.to(dev)]
    train_tok = (~held_tok).nonzero().flatten()
    n_train = int(train_tok.numel())
    k = idx.shape[1]
    flat = idx.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=n_exp).tolist()
    token_of = order // k

    h_up = torch.zeros(n_exp, d, d, dtype=torch.float32, device=dev)
    h_down = torch.zeros(n_exp, m, m, dtype=torch.float32, device=dev)
    n_tokens = torch.zeros(n_exp, dtype=torch.long)
    n_held = torch.zeros(n_exp, dtype=torch.long)
    clip_up = torch.zeros(n_exp)
    clip_down = torch.zeros(n_exp)
    held_x: list[torch.Tensor] = []

    x_train = x[train_tok]
    if cfg.mode != "routed" and n_train > 0:
        gram_all, _, _ = _gram(x_train, cfg.token_clip)
        gram_all = gram_all / n_train
    start = 0
    for e in range(n_exp):
        tok = token_of[start : start + counts[e]]
        start += counts[e]
        is_held = held_tok[tok]
        tr, ho = tok[~is_held], tok[is_held]
        n_tokens[e], n_held[e] = tr.numel(), ho.numel()
        held_x.append(x[ho].cpu())
        use = train_tok if cfg.mode == "all" else tr
        if use.numel() > 0 and (cfg.mode == "all" or tr.numel() > 0):
            xe = x[use]
            hu, nc_u, n_u = _gram(xe, cfg.token_clip)
            hid = _activation(xe, wu[e])
            hd, nc_d, _ = _gram(hid, cfg.token_clip)
            h_up[e], h_down[e] = hu, hd
            clip_up[e], clip_down[e] = nc_u / n_u, nc_d / n_u
        if cfg.mode == "blend" and n_train > 0 and tr.numel() > 0:
            hid_all = _activation(x_train, wu[e])
            gram_d, _, _ = _gram(hid_all, cfg.token_clip)
            h_up[e] += cfg.blend_pseudo * d * gram_all
            h_down[e] += cfg.blend_pseudo * m * gram_d / n_train
    min_tokens = cfg.effective_min_tokens()
    active, reason = [], []
    for e in range(n_exp):
        n = int(n_tokens[e])
        if n == 0:
            active.append(False)
            reason.append("no routed calibration token")
        elif n < min_tokens:
            active.append(False)
            reason.append(f"fewer than {min_tokens} routed tokens")
        else:
            active.append(True)
            reason.append("")
    return ExpertHessians(
        h_up, h_down, n_tokens, n_held, held_x, clip_up, clip_down, torch.tensor(active, dtype=torch.bool), reason
    )


# ------------------------------------------------------------------------------------------------ the hook


class MoECapture:
    """Forward pre-hook on a ``TorchMoEMLP`` that stores ``(x, expert_idxs)`` of every call.

    Call :meth:`set_windows` before each forward so that tokens know their calibration window, :meth:`finish` to get the
    :class:`LayerCapture`, and :meth:`close` to remove the hook.
    """

    def __init__(self, experts_module: torch.nn.Module) -> None:
        """Attach to ``experts_module`` (the ``TorchMoEMLP``; its forward receives ``(x, expert_p, expert_idxs)``)."""
        self._xs: list[torch.Tensor] = []
        self._idx: list[torch.Tensor] = []
        self._win: list[torch.Tensor] = []
        self._windows: torch.Tensor | None = None
        self._length = 0
        self._n_windows = 0
        self._handle = experts_module.register_forward_pre_hook(self._hook)

    def set_windows(self, window_ids: torch.Tensor, length: int) -> None:
        """Declare the calibration windows (row-major) of the next forward call and their token length."""
        self._windows, self._length = window_ids, length
        self._n_windows = max(self._n_windows, int(window_ids.max()) + 1)

    def _hook(self, _module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        x, _p, idx = args[0], args[1], args[2]
        x = x.detach().reshape(-1, x.shape[-1])
        idx = idx.detach().reshape(-1, idx.shape[-1])
        if self._windows is None or x.shape[0] != self._windows.numel() * self._length:
            raise RuntimeError("MoECapture.set_windows was not called for this batch")
        self._xs.append(x.clone())
        self._idx.append(idx.clone())
        self._win.append(self._windows.to(x.device).repeat_interleave(self._length))

    def finish(self) -> LayerCapture:
        """Concatenate everything seen so far."""
        return LayerCapture(torch.cat(self._xs), torch.cat(self._idx), torch.cat(self._win), n_windows=self._n_windows)

    def close(self) -> None:
        """Remove the hook."""
        self._handle.remove()


# ------------------------------------------------------------------------------------------------ blocks and runner


def embed_windows(config: Any, embedding_weight: torch.Tensor, windows: torch.Tensor) -> torch.Tensor:
    """Embed token windows exactly as ``OlalaModel.forward`` does (lookup, then the optional normalisations)."""
    h = F.embedding(windows, embedding_weight)
    if config.normalize_embeddings:
        h = F.normalize(h, dim=-1) * math.sqrt(config.hidden_size)
    if config.normalize_embeddings_ngpt:
        h = F.normalize(h, dim=-1)
    return h


def load_block(modeling: Any, config: Any, src: SourceCheckpoint, layer: int, device: torch.device | str) -> Any:
    """Build ``OlalaMonoBlock`` number ``layer`` and load its tensors from ``src`` with their checkpoint dtypes."""
    n_layers = len(config.layers_config)
    mlp_types = config.layers_mlp_config or ("m" * n_layers if config.moe else "d" * n_layers)
    block = modeling.OlalaMonoBlock(
        config, layer_idx=layer, layer_type=config.layers_config[layer], use_ve=False, mlp_type=mlp_types[layer]
    )
    prefix = f"model.layers.{layer}."
    # copy=True: a tensor read from a memory-mapped shard shares its pages with every later read of the same tensor,
    # and the propagation step writes into the block's parameters in place.
    state = {n[len(prefix) :]: src.get_tensor(n).to(device, copy=True) for n in src.names() if n.startswith(prefix)}
    block.load_state_dict(state, strict=True, assign=True)
    return block.eval()


@dataclass
class ExpertWeights:
    """Dequantized experts handed back by a handler for propagation to the next layer."""

    w_up: torch.Tensor
    w_down: torch.Tensor


@dataclass
class LayerInfo:
    """What the runner hands the handler for one calibrated layer."""

    layer: int
    capture: LayerCapture | None
    block: Any
    forward_seconds: float
    mixer_seconds: float
    mlp_seconds: float
    max_abs_mixer_out: float
    max_abs_mlp_out: float
    peak_memory_bytes: int | None
    rss_bytes: int


class LayerHandler(Protocol):
    """What :func:`run_sequential` needs from its caller."""

    def needs_capture(self, layer: int) -> bool:
        """True if the expert inputs of ``layer`` must be captured and :meth:`process` called."""
        ...

    def process(self, info: LayerInfo) -> ExpertWeights | None:
        """Use the capture (solve, save). Return dequantized experts to propagate, or None."""
        ...


def _sync(device: torch.device | str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


class _Timers:
    """Forward hooks that time the mixer and the MLP and track the largest output magnitude."""

    def __init__(self, block: Any, device: torch.device | str) -> None:
        self.device = device
        self.t = {"mixer": 0.0, "mlp": 0.0}
        self.max_abs = {"mixer": 0.0, "mlp": 0.0}
        self._start: dict[str, float] = {}
        self._handles = []
        for key, mod in (("mixer", block.mixer), ("mlp", block.mlp)):
            self._handles.append(mod.register_forward_pre_hook(self._pre(key)))
            self._handles.append(mod.register_forward_hook(self._post(key)))
        self._handles.append(block.mixer_proj.register_forward_hook(self._peak("mixer")))
        self._handles.append(block.mlp.register_forward_hook(self._peak("mlp")))

    def _pre(self, key: str) -> Callable[..., None]:
        def hook(_m: Any, _a: Any) -> None:
            _sync(self.device)
            self._start[key] = time.perf_counter()

        return hook

    def _post(self, key: str) -> Callable[..., None]:
        def hook(_m: Any, _a: Any, _o: Any) -> None:
            _sync(self.device)
            self.t[key] += time.perf_counter() - self._start[key]

        return hook

    def _peak(self, key: str) -> Callable[..., None]:
        def hook(_m: Any, _a: Any, out: Any) -> None:
            if torch.is_tensor(out):
                self.max_abs[key] = max(self.max_abs[key], float(out.detach().abs().amax()))

        return hook

    def close(self) -> None:
        for h in self._handles:
            h.remove()


def _forward_all(
    block: Any, hidden: torch.Tensor, device: torch.device | str, batch: int, capture: MoECapture | None
) -> torch.Tensor:
    """Run ``block`` over ``hidden [n_win, L, d]`` in batches of windows; returns the outputs on the CPU."""
    n, length, _ = hidden.shape
    pos = torch.arange(length, device=device)[None]
    outs = []
    with torch.no_grad():
        for b in range(0, n, batch):
            hb = hidden[b : b + batch].to(device)
            if capture is not None:
                capture.set_windows(torch.arange(b, b + hb.shape[0]), length)
            outs.append(block(hb, position_ids=pos).cpu())
    return torch.cat(outs)


def _fingerprint(config: Any, windows: torch.Tensor, extra: str) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(config.to_dict(), sort_keys=True, default=str).encode())
    h.update(windows.cpu().numpy().tobytes())
    h.update(extra.encode())
    return h.hexdigest()


def _atomic_save(obj: Any, path: Path) -> None:
    tmp = path.with_name(path.name + ".partial")
    torch.save(obj, tmp)
    tmp.replace(path)


def run_sequential(
    src: SourceCheckpoint,
    config: Any,
    windows: torch.Tensor,
    handler: LayerHandler,
    *,
    device: torch.device | str,
    batch_windows: int,
    modeling: Any,
    last_layer: int | None = None,
    propagate: str = "bf16",
    state_dir: Path | None = None,
    fingerprint_extra: str = "",
    log: Callable[[str], None] = lambda _s: None,
) -> torch.Tensor:
    """Run the blocks in order over ``windows [n_win, L]``; return the hidden states after the last layer run.

    Args:
        src: The bf16 checkpoint (one block is built and loaded at a time).
        config: The ``OlalaConfig``.
        windows: Token ids of the calibration windows.
        handler: Decides which layers are captured and consumes the captures.
        device: Torch device of the block forward.
        batch_windows: Windows per forward call.
        modeling: The imported vendored ``modeling_olala`` module.
        last_layer: Stop after this layer (default: run all layers).
        propagate: ``bf16`` (the next layer sees the bf16 outputs: the design's parallel Hessians) or ``quantized``
            (the handler's dequantized experts are installed and the layer is run again: true sequential).
        state_dir: If given, the boundary hidden state is saved after every layer and the run resumes from it.
        fingerprint_extra: Extra text identifying the settings that change the state (solver, Hessian, manifest).
        log: Progress callback.

    Raises:
        ResumeError: If ``state_dir`` holds a state made for other windows, model or settings.
    """
    if propagate not in ("bf16", "quantized"):
        raise ValueError(f"propagate must be 'bf16' or 'quantized', got {propagate!r}")
    n_layers = len(config.layers_config)
    stop = n_layers - 1 if last_layer is None else last_layer
    fp = _fingerprint(config, windows, f"{propagate}|{fingerprint_extra}")
    start, hidden = 0, None
    if state_dir is not None:
        state_dir.mkdir(parents=True, exist_ok=True)
        progress = state_dir / "progress.json"
        if progress.is_file():
            saved = json.loads(progress.read_text())
            if saved["fingerprint"] != fp:
                raise ResumeError(f"{progress} was made for other windows, model or settings; delete {state_dir}")
            start = saved["next_layer"]
            hidden = torch.load(state_dir / saved["hidden_file"], weights_only=True)
            log(f"resuming at layer {start}")
    if hidden is None:
        hidden = embed_windows(config, src.get_tensor("model.embedding.weight"), windows)
    for layer in range(start, stop + 1):
        block = load_block(modeling, config, src, layer, device)
        want = handler.needs_capture(layer)
        timers = _Timers(block, device)
        capture = MoECapture(block.mlp.experts) if want else None
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        restored = None if want else getattr(handler, "restored_weights", lambda _l: None)(layer)
        if propagate == "quantized" and restored is not None:
            _install(block, restored)
        _sync(device)
        t0 = time.perf_counter()
        out = _forward_all(block, hidden, device, batch_windows, capture)
        _sync(device)
        forward_s = time.perf_counter() - t0
        info = LayerInfo(
            layer=layer,
            capture=capture.finish() if capture is not None else None,
            block=block,
            forward_seconds=forward_s,
            mixer_seconds=timers.t["mixer"],
            mlp_seconds=timers.t["mlp"],
            max_abs_mixer_out=timers.max_abs["mixer"],
            max_abs_mlp_out=timers.max_abs["mlp"],
            peak_memory_bytes=torch.cuda.max_memory_allocated() if str(device).startswith("cuda") else None,
            rss_bytes=_rss_bytes(),
        )
        timers.close()
        stats_row = {
            "layer": layer,
            "captured": want,
            "forward_seconds": forward_s,
            "mixer_seconds": info.mixer_seconds,
            "mlp_seconds": info.mlp_seconds,
            "max_abs_mixer_out": info.max_abs_mixer_out,
            "max_abs_mlp_out": info.max_abs_mlp_out,
            "peak_cuda_allocated_bytes": info.peak_memory_bytes,
            "process_max_rss_bytes": info.rss_bytes,
        }
        if capture is not None:
            capture.close()
            weights = handler.process(info)
            if propagate == "quantized" and weights is not None:
                _install(block, weights)
                out = _forward_all(block, hidden, device, batch_windows, None)
        hidden = out
        del block, info
        if state_dir is not None:
            with (state_dir / "forward_stats.jsonl").open("a") as f:
                f.write(json.dumps(stats_row) + "\n")
            name = f"hidden_{layer + 1}.pt"
            _atomic_save(hidden, state_dir / name)
            (state_dir / "progress.json").write_text(
                json.dumps({"fingerprint": fp, "next_layer": layer + 1, "hidden_file": name})
            )
            for old in state_dir.glob("hidden_*.pt"):
                if old.name != name:
                    old.unlink()
        log(f"layer {layer} done ({forward_s:.1f} s forward)")
    return hidden


def _install(block: Any, weights: ExpertWeights) -> None:
    """Copy dequantized expert weights into the block (keeping the block's parameter dtype)."""
    with torch.no_grad():
        block.mlp.experts.experts.weight.copy_(weights.w_up.to(block.mlp.experts.experts.weight.device))
        block.mlp.experts.output_experts.weight.copy_(weights.w_down.to(block.mlp.experts.output_experts.weight.device))


def _rss_bytes() -> int:
    """Peak resident set size of this process in bytes (cumulative; ``ru_maxrss`` is bytes on macOS, KB on Linux)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    import sys

    return rss if sys.platform == "darwin" else rss * 1024
