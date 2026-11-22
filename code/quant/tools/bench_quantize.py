"""Time and peak-memory of the quantizers on a real-shape random expert tensor (CPU or CUDA).

Usage: python tools/bench_quantize.py [--device cpu] [--experts 256]
The tensor is random (not the model's weights): it only measures throughput and memory.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time

import torch

from olala_quant import fp8, nvfp4


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MB (macOS reports bytes, Linux kilobytes)."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e6 if sys.platform == "darwin" else raw / 1e3


def main() -> None:
    """Run both quantizers on a [E, 768, 384] bf16 tensor and print one JSON record."""
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--experts", type=int, default=256)
    args = p.parse_args()
    torch.manual_seed(0)
    w = (torch.randn(args.experts, 768, 384) * 0.02).to(torch.bfloat16).to(args.device)
    out: dict[str, object] = {
        "device": args.device,
        "shape": list(w.shape),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "rss_after_alloc_mb": peak_rss_mb(),
    }
    for name, fn in (("fp8", fp8.quantize_fp8), ("nvfp4", nvfp4.quantize_nvfp4)):
        t0 = time.perf_counter()
        r = fn(w)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        out[name] = {
            "seconds": round(dt, 3),
            "mparams_per_s": round(w.numel() / dt / 1e6, 1),
            "rel_fro_error": r.error.rel_fro,
            "peak_rss_mb": peak_rss_mb(),
        }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
