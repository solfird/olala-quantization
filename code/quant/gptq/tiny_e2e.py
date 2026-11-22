"""Forward error of a quantized tiny checkpoint against its bf16 source, on windows that no Hessian saw.

``compare`` loads three checkpoints into the vendored model (the bf16 source, the data-free N2 build and the G1-lite
build, dequantizing the NVFP4 tensors with ``olala_quant``), runs the same held-out windows through all three and
reports, for each quantized model against the bf16 one, the relative RMS error of the logits, the mean KL divergence
of the next-token distributions and the top-1 agreement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from olala_quant import nvfp4
from olala_quant.checkpoint_io import SourceCheckpoint

import tiny_model
from olala_model import load_olala_modules


def load_dequantized(model_dir: Path) -> dict[str, torch.Tensor]:
    """The state dict of a checkpoint directory with NVFP4 tensors dequantized back to bf16 weights."""
    ck = (
        SourceCheckpoint.from_shard_headers(model_dir)
        if not (model_dir / "model.safetensors").is_file()
        else (SourceCheckpoint(model_dir))
    )
    names = ck.names()
    out: dict[str, torch.Tensor] = {}
    for name in names:
        if name.endswith(".weight_packed"):
            base = name.removesuffix(".weight_packed")
            out[f"{base}.weight"] = nvfp4.dequantize_nvfp4(
                ck.get_tensor(name),
                ck.get_tensor(f"{base}.weight_scale"),
                ck.get_tensor(f"{base}.weight_global_scale"),
                torch.bfloat16,
            )
        elif not name.endswith((".weight_scale", ".weight_global_scale")):
            out[name] = ck.get_tensor(name)
    return out


def _logits(model_dir: Path, windows: torch.Tensor, modeling: Any, config_mod: Any) -> torch.Tensor:
    cfg = config_mod.OlalaConfig(**json.loads((model_dir / "config.json").read_text()))
    cfg.quantization_config = None
    model = modeling.OlalaForCausalLM(cfg)
    model.load_state_dict(load_dequantized(model_dir), strict=False, assign=True)
    model.eval()
    with torch.no_grad():
        return model(input_ids=windows).logits.float()


def load_dequantized_names(model_dir: Path) -> list[str]:
    """Tensor names of a written checkpoint (from its shard headers)."""
    return list(SourceCheckpoint.from_shard_headers(model_dir).names())


def fresh_windows(seed: int = 7, n_windows: int = 12) -> torch.Tensor:
    """Windows of NEW documents from the lexicon that ``make_calibration`` uses (no calibration token among them)."""
    import calib_data

    tok = calib_data.ByteTokenizer()
    docs = tiny_model.word_docs(20, seed=10_000 + seed, tag="fresh")
    return calib_data.pack_windows([tok.encode(d.text) for d in docs], tok.eos_id, tiny_model.TINY_WINDOW)[:n_windows]


def compare(
    src: Path,
    g1: Path,
    n2: Path,
    n_windows: int = 8,
    seed: int = 123,
    windows: torch.Tensor | None = None,
    rtn_name: str = "n2",
) -> dict[str, dict[str, float]]:
    """Forward error of the G1-lite and round-to-nearest builds against the bf16 source on ``windows``.

    ``n2`` is the directory of the round-to-nearest arm (N2, or N5 with the experts-only base), reported under
    ``rtn_name``. Without ``windows``, uniformly random token windows are used (a distribution the calibration never
    saw).
    """
    modeling, config_mod = load_olala_modules()
    if windows is None:
        windows = tiny_model.random_windows(n_windows, seed=seed)
    ref = _logits(src, windows, modeling, config_mod)
    out: dict[str, dict[str, float]] = {}
    for name, d in ((rtn_name, n2), ("g1", g1)):
        q = _logits(d, windows, modeling, config_mod)
        kl = torch.nn.functional.kl_div(
            torch.log_softmax(q, -1), torch.log_softmax(ref, -1), log_target=True, reduction="none"
        ).sum(-1)
        out[name] = {
            "rel_logit_rmse": float(((q - ref).square().mean() / ref.square().mean()).sqrt()),
            "mean_kl": float(kl.mean()),
            "top1_agreement": float((q.argmax(-1) == ref.argmax(-1)).float().mean()),
        }
    return out


def run_seeds(seeds: list[int], workdir: Path, n_calib_windows: int = 16, base_select: str = "") -> dict[str, Any]:
    """Build tiny random models, quantize each by round-to-nearest and by G1-lite, compare on held-out windows.

    Default base: N2 against G1-lite (non-experts quantized data-free in both). With ``base_select="experts"``: N5
    (``convert --select experts``, routed experts by round-to-nearest, everything else bf16) against G1-lite built with
    ``--base-select experts`` (the same routed experts by GPTQ, everything else bf16), reported under ``n5`` and ``g1``.

    ``fresh``: new documents of the calibration lexicon (same distribution). ``random_tokens``: uniformly random token
    ids (a distribution the calibration never saw). Returns every number and the per-metric means.
    """
    from olala_quant import convert as conv

    import build_g1lite
    from tests_support import make_calibration

    rtn = "n5" if base_select else "n2"
    rows = []
    for seed in seeds:
        d = workdir / f"seed{seed}"
        src = tiny_model.write_tiny_checkpoint(d / "src", seed=seed)
        calib = make_calibration(d, n_windows=n_calib_windows, seed=seed)
        ignore = d / "ignore.json"
        ignore.write_text(json.dumps(list(tiny_model.TINY_IGNORE)))
        code = build_g1lite.main(
            [
                "--src",
                str(src),
                "--calib",
                str(calib),
                "--dst",
                str(d / "g1"),
                "--ignore-file",
                str(ignore),
                "--heldout-every",
                "4",
                "--batch-windows",
                "4",
                *(["--base-select", base_select] if base_select else []),
            ]
        )
        if code != 0:
            raise RuntimeError(f"build_g1lite failed for seed {seed}")
        conv.convert(src, d / rtn, "nvfp4", tiny_model.TINY_IGNORE, select=base_select or None, log=lambda _s: None)
        rep = json.loads((d / "g1" / "g1lite_report.json").read_text())
        rows.append(
            {
                "seed": seed,
                "fresh": compare(src, d / "g1", d / rtn, windows=fresh_windows(seed), rtn_name=rtn),
                "random_tokens": compare(src, d / "g1", d / rtn, n_windows=12, seed=seed, rtn_name=rtn),
                "n_quantized_tensors": {
                    arm: sum(n.endswith("weight_packed") for n in load_dequantized_names(d / arm))
                    for arm in ("g1", rtn)
                },
                "n_gptq_experts": rep["totals"]["n_gptq"],
                "n_experts": rep["totals"]["n_experts"],
            }
        )
    means: dict[str, dict[str, dict[str, float]]] = {}
    for kind in ("fresh", "random_tokens"):
        means[kind] = {
            arm: {m: sum(r[kind][arm][m] for r in rows) / len(rows) for m in rows[0][kind][arm]} for arm in (rtn, "g1")
        }
    return {"base_select": base_select, "seeds": rows, "mean": means}


def main() -> None:
    """Run the comparison for several seeds and print the table (and write ``--out`` when given)."""
    import argparse
    import tempfile

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--out", type=Path)
    p.add_argument("--base-select", default="", help="'experts': compare N5 (RTN) with G1 on the experts-only base")
    args = p.parse_args()
    rtn = "n5" if args.base_select else "n2"
    with tempfile.TemporaryDirectory() as tmp:
        res = run_seeds(args.seeds, Path(tmp), base_select=args.base_select)
    for kind in ("fresh", "random_tokens"):
        print(f"\n{kind} windows (mean over {len(args.seeds)} seeds): arm, rel logit RMSE, mean KL, top-1 agreement")
        for arm in (rtn, "g1"):
            m = res["mean"][kind][arm]
            print(f"  {arm}: {m['rel_logit_rmse']:.4f}  {m['mean_kl']:.3e}  {m['top1_agreement']:.3f}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
