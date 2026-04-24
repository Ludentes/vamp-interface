"""Expanded corpus renderer — Strategy A (compressed mix_b range) + wider demographic net.

Re-renders a slider axis with:
  - Wider base set (all 10 CALIBRATION_PROMPTS instead of 6 BASES_FULL)
  - Wider seed set (default 5 seeds instead of 3)
  - Compressed α range (default {0, 0.10, 0.20, 0.30, 0.40} instead of {0..1.0})
    — stays in the identity-safe half of the teacher's mix_b curve

Writes to crossdemo_v2/<axis>/<axis>_inphase/<base>/s{seed}_a{alpha:.2f}.png.
Resumable (skip-if-exists).

Usage:
    uv run python src/demographic_pc/expand_corpus_v2.py \\
        --axis eye_squint --axis-prompts \\
            "a person with wide-open alert eyes" \\
            "a person with strongly squinted eyes, eyelids half-closed"

Diagnostic upstream: docs/research/2026-04-24-slider-corpus-identity-drift.md
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.demographic_pc.comfy_flux import ComfyClient  # noqa: E402
from src.demographic_pc.fluxspace_metrics import (      # noqa: E402
    CALIBRATION_PROMPTS, FLUX_GUIDANCE, FLUX_SAMPLER, FLUX_SCHEDULER,
    FLUX_STEPS, W, H,
    FLUX_CHECKPOINT, FLUX_VAE, FLUX_CLIP_L, FLUX_T5,
)

OUT_ROOT = ROOT / "output/demographic_pc/fluxspace_metrics/crossdemo_v2"

# Built-in axis definitions (extendable via --axis-prompts)
AXIS_PROMPTS = {
    "eye_squint":       ("a person with wide-open alert eyes",
                         "a person with strongly squinted eyes, eyelids half-closed"),
    "brow_lift":        ("a person with relaxed brows, neutral forehead",
                         "a person with both eyebrows raised high, forehead wrinkled upward"),
    "brow_furrow_v2":   ("relaxed eyebrows at rest in a neutral expression",
                         "both eyebrows strongly pulled down and pressed together, brows knit tightly"),
    "mouth_stretch_v2": ("lips gently together in a calm closed mouth",
                         "lips pressed firmly together and pulled wide sideways in a tight grimace, mouth remains closed"),
    "gaze_horizontal":  ("a person looking directly at the camera",
                         "a person looking sharply to the side, eyes averted"),
    # Demographic axes (for solver / composition use)
    "age":              ("a youthful person in their mid-twenties, smooth skin",
                         "an elderly person in their eighties with deep wrinkles and silver hair"),
    "gender":           ("a masculine face with a strong jawline and stubble",
                         "a feminine face with soft features and long eyelashes"),
    "hair_style":       ("a person with a very short cropped military buzz-cut hairstyle",
                         "a person with long flowing hair past the shoulders"),
    "hair_color":       ("a person with platinum blonde hair",
                         "a person with jet black hair"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--axis", required=True,
                   help="axis name; output dir and default prompts look this up")
    p.add_argument("--axis-prompts", nargs=2, default=None, metavar=("EDIT_A", "EDIT_B"),
                   help="override axis prompt pair (edit_a neutral, edit_b target)")
    p.add_argument("--bases", nargs="+",
                   default=[p[0] for p in CALIBRATION_PROMPTS],
                   help="base names (keys from CALIBRATION_PROMPTS)")
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[1337, 2026, 4242, 7777, 31337])
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[0.0, 0.10, 0.20, 0.30, 0.40])
    p.add_argument("--scale", type=float, default=1.0,
                   help="FluxSpaceEditPair scale (default 1.0; reduce α range instead)")
    p.add_argument("--start-pct", type=float, default=0.15)
    p.add_argument("--out-root", default=str(OUT_ROOT))
    p.add_argument("--concurrency", type=int, default=1,
                   help="parallel ComfyUI submissions (1 is safest)")
    return p.parse_args()


def base_workflow() -> dict:
    import os as _os
    unet_name = FLUX_CHECKPOINT if _os.environ.get("COMFY_UNET") else FLUX_CHECKPOINT.removeprefix("FLUX1/")
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": unet_name, "weight_dtype": "fp8_e4m3fn"}},
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX_VAE}},
        "3": {"class_type": "DualCLIPLoader",
              "inputs": {"clip_name1": FLUX_CLIP_L, "clip_name2": FLUX_T5, "type": "flux"}},
        "4": {"class_type": "EmptySD3LatentImage",
              "inputs": {"width": W, "height": H, "batch_size": 1}},
    }


def pair_wf(seed: int, prefix: str, base_prompt: str,
            edit_a: str, edit_b: str, alpha: float,
            scale: float, start_pct: float) -> dict:
    wf = base_workflow()
    wf["5"] = {"class_type": "CLIPTextEncode", "inputs": {"text": base_prompt, "clip": ["3", 0]}}
    wf["6"] = {"class_type": "FluxGuidance", "inputs": {"conditioning": ["5", 0], "guidance": FLUX_GUIDANCE}}
    wf["20"] = {"class_type": "CLIPTextEncode", "inputs": {"text": edit_a, "clip": ["3", 0]}}
    wf["21"] = {"class_type": "CLIPTextEncode", "inputs": {"text": edit_b, "clip": ["3", 0]}}
    wf["22"] = {
        "class_type": "FluxSpaceEditPair",
        "inputs": {
            "model": ["1", 0],
            "edit_conditioning_a": ["20", 0], "edit_conditioning_b": ["21", 0],
            "scale": float(scale), "mix_b": float(alpha),
            "start_percent": float(start_pct), "end_percent": 1.0,
            "double_blocks_only": False, "verbose": False,
        },
    }
    wf["7"] = {"class_type": "KSampler",
               "inputs": {"model": ["22", 0], "positive": ["6", 0], "negative": ["6", 0],
                          "latent_image": ["4", 0], "seed": seed, "steps": FLUX_STEPS, "cfg": 1.0,
                          "sampler_name": FLUX_SAMPLER, "scheduler": FLUX_SCHEDULER, "denoise": 1.0}}
    wf["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["2", 0]}}
    wf["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}}
    return wf


async def main():
    args = parse_args()
    base_prompts = dict(CALIBRATION_PROMPTS)
    for b in args.bases:
        if b not in base_prompts:
            raise SystemExit(f"unknown base: {b}. valid: {list(base_prompts)}")

    if args.axis_prompts:
        edit_a, edit_b = args.axis_prompts
    else:
        if args.axis not in AXIS_PROMPTS:
            raise SystemExit(f"no built-in prompts for axis '{args.axis}'. Pass --axis-prompts.")
        edit_a, edit_b = AXIS_PROMPTS[args.axis]

    out_dir = Path(args.out_root) / args.axis / f"{args.axis}_inphase"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build job list (resumable: skip if output exists)
    jobs: list[tuple[str, int, float]] = []
    for base in args.bases:
        for seed in args.seeds:
            for a in args.alphas:
                target = out_dir / base / f"s{seed}_a{a:.2f}.png"
                if not target.exists():
                    jobs.append((base, seed, a))

    total_desired = len(args.bases) * len(args.seeds) * len(args.alphas)
    print(f"[plan] {len(jobs)} to render, {total_desired - len(jobs)} already exist, {total_desired} total")
    print(f"[plan] axis='{args.axis}', scale={args.scale}, start_pct={args.start_pct}")
    print(f"[plan] edit_a: '{edit_a}'")
    print(f"[plan] edit_b: '{edit_b}'")
    print(f"[plan] output: {out_dir}")

    if not jobs:
        print("[plan] nothing to do")
        return

    sem = asyncio.Semaphore(args.concurrency)

    async def render_one(client, base, seed, a, idx):
        async with sem:
            base_prompt = base_prompts[base]
            # Flat prefix — avoids ComfyUI subfolder handling in our downloader
            prefix = f"expand_v2_{args.axis}_{base}_s{seed}_a{a:.2f}"
            wf = pair_wf(seed, prefix, base_prompt, edit_a, edit_b,
                         a, args.scale, args.start_pct)
            dest = out_dir / base / f"s{seed}_a{a:.2f}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                await client.generate(wf, dest)
                print(f"[{idx}/{len(jobs)}] {base} s={seed} α={a:.2f} -> {dest.name}")
            except Exception as e:
                print(f"[{idx}/{len(jobs)}] FAILED {base} s={seed} α={a:.2f}: {e}")

    async with ComfyClient() as client:
        await asyncio.gather(*(
            render_one(client, b, s, a, i + 1)
            for i, (b, s, a) in enumerate(jobs)
        ))

    print(f"[done] {len(jobs)} renders complete")


if __name__ == "__main__":
    asyncio.run(main())
