"""Sanity check: replay the cached delta on the EXACT capture seeds
(elderly_latin_m, seeds 2026/4242/1337). Compare vs the phase1_full live
renders at the same (base, seed) to see if cached replay even recovers the
edit on-latent.

Expected outcomes:
  - If cache works on-latent: these look like phase1_full/*.png (clear smile)
  - If cache doesn't even work on-latent: flat faces (as in phase3 cross-seed)
    → direction isn't semantically aligned, ruling out latent-specificity as
    the only ceiling.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from src.demographic_pc.comfy_flux import ComfyClient
from src.demographic_pc.fluxspace_metrics import (
    CALIBRATION_PROMPTS, FLUX_GUIDANCE, FLUX_SAMPLER, FLUX_SCHEDULER,
    FLUX_STEPS, W, H, FLUX_CHECKPOINT, FLUX_VAE, FLUX_CLIP_L, FLUX_T5,
)

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "models/blendshape_nmf/phase1_smile_delta_full.npz"
OUT = ROOT / "output/demographic_pc/phase3_onlatent_sanity"

BASE_NAME = "elderly_latin_m"
BASE_PROMPT = CALIBRATION_PROMPTS[5][1]
CAPTURE_SEEDS = [2026, 4242, 1337]
SCALES = [0.0, 1.0]  # baseline + cached replay at live-equivalent scale


def _base_workflow() -> dict:
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


def wf(seed: int, prefix: str, scale: float) -> dict:
    w = _base_workflow()
    w["5"] = {"class_type": "CLIPTextEncode", "inputs": {"text": BASE_PROMPT, "clip": ["3", 0]}}
    w["6"] = {"class_type": "FluxGuidance", "inputs": {"conditioning": ["5", 0], "guidance": FLUX_GUIDANCE}}
    w["22"] = {
        "class_type": "FluxSpaceDirectionInjectFull",
        "inputs": {
            "model": ["1", 0], "delta_full_npz_path": str(LIB),
            "scale": float(scale), "start_percent": 0.15, "end_percent": 1.0,
            "renorm_per_token": True, "verbose": False,
        },
    }
    w["7"] = {"class_type": "KSampler",
              "inputs": {"model": ["22", 0], "positive": ["6", 0], "negative": ["6", 0],
                         "latent_image": ["4", 0], "seed": seed, "steps": FLUX_STEPS, "cfg": 1.0,
                         "sampler_name": FLUX_SAMPLER, "scheduler": FLUX_SCHEDULER, "denoise": 1.0}}
    w["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["2", 0]}}
    w["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}}
    return w


async def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    async with ComfyClient() as client:
        for seed in CAPTURE_SEEDS:
            for s in SCALES:
                stem = f"{BASE_NAME}_s{seed}_x{s:+.1f}"
                dest = OUT / f"{stem}.png"
                if dest.exists():
                    print(f"[skip] {stem}")
                    continue
                print(f"[render] {stem}")
                await client.generate(wf(seed, f"p3sanity_{stem}", s), dest)
    print(f"[done] → {OUT}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if args.run:
        asyncio.run(run())
    else:
        print(f"Would render {len(CAPTURE_SEEDS)} × {len(SCALES)} = "
              f"{len(CAPTURE_SEEDS)*len(SCALES)} images")


if __name__ == "__main__":
    main()
