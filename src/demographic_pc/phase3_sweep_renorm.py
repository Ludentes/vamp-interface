"""Phase 3 repeat with per-token renorm ON. Same seeds/bases/scales as the
no-renorm sweep — direct head-to-head comparison."""
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
OUT = ROOT / "output/demographic_pc/phase3_sweep_renorm"

BASES = [
    ("elderly_latin_m", CALIBRATION_PROMPTS[5][1]),
    ("young_european_f", CALIBRATION_PROMPTS[8][1]),
]
TEST_SEED = 777
# Include both the original 0–1.5 regime AND the wide-sweep range.
SCALES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]


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


def wf(seed: int, prefix: str, base_prompt: str, scale: float) -> dict:
    w = _base_workflow()
    w["5"] = {"class_type": "CLIPTextEncode", "inputs": {"text": base_prompt, "clip": ["3", 0]}}
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
        for base_name, base_prompt in BASES:
            for s in SCALES:
                stem = f"{base_name}_s{TEST_SEED}_x{s:+.2f}"
                dest = OUT / f"{stem}.png"
                if dest.exists():
                    print(f"[skip] {stem}")
                    continue
                print(f"[render] {stem}")
                await client.generate(wf(TEST_SEED, f"p3rnm_{stem}", base_prompt, s), dest)
    print(f"[done] → {OUT}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if args.run:
        asyncio.run(run())
    else:
        print(f"Would render {len(BASES)} × {len(SCALES)} = {len(BASES)*len(SCALES)} images "
              f"(renorm ON)")


if __name__ == "__main__":
    main()
