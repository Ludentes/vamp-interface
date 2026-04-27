"""Saturated smoke test: inject library direction at EVERY (step, block) site.

Uses au_inject_all.npz (912 sites per component). Same 4 atoms × 2 bases,
narrower scale sweep since signal is now 912× denser.

Output: output/demographic_pc/au_library_smoke_all/
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
DIR_PATH = ROOT / "models/blendshape_nmf/au_inject_all.npz"
OUT = ROOT / "output/demographic_pc/au_library_smoke_all"

ATOMS = [(6, "smile"), (2, "brow_lift"), (3, "brow_furrow"), (7, "eye_squint")]
BASES = [
    ("elderly_latin_m",  CALIBRATION_PROMPTS[5][1]),
    ("young_european_f", CALIBRATION_PROMPTS[8][1]),
]
# Narrower sweep — 912 sites means each site's contribution is smaller per-site,
# but there are many more of them. Start at ±100 (≈per-site 0.1 of unit
# direction) and ±1000.
SCALES = [0.0, 100.0, 500.0, -500.0, 2000.0, -2000.0]
SEED = 2026


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


def inject_workflow(seed, prefix, base_prompt, atom_id, scale):
    wf = _base_workflow()
    wf["5"] = {"class_type": "CLIPTextEncode", "inputs": {"text": base_prompt, "clip": ["3", 0]}}
    wf["6"] = {"class_type": "FluxGuidance", "inputs": {"conditioning": ["5", 0], "guidance": FLUX_GUIDANCE}}
    wf["22"] = {"class_type": "FluxSpaceDirectionInject",
                "inputs": {"model": ["1", 0], "directions_npz_path": str(DIR_PATH),
                           "atom_id": int(atom_id), "scale": float(scale),
                           "start_percent": 0.15, "end_percent": 1.0, "verbose": False}}
    wf["7"] = {"class_type": "KSampler",
               "inputs": {"model": ["22", 0], "positive": ["6", 0], "negative": ["6", 0],
                          "latent_image": ["4", 0], "seed": seed, "steps": FLUX_STEPS, "cfg": 1.0,
                          "sampler_name": FLUX_SAMPLER, "scheduler": FLUX_SCHEDULER, "denoise": 1.0}}
    wf["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["2", 0]}}
    wf["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}}
    return wf


async def run():
    OUT.mkdir(parents=True, exist_ok=True)
    async with ComfyClient() as client:
        for atom_id, atom_name in ATOMS:
            for base_name, base_prompt in BASES:
                for s in SCALES:
                    stem = f"{atom_name}_{base_name}_s{s:+.0f}"
                    dest = OUT / f"{stem}.png"
                    if dest.exists():
                        print(f"[skip] {stem}")
                        continue
                    print(f"[render] {stem}")
                    await client.generate(
                        inject_workflow(SEED, f"au_smoke_all_{stem}", base_prompt,
                                        atom_id, s),
                        dest,
                    )
    print(f"[done] → {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if args.run:
        asyncio.run(run())
    else:
        print(f"Would render {len(ATOMS)} × {len(BASES)} × {len(SCALES)} "
              f"= {len(ATOMS)*len(BASES)*len(SCALES)} images")


if __name__ == "__main__":
    main()
