"""Re-render the two axes with broken prompts, using fixed text pairs.

mouth_stretch_v2 — tighten on closed-mouth horizontal pull (target
stays `mouthStretchL/R`, not `jawOpen`).

brow_furrow_v2 — mechanical pure-muscle phrasing that avoids "vertical
creases" (age-coded, cross-base weak).

Output: crossdemo/<axis>_v2/<axis>_v2_inphase/<base>/s<seed>_a<alpha>.png
so we don't clobber the original corpora. 180 renders total, ~90 min.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from src.demographic_pc.comfy_flux import ComfyClient
from src.demographic_pc.fluxspace_metrics import (
    FLUX_GUIDANCE, FLUX_SAMPLER, FLUX_SCHEDULER, FLUX_STEPS, W, H,
    FLUX_CHECKPOINT, FLUX_VAE, FLUX_CLIP_L, FLUX_T5,
)
from src.demographic_pc.fluxspace_intensity_sweep import BASES_FULL

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "output/demographic_pc/fluxspace_metrics/crossdemo"

AXES = [
    # mouth_stretch v2 — closed-mouth sideways pull, avoid open-mouth semantic
    ("mouth_stretch_v2",
     "a person with lips gently together in a calm closed mouth",
     "a person with lips pressed firmly together and pulled wide sideways in a tight grimace, mouth remains closed"),
    # brow_furrow v2 — pure muscle phrasing, no age-coded "wrinkles/creases"
    ("brow_furrow_v2",
     "a person with relaxed eyebrows at rest in a neutral expression",
     "a person with both eyebrows strongly pulled down and pressed together, brows knit tightly"),
]

SEEDS = [2026, 4242, 1337]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
SCALE = 1.0
START_PCT = 0.15


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


def pair_wf(seed: int, prefix: str, base_prompt: str,
            edit_a: str, edit_b: str, alpha: float) -> dict:
    wf = _base_workflow()
    wf["5"] = {"class_type": "CLIPTextEncode", "inputs": {"text": base_prompt, "clip": ["3", 0]}}
    wf["6"] = {"class_type": "FluxGuidance", "inputs": {"conditioning": ["5", 0], "guidance": FLUX_GUIDANCE}}
    wf["20"] = {"class_type": "CLIPTextEncode", "inputs": {"text": edit_a, "clip": ["3", 0]}}
    wf["21"] = {"class_type": "CLIPTextEncode", "inputs": {"text": edit_b, "clip": ["3", 0]}}
    wf["22"] = {
        "class_type": "FluxSpaceEditPair",
        "inputs": {
            "model": ["1", 0],
            "edit_conditioning_a": ["20", 0], "edit_conditioning_b": ["21", 0],
            "scale": float(SCALE), "mix_b": float(alpha),
            "start_percent": float(START_PCT), "end_percent": 1.0,
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


async def run() -> None:
    total = len(AXES) * len(BASES_FULL) * len(SEEDS) * len(ALPHAS)
    done = skipped = 0
    async with ComfyClient() as client:
        for axis_name, edit_a, edit_b in AXES:
            for base_name, base_prompt, _ in BASES_FULL:
                base_dir = OUT_ROOT / axis_name / f"{axis_name}_inphase" / base_name
                base_dir.mkdir(parents=True, exist_ok=True)
                for seed in SEEDS:
                    for alpha in ALPHAS:
                        stem = f"s{seed}_a{alpha:.2f}"
                        dest = base_dir / f"{stem}.png"
                        if dest.exists():
                            skipped += 1
                            continue
                        done += 1
                        print(f"[{done+skipped}/{total}] {axis_name} {base_name} s={seed} α={alpha:.2f}")
                        await client.generate(
                            pair_wf(seed, f"rp_{axis_name}_{base_name}_{stem}",
                                    base_prompt, edit_a, edit_b, alpha),
                            dest,
                        )
    print(f"[reprompt-done] rendered={done} skipped={skipped} → {OUT_ROOT}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if args.run:
        asyncio.run(run())
    else:
        total = len(AXES) * len(BASES_FULL) * len(SEEDS) * len(ALPHAS)
        print(f"Would render {total} images (~90 min est)")
        for name, a, b in AXES:
            print(f"  {name}")
            print(f"    A: {a}")
            print(f"    B: {b}")


if __name__ == "__main__":
    main()
