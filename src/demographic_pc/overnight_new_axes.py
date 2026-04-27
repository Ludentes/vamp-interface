"""Overnight pass: render prompt-pair sweeps for 4 new axes we lack
coverage on. Output goes to crossdemo/<axis>/<axis>_inphase/.

Axes:
  age        — demographic, Stage-5 composition prereq for smile-age cancel
  hair_style — vocabulary extension (uncaptured per topic notes)
  hair_color — vocabulary extension
  eye_squint — fills NMF atom C7 gap

Per axis: 6 cross-demo bases × 3 seeds × 5 mix_b = 90 renders. No attn
capture (pure PNG), since the cached-delta thread is falsified —
capture only when we actually need a measurement probe.

Total: 4 × 90 = 360 renders, ~2.5h at typical pair-render rate.
Resumable (skip-if-exists).

Run:
  uv run python -m src.demographic_pc.overnight_new_axes --run
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
    # Demographics
    ("age", "a youthful person in their mid-twenties, smooth skin",
            "an elderly person in their eighties with deep wrinkles and silver hair"),
    ("gender", "a masculine face with a strong jawline and stubble",
               "a feminine face with soft features and long eyelashes"),
    # Vocabulary extensions (uncaptured per topic notes)
    ("hair_style", "a person with a very short cropped military buzz-cut hairstyle",
                   "a person with long flowing hair past the shoulders"),
    ("hair_color", "a person with platinum blonde hair",
                   "a person with jet black hair"),
    ("skin_smoothness", "a person with very smooth flawless porcelain skin",
                        "a person with rough textured skin, visible pores and imperfections"),
    ("nose_shape", "a person with a small delicate button nose",
                   "a person with a prominent aquiline hooked nose"),
    # NMF-atom gap-fillers (C1/C2/C3/C5/C7 — see audit)
    ("eye_squint", "a person with wide-open alert eyes",
                   "a person with strongly squinted eyes, eyelids half-closed"),
    ("brow_lift", "a person with relaxed brows, neutral forehead",
                  "a person with both eyebrows raised high, forehead wrinkled upward"),
    ("brow_furrow", "a person with relaxed smooth forehead, brows apart",
                    "a person with brows strongly furrowed together, deep vertical creases between the eyebrows"),
    ("gaze_horizontal", "a person looking directly at the camera",
                        "a person looking sharply to the side, eyes averted"),
    ("mouth_stretch", "a person with a relaxed mouth, lips gently closed",
                      "a person with mouth stretched wide horizontally, lips pulled taut sideways"),
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
    print(f"[overnight] total planned: {total} renders "
          f"({len(AXES)} axes × {len(BASES_FULL)} bases × {len(SEEDS)} seeds × {len(ALPHAS)} alphas)")
    async with ComfyClient() as client:
        for axis_name, edit_a, edit_b in AXES:
            axis_dir = OUT_ROOT / axis_name / f"{axis_name}_inphase"
            for base_name, base_prompt, _ in BASES_FULL:
                base_dir = axis_dir / base_name
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
                            pair_wf(seed, f"on_{axis_name}_{base_name}_{stem}",
                                    base_prompt, edit_a, edit_b, alpha),
                            dest,
                        )
    print(f"[overnight-done] rendered={done} skipped={skipped} → {OUT_ROOT}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if args.run:
        asyncio.run(run())
    else:
        total = len(AXES) * len(BASES_FULL) * len(SEEDS) * len(ALPHAS)
        print(f"Would render {total} images (~2.5h est)")
        print(f"  axes: {[a[0] for a in AXES]}")
        print(f"  bases: {[b[0] for b in BASES_FULL]}")
        print(f"  seeds: {SEEDS}  alphas: {ALPHAS}")


if __name__ == "__main__":
    main()
