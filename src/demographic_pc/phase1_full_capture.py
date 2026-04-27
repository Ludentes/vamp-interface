"""Phase 1 of option-B rebuild: capture full (L, D) attention tensors.

Re-renders 3 seeds of the smile pair on elderly_latin_m with the modified
FluxSpaceEditPair node, which now persists attn_base_full + delta_mix_full
at fp16 per (step, block). ~17 GB per render → ~51 GB total.

Output npz files go to the external drive; PNGs stay local.

Run:
  uv run python -m src.demographic_pc.phase1_full_capture --run
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
PNG_DIR = ROOT / "output/demographic_pc/phase1_full"
NPZ_DIR = ROOT / "output/demographic_pc/phase1_full_captures"  # internal ext4, sparse-capable
SITES_JSON = ROOT / "output/demographic_pc/phase1_topk_sites.json"

EDIT_A = "A person with a faint closed-mouth smile."
EDIT_B = "A person with a manic wide-open grin, teeth bared."

BASE_NAME = "elderly_latin_m"
BASE_PROMPT = CALIBRATION_PROMPTS[5][1]
SEEDS = [2026, 4242, 1337]
MIX_B = 0.5
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


def full_workflow(seed: int, full_npz: str, prefix: str) -> dict:
    wf = _base_workflow()
    wf["5"] = {"class_type": "CLIPTextEncode", "inputs": {"text": BASE_PROMPT, "clip": ["3", 0]}}
    wf["6"] = {"class_type": "FluxGuidance", "inputs": {"conditioning": ["5", 0], "guidance": FLUX_GUIDANCE}}
    wf["20"] = {"class_type": "CLIPTextEncode", "inputs": {"text": EDIT_A, "clip": ["3", 0]}}
    wf["21"] = {"class_type": "CLIPTextEncode", "inputs": {"text": EDIT_B, "clip": ["3", 0]}}
    wf["22"] = {
        "class_type": "FluxSpaceEditPair",
        "inputs": {
            "model": ["1", 0],
            "edit_conditioning_a": ["20", 0],
            "edit_conditioning_b": ["21", 0],
            "scale": float(SCALE), "mix_b": float(MIX_B),
            "start_percent": float(START_PCT), "end_percent": 1.0,
            "double_blocks_only": False, "verbose": False,
            "full_measure_path": full_npz,
            "full_capture_sites_json": str(SITES_JSON) if SITES_JSON.exists() else "",
        },
    }
    wf["7"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["22", 0], "positive": ["6", 0], "negative": ["6", 0],
            "latent_image": ["4", 0], "seed": seed, "steps": FLUX_STEPS, "cfg": 1.0,
            "sampler_name": FLUX_SAMPLER, "scheduler": FLUX_SCHEDULER, "denoise": 1.0,
        },
    }
    wf["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["2", 0]}}
    wf["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}}
    return wf


async def run() -> None:
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    NPZ_DIR.mkdir(parents=True, exist_ok=True)
    async with ComfyClient() as client:
        for seed in SEEDS:
            stem = f"{BASE_NAME}_s{seed}_mix{int(MIX_B*100):02d}"
            png_dest = PNG_DIR / f"{stem}.png"
            npz_dest = NPZ_DIR / f"{stem}.full.npz"
            if png_dest.exists() and npz_dest.exists():
                print(f"[skip] {stem}")
                continue
            print(f"[render] {stem}  → {npz_dest}")
            await client.generate(
                full_workflow(seed, str(npz_dest), f"phase1_full_{stem}"),
                png_dest,
            )
    print(f"[done] pngs → {PNG_DIR}  npzs → {NPZ_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if args.run:
        asyncio.run(run())
    else:
        gated = SITES_JSON.exists()
        size_hint = "~300MB each, sparse" if gated else "~17GB each (no allow-list)"
        print(f"Would render {len(SEEDS)} samples ({size_hint})")
        print(f"  base: {BASE_NAME}  seeds: {SEEDS}  mix_b={MIX_B} scale={SCALE}")
        print(f"  allow-list: {SITES_JSON if gated else '(none — full 912 sites)'}")
        print(f"  pngs → {PNG_DIR}")
        print(f"  npzs → {NPZ_DIR}")


if __name__ == "__main__":
    main()
