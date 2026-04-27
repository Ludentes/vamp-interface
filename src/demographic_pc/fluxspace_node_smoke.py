"""Milestone-1 smoke test for FluxSpaceEditSkeleton node.

Runs a single 512x512 Flux portrait through ComfyUI, with the skeleton node
inserted between UNETLoader and KSampler. Since the skeleton is a pass-through,
the output should be bit-identical to a vanilla render at the same seed.

Checks two things:
  1. Prompt accepted + image saved (graph validates, hooks install cleanly).
  2. PNG hash equals a baseline render with no skeleton node.

Console-side check (user inspects ComfyUI's own stdout):
  [fluxspace-skel] wrap#1 hook_calls_total=~57 double_blocks=19 single_blocks=38 ...
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from src.demographic_pc.comfy_flux import (
    ComfyClient, FLUX_CHECKPOINT, FLUX_VAE, FLUX_CLIP_L, FLUX_T5,
    FLUX_STEPS, FLUX_GUIDANCE, FLUX_SAMPLER, FLUX_SCHEDULER,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "demographic_pc" / "fluxspace_node_smoke"

PROMPT = ("A photorealistic portrait photograph of an adult Latin American woman, "
          "neutral expression, plain grey background, studio lighting, sharp focus.")
SEED = 2026
W, H = 512, 512


def build_workflow(with_skeleton: bool, prefix: str) -> dict:
    import os as _os
    unet_name = FLUX_CHECKPOINT if _os.environ.get("COMFY_UNET") else FLUX_CHECKPOINT.removeprefix("FLUX1/")
    wf: dict = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "fp8_e4m3fn"}},
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX_VAE}},
        "3": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": FLUX_CLIP_L, "clip_name2": FLUX_T5, "type": "flux"}},
        "4": {"class_type": "EmptySD3LatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["3", 0]}},
        "6": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["5", 0], "guidance": FLUX_GUIDANCE}},
    }
    model_src = ["1", 0]
    if with_skeleton:
        wf["10"] = {
            "class_type": "FluxSpaceEditSkeleton",
            "inputs": {"model": ["1", 0], "verbose": True},
        }
        model_src = ["10", 0]
    wf["7"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model_src, "positive": ["6", 0], "negative": ["6", 0],
            "latent_image": ["4", 0], "seed": SEED, "steps": FLUX_STEPS, "cfg": 1.0,
            "sampler_name": FLUX_SAMPLER, "scheduler": FLUX_SCHEDULER, "denoise": 1.0,
        },
    }
    wf["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["2", 0]}}
    wf["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}}
    return wf


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    async with ComfyClient() as client:
        print("== baseline (no skeleton) ==")
        dest_a = OUT / "baseline.png"
        if not dest_a.exists():
            await client.generate(build_workflow(False, "fs_skel_baseline"), dest_a)
        print("== with FluxSpaceEditSkeleton ==")
        dest_b = OUT / "with_skeleton.png"
        if dest_b.exists():
            dest_b.unlink()
        await client.generate(build_workflow(True, "fs_skel_wrapped"), dest_b)

    h_a = hashlib.sha256(dest_a.read_bytes()).hexdigest()
    h_b = hashlib.sha256(dest_b.read_bytes()).hexdigest()
    print(f"baseline       sha256 = {h_a}")
    print(f"with_skeleton  sha256 = {h_b}")
    print("IDENTICAL" if h_a == h_b else "DIFFERENT — plumbing must not mutate tensors!")


if __name__ == "__main__":
    asyncio.run(main())
