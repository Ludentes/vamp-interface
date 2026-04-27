"""Milestone-3 test: FluxSpaceEditPair (attention-cache averaging A+B).

Hypothesis: prompt A ("A person wearing thick-rimmed eyeglasses") pushes
subject younger; prompt B (full base-prompt splice) pushes subject older.
Averaging their cached attentions cancels opposing age confounds while
reinforcing the shared glasses signal.

Grid: mix_b ∈ {0.3, 0.5, 0.7} at scale=1.0.

Usage:
    uv run python -m src.demographic_pc.fluxspace_node_pair_test --axis glasses
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from src.demographic_pc.comfy_flux import (
    ComfyClient, FLUX_CHECKPOINT, FLUX_VAE, FLUX_CLIP_L, FLUX_T5,
    FLUX_STEPS, FLUX_GUIDANCE, FLUX_SAMPLER, FLUX_SCHEDULER,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "output" / "demographic_pc" / "fluxspace_node_test"
LOG_PATH = OUT_ROOT / "experiments.jsonl"

BASE_PROMPT = (
    "A photorealistic portrait photograph of an adult Latin American woman, "
    "neutral expression, plain grey background, studio lighting, sharp focus."
)
SEEDS = [2026, 4242]
W, H = 512, 512

AXIS_PAIR_PROMPTS = {
    "glasses": {
        "A": "A person wearing thick-rimmed eyeglasses.",
        "B": ("A photorealistic portrait photograph of an adult Latin American woman "
              "wearing thick-rimmed eyeglasses, neutral expression, plain grey "
              "background, studio lighting, sharp focus."),
    },
}

MIXES = [0.3, 0.5, 0.7]
SCALE_SWEEP = [
    -2.0, -1.5, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.3, 0.0,
    0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4,
    1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5, 3.0,
]


def log_experiment(entry: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **entry}
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def build_workflow(edit_a: str, edit_b: str, scale: float, mix_b: float,
                   prefix: str, seed: int) -> dict:
    import os as _os
    unet_name = FLUX_CHECKPOINT if _os.environ.get("COMFY_UNET") else FLUX_CHECKPOINT.removeprefix("FLUX1/")
    wf: dict = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "fp8_e4m3fn"}},
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX_VAE}},
        "3": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": FLUX_CLIP_L, "clip_name2": FLUX_T5, "type": "flux"}},
        "4": {"class_type": "EmptySD3LatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": BASE_PROMPT, "clip": ["3", 0]}},
        "6": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["5", 0], "guidance": FLUX_GUIDANCE}},
        "20": {"class_type": "CLIPTextEncode", "inputs": {"text": edit_a, "clip": ["3", 0]}},
        "21": {"class_type": "CLIPTextEncode", "inputs": {"text": edit_b, "clip": ["3", 0]}},
        "22": {
            "class_type": "FluxSpaceEditPair",
            "inputs": {
                "model": ["1", 0],
                "edit_conditioning_a": ["20", 0],
                "edit_conditioning_b": ["21", 0],
                "scale": float(scale),
                "mix_b": float(mix_b),
                "start_percent": 0.15,
                "end_percent": 1.0,
                "double_blocks_only": False,
                "verbose": True,
            },
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["22", 0], "positive": ["6", 0], "negative": ["6", 0],
                "latent_image": ["4", 0], "seed": seed, "steps": FLUX_STEPS, "cfg": 1.0,
                "sampler_name": FLUX_SAMPLER, "scheduler": FLUX_SCHEDULER, "denoise": 1.0,
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["2", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}},
    }
    return wf


def build_single_workflow(edit: str | None, scale: float, prefix: str, seed: int) -> dict:
    import os as _os
    unet_name = FLUX_CHECKPOINT if _os.environ.get("COMFY_UNET") else FLUX_CHECKPOINT.removeprefix("FLUX1/")
    wf: dict = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "fp8_e4m3fn"}},
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX_VAE}},
        "3": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": FLUX_CLIP_L, "clip_name2": FLUX_T5, "type": "flux"}},
        "4": {"class_type": "EmptySD3LatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": BASE_PROMPT, "clip": ["3", 0]}},
        "6": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["5", 0], "guidance": FLUX_GUIDANCE}},
    }
    model_src = ["1", 0]
    if edit is not None and scale != 0.0:
        wf["20"] = {"class_type": "CLIPTextEncode", "inputs": {"text": edit, "clip": ["3", 0]}}
        wf["21"] = {
            "class_type": "FluxSpaceEdit",
            "inputs": {
                "model": ["1", 0], "edit_conditioning": ["20", 0], "scale": float(scale),
                "start_percent": 0.15, "end_percent": 1.0,
                "double_blocks_only": False, "verbose": False,
            },
        }
        model_src = ["21", 0]
    wf["7"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model_src, "positive": ["6", 0], "negative": ["6", 0],
            "latent_image": ["4", 0], "seed": seed, "steps": FLUX_STEPS, "cfg": 1.0,
            "sampler_name": FLUX_SAMPLER, "scheduler": FLUX_SCHEDULER, "denoise": 1.0,
        },
    }
    wf["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["2", 0]}}
    wf["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}}
    return wf


async def run_axis(axis: str) -> None:
    prompts = AXIS_PAIR_PROMPTS[axis]
    edit_a, edit_b = prompts["A"], prompts["B"]
    out_dir = OUT_ROOT / axis / "pair"
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_dir = OUT_ROOT / axis / "pair_refs"
    ref_dir.mkdir(parents=True, exist_ok=True)

    async with ComfyClient() as client:
        for seed in SEEDS:
            for tag, edit in (("base", None), ("A", edit_a), ("B", edit_b)):
                dest = ref_dir / f"{tag}_s{seed}.png"
                if dest.exists():
                    continue
                print(f"[fs-pair/{axis}/ref] seed={seed}  {tag}")
                await client.generate(
                    build_single_workflow(edit, 0.0 if tag == "base" else 1.0,
                                          f"fspair_{axis}_ref_{tag}_s{seed}", seed),
                    dest,
                )
        for seed in SEEDS:
            for mix in MIXES:
                dest = out_dir / f"pair_mix{mix:.2f}_s{seed}.png"
                if dest.exists():
                    continue
                print(f"[fs-pair/{axis}] seed={seed}  mix_b={mix:.2f}  scale=1.0")
                await client.generate(
                    build_workflow(edit_a, edit_b, 1.0, mix,
                                   f"fspair_{axis}_mix{mix:.2f}_s{seed}", seed),
                    dest,
                )
                log_experiment({
                    "run": "pair", "axis": axis, "mix_b": mix, "scale": 1.0,
                    "edit_a": edit_a, "edit_b": edit_b, "base_prompt": BASE_PROMPT,
                    "start_percent": 0.15, "end_percent": 1.0, "seed": seed,
                    "w": W, "h": H, "steps": FLUX_STEPS,
                    "output": str(dest.relative_to(ROOT)),
                })
    print(f"[fs-pair/{axis}] done → {out_dir}")


async def run_scale_sweep(axis: str) -> None:
    """Linearity probe: mix_b=0.5 fixed, scale ∈ SCALE_SWEEP, both seeds."""
    prompts = AXIS_PAIR_PROMPTS[axis]
    edit_a, edit_b = prompts["A"], prompts["B"]
    out_dir = OUT_ROOT / axis / "pair_scale_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    async with ComfyClient() as client:
        for seed in SEEDS:
            for s in SCALE_SWEEP:
                dest = out_dir / f"sweep_s{s:+.2f}_seed{seed}.png"
                if dest.exists():
                    continue
                print(f"[fs-pair/{axis}/sweep] seed={seed}  scale={s:+.2f}  mix_b=0.5")
                await client.generate(
                    build_workflow(edit_a, edit_b, s, 0.5,
                                   f"fspair_{axis}_sweep_s{s:+.2f}_seed{seed}", seed),
                    dest,
                )
                log_experiment({
                    "run": "pair_scale_sweep", "axis": axis, "mix_b": 0.5, "scale": s,
                    "edit_a": edit_a, "edit_b": edit_b, "base_prompt": BASE_PROMPT,
                    "start_percent": 0.15, "end_percent": 1.0, "seed": seed,
                    "w": W, "h": H, "steps": FLUX_STEPS,
                    "output": str(dest.relative_to(ROOT)),
                })
    print(f"[fs-pair/{axis}/sweep] done → {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", choices=list(AXIS_PAIR_PROMPTS), default="glasses")
    ap.add_argument("--scale-sweep", action="store_true",
                    help="Run linearity probe at mix_b=0.5 across SCALE_SWEEP")
    args = ap.parse_args()
    if args.scale_sweep:
        asyncio.run(run_scale_sweep(args.axis))
    else:
        asyncio.run(run_axis(args.axis))


if __name__ == "__main__":
    main()
