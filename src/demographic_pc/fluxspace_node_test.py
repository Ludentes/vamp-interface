"""Milestone-2 test for FluxSpaceEdit node.

Runs an A/B on a single portrait:
  A. baseline (no edit) — one render
  B. FluxSpaceEdit(edit_prompt="thick-rim eyeglasses") at several scales

Layout:
  output/demographic_pc/fluxspace_node_test/{axis}/
    baseline.png
    fs_scale{+0.50}.png
    fs_scale{+1.00}.png
    fs_scale{+2.00}.png
    fs_scale{+3.00}.png

Axis edit prompts mirror upstream FluxSpace templates but kept short.

Usage:
    uv run python -m src.demographic_pc.fluxspace_node_test --axis glasses
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


def log_experiment(entry: dict) -> None:
    """Append one JSONL row per render. Config only — tensors saved separately."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **entry}
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

BASE_PROMPT = (
    "A photorealistic portrait photograph of an adult Latin American woman, "
    "neutral expression, plain grey background, studio lighting, sharp focus."
)
SEED = 2026
W, H = 512, 512

AXIS_EDIT_PROMPTS = {
    "glasses": "A person wearing thick-rimmed eyeglasses.",
    "smile":   "A person with a broad smile showing teeth.",
    "age":     "An elderly person with gray hair and wrinkles.",
    "beard":   "A person with a full dense beard.",
}

SCALES = [0.5, 1.0, 2.0, 3.0]
ABLATION_SCALES = [0.5, 1.0, 1.5]

# Prompt-shape × term-weight grid (axis=glasses for now).
GLASSES_PROMPT_TEMPLATES = {
    "A": "A person wearing (thick-rimmed eyeglasses:{w:.1f}).",
    "B": ("A photorealistic portrait photograph of an adult Latin American woman "
          "wearing (thick-rimmed eyeglasses:{w:.1f}), neutral expression, plain grey "
          "background, studio lighting, sharp focus."),
    "C": "(thick-rimmed eyeglasses:{w:.1f})",
}
WEIGHTS = [0.5, 1.0, 1.5, 2.0]


def build_workflow(edit_prompt: str | None, scale: float, prefix: str,
                   double_blocks_only: bool = False,
                   start_percent: float = 0.15,
                   end_percent: float = 1.0) -> dict:
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
    if edit_prompt is not None and scale != 0.0:
        wf["20"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": edit_prompt, "clip": ["3", 0]},
        }
        wf["21"] = {
            "class_type": "FluxSpaceEdit",
            "inputs": {
                "model": ["1", 0],
                "edit_conditioning": ["20", 0],
                "scale": float(scale),
                "start_percent": float(start_percent),
                "end_percent": float(end_percent),
                "double_blocks_only": bool(double_blocks_only),
                "verbose": True,
            },
        }
        model_src = ["21", 0]
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


async def run_axis(axis: str, ablation: bool = False) -> None:
    edit_prompt = AXIS_EDIT_PROMPTS[axis]
    out_dir = OUT_ROOT / axis
    out_dir.mkdir(parents=True, exist_ok=True)

    async with ComfyClient() as client:
        dest_base = out_dir / "baseline.png"
        if not dest_base.exists():
            print(f"[fs-test/{axis}] baseline…")
            await client.generate(build_workflow(None, 0.0, f"fstest_{axis}_base"), dest_base)

        if ablation:
            # 2x3 matrix: double_blocks_only × scale
            for dbo in (False, True):
                tag = "dbo" if dbo else "all"
                for s in ABLATION_SCALES:
                    dest = out_dir / f"fs_{tag}_s+{s:.2f}.png"
                    if dest.exists():
                        continue
                    print(f"[fs-test/{axis}] {tag} scale=+{s:.2f}")
                    await client.generate(
                        build_workflow(edit_prompt, s, f"fstest_{axis}_{tag}_s{s:.2f}", dbo),
                        dest,
                    )
        else:
            for s in SCALES:
                dest = out_dir / f"fs_scale+{s:.2f}.png"
                if dest.exists():
                    continue
                print(f"[fs-test/{axis}] scale=+{s:.2f}  edit={edit_prompt!r}")
                await client.generate(
                    build_workflow(edit_prompt, s, f"fstest_{axis}_s{s:.2f}"),
                    dest,
                )
    print(f"[fs-test/{axis}] done → {out_dir}")


async def run_prompt_weight_grid(axis: str) -> None:
    """3×4 grid: prompt shape {A,B,C} × term weight {0.5,1.0,1.5,2.0}. scale=1.0."""
    if axis != "glasses":
        raise NotImplementedError("only glasses templated so far")
    out_dir = OUT_ROOT / axis / "pw_grid"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_dest = OUT_ROOT / axis / "baseline.png"

    async with ComfyClient() as client:
        if not base_dest.exists():
            await client.generate(build_workflow(None, 0.0, f"fstest_{axis}_base"), base_dest)

        for shape, template in GLASSES_PROMPT_TEMPLATES.items():
            for w in WEIGHTS:
                edit_prompt = template.format(w=w)
                dest = out_dir / f"pw_{shape}_w{w:.1f}.png"
                if dest.exists():
                    continue
                print(f"[fs-test/{axis}/pw] {shape} w={w:.1f}  edit={edit_prompt!r}")
                await client.generate(
                    build_workflow(edit_prompt, 1.0, f"fstest_{axis}_pw_{shape}_w{w:.1f}"),
                    dest,
                )
                log_experiment({
                    "run": "pw_grid", "axis": axis, "shape": shape, "weight": w,
                    "edit_prompt": edit_prompt, "base_prompt": BASE_PROMPT,
                    "scale": 1.0, "start_percent": 0.15, "end_percent": 1.0,
                    "double_blocks_only": False, "seed": SEED, "w": W, "h": H,
                    "steps": FLUX_STEPS, "output": str(dest.relative_to(ROOT)),
                })
    print(f"[fs-test/{axis}/pw] done → {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", choices=list(AXIS_EDIT_PROMPTS), default="glasses")
    ap.add_argument("--ablation", action="store_true",
                    help="Run double_blocks_only × scale matrix instead of scale sweep")
    ap.add_argument("--pw-grid", action="store_true",
                    help="Run prompt-shape × term-weight grid at scale=1.0")
    args = ap.parse_args()
    if args.pw_grid:
        asyncio.run(run_prompt_weight_grid(args.axis))
    else:
        asyncio.run(run_axis(args.axis, ablation=args.ablation))


if __name__ == "__main__":
    main()
