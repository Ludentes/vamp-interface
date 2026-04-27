"""FluxSpace-fine renders for Stage 4.5 comparison.

Runs the upstream FluxSpace pipeline (vendor/FluxSpace/) on the same 20
Stage-4.5 portraits and the same λ grid, for one axis at a time. Writes to
    output/demographic_pc/stage4_5/{axis}/renders/fluxspace_fine/{portrait_id}__lam{+0.50}.png

Axis → (positive-pole edit prompt, negative-pole edit prompt):
    age:     ("elderly",             "young child")
    gender:  ("woman",                "man")
    smile:   ("broad smiling face showing teeth", "neutral expressionless face")
    glasses: ("thick-rim eyeglasses", "no glasses")
    black:   ("Black person",         "White person")

For λ > 0 we use the positive-pole edit_prompt at edit_content_scale = |λ|·scale.
For λ < 0 we swap to the negative-pole prompt, at the same magnitude.
λ = 0 is produced with edit_start_iter=num_inference_steps and scales=0 (their
no-op recipe — it reproduces the base generation exactly).

Base model: by default loads Flux Krea from ComfyUI's local checkpoints
(`/home/newub/w/ComfyUI/models/checkpoints/FLUX1/flux1-krea-dev_fp8_scaled.safetensors`
+ local T5, CLIP-L, VAE) so we reuse ~30GB of already-downloaded weights and
exactly match Stage 4.5 Ours' base distribution. Pass `--no-local --model-id
black-forest-labs/FLUX.1-dev` to run against the paper's original model
instead (e.g. for sanity-checking). FluxSpace's block-patching only touches
`transformer_blocks` / `single_transformer_blocks`, which Krea and dev share
in structure, so the patch applies unchanged.

VRAM strategy (32GB GPU):
  - ComfyUI must be stopped first (it pins Flux in ~19GB).
  - bf16 Krea loaded naively peaks ~28GB — tight but fits with ComfyUI down.
  - Default to `pipe.enable_model_cpu_offload()` which swaps text encoders /
    VAE between CPU and GPU around the DiT call. Peak ~12GB, ~2× slower per
    image but safe and lets ComfyUI stay resident if needed.
  - Disable with `--no-cpu-offload` once we've confirmed VRAM headroom.
  - fp8 option: diffusers supports `torch_dtype=torch.float8_e4m3fn` for the
    transformer; not tested in FluxSpace's block wrapper. Defer unless needed.

Usage:
    uv run python -m src.demographic_pc.fluxspace_render --axis age
    uv run python -m src.demographic_pc.fluxspace_render --axis age --limit 2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
VENDOR_FS = ROOT / "vendor" / "FluxSpace"
OUT_DIR = ROOT / "output" / "demographic_pc"
STAGE_DIR = OUT_DIR / "stage4_5"

# Vendored from stage4_5_render / prompts so this file runs under the FluxSpace
# venv (diffusers 0.31.0 pin) without dragging Comfy/sklearn imports.
# Keep these in sync with stage4_5_render.LAMBDAS / portrait_grid() / prompts.py.
import random as _random
LAMBDAS = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]
_GENDERS = ["man", "woman", "non-binary person"]
_ETHNICITIES = [
    "East Asian", "Southeast Asian", "South Asian",
    "Black", "White", "Hispanic or Latino", "Middle Eastern",
]
_PROMPT_TEMPLATE = (
    "A photorealistic portrait photograph of a {age} {ethnicity} {gender}, "
    "neutral expression, plain grey background, studio lighting, sharp focus."
)


def portrait_grid() -> list[dict]:
    rng = _random.Random(42)
    cells = [(g, e) for g in _GENDERS for e in _ETHNICITIES]
    rng.shuffle(cells)
    picked = cells[:20]
    out = []
    for i, (gender, ethnicity) in enumerate(picked):
        seed = 2000 + i
        out.append({
            "portrait_id": f"p{i:02d}_{_GENDERS.index(gender)}_{_ETHNICITIES.index(ethnicity)}_s{seed}",
            "gender": gender, "ethnicity": ethnicity, "seed": seed,
            "prompt": _PROMPT_TEMPLATE.format(age="adult", ethnicity=ethnicity, gender=gender),
        })
    return out

AXIS_EDIT_PROMPTS: dict[str, tuple[str, str]] = {
    "age":     ("elderly",                               "young child"),
    "gender":  ("woman",                                 "man"),
    "smile":   ("broad smile showing teeth",             "neutral expressionless face"),
    "glasses": ("thick-rim eyeglasses",                  "no glasses, clear view of the eyes"),
    "black":   ("Black person",                          "White person"),
}

# FluxSpace was published against FLUX.1-dev. We match the rest of our pipeline
# (Stage 2 / Stage 4.5 Ours) by loading Flux Krea from the ComfyUI model dir
# rather than re-downloading from HF. Architectures are identical — same
# FluxTransformer2DModel double/single blocks — so FluxSpace's block-patching
# applies unchanged.
FLUX_MODEL_ID_DEFAULT = "black-forest-labs/FLUX.1-Krea-dev"

COMFY_ROOT = Path("/home/newub/w/ComfyUI/models")
COMFY_TRANSFORMER = COMFY_ROOT / "checkpoints" / "FLUX1" / "flux1-krea-dev_fp8_scaled.safetensors"
COMFY_VAE = COMFY_ROOT / "vae" / "FLUX1" / "ae.safetensors"
COMFY_CLIP_L = COMFY_ROOT / "text_encoders" / "clip_l.safetensors"
COMFY_T5 = COMFY_ROOT / "text_encoders" / "t5" / "t5xxl_fp16.safetensors"
CLIP_HF_ID = "openai/clip-vit-large-patch14"
T5_HF_ID = "google/t5-v1_1-xxl"
SCHEDULER_HF_ID = "black-forest-labs/FLUX.1-dev"  # scheduler config only; tiny

# FluxSpace's content_scale maps our unit λ to an edit magnitude.
# Their demos use 5.0 for attribute edits; 1.0/λ-unit gives us |λ|=5 ≈ strong, |λ|=1 ≈ mild.
# edit_global_scale (coarse) also contributes; leave at 0.8 (their recipe).
EDIT_CONTENT_SCALE_PER_LAMBDA = 1.0
EDIT_GLOBAL_SCALE = 0.8
EDIT_START_ITER = 3
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 3.5
ATTENTION_THRESHOLD = 0.5

WIDTH, HEIGHT = 768, 1024


def load_pipe_from_local(cpu_offload: bool):
    """Build FluxSpace pipe from ComfyUI's local Flux Krea + T5 + CLIP + VAE.

    Krea's fp8_scaled transformer is upcast to bf16 by from_single_file;
    the scaling metadata is baked in so accuracy matches ComfyUI's run.
    """
    sys.path.insert(0, str(VENDOR_FS))
    from flux_semantic_pipeline import GenerationPipelineSemantic  # type: ignore
    from diffusers.models.transformers import FluxTransformer2DModel
    from diffusers.models.autoencoders import AutoencoderKL
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from transformers import (
        CLIPTextModel, CLIPTokenizer,
        T5Config, T5EncoderModel, T5TokenizerFast,
    )
    from safetensors.torch import load_model

    for p in (COMFY_TRANSFORMER, COMFY_VAE, COMFY_CLIP_L, COMFY_T5):
        if not p.exists():
            raise FileNotFoundError(p)

    dtype = torch.bfloat16
    print(f"[fs/local] loading Krea transformer from {COMFY_TRANSFORMER.name}…")
    # low_cpu_mem_usage: meta-init then stream tensor copies — avoids 2× peak RAM
    # during load_state_dict. Critical on 64GB boxes — naive load OOM-killed us once.
    transformer = FluxTransformer2DModel.from_single_file(
        str(COMFY_TRANSFORMER), torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    # ae.safetensors is Flux's 16-latent VAE; from_single_file picks up diffusers'
    # default 4-latent AutoencoderKL config and errors on shape mismatch. Load
    # config from FLUX.1-dev's vae subfolder (small json) and weights from local.
    print(f"[fs/local] loading VAE (config from HF, weights from {COMFY_VAE.name})…")
    vae = AutoencoderKL.from_pretrained(
        SCHEDULER_HF_ID, subfolder="vae", torch_dtype=dtype,
    )
    load_model(vae, str(COMFY_VAE), strict=False)

    print("[fs/local] loading CLIP-L…")
    clip_tok = CLIPTokenizer.from_pretrained(CLIP_HF_ID)
    # CLIP-L is ~500MB; ComfyUI's clip_l.safetensors matches HF weights bit-for-bit
    # for the text tower, but loading from HF is simpler and cached.
    clip = CLIPTextModel.from_pretrained(CLIP_HF_ID, torch_dtype=dtype)

    print("[fs/local] loading T5XXL (fp16 native) from local safetensors…")
    t5_tok = T5TokenizerFast.from_pretrained(T5_HF_ID)
    cfg = T5Config.from_pretrained(T5_HF_ID)
    # Keep T5 at fp16 (its stored dtype) — halves RAM vs upcasting to bf16.
    # load_model streams tensor-by-tensor from the mmap'd file, no state_dict copy.
    t5 = T5EncoderModel(cfg).to(dtype=torch.float16)
    load_model(t5, str(COMFY_T5), strict=False)

    print("[fs/local] scheduler from HF config…")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        SCHEDULER_HF_ID, subfolder="scheduler",
    )

    pipe = GenerationPipelineSemantic(
        transformer=transformer, vae=vae,
        text_encoder=clip, tokenizer=clip_tok,
        text_encoder_2=t5, tokenizer_2=t5_tok,
        scheduler=scheduler,
    )
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")
    pipe.register_transformer_blocks()
    return pipe


def load_pipe_from_hf(model_id: str, cpu_offload: bool):
    sys.path.insert(0, str(VENDOR_FS))
    from flux_semantic_pipeline import GenerationPipelineSemantic  # type: ignore

    print(f"[fs/hf] loading {model_id} in bfloat16  (cpu_offload={cpu_offload})…")
    pipe = GenerationPipelineSemantic.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
    )
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")
    pipe.register_transformer_blocks()
    return pipe


def load_pipe(model_id: str | None, cpu_offload: bool, use_local: bool):
    if use_local:
        return load_pipe_from_local(cpu_offload=cpu_offload)
    return load_pipe_from_hf(model_id or FLUX_MODEL_ID_DEFAULT, cpu_offload=cpu_offload)


def render_one(pipe, prompt: str, edit_prompt: str, seed: int, lam: float,
               edit_scale_per_lambda: float, dest: Path,
               width: int = WIDTH, height: int = HEIGHT) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(seed)
    if abs(lam) < 1e-9:
        # no-op; reproduces base generation
        img = pipe(
            prompt=prompt, edit_prompt="",
            guidance_scale=GUIDANCE_SCALE, num_inference_steps=NUM_INFERENCE_STEPS,
            edit_start_iter=NUM_INFERENCE_STEPS, edit_global_scale=0.0,
            edit_content_scale=0.0, attention_threshold=0.0,
            height=height, width=width, generator=gen,
        ).images[0]
    else:
        img = pipe(
            prompt=prompt, edit_prompt=edit_prompt,
            guidance_scale=GUIDANCE_SCALE, num_inference_steps=NUM_INFERENCE_STEPS,
            edit_start_iter=EDIT_START_ITER,
            edit_global_scale=EDIT_GLOBAL_SCALE,
            edit_content_scale=abs(lam) * edit_scale_per_lambda,
            attention_threshold=ATTENTION_THRESHOLD,
            height=height, width=width, generator=gen,
        ).images[0]
    img.save(dest)


def run(axis: str, model_id: str | None, cpu_offload: bool, use_local: bool,
        limit: int | None = None, width: int = WIDTH, height: int = HEIGHT,
        lambdas: list[float] | None = None) -> None:
    if axis not in AXIS_EDIT_PROMPTS:
        raise ValueError(f"unknown axis {axis!r}; choose from {list(AXIS_EDIT_PROMPTS)}")
    pos_prompt, neg_prompt = AXIS_EDIT_PROMPTS[axis]

    portraits = portrait_grid()
    if limit:
        portraits = portraits[:limit]

    # Tag renders by base model source so dev/Krea/local runs don't collide.
    if use_local:
        base_tag = "krea_local"
    elif model_id and "Krea" in model_id:
        base_tag = "krea_hf"
    else:
        base_tag = "dev_hf"
    renders = STAGE_DIR / axis / "renders" / f"fluxspace_fine_{base_tag}"
    renders.mkdir(parents=True, exist_ok=True)

    pipe = load_pipe(model_id, cpu_offload=cpu_offload, use_local=use_local)

    lams = lambdas if lambdas is not None else LAMBDAS
    n_jobs = len(portraits) * len(lams)
    print(f"[fs/{axis}] {len(portraits)} portraits × {len(lams)} λ = {n_jobs} renders  ({width}×{height})")
    t0 = time.time()
    done = 0
    for p in portraits:
        for lam in lams:
            edit_prompt = pos_prompt if lam > 0 else neg_prompt
            dest = renders / f"{p['portrait_id']}__lam{lam:+.2f}.png"
            render_one(
                pipe, prompt=p["prompt"], edit_prompt=edit_prompt,
                seed=p["seed"], lam=float(lam),
                edit_scale_per_lambda=EDIT_CONTENT_SCALE_PER_LAMBDA,
                dest=dest, width=width, height=height,
            )
            done += 1
            if done % 5 == 0:
                dt = time.time() - t0
                rate = done / dt
                eta = (n_jobs - done) / rate / 60
                print(f"  [{done}/{n_jobs}] rate={rate:.2f}/s  eta={eta:.1f}min")
    print(f"[fs/{axis}] done → {renders}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", choices=list(AXIS_EDIT_PROMPTS), required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model-id", default=None,
                    help="HF model ID. Only used with --no-local. Default when set: black-forest-labs/FLUX.1-Krea-dev.")
    ap.add_argument("--no-local", action="store_true",
                    help="Pull Flux weights from HF hub instead of ComfyUI's local checkpoints.")
    ap.add_argument("--no-cpu-offload", action="store_true",
                    help="Keep the whole pipeline on GPU (faster, ~28GB VRAM peak). Default uses cpu-offload (~12GB, ~2× slower).")
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--height", type=int, default=HEIGHT)
    ap.add_argument("--lambdas", type=str, default=None,
                    help="Comma-separated λ values, e.g. '0,1'. Default: full LAMBDAS grid.")
    args = ap.parse_args()
    lams = [float(x) for x in args.lambdas.split(",")] if args.lambdas else None
    run(axis=args.axis, model_id=args.model_id,
        cpu_offload=not args.no_cpu_offload,
        use_local=not args.no_local,
        limit=args.limit, width=args.width, height=args.height, lambdas=lams)


if __name__ == "__main__":
    main()
