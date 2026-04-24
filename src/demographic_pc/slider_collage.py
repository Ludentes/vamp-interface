"""Render a scale-sweep collage for each trained Flux slider.

For each LoRA under models/flux_sliders/<axis>_v1_0/, produce one
collage image `slider_collage_<axis>.png` with rows=bases and cols=scales
∈ {0, 0.25, 0.5, 0.75, 1.0}. Labels embedded.

Strategy: load FluxPipeline once, pre-encode all prompts, drop
text encoders (~10 GB freed), keep transformer+VAE resident. Load each
LoRA in turn via PEFT with the same target regex used during training.
Scale the LoRA by mutating `module.scaling[adapter]` per render.

Usage:
    uv run python src/demographic_pc/slider_collage.py \\
        --lora-dirs models/flux_sliders/eye_squint_v1_0 \\
                    models/flux_sliders/gaze_horizontal_v1_0 \\
        --output-dir models/flux_sliders/collages
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file as load_safetensors

# Axis-to-edit-description for collage titles
AXIS_TITLES = {
    "eye_squint": "eye squint (squint intensity)",
    "gaze_horizontal": "gaze horizontal (eyes look sideways)",
    "brow_lift": "brow lift (eyebrows raise)",
    "brow_furrow_v2": "brow furrow v2 (eyebrows knit)",
    "mouth_stretch_v2": "mouth stretch v2 (lips pulled wide)",
}

BASE_PROMPTS = {
    "asian_m":         "A photorealistic portrait photograph of an adult East Asian man, neutral expression, plain grey background, studio lighting, sharp focus.",
    "black_f":         "A photorealistic portrait photograph of an adult Black woman, neutral expression, plain grey background, studio lighting, sharp focus.",
    "european_m":      "A photorealistic portrait photograph of an adult European man, neutral expression, plain grey background, studio lighting, sharp focus.",
    "elderly_latin_m": "A photorealistic portrait photograph of an elderly Latin American man, neutral expression, plain grey background, studio lighting, sharp focus.",
    "young_european_f":"A photorealistic portrait photograph of a young European woman, neutral expression, plain grey background, studio lighting, sharp focus.",
    "southasian_f":    "A photorealistic portrait photograph of an adult South Asian woman, neutral expression, plain grey background, studio lighting, sharp focus.",
}

# Same regex as trainer
FLUX_XATTN_TARGETS = (
    r"(?:transformer_blocks\.\d+\.attn\.(?:to_q|to_k|to_v|to_out\.0|add_q_proj|add_k_proj|add_v_proj|to_add_out))"
    r"|(?:single_transformer_blocks\.\d+\.(?:attn\.(?:to_q|to_k|to_v)|proj_out))"
)

SCALES = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFAULT_COLLAGE_BASES = ["european_m", "young_european_f"]  # held-out + trained
DEFAULT_SEED = 2026


# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lora-dirs", nargs="+", required=True,
                   help="directories under models/flux_sliders containing slider_step*.safetensors")
    p.add_argument("--ckpt-step", type=int, default=1000,
                   help="which checkpoint step to use (default 1000)")
    p.add_argument("--bases", nargs="+", default=DEFAULT_COLLAGE_BASES)
    p.add_argument("--scales", type=float, nargs="+", default=SCALES)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--flux-hf-id", default="black-forest-labs/FLUX.1-Krea-dev")
    p.add_argument("--flux-transformer",
                   default="/home/newub/w/ComfyUI/models/diffusion_models/flux1-krea-dev.safetensors")
    p.add_argument("--num-inference-steps", type=int, default=25)
    p.add_argument("--guidance-scale", type=float, default=3.5)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    return p.parse_args()


# ------------------------------------------------------------------

def build_pipeline(args, device):
    from diffusers import FluxPipeline, FluxTransformer2DModel

    # Load local transformer FIRST, then pass to pipeline so from_pretrained
    # skips the 23 GB transformer download (we already have the weights).
    print("[pipeline] loading local Flux transformer (bf16, 23 GB)")
    transformer = FluxTransformer2DModel.from_single_file(
        args.flux_transformer, torch_dtype=torch.bfloat16,
    )
    print("[pipeline] building FluxPipeline (transformer injected, others from HF cache)")
    pipe = FluxPipeline.from_pretrained(
        args.flux_hf_id, transformer=transformer, torch_dtype=torch.bfloat16,
    )
    # Use model CPU offload: each component moves to GPU only when called.
    # Fits our 32 GB 5090 where ~34 GB of weights would OOM if all resident.
    pipe.enable_model_cpu_offload(gpu_id=0)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def encode_all_prompts(pipe, bases, device):
    """Pre-encode all base prompts once, return dict[base] -> (pe, pp, ti)."""
    encoded = {}
    with torch.no_grad():
        for base in bases:
            prompt = BASE_PROMPTS[base]
            prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                prompt=prompt, prompt_2=prompt,
                device=device, num_images_per_prompt=1,
                max_sequence_length=512,
            )
            encoded[base] = (prompt_embeds, pooled_prompt_embeds, text_ids)
    return encoded


def free_text_encoders(pipe):
    """With model_cpu_offload, text encoders are already on CPU.
    Nothing to do here — offload handles residency automatically."""
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[pipeline] GPU alloc = "
          f"{torch.cuda.memory_allocated() / 1e9:.2f} GB (offload active)")


# ------------------------------------------------------------------

def load_or_swap_lora(transformer, lora_state_path, args, adapter_name="slider"):
    """Attach adapter on first call, swap state on subsequent calls."""
    from peft import LoraConfig, inject_adapter_in_model
    from peft.utils import set_peft_model_state_dict

    state = load_safetensors(str(lora_state_path))

    # First-time: attach
    existing = getattr(transformer, "peft_config", {})
    if adapter_name not in existing:
        config = LoraConfig(
            r=args.rank, lora_alpha=args.lora_alpha,
            target_modules=FLUX_XATTN_TARGETS,
            init_lora_weights="gaussian", bias="none",
        )
        if hasattr(transformer, "add_adapter"):
            transformer.add_adapter(config, adapter_name=adapter_name)
        else:
            inject_adapter_in_model(config, transformer, adapter_name=adapter_name)
        if hasattr(transformer, "enable_adapters"):
            transformer.enable_adapters()

    # Load state dict (overwrites if swapping)
    set_peft_model_state_dict(transformer, state, adapter_name=adapter_name)


def set_slider_scale(transformer, scale: float, adapter="slider"):
    """Mutate PEFT scaling dict so LoRA output is multiplied by `scale`."""
    for _, mod in transformer.named_modules():
        if hasattr(mod, "scaling") and isinstance(mod.scaling, dict) and adapter in mod.scaling:
            if not hasattr(mod, "_base_scaling"):
                mod._base_scaling = {k: v for k, v in mod.scaling.items()}
            mod.scaling[adapter] = mod._base_scaling[adapter] * scale


# ------------------------------------------------------------------

def render_scale_sweep(pipe, encoded, axis_name, scales, args, device):
    """Render the full scales × bases grid for one LoRA. Returns dict[base][scale]->PIL."""
    results: dict[str, dict[float, Image.Image]] = {}
    transformer = pipe.transformer

    for base, (pe, pp, ti) in encoded.items():
        results[base] = {}
        # Stable seed per (axis, base) — scale sweeps share same starting noise
        seed = args.seed + (hash(axis_name + base) % 100000)
        for scale in scales:
            set_slider_scale(transformer, scale)
            gen = torch.Generator(device=device).manual_seed(seed)
            with torch.no_grad():
                img = pipe(
                    prompt_embeds=pe,
                    pooled_prompt_embeds=pp,
                    height=args.resolution, width=args.resolution,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=gen,
                ).images[0]
            results[base][scale] = img
            print(f"  [{axis_name}] base={base} scale={scale:.2f} rendered")
    # Reset slider scale to 0 so next LoRA starts clean
    set_slider_scale(transformer, 0.0)
    return results


# ------------------------------------------------------------------

def compose_collage(results, axis_name, bases, scales, out_path: Path):
    """Grid: rows=bases, cols=scales, with labels and axis title header."""
    thumb = next(iter(next(iter(results.values())).values())).size[0]  # assume square
    pad = 8
    label_h = 28
    title_h = 40

    n_rows = len(bases)
    n_cols = len(scales)
    row_label_w = 180  # left label column for base name
    W = row_label_w + n_cols * thumb + (n_cols + 1) * pad
    H = title_h + label_h + n_rows * thumb + (n_rows + 1) * pad

    canvas = Image.new("RGB", (W, H), (22, 22, 22))
    draw = ImageDraw.Draw(canvas)

    try:
        font_big = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font_big = ImageFont.load_default()
        font = ImageFont.load_default()

    # Title
    title = AXIS_TITLES.get(axis_name, axis_name)
    draw.text((pad * 2, 10), f"Slider: {title}", fill=(230, 230, 230), font=font_big)

    # Column labels (scales)
    for ci, s in enumerate(scales):
        x = row_label_w + ci * (thumb + pad) + pad
        y = title_h + 6
        draw.text((x, y), f"scale = {s:.2f}", fill=(200, 200, 200), font=font)

    # Rows
    for ri, base in enumerate(bases):
        y = title_h + label_h + ri * (thumb + pad) + pad
        # Row label
        draw.text((pad * 2, y + thumb // 2 - 8), base, fill=(200, 200, 200), font=font)
        for ci, s in enumerate(scales):
            x = row_label_w + ci * (thumb + pad) + pad
            img = results[base][s]
            canvas.paste(img, (x, y))

    canvas.save(str(out_path))
    print(f"[collage] wrote {out_path}")


# ------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = build_pipeline(args, device)
    encoded = encode_all_prompts(pipe, args.bases, device)
    free_text_encoders(pipe)

    for lora_dir in args.lora_dirs:
        lora_dir = Path(lora_dir)
        axis_name = lora_dir.name.replace("_v1_0", "")
        ckpt = lora_dir / f"slider_step{args.ckpt_step:06d}.safetensors"
        if not ckpt.exists():
            print(f"[skip] {ckpt} missing")
            continue
        print(f"[lora] loading {ckpt}")
        load_or_swap_lora(pipe.transformer, ckpt, args)

        results = render_scale_sweep(
            pipe, encoded, axis_name, args.scales, args, device
        )
        out = out_dir / f"slider_collage_{axis_name}.png"
        compose_collage(results, axis_name, args.bases, args.scales, out)


if __name__ == "__main__":
    main()
