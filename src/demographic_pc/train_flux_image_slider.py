"""Flux image-pair slider trainer — v1.0 smoke test.

Trains a single LoRA slider on one axis from the crossdemo
`α ∈ {0, 0.25, 0.5, 0.75, 1.0}` corpus. Recipe:

- LoRA rank=16, alpha=1, xattn only (all attention linears in double +
  single Flux blocks).
- Multi-α supervision: random α_train ∈ {0.25, 0.5, 0.75, 1.0}; LoRA
  scale is set to α_train per step so slider magnitude = data α.
- Logit-normal timestep bias, μ=0.5, σ=1.0.
- Flow-matching velocity target `ε − z_after`.
- AdamW 8-bit, LR 2e-3, 200 warmup, 1000 steps, bf16, grad ckpt.

Deps required (not yet in pyproject):
    uv add diffusers peft bitsandbytes accelerate

Usage:
    uv run python -m demographic_pc.train_flux_image_slider \\
        --axis eye_squint \\
        --holdout-bases european_m black_f \\
        --output-dir models/flux_sliders/eye_squint

Design doc: docs/research/2026-04-24-slider-trainer-step4-hyperparams.md
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file as save_safetensors

# Hard dep imports — lazy so --help works without these installed
def _import_heavy():
    global FluxTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
    global CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
    global LoraConfig, get_peft_model_state_dict
    from diffusers import (
        FluxTransformer2DModel,
        AutoencoderKL,
        FlowMatchEulerDiscreteScheduler,
    )
    from transformers import (
        AutoTokenizer,
        CLIPTextModel, CLIPTokenizer,
        T5EncoderModel,
    )
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict


REPO_ROOT = Path(__file__).resolve().parents[2]
COMFYUI_ROOT = Path("/home/newub/w/ComfyUI/models")

# Base prompts — extended set covering v1, v2, v3, v3.1 corpora.
# adult_middle_f uses the v3.1 Lebanese override (no headscarf prior)
# regardless of which corpus a cell came from, so the trainer learns
# a coherent "this person" across mixed-corpus cells.
BASE_PROMPTS: dict[str, str] = {
    # v1 ecosystem (original 6)
    "asian_m":         "A photorealistic portrait photograph of an adult East Asian man, neutral expression, plain grey background, studio lighting, sharp focus.",
    "black_f":         "A photorealistic portrait photograph of an adult Black woman, neutral expression, plain grey background, studio lighting, sharp focus.",
    "european_m":      "A photorealistic portrait photograph of an adult European man, neutral expression, plain grey background, studio lighting, sharp focus.",
    "southasian_f":    "A photorealistic portrait photograph of an adult South Asian woman, neutral expression, plain grey background, studio lighting, sharp focus.",
    # v2 / v3 / v3.1 ecosystem (extended demographic cohort)
    "adult_latin_f":     "A photorealistic portrait photograph of an adult Latin American woman, neutral expression, plain grey background, studio lighting, sharp focus.",
    "adult_asian_m":     "A photorealistic portrait photograph of an adult East Asian man, neutral expression, plain grey background, studio lighting, sharp focus.",
    "adult_black_f":     "A photorealistic portrait photograph of an adult Black woman, neutral expression, plain grey background, studio lighting, sharp focus.",
    "adult_european_m":  "A photorealistic portrait photograph of an adult European man, neutral expression, plain grey background, studio lighting, sharp focus.",
    "adult_middle_f": (  # v3.1 Lebanese override — applies to ALL corpora
        "A photorealistic portrait photograph of a Lebanese woman with "
        "long dark wavy hair flowing past her shoulders, olive skin, dark "
        "almond eyes, defined eyebrows, neutral expression, plain grey "
        "background, studio lighting, sharp focus, close-up head-and-"
        "shoulders, face fills frame."
    ),
    "elderly_latin_m":   "A photorealistic portrait photograph of an elderly Latin American man, neutral expression, plain grey background, studio lighting, sharp focus.",
    "elderly_asian_f":   "A photorealistic portrait photograph of an elderly East Asian woman, neutral expression, plain grey background, studio lighting, sharp focus.",
    "young_black_m":     "A photorealistic portrait photograph of a young Black man, neutral expression, plain grey background, studio lighting, sharp focus.",
    "young_european_f":  "A photorealistic portrait photograph of a young European woman, neutral expression, plain grey background, studio lighting, sharp focus.",
    "adult_southasian_f":"A photorealistic portrait photograph of an adult South Asian woman, neutral expression, plain grey background, studio lighting, sharp focus.",
}
ALL_BASES = list(BASE_PROMPTS.keys())

# PEFT target modules for Flux xattn — matches any module whose name
# ends with one of these, across all 19 double + 38 single blocks.
FLUX_XATTN_TARGETS = (
    r"(?:transformer_blocks\.\d+\.attn\.(?:to_q|to_k|to_v|to_out\.0|add_q_proj|add_k_proj|add_v_proj|to_add_out))"
    r"|(?:single_transformer_blocks\.\d+\.(?:attn\.(?:to_q|to_k|to_v)|proj_out))"
)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])

    # Data
    p.add_argument("--axis", required=True,
                   help="axis name under fluxspace_metrics/crossdemo/ (e.g. eye_squint)")
    p.add_argument("--axis-subdir", default=None,
                   help="(directory mode) subdir under axis dir (default: <axis>_inphase)")
    p.add_argument("--crossdemo-root",
                   default=str(REPO_ROOT / "output/demographic_pc/fluxspace_metrics/crossdemo"),
                   help="(directory mode) corpus root walked when --manifest is not given")
    p.add_argument("--manifest", default=None,
                   help="path to curated training manifest parquet "
                        "(e.g. models/flux_sliders/training_manifest_eye_squint.parquet). "
                        "When given, supersedes directory walk; rows used directly.")
    p.add_argument("--holdout-bases", nargs="+", default=["young_european_f"],
                   help="bases to exclude from training, use for eval")
    p.add_argument("--seeds", type=int, nargs="+", default=[1337, 2026, 4242],
                   help="(directory mode only) seed list for directory walk")
    p.add_argument("--alphas-train", type=float, nargs="+",
                   default=None,
                   help="α values sampled during training; if None, auto-derived "
                        "from manifest's non-zero α values")

    # Model paths
    p.add_argument("--flux-transformer",
                   default=str(COMFYUI_ROOT / "diffusion_models/flux1-krea-dev.safetensors"))
    p.add_argument("--flux-vae",
                   default=str(COMFYUI_ROOT / "vae/FLUX1/ae.safetensors"))
    p.add_argument("--t5-weights",
                   default=str(COMFYUI_ROOT / "text_encoders/t5/t5xxl_fp16.safetensors"))
    p.add_argument("--clip-weights",
                   default=str(COMFYUI_ROOT / "text_encoders/clip_l.safetensors"))
    p.add_argument("--flux-hf-id", default="black-forest-labs/FLUX.1-Krea-dev",
                   help="HF repo id used to fetch configs (tokenizers, scheduler config); weights come from local safetensors")

    # LoRA
    # NOTE: lora_alpha == rank so base scaling = alpha/r = 1.0, making
    # per-step `set_lora_scale(α_train)` produce the LITERAL slider
    # strength α_train. This matches LoRA convention and ensures
    # ComfyUI `LoRA_strength=s` reproduces training-time α=s at inference.
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-dropout", type=float, default=0.0)

    # Optim
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--optimizer-8bit", action="store_true", default=True)

    # Schedule
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)

    # Flow-matching / timestep sampling
    p.add_argument("--logit-mean", type=float, default=0.5)
    p.add_argument("--logit-std", type=float, default=1.0)
    p.add_argument("--num-train-timesteps", type=int, default=1000)
    p.add_argument("--flux-sigma-shift", type=float, default=3.0,
                   help="Flux dev uses shift=3.0, schnell=1.0. Applied to t before transformer call.")
    p.add_argument("--guidance-scale", type=float, default=3.5,
                   help="Training-time guidance (3.5 for dev/Krea, 0 for schnell); ignored if not a guidance_embeds model")

    # Precision
    p.add_argument("--precision", choices=["bf16", "fp16"], default="bf16")

    # Logging / checkpointing
    p.add_argument("--output-dir", required=True)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--checkpoint-every", type=int, default=100,
                   help="checkpoint frequency in steps (default 100 — keeps "
                        "≤100 steps of work at risk if power dies mid-training)")
    p.add_argument("--wandb-project", default=None)

    # Reproducibility
    p.add_argument("--seed", type=int, default=42)

    # Smoke-test escape hatches
    p.add_argument("--dry-run", action="store_true",
                   help="Reduce max_steps to 5 and skip eval — plumbing check only")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--resume", action="store_true",
                   help="Resume training from latest checkpoint in output-dir if present")

    return p.parse_args()


# ------------------------------------------------------------------
# Data loading — corpus scan + PIL→tensor
# ------------------------------------------------------------------

ALPHA_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]


@dataclass
class Sample:
    base: str
    seed: int
    alpha: float
    img_path: Path


def scan_corpus(args) -> list[Sample]:
    """Directory-walk fallback when --manifest is not given."""
    axis_subdir = args.axis_subdir or f"{args.axis}_inphase"
    root = Path(args.crossdemo_root) / args.axis / axis_subdir
    if not root.exists():
        raise FileNotFoundError(f"axis root not found: {root}")

    samples: list[Sample] = []
    holdout = set(args.holdout_bases)
    for base in ALL_BASES:
        if base in holdout:
            continue
        base_dir = root / base
        if not base_dir.exists():
            print(f"[warn] missing base dir {base_dir}; skipping")
            continue
        for seed in args.seeds:
            for alpha in ALPHA_LEVELS:
                img = base_dir / f"s{seed}_a{alpha:.2f}.png"
                if img.exists():
                    samples.append(Sample(base, seed, alpha, img))
                else:
                    print(f"[warn] missing {img}")
    return samples


def load_manifest(args) -> list[Sample]:
    """Load training cells from a curated manifest parquet.

    The manifest comes from `curate_slider_training.py` and contains
    rows from multiple corpora (v1, v2, v3, v3.1) that pass identity +
    edit-effect + confound gates. Each row is uniquely identified by
    img_path; we cache latents and route prompts by base.
    """
    import pandas as pd
    mf_path = Path(args.manifest)
    if not mf_path.exists():
        raise FileNotFoundError(f"manifest not found: {mf_path}")
    df = pd.read_parquet(mf_path)
    print(f"[manifest] loaded {len(df)} rows from {mf_path.name}")

    holdout = set(args.holdout_bases)
    samples: list[Sample] = []
    skipped_holdout = 0
    skipped_no_prompt = 0
    skipped_missing = 0
    unknown_bases: set[str] = set()
    for _, row in df.iterrows():
        base = str(row["base"])
        if base in holdout:
            skipped_holdout += 1
            continue
        if base not in BASE_PROMPTS:
            unknown_bases.add(base)
            skipped_no_prompt += 1
            continue
        img_path = REPO_ROOT / row["img_path"] if not Path(row["img_path"]).is_absolute() \
                   else Path(row["img_path"])
        if not img_path.exists():
            skipped_missing += 1
            continue
        samples.append(Sample(
            base=base,
            seed=int(row["seed"]),
            alpha=float(row["alpha"]),
            img_path=img_path,
        ))
    print(f"[manifest] kept {len(samples)} samples "
          f"(holdout={skipped_holdout}, no_prompt={skipped_no_prompt}, "
          f"missing={skipped_missing})")
    if unknown_bases:
        print(f"[manifest] WARNING: bases without prompts: {sorted(unknown_bases)}")
    return samples


def load_image_tensor(path: Path, resolution: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((resolution, resolution), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) * 2.0  # [-1, 1]
    return torch.from_numpy(arr).permute(2, 0, 1)  # CHW


# ------------------------------------------------------------------
# Precompute cache — VAE latents + text embeddings
# ------------------------------------------------------------------

def _atomic_save(obj, path: Path):
    """torch.save → os.replace so a power loss mid-write can't truncate
    the destination. Writes to a sibling .tmp and atomically renames."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_save_safetensors(state_dict, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    save_safetensors(state_dict, str(tmp))
    os.replace(tmp, path)


def precompute_cache(args, samples: list[Sample], device: torch.device) -> dict:
    """Encode images with VAE and prompts with T5+CLIP. Returns cache dict."""
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    cache_dir = Path(args.output_dir) / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    latents_file = cache_dir / f"latents_{args.axis}.pt"
    embeds_file = cache_dir / "text_embeds.pt"

    samples_file = cache_dir / f"samples_{args.axis}.pt"
    if latents_file.exists() and embeds_file.exists() and samples_file.exists():
        print(f"[cache] hit: {latents_file.name}, {embeds_file.name}, {samples_file.name}")
        latents_cache = torch.load(latents_file, map_location="cpu")
        embeds_cache = torch.load(embeds_file, map_location="cpu")
        sample_meta = torch.load(samples_file, map_location="cpu")
        return {"latents": latents_cache, "embeds": embeds_cache, "samples": sample_meta}

    print("[cache] miss — encoding")

    # --- VAE ---
    print("[cache] loading VAE from HF (config + weights, ~340 MB, cached)")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        args.flux_hf_id, subfolder="vae", torch_dtype=dtype
    ).to(device).eval()
    shift = vae.config.shift_factor
    scale = vae.config.scaling_factor

    # Latent cache is keyed by str(img_path) — avoids (base, seed, α)
    # collisions when the same demographic appears in multiple corpora
    # (v3 and v3.1 both have adult_middle_f/s2026_a0.45.png at different
    # paths but the (base, seed, α) tuple is identical).
    #
    # Resumability: if a partial latents_file exists from a prior crash,
    # load it and skip already-encoded paths. Saves the full hour of VAE
    # encoding when power trips mid-cache.
    latents_cache: dict[str, torch.Tensor] = {}
    if latents_file.exists():
        try:
            latents_cache = torch.load(latents_file, map_location="cpu")
            # Validate key type — older runs cached by (base, seed, alpha)
            # tuple. A mixed-schema dict would silently corrupt state.
            if latents_cache and not isinstance(next(iter(latents_cache)), str):
                print(f"[cache] old-format keys detected (not str) — restarting")
                latents_cache = {}
            else:
                print(f"[cache] resuming partial latents: {len(latents_cache)} already encoded")
        except Exception as e:
            print(f"[cache] could not load partial {latents_file.name}: {e} — restarting")
            latents_cache = {}

    sample_meta: list[dict] = []
    flush_every = 50  # persist every N new latents
    n_new = 0
    with torch.no_grad():
        for s in samples:
            key = str(s.img_path)
            sample_meta.append({"key": key, "base": s.base, "alpha": s.alpha})
            if key in latents_cache:
                continue
            x = load_image_tensor(s.img_path, args.resolution).unsqueeze(0).to(device, dtype)
            z = vae.encode(x).latent_dist.sample()
            z = (z - shift) * scale
            latents_cache[key] = z.squeeze(0).to("cpu", torch.bfloat16)
            n_new += 1
            if n_new % flush_every == 0:
                _atomic_save(latents_cache, latents_file)
                print(f"[cache] flushed {len(latents_cache)} latents "
                      f"({n_new} new this run)")
    if n_new and n_new % flush_every != 0:
        _atomic_save(latents_cache, latents_file)
    del vae
    torch.cuda.empty_cache()

    # --- Text encoders ---
    print("[cache] loading T5 + CLIP")
    from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer, T5EncoderModel
    tokenizer_clip = CLIPTokenizer.from_pretrained(args.flux_hf_id, subfolder="tokenizer")
    text_encoder_clip = CLIPTextModel.from_pretrained(
        args.flux_hf_id, subfolder="text_encoder", torch_dtype=dtype
    ).to(device).eval()
    tokenizer_t5 = AutoTokenizer.from_pretrained(args.flux_hf_id, subfolder="tokenizer_2")
    text_encoder_t5 = T5EncoderModel.from_pretrained(
        args.flux_hf_id, subfolder="text_encoder_2", torch_dtype=dtype
    ).to(device).eval()

    embeds_cache: dict[str, dict] = {}
    bases_in_train = {s.base for s in samples}
    for base in bases_in_train:
        prompt = BASE_PROMPTS[base]
        # CLIP-L — pooled 768-d
        tok_c = tokenizer_clip(
            prompt, padding="max_length", max_length=77, truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            pooled = text_encoder_clip(tok_c.input_ids, output_hidden_states=False).pooler_output
        # T5 — sequence [≤512, 4096] → project is done inside transformer
        tok_t = tokenizer_t5(
            prompt, padding="max_length", max_length=512, truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            seq = text_encoder_t5(tok_t.input_ids)[0]
        embeds_cache[base] = {
            "pooled": pooled.squeeze(0).to("cpu", torch.bfloat16),
            "seq": seq.squeeze(0).to("cpu", torch.bfloat16),
        }
    del text_encoder_clip, text_encoder_t5
    torch.cuda.empty_cache()

    _atomic_save(latents_cache, latents_file)
    _atomic_save(embeds_cache, embeds_file)
    _atomic_save(sample_meta, samples_file)
    print(f"[cache] wrote {latents_file.name} ({len(latents_cache)} latents), "
          f"{embeds_file.name} ({len(embeds_cache)} prompts), "
          f"{samples_file.name} ({len(sample_meta)} samples)")
    return {"latents": latents_cache, "embeds": embeds_cache, "samples": sample_meta}


# ------------------------------------------------------------------
# Model build — transformer + LoRA
# ------------------------------------------------------------------

def build_transformer(args, device: torch.device):
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    print("[model] loading Flux transformer (bf16)")
    transformer = FluxTransformer2DModel.from_single_file(
        args.flux_transformer, torch_dtype=dtype
    ).to(device)
    transformer.requires_grad_(False)
    transformer.enable_gradient_checkpointing()
    return transformer


def attach_lora(transformer, args):
    from peft import LoraConfig
    config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        target_modules=FLUX_XATTN_TARGETS,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",  # PEFT default; Kaiming-like for A, zeros for B
        bias="none",
    )
    if hasattr(transformer, "add_adapter"):
        transformer.add_adapter(config, adapter_name="slider")
    else:
        from peft import inject_adapter_in_model
        inject_adapter_in_model(config, transformer, adapter_name="slider")
    if hasattr(transformer, "enable_adapters"):
        transformer.enable_adapters()
    # Ensure requires_grad only on LoRA params
    for name, p in transformer.named_parameters():
        p.requires_grad = ("lora_" in name)

    trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    total = sum(p.numel() for p in transformer.parameters())
    print(f"[model] LoRA attached: {trainable:,} trainable / {total:,} total "
          f"({100 * trainable / total:.3f}%)")
    return transformer


def set_lora_scale(transformer, scale: float, adapter_name: str = "slider"):
    """Mutate PEFT's internal scaling dict. O(#LoRA modules)."""
    for _, module in transformer.named_modules():
        if hasattr(module, "scaling") and isinstance(module.scaling, dict):
            if adapter_name in module.scaling:
                # Base scale is lora_alpha / r. Multiply by our per-step α_train.
                # PEFT stores the base; we override with (base * scale) each step.
                if not hasattr(module, "_base_scaling"):
                    module._base_scaling = {k: v for k, v in module.scaling.items()}
                module.scaling[adapter_name] = module._base_scaling[adapter_name] * scale


# ------------------------------------------------------------------
# Flow-matching training utilities
# ------------------------------------------------------------------

def sample_logit_normal(batch_size: int, mu: float, sigma: float,
                         device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """t ~ sigmoid(N(mu, sigma))  in (0, 1)."""
    u = torch.randn(batch_size, device=device) * sigma + mu
    return torch.sigmoid(u).to(dtype)


def pack_latents(z: torch.Tensor) -> torch.Tensor:
    """Flux packs (B, C, H, W) → (B, H/2 * W/2, C*4) via 2x2 unfold."""
    B, C, H, W = z.shape
    z = z.view(B, C, H // 2, 2, W // 2, 2)
    z = z.permute(0, 2, 4, 1, 3, 5)  # B, H/2, W/2, C, 2, 2
    return z.reshape(B, (H // 2) * (W // 2), C * 4)


def unpack_latents(z_packed: torch.Tensor, H: int, W: int) -> torch.Tensor:
    B, _, Cp = z_packed.shape
    C = Cp // 4
    z = z_packed.view(B, H // 2, W // 2, C, 2, 2)
    z = z.permute(0, 3, 1, 4, 2, 5)
    return z.reshape(B, C, H, W)


def prepare_latent_image_ids(H: int, W: int, device, dtype):
    """2D positional IDs for image tokens, (H/2 * W/2, 3)."""
    ids = torch.zeros(H // 2, W // 2, 3)
    ids[..., 1] = ids[..., 1] + torch.arange(H // 2)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(W // 2)[None, :]
    return ids.reshape(-1, 3).to(device, dtype)


def prepare_text_ids(seq_len: int, device, dtype):
    return torch.zeros(seq_len, 3).to(device, dtype)


# ------------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------------

def training_loop(args, transformer, cache, device):
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    # --- optimizer
    lora_params = [p for p in transformer.parameters() if p.requires_grad]
    if args.optimizer_8bit:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                lora_params, lr=args.lr, weight_decay=args.weight_decay,
                betas=(0.9, 0.999), eps=1e-8,
            )
            print("[optim] AdamW 8-bit")
        except ImportError:
            optimizer = torch.optim.AdamW(
                lora_params, lr=args.lr, weight_decay=args.weight_decay,
            )
            print("[optim] AdamW (bitsandbytes not available)")
    else:
        optimizer = torch.optim.AdamW(
            lora_params, lr=args.lr, weight_decay=args.weight_decay,
        )

    # --- LR schedule: linear warmup → constant
    def lr_lambda(step):
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        return 1.0
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- sample index
    latents = cache["latents"]    # str(img_path) -> z (CHW)
    embeds = cache["embeds"]       # base -> {pooled, seq}
    samples = cache["samples"]     # list[{key, base, alpha}]

    # Auto-derive alpha grid from manifest if not given. Skip α=0 because
    # the velocity target is identically zero there (anchor==edited).
    # Quantise to 4 decimals — parquet stores α as float64 from arithmetic
    # upstream, so e.g. 0.45 may round-trip as 0.45000000000001 and silently
    # miss exact-equality lookup. 4 decimals fits all our α grids
    # (0.05, 0.10, 0.15, …) without ambiguity.
    def _q(a) -> float:
        return round(float(a), 4)

    if args.alphas_train is None:
        all_alphas = sorted({_q(s["alpha"]) for s in samples if _q(s["alpha"]) > 0})
        args.alphas_train = all_alphas
        print(f"[data] auto α grid from manifest: {args.alphas_train}")
    else:
        args.alphas_train = [_q(a) for a in args.alphas_train]

    keys_by_alpha: dict[float, list] = {_q(a): [] for a in args.alphas_train}
    n_unmatched = 0
    for s in samples:
        a = _q(s["alpha"])
        if a in keys_by_alpha:
            keys_by_alpha[a].append((s["key"], s["base"]))
        elif a > 0:
            n_unmatched += 1
    if n_unmatched:
        print(f"[data] {n_unmatched} non-anchor samples didn't match any α bucket "
              f"(α grid={args.alphas_train})")

    n_per_alpha = {a: len(v) for a, v in keys_by_alpha.items()}
    print(f"[data] samples per α: {n_per_alpha}")
    empty = [a for a, v in n_per_alpha.items() if v == 0]
    if empty:
        # Drop empty buckets rather than crash — manifest may not cover
        # every requested α.
        print(f"[data] dropping empty α buckets: {empty}")
        for a in empty:
            del keys_by_alpha[a]
        args.alphas_train = sorted(keys_by_alpha.keys())
        if not args.alphas_train:
            raise RuntimeError("no non-empty α buckets — check manifest / α grid")

    # --- training loop setup
    max_steps = 5 if args.dry_run else args.max_steps
    transformer.train()
    rng = random.Random(args.seed)

    # --- resume from checkpoint if requested
    start_step = 0
    resumed = False
    if args.resume:
        latest = find_latest_checkpoint(args.output_dir)
        if latest is not None:
            load_checkpoint(transformer, args.output_dir, latest, optimizer, lr_scheduler, py_rng=rng)
            start_step = latest
            resumed = True
            # CLI --lr overrides whatever was saved in the optimizer state.
            for pg in optimizer.param_groups:
                if abs(pg["lr"] - args.lr) > 1e-12:
                    print(f"[resume] overriding saved lr={pg['lr']:.2e} → CLI lr={args.lr:.2e}")
                    pg["lr"] = args.lr
                    pg["initial_lr"] = args.lr
        else:
            print("[resume] no checkpoint found; starting fresh")

    # --- CSV log: append whenever --resume was passed AND a log exists,
    # regardless of whether a checkpoint was found. Avoids overwriting
    # history if power died before the first checkpoint at step 100.
    log_path = Path(args.output_dir) / "train_log.csv"
    log_mode = "a" if args.resume and log_path.exists() else "w"
    log_f = open(log_path, log_mode)
    log_csv = csv.writer(log_f)
    if log_mode == "w":
        log_csv.writerow(["step", "loss", "lr", "alpha_train", "t_mean", "grad_norm", "step_time"])
    else:
        print(f"[log] appending to existing {log_path.name}")

    t0 = time.time()
    for step in range(start_step, max_steps):
        # --- sample
        alpha_train = rng.choice(args.alphas_train)
        picks = rng.sample(
            keys_by_alpha[alpha_train],
            k=min(args.batch_size, len(keys_by_alpha[alpha_train])),
        )
        z_after = torch.stack([latents[k] for (k, _b) in picks]).to(device, dtype)
        # z_after is stored as [C, H, W] per-sample from precompute
        B, C, H, W = z_after.shape

        pooled = torch.stack([embeds[b]["pooled"] for (_k, b) in picks]).to(device, dtype)
        seq = torch.stack([embeds[b]["seq"] for (_k, b) in picks]).to(device, dtype)

        # --- timestep sampling (logit-normal) + Flux sigma shift
        u = sample_logit_normal(B, args.logit_mean, args.logit_std, device, dtype)
        # Apply Flux sigma shift: sigma = shift * u / (1 + (shift - 1) * u)
        shift = args.flux_sigma_shift
        t_cont = (shift * u) / (1 + (shift - 1) * u)

        t_scalar = t_cont.view(B, 1, 1, 1)
        noise = torch.randn_like(z_after)
        z_t = (1 - t_scalar) * z_after + t_scalar * noise
        v_target = noise - z_after  # straight-line velocity

        # --- pack for transformer input
        z_t_packed = pack_latents(z_t)
        # Force gradient flow through grad-checkpointed frozen trunk.
        # Without this, the trunk has no requires_grad=True inputs and
        # PyTorch's checkpoint can skip recomputation → zero LoRA grads.
        z_t_packed.requires_grad_(True)
        img_ids = prepare_latent_image_ids(H, W, device, dtype)
        txt_ids = prepare_text_ids(seq.shape[1], device, dtype)

        # --- set LoRA scale = α_train
        set_lora_scale(transformer, alpha_train)

        # --- guidance (conditional on model type)
        if getattr(transformer.config, "guidance_embeds", False):
            guidance = torch.full((B,), args.guidance_scale, device=device, dtype=dtype)
        else:
            guidance = None

        # --- forward
        with torch.autocast("cuda", dtype=dtype):
            v_hat_packed = transformer(
                hidden_states=z_t_packed,
                timestep=t_cont,  # already in [0, 1]
                guidance=guidance,
                pooled_projections=pooled,
                encoder_hidden_states=seq,
                txt_ids=txt_ids,
                img_ids=img_ids,
                return_dict=False,
            )[0]
        v_hat = unpack_latents(v_hat_packed, H, W)

        loss = F.mse_loss(v_hat.float(), v_target.float()) / args.grad_accum

        loss.backward()

        # --- grad accum
        if (step + 1) % args.grad_accum == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, args.max_grad_norm).item()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()
        else:
            grad_norm = float("nan")

        # --- log
        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0
        loss_val = loss.item() * args.grad_accum
        if step % args.log_every == 0:
            print(f"step {step:5d} | loss {loss_val:.4f} | lr {lr_now:.2e} | "
                  f"α {alpha_train:.2f} | t̄ {t_cont.mean().item():.2f} | "
                  f"gn {grad_norm:.3f} | {dt/(step+1):.2f} s/step")
        log_csv.writerow([step, loss_val, lr_now, alpha_train,
                          t_cont.mean().item(), grad_norm, dt / (step + 1)])
        log_f.flush()

        # --- checkpoint
        if (step + 1) % args.checkpoint_every == 0 and not args.dry_run:
            save_lora_checkpoint(transformer, args.output_dir, step + 1,
                                 optimizer, lr_scheduler, py_rng=rng)

    log_f.close()
    if not args.dry_run:
        save_lora_checkpoint(transformer, args.output_dir, max_steps,
                             optimizer, lr_scheduler, py_rng=rng)
    print(f"[done] {max_steps} steps in {time.time() - t0:.1f} s")


# ------------------------------------------------------------------
# Saving
# ------------------------------------------------------------------

def save_lora_checkpoint(transformer, output_dir: str, step: int,
                          optimizer=None, lr_scheduler=None,
                          py_rng: Optional[random.Random] = None):
    from peft.utils import get_peft_model_state_dict
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # LoRA weights → safetensors (bf16, merge-ready). Atomic write so a
    # power loss mid-save can't truncate and break the resume path.
    state = get_peft_model_state_dict(transformer, adapter_name="slider")
    state_bf16 = {k: v.detach().to(torch.bfloat16).contiguous() for k, v in state.items()}
    ckpt_path = out / f"slider_step{step:06d}.safetensors"
    _atomic_save_safetensors(state_bf16, ckpt_path)
    # Training state (optimizer + LR sched + RNGs + step) → separate .pt
    if optimizer is not None or lr_scheduler is not None:
        train_state = {"step": step}
        if optimizer is not None:
            train_state["optimizer"] = optimizer.state_dict()
        if lr_scheduler is not None:
            train_state["lr_scheduler"] = lr_scheduler.state_dict()
        train_state["torch_rng"] = torch.get_rng_state()
        train_state["cuda_rng"] = torch.cuda.get_rng_state()
        if py_rng is not None:
            train_state["py_rng"] = py_rng.getstate()
        state_path = out / f"training_state_step{step:06d}.pt"
        _atomic_save(train_state, state_path)
        print(f"[save] {ckpt_path.name} + {state_path.name}")
    else:
        print(f"[save] {ckpt_path.name} ({len(state_bf16)} tensors)")


def find_latest_checkpoint(output_dir: str) -> Optional[int]:
    """Return the highest step where BOTH the LoRA safetensors and the
    training_state .pt exist AND pass a torch.load smoke-test. Skips
    incomplete/corrupt pairs left by a mid-save power loss. Filenames are
    zero-padded (`slider_step000500`) so lexicographic sort == numeric."""
    from safetensors.torch import load_file as load_safetensors
    out = Path(output_dir)
    if not out.exists():
        return None
    ckpts = sorted(out.glob("slider_step*.safetensors"), reverse=True)
    for ckpt in ckpts:
        name = ckpt.stem
        try:
            step = int(name.replace("slider_step", ""))
        except ValueError:
            continue
        state_path = out / f"training_state_step{step:06d}.pt"
        # Smoke-test: both files load without error.
        try:
            load_safetensors(str(ckpt))
        except Exception as e:
            print(f"[resume] skipping corrupt {ckpt.name}: {e}")
            continue
        if state_path.exists():
            try:
                torch.load(state_path, map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"[resume] skipping {ckpt.name} — training state {state_path.name} corrupt: {e}")
                continue
        return step
    return None


def load_checkpoint(transformer, output_dir: str, step: int, optimizer, lr_scheduler,
                    py_rng: Optional[random.Random] = None):
    from safetensors.torch import load_file as load_safetensors
    from peft.utils import set_peft_model_state_dict
    out = Path(output_dir)
    lora_path = out / f"slider_step{step:06d}.safetensors"
    state_path = out / f"training_state_step{step:06d}.pt"
    if not lora_path.exists():
        raise FileNotFoundError(lora_path)

    lora_state = load_safetensors(str(lora_path))
    set_peft_model_state_dict(transformer, lora_state, adapter_name="slider")
    print(f"[resume] loaded LoRA from {lora_path.name}")

    if state_path.exists():
        try:
            train_state = torch.load(state_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[resume] WARNING: training state {state_path.name} corrupt: {e}")
            print(f"[resume] continuing with fresh optimizer/LR/RNG (LoRA was loaded OK)")
            return
        if "optimizer" in train_state:
            optimizer.load_state_dict(train_state["optimizer"])
        if "lr_scheduler" in train_state:
            lr_scheduler.load_state_dict(train_state["lr_scheduler"])
        if "torch_rng" in train_state:
            torch.set_rng_state(train_state["torch_rng"])
        if "cuda_rng" in train_state:
            torch.cuda.set_rng_state(train_state["cuda_rng"])
        if "py_rng" in train_state and py_rng is not None:
            py_rng.setstate(train_state["py_rng"])
        print(f"[resume] optimizer + LR sched + RNG restored from {state_path.name}")
    else:
        print(f"[resume] no training state at {state_path}; optimizer starts fresh")


# ------------------------------------------------------------------
# Entry
# ------------------------------------------------------------------

def main():
    args = parse_args()
    _import_heavy()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda")

    # --- data
    if args.manifest:
        samples = load_manifest(args)
        mode = f"manifest={args.manifest}"
    else:
        samples = scan_corpus(args)
        mode = f"directory={args.crossdemo_root}"
    print(f"[data] {len(samples)} training samples from {len(set(s.base for s in samples))} bases  ({mode})")
    if not samples:
        raise RuntimeError("no training samples found — check --manifest / --crossdemo-root")

    # --- cache (VAE + text encode, one-shot)
    cache = precompute_cache(args, samples, device)

    # --- model
    transformer = build_transformer(args, device)
    transformer = attach_lora(transformer, args)

    # --- train
    training_loop(args, transformer, cache, device)


if __name__ == "__main__":
    main()
