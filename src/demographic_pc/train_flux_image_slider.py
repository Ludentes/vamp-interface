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

# Base prompts — copied verbatim from fluxspace_metrics.CALIBRATION_PROMPTS
# to keep this file self-contained (no cross-import during training).
BASE_PROMPTS: dict[str, str] = {
    "asian_m":         "A photorealistic portrait photograph of an adult East Asian man, neutral expression, plain grey background, studio lighting, sharp focus.",
    "black_f":         "A photorealistic portrait photograph of an adult Black woman, neutral expression, plain grey background, studio lighting, sharp focus.",
    "european_m":      "A photorealistic portrait photograph of an adult European man, neutral expression, plain grey background, studio lighting, sharp focus.",
    "elderly_latin_m": "A photorealistic portrait photograph of an elderly Latin American man, neutral expression, plain grey background, studio lighting, sharp focus.",
    "young_european_f":"A photorealistic portrait photograph of a young European woman, neutral expression, plain grey background, studio lighting, sharp focus.",
    "southasian_f":    "A photorealistic portrait photograph of an adult South Asian woman, neutral expression, plain grey background, studio lighting, sharp focus.",
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
                   help="subdir under axis dir (default: <axis>_inphase)")
    p.add_argument("--crossdemo-root",
                   default=str(REPO_ROOT / "output/demographic_pc/fluxspace_metrics/crossdemo"))
    p.add_argument("--holdout-bases", nargs="+", default=["european_m"],
                   help="bases to exclude from training, use for eval")
    p.add_argument("--seeds", type=int, nargs="+", default=[1337, 2026, 4242])
    p.add_argument("--alphas-train", type=float, nargs="+",
                   default=[0.25, 0.5, 0.75, 1.0],
                   help="α values sampled during training (α=0 target is tautologically 0)")

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
    p.add_argument("--checkpoint-every", type=int, default=500)
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


def load_image_tensor(path: Path, resolution: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((resolution, resolution), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) * 2.0  # [-1, 1]
    return torch.from_numpy(arr).permute(2, 0, 1)  # CHW


# ------------------------------------------------------------------
# Precompute cache — VAE latents + text embeddings
# ------------------------------------------------------------------

def precompute_cache(args, samples: list[Sample], device: torch.device) -> dict:
    """Encode images with VAE and prompts with T5+CLIP. Returns cache dict."""
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16

    cache_dir = Path(args.output_dir) / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    latents_file = cache_dir / f"latents_{args.axis}.pt"
    embeds_file = cache_dir / "text_embeds.pt"

    if latents_file.exists() and embeds_file.exists():
        print(f"[cache] hit: {latents_file.name}, {embeds_file.name}")
        latents_cache = torch.load(latents_file, map_location="cpu")
        embeds_cache = torch.load(embeds_file, map_location="cpu")
        return {"latents": latents_cache, "embeds": embeds_cache}

    print("[cache] miss — encoding")

    # --- VAE ---
    print("[cache] loading VAE from HF (config + weights, ~340 MB, cached)")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        args.flux_hf_id, subfolder="vae", torch_dtype=dtype
    ).to(device).eval()
    shift = vae.config.shift_factor
    scale = vae.config.scaling_factor

    latents_cache: dict[tuple[str, int, float], torch.Tensor] = {}
    with torch.no_grad():
        for s in samples:
            x = load_image_tensor(s.img_path, args.resolution).unsqueeze(0).to(device, dtype)
            z = vae.encode(x).latent_dist.sample()
            z = (z - shift) * scale
            latents_cache[(s.base, s.seed, s.alpha)] = z.squeeze(0).to("cpu", torch.bfloat16)
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

    torch.save(latents_cache, latents_file)
    torch.save(embeds_cache, embeds_file)
    print(f"[cache] wrote {latents_file.name} ({len(latents_cache)} latents), "
          f"{embeds_file.name} ({len(embeds_cache)} prompts)")
    return {"latents": latents_cache, "embeds": embeds_cache}


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
    latents = cache["latents"]   # (base, seed, alpha) -> z (packed latent CHW)
    embeds = cache["embeds"]      # base -> {pooled, seq}
    keys_by_alpha: dict[float, list] = {a: [] for a in args.alphas_train}
    for (base, seed, alpha), _ in latents.items():
        if alpha in keys_by_alpha:
            keys_by_alpha[alpha].append((base, seed, alpha))

    n_per_alpha = {a: len(v) for a, v in keys_by_alpha.items()}
    print(f"[data] samples per α: {n_per_alpha}")
    if any(v == 0 for v in n_per_alpha.values()):
        raise RuntimeError(f"empty α bucket: {n_per_alpha}")

    # --- CSV log
    log_path = Path(args.output_dir) / "train_log.csv"
    log_f = open(log_path, "w")
    log_csv = csv.writer(log_f)
    log_csv.writerow(["step", "loss", "lr", "alpha_train", "t_mean", "grad_norm", "step_time"])

    # --- training loop
    max_steps = 5 if args.dry_run else args.max_steps
    transformer.train()
    rng = random.Random(args.seed)

    # --- resume from checkpoint if requested
    start_step = 0
    if args.resume:
        latest = find_latest_checkpoint(args.output_dir)
        if latest is not None:
            load_checkpoint(transformer, args.output_dir, latest, optimizer, lr_scheduler)
            start_step = latest
        else:
            print("[resume] no checkpoint found; starting fresh")

    t0 = time.time()
    for step in range(start_step, max_steps):
        # --- sample
        alpha_train = rng.choice(args.alphas_train)
        picks = rng.sample(
            keys_by_alpha[alpha_train],
            k=min(args.batch_size, len(keys_by_alpha[alpha_train])),
        )
        z_after = torch.stack([latents[k] for k in picks]).to(device, dtype)
        # z_after is stored as [C, H, W] per-sample from precompute
        B, C, H, W = z_after.shape

        pooled = torch.stack([embeds[k[0]]["pooled"] for k in picks]).to(device, dtype)
        seq = torch.stack([embeds[k[0]]["seq"] for k in picks]).to(device, dtype)

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
                                 optimizer, lr_scheduler)

    log_f.close()
    if not args.dry_run:
        save_lora_checkpoint(transformer, args.output_dir, max_steps,
                             optimizer, lr_scheduler)
    print(f"[done] {max_steps} steps in {time.time() - t0:.1f} s")


# ------------------------------------------------------------------
# Saving
# ------------------------------------------------------------------

def save_lora_checkpoint(transformer, output_dir: str, step: int,
                          optimizer=None, lr_scheduler=None):
    from peft.utils import get_peft_model_state_dict
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # LoRA weights → safetensors (bf16, merge-ready)
    state = get_peft_model_state_dict(transformer, adapter_name="slider")
    state_bf16 = {k: v.detach().to(torch.bfloat16).contiguous() for k, v in state.items()}
    ckpt_path = out / f"slider_step{step:06d}.safetensors"
    save_safetensors(state_bf16, str(ckpt_path))
    # Training state (optimizer + LR sched + step) → separate .pt for resume
    if optimizer is not None or lr_scheduler is not None:
        train_state = {"step": step}
        if optimizer is not None:
            train_state["optimizer"] = optimizer.state_dict()
        if lr_scheduler is not None:
            train_state["lr_scheduler"] = lr_scheduler.state_dict()
        train_state["torch_rng"] = torch.get_rng_state()
        train_state["cuda_rng"] = torch.cuda.get_rng_state()
        state_path = out / f"training_state_step{step:06d}.pt"
        torch.save(train_state, state_path)
        print(f"[save] {ckpt_path.name} + {state_path.name}")
    else:
        print(f"[save] {ckpt_path.name} ({len(state_bf16)} tensors)")


def find_latest_checkpoint(output_dir: str) -> Optional[int]:
    out = Path(output_dir)
    if not out.exists():
        return None
    ckpts = sorted(out.glob("slider_step*.safetensors"))
    if not ckpts:
        return None
    # Extract step number from filename
    name = ckpts[-1].stem  # slider_step000500
    try:
        return int(name.replace("slider_step", ""))
    except ValueError:
        return None


def load_checkpoint(transformer, output_dir: str, step: int, optimizer, lr_scheduler):
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
        train_state = torch.load(state_path, map_location="cpu", weights_only=False)
        if "optimizer" in train_state:
            optimizer.load_state_dict(train_state["optimizer"])
        if "lr_scheduler" in train_state:
            lr_scheduler.load_state_dict(train_state["lr_scheduler"])
        if "torch_rng" in train_state:
            torch.set_rng_state(train_state["torch_rng"])
        if "cuda_rng" in train_state:
            torch.cuda.set_rng_state(train_state["cuda_rng"])
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
    samples = scan_corpus(args)
    print(f"[data] {len(samples)} training samples from {len(set(s.base for s in samples))} bases")
    if not samples:
        raise RuntimeError("no training samples found — check --crossdemo-root and --axis")

    # --- cache (VAE + text encode, one-shot)
    cache = precompute_cache(args, samples, device)

    # --- model
    transformer = build_transformer(args, device)
    transformer = attach_lora(transformer, args)

    # --- train
    training_loop(args, transformer, cache, device)


if __name__ == "__main__":
    main()
