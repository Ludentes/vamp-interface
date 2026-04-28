"""VAE-encode the aligned 112-px FFHQ crops in compact.pt to (16, 14, 14) latents.

Reads:  compact.pt {images_u8 (N,3,112,112), arcface (N,512), shas, ...}
Writes: compact_latent.pt {latents (N,16,14,14) bf16, arcface (N,512) fp32, shas, ...}

The Flux VAE is stride-8: 112 -> 14 spatially. Identity-relevant geometry now
flows through the VAE bottleneck on aligned crops, parallel to how Pixel-A
sees the same crops at full pixel resolution.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from diffusers import AutoencoderKL  # type: ignore
from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint  # type: ignore
from safetensors.torch import load_file  # type: ignore

FLUX_VAE_CONFIG = {
    "_class_name": "AutoencoderKL",
    "_diffusers_version": "0.30.0",
    "act_fn": "silu",
    "block_out_channels": [128, 256, 512, 512],
    "down_block_types": ["DownEncoderBlock2D"] * 4,
    "force_upcast": True,
    "in_channels": 3,
    "latent_channels": 16,
    "layers_per_block": 2,
    "mid_block_add_attention": True,
    "norm_num_groups": 32,
    "out_channels": 3,
    "sample_size": 1024,
    "scaling_factor": 0.3611,
    "shift_factor": 0.1159,
    "up_block_types": ["UpDecoderBlock2D"] * 4,
    "use_post_quant_conv": False,
    "use_quant_conv": False,
}


def build_flux_vae(weights_path: Path, dtype: torch.dtype, device: str):
    vae = AutoencoderKL.from_config(FLUX_VAE_CONFIG)
    state = load_file(str(weights_path))
    converted = convert_ldm_vae_checkpoint(state, FLUX_VAE_CONFIG)
    missing, unexpected = vae.load_state_dict(converted, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"VAE state_dict mismatch: {len(missing)} missing, {len(unexpected)} unexpected")
    return vae.to(device, dtype).eval()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--compact", type=Path, required=True,
                   help="Input compact.pt with images_u8 (N,3,112,112) RGB.")
    p.add_argument("--vae", type=Path, required=True,
                   help="Flux VAE safetensors (e.g. ae.safetensors).")
    p.add_argument("--out", type=Path, required=True,
                   help="Output compact_latent.pt path.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16

    print(f"loading compact: {args.compact}")
    blob = torch.load(args.compact, map_location="cpu", weights_only=False)
    images = blob["images_u8"]   # (N, 3, 112, 112) uint8
    arcface = blob["arcface"]    # (N, 512) fp32
    shas = blob["shas"]
    n = images.size(0)
    print(f"  N={n}  images={tuple(images.shape)}  arcface={tuple(arcface.shape)}")
    if images.size(-1) != 112 or images.size(-2) != 112:
        raise ValueError(f"expected 112x112, got {tuple(images.shape)}")

    print(f"building Flux VAE from {args.vae}")
    vae = build_flux_vae(args.vae, dtype, args.device)
    shift = float(FLUX_VAE_CONFIG["shift_factor"])
    scale = float(FLUX_VAE_CONFIG["scaling_factor"])
    print(f"  shift={shift} scale={scale}")

    latents = torch.empty((n, 16, 14, 14), dtype=torch.bfloat16)
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            end = min(n, start + args.batch_size)
            x = images[start:end].to(torch.float32) / 255.0          # (B,3,112,112) [0,1]
            x = (x - 0.5) * 2.0                                      # [-1, 1]
            x = x.to(args.device, dtype)
            z = vae.encode(x).latent_dist.sample()                   # (B,16,14,14)
            z = (z - shift) * scale
            latents[start:end] = z.to("cpu", torch.bfloat16)
            if (start // args.batch_size) % 10 == 0:
                elapsed = time.time() - t0
                rate = end / max(elapsed, 1e-6)
                print(f"  {end}/{n}  ({rate:.1f} img/s, {elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"encoded {n} rows in {elapsed:.1f}s ({n/elapsed:.1f} img/s)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "latents": latents,
        "arcface": arcface,
        "shas": shas,
        "resolution": 14,
        "format_version": 1,
    }, args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
