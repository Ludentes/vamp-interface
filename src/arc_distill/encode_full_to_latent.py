"""VAE-encode FFHQ-512² full images to (16, 64, 64) latents.

Pairs with the arcface teacher embeddings in compact.pt (computed on the
aligned 112² crop of the same image). For each SHA in compact.pt, locate
the original PNG bytes in the 190-shard FFHQ parquet, decode → resize 512²
LANCZOS → Flux VAE encode → store (16, 64, 64) bf16.

Output: <out> with {latents (N, 16, 64, 64) bf16, arcface (N, 512) fp32,
shas, found (N,) bool, resolution=64, format_version=2, source_resolution=512}.

Slider training operates at 512² (verified in
ai-toolkit/config/glasses_slider_v0.yaml: width=512, height=512), giving
the same (16, 64, 64) Flux VAE latent shape this script produces.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image  # type: ignore

from arc_distill.encode_aligned_to_latent import FLUX_VAE_CONFIG, build_flux_vae


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--compact", type=Path, required=True,
                   help="compact.pt with shas + arcface (target rows + teacher).")
    p.add_argument("--ffhq-parquet-dir", type=Path, required=True,
                   help="Directory with HF FFHQ shards (train-NNNNN-of-XXXXX.parquet).")
    p.add_argument("--vae", type=Path, required=True,
                   help="Flux VAE safetensors (e.g. ae.safetensors).")
    p.add_argument("--out", type=Path, required=True,
                   help="Output compact.pt (full-image latents).")
    p.add_argument("--resolution", type=int, default=512,
                   help="Resize target before VAE encode. 512 → (16,64,64); 1024 → (16,128,128).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.resolution % 8 != 0:
        raise ValueError(f"resolution {args.resolution} not divisible by 8 (Flux VAE stride)")
    LH = args.resolution // 8

    print(f"loading compact: {args.compact}")
    blob = torch.load(args.compact, map_location="cpu", weights_only=False)
    target_shas: list[str] = list(blob["shas"])
    arcface: torch.Tensor = blob["arcface"]  # (N, 512) fp32
    n = len(target_shas)
    sha_to_idx = {s: i for i, s in enumerate(target_shas)}
    print(f"  N={n}  arcface={tuple(arcface.shape)}")

    print(f"building Flux VAE from {args.vae}")
    vae = build_flux_vae(args.vae, torch.bfloat16, args.device)
    shift = float(FLUX_VAE_CONFIG["shift_factor"])
    scale = float(FLUX_VAE_CONFIG["scaling_factor"])
    print(f"  shift={shift} scale={scale}  -> latent ({LH},{LH})")

    latents = torch.zeros((n, 16, LH, LH), dtype=torch.bfloat16)
    found = torch.zeros(n, dtype=torch.bool)

    pending_imgs: list[torch.Tensor] = []
    pending_idx: list[int] = []

    @torch.no_grad()
    def flush() -> None:
        if not pending_imgs:
            return
        x = torch.stack(pending_imgs).to(args.device, torch.bfloat16)
        z = vae.encode(x).latent_dist.sample()
        z = (z - shift) * scale
        for j, idx in enumerate(pending_idx):
            latents[idx] = z[j].to("cpu", torch.bfloat16)
            found[idx] = True
        pending_imgs.clear()
        pending_idx.clear()

    shards = sorted(args.ffhq_parquet_dir.glob("train-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no FFHQ shards under {args.ffhq_parquet_dir}")
    print(f"shards={len(shards)}")

    n_seen = 0
    t0 = time.time()
    for s_idx, s_path in enumerate(shards):
        if found.all():
            break
        table = pq.read_table(s_path, columns=["image"])
        img_col = table.column("image").to_pylist()
        for row in img_col:
            n_seen += 1
            if not row:
                continue
            img_bytes = row.get("bytes") if isinstance(row, dict) else row
            if not img_bytes:
                continue
            sha = sha256_bytes(img_bytes)
            idx = sha_to_idx.get(sha)
            if idx is None or found[idx]:
                continue
            try:
                im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                if im.size != (args.resolution, args.resolution):
                    im = im.resize((args.resolution, args.resolution), Image.LANCZOS)
            except Exception:
                continue
            arr = np.asarray(im, dtype=np.float32) / 255.0          # (H, H, 3) [0,1]
            t_im = torch.from_numpy(arr).permute(2, 0, 1)           # (3, H, H)
            t_im = (t_im - 0.5) * 2.0                               # [-1, 1]
            pending_imgs.append(t_im)
            pending_idx.append(idx)
            if len(pending_imgs) >= args.batch_size:
                flush()
        elapsed = time.time() - t0
        n_found = int(found.sum())
        rate = n_seen / max(elapsed, 1e-6)
        print(f"  shard {s_idx+1}/{len(shards)} {s_path.name}: "
              f"seen={n_seen} found={n_found}/{n} ({rate:.0f} scan/s, {elapsed:.0f}s)")
    flush()

    elapsed = time.time() - t0
    n_found = int(found.sum())
    print(f"done: found {n_found}/{n} in {elapsed:.0f}s ({n_found/max(elapsed,1e-6):.1f} enc/s)")
    if n_found < n:
        missing = [s for s, f in zip(target_shas, found.tolist()) if not f]
        print(f"WARNING: {len(missing)} target SHAs not found.")
        print(f"  first 5 missing: {missing[:5]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "latents": latents,
        "arcface": arcface,
        "shas": target_shas,
        "found": found,
        "resolution": LH,
        "source_resolution": args.resolution,
        "format_version": 2,
    }, args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
