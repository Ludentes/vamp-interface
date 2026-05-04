"""VAE-encode local Flux-rendered PNGs to (16, 64, 64) latents for v2c.

Pairs `flux_corpus_v3` rows in `reverse_index.parquet` (7,772 rendered images
with cached MediaPipe blendshapes from prior NMF work) with VAE latents at
512² resolution — same VAE settings as `arc_distill/encode_full_to_latent.py`,
so the resulting latents drop into the same training loop as the FFHQ corpus.

Output: `compact_rendered.pt` with {latents (N, 16, 64, 64) bf16,
blendshapes (N, 52) fp32, detected (N,) bool, channel_names list[str],
shas list[str], img_paths list[str], format_version 1}.

This is the "wider expression coverage" data that v1 didn't see — slider
sweeps deliberately induced expressions (smile, jaw, brow, anger, surprise,
disgust, pucker, lip_press) that hide in FFHQ's tail.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image  # type: ignore

from arc_distill.encode_aligned_to_latent import FLUX_VAE_CONFIG, build_flux_vae
from .build_compact_blendshapes import BLENDSHAPE_CHANNELS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reverse-index", type=Path,
                   default=Path("models/blendshape_nmf/sample_index.parquet"),
                   help="parquet with img_path, image_sha256, and bs_* columns. "
                        "Default sample_index.parquet which carries img_path; "
                        "reverse_index.parquet does not.")
    p.add_argument("--vae", type=Path, required=True,
                   help="Flux VAE safetensors (e.g. ae.safetensors)")
    p.add_argument("--out", type=Path, required=True,
                   help="output .pt path, e.g. compact_rendered.pt")
    p.add_argument("--source", default=None,
                   help="optional reverse_index source filter; default is "
                        "all rows in the parquet")
    p.add_argument("--resolution", type=int, default=512,
                   help="resize PNGs to this square before VAE encode (must be /8)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.resolution % 8 != 0:
        raise ValueError(f"resolution {args.resolution} not divisible by 8")
    LH = args.resolution // 8

    bs_cols = [f"bs_{c}" for c in BLENDSHAPE_CHANNELS]

    print(f"loading reverse_index: {args.reverse_index}")
    df = pd.read_parquet(args.reverse_index)
    if args.source:
        df = df[df["source"] == args.source].copy().reset_index(drop=True)
    else:
        df = df.copy().reset_index(drop=True)
    print(f"  rows after filter (source={args.source}): {len(df)}")

    missing = [c for c in bs_cols + ["img_path", "image_sha256"]
               if c not in df.columns]
    if missing:
        raise ValueError(f"parquet missing columns: {missing[:5]}")
    has_detected_col = "bs_detected" in df.columns

    # Verify image paths exist (rendered images live in output/demographic_pc/...).
    n_missing = 0
    for p in df["img_path"].head(20):
        if not Path(p).exists():
            n_missing += 1
    if n_missing > 0:
        raise FileNotFoundError(f"{n_missing}/20 sample img_paths missing")

    print(f"building Flux VAE from {args.vae}")
    vae = build_flux_vae(args.vae, torch.bfloat16, args.device)
    shift = float(FLUX_VAE_CONFIG["shift_factor"])
    scale = float(FLUX_VAE_CONFIG["scaling_factor"])
    print(f"  shift={shift} scale={scale}  -> latent ({LH},{LH})")

    n = len(df)
    latents = torch.zeros((n, 16, LH, LH), dtype=torch.bfloat16)
    found = torch.zeros(n, dtype=torch.bool)

    pending: list[torch.Tensor] = []
    pending_idx: list[int] = []

    @torch.no_grad()
    def flush():
        if not pending:
            return
        x = torch.stack(pending).to(args.device, torch.bfloat16)
        z = vae.encode(x).latent_dist.sample()
        z = (z - shift) * scale
        for j, idx in enumerate(pending_idx):
            latents[idx] = z[j].to("cpu", torch.bfloat16)
            found[idx] = True
        pending.clear()
        pending_idx.clear()

    t0 = time.time()
    for i, row in df.iterrows():
        path = Path(row["img_path"])
        try:
            im = Image.open(path).convert("RGB")
            if im.size != (args.resolution, args.resolution):
                im = im.resize((args.resolution, args.resolution),
                               Image.Resampling.LANCZOS)
        except Exception:
            continue
        arr = np.asarray(im, dtype=np.float32) / 255.0
        t_im = torch.from_numpy(arr).permute(2, 0, 1)
        t_im = (t_im - 0.5) * 2.0
        pending.append(t_im)
        pending_idx.append(int(i))
        if len(pending) >= args.batch_size:
            flush()
        if (int(i) + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  encoded {int(i)+1}/{n} ({(int(i)+1)/elapsed:.1f}/s)")
    flush()

    blendshapes = torch.from_numpy(df[bs_cols].to_numpy(dtype="float32"))
    if has_detected_col:
        detected = torch.from_numpy(df["bs_detected"].to_numpy(dtype=bool))
    else:
        # sample_index.parquet doesn't carry bs_detected — but verified earlier
        # that all 7,772 rows have populated blendshapes (sum > 0). Treat all
        # as detected, gated only by VAE-encode success.
        detected = torch.ones(n, dtype=torch.bool)
    detected = detected & found

    elapsed = time.time() - t0
    print(f"done: encoded {int(found.sum())}/{n} in {elapsed:.0f}s "
          f"({int(found.sum())/max(elapsed,1e-6):.1f}/s)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "latents": latents,
        "blendshapes": blendshapes,
        "detected": detected,
        "channel_names": BLENDSHAPE_CHANNELS,
        "shas": df["image_sha256"].tolist(),
        "img_paths": df["img_path"].tolist(),
        "resolution": LH,
        "source_resolution": args.resolution,
        "format_version": 1,
        "source_tag": args.source,
    }, args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
