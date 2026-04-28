"""Pre-pack detected FFHQ rows into one compact file for arc_pixel training.

Scans every shard pair (parquet + encoded .pt), decodes + resizes detected
images once, and writes a single .pt holding:
  images_u8:   (N, 3, 224, 224) uint8
  arcface:     (N, 512) float32
  shas:        list[str] of length N
  format_version: 1

At 224² uint8 + fp32 target this is ~3.7 GB for ~25k detected FFHQ rows —
fits in RAM, so subsequent training avoids per-step parquet/PIL costs entirely.
"""
from __future__ import annotations

import argparse
import io
import re
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

_SHARD_RE = re.compile(r"train-(\d{5})-of-\d{5}\.parquet$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-dir", type=Path, required=True)
    ap.add_argument("--encoded-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--limit-shards", type=int, default=0,
                    help="Process only the first N shards (0=all). Smoke flag.")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    shards = sorted(args.shards_dir.glob("train-*-of-*.parquet"))
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    print(f"scanning {len(shards)} shards under {args.shards_dir}")

    out_imgs: list[np.ndarray] = []
    out_arc: list[torch.Tensor] = []
    out_shas: list[str] = []

    t_start = time.time()
    for s_idx, p in enumerate(shards):
        m = _SHARD_RE.search(p.name)
        if not m:
            continue
        pt_path = args.encoded_dir / (p.stem + ".pt")
        if not pt_path.exists():
            print(f"  [{s_idx+1}/{len(shards)}] skip {p.name}: no encoded .pt")
            continue

        pt = torch.load(pt_path, map_location="cpu", weights_only=False)
        shas = pt["image_sha256"]
        arc = pt["arcface_fp32"]
        det = pt["detected"].tolist()

        table = pq.read_table(p, columns=["image"])
        img_col = table.column("image").to_pylist()

        if len(img_col) != len(shas):
            raise ValueError(f"row count mismatch in {p.name}")

        kept = 0
        for i, (sha, d) in enumerate(zip(shas, det)):
            if not d:
                continue
            img = Image.open(io.BytesIO(img_col[i]["bytes"])).convert("RGB")
            img = img.resize((args.resolution, args.resolution), Image.Resampling.BILINEAR)
            arr = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)  # CHW
            out_imgs.append(arr)
            out_arc.append(arc[i].clone())
            out_shas.append(sha)
            kept += 1

        # release the parquet table before next iter
        del table, img_col
        elapsed = time.time() - t_start
        print(f"  [{s_idx+1}/{len(shards)}] {p.name} kept={kept} "
              f"total={len(out_imgs)} elapsed={elapsed:.1f}s")

    print(f"packing {len(out_imgs)} rows into tensors...")
    images_u8 = torch.from_numpy(np.stack(out_imgs))
    arcface = torch.stack(out_arc).to(torch.float32)
    print(f"images_u8: {images_u8.shape} {images_u8.dtype} "
          f"({images_u8.numel() / 1e9:.2f} GB)")
    print(f"arcface:   {arcface.shape}")

    torch.save({
        "images_u8": images_u8,
        "arcface": arcface,
        "shas": out_shas,
        "resolution": args.resolution,
        "format_version": 1,
    }, args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
