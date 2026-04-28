"""Pre-pack FFHQ rows into one compact file for arc_pixel training.

Two modes:

  default (no --align):
    decode + resize detected rows to --resolution × --resolution uint8.
    Writes (N, 3, R, R) uint8.

  --align (canonical ArcFace input):
    re-run InsightFace SCRFD detection on each image to get 5 landmarks,
    then `face_align.norm_crop(bgr, kps, 112)` to produce the same aligned
    112² BGR crop ArcFace consumes internally. Convert BGR→RGB. Writes
    (N, 3, 112, 112) uint8. Drops rows where current detection misses
    (may diverge slightly from the original `detected` mask).

In either mode the output also carries the matching arcface_fp32 teacher
target per kept row.
"""
from __future__ import annotations

import argparse
import io
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

_SHARD_RE = re.compile(r"train-(\d{5})-of-\d{5}\.parquet$")


def _build_aligner(ctx_id: int = 0):
    """Return (FaceAnalysis, norm_crop_fn). detection-only buffalo_l."""
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align

    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app, face_align.norm_crop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-dir", type=Path, required=True)
    ap.add_argument("--encoded-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resolution", type=int, default=224,
                    help="Output side length in default (no-align) mode.")
    ap.add_argument("--align", action="store_true",
                    help="Produce 112² ArcFace-aligned crops via InsightFace "
                         "SCRFD landmarks + norm_crop.")
    ap.add_argument("--limit-shards", type=int, default=0,
                    help="Process only the first N shards (0=all). Smoke flag.")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    shards = sorted(args.shards_dir.glob("train-*-of-*.parquet"))
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    print(f"scanning {len(shards)} shards under {args.shards_dir}")
    print(f"mode: {'aligned 112x112' if args.align else f'resize {args.resolution}x{args.resolution}'}")

    aligner = None
    norm_crop = None
    if args.align:
        aligner, norm_crop = _build_aligner()
        print("InsightFace detector ready")

    out_imgs: list[np.ndarray] = []
    out_arc: list[torch.Tensor] = []
    out_shas: list[str] = []
    miss_realign = 0

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
            rgb = np.asarray(
                Image.open(io.BytesIO(img_col[i]["bytes"])).convert("RGB"),
                dtype=np.uint8,
            )

            if args.align:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                faces = aligner.get(bgr)
                if not faces:
                    miss_realign += 1
                    continue
                # largest face by bbox area
                f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                aligned_bgr = norm_crop(bgr, f.kps, image_size=112)
                aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
                arr = aligned_rgb.transpose(2, 0, 1)  # CHW
            else:
                resized = Image.fromarray(rgb).resize(
                    (args.resolution, args.resolution), Image.Resampling.BILINEAR
                )
                arr = np.asarray(resized, dtype=np.uint8).transpose(2, 0, 1)

            out_imgs.append(arr)
            out_arc.append(arc[i].clone())
            out_shas.append(sha)
            kept += 1

        del table, img_col
        elapsed = time.time() - t_start
        print(f"  [{s_idx+1}/{len(shards)}] {p.name} kept={kept} "
              f"total={len(out_imgs)} miss_realign={miss_realign} "
              f"elapsed={elapsed:.1f}s")

    print(f"packing {len(out_imgs)} rows into tensors...")
    images_u8 = torch.from_numpy(np.stack(out_imgs))
    arcface = torch.stack(out_arc).to(torch.float32)
    print(f"images_u8: {images_u8.shape} {images_u8.dtype} "
          f"({images_u8.numel() / 1e9:.2f} GB)")
    print(f"arcface:   {arcface.shape}")
    if args.align:
        print(f"miss_realign (was detected at encode time but not now): {miss_realign}")

    torch.save({
        "images_u8": images_u8,
        "arcface": arcface,
        "shas": out_shas,
        "resolution": 112 if args.align else args.resolution,
        "aligned": args.align,
        "format_version": 1,
    }, args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
