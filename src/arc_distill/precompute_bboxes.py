"""Re-run InsightFace buffalo_l on each 512² FFHQ image, save all face attrs.

For each SHA in compact.pt: load image bytes from FFHQ parquet, resize to 512²
LANCZOS (matching encode_full_to_latent's resize), run InsightFace, take largest
face by bbox area. Save:
  - bboxes (N, 4) in latent (16, 64, 64) coords [0, 64]
  - bboxes_512 (N, 4) in 512² pixel coords [0, 512]
  - kps_5 (N, 5, 2) in 512² pixel coords (5 facial landmarks: eyes, nose, mouth)
  - det_score (N,) detection confidence
  - landmark_2d_106 (N, 106, 2) in 512² pixel coords (fine landmarks)
  - pose (N, 3) [pitch, yaw, roll] degrees
  - age (N,) estimated age in years
  - gender (N,) 0=female, 1=male
  - found (N,) bool — True if detection succeeded
  - n_faces (N,) int — number of faces detected (≥2 may indicate group photos)

Rows where re-detection misses retain a centered fallback bbox (12, 12, 52, 52)
in latent coords; other fields are zero. found=False marks them.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--compact", type=Path, required=True)
    p.add_argument("--ffhq-parquet-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    from insightface.app import FaceAnalysis
    # Full buffalo_l pipeline minus recognition (already have arcface in compact).
    app = FaceAnalysis(name="buffalo_l",
                       allowed_modules=["detection", "landmark_2d_106", "genderage"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("InsightFace detector + landmark_2d_106 + genderage ready")

    blob = torch.load(args.compact, map_location="cpu", weights_only=False)
    target_shas: list[str] = list(blob["shas"])
    n = len(target_shas)
    sha_to_idx = {s: i for i, s in enumerate(target_shas)}
    print(f"  N={n}")

    bboxes_lat = torch.zeros((n, 4), dtype=torch.float32)
    bboxes_512 = torch.zeros((n, 4), dtype=torch.float32)
    kps_5 = torch.zeros((n, 5, 2), dtype=torch.float32)
    det_score = torch.zeros(n, dtype=torch.float32)
    landmark_2d_106 = torch.zeros((n, 106, 2), dtype=torch.float32)
    pose = torch.zeros((n, 3), dtype=torch.float32)
    age = torch.zeros(n, dtype=torch.float32)
    gender = torch.full((n,), -1, dtype=torch.int8)
    n_faces_per_row = torch.zeros(n, dtype=torch.int16)
    found = torch.zeros(n, dtype=torch.bool)
    centered_fallback = torch.tensor([12.0, 12.0, 52.0, 52.0], dtype=torch.float32)
    bboxes_lat[:] = centered_fallback

    shards = sorted(args.ffhq_parquet_dir.glob("train-*.parquet"))
    print(f"shards={len(shards)}")
    n_seen, n_redetect_miss = 0, 0
    t0 = time.time()

    for s_idx, s_path in enumerate(shards):
        if found.all():
            break
        table = pq.read_table(s_path, columns=["image"])
        for row in table.column("image").to_pylist():
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
                if im.size != (512, 512):
                    im = im.resize((512, 512), Image.LANCZOS)
                rgb = np.asarray(im, dtype=np.uint8)
            except Exception:
                continue
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            faces = app.get(bgr)
            n_faces_per_row[idx] = len(faces)
            if not faces:
                n_redetect_miss += 1
                found[idx] = True  # mark visited; bbox stays at centered fallback
                continue
            f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            bb = f.bbox.astype(np.float32)
            bboxes_512[idx] = torch.from_numpy(bb)
            x1, y1, x2, y2 = bb / 8.0
            bboxes_lat[idx] = torch.tensor([
                float(np.clip(x1, 0, 64)), float(np.clip(y1, 0, 64)),
                float(np.clip(x2, 0, 64)), float(np.clip(y2, 0, 64)),
            ])
            kps = getattr(f, "kps", None)
            if kps is not None:
                kps_5[idx] = torch.from_numpy(kps.astype(np.float32))
            ds = getattr(f, "det_score", None)
            if ds is not None:
                det_score[idx] = float(ds)
            lm = getattr(f, "landmark_2d_106", None)
            if lm is not None:
                landmark_2d_106[idx] = torch.from_numpy(lm.astype(np.float32))
            ps = getattr(f, "pose", None)
            if ps is not None:
                pose[idx] = torch.from_numpy(np.asarray(ps, dtype=np.float32))
            ag = getattr(f, "age", None)
            if ag is not None:
                age[idx] = float(ag)
            gd = getattr(f, "gender", None)
            if gd is not None:
                gender[idx] = int(gd)
            found[idx] = True

        elapsed = time.time() - t0
        n_found = int(found.sum())
        print(f"  shard {s_idx+1}/{len(shards)}: seen={n_seen} found={n_found}/{n} "
              f"redetect_miss={n_redetect_miss} ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    n_found = int(found.sum())
    print(f"done: found {n_found}/{n} (redetect_miss={n_redetect_miss}) in {elapsed:.0f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "bboxes_latent": bboxes_lat,
        "bboxes_512": bboxes_512,
        "kps_5": kps_5,
        "det_score": det_score,
        "landmark_2d_106": landmark_2d_106,
        "pose": pose,
        "age": age,
        "gender": gender,
        "n_faces_per_row": n_faces_per_row,
        "found": found,
        "shas": target_shas,
        "redetect_miss": int(n_redetect_miss),
        "format_version": 1,
        "source_resolution": 512,
        "latent_resolution": 64,
    }, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
