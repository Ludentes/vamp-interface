"""Audit ArcFace face-detection rate across three input variants.

Goal: decide whether `cfm.precompute` should run ArcFace on the 512² face crop
(current) or on the original full RGB (proposal).

Variants:
  A. crop 512²  + det_size=(512, 512)  — what precompute.py does now
  B. full RGB   + det_size=(640, 640)  — insightface default
  C. crop 512²  + det_size=(640, 640)  — the failing variant the implementer hit

Sample: 200 pose-detected shas from the head of pose_cache.parquet.
Reports per-variant id_ok rate. No writes outside stdout.

Run: uv run python scripts/cfm_arcface_detection_audit.py
"""
from __future__ import annotations

import glob
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arkit_controlnet.build_ffhq_index import SHARD_GLOB  # noqa: E402
from arkit_controlnet.cfm.precompute import face_crop_resize  # noqa: E402

N = 200


def main() -> None:
    from insightface.app import FaceAnalysis

    pc = pd.read_parquet(
        "output/flame_pose_cache/pose_cache.parquet",
        columns=["image_sha256", "bbox_cx", "bbox_cy", "bbox_w", "bbox_h",
                 "pose_detected"],
    )
    pc = pc[pc.pose_detected].head(N).reset_index(drop=True)
    idx = pd.read_parquet("output/ffhq_index/ffhq_sha_index.parquet")
    rows = pc.merge(idx, on="image_sha256", how="inner")
    print(f"sampled {len(rows)} pose-detected shas")

    shards = sorted(glob.glob(SHARD_GLOB))

    def make_app(det):
        app = FaceAnalysis(name="buffalo_l",
                           providers=["CUDAExecutionProvider"])
        app.prepare(ctx_id=0, det_size=det)
        return app

    app_512 = make_app((512, 512))
    app_640 = make_app((640, 640))

    n_A = n_B = n_C = 0
    by_shard = rows.groupby("shard_idx")
    for shard_idx, group in by_shard:
        shard_df = pd.read_parquet(shards[shard_idx])
        for _, r in group.iterrows():
            cell = shard_df["image"].iloc[int(r["row_idx"])]
            rgb = np.asarray(
                Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))
            crop = face_crop_resize(
                rgb, float(r["bbox_cx"]), float(r["bbox_cy"]),
                float(r["bbox_w"]), float(r["bbox_h"]), out_size=512)
            # ArcFace expects BGR uint8.
            crop_bgr = crop[:, :, ::-1]
            full_bgr = rgb[:, :, ::-1]
            if len(app_512.get(crop_bgr)) > 0:
                n_A += 1
            if len(app_640.get(full_bgr)) > 0:
                n_B += 1
            if len(app_640.get(crop_bgr)) > 0:
                n_C += 1

    total = len(rows)
    print()
    print(f"  A (crop 512, det 512): {n_A}/{total}  = {n_A/total:.1%}")
    print(f"  B (full RGB, det 640): {n_B}/{total}  = {n_B/total:.1%}")
    print(f"  C (crop 512, det 640): {n_C}/{total}  = {n_C/total:.1%}")


if __name__ == "__main__":
    main()
