"""Similarity-warp (16, 64, 64) full-image latents → (16, 14, 14) aligned crops.

Mirrors insightface's `face_align.norm_crop` which the teacher uses, but
operates in latent space. For each row:
  - src kps: kps_5 in 512 pixel coords (from face_attrs.pt)
  - dst kps: insightface canonical 5-point destination in 112 pixel coords
  - Both divided by 8 → 14² latent destination
  - Solve similarity transform via cv2.estimateAffinePartial2D
  - Apply cv2.warpAffine to the (16, 64, 64) latent, channels-in-batches-of-4

Output: compact_aligned.pt with (N, 16, 14, 14) bf16 latents + arcface targets
+ found mask. Drops re-detection-miss rows (no valid kps).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch


# Insightface canonical 5-keypoint destination in 112² pixel space.
# (left_eye, right_eye, nose, left_mouth, right_mouth)
ARCFACE_DST_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--compact", type=Path, required=True,
                   help="compact.pt with (N, 16, 64, 64) latents + arcface")
    p.add_argument("--face-attrs", type=Path, required=True,
                   help="face_attrs.pt with kps_5 in 512 pixel coords")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--latent-size", type=int, default=64)
    p.add_argument("--dst-size", type=int, default=14)
    args = p.parse_args()

    print(f"loading compact: {args.compact}")
    blob = torch.load(args.compact, map_location="cpu", weights_only=False)
    latents = blob["latents"]  # (N, 16, 64, 64) bf16
    arcface = blob["arcface"]  # (N, 512) fp32
    shas = list(blob["shas"])
    n = len(shas)
    print(f"  N={n}  latents={tuple(latents.shape)}")

    print(f"loading face_attrs: {args.face_attrs}")
    attrs = torch.load(args.face_attrs, map_location="cpu", weights_only=False)
    if list(attrs["shas"]) != shas:
        raise ValueError("compact + face_attrs SHA order mismatch")
    kps_512 = attrs["kps_5"].numpy()  # (N, 5, 2) in 512² pixel coords
    bboxes_512 = attrs["bboxes_512"].numpy()
    valid = (bboxes_512[:, 2] > bboxes_512[:, 0]) & (bboxes_512[:, 3] > bboxes_512[:, 1])
    print(f"  valid (re-detected): {valid.sum()}/{n}")

    # Latent-space src/dst keypoints: pixel / 8.
    dst_latent = ARCFACE_DST_112 / 8.0  # (5, 2) in [0, 14]

    aligned_latents = torch.zeros((n, 16, args.dst_size, args.dst_size), dtype=torch.bfloat16)
    found = torch.zeros(n, dtype=torch.bool)
    n_warp_fail = 0

    for i in range(n):
        if not valid[i]:
            continue
        src_latent = kps_512[i] / 8.0  # (5, 2) in [0, 64]
        # cv2.estimateAffinePartial2D solves for similarity (rotation + uniform
        # scale + translation) — exactly what we want.
        M, _ = cv2.estimateAffinePartial2D(src_latent, dst_latent.astype(np.float32),
                                           method=cv2.LMEDS)
        if M is None:
            n_warp_fail += 1
            continue
        lat = latents[i].to(torch.float32).numpy()  # (16, 64, 64)
        warped = np.zeros((16, args.dst_size, args.dst_size), dtype=np.float32)
        # cv2.warpAffine handles up to 4 channels at a time. (H, W, C) layout.
        for c0 in range(0, 16, 4):
            block_chw = lat[c0:c0 + 4]                      # (4, 64, 64)
            block_hwc = block_chw.transpose(1, 2, 0)        # (64, 64, 4)
            warped_hwc = cv2.warpAffine(
                block_hwc, M, (args.dst_size, args.dst_size),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            if warped_hwc.ndim == 2:  # cv2 squeezes singleton last dim
                warped_hwc = warped_hwc[..., None]
            warped[c0:c0 + 4] = warped_hwc.transpose(2, 0, 1)
        aligned_latents[i] = torch.from_numpy(warped).to(torch.bfloat16)
        found[i] = True
        if (i + 1) % 2000 == 0:
            print(f"  warped {i+1}/{n} found={int(found.sum())}")

    print(f"done: warped {int(found.sum())}/{n} (warp_fail={n_warp_fail})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "latents": aligned_latents,
        "arcface": arcface,
        "shas": shas,
        "found": found,
        "resolution": args.dst_size,
        "source_resolution": args.latent_size,
        "format_version": 3,
        "alignment": "similarity_5pt_arcface",
    }, args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
