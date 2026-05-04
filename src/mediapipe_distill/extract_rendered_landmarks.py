"""Run InsightFace buffalo_l on rendered PNGs from compact_rendered.pt; save
aligned landmark_2d_106 + found mask in the same row order.

Output: same shape conventions as arc_distill/precompute_bboxes.py — landmarks
in pixel coords of the source image (we resize to 512² LANCZOS to match the
FFHQ extraction path before detection).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rendered", type=Path, required=True,
                   help="compact_rendered.pt produced by encode_rendered_to_latent")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--det-size", type=int, default=640)
    args = p.parse_args()

    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l",
                       allowed_modules=["detection", "landmark_2d_106"])
    app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))
    print("InsightFace detector + landmark_2d_106 ready")

    blob = torch.load(args.rendered, map_location="cpu", weights_only=False)
    img_paths = list(blob["img_paths"])
    n = len(img_paths)
    print(f"  N={n}")

    landmark_2d_106 = torch.zeros((n, 106, 2), dtype=torch.float32)
    det_score = torch.zeros(n, dtype=torch.float32)
    found = torch.zeros(n, dtype=torch.bool)
    n_redetect_miss = 0
    t0 = time.time()

    for i, rel in enumerate(img_paths):
        try:
            im = Image.open(rel).convert("RGB")
            if im.size != (512, 512):
                im = im.resize((512, 512), Image.LANCZOS)
            rgb = np.asarray(im, dtype=np.uint8)
        except Exception:
            n_redetect_miss += 1
            continue
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        faces = app.get(bgr)
        if not faces:
            n_redetect_miss += 1
            continue
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        lm = getattr(f, "landmark_2d_106", None)
        if lm is not None:
            landmark_2d_106[i] = torch.from_numpy(lm.astype(np.float32))
            ds = getattr(f, "det_score", None)
            if ds is not None:
                det_score[i] = float(ds)
            found[i] = True
        else:
            n_redetect_miss += 1

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-3)
            eta = (n - (i + 1)) / max(rate, 1e-3)
            print(f"  {i+1}/{n} found={int(found.sum())} miss={n_redetect_miss} "
                  f"[{elapsed:.0f}s, {rate:.1f}/s, eta={eta:.0f}s]")

    elapsed = time.time() - t0
    n_found = int(found.sum())
    print(f"done: found {n_found}/{n} (miss={n_redetect_miss}) in {elapsed:.0f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "landmark_2d_106": landmark_2d_106,
        "det_score": det_score,
        "found": found,
        "img_paths": img_paths,
        "shas": list(blob["shas"]),
        "format_version": 1,
        "source_resolution": 512,
    }, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
