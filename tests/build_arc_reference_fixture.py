"""Build tests/fixtures/arc_reference.npz from FFHQ images.

Pre-flight #1 of the arc_latent distillation plan. Runs the canonical
InsightFaceClassifier pipeline (buffalo_l: SCRFD detect → 5-pt similarity
align → 112x112 BGR crop → (x-127.5)/127.5 → R50 → L2-normed 512-d) on
10 FFHQ images and stores the embeddings as a fixture, so the future
distillation corpus builder can round-trip-check itself.

Usage:
    uv run python tests/build_arc_reference_fixture.py \
        --parquet /tmp/ffhq_sample/train-00000-of-00190.parquet \
        --out tests/fixtures/arc_reference.npz \
        --n 10
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from demographic_pc.classifiers import InsightFaceClassifier


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=10)
    args = p.parse_args()

    table = pq.read_table(args.parquet, columns=["image"])
    rows = table.to_pylist()[: args.n]
    print(f"loaded {len(rows)} rows from {args.parquet.name}")

    clf = InsightFaceClassifier(ctx_id=0, with_embedding=True)

    image_sha256: list[str] = []
    embeddings: list[np.ndarray] = []
    detected: list[bool] = []
    bboxes: list[tuple[int, int, int, int] | None] = []

    for i, row in enumerate(rows):
        png_bytes: bytes = row["image"]["bytes"]
        sha = hashlib.sha256(png_bytes).hexdigest()
        rgb = np.array(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        out = clf.predict(bgr)
        emb = out["embedding"]
        image_sha256.append(sha)
        detected.append(bool(out["detected"]))
        bboxes.append(out["bbox"])
        if emb is None:
            embeddings.append(np.zeros(512, dtype=np.float32))
            print(f"  [{i:02d}] sha={sha[:12]} NO_FACE")
        else:
            embeddings.append(emb)
            norm = float(np.linalg.norm(emb))
            print(f"  [{i:02d}] sha={sha[:12]} bbox={out['bbox']} ||emb||={norm:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        image_sha256=np.array(image_sha256),
        embeddings=np.stack(embeddings),
        detected=np.array(detected, dtype=bool),
        bboxes=np.array([b if b else (-1, -1, -1, -1) for b in bboxes], dtype=np.int32),
        teacher="insightface_buffalo_l_w600k_r50",
        pipeline="SCRFD+norm_crop2(112,bgr)+(x-127.5)/127.5+R50+L2",
    )
    print(f"saved {args.out} ({sum(detected)}/{len(rows)} detected)")


if __name__ == "__main__":
    main()
