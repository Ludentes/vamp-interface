"""Run FairFace race classifier on Stage 2b black-* samples.

Produces labels_black.parquet with fairface_race for the 180 black renders,
so build_black_edits can combine them with Stage 2's labels.

GPU-optional: pass --cpu to keep off the GPU while Stage 4.5 renders.

Usage:
    uv run python -m src.demographic_pc.label_black --cpu
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import pandas as pd

from src.demographic_pc.classifiers import FairFaceClassifier

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "output" / "demographic_pc" / "samples_2b"
OUT = ROOT / "output" / "demographic_pc" / "labels_black.parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    paths = sorted(SAMPLES.glob("black-*.png"))
    if args.limit:
        paths = paths[: args.limit]
    print(f"[label_black] {len(paths)} samples  device={'cpu' if args.cpu else 'auto'}")

    ff = FairFaceClassifier(device="cpu" if args.cpu else None)

    rows: list[dict] = []
    t0 = time.time()
    for i, p in enumerate(paths):
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        r = ff.predict(bgr)
        rows.append({
            "sample_id": p.stem,
            "fairface_detected": r["detected"],
            "fairface_age_bin": r.get("age_bin"),
            "fairface_gender": r.get("gender"),
            "fairface_race": r.get("race"),
            "fairface_race_probs": r.get("race_probs"),
        })
        if (i + 1) % 20 == 0:
            dt = time.time() - t0
            rate = (i + 1) / dt
            print(f"  [{i+1}/{len(paths)}]  {rate:.2f}/s  eta {(len(paths)-i-1)/rate/60:.1f}min")

    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)
    det = df["fairface_detected"].mean()
    print(f"[label_black] wrote {OUT}  ({len(df)} rows, det={det:.2%})")
    print("\nrace distribution:")
    print(df["fairface_race"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
