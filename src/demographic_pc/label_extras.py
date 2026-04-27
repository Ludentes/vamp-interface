"""Label Stage 2 renders with MediaPipe blendshapes + CLIP zero-shot glasses.

Reads PNGs from output/demographic_pc/samples/{sample_id}.png and writes
output/demographic_pc/labels_extras.parquet with columns:
    sample_id, smile, brow_raise, eye_open, jaw_open, glasses_prob,
    blendshapes_detected.

Usage:
    uv run python -m src.demographic_pc.label_extras
    uv run python -m src.demographic_pc.label_extras --limit 10
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import pandas as pd

from src.demographic_pc.extras_classifiers import CLIPZeroShotGlasses, MediaPipeBlendshapes

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "output" / "demographic_pc" / "samples"
OUT = ROOT / "output" / "demographic_pc" / "labels_extras.parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--samples-dir", type=Path, default=SAMPLES)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    paths = sorted(args.samples_dir.glob("*.png"))
    if args.limit:
        paths = paths[: args.limit]
    print(f"[label_extras] {len(paths)} samples")

    blend = MediaPipeBlendshapes()
    glasses = CLIPZeroShotGlasses()

    rows: list[dict] = []
    t0 = time.time()
    for i, p in enumerate(paths):
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        b = blend.predict(bgr)
        g = glasses.predict(bgr)
        rows.append({
            "sample_id": p.stem,
            "blendshapes_detected": bool(b["detected"]),
            "smile": b["smile"],
            "brow_raise": b["brow_raise"],
            "eye_open": b["eye_open"],
            "jaw_open": b["jaw_open"],
            "glasses_prob": g["glasses_prob"],
        })
        if (i + 1) % 50 == 0:
            dt = time.time() - t0
            rate = (i + 1) / dt
            eta = (len(paths) - i - 1) / rate / 60
            print(f"  [{i+1}/{len(paths)}]  {rate:.1f}/s  eta {eta:.1f}min")

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    det = df["blendshapes_detected"].mean()
    print(f"[label_extras] wrote {args.out}  ({len(df)} rows, det={det:.2%})")
    print(df[["smile", "brow_raise", "eye_open", "jaw_open", "glasses_prob"]].describe().round(3))


if __name__ == "__main__":
    main()
