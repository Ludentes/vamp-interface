"""Stage 3 — run all three classifiers on every Stage 2 sample.

Input: output/demographic_pc/samples/{sample_id}.png (1785 images)
Output: output/demographic_pc/labels.parquet — one row per sample_id with
all classifier outputs plus the prompt-attribute ground truth.

Columns:
    sample_id, prompt_age, prompt_gender, prompt_ethnicity,
    mivolo_age, mivolo_gender, mivolo_gender_conf,
    fairface_detected, fairface_age_bin, fairface_gender, fairface_race,
    fairface_age_probs (9-vec), fairface_gender_probs (2-vec), fairface_race_probs (7-vec),
    insightface_detected, insightface_age, insightface_gender

Usage:
    uv run python -m src.demographic_pc.stage3_classify
    uv run python -m src.demographic_pc.stage3_classify --limit 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.demographic_pc.classifiers import (
    FairFaceClassifier,
    InsightFaceClassifier,
    MiVOLOClassifier,
    predict_all,
)
from src.demographic_pc.prompts import full_grid

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output" / "demographic_pc"
SAMPLES_DIR = OUT_DIR / "samples"
LABELS_PATH = OUT_DIR / "labels.parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = full_grid()
    if args.limit:
        rows = rows[: args.limit]
    print(f"[stage3] classifying {len(rows)} samples")

    print("[stage3] loading classifiers…")
    mv = MiVOLOClassifier()
    ff = FairFaceClassifier()
    ins = InsightFaceClassifier()

    records: list[dict] = []
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        img_path = SAMPLES_DIR / f"{r.sample_id}.png"
        if not img_path.exists():
            print(f"[stage3] MISSING {img_path}; skipping")
            continue
        bgr = cv2.imread(str(img_path))
        rec = predict_all(bgr, mv, ff, ins)
        records.append({
            "sample_id": r.sample_id,
            "prompt_age": r.age,
            "prompt_gender": r.gender,
            "prompt_ethnicity": r.ethnicity,
            "mivolo_age": rec.mivolo_age,
            "mivolo_gender": rec.mivolo_gender,
            "mivolo_gender_conf": rec.mivolo_gender_conf,
            "fairface_detected": rec.fairface_detected,
            "fairface_age_bin": rec.fairface_age_bin,
            "fairface_gender": rec.fairface_gender,
            "fairface_race": rec.fairface_race,
            "fairface_age_probs": rec.fairface_age_probs,
            "fairface_gender_probs": rec.fairface_gender_probs,
            "fairface_race_probs": rec.fairface_race_probs,
            "insightface_detected": rec.insightface_detected,
            "insightface_age": rec.insightface_age,
            "insightface_gender": rec.insightface_gender,
        })
        if i % 50 == 0 or i == len(rows):
            dt = time.time() - t0
            rate = i / dt
            eta = (len(rows) - i) / rate if rate > 0 else 0
            print(f"  [{i:4d}/{len(rows)}] rate={rate:.2f}/s  dt={dt:.0f}s  eta={eta/60:.1f}min")

    df = pd.DataFrame.from_records(records)
    df.to_parquet(LABELS_PATH, index=False)
    print(f"[stage3] wrote {LABELS_PATH}  rows={len(df)}")

    # Short coverage + inter-classifier report
    print("\n--- coverage ---")
    print(f"  fairface_detected:    {df.fairface_detected.sum()}/{len(df)}")
    print(f"  insightface_detected: {df.insightface_detected.sum()}/{len(df)}")

    both = df[df.fairface_detected & df.insightface_detected]
    if len(both):
        agree_ff_mv = (both.fairface_gender == both.mivolo_gender).mean()
        agree_ff_ins = (both.fairface_gender == both.insightface_gender).mean()
        agree_mv_ins = (both.mivolo_gender == both.insightface_gender).mean()
        print("\n--- gender pairwise agreement (both-detected subset) ---")
        print(f"  FairFace↔MiVOLO:     {agree_ff_mv:.3f}")
        print(f"  FairFace↔InsightFace:{agree_ff_ins:.3f}")
        print(f"  MiVOLO↔InsightFace:  {agree_mv_ins:.3f}")

    summary = {
        "n": len(df),
        "fairface_detected": int(df.fairface_detected.sum()),
        "insightface_detected": int(df.insightface_detected.sum()),
    }
    with open(OUT_DIR / "labels_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
