"""One-pass MiVOLO + FairFace + InsightFace(+ArcFace emb) scorer.

Iterates every row of `sample_index.parquet`, scores each img_path, and
writes a resumable sidecar parquet `classifier_scores.parquet` keyed by
img_path. Merge into main index at the end with `--merge`.

Columns written:
    mv_age, mv_gender, mv_gender_conf
    ff_age_bin, ff_gender, ff_race, ff_detected
    ff_age_probs, ff_gender_probs, ff_race_probs            (list[float])
    ins_age, ins_gender, ins_detected
    ins_embedding                                           (list[float], 512-d)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.demographic_pc.classifiers import (
    FairFaceClassifier, InsightFaceClassifier, MiVOLOClassifier,
)

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "models/blendshape_nmf/sample_index.parquet"
SIDECAR = ROOT / "output/demographic_pc/classifier_scores.parquet"
CHECKPOINT_EVERY = 200

EMPTY_ROW = {
    "mv_age": np.nan, "mv_gender": None, "mv_gender_conf": np.nan,
    "ff_age_bin": None, "ff_gender": None, "ff_race": None, "ff_detected": False,
    "ff_age_probs": None, "ff_gender_probs": None, "ff_race_probs": None,
    "ins_age": np.nan, "ins_gender": None, "ins_detected": False,
    "ins_embedding": None,
}


def score_one(bgr: np.ndarray, mv, ff, ins) -> dict:
    row = dict(EMPTY_ROW)
    try:
        m = mv.predict(bgr)
        row["mv_age"] = m["age"]
        row["mv_gender"] = m["gender"]
        row["mv_gender_conf"] = m["gender_conf"]
    except Exception:
        pass
    try:
        f = ff.predict(bgr)
        row["ff_detected"] = f["detected"]
        if f["detected"]:
            row["ff_age_bin"] = f["age_bin"]
            row["ff_gender"] = f["gender"]
            row["ff_race"] = f["race"]
            row["ff_age_probs"] = f["age_probs"].tolist()
            row["ff_gender_probs"] = f["gender_probs"].tolist()
            row["ff_race_probs"] = f["race_probs"].tolist()
    except Exception:
        pass
    try:
        i = ins.predict(bgr)
        row["ins_detected"] = i["detected"]
        if i["detected"]:
            row["ins_age"] = i["age"]
            row["ins_gender"] = i["gender"]
            if i["embedding"] is not None:
                row["ins_embedding"] = i["embedding"].tolist()
    except Exception:
        pass
    return row


def save(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    SIDECAR.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SIDECAR, index=False, compression="zstd")


def run_score() -> None:
    idx = pd.read_parquet(INDEX)
    todo = idx["img_path"].tolist()
    done = set()
    existing = []
    if SIDECAR.exists():
        prior = pd.read_parquet(SIDECAR)
        done = set(prior["img_path"])
        existing = prior.to_dict("records")
        print(f"[resume] {len(done)} already scored, {len(todo) - len(done)} remaining")

    print("[load] MiVOLO…"); mv = MiVOLOClassifier()
    print("[load] FairFace…"); ff = FairFaceClassifier()
    print("[load] InsightFace (with ArcFace r50)…"); ins = InsightFaceClassifier(with_embedding=True)

    rows = list(existing)
    t0 = time.time()
    n_new = 0
    for rel in todo:
        if rel in done:
            continue
        abs_path = ROOT / rel
        bgr = cv2.imread(str(abs_path))
        if bgr is None:
            row = dict(EMPTY_ROW)
        else:
            row = score_one(bgr, mv, ff, ins)
        row["img_path"] = rel
        rows.append(row)
        n_new += 1
        if n_new % 50 == 0:
            dt = time.time() - t0
            rate = n_new / dt
            remain = (len(todo) - len(done) - n_new) / rate if rate > 0 else 0
            print(f"  [{n_new}/{len(todo) - len(done)}] {rate:.2f} img/s  eta {remain:.0f}s")
        if n_new % CHECKPOINT_EVERY == 0:
            save(rows)
    save(rows)
    print(f"[save] → {SIDECAR}  rows={len(rows)}")


def merge() -> None:
    idx = pd.read_parquet(INDEX)
    sc = pd.read_parquet(SIDECAR)
    new_cols = [c for c in sc.columns if c != "img_path"]
    for c in new_cols:
        if c in idx.columns:
            idx = idx.drop(columns=c)
    merged = idx.merge(sc, on="img_path", how="left")
    n_mv = merged["mv_age"].notna().sum()
    n_ff = merged["ff_detected"].sum()
    n_ins = merged["ins_detected"].sum()
    print(f"[merge] rows={len(merged)}  mv_age={n_mv}  ff_det={n_ff}  ins_det={n_ins}")
    merged.to_parquet(INDEX, index=False, compression="zstd")
    print(f"[save] → {INDEX}  ({INDEX.stat().st_size / 1024 / 1024:.2f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.score:
        run_score()
    if args.merge:
        merge()
    if not (args.score or args.merge):
        ap.print_help()


if __name__ == "__main__":
    main()
