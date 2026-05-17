"""Apparent-age analysis across the importer corpus via insightface buffalo_l.

The chibi style LoRA imposes neotenic (baby-schema) proportions; this
quantifies the resulting apparent-age shift. For every render in every
style x batch, plus every anchor portrait, this runs buffalo_l's genderage
head and records the predicted age.

Reported:
  - Detection rate per style — does genderage even fire on a toy face.
  - Anchor calibration — buffalo age vs the manifest's intended age_bin.
    If buffalo is accurate on the (in-distribution) anchors, its chibi
    readings are at least directionally trustworthy.
  - Age drop — per id_idx, photoreal mean age minus chibi mean age, i.e.
    how many years the chibi LoRA strips off.
  - Join with the neoteny metrics from chibi_scores.csv (lower_face_ratio)
    to see whether the chin geometry tracks apparent age.

Anchor lookup: refs/ used data/importer/identities/; refs_v2 and refs_v3
used data/importer/identities_flux/.

Usage:
    uv run python scripts/analyze_age.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2  # type: ignore[import-not-found]
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis

ROOT = Path(__file__).resolve().parents[1]
IMPORTER = ROOT / "data" / "importer"
STYLES = ["photoreal", "chibi", "furry", "chibi_furry"]

# (batch name, anchor portrait dir).
BATCHES = [
    ("refs", IMPORTER / "identities"),
    ("refs_v2", IMPORTER / "identities_flux"),
    ("refs_v3", IMPORTER / "identities_flux"),
]

ID_RE = re.compile(r"id_(\d+)")
AGE_MID = {"20-29": 25.0, "30-39": 35.0, "40-49": 45.0, "50-59": 55.0}


def biggest_face(app: FaceAnalysis, img_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    faces = app.get(img)
    if not faces:
        return None
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return faces[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "exp_output/chibi_score/age_scores.csv")
    ap.add_argument("--neoteny-csv", type=Path,
                    default=ROOT / "exp_output/chibi_score/chibi_scores.csv")
    args = ap.parse_args()

    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "genderage"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # manifests: id_idx -> intended age_bin
    intended: dict[Path, dict[int, str]] = {}
    for _, anchor_dir in BATCHES:
        if anchor_dir not in intended:
            mf = pd.read_csv(anchor_dir / "manifest.csv")
            intended[anchor_dir] = dict(zip(mf["id_idx"], mf["age_bin"]))

    rows: list[dict] = []
    anchor_done: set[Path] = set()

    for batch, anchor_dir in BATCHES:
        age_bin = intended[anchor_dir]

        # anchors — score each pool once
        if anchor_dir not in anchor_done:
            anchor_done.add(anchor_dir)
            for idx, abin in age_bin.items():
                f = biggest_face(app, anchor_dir / f"id_{idx:02d}.png")
                rows.append({
                    "batch": anchor_dir.name, "style": "anchor", "id_idx": idx,
                    "render": f"id_{idx:02d}.png", "age_bin": abin,
                    "age_mid": AGE_MID.get(abin, np.nan),
                    "detected": f is not None,
                    "buffalo_age": float(f.age) if f is not None else np.nan,
                    "buffalo_gender": ("M" if f is not None and f.sex == "M" else
                                       "F" if f is not None else ""),
                })

        # renders
        for style in STYLES:
            d = IMPORTER / batch / style
            pngs = sorted(d.glob("*.png"))
            if not pngs:
                continue
            print(f"[{batch}/{style}] {len(pngs)} renders")
            for p in pngs:
                m = ID_RE.search(p.name)
                if not m:
                    continue
                idx = int(m.group(1))
                f = biggest_face(app, p)
                abin = age_bin.get(idx, "")
                rows.append({
                    "batch": batch, "style": style, "id_idx": idx,
                    "render": p.name, "age_bin": abin,
                    "age_mid": AGE_MID.get(abin, np.nan),
                    "detected": f is not None,
                    "buffalo_age": float(f.age) if f is not None else np.nan,
                    "buffalo_gender": ("M" if f is not None and f.sex == "M" else
                                       "F" if f is not None else ""),
                })

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n=== {len(df)} faces scored → {args.out} ===\n")

    # --- detection rate + mean apparent age per style ---
    print("style x batch:  detection / mean buffalo age / mean intended age")
    for style in ["anchor"] + STYLES:
        s = df[df["style"] == style]
        if s.empty:
            continue
        for batch in s["batch"].unique():
            sb = s[s["batch"] == batch]
            det = sb[sb["detected"]]
            age = det["buffalo_age"].mean() if len(det) else np.nan
            intend = sb["age_mid"].mean()
            print(f"  {style:12s} {batch:14s} {sb['detected'].mean()*100:5.1f}%  "
                  f"age {age:5.1f}  (intended {intend:4.1f})")

    # --- anchor calibration ---
    anc = df[(df["style"] == "anchor") & df["detected"]]
    if len(anc):
        err = anc["buffalo_age"] - anc["age_mid"]
        print(f"\nanchor calibration: buffalo - intended  "
              f"mean {err.mean():+.1f}  MAE {err.abs().mean():.1f}  "
              f"(n={len(anc)})")

    # --- age drop: photoreal vs chibi, per id ---
    print("\nage drop (photoreal mean age - style mean age), per render-batch:")
    for batch in ["refs", "refs_v2", "refs_v3"]:
        b = df[(df["batch"] == batch) & df["detected"]]
        if b.empty:
            continue
        pr = b[b["style"] == "photoreal"].groupby("id_idx")["buffalo_age"].mean()
        line = f"  {batch:9s}"
        for style in ["chibi", "furry", "chibi_furry"]:
            st = b[b["style"] == style].groupby("id_idx")["buffalo_age"].mean()
            common = pr.index.intersection(st.index)
            if len(common):
                drop = (pr[common] - st[common]).mean()
                line += f"   {style} {drop:+5.1f}y"
        print(line)

    # --- join chibi apparent age with neoteny lower_face_ratio ---
    if args.neoteny_csv.exists():
        neo = pd.read_csv(args.neoteny_csv)
        neo = neo.rename(columns={"set": "batch"})
        ch = df[(df["style"] == "chibi") & df["detected"]]
        j = ch.merge(neo[["batch", "render", "lower_face_ratio", "eye_vpos"]],
                     on=["batch", "render"], how="inner")
        if len(j) > 2:
            r_chin = j["buffalo_age"].corr(j["lower_face_ratio"])
            r_eye = j["buffalo_age"].corr(j["eye_vpos"])
            print(f"\nchibi apparent age vs geometry (n={len(j)}):  "
                  f"corr(age, lower_face_ratio) {r_chin:+.3f}   "
                  f"corr(age, eye_vpos) {r_eye:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
