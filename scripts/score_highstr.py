"""Score the chibi high-strength + conditioner-ablation sweep.

The headline metric is ArcFace detection rate, NOT apparent age. The prompt
sweep showed creepiness is a partial-stylization artifact; ArcFace failing
to find a human face is the proxy for "committed to the figurine read /
out of the uncanny valley". Apparent age + neoteny geometry are reported
as secondary.

Reports, by strength and by ablation (full / no_cn / no_pulid):
  - ArcFace detection rate  (low = stylised past the human-face read)
  - MediaPipe detection rate (measurability floor for the corpus arm)
  - mean apparent age + ref_dist to the id_14 reference vector

Usage:
    uv run python scripts/score_highstr.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from build_render_features import geometry, make_landmarker, biggest  # noqa
import cv2  # type: ignore[import-not-found]
import mediapipe as mp
from insightface.app import FaceAnalysis

ROOT = Path(__file__).resolve().parents[1]
IMPORTER = ROOT / "data" / "importer"
SWEEP = IMPORTER / "refs_highstr" / "chibi"
MANIFEST = IMPORTER / "manifest_highstr.parquet"
REFERENCE = IMPORTER / "refs" / "chibi" / "id_14_seed28980065.png"
OUT = ROOT / "exp_output" / "chibi_score" / "highstr_features.parquet"

DIST_FEATS = ["buffalo_age", "eye_vpos", "eye_open", "eye_size", "iris_ratio"]


def score_png(lm, app, path: Path) -> dict:
    row: dict = {}
    img = cv2.imread(str(path))
    if img is None:
        return {"mp_detected": False, "arc_detected": False}
    f = biggest(app, img)
    row["arc_detected"] = f is not None
    if f is not None:
        row["buffalo_age"] = float(f.age)
        row["buffalo_gender"] = "M" if f.sex == "M" else "F"
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                      data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    res = lm.detect(mp_img)
    row["mp_detected"] = bool(res.face_landmarks)
    if res.face_landmarks:
        row.update(geometry(res.face_landmarks[0]))
        for c in res.face_blendshapes[0]:
            row[f"bs_{c.category_name}"] = float(c.score)
    return row


def main() -> int:
    man = pd.read_parquet(MANIFEST)
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "genderage"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    rows: list[dict] = []
    with make_landmarker() as lm:
        ref = {f"ref_{k}": v for k, v in score_png(lm, app, REFERENCE).items()}
        for _, m in man.iterrows():
            p = SWEEP / m["render"]
            if not p.exists():
                continue
            rows.append({**m.to_dict(), **score_png(lm, app, p)})

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\n=== {len(df)} cells → {OUT} ===")

    det = df[df["mp_detected"] & df["arc_detected"]].copy()
    refvec = np.array([ref[f"ref_{f}"] for f in DIST_FEATS], dtype=float)
    if len(det):
        mat = det[DIST_FEATS].to_numpy(dtype=float)
        sd = mat.std(0) + 1e-9
        det["ref_dist"] = np.linalg.norm((mat - refvec) / sd, axis=1)

    print("\ndetection rate (ArcFace low = past the human-face read = good):")
    for col in ["strength", "ablation"]:
        print(f"\n  by {col}:")
        g = df.groupby(col).agg(arc=("arc_detected", "mean"),
                                mp=("mp_detected", "mean"),
                                n=("arc_detected", "size"))
        for k, r in g.iterrows():
            d = det[det[col] == k]
            age = d["buffalo_age"].mean() if len(d) else float("nan")
            dist = d["ref_dist"].mean() if len(d) else float("nan")
            print(f"    {str(k):10s} arc {r['arc']*100:5.1f}%  mp {r['mp']*100:5.1f}%  "
                  f"age {age:5.1f}  ref_dist {dist:5.2f}  n={int(r['n'])}")

    print("\n  strength × ablation (ArcFace detect %):")
    piv = df.pivot_table(index="strength", columns="ablation",
                         values="arc_detected", aggfunc="mean") * 100
    print(piv.round(0).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
