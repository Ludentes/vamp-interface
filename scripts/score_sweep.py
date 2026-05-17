"""Score the chibi grown-up sweep: which treatment/schedule/strength reads adult.

For every sweep render this measures buffalo_l apparent age + the MediaPipe
FaceMesh neoteny geometry, joins to manifest_sweep.parquet, and reports:

  - marginal mean apparent age by treatment / schedule / strength
  - the single best cell (oldest apparent age, MediaPipe-detected)
  - feature distance to the id_14 reference render -- the render the user
    hand-picked as the one chibi that does not read as a child. Distance is
    z-scored Euclidean over the age-driving features the corpus analysis
    surfaced (eye_vpos, eye_open, eye_size, iris_ratio + buffalo_age).

Usage:
    uv run python scripts/score_sweep.py
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
SWEEP = IMPORTER / "refs_sweep" / "chibi"
MANIFEST = IMPORTER / "manifest_sweep.parquet"
REFERENCE = IMPORTER / "refs" / "chibi" / "id_14_seed28980065.png"
OUT = ROOT / "exp_output" / "chibi_score" / "sweep_features.parquet"

# Features the corpus correlation surfaced as the apparent-age carriers.
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
        print(f"reference id_14: age={ref.get('ref_buffalo_age')} "
              f"eye_vpos={ref.get('ref_eye_vpos')}")
        for _, m in man.iterrows():
            p = SWEEP / m["render"]
            if not p.exists():
                continue
            rows.append({**m.to_dict(), **score_png(lm, app, p)})

    df = pd.DataFrame(rows)
    det = df[df["mp_detected"] & df["arc_detected"]].copy()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\n=== {len(df)} cells, {len(det)} fully detected → {OUT} ===")

    # z-scored distance to the reference vector
    refvec = np.array([ref[f"ref_{f}"] for f in DIST_FEATS], dtype=float)
    mat = det[DIST_FEATS].to_numpy(dtype=float)
    sd = mat.std(0) + 1e-9
    det["ref_dist"] = np.linalg.norm((mat - refvec) / sd, axis=1)

    print("\nmarginal mean apparent age (n detected):")
    for col in ["treatment", "schedule", "strength"]:
        print(f"\n  by {col}:")
        g = det.groupby(col).agg(age=("buffalo_age", "mean"),
                                 dist=("ref_dist", "mean"),
                                 n=("buffalo_age", "size"))
        for k, r in g.iterrows():
            print(f"    {str(k):14s} age {r['age']:5.1f}  "
                  f"ref_dist {r['dist']:5.2f}  n={int(r['n'])}")

    print("\ntop 8 cells by apparent age:")
    for _, r in det.nlargest(8, "buffalo_age").iterrows():
        print(f"  {r['render']:54s} age {r['buffalo_age']:5.1f}  "
              f"ref_dist {r['ref_dist']:5.2f}")
    print("\ntop 8 cells closest to id_14 reference vector:")
    for _, r in det.nsmallest(8, "ref_dist").iterrows():
        print(f"  {r['render']:54s} age {r['buffalo_age']:5.1f}  "
              f"ref_dist {r['ref_dist']:5.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
