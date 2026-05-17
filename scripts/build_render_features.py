"""Build a per-render feature parquet and rank what drives apparent age.

For every render in every style x batch (plus anchors) this records, in one
parquet:
  - buffalo_l genderage  -> apparent age, gender
  - MediaPipe FaceLandmarker -> 52 ARKit blendshapes (bs_* columns)
  - geometry from the FaceMesh landmarks:
      eye_spacing   inter-ocular distance / face width   (eyes far apart)
      eye_size      eye width / face width               (big eyes, lateral)
      eye_open      palpebral height / eye width         (big eyes, vertical)
      iris_ratio    iris diameter / eye width            (big iris)
      chin_ratio    chin-to-mouth / face height          (big chin)
      eye_vpos      eye line position top->chin          (low-set eyes)
      face_round    face width / face height             (round face)

It then correlates apparent age against every feature -- within chibi
renders and pooled across styles -- and ranks them, to answer which cue
(big eyes / eye spacing / big chin / ...) actually carries the age signal.

Supersedes score_chibi_detection.py + analyze_age.py (CSV precursors).

Usage:
    uv run python scripts/build_render_features.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2  # type: ignore[import-not-found]
import mediapipe as mp
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "mediapipe" / "face_landmarker.task"
IMPORTER = ROOT / "data" / "importer"
STYLES = ["photoreal", "chibi", "furry", "chibi_furry"]
BATCHES = [
    ("refs", IMPORTER / "identities"),
    ("refs_v2", IMPORTER / "identities_flux"),
    ("refs_v3", IMPORTER / "identities_flux"),
]
ID_RE = re.compile(r"id_(\d+)")
AGE_MID = {"20-29": 25.0, "30-39": 35.0, "40-49": 45.0, "50-59": 55.0}

# Canonical MediaPipe FaceMesh landmark indices.
R_EYE_OUTER, R_EYE_INNER = 33, 133
L_EYE_INNER, L_EYE_OUTER = 362, 263
R_EYE_TOP, R_EYE_BOT = 159, 145
L_EYE_TOP, L_EYE_BOT = 386, 374
FACE_TOP, FACE_CHIN, MOUTH_LOW = 10, 152, 17
FACE_R, FACE_L = 234, 454
R_IRIS, L_IRIS = (468, 472), (473, 477)  # center index, +rim


def make_landmarker():
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


def _p(lms, i: int) -> np.ndarray:
    return np.array([lms[i].x, lms[i].y], dtype=np.float64)


def geometry(lms) -> dict[str, float]:
    r_eye = (_p(lms, R_EYE_OUTER) + _p(lms, R_EYE_INNER)) / 2
    l_eye = (_p(lms, L_EYE_INNER) + _p(lms, L_EYE_OUTER)) / 2
    top, chin, mouth = _p(lms, FACE_TOP), _p(lms, FACE_CHIN), _p(lms, MOUTH_LOW)

    iod = float(np.linalg.norm(l_eye - r_eye))
    face_w = float(np.linalg.norm(_p(lms, FACE_L) - _p(lms, FACE_R)))
    face_h = float(abs(chin[1] - top[1]))
    eye_w = float((np.linalg.norm(_p(lms, R_EYE_INNER) - _p(lms, R_EYE_OUTER))
                   + np.linalg.norm(_p(lms, L_EYE_OUTER) - _p(lms, L_EYE_INNER))) / 2)
    eye_h = float((abs(_p(lms, R_EYE_TOP)[1] - _p(lms, R_EYE_BOT)[1])
                   + abs(_p(lms, L_EYE_TOP)[1] - _p(lms, L_EYE_BOT)[1])) / 2)
    eye_y = (r_eye[1] + l_eye[1]) / 2

    def _safe(n: float, d: float) -> float:
        return n / d if d > 1e-6 else float("nan")

    g = {
        "eye_spacing": _safe(iod, face_w),
        "eye_size": _safe(eye_w, face_w),
        "eye_open": _safe(eye_h, eye_w),
        "chin_ratio": _safe(chin[1] - mouth[1], face_h),
        "eye_vpos": _safe(eye_y - top[1], chin[1] - top[1]),
        "face_round": _safe(face_w, face_h),
        "iris_ratio": float("nan"),
    }
    if len(lms) >= 478:  # FaceLandmarker emits iris landmarks
        iris_d = []
        for c, last in [R_IRIS, L_IRIS]:
            center = _p(lms, c)
            rim = [np.linalg.norm(_p(lms, j) - center) for j in range(c + 1, last + 1)]
            iris_d.append(2 * float(np.mean(rim)))
        g["iris_ratio"] = _safe(float(np.mean(iris_d)), eye_w)
    return g


def biggest(app: FaceAnalysis, img):
    faces = app.get(img)
    if not faces:
        return None
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return faces[-1]


def score(lm, app, path: Path) -> dict:
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


def correlate(df: pd.DataFrame, label: str) -> None:
    feats = [c for c in df.columns
             if c.startswith("bs_") or c in {
                 "eye_spacing", "eye_size", "eye_open", "iris_ratio",
                 "chin_ratio", "eye_vpos", "face_round"}]
    sub = df[df["mp_detected"] & df["arc_detected"]]
    if len(sub) < 5:
        print(f"  [{label}] too few rows ({len(sub)})")
        return
    age = sub["buffalo_age"]
    scored = []
    for c in feats:
        col = sub[c]
        if col.std(skipna=True) < 1e-5 or col.notna().sum() < 5:
            continue
        r = float(age.corr(col, method="spearman"))
        if not np.isnan(r):
            scored.append((c, r))
    scored.sort(key=lambda t: -abs(t[1]))
    print(f"\n  [{label}] n={len(sub)} — age vs feature (Spearman, |r| desc):")
    for c, r in scored[:12]:
        print(f"    {c:24s} {r:+.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=ROOT / "exp_output/chibi_score/render_features.parquet")
    args = ap.parse_args()

    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "genderage"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    manifests: dict[Path, dict[int, str]] = {}
    for _, adir in BATCHES:
        if adir not in manifests:
            mf = pd.read_csv(adir / "manifest.csv")
            manifests[adir] = dict(zip(mf["id_idx"], mf["age_bin"]))

    rows: list[dict] = []
    anchors_done: set[Path] = set()
    with make_landmarker() as lm:
        for batch, adir in BATCHES:
            abins = manifests[adir]
            if adir not in anchors_done:
                anchors_done.add(adir)
                for idx, abin in abins.items():
                    base = {"batch": adir.name, "style": "anchor", "id_idx": idx,
                            "render": f"id_{idx:02d}.png", "age_bin": abin,
                            "age_mid": AGE_MID.get(abin, np.nan)}
                    rows.append({**base, **score(lm, app, adir / f"id_{idx:02d}.png")})
            for style in STYLES:
                pngs = sorted((IMPORTER / batch / style).glob("*.png"))
                if not pngs:
                    continue
                print(f"[{batch}/{style}] {len(pngs)}")
                for p in pngs:
                    m = ID_RE.search(p.name)
                    if not m:
                        continue
                    idx = int(m.group(1))
                    base = {"batch": batch, "style": style, "id_idx": idx,
                            "render": p.name, "age_bin": abins.get(idx, ""),
                            "age_mid": AGE_MID.get(abins.get(idx, ""), np.nan)}
                    rows.append({**base, **score(lm, app, p)})

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"\n=== {len(df)} rows, {len(df.columns)} cols → {args.out} ===")

    det = df[df["mp_detected"] & df["arc_detected"]]
    print("\nmean apparent age by style:")
    for style in ["anchor"] + STYLES:
        s = det[det["style"] == style]
        if len(s):
            print(f"  {style:12s} age {s['buffalo_age'].mean():5.1f}  (n={len(s)})")

    correlate(det[det["style"] == "chibi"], "chibi only")
    correlate(det[det["style"].isin(STYLES)], "all styles pooled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
