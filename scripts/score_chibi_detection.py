"""Score chibi renders: do MediaPipe + ArcFace even work on them?

For every chibi PNG in the importer ref sets this measures:

  - MediaPipe FaceLandmarker — detection success + 52 ARKit blendshapes.
    Detection rate answers whether the chibi corpus arm can be measured at
    all; blendshape distance to the (neutral) anchor flags garbage fits.
  - ArcFace (insightface buffalo_l) — detection success + identity cosine
    vs the source anchor portrait. A cosine that collapses toward 0 means
    ArcFace is reading noise off the stylised face.
  - Neoteny proxies from the FaceMesh landmarks (only where landmarks fit):
    inter-ocular / face-width, vertical eye position, lower-face ratio.
    These rank renders by how child-like the geometry reads — secondary,
    and only trustworthy on renders MediaPipe actually fit.

Anchor lookup: refs/ used the FFHQ id pool (data/importer/identities/),
refs_v2 and refs_v3 used the Flux id pool (data/importer/identities_flux/).

Usage:
    uv run python scripts/score_chibi_detection.py
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

# (chibi render dir, anchor portrait dir) per batch.
SETS = [
    ("refs", IMPORTER / "refs" / "chibi", IMPORTER / "identities"),
    ("refs_v2", IMPORTER / "refs_v2" / "chibi", IMPORTER / "identities_flux"),
    ("refs_v3", IMPORTER / "refs_v3" / "chibi", IMPORTER / "identities_flux"),
]

ID_RE = re.compile(r"id_(\d+)")

# Canonical MediaPipe FaceMesh landmark indices.
R_EYE_OUTER, R_EYE_INNER = 33, 133
L_EYE_INNER, L_EYE_OUTER = 362, 263
FACE_TOP, FACE_CHIN = 10, 152
FACE_R, FACE_L = 234, 454
MOUTH_LOW = 17  # lower lip bottom


def make_landmarker():
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


def _pt(landmarks, idx: int) -> np.ndarray:
    lm = landmarks[idx]
    return np.array([lm.x, lm.y], dtype=np.float64)


def neoteny_metrics(landmarks) -> dict[str, float]:
    """Geometric child-vibe proxies. Normalised, so resolution-independent."""
    r_eye = (_pt(landmarks, R_EYE_OUTER) + _pt(landmarks, R_EYE_INNER)) / 2
    l_eye = (_pt(landmarks, L_EYE_INNER) + _pt(landmarks, L_EYE_OUTER)) / 2
    top = _pt(landmarks, FACE_TOP)
    chin = _pt(landmarks, FACE_CHIN)
    mouth = _pt(landmarks, MOUTH_LOW)

    iod = float(np.linalg.norm(l_eye - r_eye))
    face_w = float(np.linalg.norm(_pt(landmarks, FACE_L) - _pt(landmarks, FACE_R)))
    face_h = float(np.linalg.norm(chin - top))
    eye_y = (r_eye[1] + l_eye[1]) / 2

    # eye_vpos: 0 = eyes at forehead, 1 = eyes at chin. Childlike skews high
    # (eyes sit below the head midline; big-forehead baby schema).
    eye_vpos = float((eye_y - top[1]) / (chin[1] - top[1])) if face_h > 1e-6 else float("nan")
    # lower_face: chin-to-mouth share of face height. Small = short chin = childlike.
    lower_face = float((chin[1] - mouth[1]) / (chin[1] - top[1])) if face_h > 1e-6 else float("nan")
    return {
        "iod_over_facew": iod / face_w if face_w > 1e-6 else float("nan"),
        "eye_vpos": eye_vpos,
        "lower_face_ratio": lower_face,
    }


def arcface_embed(app: FaceAnalysis, img_path: Path) -> np.ndarray | None:
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    faces = app.get(img)
    if not faces:
        return None
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return faces[-1].normed_embedding


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "exp_output/chibi_score/chibi_scores.csv")
    args = ap.parse_args()

    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    anchor_emb: dict[Path, np.ndarray | None] = {}
    rows: list[dict] = []

    with make_landmarker() as lm:
        for set_name, chibi_dir, anchor_dir in SETS:
            pngs = sorted(chibi_dir.glob("*.png"))
            if not pngs:
                print(f"[{set_name}] no PNGs at {chibi_dir} — skipping")
                continue
            print(f"[{set_name}] {len(pngs)} chibi renders")

            for p in pngs:
                m = ID_RE.search(p.name)
                if not m:
                    print(f"  [skip] {p.name}: no id_NN")
                    continue
                idx = int(m.group(1))
                row: dict = {"set": set_name, "render": p.name, "id_idx": idx}

                # --- MediaPipe ---
                img = cv2.imread(str(p))
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                res = lm.detect(mp_img)
                mp_ok = bool(res.face_landmarks)
                row["mp_detected"] = mp_ok
                if mp_ok:
                    row.update(neoteny_metrics(res.face_landmarks[0]))
                    bs = {c.category_name: c.score for c in res.face_blendshapes[0]}
                    # neutral-expression sanity channels
                    row["bs_jawOpen"] = float(bs.get("jawOpen", float("nan")))
                    row["bs_eyeBlink"] = float(
                        (bs.get("eyeBlinkLeft", 0) + bs.get("eyeBlinkRight", 0)) / 2)
                    row["bs_mean_abs"] = float(np.mean(np.abs(list(bs.values()))))

                # --- ArcFace ---
                anchor_path = anchor_dir / f"id_{idx:02d}.png"
                if anchor_path not in anchor_emb:
                    anchor_emb[anchor_path] = (
                        arcface_embed(app, anchor_path) if anchor_path.exists() else None)
                a_emb = anchor_emb[anchor_path]
                c_emb = arcface_embed(app, p)
                row["arc_detected"] = c_emb is not None
                row["arc_anchor_ok"] = a_emb is not None
                if c_emb is not None and a_emb is not None:
                    row["arc_cosine"] = float(np.dot(c_emb, a_emb))

                rows.append(row)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # --- summary ---
    print(f"\n=== summary ({len(df)} renders) → {args.out} ===")
    for set_name in df["set"].unique():
        s = df[df["set"] == set_name]
        n = len(s)
        mp_rate = s["mp_detected"].mean()
        arc_rate = s["arc_detected"].mean()
        line = (f"{set_name:9s} n={n:3d}  "
                f"MediaPipe {mp_rate*100:5.1f}%  ArcFace {arc_rate*100:5.1f}%")
        cos = s["arc_cosine"].dropna() if "arc_cosine" in s.columns else pd.Series(dtype=float)  # type: ignore
        if len(cos):
            line += f"  cosΜ={cos.mean():.3f} med={cos.median():.3f} [{cos.min():.3f},{cos.max():.3f}]"
        print(line)
        det = s[s["mp_detected"]]
        if len(det) and "eye_vpos" in det:
            print(f"          neoteny  eye_vpos {det['eye_vpos'].mean():.3f}  "
                  f"lower_face {det['lower_face_ratio'].mean():.3f}  "
                  f"iod/facew {det['iod_over_facew'].mean():.3f}  "
                  f"bs_mean_abs {det['bs_mean_abs'].mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
