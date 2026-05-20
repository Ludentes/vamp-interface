"""Spike #4 — full cell: render → swap → color-match → score on id_00.

Skips the refine pass (separate workflow; revisit if scores look promising).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import insightface
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts import swap_core  # noqa: E402

OUT = ROOT / "exp_output" / "photobooth_spike"
PHOTO_IDS = ["id_00", "id_11", "id_16", "id_01"]

OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
        379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
        234, 127, 162, 21, 54, 103, 67, 109]

_FL = None


def fl() -> vision.FaceLandmarker:
    global _FL
    if _FL is None:
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(ROOT / "models/mediapipe/face_landmarker.task")),
            output_face_blendshapes=False, num_faces=1,
            running_mode=vision.RunningMode.IMAGE)
        _FL = vision.FaceLandmarker.create_from_options(opts)
    return _FL


def oval_mask(bgr: np.ndarray) -> np.ndarray | None:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = fl().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.face_landmarks:
        return None
    h, w = bgr.shape[:2]
    lm = res.face_landmarks[0]
    poly = np.array([[lm[i].x * w, lm[i].y * h] for i in OVAL], dtype=np.float32)
    hull = cv2.convexHull(poly.astype(np.int32))
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [hull], 255)
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    return m


def lab_transfer(src_bgr: np.ndarray, src_mask: np.ndarray,
                 dst_bgr: np.ndarray, dst_mask: np.ndarray) -> np.ndarray:
    """Match a/b channel mean+std of dst (in mask) to src (in mask).
    L is preserved (avoid muddying highlights / shadows)."""
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    dst_lab = cv2.cvtColor(dst_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = dst_lab.copy()
    for ch in (1, 2):  # a, b only
        sm = src_lab[..., ch][src_mask > 0]
        dm = dst_lab[..., ch][dst_mask > 0]
        if sm.size < 100 or dm.size < 100:
            continue
        s_mean, s_std = sm.mean(), sm.std() + 1e-6
        d_mean, d_std = dm.mean(), dm.std() + 1e-6
        shifted = (dst_lab[..., ch] - d_mean) * (s_std / d_std) + s_mean
        # Apply only inside dst mask
        out[..., ch] = np.where(dst_mask > 0, shifted, dst_lab[..., ch])
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def lab_delta_ab(a_bgr: np.ndarray, a_mask: np.ndarray,
                 b_bgr: np.ndarray, b_mask: np.ndarray) -> float:
    """Mean (a,b)-channel L2 distance inside intersect of masks."""
    al = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    bl = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    if a_bgr.shape != b_bgr.shape:
        # Resize b to a — both masked
        bl = cv2.resize(bl, (a_bgr.shape[1], a_bgr.shape[0]))
        b_mask = cv2.resize(b_mask, (a_bgr.shape[1], a_bgr.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
    am = (a_mask > 0) & (b_mask > 0)
    if am.sum() < 100:
        return float("nan")
    da = al[..., 1] - bl[..., 1]
    db = al[..., 2] - bl[..., 2]
    d = np.sqrt(da[am] ** 2 + db[am] ** 2)
    return float(d.mean())


def main():
    app = swap_core.make_face_app()
    swapper = swap_core.load_swapper()

    rows = []
    for pid in PHOTO_IDS:
        src = cv2.imread(str(ROOT / f"data/importer/identities/{pid}.png"))
        render = cv2.imread(str(OUT / f"render_{pid}.png"))

        src_face = swap_core.detect_source(app, src)
        if src_face is None:
            print(f"{pid}: no src face"); continue
        src_emb = src_face.normed_embedding

        swap, mode, det = swap_core.swap_identity(
            app, swapper, render, src_face, collapse=True, restore=False)

        # Masks for color-match + scoring
        src_mask = oval_mask(src)
        swap_mask = oval_mask(swap)
        if src_mask is None or swap_mask is None:
            print(f"{pid}: mask fail"); continue

        # Color-match (Lab a/b only, scoped to oval mask)
        cmatch = lab_transfer(src, src_mask, swap, swap_mask)

        # id_cos via insightface on the swap crop
        faces = app.get(swap)
        id_swap = float(np.dot(max(faces, key=lambda f: f.det_score).normed_embedding,
                               src_emb)) if faces else float("nan")
        faces2 = app.get(cmatch)
        id_cmatch = float(np.dot(max(faces2, key=lambda f: f.det_score).normed_embedding,
                                 src_emb)) if faces2 else float("nan")

        # Lab delta_ab: src skin vs swap skin (lower = better match)
        d_swap = lab_delta_ab(src, src_mask, swap, swap_mask)
        d_cmatch = lab_delta_ab(src, src_mask, cmatch, swap_mask)

        cv2.imwrite(str(OUT / f"swap_{pid}.png"), swap)
        cv2.imwrite(str(OUT / f"cmatch_{pid}.png"), cmatch)
        rows.append((pid, mode, det, id_swap, id_cmatch, d_swap, d_cmatch))
        print(f"{pid}  mode={mode}  det={det:.2f}  "
              f"id_swap={id_swap:.3f}  id_cmatch={id_cmatch:.3f}  "
              f"Δab_swap={d_swap:.2f}  Δab_cmatch={d_cmatch:.2f}")

    # Collage: src | render | swap | cmatch  per row
    cols = []
    for pid in PHOTO_IDS:
        tiles = []
        for kind, p in [("src", f"data/importer/identities/{pid}.png"),
                        ("rnd", str(OUT / f"render_{pid}.png")),
                        ("swp", str(OUT / f"swap_{pid}.png")),
                        ("cmt", str(OUT / f"cmatch_{pid}.png"))]:
            img = cv2.imread(p)
            if img is None:
                img = np.zeros((256, 256, 3), np.uint8)
            tiles.append(cv2.resize(img, (256, 256)))
        cols.append(np.concatenate(tiles, axis=1))
    cv2.imwrite(str(OUT / "fullcell_collage.png"), np.concatenate(cols, axis=0))
    print("wrote fullcell_collage.png")


if __name__ == "__main__":
    main()
