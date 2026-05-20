"""Align a posed FLAME mesh to a photo by anchoring named facial landmarks.

The bbox-based isotropic fit (`flame_render._project`) maps the FLAME `face`
mask extent to the MediaPipe bbox extent — but the two cover *different*
anatomy (FLAME `face` includes more forehead proportionally), so the rendered
eyes/nose/mouth land too high. Instead, anchor 6 stable FLAME iBUG-70
landmarks to their MediaPipe-478 counterparts and solve a 2D similarity
transform (scale + translation + small in-plane rotation correction). Head
3D rotation stays in the supplied `rotation` matrix; this only fixes the 2D
fit on top of it.

This module is intentionally pose_cache-schema-agnostic: it takes the 6
MediaPipe landmarks in image-pixel space as input. The collage path runs
MediaPipe live; the training path will get them from an extended pose cache.
"""
from pathlib import Path

import numpy as np

# iBUG-70 ↔ MediaPipe-478 correspondences for 6 stable, expression-robust
# features. (iBUG-68 indexing + 2 extra eye centres in the FLAME embedding.)
IBUG_NOSE_TIP = 30
IBUG_LEFT_EYE_OUTER = 36
IBUG_RIGHT_EYE_OUTER = 45
IBUG_MOUTH_LEFT = 48
IBUG_MOUTH_RIGHT = 54
IBUG_CHIN = 8

MP_NOSE_TIP = 1
MP_LEFT_EYE_OUTER = 33
MP_RIGHT_EYE_OUTER = 263
MP_MOUTH_LEFT = 61
MP_MOUTH_RIGHT = 291
MP_CHIN = 152

# Pairs (mp_index, ibug_index) used to solve the similarity transform.
LANDMARK_PAIRS = [
    (MP_NOSE_TIP, IBUG_NOSE_TIP),
    (MP_LEFT_EYE_OUTER, IBUG_LEFT_EYE_OUTER),
    (MP_RIGHT_EYE_OUTER, IBUG_RIGHT_EYE_OUTER),
    (MP_MOUTH_LEFT, IBUG_MOUTH_LEFT),
    (MP_MOUTH_RIGHT, IBUG_MOUTH_RIGHT),
    (MP_CHIN, IBUG_CHIN),
]

_LMK_EMBED_PATH = Path(
    "/home/newub/w/LAM/model_zoo/human_parametric_models/flame_assets/"
    "flame/landmark_embedding_with_eyes.npy")
_lmk_embed: dict | None = None


def _load_lmk_embed() -> tuple[np.ndarray, np.ndarray]:
    """(faces_idx (70,), bary (70, 3)) — iBUG-70 landmark embedding on FLAME."""
    global _lmk_embed
    if _lmk_embed is None:
        d = np.load(_LMK_EMBED_PATH, allow_pickle=True).item()
        _lmk_embed = {
            "faces_idx": np.asarray(d["full_lmk_faces_idx"]).reshape(-1),
            "bary": np.asarray(d["full_lmk_bary_coords"]).reshape(-1, 3),
        }
    return _lmk_embed["faces_idx"], _lmk_embed["bary"]


def flame_landmarks_3d(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """The 70 iBUG-style 3D landmark positions on a deformed FLAME mesh.

    verts: (5023, 3); faces: (n_faces, 3) triangle vertex indices.
    Returns: (70, 3) float64.
    """
    faces_idx, bary = _load_lmk_embed()
    tris = faces[faces_idx]                       # (70, 3) vertex indices
    tri_verts = verts[tris]                       # (70, 3, 3)
    return np.einsum("lij,li->lj", tri_verts.astype(np.float64), bary)


def solve_similarity(src: np.ndarray, dst: np.ndarray
                     ) -> tuple[float, np.ndarray, np.ndarray]:
    """2D similarity transform `dst ≈ s · R · src + t`, least-squares over N>=2.

    src, dst: (N, 2). Returns (scale, rotation_2x2, translation_2).
    Allows a small in-plane rotation correction on top of the 3D head pose.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    s_c, d_c = src - mu_s, dst - mu_d
    H = s_c.T @ d_c                               # (2, 2)
    U, S, Vt = np.linalg.svd(H)
    R = (Vt.T @ U.T)
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    var_s = (s_c ** 2).sum() / len(src)
    scale = S.sum() / (var_s * len(src))
    t = mu_d - scale * R @ mu_s
    return float(scale), R, t


def aligned_pixels(verts: np.ndarray, rotation: np.ndarray,
                   mp_landmarks_px: np.ndarray, faces: np.ndarray
                   ) -> np.ndarray:
    """Project FLAME verts to image-pixel space, aligned to MediaPipe landmarks.

    verts:            (5023, 3) deformed FLAME vertices.
    rotation:         (3, 3) head rotation (the same matrix used by `render`).
    mp_landmarks_px:  (478, 2) MediaPipe landmark positions in image pixels.
    faces:            (n_faces, 3) FLAME triangle indices (from FlameAssets).
    Returns:          (5023, 2) float pixel coordinates.
    """
    verts = np.asarray(verts, dtype=np.float64)
    R = np.asarray(rotation, dtype=np.float64)
    vr = verts @ R.T
    xy_all = vr[:, :2].copy()
    xy_all[:, 1] *= -1.0                          # FLAME +Y up → image +Y down

    # 3D FLAME landmarks → rotated 2D, in FLAME's own units.
    lm3d = flame_landmarks_3d(verts, faces)
    lm_rot = lm3d @ R.T
    src_xy = lm_rot[:, :2].copy()
    src_xy[:, 1] *= -1.0

    mp_idx = [p[0] for p in LANDMARK_PAIRS]
    ibug_idx = [p[1] for p in LANDMARK_PAIRS]
    src_pts = src_xy[ibug_idx]                    # (6, 2) FLAME-2D
    dst_pts = mp_landmarks_px[mp_idx]             # (6, 2) image-px

    scale, R2, t = solve_similarity(src_pts, dst_pts)
    return scale * xy_all @ R2.T + t
