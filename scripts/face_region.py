"""Face-region inpaint mask for the matryoshka swap-target refine pass.

The generation arm produces a doll with a flat painted face. Before the
downstream inswapper swap, the face region is re-diffused into a face-like
structure (see scripts/matryoshka_refine.py). This module builds the mask
that confines that re-diffusion to the face -- the doll body and lacquer are
left untouched.
"""
from __future__ import annotations

import cv2
import numpy as np

from swap_core import _crop_region, mediapipe_kps_bbox


def build_face_mask(doll_bgr: np.ndarray, feather_frac: float = 0.10,
                    margin_frac: float = 0.35) -> np.ndarray | None:
    """Single-channel uint8 mask: white over the (feathered) doll face region.

    Detects the doll face with MediaPipe, expands its bbox by margin_frac
    (a smaller margin than the swap crop -- the inpaint should not bleed into
    the headscarf), and feathers the rectangle edge so the inpaint has no hard
    seam. Returns None if no face is detected, so the caller can pass the doll
    through unrefined.
    """
    kps, bbox = mediapipe_kps_bbox(doll_bgr)
    if kps is None or bbox is None:
        return None
    h, w = doll_bgr.shape[:2]
    x0, y0, x1, y1 = _crop_region(bbox, doll_bgr.shape, margin_frac=margin_frac)
    mask = np.zeros((h, w), dtype=np.float32)
    mask[y0:y1, x0:x1] = 1.0
    inset = int(round(min(x1 - x0, y1 - y0) * feather_frac))
    if inset > 0:
        k = 2 * inset + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)
