"""Painter's-order alpha-composite RGBA dolls onto an opaque background.

Polish stages — Lab color matching and elliptical drop shadow — are
exposed as standalone functions so the driver can opt in/out per run.
"""
from __future__ import annotations

import cv2
import numpy as np

from group_photobooth import Placement


def composite(background_bgr: np.ndarray,
              dolls_rgba: list[np.ndarray],
              placements: list[Placement]) -> np.ndarray:
    """Paste each doll onto a copy of the background at its placement.

    `dolls_rgba` and `placements` must be parallel lists. Placements are
    sorted by `z_order` ascending and pasted in that order.
    """
    assert len(dolls_rgba) == len(placements), \
        "dolls and placements must be parallel lists"
    out = background_bgr.copy()
    bh, bw = out.shape[:2]
    paired = sorted(zip(dolls_rgba, placements), key=lambda p: p[1].z_order)
    for doll, pl in paired:
        dh, dw = doll.shape[:2]
        new_h = max(1, pl.height)
        new_w = max(1, int(round(dw * new_h / dh)))
        dol = cv2.resize(doll, (new_w, new_h), interpolation=cv2.INTER_AREA)
        x0 = pl.x_center - new_w // 2
        y0 = pl.y_bottom - new_h
        sx0 = max(0, -x0); sy0 = max(0, -y0)
        sx1 = new_w - max(0, (x0 + new_w) - bw)
        sy1 = new_h - max(0, (y0 + new_h) - bh)
        if sx0 >= sx1 or sy0 >= sy1:
            continue
        dx0 = max(0, x0); dy0 = max(0, y0)
        dx1 = dx0 + (sx1 - sx0); dy1 = dy0 + (sy1 - sy0)
        src = dol[sy0:sy1, sx0:sx1]
        alpha = src[..., 3:4].astype(np.float32) / 255.0
        out[dy0:dy1, dx0:dx1] = (
            src[..., :3].astype(np.float32) * alpha
            + out[dy0:dy1, dx0:dx1].astype(np.float32) * (1.0 - alpha)
        ).astype(np.uint8)
    return out


def lab_match(bgr: np.ndarray, alpha: np.ndarray,
              target_lab: tuple[float, float, float],
              max_dL: float = 8.0,
              max_dab: float = 4.0) -> np.ndarray:
    """Shift mean Lab of foreground pixels (alpha>0) toward `target_lab`.

    Caps the shift to (max_dL, max_dab, max_dab) so dolls keep their character.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    fg = alpha > 0
    if not fg.any():
        return bgr
    mean = np.array([lab[..., c][fg].mean() for c in range(3)])
    diff = np.array(target_lab) - mean
    diff[0] = np.clip(diff[0], -max_dL, max_dL)
    diff[1] = np.clip(diff[1], -max_dab, max_dab)
    diff[2] = np.clip(diff[2], -max_dab, max_dab)
    for c in range(3):
        lab[..., c] = np.where(fg, lab[..., c] + diff[c], lab[..., c])
    lab[..., 0] = np.clip(lab[..., 0], 0, 100)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def drop_shadow(background_bgr: np.ndarray, placement: Placement,
                doll_w: int, opacity: float = 0.4,
                offset_yx: tuple[int, int] = (10, 0),
                blur_px: int = 16) -> np.ndarray:
    """Render an elliptical drop shadow under a doll's foot anchor."""
    out = background_bgr.copy()
    bh, bw = out.shape[:2]
    cx = placement.x_center + offset_yx[1]
    cy = placement.y_bottom + offset_yx[0]
    ax = max(4, doll_w // 2)
    ay = max(2, doll_w // 6)
    mask = np.zeros((bh, bw), np.uint8)
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    if blur_px > 0:
        k = 2 * blur_px + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    factor = 1.0 - (mask.astype(np.float32) / 255.0) * opacity
    out = (out.astype(np.float32) * factor[..., None]).astype(np.uint8)
    return out
