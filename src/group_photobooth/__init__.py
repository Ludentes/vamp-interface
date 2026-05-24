"""Group photobooth — multi-person → matryoshka group portrait.

Approach A from docs/superpowers/specs/2026-05-21-group-photobooth-pipeline-design.md:
per-face render via the existing single-face photobooth, then rembg + composite
onto a pre-rendered background in painter's order.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Person:
    """A detected person in the input photo.

    body_bbox: (x1, y1, x2, y2) in photo pixel coords.
    body_mask: uint8 HxW binary mask (0 or 255), same dims as the input photo.
    face_bbox: (x1, y1, x2, y2) face crop in photo pixel coords, None if no
        face was detected (person is then skipped by the driver).
    """
    body_bbox: tuple[int, int, int, int]
    body_mask: np.ndarray
    face_bbox: tuple[int, int, int, int] | None


@dataclass(frozen=True)
class Placement:
    """A doll's placement on the background canvas.

    x_center, y_bottom: foot anchor in background-canvas pixel coords.
    height: doll's height in background-canvas pixels.
    z_order: painter's-algorithm order. Lower = further back, painted first.
    """
    x_center: int
    y_bottom: int
    height: int
    z_order: int


__all__ = ["Person", "Placement"]
