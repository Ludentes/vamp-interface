"""FLAME expression render for the CFM conditioning channel.

Pure geometry + rasterization: deform the FLAME template by 52 ARKit
blendshapes, pose it, and flat-shade it to a control image. No photo I/O, no
MediaPipe. See docs/superpowers/specs/2026-05-18-cfm-render-cache-design.md.
"""
from pathlib import Path

import cv2
import numpy as np

from arkit_controlnet.eval_spike import ARKIT_BLENDSHAPE_NAMES

# The FLAME basis axis-0 order: ARKit's 52 blendshapes, alphabetical. MediaPipe
# emits `_neutral` + 51 expression names and never `tongueOut`; the basis is
# those 51 plus `tongueOut`, sorted.
_MP_EXPR = [n for n in ARKIT_BLENDSHAPE_NAMES if n != "_neutral"]
BASIS_CHANNEL_NAMES = sorted([*_MP_EXPR, "tongueOut"])
assert len(BASIS_CHANNEL_NAMES) == 52

# index in BASIS_CHANNEL_NAMES for each MediaPipe expression name
_MP_TO_BASIS = {n: BASIS_CHANNEL_NAMES.index(n) for n in _MP_EXPR}


def mediapipe_to_basis_vector(mp_blendshapes: dict[str, float]) -> np.ndarray:
    """Reorder a MediaPipe blendshape dict into a 52-d basis-channel vector.

    `_neutral` is dropped; `tongueOut` stays 0 (MediaPipe never emits it).
    """
    vec = np.zeros(52, dtype=np.float64)
    for name, basis_idx in _MP_TO_BASIS.items():
        vec[basis_idx] = float(mp_blendshapes.get(name, 0.0))
    return vec
