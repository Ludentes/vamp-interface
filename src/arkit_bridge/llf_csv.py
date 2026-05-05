"""Load Live Link Face's per-frame ARKit b₆₁ from MySlate_*_iPhone.csv.

Live Link Face exports one CSV row per video frame with 63 columns:
    Timecode, BlendshapeCount, 52 blendshapes (Apple order), HeadYaw, HeadPitch,
    HeadRoll, LeftEyeYaw, LeftEyePitch, LeftEyeRoll, RightEyeYaw, RightEyePitch,
    RightEyeRoll. Rotations are in radians.

This module produces a (N, 61) float32 array aligned with the MOV's frame index.
We re-permute the 52 blendshapes into the canonical order used by
`arkit_bridge.extractors.ARKIT_BLENDSHAPE_NAMES` so the student's input layout
matches whether the source is Live Link or MediaPipe.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from arkit_bridge.extractors import ARKIT_BLENDSHAPE_NAMES


_NAME_LOOKUP = {n.lower(): i for i, n in enumerate(ARKIT_BLENDSHAPE_NAMES)}


def load_llf_b61(csv_path: str | Path) -> np.ndarray:
    """Return a (N, 61) float32 array aligned with the MOV's frame index."""
    csv_path = Path(csv_path)
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        # Build column->canonical-index map for blendshapes.
        bs_cols: list[tuple[int, int]] = []  # (csv_col_idx, canonical_idx)
        rot_cols: dict[str, int] = {}
        for col_i, name in enumerate(header):
            key = name.strip().lower()
            if key in _NAME_LOOKUP:
                bs_cols.append((col_i, _NAME_LOOKUP[key]))
            elif key in {
                "headyaw", "headpitch", "headroll",
                "lefteyeyaw", "lefteyepitch", "lefteyeroll",
                "righteyeyaw", "righteyepitch", "righteyeroll",
            }:
                rot_cols[key] = col_i
        if len(bs_cols) != 52:
            raise ValueError(f"expected 52 blendshape columns, got {len(bs_cols)}")
        if len(rot_cols) != 9:
            raise ValueError(f"expected 9 rotation columns, got {len(rot_cols)}")
        rot_order = [
            "headyaw", "headpitch", "headroll",
            "lefteyeyaw", "lefteyepitch", "lefteyeroll",
            "righteyeyaw", "righteyepitch", "righteyeroll",
        ]
        rows = list(reader)
    out = np.zeros((len(rows), 61), dtype=np.float32)
    for ri, row in enumerate(rows):
        for col_i, can_i in bs_cols:
            out[ri, can_i] = float(row[col_i])
        for k, key in enumerate(rot_order):
            out[ri, 52 + k] = float(row[rot_cols[key]])
    return out
