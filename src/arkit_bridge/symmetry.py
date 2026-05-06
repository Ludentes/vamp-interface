"""Horizontal-flip symmetry transform for b_expr.

When the input image is mirrored horizontally, the ARKit-58 blendshape
vector + 6 eye-rotation tail must be permuted (L↔R) and certain signs
flipped (eye yaw, eye roll). Indices 0-51 follow the canonical
ARKIT_BLENDSHAPE_NAMES order from llf_csv.py; indices 52-57 are
[LeftEyeYaw, LeftEyePitch, LeftEyeRoll, RightEyeYaw, RightEyePitch,
RightEyeRoll].

Mirror-invariant blendshapes (jawForward/Open, mouthClose/Funnel/Pucker,
mouthRoll*/Shrug*, browInnerUp, cheekPuff, tongueOut) stay put.

L/R blendshape pairs (20 of them) swap. Direction-coded scalars
(jawLeft↔jawRight, mouthLeft↔mouthRight) are pair-swapped — under
mirror, "jaw shifted to subject-left" becomes "jaw shifted to
subject-right", so swapping the values is correct.

Eye rotations: positions swap, yaw and roll negate, pitch unchanged.
"""

from __future__ import annotations

import numpy as np

# (i, j) pairs to swap (canonical ARKit-52 indices). Symmetric: swap is
# an involution.
_BS_PAIRS = [
    (0, 7),    # eyeBlink
    (1, 8),    # eyeLookDown
    (2, 9),    # eyeLookIn
    (3, 10),   # eyeLookOut
    (4, 11),   # eyeLookUp
    (5, 12),   # eyeSquint
    (6, 13),   # eyeWide
    (15, 16),  # jawLeft <-> jawRight
    (21, 22),  # mouthLeft <-> mouthRight
    (23, 24),  # mouthSmile
    (25, 26),  # mouthFrown
    (27, 28),  # mouthDimple
    (29, 30),  # mouthStretch
    (35, 36),  # mouthPress
    (37, 38),  # mouthLowerDown
    (39, 40),  # mouthUpperUp
    (41, 42),  # browDown
    (44, 45),  # browOuterUp
    (47, 48),  # cheekSquint
    (49, 50),  # noseSneer
]


def flip_b_expr(b: np.ndarray) -> np.ndarray:
    """Apply L↔R swap + eye-rotation sign flips for horizontal mirror.

    Accepts shape (58,) or (..., 58). Returns same dtype/shape.
    """
    out = b.copy()
    for i, j in _BS_PAIRS:
        out[..., i], out[..., j] = b[..., j].copy(), b[..., i].copy()
    # Eye rotations: indices 52-57 = [Lyaw, Lpitch, Lroll, Ryaw, Rpitch, Rroll]
    # Swap L<->R; negate yaw and roll.
    out[..., 52] = -b[..., 55]   # new L_yaw  = -old R_yaw
    out[..., 53] =  b[..., 56]   # new L_pitch =  old R_pitch
    out[..., 54] = -b[..., 57]   # new L_roll  = -old R_roll
    out[..., 55] = -b[..., 52]   # new R_yaw   = -old L_yaw
    out[..., 56] =  b[..., 53]   # new R_pitch =  old L_pitch
    out[..., 57] = -b[..., 54]   # new R_roll  = -old L_roll
    return out
