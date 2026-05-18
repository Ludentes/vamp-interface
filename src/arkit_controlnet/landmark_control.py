"""Expression-exemplar selection and MediaPipe mesh rendering for the
landmark-control spike.

Exemplars are real FFHQ images chosen by their stored ARKit blendshape
coefficients (`bs_*` columns in reverse_index.parquet) — we do NOT synthesize
blendshape geometry. The selected image's MediaPipe face mesh becomes the
InfuseNet spatial-control image.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION

from arkit_controlnet.axes import AXES
from arkit_controlnet.eval_spike import ARKIT_BLENDSHAPE_NAMES, face_landmarks_xy

REVERSE_INDEX = Path("output/reverse_index/reverse_index.parquet")
FFHQ_IMAGES = Path("output/ffhq_images")

# the 51 expression channels (every bs_ column except the `_neutral` summary)
_EXPR_COLS = [f"bs_{n}" for n in ARKIT_BLENDSHAPE_NAMES if n != "_neutral"]


def _ffhq_frame() -> pd.DataFrame:
    """FFHQ rows with a detected face and a PNG on disk; bs columns only."""
    on_disk = {p.stem for p in FFHQ_IMAGES.glob("*.png")}
    df = pd.read_parquet(
        REVERSE_INDEX,
        columns=["image_sha256", "source", "bs_detected", *_EXPR_COLS],
    )
    return df[
        (df["source"] == "ffhq")
        & (df["bs_detected"] == True)  # noqa: E712 — pandas mask, not identity
        & (df["image_sha256"].isin(list(on_disk)))
    ].copy()


def _paths(shas) -> list[Path]:
    return [FFHQ_IMAGES / f"{s}.png" for s in shas]


def select_exemplars(axis_name: str, k: int = 3) -> list[Path]:
    """Top-k FFHQ images by the sum of `axis`'s ARKit target channels.

    Returns k candidates (not 1) so the caller can fall through if MediaPipe
    fails to re-detect an exemplar at mesh-render time.
    """
    cols = [f"bs_{c}" for c in AXES[axis_name].target_channels]
    df = _ffhq_frame()
    df["score"] = df[cols].sum(axis=1)
    return _paths(df.nlargest(k, "score")["image_sha256"])


def select_neutral(k: int = 3) -> list[Path]:
    """The k FFHQ images with the least total expression energy (baseline)."""
    df = _ffhq_frame()
    df["energy"] = df[_EXPR_COLS].abs().sum(axis=1)
    return _paths(df.nsmallest(k, "energy")["image_sha256"])


# control-image canvas — matches EmptyLatentImage in the spike workflow
_CANVAS_W, _CANVAS_H = 864, 1152
_FACE_FRAC = 0.45   # face bbox height as a fraction of canvas height
_CENTER_Y = 0.42    # vertical placement of the face centre (portrait framing)


def render_landmark_mesh(image_path: Path) -> np.ndarray:
    """Render an image's MediaPipe face mesh as a normalized control image.

    The mesh is recentred and isotropically scaled so the face occupies a
    fixed fraction of an 864x1152 canvas — so the control image also pins head
    size and position, not just expression. White tessellation edges on black.
    Returns an (H, W, 3) uint8 array. Raises ValueError if no face is detected.
    """
    lm = face_landmarks_xy(image_path)            # (478, 2) in [0, 1]
    xs, ys = lm[:, 0], lm[:, 1]
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    bh = ys.max() - ys.min()
    scale = (_FACE_FRAC * _CANVAS_H) / max(bh, 1e-6)
    px = (xs - cx) * scale + _CANVAS_W / 2
    py = (ys - cy) * scale + _CENTER_Y * _CANVAS_H
    pts = np.stack([px, py], axis=1).astype(np.int32)

    canvas = np.zeros((_CANVAS_H, _CANVAS_W, 3), dtype=np.uint8)
    for a, b in FACEMESH_TESSELATION:
        cv2.line(canvas, tuple(pts[a]), tuple(pts[b]), (255, 255, 255), 1,
                 lineType=cv2.LINE_AA)
    return canvas
