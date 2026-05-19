"""FLAME expression render for the CFM conditioning channel.

Pure geometry + rasterization: deform the FLAME template by 52 ARKit
blendshapes, pose it, and flat-shade it to a control image. No photo I/O, no
MediaPipe. See docs/superpowers/specs/2026-05-18-cfm-render-cache-design.md.
"""
from dataclasses import dataclass
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


_ASSETS_DIR = Path("output/flame_assets")


@dataclass
class FlameAssets:
    v_template: np.ndarray   # (5023, 3) float32
    faces: np.ndarray        # (n_faces, 3) int32
    arkit_basis: np.ndarray  # (52, 5023, 3) float64


_assets: FlameAssets | None = None


def load_flame_assets() -> FlameAssets:
    """Load and module-cache the FLAME template, faces, and ARKit basis."""
    global _assets
    if _assets is None:
        npz = _ASSETS_DIR / "flame_base.npz"
        basis = _ASSETS_DIR / "flame_arkit_bs.npy"
        if not npz.exists() or not basis.exists():
            raise FileNotFoundError(
                f"{npz} / {basis} missing — run "
                "`python -m arkit_controlnet.prep_flame_assets` first")
        d = np.load(npz)
        _assets = FlameAssets(v_template=d["v_template"], faces=d["faces"],
                              arkit_basis=np.load(basis))
    return _assets


def deform(basis_coeffs: np.ndarray) -> np.ndarray:
    """FLAME vertices for a 52-d basis-channel coefficient vector.

    `basis_coeffs` must be ordered per BASIS_CHANNEL_NAMES (use
    `mediapipe_to_basis_vector`). Returns (5023, 3) float32.
    """
    coeffs = np.asarray(basis_coeffs, dtype=np.float64)
    if coeffs.shape != (52,):
        raise ValueError(f"expected 52 coeffs, got {coeffs.shape}")
    if not np.all(np.isfinite(coeffs)):
        raise ValueError("non-finite blendshape coefficients")
    a = load_flame_assets()
    disp = np.einsum("k,kij->ij", coeffs, a.arkit_basis)
    return (a.v_template + disp).astype(np.float32)


def _face_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Unit normal per face from its three vertices."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.clip(norm, 1e-12, None)


def render(verts: np.ndarray, rotation: np.ndarray,
           bbox: tuple[float, float, float, float],
           modality: str = "normals", H: int = 512, W: int = 512) -> np.ndarray:
    """Rasterize posed FLAME geometry into an (H, W, 3) uint8 control image.

    verts:    (5023, 3) FLAME-space vertices (from `deform`).
    rotation: (3, 3) head rotation (MediaPipe transformation-matrix block).
    bbox:     (cx, cy, w, h) target face box, normalized to the [0,1] image.
    modality: "normals" — per-face normal mapped to RGB.
    """
    if modality != "normals":
        raise ValueError(f"unsupported modality {modality!r}")
    cx, cy, bw, bh = bbox
    if bw <= 0 or bh <= 0:
        raise ValueError(f"degenerate bbox {bbox}")

    a = load_flame_assets()
    faces = a.faces
    vr = verts @ np.asarray(rotation, dtype=np.float64).T   # rotate into camera

    # orthographic projection: x,y to pixels; fit mesh xy-extent into the bbox
    xy = vr[:, :2].copy()
    xy[:, 1] *= -1.0                       # FLAME +Y up -> image +Y down
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    extent = np.maximum(hi - lo, 1e-9)
    scale = min(bw * W / extent[0], bh * H / extent[1])    # isotropic, fits box
    px = (xy - (lo + hi) / 2) * scale
    px[:, 0] += cx * W
    px[:, 1] += cy * H
    pts = px.astype(np.int32)

    normals = _face_normals(vr, faces)
    shade = ((normals * 0.5 + 0.5) * 255).astype(np.uint8)   # (n_faces, 3) RGB
    order = np.argsort(vr[faces, 2].mean(axis=1))            # near (small z) last

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    for i in order:
        tri = faces[i]
        col = (int(shade[i, 2]), int(shade[i, 1]), int(shade[i, 0]))  # cv2 BGR
        cv2.fillConvexPoly(canvas, pts[tri], col, lineType=cv2.LINE_8)
    return canvas
