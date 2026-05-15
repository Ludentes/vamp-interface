"""FLAME landmark positions and the chibi quarter-grid targets.

The disk loaders below are wrapped in functools.lru_cache so the fit loop
(which calls landmark_positions every step) parses the template .obj and
embedding .npy exactly once.

A landmark is a fixed barycentric point on a template triangle, so its
position is a differentiable function of the (possibly deformed) template
vertices. We use the 70-point `full_lmk` embedding; indices follow the
standard 68-point layout (chin=8, nose tip=30, eyes 36-47, brows 17-26,
mouth 48-67).
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
import torch

FLAME_TEMPLATE = ("/home/newub/w/LAM/model_zoo/human_parametric_models/"
                  "flame_assets/flame/head_template_mesh.obj")

# Landmark index groups within the 70-point full_lmk layout.
GROUPS = {"brow": list(range(17, 27)), "eye": list(range(36, 48)),
          "nose": [30], "mouth": list(range(48, 68)), "chin": [8]}

# Chibi quarter-grid targets (research doc 2026-05-15-chibi-painter-proportion-rules).
QUARTER_GRID_TARGETS = {
    "lines": {"eye": 0.50, "nose": 0.625, "mouth": 0.75},
    # feature size targets, as fractions / multipliers (see fit.py for use):
    "eye_height_u": 0.25,   # eye bbox height ~ 1/4 head
    "nose_depth_mul": 0.45, # collapse nose z-depth to 45% of original
    "mouth_height_mul": 0.55,  # compress mouth height to a strip
}


@lru_cache(maxsize=4)
def load_landmark_embedding(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (faces_idx (70,), bary (70,3))."""
    d = np.load(path, allow_pickle=True).item()
    faces_idx = np.asarray(d["full_lmk_faces_idx"]).reshape(-1).astype(np.int64)
    bary = np.asarray(d["full_lmk_bary_coords"]).reshape(-1, 3).astype(np.float64)
    return faces_idx, bary


@lru_cache(maxsize=1)
def _template_faces() -> np.ndarray:
    faces = []
    for L in open(FLAME_TEMPLATE):
        if L.startswith("f "):
            idx = [int(p.split("/")[0]) - 1 for p in L.split()[1:]]
            if len(idx) >= 3:
                faces.append(idx[:3])
    return np.asarray(faces, dtype=np.int64)


def landmark_positions(verts: torch.Tensor) -> torch.Tensor:
    """Differentiable (70,3) landmark positions from template verts (5023,3)."""
    faces_idx, bary = load_landmark_embedding(
        FLAME_TEMPLATE.replace("head_template_mesh.obj",
                               "landmark_embedding_with_eyes.npy"))
    faces = _template_faces()
    tri = torch.as_tensor(faces[faces_idx], dtype=torch.long)   # (70,3)
    b = torch.as_tensor(bary, dtype=verts.dtype)                # (70,3)
    corners = verts[tri]                                        # (70,3,3)
    return (corners * b[:, :, None]).sum(1)                     # (70,3)


def _u(y: torch.Tensor, y_crown: torch.Tensor, y_chin: torch.Tensor) -> torch.Tensor:
    return ((y_crown - y) / (y_crown - y_chin).clamp_min(1e-6))


def landmark_lines(verts: torch.Tensor) -> dict:
    """Mean u-position of each landmark group. u from this mesh's own
    crown (max y) and chin (landmark 8)."""
    lm = landmark_positions(verts)
    y_crown = verts[:, 1].max()
    y_chin = lm[8, 1]
    return {name: _u(lm[idx, 1], y_crown, y_chin).mean()
            for name, idx in GROUPS.items()}
