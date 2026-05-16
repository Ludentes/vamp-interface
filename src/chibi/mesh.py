"""ChibiMesh — the renderer-agnostic interface for the chibi mesh pivot.

A textured triangle mesh: vertices, faces, per-vertex colour. Produced by
mesh_extract, transformed by mesh_deform, consumed by mesh_render. Keeping it
free of any pytorch3d type is what lets v2 (UV bake / Blender export) reuse the
extract and deform units unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class ChibiMesh:
    """verts: (V,3) float64 positions. faces: (F,3) int64, 0-based triangle
    indices. rgb: (V,3) float32 per-vertex colour in [0,1]. No opacity — a mesh
    is opaque, and the source data carries none."""
    verts: torch.Tensor
    faces: torch.Tensor
    rgb: torch.Tensor

    def __post_init__(self) -> None:
        assert self.verts.ndim == 2 and self.verts.shape[1] == 3, \
            f"verts must be (V,3), got {tuple(self.verts.shape)}"
        assert self.faces.ndim == 2 and self.faces.shape[1] == 3, \
            f"faces must be (F,3), got {tuple(self.faces.shape)}"
        assert self.rgb.shape == self.verts.shape, \
            f"rgb {tuple(self.rgb.shape)} must match verts {tuple(self.verts.shape)}"
        assert int(self.faces.min()) >= 0 and \
            int(self.faces.max()) < self.verts.shape[0], \
            "face index out of range [0, V)"
