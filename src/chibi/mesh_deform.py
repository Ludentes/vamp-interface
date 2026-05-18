"""Deform a mesh's vertices with the staged ChibiPipeline.

The deformation moves verts only — faces and appearance (per-vertex rgb for a
ChibiMesh, uv/uv_faces/texture for a TexturedMesh) pass through unchanged. That
invariance is the point of the mesh pivot: appearance bound to a fixed-topology
mesh is deformation-safe (a stretched triangle still interpolates its three
corner colours / re-samples the same UV patch), unlike a baked splat cloud.

Runs in float64: the searchsorted/interp/exp path in the stages carries
~1e-5 error at float32.
"""
from __future__ import annotations
import torch

from chibi.mesh import ChibiMesh, TexturedMesh


def _deform_verts(verts: torch.Tensor, faces: torch.Tensor,
                  masks_path: str, params_path: str | None) -> torch.Tensor:
    """Deform (V,3) verts with the staged ChibiPipeline."""
    from chibi.pipeline import ChibiPipeline
    pipe = ChibiPipeline(masks_path, params_path)
    return pipe.run(verts.to(torch.float64), faces)


def apply_chibi(mesh, field_params_path: str | None, masks_path: str):
    """Apply the staged ChibiPipeline to `mesh` (ChibiMesh or TexturedMesh).
    `field_params_path` is the optional pipeline-params JSON (None -> stage
    defaults). Returns the same mesh type with deformed verts; faces and
    appearance pass through unchanged."""
    assert mesh.verts.shape[0] in (5023, 20018), (
        f"apply_chibi expects FLAME topology (5023 or 20018 verts); "
        f"got {mesh.verts.shape[0]}")
    deformed = _deform_verts(mesh.verts, mesh.faces, masks_path,
                             field_params_path)
    if isinstance(mesh, TexturedMesh):
        return TexturedMesh(verts=deformed, faces=mesh.faces, uv=mesh.uv,
                            uv_faces=mesh.uv_faces, texture=mesh.texture)
    return ChibiMesh(verts=deformed, faces=mesh.faces, rgb=mesh.rgb)
