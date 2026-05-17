"""Deform a mesh's vertices with a fitted ChibiField.

The deformation moves verts only — faces and appearance (per-vertex rgb for a
ChibiMesh, uv/uv_faces/texture for a TexturedMesh) pass through unchanged. That
invariance is the point of the mesh pivot: appearance bound to a fixed-topology
mesh is deformation-safe (a stretched triangle still interpolates its three
corner colours / re-samples the same UV patch), unlike a baked splat cloud.

Mirrors scripts/chibi_make_assets.py:deform_with_field, minus the secant-basis
return that only the ARKit asset path needs. Runs in float64: the
searchsorted/interp/exp path in ChibiField carries ~1e-5 error at float32.
"""
from __future__ import annotations
import torch

from chibi.mesh import ChibiMesh, TexturedMesh
from chibi.fit import load_field_params
from chibi.landmarks import region_falloff_weights, landmark_positions


def _deform_verts(verts: torch.Tensor, field_params_path: str,
                  masks_path: str) -> torch.Tensor:
    """Deform (V,3) verts with the fitted field. The field is re-framed to
    THESE verts' own crown/chin (fitted params are frame-independent); the
    first 5023 verts must be the original FLAME verts, which landmark_positions
    and the FLAME masks index into."""
    field = load_field_params(field_params_path).double()
    xyz = verts.to(torch.float64)
    with torch.no_grad():
        field.y_crown.copy_(xyz[:, 1].max())
        field.y_chin.copy_(landmark_positions(xyz[:5023])[8, 1])
        field.z_center.copy_(xyz[:, 2].mean())
    rw = region_falloff_weights(xyz, masks_path)
    with torch.no_grad():
        return field(xyz, region_weights=rw)


def apply_chibi(mesh, field_params_path: str, masks_path: str):
    """Apply the fitted ChibiField at `field_params_path` to `mesh`.

    `mesh` is a ChibiMesh or a TexturedMesh; the return is the same type with
    deformed verts and unchanged faces + appearance. `mesh` must be
    FLAME-topology — its first 5023 verts the original FLAME verts.
    """
    assert mesh.verts.shape[0] in (5023, 20018), (
        f"apply_chibi expects the 5023 FLAME template or the 20018 baked mesh; "
        f"got {mesh.verts.shape[0]} verts")
    deformed = _deform_verts(mesh.verts, field_params_path, masks_path)
    if isinstance(mesh, TexturedMesh):
        return TexturedMesh(verts=deformed, faces=mesh.faces, uv=mesh.uv,
                            uv_faces=mesh.uv_faces, texture=mesh.texture)
    return ChibiMesh(verts=deformed, faces=mesh.faces, rgb=mesh.rgb)
