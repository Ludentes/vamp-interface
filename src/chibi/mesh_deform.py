"""Deform a ChibiMesh's vertices with a fitted ChibiField.

The deformation moves verts only — faces and per-vertex rgb pass through
unchanged. That invariance is the point of the mesh pivot: per-vertex colour
on a fixed-topology mesh is deformation-safe (a stretched triangle still
interpolates its three corner colours), unlike a baked splat cloud.

Mirrors scripts/chibi_make_assets.py:deform_with_field, minus the secant-basis
return that only the ARKit asset path needs. Runs in float64: the
searchsorted/interp/exp path in ChibiField carries ~1e-5 error at float32.
"""
from __future__ import annotations
import torch

from chibi.mesh import ChibiMesh
from chibi.fit import load_field_params
from chibi.landmarks import region_falloff_weights, landmark_positions


def apply_chibi(mesh: ChibiMesh, field_params_path: str,
                masks_path: str) -> ChibiMesh:
    """Apply the fitted ChibiField at `field_params_path` to `mesh`.

    The field is re-framed to THIS mesh's own crown/chin (fitted params are
    frame-independent). `mesh` must be FLAME-topology — the first 5023 verts
    are the original FLAME verts, which landmark_positions and the FLAME masks
    index into.
    """
    assert mesh.verts.shape[0] in (5023, 20018), (
        f"apply_chibi expects the 5023 FLAME template or the 20018 baked mesh; "
        f"got {mesh.verts.shape[0]} verts")
    field = load_field_params(field_params_path).double()
    xyz = mesh.verts.to(torch.float64)
    with torch.no_grad():
        field.y_crown.copy_(xyz[:, 1].max())
        field.y_chin.copy_(landmark_positions(xyz[:5023])[8, 1])
        field.z_center.copy_(xyz[:, 2].mean())
    rw = region_falloff_weights(xyz, masks_path)
    with torch.no_grad():
        deformed = field(xyz, region_weights=rw)
    return ChibiMesh(verts=deformed, faces=mesh.faces, rgb=mesh.rgb)
