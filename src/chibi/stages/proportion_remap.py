"""Stage 2 — quarter-grid vertical remap.

The one operator kept from the superseded ChibiField: a monotone piecewise-
linear remap of the normalized head fraction u, placing the feature bands on
the chibi quarter grid (big forehead, eye band on the 1/2 line, mouth on 3/4,
generous rounded chin). Reuses ChibiField.remap so the remap math has a single
source of truth.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import torch

from chibi.field import ChibiField
from chibi.landmarks import landmark_positions


@dataclass
class RemapParams:
    # 5 log-increments for the 6-knot monotone remap; zeros = identity.
    # Defaults solve `incr = log(target_seg / realistic_seg)` so the remap
    # maps realistic_knots (0,.39,.48,.60,.78,1) onto the chibi quarter grid
    # chibi_knots = (0, .375, .5, .625, .75, 1) — brow/eye/nose/mouth bands.
    remap_incr: list = field(
        default_factory=lambda: [-0.0392, 0.3285, 0.0408, -0.3646, 0.1278])


def proportion_remap(verts: torch.Tensor, params: RemapParams) -> torch.Tensor:
    """Return verts with the y-coordinate remapped to the chibi quarter grid;
    x and z unchanged. Vert count unchanged."""
    v = verts.to(torch.float64)
    y_crown = float(v[:, 1].max())
    y_chin = float(landmark_positions(v[:5023])[8, 1])
    fieldm = ChibiField(y_crown=y_crown, y_chin=y_chin,
                        z_center=float(v[:, 2].mean())).double()
    with torch.no_grad():
        fieldm.remap_incr.copy_(torch.tensor(params.remap_incr,
                                             dtype=torch.float64))
    with torch.no_grad():
        u = fieldm.u_of(v)
        u_chibi = fieldm.remap(u)
        new_y = fieldm.y_crown - u_chibi * (fieldm.y_crown - fieldm.y_chin)
    out = v.clone()
    out[:, 1] = new_y
    return out
