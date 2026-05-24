"""Map per-person photo-coord geometry to background-canvas placements.

Letterboxes the photo's aspect into the background's aspect (preserves
proportions, never stretches). Painter's order = ascending y_bottom so
closer-to-camera dolls (lower in image) paint last.
"""
from __future__ import annotations

from group_photobooth import Person, Placement


def solve_placements(persons: list[Person],
                     photo_hw: tuple[int, int],
                     bg_hw: tuple[int, int]) -> list[Placement]:
    """Project each person's body bbox to background-canvas coords.

    photo_hw, bg_hw are (height, width). The photo is letterboxed into the
    background — uniform scale by `min(bg_h/photo_h, bg_w/photo_w)` and
    centered on the off-axis.
    """
    if not persons:
        return []
    ph, pw = photo_hw
    bh, bw = bg_hw
    s = min(bh / ph, bw / pw)
    dx = (bw - pw * s) / 2.0
    dy = (bh - ph * s) / 2.0

    indexed: list[tuple[Person, int, int, int]] = []
    for person in persons:
        x1, y1, x2, y2 = person.body_bbox
        cx_photo = (x1 + x2) / 2.0
        yb_photo = float(y2)
        h_photo = float(y2 - y1)
        x_center = int(round(dx + cx_photo * s))
        y_bottom = int(round(dy + yb_photo * s))
        height = int(round(h_photo * s))
        indexed.append((person, x_center, y_bottom, height))

    indexed.sort(key=lambda t: t[2])
    return [Placement(x_center=xc, y_bottom=yb, height=h, z_order=z)
            for z, (_, xc, yb, h) in enumerate(indexed)]
