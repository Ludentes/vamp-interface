"""Generate the Canny doll-silhouette template for the matryoshka sweep.

One generic matryoshka outline: hemispherical head merging into an ovoid
body, flat base, vertically symmetric. ControlNet consumes the edge map so
the render keeps the doll form instead of drifting into a portrait.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "importer" / "refs" / "matryoshka" / "template_canny.png"
SIZE = 1024


def main() -> int:
    canvas = np.full((SIZE, SIZE), 255, dtype=np.uint8)
    cx = SIZE // 2
    # body: tall ellipse, lower 2/3 of the canvas
    cv2.ellipse(canvas, (cx, int(SIZE * 0.62)), (int(SIZE * 0.30), int(SIZE * 0.34)),
                0, 0, 360, 0, -1)
    # head: smaller ellipse overlapping the body top, slightly narrower
    cv2.ellipse(canvas, (cx, int(SIZE * 0.30)), (int(SIZE * 0.22), int(SIZE * 0.24)),
                0, 0, 360, 0, -1)
    # flat base: clip the bottom of the body to a straight edge
    cv2.rectangle(canvas, (0, int(SIZE * 0.92)), (SIZE, SIZE), 255, -1)
    edges = cv2.Canny(canvas, 50, 150)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), edges)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
