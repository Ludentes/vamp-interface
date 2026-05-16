"""Per-anchor contact sheets for the matryoshka swap sweep.

One PNG per anchor: rows = prompt finish (glossy/satin/matte), columns =
the 4 (cn_strength, pulid_start) combinations, cells = the swapped doll.
Run after matryoshka_swap_sweep.py completes.

    python scripts/matryoshka_swap_montage.py \
        --sweep-dir refs_matryoshka/swap_sweep
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

FINISHES = ["glossy", "satin", "matte"]
COLS = [(0.0, 0.0), (0.0, 0.1), (0.5, 0.0), (0.5, 0.1)]  # (cn, pulid_start)
CELL = 360
LABEL = 28


def _find(swapped_dir: Path, anchor: str, finish: str,
          cn: float, ps: float):
    cn_tag = f"cn{int(round(cn * 100)):03d}"
    ps_tag = f"ps{int(round(ps * 100)):03d}"
    prefix = f"{anchor}_{finish}_{cn_tag}_{ps_tag}_seed"
    hits = sorted(swapped_dir.glob(f"{prefix}*.png"))
    return hits[0] if hits else None


def build_sheet(swapped_dir: Path, anchor: str, out_path: Path) -> bool:
    rows, cols = len(FINISHES), len(COLS)
    canvas = np.full((rows * (CELL + LABEL), cols * CELL, 3), 240,
                     dtype=np.uint8)
    any_cell = False
    for r, finish in enumerate(FINISHES):
        for c, (cn, ps) in enumerate(COLS):
            x, y = c * CELL, r * (CELL + LABEL)
            cv2.rectangle(canvas, (x, y), (x + CELL, y + LABEL),
                          (20, 20, 20), -1)
            cv2.putText(canvas, f"{finish} cn{cn} ps{ps}", (x + 6, y + 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            path = _find(swapped_dir, anchor, finish, cn, ps)
            if path is None:
                continue
            im = cv2.resize(cv2.imread(str(path)), (CELL, CELL))
            canvas[y + LABEL:y + LABEL + CELL, x:x + CELL] = im
            any_cell = True
    if any_cell:
        cv2.imwrite(str(out_path), canvas)
    return any_cell


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", type=Path,
                    default=Path("refs_matryoshka/swap_sweep"))
    args = ap.parse_args()

    swapped_dir = args.sweep_dir / "swapped"
    montage_dir = args.sweep_dir / "montage"
    montage_dir.mkdir(parents=True, exist_ok=True)

    anchors = sorted({"_".join(p.name.split("_")[:2])
                      for p in swapped_dir.glob("*.png")})
    written = 0
    for anchor in anchors:
        out = montage_dir / f"{anchor}.png"
        if build_sheet(swapped_dir, anchor, out):
            written += 1
            print(f"  wrote {out}")
    print(f"[montage] {written} contact sheet(s) -> {montage_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
