"""Derive a Canny edge ControlNet conditioning image per FFHQ identity.

Reads the 20 identities from data/importer/identities/, runs Canny edge
detection, saves data/importer/cn_canny/id_NN_canny.png at 1024x1024.

For the Track-2-direct importer corpus the simplest pose-lock is to use the
identity's own portrait as its CN input — pose matches identity, paired
(photoreal, stylized) generations share the same skeleton by construction.
Pose diversity comes from seed variation per (ID, style), not from CN
templates (deferred until corpus expansion).

Usage:
    uv run python scripts/importer_build_cn_canny.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2  # type: ignore[import-not-found]
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id-dir", type=Path, default=REPO / "data/importer/identities")
    ap.add_argument("--out-dir", type=Path, default=REPO / "data/importer/cn_canny")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pngs = sorted(args.id_dir.glob("id_*.png"))
    if not pngs:
        print(f"[cn_canny] no identities found in {args.id_dir}")
        return 1

    for p in pngs:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  [skip] {p.name} unreadable")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Standard ControlNet Canny defaults; thresholds tuned for portraits
        edges = cv2.Canny(gray, 100, 200)
        # Comfy convention: white edges on black background, 3-channel
        edges_rgb = np.stack([edges, edges, edges], axis=-1)
        out = args.out_dir / p.name.replace(".png", "_canny.png")
        cv2.imwrite(str(out), edges_rgb)
        print(f"  [ok] {out.name}")

    print(f"[cn_canny] wrote {len(pngs)} canny maps to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
