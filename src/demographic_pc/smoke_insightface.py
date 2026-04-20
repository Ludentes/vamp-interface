"""InsightFace age + gender smoke test.

Uses buffalo_l pack (SCRFD detection + ArcFace + genderage). Unlike the other
two classifiers, insightface runs its own face detection + alignment, so we
feed whole images.

Usage:
    uv run python src/demographic_pc/smoke_insightface.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

ROOT = Path(__file__).resolve().parents[2]


def list_samples() -> list[Path]:
    out = [ROOT / "output" / "phase1" / "phase1_anchor.png"]
    for d in ["courier_legit", "office_legit", "scam_critical"]:
        p = ROOT / "output" / "phase1" / d
        if p.is_dir():
            imgs = sorted(p.glob("*.png"))
            if imgs:
                out.append(imgs[0])
    return out


def main() -> None:
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "genderage"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    for p in list_samples():
        if not p.exists():
            print(f"skip: {p}")
            continue
        img = cv2.imread(str(p))
        faces = app.get(img)
        if not faces:
            print(f"{p.name:40s}  NO FACE")
            continue
        f = faces[0]
        gender = "male" if f.sex == "M" else "female"
        print(f"{p.name:40s}  age={f.age:5.1f}  gender={gender}  "
              f"bbox={tuple(int(x) for x in f.bbox)}")


if __name__ == "__main__":
    main()
