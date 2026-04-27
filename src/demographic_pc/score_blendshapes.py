"""Score PNG directories with MediaPipe FaceLandmarker blendshapes.

Outputs a JSON map {png_path: {blendshape_name: score}} for downstream ridge
fits and α-interpolation analysis. Uses the face_landmarker.task model and
returns the standard 52-channel ARKit-aligned blendshape vector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "mediapipe" / "face_landmarker.task"


def make_landmarker():
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


def score_png(lm, path: Path) -> dict | None:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
    res = lm.detect(mp_image)
    if not res.face_blendshapes:
        return None
    return {c.category_name: float(c.score) for c in res.face_blendshapes[0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, type=Path,
                    help="directory with PNGs (recursed)")
    ap.add_argument("--output", required=True, type=Path,
                    help="path to output JSON")
    args = ap.parse_args()

    pngs = sorted(args.input_dir.rglob("*.png"))
    print(f"[blendshapes] {len(pngs)} PNGs under {args.input_dir}")
    out: dict[str, dict] = {}

    with make_landmarker() as lm:
        for i, p in enumerate(pngs):
            rel = str(p.relative_to(args.input_dir))
            scores = score_png(lm, p)
            if scores is None:
                print(f"  no face detected: {rel}")
                continue
            out[rel] = scores
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(pngs)}] last: {rel}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"[blendshapes] scored {len(out)}/{len(pngs)} → {args.output}")


if __name__ == "__main__":
    main()
