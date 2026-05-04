"""Empirically determine MediaPipe FaceLandmarker's hflip semantics.

Take K FFHQ PNGs locally, run blendshape detection on (image, hflip(image))
pairs, compare. Three possible outcomes:

  A. Identical (anatomical convention; flipping image doesn't change the
     subject's anatomical left/right; MediaPipe should output the same).
  B. Swapped under a left/right channel permutation (image-space convention).
  C. Something else (asymmetric model, broken on flipped input).

We try identity match, then a candidate permutation matching the validation
script's `LEFT_RIGHT_PAIRS + DIRECTIONAL_PAIRS`, and report L2 mean distance
under each. The smallest one tells us the convention.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image


# Same canonical 52-channel order as build_compact_blendshapes.py.
BLENDSHAPE_CHANNELS = [
    "_neutral",
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
]


def make_mirror_perm_v1() -> list[int]:
    """Validation script's permutation: simple left↔right swap, with the
    eyeLookIn↔eyeLookOut cross-flip. (The cross-flip is the part most likely
    to be wrong — if MediaPipe sees a mirrored image, the subject's left eye
    looking-in might still be labeled as "looking-in.")"""
    pairs = [
        ("browDownLeft", "browDownRight"),
        ("browOuterUpLeft", "browOuterUpRight"),
        ("cheekSquintLeft", "cheekSquintRight"),
        ("eyeBlinkLeft", "eyeBlinkRight"),
        ("eyeLookDownLeft", "eyeLookDownRight"),
        ("eyeLookInLeft", "eyeLookOutRight"),    # cross-flip
        ("eyeLookOutLeft", "eyeLookInRight"),    # cross-flip
        ("eyeLookUpLeft", "eyeLookUpRight"),
        ("eyeSquintLeft", "eyeSquintRight"),
        ("eyeWideLeft", "eyeWideRight"),
        ("mouthDimpleLeft", "mouthDimpleRight"),
        ("mouthFrownLeft", "mouthFrownRight"),
        ("mouthLowerDownLeft", "mouthLowerDownRight"),
        ("mouthPressLeft", "mouthPressRight"),
        ("mouthSmileLeft", "mouthSmileRight"),
        ("mouthStretchLeft", "mouthStretchRight"),
        ("mouthUpperUpLeft", "mouthUpperUpRight"),
        ("noseSneerLeft", "noseSneerRight"),
        ("jawLeft", "jawRight"),
        ("mouthLeft", "mouthRight"),
    ]
    return _build_perm(pairs)


def make_mirror_perm_v2() -> list[int]:
    """Same as v1 but WITHOUT the eyeLookIn↔eyeLookOut cross-flip. If MediaPipe
    uses anatomical labels for "in/out direction" (relative to the *eye*, not
    the image), then a hflip just swaps left↔right while preserving in/out."""
    pairs = [
        ("browDownLeft", "browDownRight"),
        ("browOuterUpLeft", "browOuterUpRight"),
        ("cheekSquintLeft", "cheekSquintRight"),
        ("eyeBlinkLeft", "eyeBlinkRight"),
        ("eyeLookDownLeft", "eyeLookDownRight"),
        ("eyeLookInLeft", "eyeLookInRight"),     # straight swap
        ("eyeLookOutLeft", "eyeLookOutRight"),   # straight swap
        ("eyeLookUpLeft", "eyeLookUpRight"),
        ("eyeSquintLeft", "eyeSquintRight"),
        ("eyeWideLeft", "eyeWideRight"),
        ("mouthDimpleLeft", "mouthDimpleRight"),
        ("mouthFrownLeft", "mouthFrownRight"),
        ("mouthLowerDownLeft", "mouthLowerDownRight"),
        ("mouthPressLeft", "mouthPressRight"),
        ("mouthSmileLeft", "mouthSmileRight"),
        ("mouthStretchLeft", "mouthStretchRight"),
        ("mouthUpperUpLeft", "mouthUpperUpRight"),
        ("noseSneerLeft", "noseSneerRight"),
        ("jawLeft", "jawRight"),
        ("mouthLeft", "mouthRight"),
    ]
    return _build_perm(pairs)


def _build_perm(pairs: list[tuple[str, str]]) -> list[int]:
    name_to_idx = {n: i for i, n in enumerate(BLENDSHAPE_CHANNELS)}
    perm = list(range(len(BLENDSHAPE_CHANNELS)))
    for a, b in pairs:
        ai, bi = name_to_idx[a], name_to_idx[b]
        perm[ai], perm[bi] = bi, ai
    return perm


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ffhq-dir", type=Path,
                   default=Path("output/ffhq_images_v2_delta"))
    p.add_argument("--task-file", type=Path,
                   default=Path("models/mediapipe/face_landmarker.task"))
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", type=Path,
                   default=Path("output/mediapipe_distill/hflip_diagnostic.json"))
    args = p.parse_args()

    pngs = sorted(args.ffhq_dir.glob("*.png"))
    random.seed(args.seed)
    sample = random.sample(pngs, min(args.n, len(pngs)))

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(args.task_file)),
        running_mode=VisionRunningMode.IMAGE,
        output_face_blendshapes=True,
        num_faces=1,
    )

    pairs_orig = []
    pairs_flip = []
    with FaceLandmarker.create_from_options(options) as lm:
        for path in sample:
            img = Image.open(path).convert("RGB")
            if img.size != (512, 512):
                img = img.resize((512, 512), Image.Resampling.LANCZOS)
            arr = np.asarray(img)
            arr_flip = arr[:, ::-1].copy()                # hflip

            for image_arr, dest in [(arr, pairs_orig), (arr_flip, pairs_flip)]:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_arr)
                res = lm.detect(mp_image)
                if not res.face_blendshapes:
                    dest.append(None)
                    continue
                vec = np.array([c.score for c in res.face_blendshapes[0]],
                               dtype=np.float32)
                dest.append(vec)

    # Keep only rows where both detections succeeded.
    valid = [(o, f) for o, f in zip(pairs_orig, pairs_flip) if o is not None and f is not None]
    print(f"{len(valid)} / {len(sample)} pairs detected on both sides")
    if not valid:
        raise RuntimeError("zero successful pairs")

    O = np.stack([v[0] for v in valid], axis=0)
    F = np.stack([v[1] for v in valid], axis=0)

    # Hypothesis A: identical (anatomical convention).
    l2_identity = np.linalg.norm(O - F, axis=1).mean()
    # Hypothesis B (v1): permuted with eyeLookIn↔eyeLookOut cross-flip.
    perm_v1 = np.asarray(make_mirror_perm_v1())
    l2_perm_v1 = np.linalg.norm(O - F[:, perm_v1], axis=1).mean()
    # Hypothesis B (v2): permuted with straight in↔in / out↔out swap.
    perm_v2 = np.asarray(make_mirror_perm_v2())
    l2_perm_v2 = np.linalg.norm(O - F[:, perm_v2], axis=1).mean()

    # Per-channel diff (where do they differ most?)
    abs_diff = np.abs(O - F).mean(axis=0)
    top_diff = np.argsort(-abs_diff)[:8]
    top_diff_names = [(BLENDSHAPE_CHANNELS[i], float(abs_diff[i])) for i in top_diff]

    out = {
        "n_pairs": len(valid),
        "L2_mean_identity (no permutation)": float(l2_identity),
        "L2_mean_perm_v1 (validation script's permutation)": float(l2_perm_v1),
        "L2_mean_perm_v2 (in↔in / out↔out, no cross-flip)": float(l2_perm_v2),
        "top_8_channels_by_orig_vs_flip_diff": top_diff_names,
        "verdict": _verdict(l2_identity, l2_perm_v1, l2_perm_v2),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


def _verdict(l2_id: float, l2_v1: float, l2_v2: float) -> str:
    best = min(l2_id, l2_v1, l2_v2)
    if l2_id == best:
        return ("MediaPipe is hflip-INVARIANT (no permutation). "
                "Anatomical convention; student should output identical "
                "blendshapes for original and hflipped image.")
    elif l2_v2 == best:
        return ("MediaPipe uses image-space convention with straight L↔R swap "
                "(no eyeLookIn↔Out cross-flip). Validation script's perm_v1 "
                "is wrong; use perm_v2.")
    else:
        return ("MediaPipe uses image-space convention WITH eyeLookIn↔Out "
                "cross-flip. Validation script's perm_v1 is correct in shape "
                "but the magnitude of mismatch suggests asymmetric learning.")


if __name__ == "__main__":
    main()
