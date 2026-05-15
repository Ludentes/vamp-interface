#!/usr/bin/env python3
"""Convert iPhone Live Link Face CSV → LAM ARKit-driven motion folder.

Input:  LLF CSV (60 fps, wire-order blendshapes + HeadYaw/Pitch/Roll + per-eye Euler)
Source: an existing VHAP-tracked motion folder for the same take (provides the
        transforms.json camera matrices + canonical_flame_param.npz identity slot)
Output: a mirror motion folder with flame_param/*.npz overwritten to carry
        ARKit-52 expression + LLF head pose, and transforms.json truncated to
        the min of the two frame counts.

The output is consumed by LAM inference run with LAM_USE_ARKIT=1, which makes
gs_renderer.py import FlameHeadSubdivided from flame_arkit instead of flame.
"""

from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ helpers

# 52 ARKit blendshapes in the wire order Apple's Live Link Face protocol streams.
# (Same as the CSV column headers, case-normalized to camelCase.)
LLF_WIRE_ORDER = [
    "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft",
    "eyeLookUpLeft", "eyeSquintLeft", "eyeWideLeft",
    "eyeBlinkRight", "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight",
    "eyeLookUpRight", "eyeSquintRight", "eyeWideRight",
    "jawForward", "jawRight", "jawLeft", "jawOpen",
    "mouthClose", "mouthFunnel", "mouthPucker", "mouthRight", "mouthLeft",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthDimpleLeft", "mouthDimpleRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthPressLeft", "mouthPressRight",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "noseSneerLeft", "noseSneerRight",
    "tongueOut",
]
assert len(LLF_WIRE_ORDER) == 52

# Alphabetical order used by LAM's flame_arkit_bs.npy (matches LAM_WebRender's
# test_expression_1s.json `names` field).
LAM_ALPHA_ORDER = sorted(LLF_WIRE_ORDER, key=str.lower)
assert len(LAM_ALPHA_ORDER) == 52
WIRE_TO_ALPHA = np.array(
    [LLF_WIRE_ORDER.index(n) for n in LAM_ALPHA_ORDER], dtype=np.int32
)


def load_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (arkit52_alphabetical [N,52], head_ypr [N,3], eyes_ypr [N,6])
    where head/eyes are HeadYaw/HeadPitch/HeadRoll and LeftEyeYaw/LeftEyePitch/
    LeftEyeRoll/RightEyeYaw/RightEyePitch/RightEyeRoll in degrees."""
    df = pd.read_csv(csv_path)
    # Column lookup is case-insensitive; LLF CSV is TitleCase.
    cols = {c.lower(): c for c in df.columns}

    # Pull blendshapes in wire order, then permute to alphabetical.
    wire = np.zeros((len(df), 52), dtype=np.float32)
    for i, name in enumerate(LLF_WIRE_ORDER):
        col = cols.get(name.lower())
        if col is None:
            raise KeyError(f"missing blendshape column for {name}")
        wire[:, i] = df[col].to_numpy(dtype=np.float32)
    arkit_alpha = wire[:, WIRE_TO_ALPHA]

    head = np.stack(
        [
            df[cols["headyaw"]].to_numpy(np.float32),
            df[cols["headpitch"]].to_numpy(np.float32),
            df[cols["headroll"]].to_numpy(np.float32),
        ],
        axis=1,
    )
    eyes = np.stack(
        [
            df[cols["lefteyeyaw"]].to_numpy(np.float32),
            df[cols["lefteyepitch"]].to_numpy(np.float32),
            df[cols["lefteyeroll"]].to_numpy(np.float32),
            df[cols["righteyeyaw"]].to_numpy(np.float32),
            df[cols["righteyepitch"]].to_numpy(np.float32),
            df[cols["righteyeroll"]].to_numpy(np.float32),
        ],
        axis=1,
    )
    return arkit_alpha, head, eyes


def head_to_neck_pose(head_ypr: np.ndarray) -> np.ndarray:
    """LLF HeadYaw/HeadPitch/HeadRoll (radians, signed Apple convention) →
    FLAME neck_pose axis-angle [pitch, yaw, roll].

    LLF emits radians despite the legacy field naming. Empirically calibrated
    on the PersonaLive ARKit bridge as EULER_SIGNS=(+1,-1,+1) F=I (the y-axis
    is flipped between LLF and our downstream convention). Same closed-form
    applies to FLAME neck_pose because both use a right-handed axis-angle
    head-orientation rotation. If a render comes back with reversed yaw,
    flip the sign of the second column here.
    """
    yaw, pitch, roll = head_ypr[:, 0], head_ypr[:, 1], head_ypr[:, 2]
    neck = np.stack([pitch, -yaw, roll], axis=1).astype(np.float32)
    return neck


def eyes_to_flame_eyes(eyes_ypr: np.ndarray) -> np.ndarray:
    """LLF per-eye Euler (6 floats, L Yaw/Pitch/Roll then R) → FLAME eyes_pose
    (6 floats, axis-angle pair). FLAME's eyes_pose treats each eye as an
    axis-angle rotation in joint frame. Sign convention is best-effort; if
    the eyes track gaze backwards we flip per-axis here."""
    # Empirically use the same sign flip on yaw as the head; pitch / roll pass.
    le_y, le_p, le_r, re_y, re_p, re_r = [eyes_ypr[:, i] for i in range(6)]
    return np.stack([le_p, -le_y, le_r, re_p, -re_y, re_r], axis=1).astype(np.float32)


# ------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="LLF CSV path")
    ap.add_argument(
        "--vhap_motion_dir",
        required=True,
        help="Source VHAP-tracked motion dir (provides transforms.json + canonical + flame_param/ scaffold)",
    )
    ap.add_argument(
        "--out_motion_dir",
        required=True,
        help="Destination ARKit-driven motion dir (will be created/overwritten)",
    )
    ap.add_argument(
        "--decimate",
        type=int,
        default=2,
        help="LLF→LAM frame decimation (LLF=60fps, LAM=30fps → 2)",
    )
    args = ap.parse_args()

    csv = Path(args.csv)
    src = Path(args.vhap_motion_dir)
    dst = Path(args.out_motion_dir)

    # Sanity on the VHAP source
    src_canonical = src / "canonical_flame_param.npz"
    src_transforms = src / "transforms.json"
    src_flame_dir = src / "flame_param"
    for p in (src_canonical, src_transforms, src_flame_dir):
        if not p.exists():
            raise FileNotFoundError(f"missing in source VHAP dir: {p}")

    # Load LLF
    arkit_alpha, head_ypr, eyes_ypr = load_csv(csv)
    print(f"loaded {len(arkit_alpha)} LLF frames @ 60 fps from {csv.name}")
    # Decimate 60→30
    arkit_alpha = arkit_alpha[:: args.decimate]
    head_ypr = head_ypr[:: args.decimate]
    eyes_ypr = eyes_ypr[:: args.decimate]
    print(f"decimated to {len(arkit_alpha)} frames @ {60 // args.decimate} fps")

    # Source frame inventory
    src_frames = sorted(src_flame_dir.glob("*.npz"))
    print(f"VHAP source has {len(src_frames)} frames in flame_param/")

    n = min(len(arkit_alpha), len(src_frames))
    print(f"using min count = {n} frames for ARKit motion folder")

    # Reset dest
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    (dst / "flame_param").mkdir()

    # Copy canonical (identity slot, will be overridden by anchor downstream)
    shutil.copy2(src_canonical, dst / "canonical_flame_param.npz")

    # Compute pose tensors
    neck_pose = head_to_neck_pose(head_ypr)  # [N,3]
    eyes_pose = eyes_to_flame_eyes(eyes_ypr)  # [N,6]

    # Per-frame .npz writes — mirror src filename scheme so transforms.json
    # references stay valid.
    static_offset_shape = None
    for i, src_frame in enumerate(src_frames[:n]):
        d = dict(np.load(src_frame, allow_pickle=True))
        if static_offset_shape is None and "static_offset" in d:
            static_offset_shape = d["static_offset"].shape
        # Overwrite fields with ARKit-driven values
        d["expr"] = arkit_alpha[i : i + 1].astype(np.float32)  # (1, 52)
        d["neck_pose"] = neck_pose[i : i + 1].astype(np.float32)  # (1, 3)
        d["eyes_pose"] = eyes_pose[i : i + 1].astype(np.float32)  # (1, 6)
        # Zero what LLF doesn't provide; head orientation already in neck.
        d["jaw_pose"] = np.zeros((1, 3), dtype=np.float32)
        d["rotation"] = np.zeros((1, 3), dtype=np.float32)
        d["translation"] = np.zeros((1, 3), dtype=np.float32)
        # Keep shape (300,) and static_offset from VHAP source (overridden later
        # by anchor identity in lam.py infer_single, but file must have key).
        np.savez(dst / "flame_param" / src_frame.name, **d)

    # Truncate transforms.json to first n frames, preserving the camera setup.
    with open(src_transforms) as fp:
        tr = json.load(fp)
    tr_frames = sorted(tr["frames"], key=lambda x: x["flame_param_path"])[:n]
    tr["frames"] = tr_frames
    with open(dst / "transforms.json", "w") as fp:
        json.dump(tr, fp, indent=2)

    # Optional .wav (audio mux at end of LAM inference). Copy if present.
    wav_candidates = list(src.glob("*.wav"))
    for w in wav_candidates:
        shutil.copy2(w, dst / w.name)

    print(f"✅ wrote {n} ARKit-driven frames → {dst}")
    print(f"   expr:      52-d ARKit alphabetical (LAM_USE_ARKIT=1 to consume)")
    print(f"   neck_pose: LLF Head Y/P/R via EULER_SIGNS=(+1,-1,+1)")
    print(f"   eyes_pose: LLF per-eye Y/P/R (sign convention best-effort)")
    print(f"   jaw_pose / rotation / translation: zeros")


if __name__ == "__main__":
    main()
