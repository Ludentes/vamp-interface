"""Shared mediapipe FaceLandmarker helper.

Produces 51-d blendshape vectors (mediapipe's 52 categories minus the
leading `_neutral`) and 3-d (yaw, pitch, roll) head pose. NaN where the
face wasn't detected.

Pin: mediapipe 0.10.20 — 0.10.21 has a broken protobuf graph parse.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np


def make_landmarker():
    from mediapipe.tasks.python import vision as mp_vis, BaseOptions
    asset = Path.home() / ".cache/mediapipe/face_landmarker.task"
    if not asset.exists():
        # Fallback to repo-local
        for cand in [
            Path(__file__).resolve().parents[1] / "models/mediapipe/face_landmarker.task",
            Path(__file__).resolve().parents[1] / "vendor/weights/mediapipe/face_landmarker.task",
        ]:
            if cand.exists():
                asset = cand
                break
    if not asset.exists():
        raise FileNotFoundError("face_landmarker.task not found in ~/.cache or repo")
    opts = mp_vis.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(asset)),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return mp_vis.FaceLandmarker.create_from_options(opts)


def extract_one(lm, rgb_uint8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (blendshapes[51], ypr[3]) in radians; NaN on failure."""
    import mediapipe as mp
    bs = np.full(51, np.nan, dtype=np.float32)
    ypr = np.full(3, np.nan, dtype=np.float32)
    img = mp.Image(image_format=mp.ImageFormat.SRGB,
                   data=np.ascontiguousarray(rgb_uint8))
    res = lm.detect(img)
    if not res.face_blendshapes:
        return bs, ypr
    cats = res.face_blendshapes[0]
    bs[:] = np.array([c.score for c in cats[1:52]], dtype=np.float32)
    if res.facial_transformation_matrixes:
        M = np.asarray(res.facial_transformation_matrixes[0], dtype=np.float32)
        # Decompose ZYX → yaw(Y), pitch(X), roll(Z)
        sy = float(np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
        if sy > 1e-6:
            pitch = float(np.arctan2(-M[2, 0], sy))
            yaw   = float(np.arctan2(M[1, 0], M[0, 0]))
            roll  = float(np.arctan2(M[2, 1], M[2, 2]))
        else:
            pitch = float(np.arctan2(-M[2, 0], sy))
            yaw   = float(np.arctan2(-M[0, 1], M[1, 1]))
            roll  = 0.0
        ypr[:] = (yaw, pitch, roll)
    return bs, ypr


# 51 mediapipe blendshape names in order (cats[1:52] from FaceLandmarker)
NAMES_51 = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
]
