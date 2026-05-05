"""Per-frame extractors: MediaPipe ARKit-52 + head/eye rotation; DWPose image.

MediaPipe FaceLandmarker emits ARKit-style blendshapes + a 4x4 facial
transformation matrix per detected face. We pull the 52 blendshapes plus
decompose the matrix into head Euler angles. Eye rotations are approximated
from the eyeLook* blendshapes (true Live Link eye SO(3) is not exposed by
MediaPipe).

DWPose render: controlnet_aux DWposeDetector produces the same RGB stick image
PersonaLive's PoseGuider was trained on.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# Canonical ARKit-52 (Apple ordering).
ARKIT_BLENDSHAPE_NAMES = [
    "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft",
    "eyeLookUpLeft", "eyeSquintLeft", "eyeWideLeft",
    "eyeBlinkRight", "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight",
    "eyeLookUpRight", "eyeSquintRight", "eyeWideRight",
    "jawForward", "jawLeft", "jawRight", "jawOpen",
    "mouthClose", "mouthFunnel", "mouthPucker", "mouthLeft", "mouthRight",
    "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthPressLeft", "mouthPressRight", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "noseSneerLeft", "noseSneerRight", "tongueOut",
]
assert len(ARKIT_BLENDSHAPE_NAMES) == 52

_MP_MODEL_PATH = Path(
    "/home/newub/w/vamp-interface/models/mediapipe/face_landmarker.task"
)


_landmarker = None


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        import mediapipe as mp
        from mediapipe.tasks.python import vision

        opts = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(_MP_MODEL_PATH)),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        _landmarker = vision.FaceLandmarker.create_from_options(opts)
    return _landmarker


def _euler_from_matrix(m: np.ndarray) -> tuple[float, float, float]:
    """ZYX intrinsic Euler from a 3x3 rotation matrix. Returns (yaw, pitch, roll)."""
    sy = math.sqrt(m[0, 0] ** 2 + m[1, 0] ** 2)
    if sy > 1e-6:
        roll = math.atan2(m[2, 1], m[2, 2])
        pitch = math.atan2(-m[2, 0], sy)
        yaw = math.atan2(m[1, 0], m[0, 0])
    else:
        roll = math.atan2(-m[1, 2], m[1, 1])
        pitch = math.atan2(-m[2, 0], sy)
        yaw = 0.0
    return yaw, pitch, roll


def _eye_yaw_pitch(blendshapes: dict, side: str) -> tuple[float, float, float]:
    """Approximate eye yaw/pitch from eyeLook* blendshapes (in radians).

    Roll is set to 0 — MediaPipe has no eye-roll signal.
    """
    s = side.capitalize()
    up = blendshapes.get(f"eyeLookUp{s}", 0.0)
    down = blendshapes.get(f"eyeLookDown{s}", 0.0)
    in_ = blendshapes.get(f"eyeLookIn{s}", 0.0)
    out = blendshapes.get(f"eyeLookOut{s}", 0.0)
    pitch = math.radians(30.0) * (up - down)
    yaw_sign = +1.0 if side == "right" else -1.0
    yaw = math.radians(30.0) * yaw_sign * (in_ - out)
    return yaw, pitch, 0.0


def extract_arkit_61(image_rgb: np.ndarray) -> np.ndarray | None:
    """Run MediaPipe FaceLandmarker; return the 61-float ARKit vector or None."""
    import mediapipe as mp

    landmarker = _get_landmarker()
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=image_rgb.astype(np.uint8))
    res = landmarker.detect(mp_image)
    if not res.face_blendshapes or not res.facial_transformation_matrixes:
        return None
    bs_list = res.face_blendshapes[0]
    bs = {b.category_name: b.score for b in bs_list}
    out = np.zeros(61, dtype=np.float32)
    for i, name in enumerate(ARKIT_BLENDSHAPE_NAMES):
        out[i] = bs.get(name, 0.0)
    M = np.array(res.facial_transformation_matrixes[0])  # 4x4
    yaw, pitch, roll = _euler_from_matrix(M[:3, :3])
    out[52:55] = (yaw, pitch, roll)
    out[55:58] = _eye_yaw_pitch(bs, "left")
    out[58:61] = _eye_yaw_pitch(bs, "right")
    return out


_dwpose = None


def get_dwpose_image(image_rgb: np.ndarray) -> np.ndarray:
    """Run DWPose; return its rendered RGB stick image at 512x512."""
    global _dwpose
    if _dwpose is None:
        from controlnet_aux import DWposeDetector
        _dwpose = DWposeDetector()
    from PIL import Image
    pil = Image.fromarray(image_rgb)
    out = _dwpose(pil, output_type="np", detect_resolution=512,
                  image_resolution=512)
    return out


def loose_face_crop(image_rgb: np.ndarray, target: int = 512) -> np.ndarray | None:
    """Pad/crop to target² with the detected face roughly centered.

    "Loose" = translate+scale only, head pose intact (no FFHQ template warp).
    """
    import cv2

    landmarker = _get_landmarker()
    import mediapipe as mp

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=image_rgb.astype(np.uint8))
    res = landmarker.detect(mp_image)
    if not res.face_landmarks:
        return None
    h, w = image_rgb.shape[:2]
    pts = np.array([(p.x * w, p.y * h) for p in res.face_landmarks[0]])
    cx, cy = pts.mean(axis=0)
    bbox = pts.max(axis=0) - pts.min(axis=0)
    side = float(max(bbox)) * 1.6
    side = max(side, 64.0)
    half = side / 2.0
    x0 = int(round(cx - half)); y0 = int(round(cy - half))
    x1 = int(round(cx + half)); y1 = int(round(cy + half))
    pad_l = max(0, -x0); pad_t = max(0, -y0)
    pad_r = max(0, x1 - w); pad_b = max(0, y1 - h)
    img2 = np.pad(
        image_rgb,
        ((pad_t, pad_b), (pad_l, pad_r), (0, 0)),
        mode="edge",
    )
    x0 += pad_l; x1 += pad_l; y0 += pad_t; y1 += pad_t
    crop = img2[y0:y1, x0:x1]
    return cv2.resize(crop, (target, target), interpolation=cv2.INTER_AREA)
