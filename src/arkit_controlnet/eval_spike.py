"""Spike metrics: identity drift (ArcFace cosine) and expression (blendshapes).

Both encoders are applied fresh to both images. We do NOT compare against the
stored reverse_index `arcface_fp32` column — that column and this harness may
use different ArcFace models, and cosine is only valid within one encoder.
"""
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from insightface.app import FaceAnalysis
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

ARKIT_BLENDSHAPE_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight", "cheekPuff",
    "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
]
assert len(ARKIT_BLENDSHAPE_NAMES) == 52

_MP_MODEL = Path("models/mediapipe/face_landmarker.task")
_landmarker = None
_arcface = None


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(_MP_MODEL)),
            output_face_blendshapes=True,
            num_faces=1,
        )
        _landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
    return _landmarker


def _get_arcface():
    global _arcface
    if _arcface is None:
        app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
        app.prepare(ctx_id=0)
        _arcface = app
    return _arcface


def _imread(path: Path):
    """cv2.imread that fails loudly instead of returning None on a bad path."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


def bs_read(image_path: Path) -> dict[str, float]:
    """52-d ARKit blendshapes from an image. Missing detection -> all zeros."""
    arr = cv2.cvtColor(_imread(image_path), cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
    res = _get_landmarker().detect(mp_image)
    if not res.face_blendshapes:
        return {n: 0.0 for n in ARKIT_BLENDSHAPE_NAMES}
    got = {c.category_name: float(c.score) for c in res.face_blendshapes[0]}
    return {n: got.get(n, 0.0) for n in ARKIT_BLENDSHAPE_NAMES}


def arcface_cos(path_a: Path, path_b: Path) -> float:
    """Cosine similarity of buffalo_l embeddings. -1.0 if either has no face."""
    app = _get_arcface()
    fa = app.get(_imread(path_a))
    fb = app.get(_imread(path_b))
    if not fa or not fb:
        return -1.0
    ea, eb = fa[0].normed_embedding, fb[0].normed_embedding
    return float(np.dot(ea, eb))


def bs_delta(output_path: Path, identity_path: Path, channels: list[str]) -> float:
    """Mean increase in the axis's target channels, output minus identity."""
    out, idn = bs_read(output_path), bs_read(identity_path)
    return float(np.mean([out[c] - idn[c] for c in channels]))


def bs_vector(image_path: Path) -> np.ndarray:
    """52-d ARKit blendshape vector, ordered by ARKIT_BLENDSHAPE_NAMES."""
    got = bs_read(image_path)
    return np.array([got[n] for n in ARKIT_BLENDSHAPE_NAMES], dtype=np.float64)


def expr_cos(path_a: Path, path_b: Path) -> float:
    """Cosine of two 52-d blendshape vectors, excluding the `_neutral` channel.

    Measures expression similarity. `_neutral` (index 0) is dropped — it is an
    inverse summary of all the others and would wash out the signal. Returns
    -1.0 if either vector is all-zero (no face detected -> bs_read all zeros).
    """
    a = bs_vector(path_a)[1:]
    b = bs_vector(path_b)[1:]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return -1.0
    return float(np.dot(a, b) / (na * nb))


def face_landmarks_xy(image_path: Path) -> np.ndarray:
    """(478, 2) array of normalized [0,1] (x, y) MediaPipe face landmarks.

    Raises ValueError if no face is detected — callers select among several
    candidates, so a hard failure is correct here.
    """
    arr = cv2.cvtColor(_imread(image_path), cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
    res = _get_landmarker().detect(mp_image)
    if not res.face_landmarks:
        raise ValueError(f"no face detected in {image_path}")
    return np.array([[p.x, p.y] for p in res.face_landmarks[0]], dtype=np.float64)


def face_landmarks_xyz(image_path: Path) -> np.ndarray:
    """(478, 3) MediaPipe face landmarks: normalized [0,1] (x, y) plus raw
    head-centred relative depth z (more negative = closer to the camera).

    Raises ValueError if no face is detected — callers select among several
    candidates, so a hard failure is correct here.
    """
    arr = cv2.cvtColor(_imread(image_path), cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
    res = _get_landmarker().detect(mp_image)
    if not res.face_landmarks:
        raise ValueError(f"no face detected in {image_path}")
    return np.array([[p.x, p.y, p.z] for p in res.face_landmarks[0]],
                    dtype=np.float64)
