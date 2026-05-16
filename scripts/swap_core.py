"""Detect-and-swap core for the matryoshka identity pipeline.

No sweep knowledge. PuLID generates a generic doll; this module detects the
doll's painted face (insightface SCRFD, MediaPipe fallback) and swaps a real
identity onto it with inswapper_128. CPU-only -- the swap never contends
with ComfyUI for GPU VRAM.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_CPU = ["CPUExecutionProvider"]

# MediaPipe Tasks-API face model -- shipped next to this file (the legacy
# mediapipe.solutions API is absent from current wheels).
_LANDMARKER_TASK = Path(__file__).with_name("face_landmarker.task")


def make_face_app():
    """insightface buffalo_l, detection + recognition, CPU only."""
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l",
                       allowed_modules=["detection", "recognition"],
                       providers=_CPU)
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def load_swapper(model_path):
    """inswapper_128 swapper model, CPU only."""
    from insightface.model_zoo import get_model
    return get_model(model_path, download=False, providers=_CPU)


def detect_source(app, img_bgr):
    """Highest-det_score face in img_bgr, or None if no face."""
    faces = app.get(img_bgr)
    if not faces:
        return None
    return max(faces, key=lambda f: f.det_score)


def mediapipe_kps_bbox(img_bgr):
    """(kps[5,2], bbox[4]) from MediaPipe FaceLandmarker, else (None, None).

    Fallback landmarker for the flat painted doll faces SCRFD cannot see --
    MediaPipe is stylization-tolerant where SCRFD is not. Uses the Tasks API
    (the legacy mediapipe.solutions API is absent from current wheels) and
    needs face_landmarker.task next to this file; the model returns 478
    landmarks (0-467 face mesh, 468-477 iris). kps order is [eyeL, eyeR,
    nose, mouthL, mouthR] (arcface convention, image-left first). Returns
    (None, None) if mediapipe or the model is unavailable -- never crashes.
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
    except ImportError:
        return None, None
    if not _LANDMARKER_TASK.exists():
        return None, None

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]
    opts = vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(_LANDMARKER_TASK)),
        num_faces=1, min_face_detection_confidence=0.1)
    landmarker = vision.FaceLandmarker.create_from_options(opts)
    try:
        res = landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    finally:
        landmarker.close()
    if not res.face_landmarks:
        return None, None
    lm = res.face_landmarks[0]

    def pt(i):
        return np.array([lm[i].x * w, lm[i].y * h])

    eye_a = np.mean([pt(i) for i in range(468, 473)], axis=0)  # iris rings
    eye_b = np.mean([pt(i) for i in range(473, 478)], axis=0)
    nose, m_a, m_b = pt(1), pt(61), pt(291)
    eyes = sorted([eye_a, eye_b], key=lambda p: p[0])
    mouth = sorted([m_a, m_b], key=lambda p: p[0])
    kps = np.array([eyes[0], eyes[1], nose, mouth[0], mouth[1]],
                   dtype=np.float32)
    xs = [lm[i].x * w for i in range(468)]   # 0..467 = face mesh, no iris
    ys = [lm[i].y * h for i in range(468)]
    bbox = np.array([min(xs), min(ys), max(xs), max(ys)], dtype=np.float32)
    return kps, bbox


def swap_identity(app, swapper, doll_bgr, source_face):
    """Swap source_face's identity onto the doll.

    Returns (result_bgr, mode, det_score). mode is 'default' (SCRFD found
    the doll face), 'forced' (MediaPipe synthetic-kps fallback), or 'failed'
    (neither -- the un-swapped doll is returned unchanged).
    """
    det = app.get(doll_bgr)
    if det:
        target = max(det, key=lambda f: f.det_score)
        result = swapper.get(doll_bgr.copy(), target, source_face,
                             paste_back=True)
        return result, "default", float(target.det_score)

    kps, bbox = mediapipe_kps_bbox(doll_bgr)
    if kps is not None:
        from insightface.app.common import Face
        target = Face(bbox=bbox, kps=kps, det_score=1.0)
        result = swapper.get(doll_bgr.copy(), target, source_face,
                             paste_back=True)
        return result, "forced", 0.0

    return doll_bgr, "failed", 0.0
