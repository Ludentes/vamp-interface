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


# Eye-collapse parameters -- all fractions of the inter-eye distance.
_EYE_WIN_FRAC = 0.40    # disk radius around each eye centre to repaint
_EYE_DOT_FRAC = 0.06    # radius of the small replacement folk-art eye dot
_EYE_DARK_T = 110       # grayscale threshold below which a pixel is "eye paint"
_EYE_OVERSIZE_FRAC = 0.18  # collapse only if eye paint fills >this much of disk


def collapse_eyes(img_bgr, kps):
    """Shrink the doll's oversized painted eyes to small folk-art dots.

    inswapper_128 is an identity-only swap: it preserves the target's
    eye-region geometry. A doll generated with huge black painted eyes
    therefore yields a swap that keeps them unchanged. Collapsing each eye to
    a small dot BEFORE the swap makes the swapper carry the small eyes
    through, so the swapped identity reads cleanly.

    kps is the [5,2] arcface keypoint array; kps[0],kps[1] are the eye
    centres. An eye is collapsed only when its dark "eye paint" fills more
    than _EYE_OVERSIZE_FRAC of the search disk -- dolls that already painted
    small folk-art eyes are left untouched. For an oversized eye the dark
    pixels are repainted with the surrounding skin colour and a small dot is
    drawn at the centre. Returns a NEW BGR image; the input is not modified.
    If kps is unusable an unchanged copy is returned.
    """
    out = img_bgr.copy()
    eye_l = np.asarray(kps[0], dtype=float)
    eye_r = np.asarray(kps[1], dtype=float)
    inter = float(np.linalg.norm(eye_r - eye_l))
    if inter < 8.0:
        return out
    win = int(round(inter * _EYE_WIN_FRAC))
    dot_r = max(2, int(round(inter * _EYE_DOT_FRAC)))
    h, w = out.shape[:2]
    for center in (eye_l, eye_r):
        cx, cy = int(round(center[0])), int(round(center[1]))
        x0, y0 = max(cx - win, 0), max(cy - win, 0)
        x1, y1 = min(cx + win, w), min(cy + win, h)
        patch = out[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= win * win
        blob = (gray < _EYE_DARK_T) & disk
        disk_area = int(disk.sum())
        if disk_area == 0:
            continue
        # leave eyes that already painted small -- only fix oversized ones
        if int(blob.sum()) / disk_area < _EYE_OVERSIZE_FRAC:
            continue
        skin = patch[~blob]
        if skin.size == 0:
            continue
        skin_col = np.median(skin.reshape(-1, 3), axis=0)
        patch[blob] = skin_col
        out[y0:y1, x0:x1] = patch
        cv2.circle(out, (cx, cy), dot_r, (45, 35, 30), -1, cv2.LINE_AA)
    return out


def swap_identity(app, swapper, doll_bgr, source_face, collapse=True):
    """Swap source_face's identity onto the doll.

    When collapse is True (default), the doll's oversized painted eyes are
    shrunk to folk-art dots before the swap (see collapse_eyes) -- inswapper
    preserves target eye geometry, so the eyes must be fixed on the input.

    Returns (result_bgr, mode, det_score). mode is 'default' (SCRFD found
    the doll face), 'forced' (MediaPipe synthetic-kps fallback), or 'failed'
    (neither -- the un-swapped doll is returned unchanged).
    """
    kps, bbox = mediapipe_kps_bbox(doll_bgr)
    work = doll_bgr
    if collapse and kps is not None:
        work = collapse_eyes(doll_bgr, kps)

    det = app.get(work)
    if det:
        target = max(det, key=lambda f: f.det_score)
        result = swapper.get(work.copy(), target, source_face,
                             paste_back=True)
        return result, "default", float(target.det_score)

    if kps is not None:
        from insightface.app.common import Face
        target = Face(bbox=bbox, kps=kps, det_score=1.0)
        result = swapper.get(work.copy(), target, source_face,
                             paste_back=True)
        return result, "forced", 0.0

    return doll_bgr, "failed", 0.0
