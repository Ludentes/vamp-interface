"""Detect-and-swap core for the matryoshka identity pipeline.

No sweep knowledge. The generator produces a generic doll; this module detects
the doll's painted face (insightface SCRFD, MediaPipe fallback) and swaps a
real identity onto it. CPU-only -- the swap never contends with ComfyUI for
GPU VRAM.

The default swapper is HyperSwap 1c (FaceFusion Labs, 256px) -- chosen over
inswapper_128 in the 2026-05-18 swap-stage bake-off
(`docs/research/2026-05-18-face-swapper-landscape.md`). load_swapper still
loads inswapper_128 if handed that path, so the bake-off harness can A/B them.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from face_restore import restore_face

_CPU = ["CPUExecutionProvider"]

# MediaPipe Tasks-API face model -- shipped next to this file (the legacy
# mediapipe.solutions API is absent from current wheels).
_LANDMARKER_TASK = Path(__file__).with_name("face_landmarker.task")

# Default identity swapper -- HyperSwap 1c, staged next to inswapper_128.
DEFAULT_SWAPPER = str(Path("~/w/ComfyUI/models/insightface/"
                           "hyperswap_1c_256.onnx").expanduser())

# FaceFusion WARP_TEMPLATE_SET['arcface_128'] -- normalised 5-point template
# (eyeL, eyeR, nose, mouthL, mouthR), scaled by crop size at warp time.
_ARCFACE_128 = np.array([
    [0.36167656, 0.40387734], [0.63696719, 0.40235469],
    [0.50019687, 0.56044219], [0.38710391, 0.72160547],
    [0.61507734, 0.72034453]], dtype=np.float32)


def make_face_app():
    """insightface buffalo_l, detection + recognition, CPU only."""
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l",
                       allowed_modules=["detection", "recognition"],
                       providers=_CPU)
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def load_swapper(model_path=None):
    """Identity swapper, CPU only. Defaults to HyperSwap 1c (DEFAULT_SWAPPER).

    Dispatches on the model filename: a 'hyperswap' file loads as a HyperSwap
    (FaceFusion port); anything else loads as an InsightFace INSwapper-contract
    model (inswapper_128). Both expose the same
    `get(img, target_face, source_face, paste_back=True)` signature, so
    swap_identity is agnostic to which backend it holds.
    """
    path = model_path or DEFAULT_SWAPPER
    if "hyperswap" in Path(path).name.lower():
        return HyperSwap(path)
    from insightface.model_zoo import get_model
    return get_model(path, download=False, providers=_CPU)


class HyperSwap:
    """FaceFusion hyperswap_*_256 swapper behind the INSwapper.get signature.

    Port of facefusion/processors/modules/face_swapper/core.py: warp the
    target face to a 256 crop via the arcface_128 template, feed it
    [-1,1]-normalised alongside the L2-normalised ArcFace source embedding,
    de-normalise the output, and paste it back through the model's own mask.
    face_swapper_weight is left at its default 0.5 -> embedding balance is a
    no-op, so it is omitted.
    """

    def __init__(self, onnx_path):
        self.sess = ort.InferenceSession(onnx_path, providers=_CPU)
        self.size = 256

    def get(self, img, target_face, source_face, paste_back=True):
        kps = np.asarray(target_face.kps, dtype=np.float32)
        tmpl = _ARCFACE_128 * self.size
        affine = cv2.estimateAffinePartial2D(
            kps, tmpl, method=cv2.RANSAC, ransacReprojThreshold=100)[0]
        crop = cv2.warpAffine(img, affine, (self.size, self.size),
                              borderMode=cv2.BORDER_REPLICATE,
                              flags=cv2.INTER_AREA)

        blob = crop[:, :, ::-1].astype(np.float32) / 255.0     # BGR->RGB, 0..1
        blob = (blob - 0.5) / 0.5                               # -> [-1, 1]
        blob = blob.transpose(2, 0, 1)[None]
        src = source_face.normed_embedding.reshape(1, -1).astype(np.float32)

        out, mask = self.sess.run(None, {"source": src, "target": blob})
        out = out[0].transpose(1, 2, 0)                         # CHW -> HWC
        out = np.clip(out * 0.5 + 0.5, 0, 1)[:, :, ::-1] * 255  # RGB->BGR
        out = out.astype(np.float32)
        face_mask = np.clip(mask[0, 0], 0, 1).astype(np.float32)

        if not paste_back:
            return out.astype(np.uint8)

        inv = cv2.invertAffineTransform(affine)
        h, w = img.shape[:2]
        warped = cv2.warpAffine(out, inv, (w, h),
                                borderMode=cv2.BORDER_REPLICATE)
        warped_mask = cv2.warpAffine(face_mask, inv, (w, h))[..., None]
        warped_mask = np.clip(warped_mask, 0, 1)
        blended = warped * warped_mask + img.astype(np.float32) * (1 - warped_mask)
        return np.clip(blended, 0, 255).astype(np.uint8)


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


def _crop_region(bbox, img_shape, margin_frac=0.45):
    """Square region around the MediaPipe face bbox, clamped to the image.

    Expands the larger bbox side by margin_frac on each side so the crop
    carries enough context for detection + restoration. Returns integer
    (x0, y0, x1, y1); the region may be non-square if clamped at a border.
    """
    x0, y0, x1, y1 = (float(v) for v in bbox)
    h, w = img_shape[:2]
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = max(x1 - x0, y1 - y0) * (0.5 + margin_frac)
    rx0 = int(max(0, round(cx - half)))
    ry0 = int(max(0, round(cy - half)))
    rx1 = int(min(w, round(cx + half)))
    ry1 = int(min(h, round(cy + half)))
    return rx0, ry0, rx1, ry1


def _feathered_mask(h, w, feather_frac=0.12):
    """Float mask, 1.0 in the interior, Gaussian-feathered to 0.0 at edges.

    Used for seam-free paste-back of the swapped crop. feather_frac is the
    inset (as a fraction of the shorter side) blurred away. Tiny regions
    where no inset fits are returned as all-ones.
    """
    mask = np.zeros((h, w), dtype=np.float32)
    inset = int(round(min(h, w) * feather_frac))
    inset = max(0, min(inset, min(h, w) // 2 - 1))
    if inset <= 0:
        mask[:] = 1.0
        return mask
    mask[inset:h - inset, inset:w - inset] = 1.0
    k = 2 * inset + 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return np.clip(mask, 0.0, 1.0)


def crop_and_upscale(img_bgr, bbox, target=512):
    """Crop the face region around bbox and Lanczos-upscale its longer side.

    The doll face is a small image patch; isolating and upscaling it to
    `target` px is what lets SCRFD detect it and gives the swap real pixels.
    Returns (up_bgr, (x0, y0, x1, y1)), or (None, None) on an empty crop.
    Shared by swap_identity and the eval harness so the metric's detection
    path matches the pipeline's.
    """
    rx0, ry0, rx1, ry1 = _crop_region(bbox, img_bgr.shape)
    crop = img_bgr[ry0:ry1, rx0:rx1]
    if crop.size == 0:
        return None, None
    ch, cw = crop.shape[:2]
    s = float(target) / max(ch, cw)
    up = cv2.resize(crop, (max(1, round(cw * s)), max(1, round(ch * s))),
                    interpolation=cv2.INTER_LANCZOS4)
    return up, (rx0, ry0, rx1, ry1)


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


def swap_identity(app, swapper, doll_bgr, source_face, collapse=True,
                  restore=False):
    """Swap source_face's identity onto the doll's painted face.

    The doll face is a small patch of a large image -- too few pixels for
    SCRFD to detect or for the swapper to align well. So the face region is
    cropped, upscaled to 512 px, and the detect/collapse/swap/restore
    pipeline runs on that isolated high-res crop; the restored crop is then
    feathered back into a copy of the doll. Backend-agnostic: `swapper` is
    whatever load_swapper returned (HyperSwap 1c by default, or inswapper).

    When collapse is True the doll's oversized painted eyes are shrunk to
    folk-art dots before the swap. The collapse step was introduced because
    inswapper is identity-only and carries the target's eye geometry through
    unchanged; it is kept on for HyperSwap too -- the 2026-05-18 swap bake-off
    measured HyperSwap *with* collapse on (id_cos 0.796), so it stays on to
    match that measured configuration. It is at worst redundant if the
    backend regenerates eyes, never harmful.

    restore (GFPGAN over the swapped crop) defaults False: the swap-test A/B
    measured GFPGAN *lowering* median identity cosine 0.773 -> 0.525 -- it
    regularizes the swapped face toward a generic restoration prior. The
    crop->upscale restructure alone carries the identity lift.

    Returns (result_bgr, mode, det_score). mode is 'default' (SCRFD found
    the crop face), 'forced' (MediaPipe synthetic-kps fallback), or 'failed'
    (no detectable doll face -- the un-swapped doll is returned unchanged).
    """
    kps_full, bbox_full = mediapipe_kps_bbox(doll_bgr)
    if kps_full is None or bbox_full is None:
        return doll_bgr, "failed", 0.0

    up, region = crop_and_upscale(doll_bgr, bbox_full)
    if up is None:
        return doll_bgr, "failed", 0.0
    rx0, ry0, rx1, ry1 = region

    kps_up, bbox_up = mediapipe_kps_bbox(up)
    work = up
    if collapse and kps_up is not None:
        work = collapse_eyes(up, kps_up)

    det = app.get(work)
    if det:
        target = max(det, key=lambda f: f.det_score)
        mode, det_score = "default", float(target.det_score)
    elif kps_up is not None:
        from insightface.app.common import Face
        target = Face(bbox=bbox_up, kps=kps_up, det_score=1.0)
        mode, det_score = "forced", 0.0
    else:
        return doll_bgr, "failed", 0.0

    swapped = swapper.get(work.copy(), target, source_face, paste_back=True)
    if restore:
        swapped = restore_face(swapped)

    patch = cv2.resize(swapped, (rx1 - rx0, ry1 - ry0),
                       interpolation=cv2.INTER_LANCZOS4)
    mask = _feathered_mask(ry1 - ry0, rx1 - rx0)[..., None]
    out = doll_bgr.copy()
    region = out[ry0:ry1, rx0:rx1].astype(np.float32)
    blended = patch.astype(np.float32) * mask + region * (1.0 - mask)
    out[ry0:ry1, rx0:rx1] = np.clip(blended, 0, 255).astype(np.uint8)
    return out, mode, det_score
