"""Cell scoring: id_cos, det_score, face_frac, clip_style.

All metrics computed on the *refined* image (the final pipeline output).
"""
from __future__ import annotations

from typing import Any

import numpy as np


def id_cos(app, bgr: np.ndarray, src_emb: np.ndarray) -> tuple[float, float]:
    """Returns (id_cos, det_score). Both NaN on failure."""
    faces = app.get(bgr)
    if not faces:
        return float("nan"), float("nan")
    f = max(faces, key=lambda x: x.det_score)
    cos = float(np.dot(f.normed_embedding, src_emb))
    return cos, float(f.det_score)


def face_frac(app, bgr: np.ndarray) -> float:
    """Largest face bbox area / image area."""
    faces = app.get(bgr)
    if not faces:
        return 0.0
    f = max(faces, key=lambda x: x.det_score)
    x0, y0, x1, y1 = f.bbox.astype(float).tolist()
    area = max(0.0, (x1 - x0) * (y1 - y0))
    return area / (bgr.shape[0] * bgr.shape[1])


_CLIP = None
_CLIP_PREPROC = None
_CLIP_DEV = None


def _clip():
    global _CLIP, _CLIP_PREPROC, _CLIP_DEV
    if _CLIP is None:
        import open_clip
        # Force CPU — driver runs alongside other local GPU workloads and
        # CLIP eats ~500 MB VRAM, which OOM-killed the sweep silently
        # mid-run on 2026-05-20.
        _CLIP_DEV = "cpu"
        model, _, preproc = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=_CLIP_DEV)
        model.eval()
        _CLIP = model
        _CLIP_PREPROC = preproc
    return _CLIP, _CLIP_PREPROC, _CLIP_DEV


def clip_embed(bgr: np.ndarray) -> np.ndarray:
    import cv2
    import torch
    from PIL import Image
    model, preproc, dev = _clip()
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    with torch.no_grad():
        x = preproc(pil).unsqueeze(0).to(dev)
        e = model.encode_image(x)
        e = e / e.norm(dim=-1, keepdim=True)
    return e.cpu().numpy()[0]


def clip_style(anchor_emb: np.ndarray, bgr: np.ndarray) -> float:
    e = clip_embed(bgr)
    return float(np.dot(e, anchor_emb))


def score_cell(app, *, refined_bgr: np.ndarray, src_emb: np.ndarray,
               anchor_emb: np.ndarray, det_mode: str,
               wall_clock: float) -> dict[str, Any]:
    cos, det = id_cos(app, refined_bgr, src_emb)
    ff = face_frac(app, refined_bgr)
    cs = clip_style(anchor_emb, refined_bgr)
    return {
        "id_cos": cos,
        "det_score": det,
        "det_mode": det_mode,
        "face_frac": ff,
        "clip_style": cs,
        "wall_clock": wall_clock,
    }
