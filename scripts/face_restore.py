"""Face restoration for the matryoshka swap pipeline -- GFPGAN, lazy + optional."""
from __future__ import annotations

from pathlib import Path

import numpy as np

_GFPGAN_WEIGHTS = Path.home() / "w/ComfyUI/models/facerestore/GFPGANv1.4.pth"

_restorer = None
_load_failed = False


def _load():
    global _restorer, _load_failed
    if _restorer is not None or _load_failed:
        return _restorer
    try:
        from gfpgan import GFPGANer
        _restorer = GFPGANer(model_path=str(_GFPGAN_WEIGHTS), upscale=1,
                             arch="clean", channel_multiplier=2, bg_upsampler=None)
    except Exception as e:
        print(f"[face_restore] GFPGAN unavailable, restoration skipped: {e}")
        _load_failed = True
    return _restorer


def restore_face(crop_bgr: np.ndarray) -> np.ndarray:
    restorer = _load()
    if restorer is None:
        return crop_bgr
    try:
        _, _, restored = restorer.enhance(crop_bgr, has_aligned=False,
                                          only_center_face=True, paste_back=True)
    except Exception as e:
        print(f"[face_restore] enhance failed, returning input: {e}")
        return crop_bgr
    if restored is None:
        return crop_bgr
    return np.ascontiguousarray(restored, dtype=np.uint8)
