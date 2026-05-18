"""face_region.build_face_mask -- inpaint mask geometry, no real models."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import face_region


def test_build_face_mask_marks_face_region(monkeypatch):
    # synthetic 400x300 doll; force a known central face bbox
    doll = np.full((400, 300, 3), 200, dtype=np.uint8)
    kps = np.zeros((5, 2), dtype=np.float32)
    bbox = np.array([120, 150, 180, 230], dtype=np.float32)
    monkeypatch.setattr(face_region, "mediapipe_kps_bbox", lambda img: (kps, bbox))
    mask = face_region.build_face_mask(doll)
    assert mask is not None
    assert mask.shape == (400, 300)
    assert mask.dtype == np.uint8
    assert mask[190, 150] == 255          # bbox centre is white
    assert mask[0, 0] == 0                # far corner is black
    assert set(np.unique(mask)).issubset(set(range(256)))


def test_build_face_mask_returns_none_when_no_face(monkeypatch):
    doll = np.full((400, 300, 3), 200, dtype=np.uint8)
    monkeypatch.setattr(face_region, "mediapipe_kps_bbox", lambda img: (None, None))
    assert face_region.build_face_mask(doll) is None
