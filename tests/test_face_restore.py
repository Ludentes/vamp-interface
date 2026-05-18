"""face_restore.restore_face -- restoration is optional and never a hard dependency."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import face_restore


def test_restore_face_returns_input_when_restorer_unavailable(monkeypatch):
    monkeypatch.setattr(face_restore, "_load", lambda: None)
    crop = np.full((64, 64, 3), 128, dtype=np.uint8)
    out = face_restore.restore_face(crop)
    assert out is crop


def test_restore_face_returns_input_when_enhance_raises(monkeypatch):
    class _Boom:
        def enhance(self, *a, **k):
            raise RuntimeError("boom")
    monkeypatch.setattr(face_restore, "_load", lambda: _Boom())
    crop = np.full((64, 64, 3), 200, dtype=np.uint8)
    out = face_restore.restore_face(crop)
    assert np.array_equal(out, crop)
