"""RGBGrabber stability checks (no cv2 / mediapipe / cuda required).

We monkey-patch the cv2 capture and the cropper builder to use synthetic
inputs, then drive the grab thread through a sequence of frames where
the underlying "detection" jitters. The EMA-strategy cropper that
RGBGrabber uses (PersonaLive's StabilizedFaceCropper) must absorb the
jitter — so the produced 512x512 PIL images should differ much less
than the input crop boxes do.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
from PIL import Image

# Make src importable regardless of where pytest is run from.
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class _FakeCap:
    """cv2.VideoCapture-shaped stub fed by an external frame iterator."""

    def __init__(self, frames):
        self._frames = list(frames)
        self._i = 0
        self._released = False

    def read(self):
        if self._released or self._i >= len(self._frames):
            # Stall: return a zero frame so the grab thread keeps spinning
            # without raising. The test joins via stop() shortly after.
            return True, np.zeros((360, 640, 3), dtype=np.uint8)
        f = self._frames[self._i]
        self._i += 1
        return True, f

    def set(self, *_args, **_kwargs):
        return True

    def release(self):
        self._released = True


class _JitterCropper:
    """Stand-in for StabilizedFaceCropper(strategy='ema').

    Detection bbox jitters around a base center; an internal EMA over
    (cx, cy, sz) smooths it. Output is the EMA-cropped patch.
    """

    def __init__(self, ema_alpha: float = 0.1, **_):
        self.ema_alpha = ema_alpha
        self._cx = None
        self._cy = None
        self._sz = None
        self.detections: list[Tuple[float, float, float]] = []
        self.outputs: list[Tuple[float, float, float]] = []

    def __call__(self, image_pil) -> np.ndarray:
        image = np.array(image_pil)
        h, w = image.shape[:2]
        # The fake "detection" is encoded into the top-left pixel of the
        # synthetic frame. See _make_frame below. This lets us inject
        # deterministic jitter from the test thread.
        cx = float(image[0, 0, 0]) + (w * 0.5)
        cy = float(image[0, 0, 1]) + (h * 0.5)
        sz = 200.0 + float(image[0, 0, 2])
        self.detections.append((cx, cy, sz))
        a = self.ema_alpha
        if self._cx is None:
            self._cx, self._cy, self._sz = cx, cy, sz
        else:
            self._cx = a * cx + (1 - a) * self._cx
            self._cy = a * cy + (1 - a) * self._cy
            self._sz = a * sz + (1 - a) * self._sz
        self.outputs.append((self._cx, self._cy, self._sz))
        half = self._sz * 0.5
        left = max(0, int(self._cx - half))
        top = max(0, int(self._cy - half))
        right = min(w - 1, int(self._cx + half))
        bot = min(h - 1, int(self._cy + half))
        if right <= left or bot <= top:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return image[top:bot, left:right]


def _make_frame(jitter_x: int, jitter_y: int, sz_jitter: int) -> np.ndarray:
    """Synthetic 360x640 BGR frame with detection jitter encoded in pixel 0,0."""
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    # Encode jitter via top-left pixel; clamp to uint8 range.
    img[0, 0, 0] = max(-127, min(127, jitter_x)) % 256
    img[0, 0, 1] = max(-127, min(127, jitter_y)) % 256
    img[0, 0, 2] = max(0, min(255, sz_jitter))
    # Fill the rest with a gradient so cropping produces non-degenerate output.
    img[..., :] = np.linspace(0, 200, 360 * 640 * 3, dtype=np.uint8).reshape(
        360, 640, 3
    )
    return img


def test_ema_stabilizes_jittered_detections():
    """Jittered per-frame detections -> small EMA output variance.

    Threshold: ema variance must be < 25% of detection variance across
    cx, cy, sz over 10 frames with ±10px jitter.
    """
    from arkit_bridge.rgb_grabber import RGBGrabber

    rng = np.random.default_rng(1234)
    jx = rng.integers(-10, 11, size=10)
    jy = rng.integers(-10, 11, size=10)
    jsz = rng.integers(0, 20, size=10)
    frames = [_make_frame(int(jx[i]), int(jy[i]), int(jsz[i])) for i in range(10)]

    g = RGBGrabber(device_index=0, ring_size=32, ema_alpha=0.1)
    fake_cap = _FakeCap(frames)
    fake_cropper = _JitterCropper(ema_alpha=0.1)
    g._open_capture = lambda: fake_cap  # type: ignore[method-assign]
    g._build_cropper = lambda: (fake_cropper, None)  # type: ignore[method-assign]

    g.start()
    # Wait until at least 10 frames have been processed.
    deadline = time.monotonic() + 5.0
    while g.frames_captured < 10 and time.monotonic() < deadline:
        time.sleep(0.02)
    g.stop()

    assert len(fake_cropper.detections) >= 10, (
        f"cropper saw {len(fake_cropper.detections)} detections; expected >=10"
    )

    det = np.array(fake_cropper.detections[:10])
    out = np.array(fake_cropper.outputs[:10])
    det_var = det.var(axis=0)
    out_var = out.var(axis=0)
    # EMA must dampen all three channels.
    for ch, name in enumerate(["cx", "cy", "sz"]):
        assert out_var[ch] < 0.25 * det_var[ch] + 1e-6, (
            f"EMA failed to stabilize {name}: "
            f"det_var={det_var[ch]:.2f} out_var={out_var[ch]:.2f}"
        )


def test_pop_window_returns_pil_images():
    """pop_window(k) returns exactly k PIL.Image at the configured size."""
    from arkit_bridge.rgb_grabber import RGBGrabber

    frames = [_make_frame(0, 0, 0) for _ in range(20)]
    g = RGBGrabber(device_index=0, width=512, height=512, ring_size=32)
    fake_cap = _FakeCap(frames)
    fake_cropper = _JitterCropper(ema_alpha=0.1)
    g._open_capture = lambda: fake_cap  # type: ignore[method-assign]
    g._build_cropper = lambda: (fake_cropper, None)  # type: ignore[method-assign]

    g.start()
    win = g.pop_window(k=8, timeout_s=5.0)
    g.stop()

    assert len(win) == 8, f"expected 8 frames, got {len(win)}"
    for f in win:
        assert isinstance(f, Image.Image)
        assert f.size == (512, 512)
        assert f.mode == "RGB"


def test_missing_camera_raises_at_start():
    """If cv2.VideoCapture returns no frame on probe, start() raises."""
    from arkit_bridge.rgb_grabber import RGBGrabber

    class _DeadCap:
        def read(self):
            return False, None

        def set(self, *a, **k):
            return True

        def release(self):
            pass

    g = RGBGrabber(device_index=99)
    g._open_capture = lambda: _DeadCap()  # type: ignore[method-assign]
    g._build_cropper = lambda: (_JitterCropper(), None)  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="no frames"):
        g.start()
