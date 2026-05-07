"""RGB webcam grabber with stabilized FaceMesh crop.

Threaded `cv2.VideoCapture` reader + per-frame MediaPipe FaceMesh crop
(StabilizedFaceCropper, EMA strategy) -> 512x512 RGB PIL.Image queue. The
streaming daemon owns one `RGBGrabber` and pulls fixed-size windows via
`pop_window(k)`, mirroring `LLFReceiver.pop_window` so the daemon's
batch-loop is driver-agnostic.

Design notes
------------
- Single capture thread reads at the camera's native rate, pushes
  cropped frames to a bounded ring buffer; consumers see only the most
  recent frames (drops oldest on overflow). This matches LLF semantics:
  steady-state should never drain the queue empty; pathological lag
  drops oldest data, not newest.
- Crop wobble (memory `project_facemesh_crop_wobble.md`): per-frame
  bbox recompute jitters. We use `StabilizedFaceCropper(strategy="ema")`
  exactly as render-side does — a causal EMA over (cx, cy, sz) keeps
  the crop locked even on noisy detections.
- MediaPipe 0.10.x still ships the legacy `mp.solutions.face_mesh.FaceMesh`
  API (verified 0.10.20 in PersonaLive venv), so we use it directly to
  match `scripts/apply_bridge_to_personalive.py:290`. If a future bump
  drops the legacy path, swap to the Tasks API (FaceLandmarker) inside
  this module — the daemon doesn't care.
- We DO NOT support a video-file fallback. Webcam-or-bust is the actual
  deployment path; a missing camera should surface as a clear startup
  error.
"""
from __future__ import annotations

import collections
import threading
import time
from typing import Optional

import numpy as np
from PIL import Image


class RGBGrabber:
    """cv2.VideoCapture + StabilizedFaceCropper -> bounded ring of PIL Images.

    Consumers call `pop_window(k, timeout_s)` -> list[PIL.Image] of length
    exactly `k` (the most recent k frames in capture order). On underflow
    waits up to `timeout_s` for the queue to fill; returns [] if it cannot.

    Lifecycle: construct -> start() -> pop_window() ... -> stop().
    """

    def __init__(
        self,
        device_index: int = 0,
        width: int = 512,
        height: int = 512,
        ema_alpha: float = 0.1,
        forehead_bias_frac: float = 0.10,
        ring_size: int = 96,
        cap_width: int = 1280,
        cap_height: int = 720,
    ):
        self._device_index = device_index
        self._out_w = width
        self._out_h = height
        self._ema_alpha = ema_alpha
        self._forehead_bias = forehead_bias_frac
        self._ring_size = ring_size
        self._cap_w = cap_width
        self._cap_h = cap_height

        self._cap = None
        self._face_mesh = None
        self._cropper = None
        self._buf: collections.deque[Image.Image] = collections.deque(maxlen=ring_size)
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.frames_captured = 0
        self.frames_dropped = 0
        self.detect_failures = 0

    # --- subclass / test hook ------------------------------------------
    def _open_capture(self):
        """Open the webcam. Override in tests to inject a fake source.

        Returns an object with `.read() -> (bool, BGR ndarray)` and
        `.release()` semantics (cv2.VideoCapture-compatible).
        """
        import cv2  # local import: tests can monkey-patch before start()

        cap = cv2.VideoCapture(self._device_index)
        # Best-effort capture-resolution hints; webcams routinely ignore them.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cap_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cap_h)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _build_cropper(self):
        # PersonaLive utility — same class the offline render path uses, so
        # crop semantics are bit-identical between training corpus and live.
        from src.utils.util import StabilizedFaceCropper  # type: ignore
        import mediapipe as mp  # type: ignore

        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1
        )
        cropper = StabilizedFaceCropper(
            strategy="ema",
            ema_alpha=self._ema_alpha,
            forehead_bias_frac=self._forehead_bias,
            face_mesh=face_mesh,
        )
        return cropper, face_mesh

    # --- lifecycle ------------------------------------------------------
    def start(self) -> None:
        cap = self._open_capture()
        # Probe one read so missing cameras fail at start(), not mid-loop.
        ok, _ = cap.read()
        if not ok:
            try:
                cap.release()
            except Exception:
                pass
            raise RuntimeError(
                f"RGBGrabber: cv2.VideoCapture({self._device_index}) returned "
                "no frames. Is a webcam connected and accessible? (Try "
                "`v4l2-ctl --list-devices` on Linux.)"
            )
        self._cap = cap
        self._cropper, self._face_mesh = self._build_cropper()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="RGBGrabber", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        if self._face_mesh is not None:
            try:
                self._face_mesh.close()
            except Exception:
                pass
            self._face_mesh = None
        self._cropper = None

    # --- internals ------------------------------------------------------
    def _process_bgr(self, bgr: np.ndarray) -> Optional[Image.Image]:
        """BGR ndarray -> 512x512 RGB PIL via StabilizedFaceCropper.

        Returns None if the cropper has no detection AND no EMA history.
        """
        # cv2 returns BGR; PersonaLive crop_face / StabilizedFaceCropper
        # consumes RGB ndarray via PIL.Image, so convert once.
        rgb = bgr[:, :, ::-1]
        pil = Image.fromarray(rgb)
        try:
            crop = self._cropper(pil)  # ndarray, may be non-square edge clip
        except (TypeError, IndexError):
            self.detect_failures += 1
            return None
        # StabilizedFaceCropper clamps to image bounds, so the returned
        # patch can be slightly non-square near frame edges; resize to
        # the canonical 512x512 the pipe expects.
        return Image.fromarray(crop).convert("RGB").resize(
            (self._out_w, self._out_h), Image.LANCZOS
        )

    def _run(self) -> None:
        assert self._cap is not None
        while not self._stop.is_set():
            ok, bgr = self._cap.read()
            if not ok or bgr is None:
                # Camera disconnect or transient failure; back off briefly
                # so we don't busy-spin against a dead device.
                time.sleep(0.01)
                continue
            try:
                pil = self._process_bgr(bgr)
            except Exception:
                # A crash in MediaPipe must not kill the grab thread —
                # the daemon would hang on pop_window forever. Swallow,
                # count as a failure, keep going.
                self.detect_failures += 1
                continue
            if pil is None:
                continue
            with self._cv:
                if len(self._buf) == self._buf.maxlen:
                    self.frames_dropped += 1
                self._buf.append(pil)
                self.frames_captured += 1
                self._cv.notify_all()

    # --- consumer API ---------------------------------------------------
    def pop_window(self, k: int, timeout_s: float = 2.0) -> list[Image.Image]:
        """Return the most recent `k` frames (oldest-first) or [] on timeout.

        Drains exactly k frames from the head of the ring; remaining
        older frames stay so the next pop_window starts where this left
        off. Mirrors `LLFReceiver.pop_window` semantics.
        """
        deadline = time.monotonic() + timeout_s
        with self._cv:
            while len(self._buf) < k:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return []
                self._cv.wait(timeout=remaining)
            out = [self._buf.popleft() for _ in range(k)]
        return out

    @property
    def captured(self) -> int:
        return self.frames_captured

    @property
    def dropped(self) -> int:
        return self.frames_dropped
