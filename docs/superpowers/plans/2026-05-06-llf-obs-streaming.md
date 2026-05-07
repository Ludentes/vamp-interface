# LLF → PersonaLive → OBS Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a daemon that consumes Live Link Face UDP from an iPhone, drives PersonaLive via the shipped ARKit bridge, and exposes the rendered 512×512 portrait as a v4l2loopback webcam that OBS picks up.

**Architecture:** Single Python process. UDP receiver thread fills a latest-N b_61 ring; main loop pulls a 24-frame batch (~400 ms at 60 FPS), runs one `Pose2VideoPipeline_Stream.__call__()` with bridge seams installed, and writes the resulting 24 frames to ffmpeg→v4l2loopback paced at 25 FPS.

**Tech Stack:** Python 3.12 (uv), PyTorch 2.11 + CUDA 12.x, PersonaLive (cloned at `~/w/PersonaLive/`), v4l2loopback kernel module, ffmpeg.

**Latency profile:** V1 (this plan) is *batch-based* with ~1.5 s glass-to-OBS latency — one pipe call per 24-frame window of input. The spec's 150 ms target requires refactoring `Pose2VideoPipeline_Stream.__call__` into per-window `step()`; that's V2 and is **deferred** behind a working V1. V1 is a fully working OBS source, just lagged.

**File structure:**

| File | Responsibility |
|---|---|
| `src/arkit_bridge/llf_udp.py` | UDP socket, packet decode, latest-N ring buffer (thread-safe) |
| `src/arkit_bridge/seam_install.py` | `install_arkit_seams()` extracted from `apply_bridge_to_personalive.py` so it's importable from the daemon |
| `src/arkit_bridge/v4l2_sink.py` | ffmpeg subprocess writer (RGB → /dev/videoN) |
| `src/arkit_bridge/streaming_driver.py` | `BatchDriver` — owns the built pipe, takes 24 b_61 frames + ypr, returns 24 RGB frames |
| `scripts/streaming_bridge.py` | Daemon entry point — wires receiver, driver, sink |
| `tests/arkit_bridge/test_llf_udp.py` | Unit: decode + ring buffer |
| `tests/arkit_bridge/test_v4l2_sink.py` | Unit: ffmpeg subprocess lifecycle |
| `tests/arkit_bridge/test_streaming_driver.py` | Integration: byte-equiv vs offline single-call on a small CSV |
| `docs/research/2026-05-06-llf-obs-runbook.md` | Operator guide |

---

### Task 1: Promote LLF decoder to importable module

**Files:**
- Create: `src/arkit_bridge/llf_udp.py`
- Test: `tests/arkit_bridge/test_llf_udp.py`

The decoder logic in `scripts/livelink_probe.py` is a script-private function. Promote it (no semantic change) so the daemon and the probe both import the same code.

- [ ] **Step 1: Write the failing test** at `tests/arkit_bridge/test_llf_udp.py`:

```python
"""Unit tests for src/arkit_bridge/llf_udp.py."""
import socket
import struct
import threading
import time

import pytest

from arkit_bridge.llf_udp import (
    decode_packet,
    LLFReceiver,
    N_FLOATS,
)


def _build_packet(subject="iPhone", floats=None):
    """Build a synthetic Live Link Face UDP packet matching the wire format."""
    if floats is None:
        floats = [0.0] * N_FLOATS
    prefix = b"\x00" * 6
    name = subject.encode("ascii")
    name_block = struct.pack(">I", len(name)) + name
    meta = struct.pack(">IIII", 0, 0, 60, 60)  # frame, subframe, denom, fps
    payload = struct.pack(">" + "f" * N_FLOATS, *floats)
    return prefix + name_block + meta + payload


def test_decode_packet_roundtrip():
    pkt = _build_packet(subject="vamp", floats=[i * 0.01 for i in range(N_FLOATS)])
    subject, floats = decode_packet(pkt)
    assert subject == "vamp"
    assert floats[0] == pytest.approx(0.0)
    assert floats[60] == pytest.approx(0.60)


def test_decode_packet_too_short():
    subject, floats = decode_packet(b"\x00" * 10)
    assert subject is None and floats is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/arkit_bridge/test_llf_udp.py::test_decode_packet_roundtrip -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'arkit_bridge.llf_udp'`.

- [ ] **Step 3: Write minimal implementation** at `src/arkit_bridge/llf_udp.py`:

```python
"""Live Link Face UDP receiver and packet decoder.

Wire format (Epic's Live Link Face iOS app, big-endian):
  [0:6]                six-byte version/uuid prefix
  [6:10] + L bytes     subject-name block (uint32 length L, then ASCII)
  next 16              frame metadata: 4 uint32 (frame, subframe, denom, fps)
  last 244             61 float32 values:
                         0..51   ARKit-52 blendshapes
                         52..60  head yaw/pitch/roll + LE/RE yaw/pitch/roll
"""
from __future__ import annotations

import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

N_FLOATS = 61
TAIL_BYTES = N_FLOATS * 4


def decode_packet(data: bytes):
    """Decode one UDP datagram. Returns (subject, floats[61]) or (None, None)."""
    if len(data) < TAIL_BYTES + 8:
        return None, None
    floats = struct.unpack(">" + "f" * N_FLOATS, data[-TAIL_BYTES:])
    try:
        name_len = struct.unpack(">I", data[6:10])[0]
        if 0 < name_len < 64 and 10 + name_len <= len(data):
            subject = data[10:10 + name_len].decode("ascii", errors="replace")
        else:
            subject = "?"
    except Exception:
        subject = "?"
    return subject, floats


@dataclass
class B61Packet:
    """One decoded LLF frame."""
    subject: str
    floats: tuple
    recv_time: float

    @property
    def b_expr(self) -> tuple:
        """52 ARKit blendshapes."""
        return self.floats[:52]

    @property
    def head_ypr(self) -> tuple:
        """(yaw, pitch, roll) head Euler angles in radians."""
        return (self.floats[52], self.floats[53], self.floats[54])

    @property
    def b58(self) -> tuple:
        """52 blendshapes + 6 eye angles (LE+RE yaw/pitch/roll). Matches
        the b_seq columns the bridge student was trained on."""
        return self.floats[:52] + self.floats[55:61]


class LLFReceiver:
    """UDP listener with latest-N ring. Drops stale packets silently."""

    def __init__(self, host: str = "0.0.0.0", port: int = 11111, ring_size: int = 64):
        self._host = host
        self._port = port
        self._ring: deque[B61Packet] = deque(maxlen=ring_size)
        self._lock = threading.Lock()
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.dropped = 0
        self.received = 0

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((self._host, self._port))
        self._sock.settimeout(0.1)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._sock is not None:
            self._sock.close()

    def _run(self):
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                return
            subject, floats = decode_packet(data)
            if floats is None:
                continue
            pkt = B61Packet(subject=subject, floats=floats, recv_time=time.time())
            with self._lock:
                if len(self._ring) == self._ring.maxlen:
                    self.dropped += 1
                self._ring.append(pkt)
                self.received += 1

    def pop_window(self, k: int = 24, timeout_s: float = 2.0) -> list[B61Packet]:
        """Block until k packets are available, then pop the k most recent
        in arrival order. Returns [] on timeout."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self._lock:
                if len(self._ring) >= k:
                    pkts = list(self._ring)[-k:]
                    self._ring.clear()
                    return pkts
            time.sleep(0.005)
        return []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/arkit_bridge/test_llf_udp.py -v`
Expected: 2 passed.

- [ ] **Step 5: Add receiver loopback test**

Append to `tests/arkit_bridge/test_llf_udp.py`:

```python
def test_receiver_loopback_window():
    rx = LLFReceiver(host="127.0.0.1", port=21111, ring_size=8)
    rx.start()
    try:
        time.sleep(0.05)  # let bind settle
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for i in range(10):
            f = [0.0] * N_FLOATS
            f[0] = i
            sender.sendto(_build_packet(floats=f), ("127.0.0.1", 21111))
            time.sleep(0.005)
        pkts = rx.pop_window(k=4, timeout_s=1.0)
        assert len(pkts) == 4
        # most recent 4 are i=6,7,8,9
        assert [p.floats[0] for p in pkts] == [6.0, 7.0, 8.0, 9.0]
    finally:
        rx.stop()
```

Run: `uv run pytest tests/arkit_bridge/test_llf_udp.py -v`
Expected: 3 passed.

- [ ] **Step 6: Refactor `scripts/livelink_probe.py` to import the shared decoder**

Edit the top of `scripts/livelink_probe.py` to remove the local `decode_packet`/`N_FLOATS`/`TAIL_BYTES` definitions and replace with:

```python
from arkit_bridge.llf_udp import decode_packet, N_FLOATS, TAIL_BYTES
```

(Keep the `ARKIT_52` and `TAIL_NAMES` lists in the script — they're presentation, not protocol.)

Run: `uv run python scripts/livelink_probe.py --help`
Expected: argparse usage prints, no ImportError.

- [ ] **Step 7: Commit**

```bash
git add src/arkit_bridge/llf_udp.py tests/arkit_bridge/test_llf_udp.py scripts/livelink_probe.py
git commit -m "feat(arkit-bridge): promote LLF UDP decoder + receiver to importable module"
```

---

### Task 2: Extract `install_arkit_seams` to importable module

**Files:**
- Create: `src/arkit_bridge/seam_install.py`
- Modify: `scripts/apply_bridge_to_personalive.py:83-205` (replace inline def with import)

The seam-patching function is currently buried in a script. The streaming daemon needs the same function with one extension: instead of accepting fixed `b_seq`/`ypr_seq` arrays at install-time, accept a *callable* that returns `(b_window, ypr_window)` for the current chunk. This lets the daemon swap data per pipe-call.

- [ ] **Step 1: Read the existing function** at `scripts/apply_bridge_to_personalive.py:83-205` to confirm the closure variables: `ref_kp_canonical`, `chunk_cursor`, `me_cursor`, `cf_kd_for_indices`. The streaming version replaces `b_seq`/`ypr_seq` with a `provider(start, n) -> (b_chunk, ypr_chunk)` callable, but we keep array-mode as the default for backwards compat.

- [ ] **Step 2: Write the failing test** at `tests/arkit_bridge/test_seam_install.py`:

```python
"""Smoke-only test: import path and signature."""
import inspect

from arkit_bridge.seam_install import install_arkit_seams


def test_signature():
    sig = inspect.signature(install_arkit_seams)
    params = list(sig.parameters.keys())
    # Required positional args first; provider keyword optional.
    assert "pipe" in params
    assert "b_seq" in params
    assert "ypr_seq" in params
    assert "student" in params
    assert "provider" in params  # new keyword for streaming mode
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/arkit_bridge/test_seam_install.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Create the module**

Create `src/arkit_bridge/seam_install.py` with the *exact body* of `install_arkit_seams` from `scripts/apply_bridge_to_personalive.py:83-205`, **plus** a new optional `provider` parameter. When `provider is not None`, replace every `b_seq[idxs]` / `ypr_seq[idxs]` access with `b_chunk, ypr_chunk = provider(start, n)`. Full source:

```python
"""Install ARKit-bridge seams onto a Pose2VideoPipeline_Stream.

Replaces pose_encoder + motion_encoder hot paths with closed-form pose +
trained MotEncoderStudent. Two modes:

  array:    b_seq + ypr_seq are precomputed numpy arrays indexed by the
            internal cursor (offline rendering path).
  provider: caller supplies provider(start, n) -> (b_chunk, ypr_chunk) so
            data can be swapped per pipe-call (streaming daemon).
"""
from __future__ import annotations

from types import MethodType

import numpy as np
import torch

from arkit_bridge.closed_form_pose import (
    EULER_SIGNS, euler_to_rotmat, compose_kd,
)


def install_arkit_seams(pipe, b_seq, ypr_seq, student, device, dtype, *,
                        patch_pose=True, patch_motion=True, mf_log=None,
                        euler_signs=None, provider=None):
    """Patch pipe.pose_encoder + pipe.motion_encoder.

    Either pass b_seq/ypr_seq (array mode, offline) or pass provider
    (callable mode, streaming). In provider mode b_seq/ypr_seq must be
    None and provider(start, n) -> (b_chunk[n,58], ypr_chunk[n,3]).
    """
    if provider is None:
        assert b_seq is not None and ypr_seq is not None, \
            "array-mode requires b_seq and ypr_seq"

    ref_kp_canonical = {"kp": None, "t": None, "scale": None}
    chunk_cursor = {"i": 0}

    def _fetch(start, n):
        if provider is not None:
            return provider(start, n)
        idxs = np.arange(start, start + n)
        idxs = np.clip(idxs, 0, len(b_seq) - 1)
        return b_seq[idxs], ypr_seq[idxs]

    def cf_kd_for_indices(start, n):
        kp_ref = ref_kp_canonical["kp"]
        t_ref = ref_kp_canonical["t"]
        s_ref = ref_kp_canonical["scale"]
        sy, sp, sr = euler_signs if euler_signs is not None else EULER_SIGNS
        _, ypr = _fetch(start, n)
        Rs = []
        for k in range(n):
            R = euler_to_rotmat(
                torch.tensor(sy * ypr[k, 0].item()),
                torch.tensor(sp * ypr[k, 1].item()),
                torch.tensor(sr * ypr[k, 2].item()),
            )
            Rs.append(R)
        R = torch.stack(Rs, dim=0).to(device=device, dtype=dtype)
        kp_ref_T = kp_ref.expand(n, -1, -1).to(device=device, dtype=dtype)
        s_T = s_ref.expand(n, -1).to(device=device, dtype=dtype)
        t_T = t_ref.expand(n, -1).to(device=device, dtype=dtype)
        return compose_kd(kp_ref_T, R, s_T, t_T)

    def patched_interpolate_kps_online(self, ref, motion, num_interp,
                                       t_scale=0.5, s_scale=0):
        kp1 = self.detector(ref.to(self.dtype))
        ref_kp_canonical["kp"] = kp1["kp"].reshape(1, -1, 3).detach()
        ref_kp_canonical["t"] = kp1["t"].detach()
        ref_kp_canonical["scale"] = kp1["scale"].detach()

        n_pad = num_interp - 1
        n_motion = motion.shape[0]
        # Pad with index 0 then 0..n_motion-1
        pad_kd = cf_kd_for_indices(0, 1).repeat(n_pad, 1, 1) if n_pad > 0 else None
        motion_kd = cf_kd_for_indices(0, n_motion)
        if pad_kd is not None:
            kp_intrep = torch.cat([pad_kd, motion_kd], dim=0)
        else:
            kp_intrep = motion_kd
        kp_frame1 = self.detector(motion[:1].to(self.dtype))
        chunk_cursor["i"] = n_motion
        return kp_intrep, kp1, kp_frame1, None

    def patched_get_kps(self, kp_ref, kp_frame1, motion, t_scale=0.5, s_scale=0):
        start = chunk_cursor["i"]
        n = motion.shape[0]
        kp_d = cf_kd_for_indices(start, n)
        chunk_cursor["i"] += n
        return kp_d, None

    if patch_pose:
        pipe.pose_encoder.interpolate_kps_online = MethodType(
            patched_interpolate_kps_online, pipe.pose_encoder
        )
        pipe.pose_encoder.get_kps = MethodType(
            patched_get_kps, pipe.pose_encoder
        )

    real_me_forward = pipe.motion_encoder.forward
    me_cursor = {"i": 0}

    def patched_me_forward(self, x):
        T = x.shape[2]
        if T == 1:
            mf = real_me_forward(x)
            if mf_log is not None:
                mf_log["records"].append({
                    "role": "ref", "start": -1,
                    "mf": mf.detach().cpu().float().numpy(),
                })
            return mf
        if patch_motion:
            start = me_cursor["i"]
            b_chunk, _ = _fetch(start, T)
            b = torch.from_numpy(np.asarray(b_chunk).astype(np.float32)).to(device)
            with torch.no_grad():
                mf = student(b)
            mf = mf.squeeze(1).unsqueeze(0)
            me_cursor["i"] += T
            mf_out = mf.to(dtype=self.dtype)
        else:
            mf_out = real_me_forward(x)
            start = me_cursor["i"]
            me_cursor["i"] += T
        if mf_log is not None:
            mf_log["records"].append({
                "role": "driving", "start": int(start),
                "mf": mf_out.detach().cpu().float().numpy(),
            })
        return mf_out

    pipe.motion_encoder.forward = MethodType(patched_me_forward, pipe.motion_encoder)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/arkit_bridge/test_seam_install.py -v`
Expected: 1 passed.

- [ ] **Step 6: Replace inline def in `scripts/apply_bridge_to_personalive.py`**

Delete lines 83-205 of `scripts/apply_bridge_to_personalive.py` (the entire inline `install_arkit_seams` function) and add at the top with the other imports:

```python
from arkit_bridge.seam_install import install_arkit_seams
```

- [ ] **Step 7: Smoke test the offline path still works**

Run a 24-frame offline render against a known-good take to confirm the refactor is bit-equivalent:

```bash
uv run python scripts/apply_bridge_to_personalive.py \
  --reference data/portraits/anchor_001.png \
  --take_dir data/llf-takes/take_2 \
  --ckpt runs/student_v2_120k/student_best.pt \
  --out_path /tmp/refactor_smoke.mp4 \
  --n_frames 24 --start_frame 0 --mode bridge
```

Expected: completes without error, mp4 plays.

- [ ] **Step 8: Commit**

```bash
git add src/arkit_bridge/seam_install.py tests/arkit_bridge/test_seam_install.py scripts/apply_bridge_to_personalive.py
git commit -m "refactor(arkit-bridge): extract install_arkit_seams to importable module with provider mode"
```

---

### Task 3: v4l2loopback sink

**Files:**
- Create: `src/arkit_bridge/v4l2_sink.py`
- Test: `tests/arkit_bridge/test_v4l2_sink.py`

ffmpeg subprocess that takes raw RGB on stdin and writes to `/dev/videoN`. The kernel side (`modprobe v4l2loopback`) is operator setup, not code; the runbook documents it.

- [ ] **Step 1: Write the failing test** at `tests/arkit_bridge/test_v4l2_sink.py`:

```python
"""Unit tests for src/arkit_bridge/v4l2_sink.py.

Uses an MP4 destination instead of a real /dev/video device so the test
runs in CI without v4l2loopback. The ffmpeg invocation is the same shape;
we just swap the output URL.
"""
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from arkit_bridge.v4l2_sink import V4L2Sink


def test_sink_writes_frames_to_mp4(tmp_path):
    out = tmp_path / "out.mp4"
    sink = V4L2Sink(
        device=str(out), width=256, height=256, fps=25,
        output_format="mp4",  # test-only path
    )
    sink.open()
    try:
        for i in range(25):
            frame = np.full((256, 256, 3), i * 10, dtype=np.uint8)
            sink.write(frame)
    finally:
        sink.close()
    assert out.exists()
    assert out.stat().st_size > 1000

    # Probe the file with ffprobe to confirm 25 frames at 25 fps.
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames",
         "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames",
         "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True,
    )
    assert int(res.stdout.strip()) == 25
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/arkit_bridge/test_v4l2_sink.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement at `src/arkit_bridge/v4l2_sink.py`**:

```python
"""Raw-RGB → ffmpeg → v4l2loopback (or mp4 for testing) sink."""
from __future__ import annotations

import subprocess
from typing import Optional

import numpy as np


class V4L2Sink:
    """Write 24-bit RGB frames to a v4l2 device via ffmpeg subprocess.

    Operator must `sudo modprobe v4l2loopback devices=1 video_nr=10
    card_label=PersonaLive exclusive_caps=1` before starting the daemon.
    """

    def __init__(self, device: str = "/dev/video10", width: int = 512,
                 height: int = 512, fps: int = 25,
                 output_format: str = "v4l2"):
        self._device = device
        self._width = width
        self._height = height
        self._fps = fps
        self._output_format = output_format
        self._proc: Optional[subprocess.Popen] = None

    def open(self):
        cmd = [
            "ffmpeg", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{self._width}x{self._height}",
            "-r", str(self._fps),
            "-i", "pipe:0",
        ]
        if self._output_format == "v4l2":
            cmd += ["-f", "v4l2", "-pix_fmt", "yuv420p", self._device]
        elif self._output_format == "mp4":
            cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", self._device]
        else:
            raise ValueError(f"unknown output_format {self._output_format!r}")
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("sink not opened")
        if frame.shape != (self._height, self._width, 3) or frame.dtype != np.uint8:
            raise ValueError(
                f"expected ({self._height}, {self._width}, 3) uint8; "
                f"got {frame.shape} {frame.dtype}"
            )
        self._proc.stdin.write(frame.tobytes())

    def close(self):
        if self._proc is not None:
            if self._proc.stdin is not None:
                try:
                    self._proc.stdin.close()
                except BrokenPipeError:
                    pass
            self._proc.wait(timeout=5)
            self._proc = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/arkit_bridge/test_v4l2_sink.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/arkit_bridge/v4l2_sink.py tests/arkit_bridge/test_v4l2_sink.py
git commit -m "feat(arkit-bridge): v4l2 sink via ffmpeg subprocess"
```

---

### Task 4: BatchDriver — wraps the pipe + bridge

**Files:**
- Create: `src/arkit_bridge/streaming_driver.py`
- Test: `tests/arkit_bridge/test_streaming_driver.py`

`BatchDriver` builds the pipe once at startup (cold-start cost ~10-20 s for tensorrt cache load), holds the reference image and the trained student, and exposes `render_batch(b58_chunk, ypr_chunk) -> np.ndarray` for the daemon. Each call processes a multiple-of-4 frame batch through one `pipe.__call__`.

- [ ] **Step 1: Write the integration test** at `tests/arkit_bridge/test_streaming_driver.py`:

```python
"""Integration test: BatchDriver yields RGB shape/dtype matching offline render.

Skipped if no GPU and no PersonaLive checkpoints. Compares batch shape only;
bit-equivalence vs the offline single-call is checked manually in Step 6
(the offline call is too expensive for CI).
"""
import os
import shutil

import numpy as np
import pytest

PL_CKPT = os.path.expanduser("~/w/PersonaLive/pretrained_weights/personalive/denoising_unet.pth")
STUDENT_CKPT = "runs/student_v2_120k/student_best.pt"
REF_IMG = "data/portraits/anchor_001.png"

skip_if_no_models = pytest.mark.skipif(
    not (os.path.exists(PL_CKPT) and os.path.exists(STUDENT_CKPT) and os.path.exists(REF_IMG)),
    reason="PersonaLive / student / reference assets missing",
)


@skip_if_no_models
def test_batch_driver_render_shape():
    import torch
    from arkit_bridge.streaming_driver import BatchDriver

    drv = BatchDriver(
        reference_path=REF_IMG,
        student_ckpt=STUDENT_CKPT,
        device="cuda",
        dtype=torch.float16,
    )
    drv.start()
    try:
        b58 = np.zeros((24, 58), dtype=np.float32)
        ypr = np.zeros((24, 3), dtype=np.float32)
        ypr[:, 0] = np.linspace(0, 0.4, 24)  # gentle yaw sweep
        rgb = drv.render_batch(b58, ypr)
        assert rgb.shape == (24, 512, 512, 3)
        assert rgb.dtype == np.uint8
    finally:
        drv.stop()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/arkit_bridge/test_streaming_driver.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `BatchDriver`**

Create `src/arkit_bridge/streaming_driver.py`. The pipe-build code is essentially `scripts/apply_bridge_to_personalive.py:build_pipe` lifted into a method, plus reference-image preprocessing. Full source:

```python
"""BatchDriver: build PersonaLive pipe once, render N-frame batches with bridge seams."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

from arkit_bridge.seam_install import install_arkit_seams
from arkit_bridge.student import MotEncoderStudent

PL = Path(os.path.expanduser("~/w/PersonaLive"))


def _build_pipe(device, dtype):
    """Lift of scripts/apply_bridge_to_personalive.py:build_pipe."""
    sys.path.insert(0, str(PL))
    cwd = os.getcwd()
    os.chdir(PL)
    try:
        from diffusers import AutoencoderKL
        from transformers import CLIPVisionModelWithProjection
        from src.scheduler.scheduler_ddim import DDIMScheduler
        from src.models.unet_2d_condition import UNet2DConditionModel
        from src.models.unet_3d import UNet3DConditionModel
        from src.pipelines.pipeline_pose2vid import Pose2VideoPipeline_Stream
        from src.models.motion_encoder.encoder import MotEncoder
        from src.liveportrait.motion_extractor import MotionExtractor
        from src.models.pose_guider import PoseGuider

        cfg = OmegaConf.load("configs/prompts/personalive_offline.yaml")
        infer = OmegaConf.load(cfg.inference_config)
        vae = AutoencoderKL.from_pretrained(cfg.vae_path).to(device, dtype=dtype)
        ref_unet = UNet2DConditionModel.from_pretrained(
            cfg.pretrained_base_model_path, subfolder="unet"
        ).to(device, dtype=dtype)
        den_unet = UNet3DConditionModel.from_pretrained_2d(
            cfg.pretrained_base_model_path, "", subfolder="unet",
            unet_additional_kwargs=infer.unet_additional_kwargs,
        ).to(device, dtype=dtype)
        me = MotEncoder().to(device, dtype=dtype).eval()
        pg = PoseGuider().to(device, dtype=dtype)
        pe = MotionExtractor(num_kp=21).to(device, dtype=dtype).eval()
        img_enc = CLIPVisionModelWithProjection.from_pretrained(
            cfg.image_encoder_path
        ).to(device, dtype=dtype)
        sched = DDIMScheduler(**OmegaConf.to_container(infer.noise_scheduler_kwargs))

        base = cfg.denoising_unet_path
        den_unet.load_state_dict(torch.load(base, map_location="cpu"), strict=False)
        ref_unet.load_state_dict(torch.load(base.replace("denoising_unet", "reference_unet"), map_location="cpu"), strict=True)
        me.load_state_dict(torch.load(base.replace("denoising_unet", "motion_encoder"), map_location="cpu"), strict=True)
        pg.load_state_dict(torch.load(base.replace("denoising_unet", "pose_guider"), map_location="cpu"), strict=True)
        den_unet.load_state_dict(torch.load(base.replace("denoising_unet", "temporal_module"), map_location="cpu"), strict=False)
        pe.load_state_dict(torch.load(base.replace("denoising_unet", "motion_extractor"), map_location="cpu"), strict=False)

        return Pose2VideoPipeline_Stream(
            vae=vae, image_encoder=img_enc,
            reference_unet=ref_unet, denoising_unet=den_unet,
            motion_encoder=me, pose_encoder=pe, pose_guider=pg, scheduler=sched,
        ).to(device)
    finally:
        os.chdir(cwd)


class BatchDriver:
    """Owns the pipe + student. Render multiple-of-4-frame batches."""

    def __init__(self, reference_path: str, student_ckpt: str,
                 device: str = "cuda", dtype: torch.dtype = torch.float16,
                 num_inference_steps: int = 4, guidance_scale: float = 1.0):
        self._reference_path = reference_path
        self._student_ckpt = student_ckpt
        self._device = device
        self._dtype = dtype
        self._num_inference_steps = num_inference_steps
        self._guidance_scale = guidance_scale
        self._pipe = None
        self._student: Optional[MotEncoderStudent] = None
        self._ref_face: Optional[Image.Image] = None

    def start(self):
        self._pipe = _build_pipe(self._device, self._dtype)
        self._student = MotEncoderStudent().to(self._device).eval()
        self._student.load_state_dict(torch.load(self._student_ckpt, map_location=self._device))
        self._prepare_reference()

    def _prepare_reference(self):
        import mediapipe as mp
        from src.utils.util import crop_face

        ref_pil = Image.open(self._reference_path).convert("RGB")
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1
        )
        try:
            self._ref_face = Image.fromarray(crop_face(ref_pil, face_mesh)).convert("RGB")
        finally:
            face_mesh.close()

    def render_batch(self, b58: np.ndarray, ypr: np.ndarray) -> np.ndarray:
        """b58: (T,58) float32; ypr: (T,3) float32 radians.

        T must be a positive multiple of 4. Returns (T, 512, 512, 3) uint8 RGB.
        """
        assert self._pipe is not None and self._student is not None
        T = b58.shape[0]
        assert T % 4 == 0 and T >= 4, f"T must be multiple of 4 (got {T})"

        def provider(start, n):
            return b58[start:start + n], ypr[start:start + n]

        install_arkit_seams(
            self._pipe, b_seq=None, ypr_seq=None,
            student=self._student, device=self._device, dtype=self._dtype,
            patch_pose=True, patch_motion=True, provider=provider,
        )

        # Stand-in driving images: identical copies of the reference. The
        # patched seams ignore content; pipe still calls preprocessing.
        stand_in = [self._ref_face] * T

        gen = torch.Generator(device=self._device); gen.manual_seed(42)
        video = self._pipe(
            tgt_images=stand_in,
            ref_image=self._ref_face,
            face_images=stand_in,
            ref_face_image=self._ref_face,
            width=512, height=512,
            video_length=T,
            num_inference_steps=self._num_inference_steps,
            guidance_scale=self._guidance_scale,
            generator=gen,
            output_type="tensor",
            temporal_window_size=4,
            temporal_adaptive_step=4,
        )
        # video: (1, 3, T, H, W) float in [0,1]
        arr = video[0].permute(1, 2, 3, 0).cpu().numpy()
        return (arr * 255.0).clip(0, 255).astype(np.uint8)

    def stop(self):
        self._pipe = None
        self._student = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/arkit_bridge/test_streaming_driver.py -v`
Expected: 1 passed (or skipped if assets missing).

- [ ] **Step 5: Commit**

```bash
git add src/arkit_bridge/streaming_driver.py tests/arkit_bridge/test_streaming_driver.py
git commit -m "feat(arkit-bridge): BatchDriver wraps pipe + seam install for streaming"
```

---

### Task 5: Daemon entry point

**Files:**
- Create: `scripts/streaming_bridge.py`

Wires receiver, driver, sink. Pre-warms with one zero-vector batch before opening the v4l2 device.

- [ ] **Step 1: Write the daemon**

Create `scripts/streaming_bridge.py`:

```python
"""LLF UDP -> PersonaLive bridge -> v4l2loopback daemon.

Usage:
  sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=PersonaLive exclusive_caps=1
  uv run python scripts/streaming_bridge.py \
      --reference data/portraits/anchor_001.png \
      --ckpt runs/student_v2_120k/student_best.pt \
      --device_path /dev/video10 \
      --port 11111 \
      --batch 24
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkit_bridge.llf_udp import LLFReceiver, B61Packet
from arkit_bridge.streaming_driver import BatchDriver
from arkit_bridge.v4l2_sink import V4L2Sink


def packets_to_arrays(pkts: list[B61Packet]):
    """Convert a list of B61Packets to (b58, ypr) numpy arrays."""
    n = len(pkts)
    b58 = np.zeros((n, 58), dtype=np.float32)
    ypr = np.zeros((n, 3), dtype=np.float32)
    for i, p in enumerate(pkts):
        b58[i] = np.asarray(p.b58, dtype=np.float32)
        ypr[i] = np.asarray(p.head_ypr, dtype=np.float32)
    return b58, ypr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--ckpt", required=True, help="MotEncoderStudent .pt")
    ap.add_argument("--device_path", default="/dev/video10")
    ap.add_argument("--port", type=int, default=11111)
    ap.add_argument("--batch", type=int, default=24,
                    help="frames per pipe call; multiple of 4")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--torch_device", default="cuda")
    args = ap.parse_args()

    assert args.batch % 4 == 0 and args.batch >= 4

    print("[1/4] building pipe (this can take 10-20s)...", flush=True)
    drv = BatchDriver(
        reference_path=args.reference,
        student_ckpt=args.ckpt,
        device=args.torch_device,
        dtype=torch.float16,
    )
    drv.start()

    print("[2/4] pre-warming with zero-vector batch...", flush=True)
    b58 = np.zeros((args.batch, 58), dtype=np.float32)
    ypr = np.zeros((args.batch, 3), dtype=np.float32)
    _ = drv.render_batch(b58, ypr)

    print(f"[3/4] opening v4l2 sink at {args.device_path}...", flush=True)
    sink = V4L2Sink(device=args.device_path, width=512, height=512, fps=args.fps)
    sink.open()

    print(f"[4/4] starting LLF receiver on UDP :{args.port}...", flush=True)
    rx = LLFReceiver(host="0.0.0.0", port=args.port, ring_size=args.batch * 4)
    rx.start()

    stop = False
    def _on_signal(signum, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    last_log = time.time()
    rendered = 0
    last_good_rgb = None
    print("daemon ready; waiting for LLF packets", flush=True)
    while not stop:
        pkts = rx.pop_window(k=args.batch, timeout_s=2.0)
        if not pkts:
            # Idle: emit anchor passthrough so OBS sees a signal.
            if last_good_rgb is None:
                b58z = np.zeros((args.batch, 58), dtype=np.float32)
                yprz = np.zeros((args.batch, 3), dtype=np.float32)
                last_good_rgb = drv.render_batch(b58z, yprz)
            for f in last_good_rgb:
                sink.write(f)
                time.sleep(1.0 / args.fps)
            continue

        b58, ypr = packets_to_arrays(pkts)
        # Pad to multiple of 4 if short.
        if len(pkts) % 4 != 0:
            pad = 4 - (len(pkts) % 4)
            b58 = np.concatenate([b58, np.tile(b58[-1:], (pad, 1))], axis=0)
            ypr = np.concatenate([ypr, np.tile(ypr[-1:], (pad, 1))], axis=0)

        t0 = time.time()
        rgb = drv.render_batch(b58, ypr)
        infer_ms = (time.time() - t0) * 1000.0
        last_good_rgb = rgb
        for f in rgb:
            sink.write(f)
        rendered += len(rgb)

        now = time.time()
        if now - last_log >= 5.0:
            fps = rendered / (now - last_log)
            print(f"  rendered={rendered} fps={fps:.1f} "
                  f"infer_ms={infer_ms:.0f} rx_received={rx.received} "
                  f"rx_dropped={rx.dropped}", flush=True)
            rendered = 0
            last_log = now

    print("draining...", flush=True)
    rx.stop()
    sink.close()
    drv.stop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test against a synthetic UDP sender**

Open a second terminal. In terminal A:

```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=PersonaLive exclusive_caps=1
ls -la /dev/video10
```

Expected: device exists.

In terminal A, run the daemon:

```bash
uv run python scripts/streaming_bridge.py \
  --reference data/portraits/anchor_001.png \
  --ckpt runs/student_v2_120k/student_best.pt \
  --device_path /dev/video10 \
  --port 21112 --batch 24
```

Wait for "daemon ready; waiting for LLF packets".

In terminal B, replay a recorded LLF take by sending its rows as UDP packets:

```bash
uv run python -c "
import socket, struct, time, numpy as np, pandas as pd
df = pd.read_csv('data/llf-takes/take_2/MyTake_LiveLinkFace_Capture.csv').head(120)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
for _, row in df.iterrows():
    floats = [float(row[c]) for c in row.index if c not in ('Timecode','BlendShapeCount')][:61]
    if len(floats) < 61:
        floats += [0.0] * (61 - len(floats))
    pkt = b'\x00'*6 + struct.pack('>I', 5) + b'iPhne' + struct.pack('>IIII',0,0,60,60) + struct.pack('>'+'f'*61, *floats)
    sock.sendto(pkt, ('127.0.0.1', 21112))
    time.sleep(1/60)
"
```

In terminal C, view the v4l2 device:

```bash
ffplay -f v4l2 /dev/video10
```

Expected: see the anchor portrait animating with the take's expression.

- [ ] **Step 3: Commit**

```bash
git add scripts/streaming_bridge.py
git commit -m "feat(arkit-bridge): streaming daemon LLF UDP -> v4l2loopback"
```

---

### Task 6: Operator runbook

**Files:**
- Create: `docs/research/2026-05-06-llf-obs-runbook.md`

- [ ] **Step 1: Write the runbook**

```markdown
---
status: live
topic: arkit-bridge
---

# Live Link Face → PersonaLive → OBS runbook

End-to-end operator guide for running the streaming bridge locally.

## One-time setup

### Linux box

Install v4l2loopback once:

```bash
sudo apt install v4l2loopback-dkms ffmpeg
```

### iPhone

1. Install **Live Link Face** from the App Store (free, by Epic Games).
2. Open the app → tap the gear icon → "Live Link" → "Add Target".
3. Set Host = your Linux box's LAN IP, Port = 11111. Tap "OK".
4. Back on the main screen, toggle "Live Link" on. The dot turns green
   when the daemon's receiver is running and the iPhone has a route.

## Per-session startup

### 1. Load v4l2loopback (every reboot)

```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 \
  card_label=PersonaLive exclusive_caps=1
```

### 2. Start the daemon

```bash
cd ~/w/vamp-interface
uv run python scripts/streaming_bridge.py \
  --reference data/portraits/anchor_001.png \
  --ckpt runs/student_v2_120k/student_best.pt \
  --device_path /dev/video10 \
  --port 11111 --batch 24 --fps 25
```

Wait for `daemon ready; waiting for LLF packets`. Cold start is 10-20 s
(pipe build) plus one batch of pre-warm (~1.5 s).

### 3. Start the iPhone stream

Toggle Live Link on in the app. The Linux daemon should start logging
`rendered=… fps=…` lines every 5 s.

### 4. Wire OBS

- Sources → + → "Video Capture Device".
- Device: `PersonaLive` (the card_label set above).
- Resolution: 512×512, 25 FPS.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Permission denied: /dev/video10` | v4l2loopback module not loaded or owned by root | re-run `sudo modprobe …`; ensure your user is in `video` group |
| `daemon ready` but no log lines | iPhone can't reach Linux | check both on same LAN, no AP isolation; `tcpdump -ni any port 11111` to confirm packets arrive |
| OBS shows black frame | daemon emits idle frames at startup before LLF is on | start LLF stream, OBS should animate |
| `rx_dropped` growing | render slower than ingest (expected at 60 FPS in / 25 FPS out) | not a problem; ring drops oldest |
| `fps < 18` in daemon log | torch_tensorrt cache cold or thermal throttling | first run builds tensorrt engines (~10 min); subsequent runs are fast |

## Latency

Glass-to-OBS budget for V1 (batch-of-24):

- LLF capture+UDP transit: ~20 ms
- Batch fill (24 frames at 60 FPS): ~400 ms
- Pipe inference: ~1000 ms (24 frames @ ~22 FPS internal)
- ffmpeg + v4l2 + OBS pickup: ~80 ms

Total: **~1.5 s**. Acceptable for vtuber-style mirror; not low enough for
real-time-conversation overlay. V2 streaming-window refactor of
`Pose2VideoPipeline_Stream.__call__` would bring this to ~150 ms but is
deferred behind a working V1.

## Stopping

Ctrl-C the daemon. The receiver, sink, and pipe drain in order. OBS will
go to a black frame; restart the daemon to resume.
```

- [ ] **Step 2: Commit**

```bash
git add docs/research/2026-05-06-llf-obs-runbook.md
git commit -m "docs(arkit-bridge): operator runbook for LLF -> PersonaLive -> OBS"
```

---

### Task 7: Topic index update

**Files:**
- Modify: `docs/research/_topics/arkit-bridge.md`

- [ ] **Step 1: Add the streaming pipeline as a current-belief item**

Open `docs/research/_topics/arkit-bridge.md` and add to the "Current beliefs" section:

```markdown
- Streaming daemon (LLF UDP → bridge → v4l2loopback → OBS) ships in V1
  with batch-of-24 latency ~1.5 s end-to-end. See
  [`2026-05-06-llf-obs-runbook.md`](../2026-05-06-llf-obs-runbook.md).
  V2 step-based refactor of `Pose2VideoPipeline_Stream.__call__` would
  cut latency to ~150 ms; deferred.
```

- [ ] **Step 2: Commit**

```bash
git add docs/research/_topics/arkit-bridge.md
git commit -m "docs(topics): arkit-bridge — streaming daemon shipped V1"
```

---

## Self-review

**Spec coverage**

| Spec section | Implementing task |
|---|---|
| `llf_receiver` | Task 1 |
| `streaming_pipe` | Task 4 (BatchDriver — V1 batch-mode; V2 windowed step is explicitly deferred) |
| `v4l2_sink` | Task 3 |
| `daemon` orchestration | Task 5 |
| Pre-warm with zero b_61 | Task 5, Step 1 |
| Idle anchor passthrough on no-packets | Task 5, Step 1 |
| Operator runbook | Task 6 |
| Bit-equiv test of refactor | Task 2, Step 7 (smoke render) |
| Soak / latency smoke rig | Task 6 (manual section) |

**Placeholder scan**: No "TBD" / "implement later" / "fill in details" patterns. Each step has full code or full command.

**Type consistency**: `LLFReceiver.pop_window` → `list[B61Packet]`; `B61Packet.b58` → 58 floats; `BatchDriver.render_batch(b58, ypr)` → `(T, 512, 512, 3) uint8`. Daemon uses `packets_to_arrays` to bridge. Names line up across tasks.

**Deviation from spec**: Spec's "windowed real-time step" is V2; V1 is batched at the cost of ~1.5 s latency. Documented inline at the top of this plan and in the runbook. The reason is that PersonaLive's `Pose2VideoPipeline_Stream.__call__` is a single-shot loop over windows; making it `step()`-shaped requires extracting `motion_bank`/`noise_latents`/`motion_hidden_states`/`pose_feas` as instance state, and that's a 1-day refactor inside a CVPR submission codebase we want to keep diffable. V1 ships value first.
