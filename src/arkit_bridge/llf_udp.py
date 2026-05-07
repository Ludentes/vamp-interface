"""Live Link Face UDP receiver and packet decoder.

Wire format (Epic's Live Link Face iOS app — see
`docs/research/2026-05-07-llf-udp-protocol-verification.md`):

  offset       size  type           field
  -------      ----  -------------  ----------------------------------
  0            4     LE uint32      version
  4            37    bytes          UUID (ASCII string starting with `$`)
  41           4     BE int32       subject-name byte length L
  45           L     ASCII          subject name
  45+L         4     BE uint32      frame
  49+L         4     BE float32     sub_frame    (NB: float, not uint)
  53+L         4     BE uint32      fps
  57+L         4     BE uint32      denom
  61+L         1     BE uint8       data_length  (always 61)
  62+L         244   61× BE float32 payload

  total: 306 + L bytes

Payload (61 floats):
  [0:52]   ARKit-52 blendshapes in ARFaceAnchor.BlendShapeLocation order
  [52:55]  HeadYaw, HeadPitch, HeadRoll          (radians)
  [55:58]  LeftEyeYaw,  LeftEyePitch,  LeftEyeRoll  (radians)
  [58:61]  RightEyeYaw, RightEyePitch, RightEyeRoll (radians)

Hazards (don't repeat these):
  - Unity's `Unity.LiveCapture.ARKitFaceCapture.FaceBlendShape` enum is
    alphabetical and does NOT match the wire-format order. Reference
    `scripts/livelink_probe.py:ARKIT_52` for the canonical list.
  - sub_frame is float32, not uint32.
  - The fps/denom order is fps-then-denom on the wire (Epic mixes LE
    uint32 version + BE everything else; we match the wire layout).
  - Head/eye Euler sign convention is undocumented in Epic's sources.
    Empirical: `EULER_SIGNS=(+1,-1,+1)` with `F_KP_REF=I` lands the
    PersonaLive renderer correctly (see
    `docs/research/2026-05-06-arkit-bridge-final-pose-config.md`).
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

# Length of the fixed header preamble that precedes the variable-length
# subject name: 4-byte LE version + 37-byte UUID + 4-byte BE name-length.
_NAME_LEN_OFFSET = 4 + 37  # = 41
_NAME_OFFSET = _NAME_LEN_OFFSET + 4  # = 45
# After the name come 4×u32 + f32 = 16 bytes of frame metadata + 1 byte
# data_length, then the payload.
_FRAME_META_BYTES = 4 + 4 + 4 + 4 + 1  # = 17

_MIN_HEADER_BYTES = _NAME_OFFSET + _FRAME_META_BYTES  # 62, name length zero


def decode_packet(data: bytes) -> tuple[Optional[str], Optional[tuple[float, ...]]]:
    """Decode one UDP datagram. Returns (subject, floats[61]) or (None, None).

    Floats are read from the trailing 244 bytes — robust to any
    upstream header drift. Subject is parsed from the BE int32 length
    at offset 41 followed by ASCII bytes; falls back to "?" on any
    inconsistency rather than failing.
    """
    if len(data) < _MIN_HEADER_BYTES + TAIL_BYTES:
        return None, None
    floats = struct.unpack(">" + "f" * N_FLOATS, data[-TAIL_BYTES:])
    subject = "?"
    try:
        (name_len,) = struct.unpack(">i", data[_NAME_LEN_OFFSET:_NAME_LEN_OFFSET + 4])
        if 0 < name_len < 64 and _NAME_OFFSET + name_len <= len(data) - TAIL_BYTES:
            subject = data[_NAME_OFFSET:_NAME_OFFSET + name_len].decode(
                "ascii", errors="replace"
            )
    except Exception:
        pass
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
            pkt = B61Packet(
                subject=subject if subject is not None else "?",
                floats=floats,
                recv_time=time.time(),
            )
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
