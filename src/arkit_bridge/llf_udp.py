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
