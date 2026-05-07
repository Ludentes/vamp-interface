"""Unit tests for src/arkit_bridge/llf_udp.py."""
import socket
import struct
import time

import pytest

from arkit_bridge.llf_udp import (  # noqa: F401  (LLFReceiver re-exported for harness)
    decode_packet,
    LLFReceiver,
    N_FLOATS,
)


def _build_packet(subject="iPhone", floats=None,
                  frame=0, sub_frame=0.0, fps=60, denom=1):
    """Build a synthetic Live Link Face UDP packet that matches the
    real wire format (per
    `docs/research/2026-05-07-llf-udp-protocol-verification.md`).

    Layout:
      4   LE uint32 version
      37  bytes UUID
      4   BE int32 subject-name length L
      L   ASCII name
      4   BE uint32 frame
      4   BE float32 sub_frame
      4   BE uint32 fps
      4   BE uint32 denom
      1   BE uint8 data_length (61)
      244 61× BE float32 payload
    """
    if floats is None:
        floats = [0.0] * N_FLOATS
    version = struct.pack("<I", 1)            # LE u32
    uuid = b"$" + b"0" * 36                   # 37 bytes, must start with `$`
    name = subject.encode("ascii")
    name_block = struct.pack(">i", len(name)) + name
    meta = (struct.pack(">I", frame)
            + struct.pack(">f", sub_frame)
            + struct.pack(">I", fps)
            + struct.pack(">I", denom)
            + struct.pack(">B", N_FLOATS))
    payload = struct.pack(">" + "f" * N_FLOATS, *floats)
    return version + uuid + name_block + meta + payload


def test_decode_packet_roundtrip():
    pkt = _build_packet(subject="vamp", floats=[i * 0.01 for i in range(N_FLOATS)])
    subject, floats = decode_packet(pkt)
    assert subject == "vamp"
    assert floats is not None
    assert floats[0] == pytest.approx(0.0)
    assert floats[60] == pytest.approx(0.60)


def test_decode_packet_too_short():
    subject, floats = decode_packet(b"\x00" * 10)
    assert subject is None and floats is None


def test_decode_packet_subject_fallback_on_garbage_header():
    """A packet whose name-length field is implausible should still
    yield the float payload — we never want to drop valid floats over
    a metadata wobble — but the subject falls back to '?'."""
    pkt = _build_packet(floats=[0.5] * N_FLOATS)
    # smash the name-length field at offset 41
    bad = bytearray(pkt)
    bad[41:45] = b"\xff\xff\xff\xff"
    subject, floats = decode_packet(bytes(bad))
    assert subject == "?"
    assert floats is not None and floats[0] == pytest.approx(0.5)


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
        # subject decoded correctly through the full pipe
        assert all(p.subject == "iPhone" for p in pkts)
    finally:
        rx.stop()
