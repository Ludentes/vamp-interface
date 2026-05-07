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
