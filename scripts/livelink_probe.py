"""Live Link Face UDP receiver probe.

Listens for packets from Epic's Live Link Face iOS app and decodes the
ARKit-52 + head + eye rotation float stream. Use this to verify the iPhone
can reach this Linux box and that the stream is parseable before wiring it
into PersonaLive.

iPhone setup:
  Live Link Face -> Settings -> Live Link -> Add Target
  Host = <this machine's LAN IP>, Port = 11111 (default)
  Back to main, toggle "Live Link" on (LED dot turns green when connected).

Run:
  python scripts/livelink_probe.py            # listens on 0.0.0.0:11111
  python scripts/livelink_probe.py --port 11111 --duration 10

Protocol (PyLiveLinkFace / JimWest reference):
  Each UDP datagram is one frame. Layout, big-endian:
    [0:6]    six-byte version/uuid prefix (varies; ignore for probe)
    [6:6+N]  subject-name block: 4-byte length L then L bytes ASCII
    then     frame metadata: 4 ints (frame, subframe, denom, fps) = 16 bytes
    then     61 float32 big-endian values:
               0..51  ARKit-52 blendshapes (eyeBlinkLeft, eyeLookDownLeft, ...)
               52..60 head yaw/pitch/roll + leftEye yaw/pitch/roll + rightEye yaw/pitch/roll
  We locate the float block by scanning from the end: 61 * 4 = 244 bytes.
"""

import argparse
import socket
import struct
import time

ARKIT_52 = [
    "eyeBlinkLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft",
    "eyeLookUpLeft", "eyeSquintLeft", "eyeWideLeft",
    "eyeBlinkRight", "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight",
    "eyeLookUpRight", "eyeSquintRight", "eyeWideRight",
    "jawForward", "jawLeft", "jawRight", "jawOpen",
    "mouthClose", "mouthFunnel", "mouthPucker", "mouthLeft", "mouthRight",
    "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthPressLeft", "mouthPressRight", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "noseSneerLeft", "noseSneerRight", "tongueOut",
]
TAIL_NAMES = [
    "headYaw", "headPitch", "headRoll",
    "leftEyeYaw", "leftEyePitch", "leftEyeRoll",
    "rightEyeYaw", "rightEyePitch", "rightEyeRoll",
]

N_FLOATS = 61
TAIL_BYTES = N_FLOATS * 4


def decode_packet(data: bytes):
    """Return (subject, floats[61]) or (None, None) if too short / malformed."""
    if len(data) < TAIL_BYTES + 8:
        return None, None
    floats = struct.unpack(">" + "f" * N_FLOATS, data[-TAIL_BYTES:])
    # Subject name: skip 6-byte prefix, read 4-byte length, then ASCII.
    try:
        name_len = struct.unpack(">I", data[6:10])[0]
        if 0 < name_len < 64 and 10 + name_len <= len(data):
            subject = data[10:10 + name_len].decode("ascii", errors="replace")
        else:
            subject = "?"
    except Exception:
        subject = "?"
    return subject, floats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=11111)
    ap.add_argument("--duration", type=float, default=0.0,
                    help="seconds (0 = forever)")
    ap.add_argument("--print-every", type=int, default=30,
                    help="print one decoded frame every N packets")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))
    sock.settimeout(1.0)

    print(f"listening on {args.host}:{args.port}")
    print("point Live Link Face at this host:port and toggle Live Link on")
    print("ctrl-c to stop")

    t0 = time.time()
    n = 0
    last_print_t = t0
    last_floats = None
    while True:
        if args.duration and (time.time() - t0) > args.duration:
            break
        try:
            data, addr = sock.recvfrom(2048)
        except socket.timeout:
            print(f"  [no packets in last 1.0s, total={n}]")
            continue
        n += 1
        subject, floats = decode_packet(data)
        last_floats = floats
        if n == 1 or (n % args.print_every == 0):
            now = time.time()
            rate = args.print_every / max(1e-6, now - last_print_t) if n > 1 else 0
            last_print_t = now
            print(f"\n[#{n} from {addr[0]} subj={subject} len={len(data)}B "
                  f"~{rate:.1f} pkt/s]")
            if floats is not None:
                top_bs = sorted(
                    [(ARKIT_52[i], floats[i]) for i in range(52)],
                    key=lambda kv: -abs(kv[1])
                )[:6]
                print("  top blendshapes:", ", ".join(
                    f"{k}={v:+.2f}" for k, v in top_bs))
                tail = floats[52:61]
                print("  head/eye:", ", ".join(
                    f"{TAIL_NAMES[i]}={tail[i]:+.2f}" for i in range(9)))

    print(f"\ntotal packets: {n}, elapsed: {time.time() - t0:.1f}s")
    if last_floats is not None:
        nz = sum(1 for v in last_floats[:52] if abs(v) > 0.01)
        print(f"last frame: {nz}/52 blendshapes non-zero")


if __name__ == "__main__":
    main()
