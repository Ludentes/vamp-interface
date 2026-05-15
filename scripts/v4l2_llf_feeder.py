"""v4l2 LLF feeder — receives Live Link Face UDP, displays values as text.

Validates the LLF receive path independent of PersonaLive. Shows:
- jawOpen (ARKit b52 idx 17) as text + horizontal bar
- mouthSmileLeft (idx 23) as text + bar
- head yaw/pitch/roll (degrees)
- packet counter, freshness (ms since last packet)

Renders at fixed FPS via PacedSinkWriter so OBS sees a steady stream
even if iPhone toggles off briefly. If no LLF packet has arrived yet,
shows "waiting for packets..." overlay.
"""
from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from arkit_bridge.llf_udp import LLFReceiver  # noqa: E402
from arkit_bridge.v4l2_sink import V4L2Sink  # noqa: E402
from streaming_bridge import PacedSinkWriter  # noqa: E402


# ARKit b52 indices (alphabetical order Apple ships).
IDX_JAW_OPEN = 17
IDX_MOUTH_SMILE_L = 23
IDX_MOUTH_SMILE_R = 24
IDX_EYE_BLINK_L = 0
IDX_EYE_BLINK_R = 7


def _load_font(size: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_overlay(base_arr, pkt, n_received, n_dropped, age_ms, frame_idx,
                   font_big, font_small):
    img = Image.fromarray(base_arr.copy())
    draw = ImageDraw.Draw(img)

    # Animated heartbeat dot proves the renderer is alive even when iPhone
    # is silent. Same orbit as burst feeder — 1 rev / 24 frames.
    ang = 2 * math.pi * frame_idx / 24
    cx = 460 + int(20 * math.cos(ang))
    cy = 30 + int(20 * math.sin(ang))
    draw.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), fill=(0, 255, 0))

    if pkt is None:
        draw.text((20, 220), "waiting for LLF packets...",
                  fill=(255, 200, 0), font=font_big)
        draw.text((20, 260), f"frame={frame_idx}",
                  fill=(180, 180, 180), font=font_small)
        return np.asarray(img, dtype=np.uint8)

    bs = pkt.b_expr
    yaw, pitch, roll = pkt.head_ypr
    jaw = bs[IDX_JAW_OPEN]
    smile_l = bs[IDX_MOUTH_SMILE_L]
    smile_r = bs[IDX_MOUTH_SMILE_R]
    blink_l = bs[IDX_EYE_BLINK_L]
    blink_r = bs[IDX_EYE_BLINK_R]

    # Header
    fresh_color = (0, 255, 0) if age_ms < 100 else (
        (255, 200, 0) if age_ms < 500 else (255, 80, 80))
    draw.text((20, 12),
              f"LLF rx={n_received} drop={n_dropped} age={age_ms:>4d}ms "
              f"subj={pkt.subject}",
              fill=fresh_color, font=font_small)

    # Blendshape bars
    def bar(y, label, val):
        draw.text((20, y), f"{label:>14s}", fill=(220, 220, 220),
                  font=font_small)
        x0 = 200
        bw = 280
        draw.rectangle((x0, y + 4, x0 + bw, y + 24),
                       outline=(120, 120, 120), width=1)
        v = max(0.0, min(1.0, float(val)))
        draw.rectangle((x0 + 2, y + 6, x0 + 2 + int((bw - 4) * v), y + 22),
                       fill=(80, 200, 255))
        draw.text((x0 + bw + 8, y), f"{val:+.3f}",
                  fill=(255, 255, 0), font=font_small)

    bar(60, "jawOpen", jaw)
    bar(95, "mouthSmile_L", smile_l)
    bar(130, "mouthSmile_R", smile_r)
    bar(165, "eyeBlink_L", blink_l)
    bar(200, "eyeBlink_R", blink_r)

    # Head pose
    draw.text((20, 250), "head pose (deg):",
              fill=(220, 220, 220), font=font_small)
    deg = lambda r: math.degrees(r)
    draw.text((20, 280),
              f"yaw  ={deg(yaw):+7.2f}",
              fill=(255, 255, 0), font=font_big)
    draw.text((20, 315),
              f"pitch={deg(pitch):+7.2f}",
              fill=(255, 255, 0), font=font_big)
    draw.text((20, 350),
              f"roll ={deg(roll):+7.2f}",
              fill=(255, 255, 0), font=font_big)

    # Live big-mouth indicator (visual sanity check that values move)
    mouth_h = int(80 * jaw)
    mw = 60
    mx0, my0 = 380, 320
    draw.ellipse((mx0, my0, mx0 + mw, my0 + max(8, mouth_h)),
                 fill=(200, 60, 60), outline=(255, 255, 255), width=2)
    draw.text((mx0 - 6, my0 + max(8, mouth_h) + 6),
              "jawOpen", fill=(220, 220, 220), font=font_small)

    return np.asarray(img, dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=str(
        ROOT / "data/llf-phase2/asian_m__06_neutral.midframe.png"))
    ap.add_argument("--device_path", default="/dev/video10")
    ap.add_argument("--port", type=int, default=11111)
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()

    base = Image.open(args.reference).convert("RGB").resize((512, 512))
    # Dim the reference so text reads well
    base_arr = (np.asarray(base, dtype=np.float32) * 0.35).clip(0, 255).astype(
        np.uint8)

    font_big = _load_font(28)
    font_small = _load_font(18)

    rx = LLFReceiver(host="0.0.0.0", port=args.port, ring_size=64)
    rx.start()

    sink = V4L2Sink(device=args.device_path, width=512, height=512,
                    fps=args.fps, output_format="v4l2")
    sink.open()
    writer = PacedSinkWriter(sink, fps=args.fps)
    writer.start()

    stop = {"flag": False}

    def _on_signal(*_a):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    print(f"LLF on udp:0.0.0.0:{args.port} → {args.device_path} @ {args.fps}fps",
          flush=True)
    print("send Live Link to this box. Ctrl-C to stop.", flush=True)

    frame_idx = 0
    last_log = time.time()
    dt = 1.0 / args.fps
    next_t = time.time()
    try:
        while not stop["flag"]:
            pkt = rx.peek_latest()
            age_ms = int((time.time() - pkt.recv_time) * 1000) if pkt else 9999
            frame = render_overlay(
                base_arr, pkt, rx.received, rx.dropped, age_ms, frame_idx,
                font_big, font_small,
            )
            writer.push([frame])
            frame_idx += 1

            now = time.time()
            if now - last_log >= 5.0:
                print(f"  frames={frame_idx} rx={rx.received} "
                      f"drop={rx.dropped} latest_age_ms={age_ms}",
                      flush=True)
                last_log = now

            next_t += dt
            slack = next_t - time.time()
            if slack > 0:
                time.sleep(slack)
            else:
                next_t = time.time()
    finally:
        writer.stop()
        sink.close()
        rx.stop()
        print(f"stopped after {frame_idx} frames, rx={rx.received} "
              f"drop={rx.dropped}", flush=True)


if __name__ == "__main__":
    main()
