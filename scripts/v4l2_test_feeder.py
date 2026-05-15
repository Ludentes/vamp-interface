"""Minimal v4l2loopback test feeder — diagnoses OBS↔v4l2 path independent
of PersonaLive / LLF.

Loads the reference face, draws a moving red dot that orbits the center,
and writes 512×512 RGB frames to /dev/video10 at a fixed FPS via the same
ffmpeg subprocess the daemon uses. No GPU, no model, no UDP.

Usage:
    /usr/bin/python3 scripts/v4l2_test_feeder.py --fps 8
    /usr/bin/python3 scripts/v4l2_test_feeder.py --fps 25
    /usr/bin/python3 scripts/v4l2_test_feeder.py --fps 8 --mp4_out /tmp/test.mp4

The dot completes one orbit per second so jerk/freeze is obvious. If OBS
shows smooth motion of the dot at the configured FPS, the v4l2 path is
healthy and any jerkiness in the real daemon is upstream of V4L2Sink. If
OBS still shows the same jerk/freeze pattern, the problem is in the
v4l2loopback ↔ OBS handoff (resolution/format mismatch, wrong source
FPS, USB bandwidth, etc).
"""
from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from arkit_bridge.v4l2_sink import V4L2Sink  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=str(
        ROOT / "data/llf-phase2/asian_m__06_neutral.midframe.png"))
    ap.add_argument("--device_path", default="/dev/video10")
    ap.add_argument("--mp4_out", default=None)
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    base = Image.open(args.reference).convert("RGB").resize((512, 512))
    base_arr = np.asarray(base, dtype=np.uint8)

    sink_kind = "mp4" if args.mp4_out else "v4l2"
    sink_path = args.mp4_out if args.mp4_out else args.device_path
    sink = V4L2Sink(
        device=sink_path, width=512, height=512, fps=args.fps,
        output_format=sink_kind,
    )
    sink.open()

    stop = {"flag": False}

    def _on_signal(*_a):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    print(f"feeding {sink_path} at {args.fps} FPS — Ctrl-C to stop",
          flush=True)
    dt = 1.0 / args.fps
    t0 = time.time()
    n_frames = 0
    last_log = time.time()
    next_t = time.time()
    try:
        while not stop["flag"]:
            elapsed = time.time() - t0
            # Orbit angle: 1 revolution per second.
            ang = 2 * math.pi * elapsed
            cx = 256 + int(80 * math.cos(ang))
            cy = 256 + int(80 * math.sin(ang))

            img = Image.fromarray(base_arr.copy())
            draw = ImageDraw.Draw(img)
            r = 20
            draw.ellipse((cx - r, cy - r, cx + r, cy + r),
                         fill=(255, 0, 0))
            draw.text((10, 10), f"{n_frames:05d} t={elapsed:.2f}s",
                      fill=(255, 255, 0))

            sink.write(np.asarray(img, dtype=np.uint8))
            n_frames += 1

            now = time.time()
            if now - last_log >= 5.0:
                eff_fps = n_frames / (now - t0)
                print(f"  wrote={n_frames} eff_fps={eff_fps:.2f}",
                      flush=True)
                last_log = now

            next_t += dt
            slack = next_t - time.time()
            if slack > 0:
                time.sleep(slack)
            else:
                next_t = time.time()
    finally:
        sink.close()
        print(f"stopped after {n_frames} frames", flush=True)


if __name__ == "__main__":
    main()
