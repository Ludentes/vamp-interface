"""v4l2 burst feeder — simulates the streaming daemon's queue dynamics.

Pre-renders 24 frames (animated red dot orbit), then every ``--render_s``
seconds pushes those 24 frames into a PacedSinkWriter via writer.push().
The writer drains at ``--fps`` to v4l2loopback. No PersonaLive, no LLF.

Bisects: if the daemon's "burst then freeze" pattern reproduces here too,
the writer-thread architecture is at fault. If this looks smooth, the
real daemon's jerk is from render-time variance (the renders that happen
to take 3.5 s starve the writer; OBS sees the last frame for 0.5 s).
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
sys.path.insert(0, str(ROOT / "scripts"))
from arkit_bridge.v4l2_sink import V4L2Sink  # noqa: E402
from streaming_bridge import PacedSinkWriter  # noqa: E402


def render_batch(batch_idx: int, n: int, base_arr: np.ndarray):
    """Make ``n`` frames where the dot continues its orbit smoothly across
    batches (batch_idx * n + i is the global frame index)."""
    frames = []
    for i in range(n):
        gi = batch_idx * n + i
        # 1 rev per (n) frames — slower so jerk is easier to see at 8 fps.
        ang = 2 * math.pi * gi / n
        cx = 256 + int(80 * math.cos(ang))
        cy = 256 + int(80 * math.sin(ang))
        img = Image.fromarray(base_arr.copy())
        draw = ImageDraw.Draw(img)
        draw.ellipse((cx - 20, cy - 20, cx + 20, cy + 20), fill=(255, 0, 0))
        draw.text((10, 10), f"batch={batch_idx} gi={gi:04d}",
                  fill=(255, 255, 0))
        frames.append(np.asarray(img, dtype=np.uint8))
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=str(
        ROOT / "data/llf-phase2/asian_m__06_neutral.midframe.png"))
    ap.add_argument("--device_path", default="/dev/video10")
    ap.add_argument("--fps", type=int, default=8,
                    help="writer drain rate (= sink rate)")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--render_s", type=float, default=3.0,
                    help="simulated render time per batch (push interval)")
    args = ap.parse_args()

    base = Image.open(args.reference).convert("RGB").resize((512, 512))
    base_arr = np.asarray(base, dtype=np.uint8)

    sink = V4L2Sink(
        device=args.device_path, width=512, height=512, fps=args.fps,
        output_format="v4l2",
    )
    sink.open()
    writer = PacedSinkWriter(sink, fps=args.fps)
    writer.start()

    stop = {"flag": False}

    def _on_signal(*_a):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    print(f"feeding {args.device_path}: writer={args.fps}fps, "
          f"push {args.batch} frames every {args.render_s}s "
          f"(simulated render). Ctrl-C to stop.", flush=True)
    print(f"queue/writer math: {args.batch}/{args.fps}={args.batch/args.fps:.2f}s "
          f"of motion per push, {args.render_s}s between pushes → "
          f"{(args.render_s - args.batch/args.fps):.2f}s frozen tail per cycle "
          f"(<0 means writer can't keep up).", flush=True)

    batch_idx = 0
    next_push = time.time()
    try:
        while not stop["flag"]:
            now = time.time()
            if now < next_push:
                time.sleep(min(0.05, next_push - now))
                continue
            frames = render_batch(batch_idx, args.batch, base_arr)
            writer.push(frames)
            batch_idx += 1
            next_push += args.render_s
    finally:
        writer.stop()
        sink.close()
        print(f"stopped after {batch_idx} batches", flush=True)


if __name__ == "__main__":
    main()
