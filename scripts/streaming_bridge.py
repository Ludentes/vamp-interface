"""LLF UDP -> PersonaLive bridge -> v4l2loopback daemon.

Run via the PersonaLive venv (project venv has incompatible diffusers):

  sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=PersonaLive exclusive_caps=1

  PYTHONPATH=src /home/newub/w/PersonaLive/.venv/bin/python \
      scripts/streaming_bridge.py \
      --reference data/llf-phase2/asian_m__06_neutral.midframe.png \
      --ckpt runs/student_v2_120k/student_best.pt \
      --device_path /dev/video10 \
      --port 11111 \
      --batch 24

When `--device_path` doesn't exist (smoke testing without v4l2loopback)
pass `--mp4_out path.mp4` to write a file instead.
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
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from arkit_bridge.llf_udp import LLFReceiver, B61Packet  # noqa: E402
from arkit_bridge.streaming_driver import BatchDriver  # noqa: E402
from arkit_bridge.v4l2_sink import V4L2Sink  # noqa: E402


def packets_to_arrays(pkts: list[B61Packet]):
    """(B61Packet list) -> (b58 (n,58) float32, ypr (n,3) float32 radians)."""
    n = len(pkts)
    b58 = np.zeros((n, 58), dtype=np.float32)
    ypr = np.zeros((n, 3), dtype=np.float32)
    for i, p in enumerate(pkts):
        b58[i] = np.asarray(p.b58, dtype=np.float32)
        ypr[i] = np.asarray(p.head_ypr, dtype=np.float32)
    return b58, ypr


def _pad_to_multiple_of_4(b58: np.ndarray, ypr: np.ndarray):
    n = b58.shape[0]
    if n % 4 == 0:
        return b58, ypr
    pad = 4 - (n % 4)
    b58 = np.concatenate([b58, np.tile(b58[-1:], (pad, 1))], axis=0)
    ypr = np.concatenate([ypr, np.tile(ypr[-1:], (pad, 1))], axis=0)
    return b58, ypr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--ckpt", required=True, help="MotEncoderStudent .pt")
    ap.add_argument("--device_path", default="/dev/video10")
    ap.add_argument("--mp4_out", default=None,
                    help="if set, write to mp4 instead of v4l2 (smoke testing)")
    ap.add_argument("--port", type=int, default=11111)
    ap.add_argument("--batch", type=int, default=24,
                    help="frames per pipe call; multiple of 4")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--torch_device", default="cuda")
    ap.add_argument("--num_inference_steps", type=int, default=4)
    args = ap.parse_args()

    assert args.batch % 4 == 0 and args.batch >= 4

    print("[1/4] building pipe (this can take 10-20s)...", flush=True)
    drv = BatchDriver(
        reference_path=args.reference,
        student_ckpt=args.ckpt,
        device=args.torch_device,
        dtype=torch.float16,
        num_inference_steps=args.num_inference_steps,
    )
    drv.start()

    print("[2/4] pre-warming with zero-vector batch...", flush=True)
    b58_warm = np.zeros((args.batch, 58), dtype=np.float32)
    ypr_warm = np.zeros((args.batch, 3), dtype=np.float32)
    t0 = time.time()
    last_good_rgb = drv.render_batch(b58_warm, ypr_warm)
    print(f"  pre-warm render={time.time() - t0:.1f}s", flush=True)

    sink_kind = "mp4" if args.mp4_out else "v4l2"
    sink_path = args.mp4_out if args.mp4_out else args.device_path
    print(f"[3/4] opening {sink_kind} sink at {sink_path}...", flush=True)
    sink = V4L2Sink(
        device=sink_path, width=512, height=512, fps=args.fps,
        output_format=sink_kind,
    )
    sink.open()

    print(f"[4/4] starting LLF receiver on UDP :{args.port}...", flush=True)
    rx = LLFReceiver(host="0.0.0.0", port=args.port, ring_size=args.batch * 4)
    rx.start()

    stop = {"flag": False}

    def _on_signal(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    last_log = time.time()
    rendered = 0
    print("daemon ready; waiting for LLF packets", flush=True)
    try:
        while not stop["flag"]:
            pkts = rx.pop_window(k=args.batch, timeout_s=2.0)
            if not pkts:
                # Idle: emit anchor passthrough so OBS sees a signal.
                for f in last_good_rgb:
                    sink.write(f)
                    time.sleep(1.0 / args.fps)
                continue

            b58, ypr = packets_to_arrays(pkts)
            b58, ypr = _pad_to_multiple_of_4(b58, ypr)

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
                print(
                    f"  rendered={rendered} fps={fps:.1f} "
                    f"infer_ms={infer_ms:.0f} rx_received={rx.received} "
                    f"rx_dropped={rx.dropped}",
                    flush=True,
                )
                rendered = 0
                last_log = now
    finally:
        print("draining...", flush=True)
        rx.stop()
        sink.close()
        drv.stop()


if __name__ == "__main__":
    main()
