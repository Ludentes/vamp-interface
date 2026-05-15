"""LLF / RGB-webcam -> PersonaLive -> v4l2loopback daemon.

Two driver paths:

* ``--driver llf`` (default): UDP B61 packets -> bridge seams -> pipe.
  Same path that has shipped since Task 9.
* ``--driver rgb``: webcam frames -> StabilizedFaceCropper -> pipe in
  teacher_full mode (no bridge seams; pipe's real motion_encoder and
  pose_encoder consume RGB directly). V1 batch render only — V2 cohort
  streaming is not wired for the RGB path.

Run via the PersonaLive venv (project venv has incompatible diffusers):

  sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=PersonaLive exclusive_caps=1

  # LLF driver (default)
  PYTHONPATH=src /home/newub/w/PersonaLive/.venv/bin/python \\
      scripts/streaming_bridge.py \\
      --reference data/llf-phase2/asian_m__06_neutral.midframe.png \\
      --ckpt runs/student_v2_120k/student_best.pt \\
      --device_path /dev/video10 \\
      --port 11111 \\
      --batch 24

  # RGB driver (webcam)
  PYTHONPATH=src /home/newub/w/PersonaLive/.venv/bin/python \\
      scripts/streaming_bridge.py \\
      --driver rgb --cam_index 0 \\
      --reference data/llf-phase2/asian_m__06_neutral.midframe.png \\
      --ckpt runs/student_v2_120k/student_best.pt \\
      --device_path /dev/video10 \\
      --batch 8

When ``--device_path`` doesn't exist (smoke testing without v4l2loopback)
pass ``--mp4_out path.mp4`` to write a file instead.
"""
from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch


class PacedSinkWriter:
    """Background thread that writes frames to a sink at a fixed FPS.

    Decouples render rate from sink rate: while the daemon renders the
    next batch (~3 s on RTX 5090), this thread is consuming the previous
    batch from the queue and feeding v4l2loopback at the configured FPS.
    If the queue empties before the next batch arrives, the writer
    repeats the last good frame so v4l2 stays alive (no frozen-then-burst
    pattern that confuses OBS).
    """

    def __init__(self, sink, fps: int, max_queue: int = 256,
                 prebuffer: int = 0):
        self._sink = sink
        self._dt = 1.0 / fps
        self._q: deque = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._last = None
        self._max_queue = max_queue
        # prebuffer: don't start draining until queue reaches this depth
        # the FIRST time. Absorbs render variance (cohort-time + prep-time
        # at batch boundary). After the initial fill the writer drains
        # normally; if the queue empties, it repeats the last frame.
        self._prebuffer = prebuffer
        self._primed = prebuffer == 0
        self._thr = threading.Thread(target=self._run, daemon=True)
        # Objective sink-side counters (read with snapshot()).
        self._writes = 0          # total frames written to sink
        self._fresh_writes = 0    # writes that drained a real queued frame
        self._repeat_writes = 0   # writes that repeated _last (queue empty)
        self._dropped = 0         # frames evicted from queue (overflow)
        self._depth_sum = 0       # for mean queue depth
        self._depth_samples = 0
        self._depth_max = 0

    def snapshot(self):
        """Atomic copy of writer counters; safe to call from any thread."""
        with self._lock:
            return {
                "writes": self._writes,
                "fresh": self._fresh_writes,
                "repeat": self._repeat_writes,
                "dropped": self._dropped,
                "depth_mean": (self._depth_sum / self._depth_samples
                               if self._depth_samples else 0.0),
                "depth_max": self._depth_max,
            }

    def reset_window(self):
        """Reset rolling counters (call after each log line)."""
        with self._lock:
            self._writes = 0
            self._fresh_writes = 0
            self._repeat_writes = 0
            self._dropped = 0
            self._depth_sum = 0
            self._depth_samples = 0
            self._depth_max = 0

    def start(self):
        self._thr.start()

    def stop(self, timeout: float = 2.0):
        self._stop.set()
        self._thr.join(timeout=timeout)

    def push(self, frames):
        """Append frames (iterable of HWC uint8) to the write queue.

        Drops oldest if the queue is fuller than max_queue — better to
        skip a stale render than to keep growing latency unboundedly.
        """
        with self._lock:
            for f in frames:
                if len(self._q) >= self._max_queue:
                    self._q.popleft()
                    self._dropped += 1
                self._q.append(f)

    def _run(self):
        next_t = time.time()
        while not self._stop.is_set():
            with self._lock:
                depth = len(self._q)
                # Hold off draining until the prebuffer has filled once.
                # After that we always drain (or repeat last frame on empty).
                if not self._primed:
                    if depth >= self._prebuffer:
                        self._primed = True
                if not self._primed:
                    f = self._last  # still warming; sink keeps last frame
                    fresh = False
                elif self._q:
                    f = self._q.popleft()
                    fresh = True
                else:
                    f = self._last
                    fresh = False
                self._depth_sum += depth
                self._depth_samples += 1
                if depth > self._depth_max:
                    self._depth_max = depth
            if f is not None:
                try:
                    self._sink.write(f)
                    self._last = f
                    with self._lock:
                        self._writes += 1
                        if fresh:
                            self._fresh_writes += 1
                        else:
                            self._repeat_writes += 1
                except (BrokenPipeError, OSError):
                    return
            next_t += self._dt
            slack = next_t - time.time()
            if slack > 0:
                time.sleep(slack)
            else:
                # Fell behind; reset cadence to "now" to avoid burst.
                next_t = time.time()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from arkit_bridge.llf_udp import LLFReceiver, B61Packet  # noqa: E402
from arkit_bridge.rgb_grabber import RGBGrabber  # noqa: E402
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
    ap.add_argument("--fps", type=int, default=25,
                    help="sink advertise rate AND writer drain rate; lower "
                         "this (e.g. 8) when render rate is below 25 to "
                         "spread each fresh frame evenly instead of "
                         "bursting a cohort then sitting on last-frame")
    ap.add_argument("--sample_span_s", type=float, default=3.0,
                    help="window of real time each batch represents. "
                         "k=batch packets are sampled evenly across this "
                         "span, so output played at fps=k/span_s shows "
                         "real-time-speed motion. Set this to your "
                         "expected render time per batch (~3.0s for "
                         "batch=24 V2 on 5090).")
    ap.add_argument("--writer_prebuffer", type=int, default=0,
                    help="frames to accumulate before writer starts draining "
                         "the FIRST time. Adds slack to absorb cohort/prep "
                         "variance at batch boundaries. 0 = no buffer "
                         "(writer starts immediately, may stutter if "
                         "render is slower than drain)")
    ap.add_argument("--torch_device", default="cuda")
    ap.add_argument("--num_inference_steps", type=int, default=4)
    ap.add_argument("--mode", choices=["v1", "v2"], default="v1",
                    help="v1: render full batch then sink; v2: cohort-stream "
                         "(prepare per batch + step per 4 frames). v2 only "
                         "applies to --driver llf.")
    ap.add_argument("--driver", choices=["llf", "rgb"], default="llf",
                    help="frame source: llf (UDP B61 packets, bridge mode) "
                         "or rgb (webcam, teacher_full mode)")
    ap.add_argument("--cam_index", type=int, default=0,
                    help="cv2.VideoCapture device index (--driver rgb only)")
    args = ap.parse_args()

    assert args.batch % 4 == 0 and args.batch >= 4
    if args.driver == "rgb" and args.mode == "v2":
        # No teacher_full V2 path yet — render_batch_rgb is V1-only.
        raise SystemExit(
            "--driver rgb does not support --mode v2 (V1 batch render only)"
        )

    print("[1/4] building pipe (this can take 10-20s)...", flush=True)
    drv = BatchDriver(
        reference_path=args.reference,
        student_ckpt=args.ckpt,
        device=args.torch_device,
        dtype=torch.float16,
        num_inference_steps=args.num_inference_steps,
    )
    drv.start()

    print(f"[2/4] pre-warming with zero-vector batch "
          f"(driver={args.driver}, mode={args.mode})...", flush=True)
    t0 = time.time()
    if args.driver == "llf":
        b58_warm = np.zeros((args.batch, 58), dtype=np.float32)
        ypr_warm = np.zeros((args.batch, 3), dtype=np.float32)
        # Pre-warm always uses V1 path so we have a `last_good_rgb` anchor
        # frame buffer for idle passthrough regardless of selected mode.
        last_good_rgb = drv.render_batch(b58_warm, ypr_warm)
    else:
        # RGB pre-warm: feed the reference face as every input frame.
        # Forces all CUDA allocations + JIT compilations to happen now,
        # not on the first real webcam batch.
        ref = drv._ref_face  # type: ignore[attr-defined]
        warm_frames = [ref] * args.batch
        last_good_rgb = drv.render_batch_rgb(warm_frames)
    print(f"  pre-warm render={time.time() - t0:.1f}s", flush=True)

    sink_kind = "mp4" if args.mp4_out else "v4l2"
    sink_path = args.mp4_out if args.mp4_out else args.device_path
    print(f"[3/4] opening {sink_kind} sink at {sink_path}...", flush=True)
    sink = V4L2Sink(
        device=sink_path, width=512, height=512, fps=args.fps,
        output_format=sink_kind,
    )
    sink.open()
    # Paced writer thread: decouples render rate from sink rate. The
    # daemon's render loop pushes frames into the writer's queue and
    # never blocks on sink I/O. The writer drains at args.fps,
    # repeating the last frame when the queue is empty (OBS sees a
    # continuous stream instead of bursts of frames-faster-than-OBS-reads
    # interspersed with frozen gaps).
    writer = PacedSinkWriter(sink, fps=args.fps,
                             prebuffer=args.writer_prebuffer)
    writer.start()

    if args.driver == "llf":
        print(f"[4/4] starting LLF receiver on UDP :{args.port}...", flush=True)
        # Ring must hold at least 2 × span × 60Hz LLF packets so the
        # evenly-spaced sampler always has both endpoints of its window.
        ring_size = max(args.batch * 4, int(2.5 * args.sample_span_s * 60))
        rx = LLFReceiver(host="0.0.0.0", port=args.port, ring_size=ring_size)
        rx.start()
        grabber = None
    else:
        print(f"[4/4] starting RGB grabber on cam_index={args.cam_index}...",
              flush=True)
        grabber = RGBGrabber(
            device_index=args.cam_index,
            width=512, height=512,
            ring_size=max(args.batch * 4, 32),
        )
        grabber.start()
        rx = None

    stop = {"flag": False}

    def _on_signal(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    last_log = time.time()
    rendered = 0
    last_cohort_stats = ""
    if args.driver == "llf":
        print("daemon ready; waiting for LLF packets", flush=True)
    else:
        print("daemon ready; pulling webcam frames", flush=True)
    try:
        while not stop["flag"]:
            if args.driver == "rgb":
                assert grabber is not None
                frames = grabber.pop_window(k=args.batch, timeout_s=2.0)
                if not frames:
                    # Writer thread holds the last frame autonomously.
                    time.sleep(0.1)
                    continue
                t0 = time.time()
                rgb = drv.render_batch_rgb(frames)
                infer_ms = (time.time() - t0) * 1000.0
                last_good_rgb = rgb
                writer.push(rgb)
                rendered += len(rgb)
                now = time.time()
                if now - last_log >= 5.0:
                    fps = rendered / (now - last_log)
                    print(
                        f"  rendered={rendered} fps={fps:.1f} "
                        f"infer_ms={infer_ms:.0f} cap={grabber.captured} "
                        f"drop={grabber.dropped} det_fail={grabber.detect_failures}",
                        flush=True,
                    )
                    rendered = 0
                    last_log = now
                continue

            assert rx is not None
            pkts = rx.pop_window_evenly_spaced(
                k=args.batch, span_s=args.sample_span_s, timeout_s=6.0,
            )
            if not pkts:
                # Writer thread holds the last frame; just idle briefly.
                time.sleep(0.1)
                continue

            b58, ypr = packets_to_arrays(pkts)
            b58, ypr = _pad_to_multiple_of_4(b58, ypr)

            if args.mode == "v1":
                t0 = time.time()
                rgb = drv.render_batch(b58, ypr)
                infer_ms = (time.time() - t0) * 1000.0
                last_good_rgb = rgb
                writer.push(rgb)
                rendered += len(rgb)
            else:
                # V2: prepare + per-cohort step. Push each cohort to the
                # writer as soon as it's decoded so the user sees motion
                # ~500 ms after the batch fills, not after the full 3 s
                # render.
                t0 = time.time()
                windows = drv.prepare_v2(b58, ypr)
                prep_ms = (time.time() - t0) * 1000.0
                rgb_blocks = []
                step_ms_total = 0.0
                cohort_dts = []
                for _ in range(windows):
                    ts = time.time()
                    block = drv.step_v2(n=4)
                    step_ms_total += (time.time() - ts) * 1000.0
                    cohort_dts.append((time.time() - ts) * 1000.0)
                    rgb_blocks.append(block)
                    writer.push(block)
                rgb = np.concatenate(rgb_blocks, axis=0)
                infer_ms = prep_ms + step_ms_total
                last_good_rgb = rgb
                rendered += len(rgb)
                if cohort_dts:
                    cd = sorted(cohort_dts)
                    last_cohort_stats = (
                        f"prep={prep_ms:.0f} cohort_ms="
                        f"min={cd[0]:.0f}/p50={cd[len(cd)//2]:.0f}/"
                        f"max={cd[-1]:.0f}/n={len(cd)}"
                    )
                else:
                    last_cohort_stats = "no cohorts"

            now = time.time()
            if now - last_log >= 5.0:
                dt_log = now - last_log
                fps = rendered / dt_log
                snap = writer.snapshot()
                writer.reset_window()
                sink_fps = snap["writes"] / dt_log
                fresh_fps = snap["fresh"] / dt_log
                repeat_fps = snap["repeat"] / dt_log
                print(
                    f"  rendered={rendered} fps={fps:.1f} "
                    f"infer_ms={infer_ms:.0f} rx_received={rx.received} "
                    f"rx_dropped={rx.dropped} cursor_skips={rx.cursor_skips}",
                    flush=True,
                )
                print(
                    f"    sink: write_fps={sink_fps:.1f} "
                    f"fresh_fps={fresh_fps:.1f} repeat_fps={repeat_fps:.1f} "
                    f"q_mean={snap['depth_mean']:.1f} q_max={snap['depth_max']} "
                    f"dropped={snap['dropped']}",
                    flush=True,
                )
                if last_cohort_stats:
                    print(f"    {last_cohort_stats}", flush=True)
                rendered = 0
                last_log = now
    finally:
        print("draining...", flush=True)
        if rx is not None:
            rx.stop()
        if grabber is not None:
            grabber.stop()
        writer.stop()
        sink.close()
        drv.stop()


if __name__ == "__main__":
    main()
