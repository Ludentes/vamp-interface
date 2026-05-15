"""Paced background-thread writer for v4l2-style sinks.

Decouples render rate from sink rate. Single-buffered v4l2loopback drops
all but the last frame of a burst write, so the daemon must NOT call
``sink.write()`` from the render loop directly. Instead the render loop
``push()``es onto a thread-safe deque; this writer drains it at a fixed
FPS, repeating the last good frame on starvation.

Originally lived in ``scripts/streaming_bridge.py`` (PersonaLive daemon);
extracted so the LivePortrait daemon can reuse it.

Sink contract: any object with ``write(frame: ndarray (H,W,3) uint8)``.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Iterable


class PacedSinkWriter:
    """Background thread that writes frames to a sink at a fixed FPS.

    Parameters
    ----------
    sink : object with ``.write(frame_uint8_HW3)``
    fps : int
        Drain rate in frames/sec. Should match the producer's *steady-
        state unique-frame rate* — higher just adds visible repeat-stutter
        without increasing perceived smoothness.
    max_queue : int
        Drop oldest if exceeded (better than unbounded latency growth).
    prebuffer : int
        Hold off draining until the queue first reaches this depth.
        Absorbs producer-side variance at startup. After the first fill,
        starvation falls back to last-frame-repeat (graceful, not pause).
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
        self._prebuffer = prebuffer
        self._primed = prebuffer == 0
        self._thr = threading.Thread(target=self._run, daemon=True)
        # Objective sink-side counters (read with snapshot()).
        self._writes = 0
        self._fresh_writes = 0
        self._repeat_writes = 0
        self._dropped = 0
        self._depth_sum = 0
        self._depth_samples = 0
        self._depth_max = 0

    # --- counters -------------------------------------------------------
    def snapshot(self) -> dict:
        """Atomic copy of writer counters; safe from any thread."""
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

    def reset_window(self) -> None:
        """Reset rolling counters (call after each log line)."""
        with self._lock:
            self._writes = 0
            self._fresh_writes = 0
            self._repeat_writes = 0
            self._dropped = 0
            self._depth_sum = 0
            self._depth_samples = 0
            self._depth_max = 0

    # --- lifecycle ------------------------------------------------------
    def start(self) -> None:
        self._thr.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._thr.join(timeout=timeout)

    # --- producer API ---------------------------------------------------
    def push(self, frames: Iterable) -> None:
        """Append frames to the write queue. Drops oldest on overflow."""
        with self._lock:
            for f in frames:
                if len(self._q) >= self._max_queue:
                    self._q.popleft()
                    self._dropped += 1
                self._q.append(f)

    # --- background -----------------------------------------------------
    def _run(self) -> None:
        next_t = time.time()
        while not self._stop.is_set():
            with self._lock:
                depth = len(self._q)
                if not self._primed and depth >= self._prebuffer:
                    self._primed = True
                if not self._primed:
                    f = self._last
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
