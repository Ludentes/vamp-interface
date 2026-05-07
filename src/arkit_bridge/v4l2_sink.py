"""Raw-RGB → ffmpeg → v4l2loopback (or mp4 for testing) sink."""
from __future__ import annotations

import subprocess
from typing import Optional

import numpy as np


class V4L2Sink:
    """Write 24-bit RGB frames to a v4l2 device via ffmpeg subprocess.

    Operator must `sudo modprobe v4l2loopback devices=1 video_nr=10
    card_label=PersonaLive exclusive_caps=1` before starting the daemon.
    """

    def __init__(self, device: str = "/dev/video10", width: int = 512,
                 height: int = 512, fps: int = 25,
                 output_format: str = "v4l2"):
        self._device = device
        self._width = width
        self._height = height
        self._fps = fps
        self._output_format = output_format
        self._proc: Optional[subprocess.Popen] = None

    def open(self):
        cmd = [
            "ffmpeg", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{self._width}x{self._height}",
            "-r", str(self._fps),
            "-i", "pipe:0",
        ]
        if self._output_format == "v4l2":
            cmd += ["-f", "v4l2", "-pix_fmt", "yuv420p", self._device]
        elif self._output_format == "mp4":
            cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", self._device]
        else:
            raise ValueError(f"unknown output_format {self._output_format!r}")
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("sink not opened")
        if frame.shape != (self._height, self._width, 3) or frame.dtype != np.uint8:
            raise ValueError(
                f"expected ({self._height}, {self._width}, 3) uint8; "
                f"got {frame.shape} {frame.dtype}"
            )
        self._proc.stdin.write(frame.tobytes())

    def close(self):
        if self._proc is not None:
            if self._proc.stdin is not None:
                try:
                    self._proc.stdin.close()
                except BrokenPipeError:
                    pass
            self._proc.wait(timeout=5)
            self._proc = None
