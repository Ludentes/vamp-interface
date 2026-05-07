"""Unit tests for src/arkit_bridge/v4l2_sink.py.

Uses an MP4 destination instead of a real /dev/video device so the test
runs in CI without v4l2loopback. The ffmpeg invocation is the same shape;
we just swap the output URL.
"""
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from arkit_bridge.v4l2_sink import V4L2Sink


def test_sink_writes_frames_to_mp4(tmp_path):
    out = tmp_path / "out.mp4"
    sink = V4L2Sink(
        device=str(out), width=256, height=256, fps=25,
        output_format="mp4",  # test-only path
    )
    sink.open()
    try:
        for i in range(25):
            frame = np.full((256, 256, 3), i * 10, dtype=np.uint8)
            sink.write(frame)
    finally:
        sink.close()
    assert out.exists()
    assert out.stat().st_size > 1000

    # Probe the file with ffprobe to confirm 25 frames at 25 fps.
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames",
         "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames",
         "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True,
    )
    assert int(res.stdout.strip()) == 25


def test_sink_rejects_bad_frame_shape(tmp_path):
    sink = V4L2Sink(device=str(tmp_path / "x.mp4"), width=64, height=64,
                   fps=25, output_format="mp4")
    sink.open()
    try:
        with pytest.raises(ValueError, match="expected"):
            sink.write(np.zeros((32, 32, 3), dtype=np.uint8))   # wrong size
        with pytest.raises(ValueError, match="expected"):
            sink.write(np.zeros((64, 64, 3), dtype=np.float32)) # wrong dtype
    finally:
        sink.close()


def test_sink_write_before_open_raises(tmp_path):
    sink = V4L2Sink(device=str(tmp_path / "x.mp4"), width=64, height=64,
                   fps=25, output_format="mp4")
    with pytest.raises(RuntimeError, match="not opened"):
        sink.write(np.zeros((64, 64, 3), dtype=np.uint8))
