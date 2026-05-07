"""Integration test: BatchDriver yields RGB shape/dtype matching offline render.

Skipped if no GPU and no PersonaLive checkpoints. Compares batch shape only.

Run via:
    PYTHONPATH=src /home/newub/w/PersonaLive/.venv/bin/python \\
        -m pytest tests/arkit_bridge/test_streaming_driver.py -v -s

The project's `uv run` venv has diffusers>=0.37 which is incompatible with
PersonaLive's MotEncoder (calls deprecated `output_type='np'` API removed
in diffusers 0.34). Use the PersonaLive venv for this test.
"""
import os
import time

import numpy as np
import pytest

PL_CKPT = os.path.expanduser(
    "~/w/PersonaLive/pretrained_weights/personalive/denoising_unet.pth"
)
STUDENT_CKPT = "runs/student_v2_120k/student_best.pt"
REF_IMG = "data/llf-phase2/asian_m__06_neutral.midframe.png"


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


skip_if_no_models = pytest.mark.skipif(
    not (
        os.path.exists(PL_CKPT)
        and os.path.exists(STUDENT_CKPT)
        and os.path.exists(REF_IMG)
    ),
    reason="PersonaLive / student / reference assets missing",
)
skip_if_no_cuda = pytest.mark.skipif(
    not _cuda_available(), reason="CUDA not available"
)


@skip_if_no_models
@skip_if_no_cuda
def test_batch_driver_render_shape():
    import torch
    from arkit_bridge.streaming_driver import BatchDriver

    drv = BatchDriver(
        reference_path=REF_IMG,
        student_ckpt=STUDENT_CKPT,
        device="cuda",
        dtype=torch.float16,
        precropped=True,
    )
    t0 = time.time()
    drv.start()
    t_start = time.time() - t0
    try:
        b58 = np.zeros((24, 58), dtype=np.float32)
        ypr = np.zeros((24, 3), dtype=np.float32)
        ypr[:, 0] = np.linspace(0, 0.4, 24)  # gentle yaw sweep
        t1 = time.time()
        rgb = drv.render_batch(b58, ypr)
        t_render = time.time() - t1
        print(
            f"\n[time-to-first-frame] start={t_start:.1f}s "
            f"render={t_render:.1f}s total={t_start + t_render:.1f}s",
            flush=True,
        )
        assert rgb.shape == (24, 512, 512, 3)
        assert rgb.dtype == np.uint8
    finally:
        drv.stop()
