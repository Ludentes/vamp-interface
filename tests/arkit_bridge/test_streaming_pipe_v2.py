"""Bit-equivalence test for the V2 prepare()/step()/decode() refactor.

Compares the upstream PersonaLive ``Pose2VideoPipeline_Stream.__call__``
against our vendored copy's backward-compat ``__call__`` (which is now
implemented as ``prepare()`` + N x ``step()`` + ``decode()``). With identical
seeds and inputs and shared sub-modules, the two pipelines must produce
the same uint8 grid within fp16 noise tolerance.

Skipped if no GPU and no PersonaLive checkpoints. Mirrors the gating in
``test_streaming_driver.py``.

Run via:
    PYTHONPATH=src /home/newub/w/PersonaLive/.venv/bin/python \
        -m pytest tests/arkit_bridge/test_streaming_pipe_v2.py -v -s

The project's ``uv run`` venv has diffusers>=0.37 which is incompatible
with PersonaLive's MotEncoder. Use the PersonaLive venv for this test.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

PL_CKPT = os.path.expanduser(
    "~/w/PersonaLive/pretrained_weights/personalive/denoising_unet.pth"
)
REF_IMG = "data/llf-phase2/asian_m__06_neutral.midframe.png"


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


skip_if_no_models = pytest.mark.skipif(
    not (os.path.exists(PL_CKPT) and os.path.exists(REF_IMG)),
    reason="PersonaLive checkpoints / reference asset missing",
)
skip_if_no_cuda = pytest.mark.skipif(
    not _cuda_available(), reason="CUDA not available"
)


def _run_pipeline(pipe, ref_pil, ref_face, tgt_images, face_images, seed: int):
    """Run a pipeline ``__call__`` with a deterministic generator + seeded torch."""
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    gen = torch.Generator(device=pipe._execution_device).manual_seed(seed)
    out = pipe(
        list(tgt_images),  # call mutates by extending; pass a copy
        ref_pil,
        list(face_images),
        ref_face,
        width=512,
        height=512,
        video_length=16,  # 4 windows of size 4 -> total cohort iters = 4 + 3 = 7
        # Note: video_length must be >= padding_num+1 = 13 because the pipe's
        # tail-mirror extension `tgt_images.extend(tgt_images[-padding_num-1:-1][::-1])`
        # clamps to the available length and underfills pose_feas otherwise.
        num_inference_steps=4,
        guidance_scale=1.0,
        generator=gen,
        temporal_window_size=4,
        temporal_adaptive_step=4,
        temporal_kv_cache=True,
    )
    # Output is Pose2VideoPipelineOutput(videos=...) with video shape
    # (b, c, f, h, w) in float [0, 1].
    return out.videos.detach().cpu().float().numpy()


@skip_if_no_models
@skip_if_no_cuda
def test_v2_split_bit_equivalent_to_upstream():
    import sys
    import torch
    from PIL import Image

    from arkit_bridge.streaming_driver import _build_pipe, PL

    # _build_pipe inserts PL on sys.path; ensure vendor is also importable.
    # Build the upstream pipe (this is the baseline).
    upstream_pipe = _build_pipe("cuda", torch.float16)

    # Build a vendored-class wrapper that SHARES every sub-module with the
    # upstream pipe. The vendored module imports from ``src.pipelines.utils``
    # which resolves at runtime via PL on sys.path (set by _build_pipe).
    if str(PL) not in sys.path:
        sys.path.insert(0, str(PL))
    # Importing vendor.personalive.pipeline_pose2vid_streaming triggers
    # ``from src.pipelines.utils import ...`` — works because PL is on path.
    from vendor.personalive.pipeline_pose2vid_streaming import (
        Pose2VideoPipeline_Stream as VendoredStream,
    )

    vendored_pipe = VendoredStream(
        vae=upstream_pipe.vae,
        image_encoder=upstream_pipe.image_encoder,
        reference_unet=upstream_pipe.reference_unet,
        denoising_unet=upstream_pipe.denoising_unet,
        motion_encoder=upstream_pipe.motion_encoder,
        pose_encoder=upstream_pipe.pose_encoder,
        pose_guider=upstream_pipe.pose_guider,
        scheduler=upstream_pipe.scheduler,
    ).to("cuda")

    # Inputs: a single neutral portrait, repeated for both driving and face
    # streams. We're testing pipeline equivalence, not bridge fidelity, so
    # static inputs are fine and converge fastest.
    ref_pil = Image.open(REF_IMG).convert("RGB")
    ref_face = ref_pil.copy()
    tgt_images = [ref_pil.copy() for _ in range(16)]
    face_images = [ref_pil.copy() for _ in range(16)]

    seed = 12345
    a = _run_pipeline(upstream_pipe, ref_pil, ref_face, tgt_images, face_images, seed)
    b = _run_pipeline(vendored_pipe, ref_pil, ref_face, tgt_images, face_images, seed)

    assert a.shape == b.shape, f"shape mismatch: upstream={a.shape} vendored={b.shape}"

    # Compare in uint8 space (mirrors the deliverable check in the plan).
    a_u8 = np.clip(a * 255.0, 0, 255).astype(np.uint8)
    b_u8 = np.clip(b * 255.0, 0, 255).astype(np.uint8)
    diff = np.abs(a_u8.astype(np.int32) - b_u8.astype(np.int32))
    mean_abs = float(diff.mean())
    max_abs = int(diff.max())
    print(
        f"\n[v2 bit-equiv] mean_abs_diff={mean_abs:.4f} max_abs_diff={max_abs} "
        f"shape={a_u8.shape}",
        flush=True,
    )
    # Tolerance: fp16 + nondeterministic CUDA kernels can produce small
    # per-pixel differences even with identical seeds. The plan asks for
    # mean abs diff < 2.0 in uint8 space.
    assert mean_abs < 2.0, f"mean abs diff {mean_abs} >= 2.0 (uint8)"
