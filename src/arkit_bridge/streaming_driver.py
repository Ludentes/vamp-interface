"""BatchDriver: build PersonaLive pipe once, render N-frame batches with bridge seams.

Single-threaded by design. The pipe holds CUDA state and the seams are
patched onto the pipe instance — concurrent calls into render_batch from
multiple threads would corrupt the seam state. The streaming daemon should
own one BatchDriver per worker.

**Process-global side effect:** `render_batch` monkey-patches `torch.tensor`
for the duration of the pipe call to swap PersonaLive's hardcoded 4-step
DDIM schedule (`[999, 666, 333, 0]`) for our trailing schedule. The patch
is restored in a `finally`. If any other thread in the same process calls
`torch.tensor` with that exact list literal during the pipe call, it will
be intercepted. We accept this for V1 because the daemon is single-threaded
and no other code path in this project produces that literal.

The pipe call signature mirrors `scripts/apply_bridge_to_personalive.py:385`
exactly (positional args + a fixed set of keyword args). The PersonaLive
pipeline `__call__` uses positional `tgt_images, ref_image, face_images,
ref_face_image, width, height, video_length, num_inference_steps,
guidance_scale` — see `~/w/PersonaLive/src/pipelines/pipeline_pose2vid.py:440`.

`temporal_window_size=4` and `temporal_adaptive_step=4` match the
`apply_bridge_to_personalive.py` defaults; they also satisfy the pipeline's
`num_inference_steps % temporal_adaptive_step == 0` and
`video_length % temporal_window_size == 0` invariants for our 4-step
schedule and multiples-of-4 batches.
"""
from __future__ import annotations

import gc
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from arkit_bridge.seam_install import install_arkit_seams
from arkit_bridge.student import MotEncoderStudent

PL = Path(os.path.expanduser("~/w/PersonaLive"))
ROOT = Path(__file__).resolve().parents[2]
VENDOR_PL = ROOT / "vendor" / "personalive"


def _build_pipe(device, dtype):
    """Lift of scripts/apply_bridge_to_personalive.py:build_pipe.

    Inserts PL on sys.path inside the function (daemon-safe; avoids
    polluting other workers' import resolution at module import time)
    and chdirs into PL while loading because configs use relative paths.
    Restores cwd in a finally — the original `build_pipe` does not, which
    is a bug we deliberately don't reproduce here.
    """
    if str(PL) not in sys.path:
        sys.path.insert(0, str(PL))
    if str(VENDOR_PL) not in sys.path:
        sys.path.insert(0, str(VENDOR_PL))
    cwd = os.getcwd()
    os.chdir(PL)
    try:
        from diffusers import AutoencoderKL  # noqa: WPS433
        from transformers import CLIPVisionModelWithProjection  # noqa: WPS433
        from src.scheduler.scheduler_ddim import DDIMScheduler  # noqa: WPS433
        from src.models.unet_2d_condition import UNet2DConditionModel  # noqa: WPS433
        from src.models.unet_3d import UNet3DConditionModel  # noqa: WPS433
        # Use the *vendored* Pose2VideoPipeline_Stream so prepare()/step()/
        # decode() are available for V2 cohort streaming. V1 still calls
        # __call__, which the vendored class implements as a wrapper around
        # the new methods (bit-equivalent — verified by
        # tests/arkit_bridge/test_streaming_pipe_v2.py).
        from pipeline_pose2vid_streaming import Pose2VideoPipeline_Stream  # noqa: WPS433
        from src.models.motion_encoder.encoder import MotEncoder  # noqa: WPS433
        from src.liveportrait.motion_extractor import MotionExtractor  # noqa: WPS433
        from src.models.pose_guider import PoseGuider  # noqa: WPS433

        cfg = OmegaConf.load("configs/prompts/personalive_offline.yaml")
        infer = OmegaConf.load(cfg.inference_config)
        vae = AutoencoderKL.from_pretrained(cfg.vae_path).to(device, dtype=dtype)
        ref_unet = UNet2DConditionModel.from_pretrained(
            cfg.pretrained_base_model_path, subfolder="unet"
        ).to(device, dtype=dtype)
        den_unet = UNet3DConditionModel.from_pretrained_2d(
            cfg.pretrained_base_model_path, "", subfolder="unet",
            unet_additional_kwargs=infer.unet_additional_kwargs,
        ).to(device, dtype=dtype)
        me = MotEncoder().to(device, dtype=dtype).eval()
        pg = PoseGuider().to(device, dtype=dtype)
        pe = MotionExtractor(num_kp=21).to(device, dtype=dtype).eval()
        img_enc = CLIPVisionModelWithProjection.from_pretrained(
            cfg.image_encoder_path
        ).to(device, dtype=dtype)
        sched = DDIMScheduler(**OmegaConf.to_container(infer.noise_scheduler_kwargs))

        base = cfg.denoising_unet_path
        den_unet.load_state_dict(torch.load(base, map_location="cpu"), strict=False)
        ref_unet.load_state_dict(
            torch.load(base.replace("denoising_unet", "reference_unet"),
                       map_location="cpu"),
            strict=True,
        )
        me.load_state_dict(
            torch.load(base.replace("denoising_unet", "motion_encoder"),
                       map_location="cpu"),
            strict=True,
        )
        pg.load_state_dict(
            torch.load(base.replace("denoising_unet", "pose_guider"),
                       map_location="cpu"),
            strict=True,
        )
        den_unet.load_state_dict(
            torch.load(base.replace("denoising_unet", "temporal_module"),
                       map_location="cpu"),
            strict=False,
        )
        pe.load_state_dict(
            torch.load(base.replace("denoising_unet", "motion_extractor"),
                       map_location="cpu"),
            strict=False,
        )

        return Pose2VideoPipeline_Stream(
            vae=vae, image_encoder=img_enc,
            reference_unet=ref_unet, denoising_unet=den_unet,
            motion_encoder=me, pose_encoder=pe, pose_guider=pg, scheduler=sched,
        ).to(device)
    finally:
        os.chdir(cwd)


class BatchDriver:
    """Owns the pipe + student. Render multiple-of-4-frame batches.

    Single-threaded. Construct, then call start() once, then render_batch()
    repeatedly, then stop() at shutdown.
    """

    def __init__(
        self,
        reference_path: str,
        student_ckpt: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        num_inference_steps: int = 4,
        guidance_scale: float = 1.0,
        precropped: bool = True,
        seed: int = 42,
    ):
        self._reference_path = reference_path
        self._student_ckpt = student_ckpt
        self._device = device
        self._dtype = dtype
        self._num_inference_steps = num_inference_steps
        self._guidance_scale = guidance_scale
        self._precropped = precropped
        self._seed = seed
        self._pipe = None
        self._student: Optional[MotEncoderStudent] = None
        self._ref_pil: Optional[Image.Image] = None
        self._ref_face: Optional[Image.Image] = None

    def start(self) -> None:
        self._pipe = _build_pipe(self._device, self._dtype)
        student = MotEncoderStudent().to(self._device).eval()
        student.load_state_dict(
            torch.load(self._student_ckpt, map_location=self._device)
        )
        self._student = student
        self._prepare_reference()
        self._install_step_schedule()

    def _install_step_schedule(self) -> None:
        """Configure scheduler for the N-step trailing DDIM ship config.

        Mirrors scripts/apply_bridge_to_personalive.py:350-371. Pre-loads
        ``num_inference_steps`` and casts ``alphas_cumprod`` to UNet dtype so
        scheduler.step() doesn't re-cast fp16 latents to fp32 mid-loop.
        The torch.tensor schedule monkey-patch is applied per render_batch call.
        """
        assert self._pipe is not None
        N = self._num_inference_steps
        step = 1000 // N
        self._SCHEDULE = [999 - i * step for i in range(N)]
        sched = self._pipe.scheduler
        sched.num_inference_steps = N
        sched.step_length = None
        sched.set_step_length = lambda _x: None
        sched.alphas_cumprod = sched.alphas_cumprod.to(dtype=self._dtype)
        sched.final_alpha_cumprod = sched.final_alpha_cumprod.to(dtype=self._dtype)

    def _prepare_reference(self) -> None:
        if self._precropped:
            img = Image.open(self._reference_path).convert("RGB").resize((512, 512))
            self._ref_pil = img
            self._ref_face = img
            return
        # Legacy mediapipe face_mesh API was dropped in mediapipe 0.10.x;
        # the cropping path needs the new Tasks API. Not implemented yet.
        raise NotImplementedError(
            "legacy mediapipe face_mesh API not available; use precropped=True"
        )

    def render_batch(self, b58: np.ndarray, ypr: np.ndarray) -> np.ndarray:
        """b58: (T,58) float32; ypr: (T,3) float32 radians.

        T must be a positive multiple of 4. Returns (T, 512, 512, 3) uint8 RGB.
        Each call rebuilds a fresh torch.Generator with the configured seed,
        so identical (b58, ypr) inputs produce bit-identical output across
        calls — required by the daemon's resync semantics.
        """
        assert self._pipe is not None, "BatchDriver.start() not called"
        assert self._student is not None, "BatchDriver.start() not called"
        assert self._ref_face is not None and self._ref_pil is not None
        T = b58.shape[0]
        assert T % 4 == 0 and T >= 4, f"T must be positive multiple of 4 (got {T})"
        assert ypr.shape == (T, 3), f"ypr shape {ypr.shape} != ({T}, 3)"
        assert b58.shape == (T, 58), f"b58 shape {b58.shape} != ({T}, 58)"
        # padding_num=12 (= (4-1)*4). Below that, the tail-mirror code path
        # below would need to wrap content cyclically; we have not validated
        # output for those tiny batches. V1 daemon batches at 24; lift this
        # when V2 cohort-stream is wired.
        assert T >= 13, f"T must be >= 13 (padding_num+1) for V1 tail-mirror; got {T}"

        # The pipe extends tgt_images internally by padding_num reversed-tail
        # copies (pipeline_pose2vid.py:845), then asks the seam for indices
        # up to T + padding_num. Mirror that padding here so provider() never
        # returns an empty slice.
        padding_num = (4 - 1) * 4  # (temporal_adaptive_step - 1) * temporal_window_size
        # T >= padding_num + 1 enforced above, so the slice is exactly
        # padding_num rows — no resize/wrap needed.
        tail_b = b58[-padding_num - 1:-1][::-1]
        tail_y = ypr[-padding_num - 1:-1][::-1]
        b58_local = np.concatenate([b58, tail_b], axis=0)
        ypr_local = np.concatenate([ypr, tail_y], axis=0)
        N_total = b58_local.shape[0]

        def provider(start, n):
            s = max(0, min(start, N_total))
            e = max(s, min(start + n, N_total))
            return b58_local[s:e], ypr_local[s:e]

        install_arkit_seams(
            self._pipe,
            b_seq=None,
            ypr_seq=None,
            student=self._student,
            device=self._device,
            dtype=self._dtype,
            patch_pose=True,
            patch_motion=True,
            provider=provider,
        )

        # Stand-in driving images: identical copies of the reference face.
        # The patched seams ignore content, but the pipe still calls
        # preprocessing on these tensors.
        stand_in_pose = [self._ref_face] * T
        stand_in_faces = [self._ref_face] * T

        gen = torch.Generator(device=self._device)
        gen.manual_seed(self._seed)

        # Patch torch.tensor to swap the hardcoded [999, 666, 333, 0] schedule
        # for our N-step trailing DDIM. Strict literal match — only the
        # exact PersonaLive 4-step sentinel triggers replacement. Restored in
        # finally so other code in this venv stays unaffected.
        _orig = torch.tensor
        SENTINEL = [999, 666, 333, 0]
        SCHEDULE = self._SCHEDULE
        def _patched_tensor(data, *a, **kw):
            if isinstance(data, list) and data == SENTINEL:
                return _orig(SCHEDULE, *a, **kw)
            return _orig(data, *a, **kw)
        torch.tensor = _patched_tensor  # type: ignore[assignment]
        try:
            out = self._pipe(
                stand_in_pose, self._ref_pil, stand_in_faces, self._ref_face,
                512, 512, T,
                num_inference_steps=self._num_inference_steps,
                guidance_scale=self._guidance_scale,
                generator=gen,
                temporal_window_size=4,
                temporal_adaptive_step=4,
            )
        finally:
            torch.tensor = _orig  # type: ignore[assignment]
        video = out.videos  # (1, 3, T, H, W) float in [0, 1]
        arr = video[0].permute(1, 2, 3, 0).cpu().float().numpy()
        return (arr * 255.0).clip(0, 255).astype(np.uint8)

    def _prepare_for_call(self, b58: np.ndarray, ypr: np.ndarray):
        """Common to V1 and V2: validate, tail-mirror pad, install seams,
        build stand-in lists + RNG. Returns (T, stand_in_pose, stand_in_faces, gen)."""
        assert self._pipe is not None, "BatchDriver.start() not called"
        assert self._student is not None, "BatchDriver.start() not called"
        assert self._ref_face is not None and self._ref_pil is not None
        T = b58.shape[0]
        assert T % 4 == 0 and T >= 4, f"T must be positive multiple of 4 (got {T})"
        assert ypr.shape == (T, 3), f"ypr shape {ypr.shape} != ({T}, 3)"
        assert b58.shape == (T, 58), f"b58 shape {b58.shape} != ({T}, 58)"
        assert T >= 13, f"T must be >= 13 (padding_num+1) for tail-mirror; got {T}"

        padding_num = (4 - 1) * 4
        tail_b = b58[-padding_num - 1:-1][::-1]
        tail_y = ypr[-padding_num - 1:-1][::-1]
        b58_local = np.concatenate([b58, tail_b], axis=0)
        ypr_local = np.concatenate([ypr, tail_y], axis=0)
        N_total = b58_local.shape[0]

        def provider(start, n):
            s = max(0, min(start, N_total))
            e = max(s, min(start + n, N_total))
            return b58_local[s:e], ypr_local[s:e]

        install_arkit_seams(
            self._pipe, b_seq=None, ypr_seq=None,
            student=self._student, device=self._device, dtype=self._dtype,
            patch_pose=True, patch_motion=True, provider=provider,
        )

        stand_in_pose = [self._ref_face] * T
        stand_in_faces = [self._ref_face] * T
        gen = torch.Generator(device=self._device)
        gen.manual_seed(self._seed)
        return T, stand_in_pose, stand_in_faces, gen

    def prepare_v2(self, b58: np.ndarray, ypr: np.ndarray) -> int:
        """V2 cohort-stream prepare. Calls vendored ``pipe.prepare(...)`` and
        runs the warmup steps internally so ``step_v2()`` always emits frames.

        Returns the number of *productive* step_v2 calls the caller should
        make (= ``windows = T // temporal_window_size``).
        """
        T, stand_in_pose, stand_in_faces, gen = self._prepare_for_call(b58, ypr)

        _orig = torch.tensor
        SENTINEL = [999, 666, 333, 0]
        SCHEDULE = self._SCHEDULE

        def _patched_tensor(data, *a, **kw):
            if isinstance(data, list) and data == SENTINEL:
                return _orig(SCHEDULE, *a, **kw)
            return _orig(data, *a, **kw)

        torch.tensor = _patched_tensor  # type: ignore[assignment]
        try:
            self._pipe.prepare(
                stand_in_pose, self._ref_pil, stand_in_faces, self._ref_face,
                512, 512, T,
                num_inference_steps=self._num_inference_steps,
                guidance_scale=self._guidance_scale,
                generator=gen,
                temporal_window_size=4,
                temporal_adaptive_step=4,
            )
            # Warmup: temporal_adaptive_step - 1 = 3 cohort iters that
            # don't emit decoded frames. Run them now so each step_v2()
            # produces exactly 4 decoded frames.
            for _ in range(4 - 1):
                self._pipe.step()
        finally:
            torch.tensor = _orig  # type: ignore[assignment]

        self._v2_windows = T // 4
        self._v2_emitted = 0
        # Track frames already decoded so step_v2 can index into the list.
        self._v2_decode_cursor = len(self._pipe._stream_final_videos)
        return self._v2_windows

    def step_v2(self, n: int = 4) -> np.ndarray:
        """V2 cohort step: one productive cohort -> (n, 512, 512, 3) uint8.

        ``n`` must equal ``temporal_window_size`` (4); the kwarg exists to
        make the streaming daemon's intent explicit.
        """
        assert n == 4, f"V2 step emits exactly 4 frames per cohort; got {n}"
        assert self._pipe is not None
        assert self._v2_emitted < self._v2_windows, (
            f"V2 cohort drained ({self._v2_emitted}/{self._v2_windows}); "
            "call prepare_v2 again before more steps"
        )

        _orig = torch.tensor
        SENTINEL = [999, 666, 333, 0]
        SCHEDULE = self._SCHEDULE

        def _patched_tensor(data, *a, **kw):
            if isinstance(data, list) and data == SENTINEL:
                return _orig(SCHEDULE, *a, **kw)
            return _orig(data, *a, **kw)

        torch.tensor = _patched_tensor  # type: ignore[assignment]
        try:
            self._pipe.step()
        finally:
            torch.tensor = _orig  # type: ignore[assignment]

        # The vendored step() appends a (1, 3, 4, 512, 512) tensor to
        # _stream_final_videos when i > temporal_adaptive_step - 2. After
        # warmup, every call appends one block.
        block = self._pipe._stream_final_videos[self._v2_decode_cursor]
        self._v2_decode_cursor += 1
        self._v2_emitted += 1
        arr = block[0].permute(1, 2, 3, 0).cpu().float().numpy()
        return (arr * 255.0).clip(0, 255).astype(np.uint8)

    def stop(self) -> None:
        self._pipe = None
        self._student = None
        self._ref_pil = None
        self._ref_face = None
        # Force collection of pipe + sub-modules + closures before asking
        # CUDA to release cached blocks. Without gc.collect() the freeing is
        # deferred to whenever the cyclic collector next runs.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
