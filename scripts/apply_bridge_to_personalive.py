"""Drop-in PersonaLive smoke render driven entirely by ARKit b_61.

Replaces two seams inside Pose2VideoPipeline_Stream:
  - pose_encoder.interpolate_kps_online / get_kps -> closed-form k_d from
    ARKit (yaw, pitch, roll) using EULER_SIGNS calibration.
  - motion_encoder driving-side calls -> MotEncoderStudent(b_expr).
    Reference-side calls (single-frame, used for neg_motion_hidden_states)
    still go through the real motion_encoder so identity is preserved.
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
PL = Path(os.path.expanduser("~/w/PersonaLive"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(PL))

from arkit_bridge.llf_csv import load_llf_b61
from arkit_bridge.seam_install import install_arkit_seams
from arkit_bridge.student import MotEncoderStudent


def build_pipe(device, dtype):
    """Mirror inference_offline.py's pipeline construction."""
    os.chdir(PL)  # configs use relative paths
    from diffusers import AutoencoderKL
    from transformers import CLIPVisionModelWithProjection
    from src.scheduler.scheduler_ddim import DDIMScheduler
    from src.models.unet_2d_condition import UNet2DConditionModel
    from src.models.unet_3d import UNet3DConditionModel
    from src.pipelines.pipeline_pose2vid import Pose2VideoPipeline_Stream
    from src.models.motion_encoder.encoder import MotEncoder
    from src.liveportrait.motion_extractor import MotionExtractor
    from src.models.pose_guider import PoseGuider

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
    ref_unet.load_state_dict(torch.load(base.replace("denoising_unet", "reference_unet"), map_location="cpu"), strict=True)
    me.load_state_dict(torch.load(base.replace("denoising_unet", "motion_encoder"), map_location="cpu"), strict=True)
    pg.load_state_dict(torch.load(base.replace("denoising_unet", "pose_guider"), map_location="cpu"), strict=True)
    den_unet.load_state_dict(torch.load(base.replace("denoising_unet", "temporal_module"), map_location="cpu"), strict=False)
    pe.load_state_dict(torch.load(base.replace("denoising_unet", "motion_extractor"), map_location="cpu"), strict=False)

    pipe = Pose2VideoPipeline_Stream(
        vae=vae, image_encoder=img_enc,
        reference_unet=ref_unet, denoising_unet=den_unet,
        motion_encoder=me, pose_encoder=pe, pose_guider=pg, scheduler=sched,
    ).to(device)
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="reference RGB image")
    ap.add_argument("--reference_clip", default=None,
                    help="Optional second reference image used ONLY for the "
                         "CLIP-image global embed (the encoder_hidden_states "
                         "cross-attended at every step). The spatial channel "
                         "(reference_unet writer + init_latents) keeps using "
                         "--reference. Decouples the two image-conditioning "
                         "paths to test whether style transfer can ride the "
                         "CLIP channel while RefNet anchors anatomy. "
                         "(2026-05-06 decoupled-channel probe.)")
    ap.add_argument("--take_dir", required=True, help="LLF take dir for ARKit b_61 + RGB stand-ins")
    ap.add_argument("--ckpt", required=True, help="MotEncoderStudent .pt")
    ap.add_argument("--out_path", required=True, help="output mp4")
    ap.add_argument("--n_frames", type=int, default=24)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--start_frame", type=int, default=0,
                    help="ARKit-frame offset into the take")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", default="bridge",
                    choices=["bridge", "teacher_motion", "teacher_full"],
                    help="bridge: student m_f + closed-form pose. "
                         "teacher_motion: real motion_encoder + closed-form pose. "
                         "teacher_full: vanilla PersonaLive.")
    ap.add_argument("--save_mf", default=None,
                    help="path to .npz to save per-batch m_f records (optional)")
    ap.add_argument("--rotate_iphone", action="store_true", default=True,
                    help="rotate driving MOV frames 90° CW (iPhone Live Link "
                         "writes rotation=-90 metadata that cv2.VideoCapture "
                         "ignores, so frames arrive sideways)")
    ap.add_argument("--no_rotate_iphone", dest="rotate_iphone",
                    action="store_false")
    ap.add_argument("--reference_precropped", action="store_true",
                    help="Skip MediaPipe FaceMesh on the reference. Use for "
                         "stylized / non-human anchors (anime, cartoon, "
                         "creature) where face_mesh returns no landmarks. "
                         "Image must already be a square crop centred on the "
                         "head; will be resized to 512².")
    ap.add_argument("--crop_strategy", default="ema",
                    choices=["perframe", "nocrop", "ema"],
                    help="EMA stabilises bbox across frames; matches render_take.py")
    ap.add_argument("--ema_alpha", type=float, default=0.2)
    ap.add_argument("--forehead_bias", type=float, default=0.10)
    ap.add_argument("--euler_signs", default=None,
                    help="Override closed-form pose Euler signs as 'sy,sp,sr' "
                         "(e.g. '-1,-1,-1'). Default: use compiled-in EULER_SIGNS.")
    ap.add_argument("--lora_path", default=None,
                    help="Optional Kohya-format SD1.5 LoRA .safetensors to merge "
                         "into UNet attention layers before rendering.")
    ap.add_argument("--lora_alpha", type=float, default=1.0,
                    help="Scalar multiplier on top of the LoRA's own alpha/rank "
                         "scaling. 1.0 = baseline strength.")
    ap.add_argument("--lora_targets", default="den",
                    choices=["den", "ref", "both"],
                    help="Which UNet(s) to merge the LoRA into. 'den' = "
                         "denoising_unet only (pass-1 hypothesis test). "
                         "'ref' = reference_unet only. 'both' = both.")
    ap.add_argument("--num_inference_steps", type=int, default=4,
                    help="Diffusion step count. PersonaLive ships at 4 (distilled). "
                         "Higher counts give off-the-shelf SD1.5 LoRAs more budget "
                         "to fire but cost FPS proportionally. Must be divisible "
                         "by --temporal_adaptive_step.")
    ap.add_argument("--temporal_adaptive_step", type=int, default=4,
                    help="Must divide num_inference_steps. Default 4 matches "
                         "PersonaLive ship config.")
    args = ap.parse_args()
    # Resolve to absolute *before* build_pipe chdir's into PersonaLive.
    args.reference = str(Path(args.reference).resolve())
    args.take_dir = str(Path(args.take_dir).resolve())
    args.ckpt = str(Path(args.ckpt).resolve())
    args.out_path = str(Path(args.out_path).resolve())
    if args.lora_path:
        args.lora_path = str(Path(args.lora_path).resolve())
    if args.reference_clip:
        args.reference_clip = str(Path(args.reference_clip).resolve())

    device = args.device
    dtype = torch.float16

    take = Path(args.take_dir)
    csv = next(take.glob("*_iPhone.csv"))
    # Accept both raw .mov (data/llf-takes/) and reencoded .mp4
    # (data/llf-takes-small/, data/llf-clips-auto/) — same take_dir contract.
    mov = next((p for ext in ("mov", "mp4") for p in take.glob(f"*_iPhone.{ext}")), None)
    if mov is None:
        raise FileNotFoundError(f"no *_iPhone.{{mov,mp4}} in {take}")
    b_all = load_llf_b61(csv)  # (N, 61)
    # b_expr layout: [0:52]=blendshapes, [52:55]=LeftEye yaw/pitch/roll,
    # [55:58]=RightEye yaw/pitch/roll. Source CSV b_all[:, 52]=HeadYaw,
    # b_all[:, 53]=HeadPitch, b_all[:, 54]=HeadRoll, b_all[:, 55:58]=LE,
    # b_all[:, 58:61]=RE.
    idxs = args.start_frame + np.arange(args.n_frames) * args.stride
    idxs = idxs[idxs < len(b_all)]
    b_seq = np.concatenate([b_all[idxs, :52], b_all[idxs, 55:61]], axis=1).astype(np.float32)
    ypr_seq = b_all[idxs, 52:55].astype(np.float32)
    n = len(idxs)
    print(f"loaded {n} ARKit frames from {csv.name}", flush=True)

    student = MotEncoderStudent().to(device).eval()
    student.load_state_dict(torch.load(args.ckpt, map_location=device))
    print(f"loaded student from {args.ckpt}", flush=True)

    pipe = build_pipe(device, dtype)
    print("pipe built; loading driving stand-in RGB frames", flush=True)

    if args.lora_path:
        from arkit_bridge.lora_inject import apply_kohya_lora_to_unet
        # Belt-and-suspenders: assert the attribute names match what we
        # threaded through build_pipe(). (Reviewer I4, 2026-05-06.)
        assert hasattr(pipe, "denoising_unet"), \
            f"pipe missing denoising_unet, has {dir(pipe)}"
        assert hasattr(pipe, "reference_unet"), \
            f"pipe missing reference_unet, has {dir(pipe)}"
        targets = []
        if args.lora_targets in ("den", "both"):
            targets.append(("denoising_unet", pipe.denoising_unet))
        if args.lora_targets in ("ref", "both"):
            targets.append(("reference_unet", pipe.reference_unet))
        for name, unet in targets:
            print(f"[lora] merging {Path(args.lora_path).name} → {name} "
                  f"@ alpha={args.lora_alpha}", flush=True)
            apply_kohya_lora_to_unet(unet, args.lora_path,
                                     alpha=args.lora_alpha, verbose=True)

    if args.reference_clip:
        # Decoupled-channel probe (2026-05-06). Compute image_prompt_embeds
        # from a *different* reference, then short-circuit pipe.image_encoder
        # to return those embeds regardless of input. The pipeline's spatial
        # channel (ref_image_processor → VAE → reference_unet writer +
        # init_latents) is unaffected and still uses --reference.
        from PIL import Image as _PILImage
        clip_ref = _PILImage.open(args.reference_clip).convert("RGB")
        _proc = pipe.clip_image_processor.preprocess(
            clip_ref.resize((224, 224)), return_tensors="pt"
        ).pixel_values
        with torch.no_grad():
            _embeds = pipe.image_encoder(
                _proc.to(device, dtype=pipe.image_encoder.dtype)
            ).image_embeds.detach()

        class _FixedClipEmbeds(torch.nn.Module):
            """Drop-in replacement for pipe.image_encoder; ignores its input
            and returns precomputed embeds. nn.Module so pipe.eval()/.to()
            traversals don't choke. (Reviewer 2026-05-06.)"""
            def __init__(self, embeds, dtype):
                super().__init__()
                self.register_buffer("_embeds", embeds)
                self._cached_dtype = dtype
                self.device = embeds.device
            @property
            def dtype(self):
                return self._cached_dtype
            def forward(self, _x):
                class _Out:
                    pass
                out = _Out()
                out.image_embeds = self._embeds
                return out

        _orig = pipe.image_encoder
        _orig_dtype = _orig.dtype
        pipe.image_encoder = _FixedClipEmbeds(_embeds, _orig_dtype)
        # Diffusers keeps a separate handle in pipe.components — attribute
        # reassignment alone leaves the original encoder GPU-resident and
        # reachable via .components traversal. Evict explicitly.
        try:
            pipe.components.pop("image_encoder", None)
        except Exception:
            pass
        del _orig
        torch.cuda.empty_cache()
        print(f"[decouple] CLIP-image channel ← {Path(args.reference_clip).name} "
              f"(spatial channel still ← {Path(args.reference).name})",
              flush=True)

    # Build RGB stand-in frames from the same MOV — pipe needs them for
    # cond_image_processor.preprocess (the seam is downstream of preprocessing
    # but we replace the actual model calls). Frame content is ignored by
    # patched seams.
    import cv2
    cap = cv2.VideoCapture(str(mov))
    raw_frames = []
    cap_idx = 0
    fetch_set = set(int(i) for i in idxs)
    last_ok = None
    while cap_idx <= max(idxs):
        ok, frame = cap.read()
        if not ok:
            break
        if cap_idx in fetch_set:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if args.rotate_iphone:
                # Live Link Face MOVs carry rotation=-90 metadata that
                # cv2.VideoCapture ignores. ffmpeg-transcoded clips
                # (data/llf-clips/*) come out upright; reading the raw
                # MOV needs a 90° CW rotation to get a portrait face.
                rgb = np.rot90(rgb, k=-1).copy()
            raw_frames.append(rgb)
            last_ok = raw_frames[-1]
        cap_idx += 1
    cap.release()
    while len(raw_frames) < n:
        raw_frames.append(last_ok if last_ok is not None else np.zeros((720, 1280, 3), np.uint8))
    print(f"loaded {len(raw_frames)} RGB stand-in frames", flush=True)

    # PIL conversions for pipe API. Use StabilizedFaceCropper (ema) per
    # render_take.py to avoid per-frame crop wobble.
    from src.utils.util import crop_face, StabilizedFaceCropper
    import mediapipe as mp
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    ref_pil = Image.open(args.reference).convert("RGB")
    if args.reference_precropped:
        # Stylized/non-human anchor: face_mesh fails. Trust the operator's
        # manual head crop; just resize to the canonical 512². crop_face's
        # landmarks are only used to compute the bbox — nothing downstream
        # consumes them on the reference path (verified architecture dive
        # 2026-05-06).
        ref_face = ref_pil.resize((512, 512), Image.LANCZOS).convert("RGB")
    else:
        ref_face = Image.fromarray(crop_face(ref_pil, face_mesh)).convert("RGB")

    cropper = StabilizedFaceCropper(
        strategy=args.crop_strategy,
        ema_alpha=args.ema_alpha,
        forehead_bias_frac=args.forehead_bias,
        face_mesh=face_mesh,
    )
    ori_pose_images = [Image.fromarray(f).convert("RGB") for f in raw_frames]
    dri_faces = []
    last_crop = None
    for pil in ori_pose_images:
        try:
            crop = cropper(pil)
            last_crop = crop
        except (TypeError, IndexError):
            crop = last_crop if last_crop is not None else np.array(pil)
        dri_faces.append(Image.fromarray(crop).convert("RGB"))
    if last_crop is None:
        raise RuntimeError("face_mesh found no faces in any driving frame")

    mf_log = {"records": []} if args.save_mf else None
    patch_pose = args.mode in ("bridge", "teacher_motion")
    patch_motion = args.mode == "bridge"
    eu_override = None
    if args.euler_signs:
        eu_override = tuple(float(x) for x in args.euler_signs.split(","))
        assert len(eu_override) == 3, "--euler_signs needs 3 comma-separated floats"
        print(f"  euler_signs override: {eu_override}", flush=True)
    install_arkit_seams(
        pipe, b_seq, ypr_seq, student, device, dtype,
        patch_pose=patch_pose, patch_motion=patch_motion, mf_log=mf_log,
        euler_signs=eu_override,
    )
    print(f"seams installed (mode={args.mode}, "
          f"patch_pose={patch_pose}, patch_motion={patch_motion}); running pipe()",
          flush=True)

    gen = torch.Generator(device=device); gen.manual_seed(42)
    L = (n // 4) * 4
    if L < 4:
        raise RuntimeError(f"need >=4 frames after //4 trim; got {L}")

    # Hail-Mary monkey-patch (2026-05-06): override PersonaLive's hardcoded
    # 4-step schedule [999, 666, 333, 0] + set_step_length(333) to vanilla
    # N-step DDIM. The distilled UNet runs at off-anchor timesteps, but the
    # underlying SD1.5 prior may surface cleanly enough that an N-step LoRA
    # budget produces visible style transfer. Auto-engages on != 4 steps.
    _restore_torch_tensor = None
    if args.num_inference_steps != 4:
        N = args.num_inference_steps
        # Vanilla DDIM trailing schedule: linspace from 999 down to 999/N rounded.
        # Matches scheduler.set_timesteps(N) output, so step_length=None gives
        # consistent prev_timestep = t - 1000//N at every step.
        step = 1000 // N
        SCHEDULE = [999 - i * step for i in range(N)]
        # Pre-configure the scheduler. We avoid set_timesteps() because it
        # rebuilds alphas_cumprod-related tensors at fp32 and breaks the
        # add_noise dtype contract with fp16 UNet weights. We only need
        # num_inference_steps populated so the step_length=None fallback
        # computes prev_timestep = t - 1000//N. Other state is unused here
        # because the loop iterates over our injected `timesteps` tensor.
        pipe.scheduler.num_inference_steps = N
        pipe.scheduler.step_length = None
        pipe.scheduler.set_step_length = lambda _x: None
        # scheduler.step() reads alphas_cumprod directly without dtype casting;
        # in the 4-step ship config jump=1 so the fp32 latents leaving step()
        # get cast back to fp16 at the end of the outer iter. With jump>=2 the
        # second inner iteration feeds fp32 into the fp16 UNet → conv_in dies.
        # Cast alphas_cumprod to match the UNet dtype once.
        pipe.scheduler.alphas_cumprod = pipe.scheduler.alphas_cumprod.to(dtype=dtype)
        pipe.scheduler.final_alpha_cumprod = pipe.scheduler.final_alpha_cumprod.to(dtype=dtype)
        # Patch torch.tensor to swap the hardcoded list. Strict pattern match —
        # only the exact `[999, 666, 333, 0]` literal triggers replacement.
        import torch as _t
        _orig_tensor = _t.tensor
        _SENTINEL = [999, 666, 333, 0]
        def _patched_tensor(data, *a, **kw):
            if isinstance(data, list) and data == _SENTINEL:
                return _orig_tensor(SCHEDULE, *a, **kw)
            return _orig_tensor(data, *a, **kw)
        _t.tensor = _patched_tensor
        _restore_torch_tensor = _orig_tensor
        print(f"[hail-mary] N={N} schedule={SCHEDULE} step_length=None", flush=True)

    out = pipe(
        ori_pose_images[:L], ref_pil, dri_faces[:L], ref_face,
        512, 512, L,
        num_inference_steps=args.num_inference_steps, guidance_scale=1.0, generator=gen,
        temporal_window_size=4, temporal_adaptive_step=args.temporal_adaptive_step,
    ).videos

    if _restore_torch_tensor is not None:
        torch.tensor = _restore_torch_tensor

    from src.utils.util import save_videos_grid
    out_path = Path(args.out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    save_videos_grid(out, str(out_path), n_rows=1, fps=25)
    print(f"wrote {out_path}", flush=True)

    if mf_log is not None:
        mf_path = Path(args.save_mf).resolve()
        mf_path.parent.mkdir(parents=True, exist_ok=True)
        # Concatenate driving records along T axis; keep ref records separate.
        drv = [r for r in mf_log["records"] if r["role"] == "driving"]
        ref = [r for r in mf_log["records"] if r["role"] == "ref"]
        drv_mf = np.concatenate([r["mf"] for r in drv], axis=1) if drv else np.zeros((1, 0, 32, 16), np.float32)
        drv_starts = np.array([r["start"] for r in drv], dtype=np.int64)
        ref_mf = (np.concatenate([r["mf"] for r in ref], axis=0)
                  if ref else np.zeros((0, 1, 32, 16), np.float32))
        np.savez(
            mf_path,
            driving_mf=drv_mf,           # (1, T_total, 32, 16)
            driving_starts=drv_starts,   # batch boundaries in driving order
            ref_mf=ref_mf,               # (n_ref_calls, 1, 32, 16)
            mode=args.mode,
            ckpt=args.ckpt if args.mode == "bridge" else "",
            arkit_indices=idxs,
        )
        print(f"wrote {mf_path}  (driving_mf={drv_mf.shape}, ref_mf={ref_mf.shape})",
              flush=True)


if __name__ == "__main__":
    main()
