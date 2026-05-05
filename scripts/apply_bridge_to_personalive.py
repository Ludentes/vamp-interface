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
from types import MethodType

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
PL = Path(os.path.expanduser("~/w/PersonaLive"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(PL))

from arkit_bridge.closed_form_pose import (
    EULER_SIGNS, euler_to_rotmat, compose_kd,
)
from arkit_bridge.llf_csv import load_llf_b61
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


def install_arkit_seams(pipe, b_seq, ypr_seq, student, device, dtype):
    """Patch pipe.pose_encoder + pipe.motion_encoder to consume ARKit data.

    b_seq: (T, 58) float32 cpu — chronological per driving frame
    ypr_seq: (T, 3) float32 — head (yaw, pitch, roll) radians
    student: trained MotEncoderStudent on device, dtype=float32
    """
    # State threaded through closures.
    ref_kp_canonical = {"kp": None, "t": None, "scale": None}
    chunk_cursor = {"i": 0}

    real_pe_get_kp = pipe.pose_encoder.get_kp  # for ref pose-vector sampling

    def cf_kd_for_indices(indices):
        """Run closed-form compose_kd for a list of frame indices into b_seq."""
        kp_ref = ref_kp_canonical["kp"]   # (1, 21, 3) on device
        t_ref = ref_kp_canonical["t"]      # (1, 3)
        s_ref = ref_kp_canonical["scale"]  # (1, 1)
        sy, sp, sr = EULER_SIGNS
        ypr = ypr_seq[indices]  # (T, 3)
        T = ypr.shape[0]
        Rs = []
        for k in range(T):
            R = euler_to_rotmat(
                torch.tensor(sy * ypr[k, 0].item()),
                torch.tensor(sp * ypr[k, 1].item()),
                torch.tensor(sr * ypr[k, 2].item()),
            )
            Rs.append(R)
        R = torch.stack(Rs, dim=0).to(device=device, dtype=dtype)  # (T, 3, 3)
        kp_ref_T = kp_ref.expand(T, -1, -1).to(device=device, dtype=dtype)
        s_T = s_ref.expand(T, -1).to(device=device, dtype=dtype)
        t_T = t_ref.expand(T, -1).to(device=device, dtype=dtype)
        return compose_kd(kp_ref_T, R, s_T, t_T)  # (T, 21, 3)

    def patched_interpolate_kps_online(self, ref, motion, num_interp, t_scale=0.5, s_scale=0):
        # Compute the canonical reference once from the actual ref RGB.
        kp1 = self.detector(ref.to(self.dtype))
        ref_kp_canonical["kp"] = kp1["kp"].reshape(1, -1, 3).detach()
        ref_kp_canonical["t"] = kp1["t"].detach()
        ref_kp_canonical["scale"] = kp1["scale"].detach()

        # Driving: motion has padding_num+1 frames; map to ARKit indices
        # The first chunk receives padding_num+1 stand-in frames; we want
        # the *last* of those to be "frame 0" of our ARKit sequence and
        # the preceding (num_interp) to interpolate from ref pose to it.
        # PersonaLive's interpolate_tensors returns num-1 elements (drops
        # last), so total output is (num_interp-1) + motion.shape[0].
        idxs = np.array([0] * (num_interp - 1) + list(range(motion.shape[0])))
        kp_intrep = cf_kd_for_indices(idxs)  # (n, 21, 3)
        # Real method also returns (kp_intrep, kp1, kp_frame1, kp_dri); the
        # consumers only use kp_intrep (line 873), kp_ref/kp_frame1 (line
        # 871 next iter via get_kps). Pass through real kp1/kp_frame1 dicts
        # so subsequent get_kps still has expected pitch/yaw/roll keys.
        kp_frame1 = self.detector(motion[:1].to(self.dtype))
        chunk_cursor["i"] = motion.shape[0]
        return kp_intrep, kp1, kp_frame1, None

    def patched_get_kps(self, kp_ref, kp_frame1, motion, t_scale=0.5, s_scale=0):
        start = chunk_cursor["i"]
        n = motion.shape[0]
        idxs = np.arange(start, start + n)
        idxs = np.clip(idxs, 0, len(b_seq) - 1)
        kp_d = cf_kd_for_indices(idxs)
        chunk_cursor["i"] += n
        return kp_d, None

    pipe.pose_encoder.interpolate_kps_online = MethodType(
        patched_interpolate_kps_online, pipe.pose_encoder
    )
    pipe.pose_encoder.get_kps = MethodType(
        patched_get_kps, pipe.pose_encoder
    )

    # motion_encoder seam: dispatch ref vs driving by time-dim.
    real_me_call = pipe.motion_encoder.__call__
    real_me_forward = pipe.motion_encoder.forward
    me_cursor = {"i": 0}

    def patched_me_forward(self, x):
        # x shape: (B, C, T, H, W). T==1 -> reference path; T>=2 -> driving.
        T = x.shape[2]
        if T == 1:
            return real_me_forward(x)
        # Driving: pull T b_expr from b_seq starting at me_cursor.
        start = me_cursor["i"]
        idxs = np.arange(start, start + T)
        idxs = np.clip(idxs, 0, len(b_seq) - 1)
        b = torch.from_numpy(b_seq[idxs].astype(np.float32)).to(device)
        with torch.no_grad():
            mf = student(b)  # (T, 1, 32, 16)
        # Real motion_encoder returns (B=1, T, 32, 16).
        mf = mf.squeeze(1).unsqueeze(0)  # (1, T, 32, 16)
        me_cursor["i"] += T
        return mf.to(dtype=self.dtype)

    pipe.motion_encoder.forward = MethodType(patched_me_forward, pipe.motion_encoder)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="reference RGB image")
    ap.add_argument("--take_dir", required=True, help="LLF take dir for ARKit b_61 + RGB stand-ins")
    ap.add_argument("--ckpt", required=True, help="MotEncoderStudent .pt")
    ap.add_argument("--out_path", required=True, help="output mp4")
    ap.add_argument("--n_frames", type=int, default=24)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    # Resolve to absolute *before* build_pipe chdir's into PersonaLive.
    args.reference = str(Path(args.reference).resolve())
    args.take_dir = str(Path(args.take_dir).resolve())
    args.ckpt = str(Path(args.ckpt).resolve())
    args.out_path = str(Path(args.out_path).resolve())

    device = args.device
    dtype = torch.float16

    take = Path(args.take_dir)
    csv = next(take.glob("*_iPhone.csv"))
    mov = next(take.glob("*_iPhone.mov"))
    b_all = load_llf_b61(csv)  # (N, 61)
    # b_expr layout: [0:52]=blendshapes, [52:55]=LeftEye yaw/pitch/roll,
    # [55:58]=RightEye yaw/pitch/roll. Source CSV b_all[:, 52]=HeadYaw,
    # b_all[:, 53]=HeadPitch, b_all[:, 54]=HeadRoll, b_all[:, 55:58]=LE,
    # b_all[:, 58:61]=RE.
    idxs = np.arange(args.n_frames) * args.stride
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
            raw_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            last_ok = raw_frames[-1]
        cap_idx += 1
    cap.release()
    while len(raw_frames) < n:
        raw_frames.append(last_ok if last_ok is not None else np.zeros((720, 1280, 3), np.uint8))
    print(f"loaded {len(raw_frames)} RGB stand-in frames", flush=True)

    # PIL conversions for pipe API
    from src.utils.util import crop_face
    import mediapipe as mp
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    ref_pil = Image.open(args.reference).convert("RGB")
    ref_face = Image.fromarray(crop_face(ref_pil, face_mesh)).convert("RGB")

    ori_pose_images = [Image.fromarray(f).convert("RGB") for f in raw_frames]
    dri_faces = []
    last_crop = None
    for pil in ori_pose_images:
        try:
            crop = crop_face(pil, face_mesh)
            last_crop = crop
        except (TypeError, IndexError):
            crop = last_crop if last_crop is not None else np.array(pil)
        dri_faces.append(Image.fromarray(crop).convert("RGB"))
    # Patch in any leading misses with the first successful crop.
    if last_crop is None:
        raise RuntimeError("face_mesh found no faces in any driving frame")
    for i, pil in enumerate(dri_faces):
        if pil.size == ori_pose_images[i].size and last_crop is not None:
            # placeholder (full frame) — replace with last_crop
            dri_faces[i] = Image.fromarray(last_crop).convert("RGB")

    install_arkit_seams(pipe, b_seq, ypr_seq, student, device, dtype)
    print("seams installed; running pipe()", flush=True)

    gen = torch.Generator(device=device); gen.manual_seed(42)
    L = (n // 4) * 4
    if L < 4:
        raise RuntimeError(f"need >=4 frames after //4 trim; got {L}")
    out = pipe(
        ori_pose_images[:L], ref_pil, dri_faces[:L], ref_face,
        512, 512, L,
        num_inference_steps=4, guidance_scale=1.0, generator=gen,
        temporal_window_size=4, temporal_adaptive_step=4,
    ).videos

    from src.utils.util import save_videos_grid
    out_path = Path(args.out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    save_videos_grid(out, str(out_path), n_rows=1, fps=25)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
