"""Extract (b_expr, m_f) pairs by walking takes through the real
PersonaLive pipe with the diffusion UNet stubbed out.

Why: motion_encoder uses temporal context across 4-frame chunks and the
pipe applies preprocessing (StabilizedFaceCropper, resize, normalisation)
that we cannot easily replicate by calling motion_encoder directly. The
v1 corpus extracted by calling motion_encoder on single 224 crops
produced m_f orthogonal (cos≈0.01) to what the pipe emits at inference
on the same b_61. Re-extracting via the pipe path eliminates this gap.

UNet is replaced by a no-op so we pay only motion_encoder + CLIP + VAE
encode + pose_encoder cost per chunk. CLIP/VAE encode the reference
once. Per-chunk compute reduces from ~0.55s (4-step UNet) to a few ms.

For each take, walks frames in 60-frame batches (matching apply_bridge),
patches motion_encoder.forward to record m_f, runs pipe.__call__, and
splits the saved (1, T, 32, 16) tensor back into per-frame pkls aligned
with the sampled ARKit indices.
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PL = Path(os.path.expanduser("~/w/PersonaLive"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(PL))

from arkit_bridge.llf_csv import load_llf_b61                       # noqa: E402


def _build_pipe(device, dtype):
    os.chdir(PL)
    from omegaconf import OmegaConf
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
        cfg.pretrained_base_model_path, subfolder="unet").to(device, dtype=dtype)
    den_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.pretrained_base_model_path, "", subfolder="unet",
        unet_additional_kwargs=infer.unet_additional_kwargs,
    ).to(device, dtype=dtype)
    me = MotEncoder().to(device, dtype=dtype).eval()
    pg = PoseGuider().to(device, dtype=dtype)
    pe = MotionExtractor(num_kp=21).to(device, dtype=dtype).eval()
    img_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path).to(device, dtype=dtype)
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


def _stub_denoising_unet(pipe):
    """Replace UNet3DConditionModel.forward with a zero-returning no-op so
    the pipe's diffusion loop completes instantly. We only care about
    motion_encoder side effects (mf_log captures the output)."""
    orig_forward = pipe.denoising_unet.forward
    class _StubOut(tuple):
        def __new__(cls, sample):
            obj = super().__new__(cls, (sample,))
            obj.sample = sample
            return obj
    def stub(self, sample, *args, **kwargs):
        return _StubOut(torch.zeros_like(sample))
    pipe.denoising_unet.forward = MethodType(stub, pipe.denoising_unet)
    return orig_forward


def _install_capture(pipe, mf_log):
    """Patch motion_encoder.forward to log every output (keep ref + driving)."""
    real_forward = pipe.motion_encoder.forward
    def capture(self, x):
        out = real_forward(x)
        mf_log["records"].append({
            "T": int(x.shape[2]),
            "mf": out.detach().cpu().float().numpy(),
        })
        return out
    pipe.motion_encoder.forward = MethodType(capture, pipe.motion_encoder)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--takes_root", default="data/llf-takes",
                    help="root containing 20260505_MySlate_<n> dirs")
    ap.add_argument("--takes", nargs="+", type=int, required=True)
    ap.add_argument("--out_dir", default="data/arkit_bridge_pairs/all")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--batch_frames", type=int, default=60,
                    help="ARKit frames per pipe call (must be multiple of 4)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_frames_per_take", type=int, default=0)
    args = ap.parse_args()
    args.reference = str(Path(args.reference).resolve())
    args.takes_root = str(Path(args.takes_root).resolve())
    args.out_dir = str(Path(args.out_dir).resolve())

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    device = args.device
    dtype = torch.float16
    pipe = _build_pipe(device, dtype)
    print("pipe built", flush=True)

    # face_mesh + cropper (per render_take.py / apply_bridge)
    import cv2
    import mediapipe as mp
    from src.utils.util import crop_face, StabilizedFaceCropper
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1)
    ref_pil = Image.open(args.reference).convert("RGB")
    ref_face = Image.fromarray(crop_face(ref_pil, face_mesh)).convert("RGB")

    ref_kept = None  # to skip stubbed UNet output, reuse first call's tensor

    for take_n in args.takes:
        take_dir = Path(args.takes_root) / f"20260505_MySlate_{take_n}"
        take_name = take_dir.name
        mov = next(take_dir.glob("*_iPhone.mov"))
        csv = next(take_dir.glob("*_iPhone.csv"))
        b_all = load_llf_b61(csv)
        n_total = len(b_all)
        print(f"\n=== take {take_n}: {n_total} ARKit rows ===", flush=True)

        cap = cv2.VideoCapture(str(mov))
        if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        n_frames_in_mov = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n_have = min(n_frames_in_mov, n_total)
        if args.max_frames_per_take and args.max_frames_per_take < n_have:
            n_have = args.max_frames_per_take
        # sampled indices at stride
        idxs_all = list(range(0, n_have, args.stride))
        # process in chunks of batch_frames
        cropper = StabilizedFaceCropper(
            strategy="ema", ema_alpha=0.2, forehead_bias_frac=0.10,
            face_mesh=face_mesh,
        )
        t0 = time.time()
        n_kept = 0
        cur_idx = 0  # next frame index cap.read() will return
        for chunk_start in range(0, len(idxs_all), args.batch_frames):
            chunk_idxs = idxs_all[chunk_start:chunk_start + args.batch_frames]
            # need multiple of 4
            L = (len(chunk_idxs) // 4) * 4
            if L < 4: continue
            chunk_idxs = chunk_idxs[:L]

            # Quick skip if all output pkls already exist for this chunk.
            all_exist = all(
                (Path(args.out_dir) / f"{take_name}_frame_{fi:06d}.pkl").exists()
                for fi in chunk_idxs)
            if all_exist:
                # still need to advance cap so cur_idx tracks
                while cur_idx <= chunk_idxs[-1]:
                    ok, _ = cap.read()
                    if not ok: break
                    cur_idx += 1
                continue

            # Read only the frames this chunk needs (sequential forward).
            wanted = set(chunk_idxs)
            target_max = chunk_idxs[-1]
            chunk_frames = {}
            while cur_idx <= target_max:
                ok, f = cap.read()
                if not ok: break
                if cur_idx in wanted:
                    chunk_frames[cur_idx] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                cur_idx += 1
            if len(chunk_frames) != L:
                continue
            ori_pose_images = [Image.fromarray(chunk_frames[fi]).convert("RGB")
                               for fi in chunk_idxs]
            dri_faces = []
            last = None
            for pil in ori_pose_images:
                try: c = cropper(pil); last = c
                except (TypeError, IndexError): c = last
                if c is None:
                    continue
                dri_faces.append(Image.fromarray(c).convert("RGB"))
            if len(dri_faces) != L:
                # alignment broken; drop
                continue

            mf_log = {"records": []}
            _install_capture(pipe, mf_log)
            gen = torch.Generator(device=device); gen.manual_seed(42)
            try:
                _ = pipe(
                    ori_pose_images, ref_pil, dri_faces, ref_face,
                    512, 512, L,
                    num_inference_steps=4, guidance_scale=1.0, generator=gen,
                    temporal_window_size=4, temporal_adaptive_step=4,
                ).videos
            except Exception as e:
                print(f"chunk start={chunk_start} pipe error: {e}", flush=True)
                continue

            # Stream pipe calls motion_encoder once per 4-frame window
            # (T=4 records). With temporal_adaptive_step=4 and L=60, the loop
            # iterates 18 windows so we get 18 T=4 records (= 72 frames; last
            # 12 are reverse-padding the pipe adds for lookahead). The ref
            # call gives T=1. Concatenate all T=4 records in order and keep
            # the first L = real-driving frames.
            drv_records = [r for r in mf_log["records"] if r["T"] == 4]
            if len(drv_records) * 4 < L:
                print(f"chunk start={chunk_start}: only {len(drv_records)} T=4 records, need >= {L//4}", flush=True)
                continue
            mf_concat = np.concatenate([r["mf"] for r in drv_records], axis=1)  # (1, >=L, 32, 16)
            mf_per_frame = mf_concat[0, :L]  # (L, 32, 16)
            assert mf_per_frame.shape == (L, 32, 16), f"unexpected mf shape {mf_per_frame.shape}"

            for k, fi in enumerate(chunk_idxs):
                out_path = Path(args.out_dir) / f"{take_name}_frame_{fi:06d}.pkl"
                if out_path.exists():
                    continue
                b_expr = np.concatenate(
                    [b_all[fi, :52], b_all[fi, 55:61]], axis=0
                ).astype(np.float32)
                with open(out_path, "wb") as f:
                    pickle.dump({
                        "b_expr": b_expr,
                        "m_f": mf_per_frame[k:k+1].astype(np.float16),
                        "frame_idx": int(fi),
                    }, f)
                n_kept += 1
            elapsed = time.time() - t0
            done = (chunk_start + L)
            rate = done / max(1e-3, elapsed)
            print(f"  take {take_n}: {done}/{len(idxs_all)*args.stride} frames "
                  f"({rate:.1f} f/s, kept={n_kept})", flush=True)
        cap.release()
        print(f"take {take_n} done: kept={n_kept} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
