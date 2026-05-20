"""CFM evaluation: per-step sample collage dump + full held-out metrics.

`dump_samples` is called from the training loop. `metrics` is the held-out
quality probe (ArcFace identity cosine + MediaPipe blendshape cosine, with a
zero-blendshape neutral-control baseline for an expression-control floor).
"""
from __future__ import annotations

import csv
import glob
import hashlib
import io
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image

from arkit_controlnet.cfm.dataset import CfmPairDataset
from arkit_controlnet.cfm.model import (
    pack_latents, unpack_latents, prepare_latent_image_ids, prepare_text_ids,
)
from arkit_controlnet.cfm.train import _build_vae, _load_text_embeds, _vae_encode
from arkit_controlnet.eval_spike import ARKIT_BLENDSHAPE_NAMES
from arkit_controlnet.flame_render import (
    BASIS_CHANNEL_NAMES, deform, mediapipe_to_basis_vector,
    render_landmark_aligned, render,
)
from arkit_controlnet.cfm.precompute import face_crop_resize

CTRL_SIZE = 512
LATENT_H = LATENT_W = CTRL_SIZE // 8
SHIFT = 3.0
FLUX_HF_ID = "black-forest-labs/FLUX.1-dev"
TEXT_EMBEDS_PATH = "output/cfm_precompute/text_embeds.pt"
PRECOMPUTE_DIR = "output/cfm_precompute"


def _sample(model, z_T, sigma_seq, id_tokens, t5_seq, pooled, ctrl_packed,
            img_ids, txt_ids, guidance, cs):
    """Euler integration of the FLUX flow with InfuseNet residuals @ scale `cs`."""
    z = z_T
    for i in range(len(sigma_seq) - 1):
        s = sigma_seq[i]
        s_next = sigma_seq[i + 1]
        eh = torch.cat([id_tokens, t5_seq], dim=1)
        cn_d, cn_s = model.infusenet(
            hidden_states=z, controlnet_cond=ctrl_packed,
            conditioning_scale=cs, encoder_hidden_states=eh,
            pooled_projections=pooled, timestep=s.expand(z.shape[0]),
            img_ids=img_ids, txt_ids=txt_ids, guidance=guidance,
            return_dict=False)
        v = model.flux(
            hidden_states=z, timestep=s.expand(z.shape[0]),
            guidance=guidance, pooled_projections=pooled,
            encoder_hidden_states=eh, txt_ids=txt_ids, img_ids=img_ids,
            controlnet_block_samples=cn_d,
            controlnet_single_block_samples=cn_s,
            return_dict=False)[0]
        z = z + (s_next - s) * v
    return z


def _build_sigma_seq(sample_steps: int, device, dtype) -> torch.Tensor:
    from diffusers import FlowMatchEulerDiscreteScheduler
    sched = FlowMatchEulerDiscreteScheduler(shift=SHIFT)
    sched.set_timesteps(sample_steps, device=device)
    return sched.sigmas.to(device=device, dtype=dtype)


def _decode(vae, z_packed: torch.Tensor) -> torch.Tensor:
    """Unpack + VAE decode. Returns uint8 (B,3,H,W) on CPU."""
    z = unpack_latents(z_packed, LATENT_H * 2, LATENT_W * 2)
    sf = vae.config.scaling_factor
    sh = getattr(vae.config, "shift_factor", 0.0) or 0.0
    with torch.no_grad():
        x = vae.decode((z.to(vae.dtype) / sf) + sh).sample
    x = x.clamp(-1.0, 1.0)
    x = ((x + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).cpu()
    return x


def _load_photo_crop_uint8(sha: str, row, ffhq_sha_index_df: pd.DataFrame,
                           shards: list[str]) -> np.ndarray:
    """Load FFHQ image by sha, face-crop to 512 matching precompute."""
    idx_row = ffhq_sha_index_df[ffhq_sha_index_df.image_sha256 == sha].iloc[0]
    shard_path = shards[int(idx_row.shard_idx)]
    df = pd.read_parquet(shard_path)
    cell = df["image"].iloc[int(idx_row.row_idx)]
    rgb = np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))
    return face_crop_resize(
        rgb,
        bbox_cx=float(row.bbox_cx), bbox_cy=float(row.bbox_cy),
        bbox_w=float(row.bbox_w), bbox_h=float(row.bbox_h),
        out_size=CTRL_SIZE,
    )


def _ffhq_index_and_shards():
    from arkit_controlnet.build_ffhq_index import SHARD_GLOB
    shards = sorted(glob.glob(SHARD_GLOB))
    idx = pd.read_parquet("output/ffhq_index/ffhq_sha_index.parquet")
    if "shard_idx" not in idx.columns:
        for alt in ("shard", "shard_id"):
            if alt in idx.columns:
                idx = idx.rename(columns={alt: "shard_idx"})
                break
    if "row_idx" not in idx.columns:
        for alt in ("row", "row_id"):
            if alt in idx.columns:
                idx = idx.rename(columns={alt: "row_idx"})
                break
    return idx, shards


def _picks(out_dir: Path, n: int) -> list[str]:
    """First-call: pick n eval shas (lowest by sha string) and persist.
    Subsequent calls: reload from disk so samples are comparable across steps.
    """
    picks_path = out_dir / "eval_picks.json"
    if picks_path.exists():
        return json.loads(picks_path.read_text())
    ds = CfmPairDataset(split="eval")
    shas = sorted(ds.df.image_sha256.tolist())[:n]
    out_dir.mkdir(parents=True, exist_ok=True)
    picks_path.write_text(json.dumps(shas))
    return shas


def _render_control_from_bs(bs_vec_dict: dict[str, float], row,
                            H: int = CTRL_SIZE, W: int = CTRL_SIZE) -> np.ndarray:
    """Render FLAME control image (uint8 HWC RGB) from a blendshape dict."""
    arkit52 = mediapipe_to_basis_vector(bs_vec_dict)
    verts = deform(arkit52)
    rot = np.array(row.rotation, dtype=np.float64).reshape(3, 3)
    try:
        lm_norm = np.array(row.landmarks_xy, dtype=np.float64).reshape(478, 2)
        lm_px = lm_norm * np.array([W, H])
        return render_landmark_aligned(verts, rot, lm_px, H=H, W=W)
    except Exception:
        return render(verts, rot,
                      (row.bbox_cx, row.bbox_cy, row.bbox_w, row.bbox_h),
                      H=H, W=W)


def _ctrl_uint8_to_minus1_1(ctrl_rgb_uint8: np.ndarray) -> torch.Tensor:
    return (torch.from_numpy(ctrl_rgb_uint8).float().permute(2, 0, 1)
            / 127.5 - 1.0)


def _setup(model, vae, text_embeds, device, dtype):
    if vae is None:
        vae = _build_vae(FLUX_HF_ID, device, dtype)
    if text_embeds is None:
        t5_seq, pooled = _load_text_embeds(TEXT_EMBEDS_PATH, device, dtype)
    else:
        t5_seq, pooled = text_embeds
    img_ids = prepare_latent_image_ids(LATENT_H, LATENT_W, device, dtype)
    txt_ids = prepare_text_ids(8 + t5_seq.shape[1], device, dtype)
    guidance = torch.full((1,), 3.5, device=device, dtype=dtype)
    return vae, t5_seq, pooled, img_ids, txt_ids, guidance


@torch.no_grad()
def dump_samples(
    model,
    step: int,
    out_dir: str,
    n: int = 8,
    sample_steps: int = 25,
    vae=None,
    text_embeds=None,
):
    """Render n eval picks: [target | control | id-only | conditioned]."""
    out = Path(out_dir)
    samples_dir = out / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda"
    dtype = torch.bfloat16

    vae, t5_seq, pooled, img_ids, txt_ids, guidance = _setup(
        model, vae, text_embeds, device, dtype)

    shas = _picks(out, n)
    ds = CfmPairDataset(split="eval")
    sub = ds.df[ds.df.image_sha256.isin(shas)].copy()
    # Preserve picks order.
    sub["__order"] = sub.image_sha256.map({s: i for i, s in enumerate(shas)})
    sub = sub.sort_values("__order").reset_index(drop=True)
    ds.df = sub.drop(columns="__order")

    ffhq_idx, shards = _ffhq_index_and_shards()
    sigma_seq = _build_sigma_seq(sample_steps, device, dtype)

    rows: list[np.ndarray] = []
    for i in range(len(ds)):
        item = ds[i]
        sha = item["sha"]
        row = ds.df.iloc[i]
        photo_latent = item["photo_latent"].to(device, dtype).unsqueeze(0)
        id_tokens = item["id_tokens"].to(device, dtype).unsqueeze(0)
        ctrl = item["control_rgb"].to(device, dtype).unsqueeze(0)
        ctrl_latent = _vae_encode(vae, ctrl, dtype)
        ctrl_packed = pack_latents(ctrl_latent)

        # σ_max scaled noise (FLUX flow starts at z_T = ε since (1-σ)·z0+σ·ε
        # at σ=σ_max≈1 is essentially noise). Use plain randn for the eval seed
        # — comparable across rows for fixed sample_steps.
        gen = torch.Generator(device=device).manual_seed(
            int(hashlib.md5(sha.encode()).hexdigest()[:8], 16))
        z_T_unpacked = torch.randn(
            photo_latent.shape, generator=gen, device=device, dtype=dtype)
        z_T = pack_latents(z_T_unpacked)

        z_id = _sample(model, z_T, sigma_seq, id_tokens, t5_seq, pooled,
                       ctrl_packed, img_ids, txt_ids, guidance, cs=0.0)
        z_cn = _sample(model, z_T, sigma_seq, id_tokens, t5_seq, pooled,
                       ctrl_packed, img_ids, txt_ids, guidance, cs=1.0)

        gen_id = _decode(vae, z_id)[0].permute(1, 2, 0).numpy()       # (H,W,3)
        gen_cn = _decode(vae, z_cn)[0].permute(1, 2, 0).numpy()

        target = _load_photo_crop_uint8(sha, row, ffhq_idx, shards)
        ctrl_uint8 = ((ctrl[0].float().cpu().permute(1, 2, 0).numpy() + 1.0)
                      * 127.5).clip(0, 255).astype(np.uint8)
        rows.append(np.concatenate([target, ctrl_uint8, gen_id, gen_cn], axis=1))

    collage = np.concatenate(rows, axis=0)
    Image.fromarray(collage).save(samples_dir / f"step_{step:06d}.png")


def _arcface_app():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(512, 512))
    return app


def _arcface_embed(app, rgb_uint8: np.ndarray) -> Optional[np.ndarray]:
    bgr = rgb_uint8[:, :, ::-1].copy()
    faces = app.get(bgr)
    if not faces:
        return None
    return faces[0].normed_embedding.astype(np.float32)


def _mp_landmarker_with_blendshapes():
    import mediapipe as mp  # noqa: F401
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path="models/mediapipe/face_landmarker.task"),
        output_face_blendshapes=True,
        num_faces=1,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


def _mp_bs_vector(landmarker, rgb_uint8: np.ndarray) -> Optional[np.ndarray]:
    """52-d ARKit bs vector (BASIS_CHANNEL_NAMES order) or None if no detection."""
    import mediapipe as mp
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=np.ascontiguousarray(rgb_uint8))
    res = landmarker.detect(mp_image)
    if not res.face_blendshapes:
        return None
    got = {c.category_name: float(c.score) for c in res.face_blendshapes[0]}
    bs_dict = {n: got.get(n, 0.0) for n in ARKIT_BLENDSHAPE_NAMES}
    return mediapipe_to_basis_vector(bs_dict)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


@torch.no_grad()
def metrics(
    model,
    step: int,
    out_dir: str,
    vae=None,
    text_embeds=None,
    sample_steps: int = 25,
) -> dict:
    """Iterate full eval split → ArcFace id cos + MP bs cos (vs intended +
    vs neutral-control baseline). Appends to <out_dir>/metrics.csv.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device, dtype = "cuda", torch.bfloat16

    vae, t5_seq, pooled, img_ids, txt_ids, guidance = _setup(
        model, vae, text_embeds, device, dtype)
    sigma_seq = _build_sigma_seq(sample_steps, device, dtype)

    ds = CfmPairDataset(split="eval")
    ffhq_idx, shards = _ffhq_index_and_shards()
    arcface = _arcface_app()
    landmarker = _mp_landmarker_with_blendshapes()

    neutral_bs = {n: 0.0 for n in ARKIT_BLENDSHAPE_NAMES}

    id_cos_list: list[float] = []
    bs_cos_list: list[float] = []
    bs_cos_neutral_list: list[float] = []
    skipped = 0

    for i in range(len(ds)):
        item = ds[i]
        sha = item["sha"]
        row = ds.df.iloc[i]
        id_tokens = item["id_tokens"].to(device, dtype).unsqueeze(0)
        photo_latent_shape = item["photo_latent"].unsqueeze(0).shape

        # Real (conditioned) control.
        ctrl_real = item["control_rgb"].to(device, dtype).unsqueeze(0)
        ctrl_real_lat = _vae_encode(vae, ctrl_real, dtype)
        ctrl_real_packed = pack_latents(ctrl_real_lat)

        # Neutral control: render with all-zero blendshapes, same pose.
        ctrl_neut_uint8 = _render_control_from_bs(neutral_bs, row)
        ctrl_neut = _ctrl_uint8_to_minus1_1(ctrl_neut_uint8).to(
            device, dtype).unsqueeze(0)
        ctrl_neut_lat = _vae_encode(vae, ctrl_neut, dtype)
        ctrl_neut_packed = pack_latents(ctrl_neut_lat)

        gen = torch.Generator(device=device).manual_seed(
            int(hashlib.md5(sha.encode()).hexdigest()[:8], 16))
        z_T_unpacked = torch.randn(
            photo_latent_shape, generator=gen, device=device, dtype=dtype)
        z_T = pack_latents(z_T_unpacked)

        z_cn = _sample(model, z_T, sigma_seq, id_tokens, t5_seq, pooled,
                       ctrl_real_packed, img_ids, txt_ids, guidance, cs=1.0)
        z_neu = _sample(model, z_T, sigma_seq, id_tokens, t5_seq, pooled,
                        ctrl_neut_packed, img_ids, txt_ids, guidance, cs=1.0)

        gen_cn = _decode(vae, z_cn)[0].permute(1, 2, 0).numpy()
        gen_neu = _decode(vae, z_neu)[0].permute(1, 2, 0).numpy()

        # Target intended bs vector (BASIS_CHANNEL_NAMES order).
        bs_target_dict = {n: 0.0 for n in ARKIT_BLENDSHAPE_NAMES}
        for n in ARKIT_BLENDSHAPE_NAMES:
            if n == "tongueOut":
                continue
            bs_target_dict[n] = float(row.get(f"bs_{n}", 0.0))
        bs_target = mediapipe_to_basis_vector(bs_target_dict)
        # Drop _neutral (matches expr_cos convention in eval_spike).
        neutral_idx = BASIS_CHANNEL_NAMES.index("_neutral") \
            if "_neutral" in BASIS_CHANNEL_NAMES else None

        # ArcFace identity: cosine(gen_cn, target_photo).
        target_uint8 = _load_photo_crop_uint8(sha, row, ffhq_idx, shards)
        e_target = _arcface_embed(arcface, target_uint8)
        e_gen = _arcface_embed(arcface, gen_cn)
        if e_target is not None and e_gen is not None:
            id_cos_list.append(_cos(e_target, e_gen))
        else:
            skipped += 1

        bs_cn = _mp_bs_vector(landmarker, gen_cn)
        bs_neu = _mp_bs_vector(landmarker, gen_neu)
        if bs_cn is not None:
            a, b = bs_cn, bs_target
            if neutral_idx is not None:
                mask = np.ones(len(a), dtype=bool)
                mask[neutral_idx] = False
                a, b = a[mask], b[mask]
            bs_cos_list.append(_cos(a, b))
        if bs_neu is not None:
            a, b = bs_neu, bs_target
            if neutral_idx is not None:
                mask = np.ones(len(a), dtype=bool)
                mask[neutral_idx] = False
                a, b = a[mask], b[mask]
            bs_cos_neutral_list.append(_cos(a, b))

    def _mean(xs):
        xs = [x for x in xs if not (isinstance(x, float) and np.isnan(x))]
        return float(np.mean(xs)) if xs else float("nan")

    result = {
        "step": int(step),
        "id_cos_mean": _mean(id_cos_list),
        "bs_cos_mean": _mean(bs_cos_list),
        "bs_cos_neutral_mean": _mean(bs_cos_neutral_list),
        "n_eval": len(ds),
        "n_skipped": skipped,
    }

    csv_path = out / "metrics.csv"
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["step", "id_cos_mean", "bs_cos_mean",
                        "bs_cos_neutral_mean", "n_eval", "n_skipped"])
        w.writerow([result["step"], f"{result['id_cos_mean']:.5f}",
                    f"{result['bs_cos_mean']:.5f}",
                    f"{result['bs_cos_neutral_mean']:.5f}",
                    result["n_eval"], result["n_skipped"]])
    return result
