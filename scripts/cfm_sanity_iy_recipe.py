"""Sanity check using InfiniteYou's exact id-embed recipe.

Bypasses our precomputed id_tokens (which come from buffalo_l ArcFace — wrong
embedding space for the resampler). Instead, for each pick:
  1. Load raw FFHQ image (full, not face-cropped).
  2. antelopev2 → 5-point landmark.
  3. face_align.norm_crop → 112² ArcFace-aligned crop.
  4. facexlib init_recognition_model('arcface') → 512-d embedding.
  5. Our existing resampler weights → 8 id_tokens.
  6. Single-pass forward: id_tokens → InfuseNet, t5 → FLUX, zero controlnet_cond.

Compares the cosine between IY-recipe id_tokens and our cached id_tokens (a
quantitative sanity check), and renders [target | IY-id-tokens A=zero-cond |
IY-id-tokens B=normals-cond].
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from arkit_controlnet.cfm.dataset import CfmPairDataset
from arkit_controlnet.cfm.model import (
    build_model, pack_latents,
    prepare_latent_image_ids, prepare_text_ids,
)
from arkit_controlnet.cfm.train import _load_text_embeds
from arkit_controlnet.cfm.eval import (
    _build_vae, _decode, _load_photo_crop_uint8, _ffhq_index_and_shards,
    _build_sigma_seq,
)
from arkit_controlnet.cfm.resampler import Resampler

CTRL_SIZE = 512
LATENT_H = LATENT_W = CTRL_SIZE // 8
TEXT_EMBEDS_PATH = "output/cfm_precompute/text_embeds.pt"
RESAMPLER_WEIGHTS = "data/infiniteyou_dl_bf16/infu_flux_v1.0/sim_stage1/image_proj_model.bin"
DEFAULT_PICKS = "exp_output/cfm_train/pilot/eval_picks.json"


def _load_resampler(device, dtype):
    m = Resampler(dim=1280, depth=4, dim_head=64, heads=20, num_queries=8,
                  embedding_dim=512, output_dim=4096, ff_mult=4)
    sd = torch.load(RESAMPLER_WEIGHTS, map_location="cpu",
                    weights_only=True)["image_proj"]
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, f"{missing=} {unexpected=}"
    return m.to(device, dtype).eval()


def _build_detector():
    """buffalo_l matching precompute's det_size=(512,512) — antelopev2 + 640²
    failed to detect FFHQ thumbnails in this env."""
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l",
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(512, 512))
    return app


def _iy_id_embed(arcface_model, detector, image_bgr):
    """Replicates extract_arcface_bgr_embedding() from InfiniteYou but uses
    buffalo_l for detection+kps. Recognition path matches InfiniteYou exactly:
    5-pt norm_crop @ 112² → facexlib ArcFace IR-SE-50 → 512-d.
    """
    from insightface.utils import face_align
    face_info = detector.get(image_bgr)
    if not face_info:
        return None
    face = sorted(face_info,
                  key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))[-1]
    kps = face.kps  # (5, 2) — standard ArcFace 5-pt
    crop = face_align.norm_crop(image_bgr, landmark=np.array(kps), image_size=112)
    x = torch.from_numpy(crop).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0
    x = (2 * x - 1).cuda().contiguous()
    with torch.no_grad():
        emb = arcface_model(x)  # (1, 512)
    return emb


@torch.no_grad()
def _single_pass_velocity(model, z_t_packed, cond_packed, sigma,
                          id_tokens, t5_seq, pooled,
                          id_txt_ids, t5_txt_ids, img_ids, guidance):
    cn_d, cn_s = model.infusenet(
        hidden_states=z_t_packed, controlnet_cond=cond_packed,
        conditioning_scale=1.0, encoder_hidden_states=id_tokens,
        pooled_projections=pooled, timestep=sigma, img_ids=img_ids,
        txt_ids=id_txt_ids, guidance=guidance, return_dict=False,
    )
    v = model.flux(
        hidden_states=z_t_packed, timestep=sigma, guidance=guidance,
        pooled_projections=pooled, encoder_hidden_states=t5_seq,
        txt_ids=t5_txt_ids, img_ids=img_ids,
        controlnet_block_samples=cn_d, controlnet_single_block_samples=cn_s,
        return_dict=False,
    )[0]
    return v


@torch.no_grad()
def _sample(model, z_T, sigma_seq, cond_packed, id_tokens, t5_seq, pooled,
            id_txt_ids, t5_txt_ids, img_ids, guidance):
    z = z_T
    for i in range(len(sigma_seq) - 1):
        s = sigma_seq[i]
        s_next = sigma_seq[i + 1]
        v = _single_pass_velocity(model, z, cond_packed, s.expand(z.shape[0]),
                                  id_tokens, t5_seq, pooled,
                                  id_txt_ids, t5_txt_ids, img_ids, guidance)
        z = z + (s_next - s) * v
    return z


def _load_raw_ffhq(sha, ffhq_idx, shards):
    idx_row = ffhq_idx[ffhq_idx.image_sha256 == sha].iloc[0]
    df = pd.read_parquet(shards[int(idx_row.shard_idx)])
    cell = df["image"].iloc[int(idx_row.row_idx)]
    rgb = np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="exp_output/cfm_sanity_iy_recipe")
    ap.add_argument("--picks-file", default=DEFAULT_PICKS)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--sample-steps", type=int, default=25)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device, dtype = "cuda", torch.bfloat16

    from facexlib.recognition import init_recognition_model
    print("[sanity-iy] loading facexlib arcface + antelopev2...")
    arcface = init_recognition_model("arcface", device="cuda")
    detector = _build_detector()
    resampler = _load_resampler(device, dtype)

    print("[sanity-iy] building model (fresh)...")
    model = build_model()
    vae = _build_vae(device, dtype)
    t5_seq, pooled = _load_text_embeds(TEXT_EMBEDS_PATH, device, dtype)

    img_ids = prepare_latent_image_ids(LATENT_H, LATENT_W, device, dtype)
    id_txt_ids = prepare_text_ids(8, device, dtype)
    t5_txt_ids = prepare_text_ids(t5_seq.shape[1], device, dtype)
    guidance = torch.full((1,), 3.5, device=device, dtype=dtype)
    sigma_seq = _build_sigma_seq(args.sample_steps, device, dtype)

    shas = json.loads(Path(args.picks_file).read_text())[:args.n]
    ds = CfmPairDataset(split="eval")
    sub = ds.df[ds.df.image_sha256.isin(shas)].copy()
    sub["__order"] = sub.image_sha256.map({s: i for i, s in enumerate(shas)})
    sub = sub.sort_values("__order").reset_index(drop=True)
    ds.df = sub.drop(columns="__order")
    ffhq_idx, shards = _ffhq_index_and_shards()

    rows = []
    for i in range(len(ds)):
        item = ds[i]
        sha = item["sha"]
        row = ds.df.iloc[i]
        print(f"[sanity-iy] {i+1}/{len(ds)} sha={sha[:12]}")

        photo_latent = item["photo_latent"].to(device, dtype).unsqueeze(0)
        ctrl_latent = item["ctrl_latent"].to(device, dtype).unsqueeze(0)
        buf_id_tokens = item["id_tokens"].to(device, dtype).unsqueeze(0)

        # IY-recipe embed from the 512² face crop (matches what precompute fed
        # to buffalo_l). 5-pt aligned arcface input wants a tight face crop.
        crop_rgb = _load_photo_crop_uint8(sha, row, ffhq_idx, shards)
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        emb = _iy_id_embed(arcface, detector, crop_bgr)
        if emb is None:
            print(f"  no face detected, skipping")
            continue
        emb = emb.to(device, dtype).view(1, 1, 512)
        with torch.no_grad():
            iy_id_tokens = resampler(emb)  # (1, 8, 4096)

        # Compare to our cached buffalo_l id_tokens.
        a = iy_id_tokens.float().flatten()
        b = buf_id_tokens.float().flatten()
        cos = float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-9))
        print(f"  cos(iy_tokens, buf_tokens) = {cos:.4f} "
              f"|iy|={a.norm():.2f} |buf|={b.norm():.2f}")

        cond_zero = pack_latents(torch.zeros_like(photo_latent))
        cond_ctrl = pack_latents(ctrl_latent)

        gen = torch.Generator(device=device).manual_seed(
            int(hashlib.md5(sha.encode()).hexdigest()[:8], 16))
        z_T_unpacked = torch.randn(photo_latent.shape, generator=gen,
                                   device=device, dtype=dtype)
        z_T = pack_latents(z_T_unpacked)

        decoded = []
        for cond in (cond_zero, cond_ctrl):
            z = _sample(model, z_T, sigma_seq, cond, iy_id_tokens, t5_seq,
                        pooled, id_txt_ids, t5_txt_ids, img_ids, guidance)
            img = _decode(vae, z)[0].permute(1, 2, 0).numpy()
            decoded.append(img)

        target = _load_photo_crop_uint8(sha, row, ffhq_idx, shards)
        rows.append(np.concatenate([target] + decoded, axis=1))

    if rows:
        collage = np.concatenate(rows, axis=0)
        out_path = out / "iy_recipe_collage.png"
        Image.fromarray(collage).save(out_path)
        print(f"[sanity-iy] wrote {out_path}")
        print("[sanity-iy] columns: target | IY-id-tokens + zero-cond | IY-id-tokens + normals-cond")
    else:
        print("[sanity-iy] no rows produced")


if __name__ == "__main__":
    main()
