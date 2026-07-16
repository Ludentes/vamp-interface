"""Step-0 sanity check: is the id_tokens path producing identity transfer?

Builds the CFM model fresh (no checkpoint), bypasses our twin-pass `velocity()`,
and calls InfuseNet + FLUX directly. Three sweeps per pick:
  A) controlnet_cond = zeros           → vanilla InfiniteYou baseline
  B) controlnet_cond = ctrl_latent     → with FLAME normals (untrained LoRA)
  C) controlnet_cond = photo_latent    → reproduces the suspected bug

Saves a collage [target | A | B | C] per pick.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
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

CTRL_SIZE = 512
LATENT_H = LATENT_W = CTRL_SIZE // 8
TEXT_EMBEDS_PATH = "output/cfm_precompute/text_embeds.pt"
PRECOMPUTE_DIR = Path("output/cfm_precompute")
DEFAULT_PICKS = "exp_output/cfm_train/pilot/eval_picks.json"


@torch.no_grad()
def _single_pass_velocity(model, z_t_packed, cond_packed, sigma,
                          id_tokens, t5_seq, pooled,
                          id_txt_ids, t5_txt_ids, img_ids, guidance):
    """Canonical InfiniteYou wiring per ComfyUI_InfiniteYou/infuse_net.py L59-63:
    InfuseNet sees ONLY id_tokens as encoder_hidden_states; FLUX backbone sees
    ONLY the T5 prompt. Cross-attention contexts are not shared between them.
    """
    cn_d, cn_s = model.infusenet(
        hidden_states=z_t_packed,
        controlnet_cond=cond_packed,
        conditioning_scale=1.0,
        encoder_hidden_states=id_tokens,
        pooled_projections=pooled,
        timestep=sigma,
        img_ids=img_ids,
        txt_ids=id_txt_ids,
        guidance=guidance,
        return_dict=False,
    )
    v = model.flux(
        hidden_states=z_t_packed,
        timestep=sigma,
        guidance=guidance,
        pooled_projections=pooled,
        encoder_hidden_states=t5_seq,
        txt_ids=t5_txt_ids,
        img_ids=img_ids,
        controlnet_block_samples=cn_d,
        controlnet_single_block_samples=cn_s,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="exp_output/cfm_sanity_singlepass")
    ap.add_argument("--picks-file", default=DEFAULT_PICKS)
    ap.add_argument("--n", type=int, default=4,
                    help="number of eval shas to render")
    ap.add_argument("--sample-steps", type=int, default=25)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device, dtype = "cuda", torch.bfloat16
    print("[sanity] building model (fresh, no checkpoint load)...")
    model = build_model()
    vae = _build_vae(device, dtype)
    t5_seq, pooled = _load_text_embeds(TEXT_EMBEDS_PATH, device, dtype)

    img_ids = prepare_latent_image_ids(LATENT_H, LATENT_W, device, dtype)
    id_txt_ids = prepare_text_ids(8, device, dtype)
    t5_txt_ids = prepare_text_ids(t5_seq.shape[1], device, dtype)
    guidance = torch.full((1,), 3.5, device=device, dtype=dtype)
    sigma_seq = _build_sigma_seq(args.sample_steps, device, dtype)

    shas = json.loads(Path(args.picks_file).read_text())[: args.n]
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
        print(f"[sanity] {i+1}/{len(ds)} sha={sha[:12]}")
        row = ds.df.iloc[i]
        photo_latent = item["photo_latent"].to(device, dtype).unsqueeze(0)
        id_tokens = item["id_tokens"].to(device, dtype).unsqueeze(0)
        ctrl_latent = item["ctrl_latent"].to(device, dtype).unsqueeze(0)

        cond_zero = pack_latents(torch.zeros_like(photo_latent))
        cond_ctrl = pack_latents(ctrl_latent)
        cond_photo = pack_latents(photo_latent)

        gen = torch.Generator(device=device).manual_seed(
            int(hashlib.md5(sha.encode()).hexdigest()[:8], 16))
        z_T_unpacked = torch.randn(
            photo_latent.shape, generator=gen, device=device, dtype=dtype)
        z_T = pack_latents(z_T_unpacked)

        decoded = []
        for cond in (cond_zero, cond_ctrl, cond_photo):
            z = _sample(model, z_T, sigma_seq, cond, id_tokens, t5_seq, pooled,
                        id_txt_ids, t5_txt_ids, img_ids, guidance)
            img = _decode(vae, z)[0].permute(1, 2, 0).numpy()
            decoded.append(img)

        target = _load_photo_crop_uint8(sha, row, ffhq_idx, shards)
        rows.append(np.concatenate([target] + decoded, axis=1))

    collage = np.concatenate(rows, axis=0)
    out_path = out / "sanity_collage.png"
    Image.fromarray(collage).save(out_path)
    print(f"[sanity] wrote {out_path}")
    print("[sanity] columns: target | A=zero-cond | B=normals-cond | C=photo-cond")


if __name__ == "__main__":
    main()
