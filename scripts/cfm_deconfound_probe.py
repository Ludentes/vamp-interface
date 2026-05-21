"""Deconfound probe: same identity, multiple FLAME controls.

The pilot's per-row eval collage answers "does the model produce something
recognizable?" but not "does it use the FLAME control?" — col 3 could be
just `decoded(photo_latent)` ignoring controlnet_block_samples entirely.

This probe holds id_tokens fixed and varies the FLAME control input. If col 2
across columns changes (jaw open, smile, brow up, yaw left, yaw right) but
col 3 (generated) stays constant, FLAME is being ignored. If col 3 tracks
col 2, the controlnet path is wired and learning.

Outputs: exp_output/cfm_deconfound/step_NNNNNN/collage.png with
  2K rows × (N+1) cols
  row 2i  = [target | C_1 | C_2 | ... | C_N]   (FLAME normals per variant)
  row 2i+1= [target | G_1 | G_2 | ... | G_N]   (generated per variant)

Variants (N=8):
  id_zero        — controlnet_cond = zeros (identity baseline)
  neutral        — all bs = 0, original landmark-aligned pose
  jaw_open       — jawOpen=1.0
  mouth_smile    — mouthSmileLeft=mouthSmileRight=1.0
  brow_up        — browInnerUp=browOuterUpLeft=browOuterUpRight=1.0
  eyes_blink     — eyeBlinkLeft=eyeBlinkRight=1.0
  yaw_left       — rotation pre-multiplied by Ry(-30°), bbox-render path
  yaw_right      — rotation pre-multiplied by Ry(+30°), bbox-render path

Notes: the yaw variants use `render()` (bbox path), not the landmark-aligned
path the model trained on — small distribution shift, but it's the only way
to vary pose without synthesizing fake landmarks.
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
    build_model, pack_latents, prepare_latent_image_ids, prepare_text_ids,
    velocity,
)
from arkit_controlnet.cfm.train import _load_text_embeds
from arkit_controlnet.cfm.eval import (
    _build_vae, _build_sigma_seq, _decode, _ffhq_index_and_shards,
    _load_photo_crop_uint8, _ctrl_uint8_to_minus1_1, _vae_encode,
)
from arkit_controlnet.eval_spike import ARKIT_BLENDSHAPE_NAMES
from arkit_controlnet.flame_render import (
    deform, mediapipe_to_basis_vector, render, render_landmark_aligned,
)

CTRL_SIZE = 512
LATENT_H = LATENT_W = CTRL_SIZE // 8
TEXT_EMBEDS_PATH = "output/cfm_precompute/text_embeds.pt"


def _bs(**overrides) -> dict[str, float]:
    d = {n: 0.0 for n in ARKIT_BLENDSHAPE_NAMES}
    d.update(overrides)
    return d


def _ry(deg: float) -> np.ndarray:
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _render_variant(name: str, row) -> np.ndarray:
    """Returns uint8 HWC RGB FLAME-normals image (or zeros for id_zero)."""
    rot = np.array(row.rotation, dtype=np.float64).reshape(3, 3)
    lm_norm = np.array(row.landmarks_xy, dtype=np.float64).reshape(478, 2)
    lm_px = lm_norm * np.array([CTRL_SIZE, CTRL_SIZE])

    if name == "id_zero":
        return np.zeros((CTRL_SIZE, CTRL_SIZE, 3), dtype=np.uint8)
    if name == "neutral":
        bs = _bs()
    elif name == "jaw_open":
        bs = _bs(jawOpen=1.0)
    elif name == "mouth_smile":
        bs = _bs(mouthSmileLeft=1.0, mouthSmileRight=1.0)
    elif name == "brow_up":
        bs = _bs(browInnerUp=1.0, browOuterUpLeft=1.0, browOuterUpRight=1.0)
    elif name == "eyes_blink":
        bs = _bs(eyeBlinkLeft=1.0, eyeBlinkRight=1.0)
    elif name in ("yaw_left", "yaw_right"):
        deg = -30.0 if name == "yaw_left" else 30.0
        bs = _bs()
        verts = deform(mediapipe_to_basis_vector(bs))
        rot_yaw = _ry(deg) @ rot
        return render(verts, rot_yaw,
                      (row.bbox_cx, row.bbox_cy, row.bbox_w, row.bbox_h),
                      H=CTRL_SIZE, W=CTRL_SIZE)
    else:
        raise ValueError(name)

    verts = deform(mediapipe_to_basis_vector(bs))
    return render_landmark_aligned(verts, rot, lm_px, H=CTRL_SIZE, W=CTRL_SIZE)


VARIANTS = ["id_zero", "neutral", "jaw_open", "mouth_smile", "brow_up",
            "eyes_blink", "yaw_left", "yaw_right"]


@torch.no_grad()
def _sample(model, z_T, sigma_seq, id_tokens, t5_seq, pooled,
            id_txt_ids, t5_txt_ids, img_ids, control_packed, guidance):
    z = z_T
    for i in range(len(sigma_seq) - 1):
        s = sigma_seq[i]
        s_next = sigma_seq[i + 1]
        v = velocity(model, z, s.expand(z.shape[0]), id_tokens, t5_seq, pooled,
                     id_txt_ids, t5_txt_ids, img_ids, control_packed, guidance)
        z = z + (s_next - s) * v
    return z


def _load_step_from_ckpt(ckpt_path: Path) -> int:
    """Cheap step read — load with map_location=cpu and pull `step`."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return int(state.get("step", -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="exp_output/cfm_train/pilot/latest.pt")
    ap.add_argument("--out-dir", default="exp_output/cfm_deconfound")
    ap.add_argument("--picks-file",
                    default="exp_output/cfm_train/pilot/eval_picks.json")
    ap.add_argument("--n-ids", type=int, default=2,
                    help="how many identities to probe (more = bigger collage)")
    ap.add_argument("--sample-steps", type=int, default=25)
    args = ap.parse_args()

    device, dtype = "cuda", torch.bfloat16
    ckpt_path = Path(args.ckpt)
    step = _load_step_from_ckpt(ckpt_path)
    print(f"[deconfound] checkpoint step={step}")

    out_dir = Path(args.out_dir) / f"step_{step:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[deconfound] building model + loading LoRA state...")
    model = build_model()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.infusenet.load_state_dict(
        state["infusenet"], strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys: {unexpected[:5]}")
    print(f"[deconfound] loaded ({len(missing)} non-trainable keys not in ckpt)")

    vae = _build_vae(device, dtype)
    t5_seq, pooled = _load_text_embeds(TEXT_EMBEDS_PATH, device, dtype)
    img_ids = prepare_latent_image_ids(LATENT_H, LATENT_W, device, dtype)
    id_txt_ids = prepare_text_ids(8, device, dtype)
    t5_txt_ids = prepare_text_ids(t5_seq.shape[1], device, dtype)
    guidance = torch.full((1,), 3.5, device=device, dtype=dtype)
    sigma_seq = _build_sigma_seq(args.sample_steps, device, dtype)

    shas = json.loads(Path(args.picks_file).read_text())[:args.n_ids]
    ds = CfmPairDataset(split="eval")
    sub = ds.df[ds.df.image_sha256.isin(shas)].copy()
    sub["__order"] = sub.image_sha256.map({s: i for i, s in enumerate(shas)})
    sub = sub.sort_values("__order").reset_index(drop=True)
    ds.df = sub.drop(columns="__order")
    ffhq_idx, shards = _ffhq_index_and_shards()

    big_rows = []
    for i in range(len(ds)):
        item = ds[i]
        sha = item["sha"]
        row = ds.df.iloc[i]
        print(f"[deconfound] {i+1}/{len(ds)} sha={sha[:12]}")

        photo_latent = item["photo_latent"].to(device, dtype).unsqueeze(0)
        id_tokens = item["id_tokens"].to(device, dtype).unsqueeze(0)

        # Fixed seed per sha so cross-variant differences come ONLY from
        # the FLAME control image, not the noise sample.
        gen = torch.Generator(device=device).manual_seed(
            int(hashlib.md5(sha.encode()).hexdigest()[:8], 16))
        z_T_unpacked = torch.randn(photo_latent.shape, generator=gen,
                                   device=device, dtype=dtype)
        z_T = pack_latents(z_T_unpacked)

        target = _load_photo_crop_uint8(sha, row, ffhq_idx, shards)

        ctrl_panels = [target]
        gen_panels = [target]
        for v in VARIANTS:
            ctrl_rgb = _render_variant(v, row)
            ctrl_panels.append(ctrl_rgb)

            ctrl_lat = _vae_encode(
                vae, _ctrl_uint8_to_minus1_1(ctrl_rgb).to(device, dtype).unsqueeze(0),
                dtype,
            )
            control_packed = pack_latents(ctrl_lat)
            z = _sample(model, z_T, sigma_seq, id_tokens, t5_seq, pooled,
                        id_txt_ids, t5_txt_ids, img_ids, control_packed, guidance)
            gen_img = _decode(vae, z)[0].permute(1, 2, 0).numpy()
            gen_panels.append(gen_img)

        big_rows.append(np.concatenate(ctrl_panels, axis=1))
        big_rows.append(np.concatenate(gen_panels, axis=1))

    collage = np.concatenate(big_rows, axis=0)
    out_path = out_dir / "collage.png"
    Image.fromarray(collage).save(out_path)
    print(f"[deconfound] wrote {out_path}")
    print(f"[deconfound] cols (left→right): target, "
          + ", ".join(VARIANTS))
    print(f"[deconfound] rows alternate per identity: controls, generated")


if __name__ == "__main__":
    main()
