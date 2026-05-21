"""CFM per-image precompute: photo VAE latents + InfiniteYou identity tokens.

Outputs (under ``out_dir``):
    photo_latents/{sha}.pt   (16, 64, 64) bf16   — FLUX VAE encode of 512² crop
    id_tokens/{sha}.pt       (8, 4096)    bf16   — InfiniteYou resampler tokens
    meta.parquet                                  — per-sha tracker

Resumable: per-sha files use atomic writes; existing outputs are skipped.

CLI:
    PYTHONPATH=src uv run python -m arkit_controlnet.cfm.precompute \
        --shas-file <file>  --out-dir output/cfm_precompute
"""
from __future__ import annotations

import argparse
import glob
import io
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from arkit_controlnet.build_ffhq_index import SHARD_GLOB
from arkit_controlnet.cfm.resampler import Resampler
from arkit_controlnet.flame_render import (
    BASIS_CHANNEL_NAMES, deform, mediapipe_to_basis_vector,
    render_landmark_aligned, render,
)
from arkit_controlnet.eval_spike import ARKIT_BLENDSHAPE_NAMES

CTRL_SIZE = 512
BS_COLUMNS = [f"bs_{n}" for n in BASIS_CHANNEL_NAMES if n != "tongueOut"]

POSE_CACHE = Path("output/flame_pose_cache/pose_cache.parquet")
REVERSE_INDEX = Path("output/reverse_index/reverse_index.parquet")
FFHQ_SHA_INDEX = Path("output/ffhq_index/ffhq_sha_index.parquet")
INFU_RESAMPLER_WEIGHTS = Path(
    "data/infiniteyou_dl_bf16/infu_flux_v1.0/sim_stage1/image_proj_model.bin"
)
FLUX_HF_ID = "black-forest-labs/FLUX.1-dev"

FLUSH_EVERY = 256


def face_crop_resize(
    rgb: np.ndarray,
    bbox_cx: float,
    bbox_cy: float,
    bbox_w: float,
    bbox_h: float,
    out_size: int = 512,
    margin: float = 0.25,
) -> np.ndarray:
    """Square-crop around a normalized bbox, replicate-pad if OOB, resize."""
    H, W = rgb.shape[:2]
    cx_px = bbox_cx * W
    cy_px = bbox_cy * H
    side = max(bbox_w * W, bbox_h * H) * (1.0 + 2.0 * margin)
    half = side / 2.0

    # Desired crop bounds in pixel space (may go negative or exceed dims).
    x0 = int(round(cx_px - half))
    y0 = int(round(cy_px - half))
    x1 = int(round(cx_px + half))
    y1 = int(round(cy_px + half))

    # Pad as needed and adjust into padded frame coords.
    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - W)
    pad_b = max(0, y1 - H)
    if pad_l or pad_t or pad_r or pad_b:
        padded = cv2.copyMakeBorder(
            rgb, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE
        )
    else:
        padded = rgb
    x0p, y0p = x0 + pad_l, y0 + pad_t
    x1p, y1p = x1 + pad_l, y1 + pad_t
    crop = padded[y0p:y1p, x0p:x1p]
    if crop.shape[0] == 0 or crop.shape[1] == 0:
        raise ValueError(
            f"empty crop: bbox ({bbox_cx},{bbox_cy},{bbox_w},{bbox_h}) on {W}x{H}"
        )
    resized = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8, copy=False)


def _atomic_torch_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_parquet_write(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _load_vae(device: torch.device):
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        FLUX_HF_ID, subfolder="vae", torch_dtype=torch.bfloat16
    ).to(device).eval()
    return vae


def _load_face_detector():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    # det_size matches our 512² face crop (default 640 misses faces on
    # tightly-framed crops).
    app.prepare(ctx_id=0, det_size=(512, 512))
    return app


def _load_arcface_recognition(device: torch.device):
    # InfiniteYou's resampler was trained against facexlib IR-SE-50 ArcFace
    # over 5-pt-aligned 112² norm_crop input. buffalo_l's normed_embedding
    # lives in an orthogonal subspace — wrong recognition head silently
    # produces junk id_tokens. See feedback-infiniteyou-arcface-recipe.
    from facexlib.recognition import init_recognition_model
    return init_recognition_model("arcface", device=str(device))


def _load_resampler(device: torch.device):
    if not INFU_RESAMPLER_WEIGHTS.exists():
        raise FileNotFoundError(
            f"InfiniteYou resampler weights missing: {INFU_RESAMPLER_WEIGHTS}"
        )
    model = Resampler(
        dim=1280,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=8,
        embedding_dim=512,
        output_dim=4096,
        ff_mult=4,
    )
    sd = torch.load(INFU_RESAMPLER_WEIGHTS, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "image_proj" in sd:
        sd = sd["image_proj"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        # Surface for debugging but don't crash — InfiniteYou ckpts sometimes
        # include extra keys (e.g. ema). Strict mismatch would be a real bug.
        print(f"[precompute] resampler load: missing={missing[:3]} unexpected={unexpected[:3]}")
    model = model.to(device, dtype=torch.bfloat16).eval()
    return model


def _encode_photo_latent(vae, crop_rgb_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
    ten = torch.from_numpy(crop_rgb_uint8).permute(2, 0, 1).unsqueeze(0).to(
        device, dtype=torch.bfloat16
    )
    ten = ten / 127.5 - 1.0
    with torch.no_grad():
        # `.mode()` (deterministic) instead of `.sample()` — re-runs must
        # produce identical latents for cache invalidation to be sha-based.
        out = vae.encode(ten).latent_dist.mode()
    shift = getattr(vae.config, "shift_factor", 0.0) or 0.0
    scale = getattr(vae.config, "scaling_factor", 1.0)
    lat = (out - shift) * scale
    return lat[0]  # (16, 64, 64)


def _build_work_table(shas: list[str] | None) -> pd.DataFrame:
    if not POSE_CACHE.exists():
        raise FileNotFoundError(f"missing pose cache: {POSE_CACHE}")
    if not REVERSE_INDEX.exists():
        raise FileNotFoundError(f"missing reverse index: {REVERSE_INDEX}")
    if not FFHQ_SHA_INDEX.exists():
        raise FileNotFoundError(f"missing ffhq sha index: {FFHQ_SHA_INDEX}")

    pc = pd.read_parquet(
        POSE_CACHE,
        columns=["image_sha256", "bbox_cx", "bbox_cy", "bbox_w", "bbox_h",
                 "rotation", "landmarks_xy", "pose_detected"],
    )
    pc = pc[pc.pose_detected].drop(columns=["pose_detected"])

    ri = pd.read_parquet(REVERSE_INDEX)
    ri_cols = ["image_sha256"] + [c for c in BS_COLUMNS if c in ri.columns]
    ri = ri[ri_cols].drop_duplicates("image_sha256")
    pc = pc.merge(ri, on="image_sha256", how="inner")

    idx = pd.read_parquet(FFHQ_SHA_INDEX)
    # Tolerate column name variants.
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
    pc = pc.merge(idx[["image_sha256", "shard_idx", "row_idx"]],
                  on="image_sha256", how="inner")
    if shas is not None:
        pc = pc[pc.image_sha256.isin(set(shas))]
    return pc.reset_index(drop=True)


def run(shas: list[str] | None = None, out_dir: str = "output/cfm_precompute") -> None:
    shards = sorted(glob.glob(SHARD_GLOB))
    if not shards:
        raise FileNotFoundError(
            f"no FFHQ shards at {SHARD_GLOB} — is the Seagate drive mounted?"
        )

    out = Path(out_dir)
    pl_dir = out / "photo_latents"
    it_dir = out / "id_tokens"
    cl_dir = out / "ctrl_latents"
    pl_dir.mkdir(parents=True, exist_ok=True)
    it_dir.mkdir(parents=True, exist_ok=True)
    cl_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.parquet"

    if meta_path.exists():
        meta_existing = pd.read_parquet(meta_path)
        if "ctrl_ok" not in meta_existing.columns:
            meta_existing["ctrl_ok"] = False
        seen = set(meta_existing.image_sha256.tolist())
    else:
        meta_existing = pd.DataFrame(
            columns=["image_sha256", "id_ok", "pl_ok", "ctrl_ok"]
        )
        seen = set()

    work = _build_work_table(shas)
    if work.empty:
        print("[precompute] no rows to process")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = _load_vae(device)
    detector = _load_face_detector()
    arcface = _load_arcface_recognition(device)
    resampler = _load_resampler(device)

    new_rows: list[dict] = []
    processed_since_flush = 0
    total = len(work)
    done = 0

    for shard_idx, grp in work.groupby("shard_idx", sort=True):
        shard_path = shards[int(shard_idx)]
        df = pd.read_parquet(shard_path)
        for _, row in grp.iterrows():
            sha = row.image_sha256
            done += 1
            pl_path = pl_dir / f"{sha}.pt"
            it_path = it_dir / f"{sha}.pt"
            cl_path = cl_dir / f"{sha}.pt"

            if (pl_path.exists() and it_path.exists() and cl_path.exists()
                    and sha in seen):
                continue

            cell = df["image"].iloc[int(row.row_idx)]
            rgb = np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))
            crop = face_crop_resize(
                rgb,
                bbox_cx=float(row.bbox_cx),
                bbox_cy=float(row.bbox_cy),
                bbox_w=float(row.bbox_w),
                bbox_h=float(row.bbox_h),
                out_size=512,
            )

            pl_ok = pl_path.exists()
            if not pl_ok:
                lat = _encode_photo_latent(vae, crop, device)
                _atomic_torch_save(lat.to("cpu", dtype=torch.bfloat16), pl_path)
                pl_ok = True

            id_ok = it_path.exists()
            if not id_ok:
                from insightface.utils import face_align
                bgr = crop[:, :, ::-1].copy()
                faces = detector.get(bgr)
                if len(faces) == 0:
                    id_ok = False
                else:
                    face = sorted(
                        faces,
                        key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
                    )[-1]
                    aligned = face_align.norm_crop(
                        bgr, landmark=np.array(face.kps), image_size=112,
                    )
                    x = (
                        torch.from_numpy(aligned).unsqueeze(0)
                        .permute(0, 3, 1, 2).float() / 255.0
                    )
                    x = (2 * x - 1).to(device).contiguous()
                    with torch.no_grad():
                        emb = arcface(x)  # (1, 512), float32
                        emb_t = emb.to(dtype=torch.bfloat16).view(1, 1, 512)
                        tokens = resampler(emb_t)[0]  # (8, 4096)
                    _atomic_torch_save(
                        tokens.to("cpu", dtype=torch.bfloat16), it_path
                    )
                    id_ok = True

            ctrl_ok = cl_path.exists()
            if not ctrl_ok:
                bs = {n: 0.0 for n in ARKIT_BLENDSHAPE_NAMES}
                for n in (n for n in ARKIT_BLENDSHAPE_NAMES if n != "tongueOut"):
                    bs[n] = float(row.get(f"bs_{n}", 0.0))
                arkit52 = mediapipe_to_basis_vector(bs)
                verts = deform(arkit52)
                rot = np.array(row.rotation, dtype=np.float64).reshape(3, 3)
                try:
                    lm_norm = np.array(row.landmarks_xy, dtype=np.float64).reshape(478, 2)
                    lm_px = lm_norm * np.array([CTRL_SIZE, CTRL_SIZE])
                    ctrl_rgb = render_landmark_aligned(
                        verts, rot, lm_px, H=CTRL_SIZE, W=CTRL_SIZE)
                except Exception:
                    ctrl_rgb = render(
                        verts, rot,
                        (row.bbox_cx, row.bbox_cy, row.bbox_w, row.bbox_h),
                        H=CTRL_SIZE, W=CTRL_SIZE)
                ctrl_lat = _encode_photo_latent(vae, ctrl_rgb, device)
                _atomic_torch_save(ctrl_lat.to("cpu", dtype=torch.bfloat16), cl_path)
                ctrl_ok = True

            new_rows.append(
                {"image_sha256": sha, "id_ok": bool(id_ok),
                 "pl_ok": bool(pl_ok), "ctrl_ok": bool(ctrl_ok)}
            )
            processed_since_flush += 1
            if processed_since_flush >= FLUSH_EVERY:
                meta_existing = _flush_meta(meta_existing, new_rows, meta_path)
                new_rows = []
                processed_since_flush = 0
                print(f"[precompute] {done}/{total} flushed")

    if new_rows:
        meta_existing = _flush_meta(meta_existing, new_rows, meta_path)
    print(f"[precompute] done: {done}/{total} rows, meta -> {meta_path}")


def _flush_meta(existing: pd.DataFrame, new_rows: list[dict], path: Path) -> pd.DataFrame:
    if not new_rows:
        return existing
    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates("image_sha256", keep="last")
    _atomic_parquet_write(combined, path)
    return combined


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shas-file", default=None,
                    help="newline-separated file of image_sha256 to process; "
                         "default = full pose∩reverse∩index intersection")
    ap.add_argument("--out-dir", default="output/cfm_precompute")
    args = ap.parse_args()

    shas = None
    if args.shas_file is not None:
        with open(args.shas_file) as f:
            shas = [line.strip() for line in f if line.strip()]
    run(shas=shas, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
