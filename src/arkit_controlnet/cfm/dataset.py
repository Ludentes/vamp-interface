"""CFM training dataset — joins reverse_index ∩ pose_cache ∩ ffhq_index.

Per-item output: cached photo latent + identity tokens (from precompute) +
the live-rendered FLAME normals control image. The control image is rendered
in-process — the modality is fixed and rendering is cheap (~5 ms).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from arkit_controlnet.flame_render import (
    BASIS_CHANNEL_NAMES, deform, mediapipe_to_basis_vector,
    render_landmark_aligned, render,
)
from arkit_controlnet.eval_spike import ARKIT_BLENDSHAPE_NAMES

POSE_CACHE = "output/flame_pose_cache/pose_cache.parquet"
RI = "output/reverse_index/reverse_index.parquet"
FFHQ_INDEX = "output/ffhq_index/ffhq_sha_index.parquet"
PRECOMPUTE_DIR = "output/cfm_precompute"
CTRL_SIZE = 512

BS_COLUMNS = [f"bs_{n}" for n in BASIS_CHANNEL_NAMES if n != "tongueOut"]


def _hash_bucket(sha: str) -> int:
    return int(hashlib.md5(sha.encode()).hexdigest()[:8], 16) % 1000


class CfmPairDataset(Dataset):
    def __init__(self, split: str = "train", eval_size: int = 1024,
                 precompute_dir: str = PRECOMPUTE_DIR):
        if split not in ("train", "eval"):
            raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
        self.precompute_dir = Path(precompute_dir)
        ri = pd.read_parquet(RI)
        ri = ri[(ri.source == "ffhq") & ri.bs_detected]
        pc = pd.read_parquet(POSE_CACHE)
        pc = pc[pc.pose_detected]
        idx = pd.read_parquet(FFHQ_INDEX)
        df = (ri.merge(pc, on="image_sha256", how="inner")
                .merge(idx, on="image_sha256", how="inner")
                .sort_values("image_sha256")
                .reset_index(drop=True))
        buckets = df.image_sha256.apply(_hash_bucket).to_numpy()
        cutoff = max(1, round(eval_size * 1000 / len(df)))
        is_eval = buckets < cutoff
        self.df = (df[is_eval] if split == "eval" else df[~is_eval]
                   ).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        sha = row.image_sha256
        photo_latent = torch.load(
            self.precompute_dir / "photo_latents" / f"{sha}.pt",
            map_location="cpu")
        id_tokens = torch.load(
            self.precompute_dir / "id_tokens" / f"{sha}.pt",
            map_location="cpu")

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

        ctrl = (torch.from_numpy(ctrl_rgb).float().permute(2, 0, 1)
                / 127.5 - 1.0)
        return {
            "photo_latent": photo_latent,
            "id_tokens": id_tokens,
            "control_rgb": ctrl,
            "sha": sha,
        }
