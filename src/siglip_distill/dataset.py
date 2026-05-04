"""Joined latent ↔ SigLIP-embedding dataset.

Reads `arc_full_latent/compact.pt` (latents+arcface, 26,108 SHAs) paired with
`compact_siglip.pt` (1152-d SigLIP-2 image embeddings aligned to the same SHAs).

Train/val split: SHA-prefix == 'f' → val (matches arc_distill / mediapipe_distill
identity-held-out semantics).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset


def is_held_out(image_sha256: str) -> bool:
    return image_sha256[:1].lower() == "f"


class CompactLatentSiglipDataset(Dataset):
    def __init__(
        self,
        compact_path: Path,
        siglip_path: Path,
        split: Literal["train", "val"],
    ):
        compact = torch.load(compact_path, map_location="cpu", weights_only=False)
        sg_blob = torch.load(siglip_path, map_location="cpu", weights_only=False)

        if list(compact["shas"]) != list(sg_blob["shas"]):
            raise ValueError("compact + siglip SHA order mismatch")

        self.latents = compact["latents"]              # (N, 16, 64, 64) bf16
        self.embeddings = sg_blob["embeddings"]        # (N, 1152) fp16, L2=1
        detected = sg_blob["detected"]                 # (N,) bool
        self.emb_dim = int(sg_blob["emb_dim"])
        shas = compact["shas"]
        self.shas = shas

        self.indices = [
            i for i, sha in enumerate(shas)
            if (split == "val") == is_held_out(sha) and bool(detected[i])
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        latent = self.latents[idx].to(torch.float32)
        target = self.embeddings[idx].to(torch.float32)
        return latent, target
