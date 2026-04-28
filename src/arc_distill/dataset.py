"""FFHQ pixel + ArcFace teacher dataset for arc_pixel distillation."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset

IMAGENET_MEAN_T = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD_T = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def is_held_out(image_sha256: str) -> bool:
    """Deterministic 6.25% held-out split: SHA prefix 'f' is val."""
    return image_sha256[:1].lower() == "f"


class CompactFFHQDataset(Dataset):
    """Reads from a single packed file produced by prepare_compact.py.

    Holds images as uint8 (N,3,H,W) in shared CPU RAM; normalises per-sample
    in __getitem__. Filters by SHA-prefix train/val split. ~3.7 GB for the
    full FFHQ detected subset at 224².
    """

    def __init__(
        self,
        compact_path: Path,
        split: Literal["train", "val"],
    ):
        blob = torch.load(compact_path, map_location="cpu", weights_only=False)
        self.images_u8 = blob["images_u8"]  # (N, 3, H, W) uint8
        self.arcface = blob["arcface"]      # (N, 512) fp32
        shas = blob["shas"]
        if not (len(shas) == self.images_u8.size(0) == self.arcface.size(0)):
            raise ValueError("compact file shape mismatch")

        self.indices = [
            i for i, sha in enumerate(shas)
            if (split == "val") == is_held_out(sha)
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        x = self.images_u8[idx].to(torch.float32) / 255.0
        x = (x - IMAGENET_MEAN_T) / IMAGENET_STD_T
        return x, self.arcface[idx]
