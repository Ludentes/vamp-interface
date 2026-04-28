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
    in __getitem__. Filters by SHA-prefix train/val split.

    `normalisation`:
      - 'imagenet': (x/255 - mean) / std — for ImageNet-pretrained students.
      - 'arcface' : (x - 127.5) / 127.5 — what buffalo_l R50 expects internally.
        Use this for the frozen-backbone adapter (Pixel-A).
      - 'none'    : raw uint8 → fp32 (no scaling). For latent-mode datasets
        that do their own thing.
    """

    def __init__(
        self,
        compact_path: Path,
        split: Literal["train", "val"],
        normalisation: Literal["imagenet", "arcface", "none"] = "imagenet",
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
        self.normalisation = normalisation

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        x = self.images_u8[idx].to(torch.float32)
        if self.normalisation == "imagenet":
            x = (x / 255.0 - IMAGENET_MEAN_T) / IMAGENET_STD_T
        elif self.normalisation == "arcface":
            x = (x - 127.5) / 127.5
        # 'none': leave as fp32 0..255
        return x, self.arcface[idx]


class CompactLatentDataset(Dataset):
    """Reads from a packed latent file: (N, 16, H, W) bf16/fp32 + arcface targets."""

    def __init__(
        self,
        compact_path: Path,
        split: Literal["train", "val"],
    ):
        blob = torch.load(compact_path, map_location="cpu", weights_only=False)
        self.latents = blob["latents"]      # (N, 16, H, W)  bf16 or fp32
        self.arcface = blob["arcface"]      # (N, 512) fp32
        shas = blob["shas"]
        if not (len(shas) == self.latents.size(0) == self.arcface.size(0)):
            raise ValueError("compact_latent file shape mismatch")
        # Aligned/precomputed corpora carry a `found` mask flagging rows whose
        # source latent was zeroed out (re-detection miss). Skip those.
        found = blob.get("found")
        self.indices = [
            i for i, sha in enumerate(shas)
            if (split == "val") == is_held_out(sha)
            and (found is None or bool(found[i]))
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        return self.latents[idx].to(torch.float32), self.arcface[idx]


class CompactLatentRoiDataset(Dataset):
    """Like CompactLatentDataset, but applies RoIAlign per row using the
    SCRFD-derived bbox in the side `face_attrs.pt` artefact, returning the
    face-region crop at fixed `output_size` (default 28).

    Bboxes are expected in latent coordinates [0..H] (where H = latent side).
    RoIAlign produces output_size × output_size regardless of bbox aspect.
    """

    def __init__(
        self,
        compact_path: Path,
        attrs_path: Path,
        split: Literal["train", "val"],
        output_size: int = 28,
    ):
        from torchvision.ops import roi_align  # local import keeps the module light

        compact = torch.load(compact_path, map_location="cpu", weights_only=False)
        attrs = torch.load(attrs_path, map_location="cpu", weights_only=False)
        if list(compact["shas"]) != list(attrs["shas"]):
            raise ValueError("compact + face_attrs SHA order mismatch")
        self.latents = compact["latents"]      # (N, 16, H, W) bf16/fp32
        self.arcface = compact["arcface"]      # (N, 512) fp32
        self.bboxes = attrs["bboxes_latent"]   # (N, 4) [x1,y1,x2,y2] in [0..H]
        # bbox_valid = re-detection succeeded AND bbox has positive area
        bbox_valid = ((attrs["bboxes_512"][:, 2] > attrs["bboxes_512"][:, 0]) &
                      (attrs["bboxes_512"][:, 3] > attrs["bboxes_512"][:, 1]))
        self._roi_align = roi_align
        self._output_size = output_size

        shas = compact["shas"]
        self.indices = [
            i for i, sha in enumerate(shas)
            if (split == "val") == is_held_out(sha) and bool(bbox_valid[i])
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        latent = self.latents[idx].to(torch.float32).unsqueeze(0)   # (1, 16, H, W)
        bbox = self.bboxes[idx]                                     # (4,)
        box = torch.tensor([[0.0, float(bbox[0]), float(bbox[1]),
                             float(bbox[2]), float(bbox[3])]])
        crop = self._roi_align(latent, box, output_size=(self._output_size, self._output_size),
                               spatial_scale=1.0, aligned=True)
        return crop.squeeze(0), self.arcface[idx]
