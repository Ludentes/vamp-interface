"""Joined latent ↔ blendshape dataset for MediaPipe distillation.

Two flavours:

  CompactLatentBlendshapeDataset — reads (compact.pt, compact_blendshapes.pt)
    aligned by SHA. The original v1 dataset on FFHQ.

  CombinedLatentBlendshapeDataset — concatenates a list of single-source
    datasets (FFHQ + rendered, etc.) for v2c. Each source can carry latents
    and blendshapes in either packing (FFHQ split via separate files; rendered
    via a single self-contained .pt with both inside).

Train/val split: SHA-prefix == 'f' → val for the FFHQ source (matches arc_distill
identity-held-out semantics). For the rendered source we use the same SHA rule
so any rendered identity that happens to start with 'f' is held out too.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import ConcatDataset, Dataset


def is_held_out(image_sha256: str) -> bool:
    return image_sha256[:1].lower() == "f"


class CompactLatentBlendshapeDataset(Dataset):
    """v1: separate compact.pt (latents+arcface) + compact_blendshapes.pt (bs)."""

    def __init__(
        self,
        compact_path: Path,
        blendshapes_path: Path,
        split: Literal["train", "val"],
    ):
        compact = torch.load(compact_path, map_location="cpu", weights_only=False)
        bs_blob = torch.load(blendshapes_path, map_location="cpu", weights_only=False)

        if list(compact["shas"]) != list(bs_blob["shas"]):
            raise ValueError("compact + blendshapes SHA order mismatch")

        self.latents = compact["latents"]              # (N, 16, 64, 64) bf16
        self.blendshapes = bs_blob["blendshapes"]      # (N, 52) fp32
        detected = bs_blob["detected"]                 # (N,) bool
        self.channel_names = bs_blob["channel_names"]
        shas = compact["shas"]

        self.indices = [
            i for i, sha in enumerate(shas)
            if (split == "val") == is_held_out(sha) and bool(detected[i])
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        latent = self.latents[idx].to(torch.float32)
        target = self.blendshapes[idx]
        return latent, target


class CompactRenderedDataset(Dataset):
    """v2c: single self-contained .pt with latents + blendshapes + detected
    + shas (produced by `encode_rendered_to_latent.py`)."""

    def __init__(
        self,
        compact_rendered_path: Path,
        split: Literal["train", "val"],
    ):
        blob = torch.load(compact_rendered_path, map_location="cpu", weights_only=False)
        self.latents = blob["latents"]                 # (N, 16, 64, 64) bf16
        self.blendshapes = blob["blendshapes"]         # (N, 52) fp32
        detected = blob["detected"]                    # (N,) bool
        self.channel_names = blob["channel_names"]
        shas = blob["shas"]
        self.indices = [
            i for i, sha in enumerate(shas)
            if (split == "val") == is_held_out(sha) and bool(detected[i])
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        latent = self.latents[idx].to(torch.float32)
        target = self.blendshapes[idx]
        return latent, target


def make_combined_dataset(
    split: Literal["train", "val"],
    compact_path: Path | None = None,
    blendshapes_path: Path | None = None,
    rendered_path: Path | None = None,
) -> Dataset:
    """v2c convenience: concat FFHQ + rendered if both provided. Channel names
    must match across sources (build_compact_blendshapes + encode_rendered_to_latent
    both write the same canonical 52-channel order)."""
    parts: list[Dataset] = []
    channel_names: list[str] | None = None
    if compact_path is not None and blendshapes_path is not None:
        d = CompactLatentBlendshapeDataset(compact_path, blendshapes_path, split)
        parts.append(d)
        channel_names = d.channel_names
    if rendered_path is not None:
        d = CompactRenderedDataset(rendered_path, split)
        if channel_names is None:
            channel_names = d.channel_names
        elif d.channel_names != channel_names:
            raise ValueError("channel name mismatch between sources")
        parts.append(d)
    if not parts:
        raise ValueError("must supply at least one source")
    if len(parts) == 1:
        merged: Dataset = parts[0]
    else:
        merged = ConcatDataset(parts)
    setattr(merged, "channel_names", channel_names)
    return merged
