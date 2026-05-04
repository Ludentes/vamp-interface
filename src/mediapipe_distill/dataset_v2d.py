"""v2d dataset variants — return (latent, bs_target, lmk_target_norm).

Rows where either bs_detected or lmk_found is False are excluded entirely:
landmark-miss rows often coincide with degenerate face renders whose bs
targets are also unreliable, so we don't train on them.

lmk_target_norm: (106, 2) float32 in [0, 1] (pixel coords / 512).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import ConcatDataset, Dataset

from .dataset import is_held_out


class CompactLatentBSLmkDataset(Dataset):
    """FFHQ source: compact.pt + compact_blendshapes.pt + compact_landmarks.pt."""

    def __init__(
        self,
        compact_path: Path,
        blendshapes_path: Path,
        landmarks_path: Path,
        split: Literal["train", "val"],
    ):
        compact = torch.load(compact_path, map_location="cpu", weights_only=False)
        bs_blob = torch.load(blendshapes_path, map_location="cpu", weights_only=False)
        lmk_blob = torch.load(landmarks_path, map_location="cpu", weights_only=False)

        if list(compact["shas"]) != list(bs_blob["shas"]):
            raise ValueError("compact + blendshapes SHA order mismatch")
        if list(compact["shas"]) != list(lmk_blob["shas"]):
            raise ValueError("compact + landmarks SHA order mismatch")

        self.latents = compact["latents"]              # (N, 16, 64, 64) bf16
        self.blendshapes = bs_blob["blendshapes"]      # (N, 52)
        self.landmarks = lmk_blob["landmark_2d_106"] / 512.0  # → [0, 1]
        bs_detected = bs_blob["detected"]
        lmk_found = lmk_blob["found"]
        self.lmk_valid = lmk_found
        self.channel_names = bs_blob["channel_names"]
        shas = compact["shas"]

        # require BOTH bs detection AND landmark detection — v2d has no use
        # for rows missing either signal, and landmark-miss rows often coincide
        # with degenerate face renders that would poison the bs head too.
        self.indices = [
            i for i, sha in enumerate(shas)
            if (split == "val") == is_held_out(sha)
            and bool(bs_detected[i]) and bool(lmk_found[i])
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        latent = self.latents[idx].to(torch.float32)
        bs = self.blendshapes[idx]
        lmk = self.landmarks[idx]
        return latent, bs, lmk


class CompactRenderedBSLmkDataset(Dataset):
    """Rendered source: compact_rendered.pt + compact_rendered_landmarks.pt."""

    def __init__(
        self,
        compact_rendered_path: Path,
        landmarks_path: Path,
        split: Literal["train", "val"],
    ):
        rblob = torch.load(compact_rendered_path, map_location="cpu", weights_only=False)
        lmk_blob = torch.load(landmarks_path, map_location="cpu", weights_only=False)

        if list(rblob["shas"]) != list(lmk_blob["shas"]):
            raise ValueError("rendered + landmarks SHA order mismatch")

        self.latents = rblob["latents"]
        self.blendshapes = rblob["blendshapes"]
        self.landmarks = lmk_blob["landmark_2d_106"] / 512.0
        bs_detected = rblob["detected"]
        self.lmk_valid = lmk_blob["found"]
        self.channel_names = rblob["channel_names"]
        shas = rblob["shas"]

        self.indices = [
            i for i, sha in enumerate(shas)
            if (split == "val") == is_held_out(sha)
            and bool(bs_detected[i]) and bool(self.lmk_valid[i])
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        latent = self.latents[idx].to(torch.float32)
        bs = self.blendshapes[idx]
        lmk = self.landmarks[idx]
        return latent, bs, lmk


def make_combined_v2d_dataset(
    split: Literal["train", "val"],
    compact_path: Path | None = None,
    blendshapes_path: Path | None = None,
    landmarks_path: Path | None = None,
    rendered_path: Path | None = None,
    rendered_landmarks_path: Path | None = None,
) -> Dataset:
    parts: list[Dataset] = []
    channel_names: list[str] | None = None
    if compact_path is not None and blendshapes_path is not None and landmarks_path is not None:
        d = CompactLatentBSLmkDataset(compact_path, blendshapes_path, landmarks_path, split)
        parts.append(d)
        channel_names = d.channel_names
    if rendered_path is not None and rendered_landmarks_path is not None:
        d = CompactRenderedBSLmkDataset(rendered_path, rendered_landmarks_path, split)
        if channel_names is None:
            channel_names = d.channel_names
        elif d.channel_names != channel_names:
            raise ValueError("channel name mismatch between sources")
        parts.append(d)
    if not parts:
        raise ValueError("must supply at least one source")
    merged: Dataset = parts[0] if len(parts) == 1 else ConcatDataset(parts)
    setattr(merged, "channel_names", channel_names)
    return merged
