"""v2b dataset — returns (latent, 8-d atom target). Mirrors dataset.py but
swaps blendshapes for atom targets produced by build_compact_atoms.py."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import ConcatDataset, Dataset

from .dataset import is_held_out


class CompactLatentAtomDataset(Dataset):
    """FFHQ source: compact.pt + compact_atoms.pt."""

    def __init__(
        self,
        compact_path: Path,
        atoms_path: Path,
        split: Literal["train", "val"],
    ):
        compact = torch.load(compact_path, map_location="cpu", weights_only=False)
        atoms_blob = torch.load(atoms_path, map_location="cpu", weights_only=False)

        if list(compact["shas"]) != list(atoms_blob["shas"]):
            raise ValueError("compact + atoms SHA order mismatch")

        self.latents = compact["latents"]              # (N, 16, 64, 64) bf16
        self.atoms = atoms_blob["atoms"]               # (N, 8) fp32
        detected = atoms_blob["detected"]              # (N,) bool
        self.atom_tags = atoms_blob["atom_tags"]
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
        target = self.atoms[idx]
        return latent, target


class CompactRenderedAtomDataset(Dataset):
    """Rendered source: compact_rendered.pt (for latents) + compact_rendered_atoms.pt."""

    def __init__(
        self,
        compact_rendered_path: Path,
        rendered_atoms_path: Path,
        split: Literal["train", "val"],
    ):
        rblob = torch.load(compact_rendered_path, map_location="cpu", weights_only=False)
        ablob = torch.load(rendered_atoms_path, map_location="cpu", weights_only=False)
        if list(rblob["shas"]) != list(ablob["shas"]):
            raise ValueError("rendered + atoms SHA order mismatch")

        self.latents = rblob["latents"]
        self.atoms = ablob["atoms"]
        detected = ablob["detected"]
        self.atom_tags = ablob["atom_tags"]
        shas = rblob["shas"]

        self.indices = [
            i for i, sha in enumerate(shas)
            if (split == "val") == is_held_out(sha) and bool(detected[i])
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        latent = self.latents[idx].to(torch.float32)
        target = self.atoms[idx]
        return latent, target


def make_combined_atom_dataset(
    split: Literal["train", "val"],
    compact_path: Path | None = None,
    atoms_path: Path | None = None,
    rendered_path: Path | None = None,
    rendered_atoms_path: Path | None = None,
) -> Dataset:
    parts: list[Dataset] = []
    atom_tags: list[str] | None = None
    if compact_path is not None and atoms_path is not None:
        d = CompactLatentAtomDataset(compact_path, atoms_path, split)
        parts.append(d)
        atom_tags = d.atom_tags
    if rendered_path is not None and rendered_atoms_path is not None:
        d = CompactRenderedAtomDataset(rendered_path, rendered_atoms_path, split)
        if atom_tags is None:
            atom_tags = d.atom_tags
        elif d.atom_tags != atom_tags:
            raise ValueError("atom tag mismatch between sources")
        parts.append(d)
    if not parts:
        raise ValueError("must supply at least one source")
    merged: Dataset = parts[0] if len(parts) == 1 else ConcatDataset(parts)
    setattr(merged, "atom_tags", atom_tags)
    return merged
