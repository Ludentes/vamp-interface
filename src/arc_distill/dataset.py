"""FFHQ pixel + ArcFace teacher dataset for arc_pixel distillation."""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Literal

import pyarrow.parquet as pq
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def is_held_out(image_sha256: str) -> bool:
    """Deterministic 6.25% held-out split: SHA prefix 'f' is val."""
    return image_sha256[:1].lower() == "f"


class FFHQPixelDataset(Dataset):
    """One parquet shard joined with its matching encoded .pt by row order.

    The encoded .pt was produced by encode_ffhq.py in the same row order as
    the parquet shard. We align positionally and verify row counts match.
    Filters to detected=True and applies the SHA-prefix train/val split.
    """

    def __init__(
        self,
        parquet_path: Path,
        encoded_pt_path: Path,
        split: Literal["train", "val"],
        resolution: int = 224,
    ):
        self.parquet_path = Path(parquet_path)
        table = pq.read_table(self.parquet_path, columns=["image"])
        image_bytes_col = table.column("image").to_pylist()

        pt = torch.load(encoded_pt_path, map_location="cpu", weights_only=False)
        pt_shas = pt["image_sha256"]
        arcface = pt["arcface_fp32"]
        detected = pt["detected"]

        if len(image_bytes_col) != len(pt_shas):
            raise ValueError(
                f"row count mismatch: parquet={len(image_bytes_col)} "
                f"pt={len(pt_shas)} for {self.parquet_path}"
            )

        keep = []
        for i, (sha, det) in enumerate(zip(pt_shas, detected.tolist())):
            if not det:
                continue
            held = is_held_out(sha)
            if (split == "val") == held:
                keep.append(i)

        self.indices = keep
        self.image_bytes = image_bytes_col
        self.shas = pt_shas
        self.targets = arcface
        self.transform = T.Compose([
            T.Resize((resolution, resolution), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        rec = self.image_bytes[idx]
        img = Image.open(io.BytesIO(rec["bytes"])).convert("RGB")
        x = self.transform(img)
        y = self.targets[idx].to(torch.float32)
        return x, y


_SHARD_RE = re.compile(r"train-(\d{5})-of-\d{5}\.parquet$")


def build_ffhq_concat(
    shards_dir: Path,
    encoded_dir: Path,
    split: Literal["train", "val"],
    resolution: int = 224,
) -> ConcatDataset:
    """Build a ConcatDataset over every shard whose .parquet has a matching .pt.

    Skips shards missing their encoded .pt without raising — useful while the
    encode_ffhq.py run is still in flight.
    """
    shards_dir = Path(shards_dir)
    encoded_dir = Path(encoded_dir)
    parts: list[FFHQPixelDataset] = []
    for p in sorted(shards_dir.glob("train-*-of-*.parquet")):
        m = _SHARD_RE.search(p.name)
        if not m:
            continue
        # Encoded .pt filenames mirror the parquet basename (train-00000-of-00190.pt).
        pt = encoded_dir / (p.stem + ".pt")
        if not pt.exists():
            continue
        parts.append(FFHQPixelDataset(p, pt, split=split, resolution=resolution))
    if not parts:
        raise FileNotFoundError(f"no matched shard pairs under {shards_dir} / {encoded_dir}")
    return ConcatDataset(parts)
