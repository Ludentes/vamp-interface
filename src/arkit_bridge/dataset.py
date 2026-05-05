"""(b_expr, m_f) pair loader for the MotEncoder distill."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PairDataset(Dataset):
    def __init__(self, root: str | Path):
        self.paths = sorted(Path(root).glob("frame_*.pkl"))
        if not self.paths:
            raise FileNotFoundError(f"no frame pkls under {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        with open(self.paths[i], "rb") as f:
            d = pickle.load(f)
        return (
            torch.from_numpy(np.asarray(d["b_expr"], dtype=np.float32)),
            torch.from_numpy(np.asarray(d["m_f"], dtype=np.float32)),
        )
