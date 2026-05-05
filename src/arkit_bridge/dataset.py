"""Iterate cached (b₆₁, T) pkls as a torch Dataset."""

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class DistillPairDataset(Dataset):
    def __init__(self, root: str | Path):
        self.paths = sorted(Path(root).glob("frame_*.pkl"))
        if not self.paths:
            raise FileNotFoundError(f"no frame pkls under {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with open(self.paths[i], "rb") as f:
            d = pickle.load(f)
        b = torch.from_numpy(np.asarray(d["b"], dtype=np.float32))
        T = torch.from_numpy(np.asarray(d["T"], dtype=np.float32))
        return b, T
