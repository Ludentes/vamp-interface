"""Distill loop: per-frame MSE on (b_expr, m_f) pairs."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from arkit_bridge.dataset import PairDataset
from arkit_bridge.student import MotEncoderStudent


def train(
    pairs_dir: str | Path,
    out_dir: str | Path,
    *,
    batch_size: int = 64,
    lr: float = 5e-4,
    steps: int = 20000,
    log_every: int = 100,
    ckpt_every: int = 2000,
    device: str = "cuda",
):
    os.makedirs(out_dir, exist_ok=True)
    ds = PairDataset(pairs_dir)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=2, drop_last=True, persistent_workers=True,
    )
    s = MotEncoderStudent().to(device)
    opt = torch.optim.AdamW(s.parameters(), lr=lr)
    log: list[dict] = []
    step = 0
    t0 = time.time()
    while step < steps:
        for b, m in dl:
            b = b.to(device); m = m.to(device)
            loss = F.mse_loss(s(b), m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % log_every == 0:
                rate = step / max(1e-6, time.time() - t0)
                print(f"step {step:6d}  loss {loss.item():.5f}  ({rate:.1f}/s)")
                log.append({"step": step, "loss": float(loss)})
            if step % ckpt_every == 0 or step >= steps:
                torch.save(s.state_dict(), Path(out_dir) / f"student_step{step:06d}.pt")
                json.dump(log, open(Path(out_dir) / "log.json", "w"))
            if step >= steps:
                break
    return s
