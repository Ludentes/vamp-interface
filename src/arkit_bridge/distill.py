"""Distillation training loop. MSE between student(b₆₁) and cached teacher T."""

import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from arkit_bridge.dataset import DistillPairDataset
from arkit_bridge.encoder import ARKitParametricPoseGuider


def train(
    pairs_dir: str,
    out_dir: str,
    *,
    batch_size: int = 32,
    lr: float = 1e-3,
    steps: int = 5000,
    log_every: int = 50,
    ckpt_every: int = 1000,
    device: str = "cuda",
):
    os.makedirs(out_dir, exist_ok=True)
    ds = DistillPairDataset(pairs_dir)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    num_workers=2, drop_last=True)
    student = ARKitParametricPoseGuider().to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    log = []
    step = 0
    t0 = time.time()
    while step < steps:
        for b, T in dl:
            b = b.to(device); T = T.to(device)
            pred = student(b)
            loss = F.mse_loss(pred, T)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % log_every == 0:
                rate = step / max(1e-6, time.time() - t0)
                print(f"step {step:5d}  loss {loss.item():.5f}  "
                      f"({rate:.1f} step/s)")
                log.append({"step": step, "loss": float(loss.item())})
            if step % ckpt_every == 0 or step >= steps:
                ckpt = Path(out_dir) / f"student_step{step:06d}.pt"
                torch.save(student.state_dict(), ckpt)
                with open(Path(out_dir) / "log.json", "w") as f:
                    json.dump(log, f)
            if step >= steps:
                break
    return student
