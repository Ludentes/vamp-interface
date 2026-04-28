"""Train an AdapterStudent (frozen IResNet50 + trainable stem) by cosine
distillation against the buffalo_l ArcFace teacher embedding.

Resumable from <out-dir>/last.pt; tracks best-by-val-cosine to checkpoint.pt;
writes per-epoch jsonl to train_log.jsonl.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .adapter import AdapterStudent
from .dataset import CompactFFHQDataset, CompactLatentDataset


def cosine_distance_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = F.normalize(target, dim=-1)
    return (1.0 - (pred * target).sum(dim=-1)).mean()


def build_loaders(args):
    if args.variant == "pixel_a":
        train_ds = CompactFFHQDataset(args.compact, "train", normalisation="arcface")
        val_ds = CompactFFHQDataset(args.compact, "val", normalisation="arcface")
    else:
        train_ds = CompactLatentDataset(args.compact, "train")
        val_ds = CompactLatentDataset(args.compact, "val")

    if args.smoke:
        train_ds.indices = train_ds.indices[: args.batch_size * 4]
        val_ds.indices = val_ds.indices[: args.batch_size]

    return (
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                   num_workers=args.workers, pin_memory=True, drop_last=True),
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                   num_workers=args.workers, pin_memory=True),
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    cs = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        z = model(x)
        y = F.normalize(y, dim=-1)
        cs.append((z * y).sum(dim=-1))
    return torch.cat(cs).mean().item() if cs else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True,
                   choices=["pixel_a", "latent_a_up", "latent_a_native"])
    p.add_argument("--compact", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--onnx-path", type=Path, default=None,
                   help="Override path to w600k_r50.onnx (defaults to ~/.insightface)")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "train_log.jsonl"
    last_path = args.out_dir / "last.pt"
    best_path = args.out_dir / "checkpoint.pt"

    device = torch.device(args.device)
    print(f"variant={args.variant} device={device} compact={args.compact}")

    train_loader, val_loader = build_loaders(args)
    print(f"train rows={len(train_loader.dataset)} val rows={len(val_loader.dataset)}")

    onnx_path = args.onnx_path if args.onnx_path else None
    model = AdapterStudent(args.variant) if onnx_path is None else AdapterStudent(args.variant, onnx_path)
    model = model.to(device)
    n_train = sum(p.numel() for p in model.trainable_parameters())
    print(f"trainable params: {n_train:,}")

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    start_epoch = 0
    best_val = -float("inf")
    if last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch = ck["epoch"] + 1
        best_val = ck.get("best_val", -float("inf"))
        print(f"resumed from epoch {start_epoch} best_val={best_val:.4f}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        losses = []
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            z = model(x)
            loss = cosine_distance_loss(z, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()

        train_cos_loss = float(sum(losses) / max(1, len(losses)))
        val_cos = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        rec = {
            "epoch": epoch,
            "train_cos_loss": train_cos_loss,
            "val_cosine_mean": val_cos,
            "val_n": len(val_loader.dataset),
            "lr": opt.param_groups[0]["lr"],
            "elapsed_s": round(elapsed, 1),
        }
        print(json.dumps(rec))
        with log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

        ck = {
            "epoch": epoch, "model": model.state_dict(),
            "opt": opt.state_dict(), "sched": sched.state_dict(),
            "variant": args.variant, "best_val": best_val,
        }
        torch.save(ck, last_path)
        if val_cos > best_val:
            best_val = val_cos
            ck["best_val"] = best_val
            torch.save(ck, best_path)


if __name__ == "__main__":
    main()
