"""Train ArcStudentResNet18 on FFHQ pixels → ArcFace teacher embeddings.

Resumable: writes `checkpoint.pt` (best by val cosine) and `last.pt` every epoch.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from arc_distill.dataset import build_ffhq_concat
from arc_distill.model import ArcStudentResNet18, cosine_distance_loss


def evaluate(model, loader, device) -> dict:
    model.eval()
    cos_sum = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            p = model(x)
            yn = F.normalize(y, dim=-1)
            cos_sum += (p * yn).sum().item()
            n += x.size(0)
    return {"val_cosine_mean": cos_sum / max(n, 1), "val_n": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-dir", type=Path, required=True)
    ap.add_argument("--encoded-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--smoke", action="store_true",
                    help="Use a single sub-dataset, 1 epoch — sanity only.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "train_log.jsonl"

    print(f"[{time.strftime('%H:%M:%S')}] building datasets...")
    train_full = build_ffhq_concat(args.shards_dir, args.encoded_dir, "train", args.resolution)
    val = build_ffhq_concat(args.shards_dir, args.encoded_dir, "val", args.resolution)

    if args.smoke:
        train = train_full.datasets[0]
        epochs = 1
    else:
        train = train_full
        epochs = args.epochs

    print(f"train={len(train)} val={len(val)}")

    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    model = ArcStudentResNet18(pretrained=True).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = -1.0
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        n = 0
        for step, (x, y) in enumerate(train_loader):
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            p = model(x)
            loss = cosine_distance_loss(p, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * x.size(0)
            n += x.size(0)
            if step % 50 == 0:
                print(f"  epoch={epoch} step={step}/{len(train_loader)} "
                      f"loss={loss.item():.4f}")
        sched.step()
        train_loss = loss_sum / max(n, 1)
        val_metrics = evaluate(model, val_loader, args.device)
        elapsed = time.time() - t0
        rec = {"epoch": epoch, "train_cos_loss": train_loss,
               "elapsed_s": round(elapsed, 1), **val_metrics}
        print(f"[epoch {epoch}] {rec}")
        with log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "val_cosine_mean": val_metrics["val_cosine_mean"]},
                   args.out_dir / "last.pt")
        if val_metrics["val_cosine_mean"] > best_val:
            best_val = val_metrics["val_cosine_mean"]
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "val_cosine_mean": best_val},
                       args.out_dir / "checkpoint.pt")
            print(f"  saved best (val_cosine_mean={best_val:.4f})")

    print(f"done. best val_cosine_mean={best_val:.4f}")


if __name__ == "__main__":
    main()
