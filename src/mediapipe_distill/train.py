"""Train the MediaPipe-blendshape student on (latent, 52-d blendshape) pairs.

Loss: MSE on the sigmoid output. The 52-d target is mostly sparse
(per-channel mean ≈ 0.09 across FFHQ); MSE handles this fine — most
channels are predicted near 0 and the loss is dominated by the rare
high-amplitude channels (smile, eye_blink, mouth_stretch).

Saves a per-epoch JSONL log next to the checkpoint with per-channel R²
on val so we can monitor whether the harder channels are actually
learning.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import make_combined_dataset
from .student import BlendshapeStudent


def per_channel_r2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-channel R² on (N, C). Target variance > 0 assumed (val n=1500
    over 52 channels, all blendshapes have nonzero variance in practice)."""
    mu = target.mean(dim=0, keepdim=True)
    ss_res = ((target - pred) ** 2).sum(dim=0)
    ss_tot = ((target - mu) ** 2).sum(dim=0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def evaluate(model: BlendshapeStudent, loader: DataLoader, device: torch.device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            pred = model(x).cpu()
            preds.append(pred)
            targets.append(y)
    p = torch.cat(preds, dim=0)
    t = torch.cat(targets, dim=0)
    mse = F.mse_loss(p, t).item()
    r2 = per_channel_r2(p, t)
    return {
        "mse": mse,
        "r2_mean": float(r2.mean()),
        "r2_median": float(r2.median()),
        "r2_p05": float(r2.kthvalue(max(1, int(0.05 * r2.numel()))).values),
        "r2_min": float(r2.min()),
        "n_channels_negative_r2": int((r2 < 0).sum()),
        "per_channel_r2": r2.tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="bs_a")
    p.add_argument("--compact", type=Path, default=None,
                   help="FFHQ compact.pt (latents+arcface). Pair with --blendshapes.")
    p.add_argument("--blendshapes", type=Path, default=None,
                   help="FFHQ compact_blendshapes.pt (52-d targets aligned to compact).")
    p.add_argument("--rendered", type=Path, default=None,
                   help="Optional rendered compact_rendered.pt (latents+blendshapes self-contained). "
                        "Combine with FFHQ for v2c.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"variant={args.variant} device={device}")

    print("loading datasets …")
    ds_train = make_combined_dataset("train",
                                     compact_path=args.compact,
                                     blendshapes_path=args.blendshapes,
                                     rendered_path=args.rendered)
    ds_val = make_combined_dataset("val",
                                   compact_path=args.compact,
                                   blendshapes_path=args.blendshapes,
                                   rendered_path=args.rendered)
    print(f"  train rows={len(ds_train)} val rows={len(ds_val)}")

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = BlendshapeStudent(args.variant).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: {n_params / 1e6:.2f} M total, {n_trainable / 1e6:.2f} M trainable")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    log_path = args.out_dir / "train_log.jsonl"
    log_f = log_path.open("w", buffering=1)

    best_r2 = -1e9
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running, n_batches = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())
            n_batches += 1
        sched.step()
        train_mse = running / max(1, n_batches)

        ev = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        rec = {
            "epoch": epoch,
            "train_mse": train_mse,
            "val_mse": ev["mse"],
            "val_r2_mean": ev["r2_mean"],
            "val_r2_median": ev["r2_median"],
            "val_r2_p05": ev["r2_p05"],
            "val_n_neg_r2": ev["n_channels_negative_r2"],
            "lr": float(opt.param_groups[0]["lr"]),
            "elapsed_s": round(elapsed, 1),
        }
        log_f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec))

        torch.save({"model": model.state_dict(), "epoch": epoch, "variant": args.variant},
                   args.out_dir / "last.pt")
        if ev["r2_mean"] > best_r2:
            best_r2 = ev["r2_mean"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "variant": args.variant,
                        "val_r2_mean": ev["r2_mean"]},
                       args.out_dir / "checkpoint.pt")

    # final per-channel detail
    final = evaluate(model, val_loader, device)
    with (args.out_dir / "eval.json").open("w") as f:
        json.dump({
            "variant": args.variant,
            "val_n": len(ds_val),
            "channel_names": ds_val.channel_names,
            **final,
        }, f, indent=2)
    log_f.close()
    print(f"\nbest val r2_mean: {best_r2:.4f}")


if __name__ == "__main__":
    main()
