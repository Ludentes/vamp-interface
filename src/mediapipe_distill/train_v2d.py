"""Train the two-head BlendshapeLandmarkStudent on (latent, bs, lmk).

Loss = λ_lmk · MSE(lmk_pred, lmk_norm) + λ_bs · MSE(bs_pred, bs_target).
Default λ_lmk=0.5, λ_bs=1.0 — see student_v2d.py docstring.

Per-epoch JSONL log records both heads' losses and per-channel R² for the
blendshape head (so it stays comparable to v1 / v2c).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset_v2d import make_combined_v2d_dataset
from .student_v2d import BlendshapeLandmarkStudent


def per_channel_r2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mu = target.mean(dim=0, keepdim=True)
    ss_res = ((target - pred) ** 2).sum(dim=0)
    ss_tot = ((target - mu) ** 2).sum(dim=0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def evaluate(model: BlendshapeLandmarkStudent, loader: DataLoader, device: torch.device):
    model.eval()
    bs_preds, bs_targets = [], []
    lmk_preds, lmk_targets = [], []
    with torch.no_grad():
        for x, bs, lmk in loader:
            x = x.to(device, non_blocking=True)
            bs_p, lmk_p = model(x)
            bs_preds.append(bs_p.cpu())
            bs_targets.append(bs)
            lmk_preds.append(lmk_p.cpu().reshape(-1, 106, 2))
            lmk_targets.append(lmk)
    bp = torch.cat(bs_preds, dim=0)
    bt = torch.cat(bs_targets, dim=0)
    lp = torch.cat(lmk_preds, dim=0)
    lt = torch.cat(lmk_targets, dim=0)
    bs_mse = F.mse_loss(bp, bt).item()
    lmk_mse = F.mse_loss(lp, lt).item()
    # lmk pixel-RMSE (un-normalize ×512)
    lmk_pix_rmse = float(((lp - lt) * 512.0).square().mean().sqrt())
    r2 = per_channel_r2(bp, bt)
    return {
        "bs_mse": bs_mse,
        "lmk_mse": lmk_mse,
        "lmk_pix_rmse": lmk_pix_rmse,
        "r2_mean": float(r2.mean()),
        "r2_median": float(r2.median()),
        "r2_p05": float(r2.kthvalue(max(1, int(0.05 * r2.numel()))).values),
        "r2_min": float(r2.min()),
        "n_channels_negative_r2": int((r2 < 0).sum()),
        "per_channel_r2": r2.tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="bs_v2d")
    p.add_argument("--compact", type=Path, required=True)
    p.add_argument("--blendshapes", type=Path, required=True)
    p.add_argument("--landmarks", type=Path, required=True)
    p.add_argument("--rendered", type=Path, default=None)
    p.add_argument("--rendered-landmarks", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-lmk", type=float, default=0.5)
    p.add_argument("--lambda-bs", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"variant={args.variant} device={device} λ_lmk={args.lambda_lmk} λ_bs={args.lambda_bs}")

    print("loading datasets …")
    ds_train = make_combined_v2d_dataset("train",
                                          compact_path=args.compact,
                                          blendshapes_path=args.blendshapes,
                                          landmarks_path=args.landmarks,
                                          rendered_path=args.rendered,
                                          rendered_landmarks_path=args.rendered_landmarks)
    ds_val = make_combined_v2d_dataset("val",
                                        compact_path=args.compact,
                                        blendshapes_path=args.blendshapes,
                                        landmarks_path=args.landmarks,
                                        rendered_path=args.rendered,
                                        rendered_landmarks_path=args.rendered_landmarks)
    print(f"  train rows={len(ds_train)} val rows={len(ds_val)}")

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = BlendshapeLandmarkStudent(args.variant).to(device)
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
        running_total, running_bs, running_lmk, n_batches = 0.0, 0.0, 0.0, 0
        for x, bs, lmk in train_loader:
            x = x.to(device, non_blocking=True)
            bs = bs.to(device, non_blocking=True)
            lmk = lmk.to(device, non_blocking=True)
            bs_p, lmk_p = model(x)
            lmk_p = lmk_p.reshape(-1, 106, 2)
            loss_bs = F.mse_loss(bs_p, bs)
            loss_lmk = F.mse_loss(lmk_p, lmk)
            loss = args.lambda_bs * loss_bs + args.lambda_lmk * loss_lmk
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_total += float(loss.item())
            running_bs += float(loss_bs.item())
            running_lmk += float(loss_lmk.item())
            n_batches += 1
        sched.step()

        ev = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        rec = {
            "epoch": epoch,
            "train_loss": running_total / max(1, n_batches),
            "train_bs_mse": running_bs / max(1, n_batches),
            "train_lmk_mse": running_lmk / max(1, n_batches),
            "val_bs_mse": ev["bs_mse"],
            "val_lmk_mse": ev["lmk_mse"],
            "val_lmk_pix_rmse": ev["lmk_pix_rmse"],
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
        if ev["r2_median"] > best_r2:
            best_r2 = ev["r2_median"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "variant": args.variant,
                        "val_r2_median": ev["r2_median"]},
                       args.out_dir / "checkpoint.pt")

    final = evaluate(model, val_loader, device)
    with (args.out_dir / "eval.json").open("w") as f:
        json.dump({
            "variant": args.variant,
            "val_n": len(ds_val),
            "channel_names": ds_val.channel_names,
            **final,
        }, f, indent=2)
    log_f.close()
    print(f"\nbest val r2_median: {best_r2:.4f}")


if __name__ == "__main__":
    main()
