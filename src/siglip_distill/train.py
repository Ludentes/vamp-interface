"""Train SigLIP-distill student on (latent, 1152-d siglip emb) pairs.

Loss: 0.5 * MSE + 0.5 * (1 - cos). MSE keeps magnitude near 1; (1 - cos)
optimizes the directional quantity downstream slider losses use.

Per-epoch JSONL log with cosine + MSE on val.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import CompactLatentSiglipDataset
from .student import SigLIPStudent


def loss_fn(pred: torch.Tensor, target: torch.Tensor, variant: str) -> tuple[torch.Tensor, dict]:
    """
    sg_a: 0.5·MSE + 0.5·(1 − cos) — original v1 mixed loss; lets pred_norm
          drift, requires MSE term to keep magnitude near 1.
    sg_b: pure (1 − cos) — student is L2-normed at the head, so MSE between
          unit vectors is monotonic in cosine; drop it for clarity.
    """
    cos = F.cosine_similarity(pred, target, dim=-1).mean()
    mse = ((pred - target) ** 2).mean()
    if variant == "sg_b":
        loss = 1.0 - cos
    else:
        loss = 0.5 * mse + 0.5 * (1.0 - cos)
    return loss, {"mse": float(mse.item()), "cos": float(cos.item())}


def evaluate(model: SigLIPStudent, loader: DataLoader, device: torch.device):
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
    mse = float(F.mse_loss(p, t).item())
    cos_per = F.cosine_similarity(p, t, dim=-1)
    p_norm = p / (p.norm(dim=-1, keepdim=True) + 1e-12)
    cos_unit_per = F.cosine_similarity(p_norm, t, dim=-1)
    return {
        "mse": mse,
        "cos_mean": float(cos_per.mean()),
        "cos_median": float(cos_per.median()),
        "cos_p05": float(cos_per.kthvalue(max(1, int(0.05 * cos_per.numel()))).values),
        "cos_unit_mean": float(cos_unit_per.mean()),
        "cos_unit_median": float(cos_unit_per.median()),
        "pred_norm_mean": float(p.norm(dim=-1).mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="sg_a")
    p.add_argument("--compact", type=Path, required=True,
                   help="arc_full_latent/compact.pt (latents+arcface).")
    p.add_argument("--siglip", type=Path, required=True,
                   help="compact_siglip.pt (1152-d siglip embs aligned to compact).")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--grad-clip", type=float, default=1.0)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"variant={args.variant} device={device}")

    print("loading datasets …")
    ds_train = CompactLatentSiglipDataset(args.compact, args.siglip, "train")
    ds_val = CompactLatentSiglipDataset(args.compact, args.siglip, "val")
    print(f"  train rows={len(ds_train)} val rows={len(ds_val)}")

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = SigLIPStudent(args.variant).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params / 1e6:.2f} M total")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    log_path = args.out_dir / "train_log.jsonl"
    log_f = log_path.open("w", buffering=1)

    best_cos = -1e9
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_mse = 0.0
        running_cos = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            loss, parts = loss_fn(pred, y, args.variant)
            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            running_loss += float(loss.item())
            running_mse += parts["mse"]
            running_cos += parts["cos"]
            n_batches += 1
        sched.step()
        nb = max(1, n_batches)

        ev = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        rec = {
            "epoch": epoch,
            "train_loss": running_loss / nb,
            "train_mse": running_mse / nb,
            "train_cos": running_cos / nb,
            "val_mse": ev["mse"],
            "val_cos_mean": ev["cos_mean"],
            "val_cos_median": ev["cos_median"],
            "val_cos_p05": ev["cos_p05"],
            "val_cos_unit_mean": ev["cos_unit_mean"],
            "val_pred_norm": ev["pred_norm_mean"],
            "lr": float(opt.param_groups[0]["lr"]),
            "elapsed_s": round(elapsed, 1),
        }
        log_f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec))

        torch.save({"model": model.state_dict(), "epoch": epoch, "variant": args.variant},
                   args.out_dir / "last.pt")
        if ev["cos_mean"] > best_cos:
            best_cos = ev["cos_mean"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "variant": args.variant,
                        "val_cos_mean": ev["cos_mean"]},
                       args.out_dir / "checkpoint.pt")

    final = evaluate(model, val_loader, device)
    with (args.out_dir / "eval.json").open("w") as f:
        json.dump({
            "variant": args.variant,
            "val_n": len(ds_val),
            "emb_dim": ds_val.emb_dim,
            **final,
        }, f, indent=2)
    log_f.close()
    print(f"\nbest val cos_mean: {best_cos:.4f}")


if __name__ == "__main__":
    main()
