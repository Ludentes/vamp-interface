"""Train AtomStudent (variant `bs_v2b`) on (latent, 8-d atom) pairs.

Same trunk as v2c; head is Linear(512, 8) with no activation. Loss = MSE on
atoms. Per-epoch JSONL log records per-atom R² so you can spot if any
atom isn't learning.

20 epochs by default — v2c showed this is enough; v2e showed longer leads
to overfitting without help.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset_v2b import make_combined_atom_dataset
from .student_v2b import AtomStudent


def per_channel_r2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mu = target.mean(dim=0, keepdim=True)
    ss_res = ((target - pred) ** 2).sum(dim=0)
    ss_tot = ((target - mu) ** 2).sum(dim=0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def evaluate(model: AtomStudent, loader: DataLoader, device: torch.device, atom_tags):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            preds.append(model(x).cpu())
            targets.append(y)
    p = torch.cat(preds, dim=0)
    t = torch.cat(targets, dim=0)
    mse = F.mse_loss(p, t).item()
    r2 = per_channel_r2(p, t)
    per_atom = {tag: float(r2[i]) for i, tag in enumerate(atom_tags)}
    return {
        "mse": mse,
        "r2_mean": float(r2.mean()),
        "r2_median": float(r2.median()),
        "r2_min": float(r2.min()),
        "n_atoms_negative_r2": int((r2 < 0).sum()),
        "per_atom_r2": per_atom,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="bs_v2b")
    p.add_argument("--compact", type=Path, required=True)
    p.add_argument("--atoms", type=Path, required=True)
    p.add_argument("--rendered", type=Path, default=None)
    p.add_argument("--rendered-atoms", type=Path, default=None)
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
    print(f"variant={args.variant} device={device} epochs={args.epochs}")

    print("loading datasets …")
    ds_train = make_combined_atom_dataset("train",
                                          compact_path=args.compact,
                                          atoms_path=args.atoms,
                                          rendered_path=args.rendered,
                                          rendered_atoms_path=args.rendered_atoms)
    ds_val = make_combined_atom_dataset("val",
                                        compact_path=args.compact,
                                        atoms_path=args.atoms,
                                        rendered_path=args.rendered,
                                        rendered_atoms_path=args.rendered_atoms)
    atom_tags = ds_val.atom_tags
    print(f"  train rows={len(ds_train)} val rows={len(ds_val)} atoms={atom_tags}")

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = AtomStudent(args.variant).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params / 1e6:.2f} M")

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            running += float(loss.item())
            n_batches += 1
        sched.step()

        ev = evaluate(model, val_loader, device, atom_tags)
        elapsed = time.time() - t0
        rec = {
            "epoch": epoch,
            "train_mse": running / max(1, n_batches),
            "val_mse": ev["mse"],
            "val_r2_mean": ev["r2_mean"],
            "val_r2_median": ev["r2_median"],
            "val_r2_min": ev["r2_min"],
            "val_n_neg_r2": ev["n_atoms_negative_r2"],
            "per_atom_r2": ev["per_atom_r2"],
            "lr": float(opt.param_groups[0]["lr"]),
            "elapsed_s": round(elapsed, 1),
        }
        log_f.write(json.dumps(rec) + "\n")
        print(json.dumps({k: v for k, v in rec.items() if k != "per_atom_r2"}))

        ck_payload = {
            "model": model.state_dict(),
            "epoch": epoch,
            "variant": args.variant,
            "atom_tags": atom_tags,
            "val_r2_min": ev["r2_min"],
            "val_r2_mean": ev["r2_mean"],
        }
        torch.save(ck_payload, args.out_dir / "last.pt")
        # Best by minimum-atom R² — for slider use, you want every atom usable,
        # not just the average to be high.
        if ev["r2_min"] > best_r2:
            best_r2 = ev["r2_min"]
            torch.save(ck_payload, args.out_dir / "checkpoint.pt")

    final = evaluate(model, val_loader, device, atom_tags)
    with (args.out_dir / "eval.json").open("w") as f:
        json.dump({
            "variant": args.variant,
            "val_n": len(ds_val),
            "atom_tags": atom_tags,
            **final,
        }, f, indent=2)
    log_f.close()
    print(f"\nbest val r2_min: {best_r2:.4f}")


if __name__ == "__main__":
    main()
