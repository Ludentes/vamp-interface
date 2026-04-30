"""PGD-robustness validator for blendshape critics.

Loads a checkpoint, runs PGD-K attack at each ε ∈ EPS_BUCKETS, computes
per-channel R² of the attacked predictions vs the true teacher labels.
A robust critic shows R² roughly flat across ε; a non-robust critic
collapses (negative R²) at small ε.

Output: validation_pgd.json with shape:
  {
    "checkpoint": <path>,
    "channel_names": [...],
    "by_eps": {"eps_0.000": {...}, "eps_0.010": {...}, ...},
    "config": {"k": 10, "alpha_ratio": 0.25, "t_value": 0.0}
  }

Each by_eps entry mirrors evaluate_at_t output (mse, r2_mean, r2_median,
per_channel_r2). t is fixed at 0 for this gate (the pure adversarial
question, separate from the noise-schedule question handled by
validate_vs_t.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import make_combined_dataset
from .pgd import pgd_perturb
from .student import BlendshapeStudent

EPS_BUCKETS = [0.0, 0.01, 0.03, 0.05, 0.1]


def per_channel_r2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mu = target.mean(dim=0, keepdim=True)
    ss_res = ((target - pred) ** 2).sum(dim=0)
    ss_tot = ((target - mu) ** 2).sum(dim=0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def evaluate_under_pgd(model: BlendshapeStudent, loader: DataLoader,
                       device: torch.device, eps: float, k: int,
                       alpha_ratio: float) -> dict:
    """Run model on attacked inputs at fixed eps. eps=0.0 is clean eval."""
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y_dev = y.to(device, non_blocking=True)
        if eps == 0.0:
            with torch.no_grad():
                pred = model(x)
        else:
            delta = pgd_perturb(model, x, y_dev,
                                eps=eps,
                                alpha=eps * alpha_ratio,
                                k=k)
            with torch.no_grad():
                pred = model(x + delta)
        preds.append(pred.detach().cpu())
        targets.append(y)
    p = torch.cat(preds, dim=0)
    t = torch.cat(targets, dim=0)
    r2 = per_channel_r2(p, t)
    return {
        "eps": eps,
        "mse": float(F.mse_loss(p, t).item()),
        "r2_mean": float(r2.mean()),
        "r2_median": float(r2.median()),
        "r2_min": float(r2.min()),
        "n_channels_negative_r2": int((r2 < 0).sum()),
        "per_channel_r2": r2.tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="bs_a")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--compact", type=Path, default=None)
    p.add_argument("--blendshapes", type=Path, default=None)
    p.add_argument("--rendered", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True,
                   help="Path to validation_pgd.json output.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--pgd-k", type=int, default=10,
                   help="PGD inner-loop steps for the attack. K=10 is "
                        "stronger than training (K=5) — we want a hard test.")
    p.add_argument("--pgd-alpha-ratio", type=float, default=0.25,
                   help="alpha = eps * ratio. Default eps/4 (0.25).")
    args = p.parse_args()
    device = torch.device(args.device)

    ds_val = make_combined_dataset("val",
                                   compact_path=args.compact,
                                   blendshapes_path=args.blendshapes,
                                   rendered_path=args.rendered)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = BlendshapeStudent(args.variant).to(device)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    print(f"loaded {args.checkpoint} (epoch {ck.get('epoch', '?')})")

    by_eps = {}
    for eps in EPS_BUCKETS:
        rec = evaluate_under_pgd(model, val_loader, device, eps,
                                 k=args.pgd_k, alpha_ratio=args.pgd_alpha_ratio)
        print(f"  eps={eps:.3f}  r2_mean={rec['r2_mean']:.4f}  "
              f"r2_median={rec['r2_median']:.4f}  n_neg={rec['n_channels_negative_r2']}/52")
        by_eps[f"eps_{eps:.3f}"] = rec

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump({
            "checkpoint": str(args.checkpoint),
            "channel_names": getattr(ds_val, "channel_names", None),
            "by_eps": by_eps,
            "config": {"k": args.pgd_k, "alpha_ratio": args.pgd_alpha_ratio,
                       "t_value": 0.0},
        }, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
