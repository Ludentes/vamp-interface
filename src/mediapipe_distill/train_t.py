"""Train a noise-conditional MediaPipe-blendshape student (bs_a + random-t).

Path 1 from `docs/research/2026-04-30-noise-conditional-distill-design.md`:
same architecture as the shipped v2c (`bs_a`), but during training the input
is a partially-noised Flux VAE latent

    z_t = (1 − t) · z_0 + t · ε,    t ~ Uniform(0, 1),   ε ~ 𝒩(0, I)

The student doesn't see `t` — it learns the marginal-over-t mapping from
noisy latent to teacher 52-d blendshape vector. If this saturates (mid-t
R² far below clean R²), the next step is Path 2 (FiLM/AdaLN). For
blendshapes the inductive bias is favourable: the targets are spatially
local features (mouth corners, eye openness, brow position) that are
easier to recover from noise than the holistic identity representation
ArcFace requires.

Per-epoch eval at t ∈ {0, 0.5}; full bucket {0, 0.1, 0.25, 0.5, 0.75, 1.0}
every 5 epochs and at end. Val noise seeded for reproducibility.

Loss is MSE on sigmoid output (same as v2c). 52-d targets are mostly
sparse (mean per-channel ≈ 0.09).
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


T_BUCKETS_FULL = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]


def add_rectified_flow_noise(z_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    eps = torch.randn_like(z_0)
    t_b = t[:, None, None, None]
    return (1.0 - t_b) * z_0 + t_b * eps


def per_channel_r2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mu = target.mean(dim=0, keepdim=True)
    ss_res = ((target - pred) ** 2).sum(dim=0)
    ss_tot = ((target - mu) ** 2).sum(dim=0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


@torch.no_grad()
def evaluate_at_t(model: BlendshapeStudent, loader: DataLoader, device: torch.device,
                  t_value: float, val_seed: int) -> dict:
    model.eval()
    preds, targets = [], []
    g = torch.Generator(device="cpu").manual_seed(val_seed + int(t_value * 100000))
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        if t_value == 0.0:
            z_t = x
        else:
            eps = torch.randn(x.shape, generator=g).to(device, non_blocking=True)
            z_t = (1.0 - t_value) * x + t_value * eps
        pred = model(z_t).cpu()
        preds.append(pred)
        targets.append(y)
    p = torch.cat(preds, dim=0)
    t_emb = torch.cat(targets, dim=0)
    r2 = per_channel_r2(p, t_emb)
    return {
        "t": t_value,
        "mse": float(F.mse_loss(p, t_emb).item()),
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
    p.add_argument("--compact", type=Path, default=None)
    p.add_argument("--blendshapes", type=Path, default=None)
    p.add_argument("--rendered", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--init-from", type=Path, default=None,
                   help="Optional warm-start from clean-only v2c checkpoint.pt; "
                        "speeds up convergence since the trunk already knows "
                        "clean Flux latent → blendshape map.")
    p.add_argument("--val-seed", type=int, default=20260430)
    p.add_argument("--full-bucket-eval-every", type=int, default=5)
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
    print(f"params: {n_params / 1e6:.2f} M total")

    if args.init_from is not None:
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"warm-started from {args.init_from} (epoch {ck.get('epoch', '?')})")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    log_path = args.out_dir / "train_log.jsonl"
    log_f = log_path.open("w", buffering=1)

    best_r2_t0 = -1e9
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running, n_batches = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            t = torch.rand(x.size(0), device=device)
            z_t = add_rectified_flow_noise(x, t)
            pred = model(z_t)
            loss = F.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.item())
            n_batches += 1
        sched.step()
        train_mse = running / max(1, n_batches)

        is_full = ((epoch + 1) % args.full_bucket_eval_every == 0
                   or epoch == args.epochs - 1)
        eval_buckets = T_BUCKETS_FULL if is_full else [0.0, 0.5]
        ev_at = {tv: evaluate_at_t(model, val_loader, device, tv, args.val_seed)
                 for tv in eval_buckets}

        elapsed = time.time() - t0
        rec = {
            "epoch": epoch,
            "train_mse": train_mse,
            "val_r2_mean_at": {f"t_{tv:.2f}": ev_at[tv]["r2_mean"] for tv in eval_buckets},
            "val_r2_median_at": {f"t_{tv:.2f}": ev_at[tv]["r2_median"] for tv in eval_buckets},
            "val_n_neg_r2_at": {f"t_{tv:.2f}": ev_at[tv]["n_channels_negative_r2"]
                                for tv in eval_buckets},
            "val_full_bucket_eval": is_full,
            "lr": float(opt.param_groups[0]["lr"]),
            "elapsed_s": round(elapsed, 1),
        }
        log_f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec))

        torch.save({"model": model.state_dict(), "epoch": epoch, "variant": args.variant},
                   args.out_dir / "last.pt")
        if ev_at[0.0]["r2_mean"] > best_r2_t0:
            best_r2_t0 = ev_at[0.0]["r2_mean"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "variant": args.variant,
                        "val_r2_mean_at_t0": best_r2_t0},
                       args.out_dir / "checkpoint.pt")

    # Final full-bucket per-channel detail
    print("\nfinal per-bucket eval …")
    final = {f"t_{tv:.2f}": evaluate_at_t(model, val_loader, device, tv, args.val_seed)
             for tv in T_BUCKETS_FULL}
    channel_names = getattr(ds_val, "channel_names", None)
    with (args.out_dir / "eval.json").open("w") as f:
        json.dump({
            "variant": args.variant,
            "val_n": len(ds_val),
            "channel_names": channel_names,
            "by_t": final,
        }, f, indent=2)
    log_f.close()
    print(f"\nbest val r2_mean at t=0: {best_r2_t0:.4f}")
    for tv in T_BUCKETS_FULL:
        print(f"  final t={tv:.2f}: r2_mean={final[f't_{tv:.2f}']['r2_mean']:.4f}  "
              f"r2_median={final[f't_{tv:.2f}']['r2_median']:.4f}  "
              f"n_neg={final[f't_{tv:.2f}']['n_channels_negative_r2']}/52")


if __name__ == "__main__":
    main()
