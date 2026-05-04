"""Train an AdapterStudent (frozen IResNet50 + trainable stem) noise-conditionally.

Path 1 from `docs/research/2026-04-30-noise-conditional-distill-design.md`:
same architecture as the shipped `latent_a2_full_native_shallow` (val cos
0.881 against buffalo_l ArcFace teacher), but during training the input is
a partially-noised latent

    z_t = (1 - t) · z_0 + t · ε,    t ~ Uniform(0, 1),   ε ~ 𝒩(0, I)

The student doesn't see `t` — it just learns to produce the same teacher
embedding regardless of noise level. If this saturates (val cos at t=0.5
falls > 0.05 below t=0), the next step is Path 2 (FiLM/AdaLN, requires an
architectural change to take `t` as input).

Per-epoch eval at t ∈ {0, 0.5} cheap; full bucket {0, 0.1, 0.25, 0.5, 0.75, 1.0}
every 5 epochs and at end. Val noise is seeded so the curves are
comparable across epochs.
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
from .dataset import CompactLatentDataset


T_BUCKETS_FULL = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]


def cosine_distance_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = F.normalize(target, dim=-1)
    return (1.0 - (pred * target).sum(dim=-1)).mean()


def add_rectified_flow_noise(z_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    eps = torch.randn_like(z_0)
    t_b = t[:, None, None, None]
    return (1.0 - t_b) * z_0 + t_b * eps


@torch.no_grad()
def evaluate_at_t(model: AdapterStudent, loader: DataLoader, device: torch.device,
                  t_value: float, val_seed: int) -> float:
    model.eval()
    cs = []
    g = torch.Generator(device="cpu").manual_seed(val_seed + int(t_value * 100000))
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if t_value == 0.0:
            z_t = x
        else:
            eps = torch.randn(x.shape, generator=g).to(device, non_blocking=True)
            z_t = (1.0 - t_value) * x + t_value * eps
        z = model(z_t)
        y = F.normalize(y, dim=-1)
        cs.append((z * y).sum(dim=-1))
    return torch.cat(cs).mean().item() if cs else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="latent_a2_full_native_shallow",
                   choices=["latent_a_full_native", "latent_a2_full_native_shallow"])
    p.add_argument("--compact", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backbone-lr", type=float, default=1e-4)
    p.add_argument("--init-from", type=Path, default=None,
                   help="Optional warm-start from a clean-only checkpoint (e.g. shipped "
                        "latent_a2_full_native_shallow's checkpoint.pt). Speeds up convergence "
                        "since the stem already knows clean Flux latents.")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--val-seed", type=int, default=20260430)
    p.add_argument("--full-bucket-eval-every", type=int, default=5)
    p.add_argument("--onnx-path", type=Path, default=None)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "train_log.jsonl"
    last_path = args.out_dir / "last.pt"
    best_path = args.out_dir / "checkpoint.pt"

    device = torch.device(args.device)
    print(f"variant={args.variant} device={device} compact={args.compact}")

    train_ds = CompactLatentDataset(args.compact, "train")
    val_ds = CompactLatentDataset(args.compact, "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    print(f"train rows={len(train_ds)} val rows={len(val_ds)}")

    model = AdapterStudent(args.variant) if args.onnx_path is None \
            else AdapterStudent(args.variant, args.onnx_path)
    model = model.to(device)
    n_train = sum(p.numel() for p in model.trainable_parameters())
    print(f"trainable params: {n_train:,}")

    groups = model.parameter_groups(stem_lr=args.lr, backbone_lr=args.backbone_lr)
    for i, g in enumerate(groups):
        n = sum(pp.numel() for pp in g["params"])
        print(f"  group {i}: {n:,} params at lr={g['lr']:g}")
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    start_epoch = 0
    best_val_t0 = -float("inf")
    if last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch = ck["epoch"] + 1
        best_val_t0 = ck.get("best_val_t0", -float("inf"))
        print(f"resumed from epoch {start_epoch} best_val_t0={best_val_t0:.4f}")
    elif args.init_from is not None:
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        print(f"init from {args.init_from} (epoch {ck.get('epoch', '?')})  "
              f"missing={len(missing)} unexpected={len(unexpected)}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        losses = []
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            t = torch.rand(x.size(0), device=device)
            z_t = add_rectified_flow_noise(x, t)
            z = model(z_t)
            loss = cosine_distance_loss(z, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()

        is_full = ((epoch + 1) % args.full_bucket_eval_every == 0
                   or epoch == args.epochs - 1)
        eval_buckets = T_BUCKETS_FULL if is_full else [0.0, 0.5]
        val_at = {}
        for tv in eval_buckets:
            val_at[tv] = evaluate_at_t(model, val_loader, device, tv, args.val_seed)

        train_cos_loss = float(sum(losses) / max(1, len(losses)))
        elapsed = time.time() - t0
        rec = {
            "epoch": epoch,
            "train_cos_loss": train_cos_loss,
            "val_cos_at": {f"t_{tv:.2f}": val_at[tv] for tv in eval_buckets},
            "val_full_bucket_eval": is_full,
            "lr": opt.param_groups[0]["lr"],
            "elapsed_s": round(elapsed, 1),
        }
        print(json.dumps(rec))
        with log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

        ck = {
            "epoch": epoch, "model": model.state_dict(),
            "opt": opt.state_dict(), "sched": sched.state_dict(),
            "variant": args.variant, "best_val_t0": best_val_t0,
        }
        torch.save(ck, last_path)
        if val_at[0.0] > best_val_t0:
            best_val_t0 = val_at[0.0]
            ck["best_val_t0"] = best_val_t0
            torch.save(ck, best_path)

    print("\nfinal eval across all t buckets …")
    final = {f"t_{tv:.2f}": evaluate_at_t(model, val_loader, device, tv, args.val_seed)
             for tv in T_BUCKETS_FULL}
    with (args.out_dir / "eval.json").open("w") as f:
        json.dump({
            "variant": args.variant,
            "val_n": len(val_ds),
            "by_t_cos_mean": final,
        }, f, indent=2)
    print(json.dumps(final, indent=2))
    print(f"\nbest val cos at t=0: {best_val_t0:.4f}")


if __name__ == "__main__":
    main()
