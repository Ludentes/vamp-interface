"""Train sg_c — noise-conditional SigLIP-distill student.

Random-t rectified-flow noise per batch, on-the-fly:

    t   ~ Uniform(0, 1)
    ε   ~ 𝒩(0, I)      (same shape as z_0)
    z_t = (1 − t) · z_0 + t · ε

Loss: pure (1 − cos(pred, target)) — student is L2-normed at the head, so
MSE between unit vectors is monotonic in cosine.

Per-epoch JSONL log includes val_cos at t buckets {0, 0.25, 0.5, 0.75, 1.0}
so we can see the schedule-shape of student quality, not just the
averaged-over-t number.
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
from .student_t import SigLIPStudentT


T_BUCKETS_VAL = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]


def add_rectified_flow_noise(z_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """z_t = (1 - t) * z_0 + t * eps. Same shape; t broadcasts across CHW."""
    eps = torch.randn_like(z_0)
    t_b = t[:, None, None, None]
    return (1.0 - t_b) * z_0 + t_b * eps


@torch.no_grad()
def evaluate_at_t_bucket(model: SigLIPStudentT, loader: DataLoader, device: torch.device,
                         t_value: float, val_seed: int) -> dict:
    """Run inference at a fixed t across the whole val set with deterministic
    noise (per-row seeded from val_seed for reproducibility across epochs)."""
    model.eval()
    preds, targets = [], []
    g = torch.Generator(device="cpu").manual_seed(val_seed + int(t_value * 100000))
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        if t_value == 0.0:
            z_t = x
        else:
            eps = torch.randn(x.shape, generator=g).to(device, non_blocking=True)
            t_b = t_value
            z_t = (1.0 - t_b) * x + t_b * eps
        t_batch = torch.full((x.size(0),), t_value, device=device)
        pred = model(z_t, t_batch).cpu()
        preds.append(pred)
        targets.append(y)
    p = torch.cat(preds, dim=0)
    t_emb = torch.cat(targets, dim=0)
    cos_per = F.cosine_similarity(p, t_emb, dim=-1)
    return {
        "t": t_value,
        "cos_mean": float(cos_per.mean()),
        "cos_median": float(cos_per.median()),
        "cos_p05": float(cos_per.kthvalue(max(1, int(0.05 * cos_per.numel()))).values),
        "pred_norm_mean": float(p.norm(dim=-1).mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="sg_c")
    p.add_argument("--compact", type=Path, required=True)
    p.add_argument("--siglip", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--val-seed", type=int, default=20260430,
                   help="Seed for val noise so val_cos_vs_t is comparable across epochs.")
    p.add_argument("--full-bucket-eval-every", type=int, default=5,
                   help="Run the full {0, 0.1, 0.25, 0.5, 0.75, 1.0} bucket eval every N epochs "
                        "(intermediate epochs eval at t=0 + t=0.5 only to save wall time).")
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

    model = SigLIPStudentT(args.variant).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_film = sum(p.numel() for n, p in model.named_parameters()
                 if "film" in n or "t_embed" in n)
    print(f"params: {n_params / 1e6:.2f} M total ({n_film / 1e6:.2f} M FiLM/t-embed)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    log_path = args.out_dir / "train_log.jsonl"
    log_f = log_path.open("w", buffering=1)

    best_cos = -1e9  # tracked at t=0 so the "ship the best clean-cos" rule still holds
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_cos = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            t = torch.rand(x.size(0), device=device)
            z_t = add_rectified_flow_noise(x, t)
            pred = model(z_t, t)
            cos = F.cosine_similarity(pred, y, dim=-1).mean()
            loss = 1.0 - cos
            opt.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            running_loss += float(loss.item())
            running_cos += float(cos.item())
            n_batches += 1
        sched.step()
        nb = max(1, n_batches)

        # Eval: at every epoch hit t=0 + t=0.5 (cheap, still informative);
        # full-bucket sweep every N epochs.
        is_full_eval = ((epoch + 1) % args.full_bucket_eval_every == 0
                        or epoch == args.epochs - 1)
        eval_buckets = T_BUCKETS_VAL if is_full_eval else [0.0, 0.5]
        ev_at = {}
        for tv in eval_buckets:
            ev_at[tv] = evaluate_at_t_bucket(model, val_loader, device, tv, args.val_seed)

        elapsed = time.time() - t0
        rec = {
            "epoch": epoch,
            "train_loss": running_loss / nb,
            "train_cos": running_cos / nb,
            "val_cos_at": {f"t_{tv:.2f}": ev_at[tv]["cos_mean"] for tv in eval_buckets},
            "val_full_bucket_eval": is_full_eval,
            "lr": float(opt.param_groups[0]["lr"]),
            "elapsed_s": round(elapsed, 1),
        }
        log_f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec))

        torch.save({"model": model.state_dict(), "epoch": epoch, "variant": args.variant},
                   args.out_dir / "last.pt")
        cos_t0 = ev_at[0.0]["cos_mean"]
        if cos_t0 > best_cos:
            best_cos = cos_t0
            torch.save({"model": model.state_dict(), "epoch": epoch, "variant": args.variant,
                        "val_cos_at_t0": cos_t0},
                       args.out_dir / "checkpoint.pt")

    # Final full-bucket eval (always)
    print("final eval across all t buckets …")
    final = {}
    for tv in T_BUCKETS_VAL:
        final[f"t_{tv:.2f}"] = evaluate_at_t_bucket(model, val_loader, device, tv, args.val_seed)
    with (args.out_dir / "eval.json").open("w") as f:
        json.dump({
            "variant": args.variant,
            "val_n": len(ds_val),
            "emb_dim": ds_val.emb_dim,
            "by_t": final,
        }, f, indent=2)
    log_f.close()
    print(f"\nbest val cos_mean at t=0: {best_cos:.4f}")
    for tv in T_BUCKETS_VAL:
        print(f"  final t={tv:.2f}: cos_mean={final[f't_{tv:.2f}']['cos_mean']:.4f}")


if __name__ == "__main__":
    main()
