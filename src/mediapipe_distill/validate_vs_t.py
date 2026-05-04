"""Validate a noise-conditional MediaPipe-blendshape student at t-buckets.

Gates from `docs/research/2026-04-30-noise-conditional-distill-design.md`,
adapted to per-channel R² metric:

  T1 — Clean-input parity. r2_mean at t=0 within 0.02 of v2c clean baseline
       (median R² ~0.76; note metric noise is higher for R² than for cosine).
  T2 — Schedule coverage. r2_mean at every t bucket within 0.10 of t=0
       (looser threshold than the cosine version because R² has larger
       per-channel variance).

Outputs full per-channel R² at each t bucket so we can see which
blendshapes survive noisy latents and which collapse — relevant for
inference-time guidance which won't work on channels with R²<0 at the
operating t.
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


T_BUCKETS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
V2C_BASELINE_R2_MEAN = 0.40   # placeholder; actual v2c baseline pulled at runtime
V2C_BASELINE_R2_MEDIAN = 0.76  # from project memory


def per_channel_r2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mu = target.mean(dim=0, keepdim=True)
    ss_res = ((target - pred) ** 2).sum(dim=0)
    ss_tot = ((target - mu) ** 2).sum(dim=0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


@torch.no_grad()
def evaluate_at_t(model: BlendshapeStudent, loader: DataLoader, device: torch.device,
                  t_value: float, seed: int) -> dict:
    model.eval()
    preds, targets = [], []
    g = torch.Generator(device="cpu").manual_seed(seed + int(t_value * 100000))
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
        "n_val": int(p.size(0)),
        "mse": float(F.mse_loss(p, t_emb).item()),
        "r2_mean": float(r2.mean()),
        "r2_median": float(r2.median()),
        "r2_p05": float(r2.kthvalue(max(1, int(0.05 * r2.numel()))).values),
        "r2_min": float(r2.min()),
        "n_channels_negative_r2": int((r2 < 0).sum()),
        "per_channel_r2": r2.tolist(),
    }


def evaluate_gates(by_t: dict) -> dict:
    r2_at = {tv: by_t[f"t_{tv:.2f}"]["r2_mean"] for tv in T_BUCKETS}
    r2_med_at = {tv: by_t[f"t_{tv:.2f}"]["r2_median"] for tv in T_BUCKETS}
    t0 = r2_at[0.0]
    parity = abs(r2_med_at[0.0] - V2C_BASELINE_R2_MEDIAN) <= 0.02
    coverage = {f"t_{tv:.2f}": (t0 - r2_at[tv]) <= 0.10
                for tv in T_BUCKETS if tv != 0.0}
    return {
        "T1_clean_parity_within_0p02_of_v2c_median": parity,
        "T1_t0_r2_median_actual": r2_med_at[0.0],
        "T1_t0_r2_median_baseline": V2C_BASELINE_R2_MEDIAN,
        "T2_schedule_coverage_within_0p10": all(coverage.values()),
        "T2_per_bucket": coverage,
        "T2_r2_mean_at_each_bucket": r2_at,
        "T2_r2_median_at_each_bucket": r2_med_at,
        "all_pass": parity and all(coverage.values()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="bs_a")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--compact", type=Path, default=None)
    p.add_argument("--blendshapes", type=Path, default=None)
    p.add_argument("--rendered", type=Path, default=None)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260430)
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"variant={args.variant} device={device}")

    ds_val = make_combined_dataset("val",
                                   compact_path=args.compact,
                                   blendshapes_path=args.blendshapes,
                                   rendered_path=args.rendered)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    n_val = len(ds_val)  # type: ignore[arg-type]
    print(f"val rows={n_val}")

    model = BlendshapeStudent(args.variant).to(device)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])

    by_t = {}
    for tv in T_BUCKETS:
        t0 = time.time()
        ev = evaluate_at_t(model, val_loader, device, tv, args.seed)
        elapsed = time.time() - t0
        ev["elapsed_s"] = round(elapsed, 1)
        print(f"  t={tv:.2f}: r2_mean={ev['r2_mean']:+.3f}  "
              f"median={ev['r2_median']:+.3f}  p05={ev['r2_p05']:+.3f}  "
              f"n_neg={ev['n_channels_negative_r2']}/52  ({elapsed:.1f}s)")
        by_t[f"t_{tv:.2f}"] = ev

    gates = evaluate_gates(by_t)
    print("\ngates:", json.dumps({k: v for k, v in gates.items()
                                  if not k.startswith("T2_per_bucket")
                                  and not k.startswith("T2_r2_")},
                                 indent=2))

    channel_names = getattr(ds_val, "channel_names", None)
    report = {
        "variant": args.variant,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(ck.get("epoch", -1)),
        "n_val": n_val,
        "channel_names": channel_names,
        "t_buckets": T_BUCKETS,
        "by_t": by_t,
        "gates": gates,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out_json}")
    print(f"ALL GATES PASS: {gates['all_pass']}")


if __name__ == "__main__":
    main()
