"""Validate AdapterStudentT (Path 2) at t-buckets {0, 0.1, 0.25, 0.5, 0.75, 1.0}.

Same gates as `validate_vs_t.py` but with the (z_t, t) forward signature:

  T1 — Clean-input parity. val cos at t=0 within 0.01 of the clean-only
       baseline (latent_a2_full_native_shallow shipped at 0.881).
  T2 — Schedule coverage. val cos at every t bucket {0.1, 0.25, 0.5, 0.75, 1.0}
       within 0.05 of t=0.

Path 1 (random-t retrain, no FiLM) shipped t=0 0.869 / t=0.5 0.525 — fails
T2 by 0.34. Path 2 has to recover the schedule curve.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .adapter_t import AdapterStudentT
from .dataset import CompactLatentDataset


T_BUCKETS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
ARC_BASELINE_T0 = 0.881  # latent_a2_full_native_shallow shipped val cos


@torch.no_grad()
def evaluate_at_t(model: AdapterStudentT, loader: DataLoader, device: torch.device,
                  t_value: float, seed: int) -> dict:
    model.eval()
    cs = []
    g = torch.Generator(device="cpu").manual_seed(seed + int(t_value * 100000))
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if t_value == 0.0:
            z_t = x
        else:
            eps = torch.randn(x.shape, generator=g).to(device, non_blocking=True)
            z_t = (1.0 - t_value) * x + t_value * eps
        t_b = torch.full((x.size(0),), t_value, device=device)
        z = model(z_t, t_b)
        y_n = F.normalize(y, dim=-1)
        cs.append((z * y_n).sum(dim=-1).cpu())
    cos = torch.cat(cs)
    return {
        "t": t_value,
        "n_val": int(cos.numel()),
        "cos_mean": float(cos.mean()),
        "cos_median": float(cos.median()),
        "cos_p05": float(cos.kthvalue(max(1, int(0.05 * cos.numel()))).values),
        "cos_p01": float(cos.kthvalue(max(1, int(0.01 * cos.numel()))).values),
    }


def evaluate_gates(by_t: dict) -> dict:
    cos_at = {tv: by_t[f"t_{tv:.2f}"]["cos_mean"] for tv in T_BUCKETS}
    t0 = cos_at[0.0]
    parity = abs(t0 - ARC_BASELINE_T0) <= 0.01
    coverage = {f"t_{tv:.2f}": (t0 - cos_at[tv]) <= 0.05
                for tv in T_BUCKETS if tv != 0.0}
    return {
        "T1_clean_parity_within_0p01_of_baseline": parity,
        "T1_t0_actual": t0,
        "T1_t0_baseline": ARC_BASELINE_T0,
        "T2_schedule_coverage_within_0p05": all(coverage.values()),
        "T2_per_bucket": coverage,
        "T2_cos_at_each_bucket": cos_at,
        "all_pass": parity and all(coverage.values()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--compact", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=20260430)
    p.add_argument("--onnx-path", type=Path, default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"variant={AdapterStudentT.VARIANT_NAME} device={device}")

    val_ds = CompactLatentDataset(args.compact, "val")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    print(f"val rows={len(val_ds)}")

    model = AdapterStudentT() if args.onnx_path is None \
            else AdapterStudentT(onnx_path=args.onnx_path)
    model = model.to(device)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])

    by_t = {}
    for tv in T_BUCKETS:
        t0 = time.time()
        ev = evaluate_at_t(model, val_loader, device, tv, args.seed)
        elapsed = time.time() - t0
        ev["elapsed_s"] = round(elapsed, 1)
        print(f"  t={tv:.2f}: cos_mean={ev['cos_mean']:.4f}  "
              f"median={ev['cos_median']:.4f}  p05={ev['cos_p05']:.4f}  "
              f"({elapsed:.1f}s)")
        by_t[f"t_{tv:.2f}"] = ev

    gates = evaluate_gates(by_t)
    print("\ngates:", json.dumps(gates, indent=2))

    report = {
        "variant": AdapterStudentT.VARIANT_NAME,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(ck.get("epoch", -1)),
        "n_val": len(val_ds),
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
