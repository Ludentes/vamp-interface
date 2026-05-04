"""Validate sg_c against the t-bucket gates from the noise-conditional design doc.

For a noise-conditional SigLIP student we want three things, beyond the
standard L1/L2/L3/L4 gates `validate_as_loss.py` already covers:

  T1 — Clean-input parity. val cos at t=0 within 0.01 of the clean-only
       sg_b baseline (0.9204).
  T2 — Schedule coverage. val cos at every t bucket
       {0.1, 0.25, 0.5, 0.75, 1.0} within 0.05 of the t=0 value.
  T3 — Per-probe R² at t=0.5 within 0.10 of the t=0 value (probes are the
       downstream-relevant proxy; we want guidance gradients useful at
       mid-trajectory).

Outputs a single JSON with `by_t` blocks (L1 cosine + L2 per-probe R²
both per-bucket) plus the gate verdicts.

Run directly on the trained sg_c checkpoint after training; the per-epoch
log already covers the schedule shape, this is the post-hoc full report.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .student_t import SigLIPStudentT


SG_COLUMNS = [
    "sg_bearded_margin",
    "sg_heavy_beard_margin",
    "sg_mustache_only_margin",
    "sg_smiling_margin",
    "sg_open_mouth_margin",
    "sg_eyes_closed_margin",
    "sg_glasses_margin",
    "sg_wrinkled_margin",
    "sg_long_hair_margin",
    "sg_angry_margin",
    "sg_surprised_margin",
    "sg_puckered_lips_margin",
]

T_BUCKETS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
SG_B_CLEAN_BASELINE = 0.9204  # val_cos_mean at t=0 from sg_b (clean-only)


def is_held_out(sha: str) -> bool:
    return sha[:1].lower() == "f"


def load_data(compact_path: Path, siglip_path: Path, probes_path: Path):
    compact = torch.load(compact_path, map_location="cpu", weights_only=False)
    sg = torch.load(siglip_path, map_location="cpu", weights_only=False)
    if list(compact["shas"]) != list(sg["shas"]):
        raise ValueError("compact + siglip SHA order mismatch")
    shas = list(compact["shas"])

    probes_df = pd.read_parquet(probes_path).set_index("image_sha256")
    missing_cols = [c for c in SG_COLUMNS if c not in probes_df.columns]
    if missing_cols:
        raise ValueError(f"probes parquet missing columns: {missing_cols}")
    probes_df = probes_df[SG_COLUMNS]

    margins_full = np.full((len(shas), len(SG_COLUMNS)), np.nan, dtype=np.float32)
    for i, sha in enumerate(shas):
        if sha in probes_df.index:
            margins_full[i] = probes_df.loc[sha].values.astype(np.float32)
    has_probes = ~np.isnan(margins_full).any(axis=1)
    detected = sg["detected"].numpy().astype(bool)
    keep = has_probes & detected

    train_idx = np.array([i for i, s in enumerate(shas)
                          if not is_held_out(s) and keep[i]], dtype=np.int64)
    val_idx = np.array([i for i, s in enumerate(shas)
                        if is_held_out(s) and keep[i]], dtype=np.int64)
    return {
        "shas": shas,
        "latents": compact["latents"],
        "teacher_embs": sg["embeddings"],
        "margins": torch.from_numpy(margins_full),
        "train_idx": torch.from_numpy(train_idx),
        "val_idx": torch.from_numpy(val_idx),
    }


def build_z_t(z_0: torch.Tensor, t_value: float, seed: int) -> torch.Tensor:
    """Deterministic noisy latent batch at fixed t_value, fixed seed."""
    if t_value == 0.0:
        return z_0
    g = torch.Generator(device="cpu").manual_seed(seed + int(t_value * 100000))
    eps = torch.randn(z_0.shape, generator=g, dtype=z_0.dtype)
    return (1.0 - t_value) * z_0 + t_value * eps


@torch.no_grad()
def run_inference_at_t(model: SigLIPStudentT, latents: torch.Tensor, t_value: float,
                       device: torch.device, seed: int, batch_size: int = 128) -> torch.Tensor:
    model.eval()
    out = []
    for i in range(0, latents.size(0), batch_size):
        x = latents[i:i + batch_size].to(torch.float32)
        z_t = build_z_t(x, t_value, seed + i).to(device, non_blocking=True)
        t_b = torch.full((x.size(0),), t_value, device=device)
        out.append(model(z_t, t_b).cpu())
    return torch.cat(out, dim=0)


def cosine_summary(student: torch.Tensor, teacher: torch.Tensor) -> dict:
    cos = F.cosine_similarity(student, teacher, dim=-1)
    return {
        "n_val": int(student.size(0)),
        "cos_mean": float(cos.mean()),
        "cos_median": float(cos.median()),
        "cos_p05": float(cos.kthvalue(max(1, int(0.05 * cos.numel()))).values),
        "cos_p01": float(cos.kthvalue(max(1, int(0.01 * cos.numel()))).values),
        "pred_norm_mean": float(student.norm(dim=-1).mean()),
    }


def recover_probe_dir(teacher_train: torch.Tensor, margins_train: torch.Tensor,
                      ridge_lambda: float = 1e-3) -> torch.Tensor:
    X = teacher_train.to(torch.float32)
    y = margins_train.to(torch.float32)
    D = X.size(1)
    XtX = X.T @ X + ridge_lambda * torch.eye(D, dtype=X.dtype)
    Xty = X.T @ y
    return torch.linalg.solve(XtX, Xty)


def per_dim_r2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mu = target.mean(dim=0, keepdim=True)
    ss_res = ((target - pred) ** 2).sum(dim=0)
    ss_tot = ((target - mu) ** 2).sum(dim=0)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def per_probe_r2(student_val: torch.Tensor, probe_dirs: torch.Tensor,
                 margins_val: torch.Tensor) -> dict:
    sn = student_val / (student_val.norm(dim=-1, keepdim=True) + 1e-12)
    pred_student = sn @ probe_dirs
    r2 = per_dim_r2(pred_student, margins_val)
    by_probe = []
    for i, name in enumerate(SG_COLUMNS):
        by_probe.append({
            "probe": name,
            "r2_student": float(r2[i]),
        })
    return {
        "per_probe": by_probe,
        "r2_student_mean": float(r2.mean()),
        "r2_student_median": float(r2.median()),
        "r2_student_min": float(r2.min()),
        "n_probes_above_0p5": int((r2 >= 0.5).sum()),
        "n_probes_above_0p7": int((r2 >= 0.7).sum()),
    }


def evaluate_gates(by_t: dict) -> dict:
    cos_at = {tv: by_t[f"t_{tv:.2f}"]["L1_cosine"]["cos_mean"] for tv in T_BUCKETS}
    r2_at = {tv: by_t[f"t_{tv:.2f}"]["L2_per_probe_r2"]["r2_student_mean"] for tv in T_BUCKETS}

    t0_cos = cos_at[0.0]
    t0_r2 = r2_at[0.0]
    t05_r2 = r2_at[0.5]

    parity = abs(t0_cos - SG_B_CLEAN_BASELINE) <= 0.01
    coverage = {
        f"t_{tv:.2f}": (t0_cos - cos_at[tv]) <= 0.05
        for tv in T_BUCKETS if tv != 0.0
    }
    coverage_all = all(coverage.values())
    probe_mid = (t0_r2 - t05_r2) <= 0.10

    return {
        "T1_clean_parity_within_0p01_of_sg_b": parity,
        "T1_t0_cos_actual": t0_cos,
        "T1_t0_cos_baseline": SG_B_CLEAN_BASELINE,
        "T2_schedule_coverage_within_0p05": coverage_all,
        "T2_per_bucket": coverage,
        "T2_cos_at_each_bucket": cos_at,
        "T3_probe_r2_t0p5_within_0p10_of_t0": probe_mid,
        "T3_t0_probe_r2_mean": t0_r2,
        "T3_t05_probe_r2_mean": t05_r2,
        "T3_per_bucket": r2_at,
        "all_pass": parity and coverage_all and probe_mid,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="sg_c")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--compact", type=Path, required=True)
    p.add_argument("--siglip", type=Path, required=True)
    p.add_argument("--probes", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20260430)
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"variant={args.variant} device={device}")

    print("loading data …")
    data = load_data(args.compact, args.siglip, args.probes)
    n_train = int(data["train_idx"].numel())
    n_val = int(data["val_idx"].numel())
    print(f"  train rows usable: {n_train}, val rows: {n_val}")

    print(f"loading checkpoint: {args.checkpoint}")
    model = SigLIPStudentT(args.variant).to(device)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])

    val_latents = data["latents"][data["val_idx"]]
    teacher_train = data["teacher_embs"][data["train_idx"]].to(torch.float32)
    teacher_val = data["teacher_embs"][data["val_idx"]].to(torch.float32)
    margins_train = data["margins"][data["train_idx"]]
    margins_val = data["margins"][data["val_idx"]]

    # Probe dirs recovered ONCE from teacher_train (independent of t).
    print("recovering probe directions from teacher train …")
    probe_dirs = recover_probe_dir(teacher_train, margins_train)

    by_t = {}
    for tv in T_BUCKETS:
        t0 = time.time()
        print(f"\n=== t = {tv:.2f} ===")
        student_val = run_inference_at_t(model, val_latents, tv, device, args.seed)
        elapsed = time.time() - t0

        L1 = cosine_summary(student_val, teacher_val)
        L2 = per_probe_r2(student_val, probe_dirs, margins_val)

        print(f"  L1: cos_mean={L1['cos_mean']:.4f}  median={L1['cos_median']:.4f}  "
              f"p05={L1['cos_p05']:.4f}  pred_norm={L1['pred_norm_mean']:.4f}")
        print(f"  L2: r2_mean={L2['r2_student_mean']:.3f}  median={L2['r2_student_median']:.3f}  "
              f"≥0.5: {L2['n_probes_above_0p5']}/12  ≥0.7: {L2['n_probes_above_0p7']}/12")
        print(f"  ({elapsed:.1f}s)")
        by_t[f"t_{tv:.2f}"] = {
            "L1_cosine": L1,
            "L2_per_probe_r2": L2,
            "elapsed_s": round(elapsed, 1),
        }

    print("\nevaluating gates …")
    gates = evaluate_gates(by_t)
    print(json.dumps({k: v for k, v in gates.items() if not k.startswith("T2_per") and not k.startswith("T3_per")},
                     indent=2))

    report = {
        "variant": args.variant,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(ck.get("epoch", -1)),
        "n_val": n_val,
        "n_train_for_probe_recovery": n_train,
        "probe_columns": SG_COLUMNS,
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
