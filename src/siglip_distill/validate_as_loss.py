"""Validate a SigLIP-distill student against the gates that matter for use
as a slider/LoRA training loss.

The load-bearing question is: for the 12 SigLIP-2 probe directions in
reverse_index (sg_smiling_margin, sg_glasses_margin, etc.), does the
student preserve those directions well enough that `dot(student, text)`
tracks `dot(teacher, text)`?

Layers:

  L1   — direct (student, teacher) cosine: mean / median / p05 over val.
  L2   — per-probe R²: for each of 12 probes, recover the text direction
         via ridge from (TRAIN teacher_emb, TRAIN margin), apply to
         val student_emb, compute R² vs val ground-truth margin. Also
         report the teacher's own R² as the "ceiling" (should be ≈ 1.0
         if recovery is faithful).
  L3a  — augmentation invariance: small input noise → small cosine drop.
  L3b  — hflip invariance: cos(student(z), student(hflip(z))) on val.
         SigLIP image embeddings are approximately mirror-invariant for
         centered face crops, so this should be high.
  L4   — gradient sanity: backprop a Flux latent toward another row's
         teacher embedding under the student's loss; verify finite
         gradients + monotonic descent.

Outputs JSON.

Inputs:

  --checkpoint     path to last.pt or checkpoint.pt
  --compact        latents (compact.pt with .latents .shas)
  --siglip         compact_siglip.pt (teacher embs aligned to compact)
  --probes         small parquet w/ image_sha256 + 12 sg_* columns
                   (ship from local reverse_index)
  --out-json       write here

Train/val split is the same SHA-prefix rule as training: sha[0]=='f' → val.
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

from .student import SigLIPStudent


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


def is_held_out(sha: str) -> bool:
    return sha[:1].lower() == "f"


def load_data(compact_path: Path, siglip_path: Path, probes_path: Path):
    compact = torch.load(compact_path, map_location="cpu", weights_only=False)
    sg = torch.load(siglip_path, map_location="cpu", weights_only=False)
    if list(compact["shas"]) != list(sg["shas"]):
        raise ValueError("compact + siglip SHA order mismatch")
    shas = list(compact["shas"])

    probes_df = pd.read_parquet(probes_path)
    probes_df = probes_df.set_index("image_sha256")
    missing_cols = [c for c in SG_COLUMNS if c not in probes_df.columns]
    if missing_cols:
        raise ValueError(f"probes parquet missing columns: {missing_cols}")
    probes_df = probes_df[SG_COLUMNS]

    margins_full = np.full((len(shas), len(SG_COLUMNS)), np.nan, dtype=np.float32)
    for i, sha in enumerate(shas):
        if sha in probes_df.index:
            row = probes_df.loc[sha].values
            margins_full[i] = row.astype(np.float32)
    has_probes = ~np.isnan(margins_full).any(axis=1)
    detected = sg["detected"].numpy().astype(bool)
    keep = has_probes & detected

    train_idx = np.array([i for i, s in enumerate(shas)
                          if not is_held_out(s) and keep[i]], dtype=np.int64)
    val_idx = np.array([i for i, s in enumerate(shas)
                        if is_held_out(s) and keep[i]], dtype=np.int64)
    return {
        "shas": shas,
        "latents": compact["latents"],         # (N, 16, 64, 64) bf16
        "teacher_embs": sg["embeddings"],      # (N, 1152) fp16
        "margins": torch.from_numpy(margins_full),  # (N, 12) fp32
        "train_idx": torch.from_numpy(train_idx),
        "val_idx": torch.from_numpy(val_idx),
    }


def run_inference(model: SigLIPStudent, latents: torch.Tensor,
                  device: torch.device, batch_size: int = 128) -> torch.Tensor:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, latents.size(0), batch_size):
            x = latents[i:i + batch_size].to(torch.float32).to(device, non_blocking=True)
            out.append(model(x).cpu())
    return torch.cat(out, dim=0)


def cosine_summary(student: torch.Tensor, teacher: torch.Tensor) -> dict:
    s = student / (student.norm(dim=-1, keepdim=True) + 1e-12)
    cos = F.cosine_similarity(student, teacher, dim=-1)
    cos_unit = F.cosine_similarity(s, teacher, dim=-1)
    return {
        "n_val": int(student.size(0)),
        "cos_mean": float(cos.mean()),
        "cos_median": float(cos.median()),
        "cos_p05": float(cos.kthvalue(max(1, int(0.05 * cos.numel()))).values),
        "cos_p01": float(cos.kthvalue(max(1, int(0.01 * cos.numel()))).values),
        "cos_unit_mean": float(cos_unit.mean()),
        "cos_unit_median": float(cos_unit.median()),
        "pred_norm_mean": float(student.norm(dim=-1).mean()),
        "pred_norm_std": float(student.norm(dim=-1).std()),
    }


def recover_probe_dir(teacher_train: torch.Tensor, margins_train: torch.Tensor,
                      ridge_lambda: float = 1e-3) -> torch.Tensor:
    """Closed-form ridge: w = (X^T X + λI)^-1 X^T y. Returns (D, P) of probe
    directions in 1152-d space, recovered from teacher emb + margin pairs."""
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


def per_probe_r2(student_val: torch.Tensor, teacher_val: torch.Tensor,
                 teacher_train: torch.Tensor, margins_train: torch.Tensor,
                 margins_val: torch.Tensor,
                 student_train: torch.Tensor | None = None) -> dict:
    """Recover probe direction from teacher_train, evaluate predicted margin
    on student_val and teacher_val, compute per-probe R² vs val ground truth.

    Student outputs are L2-normalized before dotting with probe_dirs because
    that's how they'll be used at slider-loss time (margin = cos similarity
    against text embeddings on the unit hypersphere). Without normalization,
    pred_norm < 1 produces a constant-bias term in R² that scales with
    `mean²/var` of each probe and can flip negative for low-variance probes
    even when the directional representation is fine.

    If `student_train` is also provided, additionally compute per-probe R² on
    the training slice — useful to triangulate "weak probe = small variance
    in the corpus" vs "weak probe = student didn't learn this direction"."""
    probe_dirs = recover_probe_dir(teacher_train, margins_train)  # (D, P)

    student_val_n = student_val.to(torch.float32)
    student_val_n = student_val_n / (student_val_n.norm(dim=-1, keepdim=True) + 1e-12)

    pred_student = student_val_n @ probe_dirs                     # (Nv, P)
    pred_teacher = teacher_val.to(torch.float32) @ probe_dirs     # (Nv, P)

    r2_student = per_dim_r2(pred_student, margins_val)
    r2_teacher = per_dim_r2(pred_teacher, margins_val)

    r2_student_train = None
    if student_train is not None:
        student_train_n = student_train.to(torch.float32)
        student_train_n = student_train_n / (student_train_n.norm(dim=-1, keepdim=True) + 1e-12)
        pred_student_train = student_train_n @ probe_dirs
        r2_student_train = per_dim_r2(pred_student_train, margins_train)

    by_probe = []
    for i, name in enumerate(SG_COLUMNS):
        entry = {
            "probe": name,
            "r2_student": float(r2_student[i]),
            "r2_teacher_ceiling": float(r2_teacher[i]),
            "val_std": float(margins_val[:, i].std()),
            "train_std": float(margins_train[:, i].std()),
        }
        if r2_student_train is not None:
            entry["r2_student_train"] = float(r2_student_train[i])
            entry["train_minus_val_r2"] = float(r2_student_train[i] - r2_student[i])
        by_probe.append(entry)
    out = {
        "n_train_used": int(teacher_train.size(0)),
        "n_val_used": int(margins_val.size(0)),
        "per_probe": by_probe,
        "r2_student_mean": float(r2_student.mean()),
        "r2_student_median": float(r2_student.median()),
        "r2_student_min": float(r2_student.min()),
        "r2_teacher_ceiling_mean": float(r2_teacher.mean()),
        "n_probes_above_0p5": int((r2_student >= 0.5).sum()),
        "n_probes_above_0p7": int((r2_student >= 0.7).sum()),
    }
    if r2_student_train is not None:
        out["r2_student_train_mean"] = float(r2_student_train.mean())
        out["r2_student_train_median"] = float(r2_student_train.median())
    return out


def layer_3a_invariance(model: SigLIPStudent, latents_val: torch.Tensor,
                        student_val: torch.Tensor, device: torch.device,
                        seed: int = 0) -> dict:
    out = {}
    s_norm = student_val / (student_val.norm(dim=-1, keepdim=True) + 1e-12)
    for kind in ["gauss_002", "gauss_005", "shift_h", "shift_w"]:
        torch.manual_seed(seed)
        if kind == "gauss_002":
            perturbed = latents_val.to(torch.float32) + 0.02 * torch.randn_like(latents_val.to(torch.float32))
        elif kind == "gauss_005":
            perturbed = latents_val.to(torch.float32) + 0.05 * torch.randn_like(latents_val.to(torch.float32))
        elif kind == "shift_h":
            perturbed = torch.roll(latents_val.to(torch.float32), shifts=1, dims=-2)
        else:
            perturbed = torch.roll(latents_val.to(torch.float32), shifts=1, dims=-1)
        pred_p = run_inference(model, perturbed, device)
        p_norm = pred_p / (pred_p.norm(dim=-1, keepdim=True) + 1e-12)
        cos_pos = (s_norm * p_norm).sum(dim=-1)
        out[kind] = {
            "cos_mean": float(cos_pos.mean()),
            "cos_p05": float(cos_pos.kthvalue(max(1, int(0.05 * cos_pos.numel()))).values),
        }
    return out


def layer_3b_hflip(model: SigLIPStudent, latents_val: torch.Tensor,
                   student_val: torch.Tensor, device: torch.device) -> dict:
    flipped = latents_val.to(torch.float32).flip(-1)
    pred_flip = run_inference(model, flipped, device)
    s = student_val / (student_val.norm(dim=-1, keepdim=True) + 1e-12)
    p = pred_flip / (pred_flip.norm(dim=-1, keepdim=True) + 1e-12)
    cos = (s * p).sum(dim=-1)
    return {
        "cos_mean": float(cos.mean()),
        "cos_median": float(cos.median()),
        "cos_p05": float(cos.kthvalue(max(1, int(0.05 * cos.numel()))).values),
    }


def layer_4_gradient(model: SigLIPStudent, latents_val: torch.Tensor,
                     teacher_val: torch.Tensor, device: torch.device,
                     n_steps: int = 100, lr: float = 0.05, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    n = latents_val.size(0)
    src = int(torch.randint(0, n, (1,)).item())
    tgt = int(torch.randint(0, n, (1,)).item())
    while tgt == src:
        tgt = int(torch.randint(0, n, (1,)).item())
    x = latents_val[src:src + 1].to(torch.float32).clone().to(device).requires_grad_(True)
    target = teacher_val[tgt:tgt + 1].to(torch.float32).to(device)
    model.eval()
    opt = torch.optim.SGD([x], lr=lr)
    losses, grad_norms = [], []
    grad_finite = True
    for step in range(n_steps + 1):
        opt.zero_grad()
        pred = model(x)
        # the loss the student will be used as: 0.5 MSE + 0.5 (1 - cos)
        mse = ((pred - target) ** 2).mean()
        cos = F.cosine_similarity(pred, target, dim=-1).mean()
        loss = 0.5 * mse + 0.5 * (1.0 - cos)
        if step < n_steps:
            loss.backward()
            grad = x.grad
            assert grad is not None
            if not torch.isfinite(grad).all():
                grad_finite = False
                break
            grad_norms.append(float(grad.norm()))
            opt.step()
        losses.append(float(loss.item()))
    return {
        "src_idx": src,
        "tgt_idx": tgt,
        "grad_finite": grad_finite,
        "grad_norm_mean": (sum(grad_norms) / len(grad_norms)) if grad_norms else None,
        "loss_step0": losses[0],
        "loss_final": losses[-1],
        "loss_reduction_pct": (1.0 - losses[-1] / losses[0]) * 100 if losses[0] > 0 else 0.0,
        "loss_descended_5pct": losses[-1] < 0.95 * losses[0],
    }


def evaluate_gates(report: dict) -> dict:
    L1 = report["L1_cosine"]
    L2 = report["L2_per_probe_r2"]
    L3b = report["L3b_hflip"]
    L4 = report["L4_gradient"]

    floors = {
        "cos_mean_geq_0p85": L1["cos_mean"] >= 0.85,
        "n_probes_r2_geq_0p5_geq_8": L2["n_probes_above_0p5"] >= 8,
        "hflip_cos_geq_0p9": L3b["cos_mean"] >= 0.9,
        "gradient_finite_and_descends_5pct": L4["grad_finite"] and L4["loss_descended_5pct"],
    }
    targets = {
        "cos_mean_geq_0p9": L1["cos_mean"] >= 0.9,
        "cos_p05_geq_0p8": L1["cos_p05"] >= 0.8,
        "n_probes_r2_geq_0p7_geq_8": L2["n_probes_above_0p7"] >= 8,
    }
    return {
        "all_floors_pass": all(floors.values()),
        "all_targets_pass": all(targets.values()),
        "floors": floors,
        "targets": targets,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="sg_a")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--compact", type=Path, required=True)
    p.add_argument("--siglip", type=Path, required=True)
    p.add_argument("--probes", type=Path, required=True,
                   help="Parquet with image_sha256 + 12 sg_*_margin columns")
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"variant={args.variant} device={device}")

    print("loading data …")
    data = load_data(args.compact, args.siglip, args.probes)
    n_train = int(data["train_idx"].numel())
    n_val = int(data["val_idx"].numel())
    print(f"  train rows usable for probe recovery: {n_train}")
    print(f"  val rows: {n_val}")

    print(f"loading checkpoint: {args.checkpoint}")
    model = SigLIPStudent(args.variant).to(device)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])

    train_latents = data["latents"][data["train_idx"]]
    val_latents = data["latents"][data["val_idx"]]
    teacher_train = data["teacher_embs"][data["train_idx"]].to(torch.float32)
    teacher_val = data["teacher_embs"][data["val_idx"]].to(torch.float32)
    margins_train = data["margins"][data["train_idx"]]
    margins_val = data["margins"][data["val_idx"]]

    t0 = time.time()
    print("inference (val) …")
    student_val = run_inference(model, val_latents, device)
    print(f"  inference (val): {time.time() - t0:.2f}s")

    t0 = time.time()
    print("inference (train, for diagnostic R²) …")
    student_train = run_inference(model, train_latents, device)
    print(f"  inference (train): {time.time() - t0:.2f}s")

    report = {
        "variant": args.variant,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(ck.get("epoch", -1)),
        "n_val": n_val,
        "n_train_for_probe_recovery": n_train,
        "probe_columns": SG_COLUMNS,
    }

    print("L1 — direct cosine")
    report["L1_cosine"] = cosine_summary(student_val, teacher_val)
    print(json.dumps(report["L1_cosine"], indent=2))

    print("L2 — per-probe R² (recover probe dir from teacher train, eval on student val + train)")
    report["L2_per_probe_r2"] = per_probe_r2(student_val, teacher_val,
                                              teacher_train, margins_train,
                                              margins_val,
                                              student_train=student_train)
    for entry in report["L2_per_probe_r2"]["per_probe"]:
        train_r2 = entry.get("r2_student_train")
        gap = entry.get("train_minus_val_r2")
        gap_str = f"  train R²={train_r2:+.3f}  Δ(train-val)={gap:+.3f}" if train_r2 is not None else ""
        print(f"  {entry['probe']:35s} val R²={entry['r2_student']:+.3f}  "
              f"ceiling={entry['r2_teacher_ceiling']:+.3f}  "
              f"val_std={entry['val_std']:.3f}  train_std={entry['train_std']:.3f}{gap_str}")
    print(f"  → mean student R² = {report['L2_per_probe_r2']['r2_student_mean']:.3f}, "
          f"median = {report['L2_per_probe_r2']['r2_student_median']:.3f}, "
          f"min = {report['L2_per_probe_r2']['r2_student_min']:.3f}")
    print(f"  → probes ≥ 0.5: {report['L2_per_probe_r2']['n_probes_above_0p5']}/12, "
          f"≥ 0.7: {report['L2_per_probe_r2']['n_probes_above_0p7']}/12")

    print("L3a — augmentation invariance")
    report["L3a_invariance"] = layer_3a_invariance(model, val_latents, student_val, device)
    print(json.dumps(report["L3a_invariance"], indent=2))

    print("L3b — hflip invariance")
    report["L3b_hflip"] = layer_3b_hflip(model, val_latents, student_val, device)
    print(json.dumps(report["L3b_hflip"], indent=2))

    print("L4 — gradient sanity")
    report["L4_gradient"] = layer_4_gradient(model, val_latents, teacher_val, device)
    print(json.dumps(report["L4_gradient"], indent=2))

    print("evaluating gates …")
    report["gates"] = evaluate_gates(report)
    print(json.dumps(report["gates"], indent=2))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out_json}")
    print(f"FLOORS PASS: {report['gates']['all_floors_pass']}")
    print(f"TARGETS PASS: {report['gates']['all_targets_pass']}")


if __name__ == "__main__":
    main()
