"""Validate an arc_distill student as an identity-preservation loss.

Three layers (Layer 3 = real slider A/B is a separate script).

Layer 1 — Static proxies on val split:
  1.1 Teacher cosine summary (mean / median / p05 / frac>0.9) — the headline
      ArcFace-native metric. Already in eval.json; reproduced here so one report
      covers everything.
  1.2 Demographic ridge transfer: for each target in {age, gender, yaw, pitch,
      det_score} fit a ridge from student-emb (and a parallel ridge from
      teacher-emb) and report val R². Ratio R²(student) / R²(teacher) ≳ 0.9 is
      "the student preserves the demographic content the teacher captures."
  1.3 Augmentation-invariance verification: for each perturbation in
      {gaussian σ=0.02, gaussian σ=0.05, hflip, ±1 latent-px shift_h, shift_w}
      compute student-cosine on (anchor, perturbed) positive pairs and
      (anchor, random other) negative pairs. Report TAR@FAR=1e-3 and AUC.

Layer 2 — Gradient sanity:
  2.1 Backprop into a Flux latent: pick one val sample x, set target =
      teacher_emb of a random other row, loss = 1 − cos(student(x), target).
      Verify finite gradient, then run 100 SGD steps on x and check the loss
      actually descends. Confirms the student is usable as a loss term at all.

Output: a JSON report next to the checkpoint. Latents are loaded once and
reused across all sub-tests.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .adapter import AdapterStudent
from .dataset import is_held_out


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_val_latents_and_targets(compact_path: Path, attrs_path: Path):
    """Return (val_latents fp32 (N,16,H,W), teacher (N,512) un-normalised,
    attrs dict subset to val rows + found-mask, val_shas list)."""
    compact = torch.load(compact_path, map_location="cpu", weights_only=False)
    attrs = torch.load(attrs_path, map_location="cpu", weights_only=False)
    if list(compact["shas"]) != list(attrs["shas"]):
        raise ValueError("compact + face_attrs SHA order mismatch")
    shas = compact["shas"]
    val_idx = [i for i, s in enumerate(shas) if is_held_out(s)]
    val_idx_t = torch.tensor(val_idx, dtype=torch.long)

    latents = compact["latents"][val_idx_t].to(torch.float32)
    teacher = compact["arcface"][val_idx_t].to(torch.float32)
    val_shas = [shas[i] for i in val_idx]
    sub = {
        "age": attrs["age"][val_idx_t].to(torch.float32),
        "gender": attrs["gender"][val_idx_t].to(torch.float32),
        "pose": attrs["pose"][val_idx_t].to(torch.float32),  # (N,3) pitch,yaw,roll
        "det_score": attrs["det_score"][val_idx_t].to(torch.float32),
        "found": attrs["found"][val_idx_t].to(torch.bool) if "found" in attrs
                 else torch.ones(len(val_idx), dtype=torch.bool),
    }
    return latents, teacher, sub, val_shas


def run_student_inference(model: AdapterStudent, latents: torch.Tensor,
                          device: torch.device, batch_size: int = 256
                          ) -> torch.Tensor:
    """Returns L2-normalised (N, 512) student embeddings on CPU."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, latents.size(0), batch_size):
            x = latents[i:i + batch_size].to(device, non_blocking=True)
            z = model(x)  # AdapterStudent already L2-normalises
            out.append(z.cpu())
    return torch.cat(out, dim=0)


# ---------------------------------------------------------------------------
# Layer 1.1 — teacher cosine
# ---------------------------------------------------------------------------

def layer11_teacher_cosine(student_emb: torch.Tensor, teacher_emb: torch.Tensor):
    t = F.normalize(teacher_emb, dim=-1)
    s = F.normalize(student_emb, dim=-1)
    cos = (s * t).sum(dim=-1)
    n = cos.numel()
    return {
        "n": int(n),
        "cosine_mean": float(cos.mean()),
        "cosine_median": float(cos.median()),
        "cosine_p05": float(cos.kthvalue(max(1, int(0.05 * n))).values),
        "cosine_p95": float(cos.kthvalue(max(1, int(0.95 * n))).values),
        "cosine_min": float(cos.min()),
        "frac_above_0p9": float((cos > 0.9).float().mean()),
    }


# ---------------------------------------------------------------------------
# Layer 1.2 — demographic ridge transfer
# ---------------------------------------------------------------------------

def _ridge_r2_kfold(X: torch.Tensor, y: torch.Tensor, lam: float = 100.0,
                    k: int = 5) -> float | None:
    """K-fold out-of-fold R² for `Ridge(X) → y` (default lam=100; D=512 needs
    heavy smoothing on N≈1500). Aggregates predictions across folds and
    computes a single R² (more stable than mean-of-fold-R²). Returns None if
    target variance is degenerate (avoids the 0/0 → 1.0 trap)."""
    if float(y.std()) < 1e-6:
        return None
    n = X.size(0)
    fold_size = n // k
    preds = torch.zeros_like(y)
    for f in range(k):
        va_lo = f * fold_size
        va_hi = (f + 1) * fold_size if f < k - 1 else n
        va_mask = torch.zeros(n, dtype=torch.bool)
        va_mask[va_lo:va_hi] = True
        X_tr, y_tr = X[~va_mask], y[~va_mask]
        X_va = X[va_mask]

        mu_x = X_tr.mean(dim=0, keepdim=True)
        X_tr = X_tr - mu_x
        X_va = X_va - mu_x
        mu_y = y_tr.mean()
        y_tr = y_tr - mu_y

        d = X_tr.size(1)
        A = X_tr.T @ X_tr + lam * torch.eye(d, dtype=X_tr.dtype, device=X_tr.device)
        b = X_tr.T @ y_tr
        beta = torch.linalg.solve(A, b)
        preds[va_mask] = X_va @ beta + mu_y

    ss_res = ((y - preds) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12))
    return float(r2)


def layer12_ridge_transfer(student_emb: torch.Tensor, teacher_emb: torch.Tensor,
                           attrs: dict) -> dict:
    s = F.normalize(student_emb, dim=-1)
    t = F.normalize(teacher_emb, dim=-1)
    targets = {
        "age": attrs["age"],
        "gender": attrs["gender"],
        "yaw": attrs["pose"][:, 1],
        "pitch": attrs["pose"][:, 0],
        "det_score": attrs["det_score"],
    }
    found = attrs["found"]
    out = {}
    for name, y in targets.items():
        # gender/age/pose are only meaningful where re-detection succeeded;
        # det_score is meaningful for everyone.
        mask = found if name != "det_score" else torch.ones_like(found)
        s_sub = s[mask]
        t_sub = t[mask]
        y_sub = y[mask]
        r2_s = _ridge_r2_kfold(s_sub, y_sub)
        r2_t = _ridge_r2_kfold(t_sub, y_sub)
        out[name] = {
            "n": int(mask.sum()),
            "r2_student": r2_s,
            "r2_teacher": r2_t,
            # ratio is only meaningful when teacher is positive; clamp to avoid
            # blowups when teacher itself can't predict the target.
            "r2_ratio": (r2_s / r2_t) if (r2_s is not None and r2_t is not None and r2_t > 0.05) else None,
        }
    return out


# ---------------------------------------------------------------------------
# Layer 1.3 — augmentation invariance
# ---------------------------------------------------------------------------

def _perturb(latents: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "gauss_002":
        return latents + 0.02 * torch.randn_like(latents)
    if kind == "gauss_005":
        return latents + 0.05 * torch.randn_like(latents)
    if kind == "hflip":
        return latents.flip(-1)
    if kind == "shift_h":
        return torch.roll(latents, shifts=1, dims=-2)
    if kind == "shift_w":
        return torch.roll(latents, shifts=1, dims=-1)
    raise ValueError(f"unknown perturbation: {kind}")


def _tar_at_far(pos: torch.Tensor, neg: torch.Tensor, far: float) -> tuple[float, float]:
    threshold = torch.quantile(neg, 1.0 - far).item()
    tar = float((pos > threshold).float().mean())
    return tar, threshold


def _auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """Mann-Whitney U / |pos|·|neg|."""
    scores = torch.cat([pos, neg])
    labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
    order = torch.argsort(scores)
    labels = labels[order]
    # rank of each positive (1-indexed)
    ranks = torch.arange(1, scores.size(0) + 1, dtype=torch.float64)
    sum_ranks_pos = ranks[labels.bool()].sum()
    n_pos = pos.numel()
    n_neg = neg.numel()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def _hard_negative_index(teacher_emb: torch.Tensor, k_rank: int = 1,
                         seed: int = 0) -> torch.Tensor:
    """For each row i, return the index j of one of teacher's top-k nearest
    neighbours (excluding i itself). Picks deterministically from the top-k
    using the supplied seed. Output: (N,) long tensor."""
    g = torch.Generator().manual_seed(seed)
    t = F.normalize(teacher_emb, dim=-1)
    sim = t @ t.T  # (N, N)
    sim.fill_diagonal_(-2.0)  # exclude self
    topk = sim.topk(k=k_rank, dim=-1).indices  # (N, k)
    # pick a deterministic random column per row
    cols = torch.randint(0, k_rank, (sim.size(0),), generator=g)
    return topk.gather(1, cols.unsqueeze(1)).squeeze(1)


def layer13_augmentation(model: AdapterStudent, latents: torch.Tensor,
                         student_emb: torch.Tensor, teacher_emb: torch.Tensor,
                         device: torch.device, batch_size: int = 256,
                         seed: int = 0) -> dict:
    """For each perturbation, score positives (anchor vs perturbed-anchor)
    against two negative populations:
      - random: random other identity
      - hard: one of teacher's top-5 nearest neighbours (excluding self)
    Hard-negative AUC is the discriminative test: random-negatives are too
    easy because cross-identity teacher cosine clusters near 0."""
    g = torch.Generator().manual_seed(seed)
    n = latents.size(0)
    rand_perm = torch.randperm(n, generator=g)
    hard_idx = _hard_negative_index(teacher_emb, k_rank=5, seed=seed)
    out = {}
    for kind in ["gauss_002", "gauss_005", "hflip", "shift_h", "shift_w"]:
        # Reset RNG inside _perturb for deterministic gauss draws.
        torch.manual_seed(seed)
        perturbed = _perturb(latents, kind)
        emb_p = run_student_inference(model, perturbed, device, batch_size)
        emb_p = F.normalize(emb_p, dim=-1)
        emb_a = F.normalize(student_emb, dim=-1)
        pos = (emb_a * emb_p).sum(dim=-1)
        neg_rand = (emb_a * emb_p[rand_perm]).sum(dim=-1)
        neg_hard = (emb_a * emb_p[hard_idx]).sum(dim=-1)
        tar_rand, thr_rand = _tar_at_far(pos, neg_rand, far=1e-3)
        tar_hard, thr_hard = _tar_at_far(pos, neg_hard, far=1e-3)
        out[kind] = {
            "n_pos": int(pos.numel()),
            "pos_cos_mean": float(pos.mean()),
            "neg_random_cos_mean": float(neg_rand.mean()),
            "neg_hard_cos_mean": float(neg_hard.mean()),
            "tar_at_far_1e-3_random": tar_rand,
            "tar_at_far_1e-3_hard": tar_hard,
            "threshold_random": thr_rand,
            "threshold_hard": thr_hard,
            "auc_random": _auc(pos, neg_rand),
            "auc_hard": _auc(pos, neg_hard),
        }
    return out


# ---------------------------------------------------------------------------
# Layer 2 — gradient sanity
# ---------------------------------------------------------------------------

def layer2_gradient_sanity(model: AdapterStudent, latents: torch.Tensor,
                           teacher_emb: torch.Tensor, device: torch.device,
                           n_steps: int = 100, lr: float = 0.05,
                           seed: int = 0) -> dict:
    """One latent, one (different-row) target. SGD on x, track loss curve."""
    torch.manual_seed(seed)
    n = latents.size(0)
    src_idx = int(torch.randint(0, n, (1,)).item())
    tgt_idx = int(torch.randint(0, n, (1,)).item())
    while tgt_idx == src_idx:
        tgt_idx = int(torch.randint(0, n, (1,)).item())

    x = latents[src_idx:src_idx + 1].clone().to(device).requires_grad_(True)
    target = F.normalize(teacher_emb[tgt_idx:tgt_idx + 1].to(device), dim=-1)

    model.eval()
    losses = []
    grad_norms = []
    grad_finite = True
    optim = torch.optim.SGD([x], lr=lr)
    for step in range(n_steps + 1):
        optim.zero_grad()
        z = model(x)  # already L2-normalised
        loss = 1.0 - (z * target).sum()
        if step < n_steps:
            loss.backward()
            grad = x.grad
            assert grad is not None, "loss.backward() did not populate x.grad"
            if not torch.isfinite(grad).all():
                grad_finite = False
                break
            grad_norms.append(float(grad.norm()))
            optim.step()
        losses.append(float(loss.item()))
    return {
        "src_idx": src_idx,
        "tgt_idx": tgt_idx,
        "grad_finite": grad_finite,
        "grad_norm_step0": grad_norms[0] if grad_norms else None,
        "grad_norm_mean": (sum(grad_norms) / len(grad_norms)) if grad_norms else None,
        "loss_step0": losses[0],
        "loss_step50": losses[50] if len(losses) > 50 else None,
        "loss_final": losses[-1],
        "loss_descended": losses[-1] < losses[0] - 0.05,
        "loss_curve": losses,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--compact", type=Path, required=True)
    p.add_argument("--face-attrs", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"variant={args.variant} device={device}")
    print(f"loading val latents + attrs from {args.compact}")
    latents, teacher, attrs, _ = load_val_latents_and_targets(args.compact, args.face_attrs)
    print(f"  val rows: {latents.size(0)}  latent shape: {tuple(latents.shape[1:])}")

    print(f"loading checkpoint: {args.checkpoint}")
    model = AdapterStudent(args.variant).to(device)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])

    report = {
        "variant": args.variant,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(ck.get("epoch", -1)),
        "n_val": int(latents.size(0)),
    }

    t0 = time.time()
    print("inference …")
    student_emb = run_student_inference(model, latents, device, args.batch_size)
    report["inference_seconds"] = time.time() - t0

    print("layer 1.1 — teacher cosine")
    report["layer_1_1_teacher_cosine"] = layer11_teacher_cosine(student_emb, teacher)
    print(json.dumps(report["layer_1_1_teacher_cosine"], indent=2))

    print("layer 1.2 — demographic ridge transfer")
    report["layer_1_2_ridge_transfer"] = layer12_ridge_transfer(student_emb, teacher, attrs)
    print(json.dumps(report["layer_1_2_ridge_transfer"], indent=2))

    print("layer 1.3 — augmentation invariance")
    report["layer_1_3_augmentation"] = layer13_augmentation(
        model, latents, student_emb, teacher, device, args.batch_size, seed=args.seed)
    print(json.dumps(report["layer_1_3_augmentation"], indent=2))

    print("layer 2 — gradient sanity")
    report["layer_2_gradient"] = layer2_gradient_sanity(
        model, latents, teacher, device, seed=args.seed)
    print(json.dumps({k: v for k, v in report["layer_2_gradient"].items()
                      if k != "loss_curve"}, indent=2))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
