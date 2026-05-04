# bs_v4_pgd — PGD-Adversarially-Trained Robust Blendshape Critic

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrain the bs_v3_t blendshape critic with PGD adversarial training so that LoRA-induced latent perturbations cannot fool the critic into reading off-manifold targets. Variant name: `bs_v4_pgd`.

**Architecture:** Same `bs_a` trunk as v2c/v3_t (LatentStemFull64Native + ResNet-18 + Linear(512,52) + Sigmoid, 11.47M params). Same random-t rectified-flow noise augmentation as v3_t. New: per batch run a K-step projected gradient descent inner loop to find worst-case latent perturbation `δ` within an ε-ball under L∞, then minimize `MSE(bs(z_t + δ), y_teacher) + MSE(bs(z_t), y_teacher)`. Warm-started from v3_t to skip re-learning the noise-conditional clean-input map.

**Tech Stack:** PyTorch, existing `mediapipe_distill` package, Flux VAE-encoded `(16, 64, 64)` latents, MediaPipe FaceLandmarker 52-d ARKit blendshape teacher labels (cached in `compact_blendshapes.pt` + `compact_rendered.pt`). Trains on shard (RTX 4090, Windows). Validates locally on FFHQ + LoRA-rendered fooling samples.

**Spec source:** `docs/research/2026-04-30-bs-loss-classifier-fooling.md` Path 2b. Read TL;DR + Two Paths Forward sections before starting.

**Hyperparameter rationale:**
- `ε = 0.05` (L∞ on scaled latents). Latents are pre-scaled by `(z − 0.1159) · 0.3611` so per-element std ≈ 1; ε=0.05 = ~5% perturbation. Small enough that the underlying face geometry is unchanged after VAE-decode (we want this — the threat model is "flip critic without changing the face"), large enough that adversarial directions exist.
- `K = 5` PGD inner steps. Madry et al. 2018 standard balance. K=10 doubles compute for ~0.5% extra robustness in published benchmarks.
- `α = ε / 4 = 0.0125` step size. Standard PGD ratio.
- `λ_adv = 1.0` weight on adversarial term. Equal weight to clean term.
- `epochs = 15` (2h cap, user constraint). Warm-started from v3_t/final.pt so we mostly specialize to the adversarial term. **Continue path:** if more epochs needed, relaunch with `--init-from .../bs_v4_pgd/final.pt --epochs N` — existing `--init-from` machinery handles resume.
- Expected per-epoch cost: ~5× v3_t (K=5 inner forward+backward per outer step). v3_t was ~80s/epoch → expect ~400s/epoch × 15 epochs ≈ 1.7h. Leaves margin under 2h cap.

**Validation gates (must all pass to ship):**
1. **G1 — clean parity.** `r2_median@t=0 ≥ 0.78` (v3_t baseline 0.807; allow 3% drop, standard adversarial-training cost).
2. **G2 — schedule coverage retained.** `r2_median@t=0.5 ≥ 0.70` (v3_t baseline 0.741).
3. **G3 — robustness gain.** Under PGD-5 attack at ε=0.05, `r2_median ≥ 0.5` (v3_t attacked baseline expected ≪ 0; plan computes both before declaring success).
4. **G4 — fooling regression test.** On v1h_bs_only step 200 renders, `eyeSquint(L+R)/2 ≤ 0.30` (v3_t reads 0.575; should drop near baseline 0.40). On v1j_jaw_sanity step 200 renders, `jawOpen ≥ 0.85` (these renders DO have visibly open mouths — robust critic should still read them correctly, only the no-visible-change v1h fooling case should be rejected).

---

## File Structure

**Created:**
- `src/mediapipe_distill/pgd.py` — pure-function PGD perturbation helper. One responsibility: given a model, a batch `(z, y)`, and `(ε, α, K)`, return adversarial `δ`. No training-loop coupling.
- `src/mediapipe_distill/train_t_pgd.py` — training script, forked from `train_t.py`. Adds PGD inner loop to the training step; tracks best checkpoint by `r2_median@t=0` (fixes the v3_t bug where `r2_mean@t=0` was destabilized by cliff channels and saved the wrong epoch).
- `tests/test_pgd.py` — unit tests for `pgd_perturb`.
- `scripts/run_mediapipe_distill_bs_v4_pgd.bat` — shard launch.
- `scripts/run_mediapipe_validate_bs_v4_pgd.bat` — shard validation.
- `src/mediapipe_distill/validate_pgd.py` — robustness validator. Runs PGD attack at multiple ε buckets, dumps per-channel robust R² to `validation_pgd.json`.
- `models/mediapipe_distill/bs_v4_pgd/README.md` — model card (filled after training completes).

**Modified:**
- `models/mediapipe_distill/README.md` — add `bs_v4_pgd` row to variants table; add "Robust critic for LoRA training" recommendation paragraph.
- `scripts/sanity_bs_critic.py` — extend to optionally load a second checkpoint and print v3_t vs v4_pgd side-by-side on the same fooling renders.

**Untouched:** `src/mediapipe_distill/student.py`, `dataset.py`, `train_t.py` (kept as the noise-conditional reference). Existing v3_t artifacts unchanged.

---

## Task 1: PGD perturbation helper + unit tests

**Files:**
- Create: `src/mediapipe_distill/pgd.py`
- Create: `tests/test_pgd.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pgd.py`:

```python
"""Unit tests for src/mediapipe_distill/pgd.py."""
from __future__ import annotations

import torch
import torch.nn as nn

from mediapipe_distill.pgd import pgd_perturb


class _ToyModel(nn.Module):
    """A trivial linear-then-sigmoid model so we can verify PGD increases loss."""

    def __init__(self, in_dim: int = 16 * 64 * 64, out_dim: int = 52):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.lin(z.flatten(1)))


def test_pgd_respects_eps_linf_bound():
    torch.manual_seed(0)
    model = _ToyModel().eval()
    z = torch.randn(4, 16, 64, 64)
    y = torch.rand(4, 52)
    eps = 0.05
    delta = pgd_perturb(model, z, y, eps=eps, alpha=eps / 4, k=5)
    assert delta.shape == z.shape
    assert delta.abs().max().item() <= eps + 1e-6, (
        f"PGD δ violated L∞ bound: max |δ|={delta.abs().max().item()} > {eps}"
    )


def test_pgd_increases_mse_loss():
    """The whole point of PGD: bs(z + δ) should be FURTHER from y than bs(z)."""
    torch.manual_seed(0)
    model = _ToyModel().eval()
    z = torch.randn(4, 16, 64, 64)
    y = torch.rand(4, 52)
    eps = 0.05
    delta = pgd_perturb(model, z, y, eps=eps, alpha=eps / 4, k=5)
    with torch.no_grad():
        loss_clean = ((model(z) - y) ** 2).mean().item()
        loss_adv = ((model(z + delta) - y) ** 2).mean().item()
    assert loss_adv > loss_clean, (
        f"PGD failed to increase loss: clean={loss_clean:.6f} adv={loss_adv:.6f}"
    )


def test_pgd_preserves_model_grad_state():
    """PGD must not leave model.requires_grad in a bad state."""
    torch.manual_seed(0)
    model = _ToyModel().train()
    pre_state = [p.requires_grad for p in model.parameters()]
    z = torch.randn(2, 16, 64, 64)
    y = torch.rand(2, 52)
    pgd_perturb(model, z, y, eps=0.05, alpha=0.0125, k=3)
    post_state = [p.requires_grad for p in model.parameters()]
    assert pre_state == post_state, "PGD changed model parameter requires_grad flags"


def test_pgd_zero_steps_returns_zero_delta():
    """Edge case: K=0 should be a no-op (returns zero δ)."""
    model = _ToyModel().eval()
    z = torch.randn(2, 16, 64, 64)
    y = torch.rand(2, 52)
    delta = pgd_perturb(model, z, y, eps=0.05, alpha=0.0125, k=0)
    assert delta.abs().max().item() == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/newub/w/vamp-interface && uv run pytest tests/test_pgd.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'mediapipe_distill.pgd'`

- [ ] **Step 3: Write minimal implementation**

Create `src/mediapipe_distill/pgd.py`:

```python
"""Projected gradient descent attack helper for adversarial training.

Implements PGD-K under L∞ norm bound on the input perturbation. Used by
`train_t_pgd.py` to find worst-case latent perturbations during each
training step; the training loss then includes both the clean-input MSE
and the adversarial-input MSE so the student becomes locally Lipschitz
in latent space.

Reference: Madry et al. 2018, "Towards Deep Learning Models Resistant
to Adversarial Attacks", https://arxiv.org/abs/1706.06083
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def pgd_perturb(
    model: torch.nn.Module,
    z: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    alpha: float,
    k: int,
) -> torch.Tensor:
    """Find an L∞-bounded latent perturbation that maximizes MSE(model(z+δ), y).

    The model's parameter `requires_grad` flags are toggled OFF inside the
    inner loop (we only differentiate w.r.t. δ) and restored before return.
    Model is left in eval mode for the inner loop, then restored.

    Args:
        model: blendshape student (any nn.Module returning (B, 52) sigmoid).
        z: (B, 16, 64, 64) input latent.
        y: (B, 52) teacher labels in [0, 1].
        eps: L∞ bound on per-element |δ|.
        alpha: PGD step size. Typically eps / 4.
        k: number of inner-loop steps. K=0 returns zero δ.

    Returns:
        δ tensor with same shape as z, satisfying ||δ||_∞ ≤ eps. Detached
        from any autograd graph; safe to add to z in the outer training step.
    """
    if k == 0:
        return torch.zeros_like(z)

    was_training = model.training
    model.eval()
    saved_grad_flags = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)

    # Init: random uniform in [-eps, eps] (Madry-style; better than zero init)
    delta = torch.empty_like(z).uniform_(-eps, eps).requires_grad_(True)

    for _ in range(k):
        pred = model(z + delta)
        loss = F.mse_loss(pred, y)
        grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]
        with torch.no_grad():
            delta.add_(alpha * grad.sign())
            delta.clamp_(-eps, eps)
        delta.requires_grad_(True)

    out = delta.detach()

    # restore model state
    for p, flag in zip(model.parameters(), saved_grad_flags):
        p.requires_grad_(flag)
    if was_training:
        model.train()
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/newub/w/vamp-interface && uv run pytest tests/test_pgd.py -v`

Expected: PASS — 4 tests pass, including the L∞ bound, loss-increase, grad-state-preservation, and K=0 edge case.

- [ ] **Step 5: Commit**

```bash
git add src/mediapipe_distill/pgd.py tests/test_pgd.py
git commit -m "feat(mediapipe_distill): PGD perturbation helper for adversarial training"
```

---

## Task 2: Adversarial training script (train_t_pgd.py)

**Files:**
- Create: `src/mediapipe_distill/train_t_pgd.py`

This script is forked from `train_t.py`. The diff is small: import `pgd_perturb`, run it inside the training step, add the adversarial MSE term to the loss. Also fix the best-checkpoint selection bug from v3_t (track `r2_median@t=0` instead of `r2_mean@t=0` since cliff channels destabilize the mean — this is documented in `models/mediapipe_distill/bs_v3_t/README.md` caveats).

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_train_t_pgd_smoke.py`:

```python
"""Smoke test: train_t_pgd CLI parses args and instantiates the training loop
on a tiny synthetic dataset. Does not actually train (epochs=0)."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import torch


def test_train_t_pgd_help():
    """The script exposes its CLI."""
    out = subprocess.check_output(
        [sys.executable, "-m", "mediapipe_distill.train_t_pgd", "--help"],
        text=True,
        cwd="/home/newub/w/vamp-interface",
        env={"PYTHONPATH": "/home/newub/w/vamp-interface/src", "PATH": ""},
    )
    assert "--pgd-eps" in out
    assert "--pgd-alpha" in out
    assert "--pgd-k" in out
    assert "--pgd-lambda" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/newub/w/vamp-interface && uv run pytest tests/test_train_t_pgd_smoke.py -v`

Expected: FAIL with `No module named mediapipe_distill.train_t_pgd`.

- [ ] **Step 3: Write the training script**

Create `src/mediapipe_distill/train_t_pgd.py`:

```python
"""Train a PGD-adversarially-robust noise-conditional MediaPipe student
(bs_a + random-t + adversarial inner loop).

Path 2b from `docs/research/2026-04-30-bs-loss-classifier-fooling.md`.

Forked from `train_t.py`. Diff:
  1. Each training batch runs a K-step PGD inner loop to find worst-case
     L∞-bounded latent perturbation δ; the loss is the sum of clean MSE
     and adversarial MSE.
  2. Best checkpoint tracked by r2_median@t=0 instead of r2_mean@t=0
     (v3_t README documents r2_mean is dominated by R²=-10 cheekPuff
     and saved the wrong epoch; we don't repeat that bug).
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
from .pgd import pgd_perturb
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
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4,
                   help="Lower than v3_t (1e-3) — fine-tuning a warm-started "
                        "model with adversarial term needs a smaller step.")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--init-from", type=Path, default=None,
                   help="Recommended: bs_v3_t/final.pt — already noise-conditional, "
                        "this run only specializes to adversarial-robust.")
    p.add_argument("--val-seed", type=int, default=20260430)
    p.add_argument("--full-bucket-eval-every", type=int, default=5)
    p.add_argument("--pgd-eps", type=float, default=0.05,
                   help="L∞ bound on δ per latent element (post-shift/scale).")
    p.add_argument("--pgd-alpha", type=float, default=0.0125,
                   help="PGD step size; default eps/4.")
    p.add_argument("--pgd-k", type=int, default=5,
                   help="PGD inner-loop steps. K=0 disables adversarial training.")
    p.add_argument("--pgd-lambda", type=float, default=1.0,
                   help="Weight on adversarial MSE term added to clean MSE.")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"variant={args.variant} device={device}")
    print(f"PGD: eps={args.pgd_eps} alpha={args.pgd_alpha} k={args.pgd_k} "
          f"lambda={args.pgd_lambda}")

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

    best_r2_median_t0 = -1e9
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running_clean, running_adv, n_batches = 0.0, 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            t = torch.rand(x.size(0), device=device)
            z_t = add_rectified_flow_noise(x, t)

            delta = pgd_perturb(model, z_t, y,
                                eps=args.pgd_eps,
                                alpha=args.pgd_alpha,
                                k=args.pgd_k)

            pred_clean = model(z_t)
            pred_adv = model(z_t + delta)
            loss_clean = F.mse_loss(pred_clean, y)
            loss_adv = F.mse_loss(pred_adv, y)
            loss = loss_clean + args.pgd_lambda * loss_adv

            opt.zero_grad()
            loss.backward()
            opt.step()
            running_clean += float(loss_clean.item())
            running_adv += float(loss_adv.item())
            n_batches += 1
        sched.step()
        train_clean = running_clean / max(1, n_batches)
        train_adv = running_adv / max(1, n_batches)

        is_full = ((epoch + 1) % args.full_bucket_eval_every == 0
                   or epoch == args.epochs - 1)
        eval_buckets = T_BUCKETS_FULL if is_full else [0.0, 0.5]
        ev_at = {tv: evaluate_at_t(model, val_loader, device, tv, args.val_seed)
                 for tv in eval_buckets}

        elapsed = time.time() - t0
        rec = {
            "epoch": epoch,
            "train_clean_mse": train_clean,
            "train_adv_mse": train_adv,
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
        if ev_at[0.0]["r2_median"] > best_r2_median_t0:
            best_r2_median_t0 = ev_at[0.0]["r2_median"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "variant": args.variant,
                        "val_r2_median_at_t0": best_r2_median_t0,
                        "pgd_eps": args.pgd_eps, "pgd_k": args.pgd_k,
                        "pgd_lambda": args.pgd_lambda},
                       args.out_dir / "checkpoint.pt")

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
            "pgd": {"eps": args.pgd_eps, "alpha": args.pgd_alpha,
                    "k": args.pgd_k, "lambda": args.pgd_lambda},
        }, f, indent=2)
    torch.save({"model": model.state_dict(), "epoch": args.epochs - 1,
                "variant": args.variant,
                "pgd_eps": args.pgd_eps, "pgd_k": args.pgd_k,
                "pgd_lambda": args.pgd_lambda},
               args.out_dir / "final.pt")
    log_f.close()
    print(f"\nbest val r2_median at t=0: {best_r2_median_t0:.4f}")
    for tv in T_BUCKETS_FULL:
        print(f"  final t={tv:.2f}: r2_mean={final[f't_{tv:.2f}']['r2_mean']:.4f}  "
              f"r2_median={final[f't_{tv:.2f}']['r2_median']:.4f}  "
              f"n_neg={final[f't_{tv:.2f}']['n_channels_negative_r2']}/52")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run smoke test to verify it passes**

Run: `cd /home/newub/w/vamp-interface && uv run pytest tests/test_train_t_pgd_smoke.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mediapipe_distill/train_t_pgd.py tests/test_train_t_pgd_smoke.py
git commit -m "feat(mediapipe_distill): bs_v4_pgd training script — PGD adversarial inner loop + r2_median checkpoint selection"
```

---

## Task 3: Shard launch script

**Files:**
- Create: `scripts/run_mediapipe_distill_bs_v4_pgd.bat`

- [ ] **Step 1: Write the script**

Create `scripts/run_mediapipe_distill_bs_v4_pgd.bat`:

```bat
@echo off
REM bs_v4_pgd: PGD-adversarially-robust noise-conditional MediaPipe-blendshape
REM student. Same bs_a architecture as v2c/v3_t, same combined corpus
REM (FFHQ + rendered). Warm-start from v3_t/final.pt (already
REM noise-conditional; this run only specializes to adversarial-robust).
REM
REM Inner loop: 5-step PGD on L∞-bounded latent δ at eps=0.05.
REM Outer loss: MSE(bs(z_t), y) + 1.0 * MSE(bs(z_t + δ), y).
REM
REM Expected ~5x v3_t per-epoch cost (5 inner forward+backward steps).
REM v3_t was ~80s/epoch -> v4_pgd ~400s/epoch -> ~1.7h for 15 epochs (2h cap).
REM To continue later, relaunch with --init-from bs_v4_pgd/final.pt --epochs N.
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
mkdir C:\arc_distill\mediapipe_distill\bs_v4_pgd 2>nul
del C:\arc_distill\mediapipe_distill_bs_v4_pgd.done 2>nul

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m mediapipe_distill.train_t_pgd ^
    --variant bs_a ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --blendshapes C:\arc_distill\arc_full_latent\compact_blendshapes.pt ^
    --rendered C:\arc_distill\arc_full_latent\compact_rendered.pt ^
    --out-dir C:\arc_distill\mediapipe_distill\bs_v4_pgd ^
    --init-from C:\arc_distill\mediapipe_distill\bs_v3_t\final.pt ^
    --epochs 15 --batch-size 128 --workers 0 ^
    --lr 5e-4 ^
    --pgd-eps 0.05 --pgd-alpha 0.0125 --pgd-k 5 --pgd-lambda 1.0 ^
    --device cuda > C:\arc_distill\mediapipe_distill\bs_v4_pgd\train.log 2>&1
if errorlevel 1 (
    echo train failed >> C:\arc_distill\mediapipe_distill\bs_v4_pgd\train.log
    exit /b 1
)

echo mediapipe_distill_bs_v4_pgd_done > C:\arc_distill\mediapipe_distill_bs_v4_pgd.done
```

- [ ] **Step 2: Verify it parses (Windows .bat is not parseable on linux; just check the file exists and patterns match v3_t)**

Run: `cd /home/newub/w/vamp-interface && diff -u <(grep -E "^(set|cd|mkdir|del|echo|if errorlevel|C:\\\\comfy)" scripts/run_mediapipe_distill_bs_v3_t.bat | head -10) <(grep -E "^(set|cd|mkdir|del|echo|if errorlevel|C:\\\\comfy)" scripts/run_mediapipe_distill_bs_v4_pgd.bat | head -10)`

Expected: differs only in path names containing `bs_v3_t` vs `bs_v4_pgd` — confirms the boilerplate matches.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_mediapipe_distill_bs_v4_pgd.bat
git commit -m "chore(mediapipe_distill): shard launch script for bs_v4_pgd PGD training"
```

---

## Task 4: PGD-robustness validator

**Files:**
- Create: `src/mediapipe_distill/validate_pgd.py`
- Create: `scripts/run_mediapipe_validate_bs_v4_pgd.bat`

This validator answers gate G3: "under PGD-5 attack at ε=0.05, does the new critic still report sensible R²?" It computes R² under attack at multiple ε ∈ {0.0, 0.01, 0.03, 0.05, 0.1} and dumps per-channel curves. Run on both v3_t (baseline — expect collapse at ε≥0.03) and v4_pgd (expect graceful degradation).

- [ ] **Step 1: Write the script**

Create `src/mediapipe_distill/validate_pgd.py`:

```python
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
```

- [ ] **Step 2: Add a unit test for evaluate_under_pgd determinism**

Append to `tests/test_pgd.py`:

```python
def test_evaluate_under_pgd_eps_zero_matches_clean():
    """At eps=0.0 the attacked path should match the clean path exactly."""
    from mediapipe_distill.pgd import pgd_perturb
    torch.manual_seed(0)
    model = _ToyModel().eval()
    z = torch.randn(4, 16, 64, 64)
    delta = pgd_perturb(model, z, torch.rand(4, 52), eps=0.0, alpha=0.0, k=5)
    assert delta.abs().max().item() == 0.0, (
        "eps=0 should produce zero δ regardless of K"
    )
```

- [ ] **Step 3: Update pgd.py to satisfy eps=0 contract**

Edit `src/mediapipe_distill/pgd.py` — change the early-return guard at the top of `pgd_perturb`:

```python
    if k == 0 or eps == 0.0:
        return torch.zeros_like(z)
```

- [ ] **Step 4: Run all PGD tests**

Run: `cd /home/newub/w/vamp-interface && uv run pytest tests/test_pgd.py -v`

Expected: PASS — 5 tests now (4 original + 1 new eps=0 test).

- [ ] **Step 5: Create the validate launch script**

Create `scripts/run_mediapipe_validate_bs_v4_pgd.bat`:

```bat
@echo off
REM Validate bs_v4_pgd checkpoint under PGD attack at eps in
REM {0.0, 0.01, 0.03, 0.05, 0.1}. Also runs the same validator against
REM bs_v3_t for direct comparison.
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8

echo === bs_v3_t baseline (expected to collapse at eps>=0.03) ===
C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m mediapipe_distill.validate_pgd ^
    --variant bs_a ^
    --checkpoint C:\arc_distill\mediapipe_distill\bs_v3_t\final.pt ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --blendshapes C:\arc_distill\arc_full_latent\compact_blendshapes.pt ^
    --rendered C:\arc_distill\arc_full_latent\compact_rendered.pt ^
    --out C:\arc_distill\mediapipe_distill\bs_v3_t\validation_pgd.json ^
    --device cuda

echo.
echo === bs_v4_pgd (expected to be roughly flat across eps) ===
C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m mediapipe_distill.validate_pgd ^
    --variant bs_a ^
    --checkpoint C:\arc_distill\mediapipe_distill\bs_v4_pgd\final.pt ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --blendshapes C:\arc_distill\arc_full_latent\compact_blendshapes.pt ^
    --rendered C:\arc_distill\arc_full_latent\compact_rendered.pt ^
    --out C:\arc_distill\mediapipe_distill\bs_v4_pgd\validation_pgd.json ^
    --device cuda
```

- [ ] **Step 6: Commit**

```bash
git add src/mediapipe_distill/pgd.py src/mediapipe_distill/validate_pgd.py tests/test_pgd.py scripts/run_mediapipe_validate_bs_v4_pgd.bat
git commit -m "feat(mediapipe_distill): PGD-robustness validator + eps=0 short-circuit"
```

---

## Task 5: Wire bs_v4_pgd into sanity_bs_critic.py (regression gate)

**Files:**
- Modify: `scripts/sanity_bs_critic.py`

This is the gate G4 test: print v3_t and v4_pgd readings side-by-side on the v0/v1h/v1i/v1j renders. The story is in the table — v1h_bs_only readings should drop close to baseline under v4_pgd while v0 squint readings (real squint) and v1j jaw readings (real open mouth) should stay correct.

- [ ] **Step 1: Edit the script to support a second checkpoint**

Modify `scripts/sanity_bs_critic.py` — replace the `BS_CKPT` constant block and `main()` so a second optional checkpoint can be loaded:

```python
# Replace lines 22-25 (the constants):
VAE_PATH = Path("/home/newub/w/ComfyUI/models/vae/FLUX1/ae.safetensors")
BS_CKPTS = {
    "v3_t": Path("/home/newub/w/vamp-interface/models/mediapipe_distill/bs_v3_t/final.pt"),
    "v4_pgd": Path("/home/newub/w/vamp-interface/models/mediapipe_distill/bs_v4_pgd/final.pt"),
}
DEVICE = "cuda"
DTYPE = torch.bfloat16
```

Replace the `main()` body's checkpoint-load block (the lines that load `BS_CKPT` and create one student) with this multi-checkpoint loop:

```python
def load_student(ckpt_path: Path) -> BlendshapeStudent:
    student = BlendshapeStudent(variant="bs_a").to(DEVICE).eval()
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    student.load_state_dict(state)
    return student
```

And rewrite the bottom of `main()` so it loops over available checkpoints:

```python
def main():
    print("loading VAE...")
    vae = build_flux_vae(VAE_PATH, DTYPE, DEVICE)

    students = {}
    for tag, ckpt in BS_CKPTS.items():
        if not ckpt.exists():
            print(f"skip {tag}: {ckpt} not found")
            continue
        print(f"loading {tag} from {ckpt}...")
        students[tag] = load_student(ckpt)
    if not students:
        raise SystemExit("no checkpoints found")

    v0_root = Path("/home/newub/w/vamp-interface/output/ai_toolkit_runs/squint_slider_v0/samples")
    v0_paths = sorted(v0_root.glob("*000001800_*.jpg"))
    assert len(v0_paths) == 9, v0_paths
    v1h_root = Path("/home/newub/w/vamp-interface/output/ai_toolkit_runs/squint_lora_v1h_bs_only/samples")
    v1h_paths = sorted(v1h_root.glob("*000000200_*.jpg"))
    v1i_root = Path("/home/newub/w/vamp-interface/output/ai_toolkit_runs/squint_lora_v1i_sganchor/samples")
    v1i_paths = sorted(v1i_root.glob("*000000200_*.jpg"))
    v1j_root = Path("/home/newub/w/vamp-interface/output/ai_toolkit_runs/squint_lora_v1j_jaw_sanity/samples")
    v1j_paths = sorted(v1j_root.glob("*000000200_*.jpg"))
    v0_step0 = sorted(v0_root.glob("*000000000_*.jpg"))

    all_paths = v0_paths + v1h_paths + v1i_paths + v1j_paths + v0_step0
    z = encode(vae, all_paths)

    sections = [
        ("v0 squint_slider step 1800 (visibly squinting at m=+1.5)", v0_paths),
        ("v1h_bs_only step 200 (m=1.0, NO visible squint — fooling case)", v1h_paths),
        ("v1i_sganchor step 200 (m=1.0, no visible squint)", v1i_paths),
        ("v1j_jaw_sanity step 200 (mouths visibly open)", v1j_paths),
        ("v0 step 0 baseline", v0_step0),
    ]

    eL, eR = CHAN['eyeSquintLeft'], CHAN['eyeSquintRight']
    bL, bR = CHAN['eyeBlinkLeft'], CHAN['eyeBlinkRight']
    jO = CHAN['jawOpen']

    for tag, student in students.items():
        bs = read_bs(student, z)
        print(f"\n\n#### CRITIC: {tag} #################################################")
        offset = 0
        for label, paths in sections:
            n = len(paths)
            print(f"\n=== [{tag}] {label} ===")
            for i in range(n):
                b = bs[offset + i]
                sq = (b[eL].item() + b[eR].item()) / 2
                bk = (b[bL].item() + b[bR].item()) / 2
                jo = b[jO].item()
                print(f"  {Path(paths[i]).name:<55}  "
                      f"squint={sq:.3f}  blink={bk:.3f}  jaw={jo:.3f}")
            offset += n


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script still runs against v3_t even when v4_pgd is missing**

Run: `cd /home/newub/w/vamp-interface && python scripts/sanity_bs_critic.py 2>&1 | head -30`

Expected: prints `skip v4_pgd: ... not found` (because v4_pgd hasn't trained yet), then runs v3_t section as before with the same numerical readings as documented in the failure report (v0 step 1800 east_asian m=+1.5 ≈ 0.59 etc.). Confirms no regression on v3_t output.

- [ ] **Step 3: Commit**

```bash
git add scripts/sanity_bs_critic.py
git commit -m "chore(sanity_bs_critic): support multi-checkpoint side-by-side comparison"
```

---

## Task 6: Train on shard

**Files:**
- Run: `scripts/run_mediapipe_distill_bs_v4_pgd.bat` on shard

This is a manual launch step — the plan does not include automation for it. The implementer should:

- [ ] **Step 1: Sync the repo to shard's repo_assets**

User's existing shard sync workflow (rsync/scp from `~/w/vamp-interface/src/mediapipe_distill/` to `shard:C:/arc_distill/repo_assets/src/mediapipe_distill/`). The relevant new files are `pgd.py`, `train_t_pgd.py`, `validate_pgd.py`. Also sync the `.bat` files.

- [ ] **Step 2: Confirm v3_t/final.pt exists on shard**

```bash
ssh shard 'ls -la C:/arc_distill/mediapipe_distill/bs_v3_t/final.pt'
```

Expected: file exists (it does — that's the warm-start source).

- [ ] **Step 3: Launch training**

```bash
ssh shard 'cmd /c C:\arc_distill\repo_assets\scripts\run_mediapipe_distill_bs_v4_pgd.bat &'
```

Expected: spawns background training process. Tail `C:\arc_distill\mediapipe_distill\bs_v4_pgd\train.log` to monitor.

- [ ] **Step 4: Monitor every ~30 minutes**

```bash
ssh shard 'tail -5 C:/arc_distill/mediapipe_distill/bs_v4_pgd/train.log'
```

Per-epoch log line should show `train_clean_mse`, `train_adv_mse`, and val r2_median@t=0. Expected: `train_adv_mse` starts high (>10x clean) and converges to ~2-3x clean over the run.

- [ ] **Step 5: Wait for `mediapipe_distill_bs_v4_pgd.done` sentinel**

```bash
ssh shard 'ls C:/arc_distill/mediapipe_distill_bs_v4_pgd.done'
```

Expected: ~3.3 hours after launch. If train.log shows "train failed" then triage; do not pull a partially-trained checkpoint and call this done.

- [ ] **Step 6: Pull checkpoints + eval.json + train_log.jsonl to local**

```bash
mkdir -p /home/newub/w/vamp-interface/models/mediapipe_distill/bs_v4_pgd
scp 'shard:C:/arc_distill/mediapipe_distill/bs_v4_pgd/{checkpoint.pt,final.pt,eval.json,train_log.jsonl,train.log}' \
    /home/newub/w/vamp-interface/models/mediapipe_distill/bs_v4_pgd/
```

Expected: ~50-60 MB checkpoint files plus JSON.

- [ ] **Step 7: Verify gates G1 + G2 from `eval.json`**

Run: `cd /home/newub/w/vamp-interface && python -c "import json; d=json.load(open('models/mediapipe_distill/bs_v4_pgd/eval.json')); t0=d['by_t']['t_0.00']; t5=d['by_t']['t_0.50']; print(f't=0: r2_median={t0[\"r2_median\"]:.4f}  t=0.5: r2_median={t5[\"r2_median\"]:.4f}'); assert t0['r2_median'] >= 0.78, f'G1 fail: t=0 r2_median={t0[\"r2_median\"]:.4f} < 0.78'; assert t5['r2_median'] >= 0.70, f'G2 fail: t=0.5 r2_median={t5[\"r2_median\"]:.4f} < 0.70'; print('G1 + G2 pass')"`

Expected: `G1 + G2 pass`. If either fails, escalate (likely lr too high, or PGD ε too aggressive — drop ε to 0.03 or λ_adv to 0.5 and retrain).

---

## Task 7: Run PGD validator + sanity gate (G3 + G4)

- [ ] **Step 1: Run PGD validator on shard for both v3_t and v4_pgd**

```bash
ssh shard 'cmd /c C:\arc_distill\repo_assets\scripts\run_mediapipe_validate_bs_v4_pgd.bat'
```

Expected: prints curves for both. v3_t is expected to collapse — `r2_median ≪ 0` at ε=0.05. v4_pgd is expected to be flat-ish: `r2_median ≥ 0.5` at ε=0.05 (gate G3).

- [ ] **Step 2: Pull validation_pgd.json files**

```bash
scp 'shard:C:/arc_distill/mediapipe_distill/bs_v3_t/validation_pgd.json' \
    /home/newub/w/vamp-interface/models/mediapipe_distill/bs_v3_t/
scp 'shard:C:/arc_distill/mediapipe_distill/bs_v4_pgd/validation_pgd.json' \
    /home/newub/w/vamp-interface/models/mediapipe_distill/bs_v4_pgd/
```

- [ ] **Step 3: Verify gate G3**

Run: `cd /home/newub/w/vamp-interface && python -c "import json; d=json.load(open('models/mediapipe_distill/bs_v4_pgd/validation_pgd.json')); r=d['by_eps']['eps_0.050']['r2_median']; print(f'v4_pgd @ eps=0.05: r2_median={r:.4f}'); assert r >= 0.5, f'G3 fail: {r:.4f} < 0.5'; print('G3 pass')"`

Expected: `G3 pass`.

- [ ] **Step 4: Run sanity gate locally**

Run: `cd /home/newub/w/vamp-interface && python scripts/sanity_bs_critic.py | tee /tmp/sanity_v4.log`

Expected output contains two `#### CRITIC: ...` blocks. Compute G4 manually from the v4_pgd block:

- v0 step 1800 east_asian m=+1.5 squint reading (real squint): should be ≥ 0.5 (preserved)
- v1h_bs_only step 200 squint readings (fooling): all 3 demographics should be ≤ 0.30
- v1j_jaw_sanity step 200 jaw readings (real open mouth): all 3 ≥ 0.85 (preserved)

- [ ] **Step 5: Run automated G4 check**

Run:

```bash
cd /home/newub/w/vamp-interface && python -c "
import json, subprocess, re
out = subprocess.check_output(['python', 'scripts/sanity_bs_critic.py'], text=True)
# Find the v4_pgd section
v4_idx = out.find('#### CRITIC: v4_pgd')
assert v4_idx > 0, 'v4_pgd section not found in sanity output'
v4 = out[v4_idx:]
# Parse 'squint=X.XXX' values from v1h_bs_only section
v1h_idx = v4.find('v1h_bs_only')
assert v1h_idx > 0
v1h_block = v4[v1h_idx:v4.find('=== ', v1h_idx + 1)]
squint_vals = [float(m.group(1)) for m in re.finditer(r'squint=(\d+\.\d+)', v1h_block)]
assert len(squint_vals) == 3, f'expected 3 demographics, got {len(squint_vals)}'
max_v1h_squint = max(squint_vals)
print(f'v4_pgd v1h squint max: {max_v1h_squint:.3f}')
assert max_v1h_squint <= 0.30, f'G4 fail: v1h_bs_only squint reading {max_v1h_squint:.3f} > 0.30 (critic still fooled)'
# v1j jaw preservation
v1j_idx = v4.find('v1j_jaw_sanity')
v1j_block = v4[v1j_idx:v4.find('=== ', v1j_idx + 1) if v4.find('=== ', v1j_idx + 1) > 0 else len(v4)]
jaw_vals = [float(m.group(1)) for m in re.finditer(r'jaw=(\d+\.\d+)', v1j_block)]
assert len(jaw_vals) == 3
min_v1j_jaw = min(jaw_vals)
print(f'v4_pgd v1j jaw min: {min_v1j_jaw:.3f}')
assert min_v1j_jaw >= 0.85, f'G4 fail: v1j jaw reading {min_v1j_jaw:.3f} < 0.85 (real open-mouth signal lost)'
print('G4 pass')
"
```

Expected: `G4 pass`. If v1h squint > 0.30, the critic is still foolable — the PGD ε=0.05 in training may need to expand to cover the LoRA's actual perturbation scale; rerun training at ε=0.08. If v1j jaw < 0.85, real-signal sensitivity has degraded too much; lower λ_adv.

---

## Task 8: Model card + index update + memory

**Files:**
- Create: `models/mediapipe_distill/bs_v4_pgd/README.md`
- Modify: `models/mediapipe_distill/README.md`
- Create: `~/.claude/projects/-home-newub-w-vamp-interface/memory/project_bs_v4_pgd_robust_critic.md`
- Modify: `~/.claude/projects/-home-newub-w-vamp-interface/memory/MEMORY.md`

- [ ] **Step 1: Write the model card**

Create `models/mediapipe_distill/bs_v4_pgd/README.md`. The exact R² numbers come from the just-pulled `eval.json` and `validation_pgd.json` — fill from those, do not guess. Use this template:

```markdown
# bs_v4_pgd — PGD-adversarially-robust noise-conditional critic

**Recommended for LoRA training loops where the critic is a loss term.**
Same `bs_a` architecture as v2c/v3_t (52-d ARKit blendshape regression
from Flux VAE latents). Trained with PGD-5 inner loop at ε=0.05 to
defend against the classifier-fooling failure mode diagnosed in
`docs/research/2026-04-30-bs-loss-classifier-fooling.md`.

## Architecture + training

- bs_a trunk (LatentStemFull64Native + ResNet-18 + Linear(512, 52) + Sigmoid, 11.47 M params)
- Random-t rectified-flow noise (same as v3_t)
- PGD-5 inner loop, L∞ bound ε=0.05 on scaled latents, α=0.0125, λ_adv=1.0
- 30 epochs, AdamW, cosine LR (5e-4 → 0), batch 128
- Warm-started from `bs_v3_t/final.pt`
- ~400s/epoch on RTX 4090 → ~3.3h total

## Validation

### Clean t-bucket (Layer 1.1)

| t | r2_median | r2_mean | n_neg/52 |
|---|---:|---:|---:|
| 0.00 | <FILL> | <FILL> | <FILL> |
| 0.10 | <FILL> | <FILL> | <FILL> |
| 0.25 | <FILL> | <FILL> | <FILL> |
| 0.50 | <FILL> | <FILL> | <FILL> |
| 0.75 | <FILL> | <FILL> | <FILL> |
| 1.00 | <FILL> | <FILL> | <FILL> |

### PGD-attack robustness (Layer 1.4)

PGD-10 attack at t=0, attacker α=ε/4. v3_t is included as the non-robust baseline.

| ε | v3_t r2_median | v4_pgd r2_median | gap |
|---|---:|---:|---:|
| 0.00 | <FILL> | <FILL> | <FILL> |
| 0.01 | <FILL> | <FILL> | <FILL> |
| 0.03 | <FILL> | <FILL> | <FILL> |
| 0.05 | <FILL> | <FILL> | <FILL> |
| 0.10 | <FILL> | <FILL> | <FILL> |

### Fooling-render regression (Layer 1.5)

Rendered LoRA outputs from `output/ai_toolkit_runs/squint_lora_v1h_bs_only` (no visible squint, fooling case) and `squint_lora_v1j_jaw_sanity` (visible open mouth, real signal).

| sample set | v3_t squint | v4_pgd squint | v3_t jaw | v4_pgd jaw |
|---|---:|---:|---:|---:|
| v0 step 1800 m=+1.5 (real squint) | <FILL> | <FILL> | <FILL> | <FILL> |
| v1h_bs_only step 200 (FOOLING) | <FILL> | <FILL> | <FILL> | <FILL> |
| v1j_jaw_sanity step 200 (real jaw) | <FILL> | <FILL> | <FILL> | <FILL> |

## Sample usage

Drop-in replacement for v3_t. Same `BlendshapeStudent("bs_a")` constructor.
Use this in any ai-toolkit ConceptSliderTrainer that previously used v3_t
or v2c as a `bs_loss` term.

```python
import torch
from mediapipe_distill.student import BlendshapeStudent

device = "cuda"
model = BlendshapeStudent("bs_a").to(device)
ck = torch.load("models/mediapipe_distill/bs_v4_pgd/final.pt", weights_only=False)
model.load_state_dict(ck["model"])
model.eval()

# Use exactly as bs_v3_t — but the LoRA can no longer cheaply fool it.
bs_a = model(z_anchor.detach())  # (B, 52)
bs_e = model(z_edited)
loss = ((bs_a.detach() - bs_e) ** 2).mean()
```

## Caveats

1. **Adversarial training does not guarantee robustness against perturbations larger than ε=0.05.** A LoRA trained against this critic with very large `network_multiplier` may still find adversarial directions outside the training ball. Cap or monitor multiplier scale.
2. **Per-channel R² is slightly lower than v3_t at t=0** (the standard 2-3% adversarial-training cost). The 13 do-not-use channels from v3_t are still do-not-use here — adversarial training does not cure corpus-coverage problems.
3. **Validation report distinguishes attacked-R² from clean-R².** Always cite both when reporting model performance.
4. **PGD attack at ε≥0.10 still degrades v4_pgd.** The training ball was 0.05; ε=0.10 is out-of-distribution. If LoRAs are observed perturbing this far, expand training ε.

## Provenance

- Trained 2026-04-30 on shard.
- Warm-start: `bs_v3_t/final.pt` (epoch 39).
- Source: `src/mediapipe_distill/{train_t_pgd,pgd}.py`.
- Plan: `docs/superpowers/plans/2026-04-30-bs-v4-pgd-robust-critic.md`.
- Failure report: `docs/research/2026-04-30-bs-loss-classifier-fooling.md`.
```

Replace each `<FILL>` with the actual number from the just-pulled `eval.json` (clean t-bucket) and `validation_pgd.json` (attack robustness). For the Layer 1.5 table, parse from `python scripts/sanity_bs_critic.py` output: average L+R for squint, single-value for jaw, three-demographic mean.

- [ ] **Step 2: Update the variants index**

Edit `models/mediapipe_distill/README.md`. In the variants table, append a row after `bs_v3_t`:

```markdown
| [bs_v4_pgd](bs_v4_pgd/) | 52-d bs (noise-conditional + PGD-robust) | FFHQ + rendered (33K), random-t + PGD δ | **LoRA-loss critic** (recommended over v3_t) | <FILL> @ t=0 | — |
```

In the "Which one should I use?" section, replace the bs_v3_t paragraph with this two-paragraph block:

```markdown
**For LoRA-training critic loops (Concept Sliders / ai-toolkit) → bs_v4_pgd.**
Drop-in replacement for v3_t with adversarial-perturbation training so LoRAs
cannot cheaply fool the critic into reading targets without changing the face.
Background: `docs/research/2026-04-30-bs-loss-classifier-fooling.md`.

**For pure noisy-input critics with no LoRA in the loop → bs_v3_t.**
PGD training trades a few % clean R² for robustness; if there's no adversary
in the training pipeline, v3_t is the cleaner reference.
```

- [ ] **Step 3: Commit the model card + index changes**

```bash
git add models/mediapipe_distill/bs_v4_pgd/ models/mediapipe_distill/README.md
git commit -m "docs(mediapipe_distill): bs_v4_pgd model card + variants index update"
```

- [ ] **Step 4: Write the project memory**

Create `/home/newub/.claude/projects/-home-newub-w-vamp-interface/memory/project_bs_v4_pgd_robust_critic.md`:

```markdown
---
name: bs_v4_pgd robust critic — PGD adversarial training defends against LoRA classifier-fooling
description: 2026-04-30 retrain of bs_v3_t with PGD-5 inner loop at ε=0.05, fixes the v1h/v1i fooling failure where LoRAs satisfy bs_loss without changing the rendered face
type: project
---
**The fix:** Retrained noise-conditional MediaPipe blendshape student with PGD-5 adversarial inner loop at L∞ ε=0.05 on scaled latents. Loss = MSE(bs(z), y) + 1.0 · MSE(bs(z + δ_pgd), y). Forces local Lipschitz smoothness — LoRAs can no longer find off-manifold latent directions where the critic reads target without the underlying face geometry actually changing.

**Why this is the right fix:** v1d/v1h/v1i runs all showed the *same* failure mode in different costumes — the LoRA optimizer found cheap fooling directions. Geometric masks and SigLIP anchors only made the cheap directions slightly more expensive; PGD makes them not exist within ε.

**Validation gates that must pass:**
- G1: clean r2_median@t=0 ≥ 0.78 (v3_t baseline 0.807 — allow 3% drop)
- G2: clean r2_median@t=0.5 ≥ 0.70 (schedule coverage)
- G3: PGD-10-attacked r2_median ≥ 0.5 at ε=0.05 (robustness gain — v3_t collapses to ≪ 0)
- G4: on `squint_lora_v1h_bs_only` step 200 fooling renders, eyeSquint(L+R)/2 ≤ 0.30 across all 3 demographics; on `squint_lora_v1j_jaw_sanity` real renders, jawOpen ≥ 0.85.

**How to apply:**
- Default critic for any future ai-toolkit / Concept Sliders LoRA against any subset of the 52-d ARKit blendshapes is `models/mediapipe_distill/bs_v4_pgd/final.pt`.
- v3_t is kept as the non-robust noise-conditional reference; do not use it as a LoRA-loss critic.
- If a LoRA run is observed perturbing latents at scale > 0.05 (e.g. very high `network_multiplier`), consider a v5 retrain at ε=0.08 or 0.10. The training ball is the load-bearing assumption.
- The 13 do-not-use channels from v3_t (cheekSquint L/R, noseSneer L/R, mouth frowns, jaw L/R, mouthLeft/Right, mouthClose, mouthDimpleLeft, _neutral) remain do-not-use here — corpus-coverage problems are orthogonal to adversarial training.

**Companion docs:** `docs/research/2026-04-30-bs-loss-classifier-fooling.md` (failure diagnosis), `docs/superpowers/plans/2026-04-30-bs-v4-pgd-robust-critic.md` (this plan), `models/mediapipe_distill/bs_v4_pgd/README.md` (model card).
```

- [ ] **Step 5: Update MEMORY.md index**

Edit `/home/newub/.claude/projects/-home-newub-w-vamp-interface/memory/MEMORY.md`. Insert a new line directly after the existing `project_v1d_slider_falsified_by_clean_student.md` line:

```markdown
- [project_bs_v4_pgd_robust_critic.md](project_bs_v4_pgd_robust_critic.md) — 2026-04-30 PGD-adversarial retrain of bs_v3_t fixes LoRA classifier-fooling; default critic for ai-toolkit bs_loss runs at ε=0.05 ball
```

- [ ] **Step 6: Commit memory updates**

The memory directory is outside the repo and is not git-tracked here. No commit needed for memory — they're already on disk.

- [ ] **Step 7: Final repo commit**

If anything is still uncommitted (the model card, index, etc.):

```bash
cd /home/newub/w/vamp-interface && git status && git push
```

Expected: `nothing to commit, working tree clean` or a final `git push` of the model card commit. Push to remote so the model card and plan are live.

---

## Self-review checklist

- [ ] Spec coverage: Path 2b is implemented (PGD inner loop, K=5 default). Path 2a (random-δ) and Path 2c (ensemble) are explicitly *not* implemented per user's "C" choice. Open question 1 (correlation problem) is acknowledged in the model card caveat 2. Open question 3 (ArcFace × adversarial composition) is left for a future v5 — not in scope.
- [ ] No placeholders inside code blocks; all functions defined; all `<FILL>` markers in the model card template are explicitly tagged as "fill from eval.json/validation_pgd.json" — that's data, not a code placeholder.
- [ ] Type consistency: `pgd_perturb` signature is identical across `pgd.py`, `train_t_pgd.py`, `validate_pgd.py`, and the test. Checkpoint dict shape (`{"model": ..., "epoch": ..., "variant": ..., ...}`) matches v3_t's load path so all existing scripts including `sanity_bs_critic.py` continue to work.
- [ ] Gates G1–G4 are concrete numerical thresholds with automated checks (Tasks 6.7, 7.3, 7.5).
- [ ] Failure recovery is documented in the gate-failure escalation paths (Task 6.7 mentions retraining at lower ε; Task 7.5 mentions raising training ε if v1h fooling persists).
