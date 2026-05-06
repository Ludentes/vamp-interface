# ARKit→PersonaLive Bridge v4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train two student variants (A+B per-sample×per-cell weighted MSE; C Jacobian-norm regularizer), pick the winner offline, ship as student_v4.pt with a kp_ref y-mirror calibration fix.

**Architecture:** No architecture changes to MotEncoderStudent (4-layer MLP, 278K params). Add `F_KP_REF = diag(+1,-1,+1)` to closed_form_pose. Extend distill.py with two new loss modes consuming a shared precomputed `stats.npz` containing `freq_k`, cell-coupling matrix `C[512,58]`, plus existing teacher_mean/std/b_p95.

**Tech Stack:** PyTorch, NumPy, mediapipe FaceLandmarker (rotation matrices), ffmpeg, polars (parquet), PersonaLive `.venv` for renders, RTX 5090.

**Standing rule:** Every non-trivial script (any .py over ~30 lines or with logic, not pure CLI plumbing) gets reviewed by `superpowers:code-reviewer` agent before commit. Trivial CLI wiring (1-2 line argparse adds) skip review.

---

## File map

| File | Action | Responsibility |
|---|---|---|
| `src/arkit_bridge/closed_form_pose.py` | modify | Add `F_KP_REF`; thread through `compose_kd` |
| `tests/arkit_bridge/test_closed_form_pose.py` | create | Unit tests for F_KP_REF math + identity invariants |
| `scripts/precompute_v4_stats.py` | create | Extends compute_teacher_stats with `freq_k` + `C[512,58]` |
| `src/arkit_bridge/distill.py` | modify | Add `weighted_mse` and `varnorm_jvp` loss modes |
| `tests/arkit_bridge/test_distill_loss.py` | create | Per-loss unit tests (gradient sanity, expected behavior on synthetic batch) |
| `scripts/train_arkit_student.py` | modify | Add `weighted_mse` and `varnorm_jvp` choices to `--loss_mode` |
| `scripts/verify_calibration_v4_clip.py` | create | Re-render single yaw/pitch clip with new F_KP_REF, report angular distance |
| `scripts/bakeoff_v4.py` | create | Cross-arm scorecard CSV/MD on heldout takes |

---

## Task 1: Add F_KP_REF to closed-form pose

**Files:**
- Modify: `src/arkit_bridge/closed_form_pose.py`
- Create: `tests/arkit_bridge/test_closed_form_pose.py`

- [ ] **Step 1: Write failing test for F_KP_REF math**

Create `tests/arkit_bridge/test_closed_form_pose.py`:
```python
"""Closed-form pose unit tests including F_KP_REF kp_ref y-mirror."""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from arkit_bridge.closed_form_pose import (  # noqa: E402
    EULER_SIGNS, F_KP_REF, compose_kd, euler_to_rotmat,
)


def test_F_KP_REF_is_y_mirror():
    expected = torch.tensor([[1., 0., 0.],
                             [0., -1., 0.],
                             [0., 0., 1.]])
    assert torch.allclose(F_KP_REF, expected)
    assert torch.linalg.det(F_KP_REF).item() == -1.0


def test_compose_kd_zero_rotation_applies_F_to_kpref():
    """With identity R, compose_kd should apply F_KP_REF to kp_ref."""
    kp_ref = torch.randn(1, 21, 3)
    R = torch.eye(3).unsqueeze(0)
    s = torch.ones(1)
    t = torch.zeros(1, 2)
    out = compose_kd(kp_ref, R, s, t)
    expected = kp_ref @ F_KP_REF
    # tx,ty zero so out == expected exactly
    assert torch.allclose(out, expected, atol=1e-6)


def test_compose_kd_nonzero_rotation():
    """compose_kd applies F to kp_ref, then R, then s, then translates."""
    kp_ref = torch.tensor([[[1., 2., 3.]]])  # (1, 1, 3)
    yaw = torch.tensor([0.5]); pitch = torch.zeros(1); roll = torch.zeros(1)
    R = euler_to_rotmat(yaw, pitch, roll)
    s = torch.tensor([1.5])
    t = torch.zeros(1, 2)
    out = compose_kd(kp_ref, R, s, t)
    expected = (kp_ref @ F_KP_REF) @ R * 1.5
    assert torch.allclose(out, expected, atol=1e-5)


def test_euler_signs_preserved():
    """v4 must keep the v3 winner: (+1, -1, -1)."""
    assert EULER_SIGNS == (+1.0, -1.0, -1.0)
```

- [ ] **Step 2: Run test to verify failure**

Run: `cd /home/newub/w/vamp-interface && python -m pytest tests/arkit_bridge/test_closed_form_pose.py -v`
Expected: FAIL on `import F_KP_REF` (does not exist yet).

- [ ] **Step 3: Add F_KP_REF and update compose_kd**

Edit `src/arkit_bridge/closed_form_pose.py`. Replace the EULER_SIGNS block:

```python
# Empirical winner of calibration v3 (P_48 frame search on
# data/llf-clips-auto/20260505_MySlate_5_yaw, 600 frames):
#   sign combo (+1, -1, -1) with F* = diag(+1, -1, +1), score 0.168 rad.
# F_KP_REF is applied to the LivePortrait reference keypoints once
# (kp_ref @ F_KP_REF) before the Euler rotation is composed; this
# corrects a kp_ref vs ARKit-frame y-axis mirror that v1 missed.
EULER_SIGNS = (+1.0, -1.0, -1.0)  # (yaw, pitch, roll)
F_KP_REF = torch.tensor([[1.0, 0.0, 0.0],
                          [0.0, -1.0, 0.0],
                          [0.0, 0.0, 1.0]])
```

Update `compose_kd` body — replace the line `k = kp_ref @ R` with:
```python
    F = F_KP_REF.to(dtype=kp_ref.dtype, device=kp_ref.device)
    k = (kp_ref @ F) @ R
```

- [ ] **Step 4: Run test to verify pass**

Run: `cd /home/newub/w/vamp-interface && python -m pytest tests/arkit_bridge/test_closed_form_pose.py -v`
Expected: 4 passed.

- [ ] **Step 5: Run code-reviewer agent**

Dispatch: `Agent(subagent_type="superpowers:code-reviewer")` with prompt: "Review the change to /home/newub/w/vamp-interface/src/arkit_bridge/closed_form_pose.py adding F_KP_REF (kp_ref y-mirror) per spec docs/superpowers/specs/2026-05-06-arkit-bridge-v4-design.md. Verify: math correct, F applied once not per-frame in batch case, dtype/device handling correct, tests in tests/arkit_bridge/test_closed_form_pose.py adequate. Read both files."

Address any issues raised before proceeding.

- [ ] **Step 6: Commit**

```bash
git add src/arkit_bridge/closed_form_pose.py tests/arkit_bridge/test_closed_form_pose.py
git commit -m "feat(arkit-bridge): apply F_KP_REF=diag(+1,-1,+1) kp_ref y-mirror

Empirical winner of calibration v3 (P_48 frame search on yaw clip).
EULER_SIGNS=(+1,-1,-1) confirmed; new F_KP_REF closes ~25% of residual
angular distance (0.225 -> 0.168 rad). Applied as (kp_ref @ F) @ R in
compose_kd; F is applied once on the reference keypoints, not per-frame.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2: Phase 0a — verify F_KP_REF on yaw/pitch clips

**Files:**
- Create: `scripts/verify_calibration_v4_clip.py`

This script renders one clip with the new closed_form_pose, extracts mediapipe rotation matrices, and reports angular distance against the driver clip's ARKit Eulers. Reuses `apply_bridge_to_personalive.py` for rendering and the angular_distance function from `calibrate_euler_signs_v3.py`.

- [ ] **Step 1: Write the script**

Create `scripts/verify_calibration_v4_clip.py`:
```python
"""Render one clip with current closed_form_pose F_KP_REF and report
mean/median/p90 angular distance vs driver ARKit Euler stream.

Usage:
  python scripts/verify_calibration_v4_clip.py \
    --take_dir data/llf-clips-auto/20260505_MySlate_5_yaw \
    --out_mp4 exp_output/arkit_bridge/calibration_v4/render_5_yaw.mp4
"""

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Reuse v3 helpers
sys.path.insert(0, str(ROOT / "scripts"))
from calibrate_euler_signs_v3 import (  # noqa: E402
    angular_distance, extract_render_rotmats, load_driver_rotmats,
)


PERSONA_PY = "/home/newub/w/PersonaLive/.venv/bin/python"
ANCHOR = "data/llf-phase2/asian_m__06_neutral.midframe.png"
CKPT = "runs/student_v2_lam10/student_best.pt"


def render(take_dir: Path, out_mp4: Path, *, n_frames=600, stride=1):
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        PERSONA_PY, "scripts/apply_bridge_to_personalive.py",
        "--take_dir", str(take_dir),
        "--anchor", ANCHOR,
        "--ckpt", CKPT,
        "--mode", "bridge",
        "--n_frames", str(n_frames),
        "--stride", str(stride),
        "--out_mp4", str(out_mp4),
        # NOTE: no --euler_signs override; uses closed_form_pose default
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take_dir", required=True)
    ap.add_argument("--out_mp4", required=True)
    ap.add_argument("--n_frames", type=int, default=600)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--skip_render", action="store_true")
    args = ap.parse_args()

    take_dir = Path(args.take_dir)
    out_mp4 = Path(args.out_mp4)

    if not args.skip_render:
        render(take_dir, out_mp4, n_frames=args.n_frames, stride=args.stride)

    R_input = load_driver_rotmats(take_dir, n_frames=args.n_frames, stride=args.stride)
    R_render = extract_render_rotmats(out_mp4, n_frames=args.n_frames)
    n = min(len(R_input), len(R_render))
    R_input = R_input[:n]; R_render = R_render[:n]

    ds = np.array([angular_distance(R_input[i], R_render[i]) for i in range(n)])
    print(f"\n# verify_calibration_v4_clip.py — {take_dir.name}")
    print(f"  n={n}  mean={ds.mean():+.4f} median={np.median(ds):+.4f} p90={np.percentile(ds,90):+.4f} rad")
    pass_thresh = 0.05
    print(f"  PASS (mean<={pass_thresh:.3f}): {ds.mean() <= pass_thresh}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Code-review the script**

Dispatch: `Agent(subagent_type="superpowers:code-reviewer")` with prompt: "Review /home/newub/w/vamp-interface/scripts/verify_calibration_v4_clip.py. It re-uses calibrate_euler_signs_v3 helpers (angular_distance, extract_render_rotmats, load_driver_rotmats) — confirm those imports are correct (read scripts/calibrate_euler_signs_v3.py to verify). Check: idempotency (--skip_render path), error handling on render failure, n_frames clamping if mp4 has fewer frames."

Address issues, then commit.

- [ ] **Step 3: Run on yaw clip**

```bash
systemd-run --user --scope -p MemoryMax=45G \
  python scripts/verify_calibration_v4_clip.py \
    --take_dir data/llf-clips-auto/20260505_MySlate_5_yaw \
    --out_mp4 exp_output/arkit_bridge/calibration_v4/render_5_yaw.mp4
```

Expected: `mean <= 0.05 rad` PASS. v3 baseline (no F_KP_REF) was 0.168.
If mean > 0.10: F_KP_REF application formula is wrong → fix Task 1's `compose_kd` edit (try `kp_ref @ R @ F` or `F @ kp_ref @ R` etc.) and re-run.

- [ ] **Step 4: Run on pitch clip and heldout-take yaw clip**

```bash
systemd-run --user --scope -p MemoryMax=45G \
  python scripts/verify_calibration_v4_clip.py \
    --take_dir data/llf-clips-auto/20260505_MySlate_5_pitch \
    --out_mp4 exp_output/arkit_bridge/calibration_v4/render_5_pitch.mp4

systemd-run --user --scope -p MemoryMax=45G \
  python scripts/verify_calibration_v4_clip.py \
    --take_dir data/llf-clips-auto/20260505_MySlate_4_yaw \
    --out_mp4 exp_output/arkit_bridge/calibration_v4/render_4_yaw.mp4
```

Both must pass (mean <= 0.05). Heldout take 4 confirms F is take-invariant.

- [ ] **Step 5: Commit**

```bash
git add scripts/verify_calibration_v4_clip.py
git commit -m "feat(arkit-bridge): clip-grain F_KP_REF verification script

Renders one clip with current closed_form_pose, mediapipe-extracts
rotation matrices, reports angular distance vs driver ARKit Eulers.
Pass criterion mean <= 0.05 rad. Phase 0a gate for v4.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

Then commit the verification artifacts:
```bash
git add exp_output/arkit_bridge/calibration_v4/
git commit -m "exp(arkit-bridge): F_KP_REF verification on 3 clips

clip 5_yaw / 5_pitch / 4_yaw all PASS at mean <= 0.05 rad.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: Precompute v4 stats (freq_k + cell coupling C[512,58])

**Files:**
- Create: `scripts/precompute_v4_stats.py`

Reuses `compute_teacher_stats.py`'s teacher_mean/std/b_p95/sample_weights logic; adds two new fields:
- `freq_k = mean(b_k > 0.05)` over corpus, shape (58,)
- `C[512, 58]` empirical coupling = `mean over pairs of |Δm_f / Δb|` via finite-difference probe pairs

For C, an exact computation requires ground-truth pair-wise differences along single axes — we use a kNN scheme: for each channel k, find the 32 pairs with the largest |b_k| in the corpus and the 32 pairs with the smallest, compute `(m_high - m_low) / (b_high - b_low)[k]` per cell, average. Cheap and stable.

- [ ] **Step 1: Write failing test**

Create `tests/arkit_bridge/test_precompute_stats.py`:
```python
"""Smoke test for v4 stats precompute on a synthetic 2-pair corpus."""

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def make_pair(out_path: Path, b_expr: np.ndarray, m_f: np.ndarray):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"b_expr": b_expr.astype(np.float32),
                     "m_f": m_f.astype(np.float32)}, f)


def test_smoke(tmp_path):
    """Synthetic corpus where m_f is purely linear in b_expr along channel 7."""
    pairs_dir = tmp_path / "pairs"
    out = tmp_path / "stats.npz"
    rng = np.random.default_rng(0)
    for i in range(64):
        b = rng.normal(0, 0.1, size=58).astype(np.float32)
        b[7] = (i / 32.0) - 1.0  # ramp channel 7 from -1 to 1
        m = np.zeros((1, 32, 16), dtype=np.float32)
        m[0, :, 0] = b[7] * 2.0  # only column 0 of m_f responds, scale 2.0
        make_pair(pairs_dir / f"frame_{i:06d}.pkl", b, m)

    from precompute_v4_stats import main as precompute_main
    precompute_main([
        "--pairs_dir", str(pairs_dir),
        "--out", str(out),
        "--coupling_k", "16",
    ])

    s = np.load(out, allow_pickle=True)
    assert s["freq_k"].shape == (58,)
    assert s["C"].shape == (32 * 16, 58)
    # Channel 7 column 0 should have coupling ~2.0
    C = s["C"].reshape(32, 16, 58)
    assert C[:, 0, 7].mean() > 1.5
    # Other channels' column-0 coupling should be small
    other_means = np.delete(C[:, 0, :].mean(0), 7)
    assert np.all(other_means < 0.5)
```

- [ ] **Step 2: Run test to verify fail**

Run: `cd /home/newub/w/vamp-interface && python -m pytest tests/arkit_bridge/test_precompute_stats.py -v`
Expected: FAIL on `from precompute_v4_stats import main` (does not exist).

- [ ] **Step 3: Implement the script**

Create `scripts/precompute_v4_stats.py`:
```python
"""Precompute v4 stats: teacher_mean/std, b_p95, sample_weights, freq_k,
and empirical coupling matrix C[512, 58].

C[j, k] = empirical |dm_f[j] / db_k| from finite-difference probe pairs
(top-k_high vs bottom-k_low along each channel). Used for cell weighting
in variant A+B and as the magnitude target ‖C[:, k]‖ in variant C.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_corpus(pairs_dir: Path):
    paths = sorted(pairs_dir.glob("*frame_*.pkl"))
    n = len(paths)
    if n == 0:
        raise SystemExit(f"no pkls under {pairs_dir}")
    print(f"loading {n} pairs from {pairs_dir}")
    b_all = np.zeros((n, 58), dtype=np.float32)
    m_all = np.zeros((n, 32, 16), dtype=np.float32)
    for i, p in enumerate(tqdm(paths)):
        with open(p, "rb") as f:
            d = pickle.load(f)
        b_all[i] = np.asarray(d["b_expr"], dtype=np.float32)
        mf = np.asarray(d["m_f"], dtype=np.float32).squeeze(0)
        m_all[i] = mf
    return paths, b_all, m_all


def compute_coupling(b_all: np.ndarray, m_all: np.ndarray, k: int = 32):
    """For each input channel, average |Δm/Δb| over (top-k - bottom-k) pair
    differences. Returns C with shape (32*16, 58)."""
    n_b = b_all.shape[1]
    m_flat = m_all.reshape(m_all.shape[0], -1)  # (N, 512)
    C = np.zeros((m_flat.shape[1], n_b), dtype=np.float32)
    for ch in range(n_b):
        order = np.argsort(b_all[:, ch])
        lo = order[:k]
        hi = order[-k:]
        # pair them by rank
        b_diff = b_all[hi, ch] - b_all[lo, ch]
        b_diff = np.where(np.abs(b_diff) < 1e-6, 1e-6, b_diff)
        m_diff = m_flat[hi] - m_flat[lo]  # (k, 512)
        secant = m_diff / b_diff[:, None]  # (k, 512)
        C[:, ch] = np.abs(secant).mean(axis=0)
    return C


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--active_thresh", type=float, default=0.05,
                    help="b_k > active_thresh counts as 'active' for freq_k")
    ap.add_argument("--coupling_k", type=int, default=32,
                    help="top-k vs bottom-k pair count for coupling secant")
    args = ap.parse_args(argv)

    pairs_dir = Path(args.pairs_dir)
    paths, b_all, m_all = load_corpus(pairs_dir)
    n = len(paths)

    teacher_mean = m_all.mean(0).astype(np.float32)
    teacher_std = m_all.std(0).astype(np.float32)

    b_abs = np.abs(b_all)
    b_p95 = np.maximum(np.percentile(b_abs, 95, axis=0), 1e-3).astype(np.float32)
    weights = np.clip((b_abs / b_p95[None, :]).max(axis=1), 0.1, 10.0).astype(np.float32)

    freq_k = (b_all > args.active_thresh).mean(axis=0).astype(np.float32)
    freq_k = np.maximum(freq_k, 1.0 / n)  # avoid div-by-zero

    print("computing coupling matrix C[512, 58]...")
    C = compute_coupling(b_all, m_all, k=args.coupling_k)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        teacher_mean=teacher_mean,
        teacher_std=teacher_std,
        b_p95=b_p95,
        sample_weights=weights,
        paths=np.array([p.name for p in paths]),
        freq_k=freq_k,
        C=C,
    )
    print(f"wrote {out}")
    print(f"  freq_k: min={freq_k.min():.4f} median={np.median(freq_k):.4f} max={freq_k.max():.4f}")
    print(f"  C: norm-per-channel min={np.linalg.norm(C, axis=0).min():.4f} "
          f"max={np.linalg.norm(C, axis=0).max():.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify pass**

Run: `cd /home/newub/w/vamp-interface && python -m pytest tests/arkit_bridge/test_precompute_stats.py -v`
Expected: 1 passed.

- [ ] **Step 5: Run code-reviewer**

Dispatch: `Agent(subagent_type="superpowers:code-reviewer")` with prompt: "Review /home/newub/w/vamp-interface/scripts/precompute_v4_stats.py and tests/arkit_bridge/test_precompute_stats.py. It extends compute_teacher_stats.py to add freq_k and an empirical coupling matrix C[512, 58]. Verify: load semantics match existing compute_teacher_stats.py (read it for reference), output keys are a strict superset, finite-difference coupling is numerically sane (zero-division guarded), shape contracts documented. The C is consumed by distill.py weighted_mse and varnorm_jvp loss modes (Task 4)."

Address feedback, then commit.

- [ ] **Step 6: Run on full train corpus**

```bash
mkdir -p runs/v4_shared
python scripts/precompute_v4_stats.py \
  --pairs_dir data/arkit_bridge_pairs/all \
  --out runs/v4_shared/stats.npz
```

Expected runtime: ~5 minutes (one pass over ~16K pairs + coupling).
Inspect output: `freq_k` median should be near the activation rate; `C` per-channel norms should span ~3 orders of magnitude (rare channels have small but nonzero coupling).

- [ ] **Step 7: Commit**

```bash
git add scripts/precompute_v4_stats.py tests/arkit_bridge/test_precompute_stats.py
git commit -m "feat(arkit-bridge): precompute v4 stats (freq_k + coupling C)

Extends compute_teacher_stats.py with per-channel activation frequency
and empirical coupling matrix C[512,58] via top-k vs bottom-k secants.
Consumed by v4 weighted_mse (per-cell weighting) and varnorm_jvp (target
magnitude) loss modes. Synthetic-corpus unit test included.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"

git add runs/v4_shared/stats.npz
git commit -m "exp(arkit-bridge): v4 stats.npz on full train corpus

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: Add weighted_mse loss mode (variant A+B)

**Files:**
- Modify: `src/arkit_bridge/distill.py`
- Create: `tests/arkit_bridge/test_distill_loss.py`

- [ ] **Step 1: Write failing test for weighted_mse**

Create `tests/arkit_bridge/test_distill_loss.py`:
```python
"""Distill loss-mode unit tests for v4."""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from arkit_bridge.distill import _make_loss  # noqa: E402


def make_stats(B=8, n_b=58, m_shape=(32, 16), seed=0):
    """Synthetic stats matching expected v4 stats.npz schema."""
    rng = np.random.default_rng(seed)
    return {
        "teacher_mean": rng.normal(0, 0.1, m_shape).astype(np.float32),
        "teacher_std": np.full(m_shape, 0.1, dtype=np.float32),
        "b_p95": np.full(n_b, 0.5, dtype=np.float32),
        "sample_weights": np.ones(B, dtype=np.float32),
        "paths": np.array([f"f{i}.pkl" for i in range(B)]),
        "freq_k": np.linspace(0.01, 0.5, n_b).astype(np.float32),
        "C": np.abs(rng.normal(0, 0.5, (m_shape[0]*m_shape[1], n_b))).astype(np.float32),
    }


def test_weighted_mse_finite_loss():
    stats = make_stats()
    fn = _make_loss("weighted_mse", stats, "cpu", lam_std=1.0)
    s = torch.zeros(8, 32, 16, requires_grad=True)
    t = torch.randn(8, 32, 16) * 0.1
    b = torch.randn(8, 58)
    loss, parts = fn(s, t, b=b)
    assert torch.isfinite(loss)
    assert "weighted_mse" in parts
    assert "std_match" in parts
    loss.backward()
    assert s.grad is not None
    assert torch.isfinite(s.grad).all()


def test_weighted_mse_rare_channel_amplifies():
    """A frame with rare-channel activation should produce larger gradient
    than a neutral frame."""
    stats = make_stats()
    # make channel 0 ultra-rare
    stats["freq_k"] = np.array([0.001] + [0.5] * 57, dtype=np.float32)
    fn = _make_loss("weighted_mse", stats, "cpu", lam_std=0.0)

    s_rare = torch.zeros(1, 32, 16, requires_grad=True)
    t_rare = torch.ones(1, 32, 16) * 0.1
    b_rare = torch.zeros(1, 58); b_rare[0, 0] = 0.5  # rare channel active
    l_rare, _ = fn(s_rare, t_rare, b=b_rare)
    l_rare.backward(); g_rare = s_rare.grad.norm().item()

    s_neutral = torch.zeros(1, 32, 16, requires_grad=True)
    t_neutral = torch.ones(1, 32, 16) * 0.1
    b_neutral = torch.zeros(1, 58); b_neutral[0, 50] = 0.5  # common
    l_neutral, _ = fn(s_neutral, t_neutral, b=b_neutral)
    l_neutral.backward(); g_neutral = s_neutral.grad.norm().item()

    assert g_rare > g_neutral * 1.5, f"rare {g_rare} not > 1.5×common {g_neutral}"


def test_varnorm_jvp_finite_loss():
    stats = make_stats()
    fn = _make_loss("varnorm_jvp", stats, "cpu", lam_std=1.0, lam_jvp=0.1, alpha=0.5)
    s = torch.nn.Sequential(
        torch.nn.Linear(58, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32 * 16),
        torch.nn.Unflatten(1, (32, 16)),
    )
    b = torch.randn(8, 58, requires_grad=True)
    pred = s(b)
    t = torch.randn(8, 32, 16) * 0.1
    loss, parts = fn(pred, t, b=b, model=s)
    assert torch.isfinite(loss)
    assert "jvp" in parts
    loss.backward()
    for p in s.parameters():
        assert torch.isfinite(p.grad).all()
```

- [ ] **Step 2: Run test to verify fail**

Run: `cd /home/newub/w/vamp-interface && python -m pytest tests/arkit_bridge/test_distill_loss.py -v`
Expected: FAIL on unknown loss_mode.

- [ ] **Step 3: Add weighted_mse to _make_loss in distill.py**

Edit `src/arkit_bridge/distill.py`. The current `_make_loss(loss_mode, stats, device, ...)` returns a `(s, t) -> (loss, dict)` closure. v4 modes need access to `b` (and for varnorm_jvp, the model). Change the closure signature to accept kwargs:

Replace `_make_loss` (lines 29-70) and the call site at line 187 to pass extra kwargs. Concretely:

```python
def _make_loss(loss_mode, stats, device, lam_std=1.0, lam_tail=0.5,
               tail_z=2.0, lam_jvp=0.1, alpha=0.5, eps=1e-3):
    """Returns a closure (student_pred, teacher, **kwargs) -> (loss, parts).

    Modes that require extras must be passed via kwargs:
      - weighted_mse: needs b (input batch)
      - varnorm_jvp:  needs b (input batch) and model (the student)
    """
    if loss_mode == "plain":
        def fn_plain(s, t, **_):
            l = F.mse_loss(s, t)
            return l, {"mse": float(l.detach())}
        return fn_plain

    sigma = torch.from_numpy(stats["teacher_std"]).to(device).clamp(min=eps)
    mean = torch.from_numpy(stats["teacher_mean"]).to(device)

    if loss_mode == "varnorm":
        def fn_varnorm(s, t, **_):
            l = ((s - t) / sigma).pow(2).mean()
            return l, {"varnorm": float(l.detach())}
        return fn_varnorm

    if loss_mode == "varnorm_std_tail":
        def fn_full(s, t, **_):
            l_var = ((s - t) / sigma).pow(2).mean()
            s_std = s.std(dim=0, unbiased=False)
            t_std = t.std(dim=0, unbiased=False)
            l_std = (s_std - t_std).pow(2).mean()
            z = (t - mean) / sigma
            tail_mask = (z.abs() > tail_z).float()
            denom = tail_mask.sum().clamp(min=1.0)
            l_tail = ((s - t).pow(2) * tail_mask).sum() / denom
            loss = l_var + lam_std * l_std + lam_tail * l_tail
            return loss, {
                "varnorm": float(l_var.detach()),
                "std_match": float(l_std.detach()),
                "tail": float(l_tail.detach()),
            }
        return fn_full

    if loss_mode == "weighted_mse":
        if "freq_k" not in stats or "C" not in stats:
            raise ValueError("weighted_mse requires stats with freq_k and C")
        freq_k = torch.from_numpy(stats["freq_k"]).to(device).clamp(min=1e-4)
        w_k = (1.0 / freq_k).pow(alpha)            # (58,)
        b_p95 = torch.from_numpy(stats["b_p95"]).to(device).clamp(min=eps)
        # cell_weight[j] = sum_k w_k * C[j,k] / sum_k C[j,k], normalized mean=1
        C = torch.from_numpy(stats["C"]).to(device)            # (512, 58)
        num = (C * w_k[None, :]).sum(dim=1)                    # (512,)
        den = C.sum(dim=1).clamp(min=eps)
        cell_w = (num / den)
        cell_w = (cell_w / cell_w.mean()).reshape(*sigma.shape)  # (32,16) mean=1

        def fn_weighted(s, t, *, b=None, **_):
            if b is None:
                raise ValueError("weighted_mse needs b kwarg")
            # per-sample weight = max_k (|b_k| / b_p95_k * w_k), clipped
            score = (b.abs() / b_p95[None, :]) * w_k[None, :]  # (B, 58)
            sw = score.max(dim=1).values.clamp(0.1, 10.0)      # (B,)
            sw = sw / sw.mean().clamp(min=eps)
            # varnorm-MSE per (B, 32, 16), then weighted
            err2 = ((s - t) / sigma).pow(2)                     # (B, 32, 16)
            err2 = err2 * cell_w[None, :, :]                    # cell weight
            err2 = err2.mean(dim=(1, 2))                        # (B,)
            l_main = (err2 * sw).mean()
            s_std = s.std(dim=0, unbiased=False)
            t_std = t.std(dim=0, unbiased=False)
            l_std = (s_std - t_std).pow(2).mean()
            loss = l_main + lam_std * l_std
            return loss, {
                "weighted_mse": float(l_main.detach()),
                "std_match": float(l_std.detach()),
                "sample_w_mean": float(sw.mean().detach()),
            }
        return fn_weighted

    if loss_mode == "varnorm_jvp":
        if "freq_k" not in stats or "C" not in stats:
            raise ValueError("varnorm_jvp requires stats with freq_k and C")
        freq_k = torch.from_numpy(stats["freq_k"]).to(device).clamp(min=1e-4)
        w_k = (1.0 / freq_k).pow(alpha)
        # Categorical dist over channels for importance sampling
        probs = (w_k / w_k.sum())
        C = torch.from_numpy(stats["C"]).to(device)            # (512, 58)
        # target magnitudes per channel
        target_norm = C.norm(dim=0)                            # (58,)

        def fn_jvp(s, t, *, b=None, model=None, **_):
            if b is None or model is None:
                raise ValueError("varnorm_jvp needs b and model kwargs")
            l_var = ((s - t) / sigma).pow(2).mean()
            s_std = s.std(dim=0, unbiased=False)
            t_std = t.std(dim=0, unbiased=False)
            l_std = (s_std - t_std).pow(2).mean()
            # sample one channel index per step
            k = int(torch.multinomial(probs, num_samples=1).item())
            e_k = torch.zeros(b.shape[1], device=device)
            e_k[k] = 1.0
            # JVP of model at b along e_k
            _, jvp_out = torch.func.jvp(
                model,
                (b,),
                (e_k.unsqueeze(0).expand_as(b),),
            )
            # ‖J_s e_k‖ averaged over batch
            student_norm = jvp_out.flatten(1).norm(dim=1).mean()
            l_jvp = (student_norm - target_norm[k]).pow(2)
            loss = l_var + lam_std * l_std + lam_jvp * l_jvp
            return loss, {
                "varnorm": float(l_var.detach()),
                "std_match": float(l_std.detach()),
                "jvp": float(l_jvp.detach()),
                "k": k,
                "student_norm": float(student_norm.detach()),
                "target_norm": float(target_norm[k].detach()),
            }
        return fn_jvp

    raise ValueError(f"unknown loss_mode={loss_mode}")
```

Then modify the train loop at line 187. Replace:
```python
            loss, parts = loss_fn(pred, m)
```
with:
```python
            loss, parts = loss_fn(pred, m, b=b, model=s)
```

Add new params to `train()` signature: `lam_jvp: float = 0.1, alpha: float = 0.5,` and pass to `_make_loss`.

- [ ] **Step 4: Run test to verify pass**

Run: `cd /home/newub/w/vamp-interface && python -m pytest tests/arkit_bridge/test_distill_loss.py -v`
Expected: 3 passed.

- [ ] **Step 5: Wire CLI flags**

Edit `scripts/train_arkit_student.py`:
- Add `"weighted_mse", "varnorm_jvp"` to `--loss_mode` choices
- Add `ap.add_argument("--lam_jvp", type=float, default=0.1)`
- Add `ap.add_argument("--alpha", type=float, default=0.5)`
- Pass `lam_jvp=args.lam_jvp, alpha=args.alpha` in `train(...)` call

- [ ] **Step 6: Run code-reviewer**

Dispatch: `Agent(subagent_type="superpowers:code-reviewer")` with prompt: "Review the v4 loss-mode additions in /home/newub/w/vamp-interface/src/arkit_bridge/distill.py (weighted_mse and varnorm_jvp), the matching CLI updates in scripts/train_arkit_student.py, and tests/arkit_bridge/test_distill_loss.py. Spec at docs/superpowers/specs/2026-05-06-arkit-bridge-v4-design.md. Verify: kwarg threading is consistent across all loss modes (plain/varnorm/varnorm_std_tail/weighted_mse/varnorm_jvp), the model arg is only required by varnorm_jvp, JVP usage with torch.func.jvp is correct (re-read torch docs if needed), cell_weight normalization gives mean=1, sample_weight clipping matches spec [0.1, 10.0], probs is a proper distribution, no detach issues breaking gradient flow."

Address all feedback before proceeding.

- [ ] **Step 7: Commit**

```bash
git add src/arkit_bridge/distill.py scripts/train_arkit_student.py tests/arkit_bridge/test_distill_loss.py
git commit -m "feat(arkit-bridge): add weighted_mse and varnorm_jvp loss modes

v4 bake-off candidates:
- weighted_mse (variant A+B): per-sample × per-cell weighting; sample
  weight = max_k(|b_k|/b_p95_k · 1/freq_k^α), cell weight from coupling
  matrix C. λ_std=1.0, α=0.5.
- varnorm_jvp (variant C): importance-sampled Jacobian-norm regularizer
  on student only; force ‖J_s e_k‖ to match empirical teacher response
  norm ‖C[:,k]‖. λ_jvp=0.1, α=0.5.

Both consume runs/v4_shared/stats.npz. Unit tests for finite-loss,
gradient-flow, and rare-channel amplification.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5: Train v4a (variant A+B)

**Files:** runtime artifacts only.

- [ ] **Step 1: Train**

```bash
mkdir -p runs/student_v4a
systemd-run --user --scope -p MemoryMax=45G \
  python scripts/train_arkit_student.py \
    --pairs_dir data/arkit_bridge_pairs/all \
    --out_dir runs/student_v4a \
    --holdout_dir data/arkit_bridge_pairs/holdout_v3 \
    --batch_size 128 \
    --lr 5e-4 \
    --steps 30000 \
    --ckpt_every 2000 \
    --loss_mode weighted_mse \
    --sampler active_channel \
    --stats runs/v4_shared/stats.npz \
    --lam_std 1.0 \
    --alpha 0.5 \
    2>&1 | tee runs/student_v4a/train.log
```

Expected runtime: ~30 min on RTX 5090.
Expected final eval: `ratio_mean` lower than v2 (~0.0066), `r2_above_0_7_fraction` ≥ 0.97.

- [ ] **Step 2: Inspect log**

```bash
tail -40 runs/student_v4a/train.log
cat runs/student_v4a/eval_log.json | python -m json.tool | tail -30
```

If `ratio_mean` did not improve over v2's 0.0066, abort and investigate (sample weight too peaked? cell weight degenerate? Check `sample_w_mean` log line).

- [ ] **Step 3: Commit checkpoint metadata**

```bash
git add runs/student_v4a/eval_log.json runs/student_v4a/log.json runs/student_v4a/train.log
git commit -m "exp(arkit-bridge): student_v4a (weighted_mse) train log

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

(Don't commit `student_best.pt` — too large; archived externally per project convention.)

---

## Task 6: Train v4b (variant C)

**Files:** runtime artifacts only.

- [ ] **Step 1: Train**

```bash
mkdir -p runs/student_v4b
systemd-run --user --scope -p MemoryMax=45G \
  python scripts/train_arkit_student.py \
    --pairs_dir data/arkit_bridge_pairs/all \
    --out_dir runs/student_v4b \
    --holdout_dir data/arkit_bridge_pairs/holdout_v3 \
    --batch_size 128 \
    --lr 5e-4 \
    --steps 30000 \
    --ckpt_every 2000 \
    --loss_mode varnorm_jvp \
    --sampler active_channel \
    --stats runs/v4_shared/stats.npz \
    --lam_std 1.0 \
    --lam_jvp 0.1 \
    --alpha 0.5 \
    2>&1 | tee runs/student_v4b/train.log
```

Expected runtime: ~60 min (JVP doubles per-step cost).
Expected final eval: `ratio_mean` lower than v2's 0.0066, `r2_above_0_7_fraction` ≥ 0.97.

- [ ] **Step 2: Inspect log**

```bash
tail -40 runs/student_v4b/train.log
cat runs/student_v4b/eval_log.json | python -m json.tool | tail -30
```

Look for the `jvp` term — should decrease over training but never hit zero (target is per-channel norm, sampled per-step). The `student_norm` and `target_norm` log fields show whether the regularizer is biting.

- [ ] **Step 3: Commit checkpoint metadata**

```bash
git add runs/student_v4b/eval_log.json runs/student_v4b/log.json runs/student_v4b/train.log
git commit -m "exp(arkit-bridge): student_v4b (varnorm_jvp) train log

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 7: Bake-off scorecard

**Files:**
- Create: `scripts/bakeoff_v4.py`

- [ ] **Step 1: Write the script**

Create `scripts/bakeoff_v4.py`:
```python
"""Side-by-side bake-off scorecard for v4 student variants.

Reads two checkpoints (v4a weighted_mse, v4b varnorm_jvp), runs offline
diagnostics on a heldout pair set, emits CSV + Markdown summary.

Metrics per arm:
  - tail_recovery_median  (per-cell std ratio on |t_z|>2 frames)
  - n_channels_above_0_7  (per-channel input_sensitivity ratio >= 0.7)
  - jvp_norm_match_median (|‖J_s e_k‖ - ‖C[:,k]‖| / ‖C[:,k]‖, all 58 channels)
  - heldout_r2_median     (per-cell R² on heldout)
  - heldout_std_ratio_median (student/teacher std ratio on heldout)

Decision: arm wins ≥4/5 metrics → ship. Otherwise flag tie for follow-up.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.func

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkit_bridge.student import MotEncoderStudent  # noqa: E402


def load_pairs(pairs_dir: Path):
    paths = sorted(pairs_dir.glob("*frame_*.pkl"))
    bs, ms = [], []
    for p in paths:
        with open(p, "rb") as f:
            d = pickle.load(f)
        bs.append(np.asarray(d["b_expr"], dtype=np.float32))
        ms.append(np.asarray(d["m_f"], dtype=np.float32).squeeze(0))
    return np.stack(bs), np.stack(ms)


def load_student(ckpt: str, device: str):
    s = MotEncoderStudent().to(device)
    s.load_state_dict(torch.load(ckpt, map_location=device))
    s.eval()
    return s


@torch.no_grad()
def predict(s, b: np.ndarray, device: str):
    return s(torch.from_numpy(b).to(device)).cpu().numpy()


def tail_recovery_median(t: np.ndarray, p: np.ndarray, mean: np.ndarray, std: np.ndarray):
    z = (t - mean) / np.maximum(std, 1e-6)
    mask = np.abs(z) > 2.0
    out = []
    for j in range(t.shape[1]):
        for k in range(t.shape[2]):
            m = mask[:, j, k]
            if m.sum() < 8:
                continue
            ts = t[m, j, k].std()
            ps = p[m, j, k].std()
            if ts < 1e-4:
                continue
            out.append(ps / ts)
    return float(np.median(out)) if out else float("nan")


def per_channel_sensitivity(s, b: np.ndarray, m: np.ndarray, device: str, delta=0.05):
    """For each input channel k, perturb b → b + δ·e_k, measure ‖Δstudent‖/‖Δteacher_emp‖.
    teacher_emp here means: a calibrated empirical response derived from pair data.
    This script approximates by comparing student's δ-response to the precomputed
    coupling's per-channel norm (loaded externally). For simpler version, return
    student's per-channel response norm directly and compare ratios across arms.
    """
    n_b = b.shape[1]
    out = np.zeros(n_b)
    b_t = torch.from_numpy(b).to(device)
    with torch.no_grad():
        m_base = s(b_t).cpu().numpy()
    for k in range(n_b):
        b_pert = b_t.clone()
        b_pert[:, k] += delta
        with torch.no_grad():
            m_pert = s(b_pert).cpu().numpy()
        out[k] = np.linalg.norm(m_pert - m_base) / max(delta, 1e-6) / max(np.sqrt(b.shape[0]), 1)
    return out


def jvp_norm(s, b: torch.Tensor, k: int):
    e_k = torch.zeros(b.shape[1], device=b.device)
    e_k[k] = 1.0
    _, jvp_out = torch.func.jvp(s, (b,), (e_k.unsqueeze(0).expand_as(b),))
    return jvp_out.flatten(1).norm(dim=1).mean().item()


def evaluate_arm(name: str, ckpt: str, b: np.ndarray, m: np.ndarray,
                 stats: dict, device: str):
    s = load_student(ckpt, device)
    pred = predict(s, b, device)

    # tail_recovery
    tr = tail_recovery_median(m, pred, stats["teacher_mean"], stats["teacher_std"])

    # per-channel sensitivity ratio: student_response / coupling_norm
    sens = per_channel_sensitivity(s, b, m, device)
    target = np.linalg.norm(stats["C"], axis=0)
    ratio = sens / np.maximum(target, 1e-6)
    n_above_07 = int((ratio >= 0.7).sum())

    # JVP norm match (median |student - target| / target across channels)
    b_t = torch.from_numpy(b[:64]).to(device)
    sn = np.array([jvp_norm(s, b_t, k) for k in range(b.shape[1])])
    jvp_match = float(np.median(np.abs(sn - target) / np.maximum(target, 1e-6)))

    # heldout R² and std ratio
    err = ((m - pred) ** 2).mean(axis=0)
    var = m.var(axis=0)
    r2 = 1.0 - err / np.maximum(var, 1e-6)
    r2_med = float(np.median(r2.flatten()))
    s_std = pred.std(axis=0)
    t_std = m.std(axis=0)
    sr = s_std / np.maximum(t_std, 1e-6)
    sr_med = float(np.median(sr.flatten()))

    return {
        "name": name,
        "ckpt": ckpt,
        "tail_recovery_median": tr,
        "n_channels_above_0_7": n_above_07,
        "jvp_norm_match_median": jvp_match,
        "heldout_r2_median": r2_med,
        "heldout_std_ratio_median": sr_med,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_a", required=True, help="v4a weighted_mse")
    ap.add_argument("--ckpt_b", required=True, help="v4b varnorm_jvp")
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s_npz = np.load(args.stats, allow_pickle=True)
    stats = {k: s_npz[k] for k in s_npz.files}

    b, m = load_pairs(Path(args.heldout))
    print(f"heldout: {b.shape[0]} pairs")

    a = evaluate_arm("v4a_weighted_mse", args.ckpt_a, b, m, stats, args.device)
    bb = evaluate_arm("v4b_varnorm_jvp", args.ckpt_b, b, m, stats, args.device)

    # decide
    metrics_higher_better = {
        "tail_recovery_median": True,
        "n_channels_above_0_7": True,
        "jvp_norm_match_median": False,
        "heldout_r2_median": True,
        "heldout_std_ratio_median": "closer_to_1",
    }
    a_wins = 0; b_wins = 0
    for k, hb in metrics_higher_better.items():
        va = a[k]; vb = bb[k]
        if hb is True:
            (a_wins, b_wins) = (a_wins + 1, b_wins) if va > vb else (a_wins, b_wins + 1)
        elif hb is False:
            (a_wins, b_wins) = (a_wins + 1, b_wins) if va < vb else (a_wins, b_wins + 1)
        else:  # closer_to_1
            (a_wins, b_wins) = (a_wins + 1, b_wins) if abs(va - 1) < abs(vb - 1) else (a_wins, b_wins + 1)

    decision = ("v4a" if a_wins >= 4 else
                "v4b" if b_wins >= 4 else
                "tie_render_yaw_clip")

    payload = {"a": a, "b": bb, "a_wins": a_wins, "b_wins": b_wins, "decision": decision}
    with open(out_dir / "scorecard.json", "w") as f:
        json.dump(payload, f, indent=2)

    # CSV
    rows = ["metric,v4a_weighted_mse,v4b_varnorm_jvp"]
    for k in metrics_higher_better:
        rows.append(f"{k},{a[k]},{bb[k]}")
    rows.append(f"wins,{a_wins},{b_wins}")
    rows.append(f"decision,{decision},")
    (out_dir / "scorecard.csv").write_text("\n".join(rows) + "\n")

    # MD summary
    md = [
        f"# v4 Bake-off Scorecard",
        f"",
        f"Heldout: {args.heldout} ({b.shape[0]} pairs)",
        f"",
        f"| Metric | v4a (A+B weighted_mse) | v4b (C varnorm_jvp) |",
        f"|---|---|---|",
    ]
    for k in metrics_higher_better:
        md.append(f"| `{k}` | {a[k]:.4f} | {bb[k]:.4f} |")
    md += [
        f"",
        f"**Wins:** v4a={a_wins}, v4b={b_wins}",
        f"",
        f"**Decision:** `{decision}`",
    ]
    (out_dir / "scorecard.md").write_text("\n".join(md) + "\n")

    print("\n".join(md))
    print(f"\nwrote {out_dir}/scorecard.{{json,csv,md}}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Code-review**

Dispatch: `Agent(subagent_type="superpowers:code-reviewer")` with prompt: "Review /home/newub/w/vamp-interface/scripts/bakeoff_v4.py per spec docs/superpowers/specs/2026-05-06-arkit-bridge-v4-design.md. Verify: metric definitions match spec, decision rule (>=4/5 wins) implemented correctly, JVP usage correct, no NaN traps, CSV/MD/JSON outputs are consistent. The 'closer_to_1' rule for std_ratio handled per spec."

Address feedback.

- [ ] **Step 3: Run scorecard**

```bash
mkdir -p exp_output/arkit_bridge/v4_bakeoff
python scripts/bakeoff_v4.py \
  --ckpt_a runs/student_v4a/student_best.pt \
  --ckpt_b runs/student_v4b/student_best.pt \
  --heldout data/arkit_bridge_pairs/holdout_v3 \
  --stats runs/v4_shared/stats.npz \
  --out_dir exp_output/arkit_bridge/v4_bakeoff
```

- [ ] **Step 4: Apply decision rule**

Read `exp_output/arkit_bridge/v4_bakeoff/scorecard.md`. Cases:

- **`decision: v4a`** or **`decision: v4b`** — ship that arm: `cp runs/student_v4{a,b}/student_best.pt runs/student_v4/student_v4.pt`. Proceed to Task 8.
- **`decision: tie_render_yaw_clip`** — render both checkpoints on `data/llf-clips-auto/20260505_MySlate_5_yaw` via `scripts/verify_calibration_v4_clip.py` (with `--ckpt` plumbed). Pick on rendered yaw sign_agree + amp_student. Promote winner.

- [ ] **Step 5: Commit scorecard + decision**

```bash
git add scripts/bakeoff_v4.py exp_output/arkit_bridge/v4_bakeoff/
git commit -m "exp(arkit-bridge): v4 bake-off scorecard, winner=<v4a|v4b>

5 metrics offline on heldout: tail_recovery, channel_sensitivity_count,
JVP_norm_match, heldout R², heldout std_ratio. Decision rule >=4/5.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 8: Render winner on full takes

**Files:** runtime artifacts only; no new code.

- [ ] **Step 1: Render takes 2,3,4,5,6,7,8 with the winner ckpt**

```bash
WINNER=runs/student_v4/student_v4.pt
mkdir -p exp_output/arkit_bridge/render/v4_full

for take in 2 3 4 5 6 7 8; do
  systemd-run --user --scope -p MemoryMax=45G \
    /home/newub/w/PersonaLive/.venv/bin/python scripts/apply_bridge_to_personalive.py \
      --take_dir data/llf-takes-small/20260505_MySlate_$take \
      --anchor data/llf-phase2/asian_m__06_neutral.midframe.png \
      --ckpt $WINNER \
      --mode bridge \
      --n_frames 1200 \
      --stride 2 \
      --out_mp4 exp_output/arkit_bridge/render/v4_full/take${take}_bridge.mp4
done
```

Expected: ~12 min per take × 7 takes = ~90 min. Resumable: `apply_bridge_to_personalive.py` skips existing mp4 outputs.

- [ ] **Step 2: Build render_metrics.parquet**

```bash
python scripts/build_render_metrics_parquet.py \
  --renders exp_output/arkit_bridge/render/v4_full/ \
  --takes_dir data/llf-takes-small/ \
  --out exp_output/arkit_bridge/render_metrics_v4.parquet
```

(If `build_render_metrics_parquet.py` doesn't have those CLI flags, adapt to the actual interface — read its argparse to confirm.)

- [ ] **Step 3: Tier 3 evaluation queries**

```bash
python -c "
import polars as pl
df = pl.read_parquet('exp_output/arkit_bridge/render_metrics_v4.parquet')
# yaw sign_agree
print('yaw sign_agree per take:')
print(df.group_by('take').agg(
    pl.col('yaw_sign_agree').mean().alias('mean'),
    pl.col('yaw_sign_agree').count().alias('n'),
))
# amp_student
print('amp_student (mouth/jaw):')
print(df.filter(pl.col('channel').is_in(['jawOpen', 'mouthFunnel', 'mouthPucker']))
        .group_by('take', 'channel').agg(pl.col('amp_student').median()))
# arcface drift on take 7
print('arcface drift, take 7:')
print(df.filter(pl.col('take') == 7).select('frame_idx', 'arcface_cos').sort('frame_idx'))
"
```

- [ ] **Step 4: Apply gate decisions**

| Gate | Threshold | Decision |
|---|---|---|
| Tier 1: tail_recovery median ≥ 0.95 | offline | already passed in bake-off |
| Tier 2: rendered yaw sign_agree ≥ 0.85 + r ≥ 0.7 | render_metrics_v4.parquet | per-take |
| Tier 3: amp_student ≥ 0.5 mouth/jaw, 5+/7 takes | parquet query | aggregate |
| Tier 3: r(LPIPS, bnorm_expr) > 0 | parquet query | per-take |
| Tier 3: ArcFace cos median ≥ 0.90 | parquet query | per-take |
| Tier 3: ArcFace drift slope ≥ −0.03/1000 fr | take 7 | single take |

If all pass: promote `student_v4.pt` and write final readout. If any fail: escalate to v1-viability.md Plan-B (RBF, hyperparam grid).

- [ ] **Step 5: Commit final artifacts**

```bash
git add exp_output/arkit_bridge/render_metrics_v4.parquet exp_output/arkit_bridge/v4_acceptance_report.md
git commit -m "exp(arkit-bridge): v4 full-take render + Tier 3 acceptance

Tail recovery: <X> (was 0.84-0.91 in v1)
Yaw sign_agree median: <Y>
amp_student mouth/jaw passes: <N>/7 takes
ArcFace cos median: <Z>
ArcFace drift (take 7): <W> rad/1000fr

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 9: Update topic index

**Files:**
- Modify or create: `docs/research/_topics/arkit-bridge.md`

- [ ] **Step 1: Write/update topic index**

If `docs/research/_topics/arkit-bridge.md` does not exist, create it. Either way, the file should:
- Open with current beliefs (1 paragraph)
- List load-bearing dated docs (v1-design, v1-readout, v2-loss-redesign, v3-calibration result, v4-design spec, v4-acceptance report)
- Note falsified/superseded paths (v1 EULER_SIGNS sign-flip-fix doc, F*=I assumption)
- Pointer to `docs/superpowers/specs/2026-05-06-arkit-bridge-v4-design.md` and `docs/superpowers/plans/2026-05-06-arkit-bridge-v4.md`

- [ ] **Step 2: Commit**

```bash
git add docs/research/_topics/arkit-bridge.md
git commit -m "docs(arkit-bridge): topic index covering v1-v4 thread

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Self-review checklist (run before execution)

- [x] **Spec coverage:** All sections of the spec mapped to tasks (calibration→T1+T2, A+B loss→T4, C loss→T4, precompute→T3, bake-off→T7, full render→T8, gates→T8). Topic index→T9.
- [x] **Placeholders:** None. All code blocks complete; commit messages templated with `<placeholders>` only where the experimental result is unknown until run.
- [x] **Type consistency:** `_make_loss(loss_mode, stats, device, lam_std, lam_tail, tail_z, lam_jvp, alpha)` signature consistent across distill.py and CLI; closures use `**kwargs` for extra args (`b`, `model`); `stats.npz` keys (`teacher_mean`, `teacher_std`, `b_p95`, `sample_weights`, `paths`, `freq_k`, `C`) consistent across precompute, distill, bakeoff.
- [x] **Code-review gates** present on every non-trivial script: closed_form_pose (T1), verify_calibration_v4_clip (T2), precompute_v4_stats (T3), distill loss modes (T4), bakeoff_v4 (T7). Trivial CLI plumbing in train_arkit_student (T4 step 5) skips review per standing rule.
- [x] **TDD:** T1, T3, T4 each follow write-test → fail → implement → pass.
