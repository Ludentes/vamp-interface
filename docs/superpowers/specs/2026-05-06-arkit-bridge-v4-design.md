---
status: live
topic: arkit-bridge
---

# ARKit→PersonaLive Bridge v4 — Design

**Date:** 2026-05-06
**Thesis:** Calibration fix (kp_ref y-mirror) + per-channel-rarity weighted loss, **two arms trained in parallel and compared offline before render**. No architecture changes.

## Background

v1 student (4-layer MLP, 278K params) reached R²≥0.7 on 98% of output cells but suffered 16–23% tail attenuation on rare-active blendshapes. Per-cell R² mask hides per-input-channel sensitivity collapse. Calibration v3 (P_48 angular search) confirmed `EULER_SIGNS=(+1, -1, -1)` is correct and identified a previously-missed kp_ref y-mirror `F* = diag(+1, -1, +1)` worth ~25% of total residual.

## Goals

1. Recover per-input-channel sensitivity to weak/rare blendshapes (jaw, eye-look, brow channels) without losing common-channel fidelity.
2. Bake the kp_ref y-mirror into the closed-form pose path.
3. Decide between two principled approaches **on offline metrics**, not on render aesthetic.

## Out of scope

- Architecture changes to MotEncoderStudent (capacity already sufficient).
- Multi-anchor training, anchor-reinjection HKM, latent-pile blend.
- Bagging (defer to v5; addresses variance, not bias).
- Glasses leak / anchor diversity sweep (post-v4 ship; orthogonal).

## Calibration

`closed_form_pose.py` updates:

- **Keep:** `EULER_SIGNS = (+1, -1, -1)` (confirmed by P_48 search).
- **Add:** `F_KP_REF = diag(+1, -1, +1)` applied as `kp_ref @ F` once (not per-frame). Equivalent to pre-rotating the reference keypoints into the ARKit face frame's mirror.
- The Euler→rotmat path (`Rz·Ry·Rx`)^T composes with `kp_ref` via `compose_kd(kp_ref, R) = (kp_ref @ F) @ R`.

Verification on clips (Phase 0a):
- Re-render `data/llf-clips-auto/20260505_MySlate_5_yaw` and `_pitch`, plus heldout `20260505_MySlate_4_yaw`.
- Mediapipe-extract rotation matrices, compute angular distance to driver.
- Pass: mean ≤ 0.05 rad on each clip.

Parallel: investigate `input_permutation` control failure from calibration v3 (score *improved* under random axis permutation). Hypothesis: residual sign in ARKit Euler→rotmat code that F* compensates for asymmetrically. Non-blocking; if found, may simplify F*.

## Loss arms

Both arms share base `varnorm_MSE + λ_std · std_match` and `WeightedRandomSampler(b_p95)`. They differ in **one term**.

### Variant A+B — per-sample × per-cell weighted MSE

```
L_AB = varnorm_MSE(student, teacher) * sample_weight(b) * cell_weight + λ_std * std_match
```

- `sample_weight(b) = clip(max_k(b_k / freq_k**α), 0.1, 10) / batch_mean`, **α = 0.5**
- `cell_weight[j] = (Σ_k (1/freq_k**α) · C[j,k]) / Σ_k C[j,k]`, normalized so mean=1
- `freq_k = mean(b_k > 0.05)` over training corpus
- `C[j,k] = E_b |∂m_teacher,j / ∂b_k|` over 1024 calibration samples (precomputed once)
- λ_std = 1.0
- Cost: ~30 min training. Zero extra fwd/bwd passes vs MSE baseline.

Targets: each rare-channel-active example gets a big loss multiplier (sample weight); each output cell coupled to rare channels gets a big loss multiplier (cell weight). Together: rare-channel gradient flows are amplified at both ends.

### Variant C — Jacobian-norm regularizer

```
L_C = varnorm_MSE(student, teacher) + λ_std * std_match + λ_jvp * (‖J_s e_k‖ - ‖C[:,k]‖)²
```

- per training step: `k ~ Categorical(w_k normalized)`, `w_k = 1/freq_k**α`, **α = 0.5**
- `J_s e_k` via JVP on **student only** (we don't have a differentiable teacher
  from `b_expr` to `m_f` — teacher operates on RGB; stored pairs `(b_expr, m_f)`
  don't admit a teacher Jacobian along `b_expr`)
- target magnitude `‖C[:, k]‖` = empirical coupling norm of teacher's response
  to channel k, precomputed once via finite-difference probes on the train pair
  corpus
- λ_std = 1.0, **λ_jvp = 0.1**
- Cost: ~60 min training (one extra fwd+bwd per step on student).

Targets: forces student's instantaneous local response magnitude to rare
channels to equal the empirical teacher response magnitude. Constrains
*magnitude only*, not direction — strictly weaker than full Jacobian matching
but well-defined given our data, and addresses the "weak-channel response is
too small" failure directly at the local-derivative level (vs A+B which
operates on stored loss multipliers).

## Bake-off scorecard

After both arms train, score on heldout takes {4, 7} + diagnostic batch. Per-arm:

| Metric | Source | Better |
|---|---|---|
| Per-cell tail_recovery median (|t_z|>2 mask) | `diagnose_mf_attenuation.py` | higher |
| #channels with input_sensitivity ratio ≥ 0.7 (out of 52) | `diagnose_input_sensitivity.py` | higher |
| Per-channel `‖J_s e_k − J_t e_k‖ / ‖J_t e_k‖` median | new (one-shot JVP audit) | lower |
| Heldout per-cell R² median | new | higher |
| Heldout per-cell std ratio (student/teacher) median | new | closer to 1.0 |
| Train wall time (info) | log | — |

Output: `exp_output/arkit_bridge/v4_bakeoff/scorecard.csv` + `bakeoff_summary.md`.

Decision rule:
- Wins ≥ 4/5 → ship that arm as `student_v4.pt`.
- Split (e.g. A+B wins magnitude, C wins channel-uniformity): render both on `data/llf-clips-auto/20260505_MySlate_5_yaw` only (~10 min), pick on rendered-output sign-agree + amp_student. Avoid full-take render until winner picked.

## Acceptance gates (final ship)

After winner picked and rendered on full takes {2,3,4,5,6,7,8}:

| Tier | Metric | Threshold |
|---|---|---|
| 1 | tail_recovery median | ≥ 0.95 |
| 1 | input_sensitivity ratio ≥ 0.7 | on 45+/52 channels |
| 2 | rendered yaw sign_agree (active) | ≥ 0.85 |
| 2 | r(yaw_bridge, yaw_teacher) | ≥ 0.7 |
| 3 | amp_student on jaw/mouth | ≥ 0.5 on 5+/7 takes |
| 3 | r(LPIPS, bnorm_expr) | > 0 |
| 3 | ArcFace cos vs anchor median (1000+ frames) | ≥ 0.90 |
| 3 | ArcFace identity drift (take 7) | slope ≥ −0.03/1000 fr |

If any Tier-1 fails → escalate to Plan-B (RBF / hyperparam grid from `v1-viability.md`).

## Phasing

| Phase | Work | Granularity | Cost |
|---|---|---|---|
| 0a | Apply `F_KP_REF` to closed_form_pose, code-review | 3 clips | 5 min GPU |
| 0b | Investigate input_permutation control (parallel) | offline | 0 |
| 1 | Precompute `stats.npz` (teacher_mean/std, freq_k, b_p95, C[512,58]) | full train | 5 min GPU |
| 2 | Train v4a (A+B) | full train | 30 min GPU |
| 3 | Train v4b (C) | full train | 60 min GPU |
| 4 | Offline bake-off scorecard | heldout 4,7 | 2 min CPU |
| 5 | Render winner full {2,3,4,5,6,7,8}, build render_metrics.parquet | full takes | 90 min GPU |
| 6 | Anchor sweep + glasses diagnostics (post-ship) | — | 30 min GPU |

Total to decision: **~2 hrs GPU + 30 min analysis**.

## Code surface

- `src/arkit_bridge/closed_form_pose.py` — add `F_KP_REF`; minor edit to `compose_kd`.
- `scripts/precompute_v4_stats.py` — **new**, computes `stats.npz`.
- `src/arkit_bridge/distill.py` — add loss modes `weighted_mse` (A+B) and `varnorm_jvp` (C); reuse `varnorm_std` foundation; both consume `stats.npz`.
- `scripts/train_arkit_student.py` — wire `--loss_mode {weighted_mse, varnorm_jvp}` and `--stats stats.npz`.
- `scripts/bakeoff_v4.py` — **new**, computes scorecard from two checkpoints on heldout.

All non-trivial scripts pass through `superpowers:code-reviewer` before being declared done.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| F* application formula wrong (kp_ref @ F vs F @ R @ F^T) | Phase 0a clip verification fails fast → fix and re-verify before Phase 1 |
| α=0.5 too aggressive (over-amplifies one rare channel) | sample_weight clipped to [0.1, 10]; if Tier 1 fails, sweep α ∈ {0.3, 0.5, 1.0} |
| C variant unstable (JVP gradient noise) | start λ_jac=0.1; widely-known stable in distillation literature |
| input_permutation FAIL means F* itself is wrong | Phase 0a verifies on real renders; if mean angular distance doesn't drop, F* is wrong regardless of control |
| Bake-off scorecard too coarse to discriminate | Decision rule has fall-through to mini-render on yaw clip |
