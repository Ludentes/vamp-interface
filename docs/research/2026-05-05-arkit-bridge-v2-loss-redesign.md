---
status: live
topic: neural-deformation-control
---

# ARKit→PersonaLive bridge v2 — loss & training redesign (2026-05-05)

## What v1 got wrong

v1's distill objective was per-frame `MSE(student(b_expr), m_f_teacher)`
on a 16 522-pair corpus. It cleared the Tier-1 gates by a wide margin
(ratio_mean=0.0066, R²≥0.7=0.98) but the renders confirmed the textbook
L2-regression failure:

- `std(student) / std(teacher)` per cell: median 0.91, p10 0.87 — uniform
  9–13 % magnitude shrinkage.
- Tail recovery `std(student[|t_z|>2]) / std(teacher[|t_z|>2])`: median
  0.84, p10 0.77 — the cells that *carry* expression are 16–23 %
  attenuated.
- Output: take 3 `r(LPIPS, bnorm_expr) = -0.64` — expression amplitude
  inversely correlated with perceptual distance from anchor. Diffusion
  UNet snaps back to anchor when the student gives it a deflated m_f.

Per-cell R² is **scale-tolerant** when student and teacher shrink
proportionally; that's what masked this. R² ≥ 0.7 across 98 % of cells
co-exists with 16 % tail attenuation.

## Three changes for v2

Ranked by expected effect / implementation cost:

### 1. Variance-normalised MSE (cheapest, biggest expected lift)

```python
# Compute per-cell teacher std once over the corpus or per-batch.
sigma = teacher_std.clamp(min=1e-3)            # (32, 16)
loss = ((student - teacher) / sigma).pow(2).mean()
```

Equalises every cell's loss contribution regardless of magnitude. Cells
with small teacher variance (~0.05) currently dominate by sheer count
(many neutral-cluster cells); normalising lets the high-variance
expression cells (~0.18) actually drive gradients.

Implementation: ~15 lines in `distill.py`. One pre-pass to compute
`teacher_std` on the train set, save it next to the ckpt, use it as a
fixed weight (or recompute per-batch with EMA — but a fixed value
estimated once on the full corpus is fine and reproducible).

### 2. Tail-mining auxiliary term (medium cost, targets tail directly)

```python
z = (teacher - teacher_mean) / sigma
tail = (z.abs() > 2.0).float()                 # (B, 32, 16) mask
loss_tail = ((student - teacher) ** 2 * tail).sum() / tail.sum().clamp(min=1)
loss = loss_main + 0.5 * loss_tail
```

Directly upweights residuals on |z|>2 cells. Coefficient 0.5 is a
starting guess; tune so train-time `loss_tail` is ~1× to 2× `loss_main`
at convergence.

Optionally combine with #1 by computing the loss inside variance-
normalised space (gradients already balanced; tail term then
emphasises rare *but high-magnitude* events specifically).

### 3. Huber / smooth-L1 instead of MSE (cheap, smaller effect)

```python
loss = F.smooth_l1_loss(student, teacher, beta=0.1)
```

Less aggressive penalty on outliers than L2 — but in our case the
*outliers* are the very samples we want to fit better, so this is
**unlikely to help on its own**. Listed for completeness; would only
combine sensibly with tail-mining (#2).

## Other levers worth thinking about

- **Stride-1 corpus instead of stride-2**: doubles the corpus to ~33 K
  pairs at the cost of higher temporal redundancy. Probably mostly
  useless for the v2 attenuation problem (the issue is loss objective,
  not data volume), but trivial to test alongside.
- **Per-take re-balancing**: take 4 contributed only 407 pairs of 16 522
  and was the most face_mesh-fallback-heavy at extraction time. We could
  drop take 4 entirely or upweight it; current corpus is fine.
- **Augmentation in 224×224 crop**: the loose-centred crop used at pair
  extraction differs from PersonaLive's tight `crop_face` at inference.
  Probably small effect on tail attenuation but explains some of the
  remaining residual.
- **Multi-anchor training**: corpus is one driver subject (the ARKit
  blendshapes). Attempting cross-actor generalisation is a v3 question;
  v2 should fix the within-distribution attenuation first.

## Plan for the v2 pass

1. Compute `teacher_std`, `teacher_mean` over train pairs once; save as
   `runs/student_v1/teacher_stats.npz`. ~30 s.
2. Add `--loss_mode {plain|varnorm|varnorm_tail}` to `distill.train()`.
3. Train 30 K steps with `varnorm_tail` (same hyperparams as v1 — that
   schedule was clearly fine; lr=5e-4, batch=128, eval every 2 K).
4. Run the same `diagnose_mf_attenuation.py` on the new ckpt — gate:
   tail_recovery median ≥ 0.95.
5. Re-render the 3-take diagnostic set (2/3/8). Gate:
   `r(LPIPS, bnorm_expr)` strictly positive on take 3, ArcFace cos on
   take 2 ≥ 0.90.
6. If both pass, drop the existing `student_best.pt` for v2 in renders
   and proceed to long-take generation; if either fails, escalate to
   the architectural alternatives (RBF / SVR diagnostic) listed in the
   viability doc's Tier-1 escalation section.

Estimated total wall time: ~10 min training + ~5 min eval/render
diagnostic. Cheap.

## What we explicitly are *not* changing

- Architecture (4-layer MLP, 278 K params, zero-init head). It can fit;
  the loss was the wrong target.
- Corpus (16 522 pairs across 7 takes). Adequate.
- Closed-form pose path. Calibration sound (yaw L2 below the
  mediapipe-vs-ARKit noise floor on takes 2/3 in the final diagnostic).
- PersonaLive-side seams (`pose_encoder.{interpolate_kps_online,get_kps}`
  and `motion_encoder.forward`). Working as intended.
- Take-8-style identity-drift artifacts. Out-of-scope; PersonaLive
  pipeline issue, not bridge.

## Cross-references

- Readout: `2026-05-05-arkit-bridge-v1-readout.md`
- Diagnostic script: `scripts/diagnose_mf_attenuation.py`
- Output diagnostic: `scripts/diagnose_render_expression.py`
- Viability framework: `2026-05-05-arkit-bridge-v1-viability.md`
  — see "Architectural escalation options" section for plan-B if v2
  loss tuning doesn't lift tail_recovery to ≥0.95.
