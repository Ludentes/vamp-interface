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

## Updated 2026-05-05 evening — magnitude-focused v2

Full-take diagnostic across takes 2–8 confirmed two distinct problems
the v1 loss is blind to:

- **Output magnitude collapse.** Per-category amp_render/amp_driving:
  jaw 0.01–0.17, mouth 0.03–0.10, cheek 0.02–0.20 across most takes.
  Output `m_f` sits in a tight cluster around the per-cell mean.
- **Input coverage gaps.** The 16 522-pair corpus is neutral-skewed —
  most frames have jawOpen<0.05, cheek*≈0, mouthSmile<0.1. The student
  rarely sees "this channel is active" examples and never learns to
  emit the corresponding `m_f` displacement.

Varnorm-MSE (option 1 below) equalises *gradient* across cells but does
not directly penalise the deflated-mean solution. Tail-mining (old
option 2) helps but only on the output side. We need terms that target
magnitude and input coverage explicitly.

### 1. Variance-normalised MSE (foundation)

```python
sigma = teacher_std.clamp(min=1e-3)            # (32, 16)
loss_varnorm = ((student - teacher) / sigma).pow(2).mean()
```

Equalises every cell's loss contribution regardless of magnitude. Cells
with small teacher variance (~0.05) currently dominate by sheer count
(many neutral-cluster cells); normalising lets the high-variance
expression cells (~0.18) actually drive gradients. Compute
`teacher_std` once on the full train set; save next to the ckpt.

### 2. Variance-matching auxiliary (direct fix for output magnitude)

```python
# per-cell std over the batch — pushes student to *want* full variance
loss_std = (student.std(0) - teacher.std(0)).pow(2).mean()
```

Why this is needed even with varnorm: varnorm penalises pointwise
residuals scaled by σ, but a student predicting the per-cell mean has
zero gradient pressure to spread out under a pointwise loss. The std-
match term penalises any solution whose output variance is below the
teacher's, regardless of point-by-point fit.

Practical: needs batch ≥ 64 for a stable per-cell std estimate; we're
already at 128 so fine. Start λ_std=1.0; tune so train-time `loss_std`
is ~0.5× `loss_varnorm` at convergence.

### 3. Tail-mining auxiliary (upweights rare *output* events)

```python
z = (teacher - teacher_mean) / sigma
tail = (z.abs() > 2.0).float()
loss_tail = ((student - teacher) ** 2 * tail).sum() / tail.sum().clamp(min=1)
```

Same as before. Likely smaller marginal lift once #2 is in place but
cheap to keep. λ_tail=0.5.

### 4. Active-channel weighted sampling (input-side coverage)

The cheapest fix for the 61-input coverage problem is dataset-side, not
loss-side. Precompute per-frame the most-active normalised input
channel:

```python
# during pair extraction, alongside b_61 and m_f:
b_p95 = np.percentile(np.abs(b_corpus), 95, axis=0)   # (61,)
sample_weight = np.max(np.abs(b) / np.maximum(b_p95, 1e-3), axis=1)
```

Use as `WeightedRandomSampler` weights. Frames where any ARKit channel
sits in its top decile get oversampled; the neutral-cluster majority is
downweighted but still seen. Effective batch composition shifts from
~80% near-neutral to ~50/50 active/neutral.

Implementation: 5 lines in dataset construction + swap `DataLoader(...)`
to use `WeightedRandomSampler`. Reproducible; no new hyperparameter
beyond what's implicit in the percentile choice.

### 5. Per-input-channel sensitivity audit (diagnostic, not loss)

For each ARKit input dim i, measure how much `m_f` moves under a small
perturbation along that axis, ratio'd against the teacher:

```python
delta = 0.5 * b_p95[i]
ratio_i = (
    (student(b + delta * e_i) - student(b)).norm(dim=(-2,-1)) /
    (teacher(b + delta * e_i) - teacher(b)).norm(dim=(-2,-1)).clamp(min=1e-6)
).mean()
```

Channels where ratio_i < 0.5 are inputs the student is "deaf" to —
exactly the channels we expect to render flat in the output. Run on
~200 holdout samples (~1 min). Reports a 61-element vector; gates v2
acceptance per-channel rather than only per-output-cell.

### 6. Combined v2 loss

```python
loss = loss_varnorm + λ_std * loss_std + λ_tail * loss_tail
# defaults: λ_std=1.0, λ_tail=0.5
```

Plus dataset uses active-channel `WeightedRandomSampler`. Plus eval
adds the per-input sensitivity audit alongside the existing per-cell
attenuation diagnostic.

### Not changing: Huber/smooth-L1

Less aggressive on outliers than L2 — but in our case the *outliers*
are the samples we want to fit better. Skipped.

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

1. Compute `teacher_std`, `teacher_mean` over train pairs and `b_p95`
   over input channels; save as `runs/student_v1/teacher_stats.npz`. ~30 s.
2. Add `--loss_mode {plain|varnorm|varnorm_std_tail}` to `distill.train()`.
   `varnorm_std_tail` = #1 + #2 + #3.
3. Add `--sampler {uniform|active_channel}` and a `_compute_sample_weights`
   helper using `b_p95`.
4. Train 30 K steps with `--loss_mode varnorm_std_tail
   --sampler active_channel` (same hyperparams as v1 — schedule fine;
   lr=5e-4, batch=128, eval every 2 K).
5. Diagnostics on the new ckpt:
   - `diagnose_mf_attenuation.py` — gate: tail_recovery median ≥ 0.95.
   - new `diagnose_input_sensitivity.py` — gate: per-channel ratio ≥ 0.7
     on at least 45/52 ARKit blendshapes (allow 7 dead channels for
     outliers like tongueOut and rare ARKit dims).
6. Re-render takes 2/3/4/5/6/7/8 (full); gates:
   - `channel_recovery_take{n}.json` median amp_vs_driving ≥ 0.5 in
     mouth and jaw categories on at least 5/7 takes.
   - `r(LPIPS, bnorm_expr)` strictly positive on take 3.
   - ArcFace cos on take 2 ≥ 0.90.
7. If gates pass, replace `student_best.pt` with the v2 ckpt and proceed.
   If they fail, escalate to architectural alternatives (RBF / SVR
   diagnostic) listed in the viability doc's Tier-1 escalation section.

Estimated total wall time: ~10 min training + ~10 min eval/render
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
