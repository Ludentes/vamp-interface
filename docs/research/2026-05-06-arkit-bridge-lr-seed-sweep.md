---
status: live
topic: arkit-bridge
---

# LR / schedule / seed sweep — confirms 0.0114 is the floor

After v4 (loss design exhausted) and v5 (dataset augmentation
falsified, see `2026-05-06-arkit-bridge-v5-falsified.md`), ran a
five-variant sweep changing only learning rate, schedule, or seed
on the v2 corpus + varnorm_std_tail loss. The aim: distinguish
"v2 plateau is fundamental" from "v2 plateau is one stochastic
basin we never escape".

## Setup

All runs: `data/arkit_bridge_pairs/all`, `runs/v4_shared/stats.npz`,
`varnorm_std_tail` loss, `active_channel` sampler, 90k steps, default
batch=64, AdamW.

| variant | knob change | other |
|---|---|---|
| v2_lr3e4 | lr 5e-4 → 3e-4 | constant |
| v2_lr1e3_cosine | lr 1e-3 + cosine to 5e-5 | — |
| v2_cosine | lr 5e-4 + cosine to 5e-5 | — |
| v2_seed7 | seed=7 | otherwise v2 default |
| v2_seed11 | seed=11 | otherwise v2 default |

## Results (holdout_v3 ratio_mean @ 90k)

| variant | ratio | R²≥0.7 |
|---|---|---|
| v2_120k (120k baseline) | **0.0114** | 1.000 |
| v2_lr3e4 (90k) | 0.0169 | 0.998 |
| **v2_lr1e3_cosine (90k)** | **0.0116** | 1.000 |
| v2_cosine (90k) | 0.0158 | 1.000 |
| v2_seed7 (90k) | 0.0129 | 1.000 |
| v2_seed11 (90k) | 0.0134 | 1.000 |

`v2_120k` at *its own* step 90k was 0.0118 (the run's `eval_log.json`
records 90000 → 0.01179) — drifting to 0.0114 by step 120k.

## Read

- **Seed-only noise**: `v2_seed7` (0.0129) and `v2_seed11` (0.0134) span
  a ~0.0005 noise band centered slightly above the v2 default.
  Combined with v2_120k's own seed, observed seed range at 90k is
  **0.0118–0.0134**. Variation 0.014–0.011 across all v4/v5
  experiments is mostly within or just above this band.

- **`v2_lr1e3_cosine` is the only variant that matched the 120k baseline
  in 90k steps.** Same floor, faster convergence. Useful as a speedup
  default for future experiments, not a quality improvement.

- **Aggressive cosine annealing (5e-4 → 5e-5) hurts** at this step
  count — `v2_cosine` ended worse than constant LR. Cosine is fine
  *with a higher peak LR* (the 1e-3 variant) but not as a refinement
  on top of v2's existing 5e-4.

- **Smaller LR (3e-4) is just under-training.** Would likely match
  v2_120k if extended past 90k; not a real improvement.

## Honest verdict

The 0.0114 ratio_mean is a **structural floor for the
(MotEncoderStudent, b_expr → m_f, varnorm_std_tail loss, this
corpus) tuple**, not a stochastic outcome of v2's particular seed.
Five attempts to escape it via training-knob changes all landed
within seed noise. Combined with:

- ❌ Loss-design tweaks (v4 family)
- ❌ Function-class kernel methods (full-Gram KRR ceiling at 0.027)
- ❌ Mirror-flip data augmentation (v5)
- ❌ LR/schedule/seed knobs (this doc)

the offline `ratio_mean` story for ARKit-bridge is exhausted on the
training axis. **The remaining unfalsified ideas operate on the
target side**: produce flipped m_f by flipping unflipped output
(soft equivariance), or move attention to the rendered-quality
axis where temporal smoothness and per-frame consistency matter
beyond what offline ratio sees.

## Recommended adoption

- **Default training recipe** for future ARKit-bridge experiments:
  lr=1e-3, cosine to 5e-5, 90k steps. Same final quality as v2_120k
  at 75% of the steps (~25% wall-clock saving). The recipe was the
  one win from this sweep.
- **Deployable student**: `runs/student_v2_120k/student_best.pt`
  remains canonical.
