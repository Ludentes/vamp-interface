---
status: live
topic: demographic-pc-pipeline
---

# v6 Lion: optimizer-dynamics hypothesis falsified

v6 ran 800 steps to completion: Lion at lr=3e-5 with cosine→0 schedule,
α=1·η=4 (v4 known-good values), v2 dataset. **No glasses engagement on
any sample at any checkpoint, +1.5 strength.** Bundle drift visible
(earrings, makeup) on females, identical to v4-step-400 stereotype
shift.

## What we tested

v5 broke at high per-step LoRA pressure (α=16·η=2.5). The
basin-escape hypothesis said: AdamW with high pressure committed
early to the lazy-fixed-point basin in flat regions of the loss
landscape; Lion's sign-based updates would give uniform per-dim
pressure and escape. v6 was the test.

## Result

| Step | Glasses on +1.5 (any demo) | Bundle visible |
|------|---|---|
| 200  | 0/3 | minor |
| 400  | 0/3 | mild  |
| 600  | 0/3 | mild  |
| 750  | 0/3 | mild (earrings, makeup on female) |

Loss decayed normally (~7e-3 to 1e-2 range). Optimizer behaved as
expected. The training did not diverge or collapse — it just
accomplished nothing visible at the slider direction.

## Why Lion didn't help

Cumulative LR-step pressure analysis:

| Version | LR profile      | Steps | ∫ lr dt |
|---------|-----------------|-------|---------|
| v4      | 1.25e-4 const   | 1500  | 0.188   |
| v4 step 600 (first engage) | const | 600 | 0.075 |
| v5      | 1.25e-4 cos→0   | 450 (killed) | ~0.04 |
| v6      | 3e-5 cos→0      | 800   | ~0.015  |

**v6 had ~12× less cumulative gradient than v4 by step 800.** Lion's
basin-escape mechanic operates per-step, but the integral of gradient
mass still scales with `lr × steps`. Lion at 1/4 the LR plus cosine
decay has nowhere near v4's gradient budget.

Falsifies the basin-escape hypothesis as the dominant cause of v5/v6
under-engagement. Reframes the problem as **gradient budget +
loss-formulation**, not optimizer dynamics.

## Cross-reference: math doc

The local/global asymmetry analysis (see
`docs/research/2026-04-27-concept-slider-local-vs-global-math.md`)
predicts that any sub-budget run will fail the same way — global PCs
exhaust loss-mass before local features get gradient. Optimizer choice
doesn't help if the gradient budget is too small to reach Phase 2
(global PC saturation → local emergence).

## Verdict

- **v5 (α=16, AdamW, cos→0)**: high pressure, low budget. No engagement.
- **v6 (α=1, Lion, cos→0)**: low pressure, even lower budget. No engagement.
- **v4 (α=1, AdamW, const)**: low pressure, high budget. Engaged at step 600. **Known-good baseline.**

## What to take forward

1. **Optimizer dynamics aren't the bottleneck.** Stop tweaking
   optimizer/scheduler combos until budget + loss formulation are
   solved.
2. **The cosine→0 schedule starves training.** For these kinds of
   small-α LoRAs, constant LR + manual stop is the right shape.
3. **The structural fix is in the loss, not the optimizer.** Spatial
   mask (anisotropic Gaussian over eye region) shifts loss-mass to
   local features; combined with constant LR + adequate budget, this
   should let local engagement happen while reducing bundle.

## Files

- `output/ai_toolkit_runs/glasses_slider_v6/glasses_slider_v6_*.safetensors`
  (16 checkpoints kept; final = `glasses_slider_v6.safetensors`)
- `output/ai_toolkit_runs/glasses_slider_v6.log`
- Sample renders at every 50 steps in `samples/`

## Next: v7

v4-shape optimizer (adamw8bit, lr=1.25e-4 **constant**, α=1, η=4) +
v2 dataset + spatial Gaussian mask on slider MSE terms (`eye_mask_peak=5.0`).
Manual stop when glasses appear. See `config/glasses_slider_v7.yaml`.
