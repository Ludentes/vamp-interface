---
status: live
topic: demographic-pc-pipeline
---

# v7.1 result — cosine refinement falsified the mask-peakedness hypothesis

Branched from `glasses_slider_v7_000001200.safetensors` per
`docs/research/2026-04-27-v7.1-cosine-refinement-plan.md`.

Three deltas vs v7:
- `eye_mask_peak: 5.0 → 2.5` (halve loss-mass concentration)
- `lr: 1.25e-4 constant → 2e-5 cosine_with_min_lr (lr_min=5e-6)`
- `steps: 2500 → 400`

Hold-everything-else at v7 known-good: α=1, r=16, η=4, adamw8bit,
EMA off, xattn scope, anchor=null, v2 dataset, save_every=50.

Final ckpt: `output/ai_toolkit_runs/glasses_slider_v7_1/glasses_slider_v7_1.safetensors`
(tagged `glasses_slider_v7_1_000000400` for the eval pipeline).

## Eval battery results (216 cells: 9 prompts × 3 seeds × 8 strengths)

Strengths: 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75.

| Metric | v7-1200 | v7.1-400 | Δ |
|---|---|---|---|
| Spearman ρ in_dist  | +0.669 | +0.643 | -0.026 |
| Spearman ρ held_out | +0.608 | +0.570 | -0.038 |
| separation in_dist  | +0.060 | +0.057 | flat |
| separation held_out | +0.048 | +0.046 | flat |
| identity mean cos   | 0.712  | 0.719  | flat |
| identity min cos    | 0.000  | 0.017  | flat |
| identity cells <0.4 | 20/189 | 23/189 | +3 (worse) |
| frac glasses in_dist  @s=0.75 | 0.00 | 0.11 | +0.11 |
| frac glasses in_dist  @s=1.00 | 0.33 | 0.22 | -0.11 |
| frac glasses in_dist  @s=1.25 | 0.67 | 0.67 | flat |
| frac glasses in_dist  @s=1.50 | 0.89 | 1.00 | +0.11 |
| frac glasses held_out @s=1.50 | 0.78 | 0.78 | flat |

**Acceptance criteria from v7.1 plan, scored:**

- Dead zone shrinks (frac in_dist @s=0.75 ≥ 0.33): **fail** (got 0.11).
- Identity floor lifts (≤5 cells <0.4, min cos > 0.3): **fail** (23 cells, min 0.017).
- Engagement preserved (frac in_dist @s=1.5 ≥ 0.78): **pass** (got 1.00).
- Bundle does not creep (bundle_pos ≤ 0.10): pass-by-proxy (separation flat at +0.057).
- Soft target (Spearman ρ ≥ 0.85): **fail** (got +0.643).

Two of three primary criteria fail; the soft target fails by a wide
margin.

## Visual inspection (collage)

Collage: `models/sliders/glasses_v7_1/glasses_slider_v7_1_000000400/glasses_v7_1_glasses_slider_v7_1_000000400_eval_collage.png`.

First-engagement strength per prompt, v7.1 vs v7-1200:

| Row                      | v7-1200 | v7.1-400 |
|--------------------------|---------|----------|
| eastasian_f_studio       | s=1.00  | s=1.25   |
| elderly_white_f_natural  | s=1.50  | s=1.50   |
| latino_m_cafe (held-out) | s=1.50  | s=1.50   |
| middleeast_neutral       | s=1.00  | s=1.00   |
| southasian_f_studio      | s=1.00  | s=1.25   |
| young_black_m_bench (held-out) | n/a     | n/a      |
| asian_m_studio           | s=1.00  | s=1.25   |
| black_f_studio           | s=1.00  | s=1.25   |
| european_m_studio        | s=1.00  | s=1.50   |

First-engagement moved **later** by 0.25 on 5/9 prompts. The numerical
"frac in_dist @s=0.75 went 0.00 → 0.11" was a single cell, not a
population shift.

High-strength saturation tightened (s=1.5 cells have visibly thicker
frames). The slider became more *bimodal* (off at low s, fully on at
high s), not more *gradual*.

## Falsification of the mask-peakedness hypothesis

The v7.1 plan posited that the dead zone at s∈[0, 0.75] was caused by
narrow loss support (eye_mask_peak=5.0 concentrating ~5% of latent area
into 3.5× of the loss-mass), producing a high-amplitude spike direction
that only registers above s≈1.0. Halving the peak (5.0 → 2.5) was the
only structural change.

Result: the dead zone did not shrink. It got marginally worse on a
per-prompt basis. The mask-peakedness hypothesis is **falsified**.

This triggers the counter-hypothesis from the same plan (lines 64-67):

> "If v7.1 with broader mask shows *increased* bundle without fixing the
> dead zone, the diagnosis is wrong — the dead zone is a property of the
> LoRA's effective rank / capacity allocation, not the loss support — and
> v7.2 should keep mask=2.5 but raise rank or adjust α."

We did not observe a bundle increase exactly (separation is flat at
+0.057), but the **identity floor degraded slightly** (20 → 23 cells
<0.4) and the **engagement curve sharpened** rather than broadening —
both symptoms of rank-exhaustion. The LoRA spent its capacity on the
global "studio-portrait + thick frames at high s" mode and had no
headroom left for a smooth low-strength ramp.

## Held-out-scene robustness: separate finding

The `ho_young_black_m_bench` prompt produces severe identity collapse
(cos ≈ 0) in **both v7-1200 and v7.1-400**, including at s=0.0. Visual
inspection of the collage shows the rendered face at s=0 is already a
near-silhouette — harsh lighting + the demographic anchor combine to
produce an underexposed render that ArcFace scores at zero. The slider
neither caused nor fixed this; it propagates through every column.

The 23 identity cells <0.4 in v7.1 (and 20 in v7-1200) are concentrated
on this one prompt × seed × strength block. **This is a base-model +
prompt interaction, not a slider failure.** It is independent of any
v7.x iteration and should be filed separately rather than driving
slider hyperparameters.

## What this is worth

- v7-1200 remains the production checkpoint. v7.1 is not better on any
  primary criterion.
- The "broaden the mask" lever does not move the dead zone. Don't
  retry it.
- Future iterations should target rank or supervision, not loss-mass
  shaping, for the dead-zone problem.
- High-strength saturation tightening is a real (small) win that
  transfers — keep `eye_mask_peak=2.5` going forward.

## Cost

- One 400-step training run from a v7-1200 starting point (~30 min).
- One 216-cell eval battery (~50 min CPU+GPU; one OOM mid-run that took
  out a Chromium session before we wrapped the second attempt in a
  `systemd-run --user --scope -p MemoryMax=45G` cap, see
  `feedback_eval_memory_cap.md` in memory).

## Next

`v7.2` (spec: `/home/newub/w/ai-toolkit/config/glasses_slider_v7_2.yaml`)
tests the rank-capacity hypothesis directly: r=16 → r=32, peak=2.5
held forward, otherwise v7's known-good schedule (constant lr=1.25e-4,
1500 steps). Cannot resume from v7-1200 — rank change forks the LoRA
matrices. Fresh run.

If v7.2 also fails to move first-engagement, the bottleneck is the
supervision signal itself (text encoder bundle + Flux's prior), not
LoRA capacity. At that point the path forward is:
- anchor / dataset changes (the open lever from the v7 notes)
- distill differentiable metric losses (ArcFace, SigLIP, MediaPipe latent
  classifiers — see `docs/research/2026-04-27-latent-native-classifier-distillation.md`)
- accept v7-1200 as the ceiling of this approach.
