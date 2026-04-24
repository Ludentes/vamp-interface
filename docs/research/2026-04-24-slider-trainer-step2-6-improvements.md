---
status: live
topic: demographic-pc-pipeline
---

# Step 2.6 — Improvements over stock Concept-Sliders recipe

What we can add given our specific corpus, metrics, and goals. The stock
Flux Concept-Sliders recipe was built for cheap text-only supervision on
SDXL. We are **data-richer** (18 image pairs + intermediate α values per
axis, 6 bases × 3 seeds) and **goal-different** (composable cross-base
sliders with measured quality). Several upgrades exploit that gap.

## What the stock recipe uses vs what we have

| Asset | Stock recipe | Us |
|---|---|---|
| Supervision | text prompts (pos/neg) | image pairs (α=0, α=1) |
| Intermediate samples | none | α ∈ {0.25, 0.5, 0.75} |
| Per-image annotations | none | MediaPipe 52-d ARKit blendshapes, SigLIP-2 margins, ArcFace IR101 |
| Cross-demographic | one base prompt | 6 bases × 3 seeds, per-base response quantified |
| Cross-axis | single slider | cached δ from FluxSpaceEditPair on ~15 axes |
| Goals | "train a slider" | composable, cross-base consistent, magnitude-linear |

## Idea triage

### Tier 1 — ship in v1 (cheap + aligned with biggest weaknesses)

**Multi-α supervision.** Replace endpoint-only loss with random-α
sampling over {0.25, 0.5, 0.75, 1.0}, scaling the LoRA multiplier to
`α_train` inside the forward, targeting `v = z_α_train − z_before`.
Teaches **monotonic scale linearity** — the property that makes a
slider a slider. Cost: a few lines.

**Timestep sampling bias.** `FluxSpaceEditPair` uses
`start_percent=0.15`; the edit signal lives mostly in mid-trajectory
timesteps. Uniform `t ~ U[0,1]` wastes gradient budget on
near-clean/near-noise steps. Bias sampling logit-normal with peak
~t=0.3. One-line scheduler change. Expected: faster convergence.

### Tier 2 — add in v1.1 (once v1 works end-to-end)

**Identity anchor loss.** Every N steps, run base prompt with LoRA
scale=0 and without LoRA, enforce `MSE(v̂_with, v̂_without) ≈ 0`.
Guarantees the slider off-state stays stock Flux even as non-linear
softmax drift accumulates in A/B. Cost: one extra forward / N steps.

**Warm-start LoRA from cached δ.** We have thousands of
`FluxSpaceEditPair` attention caches. Per-module SVD truncation to rank
16 → non-trivial `(A_init, B_init)`. Training becomes refinement, not
discovery. Potentially 3–5× faster convergence + better data-efficiency
for 18-pair axes. **Deferred because:** cache experiments have burned
us before (2026-04-23 cached-δ replay falsification), and the 315 GB
archive adds engineering surface. Keep as explicit v1.1 experiment,
run on one axis as ablation vs pure v1.

### Tier 3 — add in v2 (composability generation)

**Cross-axis orthogonality regulariser.** If we train N sliders and
they accidentally share LoRA subspace, "smile + surprise" composition
is nonlinear. Train axes jointly with `Σ_{i≠j} ||A_i · A_jᵀ||_F`
penalty on down-projections. Composability by construction — directly
addresses what `FluxSpaceEditPairMulti` has been fighting for. Cost:
O(N²) terms, memory for N×LoRA simultaneously.

### Tier 4 — eval-only, never training loss

**Auxiliary metric losses (blendshape / SigLIP / ArcFace).** Tempting
because all scorers work. But each requires VAE decode + scorer
forward → 10× slowdown. Gradients through MediaPipe are noisy. Use as
**eval signal with early-stopping on blendshape-Δ plateau**, not as
backprop target.

### Tier 5 — skip

- Scale-conditioned LoRA architecture (unnecessary — scalar multiplier
  covers it, tier-1 item #1)
- Horizontal flip augmentation (breaks `gaze_horizontal`)
- Higher-than-512 training (face is resolved enough)
- DoRA in v1 (experimental on Flux, 1.3–1.5× slower — step 2 already
  deferred to v2)

### Base-dependent axes (e.g. asian_m on brow_furrow_v2)

We know certain axis-base pairs fail (e.g. `brow_furrow_v2` asian_m
Δ ≈ 0). Two paths:
- **Drop:** filter these pairs out of training, mark axis as "5/6 base
  coverage". Safe, sacrifices coverage.
- **Upweight:** risks overfitting base features to axis.

**v1 choice:** drop. Per-axis data-loader filter. Note coverage in the
trained-slider card.

## v1 decision

```
v1 recipe = stock Concept-Sliders Flux + Tier-1 additions
          = LoRA rank=16, α=1, xattn, bf16
          + multi-α supervision (random α_train ∈ {0.25, 0.5, 0.75, 1})
          + logit-normal timestep bias (peak ~0.3)
          + drop-mode on known-failing base-axis pairs
          + MediaPipe/SigLIP/ArcFace as eval-only
```

Training: 1000 steps, LR 2e-3, 8-bit Adam, grad ckpt, 4 forwards/step
→ 1 forward/step (thanks to image supervision). First axis:
`eye_squint`. Target: 60–90 min on 5090.

## v1.1 queued experiment (after v1 validated)

On `eye_squint` only: SVD-warm-start LoRA from cached `FluxSpaceEditPair`
attention deltas. Compare convergence + final quality vs v1. If wins,
adopt for full axis set. If loses or fails (cache experiments have been
fraught), abandon and keep v1 recipe.

## v2 plan (full slider dictionary)

Joint training of 10 axes with cross-axis orthogonality regulariser.
Composability the target, not just per-axis quality.

## Next

Step 3: validation protocol — hold-out splits, primary/guardrail
metrics, pass/fail thresholds per axis.
