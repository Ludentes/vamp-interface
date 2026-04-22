---
status: live
topic: manifold-geometry
summary: Falsifies the mixture-of-AUs hypothesis — α≈0.45 cliff appears identically in same-phase smile, same-phase jaw, and cross-phase Mona→Joker sweeps. The cliff is a FluxSpace mix_b injection threshold.
---

# Phase-1 falsification — in-phase vs cross-phase α-sweeps

**Date:** 2026-04-22
**Follow-up to:** `2026-04-22-blendshape-bridge-plan.md`
**Data:** `smile_inphase/blendshapes.json` (330), `jaw_inphase/blendshapes.json` (330), `alpha_interp/blendshapes.json` (660).
**Script:** `src/demographic_pc/analyze_inphase_monotonicity.py`

## TL;DR — the hypothesis was wrong

The α≈0.45 cliff is **not** caused by cross-phase prompt-embedding mixture
of AU axes. It appears at the same α-location in every sweep — same-phase
smile, same-phase jaw, and the Mona-Lisa → Joker cross-phase reference.
It is an **injection-schedule / activation-threshold property of FluxSpace
`mix_b`**, not a manifold-geometry property of the endpoint pair.

## Mean trajectories (cross-base, cross-seed)

### smile_inphase — two AU12-dominant smile prompts

| α | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | **0.5** | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|-----|-----|-----|-----|-----|---------|-----|-----|-----|-----|-----|
| mouthSmile | 0.006 | 0.012 | 0.019 | 0.071 | 0.187 | **0.594** | 0.769 | 0.863 | 0.900 | 0.916 | 0.929 |
| jawOpen    | 0.007 | 0.007 | 0.006 | 0.005 | 0.008 | **0.031** | 0.047 | 0.061 | 0.093 | 0.095 | 0.109 |

### jaw_inphase — two AU26-dominant jaw prompts

| α | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | **0.5** | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|-----|-----|-----|-----|-----|---------|-----|-----|-----|-----|-----|
| jawOpen    | 0.007 | 0.008 | 0.009 | 0.012 | 0.015 | **0.456** | 0.634 | 0.695 | 0.765 | 0.768 | 0.768 |
| mouthSmile | 0.006 | 0.004 | 0.005 | 0.004 | 0.004 | **0.034** | 0.044 | 0.048 | 0.066 | 0.098 | 0.104 |

### alpha_interp — Mona Lisa → Joker cross-phase

| α | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | **0.5** | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|-----|-----|-----|-----|-----|---------|-----|-----|-----|-----|-----|
| mouthSmile | 0.834 | 0.844 | 0.857 | 0.865 | 0.859 | 0.916 | 0.927 | 0.923 | 0.920 | 0.916 | 0.913 |
| jawOpen    | 0.020 | 0.024 | 0.035 | 0.044 | 0.064 | **0.314** | 0.467 | 0.501 | 0.524 | 0.538 | 0.540 |

## The one common signature

**Every sweep has a step-function jump between α=0.4 and α=0.5 on its
primary channel.** Absolute jumps:

- `smile_inphase` mouthSmile: **+0.41** (from 0.187 to 0.594).
- `jaw_inphase` jawOpen: **+0.44** (from 0.015 to 0.456).
- `alpha_interp` jawOpen: **+0.25** (from 0.064 to 0.314).

Below α≈0.45 the edit is near-invisible; above, it saturates fast.
This is a FluxSpace `mix_b` injection property, not an endpoint-pair property.

## What this falsifies

The blendshape-bridge plan's Phase-1 prediction was: in-phase sweeps are
monotonic on their primary channel with no cliff. **They are not.** They
have the same cliff at the same α-location.

Consequences for the plan:

1. **The "mixture of AUs" explanation is too clean.** AUs do co-activate
   (AU12 pulls some AU26 and vice versa), but the cross-AU coupling
   amplitude (~10%) is much smaller than the cliff amplitude (~40%).
   Mixture-of-axes is a real but secondary effect.
2. **The original α-interp writeup overstated the smile non-monotonicity.**
   In the cross-phase sweep, mouthSmile goes 0.834 → peak 0.927 → 0.913 —
   a ~1.5% reversal, not the large concave dome implied. The jaw step at
   α=0.45 is the dominant signal; the smile ripple is a small secondary.
3. **α is the wrong knob for intensity control.** Below threshold, nothing.
   Above, saturation. `scale` at fixed α is the right control.

## What this doesn't falsify

- The AU-aligned-direction idea still holds. Same-phase endpoints do
  produce cleaner primary-channel responses than cross-phase (smile |τ|
  = 0.93 vs 0.46), even if they share the injection threshold.
- Cross-axis leakage is still a concern; the blendshape-bridge vocabulary
  is still worth building.
- Fine intensity control via `scale`-sweep at fixed α (Phase 4 refined)
  is still the right experiment.

## Revised plan

Phase 1 is complete, result: hypothesis falsified in its simple form.
Revised Phase 4 in the blendshape-bridge plan:

- **Do not** α-sweep per axis. Use fixed α ≈ 0.6 (just above injection
  threshold) and sweep `scale` ∈ {-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3}.
- Intensity-dial data (`intensity_full`, 504 renders at varying scales)
  is already the right data for this. Rerun Phase 4 analysis on that
  corpus first before generating new renders.

New parallel research track (not in the original plan):

- **Characterise the α-injection threshold.** Is it at 0.45 for all
  FluxSpace edits, or does it move with `start_percent` / `scale` /
  DiT-block subset? Diagnostic sweep: small grid of
  (start_percent, scale) pairs at α ∈ {0.4, 0.5} only, measure the
  sharpness of the jump. Cheap; ~30 renders.

## Pyright housekeeping

- `analyze_inphase_monotonicity.py` has two minor Pyright issues:
  `kendalltau(...).statistic` flagged (works at runtime; suppress or
  alias); `_base`/`_seed` unused (intentional placeholders). Fix on
  next touch per the two-strikes rule.

## Artefacts

- `output/demographic_pc/fluxspace_metrics/analysis/inphase_monotonicity.json`
- Plot TODO: per-base α-trajectories, side-by-side.
