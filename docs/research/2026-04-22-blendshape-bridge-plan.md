---
status: live
topic: manifold-geometry
summary: Five-phase plan to bridge FluxSpace and blendshape ecosystem via per-AU attention-cache directions; hypothesis that α-cliff is mixture-of-AUs; stopping conditions per phase.
---

# Blendshape bridge — experiment plan

**Date:** 2026-04-22
**Goal:** Build a vocabulary of FluxSpace attention-cache directions, each
aligned with an AU-like blendshape axis, and characterise the geometry
(linearity / phase boundaries) of each axis individually. This becomes
the bridge between Flux editing and the blendshape ecosystem (ARKit-52,
FLAME, MetaHuman, FACS).

## Core hypothesis

The α≈0.45 phase cliff and non-monotonic `mouthSmile` in the
Mona-Lisa → Joker sweep are not a property of Flux's attention-cache
geometry. They are a property of the **prompt embedding being a coarse
mixture of multiple blendshape axes** whose dominant coefficient changes
across the sweep. Specifically:

- "Mona Lisa" prompt loads primarily onto AU12 (lip-corner pull) with
  AU26 (jawDrop) suppressed (closed-mouth).
- "Joker" prompt loads primarily onto AU26 (jawDrop) + AU25 (lips
  part) with AU12 weakened (horizontal stretch replaces corner pull).
- `mix_b`-linear interpolation of the two embeddings traces a curve
  through AU-coefficient space, not a line. The cliff at α≈0.45 is
  where AU26 overtakes AU12 as the dominant axis.

**Prediction:** when we interpolate between endpoints that load onto
the *same* AU (e.g. "faint closed smile" → "broader closed smile" →
both AU12-dominant), the sweep is monotonic in every measured
blendshape channel. If this holds, the non-monotonicity is a
mixture-of-axes phenomenon, not a manifold-curvature phenomenon, and
the fix is to edit on AU-aligned directions directly.

**Predictive retrofit** (evidence for the hypothesis before any new
experiment):

- Ridge on demographic labels (Stage 4.5) produced linearity R²=0.90.
  Demographics are effectively single-axis at the level we edit them
  (age, gender are smooth identity-space shifts), so no mixture
  problem.
- FluxSpace pair-averaged glasses edits generalise across six
  demographics with 93% collapse-prediction accuracy. Glasses are an
  added object, essentially a single AU-adjacent axis. Monotonic.
- FluxSpace smile edits fail at extremes. Smile prompts bundle AU12 +
  AU6 + AU25 + AU26. Multi-axis. Non-monotonic.

## Phase 1 — falsification from existing data (cheap, no new renders)

**Goal:** Test the hypothesis using renders already on disk.

**Step 1.1 — score the controls.** Run `score_blendshapes.py` on:

- `crossdemo/smile/smile_inphase/` (330 PNGs) — same-phase sweep
  ("faint closed smile" → "broader closed smile"), AU12 only.
- `crossdemo/smile/jaw_inphase/` (330 PNGs) — same-phase sweep for
  AU26 alone.

The `alpha_interp/` 660-PNG sweep (cross-phase Mona→Joker) is already
scored in `crossdemo/smile/alpha_interp/blendshapes.json`.

**Step 1.2 — monotonicity analysis.** For each (base, seed)
trajectory:

- Compute Kendall's τ per blendshape channel over the α-sweep.
- Define monotonic = |τ| ≥ 0.9 in the same direction across the
  sweep.
- Count strictly-monotonic trajectories per (base, channel).

**Prediction:**
- `smile_inphase`: ≥ 80% of trajectories monotonic in `mouthSmileLeft`
  + `mouthSmileRight`, flat in `jawOpen`.
- `jaw_inphase`: ≥ 80% monotonic in `jawOpen`, flat in
  `mouthSmile*`.
- `alpha_interp/` cross-phase: matches the existing finding —
  0% strictly monotonic in `mouthSmile*` across 60 trajectories,
  step-function in `jawOpen`.

**Falsification condition:** if in-phase sweeps show the same
non-monotonic rate as cross-phase, the hypothesis is wrong and the
geometry really does lie in Flux's attention-cache, not in the
prompt-embedding decomposition. In that case we fall back to the
local-metric estimation thread.

**Step 1.3 — per-block attention-cache decomposition.** Using the
`alpha_interp_attn/` attention pkls (660 attn states, one per
α-step), for each DiT block and step, compute the per-channel
derivative of the attention-cache projection onto the AU12 ridge
direction vs the AU26 ridge direction (ridge fits from Stage 4.5
on-hand, not yet AU-supervised — this is a coarse preliminary check).

If the AU12 projection peaks mid-α and the AU26 projection turns on
sharply around α≈0.45, the mixture-flip mechanism is **directly
visible in attention space**, without needing image-space
blendshapes.

Deliverable: `docs/research/2026-04-22-inphase-monotonicity.md` with
τ tables and the attention-projection plot.

## Phase 2 — sparse NMF on blendshape corpus

**(Revised 2026-04-22 after lit read of Tripathi 2024 PCA + DFECS. Full recipe in
`2026-04-22-blendshape-decomp-lit-read.md`. Original PCA→ICA framing below
preserved for provenance but superseded — use the NMF recipe.)**



**Goal:** Produce a 20-25 dimensional AU-like axis basis from
measured blendshapes.

**Step 2.1 — assemble corpus.** Concatenate blendshape scores from:

- `bootstrap_v1/blendshapes.json` (288)
- `alpha_interp/blendshapes.json` (660)
- `smile_inphase/blendshapes.json` (330, new)
- `jaw_inphase/blendshapes.json` (330, new)
- `calibration_expanded/` (to be scored, 300)
- `intensity_full/` (to be scored, 504)

≈ 2400 scored images.

**Step 2.2 — preprocess.** Stack the 52 channels; drop channels with
< 1% variance across the corpus (neutral/`_neutral` is expected to
be flat). Remove bilateral collapse by symmetrising
(`mouthSmileLeft` + `mouthSmileRight`) / 2 → one channel per pair
unless asymmetry variance exceeds 20% of symmetric variance. This
roughly halves 52 → ~30 channels.

**Step 2.3 — PCA whiten.** Retain components explaining 95% of
variance (literature target: ~25 components for expression data).
Centre + whiten.

**Step 2.4 — ICA on whitened.** FastICA with `n_components =` PCA
retained. Match recovered independent components to known AUs via
the ARKit-to-FACS cheat-sheet (Özel 2020) — report which ICA axis
most closely corresponds to AU1, AU2, AU4, AU6, AU9, AU12, AU15,
AU17, AU20, AU25, AU26, AU43, plus the eye-gaze 2-DOF pair.

**Step 2.5 — validation.** Reconstruct the 52-channel vector from
ICA; report per-channel reconstruction R². Drop any ICA axis with
reconstruction-contribution < 1%.

Deliverable: `src/demographic_pc/blendshape_decomposition.py` and
`docs/research/2026-04-22-blendshape-ica-axes.md` with the AU map.

## Phase 3 — per-axis attention-cache ridge fits

**Goal:** Produce one FluxSpace-compatible attention-cache direction
per ICA axis.

**Step 3.1 — training set.** For each image in the corpus, pair
(ICA coefficient vector, attention-cache features). Use the
attention caches from `bootstrap_v1`, `smile_inphase`,
`jaw_inphase`, `alpha_interp_attn`, wherever captured. Note: some
corpus images have no attention capture (early renders); those are
scoring-only and don't enter the ridge fit.

**Step 3.2 — ridge fit per axis.** For each ICA axis `k`, ridge-
regress `c_k` on the attention-cache features. Use the same
machinery as Stage 4.5 (`fit_ridge_attention.py`); targets change
from demographic labels to ICA coefficients. Report R² per axis.

**Step 3.3 — produce edit directions.** Convert each ridge weight
`w_k` to a FluxSpace-compatible attention-cache delta by the Stage 4
protocol (normalise by `‖target − base‖`, etc.). Output:
`models/blendshape_directions/au_{k}.safetensors` or similar.

Deliverable: 20-25 attention-cache directions, each producing a
single-AU edit.

## Phase 4 — per-axis linearity and geometry characterisation

**Goal:** Test each AU direction's linearity in FluxSpace and flag
axes that need geometric correction.

**Step 4.1 — single-axis sweeps.** For each AU direction, render an
α-sweep on 3 base faces × 3 seeds = 9 trajectories over
α ∈ {-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3}. Total: 20 axes × 9 × 9 =
1620 renders. (At 10s/render ≈ 4.5 hours.)

**Step 4.2 — per-axis metrics.**

- Monotonicity: Kendall's τ on measured `c_k(s)` (the target ICA
  coefficient) across the sweep.
- Linearity: R² of linear fit `c_k(s) = β₀ + β₁ s`.
- Identity drift: ArcFace cosine against the neutral base at each α.
- Collapse onset: `max_env` against the calibration prior.
- Cross-axis leakage: measured `c_j(s)` for all other axes j ≠ k.
  A clean axis has leakage below the seed-noise floor.

**Step 4.3 — geometry classification.** Per axis, categorise:

- **Class A (linear):** monotonic, R² ≥ 0.85, leakage < 20%.
  Ship as-is.
- **Class B (saturating but monotonic):** monotonic, R² < 0.85,
  clear saturation at large |s|. Apply `max_env` gating only.
- **Class C (non-monotonic / cliff):** fails monotonicity. Flag
  for Phase 5 geometric correction. Record the α-location of any
  phase boundary.
- **Class D (leaky):** monotonic but leakage > 50% onto another
  axis. Indicates the ICA decomposition didn't fully separate this
  axis; revisit Phase 2 before correcting.

Deliverable: `docs/research/2026-04-23-per-axis-linearity.md` with
the classification.

## Phase 5 — Riemannian correction where needed (optional)

**Trigger:** only run this phase for axes classified Class C.

**Step 5.1 — local metric estimation.** For the failing axis,
estimate a local metric on the attention cache using the dense
single-axis sweep data from Phase 4 (9 trajectories × 9 α values
= 81 samples per axis, plus cross-demographic transfers). Within-
seed covariance; tangent-space Jacobian at the phase-boundary
α-location. Evaluate on a held-out base/seed (B1 tautology check).

**Step 5.2 — reparametrisation attempt.** Apply String-Method-style
non-uniform α spacing around the boundary. Cheapest first. Validate
by re-measuring Kendall's τ on the corrected sweep.

**Step 5.3 — tangent-space correction (if needed).** Only if
reparametrisation doesn't restore monotonicity: project the
per-step edit onto the tangent space of the estimated metric at
each α. Expensive, avoid if possible.

Deliverable: `docs/research/2026-04-23-riemann-per-axis.md` with
per-axis correction recipes.

## Stopping conditions

- **Stop after Phase 1** if in-phase sweeps are monotonic: the
  hypothesis is confirmed and we can move to Phase 2 knowing we're
  building the right thing.
- **Stop after Phase 4** if all AU axes are Class A or Class B:
  ship the blendshape-driven FluxSpace vocabulary; Riemann becomes
  optional side-quest.
- **Pivot after Phase 1** if in-phase sweeps are non-monotonic:
  return to local-metric estimation on attention cache; blendshape
  decomposition is not the explanation.

## Downstream payoff (if this works)

- AU-driven Flux editing compatible with ARKit blendshape coefficients,
  FLAME expression parameters, and the broader VTuber / MetaHuman
  ecosystem.
- A clean experimental distinction between "mixture-of-axes" and
  "attention-cache curvature" as explanations for non-monotonicity
  in any future edit pair.
- The demographic-PC pipeline (paused at Stage 5) gets unblocked with
  a principled edit vocabulary instead of demographic-label ridge
  directions.

## Open questions this does NOT answer

- Whether the attention-cache has genuinely non-Euclidean geometry
  independent of the mixture-of-axes effect. Phase 5 touches this
  but only for axes that fail Phase 4 — we get a local answer, not
  a global one.
- Whether the ICA basis is the right decomposition vs SPLOCS or
  direct FACS-AU supervision. Phase 2 produces ICA; if reconstruction
  is poor we may need to revisit.
- Whether the per-axis directions compose additively at inference
  (edit AU6 + AU12 for Duchenne smile). Not tested in Phase 4; cheap
  follow-up.

## Artefacts and scripts

Existing:
- `src/demographic_pc/score_blendshapes.py` (MediaPipe 52-channel scoring)
- `src/demographic_pc/analyze_alpha_linearity.py` (τ / polynomial-fit analysis)
- `src/demographic_pc/fit_ridge_attention.py` (attention-cache ridge fits)
- `src/demographic_pc/fluxspace_render.py` (per-axis rendering)

New:
- `src/demographic_pc/blendshape_decomposition.py` (Phase 2)
- `src/demographic_pc/fit_au_directions.py` (Phase 3)
- `src/demographic_pc/per_axis_linearity.py` (Phase 4)
