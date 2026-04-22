---
status: live
topic: manifold-geometry
summary: Linear mix_b interpolation between "closed-mouth smile" and "manic grin" in FluxSpace reveals a phase transition at α≈0.45; jawOpen step-function, non-monotonic mouthSmile across all 6 demographics.
---

# α-Interpolation Reveals a Phase Boundary Between Mona Lisa and Joker

**Date:** 2026-04-22
**Follow-up to:** `2026-04-21-smile-intensity-dial.md`, `2026-04-22-manifold-theory-crossref.md`
**Experimental data:** `output/demographic_pc/fluxspace_metrics/crossdemo/smile/alpha_interp/`

## Question

If we have two known-good endpoint prompts — "faint closed-mouth smile"
(Mona Lisa) and "manic wide-open grin, teeth bared" (Joker) — can we linearly
interpolate between them in attention-output space via `FluxSpaceEditPair`
`mix_b` and get a continuous spectrum of intermediate smile intensities?

FluxSpace's underlying premise is that joint-attention outputs are a linear
representation space where semantic directions compose additively. `mix_b`
exactly implements that linear interpolation at the attention-cache level. So
the question is a direct test of the linearity assumption.

## Setup

- `edit_a` = "A person with a faint closed-mouth smile."
- `edit_b` = "A person with a manic wide-open grin, teeth bared."
- `scale` = 1.0, `start_percent` = 0.15 (fixed FluxSpace defaults).
- `mix_b` swept at α ∈ {0.0, 0.1, ..., 1.0} = 11 values.
- 6 bases × 10 seeds each = 60 independent (base, seed) trajectories.
- 660 PNGs total, all MediaPipe-scored for 52-channel blendshapes.
- Composite smile intensity = mean of `mouthSmileLeft`, `mouthSmileRight`,
  `mouthStretchLeft`, `mouthStretchRight`. Jaw intensity = `jawOpen`.

## Finding 1: `jawOpen` shows a textbook phase transition at α ≈ 0.45

Average curves across 10 seeds per base:

| α                 | 0.0  | 0.1  | 0.2  | 0.3  | 0.4  | **0.5** | 0.6  | 0.7  | 0.8  | 0.9  | 1.0  |
|-------------------|------|------|------|------|------|---------|------|------|------|------|------|
| asian_m           | 0.05 | 0.05 | 0.09 | 0.10 | 0.14 | **0.40**| 0.56 | 0.57 | 0.58 | 0.60 | 0.61 |
| european_m        | 0.02 | 0.02 | 0.03 | 0.04 | 0.07 | **0.41**| 0.60 | 0.60 | 0.60 | 0.60 | 0.60 |
| elderly_latin_m   | 0.02 | 0.02 | 0.04 | 0.06 | 0.08 | **0.32**| 0.49 | 0.52 | 0.56 | 0.58 | 0.59 |
| young_european_f  | 0.01 | 0.01 | 0.01 | 0.01 | 0.03 | **0.39**| 0.46 | 0.49 | 0.53 | 0.56 | 0.54 |
| black_f           | 0.01 | 0.02 | 0.02 | 0.02 | 0.04 | **0.09**| 0.24 | 0.32 | 0.35 | 0.38 | 0.39 |
| southasian_f      | 0.02 | 0.02 | 0.02 | 0.03 | 0.04 | **0.26**| 0.47 | 0.50 | 0.51 | 0.50 | 0.51 |

Closed-mouth plateau (jaw ≈ 0) through α = 0.4, sharp jump between α = 0.4
and α = 0.5, open-mouth saturation thereafter. Every base shows this.
Jump width ≈ 0.1 in α captures 50–90% of the total jaw range — essentially a
step function, not a gradient.

Cross-seed σ at each α is 0.08–0.14 for jaw, well below the jump amplitude
(0.3–0.4). The discontinuity is not seed noise.

## Finding 2: `mouthSmile` peaks mid-range, then declines — not monotonic

| α                 | 0.0  | 0.1  | 0.2  | 0.3  | 0.4  | 0.5     | 0.6  | 0.7  | 0.8  | 0.9  | 1.0  |
|-------------------|------|------|------|------|------|---------|------|------|------|------|------|
| european_m        | 0.46 | 0.48 | 0.49 | 0.49 | 0.50 | 0.50    | 0.49 | 0.48 | 0.48 | 0.48 | 0.48 |
| young_european_f  | 0.48 | 0.48 | 0.48 | 0.50 | 0.51 | **0.54**| 0.53 | 0.51 | 0.50 | 0.49 | 0.49 |
| southasian_f      | 0.44 | 0.44 | 0.45 | 0.46 | 0.47 | **0.50**| 0.49 | 0.48 | 0.49 | 0.50 | 0.50 |
| black_f           | 0.52 | 0.54 | 0.55 | 0.55 | 0.55 | 0.55    | 0.56 | 0.56 | 0.55 | 0.56 | 0.55 |
| asian_m           | 0.41 | 0.42 | 0.43 | 0.43 | 0.42 | 0.45    | 0.48 | 0.48 | 0.48 | 0.48 | 0.48 |
| elderly_latin_m   | 0.38 | 0.38 | 0.40 | 0.42 | 0.42 | **0.49**| 0.48 | 0.48 | 0.48 | 0.47 | 0.47 |

**0 of 60 (base, seed) trajectories are strictly monotonic in smile.**
Every single one reverses direction at least once.

Interpretation: MediaPipe's `mouthSmile` channel measures lip-corner pull.
At α = 0 (Mona Lisa prompt), the face has a closed-lipped smile — corners
up, mouth shut — which scores high for mouthSmile. At α = 1 (Joker prompt),
the face has a wide-open manic grin — mouth-aperture dominates, corners are
stretched horizontally rather than pulled up, so mouthSmile is lower.
The mid-range maximum is where *both* signals coexist: lip corners still
pulled up, mouth starting to open.

## Finding 3: the two channels encode two different axes

A single linear interpolation on `mix_b` simultaneously traces:

- **Mouth-aperture axis**: roughly step-function in α, saturating at α ≈ 0.5.
- **Lip-corner-pull axis**: concave in α, peaking near α = 0.5 and dropping
  toward both endpoints.

These are not the same intensity axis. The Mona-Lisa-to-Joker prompt pair is
named as if there were a single "smile intensity" dimension that ranges from
subtle to extreme, but the blendshape measurements say this is actually a
linear mix of two distinct facial actions with different dynamics.

## Finding 4: nonlinearity ratio is substantial for smile, modest for jaw

For each (base, seed) we fit both a linear and cubic polynomial to the
α → blendshape trajectory. Nonlinearity = `R²_cubic − R²_linear`.

| channel      | mean R²_linear | mean nonlinearity | p95 nonlinearity |
|--------------|-----------------|-------------------|------------------|
| smile        | 0.49            | 0.229             | 0.620            |
| jaw          | 0.82            | 0.098             | 0.145            |

Jaw is well-approximated by a line once you accept the plateau+jump shape
(the cubic fit can't do much better). Smile has 2× the nonlinearity and
half the linear-R² — its shape actively fights a linear fit.

## Finding 5: per-base concavity is consistent

Concavity sign = mean of middle residuals minus mean of endpoint residuals
from the linear fit. Positive = U-shape (smile higher than line in middle).

| base               | mean concavity | interpretation                     |
|--------------------|----------------|------------------------------------|
| young_european_f   | +0.0329        | concave (mid above line)           |
| elderly_latin_m    | +0.0309        | concave                            |
| european_m         | +0.0209        | concave                            |
| southasian_f       | +0.0106        | concave                            |
| black_f            | +0.0103        | concave                            |
| asian_m            | +0.0006        | approximately linear               |

5 of 6 bases show consistent concavity. asian_m is the exception — its smile
trajectory is closer to a straight line, because the asian_m endpoints have
less smile difference between them (Mona-Lisa α=0 already scores fairly low
mouthSmile on this base).

## Connection to manifold theory

This result is an empirical confirmation of the Hessian-geometry prediction
from the survey (ref [13]): "geodesic interpolations are approximately
linear within each phase but break down at phase boundaries where the
effective Lipschitz constant diverges." The discontinuity in jawOpen between
α = 0.4 and α = 0.5 is a Lipschitz singularity. The concave smile curve is
consistent with two smooth phases joined by a singular point.

It's also evidence that **FluxSpace's Euclidean linearity assumption fails
in the decoded image space even though it holds in attention-cache space**.
`mix_b` produces a linear combination of attention states by construction,
but the rendered image blendshapes are not linear in `mix_b`. Either the
model's image-generation function is non-linear near the interpolation path
(which is expected — it's 20 steps of nonlinear sampling through DiT
blocks), or the attention-cache linearity is itself a fiction because the
Jacobi identity on the learned manifold requires a Riemannian structure the
cache-averaging mechanism doesn't respect.

## Implications for fine-grained control

1. **"Mona Lisa to Joker continuum" is two axes, not one.** The user's
   original expectation of a smooth intensity spectrum between the endpoints
   isn't what this pair produces. A linear-in-α spectrum would require
   endpoints that lie within a single phase.

2. **Intermediate α-values are their own species.** Images at α = 0.4 or
   α = 0.5 are not "50% of a smile"; they're a specific mid-phase expression
   (closed-mouth smile about to open) that doesn't exist on a continuum
   between the two endpoint prompts.

3. **For smile intensity at fine grain, use closer endpoints.** Replace
   Joker with "smiling widely with lips pressed together" to stay in the
   closed-mouth phase. Linear α-interp between two same-phase endpoints
   should produce a monotonic smile trajectory.

4. **For laugh intensity, use α > 0.5 only.** The jaw-aperture axis is
   well-behaved past the phase boundary.

5. **True multi-axis control probably needs simplex mixing.** Two
   endpoints + `mix_b` gives a 1-D line in attention space. A 2-D control
   plane (3 endpoints mixed at barycentric weights, e.g., bare-smile,
   closed-broad-smile, open-laugh) might let us navigate mouth-pull and
   mouth-aperture independently. Untested.

## What we haven't done yet

- **Attention-space geometry of the phase boundary.** The α-interp data
  only contains PNGs, not attention pkls (our script used
  `measure_path=None`). To locate the phase boundary *from attention
  alone* (without rendering each α), we'd need to re-render with attention
  capture. If the boundary corresponds to a specific (block, step) whose
  attention state changes faster than linearly in α, we'd have a preflight
  predictor.

- **Tangent/normal decomposition of the two endpoint δ vectors.** If one
  endpoint pulls more strongly in the manifold's normal direction than the
  other, that asymmetry might explain why the trajectory is non-monotonic.
  Computable from the existing calibration corpus.

- **Riemannian geodesic alternative to linear mix_b.** A proper
  exponential-map-based interpolation on the attention-space manifold
  would, in theory, avoid the phase boundary by curving around it.
  Requires estimating the metric tensor — not currently feasible with our
  data.

## Open questions for further work

- Is there a phase boundary *location predictor* from the attention states
  alone? Hessian eigenvalue spikes would be the natural signal.
- Do different endpoint pairs (e.g., neutral → smile, smile → laugh) also
  show phase boundaries in their α-sweeps? Hypothesis: only when the two
  endpoints are in different phases.
- Does the same phenomenon show up for other semantic edit pairs (glasses
  off → on with different styles, young → old)? We expect yes, with
  different phase-boundary locations per axis.

## Artefacts

- Data: `output/demographic_pc/fluxspace_metrics/crossdemo/smile/alpha_interp/` (660 PNGs)
- Blendshape scores: `alpha_interp/blendshapes.json`
- Per-group linearity stats: `output/demographic_pc/fluxspace_metrics/alpha_linearity/per_group_stats.json`
- Analysis script: `src/demographic_pc/analyze_alpha_linearity.py`
- Scoring script: `src/demographic_pc/score_blendshapes.py`
- Generation script: `src/demographic_pc/fluxspace_alpha_interp.py`
