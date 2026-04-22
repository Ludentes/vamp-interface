---
status: live
topic: manifold-geometry
---

# Three manifold-theory papers vs. our α-interp finding — what each one actually gives us

**Date:** 2026-04-22
**Papers read:** `docs/papers/hessian-geometry-2506.10632.pdf`,
`learning-on-manifold-rjf-2602.10099.pdf`, `diffusion-string-method-2602.22122.pdf`
**Open questions this was meant to answer:**
- Does the Mona-Lisa→Joker α≈0.45 cliff match a predicted phase-boundary phenomenon?
- Is Euclidean `mix_b` linear interpolation *provably* inadequate, and do we have a drop-in replacement?
- Can we estimate a local Fisher/Riemannian metric for collapse prediction (the "within-identity Σ" question)?

## One-line applicability verdict

| Paper | Applicability to FluxSpace `mix_b` editing | Usable recipe? |
|-------|---------------------------------------------|----------------|
| Hessian Geometry (Lobashev et al.) | Conceptual match for the cliff; not validated on DiT/flow-matching; 2D latent slice only | Partial — requires picking a second axis |
| RJF (Kumar & Patel) | Training-time only; requires analytical hyperspherical manifold; does not cover test-time edits | No |
| Diffusion String Method (Moreau et al.) | Directly demonstrates non-monotonic likelihood along linear-init paths; matches smile phenomenology | Closest — but in state space, not attention-cache space |

## Hessian Geometry — Lobashev et al., ICML 2025

**What the Fisher metric actually is in this paper.** Not the Hessian of log p(x). Not the Jacobian of the generator. It is the Hessian of the **log-partition function log Z(t)** over a 2-parameter latent slice, fit by an MLP trained with Jensen-Shannon divergence against a CLIP-feature-based posterior (p.3 Eq. 8; p.4 Eq. 19; p.5 Eqs. 22–24). Eigenvalues of this Hessian diverge at phase boundaries (p.7 Fig. 6 — fractal down to 10⁻⁸ float16 precision).

**What this predicts for our cliff.** Proposition 4.1 (p.9) is a toy model where a unimodal latent maps to a bimodal image distribution via variance-preserving reverse-ODE; the Lyapunov exponent `λ = (β/2)(1 + (1−σ²)/σ⁴)` diverges as σ→0 at the midpoint between modes. **This is the formal statement of "diverging Lipschitz at a phase boundary"** and structurally matches our α≈0.45 step in `jawOpen`. Our fixed-base Mona→Joker pair, each rendered with a different base portrait (→ bimodal image distribution), is exactly their setup.

**What the paper does *not* cover.**
- **Only validated on 2D latent slices of SD 1.5 (DDIM), Ising, TASEP.** No DiT, no flow matching, no Flux.
- **Hessianizability theorem (Thm 2.1, p.3) only guarantees a Fisher metric on 2D analytic manifolds.** Our `mix_b` is 1D; no metric is even defined on a line. To apply their method, we'd need to pick an orthogonal second parameter (another FluxSpace axis, a seed perturbation, or a `start_percent` dial) to form a 2D slice.
- The learned log Z is **not required to be convex** (p.5). Positivity of the inferred metric is empirical, not structural.
- Stochastic sampling (η>0) smooths their free-energy landscape and hides phase boundaries (p.7). FluxSpace operates at η=0 rectified-flow so this is fine for us.

**Verdict.** Right conceptual match. Not a usable recipe as-is. If we want a concrete Fisher estimate, we need to define a 2D latent slice first, then run their posterior-fit pipeline. Fast pilot: 2D slice `(mix_b, start_percent)`, small grid, CLIP-distance posterior per cell.

## RJF — Kumar & Patel

**What it does.** Replaces Euclidean interpolation `x_t = (1−t)x + tε` with spherical linear interpolation (SLERP) as the training-time probability path for flow-matching DiTs on frozen representation-encoder features (DINOv2, SigLIP, MAE). Adds a Jacobi-field-derived loss reweighting `λ(t, Ω) = sinc²((1−t)Ω)` that down-weights errors near t=0 and up-weights near t=1 (p.5 Eq. 9; p.6 Eqs. 13–14).

**Applicability to us: low.**
- **Entirely a training-time recipe.** No treatment of test-time editing, inversion, or `c → c + s·δ` intervention anywhere in the paper. Our use case is out of scope.
- **Requires an analytically known manifold** — specifically a unit hypersphere `S^{d−1}` enforced by LayerNorm on the encoder output. All math uses closed-form SLERP and exponential maps (p.3 Eq. 1: `z = r · ẑ`, r ≈ √d). **Flux's joint-attention output cache has no such structural constraint** and no known analytical geodesic.
- The "chord cuts through low-density region" picture (p.2 Fig. 2) is structurally analogous to our α≈0.45 cliff — at t=0.5 they show `‖x_{0.5}‖ ≈ 0.7√d` off the √d-radius manifold — but this is a **training-pathology claim** about the learned velocity field, not a claim about monotonicity of semantic attributes under test-time editing.

**Verdict.** Mostly inapplicable. The mental model (chord vs geodesic) is useful framing for our non-monotonic-smile observation, but porting the method would require (a) discovering a local analytical structure of Flux's attention cache — we have no evidence one exists, or (b) retraining Flux from scratch, which is not on the table. **Downgrade its priority in future discussion.**

## Diffusion String Method — Moreau et al.

**What it does.** Discretises a path between two endpoint samples as N+1 images with fixed endpoints, iteratively (a) flows each inner point under a regime-dependent velocity (`b_t` alone, `b_t + γ²s_t` for MEP, or full SDE with Voronoi-walker EMA for Principal Curves), then (b) reparametrises to equal arc-length via splines. N=50–70, η=0.1–0.5, quench γ_t→0 as t→1 (p.6 Algorithm 1).

**Direct hit on our phenomenology.** Fig. 4 (p.7) and Fig. 6 (p.8) show **pronounced non-monotonic peaks in log-likelihood along linear-initialised strings** in SiT-XL VAE latent space — exactly the shape of our non-monotonic `mouthSmile` trajectory. Principal-Curve regime (T>0) flattens these peaks (Fig. 5 middle/bottom). The paper's central claim ("linear interpolation drifts off the typical set; Principal Curves stay on it") is the formal version of our working hypothesis for the α-interp result.

**Gaps before we can use it.**
- **Runs in VAE latent (SiT-XL, 4×32×32), not DiT text-to-image state space.** Explicitly listed as future work (p.11).
- **Requires both velocity `b_t` AND score `s_t`.** Flux gives us `b_t` directly; the paper notes (p.3 §2.1, refs 11–14) that a score can be derived from a rectified-flow velocity via stochastic-interpolant identities, but **does not implement this** for rectified flows.
- **Operates on model state `x`, not on attention caches or `mix_b` coordinates.** FluxSpace's `mix_b` is a linear mixture of cached attention outputs inside joint-attention blocks. The paper's score field is defined over denoised images, not over feature activations. Applying the algorithm naively to our setting requires either (i) rendering each inner point to an image and running the method on images (expensive, ~N×50 Flux renders per iteration), or (ii) deriving an analogue of the score field on attention-cache space (no theoretical grounding in the paper).

**Verdict.** Closest theoretical match and most directly usable conceptually. Two viable paths forward:
1. **Cheap path.** Use the paper's reparametrisation + regime ideas as a *scheduling heuristic* on top of `mix_b` — e.g., non-uniform α spacing around α≈0.45 to avoid the phase boundary, or a "walker"-style seed-jitter smoothing of images near the boundary. No new theory needed.
2. **Expensive path.** Implement the full String Method in image space: render ~50 inner-point images from α ∈ [0,1], integrate the Flux rectified-flow velocity backward to get score, apply Algorithm 1 in image space. ~50 iterations × 50 points × ~10s/render ≈ 7 hours per α-sweep. Feasible but heavy.

## Impact on our Mahalanobis-vs-Riemannian analysis

In `_topics/metrics-and-direction-quality.md` we argued that the Fisher metric is "a property of the model, not a sample covariance." **That's true of the formal object but misleading about estimation practice**:

- Lobashev et al. **do not** use sample covariance — they fit a scalar log Z via JSD, which sidesteps the Mahalanobis B1 tautology in a genuinely different way than we had sketched.
- But the JSD-fit introduces *different* failure modes: non-convex MLP → no structural PSD guarantee; CLIP-feature posterior → inherits CLIP's own biases; requires 2D slice, so the direct analogue to "sample Σ on N=1785 conditionings" is "sample a 2D grid of latents and render N images per grid cell," which is a much larger compute commitment than sample-Σ.

**Updated rule**: the Mahalanobis-B1 tautology (evaluate-with-same-data-as-construct) is the important lesson; Fisher-metric approaches sidestep *that specific* failure, but introduce an estimator-design problem. Our "within-identity Σ could rehabilitate Mahalanobis" experiment is still the right next step — it's cheaper than any of these papers' pipelines and tests the same core question (is local covariance informative for collapse prediction?).

## Recommended next actions, in order

1. **Publish the α-interp phase-boundary result as-is.** Frame it as empirical confirmation of the Lobashev Proposition-4.1 phenomenon in a DiT/flow-matching setting — a novel empirical contribution, since Lobashev did not test DiT/flow-matching. Cite all three papers for theoretical framing; acknowledge none directly predicts the DiT case.
2. **Try the String Method's reparametrisation-only heuristic first.** Non-uniform α spacing, phase-boundary-aware sampling. Free; reuses existing `alpha_interp_attn` data.
3. **Run the within-identity Σ experiment** from the metrics topic file. Cheap; tests whether local covariance is a usable collapse predictor.
4. **Hold RJF for training-time work only.** If we ever fine-tune Flux on our demographic corpus, it becomes relevant; otherwise skip.
5. **Hold Hessian Geometry for a 2D-slice follow-up experiment.** Viable but a separate project; not on the critical path.

## Artefacts

- Full distillations from subagent runs (not persisted; captured here).
- Open transcripts in `/tmp/claude-1000/.../tasks/` until session ends.
