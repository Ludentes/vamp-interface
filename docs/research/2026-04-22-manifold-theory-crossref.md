---
status: live
topic: manifold-geometry
summary: Maps empirical FluxSpace findings to three load-bearing papers: Tubular Neighborhoods (collapse boundary), Hessian Geometry (phase transitions), Log-Domain Smoothing (tangent-normal decomposition).
---

# Manifold Theory Cross-Reference: What Our Empirical Results Match, Extend, and Risk Missing

**Date:** 2026-04-22
**Source survey:** `docs/research/Manifold-Research.md`
**Empirical inputs:** `2026-04-21-fluxspace-synthesis.md` and the four component research docs it references.

## Question

Our FluxSpace pipeline produces a set of empirical regularities (collapse
envelopes, cross-axis geometry invariance, paired-averaging confound
cancellation) that we've treated operationally — fit a threshold, observe a
pattern, exploit it. The manifold-theory survey covers recent work formalising
the geometry that diffusion and flow-matching models learn. How do the two
map onto each other? Which of our empirical findings have theoretical backing
we can lean on, which extend what the theory says, and where might our Euclidean
framing be missing something the theory would warn us about?

## Papers worth prioritising

Three of the survey's cited papers have direct leverage on our open questions.
The rest of the survey is relevant but less load-bearing for what we're doing.

**Sakamoto et al. — Tubular Neighborhoods & Injectivity Radius** (survey ref [2]).
Formalises the region around the data manifold where the latent→data mapping
is well-conditioned. "When trajectories go outside the injectivity radius, the
mapping can self-intersect or become ill-conditioned." This is operationally
the same phenomenon as our `max_env` ratio crossing `T_ratio = 1.275` — our
threshold is an empirical estimate of that boundary in FLUX's attention-output
space. The paper likely gives a formal definition, an algorithm to estimate
the radius, and error bounds. Reading it could replace our hand-fit scalar
threshold with a principled construction.

**Hessian Geometry of Latent Space in Generative Models** (survey ref [13]).
Reports "fractal phase transitions" in latent spaces: "geodesic interpolations
are approximately linear within each phase but break down at phase boundaries
where the effective Lipschitz constant diverges." This is the exact prediction
our α-interpolation experiment (Mona Lisa ↔ Joker in 11 steps per base × 10 seeds)
is about to test. Before analysing those results we should know what signatures
distinguish a within-phase smooth trajectory from a phase-boundary crossing —
the paper presumably gives diagnostics.

**Farghly et al. — Log-Domain Smoothing & Manifold Adaptivity**
(survey refs [3, 4, 10]). Shows that score-matching implicitly smooths in
directions tangent to the data manifold, leaving normal directions less
smoothed. "Smoothing is implicitly geometry-adaptive, concentrated along
planes parallel to the data manifold." This predicts that editing directions
with large normal components collapse faster than tangential ones. Our
counter-intuitive `cos(δ, attn_base) ≈ −0.52 → wider window` result is
partially explained by simple magnitude cancellation, but a proper tangent
decomposition `δ = δ_tan + δ_normal` would give a sharper predictor. Paper
would give us the framework.

Lower-priority but worth noting: **Geometric Perspective on Diffusion Models**
(ref [1], foundational but more diffusion-specific than rectified-flow),
**Rectified-CFG++** (refs [28, 29], geometry-aware guidance — we don't use CFG
but the predictor-corrector formulation might apply to our `s`-scaling),
**Stability of Flow Matching** (ref [27], cross-model manifold invariance).

## Where our findings agree with the theory

| empirical finding | matching theoretical claim | survey source |
|---|---|---|
| `max_env(s) / max_env(0) ≤ 1.275` predicts collapse | trajectories exiting the injectivity radius cause self-intersection / ill-conditioning | Sakamoto et al. (ref [2]) |
| glasses window asymmetric; smile window symmetric | manifold curvature is direction-dependent; smoothing is geometry-adaptive | Farghly et al. (refs [3, 4]) |
| double blocks carry semantic edit signal; single blocks don't | text-image joint attention is the semantic layer, single blocks handle refinement | FLUX architecture + Flux.1 Kontext (ref [25]) |
| `cos|p95|(δ, attn_base)` ranking identical across glasses + smile axes | separate-trained flow models share the same global manifold; coarse geometry is stable | "Amazing Stability of Flow Matching" (ref [27]) |
| no CFG used; collapse-on-scale is intrinsic to δ, not guidance-induced | CFG pulls trajectories off-manifold — our edit force is analogous, scaled | Rectified-CFG++ (refs [28, 29]) |
| paired-contrast ridge |cos| with pair-averaging δ ceilings ~0.37 | linear approximation is limited near curved manifolds; Riemannian treatment tightens it | Riemannian Flow Matching (refs [17, 18]) |

The correspondence is consistent enough that we should stop framing these
results as just operational tricks. They're instances of general geometric
phenomena that happen to show up in FLUX's attention-output space.

## Where our results extend what the survey says

**Per-base local geometry is invariant across semantic axes.**
The survey's shared-manifold claim (Stability of Flow Matching, ref [27]) is
about different *training runs* or *datasets* converging on similar global
manifolds. Our `cos|p95|` result says that within one trained model, a given
*base prompt* has stable geometric character (narrow vs wide window) across
different *edit axes*. That's a more local claim than the one in the survey,
and it enables single-measurement preflight: one geometry pass per base
predicts editing latitude across every axis for that base. Not documented
anywhere we can find.

**B-ladder prompt graduation beats scale magnitude as an intensity dial.**
The survey's direction-finding work (Concept Sliders-style ridge directions,
attention manipulation per FluxSpace) uses scalar strength `λ` as the primary
knob. Our finding — that sweeping the *content* of the edit prompt (faint →
warm → broad → manic) gives cleaner intensity control than sweeping `λ` —
isn't in any of the cited work. It's our operational discovery, and it's the
empirical motivation for the α-interp experiment (which asks whether linearly
interpolating between two endpoint prompts produces continuous intensity in
attention space).

**Pair-averaging at the attention layer cancels prompt-specific confounds.**
None of the cited methods use two edit conditions averaged at mix_b=0.5 with
opposing demographic confounds. This is a second contribution layered on top
of the paper's single-prompt FluxSpace. Worth naming explicitly as an
extension if we write this up for outside audience.

## Where our Euclidean framing might be missing something

**The attention-output space is probably Riemannian, and we treat it as flat.**
Riemannian Flow Matching (ref [17]) and the Geometric Interference work
(refs [22, 23]) both argue that high-dim DiT feature spaces have nontrivial
curvature, and that Euclidean interpolation through such spaces traverses
low-density interior regions — the same failure mode that Rectified-CFG++
addresses for guidance. Our FluxSpace steering adds `s · δ` as a Euclidean
translation; if the true geometry is curved, we're approximating a geodesic
with a chord, which is fine for small `s` and wrong for large `s`. This
might be part of why our safe window has a hard edge: the chord exits the
manifold while a geodesic would stay on it.

α-interpolation is partly a test of this. Linear `mix_b` interpolation
between two cached attention states is a Euclidean line in attention space.
If blendshape progression across the sweep is monotonic-but-nonlinear (e.g.
flat then sudden jump, or concave), that's a signature of underlying
curvature — the chord is a bad approximation of whatever geodesic the model
actually follows.

**We haven't measured the intrinsic dimension or tangent structure.**
Jones' "Manifold Diffusion Geometry" (ref [19]) provides estimators for
curvature, tangent spaces, and intrinsic dimension from samples. We have
300 calibration pkls with per-(block, step) attention states — we could
apply those estimators directly and get an empirical tangent frame at each
base's location. Then the `δ = δ_tan + δ_normal` decomposition becomes
measurable. Haven't attempted this.

**Cross-model manifold transfer is claimed but untested.**
The survey says separate-trained flow models on similar data share the same
coarse manifold (ref [27]). If true for FLUX variants, then our
`cos|p95|` ranking should transfer from FLUX-Krea to FLUX-Dev to
FLUX-Schnell. Haven't tested — would need a second calibration corpus on a
different FLUX checkpoint. One overnight of shard compute.

## Concrete takeaways

1. Read Sakamoto, Hessian-Geometry, and Farghly before we do more theoretical
   extension of our work. The first two will change how we interpret
   α-interp results.

2. Add a tangent/normal decomposition of δ as a new preflight metric.
   Compute a local tangent frame at each base via SVD on nearby calibration
   samples, project δ onto tangent vs normal, report
   `‖δ_normal‖ / ‖δ‖`. Hypothesis: window width correlates better with
   `‖δ_tan‖ / ‖δ‖` than with `cos|p95|` alone. ~20 lines of code on existing
   pkls.

3. Reframe our collapse predictor as an injectivity-radius estimator rather
   than a fitted scalar threshold. If Sakamoto's algorithm is applicable we
   port it; otherwise we at least frame `T_ratio` as an empirical estimate
   of that radius rather than a universal constant.

4. α-interp results should be interpreted through the phase-transition lens:
   smooth monotonic blendshape = within-phase Euclidean interpolation is
   valid; jump/plateau = phase boundary crossed, implying nontrivial
   curvature that Euclidean `mix_b` misses.

5. Test cross-model manifold transfer when we have shard time.
   `cos|p95|` ranking on FLUX-Dev, compare to FLUX-Krea. If identical up to
   permutation, shared-manifold claim confirmed locally.

6. The overall take-home: our pipeline's core metrics (max_env, cos|p95|,
   pair-averaging δ) are instances of general geometric regularities, not
   FLUX-specific artefacts. This is both reassuring (the method should
   generalise) and a warning (the Euclidean framing has known limits).

## Open items the survey raises that we haven't addressed

- Intrinsic dimension of FLUX's attention-output space per (block, step).
  Probably much smaller than 3072; if so, ridge fits on blendshape targets
  should be projected into that reduced subspace.
- Riemannian steering (exponential-map-based) as an alternative to linear
  `attn_base + s · δ`. Would need the metric tensor.
- Entropy-weighted interpolation (String Method, refs [11, 12]) as an
  alternative to Euclidean α-mixing. Might better avoid the "unrealistic
  midpoint" failure mode if it shows up in α-interp.

## Artefacts

- Source survey: `docs/research/Manifold-Research.md`
- Empirical synthesis: `docs/research/2026-04-21-fluxspace-synthesis.md`
- Component findings: `docs/research/2026-04-21-fluxspace-{collapse-prediction,crossdemo-confirmation,smile-axis,smile-intensity-dial}.md`
