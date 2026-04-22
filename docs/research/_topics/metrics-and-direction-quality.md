## Metrics and direction quality — current belief

**Status:** live. Last updated 2026-04-22.

### TL;DR of where we landed

- **We moved away from Mahalanobis.** It failed in two distinct roles:
  as a direction-quality metric (tautological) and as a collapse
  predictor (wrong Σ). `max_env` replaced it for collapse prediction;
  no single scalar replaced it for direction quality — we now use the
  behavioural metrics directly (linearity R², identity drift, flip rate).
- **Direction extraction winner: FluxSpace pair-averaged attention
  edits.** Our ridge direction beat prompt-pair on manifold adherence
  (Stage 4.5) but FluxSpace's own coarse variant, once we implemented
  it correctly, beat ridge visually at scale 0.5–1.0.
- **Collapse predictor winner: `max_env = max|attn_base + s·δ|`.** One
  s=1 render + a 10-prompt calibration prior gives 93%/82% in/out of
  sample on the glasses axis (unconfirmed on other axes at this T).

### Why we moved away from Mahalanobis

Two independent failures recorded within ~24 hours of each other:

1. **Tautological as direction-quality metric (Stage 4.5, 2026-04-20).**
   Σ was estimated from the same 1785 conditioning vectors used to fit
   the ridge direction. Ridge + the representer theorem forces the
   weight into the span of those points with heavy shrinkage onto
   high-variance axes, so `√(wᵀΣ⁻¹w)` is low **by construction** for
   our direction and high for any externally-specified direction like
   FluxSpace's prompt-pair embedding. Claiming the 14.3× ratio
   "predicted" FluxSpace's behaviour on unseen renders was the same
   distributional fact showing up twice. Flagged as B1 in the
   adversarial review; now recorded in memory as
   `feedback_no_tautological_predictions.md`.

2. **Wrong Σ as a collapse predictor (2026-04-21).** Diagonal Σ built
   from prompt-to-prompt spread across a 10-prompt calibration corpus
   captured *between-identity* variance, which is much broader than
   the *within-identity* deformation a single render tolerates before
   collapse. Σ under-counted within-identity slack and over-counted
   between-identity slack, giving 78.57% accuracy vs `max_env`'s
   92.86% on the same glasses-axis corpus.

Practical rule: don't put Mahalanobis in a load-bearing position. The
Σ we actually have access to never matches the decision boundary we
care about. `max_env` wins for collapse because the boundary is a
saturation envelope, not a statistical-outlier distance. For direction
quality we use the behavioural metrics directly.

### What replaced it

| Role | Metric that replaced Mahalanobis |
|------|----------------------------------|
| Direction quality | Linearity R² on α-sweep, ArcFace identity drift, demographic-flip rate at matched extremes |
| Collapse / safe-window prediction | `max_env = max_d |attn_base + s·δ_mix|` with threshold `T_ratio ≈ 1.275` of calibration p99 |
| On-manifold-ness in attention space | cos(δ, attn_base − calibration mean); `max_env` as the hard envelope |

### Load-bearing docs (in priority order)

1. [2026-04-21 FluxSpace synthesis](../2026-04-21-fluxspace-synthesis.md)
   — "What we know about Mahalanobis" section is the decisive writeup.
2. [2026-04-20 Stage 4.5 adversarial review](../2026-04-20-demographic-pc-stage4_5-adversarial-review.md)
   — B1 is the tautology flag that killed Mahalanobis as a predictor.
3. [2026-04-20 Stage 4.5 comparison](../2026-04-20-demographic-pc-stage4_5-comparison.md)
   — where the 14.3× ratio was originally computed; kept as provenance,
   superseded by the synthesis doc's reframing.
4. [2026-04-21 FluxSpace collapse prediction](../2026-04-21-fluxspace-collapse-prediction.md)
   — `max_env` beats diagonal-Σ Mahalanobis 93% vs 79%.
5. [2026-04-21 FluxSpace crossdemo confirmation](../2026-04-21-fluxspace-crossdemo-confirmation.md)
   — `T_ratio = 1.275` proposed, over-predicts upper edge on 3/6 bases.

### Does the Mahalanobis critique extend to Hessian / Riemannian framings?

The Hessian-geometry paper and the RJF paper both propose what looks
like a more sophisticated version of "Mahalanobis with the right Σ":
use the Fisher information metric (intrinsic to the model) to measure
on-manifold distance and to construct geodesic interpolation paths.
The two failure modes that killed sample-Σ Mahalanobis apply
differently to this framing.

**Failure 1 — tautology. Does not apply in its original form, but
returns in a subtler one.**

Sample-Σ Mahalanobis was tautological in Stage 4.5 because Σ was
estimated from the same 1785 conditionings used to fit the ridge
direction. The Fisher metric is a property of the Flux model itself
(Jacobian of the generative map, or Hessian of log-likelihood), not
a sample covariance. You can score an arbitrary direction with it
without the same-data-twice circularity. **But** we do not have the
true Fisher metric; we would estimate it. If we estimate it from the
same attention-cache samples that built `mix_b`, and then evaluate
`mix_b` with it, the tautology returns. And if we use a Fisher-derived
metric to *construct* a geodesic and then report that the geodesic has
small Fisher path-length — that is pure B1 in new costume. **Rule:
never score a construction with a metric derived from the same thing
that produced the construction.**

**Failure 2 — wrong Σ. Applies conceptually, but the Hessian framing
handles it differently.**

Sample-Σ Mahalanobis was a single scalar and was tuned to the wrong
variance (between-identity, not within-identity). The Fisher metric
is a full tensor field, and the Hessian-geometry paper says that
phase boundaries appear precisely as **eigenvalue divergence**
(Lipschitz singularity) — qualitatively different from Euclidean
distance blowup. So Fisher *can* express saturation boundaries in
principle. **But** it is still only useful if estimated locally at
the right point. A Fisher estimate built from cross-prompt variation
repeats the "between-identity" error. The right estimate is the
tangent-space Jacobian at a single (base, prompt, step) point.

**Unified rule for upcoming Riemannian experiments**

- Estimate the metric *locally*, not from cross-prompt spread.
- Score geodesics with *independent behavioural* metrics
  (blendshape linearity, ArcFace drift, collapse), not with the
  Fisher metric itself.
- Watch for *eigenvalue spikes*, not just distances — that is
  the qualitatively new signal the Hessian framing offers.
- Our open question "could a within-identity Σ rehabilitate
  Mahalanobis for collapse prediction?" is the same experiment as
  "does a local Fisher estimate predict collapse" — two vocabularies
  for the same test.

### Open questions

- Does `max_env`'s win survive on axes other than glasses? Smile
  crossdemo found cos|p95| ranking transfers but `T_ratio` still
  over-predicts. Not tested on jaw/gender/age at this threshold.
- Could a within-identity Σ (per-base seed variation rather than
  prompt-to-prompt spread) rehabilitate Mahalanobis for collapse
  prediction? Same question as "does a local Fisher estimate predict
  collapse." Cheap to try on existing calibration data.
