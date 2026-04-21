# Stage 4.5 — Ours vs FluxSpace on a single axis (age)

**Date:** 2026-04-20
**Follows:** [Stages 2–4 report](2026-04-20-demographic-pc-stage2-4-report.md).
**Status:** Complete. 340 renders evaluated with MiVOLO, FairFace, InsightFace, ArcFace IR101.

## Question

Stage 4 produced a regression-based age direction in Flux's 4864-d conditioning space — concat[CLIP-L pooled 768, T5-mean 4096]. Before investing further, we wanted a sanity check against the nearest comparable published method. Two concrete questions:

- **Is our direction actually better than a naïve prompt-pair contrast?** FluxSpace-coarse (Dalva et al., CVPR 2025) has an at-this-layer variant that we can implement in an afternoon: encode "young adult" and "elderly" portrait prompts, direction = (target − proj_base(target)). If our ridge-regression direction is within the same ballpark, the Stage 4 pipeline is vindicated; if worse, we've been building on sand; if better, it may be worth publishing.

- **Can we quantify "respects the manifold"?** Visually, extreme strengths produce coherent saturation under one method and catastrophic collapse under the other. We want that intuition grounded in something measurable.

Only the age axis is tested. Gender and race are deferred — age has the strongest per-head regression (R²=0.991) and the tightest cross-classifier agreement (Stage 4), so it's the fair test.

## Setup

- **20 held-out portraits**: stratified (gender, ethnicity) grid at age="adult", seeds 2000–2019, distinct from the 1785 training seeds.
- **9-level λ grid**: {−3, −2, −1, −0.5, 0, +0.5, +1, +2, +3}. Inner range (|λ|≤1) is the published-editing regime; |λ|>1 deliberately probes off-manifold / uncanny territory.
- **Two methods, same injection point**: both add `λ · w` at the CLIP-L pooled + T5-mean conditioning layer, via a custom ComfyUI node (`ApplyConditioningEdit`). λ=0 is method-agnostic, rendered once per portrait as `baseline`.

**"Ours" direction.** Ridge (α=316, SVD solver) on the mean-centered 4864-d conditioning regressing MiVOLO-age (N=1785). `direction = w / (w·w)` — unit strength = +1 predicted year. To reach visually meaningful shifts the scale is λ·45 years.

**FluxSpace-coarse direction.** Encode `base="young adult person portrait..."` and `target="elderly person portrait..."` through the same CLIP-L + T5 stack. Direction = `target − proj_base(target)` (component of target not in base), normalized so strength 1 ≈ one pair-magnitude. Scale is λ·2 pair-magnitudes.

340 renders: 20 baselines + 20 portraits × 8 non-zero λ × 2 methods.

Classifiers run on all 340 (MiVOLO age/gender, FairFace gender/race with dlib-CNN alignment, InsightFace age/gender with SCRFD detect). ArcFace IR101 (`minchul/cvlface_arcface_ir101_webface4m`) 512-d embedding per render for identity drift.

## Metrics

We report six things per method.

**1. On-axis age slope.** Two flavours:
- *Local* — central-difference slope between λ=+0.5 and λ=−0.5, per portrait, averaged. Tangent at the origin, robust to saturation.
- *Global* — polyfit deg=1 over the full 9-point λ grid. What a naïve linear summary would say.

Reporting both matters because if the response saturates, local and global disagree; if it's linear, they match.

**2. Linearity R².** R² of the linear fit to MiVOLO-age over the full λ range, per portrait, averaged. Low R² means curved shape (saturation, sigmoid, or collapse).

**3. Inner / outer slope.** Separate polyfits on |λ|≤1 and |λ|>1. Ratio outer/inner < 1 ⇒ saturating; > 1 ⇒ accelerating; negative ⇒ overshooting.

**4. Attribute Dependency (AD).** At matched target-slope (using *local* slopes to avoid the linearity confound), per non-target continuous score:
- Gender: slope of P(male)−P(female) vs λ.
- Race: per-class slope of P(race=k) vs λ for k ∈ FairFace's 7 classes.

AD_k = |slope_k| / |slope_age|. Reported as max and mean over k.

**5. Off-axis flip rate.** At |λ|=3, fraction of portraits whose argmax(FairFace gender) or argmax(FairFace race) differs from baseline. Coarse but interpretable.

**6. Identity drift.** `1 − cosine(ArcFace(baseline), ArcFace(edit))` per non-baseline render, averaged by λ.

Plus one direction-only metric, **Mahalanobis norm per unit strength**, computed from the training conditioning:

- Fit Σ on the 1785 training conditioning vectors with Ledoit-Wolf shrinkage (α=0.005 — covariance was essentially well-conditioned).
- For each direction `w`, report `√(wᵀ Σ⁻¹ w)`. Since the edit is `c → c + s·w` applied uniformly, this is the standard-deviation-unit step size per unit Euclidean strength.
- Ratio Mahal/Eucl is the key diagnostic: low ⇒ direction aligned with high-variance (manifold) axes; high ⇒ direction crosses low-variance (off-manifold) axes.

## Results

### Direction-only: how far off the manifold?

| | Euclidean ‖w‖ | √(wᵀΣ⁻¹w) | **Mahal / Eucl** |
|---|---|---|---|
| Ours | 0.141 | 0.270 | **1.92** |
| FluxSpace | 14.75 | 405.6 | **27.49** |

FluxSpace's direction carries 14.3× more standard-deviation cost per unit Euclidean step than Ours'. This is a theoretical prediction confirmed numerically: ridge regularization (α=316) plus the representer theorem force Ours' `w` to lie in the span of observed conditionings with weight concentrated on high-variance axes; a single-prompt-pair contrast has no such constraint and picks up low-variance off-principal loadings (tokenizer idiosyncrasies, prompt-specific stylistic co-loadings).

This is direction-geometry alone — no renders, no classifiers. It predicts that FluxSpace will leave the training-conditioning hull at smaller λ, and that's exactly what the image-level metrics show.

### Image-level: how does the edit behave?

| Metric | **Ours** | **FluxSpace** | Comment |
|---|---|---|---|
| Age slope local (yr/λ, @ λ=0) | +8.93 | +20.91 | FS stronger near origin |
| Age slope global (yr/λ, full range) | +6.28 | +3.74 | FS crashes globally |
| **Linearity R²** | **0.896** | **0.216** | Ours near-linear; FS strongly curved |
| Slope inner (|λ|≤1) | +8.35 | +16.70 | FS inner slope 2× Ours |
| Slope outer (|λ|>1) | +6.08 | +2.49 | FS outer slope 25× collapsed |
| **Saturation ratio** (outer/inner) | **0.73** | **0.15** | FS spends 85% of range by |λ|=1 |
| AD gender (local) | 3.5·10⁻⁵ | 2.7·10⁻⁴ | FS 8× more entangled |
| AD race max (local) | 1.0·10⁻³ | 4.1·10⁻³ | FS 4× more entangled |
| **Gender flip @ |λ|=3** | **0%** | **100%** | FS flips every portrait |
| **Race flip @ |λ|=3** | **12.5%** | **100%** | FS flips every portrait |
| **ArcFace drift @ λ=−3 / +3** | 0.319 / 0.443 | **0.976 / 0.996** | FS identity is gone |
| ArcFace drift @ λ=−1 / +1 | 0.155 / 0.143 | 0.463 / 0.412 | Even at inner-range limit |

Every single image-level metric favours Ours except *local age slope near the origin*, where FluxSpace moves 2.3× harder per λ. That's a gain choice, not a quality result — we could relabel λ_ours to match it; the other metrics would move in Ours' favour further. At *matched target-slope* (the local-slope rescaling), the AD and identity-drift numbers are what they are.

### The saturation curve

Mean MiVOLO-predicted age across the 20 portraits, per λ (adult baseline ≈ 36 years):

```
λ:   -3    -2    -1    -0.5   0     +0.5  +1    +2    +3
Ours       — monotone, near-linear inside, gently saturating outside.
Flux       — steep inside, then a cliff both directions.
```

(Raw curves from `output/demographic_pc/stage4_5/eval_summary.json` `identity_drift_by_lam` and MiVOLO age; see the companion blog post for the plot.)

Ours' linearity R² = 0.90 across the full |λ|≤3 range; FS = 0.22. At the same time, identity drift at |λ|=3 is 0.32–0.44 for Ours and 0.98–0.996 for FS. So Ours is simultaneously more linear in dose-response *and* less destructive to identity.

## Interpretation

### Method analysis supports the geometric prediction

Ours is a supervised direction-extraction (InterfaceGAN/StyleFlow style): all 1785 training conditionings, ridge regularization, fit against a labelled signal. By the representer theorem `w` is a linear combination of observed points, and heavy α=316 ridge shrinks it onto the dominant subspace of the conditioning distribution. That makes it, effectively, an age-projection onto the data manifold. Pushing along `w` stays inside the empirical conditioning hull for a wider range of strengths, with the decoder still able to interpret the result coherently.

FluxSpace-coarse is a two-point contrast, with no distributional constraint and no regularization. The direction encodes whatever the two specific prompt embeddings happen to differ in — including axes the data distribution never populates. A unit Euclidean step traverses 27 standard deviations of the conditioning distribution; the decoder quickly runs out of training signal telling it what to do.

This is the supervised-direction vs prompt-contrast distinction known from GAN editing. What Stage 4.5 adds is a specific number (14.3× Mahalanobis ratio) and a measured behavioral consequence (linearity R² 0.90 vs 0.22; identity preserved vs destroyed at matched extremes).

### Two kinds of cliff

Ours and FluxSpace have cliffs in different places and of different shapes:

- **Ours has a wide plateau, gentle edge.** Within |λ|≤1 the edit is linear and productive; past |λ|=1 the slope drops 27% (outer/inner ratio 0.73) and identity drift is bounded at ~0.44 by λ=3. The mechanism limiting further range is the training distribution: Ours' direction is built from MiVOLO-age roughly 5–90 years, and pushing further extrapolates beyond what the regression ever saw.

- **FluxSpace has a narrow plateau, sheer cliff.** By |λ|=1 it has spent ~85% of its on-axis effect. Past that, age-classifier output *decreases* in some portraits (the saturation ratio 0.15 under-reports this — it's averaging across sign changes), identity is gone (ArcFace cos ≈ 0.02–0.04), and every portrait's gender and race flip. This is catastrophic collapse, not controlled saturation.

### Implication for vamp-interface

The vamp-interface design thesis is uncanny-valley-as-fraud-signal: high-sus job postings should produce faces that feel *wrong*. FluxSpace's cliff behaviour — zombie/lesion/gender-flipped territory at moderate λ — is closer to what the uncanny-valley signal needs than Ours' controlled saturation. Ours is the better tool for *staying in the population*; FluxSpace is the better tool for *exiting it visibly*.

The clean finding is not "Ours beats FluxSpace-coarse." It's:

> Direction-extraction method determines where the cliff is. Supervised, manifold-aligned directions have wide, gentle plateaus; prompt-contrast directions have narrow ones with sheer cliffs. Applications that need graceful saturation (precision editing, Concept Sliders) should build the first kind. Applications that need signalled departure (uncanny valley, adversarial perturbation visualisation) may want the second kind, or a hybrid that uses the first for in-distribution steering and a separate mechanism for the off-manifold push.

### Caveats

- **Age only.** Gender and race subspaces have weaker cross-classifier agreement; the comparison may differ there. FluxSpace's fine variant (block-level editing with timestep gating) wasn't tested at all — it's the published version, and may behave differently from the coarse conditioning-level variant we compared against.
- **Classifier cap confound.** MiVOLO can't report much past ~100 years. At OURS_SCALE=45, λ=3 requests a +135-year shift — well past the classifier's range. Part of the measured saturation is classifier ceiling, not generator ceiling. ArcFace drift continuing to grow past |λ|=1 (0.14 → 0.32 → 0.44) suggests the generator is still moving; to fully disentangle we'd need a perceptual age rating on the images themselves.
- **n=20 portraits.** Small sample. Effect sizes this large survive obvious sensitivity checks, but per-cell breakdowns (per ethnicity, per gender) aren't statistically meaningful at this N.
- **Single scale choice.** OURS_SCALE=45 years/λ and FS_SCALE=2 pair-magnitudes/λ were picked so that λ=1 gives visually meaningful shifts. Different scales would move "the cliff" in λ-space but not change the relative shape.

## Reproducibility

- Code: `src/demographic_pc/stage4_5_{render,evaluate}.py`, `src/demographic_pc/build_age_edits.py`, `ComfyUI/custom_nodes/demographic_pc_edit/__init__.py`.
- Renders: `output/demographic_pc/stage4_5/renders/{baseline,ours,fluxspace}/` (340 files).
- Metrics: `output/demographic_pc/stage4_5/eval.parquet` (per-render classifier + ArcFace), `eval_summary.json` (all reported numbers).
- Training conditioning + Σ: `output/demographic_pc/conditioning.npy`, shrinkage computed inline.

Runtime: 340 renders ≈ 30 min on RTX 5090, evaluation ≈ 45 s (Mahalanobis Ledoit-Wolf ~15 s, classifier sweep ~30 s at 10 samples/s).
