---
status: live
topic: manifold-geometry
summary: Pair-averaging recipe transfers to smile axis; cos|p95| clustering is cross-axis invariant; ratio threshold over-predicts upper edge by ~0.2.
---

# FluxSpace Smile Axis Confirmation

**Date:** 2026-04-21
**Follow-up to:** `2026-04-21-fluxspace-crossdemo-confirmation.md`

## Question

Does the FluxSpace pair-averaging recipe — derived on the glasses axis —
transfer to smile? Does the same collapse predictor, ratio threshold, and
geometry clustering hold?

## Setup

Same 6 cross-demo base prompts as glasses. A = `"A person smiling warmly."`;
per-base B replaces "neutral expression" with "smiling warmly". mix_b=0.5,
seed=2026, s=1.0 measurement + verification sweeps straddling predicted edges.

Artefacts under `output/demographic_pc/fluxspace_metrics/crossdemo/smile/`.

## Finding 1: pair averaging generalises to smile

At s=1.0, all 6 bases exhibit a clear warm smile with no visible demographic
drift. Mix_b=0.5 continues to cancel the demographic confound introduced by
the spliced B. No retuning of the recipe was needed between axes.

## Finding 2: CLIP is a stronger smile detector than glasses detector

Peak CLIP P(attribute) at s=+1.0, per base:

| base | smile P(peak) | glasses P(peak) |
|---|---|---|
| asian_m | 0.79 | 0.53 |
| black_f | 0.71 | 0.45 |
| european_m | 0.94 | 0.74 |
| elderly_latin_m | 0.84 | 0.68 |
| young_european_f | 0.85 | 0.59 |
| southasian_f | 0.71 | 0.63 |

Smile is a larger visible change on a 224×224 downsampled portrait than
glasses, and CLIP ViT-B-32 picks it up more confidently. Curve shapes remain
interpretable for both axes, but the absolute P value is not comparable
across axes.

## Finding 3: ratio-threshold predictor is systematically *too permissive* on smile

Predicted vs visually-observed edges (T_ratio=1.275, same threshold as glasses):

| base | predicted safe | visual observation | upper Δ |
|---|---|---|---|
| asian_m | [−0.40, +2.45] | upper collapse by s≈+2.25 | **≥−0.20** |
| black_f | [−0.25, +2.35] | collapse by s≈+2.15 | **≥−0.20** |
| european_m | [−0.35, +2.35] | collapse by s≈+2.15 | **≥−0.20** |
| elderly_latin_m | [−0.70, +1.35] | narrow; breakdown by s≈−0.80 and s≈+1.35–1.45 | ~nailed |
| young_european_f | [−0.25, +2.30] | collapse by s≈+2.10 | **≥−0.20** |
| southasian_f | [−0.50, +2.55] | collapse by s≈+2.35 | **≥−0.20** |

The glasses study observed ~0.25 positive-side over-prediction; smile shows a
similar or larger gap on the upper end. T_ratio=1.275 is too loose for smile —
ballpark T_ratio≈1.20 would close the gap, but the right fix is almost
certainly a two-sided threshold (T⁺, T⁻) per axis, and possibly per-axis fits
entirely.

## Finding 4: windows are wider for smile than glasses (except narrow base)

Predicted safe widths (upper − lower):

| base | glasses width | smile width | Δ |
|---|---|---|---|
| asian_m | 1.70 | 2.85 | +1.15 |
| black_f | 1.70 | 2.60 | +0.90 |
| european_m | 2.70 | 2.70 | 0 |
| elderly_latin_m | 1.65 | 2.05 | +0.40 |
| young_european_f | 2.70 | 2.55 | −0.15 |
| southasian_f | 2.90 | 3.05 | +0.15 |

Smile is generally more permissive. The previously-narrow bases (asian_m,
black_f, elderly_latin_m) gain more headroom on the smile axis than the
already-wide bases. This is plausible — smile is a geometric deformation of
lip/cheek regions; glasses is an added object, and the "objectness" pressure
collapses the face faster once the magnitude exceeds some threshold.

## Finding 5: geometry ranking is stable across axes

cos|p95|(δ_mix, attn_base), measured on smile pkls:

| base | glasses cos\|p95\| | smile cos\|p95\| | cluster |
|---|---|---|---|
| elderly_latin_m | 0.85 | 0.86 | narrow |
| asian_m | 0.85 | 0.88 | narrow |
| black_f | 0.86 | 0.90 | narrow |
| young_european_f | 0.92 | 0.91 | wide |
| european_m | 0.93 | 0.92 | wide |
| southasian_f | 0.93 | 0.93 | wide |

The per-base ranking is identical between axes, and cos|p95| values are very
close. This is the **strongest transferable signal** we have:
the (δ, base) geometry of a given demographic prompt is approximately
axis-invariant in the cos|p95| aggregate, even though the numeric safe-width
varies across axes.

**Implication:** one geometry measurement per base may predict
narrow-vs-wide-window behaviour across *all* semantic axes for that base.
If this holds on a third axis (beard, age), the per-demographic geometry
cluster becomes a prompt-engineering invariant, not an axis-specific quirk.

## Finding 6: identity drift at s=0 is zero across all bases (sanity)

ArcFace drift (1 − cos vs s=0 render) = 0.000 for all 6 bases. The pipeline
does not introduce noise at the zero-scale point; the CLIP detector and
ArcFace model are reading the same latent identity.

## Observations that did not transfer

- Smile windows have **no obvious asymmetry** — negative s directions produce
  anti-smile (frown) that collapses at roughly the same |s| as the positive
  direction does. Glasses had a strong negative asymmetry (s=−0.7 collapses
  where s=+1.4 is still borderline safe).
- Elderly_latin_m remains the narrowest base. But it is not dramatically
  narrower on smile (width 2.05) than on glasses (width 1.65).

## Open follow-ups

- Run start_percent sweep on smile for narrow bases (only elderly_latin_m
  qualifies; others are ≥2.5 wide and expected indifferent).
- Two-sided threshold fit (T⁺, T⁻) cross-axis — smile should close the
  over-prediction gap.
- Third axis (beard or age) to confirm geometry-cluster invariance.

## Artefacts

- Measurements: `output/demographic_pc/fluxspace_metrics/crossdemo/smile/measurement/*.pkl`
- Verification renders: `output/demographic_pc/fluxspace_metrics/crossdemo/smile/verify/<base>/`
- Collages: `output/demographic_pc/fluxspace_metrics/crossdemo/smile/collages/all_bases.png`
- Predictions: `output/demographic_pc/fluxspace_metrics/crossdemo/smile/predictions.json`
- Primary metrics: `output/demographic_pc/fluxspace_metrics/crossdemo/smile/primary/primary_metrics.{json,png}`
- Pipeline invocation: `uv run python -m src.demographic_pc.fluxspace_metrics --crossdemo-{measure,verify} --axis smile`
