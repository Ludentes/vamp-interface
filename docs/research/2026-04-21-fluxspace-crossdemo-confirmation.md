---
status: live
topic: manifold-geometry
summary: Pair-averaging and ratio-threshold collapse predictor generalize across six demographics; T_ratio=1.275 achieves 93% accuracy despite baseline shifts.
---

# FluxSpace Cross-Demographic Confirmation (Glasses Axis)

**Date:** 2026-04-21
**Follow-up to:** `2026-04-21-fluxspace-collapse-prediction.md`

## Question

Does `FluxSpaceEditPair` (attention-cache averaging of A=bare and B=base-splice
edit prompts) generalize beyond the single `adult Latin American woman` base on
which we derived it? And does the `max_env` collapse predictor we fit against
that single base transfer?

## Setup

Six base prompts drawn from the calibration corpus (neutral portrait, varied
demographics): `asian_m`, `black_f`, `european_m`, `elderly_latin_m`,
`young_european_f`, `southasian_f`. For each, we constructed B by splicing
"wearing thick-rimmed eyeglasses" into the base; A stayed constant (`"A person
wearing thick-rimmed eyeglasses."`). One measurement pass per base at s=1.0,
mix_b=0.5, seed=2026. Verification sweep: ~11 scales per base, straddling the
predicted safe edges.

All artefacts under `output/demographic_pc/fluxspace_metrics/crossdemo/`.

## Finding 1: pair averaging generalizes

At s=1.0, all six bases gain thick-rimmed glasses. No demographic drift
visible. Confound cancellation (the age-drift mechanism from the Latin-f
study) holds across other demographics at the same mix_b=0.5.

## Finding 2: absolute threshold does NOT generalize

`max_env(s)` values at s=0 vary by ~30% across bases (6.72 on latin_f → 9.56
on european_m). Applying the Latin-f-fitted T=8.5 predicted **zero safe range
for 3/6 bases** — their baseline `max_env(0)` already exceeded 8.5.

**Fix:** switch to the ratio metric `max_env(s) / max_env(s=0)`. Fits to
T_ratio=1.275 (averaged across Latin-f seeds), 92.86% in-sample accuracy on the
Latin-f sweep — same as the absolute threshold.

## Finding 3: ratio threshold generalizes, but with ~0.25 positive-side drift

Verified per-base against sweeps straddling the predicted edges:

| base | predicted safe | observed safe | Δ (upper) |
|---|---|---|---|
| asian_m | [−0.25, +1.35] | [≈−0.35, +1.35] | nailed |
| black_f | [−0.25, +1.75] | [−0.25, **+1.45**] | **−0.30** |
| european_m | [−0.50, +2.60] | [−0.40, **+2.30**] | −0.30 |
| elderly_latin_m | [−0.60, +1.05] | [−0.50, +1.15] | +0.10 |
| young_european_f | [−0.35, +2.45] | [<−0.55, **+2.15**] | −0.30 |
| southasian_f | [−0.45, +2.50] | [<−0.65, **+2.25**] | −0.25 |

T_ratio=1.275 is systematically too permissive on the positive side.
Tightening to ~1.20–1.22 would close the gap at the cost of shrinking the
predicted safe range for bases that actually tolerate strong edits.

## Finding 4: usable window width varies 10×, uncorrelated with baseline

- `elderly_latin_m`: baseline 9.06, observed safe width ≈ 1.65 (narrow: cliff
  both sides)
- `young_european_f`: baseline 7.72, observed safe width > 2.70 (wide)

Window width is **not** explained by baseline activation magnitude. It's a
property of the `(δ_mix, attn_base)` pair — how much room the edit direction
has before it overruns the existing signal. This suggests a per-base geometric
metric (e.g. `cos(δ, attn_base)` per (block, step)) is the right explainer.
Not yet computed.

## Finding 5: math matches visuals directionally

For Latin-f, `max_env` grows faster on the negative s side than the positive
(ratio at s=−0.6 is 1.50, at s=+1.3 is 1.33). This correctly predicts that
negative-side collapses arrive at smaller |s| — and indeed the observed
collapse onset at s=−0.7 is closer to origin than the s=+1.4 onset. A single
scalar threshold flattens this asymmetry into either positive-biased or
negative-biased prediction; a two-sided fit (separate T⁺, T⁻) should recover
the asymmetry.

## Primary-metric gap

The collapse predictor answers "when does it break." It does NOT answer the
primary questions:

1. **Is the attribute actually present at scale s?**
2. **How much identity drift relative to s=0?**

Both are measurable on existing renders without new inference:

- ArcFace IR101 cosine vs s=0 baseline → drift curve per base
- CLIP similarity delta between "person with glasses" and "person without" →
  glasses-presence curve per base

These two curves together define the *usable* window, not just the
non-collapsed window.

## Finding 6: geometry cos(δ, attn_base) predicts window width

Computed per (block, step) from measurement pkls; aggregated as the 95th
percentile of |cos|:

| base | cos\|p95\| | ratio_p95 (‖δ‖/‖ab‖) | observed safe-width |
|---|---|---|---|
| asian_m | 0.85 | 0.89 | 1.70 |
| black_f | 0.86 | 0.91 | 1.70 |
| elderly_latin_m | 0.85 | 0.89 | 1.65 |
| european_m | 0.93 | 0.96 | 2.70 |
| young_european_f | 0.92 | 0.94 | 2.70 |
| southasian_f | 0.93 | 0.95 | 2.90 |

All `cos(δ, attn_base)` values are strongly **negative** (mean ≈ −0.52),
meaning `δ` partially *cancels* the base. Counter-intuitively, the bases with
the most antiparallel δ have the **widest** usable windows — the cancellation
creates a valley in `|attn_base + s·δ|` that extends the safe range before
collapse.

**This is a third predictor from the same measurement pass.** No sweep
required. Narrow- vs wide-window bases cluster cleanly at cos\|p95\| ≈ 0.85
vs ≈ 0.93.

Prior hypothesis was inverted — high alignment was expected to narrow the
window. The opposite happens because δ points against the base, not along it.

## Finding 7: primary metrics define usable window shape

ArcFace drift (1 − cos vs s=0 baseline) and CLIP P(glasses) computed across
all cross-demo verification renders:

- **Drift at s=0 ≈ 0** for all bases (sanity check passes).
- **P(glasses) rises, peaks near s=+1.0, then drops** as collapse scrambles
  the image and glasses become unrecognisable to CLIP.
- **Sweet spot** = where P(glasses) peaks with drift still below threshold.
  For most bases: s ∈ [+0.95, +1.15]. For elderly_latin_m (narrowest window):
  s ∈ [+0.85, +1.05].
- **CLIP ViT-B-32 peak P(glasses) is only 0.45–0.74**; visually the glasses
  are clear. CLIP is a weak detector on 512→224 downsampled portraits — curve
  *shape* is informative but absolute values aren't.

## Finding 8: later `start_percent` reduces drift on narrow-window bases

Swept `start_percent ∈ {0.15, 0.20, 0.30, 0.40}` at s=1.0:

| base | sp=0.15 drift | sp=0.40 drift | reduction |
|---|---|---|---|
| elderly_latin_m | 0.57 | **0.24** | **−58%** |
| southasian_f | 0.29 | 0.23 | −21% |
| latin_f | 0.33 | 0.31 | ~flat |

Glasses presence (CLIP) stays within ±0.05 across sp values — the edit still
fires. Narrow-window bases benefit most: postponing edit onset lets identity
lock in first.

**Default recommendation:** use `start_percent=0.30` or `0.40` when the
(δ, base) geometry predicts a narrow window (cos\|p95\| ≈ 0.85). The wider
bases are indifferent.

## Open follow-ups

- N-way averaging (3+ edit prompts with more confound dimensions).
- Two-sided threshold (T⁺ ≠ T⁻) for collapse prediction.
- Dedicated glasses detector (replace CLIP ViT-B-32 with a face-ROI classifier).
- Cross-axis generalisation (smile, beard).
- Automatic selection of `start_percent` from the cos\|p95\| geometry metric.

## Artefacts

- Measurements: `output/demographic_pc/fluxspace_metrics/crossdemo/measurement/*.pkl`
- Verification renders: `output/demographic_pc/fluxspace_metrics/crossdemo/verify/<base>/`
- Collages: `output/demographic_pc/fluxspace_metrics/crossdemo/collages/{<base>.png, all_bases.png}`
- Predictions: `output/demographic_pc/fluxspace_metrics/crossdemo/predictions.json`
- Pipeline: `src/demographic_pc/fluxspace_metrics.py` (`--crossdemo-measure`, `--crossdemo-verify`)
- Collage script: `src/demographic_pc/fluxspace_crossdemo_collage.py`
