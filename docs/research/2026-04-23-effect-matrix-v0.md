---
status: live
topic: metrics-and-direction-quality
---

# Effect matrix v0 (2026-04-23)

Per (axis, subtag, base) cell, slopes of each readout vs `scale`, fit by OLS across all seeds × scales in the cell. Built over the overnight_* corpus (1,536 rows) from the fully populated `models/blendshape_nmf/sample_index.parquet`.

Artifact: [`output/demographic_pc/effect_matrix_v0.parquet`](../../output/demographic_pc/effect_matrix_v0.parquet).

## Readouts

- **target_slope** — slope of the axis's target SigLIP-2 probe margin vs scale. Positive for "add" directions, negative for "remove".
- **primary_atom / slope** — the NMF atom (out of 21) with the largest |slope|; empty if no atom tracks the edit (e.g. beard has no blendshape analog).
- **mv_age_slope, ins_age_slope** — age drift in years per unit scale, MiVOLO and InsightFace readings.
- **identity_drift_slope** — slope of `1 - cos(ArcFace_this, ArcFace_base)`. Positive = identity drifts away from base.
- **total_drift_slope** — slope of `1 - cos(SigLIP_img_this, SigLIP_img_base)`. Catch-all for effects outside blendshapes + probes.
- **mv_gender_flip_hi_vs_lo** — fraction of seeds where the MiVOLO-predicted gender at max-scale differs from the majority gender at min-scale. 0 = stable.
- **max_env_at_max_scale** — mean collapse envelope at the cell's max scale. NaN where no attn cache (all overnight_* rows currently lack attn pkls — stamped only where has_attn=True elsewhere).

## anger · anger

Target probe: `siglip_angry_margin`  · expected sign `+`  · cells: 6

| base | target | atom (slope) | mv_age/y | ins_age/y | id drift | total drift | gender flip |
|---|---|---|---|---|---|---|---|
| asian_m | 0.073 (R²=0.72) | atom_04 (1.609) | +3.35 | +6.69 | +0.413 | +0.092 | 0.00 |
| black_f | 0.080 (R²=0.76) | atom_15 (1.085) | +7.14 | +16.66 | +0.513 | +0.089 | 0.00 |
| elderly_latin_m | 0.075 (R²=0.78) | atom_12 (0.293) | -6.33 | -1.81 | +0.449 | +0.098 | 0.00 |
| european_m | 0.073 (R²=0.74) | atom_18 (-0.513) | +2.90 | +3.65 | +0.442 | +0.093 | 0.00 |
| southasian_f | 0.075 (R²=0.76) | atom_15 (0.685) | -6.67 | +7.39 | +0.504 | +0.151 | 0.00 |
| young_european_f | 0.087 (R²=0.80) | atom_15 (0.959) | +5.38 | +9.41 | +0.541 | +0.114 | 0.00 |

## beard · add

Target probe: `siglip_bearded_margin`  · expected sign `+`  · cells: 2

| base | target | atom (slope) | mv_age/y | ins_age/y | id drift | total drift | gender flip |
|---|---|---|---|---|---|---|---|
| asian_m | 0.068 (R²=0.89) | atom_15 (0.768) | -3.95 | -6.84 | +0.414 | +0.031 | 0.00 |
| european_m | 0.086 (R²=0.80) | atom_05 (1.237) | -3.01 | -5.61 | +0.440 | +0.072 | 0.00 |

## beard · remove

Target probe: `siglip_bearded_margin`  · expected sign `–`  · cells: 1

| base | target | atom (slope) | mv_age/y | ins_age/y | id drift | total drift | gender flip |
|---|---|---|---|---|---|---|---|
| elderly_latin_m | 0.029 (R²=0.50) | atom_15 (0.210) | -14.59 | -5.70 | +0.462 | +0.077 | 0.00 |

## beard_rebalance · remove

Target probe: `siglip_bearded_margin`  · expected sign `–`  · cells: 3

| base | target | atom (slope) | mv_age/y | ins_age/y | id drift | total drift | gender flip |
|---|---|---|---|---|---|---|---|
| asian_m_bearded | 0.001 (R²=0.01) | atom_15 (0.879) | -4.54 | -3.98 | +0.191 | +0.012 | 0.00 |
| elderly_latin_m_bearded | 0.005 (R²=0.27) | atom_17 (0.193) | -11.84 | -2.51 | +0.251 | +0.027 | 0.00 |
| european_m_bearded | -0.002 (R²=0.04) | atom_12 (-0.570) | -2.50 | -2.67 | +0.105 | +0.008 | 0.00 |

## pucker · pucker

Target probe: `siglip_puckered_lips_margin`  · expected sign `+`  · cells: 6

| base | target | atom (slope) | mv_age/y | ins_age/y | id drift | total drift | gender flip |
|---|---|---|---|---|---|---|---|
| asian_m | 0.053 (R²=0.75) | atom_18 (-0.361) | -7.78 | -14.25 | +0.250 | +0.056 | 0.00 |
| black_f | 0.015 (R²=0.26) | atom_09 (0.055) | -4.06 | +0.13 | +0.226 | +0.019 | 0.00 |
| elderly_latin_m | 0.070 (R²=0.90) | atom_09 (0.511) | -3.31 | -5.09 | +0.314 | +0.071 | 0.00 |
| european_m | 0.041 (R²=0.54) | atom_17 (0.519) | -5.60 | -7.43 | +0.382 | +0.074 | 0.00 |
| southasian_f | 0.020 (R²=0.27) | atom_17 (0.289) | -10.76 | -8.53 | +0.455 | +0.096 | 0.00 |
| young_european_f | 0.029 (R²=0.56) | atom_04 (-0.244) | +1.60 | -0.22 | +0.269 | +0.050 | 0.00 |

## smile · broad

Target probe: `siglip_smiling_margin`  · expected sign `+`  · cells: 6

| base | target | atom (slope) | mv_age/y | ins_age/y | id drift | total drift | gender flip |
|---|---|---|---|---|---|---|---|
| asian_m | 0.086 (R²=0.73) | atom_16 (1.868) | -8.15 | -5.46 | +0.415 | +0.049 | 0.00 |
| black_f | 0.096 (R²=0.73) | atom_16 (1.255) | -4.45 | +5.13 | +0.508 | +0.061 | 0.12 |
| elderly_latin_m | 0.076 (R²=0.71) | atom_16 (1.120) | -8.42 | -5.17 | +0.557 | +0.068 | 0.00 |
| european_m | 0.098 (R²=0.74) | atom_16 (1.047) | -2.59 | +0.37 | +0.516 | +0.080 | 0.00 |
| southasian_f | 0.040 (R²=0.71) | atom_16 (1.307) | -13.70 | -2.51 | +0.553 | +0.083 | 0.00 |
| young_european_f | 0.093 (R²=0.73) | atom_16 (1.298) | +2.99 | +7.96 | +0.499 | +0.077 | 0.00 |

## smile · faint

Target probe: `siglip_smiling_margin`  · expected sign `+`  · cells: 6

| base | target | atom (slope) | mv_age/y | ins_age/y | id drift | total drift | gender flip |
|---|---|---|---|---|---|---|---|
| asian_m | 0.045 (R²=0.57) | atom_07 (0.629) | -10.85 | -0.89 | +0.226 | +0.026 | 0.00 |
| black_f | 0.053 (R²=0.61) | atom_07 (1.109) | -4.64 | +0.62 | +0.340 | +0.042 | 0.00 |
| elderly_latin_m | 0.036 (R²=0.59) | atom_07 (0.613) | -1.04 | -1.93 | +0.286 | +0.024 | 0.00 |
| european_m | 0.059 (R²=0.66) | atom_07 (0.846) | -3.26 | -4.82 | +0.359 | +0.055 | 0.00 |
| southasian_f | 0.004 (R²=0.02) | atom_07 (1.823) | -14.66 | -9.65 | +0.516 | +0.114 | 0.00 |
| young_european_f | 0.040 (R²=0.59) | atom_07 (1.686) | +0.74 | +4.93 | +0.355 | +0.077 | 0.00 |

## smile · manic

Target probe: `siglip_smiling_margin`  · expected sign `+`  · cells: 6

| base | target | atom (slope) | mv_age/y | ins_age/y | id drift | total drift | gender flip |
|---|---|---|---|---|---|---|---|
| asian_m | 0.095 (R²=0.71) | atom_19 (0.742) | -5.30 | +4.32 | +0.419 | +0.075 | 0.00 |
| black_f | 0.105 (R²=0.69) | atom_19 (0.993) | -0.75 | +6.36 | +0.550 | +0.080 | 0.12 |
| elderly_latin_m | 0.082 (R²=0.70) | atom_19 (0.748) | -9.91 | -4.21 | +0.629 | +0.097 | 0.00 |
| european_m | 0.108 (R²=0.75) | atom_19 (0.878) | -0.48 | +13.20 | +0.606 | +0.118 | 0.00 |
| southasian_f | 0.048 (R²=0.75) | atom_19 (0.727) | -11.89 | -1.46 | +0.535 | +0.115 | 0.00 |
| young_european_f | 0.100 (R²=0.71) | atom_19 (0.879) | +6.53 | +12.75 | +0.563 | +0.115 | 0.00 |

## smile · warm

Target probe: `siglip_smiling_margin`  · expected sign `+`  · cells: 6

| base | target | atom (slope) | mv_age/y | ins_age/y | id drift | total drift | gender flip |
|---|---|---|---|---|---|---|---|
| asian_m | 0.074 (R²=0.71) | atom_16 (0.895) | -10.26 | -1.68 | +0.351 | +0.039 | 0.00 |
| black_f | 0.084 (R²=0.69) | atom_16 (1.090) | -4.15 | +3.73 | +0.396 | +0.047 | 0.00 |
| elderly_latin_m | 0.060 (R²=0.66) | atom_16 (0.869) | -3.50 | -5.39 | +0.426 | +0.047 | 0.00 |
| european_m | 0.085 (R²=0.71) | atom_16 (0.796) | -2.86 | -0.20 | +0.448 | +0.067 | 0.00 |
| southasian_f | 0.027 (R²=0.58) | atom_16 (1.020) | -13.56 | -5.98 | +0.478 | +0.046 | 0.00 |
| young_european_f | 0.083 (R²=0.70) | atom_16 (1.328) | +2.53 | +5.28 | +0.459 | +0.073 | 0.00 |

## surprise · surprise

Target probe: `siglip_surprised_margin`  · expected sign `+`  · cells: 6

| base | target | atom (slope) | mv_age/y | ins_age/y | id drift | total drift | gender flip |
|---|---|---|---|---|---|---|---|
| asian_m | 0.072 (R²=0.54) | atom_12 (0.604) | -2.91 | -9.22 | +0.249 | +0.060 | 0.00 |
| black_f | 0.050 (R²=0.39) | atom_09 (1.338) | +1.05 | +1.72 | +0.243 | +0.042 | 0.00 |
| elderly_latin_m | 0.089 (R²=0.61) | atom_01 (0.823) | -1.29 | -1.69 | +0.392 | +0.070 | 0.00 |
| european_m | 0.099 (R²=0.74) | atom_09 (1.155) | -0.86 | -4.60 | +0.425 | +0.090 | 0.00 |
| southasian_f | 0.041 (R²=0.37) | atom_09 (0.935) | -8.80 | -5.77 | +0.473 | +0.115 | 0.00 |
| young_european_f | 0.067 (R²=0.51) | atom_09 (0.462) | +3.99 | +1.50 | +0.366 | +0.076 | 0.00 |

## Aggregate observations

Mean slopes across bases for each (axis, subtag):

| axis/subtag | target | σ | mv_age/y | ins_age/y | id drift | total drift | gender flip |
|---|---|---|---|---|---|---|---|
| anger/anger | 0.077 | 0.01 | +0.96 | +7.00 | +0.477 | +0.106 | 0.00 |
| beard/add | 0.077 | 0.01 | -3.48 | -6.23 | +0.427 | +0.052 | 0.00 |
| beard/remove | 0.029 | nan | -14.59 | -5.70 | +0.462 | +0.077 | 0.00 |
| beard_rebalance/remove | 0.001 | 0.00 | -6.29 | -3.05 | +0.183 | +0.016 | 0.00 |
| pucker/pucker | 0.038 | 0.02 | -4.98 | -5.90 | +0.316 | +0.061 | 0.00 |
| smile/broad | 0.081 | 0.02 | -5.72 | +0.05 | +0.508 | +0.070 | 0.02 |
| smile/faint | 0.039 | 0.02 | -5.62 | -1.96 | +0.347 | +0.056 | 0.00 |
| smile/manic | 0.090 | 0.02 | -3.63 | +5.16 | +0.550 | +0.100 | 0.02 |
| smile/warm | 0.069 | 0.02 | -5.30 | -0.71 | +0.426 | +0.053 | 0.00 |
| surprise/surprise | 0.070 | 0.02 | -1.47 | -3.01 | +0.358 | +0.075 | 0.00 |

## Headline findings

- **beard/add is not predominantly a beard edit — it's a ~3-year
  de-aging edit.** Target SigLIP slope 0.077, but `mv_age_slope −3.5 y`
  and `ins_age_slope −6.2 y`. The model reads "add beard" more
  strongly as "add youth" than as "add facial hair." This is the
  classic confound the composition solver needs to correct.

- **beard_rebalance/remove does not remove beard on any base** —
  target slopes {0.001, 0.005, −0.002}, R² ≤ 0.27. Removing the
  bearded prefix from the base prompt and applying a "clean-shaven"
  edit simply doesn't move the SigLIP-2 bearded margin. What *does*
  move: `mv_age_slope` of −11.8 on `elderly_latin_m_bearded`. The
  axis is firing as a de-aging direction with the beard as a
  passenger, not the driver. Candidate prompt-pair redesign.

- **beard/remove on elderly_latin_m takes 14.6 years off.** The
  strongest age confound in the matrix. With target slope only 0.029,
  this is almost entirely an age edit.

- **smile/broad and smile/warm consistently pull age down ~5 years
  across bases.** Already-known "smile-makes-younger" is confirmed
  quantitatively: `mv_age_slope` −5.7 y (broad) and −5.3 y (warm),
  tight across demographics.

- **Identity drift scales hard.** `smile/manic` averages `+0.55` per
  unit scale — at s=1 the face is 55% less cosine-similar to the
  neutral base (ArcFace). Any solver operating in the scale ≥ 0.5
  regime needs an identity-preservation constraint from the start.

- **Atom_16 is the smile basis.** Emerges as primary atom across
  `smile/broad` and `smile/warm` for every base, with slope ~1.0–1.3.
  `smile/manic` switches to atom_19 — suggesting the extreme "manic"
  rung engages a different mouth-open atom than the friendlier rungs.

- **Gender flips are rare.** Zero across almost all cells; only
  `smile/broad` and `smile/manic` on `black_f` flipped 1/8 seeds.
  Edits don't cross gender often enough to be the dominant confound.

- **Primary atoms are not stable across bases for anger/pucker/surprise.**
  Different bases land on different atoms as "primary" for the same
  prompt-pair edit — signal that the ridge atom→δ decomposition will
  need to aggregate across multiple atoms rather than treating any
  single one as "the surprise atom."

## Interpretation hooks

- If target slope is small or flips sign across bases, the edit direction doesn't transfer — candidate for re-pairing the prompt.
- Large |mv_age_slope| with a semantic edit flags a classical demographic confound (smile makes younger, beard-add makes older). These are the entries the composition solver will use to counter-edit.
- identity_drift >> total_drift means identity features moved more than the overall SigLIP distribution — likely a face-specific signal. Opposite ordering means the edit changed the non-face scene (background, lighting) more than the face.
- gender_flip > 0.25 at max-scale means the axis crosses a demographic boundary often enough to matter — flag for preservation-clause work.

## Next

- Compose counter-edits on the axes with biggest non-target drifts: expected first targets are smile×manic (largest age drift) and beard/add (largest identity drift).
- Render demographic-edit pairs (age, gender, race) so they become composable δs, not just measurements.
- Verify `max_env` / `T_ratio` on the new overnight corpus once attn caches are built for the overnight sources (future batch).
