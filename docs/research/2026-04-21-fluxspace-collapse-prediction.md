---
status: live
topic: metrics-and-direction-quality
summary: max_env predictor with ratio threshold generalizes across seeds (93% accuracy) to predict safe scale windows without dense sweeps.
---

# Predicting FluxSpace Edit Collapse from One Measurement Pass

**Date:** 2026-04-21
**Axis:** glasses (FluxSpaceEditPair, attention-cache averaging)
**Base prompt:** adult Latin American woman, portrait.

## Question

For a given attribute direction we extract via attention-cache averaging, there
exists a safe range of scale `s` where the edit is clean, flanked by collapse
regions on both sides (grain/mosaic on the positive side, feature wash-out on
the negative). **Can we predict those edges from a single scale=1.0 render
plus a small calibration corpus, without sweeping?**

## Setup

- **Pair node.** `FluxSpaceEditPair` with prompts A (`"A person wearing
  thick-rimmed eyeglasses."`) and B (full base-prompt splice). At `mix_b=0.5`
  the opposing age confounds cancel. See `project_fluxspace_pair_averaging`.
- **Calibration corpus.** 10 base-only Flux renders covering diverse
  demographics (see `CALIBRATION_PROMPTS` in `src/demographic_pc/fluxspace_metrics.py`).
  Each run captures `FluxSpaceBaseMeasure` per-D reductions (mean/rms/Frobenius)
  for every (block, step). 10 runs → per-(block, step) prompt-to-prompt
  μ_d, σ_d of `attn_base.mean_d`.
- **Measurement.** `FluxSpaceEditPair` with `measure_path`, seeds 2026 and 4242,
  at scale=1.0 mix_b=0.5. Captures `attn_base`, `delta_mix` per-D.
- **Ground truth.** Dense scale sweep s ∈ {−2, −1.5, −1, −0.9, ..., −0.3, 0,
  0.3, ..., 1.3, 1.4, ..., 2, 2.5, 3} × {2026, 4242} = 56 renders, manually
  labeled safe/collapse from the montage.

Observed safe window: **s ∈ [−0.6, +1.3]** (asymmetric).

## Predictors

All are analytic in `s` from a single measurement pass:

- `steered_d = attn_base_d + s · delta_mix_d`
- `d_mahal(s)²   = Σ_d ((steered_d − μ_d) / σ_d)²`   — quadratic in s
- `zmax(s)     = max_d |(steered_d − μ_d) / σ_d|`   — piecewise linear
- `max_env(s)  = max_d |steered_d|`                   — piecewise linear
- `fro_ratio(s) = ‖steered‖₂ / ‖attn_base_cal‖_F_avg` — quadratic in s

Per predictor we aggregate by `max` over (block, step).

## Results

Threshold fit by sweeping candidate values from the seed's own metric range and
maximising safe/collapse separation accuracy:

| metric | in-sample acc | cross-seed acc | fit-T (2026 / 4242) |
|---|---|---|---|
| **max_env_max** | **92.86%** | **82.14%** | 8.93 / 8.43 |
| d_mahal_max | 78.57% | 71–79% | 172.26 / 165.78 |
| zmax_max | 78.57% | 75–79% | 44.35 / 41.18 |
| fro_ratio_max | 71.43% | 64–71% | 1.51 / 1.58 |

**Predicted safe range via analytic s-scan at fitted T (max_env):**

- seed 2026: s ∈ [−0.40, +1.30]
- seed 4242: s ∈ [−0.35, +1.25]

**Ground truth: s ∈ [−0.6, +1.3].**

## Findings

- **Positive edge pinned to grid resolution.** Both seeds predict the upper
  edge within 0.05 of the measured transition s=+1.30→+1.40.
- **Negative edge under-predicted by ~0.2.** Metric grows faster on the
  negative side than the observed image-space collapse onset. Conservative
  failure (flags usable scales) rather than dangerous (missing collapse).
- **max_env is seed-stable.** Threshold varies by 6% across seeds; the cross-
  seed accuracy is 82% (vs in-sample 93%) — one seed of measurement is enough
  to fit a threshold that generalises.
- **d_mahal is too pessimistic.** Diagonal Σ built from prompt-to-prompt spread
  in `attn_base.mean_d` is tighter than the actual on-manifold tolerance around
  any single prompt. Collapses the predicted safe window to [+0.75, +1.30].
- **fro_ratio and zmax under-perform.** Scalar Frobenius ignores per-dim
  structure; zmax is dominated by a few outlier dims with large σ.

## Recipe (preflight for any new axis/prompt)

1. Run `fluxspace_metrics --calibrate` once per base-prompt domain (10 renders,
   shareable across axes).
2. Render one `FluxSpaceEditPair` measurement at `s=1, mix_b=0.5` with
   `measure_path`.
3. For each s in a dense grid, compute `max_env(s) = max over (block, step, d)
   of |attn_base + s·δ|`. Pick the safe range as the contiguous interval
   around `s=0` where `max_env ≤ T` (default **T=8.5**; conservative).
4. Sweep at s=1.0 first; extend cautiously in either direction up to the
   predicted edge. Expect positive-edge prediction to be tight; negative-edge
   to leave ~0.2 of margin on the table.

## Limits

- **One axis, one base-prompt.** Glasses on a Latin American woman only.
  Generalisation to smile, beard, other demographics is the next experiment.
- **Fitted T, not universal T.** T=8.5 is this specific pair's max_env value;
  other axes may have different absolute scales. The fitting procedure itself
  generalises; the absolute threshold may not.
- **Grid is 0.05 wide.** Sub-grid precision not validated.

## Artefacts

- Calibration pickles: `output/demographic_pc/fluxspace_metrics/calibration/*.pkl`
- Measurement pickles: `output/demographic_pc/fluxspace_metrics/measurement/pair_s{2026,4242}.pkl`
- Analysis table: `output/demographic_pc/fluxspace_metrics/analysis/predictions.json`
- Dense sweep montage: `output/demographic_pc/fluxspace_node_test/glasses/dense_sweep_montage.png`
- Pipeline: `src/demographic_pc/fluxspace_metrics.py`
- Custom node: `~/w/ComfyUI/custom_nodes/demographic_pc_fluxspace/__init__.py`
