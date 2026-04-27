---
status: live
topic: metrics-and-direction-quality
summary: Offline fidelity analysis for cached-δ injection. Tests whether a small per-axis library (mean + top-k PCs) can reconstruct individual sample δs at single_34, whether libraries generalise across bases, and how much injecting one axis leaks into another.
---

# Cached-δ injection fidelity — 2026-04-23

Decides whether it is worth building a `FluxSpaceInjectCached` ComfyUI node that skips the `2N+1` forward-pass overhead of `FluxSpaceEditPair` by consuming a precomputed per-axis vector.

Working at each axis's peak `(step, block)` from the recon. Cache only stores channel-reduced summaries (`mean_d` shape 3072, one value per channel after averaging over tokens), so this analysis says nothing about whether a per-channel shift reproduces the live-tensor edit — it only says whether the per-channel δ itself is low-rank and portable.

## Per-axis reconstruction captured fraction

Per-sample `1 − ‖δ − δ̂‖ / ‖δ‖` using `{v_mean} ∪ top-k PCs`. Reports median and [p25, p75] over N samples.

| axis | N | peak (step, block) | k=0 (mean-only) | k=1 | k=3 | k=10 |
|---|---|---|---|---|---|---|
| smile_inphase | 330 | (6, single_34) | 0.720 [0.644, 0.776] | 0.833 [0.799, 0.853] | 0.857 [0.838, 0.877] | 0.905 [0.894, 0.914] |
| jaw_inphase | 330 | (7, single_34) | 0.809 [0.772, 0.837] | 0.846 [0.823, 0.865] | 0.872 [0.853, 0.886] | 0.912 [0.900, 0.922] |
| alpha_interp_attn | 660 | (8, single_34) | 0.779 [0.727, 0.809] | 0.841 [0.821, 0.855] | 0.858 [0.843, 0.874] | 0.899 [0.889, 0.909] |
| anger_rebalance | 180 | (8, single_34) | 0.097 [0.003, 0.639] | 0.564 [0.110, 0.734] | 0.643 [0.290, 0.854] | 0.757 [0.483, 0.935] |
| surprise_rebalance | 180 | (8, single_34) | 0.068 [0.002, 0.772] | 0.586 [0.128, 0.830] | 0.681 [0.442, 0.881] | 0.756 [0.475, 0.949] |
| disgust_rebalance | 181 | (10, single_34) | 0.090 [0.002, 0.796] | 0.617 [0.115, 0.884] | 0.703 [0.421, 0.905] | 0.790 [0.468, 0.951] |
| pucker_rebalance | 180 | (9, single_34) | 0.097 [0.002, 0.756] | 0.681 [0.077, 0.875] | 0.777 [0.420, 0.909] | 0.810 [0.464, 0.954] |
| lip_press_rebalance | 180 | (8, single_34) | 0.071 [0.003, 0.777] | 0.610 [0.123, 0.874] | 0.700 [0.435, 0.918] | 0.784 [0.487, 0.955] |

## Base-holdout portability (k=3)

Build library from 5 bases, evaluate on held-out base. `in_median` vs `out_median` = train vs test captured fraction.

### smile_inphase

| heldout base | in_median | out_median | gap |
|---|---|---|---|
| asian_m | 0.864 | 0.803 | +0.061 |
| black_f | 0.863 | 0.645 | +0.218 |
| elderly_latin_m | 0.865 | 0.763 | +0.103 |
| european_m | 0.859 | 0.845 | +0.015 |
| southasian_f | 0.860 | 0.823 | +0.036 |
| young_european_f | 0.865 | 0.821 | +0.044 |

### anger_rebalance

| heldout base | in_median | out_median | gap |
|---|---|---|---|
| asian_m | 0.640 | 0.590 | +0.049 |
| black_f | 0.637 | 0.539 | +0.098 |
| elderly_latin_m | 0.712 | 0.529 | +0.183 |
| european_m | 0.606 | 0.589 | +0.016 |
| southasian_f | 0.620 | 0.620 | -0.000 |
| young_european_f | 0.626 | 0.612 | +0.014 |

## Cross-axis leakage — cosine of axis-mean vectors

If we inject unit vector `v_A`, its projection onto `v_B` is `cos(A, B)`. This is the side-effect an injection node would produce on the *other* axis's peak channel.

| | smile_inphase | jaw_inphase | alpha_interp_attn | anger_rebalance | surprise_rebalance | disgust_rebalance | pucker_rebalance | lip_press_rebalance |
|---|---|---|---|---|---|---|---|---|
| smile_inphase | +1.000 | +0.938 | +0.985 | +0.621 | +0.500 | +0.520 | +0.500 | +0.476 |
| jaw_inphase | +0.938 | +1.000 | +0.970 | +0.764 | +0.679 | +0.688 | +0.682 | +0.656 |
| alpha_interp_attn | +0.985 | +0.970 | +1.000 | +0.676 | +0.571 | +0.586 | +0.572 | +0.547 |
| anger_rebalance | +0.621 | +0.764 | +0.676 | +1.000 | +0.978 | +0.978 | +0.969 | +0.966 |
| surprise_rebalance | +0.500 | +0.679 | +0.571 | +0.978 | +1.000 | +0.989 | +0.992 | +0.995 |
| disgust_rebalance | +0.520 | +0.688 | +0.586 | +0.978 | +0.989 | +1.000 | +0.990 | +0.989 |
| pucker_rebalance | +0.500 | +0.682 | +0.572 | +0.969 | +0.992 | +0.990 | +1.000 | +0.995 |
| lip_press_rebalance | +0.476 | +0.656 | +0.547 | +0.966 | +0.995 | +0.989 | +0.995 | +1.000 |

## Interpretation

- **If k=0 (mean-only) already captures ≥0.7 median** → a single per-axis vector is enough; rank-1 library is viable. This is the strongest possible outcome for the injection-node path.

- **If k=0 is poor but k=1..3 catches up** → we need a small basis, not a vector. Still tractable; library entries become `(mean, basis)` pairs.

- **If k=10 still leaves a large residual** → samples carry per-sample structure the axis doesn't share. Injection would reproduce only the axis-common part, which may or may not be the part that moves pixels.

- **Base-holdout gap <0.1** → one library works across bases → a flat `{axis → vector}` store is enough. **Gap >0.2** → needs per-base entries, matching the dictionary's `(axis, base)` keying.

- **Leakage table**: the high non-smile/non-smile cosines from the recon (0.97–0.99) say an injection of one emotion shifts the channel mean of the others by almost as much — expected, since the recon already showed they share the same direction.

## Caveat

All of this is on channel-reduced summaries. The real question — does injecting `scale · v` (broadcast across tokens) at `single_34` during a forward pass reproduce the prompt-pair edit's pixel output — is not answerable from cache alone. Positive signals here license building the ComfyUI node; negative signals kill it. Either outcome is a real result.
