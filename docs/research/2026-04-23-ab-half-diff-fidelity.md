---
status: live
topic: metrics-and-direction-quality
summary: Fidelity analysis on `ab_half_diff` (A/B pair asymmetry) instead of `delta_mix`. Tests whether emotion axes that looked rank-~10 on delta_mix recover axis-specific structure on the asymmetry channel.
---

# ab_half_diff injection fidelity — 2026-04-23

Rerun of the per-channel fidelity probe on `ab_half_diff.mean_d` instead of `delta_mix.mean_d`. `ab_half_diff = (attn_a - attn_b) / 2`, so it isolates what the A/B pair halves *disagree* on, while `delta_mix = (attn_a + attn_b)/2 - attn_base` captures their joint push.

## Per-axis reconstruction captured fraction

| axis | N | peak (step, block) | mean-norm | k=0 | k=1 | k=3 | k=10 |
|---|---|---|---|---|---|---|---|
| smile_inphase | 330 | (5, double_18) | 8.05 | 0.745 [0.701, 0.777] | 0.779 [0.749, 0.799] | 0.817 [0.781, 0.827] | 0.885 [0.878, 0.900] |
| jaw_inphase | 330 | (5, double_18) | 11.96 | 0.742 [0.721, 0.804] | 0.793 [0.764, 0.823] | 0.835 [0.814, 0.854] | 0.905 [0.898, 0.915] |
| alpha_interp_attn | 660 | (5, double_18) | 8.91 | 0.749 [0.723, 0.775] | 0.795 [0.753, 0.808] | 0.816 [0.801, 0.835] | 0.874 [0.861, 0.887] |
| anger_rebalance | 180 | (7, double_18) | 10.20 | 0.375 [0.338, 0.400] | 0.502 [0.406, 0.564] | 0.692 [0.668, 0.791] | 0.888 [0.872, 0.903] |
| surprise_rebalance | 180 | (7, double_18) | 10.19 | 0.355 [0.324, 0.381] | 0.445 [0.357, 0.552] | 0.665 [0.637, 0.776] | 0.883 [0.862, 0.896] |
| disgust_rebalance | 181 | (7, double_18) | 10.16 | 0.327 [0.294, 0.369] | 0.426 [0.313, 0.513] | 0.666 [0.625, 0.738] | 0.878 [0.861, 0.901] |
| pucker_rebalance | 180 | (7, double_18) | 10.53 | 0.352 [0.327, 0.393] | 0.452 [0.376, 0.577] | 0.667 [0.634, 0.783] | 0.886 [0.865, 0.901] |
| lip_press_rebalance | 180 | (7, double_18) | 9.90 | 0.335 [0.304, 0.378] | 0.445 [0.344, 0.543] | 0.653 [0.630, 0.781] | 0.884 [0.866, 0.898] |

## Effective dimensionality

| axis | ev1 | ev2 | ev3 | k80 | k95 |
|---|---|---|---|---|---|
| smile_inphase | 0.223 | 0.151 | 0.120 | 9 | 19 |
| jaw_inphase | 0.263 | 0.181 | 0.113 | 8 | 18 |
| alpha_interp_attn | 0.266 | 0.128 | 0.099 | 12 | 33 |
| anger_rebalance | 0.355 | 0.235 | 0.174 | 4 | 7 |
| surprise_rebalance | 0.337 | 0.233 | 0.179 | 4 | 8 |
| disgust_rebalance | 0.372 | 0.254 | 0.156 | 4 | 7 |
| pucker_rebalance | 0.321 | 0.239 | 0.174 | 4 | 8 |
| lip_press_rebalance | 0.334 | 0.238 | 0.173 | 4 | 7 |

## Cross-axis cosine of axis-mean `v_mean` (normalised)

| | smile_inphase | jaw_inphase | alpha_interp_attn | anger_rebalance | surprise_rebalance | disgust_rebalance | pucker_rebalance | lip_press_rebalance |
|---|---|---|---|---|---|---|---|---|
| smile_inphase | +1.000 | +0.623 | +0.535 | -0.066 | -0.063 | -0.077 | -0.022 | -0.027 |
| jaw_inphase | +0.623 | +1.000 | +0.711 | -0.264 | -0.202 | -0.250 | -0.127 | -0.090 |
| alpha_interp_attn | +0.535 | +0.711 | +1.000 | -0.192 | -0.161 | -0.261 | -0.087 | -0.045 |
| anger_rebalance | -0.066 | -0.264 | -0.192 | +1.000 | +0.918 | +0.854 | +0.828 | +0.851 |
| surprise_rebalance | -0.063 | -0.202 | -0.161 | +0.918 | +1.000 | +0.869 | +0.898 | +0.910 |
| disgust_rebalance | -0.077 | -0.250 | -0.261 | +0.854 | +0.869 | +1.000 | +0.857 | +0.839 |
| pucker_rebalance | -0.022 | -0.127 | -0.087 | +0.828 | +0.898 | +0.857 | +1.000 | +0.941 |
| lip_press_rebalance | -0.027 | -0.090 | -0.045 | +0.851 | +0.910 | +0.839 | +0.941 | +1.000 |

Mean off-diagonal cos = **+0.311** (min -0.264, max +0.941)

## Comparison to delta_mix

Delta_mix baseline (from 2026-04-23-injection-fidelity.md): emotion cluster shared cos 0.97–0.99, mean-only captured 0.07–0.10, k=10 plateau ~0.80.

This file tests whether `ab_half_diff` separates emotions that `delta_mix` collapses.
