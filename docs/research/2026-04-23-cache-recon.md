---
status: live
topic: metrics-and-direction-quality
summary: Recon pass on the compact attention cache. Locates per-axis energy, measures cross-axis distinguishability, portability across bases, and effective dimensionality at peak block.
---

# Attention-cache recon — 2026-04-23

Ran four probes over `models/blendshape_nmf/attn_cache/<tag>/` (8 axis/subtag corpora, delta_mix shape `(N, 16 steps, 57 blocks, 3072 D)` fp16).

## Probe A — peak location per axis

| axis | peak step | peak block | peak Fro |
|---|---|---|---|
| smile_inphase | 6 | single_34 | 58.613 |
| jaw_inphase | 7 | single_34 | 64.474 |
| alpha_interp_attn | 8 | single_34 | 58.510 |
| anger_rebalance | 8 | single_34 | 54.996 |
| surprise_rebalance | 8 | single_34 | 60.503 |
| disgust_rebalance | 10 | single_34 | 67.964 |
| pucker_rebalance | 9 | single_34 | 65.053 |
| lip_press_rebalance | 8 | single_34 | 66.792 |

Heatmap: ![](images/2026-04-23-cache-recon-heatmaps.png)

## Probe B — cross-axis cosine at peaks

Normalized mean δ at each axis's own peak (step, block). High off-diagonal = axes share direction; block-diagonal = distinct.

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

Mean off-diagonal cos = **+0.778** (min +0.476, max +0.995).

## Probe C — portability across bases (smile_inphase)

Per-base mean δ at smile_inphase's peak (step, block), cosine across 6 bases.

| | asian_m | black_f | elderly_latin_m | european_m | southasian_f | young_european_f |
|---|---|---|---|---|---|---|
| asian_m | +1.000 | +0.845 | +0.979 | +0.967 | +0.930 | +0.941 |
| black_f | +0.845 | +1.000 | +0.864 | +0.742 | +0.655 | +0.674 |
| elderly_latin_m | +0.979 | +0.864 | +1.000 | +0.939 | +0.903 | +0.908 |
| european_m | +0.967 | +0.742 | +0.939 | +1.000 | +0.981 | +0.988 |
| southasian_f | +0.930 | +0.655 | +0.903 | +0.981 | +1.000 | +0.989 |
| young_european_f | +0.941 | +0.674 | +0.908 | +0.988 | +0.989 | +1.000 |

Mean off-diagonal cos = **+0.887** (min +0.655, max +0.989).

## Probe D — effective dimensionality at peak block

PCA on `(N, 3072)` at peak (step, block). `k80` / `k95` = rank needed to explain 80% / 95% of variance.

| axis | ev1 | ev2 | ev3 | k80 | k95 |
|---|---|---|---|---|---|
| smile_inphase | 0.728 | 0.084 | 0.028 | 2 | 15 |
| jaw_inphase | 0.660 | 0.075 | 0.037 | 4 | 22 |
| alpha_interp_attn | 0.713 | 0.050 | 0.032 | 4 | 26 |
| anger_rebalance | 0.876 | 0.068 | 0.019 | 1 | 3 |
| surprise_rebalance | 0.949 | 0.017 | 0.014 | 1 | 2 |
| disgust_rebalance | 0.951 | 0.018 | 0.012 | 1 | 1 |
| pucker_rebalance | 0.946 | 0.027 | 0.012 | 1 | 2 |
| lip_press_rebalance | 0.952 | 0.019 | 0.014 | 1 | 1 |

## Interpretation hooks

- **If B is block-diagonal** → axes are distinct directions → a per-axis library is tractable. **If B shows broad off-diagonal mass** → many axes are entangled in the same subspace → portable library is harder.

- **If C mean-off-diag cos is high (>0.7)** → a single cross-base smile direction exists → future renders can skip live prompt-pair passes. **If low (<0.3)** → `(axis, base)` keying (the dictionary's current choice) is correct.

- **If D `ev1 > 0.6`** → axis is effectively rank-1, one vector stores it. **If k80 > 10** → richer structure; library would have to be a basis, not a vector.

## Next steps

- Extend `cache_attn_features.py` to cover `promptpair_iterate/` sweeps (the adjacency-validated smile/age/race runs) and re-run this recon on the validated corpus.

- If axes look portable (B + C pass thresholds): prototype a `FluxSpaceInjectCached` node that consumes a cached δ and reproduces the prompt-pair edit without the 2N+1 forward passes.
