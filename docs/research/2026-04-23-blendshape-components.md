---
status: live
topic: metrics-and-direction-quality
summary: Decompose the 52-d ARKit blendshape space (PCA + ICA) and regress attention channels against the discovered components. Better than regressing each of 52 blendshapes separately because the output space is highly collinear.
---

# Blendshape-space decomposition + δ regression — 2026-04-23

The 52-d ARKit blendshape output is strongly collinear (previous per-blendshape regression showed top-15 AUs across axes are mostly brow/eye). This file first decomposes the blendshape matrix itself, then regresses attention against the discovered components.

## PCA of blendshape matrix (N_all × 52 stacked across 8 corpora)

| PC | var ratio | cum var |
|---|---|---|
| PC0 | 0.466 | 0.466 |
| PC1 | 0.213 | 0.679 |
| PC2 | 0.086 | 0.764 |
| PC3 | 0.054 | 0.819 |
| PC4 | 0.035 | 0.854 |
| PC5 | 0.029 | 0.882 |
| PC6 | 0.023 | 0.905 |
| PC7 | 0.020 | 0.925 |

## PCA components: top blendshape loadings + per-corpus δ-regression R²

### PC0 — variance 0.466

Top loadings: mouthSmileRight(+0.50), mouthSmileLeft(+0.50), mouthUpperUpRight(+0.38), mouthUpperUpLeft(+0.38), mouthLowerDownRight(+0.25), mouthPucker(-0.22)

| corpus | CV R² | σ(score) |
|---|---|---|
| smile_inphase | +0.841 | 0.781 |
| jaw_inphase | +0.762 | 0.262 |
| alpha_interp_attn | +0.802 | 0.423 |
| anger_rebalance | +0.082 | 0.191 |
| surprise_rebalance | +0.536 | 0.146 |
| disgust_rebalance | +0.449 | 0.089 |
| pucker_rebalance | +0.832 | 0.105 |
| lip_press_rebalance | +0.586 | 0.140 |

### PC1 — variance 0.213

Top loadings: browOuterUpLeft(+0.40), browInnerUp(+0.35), browDownRight(-0.31), browOuterUpRight(+0.31), browDownLeft(-0.30), jawOpen(+0.29)

| corpus | CV R² | σ(score) |
|---|---|---|
| smile_inphase | +0.959 | 0.365 |
| jaw_inphase | +0.880 | 0.714 |
| alpha_interp_attn | +0.888 | 0.499 |
| anger_rebalance | +0.776 | 0.293 |
| surprise_rebalance | +0.683 | 0.383 |
| disgust_rebalance | +0.783 | 0.351 |
| pucker_rebalance | +0.929 | 0.415 |
| lip_press_rebalance | +0.954 | 0.429 |

### PC2 — variance 0.086

Top loadings: mouthPucker(+0.43), jawOpen(-0.39), browDownRight(-0.37), browDownLeft(-0.35), mouthLowerDownRight(-0.30), mouthLowerDownLeft(-0.30)

| corpus | CV R² | σ(score) |
|---|---|---|
| smile_inphase | +0.855 | 0.297 |
| jaw_inphase | +0.803 | 0.392 |
| alpha_interp_attn | +0.872 | 0.357 |
| anger_rebalance | +0.736 | 0.280 |
| surprise_rebalance | +0.401 | 0.208 |
| disgust_rebalance | +0.861 | 0.283 |
| pucker_rebalance | +0.862 | 0.301 |
| lip_press_rebalance | +0.883 | 0.324 |

### PC3 — variance 0.054

Top loadings: eyeLookDownLeft(+0.32), mouthPucker(-0.32), eyeLookDownRight(+0.32), browInnerUp(+0.30), browOuterUpLeft(+0.27), mouthUpperUpLeft(-0.27)

| corpus | CV R² | σ(score) |
|---|---|---|
| smile_inphase | +0.630 | 0.148 |
| jaw_inphase | +0.770 | 0.273 |
| alpha_interp_attn | +0.820 | 0.386 |
| anger_rebalance | +0.761 | 0.193 |
| surprise_rebalance | +0.666 | 0.200 |
| disgust_rebalance | +0.683 | 0.141 |
| pucker_rebalance | +0.872 | 0.198 |
| lip_press_rebalance | +0.862 | 0.191 |

### PC4 — variance 0.035

Top loadings: mouthPucker(+0.44), browDownLeft(+0.42), browDownRight(+0.40), mouthUpperUpLeft(+0.34), mouthUpperUpRight(+0.34), browOuterUpLeft(+0.19)

| corpus | CV R² | σ(score) |
|---|---|---|
| smile_inphase | +0.751 | 0.272 |
| jaw_inphase | +0.626 | 0.187 |
| alpha_interp_attn | +0.696 | 0.242 |
| anger_rebalance | +0.479 | 0.183 |
| surprise_rebalance | +0.546 | 0.211 |
| disgust_rebalance | -0.390 | 0.196 |
| pucker_rebalance | +0.667 | 0.163 |
| lip_press_rebalance | +0.635 | 0.167 |

### PC5 — variance 0.029

Top loadings: mouthPucker(+0.57), mouthUpperUpLeft(-0.33), mouthUpperUpRight(-0.31), mouthSmileLeft(+0.30), mouthSmileRight(+0.30), browDownRight(+0.20)

| corpus | CV R² | σ(score) |
|---|---|---|
| smile_inphase | +0.523 | 0.230 |
| jaw_inphase | +0.564 | 0.160 |
| alpha_interp_attn | +0.711 | 0.229 |
| anger_rebalance | +0.398 | 0.166 |
| surprise_rebalance | +0.070 | 0.181 |
| disgust_rebalance | -1.004 | 0.123 |
| pucker_rebalance | +0.552 | 0.129 |
| lip_press_rebalance | +0.188 | 0.130 |

### PC6 — variance 0.023

Top loadings: eyeSquintRight(+0.32), eyeBlinkLeft(+0.31), mouthLowerDownRight(+0.31), mouthLowerDownLeft(+0.29), eyeBlinkRight(+0.28), eyeLookDownLeft(+0.28)

| corpus | CV R² | σ(score) |
|---|---|---|
| smile_inphase | +0.756 | 0.126 |
| jaw_inphase | +0.568 | 0.135 |
| alpha_interp_attn | +0.710 | 0.247 |
| anger_rebalance | +0.630 | 0.141 |
| surprise_rebalance | +0.525 | 0.142 |
| disgust_rebalance | +0.754 | 0.122 |
| pucker_rebalance | +0.782 | 0.116 |
| lip_press_rebalance | +0.709 | 0.116 |

### PC7 — variance 0.020

Top loadings: eyeLookInRight(+0.65), eyeLookOutLeft(+0.63), eyeSquintRight(+0.17), eyeLookOutRight(-0.15), mouthSmileRight(-0.13), eyeSquintLeft(+0.11)

| corpus | CV R² | σ(score) |
|---|---|---|
| smile_inphase | +0.825 | 0.144 |
| jaw_inphase | +0.807 | 0.121 |
| alpha_interp_attn | +0.863 | 0.248 |
| anger_rebalance | +0.493 | 0.111 |
| surprise_rebalance | +0.657 | 0.126 |
| disgust_rebalance | +0.787 | 0.119 |
| pucker_rebalance | +0.720 | 0.099 |
| lip_press_rebalance | +0.721 | 0.100 |

## ICA components: top blendshape loadings + δ R²

### IC0

Top loadings: eyeLookInRight(+0.67), eyeLookOutLeft(+0.66), eyeLookOutRight(-0.15), browInnerUp(+0.12), eyeLookInLeft(-0.11), mouthStretchLeft(+0.10)

| corpus | CV R² | σ(score) |
|---|---|---|
| smile_inphase | +0.855 | 0.711 |
| jaw_inphase | +0.794 | 0.675 |
| alpha_interp_attn | +0.869 | 1.494 |
| anger_rebalance | +0.364 | 0.532 |
| surprise_rebalance | +0.625 | 0.696 |
| disgust_rebalance | +0.695 | 0.594 |
| pucker_rebalance | +0.671 | 0.513 |
| lip_press_rebalance | +0.656 | 0.532 |

### IC1

Top loadings: mouthPucker(+0.60), browDownLeft(+0.49), browDownRight(+0.47), eyeSquintRight(+0.17), browOuterUpLeft(+0.15), browInnerUp(+0.15)

| corpus | CV R² | σ(score) |
|---|---|---|
| smile_inphase | +0.827 | 0.963 |
| jaw_inphase | +0.609 | 0.792 |
| alpha_interp_attn | +0.812 | 0.857 |
| anger_rebalance | +0.493 | 0.875 |
| surprise_rebalance | +0.325 | 0.919 |
| disgust_rebalance | -0.349 | 1.010 |
| pucker_rebalance | +0.648 | 0.797 |
| lip_press_rebalance | +0.574 | 0.821 |

### IC2

Top loadings: browInnerUp(-0.46), browOuterUpLeft(-0.43), browOuterUpRight(-0.39), mouthPucker(+0.35), jawOpen(-0.27), browDownRight(-0.17)

| corpus | CV R² | σ(score) |
|---|---|---|
| smile_inphase | +0.857 | 0.408 |
| jaw_inphase | +0.853 | 1.653 |
| alpha_interp_attn | +0.838 | 0.753 |
| anger_rebalance | +0.468 | 0.329 |
| surprise_rebalance | +0.526 | 0.690 |
| disgust_rebalance | +0.608 | 0.255 |
| pucker_rebalance | +0.890 | 0.577 |
| lip_press_rebalance | +0.866 | 0.510 |

### IC3

Top loadings: eyeBlinkLeft(-0.42), eyeLookDownLeft(-0.41), eyeLookDownRight(-0.41), eyeBlinkRight(-0.40), eyeSquintRight(-0.33), eyeSquintLeft(-0.28)

| corpus | CV R² | σ(score) |
|---|---|---|
| smile_inphase | +0.748 | 0.470 |
| jaw_inphase | +0.714 | 0.502 |
| alpha_interp_attn | +0.814 | 1.495 |
| anger_rebalance | +0.746 | 0.723 |
| surprise_rebalance | +0.818 | 0.650 |
| disgust_rebalance | +0.805 | 0.618 |
| pucker_rebalance | +0.856 | 0.558 |
| lip_press_rebalance | +0.892 | 0.547 |

### IC4

Top loadings: mouthSmileLeft(+0.45), mouthSmileRight(+0.45), mouthUpperUpLeft(-0.37), mouthUpperUpRight(-0.36), eyeLookDownLeft(-0.24), eyeLookDownRight(-0.23)

| corpus | CV R² | σ(score) |
|---|---|---|
| smile_inphase | +0.578 | 1.270 |
| jaw_inphase | +0.672 | 0.625 |
| alpha_interp_attn | +0.708 | 1.325 |
| anger_rebalance | +0.247 | 0.548 |
| surprise_rebalance | +0.312 | 0.403 |
| disgust_rebalance | +0.460 | 0.460 |
| pucker_rebalance | +0.552 | 0.250 |
| lip_press_rebalance | +0.044 | 0.359 |

### IC5

Top loadings: mouthPucker(+0.83), mouthUpperUpRight(+0.21), mouthSmileRight(+0.19), mouthUpperUpLeft(+0.18), mouthSmileLeft(+0.17), eyeLookOutLeft(-0.16)

| corpus | CV R² | σ(score) |
|---|---|---|
| smile_inphase | +0.740 | 0.970 |
| jaw_inphase | +0.655 | 0.774 |
| alpha_interp_attn | +0.845 | 0.701 |
| anger_rebalance | +0.666 | 0.878 |
| surprise_rebalance | +0.397 | 0.971 |
| disgust_rebalance | +0.357 | 0.675 |
| pucker_rebalance | +0.752 | 0.864 |
| lip_press_rebalance | +0.707 | 0.902 |

### IC6

Top loadings: browDownRight(+0.52), browDownLeft(+0.50), mouthPucker(-0.40), mouthUpperUpRight(+0.27), mouthUpperUpLeft(+0.26), eyeLookInRight(-0.25)

| corpus | CV R² | σ(score) |
|---|---|---|
| smile_inphase | +0.875 | 1.104 |
| jaw_inphase | +0.831 | 0.650 |
| alpha_interp_attn | +0.807 | 0.740 |
| anger_rebalance | +0.735 | 0.887 |
| surprise_rebalance | +0.395 | 0.512 |
| disgust_rebalance | +0.684 | 0.843 |
| pucker_rebalance | +0.854 | 0.829 |
| lip_press_rebalance | +0.897 | 0.907 |

### IC7

Top loadings: mouthLowerDownRight(+0.39), mouthLowerDownLeft(+0.39), jawOpen(+0.38), eyeSquintRight(+0.29), mouthUpperUpLeft(-0.28), mouthUpperUpRight(-0.25)

| corpus | CV R² | σ(score) |
|---|---|---|
| smile_inphase | +0.713 | 0.819 |
| jaw_inphase | +0.606 | 0.913 |
| alpha_interp_attn | +0.712 | 1.352 |
| anger_rebalance | +0.508 | 0.501 |
| surprise_rebalance | +0.558 | 0.776 |
| disgust_rebalance | +0.705 | 0.419 |
| pucker_rebalance | +0.904 | 0.664 |
| lip_press_rebalance | +0.859 | 0.607 |

## PCA blendshape components vs 11 existing NMF atoms on δ

Cosine between each PCA component's δ-weight vector (ridge-fit at smile_inphase peak block) and each NMF atom's direction (averaged across its 24 sites, unit-normalised).

| | atom_00 | atom_01 | atom_02 | atom_03 | atom_04 | atom_05 | atom_06 | atom_07 | atom_08 | atom_09 | atom_10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PC0 | +0.03 | +0.04 | +0.13 | -0.01 | -0.02 | -0.05 | -0.02 | -0.01 | -0.02 | -0.01 | -0.06 |
| PC1 | -0.04 | -0.00 | -0.09 | -0.03 | -0.00 | -0.01 | -0.01 | +0.03 | +0.05 | +0.00 | -0.00 |
| PC2 | -0.00 | +0.04 | +0.09 | -0.06 | -0.04 | -0.01 | -0.03 | +0.00 | +0.08 | +0.01 | -0.04 |
| PC3 | +0.02 | -0.01 | +0.09 | -0.01 | +0.02 | +0.05 | +0.00 | -0.01 | -0.00 | +0.00 | +0.00 |
| PC4 | -0.02 | +0.02 | -0.05 | +0.02 | -0.01 | -0.09 | +0.01 | -0.02 | +0.02 | +0.00 | -0.02 |
| PC5 | +0.00 | -0.01 | +0.12 | -0.02 | -0.01 | +0.07 | -0.00 | -0.02 | +0.08 | -0.00 | -0.04 |
| PC6 | -0.03 | -0.03 | -0.16 | +0.00 | +0.01 | +0.02 | +0.02 | +0.02 | +0.01 | -0.00 | +0.03 |
| PC7 | +0.01 | +0.00 | -0.13 | +0.03 | +0.03 | -0.02 | +0.01 | +0.08 | -0.01 | -0.00 | +0.03 |


0 cells with |cos|>0.3; max |cos| = 0.162.

## How to read this

- **PCA/ICA var ratio tail**: how many independent blendshape patterns actually exist. If PC0 eats >50%, the output is nearly rank-1; most per-blendshape signal was noise around that one direction.

- **Top loadings**: interpret each component. A single clean AU = one blendshape dominant. A pattern ('smile') = multiple blendshapes co-load with same sign. Mixed signs = open/close style opposition.

- **R² per corpus**: tells us which corpora encode each pattern in attention. If `smile_inphase` scores high on 'smile-pattern' PC but anger corpora score low on it, attention is axis-specific.

- **ICA vs PCA**: ICA components should be closer to single AUs (non-Gaussian, parts-based). If ICA gives cleaner loadings and similar R², prefer ICA as the library basis.

- **PC ↔ NMF atom cos**: |cos|>0.5 means attention-space NMF and output-space PCA found the same direction. |cos|<0.3 everywhere means the two bases answer different questions.
