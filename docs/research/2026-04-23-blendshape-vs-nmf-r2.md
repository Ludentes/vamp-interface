---
status: live
topic: metrics-and-direction-quality
summary: Side-by-side comparison of per-blendshape ridge R² against NMF-component R² at each corpus's peak block. Tests whether a finer-grained per-blendshape library would outperform the 8-component NMF library.
---

# Per-blendshape vs NMF component R² — 2026-04-23

Fitting ridge directly on individual ARKit blendshapes at each corpus's peak block, side-by-side with the NMF-component R² from `au_library.npz`.

## Canonical AU blendshapes — best R² across corpora

| blendshape | smile_inphase | jaw_inphase | alpha_interp_attn | anger_rebalance | surprise_rebalance | disgust_rebalance | pucker_rebalance | lip_press_rebalance |
|---|---|---|---|---|---|---|---|---|
| mouthSmileLeft | +0.808 (σ=0.43) | +0.654 (σ=0.10) | +0.715 (σ=0.09) | -0.493 (σ=0.07) | +0.432 (σ=0.08) | -0.541 (σ=0.06) | +0.469 (σ=0.01) | -0.387 (σ=0.08) |
| mouthSmileRight | +0.814 (σ=0.43) | +0.633 (σ=0.11) | +0.674 (σ=0.10) | -0.637 (σ=0.05) | +0.335 (σ=0.07) | +0.431 (σ=0.06) | — | -0.404 (σ=0.06) |
| mouthPucker | +0.747 (σ=0.21) | +0.667 (σ=0.29) | +0.517 (σ=0.03) | +0.633 (σ=0.30) | +0.330 (σ=0.33) | +0.343 (σ=0.19) | +0.744 (σ=0.32) | +0.717 (σ=0.33) |
| mouthFunnel | +0.808 (σ=0.05) | +0.486 (σ=0.09) | +0.754 (σ=0.07) | +0.617 (σ=0.02) | +0.235 (σ=0.09) | — | +0.578 (σ=0.04) | +0.236 (σ=0.04) |
| mouthPressLeft | +0.782 (σ=0.10) | +0.862 (σ=0.10) | +0.748 (σ=0.07) | +0.523 (σ=0.09) | +0.188 (σ=0.08) | +0.644 (σ=0.17) | +0.403 (σ=0.03) | +0.180 (σ=0.06) |
| mouthPressRight | +0.745 (σ=0.08) | +0.633 (σ=0.05) | +0.702 (σ=0.07) | +0.555 (σ=0.06) | -0.067 (σ=0.05) | +0.629 (σ=0.10) | +0.280 (σ=0.01) | +0.372 (σ=0.02) |
| mouthUpperUpLeft | +0.748 (σ=0.38) | +0.593 (σ=0.05) | +0.749 (σ=0.33) | -0.041 (σ=0.11) | — | — | — | +0.774 (σ=0.01) |
| mouthUpperUpRight | +0.751 (σ=0.38) | +0.630 (σ=0.06) | +0.754 (σ=0.33) | -0.151 (σ=0.11) | — | — | +0.701 (σ=0.01) | +0.766 (σ=0.01) |
| mouthLowerDownLeft | +0.721 (σ=0.06) | +0.726 (σ=0.27) | +0.816 (σ=0.31) | -0.826 (σ=0.03) | — | — | — | — |
| mouthLowerDownRight | +0.744 (σ=0.11) | +0.744 (σ=0.27) | +0.825 (σ=0.34) | -0.829 (σ=0.04) | -0.269 (σ=0.01) | — | — | — |
| jawOpen | +0.759 (σ=0.08) | +0.818 (σ=0.39) | +0.864 (σ=0.27) | +0.232 (σ=0.05) | -0.131 (σ=0.17) | +0.389 (σ=0.02) | +0.464 (σ=0.03) | +0.523 (σ=0.02) |
| browInnerUp | +0.932 (σ=0.14) | +0.815 (σ=0.37) | +0.816 (σ=0.17) | +0.203 (σ=0.02) | +0.714 (σ=0.23) | +0.268 (σ=0.09) | +0.844 (σ=0.17) | +0.920 (σ=0.14) |
| browOuterUpLeft | +0.939 (σ=0.17) | +0.819 (σ=0.35) | +0.827 (σ=0.22) | +0.417 (σ=0.11) | +0.762 (σ=0.25) | +0.881 (σ=0.15) | +0.941 (σ=0.22) | +0.960 (σ=0.21) |
| browOuterUpRight | +0.928 (σ=0.11) | +0.831 (σ=0.32) | +0.797 (σ=0.15) | +0.393 (σ=0.07) | +0.738 (σ=0.20) | +0.696 (σ=0.09) | +0.936 (σ=0.16) | +0.948 (σ=0.16) |
| browDownLeft | +0.931 (σ=0.27) | +0.857 (σ=0.17) | +0.856 (σ=0.23) | +0.703 (σ=0.24) | +0.402 (σ=0.11) | +0.272 (σ=0.27) | +0.864 (σ=0.24) | +0.907 (σ=0.26) |
| browDownRight | +0.939 (σ=0.26) | +0.884 (σ=0.18) | +0.869 (σ=0.23) | +0.737 (σ=0.26) | +0.307 (σ=0.12) | +0.422 (σ=0.29) | +0.847 (σ=0.25) | +0.894 (σ=0.27) |
| noseSneerLeft | — | — | — | — | — | — | — | — |
| noseSneerRight | — | — | — | — | — | — | — | — |
| eyeSquintLeft | +0.933 (σ=0.15) | +0.858 (σ=0.17) | +0.828 (σ=0.16) | +0.598 (σ=0.12) | +0.688 (σ=0.17) | +0.681 (σ=0.14) | +0.914 (σ=0.17) | +0.939 (σ=0.18) |
| eyeSquintRight | +0.926 (σ=0.17) | +0.867 (σ=0.15) | +0.776 (σ=0.15) | +0.717 (σ=0.16) | +0.686 (σ=0.16) | +0.817 (σ=0.16) | +0.894 (σ=0.16) | +0.896 (σ=0.17) |
| eyeBlinkLeft | +0.898 (σ=0.08) | +0.922 (σ=0.06) | +0.804 (σ=0.18) | +0.733 (σ=0.08) | +0.722 (σ=0.07) | +0.808 (σ=0.08) | +0.890 (σ=0.08) | +0.878 (σ=0.08) |
| eyeBlinkRight | +0.898 (σ=0.08) | +0.906 (σ=0.05) | +0.797 (σ=0.17) | +0.651 (σ=0.08) | +0.545 (σ=0.05) | +0.818 (σ=0.06) | +0.914 (σ=0.07) | +0.911 (σ=0.08) |
| cheekPuff | — | — | — | — | — | — | — | — |
| cheekSquintLeft | — | — | — | — | — | — | — | — |
| cheekSquintRight | — | — | — | — | — | — | — | — |

## NMF component R² vs its dominant blendshape

For each component, the dominant blendshape is `argmax(H[c])`. Δ = per-blendshape R² − per-component R². Positive Δ means the direct blendshape direction beats the NMF component at this corpus.

### C0 dominant = `jawOpen`

| corpus | NMF R² | blendshape R² | Δ | σ(bs) |
|---|---|---|---|---|
| smile_inphase | +0.680 | +0.759 | +0.080 | 0.08 |
| jaw_inphase | +0.809 | +0.818 | +0.009 | 0.39 |
| alpha_interp_attn | +0.834 | +0.864 | +0.030 | 0.27 |
| anger_rebalance | -0.220 | +0.232 | +0.452 | 0.05 |
| surprise_rebalance | -0.153 | -0.131 | +0.022 | 0.17 |
| disgust_rebalance | +0.564 | +0.389 | -0.175 | 0.02 |
| pucker_rebalance | +0.085 | +0.464 | +0.379 | 0.03 |
| lip_press_rebalance | +0.179 | +0.523 | +0.343 | 0.02 |

### C1 dominant = `eyeLookDownRight`

| corpus | NMF R² | blendshape R² | Δ | σ(bs) |
|---|---|---|---|---|
| smile_inphase | +0.859 | +0.820 | -0.039 | 0.10 |
| jaw_inphase | +0.744 | +0.813 | +0.068 | 0.13 |
| alpha_interp_attn | +0.821 | +0.805 | -0.016 | 0.20 |
| anger_rebalance | +0.643 | +0.466 | -0.177 | 0.07 |
| surprise_rebalance | +0.597 | +0.654 | +0.056 | 0.09 |
| disgust_rebalance | +0.462 | +0.227 | -0.234 | 0.07 |
| pucker_rebalance | +0.713 | +0.828 | +0.115 | 0.10 |
| lip_press_rebalance | +0.853 | +0.878 | +0.026 | 0.09 |

### C2 dominant = `browOuterUpLeft`

| corpus | NMF R² | blendshape R² | Δ | σ(bs) |
|---|---|---|---|---|
| smile_inphase | +0.941 | +0.939 | -0.002 | 0.17 |
| jaw_inphase | +0.839 | +0.819 | -0.020 | 0.35 |
| alpha_interp_attn | +0.829 | +0.827 | -0.002 | 0.22 |
| anger_rebalance | +0.161 | +0.417 | +0.256 | 0.11 |
| surprise_rebalance | +0.743 | +0.762 | +0.019 | 0.25 |
| disgust_rebalance | +0.704 | +0.881 | +0.177 | 0.15 |
| pucker_rebalance | +0.940 | +0.941 | +0.001 | 0.22 |
| lip_press_rebalance | +0.961 | +0.960 | -0.001 | 0.21 |

### C3 dominant = `browDownRight`

| corpus | NMF R² | blendshape R² | Δ | σ(bs) |
|---|---|---|---|---|
| smile_inphase | +0.945 | +0.939 | -0.006 | 0.26 |
| jaw_inphase | +0.883 | +0.884 | +0.000 | 0.18 |
| alpha_interp_attn | +0.873 | +0.869 | -0.004 | 0.23 |
| anger_rebalance | +0.771 | +0.737 | -0.034 | 0.26 |
| surprise_rebalance | +0.380 | +0.307 | -0.073 | 0.12 |
| disgust_rebalance | +0.537 | +0.422 | -0.115 | 0.29 |
| pucker_rebalance | +0.864 | +0.847 | -0.017 | 0.25 |
| lip_press_rebalance | +0.910 | +0.894 | -0.016 | 0.27 |

### C4 dominant = `mouthPucker`

| corpus | NMF R² | blendshape R² | Δ | σ(bs) |
|---|---|---|---|---|
| smile_inphase | +0.755 | +0.747 | -0.008 | 0.21 |
| jaw_inphase | +0.675 | +0.667 | -0.008 | 0.29 |
| alpha_interp_attn | +0.565 | +0.517 | -0.048 | 0.03 |
| anger_rebalance | +0.650 | +0.633 | -0.017 | 0.30 |
| surprise_rebalance | +0.324 | +0.330 | +0.006 | 0.33 |
| disgust_rebalance | +0.422 | +0.343 | -0.079 | 0.19 |
| pucker_rebalance | +0.753 | +0.744 | -0.009 | 0.32 |
| lip_press_rebalance | +0.726 | +0.717 | -0.009 | 0.33 |

### C5 dominant = `mouthUpperUpLeft`

| corpus | NMF R² | blendshape R² | Δ | σ(bs) |
|---|---|---|---|---|
| smile_inphase | +0.753 | +0.748 | -0.005 | 0.38 |
| jaw_inphase | +0.575 | +0.593 | +0.018 | 0.05 |
| alpha_interp_attn | +0.757 | +0.749 | -0.008 | 0.33 |
| anger_rebalance | -0.110 | -0.041 | +0.069 | 0.11 |
| surprise_rebalance | +0.611 | — | — | 0.01 |
| disgust_rebalance | -0.295 | — | — | 0.00 |
| pucker_rebalance | +0.704 | — | — | 0.01 |
| lip_press_rebalance | +0.766 | +0.774 | +0.008 | 0.01 |

### C6 dominant = `mouthSmileLeft`

| corpus | NMF R² | blendshape R² | Δ | σ(bs) |
|---|---|---|---|---|
| smile_inphase | +0.807 | +0.808 | +0.001 | 0.43 |
| jaw_inphase | +0.636 | +0.654 | +0.018 | 0.10 |
| alpha_interp_attn | +0.652 | +0.715 | +0.063 | 0.09 |
| anger_rebalance | -0.656 | -0.493 | +0.163 | 0.07 |
| surprise_rebalance | +0.390 | +0.432 | +0.042 | 0.08 |
| disgust_rebalance | +0.043 | -0.541 | -0.584 | 0.06 |
| pucker_rebalance | +0.511 | +0.469 | -0.042 | 0.01 |
| lip_press_rebalance | -0.401 | -0.387 | +0.014 | 0.08 |

### C7 dominant = `eyeSquintLeft`

| corpus | NMF R² | blendshape R² | Δ | σ(bs) |
|---|---|---|---|---|
| smile_inphase | +0.838 | +0.933 | +0.095 | 0.15 |
| jaw_inphase | +0.853 | +0.858 | +0.005 | 0.17 |
| alpha_interp_attn | +0.839 | +0.828 | -0.012 | 0.16 |
| anger_rebalance | +0.563 | +0.598 | +0.036 | 0.12 |
| surprise_rebalance | +0.696 | +0.688 | -0.008 | 0.17 |
| disgust_rebalance | +0.061 | +0.681 | +0.620 | 0.14 |
| pucker_rebalance | +0.833 | +0.914 | +0.081 | 0.17 |
| lip_press_rebalance | +0.833 | +0.939 | +0.106 | 0.18 |

## Full 52 scan, σ-filtered, ranked by best R²

Kept 38 blendshapes with σ(y) > 0.05 in at least one corpus. CSV at `output/demographic_pc/blendshape_vs_nmf_r2.csv`. Top 20 shown here.

| blendshape | max σ | smile_inphase | jaw_inphase | alpha_interp_attn | anger_rebalance | surprise_rebalance | disgust_rebalance | pucker_rebalance | lip_press_rebalance |
|---|---|---|---|---|---|---|---|---|---|
| browOuterUpLeft | 0.35 | +0.939 | +0.819 | +0.827 | +0.417 | +0.762 | +0.881 | +0.941 | +0.960 |
| browOuterUpRight | 0.32 | +0.928 | +0.831 | +0.797 | +0.393 | +0.738 | +0.696 | +0.936 | +0.948 |
| browDownRight | 0.29 | +0.939 | +0.884 | +0.869 | +0.737 | +0.307 | +0.422 | +0.847 | +0.894 |
| eyeSquintLeft | 0.18 | +0.933 | +0.858 | +0.828 | +0.598 | +0.688 | +0.681 | +0.914 | +0.939 |
| browInnerUp | 0.37 | +0.932 | +0.815 | +0.816 | +0.203 | +0.714 | +0.268 | +0.844 | +0.920 |
| browDownLeft | 0.27 | +0.931 | +0.857 | +0.856 | +0.703 | +0.402 | +0.272 | +0.864 | +0.907 |
| eyeSquintRight | 0.17 | +0.926 | +0.867 | +0.776 | +0.717 | +0.686 | +0.817 | +0.894 | +0.896 |
| eyeLookInLeft | 0.06 | +0.924 | +0.832 | +0.486 | +0.077 | +0.492 | -0.233 | +0.411 | +0.485 |
| eyeLookOutRight | 0.06 | +0.906 | +0.923 | +0.501 | +0.063 | +0.552 | -0.052 | +0.561 | +0.689 |
| eyeBlinkLeft | 0.18 | +0.898 | +0.922 | +0.804 | +0.733 | +0.722 | +0.808 | +0.890 | +0.878 |
| eyeBlinkRight | 0.17 | +0.898 | +0.906 | +0.797 | +0.651 | +0.545 | +0.818 | +0.914 | +0.911 |
| eyeLookDownRight | 0.20 | +0.820 | +0.813 | +0.805 | +0.466 | +0.654 | +0.227 | +0.828 | +0.878 |
| eyeLookDownLeft | 0.20 | +0.833 | +0.794 | +0.801 | +0.428 | +0.654 | +0.010 | +0.821 | +0.873 |
| eyeLookOutLeft | 0.18 | +0.807 | +0.713 | +0.869 | +0.332 | +0.606 | +0.575 | +0.655 | +0.431 |
| eyeWideRight | 0.19 | — | +0.865 | +0.808 | — | +0.650 | — | — | — |
| jawOpen | 0.39 | +0.759 | +0.818 | +0.864 | +0.232 | -0.131 | +0.389 | +0.464 | +0.523 |
| eyeWideLeft | 0.13 | — | +0.863 | +0.739 | — | +0.683 | — | — | — |
| mouthPressLeft | 0.17 | +0.782 | +0.862 | +0.748 | +0.523 | +0.188 | +0.644 | +0.403 | +0.180 |
| eyeLookInRight | 0.19 | +0.840 | +0.818 | +0.862 | +0.335 | +0.592 | +0.629 | +0.643 | +0.665 |
| mouthShrugUpper | 0.05 | +0.847 | +0.695 | +0.702 | +0.151 | +0.728 | +0.656 | +0.640 | +0.802 |

## How to read

- **Δ mostly positive** → per-blendshape library beats per-component. Each blendshape direction is finer than the component's mixture, and an injection library should store per-blendshape `w` vectors.

- **Δ mostly zero** → NMF component already captures the same signal. Use the 8-component library; it's smaller and cleaner.

- **Δ negative** → the NMF component's broader loading is actually more predictable than any single blendshape (averaging noise across co-activating AUs helps).

- **σ(bs) small** → per-blendshape R² is dominated by noise; ignore.
