---
status: live
topic: metrics-and-direction-quality
summary: ARKit blendshape regression against cached delta_mix channels. Finds which of 52 blendshapes (≈FACS AUs) each axis corpus encodes linearly from attention, and how the weight vectors relate across AUs.
---

# Blendshape / AU regression from cached δ — 2026-04-23

For each axis corpus, at its peak `(step, block)`: ridge-fit each of the 52 ARKit blendshape scores (MediaPipe FaceLandmarker output; maps to FACS AUs) against the 3072-dim delta_mix channel vector. 5-fold CV R² reports how much of each blendshape's variation is encoded in the attention channel means at that block.

## smile_inphase — peak (6, single_34), N=330

Top-15 blendshapes by CV R² (blendshape / R² / σ(y) / cos(w, v_axis)):

| blendshape | R² | σ(y) | cos(w, v_axis) |
|---|---|---|---|
| browDownRight | +0.939 | 0.264 | -0.001 |
| browOuterUpLeft | +0.939 | 0.172 | +0.016 |
| eyeSquintLeft | +0.933 | 0.149 | -0.018 |
| browInnerUp | +0.932 | 0.143 | +0.009 |
| browDownLeft | +0.931 | 0.271 | +0.004 |
| browOuterUpRight | +0.928 | 0.106 | +0.019 |
| eyeSquintRight | +0.926 | 0.166 | -0.015 |
| eyeLookInLeft | +0.924 | 0.053 | -0.013 |
| eyeLookOutRight | +0.906 | 0.057 | -0.023 |
| eyeBlinkLeft | +0.898 | 0.078 | -0.007 |
| eyeBlinkRight | +0.898 | 0.080 | -0.003 |
| mouthShrugUpper | +0.847 | 0.035 | +0.011 |
| eyeWideLeft | +0.842 | 0.004 | +0.011 |
| eyeWideRight | +0.841 | 0.004 | +0.014 |
| mouthStretchLeft | +0.840 | 0.084 | +0.008 |

Cosine between top-10 blendshape weight vectors:

| | browDownRight | browOuterUpLeft | eyeSquintLeft | browInnerUp | browDownLeft | browOuterUpRight | eyeSquintRight | eyeLookInLeft | eyeLookOutRight | eyeBlinkLeft |
|---|---|---|---|---|---|---|---|---|---|---|
| browDownRight | +1.00 | -0.11 | +0.28 | -0.19 | +0.97 | -0.10 | +0.31 | -0.04 | +0.08 | +0.29 |
| browOuterUpLeft | -0.11 | +1.00 | -0.31 | +0.63 | -0.10 | +0.87 | -0.22 | +0.09 | +0.05 | -0.32 |
| eyeSquintLeft | +0.28 | -0.31 | +1.00 | -0.17 | +0.29 | -0.26 | +0.68 | -0.05 | -0.04 | +0.35 |
| browInnerUp | -0.19 | +0.63 | -0.17 | +1.00 | -0.19 | +0.59 | -0.10 | +0.10 | -0.02 | -0.29 |
| browDownLeft | +0.97 | -0.10 | +0.29 | -0.19 | +1.00 | -0.08 | +0.26 | -0.03 | +0.10 | +0.35 |
| browOuterUpRight | -0.10 | +0.87 | -0.26 | +0.59 | -0.08 | +1.00 | -0.19 | +0.07 | +0.05 | -0.21 |
| eyeSquintRight | +0.31 | -0.22 | +0.68 | -0.10 | +0.26 | -0.19 | +1.00 | -0.02 | -0.03 | +0.47 |
| eyeLookInLeft | -0.04 | +0.09 | -0.05 | +0.10 | -0.03 | +0.07 | -0.02 | +1.00 | +0.90 | +0.07 |
| eyeLookOutRight | +0.08 | +0.05 | -0.04 | -0.02 | +0.10 | +0.05 | -0.03 | +0.90 | +1.00 | +0.12 |
| eyeBlinkLeft | +0.29 | -0.32 | +0.35 | -0.29 | +0.35 | -0.21 | +0.47 | +0.07 | +0.12 | +1.00 |

## jaw_inphase — peak (7, single_34), N=330

Top-15 blendshapes by CV R² (blendshape / R² / σ(y) / cos(w, v_axis)):

| blendshape | R² | σ(y) | cos(w, v_axis) |
|---|---|---|---|
| eyeLookOutRight | +0.923 | 0.049 | -0.010 |
| eyeBlinkLeft | +0.922 | 0.062 | -0.022 |
| eyeBlinkRight | +0.906 | 0.046 | -0.017 |
| browDownRight | +0.884 | 0.179 | +0.029 |
| eyeSquintRight | +0.867 | 0.148 | +0.014 |
| eyeWideRight | +0.865 | 0.186 | +0.008 |
| eyeWideLeft | +0.863 | 0.134 | -0.010 |
| mouthPressLeft | +0.862 | 0.104 | +0.021 |
| eyeSquintLeft | +0.858 | 0.165 | +0.009 |
| browDownLeft | +0.857 | 0.173 | +0.019 |
| eyeLookInLeft | +0.832 | 0.056 | +0.014 |
| browOuterUpRight | +0.831 | 0.316 | +0.004 |
| browOuterUpLeft | +0.819 | 0.345 | -0.003 |
| jawOpen | +0.818 | 0.388 | +0.002 |
| eyeLookInRight | +0.818 | 0.103 | +0.018 |

Cosine between top-10 blendshape weight vectors:

| | eyeLookOutRight | eyeBlinkLeft | eyeBlinkRight | browDownRight | eyeSquintRight | eyeWideRight | eyeWideLeft | mouthPressLeft | eyeSquintLeft | browDownLeft |
|---|---|---|---|---|---|---|---|---|---|---|
| eyeLookOutRight | +1.00 | +0.11 | +0.13 | +0.07 | +0.01 | -0.20 | -0.15 | +0.06 | +0.08 | +0.05 |
| eyeBlinkLeft | +0.11 | +1.00 | +0.88 | +0.36 | +0.63 | -0.20 | -0.07 | +0.11 | +0.59 | +0.37 |
| eyeBlinkRight | +0.13 | +0.88 | +1.00 | +0.31 | +0.60 | -0.15 | -0.06 | +0.09 | +0.55 | +0.34 |
| browDownRight | +0.07 | +0.36 | +0.31 | +1.00 | +0.32 | -0.19 | -0.24 | +0.21 | +0.38 | +0.92 |
| eyeSquintRight | +0.01 | +0.63 | +0.60 | +0.32 | +1.00 | -0.32 | -0.20 | +0.01 | +0.77 | +0.34 |
| eyeWideRight | -0.20 | -0.20 | -0.15 | -0.19 | -0.32 | +1.00 | +0.76 | +0.00 | -0.36 | -0.12 |
| eyeWideLeft | -0.15 | -0.07 | -0.06 | -0.24 | -0.20 | +0.76 | +1.00 | -0.02 | -0.27 | -0.18 |
| mouthPressLeft | +0.06 | +0.11 | +0.09 | +0.21 | +0.01 | +0.00 | -0.02 | +1.00 | -0.03 | +0.21 |
| eyeSquintLeft | +0.08 | +0.59 | +0.55 | +0.38 | +0.77 | -0.36 | -0.27 | -0.03 | +1.00 | +0.40 |
| browDownLeft | +0.05 | +0.37 | +0.34 | +0.92 | +0.34 | -0.12 | -0.18 | +0.21 | +0.40 | +1.00 |

## alpha_interp_attn — peak (8, single_34), N=660

Top-15 blendshapes by CV R² (blendshape / R² / σ(y) / cos(w, v_axis)):

| blendshape | R² | σ(y) | cos(w, v_axis) |
|---|---|---|---|
| eyeLookOutLeft | +0.869 | 0.185 | +0.000 |
| browDownRight | +0.869 | 0.228 | +0.008 |
| jawOpen | +0.864 | 0.268 | -0.004 |
| eyeLookInRight | +0.862 | 0.186 | +0.005 |
| browDownLeft | +0.856 | 0.228 | +0.008 |
| eyeSquintLeft | +0.828 | 0.157 | +0.000 |
| browOuterUpLeft | +0.827 | 0.216 | +0.005 |
| mouthLowerDownRight | +0.825 | 0.339 | -0.008 |
| mouthLowerDownLeft | +0.816 | 0.307 | -0.004 |
| browInnerUp | +0.816 | 0.168 | +0.010 |
| eyeWideRight | +0.808 | 0.055 | +0.001 |
| eyeLookDownRight | +0.805 | 0.199 | +0.009 |
| eyeBlinkLeft | +0.804 | 0.180 | +0.001 |
| eyeLookDownLeft | +0.801 | 0.198 | +0.010 |
| browOuterUpRight | +0.797 | 0.148 | +0.008 |

Cosine between top-10 blendshape weight vectors:

| | eyeLookOutLeft | browDownRight | jawOpen | eyeLookInRight | browDownLeft | eyeSquintLeft | browOuterUpLeft | mouthLowerDownRight | mouthLowerDownLeft | browInnerUp |
|---|---|---|---|---|---|---|---|---|---|---|
| eyeLookOutLeft | +1.00 | +0.13 | -0.01 | +0.91 | +0.14 | -0.14 | -0.03 | +0.07 | +0.06 | -0.00 |
| browDownRight | +0.13 | +1.00 | -0.11 | +0.03 | +0.96 | +0.38 | -0.46 | -0.10 | -0.12 | -0.38 |
| jawOpen | -0.01 | -0.11 | +1.00 | +0.05 | -0.09 | -0.13 | +0.10 | +0.63 | +0.61 | -0.02 |
| eyeLookInRight | +0.91 | +0.03 | +0.05 | +1.00 | +0.04 | -0.16 | +0.04 | +0.14 | +0.14 | +0.04 |
| browDownLeft | +0.14 | +0.96 | -0.09 | +0.04 | +1.00 | +0.40 | -0.49 | -0.07 | -0.10 | -0.39 |
| eyeSquintLeft | -0.14 | +0.38 | -0.13 | -0.16 | +0.40 | +1.00 | -0.50 | -0.18 | -0.22 | -0.44 |
| browOuterUpLeft | -0.03 | -0.46 | +0.10 | +0.04 | -0.49 | -0.50 | +1.00 | +0.05 | +0.10 | +0.71 |
| mouthLowerDownRight | +0.07 | -0.10 | +0.63 | +0.14 | -0.07 | -0.18 | +0.05 | +1.00 | +0.96 | -0.09 |
| mouthLowerDownLeft | +0.06 | -0.12 | +0.61 | +0.14 | -0.10 | -0.22 | +0.10 | +0.96 | +1.00 | -0.02 |
| browInnerUp | -0.00 | -0.38 | -0.02 | +0.04 | -0.39 | -0.44 | +0.71 | -0.09 | -0.02 | +1.00 |

## anger_rebalance — peak (8, single_34), N=180

Top-15 blendshapes by CV R² (blendshape / R² / σ(y) / cos(w, v_axis)):

| blendshape | R² | σ(y) | cos(w, v_axis) |
|---|---|---|---|
| mouthLeft | +0.778 | 0.008 | +0.008 |
| mouthRollUpper | +0.759 | 0.120 | -0.002 |
| browDownRight | +0.737 | 0.264 | +0.022 |
| eyeBlinkLeft | +0.733 | 0.080 | +0.008 |
| eyeSquintRight | +0.717 | 0.162 | +0.023 |
| browDownLeft | +0.703 | 0.239 | +0.014 |
| mouthRollLower | +0.696 | 0.170 | -0.001 |
| mouthDimpleRight | +0.652 | 0.009 | +0.005 |
| eyeBlinkRight | +0.651 | 0.085 | +0.006 |
| mouthPucker | +0.633 | 0.298 | -0.001 |
| eyeWideLeft | +0.622 | 0.005 | -0.009 |
| mouthFunnel | +0.617 | 0.015 | -0.004 |
| eyeSquintLeft | +0.598 | 0.124 | +0.025 |
| mouthPressRight | +0.555 | 0.057 | +0.013 |
| mouthClose | +0.544 | 0.004 | +0.001 |

Cosine between top-10 blendshape weight vectors:

| | mouthLeft | mouthRollUpper | browDownRight | eyeBlinkLeft | eyeSquintRight | browDownLeft | mouthRollLower | mouthDimpleRight | eyeBlinkRight | mouthPucker |
|---|---|---|---|---|---|---|---|---|---|---|
| mouthLeft | +1.00 | -0.02 | -0.12 | +0.01 | +0.01 | -0.04 | -0.00 | -0.09 | +0.01 | +0.42 |
| mouthRollUpper | -0.02 | +1.00 | +0.03 | +0.23 | +0.10 | +0.05 | +0.95 | +0.32 | +0.24 | -0.01 |
| browDownRight | -0.12 | +0.03 | +1.00 | +0.25 | +0.43 | +0.81 | -0.00 | +0.14 | +0.27 | -0.30 |
| eyeBlinkLeft | +0.01 | +0.23 | +0.25 | +1.00 | +0.61 | +0.11 | +0.15 | +0.06 | +0.83 | -0.05 |
| eyeSquintRight | +0.01 | +0.10 | +0.43 | +0.61 | +1.00 | +0.33 | +0.00 | -0.02 | +0.55 | -0.29 |
| browDownLeft | -0.04 | +0.05 | +0.81 | +0.11 | +0.33 | +1.00 | -0.01 | +0.14 | +0.21 | -0.27 |
| mouthRollLower | -0.00 | +0.95 | -0.00 | +0.15 | +0.00 | -0.01 | +1.00 | +0.32 | +0.16 | +0.02 |
| mouthDimpleRight | -0.09 | +0.32 | +0.14 | +0.06 | -0.02 | +0.14 | +0.32 | +1.00 | +0.14 | -0.10 |
| eyeBlinkRight | +0.01 | +0.24 | +0.27 | +0.83 | +0.55 | +0.21 | +0.16 | +0.14 | +1.00 | -0.13 |
| mouthPucker | +0.42 | -0.01 | -0.30 | -0.05 | -0.29 | -0.27 | +0.02 | -0.10 | -0.13 | +1.00 |

## surprise_rebalance — peak (8, single_34), N=180

Top-15 blendshapes by CV R² (blendshape / R² / σ(y) / cos(w, v_axis)):

| blendshape | R² | σ(y) | cos(w, v_axis) |
|---|---|---|---|
| browOuterUpLeft | +0.762 | 0.247 | +0.003 |
| browOuterUpRight | +0.738 | 0.196 | +0.001 |
| mouthShrugUpper | +0.728 | 0.033 | -0.004 |
| eyeBlinkLeft | +0.722 | 0.068 | -0.000 |
| browInnerUp | +0.714 | 0.234 | +0.009 |
| eyeSquintLeft | +0.688 | 0.172 | -0.002 |
| eyeSquintRight | +0.686 | 0.163 | +0.003 |
| eyeWideLeft | +0.683 | 0.036 | -0.001 |
| eyeLookDownLeft | +0.654 | 0.098 | +0.007 |
| eyeLookDownRight | +0.654 | 0.094 | +0.006 |
| eyeWideRight | +0.650 | 0.049 | +0.002 |
| eyeLookOutLeft | +0.606 | 0.081 | +0.012 |
| eyeLookUpRight | +0.602 | 0.070 | -0.005 |
| eyeLookInRight | +0.592 | 0.085 | +0.013 |
| mouthStretchLeft | +0.581 | 0.003 | -0.008 |

Cosine between top-10 blendshape weight vectors:

| | browOuterUpLeft | browOuterUpRight | mouthShrugUpper | eyeBlinkLeft | browInnerUp | eyeSquintLeft | eyeSquintRight | eyeWideLeft | eyeLookDownLeft | eyeLookDownRight |
|---|---|---|---|---|---|---|---|---|---|---|
| browOuterUpLeft | +1.00 | +0.90 | +0.02 | -0.50 | +0.71 | -0.55 | -0.43 | +0.48 | +0.18 | +0.18 |
| browOuterUpRight | +0.90 | +1.00 | +0.04 | -0.44 | +0.74 | -0.48 | -0.43 | +0.39 | +0.21 | +0.22 |
| mouthShrugUpper | +0.02 | +0.04 | +1.00 | -0.08 | -0.08 | -0.07 | -0.10 | -0.03 | +0.03 | +0.03 |
| eyeBlinkLeft | -0.50 | -0.44 | -0.08 | +1.00 | -0.34 | +0.66 | +0.74 | -0.21 | -0.24 | -0.24 |
| browInnerUp | +0.71 | +0.74 | -0.08 | -0.34 | +1.00 | -0.31 | -0.33 | +0.24 | +0.20 | +0.21 |
| eyeSquintLeft | -0.55 | -0.48 | -0.07 | +0.66 | -0.31 | +1.00 | +0.71 | -0.40 | -0.38 | -0.37 |
| eyeSquintRight | -0.43 | -0.43 | -0.10 | +0.74 | -0.33 | +0.71 | +1.00 | -0.26 | -0.35 | -0.37 |
| eyeWideLeft | +0.48 | +0.39 | -0.03 | -0.21 | +0.24 | -0.40 | -0.26 | +1.00 | +0.03 | +0.06 |
| eyeLookDownLeft | +0.18 | +0.21 | +0.03 | -0.24 | +0.20 | -0.38 | -0.35 | +0.03 | +1.00 | +0.99 |
| eyeLookDownRight | +0.18 | +0.22 | +0.03 | -0.24 | +0.21 | -0.37 | -0.37 | +0.06 | +0.99 | +1.00 |

## disgust_rebalance — peak (10, single_34), N=181

Top-15 blendshapes by CV R² (blendshape / R² / σ(y) / cos(w, v_axis)):

| blendshape | R² | σ(y) | cos(w, v_axis) |
|---|---|---|---|
| browOuterUpLeft | +0.881 | 0.149 | +0.002 |
| eyeBlinkRight | +0.818 | 0.064 | +0.004 |
| eyeSquintRight | +0.817 | 0.160 | +0.013 |
| eyeBlinkLeft | +0.808 | 0.079 | +0.002 |
| mouthUpperUpRight | +0.797 | 0.005 | +0.002 |
| mouthUpperUpLeft | +0.783 | 0.003 | +0.002 |
| eyeWideLeft | +0.765 | 0.004 | -0.002 |
| mouthRollUpper | +0.725 | 0.139 | -0.001 |
| browOuterUpRight | +0.696 | 0.088 | +0.004 |
| eyeSquintLeft | +0.681 | 0.141 | +0.008 |
| mouthShrugUpper | +0.656 | 0.035 | -0.000 |
| mouthRollLower | +0.654 | 0.195 | -0.001 |
| mouthPressLeft | +0.644 | 0.167 | +0.008 |
| eyeLookInRight | +0.629 | 0.058 | +0.012 |
| mouthPressRight | +0.629 | 0.103 | +0.007 |

Cosine between top-10 blendshape weight vectors:

| | browOuterUpLeft | eyeBlinkRight | eyeSquintRight | eyeBlinkLeft | mouthUpperUpRight | mouthUpperUpLeft | eyeWideLeft | mouthRollUpper | browOuterUpRight | eyeSquintLeft |
|---|---|---|---|---|---|---|---|---|---|---|
| browOuterUpLeft | +1.00 | -0.10 | -0.19 | -0.14 | +0.28 | +0.34 | +0.09 | -0.07 | +0.79 | -0.18 |
| eyeBlinkRight | -0.10 | +1.00 | +0.54 | +0.80 | -0.08 | -0.11 | -0.59 | +0.29 | -0.17 | +0.33 |
| eyeSquintRight | -0.19 | +0.54 | +1.00 | +0.61 | -0.08 | -0.12 | -0.65 | +0.15 | -0.27 | +0.56 |
| eyeBlinkLeft | -0.14 | +0.80 | +0.61 | +1.00 | +0.00 | -0.01 | -0.71 | +0.26 | -0.15 | +0.24 |
| mouthUpperUpRight | +0.28 | -0.08 | -0.08 | +0.00 | +1.00 | +0.95 | -0.04 | -0.10 | +0.23 | -0.24 |
| mouthUpperUpLeft | +0.34 | -0.11 | -0.12 | -0.01 | +0.95 | +1.00 | -0.00 | -0.09 | +0.31 | -0.30 |
| eyeWideLeft | +0.09 | -0.59 | -0.65 | -0.71 | -0.04 | -0.00 | +1.00 | -0.01 | +0.13 | -0.36 |
| mouthRollUpper | -0.07 | +0.29 | +0.15 | +0.26 | -0.10 | -0.09 | -0.01 | +1.00 | -0.10 | +0.16 |
| browOuterUpRight | +0.79 | -0.17 | -0.27 | -0.15 | +0.23 | +0.31 | +0.13 | -0.10 | +1.00 | -0.10 |
| eyeSquintLeft | -0.18 | +0.33 | +0.56 | +0.24 | -0.24 | -0.30 | -0.36 | +0.16 | -0.10 | +1.00 |

## pucker_rebalance — peak (9, single_34), N=180

Top-15 blendshapes by CV R² (blendshape / R² / σ(y) / cos(w, v_axis)):

| blendshape | R² | σ(y) | cos(w, v_axis) |
|---|---|---|---|
| browOuterUpLeft | +0.941 | 0.222 | -0.008 |
| browOuterUpRight | +0.936 | 0.163 | -0.007 |
| eyeBlinkRight | +0.914 | 0.073 | +0.006 |
| eyeSquintLeft | +0.914 | 0.169 | +0.021 |
| eyeSquintRight | +0.894 | 0.160 | +0.012 |
| eyeBlinkLeft | +0.890 | 0.080 | +0.003 |
| browDownLeft | +0.864 | 0.245 | +0.012 |
| browDownRight | +0.847 | 0.249 | +0.009 |
| browInnerUp | +0.844 | 0.165 | -0.003 |
| eyeLookDownRight | +0.828 | 0.098 | -0.010 |
| eyeLookDownLeft | +0.821 | 0.096 | -0.012 |
| eyeWideLeft | +0.772 | 0.006 | -0.006 |
| eyeWideRight | +0.752 | 0.007 | -0.009 |
| mouthPucker | +0.744 | 0.318 | +0.003 |
| mouthUpperUpLeft | +0.718 | 0.008 | -0.004 |

Cosine between top-10 blendshape weight vectors:

| | browOuterUpLeft | browOuterUpRight | eyeBlinkRight | eyeSquintLeft | eyeSquintRight | eyeBlinkLeft | browDownLeft | browDownRight | browInnerUp | eyeLookDownRight |
|---|---|---|---|---|---|---|---|---|---|---|
| browOuterUpLeft | +1.00 | +0.84 | -0.21 | -0.40 | -0.16 | -0.23 | -0.38 | -0.36 | +0.71 | +0.28 |
| browOuterUpRight | +0.84 | +1.00 | -0.15 | -0.26 | -0.20 | -0.10 | -0.30 | -0.31 | +0.61 | +0.35 |
| eyeBlinkRight | -0.21 | -0.15 | +1.00 | +0.55 | +0.49 | +0.86 | +0.38 | +0.41 | -0.24 | +0.12 |
| eyeSquintLeft | -0.40 | -0.26 | +0.55 | +1.00 | +0.62 | +0.56 | +0.51 | +0.48 | -0.26 | -0.24 |
| eyeSquintRight | -0.16 | -0.20 | +0.49 | +0.62 | +1.00 | +0.48 | +0.34 | +0.48 | +0.01 | -0.26 |
| eyeBlinkLeft | -0.23 | -0.10 | +0.86 | +0.56 | +0.48 | +1.00 | +0.30 | +0.31 | -0.23 | +0.18 |
| browDownLeft | -0.38 | -0.30 | +0.38 | +0.51 | +0.34 | +0.30 | +1.00 | +0.91 | -0.28 | -0.03 |
| browDownRight | -0.36 | -0.31 | +0.41 | +0.48 | +0.48 | +0.31 | +0.91 | +1.00 | -0.29 | -0.09 |
| browInnerUp | +0.71 | +0.61 | -0.24 | -0.26 | +0.01 | -0.23 | -0.28 | -0.29 | +1.00 | +0.08 |
| eyeLookDownRight | +0.28 | +0.35 | +0.12 | -0.24 | -0.26 | +0.18 | -0.03 | -0.09 | +0.08 | +1.00 |

## lip_press_rebalance — peak (8, single_34), N=180

Top-15 blendshapes by CV R² (blendshape / R² / σ(y) / cos(w, v_axis)):

| blendshape | R² | σ(y) | cos(w, v_axis) |
|---|---|---|---|
| browOuterUpLeft | +0.960 | 0.212 | -0.002 |
| browOuterUpRight | +0.948 | 0.158 | -0.004 |
| eyeSquintLeft | +0.939 | 0.178 | +0.018 |
| browInnerUp | +0.920 | 0.142 | +0.010 |
| eyeBlinkRight | +0.911 | 0.077 | +0.007 |
| browDownLeft | +0.907 | 0.261 | +0.016 |
| eyeSquintRight | +0.896 | 0.170 | +0.009 |
| browDownRight | +0.894 | 0.269 | +0.014 |
| eyeBlinkLeft | +0.878 | 0.080 | +0.001 |
| eyeLookDownRight | +0.878 | 0.091 | -0.003 |
| eyeLookDownLeft | +0.873 | 0.088 | -0.003 |
| eyeWideRight | +0.864 | 0.006 | -0.006 |
| eyeWideLeft | +0.828 | 0.006 | -0.002 |
| mouthShrugUpper | +0.802 | 0.044 | -0.018 |
| mouthUpperUpLeft | +0.774 | 0.010 | -0.003 |

Cosine between top-10 blendshape weight vectors:

| | browOuterUpLeft | browOuterUpRight | eyeSquintLeft | browInnerUp | eyeBlinkRight | browDownLeft | eyeSquintRight | browDownRight | eyeBlinkLeft | eyeLookDownRight |
|---|---|---|---|---|---|---|---|---|---|---|
| browOuterUpLeft | +1.00 | +0.87 | -0.26 | +0.63 | -0.22 | -0.40 | -0.18 | -0.37 | -0.17 | +0.36 |
| browOuterUpRight | +0.87 | +1.00 | -0.25 | +0.57 | -0.21 | -0.36 | -0.28 | -0.37 | -0.13 | +0.41 |
| eyeSquintLeft | -0.26 | -0.25 | +1.00 | +0.01 | +0.50 | +0.51 | +0.67 | +0.47 | +0.41 | -0.33 |
| browInnerUp | +0.63 | +0.57 | +0.01 | +1.00 | -0.14 | -0.20 | +0.02 | -0.21 | -0.08 | +0.21 |
| eyeBlinkRight | -0.22 | -0.21 | +0.50 | -0.14 | +1.00 | +0.32 | +0.42 | +0.36 | +0.66 | -0.07 |
| browDownLeft | -0.40 | -0.36 | +0.51 | -0.20 | +0.32 | +1.00 | +0.42 | +0.91 | +0.30 | -0.10 |
| eyeSquintRight | -0.18 | -0.28 | +0.67 | +0.02 | +0.42 | +0.42 | +1.00 | +0.52 | +0.54 | -0.32 |
| browDownRight | -0.37 | -0.37 | +0.47 | -0.21 | +0.36 | +0.91 | +0.52 | +1.00 | +0.26 | -0.15 |
| eyeBlinkLeft | -0.17 | -0.13 | +0.41 | -0.08 | +0.66 | +0.30 | +0.54 | +0.26 | +1.00 | +0.02 |
| eyeLookDownRight | +0.36 | +0.41 | -0.33 | +0.21 | -0.07 | -0.10 | -0.32 | -0.15 | +0.02 | +1.00 |

## How to read this

- **R² > 0.5** → the blendshape is strongly encoded as a linear direction in delta_mix at that block. Candidate for an injection target.

- **cos(w, v_axis) near ±1** → the blendshape's direction is the same as the axis's mean δ direction. The axis *is* that blendshape.

- **cos(w, v_axis) near 0** → the blendshape is encoded but the axis is something else; injecting the axis-mean won't move this blendshape.

- **Top-N cos matrix**: high off-diagonal = blendshapes are collinear in attention (not separately steerable). Block-diagonal = distinct AUs live in distinct attention directions, so a per-AU edit library is tractable.
