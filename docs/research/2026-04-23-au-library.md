---
status: live
topic: metrics-and-direction-quality
summary: NMF decomposition of the 52-d ARKit blendshape matrix into K=8 parts-based AU patterns, with per-corpus ridge fits of δ→component score. Library saved to models/blendshape_nmf/au_library.npz for future cached-δ injection.
---

# AU library from blendshape NMF — 2026-04-23

K=8, reconstruction R² on Y = **0.913**. NMF is parts-based: each component is a non-negative co-activation pattern of ARKit blendshapes, not an orthogonal signed direction.

## Components

| C | semantic | top blendshape loadings | mean R² | R² range |
|---|---|---|---|---|
| C0 | jaw_open | jawOpen(2.45), mouthLowerDownRight(2.42), mouthLowerDownLeft(2.36), mouthDimpleRight(0.47) | +0.347 | [-0.220, +0.834] |
| C1 | gaze_vertical | eyeLookDownRight(1.93), eyeLookDownLeft(1.89), eyeBlinkLeft(1.73), eyeBlinkRight(1.59) | +0.711 | [+0.462, +0.859] |
| C2 | brow_lift | browOuterUpLeft(2.79), browInnerUp(2.62), browOuterUpRight(2.26), eyeLookDownLeft(0.87) | +0.765 | [+0.161, +0.961] |
| C3 | brow_furrow | browDownRight(3.81), browDownLeft(3.64), eyeSquintLeft(1.83), eyeSquintRight(1.52) | +0.770 | [+0.380, +0.945] |
| C4 | pucker | mouthPucker(5.75), browOuterUpLeft(0.83), eyeSquintLeft(0.50), mouthFunnel(0.36) | +0.609 | [+0.324, +0.755] |
| C5 | mouth_stretch | mouthUpperUpLeft(4.70), mouthUpperUpRight(4.68), mouthLowerDownRight(1.27), mouthStretchLeft(0.66) | +0.470 | [-0.295, +0.766] |
| C6 | smile | mouthSmileLeft(3.60), mouthSmileRight(3.57), eyeSquintRight(0.28), eyeSquintLeft(0.19) | +0.248 | [-0.656, +0.807] |
| C7 | gaze_vertical | eyeSquintLeft(2.86), eyeSquintRight(2.61), eyeLookInRight(1.97), eyeLookOutLeft(1.94) | +0.689 | [+0.061, +0.853] |

## Per-corpus R² matrix

| component | smile_inphase | jaw_inphase | alpha_interp_attn | anger_rebalance | surprise_rebalance | disgust_rebalance | pucker_rebalance | lip_press_rebalance |
|---|---|---|---|---|---|---|---|---|
| C0 (jaw_open) | +0.680 | +0.809 | +0.834 | -0.220 | -0.153 | +0.564 | +0.085 | +0.179 |
| C1 (gaze_vertical) | +0.859 | +0.744 | +0.821 | +0.643 | +0.597 | +0.462 | +0.713 | +0.853 |
| C2 (brow_lift) | +0.941 | +0.839 | +0.829 | +0.161 | +0.743 | +0.704 | +0.940 | +0.961 |
| C3 (brow_furrow) | +0.945 | +0.883 | +0.873 | +0.771 | +0.380 | +0.537 | +0.864 | +0.910 |
| C4 (pucker) | +0.755 | +0.675 | +0.565 | +0.650 | +0.324 | +0.422 | +0.753 | +0.726 |
| C5 (mouth_stretch) | +0.753 | +0.575 | +0.757 | -0.110 | +0.611 | -0.295 | +0.704 | +0.766 |
| C6 (smile) | +0.807 | +0.636 | +0.652 | -0.656 | +0.390 | +0.043 | +0.511 | -0.401 |
| C7 (gaze_vertical) | +0.838 | +0.853 | +0.839 | +0.563 | +0.696 | +0.061 | +0.833 | +0.833 |

## Peak (step, block) per corpus

| corpus | step | block |
|---|---|---|
| smile_inphase | 6 | single_34 |
| jaw_inphase | 7 | single_34 |
| alpha_interp_attn | 8 | single_34 |
| anger_rebalance | 8 | single_34 |
| surprise_rebalance | 8 | single_34 |
| disgust_rebalance | 10 | single_34 |
| pucker_rebalance | 9 | single_34 |
| lip_press_rebalance | 8 | single_34 |

## Library file layout

`models/blendshape_nmf/au_library.npz` contains:

- `H` — (K=8, 52) AU-pattern matrix, blendshape-space loadings per component. Non-negative.
- `W_refits` — (K, T, 3072) per-(component, corpus) ridge weight vectors. These are the cached-δ injection targets: to push component C by amount α at corpus T's peak block, add `α · W_refits[C, T]` to δ.

- `peak_coords` — (T, 2) [step_idx, block_idx] in cache order.

- `per_comp_r2` — (K, T) CV R² per (component, corpus), the reliability of each library vector.

- `tags`, `names`, `reconstruction_r2` — string labels and the blendshape-space fit quality.

## How to read this

- **NMF recon R²** tells us how lossy the basis is. ≥0.9 means the 52-d blendshape space really is 8-d in disguise. <0.7 means we cut too much and need more components.

- **Per-component mean R²**: how reliably each AU pattern is encoded in attention at the peak block. Components with mean R²>0.7 are library-grade — inject them and expect the pattern to move.

- **R² spread across corpora**: a component that works only on one corpus reflects a corpus-specific artefact. One that works on all corpora is a general attention direction worth caching.

- **Semantic tag**: heuristic over top loadings. Override by inspection if a pattern is mixed.
