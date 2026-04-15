# Research: RigFace — Technical Deep Dive

**Date:** 2026-04-12
**Source:** arxiv.org/abs/2502.02465, github.com/weimengting/RigFace
**Status:** Public — code + weights on GitHub and HuggingFace (released June 2025)

---

## What It Is

RigFace is a fully fine-tuned Stable Diffusion 1.5 model for face editing with three independent control channels: expression, head pose, and lighting. Identity is preserved via a separate encoder path. Unlike ControlNet or adapter approaches, the entire SD backbone is trained — not just a small plugin module. The paper's thesis is that full fine-tuning is necessary for tight identity preservation in face editing.

---

## Architecture: Four Components

### 1. Spatial Attribute Provider (pipeline, not a neural net)

Takes a source portrait + target expression/pose/lighting parameters → runs through DECA (3DMM reconstruction) → produces three decoupled outputs fed into the diffusion backbone:

- **3D rendered image**: Lambertian reflectance + spherical harmonics lighting model. Renders the face at target pose/lighting. Pixel-space guidance the UNet can actually interpret.
- **Expression coefficients**: FLAME blendshape parameters extracted via Deep3DRecon. Separate from pose so expression and pose don't bleed.
- **Dilated foreground mask**: Face parsing output. Tells the model where the face is vs. background. Enables background preservation when pose changes dramatically.

Why 3D renders instead of raw coefficient vectors: rendered images are in the same domain as what the diffusion UNet operates on. Raw 403-d FLAME vectors require the model to learn a mapping from abstract coefficients to pixels; a rendered image provides it directly.

### 2. Identity Encoder

An entire second copy of the SD 1.5 denoising UNet, initialized from the same pretrained weights, fully trainable. Processes the reference identity image and extracts multi-level self-attention features from each transformer block.

This is deliberately not CLIP image embedding — the paper shows CLIP loses fine-grained facial detail. The full-UNet identity encoder preserves bone structure, distinctive features, and skin characteristics.

### 3. FaceFusion (identity injection mechanism)

For each transformer block, takes the identity features from the Identity Encoder and concatenates them along the width dimension before the self-attention operation in the Denoising UNet. Only half of the concatenated output is retained. No geometric warping or spatial alignment — pure feature concatenation.

Result: identity information flows into every layer of the denoising process, not just into early or late layers.

### 4. Attribute Rigger + Denoising UNet

Lightweight convolution module that injects the spatial conditions (3D renders + expression coefficients + masks) into cross-attention layers. Injected at multiple depths (early, middle, deep) so guidance operates at all scales.

The separation of concerns is physical: **self-attention = identity (who), cross-attention = attributes (how they look right now)**.

Loss function:
```
L = E[||ε - φ_SD(z_t, t, φ_id(g), ψ, φ_col(y))||²]
```
where `φ_id(g)` = identity features from reference image `g`, `ψ` = expression coefficients, `φ_col(y)` = spatial conditions. Standard diffusion MSE on predicted noise, no explicit face consistency loss.

---

## Training

- **Dataset**: Aff-Wild video dataset, 30,000 image pairs (source/target frames from same video). Same-video pairs guarantee consistent identity, background, and lighting while naturally varying expression and pose. 20 identities held out for evaluation.
- **Hardware**: 2× AMD MI250x (~equivalent to 2× A100)
- **Steps**: 100,000 at batch size 8, LR 1e-5
- **Duration**: ~24 GPU hours

---

## What You Control Independently

| Channel | Input format | What changes |
|---|---|---|
| Expression | FLAME blendshape coefficients (from SMIRK or DECA on target image) | Mouth, eyes, brows, cheeks |
| Head pose | FLAME pose parameters → 3D render | Head rotation, neck articulation |
| Lighting | Spherical harmonics coefficients → 3D render | Light direction, intensity, shadows |
| Identity | Reference portrait image → Identity Encoder | Who the person is — preserved |

All four are independent. You can change expression without drifting the pose. You can change lighting without changing expression.

---

## Quality Signal

Perceptual study: 59.3% of RigFace expression edits rated by human evaluators as more realistic than ground truth. Indicates the model interpolates plausibly, not just copying.

---

## Weaknesses

- Breaks on extreme head poses (DECA quality degrades at profile/upward angles)
- Occlusions (hair over face, glasses, hands) degrade output
- Computational: requires training two full SD UNets simultaneously
- No speed benchmarks — inference is multi-step diffusion, ~1-2s per image
- Training data bias: Aff-Wild is relatively small and video-based

---

## Availability

- GitHub: https://github.com/weimengting/RigFace
- HuggingFace: https://huggingface.co/mengtingwei/rigface
- Requirements: Python 3.9, PyTorch 1.13, PyTorch3D, DECA model (deca_model.tar), FLAME2020 model files (free academic license from MPI-IS)
- Training time: 24 GPU hours on MI250x-equivalent

---

## Related Work (post-February 2025)

- **Arc2Face + blendshape expression adapter** (ICCVW 2025) — same two-channel architecture but uses ArcFace identity embedding instead of pixel-space identity encoder. More composable (identity is a manipulable vector), slightly less fine-grained.
- **MorphFace** (CVPR 2025) — same 3DMM+diffusion paradigm, aimed at synthetic training data generation rather than editing.
- No direct academic citations of RigFace found as of April 2026 (6-month citation lag normal).
