---
status: live
topic: demographic-pc-pipeline
supersedes: 2026-04-29-arcface-pixel-baseline.md
---

# ArcFace frozen-backbone adapter on aligned FFHQ crops: Pixel-A and Latent-A2

Distillation result for the question *"can we recover ArcFace identity geometry
from Flux VAE latents at aligned-crop resolution?"* — a diagnostic milestone
on the way to the canonical [`arc_latent` plan](2026-04-27-arc-latent-distillation-plan.md).
**Pixel-A clears the cosine-0.9 gate (val 0.960). Latent-A2-shallow at 14×14
aligned-crop latents clears a Pixel-A − 0.10 gate (val 0.882) once IResNet50's
layer-1 is unfrozen.** The 0.155 gap from Pixel-A to frozen-backbone Latent-A
is the raw "VAE identity tax" at aligned-crop resolution; layer-1 unfreeze
recovers half of it.

## Pivot from the original ResNet-18-from-scratch plan

The original plan trained a ResNet-18 from scratch on `(16, 64, 64)` Flux VAE
latents from full 1024² FFHQ images, gated at cosine R² ≥ 0.95. The pivot:

1. **Reuse ArcFace pretraining.** The teacher is IResNet50 (buffalo_l
   `w600k_r50.onnx`, R50 not torchvision-50). Training a ResNet-18 from random
   init throws away ~25M parameters of identity-relevant weights that already
   work. The frozen-backbone-with-trainable-stem pattern keeps all 25M and only
   trains the input adapter — orders of magnitude less compute.
2. **Square-1 verification first.** Earlier attempt at ImageNet-ResNet-18 on
   unaligned 224² FFHQ portraits hit val cos 0.377; suspect was alignment.
   Rebuilt the corpus with SCRFD bbox + 5-point similarity alignment to 112²
   crops; the aligned-pixel ResNet-18 gate ran on the same data and still
   under-performed. `scripts/check_arc_oracle.py` then verified pipeline
   byte-perfect against the teacher (cos = 1.0000 on 5/5 sample rows for
   teacher reproducibility, crop equality, and R50 oracle), refuting the
   "pipeline mismatch" hypothesis.
3. **Conclusion: it's not the pipeline, it's the input distribution.** The
   IResNet50 backbone expects the precise statistics of an ArcFace-aligned
   112² crop. Anything that doesn't match — wider crops, different normalisation,
   different spatial resolution — costs you. Fix: keep the backbone, train
   only the stem to map your input to the distribution layer-1 expects.

## Method

**Backbone.** `w600k_r50.onnx` converted to PyTorch via `onnx2torch`. Loaded
as `ModelProto` first to bypass `safe_shape_inference`'s NamedTemporaryFile
path, which fails on Windows (only relevant for the shard but gates the test
suite cross-platform). All parameters and buffers frozen. Forward call is
`self.Conv_0(input_1)`; replacing the `Conv_0` submodule substitutes the stem.

**Three stems trained.**

- **PixelStem** (Pixel-A, 1.9K params). Single 3×3 conv 3→64 stride 1, mirroring
  the original `Conv_0` exactly including `bias=True` (BN was absorbed into the
  conv during ONNX export). Native PRelu_1 + BatchNormalization_2 are also
  unfrozen and reset to fresh init so the gate measures "can a fresh stem
  recover R50's input distribution" rather than "is the original stem already
  optimal."
- **LatentStemUpsample** (Latent-A-up, 9.4K params). Bilinear-upsample 14→112,
  then Conv 16→64 + BN + PReLU. Bypasses native PRelu_1+BN_2 (the latent stem
  carries its own activation/normalisation).
- **LatentStemNative** (Latent-A-native, 271K params). ConvTranspose2d stride-8
  for 14→112 without bilinear interpolation; Conv 16→64 → BN → PReLU →
  ConvTranspose 64→64 stride 8 → BN → PReLU.

**A2-shallow.** Initialised from latent_a_native's checkpoint at epoch 19,
then unfreezes IResNet50 layer-1 — the three IBasicBlock residual blocks with
parameter-bearing modules `Conv_3, Conv_5, Conv_6, BatchNormalization_8,
Conv_9, Conv_11, BatchNormalization_13, Conv_14, Conv_16` and three PReLU
slope buffers (`initializers.onnx_initializer_{1,2,3}`). 226K layer-1 params
unfrozen. Two-group AdamW: stem at lr 1e-3, layer-1 at lr 1e-4 (10× ratio,
standard fine-tune). Layer-1 BatchNorm running stats stay frozen via eval mode
(only affine weight/bias trains), preventing drift on a small dataset.

**Data.** 25,696 FFHQ rows that SCRFD detected at canonical `det_thresh=0.5`
(out of ~70K full FFHQ). SHA-prefix split: `image_sha256[0]=='f'` is val
(1,514 rows; ~6.25%); rest is train (24,182 rows). Same canonical pipeline
as `InsightFaceClassifier`: SCRFD detect → 5-point similarity transform →
112² aligned crop. Latents are encoded by Flux VAE
(`encode_aligned_to_latent.py`) on the aligned 112² crops, producing
`(16, 14, 14)` bf16 latents. Teacher embeddings are L2-normalised 512-d from
the same R50 on the same aligned crops.

**Optimisation.** AdamW, weight decay 1e-4, cosine LR schedule, batch size
256, 20 epochs each variant. Resumable from `last.pt`; tracks best-by-val to
`checkpoint.pt`. Per-epoch loss is `1 − cos(student(x), teacher.detach())`.

## Results

| Variant | Trainable | val cos mean | median | p05 | p95 | min | frac > 0.9 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Pixel-A** | 1,920 | **0.960** | 0.968 | 0.924 | 0.980 | — | 97.2% |
| Latent-A-up | 9,408 | 0.577 | 0.582 | 0.422 | 0.701 | 0.175 | 0.0% |
| Latent-A-native | 271,744 | 0.805 | 0.811 | 0.712 | 0.875 | 0.509 | 0.5% |
| **Latent-A2-shallow** | 497,920 | **0.882** | 0.886 | 0.821 | 0.927 | 0.658 | 31.6% |

Gates:

- Pixel-A gate (cos ≥ 0.95): **CLEARED** at 0.960.
- Latent-A gate (Pixel-A − 0.10 = 0.86): A2-shallow **CLEARED** at 0.882; the
  two frozen-backbone-only variants (up and native) failed.

## Interpretation

**The VAE identity tax at aligned-crop resolution is ≈0.155 cosine.** Pixel-A
0.960 minus Latent-A-native 0.805 measures the gap between giving the frozen
IResNet50 its native input (aligned 112² pixels) and giving it a `(16, 14, 14)`
VAE-encoded version of the same crop. This is a real, measurable cost of
working in latent space.

**Layer-1 unfreeze recovers half of it.** A2-shallow lifts the frozen-backbone
0.805 to 0.882 with 226K additional trainable params. Read: the early
residual blocks of IResNet50 expect inputs in the precise statistical
distribution of pixel-domain stems. Once allowed to remap, the deeper backbone
(layer-2 onwards, still frozen at ArcFace pretraining) reads the result fine.
The remaining 0.078 gap to Pixel-A parity is some combination of
information genuinely lost in VAE compression at 14×14 spatial and deeper-layer
mismatches not addressed by layer-1 unfreeze.

**Capacity vs ceiling.** The up→native jump (9K → 271K params, 0.577 → 0.805)
is mostly stem capacity: the bilinear-upsample variant just doesn't have
enough learnable headroom, while the ConvTranspose stride-8 stem can render
the pixel-distribution layer-1 wants. A2-shallow's 0.077 lift over native is
a different mechanism — not stem capacity, but layer-1's freedom to remap.

**Trainable-param efficiency.** Pixel-A reaches 0.960 with 1,920 params.
A2-shallow reaches 0.882 with 497,920. Per-parameter, the Pixel-A regime is
~260× more identity-information-dense — but it is operating on the modality
the backbone was trained on. The adapter framework recovers most of the
identity from a different modality at ~250× the parameter budget; that's the
price of working in latent space.

## Code artefacts

- [src/arc_distill/backbone.py](../../src/arc_distill/backbone.py) — frozen R50 loader, `attach_stem`, `mark_stem_trainable`, `unfreeze_layer1`.
- [src/arc_distill/stems.py](../../src/arc_distill/stems.py) — three stems.
- [src/arc_distill/adapter.py](../../src/arc_distill/adapter.py) — `AdapterStudent` with four variants and `parameter_groups()` for differential LR.
- [src/arc_distill/train_adapter.py](../../src/arc_distill/train_adapter.py) — supports `--init-from` for A2's warm start from native's checkpoint.
- [src/arc_distill/eval_adapter.py](../../src/arc_distill/eval_adapter.py) — emits the table fields.
- [src/arc_distill/encode_aligned_to_latent.py](../../src/arc_distill/encode_aligned_to_latent.py) — Flux VAE encode of aligned 112² crops to `(16, 14, 14)` bf16.
- [scripts/check_arc_oracle.py](../../scripts/check_arc_oracle.py) — square-1 byte-correctness verification.
- `scripts/run_arc_adapter_*.bat` — schtasks wrappers for shard.
- 11/11 tests pass at [tests/test_arc_distill.py](../../tests/test_arc_distill.py).

## Limitations and scope

**This experiment is at aligned-crop resolution, not slider-training resolution.**
The canonical [`arc_latent` plan](2026-04-27-arc-latent-distillation-plan.md)
targets `(16, 64, 64)` latents from full 1024² FFHQ images, which is the
geometry slider training actually emits. Our `(16, 14, 14)` aligned-crop
latents have ~21× less spatial information than the slider-training-resolution
latent. The 0.155 raw VAE-tax number measured here is specific to this
aligned-crop geometry and does not directly transfer.

**A2-shallow is not yet validated as a slider-training loss head.** Cosine to
teacher is a regression metric. The downstream questions are:

- Does it discriminate same-vs-different identity at usable margin (ROC/AUC)?
- Does the existing demographic+glasses ridge battery transfer to A2 features
  with acceptable accuracy?
- Do gradients through it point sensibly back toward identity-preserving
  edits when used as a loss?

Until at least the first two are checked, A2 is a regressor, not a loss.

## Next steps

The user reframed the goal: "We want ArcNet, MediaPipe, and SigLIP to be
usable as losses without rendering." Three loss heads in latent space, all
needing to consume slider-training-resolution latents.

The decision tree:

1. **Build the canonical-plan corpus** (`(16, 64, 64)` latents from 1024²
   FFHQ encoded at slider-training resolution). Train an A2-shaped student
   on that geometry. Stem changes from `(16, 14, 14)` ConvTranspose to
   something handling the `(16, 64, 64)` → IResNet50-input conversion;
   may need a learned spatial pooling or attention to a face-sized region
   since the 64×64 latent encodes the full image, not a face crop.
2. **Verify A2 (or its 64×64 successor) as an actual loss.** Identity-pair
   discrimination AUC vs teacher; demographic ridge transfer Pearson r vs
   teacher.
3. **Replicate the recipe for MediaPipe.** Cached 52-d ARKit blendshape labels
   already exist on aligned FFHQ from earlier NMF work; direct regression
   target makes this a smaller experiment.
4. **Replicate for SigLIP.** Architecturally harder (ViT patch embed is
   pixel-grid-tied); requires either a new patch projection + first-block
   unfreeze (mirroring A2) or end-to-end distillation into a smaller student.

The 14×14 result here is foundational for the recipe, not the deliverable.
The deliverable lives at slider-training resolution.

## Reference results

```
Pixel-A           checkpoint_epoch=13  best val cos 0.960
Latent-A-up       checkpoint_epoch=19  best val cos 0.577
Latent-A-native   checkpoint_epoch=19  best val cos 0.805
Latent-A2-shallow checkpoint_epoch=19  best val cos 0.882
```

All four checkpoints live on shard at
`C:\arc_distill\arc_adapter\<variant>\checkpoint.pt`; `eval.json` next to
each.
