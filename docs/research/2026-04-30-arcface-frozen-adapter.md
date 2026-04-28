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

## Update 2026-04-28 — full-image (16, 64, 64) corpus

Built the canonical-plan corpus per [`arc_latent` plan](2026-04-27-arc-latent-distillation-plan.md):
26,108 FFHQ rows VAE-encoded at 512² → `(16, 64, 64)` bf16 latents on shard at
`C:\arc_distill\arc_full_latent\compact.pt` (3.48 GB). Same SHAs and arcface
teacher embeddings as the aligned-crop 14×14 corpus, so results are directly
comparable.

**Run log (chronological).**

| Variant | Stem | Mean | Median | Frac>0.9 | Status / note |
|---|---|---:|---:|---:|---|
| latent_a_full_pool | adaptive-pool 64→14 + LatentStemNative | 0.298 | — | — | killed at epoch 7; pool destroys spatial info |
| **latent_a_full_native** | ConvT k=8 s=2 p=3 (64→128) + crop 112 | **0.874** | 0.931 | 83.1% | 20 epoch full run, plateaued ~epoch 14 |
| **latent_a2_full_native_shallow** | latent_a_full_native + layer-1 unfrozen, init-from base | **0.881** | 0.939 | 87.6% | small lift over base; train-val gap closed |
| latent_a_full_crop | center-crop 64→32 + ConvT k=8 s=4 p=2 + crop 112 | 0.534 | — | — | killed at epoch 13; loses face content |
| latent_a_full_roi | RoIAlign 64→28 (SCRFD bbox) + ConvT k=8 s=4 p=2 (28→112) | 0.687 | 0.744 | 0.1% | 20 epoch full run; bbox crop ≠ similarity-aligned crop (teacher's frame) |

**Reading on each.**

- **full_pool fails.** The 64×64 latent encodes the whole frame; pooling
  averages background/hair/shoulders into the channel distribution and the
  frozen IResNet50 receives input that doesn't look like any face's stem
  output. Stem capacity alone can't fix it.
- **full_native works.** ConvTranspose stride-2 preserves all spatial info.
  Already higher fraction-above-0.9 than 14×14 A2-shallow (83% vs 31.6%); mean
  0.874 is dragged by a long left tail (p05 = 0.061, min = -0.13).
- **A2 over full_native.** Small +0.007 mean lift (vs +0.077 at 14×14). Train
  loss now ≈ val cos — no overfit room. Layer-1 unfreeze isn't the right
  knob anymore; the bottleneck has shifted.
- **full_crop fails.** Fixed center-crop (32×32) chops face content from many
  rows where the face isn't dead-center, throws away background that doesn't
  bother the previous setup. Net: large step backwards.
- **full_roi underperforms.** Per-row RoIAlign using the SCRFD bbox (re-run
  on 512², saved to `face_attrs.pt`) gives the student a face-centred crop
  but in a *different reference frame from the teacher*: bbox crop is
  axis-aligned + variably scaled; teacher's input is a 5-keypoint
  similarity-aligned 112² crop (rotated upright, eyes/nose at canonical
  positions). Different operation, plateaus at ~0.69.

**Diagnostic on the long left tail (full_native A2 checkpoint):**
the bottom-50 rows by val cos cluster around catastrophic predictions
(mean = -0.02), are not clustered together identity-wise (intra-band teacher
cos ≈ 0.017 — diverse failures), and visually share a structural pattern:
older men with **beards/hats/headwear**, sunglasses, scarves, costume
headgear. Failure mode is **localization**, not distribution mismatch —
the stem can't isolate the face the teacher saw on rows where non-face
content occupies a big fraction of the latent.

**Side artefact:** `face_attrs.pt` saves bbox/kps_5/landmark_2d_106/pose/
age/gender/det_score/n_faces per row, written for the RoI experiment but
useful well beyond that (latent-space similarity-warp, demographic ridge
heads, pose-conditional analyses).

Next experiment slot: similarity-warp (16, 64, 64) → (16, 14, 14) using the
5 keypoints in `face_attrs.pt` (the principled fix vs RoI bbox) and reuse
the existing latent_a_native variant. Predicted: val cos ≈ original 14×14
result (0.805 / A2: 0.882) since this reproduces the teacher's reference
frame in latent space.

## Validation as a loss head — Layer 1 + Layer 2 results

Ran [`validate_as_loss.py`](../../src/arc_distill/validate_as_loss.py) on the
`latent_a2_full_native_shallow` checkpoint (val n=1514). Full report in
`C:\arc_distill\arc_adapter\latent_a2_full_native_shallow\validation_report.json`.

**Layer 1.1 — teacher cosine.** mean 0.881 / median 0.939 / frac>0.9 = 87.6% /
p05 = 0.063 / min = -0.148. Reproduces eval.json.

**Layer 1.2 — k-fold ridge transfer (λ=100, k=5, out-of-fold R²).**

| target | R²(student) | R²(teacher) | note |
|---|---:|---:|---|
| age | 0.037 | 0.037 | linear projection to age is weak in either embedding; matched |
| gender | 0.014 | 0.011 | binary target — linear ridge is the wrong tool, both ≈0; matched |
| det_score | 0.003 | 0.003 | no signal in either; matched |
| yaw / pitch | — | — | dropped: `face_attrs.pose` is all zeros (precompute didn't enable `landmark_3d_68`) |

Demographic ridge transfer is "the student does whatever the teacher does."
Within the limits of linear ridge, no measurable transfer gap. To say anything
stronger about gender preservation needs logistic regression, which is out of
scope here.

**Layer 1.3 — augmentation invariance (TAR@FAR=1e-3, AUC).**

Hard negatives = teacher's top-5 nearest neighbours (excl. self). Random
negatives = a random other identity.

| perturbation | pos cos | rand neg | hard neg | AUC random | AUC hard |
|---|---:|---:|---:|---:|---:|
| gauss σ=0.02 | 0.999 | 0.008 | 0.186 | 1.000 | 1.000 |
| gauss σ=0.05 | 0.993 | 0.008 | 0.186 | 1.000 | 1.000 |
| hflip | 0.877 | 0.008 | 0.171 | 1.000 | 0.99994 |
| roll ±1 (H) | 0.972 | 0.007 | 0.186 | 1.000 | 0.99999996 |
| roll ±1 (W) | 0.978 | 0.008 | 0.186 | 1.000 | 1.000 |

Hard-negative AUC ≥ 0.9999 across all 5 perturbations. The student preserves
identity geometry under realistic latent-space perturbations even when scored
against teacher's nearest-neighbour identities (mean cos 0.18 vs random 0.008).
The hflip drop in pos-cos to 0.877 is realistic — real ArcFace also drops
slightly under symmetry breaks; the student inherits this.

Caveat: hard negatives here are FFHQ singletons, not multi-shot identities. A
true 1:1 verification benchmark (LFW / CFP-FP) would put genuine same-person /
different-person pairs through the student. We don't have a multi-shot dataset
on shard yet. This Layer 1.3 result is "preserves teacher discrimination on
augmented FFHQ singletons," not "passes LFW."

**Layer 2 — gradient sanity.** Single (16, 64, 64) latent x, target =
teacher_emb of a random different row, loss = `1 − cos(student(x), target)`.
- gradient finite ✓
- grad norm 0.14 (reasonable, neither vanishing nor exploding)
- 100 SGD steps at lr=0.05 reduce loss 1.081 → 0.986 (~9% drop toward an
  arbitrary cross-identity target)
- backprop through onnx2torch FX graph works as expected

**Verdict for Use A.** Green light. The student passes the cheap proxies for
"usable as a slider/LoRA training loss head" — invariance is robust, gradient
flows, demographic content tracks teacher within the limits of linear probing.
The real test is Layer 3 (slider A/B with vs without student loss), specced at
[2026-04-30-layer3-slider-ab-spec.md](2026-04-30-layer3-slider-ab-spec.md).

**Use B (inference-time guidance) untouched** — see "Possible uses" below for
why 0.881 is more concerning under operating conditions there.

## Possible uses and their fidelity demands

The 0.881 mean teacher cosine of `latent_a2_full_native_shallow` is a single
scalar that means different things to different downstream uses. The two
candidate uses we care about:

**Use A: identity-preservation loss during slider/LoRA training.**
The student is queried inside the training loop on edited and anchor latents;
loss = `1 − cos(student(x_edited), student(x_anchor))` (or similar),
backprop'd into the trainable weights. Each batch averages over many samples,
and only the *direction* of the gradient needs to track ArcFace — absolute
fidelity is much less load-bearing. **0.881 is plausibly fine here.** The
load-bearing question is whether same-source perturbed pairs rank above
cross-identity pairs at usable margin (Layer 1.3, this validation suite).

**Use B: classifier-guidance / inference-time identity steering.**
The student is queried during sampling, on a single denoising step's predicted
x₀ or latent; gradient steers the noise prediction toward an identity target.
Here the 0.119 off-direction component biases every step on every sample;
errors don't average out the way they do across a training batch. A 0.881
student may pull samples toward "ArcFace-shaped but not-quite-identity"
attractors, and those biases compound across 20–50 sampling steps. **Use B
needs a higher fidelity bar than 0.881 implies, and should be measured under
operating conditions before being trusted** — by, e.g., taking a target
identity, running classifier guidance on a fixed seed with both the
real teacher and our student, and comparing the rendered face to the target
on a held-out set with a different ArcFace instance as the judge.

The validation suite in this thread (Layer 1.1–1.3 + Layer 2 + Layer 3
slider A/B) is squarely aimed at Use A. Use B remains an open question;
nothing in `validate_as_loss.py` directly probes it.

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
