---
status: live
topic: arc-distill
---

# MediaPipe distillation plan — expression head in latent space

The third "consume teacher T as a loss without rendering" thread, after
ArcFace (shipped to L1+L2 validation) and before SigLIP. Inherits the
defaults from [2026-04-30-arc-distill-lessons.md](2026-04-30-arc-distill-lessons.md).

## Goal

A student that maps Flux VAE latents `(16, 64, 64)` to a 52-d ARKit blendshape
vector matching MediaPipe FaceLandmarker's output, **without rendering**.
Use cases:

- **Expression-preservation loss** during slider/LoRA training. When the user
  trains a "make-older" slider, this loss penalises unintended changes to
  smile/eye-open/jaw-position. Pairs naturally with the ArcFace head: ArcFace
  preserves identity, MediaPipe preserves expression.
- **Targeted expression-control loss.** Drive specific blendshapes toward
  target values during sampling or editing.
- **Cheap expression diff probe** for analysis (we already use blendshape
  diffs as an interpretive signal; running them at latent speed instead of
  render speed unlocks larger sweeps).

## Teacher

**MediaPipe FaceLandmarker** (Tasks API, blendshape mode enabled).
- Input: RGB image, internally face-detected and cropped.
- Output of interest: 52 ARKit-compatible blendshape coefficients in
  [0, 1], roughly sparse — most are near zero on most faces.
- Optional outputs: 478 3D landmarks (4-d each: xyz + visibility),
  4×4 facial transformation matrix.
- Default model file: `face_landmarker.task` from MediaPipe distribution.

The 52-d blendshape vector is the primary distillation target. Landmarks
and pose are not in scope for v1 (different units, different invariances,
larger output space, smaller use-case payoff).

## Open question — how strict is "MediaPipe"?

MediaPipe's blendshape head is a small mobile CNN trained on a proprietary
dataset; we cannot inherit its weights. So unlike arc_distill (where the
trick was reusing the frozen IResNet50), here the student backbone choice
is **architectural, not transfer-learning**.

Three candidates, ordered by ambition:

1. **Fresh-from-scratch small CNN.** Stem identical to arc_distill's
   `LatentStemFull64Native` (ConvT stride-2 → 112×112), then a 5-block
   ResNet-style head, then a 52-way linear regressor. ~5 M params, all
   trainable. Most direct; analogous to MediaPipe's own backbone size.
2. **Reuse the frozen IResNet50 backbone from arc_distill, replace the head.**
   ArcFace embeddings discard expression information by design (so do all
   verification heads — they want a *consistent* embedding regardless of
   smile). This reuse will likely fail. **Skip this path** unless validation
   surprises us.
3. **Fresh ImageNet-pretrained ResNet18 backbone + new patch projection.**
   ImageNet features include some expression/texture signal. Possibly
   stronger than (1); strictly more expensive.

**Default: option 1.** Decide after smoke-test results.

## Build sequence

### Step 0 — verify teacher works on FFHQ

Run MediaPipe FaceLandmarker on 100 FFHQ rows at 512², dump blendshapes
to JSON. Sanity-check distribution: expect most channels ≈0, a few nonzero
("eyes_open", "mouth_close" near 1 by default), some clearly load-bearing
("smile_left/right", "brow_inner_up"). If output is degenerate or systematic
detection failures, fix the precompute *before* scaling to 26k rows.

Estimated effort: 30 min. Output: `docs/research/2026-04-30-mediapipe-teacher-smoke.md`
with histogram of each channel.

### Step 1 — precompute teacher labels

Mirror `precompute_bboxes.py` shape:

- Input: same 26,108 FFHQ rows ordered by SHA in `compact.pt`.
- For each row: load 512² image, run FaceLandmarker, save 52-d blendshape
  vector + face-detection found mask.
- Output: `compact_blendshapes.pt` with `{shas, blendshapes (N, 52) fp32,
  found (N,) bool, format_version: 1}`.
- Failed-detection rows: zero vector + found=False (so downstream filters
  honour the mask, not infer from value).

Estimated effort: 1 h on shard. Cache and archive to USB drive (per
[reference_external_pkl_archive.md memory] cache-then-archive lifecycle).

### Step 2 — student v1 baseline

`src/mediapipe_distill/` mirroring `src/arc_distill/`:
- `dataset.py`: `CompactBlendshapeDataset(compact_path, blendshapes_path, split)` —
  reads latents from `compact.pt` and labels from `compact_blendshapes.pt`,
  filters on `found`.
- `stems.py`: reuse `LatentStemFull64Native` from arc_distill.
- `student.py`: stem + small head (3-5 ResNet blocks) + linear(52). Output
  passed through sigmoid (blendshapes are [0, 1]).
- `train.py`: AdamW, lr 1e-3, cosine schedule, 20 epochs. Loss = MSE on
  raw 52-d. Try BCE-with-logits as ablation (matches sigmoid output more
  cleanly).
- `eval.py`: per-channel R² on val split (k-fold-style or held-out). Mean
  R², median R², worst-5 channels by R².

Build expectation: matches arc_distill v1 baseline shape exactly.
Estimated effort: 2-3 h to write, 30-60 min training on shard.

### Step 3 — validate as a loss

Same three-layer protocol as arc_distill, instantiated for blendshapes:

**Layer 1.1 — teacher metric.** Mean per-channel R² across 52 blendshapes.
Expected baseline: 0.5-0.7. Headline number for the writeup.

**Layer 1.2 — demographic transfer.** Predict {age, gender, race} from the
student's 52-d output and from the teacher's. Both should be near-zero
(blendshapes shouldn't carry demographic information by design); if the
student carries more than the teacher, it's leaking.

**Layer 1.3 — augmentation invariance.** Same as arc_distill, but the
positives are pairs `(blendshapes(x), blendshapes(perturb(x)))` and the
metric is L2 distance not cosine. Plus an *expression-flip* test: artificially
flip the smile (target manipulation by adding +0.5 to smile_left/right) and
verify the student's prediction shifts in the corresponding direction.

**Layer 2 — gradient sanity.** Backprop through student into a Flux latent,
check finite gradient and 100-step descent toward an arbitrary target
blendshape vector.

**Layer 3 — slider A/B.** Spec mirrors
[2026-04-30-layer3-slider-ab-spec.md](2026-04-30-layer3-slider-ab-spec.md):
train an "age-up" slider with vs without `λ · MSE(student(z_edit), student(z_anchor))`
and verify at matched edit magnitude that the student-loss preserves smile,
eye openness, etc.

## Scope cuts

- **Landmarks not in v1.** 478×3 floats with no obvious normalization make
  the loss landscape harder; expression payoff is in the 52-d head. Add
  later if needed.
- **No transformation-matrix / pose target.** Use ArcFace's pose probe
  instead (when we re-precompute it with `landmark_3d_68` enabled — see
  the lessons doc).
- **No cross-dataset eval.** This is a research head, not a deployable
  classifier. FFHQ-only.
- **Sigmoid output assumed.** Blendshapes are [0, 1] in MediaPipe's spec;
  pinning the student to that range simplifies the loss and avoids learned
  scale drift. If MediaPipe occasionally emits >1 we'll find out in step 0.

## Decision points

- **After step 0 (teacher smoke):** if MediaPipe detection fails on >5%
  of FFHQ, switch to *aligned* FFHQ (the 1024² FFHQ is canonical-cropped
  already; should be fine).
- **After step 2 (v1 baseline trained):** if mean R² < 0.4, escalate to
  ImageNet-pretrained backbone or larger student. If R² > 0.7, ship as v1
  and start Layer 3.
- **After Layer 1.3 expression-flip:** if the prediction *doesn't* shift
  with target manipulation, the model has collapsed to predicting the
  population mean. Re-train with a focal-MSE-like reweighting that emphasises
  rare (high-amplitude) blendshape values.

## Lineage and cross-references

- The arc_distill writeup explicitly named MediaPipe as the next head:
  [2026-04-30-arcface-frozen-adapter.md](2026-04-30-arcface-frozen-adapter.md)
  "Next steps" section, item 3.
- We have prior MediaPipe blendshape work captured in earlier NMF research
  (see `project_blendshape_bridge_state.md` memory and
  `src/demographic_pc/fit_nmf_directions_resid.py`). Those blendshapes were
  computed on *Flux-rendered* images, not on FFHQ source — they don't
  directly substitute for step 1's precompute. But the MediaPipe extraction
  code in that thread is reusable.
- Lessons we're carrying: [2026-04-30-arc-distill-lessons.md](2026-04-30-arc-distill-lessons.md).

## Build status

Plan only. Nothing implemented yet. Estimated total time-to-Layer-3:
~1.5 days (step 0 30min, step 1 1h, step 2 4h, step 3 ~half-day excluding
slider A/B).
