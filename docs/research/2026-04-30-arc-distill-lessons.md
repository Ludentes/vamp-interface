---
status: live
topic: arc-distill
---

# Lessons from arc_distill

What we'd want any future "distill teacher X into latent space" thread (the
MediaPipe and SigLIP heads next) to start from.

## On architecture

**Frozen backbone + trainable stem is the right shape when teacher hardware
is out of reach.** The Pixel-A baseline (0.96 cos against ArcFace teacher,
trainable=stem only ≈0.27 M params) validated this before we touched latents.
Without that early signal we would have wasted weeks on full distillation.

**Geometry beats capacity.** ConvT stride-2 (no spatial discard) hit 0.87 on
`(16, 64, 64)`. Adaptive-pool 64→14 hit 0.30 — same downstream architecture,
just a different up/downsample. The deep IResNet50 wants spatial structure
preserved into its stem-output frame; lossy spatial reductions break the
contract between stem-output and what layer-1 expects.

**Match the teacher's reference frame, exactly.** SCRFD bbox-crop RoIAlign
plateaued at 0.69 because the teacher's input is *similarity-aligned* (5
keypoints, rotated upright, eyes/nose at canonical positions), not
axis-aligned bbox. RoIAlign of the wrong frame is not free. If you want to
match a teacher that works in canonical face space, replicate the canonical
transform — don't substitute a "close enough" alternative.

**Low-res + cv2.warpAffine is interpolation-bound.** The 14×14
similarity-aligned attempt lost more identity to bilinear interpolation
(1 latent px ≈ 8 image px) than the alignment recovered. Lesson: similarity-warp
into an oversized canvas (28+) if you want both alignment and detail. The
"correct" operation done at the wrong resolution is still wrong.

**Layer-1 unfreeze gets ~+0.01 mean cos and closes train/val gap.**
Diminishing returns past that on the 24k FFHQ dataset. Going deeper (layer-2)
would risk the BN-running-stat issue and probably overfit.

## On precompute & data

**Pre-computed artefacts must carry an explicit valid mask, and downstream
datasets must honour it.** Aligned14 polluted-row bug: `precompute_aligned_latents.py`
zeros out re-detection-miss rows but `CompactLatentDataset` didn't filter on
`found`. Cost: one ~13 min training run on contaminated data, killed at epoch 0
by suspicious row counts. Fix: 7-line dataset patch. Pattern: any precompute
that drops some rows must surface that as a mask, and any reader must respect
it (or default to `found is None → all valid`, which is what we did).

**Always check what the teacher actually produced before measuring it.**
`face_attrs.pose` was all zeros — InsightFace doesn't populate pose unless
`landmark_3d_68` is in `allowed_modules`, and the precompute didn't enable
it. The Layer 1.2 ridge transfer reported R²=1.000 on yaw and pitch on the
first run; that would have been an extremely embarrassing publishable
"perfect transfer" claim. Caught only because 1.000 was suspicious enough
to inspect. Generalise: any "perfect" metric needs a degeneracy check before
celebration.

**Cache encoded latents.** A wasted re-encode (33 minutes) happened in this
thread because we didn't notice `C:\arc_distill\encoded\train-*.pt` already
contained per-shard `(369, 16, 64, 64)` latents from earlier work. Future
heads should always check for existing shard caches before re-encoding.

## On evaluation

**A single scalar metric without a use-case context is misleading.** 0.881
teacher cosine is "great" or "bad" depending on whether you're using the
model as a batch-averaged loss (probably fine) or per-step inference
guidance (probably not). The "Possible uses" section in the writeup is now
a checklist for any future head.

**k-fold + heavy ridge regularization is the right diagnostic for "does
the embedding contain X."** Single train/val split with weak regularization
(λ=1.0) gave R²=1.000 on degenerate (all-zeros) yaw/pitch targets — the
fit "predicted" zero because the train mean was zero. k-fold with λ=100
plus a degenerate-target guard (`if y.std() < 1e-6: return None`) gave
honest answers.

**Random-shuffle negatives are too easy. Add hard negatives.** Layer 1.3
random-negative AUC = 1.000 was a meaningless ceiling: cross-identity
teacher cosine clusters at ≈0.008. Hard negatives = teacher's top-5 nearest
neighbours pushed the negative cosine to ≈0.18, threshold to ≈0.85 — and
the student still hit AUC ≥ 0.9999. *That* is the discriminative test.
Always add hard negatives to contrastive evals.

**Document every run, especially the failed ones.** The chronological
run-log table in the writeup was load-bearing for the eventual decision to
stop chasing aligned variants. Each failure ("full_pool: 0.30, killed",
"full_crop: 0.53, killed", "full_roi: 0.69, completed but wrong frame") was
plausible-looking at start; only by the time the table had 5 rows did the
shape of the problem become clear (geometry > capacity). Single-run
narratives lie about how predictable the result was.

## On scope & decision-making

**Distinguish "loss-mode" from "guidance-mode" early.** They have different
fidelity bars. We almost shipped a single 0.881 number without saying which
use it qualifies for. Now codified in the "Possible uses" section.

**Stop chasing similarity-aligned at 14×14 once the early curve is visible.**
0.408 at epoch 4 on the aligned14 run was clearly tracking to a worse final
than the 0.881 incumbent. Don't run experiments to completion when the
trajectory has already answered the question — kill, document, move on. We
killed within 5 epochs and saved ~10 GPU-min, the writing time was the
expensive part.

**A research thread should pin its deliverables to which ecosystem consumes
them.** "We want ArcNet, MediaPipe, SigLIP usable as losses without
rendering" — that's the framing that resolved the long-running question of
"why do we care about beating 0.88." Without it, the team chases metric
improvements with no exit condition.

## Carrying these forward

The **MediaPipe head** (next in the series) inherits these as defaults:
- start with frozen backbone where possible (likely a fresh small CNN here
  since MediaPipe FaceLandmarker uses a mobile-style backbone we don't have
  weights for in PyTorch — but inherit the *shape* of "minimal trainable")
- match its reference frame (ARKit blendshapes are computed from a canonical
  face; verify whether MediaPipe's input crop is similarity-aligned or
  axis-aligned)
- precompute teacher labels with explicit masks, including pose validity
  if pose enters the label set
- score with both random and hard negatives where applicable; for
  regression targets, k-fold ridge with degenerate-target guard
- stop when the early-epoch trajectory dominates a known incumbent
- write the use-case fidelity bar into the spec on day one
- save validation reports at `models/<head>/validation_report.json` next to
  the checkpoint, and write a model-card README colocated with it
