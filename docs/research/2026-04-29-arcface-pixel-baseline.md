---
status: live
topic: demographic-pc-pipeline
---

# ArcFace-pixel sanity baseline — gate FAILED (2026-04-29)

## Decision

**FAIL.** Held-out cosine mean = **0.377** on N=1514 val rows
(gate threshold 0.9). Train loss kept improving through epoch 7
(cos loss 0.42, equiv cos 0.58) while val plateaued at 0.377 from
epoch 5 onward — a real but unsurprising overfitting gap.

This is a *clean* failure: the loop trained, the loss decreased, and
the model genuinely learned *something* (val cos 0.38 ≫ random 0).
It just learned far too little of the ArcFace identity signal.

## Why we ran it

Step 2 of the latent-native classifier distillation roadmap
([2026-04-27-latent-native-classifier-distillation.md](2026-04-27-latent-native-classifier-distillation.md)).
The gate validates that distilling InsightFace `buffalo_l` ArcFace 512-d
embeddings into a small ResNet-18 student is well-posed *on raw pixels*
before swapping the input to 16-channel Flux VAE latents.

## What we ran

- **Data.** 70k FFHQ (190 parquet shards) joined with our
  `encode_ffhq.py` outputs (`arcface_fp32 (N, 512)` per row). Filter
  `detected=True`: 24,182 train + 1,514 val rows.
- **Compact prep.** Initial design tried lazy parquet streaming; the
  ConcatDataset over 190 shards eagerly materialised image-bytes columns
  into RAM and OOM'd. Replaced with a one-time `prepare_compact.py`
  pass that decodes + resizes detected rows to 224² uint8, packs into
  one ~3.7 GB tensor (`compact.pt`). Subsequent training reads from
  this tensor with `workers=0` (Windows uses spawn — workers >0 would
  duplicate the tensor per process).
- **Split.** Deterministic SHA-prefix: `image_sha256[0]=='f'` is val
  (~6.25%). Stable across re-runs and identical to the split arc_latent
  will use in step 3.
- **Student.** torchvision `resnet18(weights=DEFAULT)` (ImageNet
  pretrained), fc → `Linear(in_features, 512)`, output L2-normalised.
- **Teacher.** Frozen `arcface_fp32` already on disk — no teacher
  forward at train time.
- **Loss.** `1 - cos(student, target)`.
- **Schedule.** AdamW 3e-4, weight decay 1e-4, batch 128, 8 epochs,
  cosine LR. Resolution 224². Single 3090 (Windows scheduled task).
  Wall time: ~13 min compact prep + ~4 min training.

## Result

```json
{
  "n": 1514,
  "cosine_mean": 0.3768,
  "cosine_median": 0.3777,
  "cosine_p05": 0.1031,
  "cosine_p95": 0.6146,
  "cosine_min": -0.0942,
  "frac_above_0p9": 0.0,
  "gate_passed_mean_gt_0p9": false
}
```

Per-epoch trajectory (from `output/arc_pixel/train_log.jsonl`):

| epoch | train_cos_loss | val_cosine_mean |
|-------|----------------|-----------------|
| 0     | 0.850          | 0.197           |
| 1     | 0.729          | 0.268           |
| 2     | 0.648          | 0.308           |
| 3     | 0.582          | 0.343           |
| 4     | 0.525          | 0.360           |
| 5     | 0.477          | 0.372           |
| 6     | 0.443          | 0.377           |
| 7     | 0.424          | 0.377           |

## Interpretation

The dominant suspect is the **input format mismatch**: ArcFace was
trained on **aligned 112² face crops**, while we hand it whole-image
224² FFHQ portraits and ask the student to perform alignment-invariant
identity extraction in 11 M parameters. That's a substantially harder
task than the teacher itself ever solves. The other usual suspects:

- **Capacity.** ResNet-18 (~11 M) vs ArcFace R50 (~25 M). Plausible but
  unlikely to be the bottleneck given the val plateau before train
  saturation.
- **Train compute.** Train loss was still descending at epoch 7. Longer
  training would likely move val a bit but not close a 0.5 gap.
- **Augmentation.** None applied — no flips, no color jitter. ArcFace
  is partly translation-invariant by training, but the aligned-crop
  pretrain didn't have to see uncentred faces.
- **ImageNet pretrain transferability.** ImageNet features carry some
  identity-relevant geometry (eyes/nose layout) but were never
  specifically trained for face identity.

## What this means for arc_latent (step 3)

This makes step 3 less of a slam-dunk. The latent variant inherits the
same "no alignment" handicap: the Flux VAE encodes the whole 512² image
into 64×64×16 — no face crop is ever extracted. So the latent student
faces the same alignment-invariance problem on top of an even less
identity-aware input format.

**Two ways to react:**

1. **Re-run the gate with aligned crops.** We have `bbox` from
   InsightFace SCRFD per detected row in the reverse-index parquet
   (`output/reverse_index/reverse_index.parquet`); we'd need to backfill
   it into the encoded `.pt` shards (or a side file) and have
   `prepare_compact.py` produce a 112²-aligned-crop variant. If the
   aligned-crop pixel baseline crosses the gate cleanly, the failure
   above is just "wrong input format" and arc_latent's prospects depend
   on whether Flux's VAE preserves identity geometry through
   compression. Worth asking before paying for arc_latent training.
2. **Lower the gate, or change it.** ArcFace cosine 0.9 is a *very*
   high bar even for full-resolution face-crop students; the
   distillation literature commonly reports 0.7-0.85 for face-aligned
   pixel students. A revised gate of e.g. "median cosine > 0.7" is
   defensible but changes what step 3 success means.

The *original* roadmap (option 1 of step 2 from the latent-native
research note) accepted "could be 0.95 cosine R² (great) or 0.6
(problematic)" as the spread of possibilities. We landed below 0.6 —
problematic, but informative.

## Anti-goals

- Not a production model — held-out cosine alone was the gate.
- Not retuning ArcFace teacher.
- Not pre-aligning crops (deliberate, to keep parity with the latent
  variant). The choice is now visible as a load-bearing assumption.

## Next step

Recommend (1): rebuild the compact dataset using SCRFD bboxes from the
reverse-index parquet to produce 112²-aligned face crops, re-run the
gate. Cheap (~15 min) and tells us whether the failure was alignment
or task structure. Defer arc_latent until that's resolved.
