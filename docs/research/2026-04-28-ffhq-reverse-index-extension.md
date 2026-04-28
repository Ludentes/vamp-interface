---
status: live
topic: demographic-pc-pipeline
---

# FFHQ reverse-index extension (2026-04-28)

## Decision

Run the same five reverse-index metric families that we have on the
Flux-corpus side (mivolo + fairface + insightface attrs + siglip-2
attribute probes + ARKit blendshapes + NMF atom projection from
`au_library.npz`) on all 70k FFHQ images, at 512² resolution to match
Flux corpus dimensions, on the remote Windows GPU box.

## Why

- We already have the FFHQ corpus on the Windows box (95 GB, 190
  parquet shards) for the arc_latent distillation work.
- The Flux corpus is small (~7800 rows) and synthetically biased.
  A real-world reference at 10× the size lets us:
  - sanity-check the classifiers and probes on natural-image
    distributions (gender ~50/50, age curve broader, etc.);
  - compare manifold geometry of synthetic vs real on shared metrics;
  - give the arc_latent student a real-world test set even though its
    training set is also FFHQ.

## How

Single-pass extractor (`src/demographic_pc/extract_ffhq_metrics.py`)
loads all five wrappers once, holds them resident on the RTX 3090,
and processes one parquet shard at a time. Per-shard `.pt` outputs
under `C:\arc_distill\metrics\` are merged locally with the
encode_ffhq.py ArcFace outputs (`C:\arc_distill\encoded\`) into a
unified `output/reverse_index/reverse_index.parquet` keyed by
`image_sha256`.

Detection-rate caveat carried over from
`docs/research/2026-04-27-arcface-detection-threshold.md`: at the
canonical SCRFD det_thresh=0.5, FFHQ has ~40% miss rate from the
ArcFace path. MiVOLO does not detect (predicts on the whole image),
so it returns a number for every row — this is fine for "what does
the model think" but should not be interpreted as "every row is a
face." MediaPipe FaceLandmarker has its own internal detector with
different priors and recovers a different (typically wider) subset.

## Anti-goals

- Not training anything new — purely an extraction pass over existing
  models.
- Not changing the Flux corpus reverse-index schema — only backfilling
  `image_sha256` so the join key exists.
- Not extending the FluxSpace per-render measurements
  (`axis/scale/attn_*`) to FFHQ — those are render-time and don't
  apply to real photos.

## Storage

- Per-shard FFHQ metrics `.pt`: ~0.5 MB × 190 shards ≈ 100 MB
- Per-shard FFHQ encoded `.pt` (already produced by encode_ffhq.py):
  ~50 MB × 190 ≈ 9.4 GB
- Unified `reverse_index.parquet`: estimated 100-200 MB
