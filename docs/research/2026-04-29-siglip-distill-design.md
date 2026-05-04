---
status: live
topic: arc-distill
---

# SigLIP-distill design — v1 (2026-04-29)

Captures the *initial* design choices for distilling 1152-d SigLIP-2 SO400M
image embeddings from Flux VAE latents `(16, 64, 64)`, so future iterations
know where the starting point came from and which knobs are worth turning if
quality is poor.

## Goal

Make a SigLIP-loss-without-rendering primitive. At slider-training time, given
an anchor latent `z_a` and an edited latent `z_e`, predict the SigLIP image
embedding for each, dot against arbitrary text-prompt embeddings to compute
on-axis / off-axis margins, and use those as preservation losses. This is the
1152-d analogue of v2c's bs head — same trunk, much wider target.

## Data

- **Source pairs**: `output/siglip_distill/compact_siglip.pt` (26,108 rows,
  1152-d fp16, L2-normed), built from `reverse_index.siglip_img_emb_fp16` aligned
  to the SHAs in `arc_full_latent/compact.pt`.
- **Coverage**: 26,108 / 26,108 = 100% (every FFHQ SHA with a latent also has a
  SigLIP embedding).
- **Train/val split**: SHA-prefix `sha[0]=='f'` → val, ~6% — same convention as
  v1/v2c so val sets are comparable.
- **No rendered slice**: unlike v2c, v1 does NOT include the 7,772 flux_corpus_v3
  rows or the 1,344 grid renders. Reason: those don't have Flux VAE latents
  cached in `arc_full_latent/compact.pt` (they were rendered, not encoded). Could
  be added later by VAE-encoding their PNGs, but not for v1.

## Initial design (v1)

### Trunk: same as v2c

```
LatentStemFull64Native     → (B, 64, 112, 112)
ResNet-18 layers           → (B, 512, 14, 14)
AdaptiveAvgPool2d → flatten → (B, 512)
```

11 M trainable parameters. Proven on this exact latent format (shift=0.1159,
scale=0.3611) for the bs head. Reusing it lets us isolate the head/loss as the
only thing that's different from v2c.

### Head: linear, no activation

```
Linear(512, 1152)          → (B, 1152) ∈ ℝ
```

No sigmoid (teacher is in ℝ¹¹⁵² then L2-normed; sigmoid would clip the wrong
side of the distribution). At inference, L2-norm the student output to put it
on the same hypersphere as the teacher.

### Loss: combined 0.5·MSE + 0.5·(1 − cos)

```python
mse  = ((student - teacher) ** 2).mean()
cos  = F.cosine_similarity(student, teacher, dim=-1).mean()
loss = 0.5 * mse + 0.5 * (1 - cos)
```

- **MSE** keeps magnitude near 1 (teacher is unit-norm) — without it the student
  is free to drift to any scale that preserves direction, which would make the
  inference-time L2-norm step lossy.
- **(1 − cos)** directly optimizes the quantity downstream slider losses use
  (margin = `dot(siglip_img, siglip_text)` ∝ cosine when both are unit-norm).
- Equal weighting is a starting choice, not a result. If MSE saturates while
  cosine still improves (or vice versa), tune the mix.

## Iteration plan if v1 results are poor

These are the knobs in priority order. Pulling them in this order isolates one
variable per run:

| Order | Change | Why |
|---|---|---|
| 1 | **Pure (1 − cos) loss, drop MSE** | If R²/cosine plateaus with mixed loss, MSE may be over-constraining magnitude on a directional target. Standard contrastive-style finding. |
| 2 | **ResNet-50 trunk** (~25 M params, ~2× train time) | The 1152-d teacher is much higher-bandwidth than v2c's 52-d; ResNet-18's 512-d bottleneck may be the cap. |
| 3 | **Project teacher to lower-d, distill that, lift back** | If full 1152-d is too hard, distill a 256-d projection (PCA on teacher set) and recover full at inference. Loses some text probes but easier to fit. |
| 4 | **Add rendered slice** | VAE-encode the 7,772 flux_corpus_v3 PNGs to latents, append to training set. More diverse expressions/demographics; helps if val cosine is bottlenecked by FFHQ-tail coverage. |
| 5 | **Per-axis temperature on cosine** | If only specific text-axis margins (e.g., glasses, beard) are the bottleneck, can't fix at the embedding level — would need a probe-specific head. Last resort. |

Don't pull #5 without exhausting #1–#4. Don't pull #2 before #1 (knob #1 is
free; knob #2 is 2× compute).

## Validation criteria for "v1 works"

- **Cosine ≥ 0.85** on held-out val (subjective bar — SigLIP face↔face cosine
  baseline is ~0.7, so 0.85 means we're recovering most of the directional info)
- **Per-text-probe margin R² ≥ 0.5** on the 12 SigLIP probes already in
  reverse_index (bearded, smiling, glasses, etc.) — this is the actual
  downstream use, so it's the load-bearing metric
- **Round-trip sanity**: predict embedding, dot against teacher's "glasses" probe,
  compare to `sg_glasses_margin` for the same image. R² should be ≥ 0.5 on the
  shippable side.

If we hit 0.85 cosine but probe R² is < 0.3 across the board, that means cosine
is being driven by face-vs-non-face structure (easy) and not by attribute
direction (the thing we care about) — pull knob #1 first.

## Why one-shot this rather than ablation grid

We have a clean comparison target (v2c bs distill on the same trunk + latent
format) and a downstream eval rig (glasses_v9 vs glasses_v8 with the new loss).
Iterating on the loss/trunk before validating downstream impact is YAGNI — if
v1's cosine 0.85 turns into a glasses_v9 that just matches v8, no amount of
loss-formulation ablation upstream will change that. Build the simplest baseline
that's plausibly good, run the downstream test, then iterate.

## Source artifacts

- Pair file: `output/siglip_distill/compact_siglip.pt`
- Build script: `src/siglip_distill/build_compact_siglip.py`
- Extraction: `src/siglip_distill/extract_siglip_features.py`
- Merge: `src/siglip_distill/merge_into_reverse_index.py`
- Reverse-index column: `siglip_img_emb_fp16` (1152-d L2-normed fp16, 100% coverage)

## What's NOT done in v1 (deferred)

- Rendered slice (7,772 flux_corpus_v3 PNGs) — not VAE-encoded yet
- Calibration_expanded / crossdemo renders — not in reverse_index at all
- Per-probe distill heads — explicitly rejected; one general embedding is
  the design goal
