---
status: live
topic: demographic-pc-pipeline
---

# SCRFD detection threshold on FFHQ (canonical 0.5 vs sweep)

Pre-flight finding from the `arc_latent` distillation plan, while
generating the 10-image reference fixture.

## What we measured

10 first images of `bitmind/ffhq` `train-00000-of-00190.parquet` (1024×1024
RGB), run through `insightface buffalo_l` SCRFD detector at varying
`det_thresh`, `det_size=(640, 640)` (canonical):

| det_thresh | detected | notes |
|---|---|---|
| 0.5 (canonical default) | 6/10 | misses all 4 are clearly visible faces |
| 0.3 | 7/10 | one more recovered |
| 0.1 | 10/10 | all faces recovered, widths 470–585 px (sane, no false positives) |

Misses at thresh=0.5 are all obvious frontal smiling faces (e.g.
`/tmp/ffhq_sample/img_01.png`, `img_06.png`). They are not edge cases
in the human sense.

## Why 0.5 misses obvious FFHQ faces

Working hypothesis (not measured): FFHQ uses dlib's 5-point alignment
with a tight crop that puts the face at ~55–60% of frame area. SCRFD's
buffalo_l detector is trained on WIDER FACE, which has a much wider
distribution of face-to-frame ratios skewed smaller. The anchor priors
that fire most confidently are calibrated for face-fills-30%-of-frame,
not 55%. The detector still finds the face but at lower confidence,
which the 0.5 threshold drops.

Tangentially supports user observation that Flux-generated faces
detect more reliably than FFHQ at the same threshold — Flux generations
tend toward "headshot with shoulders" framing closer to SCRFD's training
distribution. This is a hypothesis to verify later, not a confirmed
finding.

## Decision: keep canonical 0.5 for the corpus builder

Rationale:

1. **Consistency with existing downstream.** Every cached embedding in
   `output/demographic_pc/classifier_scores.parquet` and every ridge
   classifier (glasses, age, gender, race) was produced at the
   `InsightFaceClassifier` default of 0.5. The distillation student
   must match the same teacher distribution. If we lower the threshold
   for the FFHQ corpus, the student learns slightly different features
   than the ridges expect.
2. **The corpus is large.** FFHQ-70k × ~60% detection rate ≈ 42k images.
   That's plenty for distilling a ResNet-18 student. The plan was
   originally over-specified.
3. **No false-positive risk.** SCRFD-at-0.1 gives 10/10 with no
   spurious detections (all face widths 470–585 px), so we *could* drop
   threshold safely if we wanted more data. We don't need to.

## Plan implications

The `arc_latent` plan's pre-flight #1 said "FFHQ is well-aligned,
detection failures should be rare (<1%)." That was wrong. Real expected
miss rate is **30–40%** at thresh=0.5. Update the plan to reflect this
and to make the "filter out None embeddings" path explicit in the
corpus builder, not an afterthought.

Also: when reporting corpus size in the validation battery, report it
as `n_detected / n_total` so the dropped fraction is auditable.

## What to verify later (not blocking)

- Actually measure detection rate over a larger FFHQ slice (e.g.
  one full parquet shard ≈ 370 images) to get a confidence interval
  on the 30–40% estimate.
- Compare to detection rate on the v3.x Flux-generated corpus at the
  same threshold. The user's expectation is that Flux generations
  detect more reliably; numbers would confirm or falsify.
- If the dropped 30–40% has demographic bias (e.g., children, dark-skin
  subjects, head-tilts disproportionately rejected), that bias gets
  baked into the student. Worth a quick correlation check before
  Step 3 (latent-input student training).

## Reference fixture

`tests/fixtures/arc_reference.npz` records the canonical-pipeline
result on these 10 images: 6 valid embeddings (||emb||=1.0000), 4
recorded as `detected=False` with zero-vector placeholder. Future
corpus builders must reproduce these embeddings bit-for-bit on the
same 10 images (verify by sha256) — silent drift here would mask
distillation-quality regressions downstream.
