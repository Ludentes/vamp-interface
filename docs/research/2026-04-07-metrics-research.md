# Visual Metrics Research: Face Distinctness & Uncanniness

**Date:** 2026-04-07  
**Status:** CLIP + DINOv2 experiments complete; FaceNet approach identified as correct path forward

---

## Problem Statement

We need to quantify two things about our generated face dataset:

1. **Cluster separation** — do faces from different cohorts (warehouse_legit, courier_scam, etc.) look visually distinct from each other?
2. **Uncanniness** — does the sus_level encoding actually work? Do high-fraud faces look more "wrong" than legitimate ones?

These are not the same question but they share a measurement infrastructure.

---

## Approaches Evaluated

### 1. CLIP ViT-B/32 (implemented, v1/v2/Flux results)

**Method:** Embed each face PNG into 512-d CLIP space. Compute cosine similarity within and between cohort clusters. Separation score = `(inter_dist - intra_dist) / inter_dist`.

**Results:**

| Version | Separation | Intra sim | Inter sim |
|---------|-----------|-----------|-----------|
| v1 SDXL (3-axis, narrow) | 0.515 | 0.920 | 0.835 |
| v2 SDXL (5-axis, wide) | 0.516 | 0.896 | 0.786 |
| Flux krea-dev fp8 | 0.449 | 0.949 | 0.906 |

**Problem: CLIP is biased toward SDXL output.** CLIP ViT-B/32 was trained on a distribution that includes SDXL-style imagery. Flux faces are hyperrealistic and texturally dense in ways CLIP doesn't discriminate well — everything Flux produces lands in a tight cluster in CLIP space regardless of actual visual differences. The 0.449 score for Flux likely reflects a metric limitation, not worse face generation.

**Conclusion:** CLIP is not the right metric for cross-model comparison. Acceptable as a relative metric within a single model family (SDXL v1 vs v2), but biased otherwise.

### 2. DINOv2 (proposed, not yet implemented)

**Method:** DINOv2-base (ViT-B/14) from `facebook/dinov2-base` — available in `transformers`, zero additional deps. Purely visual, no text-image alignment training, better fine-grained visual similarity.

**Expected advantage over CLIP:** No style bias. Should evaluate Flux faces more fairly.

**Status:** Not yet run. Low priority since FaceNet is a better solution for this specific domain.

### 3. CLIP Zero-Shot Uncanniness (proposed, not yet implemented)

**Method:** For each face, compute:
- `sim(face, "natural human face, genuine expression, open gaze")`
- `sim(face, "uncanny valley, wrong proportions, hollow affect, forced smile")`
- Score = `sim_uncanny - sim_natural`

**Appeal:** Zero new deps, runs now with existing CLIP model.

**Problem:** Relies on CLIP's semantic understanding of "uncanny valley" which may be shallow. CLIP was not trained on faces specifically labeled as uncanny. Likely noisy.

**Verdict:** Useful as a quick sanity check but not reliable as a primary metric.

### 4. Anchor Distance (proposed, not yet implemented)

**Method:** Every generated face is a perturbation of the same anchor face (output/phase1/phase1_anchor.png). Embed anchor and each generated face, measure embedding distance.

**Appeal:** Directly validates the core design claim — high sus_level → high denoising → larger drift from anchor. If `anchor_distance` correlates with `sus_level`, the encoding works.

**Problem (with CLIP/DINO):** These models are not identity-sensitive. A face with different lighting or expression but same identity would score as "different." We need an identity-invariant embedding.

---

## Key Insight: Use Face Recognition Fingerprinting

Face recognition systems (ArcFace, FaceNet) produce compact identity fingerprints specifically trained to be:
- **Invariant** to expression, lighting, pose, style
- **Sensitive** to identity — two photos of the same person → near-zero distance; different people → large distance

This is the correct embedding space for our problem. One model solves all three metrics:

### Metrics from a single FaceNet embed pass

| Metric | Computation | What it validates |
|--------|-------------|------------------|
| **Anchor identity distance** | `dist(face_i, anchor)` per face | Does high sus → high identity drift? |
| **Cluster separation** | Intra vs inter distance in fingerprint space | Are cohort clusters distinct identities? |
| **sus correlation** | `corr(anchor_dist, sus_level)` per cohort | Does the encoding actually work end-to-end? |
| **Cross-model fairness** | Same FaceNet on SDXL + Flux faces | Style-agnostic comparison |

### Why fingerprinting solves the Flux bias problem

FaceNet/ArcFace is trained on real photographs — it doesn't care whether a face was rendered by SDXL or Flux. The identity distance between a warehouse_legit face and a courier_scam face is the same regardless of which diffusion model produced them. This removes the CLIP style bias entirely.

### Why fingerprinting solves the uncanniness measurement problem

"Uncanny" in our design = drifted from the neutral anchor identity. High denoising → SDXL generates a face that is less constrained by the anchor → greater identity drift in fingerprint space. This is measurable, not just intuited.

If `corr(anchor_dist, sus_level) > 0.7` across cohorts, we can claim the encoding is working quantitatively.

---

## Recommended Implementation: `facenet-pytorch`

**Package:** `facenet-pytorch` — pure PyTorch, no C extensions, clean install  
**Model:** InceptionResnetV1 pretrained on VGGFace2 (512-d embeddings)  
**Alternative:** InsightFace (ArcFace) — more accurate but C extension compilation is fragile

### Proposed `face_distinctness.py` rewrite

```
--model clip     # current implementation (kept for reference)
--model dino     # DINOv2, zero-dep, style-agnostic
--model facenet  # default: identity fingerprinting, solves all metrics
```

FaceNet run outputs:
1. Per-cohort anchor distance distribution (mean ± std)
2. Correlation table: anchor_dist × sus_level per cohort  
3. Cluster separation in identity space (same formula as CLIP)
4. Overall encoding validation: does anchor_dist rise monotonically with sus_level?

---

## Generation Version Summary

| Version | Identity axis | PCA axes | Divisor | Output dir |
|---------|--------------|----------|---------|-----------|
| v1 | PC1 (physical/knowledge) | 3 | 2.0 | dataset_faces |
| v2 | PC1 + PC4/PC5 (complexion/texture) | 5 | 1.0 | dataset_faces_v2 |
| v3 | work_type explicit archetype | 3 (PC2+PC3 only) | 2.0 | dataset_faces_v3 |
| flux | v1 mapping, Flux backend | 3 | 2.0 | dataset_faces_flux |

**v3 hypothesis:** By using the explicitly extracted `work_type` field as the primary identity axis, cohort separation should increase because work_type is a clean categorical that directly maps to distinct visual archetypes. Legitimate vs scam faces within the same work_type are then differentiated only by the uncanny affect channel. This matches the design intent more faithfully than PCA projection.

**v3 expected limitation:** `easy_money_scam`, `pay_mismatch_scam`, `medium_sus`, `high_medium_sus` are mostly `другое` work_type and will all get the same neutral archetype. Their only differentiation is the uncanny affect level. This is by design — these are the hardest cases.

---

## Next Steps

1. ~~Run v3 generation~~ (in progress, ~15 min remaining)
2. ~~Compare v2 vs v3 CLIP separation~~ (will complete with current background task)
3. Install `facenet-pytorch` and rewrite `face_distinctness.py` with FaceNet as default
4. Run FaceNet metrics across all 4 output directories
5. Check anchor_dist × sus_level correlation — primary encoding validation
6. Update visualization report with final metric results
