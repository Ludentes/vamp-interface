# Visualization System — How It Works

**Date:** 2026-04-11  
**Status:** current (v8d)

This document describes the end-to-end system: how job postings become faces, how the map is structured, and what the measurements say.

---

## 1. The Core Idea

Each job posting is a point in a high-dimensional semantic space. Close points = similar jobs. The visualization renders each job as a photorealistic face. The face encodes two things simultaneously:

- **Who** the job is for (cluster membership → face archetype)
- **How suspicious** it is (sus_level → uncanny valley drift)

The result: a player who looks at the face map learns to *feel* which regions contain scams before they can articulate why.

---

## 2. Embedding Space

**Model:** `qwen3-embedding:0.6b` via Ollama at `localhost:11434`  
**Dimension:** 1024-d  
**Metric:** cosine similarity (the native metric for contrastive-loss trained models)

Each job posting's `raw_content` is encoded as plain text into a unit-normed 1024-d vector. These vectors live in the telejobs Postgres DB as `jobs.embedding` (pgvector). The corpus is ~23,800 jobs from ~20 Telegram channels.

**Why this model:** verified cos=1.0000 on a re-embed round-trip. The corpus was embedded once; the embedding is stable. It's the asymmetric retrieval variant — job text is encoded as documents, anchor queries are encoded with the retrieval instruction prefix.

**Space properties:** bimodal corpus — safe (sus 0–30) = 54%, fraud (sus 70–100) = 28%, ambiguous middle = 18%. The safe and fraud poles are structurally distinct in embedding space, not just label-different.

---

## 3. The Map (PaCMAP 2D Layout)

The 1024-d embeddings are projected to 2D using **PaCMAP** (`n_neighbors=15`, seed=42). This gives the game map — every job gets an (x, y) coordinate.

PaCMAP preserves local neighbourhood structure better than UMAP for this corpus. The 2D layout is a lossy but navigable picture of the semantic space. Cluster boundaries in 2D roughly correspond to semantic boundaries in 1024-d.

The map is precomputed offline. It does not change at query time.

**What the map shows:**
- Spatial regions corresponding to job categories (delivery, office, construction, etc.)
- Fraud concentration: fraud jobs cluster in certain regions but are also scattered throughout — they mimic legitimate job categories
- Source variation: different Telegram channels concentrate in different regions

---

## 4. Countries (Anchor System)

The map is divided into **10 "countries"**, each corresponding to a semantic anchor — a hand-authored job-posting query in Russian prose, encoded by qwen3-embedding:0.6b with the retrieval prefix.

| Anchor (slug) | Face archetype | Gender | Ethnicity | Age |
|---|---|---|---|---|
| пеший курьер | Food courier | M | Central Asian | 18–25 |
| разнорабочий | Construction worker | M | Middle Eastern | 35–45 |
| водитель-курьер | Car courier | M | Armenian | 35–45 |
| офис-менеджер | Office manager / secretary | F | Slavic | 25–35 |
| backend разработчик | Software developer | M | Slavic | 25–35 |
| официант-бармен | Restaurant / bar staff | M | Mediterranean | 18–25 |
| кассир | Supermarket cashier | F | Slavic | 45–55 |
| охранник | Security guard | M | Slavic | 55–65 |
| сиделка | Caregiver / nurse | F | East Asian | 45–55 |
| уборщица / горничная | Cleaner / housekeeper | F | Central Asian | 35–45 |

**Why 10:** game design sweet spot. Too few (< 7) → map feels empty; too many (> 15) → players can't learn the distinctions. 10 is the casual-GeoGuessr number.

**Why prose queries, not cluster centroids:** qwen3 is contrastive-loss trained; angles (cosine) are meaningful, Euclidean distances less so. Prose queries produce embeddings that sit naturally among real job postings. Cluster centroids in 1024-d have no guarantee of landing in a meaningful semantic region.

**Why Russian prose, not English:** the corpus is Russian. The encoder expects Russian prose; queries written as English phrases produce systematically lower cosines to the actual jobs.

### Anchor assignment (blend weights)

For each job at generation time:

1. Compute `cos(job_embedding, anchor_i)` for all 10 anchors.
2. Apply softmax with temperature **T = 0.02**: `w_i = exp(cos_i / 0.02) / Σ exp(cos_j / 0.02)`.
3. Take top-3 by weight. Renormalize to sum to 1.

At T=0.02: top-1 weight ≈ 63%, top-3 cumulative ≈ 89%, effective number of active anchors ≈ 2.3. This means each face is dominated by one anchor but smoothly blended with its 1–2 nearest neighbours. Hard Voronoi boundaries are avoided.

T=0.01 would produce near-one-hot assignments (effective_N ≈ 1.2) — hard country borders, broken smoothness. T=0.1 would produce mush across all 10 anchors (effective_N ≈ 8) — every face looks the same. T=0.02 is the calibrated sweet spot.

**Dominant country assignment:** `argmax(softmax(cos, T=0.02))` — the anchor with the highest weight. This is what the contact sheet and balanced sampler use.

---

## 5. Face Generation

**Model:** Flux.1-krea-dev (fp8_e4m3fn, img2img at denoise 0.80)  
**Base image:** one fixed neutral anchor face (`output/phase1/phase1_anchor.png`)

Every face is a perturbation of the same anchor image. This gives identity coherence across the dataset: the anchor is the "normal person", and all generated faces are variations of it.

### Prompt (CLIP/T5 conditioning)

The CLIP/T5 text conditioning is composed from the anchor blend:

1. Top-3 anchors' `face_record` fields (age, gender, ethnicity, hair, complexion, uniform, expression for the sus_band) are composed into a sentence each via `compose_sentence()`.
2. The 3 sentences are encoded to CLIP conditioning.
3. They are blended via ComfyUI `ConditioningAverage` nodes in proportion to the top-3 softmax weights.

The prompt encodes *who* the face is. It does not mention fraud or suspicion — those come through the uncanny axis.

### Uncanny axis (sus_level → LoRA strength)

The `sus_level` (0–100, from telejobs classifier) drives the uncanny valley signal through two channels:

**1. Denoising strength:** fixed at 0.80 for all jobs. The drift from the anchor happens through the LoRA mix, not denoise variation (v8d simplification).

**2. LoRA strength ramp:**
```
sus_ramp = max(0.0, (sus_level - 25) / 75.0) ** 1.2
```
This gives a delayed, convex ramp: clean jobs (sus < 25) get zero LoRA, the curve accelerates through the mid-high range, and peaks at sus=100.

**LoRA mix at sus=100 (the reptiloid peak):**

| LoRA | Strength at sus=100 | Role |
|---|---|---|
| `Cursed_LoRA_Flux.safetensors` | 0.5 | Sharpens uncanny without zombie artifacts |
| `Eerie_horror_portraits.safetensors` | 0.0 | **Disabled** — was the skin-lesion/zombie culprit |
| `Strange_and_unsettling.safetensors` (rank 2) | 1.5 | Core strangeness; needs high strength because rank 2 |
| `horror_nova.safetensors` (rank 64) | 1.5 | Core horror; trigger-gated, requires trigger text |

**Trigger text** (injected via `ConditioningConcat`, not prepended to face prompt):
```
horror-nova, eerie, creepy, haunted, macabre, nightmare,
strange and unsettling, weathered, ancient, gnarled
```

The trigger is injected via ConditioningConcat after the face conditioning blend, so it doesn't corrupt the semantic anchor blend.

At intermediate sus values (e.g. sus=60): `sus_ramp = max(0, (60-25)/75)^1.2 ≈ 0.40`. LoRA strengths are `0.5×0.40 = 0.20`, `1.5×0.40 = 0.60`, etc. The face drifts toward uncanny but isn't yet reptiloid.

**Why this produces uncanny, not horror:** at sus=100, the face reads as "something is wrong with this person" — slightly too angular, slightly too still, slightly too composed. The mix was calibrated to avoid the zombie/skin-lesion failure mode from v8c (Eerie_horror_portraits was the culprit; disabled entirely in v8d).

---

## 6. What the Metrics Say

Measured with ArcFace IR101 identity fingerprints (style-agnostic, works equally on SDXL and Flux output):

| Dataset | Separation score | Sus correlation | Verdict |
|---|---|---|---|
| v7 (SDXL, basic LoRA) | 0.075 | −0.057 | not validated |
| v8c (Flux, cursed LoRA, no triggers) | 0.047 | −0.033 | not validated |
| **v8d (Flux, reptiloid mix, triggers)** | **0.142** | **+0.353** | **encoding validated** |

**Separation score:** `(inter_dist − intra_dist) / inter_dist` over sus_bands. Higher = faces in different sus bands are more distinct from each other than faces within the same band.

**Sus correlation:** Pearson r(anchor_dist, sus_level) over all 1,268 v8d faces. `+0.353` means faces with higher sus_level are measurably farther from the neutral anchor in identity space. This is the primary validation: the encoding *works*.

**Per-band anchor distance (v8d):**

| sus_band | n | avg anchor dist | r(dist, sus) |
|---|---|---|---|
| clean (0–25) | 290 | 1.150 | +0.082 |
| low (26–49) | 264 | 1.147 | −0.233 |
| mid (50–64) | 233 | 1.127 | −0.080 |
| high (65–78) | 206 | 1.161 | +0.049 |
| **fraud (82–100)** | **275** | **1.269** | **+0.365** |

The fraud band is the signal carrier: avg anchor distance jumps from ~1.13–1.16 for clean/low/mid/high to **1.269** for fraud. The ramp delay means mid/high bands are in the curved portion — the signal builds slowly and then breaks sharply at the top.

---

## 7. Map Visualization: Countries and Sources

### Naming countries

Each country (anchor) has a human-readable name derived from the anchor query. For the game UI, the names should be evocative but neutral — not "scam cluster" but "delivery district". The face archetype *implies* the cluster without labeling it.

Potential display: hovering a region shows the anchor name + dominant job examples. The player sees the face archetype first, the label only on demand.

### Showing source data

Each job has a source Telegram channel. On the map, source can be shown as:

- **Color overlay:** each channel gets a distinct hue; opacity ∝ density. Fraud-heavy channels will visually concentrate in specific regions.
- **Filter:** "show only jobs from @channel_X" reveals the channel's semantic footprint — which countries it posts in, whether it skews toward high-sus regions.
- **Source profile:** per-channel stats: anchor distribution, sus level histogram, top clusters. Useful for the analyst persona.

The 20 channels vary substantially in corpus contribution and fraud rate. Some channels are entirely fraud (sus 90–100 only); others are entirely legitimate. The map makes this visible spatially.

### Cluster naming

Coarse PaCMAP clusters (HDBSCAN C0–C42) have natural names derivable from the dominant anchor + sus band:

- C20 (largest, ~2,500 jobs) — the biggest coherent semantic cluster; dominant anchor determines its "country" on the map
- Noise (11.6%) — jobs that don't fit cleanly into any cluster; scattered across the map; highest fraud rate

Cluster names for the game should come from the dominant face archetype, not the HDBSCAN ID. E.g., "Delivery District", "Office Quarter", "Security Zone".

---

## 8. Generation Pipeline Summary

```
telejobs DB (jobs.embedding, jobs.sus_level)
    │
    ├─ cosine to 10 anchors → softmax T=0.02 → top-3 weights
    │      → compose_sentence(face_record, sus_band) × 3
    │      → ConditioningAverage blend in CLIP space
    │
    ├─ sus_level → sus_ramp = max(0, (sus-25)/75)^1.2
    │      → 4-LoRA chain strengths (cursed, eerie, strange, horror)
    │      → ConditioningConcat trigger text (if sus_ramp > 0)
    │
    └─ ComfyUI Flux.1-krea-dev img2img (denoise=0.80, fixed anchor image)
           → face PNG (256×256)
           → manifest.json entry {sus_level, sus_band, lora_strengths, seed, ...}
```

Seed is deterministic per job_id: `seed = abs(hash(job_id)) % 2^32`. Same job always produces the same face.

**Current dataset:** v8d, ~1,270 PNGs, balanced across 10 anchors × 5 sus_bands (target 24 per cell). Encoding validated.
