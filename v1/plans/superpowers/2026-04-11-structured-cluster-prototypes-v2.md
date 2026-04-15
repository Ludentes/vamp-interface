# Structured Cluster Prototypes v2 — Spec Addendum

**Status:** addendum to [2026-04-10-manifold-preserving-face-generation-design.md](2026-04-10-manifold-preserving-face-generation-design.md)
**Date:** 2026-04-11

## Problem

Phase 1 (commits `0172d46` → `74c7e03`) produced `data/cluster_prototypes.json` with 42 entries that technically passed the distinctness gate (pairwise cosine std = 0.077, closest pair 0.918) but collapsed on manual inspection: roughly 30 of 42 clusters describe a variant of "late-twenties to early-forties, gender-neutral, wiry/sturdy build, sun-weathered or dusty skin, slightly hunched posture, weary focused gaze". Only screen-based gig clusters (C5, C6, C8, C13, C15, C16) and a handful of office/service clusters break the pattern.

Phase 3 smoke-test inspection (23 faces in `output/dataset_faces_v8/`) confirms the collapse shows up visually: within-cluster faces differ (because denoise=0.80 allows variation) but across-cluster faces look like the same archetype in different sus bands. High sus is apparent because it's driven by LoRAs (orthogonal axis), not by prompt content. The LoRA curve is also too aggressive at the low end — sus=25 already reads as cursed.

## Goals

1. Force cluster prototypes onto visibly distinct face-space axes so within-band cluster differentiation is readable by a human.
2. Soften the LoRA curve so clean postings render as normal photographs.
3. Add per-sus-band expression variation so the sus axis modulates the same person's mood, not just their weirdness.
4. Change sampling so every cluster is represented equally rather than letting the largest clusters dominate.

## Approach

### 1. Structured face-vector schema

Replace the free-form single sentence with a JSON object per cluster using 8 fixed axes and 5 per-sus-band expression values. The LLM's job is to fill in the axes; composition into a prompt sentence is deterministic.

```
Axes (per cluster, single value):
  age                 ∈ {"18-25", "25-35", "35-45", "45-55", "55-65"}
  gender              ∈ {"masculine", "feminine", "androgynous"}
  ethnicity           ∈ {"Slavic", "Central Asian", "Armenian",
                         "Mediterranean", "East Asian", "Middle Eastern"}
  hair                ∈ open string (e.g. "short dark", "shaved", "long braided blonde")
  facial_hair         ∈ open string, or "none" for non-masculine (e.g. "clean-shaven",
                        "stubble", "full black beard", "grey moustache")
  complexion          ∈ {"smooth", "pale indoor", "sun-weathered", "wind-chapped",
                         "ruddy", "oily", "waxy", "lightly scarred"}
  uniform             ∈ open string describing clothing at shoulders
                        (e.g. "bright yellow hi-vis vest over hoodie",
                         "black collared polo with company lanyard")

Expressions (per cluster, five values — one per sus band):
  expressions.clean   ∈ open short phrase (e.g. "calm neutral gaze")
  expressions.low     ∈ open short phrase
  expressions.mid     ∈ open short phrase
  expressions.high    ∈ open short phrase
  expressions.fraud   ∈ open short phrase (e.g. "hollow darting eyes, forced smile")
```

The enumerated axes (age, gender, ethnicity, complexion) use strict value sets to prevent gemma4 from inventing near-duplicates like "slightly tanned" / "lightly tanned" / "moderately tanned". The open-string axes (hair, facial_hair, uniform, expressions) need the vocabulary freedom to encode cluster specifics.

### 2. Ethnicity distribution

Realistic for a Russian-speaking labour market. The model is instructed to assign ethnicity based on the statistical realism of the cluster — delivery and construction skew Central Asian and Armenian, IT and office skew Slavic, domestic services and retail are mixed. No forced diversity; just accurate representation. "Armenian" is the Flux-friendly label for the South Caucasus demographic — the word "Caucasian" would collide with its English meaning of generic white.

### 3. LoRA curve

New curve: `sus_factor = max(0.0, (sus - 25) / 75.0) ** 1.2`

| sus | old curve | new curve |
|---|---|---|
| 0 | 0.00 | 0.00 |
| 25 | 0.33 | 0.00 |
| 40 | 0.48 | 0.146 |
| 65 | 0.72 | 0.407 |
| 80 | 0.84 | 0.616 |
| 100 | 1.00 | 1.00 |

Clean postings (sus ≤ 25) receive zero LoRA weight. The cursed/eerie weight ratio stays at 1.0× and 0.75× of sus_factor as before.

### 4. Per-sus-band expression composition

Prompt sentence template (deterministic):

```
A {age} {ethnicity} {gender_noun} with {complexion} skin,
{hair} hair, {facial_hair_clause}, wearing {uniform}, {expression_phrase}.
```

where `gender_noun` maps masculine→"man", feminine→"woman", androgynous→"person", and `facial_hair_clause` is either `facial_hair` verbatim or empty for feminine/androgynous. `expression_phrase` is looked up per sus_band from the cluster's `expressions` dict.

The LLM derives one JSON object per cluster. Composition happens at generation time based on the job's sus_band, producing 42 × 5 = 210 distinct prompt sentences from 42 JSON records.

### 5. Derivation via gemma4 with thinking enabled

Phase 0 discovered gemma4's `think: false` still burns `num_predict` tokens on hidden reasoning. Rather than fight that, we enable thinking deliberately: `think: true`, `num_predict: 4000`. Gemma4's thinking produces better structured output on reasoning-heavy tasks; we trade latency for quality. Expected cost: ~5 s per cluster × 42 clusters ≈ 4 min total.

Prompt demands strict JSON output with enumerated values quoted verbatim. Post-parse validates the axes and falls back to a neutral skeleton on parse failure for individual clusters.

### 6. Per-cluster sampling

New sampler: `select_jobs_per_cluster(n_per_cell: int)`. For each (cluster_coarse, sus_band) pair, pick `n_per_cell` jobs deterministically by (seed, cluster, band) ranking. For this iteration, `n_per_cell = 2` → 42 × 5 × 2 = 420 cells max, probably ~370 after dropping empty cells (some small clusters may not have jobs in every sus band).

Total runtime at ~20 s per Flux generation: ~2 hours.

### 7. RBFConditioner change

Current method `cluster_prototype_conditioning_nodes(embedding, clip_ref, start_node_id)` looks up prototypes by cluster only. The new method `structured_prototype_conditioning_nodes(embedding, sus_band, clip_ref, start_node_id)` accepts sus_band and composes the sentence per-cluster at call time.

Loads `data/cluster_prototypes_v2.json` in `__init__`. The old method stays for v8-era comparison; tests cover both.

## Validation

**Derivation gate (Phase 1b):**
- `data/cluster_prototypes_v2.json` has exactly 42 entries
- Every entry has all 8 required fields filled (no nulls)
- Every `expressions` dict has all 5 band keys
- Enumerated axes (age, gender, ethnicity, complexion) use only whitelisted values
- Composed sentences pairwise cosine std > 0.12 (higher bar than v1's 0.05)
- Composed sentences max pairwise cosine < 0.97 (down from 0.99)

**Regeneration gate (Phase 3b):**
- ComfyUI accepts the new workflow
- Smoke run (5 clusters × 5 bands × 1 per cell = 25 faces) produces visually distinct faces
- Full run (~370 faces) completes without error
- Contact sheet shows readable cluster-column variation within each sus-band row

**Eval gate (Phase 4, unchanged):**
- v8b beats v6 and v7 on M3 (+0.10), M4, M5 (≥7/10)
- v8b does not regress M6 vs v8
- Bonus: v8b should beat v8 on M3 (that's the whole point of this iteration)

## Migration

New artifacts (all additive — v8 stays intact for A/B):

```
src/prototype_schema.py                     — constants, compose()
src/derive_structured_prototypes.py         — gemma4 structured derivation
src/validate_structured_prototypes.py       — gate for Phase 1b
src/generate_v8b.py                         — new LoRA curve, sampler, conditioning
data/cluster_prototypes_v2.json             — 42 structured records
output/dataset_faces_v8b/                   — new batch
output/dataset_faces_v8b/manifest.json
output/dataset_faces_v8b/contact_sheet.html
```

Modified:
- `src/rbf_conditioning.py` — adds `structured_prototype_conditioning_nodes()`; existing method untouched

## Non-goals

- Recalibrating the sus axis beyond the LoRA curve softening. If the new curve still feels wrong at specific sus levels we tune later.
- Replacing gemma4. We explicitly chose to stay with gemma4 + thinking ON; gpt-oss and glm-4.7-flash are fallbacks if validation fails.
- Changing the cluster partition. The 42-cluster qwen partition stays as input.
- Changing Phase 4 metrics. The eval harness is the same — we just run it on one more dataset.
