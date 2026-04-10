# Semantic Anchor Face Generation v3 — Spec

**Status:** supersedes [2026-04-11-structured-cluster-prototypes-v2.md](2026-04-11-structured-cluster-prototypes-v2.md)
**Date:** 2026-04-11

## Problem

Two failures in v2:

1. **Unrealistic ethnicity distribution.** gemma4 derivation returned 34 Central Asian / 7 Slavic / 0 of the other 4 values — over-reading the "delivery = migrant" stereotype for a majority-Slavic labour market.
2. **No spatial coherence.** Per-cluster LLM derivation treats each cluster independently, so neighbouring clusters in qwen space can get unrelated faces. For the scam-guessr game, players must be able to learn "this region of the map = X face archetype", which requires the face function to be smooth over embedding space. Cluster-based prototypes can't provide that.

## Approach: directional anchors in qwen space

Replace cluster-based LLM prototypes with **text-query anchors** in the native qwen embedding space. Each anchor is a unit vector derived from a hand-written Russian text query encoded through qwen3-embedding:0.6b (the same model that produced `jobs.embedding`). Each job computes cosine similarity to every anchor, softmax over the scores gives blend weights, and the top-3 anchors combine through the existing ConditioningAverage infrastructure.

Why directional instead of point-based:

- **Cosine is the native metric of qwen space.** qwen3-embedding is contrastive-loss trained; angles are meaningful, Euclidean distances less so.
- **No curse of dimensionality.** In 1024-d, Euclidean makes everything look equidistant; cosine doesn't.
- **Smooth everywhere.** Softmax(cos) produces nonzero weight on every anchor for every point — no boundary artifacts.
- **Editorial control through text.** We write the queries; we get the semantic axes we care about.
- **Nameable anchors.** "Курьер доставки еды" is a human-readable name we can surface in the game UI.
- **Uses existing infrastructure.** Same encoder, same top-k selection, same ConditioningAverage blend chain as v8.

Why text queries instead of hand-placed 2D points:

- PaCMAP 2D loses information; qwen 1024-d doesn't.
- Queries are data; placements are editorial work that doesn't version-control well.
- If we swap the embedding model later, we re-encode the queries and we're done.

## Pipeline shape

```
Russian text queries (hand-written, ~30 candidates)
     │
     │ qwen embedding (localhost:11434)
     ▼
Query vectors in qwen space (1024-d, unit-normed)
     │
     │ pairwise cos, drop redundant (cos > 0.85)
     ▼
Candidate anchor set (~20-25)
     │
     │ coverage analysis: for each job, max cos over anchors
     │ knee method: pick smallest N where p50(max_cos) plateaus
     ▼
Final anchor list (N ≈ 10-22)
     │
     │ hand-author face_record per anchor (8-axis schema)
     │ target ethnicity distribution: 40% Slavic + even rest
     ▼
data/face_anchors.json
     │
     │ at generation time per job:
     │   cos(job_emb, anchor_i) for all i
     │   softmax(scale * cos) → weights
     │   top-3 → compose_sentence(anchor_record, sus_band)
     │   ConditioningAverage blend
     ▼
ComfyUI Flux workflow → face PNG
```

## Design details

### Text query language

Queries are written in **Russian**. The corpus is Russian; qwen3-embedding:0.6b is multilingual but Russian queries produce embeddings closer to Russian job postings. No translation indirection.

### Candidate pool size

~30 queries. Dedup step may drop redundant pairs (cos > 0.85) down to ~22-25, then knee analysis may reduce further.

### Deduplication threshold

Two anchors with pairwise cosine > 0.85 are considered redundant; keep whichever has higher average cosine to the corpus (i.e. the better-represented one).

### Coverage metric

For each job `j`, compute `max_cos(j) = max_i cos(job_emb(j), anchor_i)`.

Report the distribution across all jobs:
- p50 (median): how well the average job is covered
- p10: how badly the worst 10% are covered
- p90: how well the best 10% are covered

Coverage is acceptable if **p50 > 0.50 and p10 > 0.35**. If either fails with 30 candidates, we need more queries or better ones — flag and escalate, don't silently proceed.

### Knee analysis for N

Greedy anchor selection by marginal contribution:

1. Start with empty anchor set, compute `max_cos` distribution (just the zero vector baseline).
2. Add the anchor that maximises the increase in median `max_cos`.
3. Repeat until all candidates added.
4. Plot median `max_cos` vs N.
5. Knee = smallest N where adding one more anchor improves median by less than a threshold (default 0.01).

Default N parameter is auto-derived from the knee. User can override with `--num-anchors N` on both the validator and the generator.

### Face records

Each final anchor gets a hand-authored face record matching the 8-axis schema from v2 Task 1 (`prototype_schema.py` carries over unchanged):

```json
{
  "age": "25-35",
  "gender": "masculine",
  "ethnicity": "Slavic",
  "hair": "short dirty blonde",
  "facial_hair": "light stubble",
  "complexion": "smooth",
  "uniform": "black collared polo with company lanyard",
  "expressions": {
    "clean": "calm professional focus",
    "low": "slightly tense",
    "mid": "forced neutrality",
    "high": "darting anxious",
    "fraud": "hollow practiced smile"
  }
}
```

Target ethnicity distribution across N anchors: **40% Slavic, 15% each for Central Asian, Armenian, Mediterranean, East Asian, Middle Eastern**. At N=15, that's 6 Slavic + ~2 each of the others (rounded to integer counts; one bucket may be short). The validator reports actual vs target and flags if any axis is under-represented by more than one anchor.

### Blending at generation time

Per job:

1. Compute `cos(job_embedding, anchor_i)` for all i.
2. Softmax with temperature: `w_i = exp(cos_i / T) / Σ exp(cos_i / T)`, default `T = 0.1` (sharp but not hard).
3. Take top-3 by weight. Renormalise.
4. For each of the 3, `compose_sentence(anchor.face_record, sus_band=job.sus_band)`.
5. Three `CLIPTextEncode` nodes → two-stage `ConditioningAverage` chain (existing infrastructure).

### Sus axis

Unchanged from v2:

- LoRA curve: `sus_factor = max(0.0, (sus - 25) / 75.0) ** 1.2`
- Expressions vary per sus-band at composition time (from the anchor's `expressions` dict)

## Artifacts

New (additive — v8 and v8b logic untouched):

```
data/candidate_anchors.txt        — 30 Russian queries, one per line with optional English comment
data/face_anchors.json            — final anchor list: name, query, embedding, face_record
src/encode_anchors.py             — encode queries via qwen3-embedding:0.6b, save embeddings
src/analyse_anchor_coverage.py    — dedup + coverage + knee analysis
src/validate_face_anchors.py      — ethnicity distribution, schema, orthogonality checks
src/generate_v8c.py               — generation with semantic anchor conditioning
output/dataset_faces_v8c/
```

Modified:

- `src/rbf_conditioning.py` — adds `semantic_anchor_conditioning_nodes()`; v1 and v2 methods untouched

## Validation gates

**Phase 1c gate (pre-generation):**
- Candidate dedup: no surviving anchor pair with cos > 0.85
- Coverage: p50(max_cos) > 0.50 and p10(max_cos) > 0.35
- Every final anchor has a complete face_record (8 axes + 5 expression bands)
- Ethnicity distribution within ±1 anchor of the target (40% Slavic + even rest)
- Enumerated axes use whitelisted values only

**Phase 3c gate (post-smoke):**
- ComfyUI accepts the new workflow (first face generates cleanly)
- Smoke contact sheet shows readable cluster-column variation within each sus-band row
- Clean-band faces at sus ≤ 25 render without LoRA artifacts

**Phase 4 gate (eval):**
- v8c M3 > v8 M3 + 0.05 (semantic anchors actually improve cluster separation)
- v8c M6 ≥ v8 M6 − 0.05 (sus signal not regressed)
- v8c M1 in normal-photo range for clean band

## What we abandon

- `data/cluster_prototypes_v2.json` — kept in git history, not used at generation time
- `src/derive_structured_prototypes.py` — kept for reference, not part of the pipeline
- `src/validate_structured_prototypes.py` — kept for reference
- The v2 gate thresholds (they were designed for cluster-derived prototypes)

## Non-goals

- Not rewriting the 42-cluster partition. Clusters stay as a UI/analysis artifact for the game map; they're no longer a face-derivation primitive.
- Not changing the PaCMAP 2D layout. Same map, same UI.
- Not changing the embedding model. qwen3-embedding:0.6b stays (it is what `jobs.embedding` was populated with).
- Not recalibrating sus beyond the v2 LoRA curve change.
