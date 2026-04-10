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
Russian job-posting prose queries (hand-written, ~34 candidates)
     │
     │ qwen3-embedding:0.6b with retrieval instruction prefix
     ▼
Query vectors in qwen space (1024-d, unit-normed)
     │
     │ diagnostics: dedup, greedy order, blend concentration at T=0.02
     ▼
Candidate pool (up to 34; used only for ranking + diagnostics)
     │
     │ CURATE to N = 10 by hand, balancing game-map diversity
     │ (age, gender, ethnicity, role archetype) against corpus coverage
     ▼
Final anchor list (N = 10 — "countries" on the game map)
     │
     │ hand-author face_record per anchor (vivid 8-axis descriptors)
     │ target ethnicity distribution: 40% Slavic + roughly even rest
     ▼
data/face_anchors.json
     │
     │ at generation time per job:
     │   cos(job_emb, anchor_i) for all 10
     │   softmax with temperature T=0.02 → weights
     │     (top-1 ≈ 63%, top-3 ≈ 89%, effective_N ≈ 2.3)
     │   optional post-softmax power sharpening (default k=1.0)
     │   top-3 → compose_sentence(anchor_record, sus_band)
     │   optional conditioning-space amplification (default α=1.0)
     │   ConditioningAverage blend
     ▼
ComfyUI Flux workflow → face PNG
```

## Design details

### Text query language and format

Queries are written in **Russian** as short 1-2 sentence job-posting prose, not phrase-lists. Example:

> `Требуется пеший курьер для доставки еды и заказов по городу. Работа с мобильным приложением, гибкий график, оплата за каждую доставку.`

Two reasons for prose over phrase-lists:
1. The corpus is prose. The document encoder expects prose-shaped text; phrase-lists produce embeddings that are systematically further from real jobs.
2. Queries double as editorial hints to the face_record author — a vivid prose query makes it easier to write a vivid face_record.

### Query encoding: asymmetric retrieval prefix

qwen3-embedding:0.6b is a contrastive retrieval model with an **asymmetric query/document format**. Documents (the jobs in postgres) were encoded as plain text (verified cos=1.0000 on a round-trip re-embed). Queries (our anchors) must be encoded with the retrieval instruction prefix:

```
Инструкция: Найди вакансии, соответствующие роли.
Запрос: <query>
```

Without the prefix, queries behave like plain documents and the per-query→per-doc cosines are systematically lower. With the prefix, top-1 cosines for matching jobs rise by ~0.05-0.10.

### Candidate pool size and curation

- **Candidate pool:** ~34 queries (the pool used for ranking and diagnostics).
- **Final N:** **10**, hand-curated from the pool.

Why 10, hand-curated: this is a game (scam-guessr), not a retrieval system. Each anchor becomes a *country* on the player's 2D map — a learnable face archetype that occupies a visible PaCMAP neighborhood. Too few (N<7) leaves the map feeling empty; too many (N>15) becomes a pro-level geoguessr where players can't learn the distinctions. **N=10 is the casual-game sweet spot.**

Greedy coverage order is used as *input* to the curation, not as the final selector — the top-10 greedy picks over-represent delivery/warehouse/office because the corpus is skewed, and leave the ethnicity/age/gender axes underspread. A small hand-curation step picks 10 that span the visual axes.

### Deduplication threshold

Two anchors with pairwise cosine > 0.85 are considered redundant; keep whichever has higher median cosine to the corpus. In practice this triggered 0 drops at N=34 candidates — prose queries are distinct enough.

### Blend concentration metric (replaces max_cos coverage)

The v3 generation uses a **softmax top-3 blend** at T=0.02. What matters is not "does the best anchor match job X closely" (max_cos), but "when I blend the top-3 anchors for job X, does that blend concentrate on a few anchors or mush across all of them".

Metrics computed on the full job corpus:

- **Top-3 weight sum, p50:** how much of the softmax mass lands in the top-3 for a typical job. Target: **≥ 0.80**.
- **Effective N anchors, p50:** `1 / Σ(w_i²)`. Target: **≤ 4.0** (ideally 2-3).
- **Top-1 weight, p50:** how dominant the single best anchor is. Reported, no gate (follows from top-3 + eff_N).
- **Per-anchor participation:** how many jobs have this anchor in their top-3. Target: **≥ 1% of corpus per anchor** (no "dead" anchors).

At T=0.02 with our current 34 candidates these metrics are: top-3 p50 = 0.89, eff_N p50 = 2.3, all 34 participate. With the curated 10 the numbers should be at least as good (fewer anchors to split mass across).

If metrics fail: lower T slightly, expand the pool, or reduce N — but never below 7.

### Softmax temperature

**Default `T = 0.02`**. This was chosen empirically from a sweep on real job embeddings (see commit history in `src/analyse_anchor_coverage.py`). T=0.1 (the original guess) gave effective_N ≈ 24 — essentially uniform mush over all 34 anchors. T=0.01 gives effective_N ≈ 1.2 (nearly one-hot, hard Voronoi boundaries, loses smoothness). T=0.02 is the sweet spot: top-1 ≈ 63%, top-3 ≈ 89%, effective_N ≈ 2.3, still continuous.

### Smoothness preservation

v3 moved away from cluster-based prototypes specifically to get a **smooth** face function over embedding space. Changes that break smoothness are off the table:

- Lowering T below 0.02 → one-hot → hard Voronoi boundaries → broken smoothness
- Power sharpening with k > 1.5 → same effect, default **k = 1.0**
- Reducing N below 7 → transition regions become too small to see

Distinctness instead comes from:

1. **Vivid face_records.** Ethnicity/age/hair/uniform descriptors are written *extreme* ("ice-blue eyes, very fair Slavic, snub nose" not "Slavic"). The blend math is untouched; the archetypes being blended are each more vivid.
2. **Optional conditioning-space amplification** (parameter `amplify_alpha`, default `1.0` = no-op). After the 3 top anchors' sentences are encoded into CLIP/T5 conditioning, compute `base = mean(c_1, c_2, c_3)` and replace `c_i' = base + α·(c_i - base)` for α > 1. This preserves continuity (a smooth blend between two archetypes stays smooth) but pushes archetype conditionings further apart in Flux latent space. Tuned empirically on smoke batches.

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

Target ethnicity distribution at N=10: **4 Slavic (40%), 1-2 Central Asian, 1 Armenian, 1 Mediterranean, 1 East Asian, 1 Middle Eastern**. Validator allows ±1 anchor deviation per bucket.

### Blending at generation time

Per job:

1. Compute `cos(job_embedding, anchor_i)` for all 10 anchors.
2. Softmax with temperature: `w_i = exp(cos_i / T) / Σ exp(cos_i / T)`, default `T = 0.02`.
3. Optional power sharpening: `w_i ← w_i^k / Σ w_j^k`, default `k = 1.0` (no-op).
4. Take top-3 by weight. Renormalise to sum to 1.
5. For each of the 3, `compose_sentence(anchor.face_record, sus_band=job.sus_band)`.
6. Three `CLIPTextEncode` nodes → optional conditioning-space amplification (default `α = 1.0`) → two-stage `ConditioningAverage` chain.

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

**Phase 1c gate (pre-generation), computed on `data/face_anchors.json` (N=10):**

Schema:
- Exactly 10 anchors
- Every anchor has a complete `face_record` (8 axes + 5 expression bands)
- Enumerated axes (age, gender, ethnicity, complexion) use whitelisted values
- Pairwise anchor cosine < 0.85 (no redundant pair survives curation)

Ethnicity distribution at N=10:
- 4 Slavic (±1)
- 1 Armenian (±1)
- 1 Mediterranean (±1)
- 1 East Asian (±1)
- 1 Middle Eastern (±1)
- 1-2 Central Asian

Blend concentration over the full job corpus (computed with T=0.02, k=1.0):
- p50(top-3 weight sum) ≥ 0.80
- p50(effective_N) ≤ 4.0
- Every anchor is in the top-3 for ≥ 1% of the corpus (no dead anchors)

**Phase 3c gate (post-smoke):**
- ComfyUI accepts the new workflow (first face generates cleanly)
- Smoke contact sheet shows each of the 10 anchors clearly readable when it dominates
- Clean-band faces at sus ≤ 25 render without LoRA artifacts
- Transitions on the 2D PaCMAP map look *smooth* — no hard Voronoi boundaries

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
