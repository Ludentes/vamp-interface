# Manifold-Preserving Face Generation — Design

**Status:** Spec draft
**Date:** 2026-04-10
**Author:** Claude Sonnet 4.6 (+ user)
**Supersedes:** All prior face-generation experiments (v1–v7)

---

## 1. Goal

Generate one photorealistic face per job posting such that the face manifold meaningfully reflects the underlying job semantics:

- Semantically similar jobs → visually similar faces (cluster identity)
- Jobs at semantic boundaries → visually hybrid faces (smooth interpolation)
- Fine-grained semantic distances → correlated fine-grained FaceNet distances
- High-sus jobs → faces with uncanny-valley affect (orthogonal modifier, not identity)

The pipeline must be driven by **measured metrics**, not eyeballing. We will rebuild around an evaluation harness.

### Framing: qwen is an input, not the target

The qwen 1024-d embedding and its 42 HDBSCAN clusters are **convenient existing artefacts**, not the goal. They were produced by one embedding model on the original Russian text. A different embedding (CLIP-L on translations, sentence-transformers on gemma4 face descriptions, or another clustering scheme) could give a cleaner partition.

For this spec we proceed with qwen + 42 clusters because they already exist, the infrastructure is built, and we have no evidence yet that they're wrong. If the evaluation in §8 reveals that the 42-cluster partition is itself the bottleneck (e.g. all metrics fail because the clusters themselves are noisy), we reopen the question of what the source partition should be. This is the **"silent assumption we're willing to abandon"** — document it so we don't confuse "pipeline is broken" with "qwen clusters were the wrong target".

Implications:
- Phase 0's clustering metric uses gemma4 outputs against qwen clusters. A failure there means *either* the LLM projection is bad *or* the qwen partition is wrong. We interpret a soft fail (ratio 1.0–1.4) as "try a different source partition", not "redesign the LLM pipeline".
- Phase 4's M3 Pearson r uses qwen distances. This is noted as a temporary ground truth. A better ground truth would be FaceNet-vs-sentence-encoder-on-translated-text, which is embedding-model-agnostic. If v8 passes M4 and M5 but fails M3, we re-run M3 with the translated-text sentence-encoder distances before declaring failure.

---

## 2. What We've Learned (Compressed)

| Lesson | Implication |
|---|---|
| qwen 1024-d has real cluster structure (42 HDBSCAN clusters) and a sus gradient | The source manifold is usable as-is |
| CLIP-L pooled is the weakest Flux channel (~5–15% of signal) | Stop trying to project onto pooled |
| T5-XXL sequence is Flux's primary conditioning (80%+ of signal) | Target T5, not pooled |
| Hand-written archetype prompts collapse in CLIP-L and T5 (cos=1.000 across archetypes) | Don't hand-write prompts; derive from data |
| Archetype-per-cluster-group (10 archetypes for 42 clusters) loses intra-group variation | Prototype per cluster, not per archetype |
| txt2img (denoise=1.0) throws away anchor calibration | Return to img2img at denoise ~0.80 |
| Job text in Russian encodes poorly in CLIP-L (~74° approximation error) | Translations unlock English-trained encoders |
| Feature-list scam prompts + high-strength LoRAs produce robots, not uncanny valley | Tune sus as a separate, orthogonal experiment |

**The strategic mistake**: we operated on pooled CLIP-L because the `QwenPooledReplace` trick was mechanically tractable. We should have been blending T5 sequences — which Flux's attention layers actually read.

---

## 3. Architecture Overview

The pipeline has four components that operate at four time scales:

```
┌────────────────────────────────────────────────────────────────────────┐
│ ONE-TIME (per model version)                                           │
│                                                                        │
│  (a) Derive cluster prototypes                                        │
│      42 clusters × ~20 central translated jobs → gemma4 → 1 face      │
│      description per cluster (stored in data/cluster_prototypes.json)  │
│                                                                        │
│  (b) Pre-encode prototypes                                             │
│      For each prototype: encode through Flux's T5 + CLIP-L once        │
│      Save as data/cluster_conditioning.pt                              │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│ PER-JOB (at generation time, fast)                                     │
│                                                                        │
│  (c) Compute top-k RBF weights on qwen centroids                       │
│      cluster_weights = softmax(-distances / T)  → top 3                │
│                                                                        │
│  (d) Blend T5 + CLIP-L conditioning                                    │
│      T5_blended     = Σ w_i · T5_prototypes[i]                         │
│      pooled_blended = Σ w_i · pooled_prototypes[i]                     │
│      (via ComfyUI ConditioningAverage chain on T5 conditioning)        │
│                                                                        │
│  (e) Generate with Flux img2img                                        │
│      anchor image, denoise=0.80, seed=hash(job_id)                     │
│      LoRAs scale with sus_factor (sus axis, orthogonal to identity)    │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│ VALIDATION (once per pipeline change)                                  │
│                                                                        │
│  (f) Evaluation harness                                                │
│      Sample 50 jobs (stratified by cluster + manifold interior)        │
│      Generate faces with candidate pipeline                            │
│      Extract FaceNet embeddings                                        │
│      Compute: intra/inter cluster ratio, Pearson r vs qwen             │
│      Compare against baseline (v7 CLIP-L replace, v6 archetype blend)  │
└────────────────────────────────────────────────────────────────────────┘
```

**Key design principle**: blending happens at the T5 conditioning layer (where Flux's attention actually reads it), cluster prototypes are data-derived (not hand-written), and the pipeline is accepted or rejected based on metrics (not eyeballing).

---

## 4. Phase 0 — Hypothesis Validation (GO/NO-GO Gate)

**Question**: Does gemma4, when given a job posting and asked for a face description, produce outputs that cluster in sentence-embedding space according to the source qwen clusters?

**If no**: the LLM projection is too noisy and the whole approach fails. Fall back to explicit per-cluster prototypes injected as identical strings for all jobs in a cluster, or revisit.

**If yes**: proceed to Phase 1.

### Procedure

1. Sample 20 jobs: 5 from each of 4 diverse clusters (e.g. C0 доставка, C1 стройка, C5 офис, C29 торговля)
2. For each job, run gemma4 with prompt: *"Given this job posting, describe the face of a typical worker in one sentence, focusing on physical traits: age, build, demeanour, clothing, expression."*
3. Encode each output through a sentence encoder (`sentence-transformers/all-MiniLM-L6-v2`, or fall back to CLIP-L on English)
4. Compute 20×20 cosine similarity matrix
5. Compute intra-cluster mean and inter-cluster mean

### Pass Criteria

- Intra-cluster mean cosine **> 0.7**
- Inter-cluster mean cosine **< 0.5**
- Ratio (intra / inter) **> 1.4**

If these are met, Phase 1 is justified. If not, document failure and escalate for redesign.

### Deliverable

`src/test_prototype_clustering.py` — produces a report:

```
Cluster C0 (доставка):  intra-cluster cos = 0.XX
Cluster C1 (стройка):   intra-cluster cos = 0.XX
...
Overall intra-cluster mean: 0.XX
Overall inter-cluster mean: 0.XX
Ratio: X.XX
Verdict: PASS / FAIL
```

---

## 5. Phase 1 — Cluster Prototype Derivation

For each of the 42 coarse clusters (C0–C40 + noise), derive one face description from the LLM grounded in real translated examples from that cluster.

### Procedure

1. Load `output/full_layout.parquet` — all jobs with cluster assignments
2. Load `data/job_translations.parquet` — merged from both shards (local + remote)
3. For each cluster:
   - Select up to 20 central jobs (nearest to cluster centroid in qwen space, with translation available)
   - Format as numbered list: `"1. {text_en[:400]}\n2. {text_en[:400]}\n..."`
   - Prompt gemma4: *"Here are 20 example job postings from one cluster. Write a single one-sentence face description that captures the typical worker for this cluster — age, build, gender presentation, demeanour, clothing, expression. Ground your description in the examples. Output only the sentence."*
   - Store `{cluster_id: description}` in `data/cluster_prototypes.json`

### Output

```json
{
  "C0": "A young man in his twenties, slim build, alert restless eyes, wearing a reflective delivery vest, slightly sun-browned skin, looking hurried but engaged",
  "C1": "A middle-aged man in his forties, broad shoulders, weathered sun-tanned face, hard hat and dusty workwear, calloused hands, grounded confident expression",
  ...
  "noise": "An unremarkable person of indeterminate occupation, plain casual clothing, neutral affect, ordinary features"
}
```

Noise cluster gets a generic prototype.

### Deliverable

- `src/derive_cluster_prototypes.py` — reads corpus + translations, calls gemma4 per cluster, writes JSON
- `data/cluster_prototypes.json` — 42 entries

### Validation

Apply the same clustering metric from Phase 0 to the prototypes themselves: encode all 42 through a sentence encoder, compute 42×42 cosine matrix, verify that visually-similar clusters (all warehouse clusters, all courier clusters) cluster together. Sanity check before Phase 2.

---

## 6. Phase 2 — Pre-encode Prototypes Through Flux Encoders

For each cluster prototype, we need to know what T5 and CLIP-L produce so we can blend them at generation time.

**Issue**: T5 encoding happens inside ComfyUI, not in our Python code. We can't easily extract `[seq_len, 4096]` T5 tensors to disk.

**Resolution**: Don't pre-extract. Instead, let ComfyUI encode the 42 prototypes at generation-graph-construction time. The workflow builds 42 `CLIPTextEncode` nodes (one per prototype) and then a `ConditioningAverage` chain across the top-3 clusters for each job. Nodes for unused clusters are still built but not referenced; ComfyUI will lazy-eval only the ones connected.

**Alternative optimization**: Only encode the prototypes actually needed for a given job (top-3). Build 3 `CLIPTextEncode` nodes per workflow, re-encode per job. Slightly slower per job but cleaner workflow graph.

**Decision**: Use the alternative (3 encodes per job). Re-encoding 3 sentences through T5 costs ~100ms, negligible compared to 20-step diffusion.

**Output**: no pre-encoded file. The blending happens live inside the ComfyUI workflow at generation time.

### Deliverable

No new file in Phase 2. The prototype conditioning logic moves to `rbf_conditioning.py` as a new method (see Phase 3).

---

## 7. Phase 3 — Generation Pipeline (v8)

### RBFConditioner new method

Add to `src/rbf_conditioning.py`:

```python
def cluster_prototype_conditioning_nodes(
    self,
    embedding: list[float],
    clip_ref: list,
    start_node_id: int = 20,
) -> tuple[dict[str, Any], str]:
    """Build ComfyUI conditioning subgraph using data-derived cluster prototypes.

    Differs from direct_conditioning_nodes:
    - Uses cluster prototypes (42) instead of archetypes (10)
    - Blends T5 conditioning via ConditioningAverage (not pooled CLIP-L)
    - No sus_factor blending here — sus is applied via LoRAs + denoise only
    """
```

Logic:

1. Load `data/cluster_prototypes.json` once in `__init__`
2. Compute top-3 cluster weights from qwen embedding (reuse existing `top_k_weights`)
3. For each of top-3:
   - `CLIPTextEncode(prototype_text, clip_ref)` → produces conditioning with both T5 and pooled CLIP-L populated
4. Chain `ConditioningAverage` nodes to blend top-3 by weights (reuse existing logic from `build_v6_conditioning_nodes`)
5. Return `(nodes_dict, final_conditioning_id)`

The `ConditioningAverage` node in ComfyUI blends both T5 sequences AND pooled CLIP-L in lockstep — no need to handle them separately.

### generate_v8.py

New entry point, structured like `generate_v7.py` but:
- Calls `rbf.cluster_prototype_conditioning_nodes()` (not `direct_conditioning_nodes()`)
- **img2img with denoise=0.80** (not txt2img denoise=1.0)
- `seed = hash(job_id) % 2^32` (unchanged — per-job variation)
- LoRAs scale with `sus_factor` (unchanged, but see §9 for recalibration)
- Output: `output/dataset_faces_v8/<job_id>.png` + manifest.json

### Sus axis — orthogonal modifier

Sus conditioning does NOT go into the prototype blend. Sus is applied via:

1. **LoRAs**: `lora_cursed = sus_factor * C_cursed`, `lora_eerie = sus_factor * C_eerie` where C_* are tuned per-LoRA (see §9)
2. **Optional denoise boost**: `denoise = 0.80 + sus_factor * 0.10` — fraud faces get slightly more freedom from the anchor

Prompt-level sus descriptions are REMOVED in v8 — they conflicted with LoRAs and produced robot-style outputs. Sus is purely a rendering modifier.

### Deliverable

- `src/rbf_conditioning.py` — add `cluster_prototype_conditioning_nodes()` method (does NOT remove existing methods)
- `src/generate_v8.py` — new large-batch generation script
- `data/cluster_prototypes.json` — from Phase 1

---

## 8. Phase 4 — Evaluation Harness

The pipeline is only as good as the metric that validates it. This phase is the acceptance gate.

### Sample Set

**Deterministic ~50 jobs** chosen once, reused across all evaluation runs:

- **Cluster centres**: 3 jobs from each of the 10 largest clusters, closest to centroid (30 jobs)
- **Cluster boundaries**: 10 jobs chosen from the middle of PaCMAP distance to the nearest 2 cluster centroids (detect smoothness)
- **Sus extremes**: 10 jobs split 5 sus≤20 + 5 sus≥90 (detect uncanny modifier effect)

Save as `data/eval_sample.json` — frozen after first creation. Do not modify.

### Metrics

For each candidate pipeline, generate all 50 faces. Then:

**M1. FaceNet distance matrix** (50×50): cosine distances between FaceNet-512 embeddings of the generated faces.

**M2. qwen distance matrix** (50×50): cosine distances in the source embedding space.

**M3. Pearson r** between M1 and M2 (flattened upper-triangular). Higher = pipeline preserves fine-grained geometry.

**M4. Intra/inter cluster ratio**:
- `intra_mean` = mean FaceNet cosine within each cluster's 3 samples, averaged across clusters
- `inter_mean` = mean FaceNet cosine between samples from different clusters
- `ratio = intra_mean / inter_mean` (higher = clusters more distinct in face space)

**M5. Boundary smoothness**: for the 10 boundary jobs, check that each boundary face lies approximately between its two nearest cluster centres in face space. Specifically: compute FaceNet distance from boundary-face to each-of-its-two-nearest-cluster-centre-faces. The boundary face should be closer to both than cluster centres are to each other. Pass if >7/10 boundary jobs satisfy this.

**M6. Sus separation**: FaceNet mean distance between sus≤20 and sus≥90 samples should be significantly greater than within-band distance. Report the ratio.

### Baselines

Generate the same 50 faces with:
- **v7 baseline** (current): CLIP-L pooled replace, txt2img
- **v6 baseline** (archetype blend): text archetype ConditioningAverage, img2img

…and the new **v8 candidate**: cluster prototype T5 blend, img2img denoise 0.80.

Each baseline generates 50 faces = 150 total. At ~20s/face on Flux Dev = 50 min per baseline.

### Report

```
Metric                    v7 (pooled)  v6 (archetype)  v8 (prototype)  Δ
M3 Pearson r vs qwen      0.XX         0.XX            0.XX            +0.XX
M4 intra/inter ratio      X.XX         X.XX            X.XX            +X.XX
M5 boundary pass rate     X/10         X/10            X/10            +X
M6 sus separation         X.XX         X.XX            X.XX            +X.XX
```

### Acceptance

v8 is **accepted** if:
- M3 improves over both baselines by at least 0.10
- M4 improves over both baselines
- M5 ≥ 7/10
- M6 does not regress

Otherwise document findings and iterate (tune temperature, prototype length, denoise, etc.) OR escalate for redesign.

### Deliverable

- `src/eval_manifold_preservation.py` — runs all three pipelines on the frozen sample, computes all metrics, writes report
- `data/eval_sample.json` — 50 frozen sample jobs
- `output/eval_results/<pipeline_version>/` — generated faces + metrics JSON

---

## 9. Sus Axis Recalibration (Separate Experiment)

Sus appearance is currently broken (robot close-ups). This is **orthogonal** to the manifold work and will be addressed after v8 passes the manifold eval. Noted here as follow-up, not blocking:

- LoRA strengths currently at 1.0 (cursed) and 0.75 (eerie). Lower to 0.3 and 0.2.
- Remove feature-list scam prompts entirely (they're gone in v8 anyway, since prototype prompts don't mention sus)
- Experiment: add `"uncanny valley, subtle unease, photorealistic"` as a T5 suffix at high sus, not a separate prompt
- Test range: sus=0, 30, 50, 70, 90, 100 with same cluster → visually coherent drift?

This is a separate spec + plan after v8 is validated. **Do not implement during v8 rollout.**

---

## 10. Data Flow Diagram

```
                    ┌────────────────────────────┐
                    │  telejobs DB (jobs table)  │
                    │  id, embedding, sus_level  │
                    └────────────┬───────────────┘
                                 │
                  ┌──────────────┴──────────────┐
                  │                             │
       full_layout.parquet          job_translations.parquet
       (cluster_coarse)             (text_en)
                  │                             │
                  └──────────────┬──────────────┘
                                 │
                    ┌────────────▼────────────────┐
                    │  derive_cluster_prototypes   │
                    │  (gemma4 per cluster)        │
                    └────────────┬─────────────────┘
                                 │
                    ┌────────────▼────────────────┐
                    │  cluster_prototypes.json     │
                    │  {C0: "...", C1: "...", ...} │
                    └────────────┬─────────────────┘
                                 │
 ┌───────────────────────────────┼───────────────────────────────┐
 │                               │                               │
 │ generate_v8.py                │    eval_manifold_preservation │
 │                               │                               │
 │   for job in jobs:            │    for job in eval_sample:    │
 │     top_k_weights(qwen)       │      [same as generate_v8]    │
 │     top_prototypes = [...]    │      extract FaceNet emb      │
 │     cluster_prototype_        │    compute M1..M6             │
 │       conditioning_nodes()    │    report                     │
 │     flux_v6_workflow()        │                               │
 │     → PNG                     │                               │
 └───────────────────────────────┴───────────────────────────────┘
```

---

## 11. File Layout

### New files

| File | Purpose |
|---|---|
| `src/test_prototype_clustering.py` | Phase 0 GO/NO-GO smoke test |
| `src/derive_cluster_prototypes.py` | Phase 1 — LLM-derive per-cluster descriptions |
| `src/generate_v8.py` | Phase 3 — new large-batch generation |
| `src/eval_manifold_preservation.py` | Phase 4 — evaluation harness |
| `data/cluster_prototypes.json` | Phase 1 output — 42 face descriptions |
| `data/eval_sample.json` | Phase 4 — frozen 50 evaluation jobs |

### Modified files

| File | Change |
|---|---|
| `src/rbf_conditioning.py` | Add `cluster_prototype_conditioning_nodes()` method (do NOT remove existing) |

### Untouched

- `data/archetypes.json` — kept for v6 baseline in eval
- `data/clipl_centroids.pt` — kept for v7 baseline in eval
- `data/cluster_centroids.pt` — unchanged (still the source of RBF weights)
- `src/generate_v7.py`, `src/test_v6.py` — kept for baseline eval runs
- `src/rbf_conditioning.py` other methods — kept for baselines

---

## 12. Error Handling & Edge Cases

**Noise cluster jobs**: 2,757 jobs (11%) land in HDBSCAN noise. Treat `noise` as its own cluster with its own prototype. RBF finds `noise` as a nearest cluster when appropriate.

**Missing translation**: if a job has no `text_en` (translation failed or empty), skip it in `derive_cluster_prototypes.py` sampling. At generation time, missing translation is not an issue because we use the pre-derived prototypes, not per-job LLM.

**Cluster with <20 jobs**: use all available jobs, no padding. Small clusters exist (стройка is C1 only, ~151 jobs — plenty; edge-case clusters might have fewer).

**gemma4 thinking accidentally re-enabled**: the `think: false` option must be in the Ollama call. Verify in smoke test.

**Prototype JSON edit conflicts**: `data/cluster_prototypes.json` is regenerated wholesale. If a user edits it manually, re-running `derive_cluster_prototypes.py` overwrites. Document in script header.

**ComfyUI workflow validation**: `cluster_prototype_conditioning_nodes()` must produce a workflow that passes ComfyUI's `/prompt` validation. Test with dry-run on 1 job before full batch.

**Seed collision**: `hash(job_id) % 2^32` has effectively zero collision probability at 27k jobs. No action needed.

---

## 13. Testing Strategy

### Unit tests (new)

- `tests/test_cluster_prototype_conditioning.py`:
  - Given a mock qwen embedding, verify `cluster_prototype_conditioning_nodes()` returns a dict with 3 `CLIPTextEncode` nodes and a `ConditioningAverage` chain
  - Verify final node ID is returned
  - Verify weights sum to 1.0 after normalization
  - Verify top-3 accumulation when multiple clusters share a top spot

### Integration tests

- `src/test_prototype_clustering.py` (Phase 0) — end-to-end gate on 20 jobs
- `src/eval_manifold_preservation.py` (Phase 4) — end-to-end gate on 50 jobs × 3 pipelines

### Smoke tests

Before committing any pipeline change:
```bash
uv run src/generate_v8.py --dry-run --count 3
uv run src/generate_v8.py --count 3
```

---

## 14. Sequencing & Gates

```
Phase 0: test_prototype_clustering ──► GATE 1 (ratio > 1.4?)
                                           │ PASS
                                           ▼
Phase 1: derive_cluster_prototypes ──► GATE 2 (all 42 clusters have prototypes?)
                                           │ PASS
                                           ▼
Phase 3: generate_v8 (smoke, 20 faces) ──► GATE 3 (no crashes, faces look different?)
                                           │ PASS
                                           ▼
Phase 4: eval_manifold_preservation ──► GATE 4 (v8 beats both baselines by §8 criteria?)
                                           │ PASS
                                           ▼
Full batch: generate_v8 (2000+ faces, background run)
```

Each gate is a checkpoint where we decide: proceed, iterate on parameters, or abandon and redesign. Gates 1 and 4 are the most critical — failure there means the approach is wrong, not just miscalibrated.

---

## 15. What This Design Does NOT Do

- **Does not fix the sus axis appearance** — see §9, separate experiment
- **Does not generate faces for all 27k jobs immediately** — first 50 for eval, then scale to 2k, then if everything works, 27k in background
- **Does not retrain any model** — pure conditioning + workflow-level change
- **Does not add new infrastructure** — uses existing ComfyUI, Flux, FaceNet, gemma4, Ollama
- **Does not touch the Map UI** — the map consumes PNGs; pipeline is transparent to it
- **Does not remove existing pipelines** — v6 and v7 remain as baselines for eval
- **Does not modify anchor** — `output/phase1/phase1_anchor.png` stays as the calibration face

---

## 16. Open Questions / Risks

### Risks

- **gemma4 prototypes collapse**: the 42 prototypes may all look similar because gemma4 writes "a person of average build, neutral expression" for every cluster. Phase 0 catches this. Mitigation: increase temperature, sharper prompt, use gpt-oss instead.

- **T5 blending doesn't actually blend**: ConditioningAverage on T5 sequences may produce weird concatenated outputs instead of smooth hybrids. Mitigation: Phase 3 smoke test on a few boundary jobs. If broken, fall back to "dominant cluster wins" with weights only affecting pooled.

- **Anchor dominance at denoise 0.80**: faces may still look too similar because anchor structure persists. Mitigation: try denoise 0.85, 0.90, measure M4 across values.

- **FaceNet eval doesn't reflect human perception**: FaceNet metrics may pass while humans still see identical faces. Mitigation: human spot-check after each eval run (user reviews the 50-face grid).

- **2k generation cost**: At 20s/face Flux Dev, 2000 faces = 11 hours. Acceptable for background run, but any bugs discovered mid-run waste time. Mitigation: 50-face smoke run first, 2k only after all gates pass.

### Open Questions

- **Should we use PaCMAP distance or qwen cosine for RBF weights?** qwen cosine is what the pipeline has always used. PaCMAP may be closer to human-perceived similarity. Noted as a follow-up experiment after v8 validates.

- **Does ConditioningAverage on T5 with different seq_lens work?** ComfyUI may auto-pad or throw. Need smoke test.

- **Sentence encoder choice for Phase 0**: `all-MiniLM-L6-v2` is 22MB and English-only. Will work on LLM output which is English. Sticking with this unless it proves too crude.

### Assumptions we're making

- gemma4 is capable of writing visual face descriptions grounded in job context (not generic filler)
- Within-cluster qwen variation maps somewhat linearly to within-cluster face variation (else RBF blending is meaningless)
- Flux's attention will treat blended T5 sequences as coherent prompts, not as concatenated nonsense
- 3 prototypes (top-k=3) is enough to interpolate smoothly across cluster boundaries
- **The qwen 42-cluster partition is "good enough" as a source** — we're willing to proceed without validating this, but will revisit if evaluation metrics suggest the clusters themselves are the bottleneck rather than the pipeline

---

## 17. Success Criteria (Final Form)

At the end of this spec's implementation, we have:

1. 42 data-derived cluster prototype descriptions in `data/cluster_prototypes.json`
2. A `generate_v8.py` pipeline that produces faces using T5 blending
3. A frozen `eval_sample.json` of 50 jobs and a `eval_manifold_preservation.py` harness
4. Measured metrics showing v8 beats v6 and v7 on M3, M4, M5
5. A human spot-check confirming faces look visually distinct across clusters and smooth at boundaries
6. A decision: scale v8 to 2k, then 27k — or iterate on parameters — or redesign

The spec is successful if we walk away with either **"v8 works, scale it up"** or **"v8 fails for reason X, here's what to try next"**. Both are valuable outcomes; the current state ("no idea what works") is not.
