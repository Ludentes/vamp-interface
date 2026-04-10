# Embedding Space & Face Conditioning

> Context for Map UI development. Describes what the embedding space is, how it is
> clustered, what archetypes mean, and how job embeddings drive face generation.

---

## 1. Embedding Model

**Model:** `mxbai-embed-large` (Mistral-based, 1024-d)
**Served via:** Ollama at `COMFY_HOST:11434`
**Input:** Raw job posting text (`raw_content` column in telejobs DB)
**Output:** 1024-dimensional float32 vector, stored in `jobs.embedding` (pgvector)

The embedding captures semantic content: similar job types land near each other in 1024-d space. Physical proximity = semantic similarity. The geometry is meaningful — this is what the Map renders.

---

## 2. 2D Layout

**Algorithm:** PaCMAP (Pairwise Controlled Manifold Approximation)
- Chosen over UMAP for better local+global structure preservation, deterministic output
- Parameters: `n_neighbors=10`, `random_state=42`
- Output normalised to [0, 1] on both axes

**Corpus:** 23,777 jobs
**Layout file:** `output/full_layout.parquet`
**Columns:** `id`, `x`, `y`, `sus_level`, `sus_category`, `work_type`, `cluster`, `cluster_label`, `cluster_coarse`

The (x, y) coordinates are the direct input to the Map scatter plot. They preserve neighbourhood structure: jobs that are close in 2D were semantically close in 1024-d space.

---

## 3. Clustering

**Algorithm:** HDBSCAN applied to the PaCMAP 2D coordinates
**Output:** 42 coarse cluster labels (`C0`–`C40` + `noise`)

Each coarse cluster is then hand-assigned to one of 10 archetypes (see §4). Noise points (HDBSCAN outliers) are assigned to `другое_fraud` — they tend to be structurally anomalous postings.

### Cluster sizes (jobs per archetype)

| Archetype | Clusters | Jobs |
|-----------|----------|------|
| доставка | C0, C3, C9, C11, C14, C16, C33 | ~2,760 |
| уборка | C4, C7, C18, C35 | ~1,411 |
| другое_legit | C19–C40 (12 clusters) | ~12,668 |
| другое_fraud | C2, C10, C12, C13, C15, C17, C22, C24, C25, C31, noise | ~5,450 |
| офис | C5, C6 | ~404 |
| погрузка | C36, C37 | ~402 |
| торговля | C29 | ~213 |
| удалёнка | C8 | ~161 |
| склад | C34 | ~157 |
| стройка | C1 | ~151 |

`другое_legit` and `другое_fraud` are catch-all categories. `другое_legit` is the largest group — most postings don't cleanly fit a specific work type. `другое_fraud` captures high-sus postings that don't cluster into a recognisable job type.

---

## 4. Archetypes

Ten semantic categories derived from manual inspection of the clusters. Each archetype has two face prompts (clean vs scam) used for CLIP-L encoding.

### доставка — Courier / Delivery

| Variant | Prompt |
|---------|--------|
| clean | `young man, stocky build, delivery courier uniform, worn jacket, weathered skin, healthy natural expression, direct gaze, outdoor worker` |
| scam | `person, delivery jacket, hollow sunken eyes, vacant distant stare, pale waxy skin, unnaturally still expression, slightly off proportions` |

### стройка — Construction

| Variant | Prompt |
|---------|--------|
| clean | `middle-aged man, broad shoulders, construction worker, hard hat, dusty workwear, calloused hands, sun-weathered skin, grounded confident look` |
| scam | `person, construction vest, glassy unfocused eyes, slack jaw, wrong skin texture, too-smooth face, something deeply unsettling about the expression` |

### уборка — Cleaning / Janitorial

| Variant | Prompt |
|---------|--------|
| clean | `woman, practical cleaning uniform, middle-aged, tired honest eyes, hair pulled back, sensible appearance, straightforward gaze` |
| scam | `person, cleaning uniform, blank mechanical expression, eyes that don't quite focus, subtle wrongness in the face geometry, uncanny valley` |

### склад — Warehouse

| Variant | Prompt |
|---------|--------|
| clean | `heavy-set man, warehouse worker, padded vest, indoor pallor, practical no-nonsense look, slightly tired, trustworthy face` |
| scam | `person, warehouse vest, hollow cheekbones, eyes too wide or too narrow, skin with wrong undertone, subtle dread in the composition` |

### офис — Office Worker

| Variant | Prompt |
|---------|--------|
| clean | `professional, business casual clothing, smooth skin, groomed appearance, confident but approachable expression, office background` |
| scam | `person, business attire, smile that doesn't reach the eyes, too-perfect features, uncanny sheen to skin, something wrong about the face` |

### удалёнка — Remote Worker

| Variant | Prompt |
|---------|--------|
| clean | `young person, casual hoodie, pale indoor complexion, slightly tired, tech worker aesthetic, honest distracted expression` |
| scam | `person, casual clothes, vacant screen-reflected eyes, hollow face, too-uniform skin, wrong ambient lighting on face, deeply unsettling` |

### торговля — Retail / Sales

| Variant | Prompt |
|---------|--------|
| clean | `neat person, retail or sales clothing, customer-facing smile, approachable, normal healthy appearance` |
| scam | `person, neat clothes, plastic customer-service smile, eyes that slide away, wrong face geometry, uncanny professional mask` |

### погрузка — Loading / Materials Handling

| Variant | Prompt |
|---------|--------|
| clean | `strong man, loading dock worker, heavy gloves, muscular, outdoor-worn appearance, direct practical gaze` |
| scam | `person, work gloves, unfocused gaze, skin with wrong texture, hollow expression, something off about the proportions` |

### другое_legit — Other (Legitimate)

| Variant | Prompt |
|---------|--------|
| clean | `ordinary person, nondescript casual clothing, unremarkable honest appearance, average build, natural expression` |
| scam | `person, casual clothes, subtly wrong face, vacant expression, uncanny valley, something not quite right about the human quality` |

### другое_fraud — Other (Fraud-pattern)

| Variant | Prompt |
|---------|--------|
| clean | `person, plain clothing, somewhat blank expression, average features, neutral affect` |
| scam | `person, hollow eyes, wrong skin texture, mechanical expression, deeply disturbing face, uncanny valley at maximum, something fundamentally wrong` |

---

## 5. Face Generation Pipeline

### Input

- **Job embedding** (1024-d qwen vector from DB)
- **sus_level** (0–100 integer, from fraud classifier)

### Step 1: RBF archetype weights

Top-3 nearest cluster centroids are found by squared L2 distance in 1024-d space. Softmax is applied to negative distances (RBF kernel):

```
logits_k = -sq_dist(embedding, centroid_k) / temperature
weights   = softmax(logits)   # sum = 1.0
```

Default `temperature=1.5` (softer blending). Lower temperature → sharper archetype assignment.

Cluster weights are accumulated per archetype (if two top-3 clusters share an archetype, their weights are summed), then renormalized.

### Step 2: sus_factor

```python
sus_factor = (sus_level / 100.0) ** 0.8
```

Power < 1 compresses the range: legit jobs (sus ≤ 30) → sus_factor ≤ 0.25; fraud jobs (sus ≥ 80) → sus_factor ≥ 0.74.

### Step 3: T5 text (primary Flux conditioning)

The dominant archetype (highest RBF weight) drives the T5 prompt:

```python
if sus_factor >= 0.5:
    t5_text = f"photorealistic portrait, upper body, chest up, {archetype.scam}, soft studio lighting, plain background, sharp focus"
else:
    t5_text = f"photorealistic portrait, upper body, chest up, {archetype.clean}, soft studio lighting, plain background, sharp focus"
```

This is a hard switch at 0.5 (sus ≈ 60). Below that: clean appearance. Above: uncanny/scam appearance.

### Step 4: CLIP-L pooled vector (secondary Flux conditioning)

Each archetype has a precomputed CLIP-L pooled centroid (768-d, L2-normalised) for both clean and scam variants, stored in `data/clipl_centroids.pt`.

The pooled vector is an RBF-weighted blend, then sus-blended:

```python
pooled_clean = Σ_k weight_k * clipl_centroid[archetype_k, clean]
pooled_scam  = Σ_k weight_k * clipl_centroid[archetype_k, scam]
pooled       = normalize((1 - sus_factor) * pooled_clean + sus_factor * pooled_scam)
```

### Step 5: ComfyUI nodes

Two nodes injected into the Flux workflow:

```
CLIPTextEncode(t5_text)            → T5 sequence [1, seq_len, 4096]
QwenPooledReplace(above, pooled)   → swaps pooled_output with our 768-d vector
```

`QwenPooledReplace` is a custom ComfyUI node (repo: `vamp_conditioning`) that intercepts the conditioning tuple and replaces `pooled_output` without touching the T5 sequence.

### Generation

**Model:** Flux.1-dev (fp8 scaled, 20 steps, guidance=3.5)
**Mode:** txt2img (denoise=1.0), `seed = hash(job_id) % 2^32`
**LoRAs:** Cursed_LoRA_Flux (strength = sus_factor) + Eerie_horror (strength = 0.75 × sus_factor)

Same job_id always produces the same face. The only variation between jobs is conditioning.

---

## 6. Known Limitations (from conditioning analysis)

Measured via `src/analyse_conditioning.py` on 20 test jobs (10 archetypes × 2 sus bands):

| Metric | Value |
|--------|-------|
| Cross-archetype CLIP-L angle | 23.8° |
| Same-archetype CLIP-L angle | 25.0° |
| Cross-sus-band angle | 26.9° |

**The pooled vectors do not reliably discriminate archetypes.** The sus axis (clean↔scam) creates more separation than different work types. Several archetype pairs have near-identical CLIP-L centroids:
- торговля_scam ≈ погрузка_scam (cos = 1.000)
- стройка_clean = склад_clean = торговля_clean = погрузка_clean = другое_legit_clean in T5-proxy space

T5 text is the primary variation driver. Work-type visual variation in current faces comes mainly from T5, not from the CLIP-L pooled channel.

**Implication for Map UI:** Do not assume visual face similarity == embedding proximity. The sus axis is encoded more strongly than the work-type axis in the current generation setup.

---

## 7. Data Files Reference

| File | Description |
|------|-------------|
| `output/full_layout.parquet` | 23,777 jobs: (x, y) PaCMAP coords + metadata |
| `data/cluster_centroids.pt` | 42 cluster mean qwen embeddings [42, 1024] |
| `data/clipl_centroids.pt` | Archetype CLIP-L pooled centroids [10, 768] × {clean, scam} |
| `data/archetypes.json` | Archetype name → {clean, scam} prompt text |
| `data/job_clipl_embeddings.parquet` | Per-job CLIP-L pooled vectors for all 27k+ jobs |

---

## 8. Key Constants for Map UI

```python
# sus_factor threshold for clean→scam switch in T5
SUS_FACTOR_THRESHOLD = 0.5   # sus_level ≈ 60

# sus_level thresholds for display
LEGIT_THRESHOLD   = 30   # sus ≤ 30 → legitimate
FRAUD_THRESHOLD   = 80   # sus ≥ 80 → likely fraud

# RBF temperature (influences archetype blending sharpness)
DEFAULT_TEMPERATURE = 1.5

# Cluster labels in layout parquet
ARCHETYPE_ORDER = [
    "доставка", "стройка", "уборка", "склад",
    "офис", "удалёнка", "торговля", "погрузка",
    "другое_legit", "другое_fraud",
]
```
