# Face Visualization — Diffusion-Based Technical Design

**Date:** 2026-04-06
**Depends on:** `scenarios.md`

---

## The Core Requirement

The face generation function `g` must be **continuous with respect to its input**:

```
d(face_A, face_B) ≈ C · d(embedding_A, embedding_B)
```

Close job postings in embedding space → similar faces on screen. This is the entire technical problem. Everything else is engineering.

---

## Why Diffusion Models Satisfy This

Diffusion models generate images by iteratively denoising, guided at each step by a conditioning vector via cross-attention. The cross-attention operation is a smooth function of the conditioning input — nearby conditioning vectors produce similar attention weights, which guide the denoising similarly, producing similar outputs.

This continuity is not guaranteed in theory but holds robustly in practice for all major diffusion models (SDXL, FLUX, SD3). It's why prompt interpolation works: you can linearly interpolate two CLIP embeddings and get a smooth face morphing between two prompts.

The constraint is that your input must reach the model **in the right space and distribution**. Raw mxbai-embed-large vectors cannot be fed directly into SDXL cross-attention — wrong dimensions (1024 vs 768/2048), wrong distribution (text similarity space vs CLIP image-aligned space). The projection step is the entire technical challenge.

---

## The Neighbourhood-Preservation Problem

mxbai-embed-large embeds job posts in a space where nearby points share semantic content: courier jobs cluster together, IT jobs cluster, easy-money scam posts cluster. This is the property we want to preserve in face space.

Two things can break it:

1. **Non-smooth projection** — a projection that folds, rotates, or collapses nearby regions. Linear projections are smooth by definition. MLPs are smooth if well-conditioned.

2. **Distribution mismatch** — if your projected vectors land in a low-density region of the conditioning space, the model generates incoherent or degenerate outputs. Distribution matching (whitening + covariance alignment) prevents this.

---

## The Uncanny Valley as Signal

The key design decision: **the primary signal is not a specific expression, it is deviation from naturalistic.**

Legitimate postings should look boring — unremarkable, comfortable, natural. High-fraud postings drift away from the anchor into uncomfortable territory. The viewer's "something is wrong with this face" reaction is pre-cognitive and requires no calibration.

This means **denoising strength is the sus_level dial**, not the prompt:

```
sus_level  0  → denoising strength 0.05 → barely deviates → natural
sus_level 50  → denoising strength 0.25 → slight wrongness
sus_level 90  → denoising strength 0.55 → uncanny
```

At higher denoising strength the model is pulled further from the anchor toward the conditioning vector. The result doesn't resolve cleanly into a natural face — it produces characteristic artifacts: expression-geometry mismatch, slight asymmetry, eyes that don't cohere, a smile the rest of the face isn't participating in.

The factor vector determines the **flavor** of wrongness. The sus_level determines the **magnitude**.

---

## Three Architectural Options

### Option 1: Linear Projection into CLIP Space

**Mechanism:** Whiten the job embedding corpus (zero-mean, unit covariance), then apply a fixed linear map to project 1024-d → 768-d (or 2048-d for SDXL dual-encoder). Feed the result as text conditioning to a frozen SDXL/FLUX model.

**Projection construction (no training required):**
```python
from sklearn.decomposition import PCA
import numpy as np

# 1. Compute PCA basis on the corpus of job embeddings
pca = PCA(n_components=768, whiten=True)
pca.fit(job_embeddings)  # shape: [N, 1024]

# 2. Project a new embedding
def project(emb):
    return pca.transform(emb[None])[0]  # → [768]

# 3. Rescale to match CLIP embedding distribution statistics
def align_distribution(projected, clip_mean, clip_std):
    return projected * clip_std + clip_mean
```

**Neighbourhood preservation:** Linear maps preserve linear structure perfectly. PCA preserves the directions of maximum variance. Close job posts stay close after projection.

**Semantic coherence:** The projected vectors carry no face-semantic content. The model produces faces whose variation reflects the statistical structure of the job embedding space, not human-interpretable face dimensions. Similar clusters → similar faces, but the "character" of the cluster is arbitrary.

**Verdict:** Correct for the analyst (cluster structure visible). Needs the expression layer for legibility.

---

### Option 2: Fixed Anchor + img2img Conditioning

**Mechanism:** Use one neutral anchor face as a visual baseline. Run SDXL img2img with the projected job embedding as conditioning. Denoising strength scales with `sus_level`.

```
anchor_face → VAE encode → noisy latent (strength = f(sus_level)) → denoise with job_embedding conditioning → face
```

**Advantages:**
- All generated faces share a base identity (same person) — viewer calibrates to one face, not random individuals
- Denoising strength is the sus dial — continuous, intuitive
- img2img is already in ComfyUI

**Neighbourhood preservation:** The anchor constrains identity variation; the conditioning vector steers the remaining variation. Nearby embeddings → similar denoising trajectories → similar modifications to anchor.

**Verdict:** Best practical starting point. Controls visual identity drift. Naturally produces a "family of faces" from one prototype.

---

### Option 3: Two-Channel Conditioning (Hybrid — Recommended)

**Mechanism:** Split encoding across two independent conditioning channels:

| Channel | Source | Drives | Purpose |
|---|---|---|---|
| Identity | Text embedding (mxbai) | Face shape, proportions, character | Cluster membership — similar posts look alike |
| Expression | Factor vector (16-d) | Brow, eyes, mouth tension, gaze | Flavor of wrongness — compound fraud → compound expression conflict |

Implementation: two sequential img2img passes.

```
Pass 1 (identity):
  anchor_face + embedding_projection → img2img at strength 0.25 → base face

Pass 2 (expression):
  base_face + expression_prompt(factors) → img2img at strength = f(sus_level) → final face
```

Expression prompt construction from factor vector:

```python
def factors_to_expression_prompt(factors: dict, sus_level: int) -> str:
    signals = []

    # Compound fraud signals → conflicting expression (smile + cold eyes = uncanny)
    if factors.get("mentions_easy_money") and factors.get("only_dm_contact"):
        signals.append("performative warm smile, cold calculating eyes, uncanny")
    elif factors.get("mentions_easy_money") or factors.get("bot_text_patterns"):
        signals.append("forced smile, cold eyes")

    if factors.get("targets_minors"):
        signals.append("unsettling expression")
    if factors.get("no_details_at_all"):
        signals.append("blank vacant stare, empty expression")
    if factors.get("urgency_pressure"):
        signals.append("furrowed brows, tense jaw")
    if factors.get("suspicious_delivery") and factors.get("only_dm_contact"):
        signals.append("evasive gaze, tense")

    # Legitimacy signals → natural, open
    if factors.get("has_company_or_org") and factors.get("has_specific_address"):
        signals.append("open warm expression, direct gaze")
    if factors.get("grammar_quality", 0) > 0.8 and sus_level < 40:
        signals.append("composed, natural")

    return ", ".join(signals) if signals else "neutral expression"
```

The expression-geometry conflict is the key mechanism: prompting for warm smile + cold eyes simultaneously forces the model to produce something that satisfies both, resulting in a face that reads as performatively friendly — which maps exactly to the easy-money scam type.

**Neighbourhood preservation:** Identity channel preserves embedding geometry. Expression channel is driven independently by factors. The two are separable.

**Verdict:** Highest value. Serves scam hunter and analyst. Uncanny valley effect is maximized by compound conflicting expression signals.

---

## Recommended Implementation Path

### Phase 1: Embedding → Face (analyst + hunter)

1. Compute corpus embeddings — 23k jobs × mxbai-embed-large → 1024-d, stored in pgvector
2. Fit PCA projection — 1024-d → 768-d, whitened, on full corpus
3. Align distribution to CLIP statistics (sample real CLIP embeddings of neutral portrait prompts for mean/std)
4. Choose anchor face — generate one neutral portrait via ComfyUI. This is the prototype for the whole dataset.
5. Build ComfyUI workflow — img2img: anchor + projected embedding at strength 0.25
6. Generate 23k faces, cache as 256×256 PNG keyed by job_id

Expected output: a grid where spatial proximity reflects semantic similarity. No expression meaning yet — just identity variation.

**First test:** Take 20 jobs spanning 4 semantic clusters (e.g., courier jobs, IT jobs, retail, easy-money scams). Generate faces. Do the courier jobs look like each other? Do the scam posts look like each other? If yes, the projection is working. If they all look random, the projection is useless.

### Phase 2: Factors → Expression

1. Map factor vector → expression prompt per the function above
2. Add second img2img pass at strength `sus_level / 100 * 0.5 + 0.05` (range 0.05–0.55)
3. Tune on golden dataset — manually verify high-sus faces feel uncomfortable
4. First user test: show 10 faces (5 high-sus, 5 low-sus) to a naive viewer. Can they sort correctly without instructions?

### Phase 3: Discordance Visualization (analyst)

1. Compute per-job discordance: `|mean(sus_level of 10 embedding-nearest-neighbors) - job_sus_level|`
2. Surface high-discordance faces prominently in analyst view
3. These are model blind spots — highest value for golden dataset expansion

---

## Implementation Notes

**ComfyUI:** Both img2img passes map directly to ComfyUI workflows. Embedding projection runs offline in Python. The projected conditioning vector can be injected via a custom ComfyUI node or the REST API's latent input.

**Model choice:** SDXL for Phase 1 — well-supported, good face quality at 512px, fast batch generation. FLUX for later if quality matters (slower). Faces are pre-generated and cached, so throughput only matters for batch jobs.

**Resolution:** 256×256 grid thumbnails. 512×512 or 1024×1024 on click/expand.

**Determinism:** Seed fixed per job_id. Same job → same face every time. Critical for building familiarity.

**Anchor face:** Should be demographically neutral — average features, neutral expression, no distinctive characteristics. The entire visual vocabulary of the visualization is deviations from this face. Choose carefully; it cannot be changed after the cache is generated.

---

## Open Questions

1. **Does PCA projection produce face variation or noise?** The first empirical test. May require non-linear projection (MLP trained with contrastive loss) if linear fails.

2. **What denoising strength range actually produces uncanny vs random?** Expected: 0.30–0.55 is the productive range. Below 0.30 is too subtle. Above 0.55 the face loses coherence. Needs tuning.

3. **Can the expression prompt mapping be automated?** Current design hand-codes factor → prompt. Alternative: few-shot learn factor vectors → CLIP directions from the golden dataset. Requires ~50 labeled examples.

4. **Anchor face selection.** A more "average" anchor may produce more legible deviations. Consider generating the anchor as the mean of several neutral portrait generations rather than a single draw.
