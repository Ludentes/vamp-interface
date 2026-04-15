# vamp-interface V1 (archived)

V1 is the shipped version of vamp-interface. It was built quickly on an early version of the idea, iterated across five generation passes, and reached a measurable result on a real corpus. V1 code and plans live here; this directory is frozen reference material, not the current work.

The V2 rebuild is in `../v2/`. V2 is a deliberate do-over using the formal framework that V1 did not have.

---

## What V1 is

- **Python 3.12 / uv / pgvector / Flux via ComfyUI**. Job postings come from the `telejobs` DB, get embedded with `qwen3-embedding:0.6b` (1024-d), optionally PCA-whitened, and fed into a ComfyUI Flux workflow that produces one photorealistic face per job.
- **Fixed seed per `job_id`** (`seed = hash(job_id) % 2^32`) so the same posting always produces the same face.
- **LoRA-based drift mechanism** (`Cursed_LoRA_Flux + Eerie_horror` at strength proportional to `sus_level`) — the face drifts from neutral into uncanny territory as fraud score rises.
- **Anchor-bridge architecture**: qwen embedding → HDBSCAN cluster → hand-curated archetype prompt → Flux generation. The archetypes were picked by hand; this is the "editorial channel" that V2's framework credits as first-class.

---

## Measured results (2026-04-07, 543-job benchmark corpus)

| Version | Backend | Drift r(anchor_dist, sus_level) | ArcFace cluster sep |
|---|---|---|---|
| v1 SDXL | SDXL, 1-axis | +0.688 | 0.381 |
| v2 SDXL | SDXL, 16-axis factors | +0.724 | 0.418 |
| v3 SDXL | SDXL, archetypes | +0.701 | 0.396 |
| **Flux v3 (anchor-bridge)** | **Flux + archetypes** | **+0.914** | **0.2179** |
| Flux v4 (continuous) | Flux + PaCMAP | +0.847 | 0.2294 |

**Flux v3 is the production baseline.** Its `r=+0.914` is the bar any V2 candidate must clear. Its ArcFace cluster separation is tight (0.2179 vs SDXL ~0.4) because hyperrealistic Flux faces share biometric geometry — this is a real property of the model, not a measurement artifact.

Measurement tool: **ArcFace IR101** via `minchul/cvlface_arcface_ir101_webface4m` (512-d, LFW 99.83%, IJB-C 97.25%). Script: `v1/src/face_distinctness.py --model arcface`. Anchor: `output/phase1/phase1_anchor.png` (seed=42 neutral portrait). Anchor distance = `sqrt(2 * (1 − cosine_sim(face, anchor)))`.

Full 2026-04-07 findings: [`../docs/research/2026-04-07-final-findings.md`](../docs/research/2026-04-07-final-findings.md).

---

## Contents

```
src/               11 Python scripts (embedding, clustering, generation, scoring, layout)
plans/
  rebuild-plan-draft.md        early draft of the V2 rebuild plan (PRE-framework, not load-bearing)
  deeper-research-queue.md     research questions queued pre-framework
  superpowers/                 numbered phase plans from the V1 iteration cycles (v6/v7/v8 manifold-face-gen, cluster prototypes, semantic anchors)
```

### src/ — scripts

| Script | What it does |
|---|---|
| `embed_jobs.py` | Embeds `jobs.raw_content` via qwen3 into `jobs.embedding` (pgvector column) |
| `smoke_test_embeddings.py` | Sanity check on embedding pipeline |
| `phase1_cluster_test.py` | HDBSCAN cluster discovery on qwen embeddings |
| `phase2_generation.py` | Flux generation from clustered archetypes |
| `generate_dataset.py` | Main generation driver — orchestrates ComfyUI calls |
| `build_test_dataset.py` | Pulls the benchmark 543-job subset |
| `build_full_layout.py` | 2D layout for visualization grid |
| `build_pacmap_layout.py` | PaCMAP-based continuous identity axis (Flux v4 experiment) |
| `face_distinctness.py` | **ArcFace IR101 measurement tool** — the canonical scoring script |
| `score_sus.py` | LLM-based re-scoring of jobs |
| `compare_scorers.py` | Compare different sus-scoring approaches |

These scripts work. They are not broken. They just predate the framework and their architectural choices are being re-evaluated in V2.

---

## Why V2 exists

V1 works, but it was built without:

- A formal definition of what "similar jobs → similar faces" actually means mathematically
- Clear criteria for what a "good" face generation function is vs. what a "working" one is
- A grounded argument for *why* Flux rather than StyleGAN, or for *why* the LoRA drift rather than h-space direction finding
- A way to compare candidate tools without the conversation dissolving into vibes

Three independent session passes re-derived the same conclusions without citing prior decisions. We were walking in circles. The fix was: stop iterating, write the framework, *then* evaluate.

The framework is at [`../v2/framework/math-framework.md`](../v2/framework/math-framework.md). V1 is preserved here because:

1. Its measurements are still the baseline for V2 comparisons.
2. Its scripts are still the working reference implementation for embedding / clustering / generation / scoring.
3. Its blind alleys (documented in [`../v2/framework/sources/rebuild-blind-alleys.md`](../v2/framework/sources/rebuild-blind-alleys.md)) are the cost of not having had the framework earlier.

Do not delete V1. Do not modify V1. When V2 produces something measurably better, V1 becomes an annotated historical artifact.
