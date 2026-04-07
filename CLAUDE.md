# vamp-interface — Project Instructions

## What This Is

A data visualization experiment. Each job posting from the telejobs corpus is rendered as a photorealistic AI-generated face. The face encodes fraud signals via the uncanny valley: high-fraud postings produce faces that feel wrong without the viewer being able to say why.

This is a standalone project. It consumes data from telejobs and draws conceptual inspiration from portrait-to-live2d, but owns its own stack entirely.

## Core Hypothesis

The face generation function must be **continuous**: close points in embedding space → similar faces. `d(face_A, face_B) ≈ C · d(embedding_A, embedding_B)`.

The uncanny valley is the primary signal mechanism, not feature-by-feature reading. High `sus_level` → higher denoising strength → face drifts from neutral anchor into uncomfortable territory.

## Stack

- **Python 3.12**, uv for package management
- **SDXL via ComfyUI** — img2img, two-pass generation (identity pass + expression pass)
- **mxbai-embed-large** on server COMFY_HOST:11434 — 1024-d job post embeddings
- **pgvector** in telejobs DB — embedding storage
- **PCA projection** — 1024-d → CLIP conditioning space, whitened

## Data Sources

- telejobs DB — `jobs` table: `raw_content`, `sus_level`, `sus_factors` (16-d factor vector), structured metadata
- ComfyUI REST API — same server infrastructure as portrait-to-live2d

## Key Design Decisions

- **Two-channel encoding**: identity from text embedding (cluster membership), expression from factor vector (verdict)
- **Fixed anchor face**: one neutral portrait; all generated faces are perturbations of it. Same face → calibration anchor.
- **Denoising strength as the sus dial**: 0.05 (legit) → 0.55 (critical). Not the prompt.
- **Seed fixed per job_id**: same job always produces same face
- **Pre-generate and cache**: 256×256 PNG per job, generated offline, served statically

## Three Users

See `docs/design/scenarios.md`. In priority order for design decisions:
1. **Scam hunter** — needs cluster membership + uncanny verdict fast
2. **Analyst** — needs embedding geometry preserved, discordance visible
3. **Student** — needs intuitive safe/unsafe, no calibration

## Project Layout

```
docs/
  design/          — scenarios, technical design
  research/        — literature review, data landscape
comfyui/
  workflows/       — ComfyUI workflow JSON files
src/               — code (projection, generation pipeline, UI)
```

## Conventions

- uv for Python
- Conventional commits: docs:, feat:, fix:, chore:
- Research docs → docs/research/YYYY-MM-DD-<topic>.md
- Commit and push after each logical unit of work
