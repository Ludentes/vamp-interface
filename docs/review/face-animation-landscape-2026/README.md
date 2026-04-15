# Face Animation Landscape 2026 — A Thesis-Style Review

**Started:** 2026-04-13
**Status:** In progress, chapters delivered sequentially
**Scope:** Unified review of the face representation, animation, and generation landscape as it stands in early 2026 — covering 2D rigged systems (Live2D), landmark/blendshape capture stacks (MediaPipe, ARKit), 3D morphable models (FLAME and relatives), neural deformation methods (LivePortrait and heirs), audio-driven talking heads, 3D Gaussian Splatting avatars, and diffusion-based parametric generation.

The review draws on ~50 research notes accumulated across two sibling projects — `vamp-interface` (data visualization via generated faces) and `portrait-to-live2d` (automated rigging of static portraits) — and extends them with targeted new research where gaps exist, especially in neural deformation and talking-head synthesis.

## Purpose

The field is fragmented. A researcher or product builder encountering it in 2026 has to piece together their worldview from non-overlapping communities: the Live2D anime-rigging scene, the academic 3DMM lineage, the VFX/games pipeline anchored on ARKit, the diffusion-native image-editing community, and the fast-moving 3DGS avatar research front. Each community knows its own tooling deeply and the others shallowly or not at all. Bridges exist but are mostly built ad hoc. Decisions are routinely made against an incomplete map.

This review's goal is to produce that map — a single document that an engineer or researcher can read end-to-end to understand what exists, how the pieces relate, what is production-ready versus research-only, what is converging versus diverging, and most importantly: **given a concrete use case, which stack to reach for and why**. The final chapters fold the survey into explicit decision trees.

## Chapter Index

| # | File | Chapter | Status |
|---|---|---|---|
| 00 | [00-introduction.md](00-introduction.md) | Introduction — the three worlds and the bridges between them | draft |
| 01 | [01-taxonomy.md](01-taxonomy.md) | A Taxonomy of Face Representations | draft |
| 02 | [02-live2d-world.md](02-live2d-world.md) | The Live2D World — 2D Rigged Animation in Production | draft |
| 03 | [03-mediapipe-arkit-world.md](03-mediapipe-arkit-world.md) | The MediaPipe/ARKit World — Landmark and Blendshape Capture as Lingua Franca | draft |
| 04 | [04-flame-3dmm-world.md](04-flame-3dmm-world.md) | The FLAME World — 3D Morphable Models and the Research Lineage | draft |
| 05 | [05-neural-deformation.md](05-neural-deformation.md) | Neural Image Deformation — LivePortrait and the Implicit-Keypoint Lineage | draft |
| 06 | [06-talking-heads.md](06-talking-heads.md) | Talking Heads — Audio-Driven Face Synthesis 2024-2026 | draft |
| 07 | [07-3dgs-avatars.md](07-3dgs-avatars.md) | 3D Gaussian Splatting Avatars — The Real-Time Photorealism Track | draft |
| 08 | [08-diffusion-parametric.md](08-diffusion-parametric.md) | Diffusion-Based Parametric Face Generation | draft |
| 09 | [09-bridges-and-conversions.md](09-bridges-and-conversions.md) | Bridges — Conversions Between Representations | draft |
| 10 | [10-decision-trees.md](10-decision-trees.md) | Decision Trees by Use Case | draft |
| 11 | [11-market-and-community.md](11-market-and-community.md) | Market, Community, and Tooling Reality | draft |
| 12 | [12-conclusions.md](12-conclusions.md) | Conclusions, Open Problems, and a 12-24 Month Horizon | draft |

**Full first-pass draft complete (2026-04-13).** All 13 chapters drafted end-to-end. Open for review, restructuring, corrections, and material updates.

## Methodology

Each chapter stands on existing research notes from both projects plus targeted new literature search where gaps exist. Claims are cited to specific source documents or external references at end-of-chapter. Where two sources disagree or one is outdated, the discrepancy is noted explicitly rather than papered over. Where the author had to make judgment calls (e.g., "which standard to prefer"), reasoning is laid out rather than asserted.

The structure is meant to be read linearly but chapters are also designed to function as standalone references.

## Conventions

- **Citations:** inline `[#]` markers, full references at chapter end, organized by paper/tool/code repo
- **Terminology:** introduced formally in Chapter 01 and used consistently afterward
- **Quality labels:** "production-ready" means public code + public weights + documented usage, "research" means paper exists but one of those is missing
- **Dates:** all dates are absolute (YYYY-MM-DD or Month YYYY), never relative

## Relationship to Prior Research

This review does not duplicate the underlying research notes — it synthesizes them. Readers interested in primary-source depth on any topic should consult:

**`vamp-interface/docs/research/`** — embedding→face generation, FLAME + diffusion, VTuber market scan, parametric real-time synthesis, 3DGS avatars
**`portrait-to-live2d/docs/research/`** — Live2D Cubism internals, moc3 format, texture atlas handling, LivePortrait deep dives, rigging automation, headless rendering, real-time portrait animation

Both project research corpora are referenced throughout this review by filename where relevant.
