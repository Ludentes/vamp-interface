## Demographic-PC pipeline — current belief

**Status:** live but paused at Stage 4.5 while FluxSpace work supersedes
Stage 5. Last updated 2026-04-22.

### TL;DR

- **What it is:** a staged pipeline to extract demographic-attribute
  axes (gender, age, glasses, smile, race, etc.) from Flux
  conditioning space and use them for controlled edits.
- **Where we are:** Stage 4.5 complete. Ridge direction beats
  prompt-pair contrast on manifold adherence; linearity R² 0.90 vs
  0.22; identity preserved at matched extremes where prompt-pair
  destroys it. Stage 5 was unblocked in principle.
- **Why Stage 5 is paused:** FluxSpace's own pair-averaged attention
  edits (discovered 2026-04-21) beat our ridge direction visually at
  scale 0.5–1.0. The `_topics/manifold-geometry.md` and
  `_topics/metrics-and-direction-quality.md` threads are the active
  continuation; Stage 5 (large-scale attribute extraction + downstream
  scam-hunting use) is queued behind a FluxSpace-vs-ridge combined
  approach.

### Stage map

| Stage | Purpose | Status |
|-------|---------|--------|
| 0 | Unified classifier API + prompt grid | done |
| 1 | 50-sample Flux-transfer sanity check | done, r(sus) strong |
| 2 | Full-scale generation + conditioning capture | done |
| 2b | Conditioning capture on black/glasses attribute-targeted prompts | done |
| 3 | Demographic classification on generated corpus | done |
| 4 | Direction extraction via ridge on 1785 conditionings | done |
| 4.5 | Ridge vs FluxSpace-coarse prompt-pair head-to-head | done; ridge wins |
| 5 | Large-scale axis extraction + downstream app | paused, see above |

### Load-bearing dated docs (in priority order)

1. [2026-04-20 Stage 4.5 comparison](../2026-04-20-demographic-pc-stage4_5-comparison.md)
   — the head-to-head numbers; current canonical result for this
   pipeline's Stage-4 output.
2. [2026-04-20 Stage 4.5 adversarial review](../2026-04-20-demographic-pc-stage4_5-adversarial-review.md)
   — B1/B2/B3 critiques that reshaped how we report Stage 4.5.
3. [2026-04-20 extraction plan](../2026-04-20-demographic-pc-extraction-plan.md)
   — the original pipeline plan. Superseded in parts by FluxSpace
   integration; still the canonical staging reference.
4. [2026-04-20 Stage 2-4 report](../2026-04-20-demographic-pc-stage2-4-report.md)
   — conditioning capture, classification, direction extraction
   mechanics.
5. [2026-04-20 Stage 1 report](../2026-04-20-demographic-pc-stage1-report.md)
   — sanity-check rejection of pre-Stage-1 assumptions.
6. [2026-04-20 demographic classifiers](../2026-04-20-demographic-classifiers.md)
   — classifier choices driving Stage 3.
7. [2026-04-20 Stage 1 report](../2026-04-20-demographic-pc-stage1-report.md)
   and [install log](../2026-04-20-demographic-pc-install-log.md)
   — provenance; read only if reproducing.

### Cross-thread links

- Direction-quality metrics that evaluate this pipeline's output:
  [metrics-and-direction-quality](metrics-and-direction-quality.md).
- Why FluxSpace superseded Stage 5 execution:
  [manifold-geometry](manifold-geometry.md).

### Open questions

- Can ridge (Stage 4) and FluxSpace (2026-04-21+) be combined into a
  single edit pipeline? The memory file
  `feedback_fluxspace_fallback.md` suggests yes — use FluxSpace for
  coarse on-manifold motion, ridge for the semantic projection. Not
  yet tried.
- Does Stage 5's downstream scam-hunting use case survive the
  manifold-geometry pivot? Still planned but paused; see
  `project_perception_curriculum_pivot.md` in memory.
