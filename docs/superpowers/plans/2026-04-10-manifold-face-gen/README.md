# Manifold-Preserving Face Generation — Plans

**Spec:** [../../specs/2026-04-10-manifold-preserving-face-generation-design.md](../../specs/2026-04-10-manifold-preserving-face-generation-design.md)

This directory contains four phase plans that implement the spec sequentially. Each phase ends with a gate: a measured decision point where the work either proceeds, iterates, or stops.

## Phases

| # | File | Gate | Blocking? |
|---|---|---|---|
| 0 | [phase-0-prototype-clustering-test.md](phase-0-prototype-clustering-test.md) | LLM face descriptions cluster by source cluster (ratio > 1.4) | Recommended — failure invalidates Phase 1+ |
| 1 | [phase-1-derive-cluster-prototypes.md](phase-1-derive-cluster-prototypes.md) | All 42 clusters have non-empty prototypes | Blocks Phase 3 |
| 3 | [phase-3-generate-v8.md](phase-3-generate-v8.md) | Workflow validates, 20-face smoke run produces visually distinct faces | Blocks Phase 4 |
| 4 | [phase-4-eval-harness.md](phase-4-eval-harness.md) | v8 beats v6 and v7 on M3 (+0.10), M4, M5 (≥7/10) without regressing M6 | Final accept/reject for v8 |

Phase 2 from the spec (pre-encoding prototypes) collapses into Phase 3 at runtime — no standalone plan needed.

## Prerequisites

- Translations complete: `data/job_translations.parquet` (27,212 rows, ~24k with non-empty `body_en`)
- Layout parquet present: `/home/newub/w/vamp-interface/output/full_layout.parquet`
- Cluster centroids present: `data/cluster_centroids.pt`
- Ollama at `localhost:11434` with `gemma4:latest` loaded
- ComfyUI at `localhost:8188` with Flux.1-dev checkpoint
- Working directory for all commands: `/home/newub/w/telejobs/tools/face-pipeline`

## Execution Notes

Each phase is meant to be runnable independently by a subagent. Phases 1 → 3 → 4 are sequential; Phase 0 is a gate before Phase 1 but doesn't depend on its output. Do not skip Phase 0 unless you've already validated the LLM clustering hypothesis another way.
