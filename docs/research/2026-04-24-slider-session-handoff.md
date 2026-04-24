---
status: superseded
topic: demographic-pc-pipeline
superseded_by: 2026-04-24-slider-session-evening-handoff
---

# Slider trainer session handoff — compaction-safe snapshot

Full context for the next session to pick up where this one left off.
Produced just before `/compact`.

## What was built this session (2026-04-24)

**Training stack (v1.0 — works, validated on 5 axes):**

- `src/demographic_pc/train_flux_image_slider.py` — PEFT LoRA trainer
  on Flux-Krea-dev (bf16). Rank 16, α=16 (literal slider scale),
  xattn target list (regex) including single-block `proj_out`,
  logit-normal timestep bias (μ=0.5), multi-α supervision, Flux sigma
  shift, grad checkpointing, 8-bit Adam, resume flag, flow-matching
  velocity target `ε − z_after`. ~55 min for 1000 steps on 5090.
- `src/demographic_pc/slider_collage.py` — inference + collage via
  diffusers `FluxPipeline` with `enable_model_cpu_offload()`. Pass
  local transformer in via `transformer=` arg to skip HF re-download.
- `src/demographic_pc/preview_corpus_collage.py` — pure-PIL grid of
  rendered corpus PNGs (no model load).

**Five v1.0 sliders trained** at `models/flux_sliders/<axis>_v1_0/`:
eye_squint, gaze_horizontal, brow_lift, mouth_stretch_v2, brow_furrow_v2.

**Problem found via collage + Stage 0 diagnostic:**

- Sliders reproduced the teacher corpus's identity drift.
- ArcFace scoring of corpus: **46–100% of α=1.0 training samples
  have cos < 0.75 vs their α=0 anchor** (eye_squint 94%, brow_lift
  100%, gaze_horizontal 47%).
- Corpus is the problem, not the trainer.

**Diagnostic + index infrastructure:**

- `src/demographic_pc/score_corpus_identity.py` — ArcFace
  cos vs α=0 anchor per (base, seed, α).
- `src/demographic_pc/extend_sample_index_v2.py` — extended the
  canonical `models/blendshape_nmf/sample_index.parquet` to **6577
  rows × 91 cols** with new sources (overnight axes, reprompt v2,
  crossdemo_v2). Added derived columns: `identity_cos_to_base`,
  `identity_pass_075`, `alpha`, `corpus_version`.
- Updated `output/demographic_pc/classifier_scores.parquet` with
  ArcFace embeddings for new rows. Backups at
  `*.parquet.bak.20260424_*`.

## Strategy ladder for corpus regeneration

Three strategies to fix identity-drifted corpus (per
`docs/research/2026-04-24-slider-corpus-identity-drift.md`):

- **A** — teacher-scale compression: `mix_b ∈ {0, 0.10, 0.20, 0.30, 0.40}`
  instead of {0..1.0}. Stays in the identity-safe half of the teacher
  curve. Implemented as `expand_corpus_v2.py`.
- **B** — filter: drop existing samples with cos < 0.75. Loses α
  coverage.
- **C** — pair-averaging: two prompt pairs at 50/50 via
  `FluxSpaceEditPairMulti`, per 2026-04-21 glasses axis memory.
  Hypothesis: opposing identity drifts cancel, let us use wider
  mix_b. Implemented as `expand_corpus_v3_multipair.py`.

## Pilot results on eye_squint (partial v3 in progress)

**v2 (Strategy A, mix_b max 0.40)** — 250 PNGs, complete:
- Identity pass rate α=0.40: **92%** (vs v1 α=1.00: 6%)
- **But axis effect barely visible at α=0.40** — too subtle in preview collage
- Output: `output/demographic_pc/fluxspace_metrics/crossdemo_v2/eye_squint/eye_squint_inphase/`
- Preview: `models/flux_sliders/collages/preview_eye_squint_v2.png`

**v3 (Strategy C, pair-averaging, mix_b 0.00 → 1.00)** — RENDERING NOW:
- Task ID: `bsuscf18h` (background), ~400 renders, ETA ~100 min total
- Output: `output/demographic_pc/fluxspace_metrics/crossdemo_v3/eye_squint/eye_squint_inphase/`
- α sweep: {0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.00}
- Pair 1 (aggressive): `wide-open alert eyes` → `strongly squinted eyes, eyelids half-closed`
- Pair 2 (warm): `eyes fully open in a neutral expression` → `gently narrowed eyes crinkled at the corners as if smiling warmly`
- per-slot scale = 0.5 (sum ≈ 1.0)
- Hypothesis to test: does averaging keep identity ≥0.75 at higher mix_b?

## Immediate next moves when v3 render completes

1. **Rescore identity:** rerun `extend_sample_index_v2.py` (will detect
   new v3 source, drop stale blendshapes.json first if it exists).
2. **Preview collage** at seed=2026 across all 10 bases and all 8 α values:
   ```
   uv run python src/demographic_pc/preview_corpus_collage.py \\
     --root output/demographic_pc/fluxspace_metrics/crossdemo_v3 \\
     --axis eye_squint --seed 2026 \\
     --alphas 0.0 0.15 0.30 0.45 0.60 0.75 0.90 1.00 \\
     --out models/flux_sliders/collages/preview_eye_squint_v3.png
   ```
3. **Curate training corpus** — select α cutoff from v3 identity-pass
   curve. Expected shape: cos=1.0 at α=0, decreasing as α grows. Pick
   the largest α where pass rate stays ≥ 80%. Train slider on v3
   samples with α up to that cutoff.
4. **Retrain eye_squint v1.1** on v3 curated corpus. Same trainer,
   new data. Output dir `models/flux_sliders/eye_squint_v1_1/`.
5. **Compare sliders** side-by-side:
   - v1.0 (trained on v1 corpus with drift)
   - v2.0 if we pivoted (trained on v2 narrow corpus)
   - v1.1 (trained on v3 curated corpus)

## Research trail (read in order for context)

1. `docs/research/2026-04-24-slider-trainer-step1-inventory.md`
2. `docs/research/2026-04-24-slider-trainer-step2-lora-family.md`
3. `docs/research/2026-04-24-slider-trainer-step2-5-flow-explainer.md`
4. `docs/research/2026-04-24-slider-trainer-step2-6-improvements.md`
5. `docs/research/2026-04-24-slider-trainer-step3-validation.md`
6. `docs/research/2026-04-24-slider-trainer-step4-hyperparams.md`
7. `docs/research/2026-04-24-slider-trainer-phased-plan.md`
8. `docs/research/2026-04-24-slider-corpus-identity-drift.md` ← diagnostic doc
9. This file — session handoff

## Key numerical facts to remember

- 5090 has 32 GB; Flux-Krea-dev bf16 23.8 GB → training fits only with
  ComfyUI killed, text encoders pre-encoded and dropped.
- Trainer step time: 0.5 s/step on 5090, 1000 steps = 8.3 min.
- LoRA: **37.4M trainable / 11.9B total** (0.313%). With `proj_out`
  included on single blocks.
- Sample index path: `models/blendshape_nmf/sample_index.parquet`.
  Columns: source, base, seed, alpha, identity_cos_to_base,
  identity_pass_075, 52 bs_*, 20 atom_*, + others.
- Identity threshold: τ = 0.75 (ArcFace cos).

## Operational invariants

- **ComfyUI + training are VRAM-incompatible on 32 GB.** Rendering
  corpora requires ComfyUI up; training requires ComfyUI down.
- **Rendering scripts use flat prefixes** (`expand_v2_<axis>_<base>_...`)
  not `/`-separated — avoids ComfyUI subfolder download bug in
  `ComfyClient.download()`.
- **Classifier_scores embeddings must be float32** for pyarrow
  parquet write. Fixed in `extend_sample_index_v2.py`.

## Files the next session may need

```
src/demographic_pc/train_flux_image_slider.py     trainer
src/demographic_pc/slider_collage.py              inference/collage
src/demographic_pc/preview_corpus_collage.py      corpus preview (PIL)
src/demographic_pc/score_corpus_identity.py       Stage 0 diagnostic
src/demographic_pc/extend_sample_index_v2.py      index extender
src/demographic_pc/expand_corpus_v2.py            Strategy A renderer
src/demographic_pc/expand_corpus_v3_multipair.py  Strategy C renderer
models/flux_sliders/<axis>_v1_0/                  5 v1.0 LoRAs
models/blendshape_nmf/sample_index.parquet        canonical index
output/demographic_pc/fluxspace_metrics/crossdemo_v2/  Strategy A corpus
output/demographic_pc/fluxspace_metrics/crossdemo_v3/  Strategy C corpus (in progress)
```
