# Parquet index

Quick reference for the parquet files scattered across this project. One row
per **canonical** file (per-iteration / per-checkpoint copies are mentioned but
not enumerated). Sizes / row counts are 2026-04-30 snapshots.

## Image-level catalogues (the "indices of indices")

| Path | Rows | Key | What it is |
|---|---|---|---|
| `output/reverse_index/reverse_index.parquet` | 79,116 | `image_sha256` | **Master image catalogue.** All FFHQ + Flux corpus + Solver-A grid renders, with FairFace / MiVOLO / InsightFace age/gender/race, ArcFace embedding (`arcface_fp32`), full 52-d ARKit blendshapes (`bs_*`) + `bs_detected` flag, 20 NMF atom coords (`atom_*`), 12 SigLIP probe margins (`sg_*`), **1152-d L2-normed SigLIP-2 SO400M image embedding (`siglip_img_emb_fp16`, added 2026-04-29 — 100% coverage).** Joins everything else. `source` ∈ {ffhq=70,000, flux_corpus_v3=7,772, flux_solver_a_grid_squint=1,344}. **Confirmed 2026-04-30: all 26,108 SHAs in `arc_full_latent/compact.pt` are a strict subset of the ffhq slice — joining on `image_sha256` gives the (16, 64, 64) latent ↔ 52-d blendshape pairs needed for `mediapipe_distill` directly, no MediaPipe re-extraction required.** |
| `models/blendshape_nmf/sample_index.parquet` | 7,772 | `rel`, `attn_row` | Same per-row schema as reverse_index (90 cols of bs_* + atom_*) **plus** attention-cache pointers (`has_attn`, `attn_tag`, `attn_row`) for FluxSpace measurement renders. The "joinable into reverse_index" partner for prompt-pair rendered samples. |

## Squint Path B pipeline (current axis)

| Path | Rows | What it is |
|---|---|---|
| `output/solver_c/squint_pairs.parquet` | 4.87M | Solver C all-pairs (FFHQ + grid) over θ_squint difference; raw candidate list. |
| `output/solver_c_ffhq/ffhq_squint_pairs.parquet` | 762k | FFHQ-only subset, same schema. |
| `output/solver_c_ffhq_v2/ffhq_squint_pairs.parquet` | 525k | v2 with contamination filters (closed-eye, glasses, gaze). **Live one for FFHQ pair selection.** |
| `output/solver_c_ffhq_v2/ffhq_squint_v2_index.parquet` | 18,348 | sha → row_id lookup for the v2 pairs. |
| `output/squint_path_b/pair_manifest.parquet` | 2,235 | **Live curated training pairs** (cluster_cap=1, closed-eye filtered). 20 cols incl. `kept`, `drop_reason`, `source`, `arc_cos`, `theta_*`. Drives `emit_squint_dataset.py`. |
| `output/demographic_pc/solver_a_squint_grid/scores.parquet` | 672 | Per-pair scoring of the 344-pair Solver-A Flux grid (anchor/edit blendshapes + SigLIP probes + ArcFace cos). |

## Demographic-PC measurement layer

| Path | Rows | What it is |
|---|---|---|
| `output/demographic_pc/siglip_img_features.parquet` | 5,038 | Raw 768-d SigLIP image features keyed by `img_path`. |
| `output/demographic_pc/classifier_scores.parquet` | 7,442 | Pre-reverse-index classifier outputs (FairFace / MiVOLO / InsightFace). Subset of what's now in reverse_index — kept for build-graph re-runs. |
| `output/demographic_pc/clip_probes_siglip2.parquet` | 1,536 | 12 SigLIP-2 binary probe margins per image. Subset of what's now in reverse_index. |
| `output/demographic_pc/labels.parquet` | 1,785 | Phase-1 base labels (`prompt_*` + classifier outputs) for the original FluxSpace measurement corpus. |
| `output/demographic_pc/labels_extras.parquet` | 1,785 | MediaPipe blendshape probes (smile / brow_raise / eye_open / jaw_open / glasses_prob) joined to labels. |
| `output/demographic_pc/labels_extras_2b.parquet` | 385 | Same schema as labels_extras for the 2b sub-corpus. |
| `output/demographic_pc/labels_black.parquet` | 180 | Targeted FairFace re-classification of the "black" race subset. |
| `output/demographic_pc/effect_matrix_dictionary.parquet` | 99 | Per-iteration prompt-pair effect-on-atoms slopes (the editing dictionary). 99 rows = (iters × axes), 110 cols. |
| `output/demographic_pc/effect_matrix_v0.parquet` | 48 | v0 of the same idea, smaller: 48 (axis,subtag,base) tuples × 22 cols. |
| `output/demographic_pc/overnight_siglip_probes.parquet` | 990 | 6 demographic-shift SigLIP margins for overnight drift renders. |
| `output/demographic_pc/stage4_5/eval.parquet` | 340 | Stage 4.5 ridge-vs-prompt-pair evaluation results — ArcFace cos + MiVOLO age + FairFace at multiple lam/strength settings. |
| `output/demographic_pc/stage4_5/gender/eval.parquet` | 340 | Same but for the gender axis. |
| `output/demographic_pc/promptpair_iterate/<axis>/iter_NN/results.parquet` | ~96 each | Per-image full atom / blendshape readout for one iteration of prompt-pair search. ~30 such files across {smile, age, race} × iter_01..08. |
| `output/demographic_pc/promptpair_iterate/<axis>/iter_NN/per_base.parquet` | ~12 each | Per-base summary (slope / r² / atom purity) of the same iteration. |

## Slider-quality eval (`measure_slider`)

| Path | Rows | What it is |
|---|---|---|
| `models/sliders/<name>/<ckpt_tag>/eval.parquet` | 216 typ. | Per-cell scoring for one checkpoint × one slider name. Schema mirrors sample_index + `prompt_id`, `prompt_pool`, `slider_name`, `checkpoint`, observed-axis classifier outputs. **One file per scored checkpoint.** Examples: `models/sliders/glasses_v8/glasses_slider_v8_000002450/eval.parquet`, several under `glasses_v4/`, `glasses_v7/`, `glasses_v7_1/`. squint_slider_v0/{550,1550,1800} files land here when the eval battery completes. |
| `models/sliders/glasses_v7/checkpoint_index.parquet` | 27 | One row per glasses-v7 checkpoint with axis-level summary metrics (engagement, identity drift, bundle confounds, balanced score). The "did this checkpoint get better?" view. |
| `models/sliders/glasses_v7/checkpoint_index_cells.parquet` | 243 | Same data unrolled to per-cell (step × demo × strength). Used to plot strength curves. |

## Curator / candidate selection (Path A pipeline)

| Path | Rows | What it is |
|---|---|---|
| `models/flux_sliders/training_manifest_eye_squint.parquet` | 409 | Path-A scatter-gather curator output — per-render rankable scores (`edit_effect`, `confound_smile`, `confound_gaze`, `identity_cos_to_base`). Older. |
| `models/flux_sliders/training_manifest_eye_squint_v1_*.parquet` | ~300 | Variants. v1_5 / v1_2 / v1_3 are tighter selections. |
| `models/flux_sliders/candidate_top25.parquet` | 25 | Top-25 by composite score from the curator. |
| `models/flux_sliders/candidate_balanced.parquet` | ~25 | Balanced-by-demographic version of the same. |

## App-side (vamp UI)

| Path | Rows | What it is |
|---|---|---|
| `output/full_layout.parquet` | 23,777 | Full PaCMAP layout for the scam-guessr UI: `id, x, y, sus_level, sus_category, work_type, source_name, sender_id, contact_telegram, text, created_at, cluster, cluster_label, cluster_coarse`. |
| `output/layout.parquet` | 543 | Old subsampled layout (pre-rebalance). Kept for reference. |

## Calibration / reference

| Path | Rows | What it is |
|---|---|---|
| `models/blendshape_nmf/calibration_table.parquet` | 576 | Per-(atom, axis, base) calibration: slope, r², monotonicity, usability flag. Drives "is this prompt-pair clean enough to publish?" gating. |

## Test fixtures

| Path | Rows | What it is |
|---|---|---|
| `tests/fixtures/ffhq_shard0.parquet` | 369 | Single FFHQ shard for `tests/test_ffhq_extractor.py` and the arc-distill reference fixture build. 477 MB — keep on disk. |
| `tests/fixtures/ffhq_smoke.parquet` | 6 | 6-row smoke test fixture. |

## Conventions

- Image rows are keyed by `image_sha256` everywhere except where the row pre-dates the SHA-keying refactor (`labels.parquet` uses `sample_id`).
- `arcface_fp32` is the InsightFace `buffalo_l` (IResNet50) embedding — the teacher behind `models/arc_distill/`.
- `bs_*` columns are 52-d ARKit blendshapes from MediaPipe FaceLandmarker. `atom_*` are NMF(k=20) projections of the 52-d space.
- `sg_*` margins in reverse_index, `siglip_*_margin` columns elsewhere — same SigLIP-2 binary probe margins, just renamed across rebuilds.
- When you add a parquet, **add a row to this table in the same commit.**
