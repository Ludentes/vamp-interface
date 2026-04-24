---
status: live
topic: demographic-pc-pipeline
---

# Stage 0 — corpus identity-drift diagnostic

The v1.0 sliders showed identity drift in the collage. Stage 0 scores
the existing FluxSpaceEditPair training corpus with ArcFace IR101
(buffalo_l `w600k_r50`) to determine whether the slider inherited the
drift from the corpus, or introduced it during training.

## Method

For each (axis, base, seed, α) in the crossdemo corpus, compute
`cos(emb_α, emb_α=0)` where `emb_*` are normalised ArcFace embeddings.
Threshold τ=0.75 chosen as the step-3 protocol G1 pass line
(identity-preserved).

Script: `src/demographic_pc/score_corpus_identity.py`.

## Results

### eye_squint

| α | n | mean cos | median | min | below τ=0.75 |
|---|---:|---:|---:|---:|---:|
| 0.25 | 17 | 0.943 | 0.954 | 0.842 | **0 / 17 (0%)** |
| 0.50 | 17 | 0.799 | 0.826 | 0.581 | 3 / 17 (18%) |
| 0.75 | 17 | 0.673 | 0.692 | 0.427 | 12 / 17 (71%) |
| 1.00 | 17 | 0.607 | 0.647 | **0.290** | **16 / 17 (94%)** |

Overall 31/68 (46%) below τ. elderly_latin_m worst (5/8 below). One sample
has cos=0.29 — effectively a different person.

### brow_lift

| α | n | mean cos | below τ |
|---|---:|---:|---:|
| 0.25 | 17 | 0.902 | 1/17 |
| 0.50 | 17 | 0.764 | 5/17 |
| 0.75 | 17 | 0.617 | 14/17 |
| 1.00 | 17 | **0.554** | **17/17** |

**100% of α=1.00 samples fail identity threshold.**

### gaze_horizontal

| α | n | mean cos | below τ |
|---|---:|---:|---:|
| 0.25 | 17 | 0.938 | 1/17 |
| 0.50 | 17 | 0.863 | 3/17 |
| 0.75 | 17 | 0.776 | 6/17 |
| 1.00 | 17 | **0.730** | 8/17 (47%) |

Noticeably better than the other two — gaze sideways doesn't carry as
much face-restructuring signal. Confirms our collage impression.

## Interpretation

Conclusively: **the corpus is the problem**, not the trainer.

- Slider at scale=1.0 learned to produce "a squinting/brow-lifting
  *person*", because 94–100% of its α=1 training samples were
  identity-swapped versions. It faithfully reproduces this.
- Slider at scale=0.25 would probably look fine if we had trained at
  lower scales — the α=0.25 corpus is clean (0% drift for eye_squint).
- The failure is **axis-dependent**: gaze_horizontal preserves much
  better because the edit doesn't restructure face geometry.

## Per-base pattern

Elderly bases drift more across all three axes tested:
- elderly_latin_m: consistently the worst (mean 0.61–0.70)
- young_european_f: also bad (0.68–0.81)
- asian_m, black_f: best preserved (0.78–0.95)

Hypothesis: the `FluxSpaceEditPair` edit pulls all bases toward a
canonical face-prior that matches "adult East Asian / Black" more than
"elderly Latin / young European". This is a Flux-prior entanglement
flagged in the solver doc.

## Why compose_iterative with demographic counters won't fully fix it

The drift isn't uniformly demographic. Compose_iterative cancels
age/gender/hair drift via counter-pairs, but the residual
"edit-adjacent identity restructuring" doesn't have a clean counter-
prompt. Candidate approaches:

1. **Pair-averaging** (`FluxSpaceEditPairMulti`) — two different
   prompt pairs targeting the same AU from different angles. Identity
   drifts in different directions for each; averaging cancels them.
   Per `project_fluxspace_pair_averaging.md` memory, this worked on
   the glasses axis.

2. **Teacher-scale reduction** — render at `SCALE=0.5` instead of 1.0.
   α=0.25 (which is clean at SCALE=1.0) corresponds to mix_b=0.5 at
   SCALE=0.5 — same effective perturbation, but ensures the student
   slider's full scale range maps to the "clean" end of the teacher.

3. **Corpus filtering** — drop (base, seed, α) samples where cos<τ.
   Keeps the existing render, retrains on what's left. Loses coverage
   at α=0.75, α=1.00, and disproportionately loses elderly bases.

4. **Compose_iterative with identity threshold** — probably partial
   fix; counter-pairs can cancel some drift but not all.

## Recommended sequence

**First test (Option 3, 30 min):** drop α=0.75 and α=1.00 entirely
from training; retrain eye_squint on α ∈ {0.25, 0.5} only. Slider at
scale=1.0 now reproduces α=0.5 which is ~80% identity-preserved. If
the collage shows clean sliders with proportionally smaller edits, the
diagnostic is confirmed actionable and we proceed to Option 1 or 2 for
full-coverage fix.

**Second (Option 1, ~2h):** pair-averaging regeneration. Design a
second prompt pair per axis, use `FluxSpaceEditPairMulti` at 50/50,
re-render α=0.75 and α=1.00 samples. Re-measure. Retrain.

**Third (Option 2, if needed):** reduce `SCALE` to 0.5 and use
`mix_b ∈ {0.5, 0.75, 1.0, 1.25}` for training, mapping to slider
α ∈ {0.25, 0.5, 0.75, 1.0}. Samples the lower half of teacher curve
which is where identity preserves.

## Artefacts

- `output/demographic_pc/stage0_identity_eye_squint.csv`
- `output/demographic_pc/stage0_identity_brow_lift.csv` (not yet written — rerun with `--output`)
- `output/demographic_pc/stage0_identity_gaze_horizontal.csv`
