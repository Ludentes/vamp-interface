---
status: live
topic: demographic-pc-pipeline
---

# Slider quality measurement procedure

A LoRA checkpoint is not a slider until it passes a defined battery. This
doc specifies that battery: the rendering grid, the measurements, the
output schema, and the pass criteria. Apply it to every candidate slider
before declaring success.

## Why a procedure

Eyeballing a 9-prompt training collage is how we got fooled across v0-v4.
"Glasses appear at +1.5" is necessary but not sufficient. The slider must:

- be **monotonic** in strength, not a switch
- work in **both directions** (negative removes, positive adds)
- preserve **identity** across the usable range
- **generalize** to prompts the trainer never saw
- have a known **artifact cliff** so users know the safe range

A casual visual check covers maybe one of these. The procedure forces all
five.

## Inputs

For each candidate slider:

- `slider_name` — e.g. `glasses_v4`, `glasses_v5`
- `checkpoint_path` — the .safetensors to evaluate (typically the last,
  but solver work may evaluate multiple)
- `intended_axis` — what this slider is supposed to control: `glasses`,
  `smile`, `eye_squint`, etc. Determines which measurement is "intended"
  vs "drift"

## Render grid

The grid is `prompts × strengths × seeds`.

**Prompts.** Two pools:

- **In-distribution (9):** the trainer's sample prompts — three demographics
  (east_asian_m, black_f, european_m) × the three strengths the trainer
  itself rendered. Rendering these at our broader strength sweep tells us
  whether the trainer's reported behaviour holds at intermediate strengths.
- **Held-out (≥6):** prompts the trainer never saw. Vary along
  demographic, lighting, framing. Examples:
  - `a portrait photograph of a south asian woman, plain grey background, studio lighting`
  - `a portrait photograph of a latino man, soft window light, cafe background`
  - `a portrait photograph of an elderly white woman, natural daylight`
  - `a portrait photograph of a young black man, harsh side lighting, plain wall`
  - `a portrait photograph of a middle eastern person, neutral light, indoor`
  - `a portrait photograph of an east asian woman, plain grey background, studio lighting`

Held-out catches memorization. If the slider works on training prompts
only, it overfit a direction that does not generalize.

**Strengths.** `s ∈ {-2.5, -1.5, -1.0, -0.5, 0, +0.5, +1.0, +1.5, +2.5}`.

The training samples used ±1.5. The procedure must extend past that to
find the artifact cliff — `|s|=2.5` is the "where does it break" probe.
`s=0` is the byte-identical baseline (verified bit-exact via
`network_mixins.py:274-279` short-circuit) and exists as the explicit
drift anchor.

**Seeds.** ≥3 seeds per `(prompt, strength)` cell. Without multi-seed,
slider effects and seed luck are inseparable. Seeds fixed across the
sweep — same seed list for every (prompt, strength) so seed × strength
trajectories are comparable.

## Measurements

For each rendered image, compute:

| Field | Source | Notes |
|---|---|---|
| `intended_metric` | per-axis (see below) | The thing the slider is supposed to move |
| `identity_cos_to_base` | InsightFace ArcFace (buffalo_l/w600k_r50) | Identity preservation; baseline = same prompt, same seed, scale=0 |
| `identity_pass_075` | derived | 1 if `identity_cos_to_base ≥ 0.75` |
| `siglip_<probe>_margin` | SigLIP-2-so400m zero-shot | One per probe in `SIGLIP_PROBES` (glasses, hair_long, hair_curly, earrings, formal_clothing, beard); always measured regardless of axis |
| `bs_<channel>` | MediaPipe FaceLandmarker | 52 ARKit blendshapes incl. `bs__neutral`, prefixed `bs_` to match `sample_index.parquet` |
| `atom_NN` | NMF basis from `models/blendshape_nmf/W_nmf_resid.npy` | Pinv projection from blendshapes onto the 20-atom basis used by `sample_index.parquet`; `atom_source = "pinv_lora"` |
| `observed_age_mivolo` | MiVOLO IMDB-cleaned | Continuous age estimate |
| `observed_age_insightface` | InsightFace genderage | Continuous age estimate |
| `observed_gender_mivolo` / `observed_gender_insightface` / `observed_gender_fairface` | three sources | M/F |
| `observed_age_bin_fairface` | FairFace dlib-aligned | 9-bin |
| `observed_ethnicity_fairface` | FairFace dlib-aligned | 7-class race |
| `lighting_contrast` | luminance std | Pixel stat |
| `lighting_gradient` | mean ‖∇L‖ | Pixel stat |

Per-axis intended metric:

- `glasses` → `siglip_glasses`
- `smile` → `mouthSmile_L + mouthSmile_R` (MediaPipe blendshape)
- `eye_squint` → `eyeSquint_L + eyeSquint_R`
- generic axes → declare the metric in the slider's spec before evaluating

## Output schema

Single parquet file per `(slider_name, checkpoint)`. The schema is the
canonical FluxSpace-corpus schema (matches `sample_index.parquet`) plus
slider-eval-specific provenance, so the LoRA-eval and FluxSpace
measurement corpora can be concatenated for downstream solvers (Path B
checkpoint mixing, Path C training-pair selection, cross-axis
composition queries). One row per `(slider_name, checkpoint,
prompt_id, scale, seed)`:

```
# canonical (matches sample_index.parquet)
source                str    # "<slider_name>/<ckpt_tag>"
rel                   str    # "<prompt_id>/seed{N}_str{V}.png"
base                  str|None  # closest BASE_META key; None for held-out demos outside the NMF fit
ethnicity             str    # intended demographic (asian|black|european|southasian|latin|middleeast)
gender                str    # m|f|u
age                   str    # young|adult|elderly
axis                  str    # the slider's intended axis
scale                 float  # LoRA strength (matches sample_index `scale`)
seed                  int
start_pct             float  # NaN for LoRA renders (no start-percent dial)
has_attn              bool   # False for LoRA renders
attn_tag              str|None
attn_row              int    # -1
atom_source           str    # "pinv_lora"
img_path              str    # relative to repo root
corpus_version        str    # "lora_eval_v1"
alpha                 float  # NaN for LoRA renders

# per-image observations (canonical naming)
identity_cos_to_base  float  # ArcFace cos vs same-prompt-same-seed scale=0
identity_pass_075     int8   # derived
siglip_<probe>_margin float  # one per SIGLIP_PROBES entry
bs_<channel>          float  # 52 MediaPipe blendshapes
atom_00..atom_NN      float  # NMF projection (N=20 currently)
observed_age_*        float  # MiVOLO + InsightFace
observed_gender_*     str    # MiVOLO + InsightFace + FairFace
observed_age_bin_fairface str
observed_ethnicity_fairface str
fairface_detected     bool
face_detected         bool   # InsightFace
lighting_contrast     float
lighting_gradient     float

# slider-eval-specific provenance (kept separately from canonical fields)
slider_name           str
checkpoint            str
prompt_id             str
prompt_pool           str    # "in_distribution" | "held_out"
intended_metric       float  # axis-specific; alias of one of the above
```

Stored at `models/sliders/<slider_name>/<ckpt_tag>/eval.parquet`.

**Fusion rule:** `pd.concat([sample_index, lora_eval], join="outer")`
gives one global corpus where any solver can query
`(axis, ethnicity, gender, age, scale)` ranges across both FluxSpace
and LoRA-render samples. This is the seed of the unified reverse
index for Path B and Path C; the canonical-schema requirement is the
load-bearing decision, not the column ordering. Adding a new slider
axis or measurement source must keep this fusion working.

## Pass criteria

A LoRA is a usable slider when **all** of:

- **Monotonicity.** Spearman correlation of `intended_metric` vs
  `strength` ≥ 0.9, computed across all (prompt, seed) cells. Both pools
  separately; held-out must also pass.
- **Bidirectional.** Negative-direction effect: `mean intended_metric at
  s=-1.5` < `mean intended_metric at s=0` < `mean intended_metric at
  s=+1.5`, with at least 0.3 SigLIP-margin (or per-axis equivalent)
  separation. If the slider only works in one direction, it's half a
  slider.
- **Identity preservation.** `arcface_cos_to_baseline ≥ 0.4` for all
  cells in the usable range (`|s| ≤ 1.5`). Below 0.4, the model lost the
  person; the slider may still draw glasses on someone, but not on the
  prompted demographic.
- **Artifact-free usable range.** Define `usable_range = max |s|` such
  that no cell shows artifact cliff (operationally: cells where
  `intended_metric` decreases as `|s|` increases past that point, or
  cells where SigLIP glasses score collapses to <0 from a positive
  region). Required: `usable_range ≥ 1.0`. We accept ±2.5 ringing but
  not ±1.0 collapse.
- **Generalization.** Held-out and in-distribution monotonicity Spearman
  must agree within 0.15. If in-dist is 0.95 but held-out is 0.6, we
  memorized the training distribution.

If any single criterion fails, the slider is **not** usable. The eval
should report which criterion failed first so the next training iteration
knows what to fix (e.g. "monotonicity passes, identity fails at +1.5"
points at η too high, not LR too high).

## One-sided axes

Some axes are inherently one-sided — adding glasses to a face is a concept,
"removing glasses" from a baseline that has none isn't, so the negative
direction has no semantic content and the slider just drifts away in
whatever direction it can find. Glasses is the canonical example. Eye
colour, scar presence, hat-wearing are likely the same shape. A slider
can still be useful as a one-way knob; the procedure should not declare
it broken for failing a bidirectional separation test it was never
designed for.

For axes in `ONE_SIDED_AXES` (defined in `measure_slider.py`):

- **Monotonicity** is computed only on `s ∈ [0, s_max]`. Held-out and
  in-distribution Spearman both reported.
- **Separation** is `mean(intended @ s_max) − mean(intended @ 0)`. The
  ≥ 0.3 threshold from the bidirectional case does not transfer; the
  axis-specific cosine scale matters. For glasses, SigLIP bipolar margin
  Δ ≈ +0.05 corresponds to "glasses now visible on most cells" — small
  in cosine units but unambiguous in fraction-with-feature.
- **Fraction with feature**: a binary proxy `siglip_<feature> > 0` per
  cell, aggregated to a per-strength fraction. This tracks the human eye
  better than the SigLIP margin. The slider's *engagement strength*
  (where fraction first crosses ~0.5) and *coverage strength* (where it
  saturates near 1.0) are the operationally interesting quantities.
- **Identity preservation** is reported only on `s ∈ [0, +1.5]`. The
  negative half is rendered and stored but excluded from pass criteria.
- **Generalization gap** is in-dist vs held-out at the engagement
  strength, on the *fraction-with-feature* metric — not on the SigLIP
  margin (too small in absolute terms to compare gap meaningfully).

Cliff/usable-range logic still applies: the strength at which ArcFace
identity drops below 0.4 caps the usable range from above.

When adding a new axis: default to bidirectional. Only add to
`ONE_SIDED_AXES` after observing that the slider's negative direction
doesn't move the intended metric coherently — i.e. after running the
full grid once and seeing the negative half is noise. Don't pre-classify;
let the data classify.

## Using the procedure to drive training decisions

The eval procedure also tells the training loop what to do next. When
a checkpoint comes off the trainer (e.g. v7 every 50 steps), score it
against the criteria above and route to one of three actions.

**Ship.** All five criteria pass on the held-out pool. Stop training,
package the checkpoint, run the full battery once more for the
artifact. Don't keep training "for safety" — additional steps at
constant LR keep accreting bundle drift (earrings, makeup, vibe shift)
that shows up as identity-preservation regressions on the next eval.

**Continue at constant LR.** Engagement is partial — `intended_metric`
is climbing checkpoint-over-checkpoint but monotonicity Spearman is
still <0.9, or only some demographics show the feature. As long as
identity preservation isn't degrading and bundle (off-axis SigLIP
probes, off-axis blendshapes) isn't growing, the local feature is
still emerging in Phase 2 and needs more gradient budget. Keep going.

**Branch to lower LR + cosine refinement.** Engagement passes (feature
present across demos, monotonic) **and** bundle is creeping up between
checkpoints — off-axis SigLIP probes drifting positive, identity
cosine sliding toward 0.4, "academic vibe" / earring stereotypes
visible in the collage. The interpretation: global PCs have already
been captured, additional constant-LR steps are now spending gradient
on bundle features rather than refining the local feature. Branch from
the chosen checkpoint, drop LR by ~5–10× (e.g. 1.25e-4 → 2e-5), cosine
to a small floor (e.g. 5e-6) over 200–400 steps, same dataset and
mask. Re-eval; the refined checkpoint should hold the engagement
metric while pulling identity and off-axis probes back toward baseline.

Why a small floor and not zero: v6 falsified cosine→0 — gradient
budget collapses faster than the schedule predicts, training stalls
before refinement completes (see
`2026-04-27-v6-lion-falsified.md`). The floor keeps the integral
∫lr·dt non-trivial across the refinement window.

**Why we don't decide on a scalar trigger.** Bundle creep is a gestalt
property of the rendered grid — the eval collage shows it before any
single metric crosses a threshold. The procedure provides the metrics
to confirm a switch decision (off-axis SigLIP up, identity cosine
down) but the trigger is "I can see it in the held-out collage". If
the collage looks clean and no metric flags, don't preemptively
refine; refinement isn't free, and a clean constant-LR checkpoint that
ships is worth more than a refined one that adds a day of compute.

**Falsification clause.** If a checkpoint fails the engagement
criteria entirely (no cell shows the intended feature on +s_max
across multiple consecutive evals), refinement is the wrong move —
that's a structural failure, not a polish problem. Refining a
checkpoint that never engaged just smooths nothing into nothing. Go
back to the loss formulation / mask / data, not the schedule.

## Decisions deliberately deferred

- **Multi-checkpoint solver.** This procedure measures *one* checkpoint
  at a time. Composition / mixing across checkpoints is a separate doc
  (Path E in the measurement-grounded plan) and only matters if no single
  checkpoint passes.
- **Cross-slider comparison.** Premature — we have one slider in
  progress. Build the comparison schema after the second slider's
  parquet exists.
- **Threshold tuning.** The 0.9 / 0.4 / 1.0 / 0.15 numbers above are
  starting points. After running the eval on v4 best + v5 best they may
  need adjustment based on what passes-but-looks-bad and what fails-but-
  looks-fine. Tune once, then freeze, before declaring any future slider
  "passed".

## Implementation order

When this is built (separate from this doc):

- A `measure_slider.py` script that takes `slider_name, checkpoint_path,
  intended_axis` and produces the parquet.
- Renders are produced through ComfyUI with the existing slider-loading
  workflow. Scripts read `eval.parquet` for analysis; the render step
  writes both the PNG and the row.
- Visual collage (`<slider_name>_eval_collage.png`) generated alongside
  the parquet for fast human inspection — held-out prompts on top half,
  in-dist on bottom, columns are strengths.
