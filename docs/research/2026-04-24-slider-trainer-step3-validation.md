---
status: live
topic: demographic-pc-pipeline
---

# Step 3 — Validation protocol

How we decide whether a trained slider is good, per axis, before
shipping it to the dictionary. Ties hyperparameter choices (step 4) to
concrete pass/fail signals.

## Philosophy

- **Benchmark the slider against the prompt-pair** it was distilled
  from. Prompt-pair `FluxSpaceEditPair` at `scale=1, start_percent=0.15`
  is the *teacher*; the slider at `scale=1` is the *student*. Student
  should match or beat teacher with lower inference cost.
- **Three axes of evaluation** per slider: direction (does it edit?),
  linearity (does magnitude track?), identity (does the face survive?).
- **Hold-out by demographic base**, not by seed — we care about
  cross-demographic generalisation more than cross-noise.
- **No leaderboard shopping.** Metrics, thresholds, and hold-outs are
  locked *before* training. Retuning recipe on test-set scores
  invalidates the protocol.

## Data splits

Corpus per axis: 6 bases × 3 seeds × 5 α = 90 images; 18 (α=0, α=1)
endpoint pairs.

### Primary split — leave-one-base-out (LOBO)

- **Train on 5 bases** (15 endpoint pairs + 30 intermediate-α samples).
- **Hold out 1 base** entirely (3 seeds × 5 α = 15 images; 3 pairs).
- 6 folds possible per axis; run **2 folds** per axis (one
  "representative" base held out + one "hardest" base from screening
  — e.g. `asian_m` for `brow_furrow_v2`).

Rationale: bases are our demographic-generalisation proxy. Holding out
a base tests whether the slider learned the *axis* vs whether it
learned a shortcut tied to a specific demographic.

### Seed and α usage

- All 3 seeds of train-bases go into training.
- All 5 α values of train-bases go into training (multi-α supervision,
  tier-1 improvement).
- Held-out base: all 5 α × 3 seeds used for **eval only**.

### Why not hold-out seed?

Considered; rejected. Seeds in our corpus share the same base prompt →
noise-level generalisation is tested automatically during inference (we
pick fresh noise). Base generalisation is the harder, more useful test.

## Metrics

### Primary (target-axis response)

For axes with a **direct ARKit blendshape target** (7 of 10 in the v1
queue — `eye_squint`, `gaze_horizontal`, `brow_lift`, `brow_furrow_v2`,
`mouth_stretch_v2`, plus `smile`, `jaw` from prior corpora):

- **P1. Target-blendshape Δ at scale=1, held-out bases**
  `Δ = blendshape_target(slider@1) − blendshape_target(slider@0)`,
  averaged over held-out-base seeds+α samples.

For axes with **no direct blendshape** (`age`, `gender`, `hair_style`,
`hair_color`, `glasses`):

- **P1'. SigLIP-2 target-concept margin Δ at scale=1**
  e.g. for `age` axis, margin `= SigLIP(image | "elderly") −
  SigLIP(image | "young")`; report `Δ = margin(s=1) − margin(s=0)`.

### Linearity (does magnitude track?)

- **P2. Scale-response monotonicity (Spearman ρ).** At held-out bases,
  render at scale ∈ {0, 0.25, 0.5, 0.75, 1.0}, compute target
  blendshape or SigLIP margin at each, report Spearman rank correlation
  with scale.
- **P3. Linear-fit R²** on the same 5 points.

### Identity preservation

- **G1. ArcFace IR101 cosine(slider@0, slider@1)** at held-out bases.
  "Does the person still look like the same person after the edit?"

### Off-target AU leakage (axes with blendshape target only)

- **G2. Non-target blendshape L¹ drift.** Sum of `|Δ|` over the 51
  non-target blendshapes, divided by 51. Spiking this means the slider
  is yanking on unrelated AUs.

### Anchor integrity

- **G3. LoRA-off == stock-Flux.** At scale=0 on a held-out base, render
  with LoRA loaded vs LoRA unloaded. Pixel-space PSNR should be ≥ 40 dB
  (effectively identical) if the tier-1.1 anchor loss is doing its job.
  In v1 without anchor loss, this may drift; report as a diagnostic.

## Pass / fail thresholds

Benchmark = the prompt-pair teacher's overnight screening numbers.
Slider must beat or closely match these without the 2× inference cost.

| Metric | Pass (ship to dictionary) | Warn | Fail (retrain) |
|---|---|---|---|
| **P1** Target Δ (held-out bases) | ≥ 0.8 × prompt-pair effect | [0.5, 0.8) × | < 0.5 × |
| **P2** Monotonicity ρ | ≥ 0.85 | [0.6, 0.85) | < 0.6 |
| **P3** Linearity R² | ≥ 0.80 | [0.5, 0.80) | < 0.5 |
| **G1** ArcFace cosine | ≥ 0.70 | [0.55, 0.70) | < 0.55 |
| **G2** Off-target L¹ drift | ≤ 0.03 | (0.03, 0.05] | > 0.05 |
| **G3** Anchor PSNR (s=0) | ≥ 40 dB | [30, 40) dB | < 30 dB (v1.1 blocker) |

**Ship rule:** all P-metrics in **pass** or better, **at least one**
guardrail in **pass**, no guardrail in **fail**. "Warn" across the
board is a "ship but note in card" verdict.

### Prompt-pair baseline reference (from overnight screening)

v1 prompt-pair `FluxSpaceEditPair` effects we're distilling:

| Axis | P-pair effect | P1 pass threshold (0.8×) |
|---|---:|---:|
| eye_squint | +0.90 | ≥ 0.72 |
| gaze_horizontal | +1.05 | ≥ 0.84 |
| brow_lift | +0.41 | ≥ 0.33 |
| brow_furrow_v2 | +0.26 | ≥ 0.21 |
| mouth_stretch_v2 | +0.044 | ≥ 0.035 |
| age (SigLIP) | +0.185 | ≥ 0.148 |
| gender (SigLIP) | +0.205 | ≥ 0.164 |
| hair_style (SigLIP) | +0.148 | ≥ 0.118 |
| hair_color (SigLIP) | +0.094 | ≥ 0.075 |

(`mouth_stretch_v2` threshold is small because the v1 prompt-pair
effect is small — this is fine; slider beating prompt-pair matters more
than the absolute number.)

## Evaluation procedure per slider

Once training finishes:

```
for held_out_base in [representative_base, hardest_base]:
    for seed in [1337, 2026, 4242]:
        for scale in [0, 0.25, 0.5, 0.75, 1.0]:
            image = flux(base_prompt, seed, slider_scale=scale)
            blendshapes = mediapipe(image)      # 52-d
            siglip_margin = siglip_probe(image, axis_concept)
            arcface_emb = arcface(image)
        # per-seed: Δ target blendshape, monotonicity ρ, R², off-target L¹, ArcFace cos(s=0, s=1)
# aggregate: mean ± std over seeds × 2 bases
```

Total cost per slider eval: 2 bases × 3 seeds × 5 scales = **30 renders**.
At ~6 s/render on 5090 (Flux-dev 30 steps, 512×512) ≈ **3 min eval**
per slider.

Baseline reference renders (stock Flux, no slider) already exist in the
overnight corpus — use those for "teacher" comparison.

## Output per trained slider — the "slider card"

Markdown stub saved to `models/flux_sliders/<axis>/card.md`:

```
# Slider: <axis>

## Config
LoRA rank=16, alpha=1, xattn, bf16
Trained 2026-04-24, 1000 steps, held-out bases: [<base1>, <base2>]

## Results (mean ± std over 2 held-out bases × 3 seeds)

P1 target-Δ:      +0.XX ± 0.XX   (pass threshold ≥ +0.XX)
P2 monotonicity:  ρ=0.XX
P3 linearity:     R²=0.XX
G1 ArcFace cos:   0.XX
G2 off-target:    L¹=0.XXX
G3 anchor PSNR:   XX.X dB

Verdict: [SHIP | SHIP-WITH-WARN | RETRAIN]

## Per-base breakdown
| base | P1 Δ | P2 ρ | G1 cos |
|---|---|---|---|
| asian_m | ... | ... | ... |
| ... | ... | ... | ... |

## Known failure modes
- <e.g. "bleeds jawOpen at scale > 0.75, mitigated by scale cap at 0.8">
```

## What this protocol explicitly does NOT test

- **Composability across axes.** Tier-3 / v2 concern. A solo slider
  passing here doesn't mean `smile + eye_squint` composes cleanly.
- **Non-face edits.** Our corpus is face-portrait. Slider behaviour on
  full-body or object images is undefined and uninteresting.
- **Out-of-corpus prompts.** Held-out *base* is still one of our 6
  demographic templates. Completely novel prompts ("photo of a cat")
  are outside scope.
- **Long-term drift / stacked edits.** Applying one slider then another
  in sequence isn't tested; that's a composability question.

## Calibration pass before training begins

Before training the first slider, compute the **prompt-pair teacher's
own metrics on the same eval split** to confirm the thresholds make
sense:
- Render `FluxSpaceEditPair` at scale=1 on held-out-base seeds.
- Score with MediaPipe + SigLIP + ArcFace.
- Check: are prompt-pair numbers what the overnight corpus said? Does
  ArcFace cos ≥ 0.7 at s=1 for the teacher? (If not, our identity
  threshold is too tight.)

This takes ~6 min (2 held-out bases × 3 seeds × 1 scale × 10 axes ≈ 60
renders) and catches threshold-calibration mistakes before they cost a
training run.

## Next

Step 4: slider hyperparameters (rank, α, target modules confirmed,
LR schedule, α-sampling distribution, timestep weighting, batch
composition, number of steps).
