---
status: live
topic: demographic-pc-pipeline
---

# v7 step 550: glasses engagement border before identity recovery

Sample inspection of `output/ai_toolkit_runs/glasses_slider_v7/samples/`
at step 550. Grid is 3 demographics (east_asian_m, black_f, european_m)
× 3 strengths (−1.5, 0, +1.5), seed 1337, network_multiplier 1.0.

## What's at 550

| Demo | s = −1.5 | s = 0 | s = +1.5 |
|---|---|---|---|
| east_asian_m | no glasses; buzz cut, shirtless ("gym" stereotype) | no glasses; neutral, light stubble | **clear glasses** (thin metal frames); beard, dark moody studio light, "academic" bundle |
| black_f | no glasses; **shaved head** (severe identity shift on negative) | no glasses; neutral, natural hair | **clear glasses** (thick black frames); formal blazer, makeup, salon hair — strong bundle |
| european_m | no glasses; buzz cut, shirtless | no glasses; light stubble, neutral tee | **clear glasses** (tortoiseshell); full beard, turtleneck, "professor" bundle |

Glasses engage cleanly across all three demographics on +1.5. The
slider is *working* in the engagement-criterion sense (feature present,
demographics-consistent).

## Why this is the interesting border

User observation: after step 550 we lose some of the glasses in
subsequent checkpoints while identity preservation improves.

Read against the v7 hypothesis (spatial mask shifts loss-mass to eye
region): at 550 the LoRA has accumulated enough gradient to express
glasses but is also still pulling in the global bundle (formal
clothing, makeup, "academic" stylings, hair changes on female
demographic, shaved-head/shirtless on the negative direction). After
550, the constant-LR trajectory keeps spending gradient — and that
budget appears to **trade glasses-engagement back for identity
coherence** rather than refining the local feature in place.

This is a different failure mode than v4–v6:

- **v4** engaged late (step 600) with bundle that grew monotonically
  from there. No "engagement falls off" phase observed within budget.
- **v5/v6** never engaged glasses at all — gradient-budget falsified.
- **v7 step 550** engages early *and then partially backs off*. The
  spatial mask appears to be doing real work — the loss landscape now
  has a competing pull toward "match the unedited target via
  identity-preserving features," because the eye-region weighting
  amplifies the cost of every misaligned cell in that small support.

If confirmed by the eval procedure, step 550 is a **reverse cliff**:
the *peak engagement* checkpoint, with engagement degrading in both
directions — backward (insufficient gradient pre-550) and forward
(re-allocation post-550 toward identity).

## Step 750 (visual)

Same grid, same seed.

| Demo | s = −1.5 | s = 0 | s = +1.5 |
|---|---|---|---|
| east_asian_m | shirtless, buzz cut (unchanged) | neutral | **glasses LOST**; light beard, blue shirt — bundle still present, glasses gone |
| black_f | neutral, no glasses (identity recovered vs 550 — hair not shaved) | neutral | **glasses LOST**; cropped curly hair + makeup/eyeliner — bundle remains, glasses absent |
| european_m | shirt + buzz cut (less shirtless than 550) | neutral | tortoiseshell glasses still present; full beard, denim shirt |

**Asian m and black f drop glasses by step 750.** European m holds.
Negative-direction identity stops degrading (no shaved-head female).

## Step 800 (visual)

| Demo | s = −1.5 | s = 0 | s = +1.5 |
|---|---|---|---|
| east_asian_m | grey t-shirt, buzz cut — clothed, identity-coherent baseline | neutral | **still no glasses**; goatee, blue shirt |
| black_f | neutral, no glasses, hair back — clean identity | neutral | **glasses BACK** (thick tortoiseshell); large afro + hoop earrings + heavy makeup — bundle stronger than 550 |
| european_m | clothed (less shirtless) | neutral | tortoiseshell glasses, full beard, denim shirt |

The asynchrony across demographics is the load-bearing observation:
- **east_asian_m**: engagement *lost* between 550 and 750, not recovered by 800.
- **black_f**: engagement *lost* at 750, *recovered* at 800 with stronger bundle.
- **european_m**: engagement *held continuously* from 550 onward.

This is not a clean monotone "feature falls off" curve — it's
demographic-dependent reshuffling. The constant-LR trajectory is
trading off across demographics: gradient budget that was producing
glasses on east_asian_m at 550 is being spent on something else at
750/800. Possible explanations:

1. **Bundle competition.** East-asian male's "academic" bundle at 550
   was already weaker than female/european-male versions; the LoRA
   may have given up on the glasses-on-east-asian path and consolidated
   the bundle direction on the other two demographics.
2. **Identity-recovery cost.** Negative direction at 550 had severe
   identity drift (shaved head, shirtless). 750/800 negatives are
   visibly cleaner. The mechanism that pulls negatives back toward
   identity may be eroding the positive-direction glasses cells as
   collateral.
3. **Anchor proximity.** "Person without glasses" is closer to the
   identity manifold for east-asian baseline (frequent in pretraining)
   than for black female with glasses. The trainer may be falling into
   the lower-loss but glasses-missing solution preferentially on the
   demographic where it's easiest.

The eval battery on 800 will quantify this — held-out prompts will
tell us whether the glasses signal generalises beyond training prompts
at all post-550, or whether v7's effective output is "european-male
glasses slider with demographic-specific dropouts".

## Step 900 (visual) — best yet

| Demo | s = −1.5 | s = 0 | s = +1.5 |
|---|---|---|---|
| east_asian_m | shirtless, buzz cut (negative-side bundle still present) | neutral | **glasses BACK** — thin gold metal frames; goatee, light button-up shirt; cleaner than the 550 thicker-frame "academic" |
| black_f | clean neutral, hair back, no glasses | neutral | tortoiseshell glasses; large afro + hoop earrings + makeup — bundle similar to 800 |
| european_m | shirtless + shaved (negative bundle re-emerged vs 750/800) | neutral | tortoiseshell glasses; beard, denim shirt (consistent with 550–800) |

**All three demos engage glasses on +1.5.** East_asian_m re-engages
with a different (thinner, gold) frame style than the chunky black
frames at 550 — suggests the LoRA found a second glasses path on this
demographic, not a recovery of the 550 path. Black_f and european_m
hold their respective frame styles from earlier checkpoints.

**Negative-side bundle re-emerged at 900** (european_m back to
shirtless+shaved; east_asian_m stayed shirtless throughout). The
brief "negatives clean up" window at 750/800 was not a stable
trajectory.

Working hypothesis: between 800 and 900 the LoRA reorganised — east
asian male's glasses path replaced; black female's path strengthened;
european male's path stable. The reshuffling cost was paid by the
negative-direction identity coherence we briefly had at 800.

This is the strongest candidate for shipping so far. Eval battery
pending (deferred until training releases the GPU; v7 still running
toward step 2500).

## Recommended action

Apply the rubric from
`docs/research/2026-04-26-slider-quality-measurement.md` →
*Using the procedure to drive training decisions*:

1. **Score 550 with the full eval battery** (held-out prompts ×
   strength sweep × ≥3 seeds). Confirm engagement holds outside the
   training prompts; confirm identity cosine in the usable range.
2. **Score 600, 650, 700** the same way. Quantify the
   glasses-vs-identity tradeoff numerically — `siglip_glasses` margin
   per checkpoint vs `identity_cos_to_base` per checkpoint.
3. If 550 is the engagement peak and later checkpoints sacrifice it
   for identity, **branch from 550** with cosine refinement (lr=2e-5
   → 5e-6 over 200–400 steps, same dataset and mask). Goal: keep
   engagement fixed at the 550 level while letting identity recover —
   moving along the off-axis dimension only.
4. If later checkpoints simply lose engagement without identity
   recovering proportionally, that's a regression, not a tradeoff —
   ship 550 directly.

## Bundle pattern (record for cross-axis comparison)

The bundle direction at +1.5 is consistent across demographics:
formal clothing + grooming + studio-lighting shift toward "intellectual
professional" stereotype. Female demographic also picks up makeup and
hair styling. Male demographics pick up facial hair (beard/stubble).
This is the same "academic vibe" bundle observed in v4, suggesting
the loss-formulation fix (mask) reduced bundle *magnitude* but did
not change its *direction* — the bundle direction is a property of
the dataset's pos/neg pair distribution, not the loss.

If true, the data-level fix (pixel-aligned pairs, FluxSpace edits at
fixed identity) remains necessary even with a working mask — the mask
gates *how much* bundle gets baked in, not *which* bundle.

## Files

- Samples: `output/ai_toolkit_runs/glasses_slider_v7/samples/*_000000550_*.jpg`
- Checkpoint: `output/ai_toolkit_runs/glasses_slider_v7/glasses_slider_v7_000000550.safetensors`
- Config: `/home/newub/w/ai-toolkit/config/glasses_slider_v7.yaml`
