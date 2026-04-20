# Demographic-PC — Stage 1 Sanity Report

**Date:** 2026-04-20
**Related:** `2026-04-20-demographic-pc-extraction-plan.md` (plan), `2026-04-20-demographic-pc-install-log.md` (Stage 0)
**Data:** `output/demographic_pc/stage1/sanity_check_50.json` + 50 Flux Krea portraits at 768×1024
**Generation:** Flux Krea v3 img2img from a fresh neutral 768×1024 anchor, denoise=0.9, guidance=3.5, euler/simple, 20 steps, ~5.5 min for 50 samples.

## Headline

**Gate passes. Stage 2 is go.** Classifiers transfer cleanly to Flux v3 output with predictable, structured failure modes — none severe enough to invalidate regression-based direction extraction on 1800 samples.

## Numbers

| Axis | FairFace | MiVOLO | InsightFace |
|---|---|---|---|
| Face detection | 50/50 | n/a (whole image) | 50/50 |
| Gender (binary prompts, N=34) | **100%** | 94% | 77% |
| Age within ±12y of prompt midpoint | **86%** | 82% | 62% |
| Ethnicity vs prompt (7-way) | 62% overall | — | — |

Inter-classifier gender agreement: **42/50 all three unanimous**, 6 MiVOLO=FairFace disagree with InsightFace, 2 MiVOLO=InsightFace disagree with FairFace.

## General Picture

**Face detection is solved.** 50/50 for both FairFace (dlib CNN-detect + 5-landmark align) and InsightFace (SCRFD). Stage 2 carries no "NaN row" risk from missing detections.

**Gender is the strongest signal.** FairFace nails every binary-prompt face. MiVOLO misses twice, both on elderly-woman prompts where grey hair + androgynous bone structure fooled it. InsightFace's 77% reflects its 96² internal input resolution — it makes confident-but-wrong sex calls on children and a few edge cases (e.g. a Black child prompt comes out female twice).

**Age is noisy but structured, not random.** MiVOLO and FairFace cluster tightly around the prompt-age midpoint (82–86%). InsightFace runs **systematically 10–15 years older** on adult/elderly prompts (e.g. "adult" prompts read as 48–58, "elderly" as 74–84 vs FairFace 50–59 / 60–69). A consistent bias direction is actually what we want — regression learns around it. Random noise would fail; structured bias gets absorbed by the learned weight matrix.

**FairFace ethnicity fails where it was pre-declared to fail.** The "dark" regions of the 7-class map are perfect: Black 6/6, South Asian 6/6, East Asian 5/6. The "light-skin" middle is mush: White 8/14, Hispanic/Latino 2/6, Middle Eastern 1/6 — the three swap among each other as expected from FairFace's known confusion profile. This was flagged in the plan as concentration risk; it's now operational, not hypothetical.

**Non-binary prompts produce binary predictions.** The classifiers have no third class, and split prompts that are visually androgynous ~arbitrarily into M/F. Kept in the grid because:
1. label-distribution coverage for the regression is more important than ground-truth agreement per sample,
2. dropping them would bump us from 105 to 70 cells, making seeds-per-cell awkward,
3. even "arbitrary" M/F assignments on ambiguous faces carry signal about how Flux responds to "non-binary" in the prompt.

## What This Means For Stage 2

Regression on 1800 samples will likely find:

- **Strong age direction** (MiVOLO + FairFace both informative) — probably 3–5 dims of useful signal.
- **Strong gender direction** (all three classifiers) — 1–2 dims.
- **Strong dark/light ethnicity axis** (Black / East Asian / South Asian clearly separated from rest) — 2–3 dims.
- **Weak Western-light-skin sub-structure** (White / Hispanic / Middle Eastern poorly separated) — the race direction quality will be driven by the dark-cluster contrasts, which is fine for the curriculum but means race is the noisiest axis.

The combined demographic subspace should land in the planned **10–15 dims**, with age and gender contributing clean signal and race contributing a strong "non-Western / Western" axis plus weaker within-Western structure.

## Qualitative Observations From The Sample Sheet

- Anchor portrait (neutral prompt at seed 42) is an adult-looking man with medium skin tone and short dark hair. At denoise=0.9, this anchor's geometric scaffolding survives barely enough to stabilize composition — ethnicity, age, and gender are overwhelmingly driven by the prompt, which is the desired regime for Stage 2.
- Middle-aged-White-woman prompts come out consistently with grey hair and strong age markers; this is the sample family where MiVOLO/FairFace are most in agreement on "older than prompt midpoint 50."
- South Asian and Middle Eastern prompts overlap in FairFace's output — both frequently map to "Indian" or "Middle Eastern" with confidence, so the 6/6 South Asian score is half "correct" and half "FairFace confuses MENA and South Asian similarly in both directions." This is a known feature of res34-7.
- InsightFace's systematic age-up is most pronounced for "young adult" prompts — a 22-year-old prompt reads as 30–45 on InsightFace. This explains the 62% age score: it isn't that InsightFace is "wrong," it's that its definition of "young adult" sits higher on the age axis than the other two classifiers.

## Decisions For Stage 2

1. **Keep non-binary prompts** in the full grid (see rationale above).
2. **Conditioning capture:** do generation first (3.5h unattended), then a second pass through prompts.py with a stand-alone CLIP-L + T5XXL encoder — decouples two failure surfaces.
3. **Resolution:** 768×1024, as planned.
4. **Anchor:** reuse the one generated in Stage 1 (`output/demographic_pc/stage1/anchor_768x1024.png`). No need to re-roll.
5. **Denoise:** stay at 0.9. Stage 1 confirms this is enough for demographics to drive the image; lower risks anchor-leakage into the regression signal.

## Known Risks Still Carried

- FairFace race direction is weakest; White/Hispanic/Middle-Eastern sub-axis will contribute noise, not signal. The orthogonalization in Stage 5 should still work because the strong dark/light axis dominates variance.
- InsightFace age bias may couple with gender in regression — its older-skew is slightly stronger on male-prompt faces than female. Worth checking per-head residual plots in Stage 4.
- Classifier failure modes are now known *descriptively*; what we haven't checked is whether they're *systematically correlated with any prompt dim other than the obvious one*. That's Stage 4's job (residual analysis after regression).
