---
status: live
topic: manifold-geometry
summary: B-ladder splice prompts control smile kind; scale is quantity knob with hard ceiling near collapse; start_percent=0.40 kills edit signal entirely.
---

# Smile Intensity Dial — Mona Lisa vs Joker

**Date:** 2026-04-21
**Follow-up to:** `2026-04-21-fluxspace-smile-axis.md`

## Question

Can we get fine-grained control over smile intensity — from a closed-mouth
Mona-Lisa hint to a full manic Joker grin — without collapsing the face?
Three control knobs are available:

- `B-ladder` — the splice prompt: `faint closed-mouth` → `warm` → `broad toothy`
  → `manic wide-open grin`. Four rungs.
- `scale s` — FluxSpace magnitude. Five values from 0.4 (subtle) to 2.1
  (near-collapse).
- `start_percent` — step at which the edit begins. Three values: 0.15 (default),
  0.25 (later onset), 0.40 (identity-preserving).

180 renders: 3 bases × 4 ladder × 5 scale × 3 start_pct, `mix_b=0.5`,
seed=2026, pair-averaging recipe unchanged.

## Finding 1: the B-ladder is the real intensity dial

At `(sp=0.15, s=1.0)` the four ladder rungs produce a clean progression from
closed-mouth to teeth-bared open grin, on all three bases. `asian_m` is the
cleanest demonstration: rung 1 shows a near-neutral slight smile, rung 4 shows
a wide-open laugh with visible teeth and a distinct head-tilt. The ladder
separates *kind* of smile where scale only separates *degree*.

This is the Mona Lisa vs Joker axis. Keep s ≈ 1.0 and vary B.

## Finding 2: scale past ~1.4 saturates into noise, not into a "more extreme" edit

At `s=1.4` a speckle texture begins overlaying the whole image (face still
identifiable). At `s=1.8` and `s=2.1` the image is noise-dominated across all
four ladder rungs and all three bases. Crucially, the ladder differentiation
that was crisp at `s=1.0` disappears at `s≥1.4` — high s doesn't amplify the
chosen rung's identity, it just saturates the attention envelope.

Implication: s is a *quantity* knob with a hard ceiling near the predicted
collapse edge, not a *quality* knob. The collapse predictor (max_env, T_ratio)
was right that these scales approach the manifold edge; the new observation is
that you lose prompt-directed control well before you lose the face.

## Finding 3: start_percent=0.40 kills the edit outright

Across all three bases at `sp=0.40`, the grid shows near-identical renders
regardless of ladder rung or s: the face is essentially the base prompt. The
edit budget is compressed into the final 60% of sampling, and with mix_b=0.5
pair-averaging the effective signal is too weak to land.

This contradicts my earlier prior from the glasses study, where
`sp=0.15 → 0.40` cut identity drift 58% on narrow bases while preserving the
attribute. Two possible reasons:

1. Smile is a **geometric deformation of existing geometry** (lip, cheek,
   eye-corner pixels) and must be established in the low-step structural phase.
   Glasses is an **added object** that can still be placed after structure.
2. Pair averaging at mix_b=0.5 attenuates δ by ~2× relative to the single-B
   baseline, so the edit is weaker to begin with and the timing budget matters
   more.

Practical: for smile, `start_percent ∈ {0.15, 0.25}` is usable; ≥0.40 is not.
`0.15` and `0.25` look nearly identical on all three bases — the edit has
already taken effect by step 25%.

## Finding 4: the narrow-window base (elderly_latin_m) saturates earlier

elderly_latin_m's speckle onset begins by `s≈1.4` on all rungs, matching the
pair-averaged prediction from the earlier cross-demographic run (predicted
upper edge +1.35). `asian_m` and `european_m` cleanly render through `s=1.4`
with only light texture, matching their wider-window classification. The
base-geometry cos|p95| clustering observed in the smile axis study continues
to predict where saturation begins.

## Recommendations for fine-grained control

1. **Default "dial" is the B-ladder at `(sp=0.15, s=1.0)`.** Four rungs give
   well-separated smile kinds: Mona Lisa, warm, toothy, manic.
2. **For continuous interpolation between rungs,** interpolate the splice
   text embedding (CLIP+T5 conditioning), not the scale. Two adjacent rungs
   averaged at weight α mid-step produces an intermediate kind that scale
   cannot.
3. **Do not push s past 1.2 to reach more extreme Joker** — you will get
   noise, not a stronger edit. If you need more extremity, climb the ladder
   (design rung 5 = "cackling wide-open, tongue visible") and keep s=1.0.
4. **Leave start_percent at 0.15** for smile. The 0.25 renders are
   indistinguishable from 0.15 at a much smaller identity-preservation
   benefit; 0.40 is unusable.
5. **Per-base scale ceiling** remains approximately where cos|p95| predicts:
   narrow bases (elderly_latin_m, asian_m, black_f) cap at s≈1.2–1.4 before
   noise, wider bases (european_m, southasian_f) cap at s≈1.4–1.6.

## What this means for the collapse predictor

The earlier work framed the safe window as "s where the image is still a
recognizable face". That predictor is still correct, but under-specifies
useful control: there is a **second, narrower window where the ladder rung
is still distinguishable**. We now have both bounds for smile:

- Ladder-readable window: s ∈ [~0.4, ~1.2]
- Face-recognizable window: s ∈ [~-0.7, ~1.4–2.1] depending on base

The ladder-readable window is what matters for application control. The
face-recognizable window is the absolute physical ceiling.

## Open follow-ups

- Fifth ladder rung past "manic" — can we push the splice further before
  the model refuses? ("cackling with visible tongue", "screaming laugh")
- CLIP P(smile) + AU12/AU25 scoring per tile to objectively rank the
  ladder and calibrate a (rung, s) → perceptual-intensity curve.
- Same 3-axis sweep on glasses (open-mouth-smile territory has no glasses
  analog, but thin-rim → thick-rim → aviator → protective-goggles is the
  ladder equivalent).
- Full 6-demographic replicate running on the Windows shard
  (`intensity_full/`, 504 renders, pending).

## Artefacts

- Renders: `output/demographic_pc/fluxspace_metrics/crossdemo/smile/intensity/<base>/<ladder>/sp{sp}_s{s}.png`
- Collages: `output/demographic_pc/fluxspace_metrics/crossdemo/smile/intensity/collages/<base>_sp{sp}.png`
- Sweep script: `src/demographic_pc/fluxspace_intensity_sweep.py`
- Collage builder: `src/demographic_pc/fluxspace_intensity_collage.py`
- Pipeline: `uv run python -m src.demographic_pc.fluxspace_intensity_sweep --run`
- Full-shard variant: `COMFY_URL=http://192.168.87.25:8188 COMFY_UNET='FLUX1\flux1-krea-dev_fp8_scaled.safetensors' ... --run --full`
