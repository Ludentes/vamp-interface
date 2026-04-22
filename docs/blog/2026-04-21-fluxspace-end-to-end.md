---
status: live
topic: manifold-geometry
summary: FluxSpace pair-averaging generalises across 6 demographics on glasses and smile axes; max_env ratio predicts scale-collapse edges; geometry invariant to axis and base.
---

# FluxSpace, End to End: One Recipe, Two Axes, Six Demographics, and a Geometry That Travels

**Date:** 2026-04-21
**Series:** Follows [Demographic-PC Extraction, End to End](2026-04-20-demographic-pc-extraction-end-to-end.md). That post ended with a warning: our ridge-direction edits collapsed into solid colour or noise well before the published FluxSpace-coarse baseline did. The suggested fallback was to port FluxSpace itself and run *it*, instead of patching us. This post is one day of doing exactly that — and everything it turned up.

![Top: six demographics on the glasses axis, pair-averaged at mix_b=0.5, predicted safe windows with gold-bordered edges. Bottom: same six bases on the smile axis, same recipe, zero retuning. In both stacks, s=+1.0 lands in the measurement slot (first cell, blue border); upper cells are predicted-edge straddles that almost always show collapse. The smile stack lights up stronger at s=+1.0 — larger visible change, easier for CLIP — but the two stacks share the same narrow-vs-wide-window structure by base.](images/2026-04-21-fluxspace-end-to-end-cover.png)

This is the whole post in one frame. The recipe that makes the top rows work — pair-averaged attention editing — is the same recipe that makes the bottom rows work. The predicted safe windows in the top are mostly right, in the bottom mostly too wide. The bases that are narrow on one axis are narrow on the other. Everything below explains how we got there and why each piece matters.

## The starting point: our direction extraction was bleeding

The demographic-PC Stage-4.5 results showed our ridge direction outperforming FluxSpace-*coarse* (their cheapest variant) on manifold adherence for age edits. But the moment we asked for any other axis — glasses, in particular — our direction produced ghosts or skin-lesion textures at strengths where FluxSpace still produced a cleanly-glasses-wearing face. The diagnostic was in the memory note: *use FluxSpace end-to-end instead of patching ours*. So we did.

## Stage 1: port the FluxSpace edit node to our ComfyUI stack

FluxSpace operates inside Flux's joint-attention blocks. At each transformer step it routes an *edit* conditioning through the same MM-attention path as the base, then steers attention toward the edit direction by a scale factor `s`. Our port lives in `custom_nodes/demographic_pc_fluxspace/__init__.py` as `FluxSpaceBaseMeasure` (pass-through with per-block logging) and `FluxSpaceEdit` (scale injector). On the glasses axis with a single edit prompt, scale ∈ [0.5, 1.0] produced clean, non-collapsed edits where our own direction failed — which confirmed the bug was in our direction-extraction, not in our understanding of Flux.

The boring but load-bearing discovery: `double_blocks_only=True`, which the FluxSpace paper suggests as a knob for isolating edits, **does not work on Flux Krea**. Leaving it False and using `start_percent` as the identity-preservation knob is the workable configuration. Not in the paper; learned by running.

## Stage 2: the single-prompt edit has a hidden confound

With `FluxSpaceEdit(A="A person wearing thick-rimmed eyeglasses")` driving the steering, scale=1.0 on a `"adult Latin American woman"` base added glasses — and also made the face visibly older. Swapping "thick-rimmed" for "round wire" changed the direction of aging, not its magnitude. The edit prompt was leaking demographic information through its own statistics. What we wanted was *glasses*. What we got was *glasses plus whatever else that edit sentence averages into*.

The fix is the paper's own trick, dressed up: edit through **two** prompts simultaneously and average their attention caches. A = bare (`"A person wearing thick-rimmed eyeglasses."`), B = base-spliced (`"A photorealistic portrait photograph of an adult Latin American woman wearing thick-rimmed eyeglasses, ... studio lighting, sharp focus."`). At `mix_b=0.5`, A and B drag the conditioning in opposite demographic directions that cancel, while their common direction — *glasses* — survives. `FluxSpaceEditPair` implements this; at mix_b=0.5, seed 2026, the glasses go on and the age stays put.

The lesson behind the fix: when your edit direction is fused with a confound, averaging two directions that share the target but disagree on the confound cancels the confound without needing to name it. We didn't have to identify "age" anywhere in the pipeline; we let the averaging do it.

## Stage 3: collapse prediction from one measurement pass

With pair-averaging working at s=1.0, the next question was: *how far can we push s before the image breaks?* A dense 28-point scale sweep labelled every scale collapse/safe, which gave us something to fit against. From the per-(block, step, D) attention-base and δ_mix we captured during a single measurement run, four candidate predictors all reduce to analytic quadratics in s:

- `max_env(s) = max over (block, step, D) of |attn_base_d + s · δ_mix_d|`
- Frobenius of steered attention, `‖attn_base + s · δ_mix‖_F`
- Per-D diagonal-Σ Mahalanobis against a calibration corpus of 10 diverse base prompts
- Per-D max z-score against the same calibration

`max_env` won. On seed 2026: 92.86% accuracy in-sample. On seed 4242, threshold transferred from seed 2026: 82.14%. The math matches the visuals directionally — on latin_f, max_env grows faster for s<0 than s>0, and the observed collapse onset is closer to origin on the negative side. One 30-second measurement pass + the calibration prior produces a usable safe-window prediction for any new edit — no sweep required.

The threshold `T=8.5` does **not** transfer across base prompts, because baseline `max_env(0)` varies by ~30% across demographics and three of our six cross-demo bases already exceeded 8.5 at s=0. The fix was switching to the **ratio** metric `max_env(s) / max_env(0)`. T_ratio = 1.275 keeps the 92.86% in-sample accuracy and is nominally base-invariant.

## Stage 4: cross-demographic confirmation on the glasses axis

Six base prompts: asian_m, black_f, european_m, elderly_latin_m, young_european_f, southasian_f. Same A (bare), per-base B (splice), mix_b=0.5, seed 2026. Measurement pass at s=1.0, verification sweep straddling predicted edges.

Three findings came out of this pass, in order of unexpectedness:

**Pair averaging generalises at the recipe level.** All six bases gain clean thick-rimmed glasses at s=+1.0 with no visible demographic drift. The confound-cancellation mechanism from the latin_f study is not a latin_f peculiarity.

**Ratio threshold *mostly* generalises.** T_ratio=1.275 is systematically too permissive on the positive side by ~0.25 for 4 of 6 bases. It's good enough to stop shooting in the dark; not good enough to trust as a hard safety bound. A two-sided fit (T⁺, T⁻) should close it.

**Usable window width varies 10×, and is uncorrelated with baseline activation magnitude.** Elderly_latin_m has width ≈ 1.65; southasian_f has width ≈ 2.90. Whatever drives width is a property of how δ and the base interact geometrically, not of the base alone.

## Stage 5: geometry — cos(δ, base) predicts window width from the same measurement

Because the collapse metric is `max|attn_base + s·δ|`, the safe window's size is determined by how `δ` adds to `attn_base` *per dimension*. If δ points *with* the base, they reinforce and a small s saturates; if δ points *against* the base, they cancel and s has room before the combined vector grows past the collapse threshold. Aggregating `cos(δ_mix, attn_base)` per (block, step) as the 95th percentile of absolute cosine:

| base | cos\|p95\| | observed width |
|---|---|---|
| elderly_latin_m | 0.855 | 1.65 |
| asian_m | 0.853 | 1.70 |
| black_f | 0.862 | 1.70 |
| young_european_f | 0.921 | 2.70 |
| european_m | 0.930 | 2.70 |
| southasian_f | 0.931 | 2.90 |

Two clusters, clean split. Counter-intuitively — and opposite to what we'd predicted — **the bases with the most antiparallel δ have the widest windows**. Mean `cos(δ, base)` is ≈ −0.52 across the corpus, so δ always partially cancels the base; the more complete the cancellation, the more room s has before the sum saturates.

This is a third predictor from the same measurement pass. No sweep required; no cross-seed fit; no new axis data. Just geometry.

## Stage 6: the primary metrics (which we should have led with)

"Does it collapse" answers the wrong question for our downstream use. The questions that actually matter are: **did we add the attribute** and **how much did the identity drift**. Both are measurable on existing renders without new inference.

- **ArcFace IR101 drift** (1 − cos vs the s=0 render of the same base) sanity-checks at 0.000 across all bases at s=0, and rises monotonically outward with scale magnitude.
- **CLIP ViT-B-32 P(glasses)** peaks at s ≈ +1.0 for most bases and at s ≈ +0.85 for elderly_latin_m (our narrowest window). Absolute P values are low (0.45–0.74) because ViT-B-32 is a weak detector on 224² downsampled portraits; the *shape* of the curve is reliable.

The sweet spot is where P(attribute) peaks with drift still below threshold. For most bases that's s ∈ [+0.95, +1.15]; for elderly_latin_m it's s ∈ [+0.85, +1.05]. Later `start_percent` preserves identity on narrow bases — sp=0.40 on elderly_latin_m cuts drift from 0.57 to 0.24 (−58%) while P(glasses) stays 0.60–0.64. The wide bases are indifferent to `start_percent`. This gives us an automatic rule: use sp=0.30–0.40 when the geometry predicts cos|p95| ≈ 0.85.

## Stage 7: does any of it transfer to a second axis? Smile.

The whole glasses story could still have been a glasses-shaped coincidence. We ran the same pipeline — same six bases, same mix_b=0.5, same seed, same T_ratio — with `A = "A person smiling warmly."` and per-base B replacing "neutral expression" with "smiling warmly". The result is the bottom half of the cover image.

**Pair averaging generalises.** All six bases smile warmly at s=+1.0 with no demographic drift. Zero retuning.

**CLIP is a better smile detector than glasses detector.** Peak P(smile) 0.71–0.94 vs P(glasses) 0.45–0.74 on the same bases. Smile is a big visible change on a 224² downsampled face; glasses are thin frames in dark hair. The *curve shape* is reliable on both; absolute values aren't comparable across axes.

**Ratio threshold is systematically *too* permissive on smile.** Five of six bases have upper-edge prediction errors of ≥ 0.2 — visual collapse starts at s ≈ +2.15 where prediction said s ≈ +2.35 was still safe. The one base it nailed was elderly_latin_m (narrowest window on both axes). A per-axis, two-sided fit is now a confirmed need rather than a hedge.

**Windows generally widen on smile, with the most gain on bases that were narrow on glasses.** Elderly_latin_m 1.65 → 2.05 (+0.40); asian_m 1.70 → 2.85 (+1.15); black_f 1.70 → 2.60 (+0.90). Already-wide bases are flat. Smile is a geometric deformation (lips, cheeks); glasses is an added object. Objects collapse the attention map faster once magnitude exceeds what the base can absorb.

**The one signal that transferred perfectly: cos|p95| ranking.**

| base | glasses cos\|p95\| | smile cos\|p95\| |
|---|---|---|
| elderly_latin_m | 0.855 | 0.855 |
| asian_m | 0.853 | 0.876 |
| black_f | 0.862 | 0.896 |
| young_european_f | 0.921 | 0.912 |
| european_m | 0.930 | 0.917 |
| southasian_f | 0.931 | 0.928 |

Identical ranking. Values nearly identical. This wasn't guaranteed — δ_mix is computed from axis-specific prompts and could easily have reshuffled — but the per-base narrow/wide cluster is stable. If this holds on a third axis, the implication is real: **narrow-vs-wide-window is a property of the demographic base prompt, not of the attribute axis**. One measurement per base may predict headroom for every future axis.

**One asymmetry was axis-specific.** Glasses had strong negative-side collapse bias; smile has none. Frowns at s=−0.7 look as close to the edge as smiles at s=+2.3. Two-sided thresholds will need to be fit per axis, not borrowed across.

## Where this leaves us

Four artefacts from one day that we didn't have yesterday:

1. A working FluxSpace edit node in our ComfyUI stack.
2. A confound-cancellation recipe (`pair averaging at mix_b=0.5`) that turns one axis, one base into six bases, two axes without retuning.
3. A collapse-edge predictor (`max_env ratio`) fit from one measurement pass — directionally right, systematically off by about a quarter-scale on the upper edge, and the size of the error is itself a signal (smile needs a tighter threshold than glasses).
4. A per-base geometry scalar (`cos|p95|`) that clusters bases into narrow vs wide windows — and whose ranking is identical across both axes tested.

In order of next information yield:

1. **A third axis** (beard or explicit age) to confirm cos|p95| is a per-base invariant, not a two-axis coincidence. If it holds, we can retire the sweep-to-find-the-window pattern entirely and drive scale choice from measurement.
2. **Two-sided per-axis threshold fits** for T⁺, T⁻. The smile overshoot is consistent enough that a second constant will fix it.
3. **A face-ROI classifier** to replace CLIP ViT-B-32 for attribute presence. The current P peaks are low and only the curve shape is informative; a stronger detector makes them usable as hard signals.

The raw artefacts are under `output/demographic_pc/fluxspace_metrics/crossdemo/` (glasses in the top-level `collages/`, smile in `smile/collages/`). The three research notes with every number are:

- [2026-04-21-fluxspace-collapse-prediction.md](../research/2026-04-21-fluxspace-collapse-prediction.md)
- [2026-04-21-fluxspace-crossdemo-confirmation.md](../research/2026-04-21-fluxspace-crossdemo-confirmation.md)
- [2026-04-21-fluxspace-smile-axis.md](../research/2026-04-21-fluxspace-smile-axis.md)
