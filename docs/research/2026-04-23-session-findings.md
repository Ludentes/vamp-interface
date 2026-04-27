---
status: live
topic: metrics-and-direction-quality
---

# Session findings — 2026-04-23 (evening)

Six structural findings from today. These survive next week; they are not session diary.

## 1. Pair-averaging is not subtraction; it is averaging

Both halves of a `FluxSpaceEditPair` must push the target in the same direction with *adjacent*, not diametric, ambient biases on the confound being cancelled.

- Categorical opposites (Latin American vs East Asian) produce a mixture dominated by Flux's default for the target token, not a cancellation. Confirmed by `race_iter_01` v2/v3 (→ East Asian), `smile_iter_06` v_la_vs_ea (→ East Asian on every base).
- Adjacent halves cancel. `smile_iter_07` v_la_me (Latin American + Middle Eastern) on `elderly_latin_m`: Latino → Latino, no flip, identity drift 0.446 — first clean smile in the whole thread.
- Glasses axis worked with generic-short + full-base-splice because those prompt styles are adjacent on the **age** axis in Flux (short reads ~young, full portrait reads ~older). They are *not* adjacent on the race axis, which is why the glasses template translated imperfectly to race in `smile_iter_05`.

## 2. Shared tokens in both halves cancel in the δ; base anchoring is separate

Putting `{ethnicity}` and `{gender}` in both pos and neg (`smile_iter_04` v6_full_demo) did not cancel race drift — shared tokens produce zero contribution to the δ. Demographic anchoring in the *base* prompt pins the anchor at scale=0; anchoring inside both halves of the edit is cosmetic.

Implication: when designing a pair, the interesting part is the **asymmetry between the two halves**, not the words both halves share.

## 3. Graph-level chaining of edit nodes does not compose

`model.set_model_attn1_output_patch` is a single slot in ComfyUI. Chaining two `FluxSpaceEditPair` nodes overwrites the inner node's hook with the outer node's. The 2D sweep probe (`smile ∈ {0.3, 0.5}` × `race ∈ {0.0, 0.1, 0.2, 0.3, 0.5}`) showed race ≥ 0.1 completely kills the smile — the outer (race) node wins entirely.

Multi-axis composition therefore requires either:
- A `FluxSpaceEditPairMulti` node that combines N pairs inside one attention patch, or
- Sequential img2img renders (costs a VAE round-trip + 2× diffusion steps per composed edit).

## 4. Counter-edit pairs are base-specific

The winning smile pair on `elderly_latin_m` is `v_la_me` (Latin American + Middle Eastern halves). The same pair on European bases flips White → Latino_Hispanic. Pair choice depends on the base's position on the confound axis relative to Flux's default drift direction for the target.

Implication for the dictionary: rows must be keyed on `(axis, base_race, ...)` not just `axis`. The solver's existing `base` filter already enforces this at lookup time; the generation side needs to produce rows per base.

## 5. Full-resolution attention caches were being thrown away; now they are not

Caches are `(N_samples × N_steps, N_blocks=16, N_tokens=57, dim=3072)` fp16, with `delta_mix` pre-computed. MediaPipe-era workflows cached; `promptpair_iterate` and `execute_composition` were passing `measure_path=None`, silently dropping the δ for every smile, age, and race iteration 01–04.

Fixed 2026-04-23. From `smile_iter_05` onward every render writes a ~110 MB pkl alongside the PNG.

Unblocks:
- Cache-δ axis extraction (procedure milestone 8).
- Ridge-vs-FluxSpace post-mortem: `directions_k11.npz` is a `(24, 3072)` sparse projection — a different subspace from the live FluxSpace cache. We can now compute the cosine between the two subspaces on shared axes.

## 6. Residual confounds are additive; identity drift is the current ceiling

Even `v_la_me` at s=0.5 on `elderly_latin_m` still has:

- Identity drift ≈ 0.446 (non-trivial face change).
- Minor beard accumulation (noticed 2026-04-23 evening; not scored automatically).
- Small age slope (~−4.4 y/scale) that wasn't fully cancelled.

Each of these is another pair-design problem of the same shape, chippable one at a time. But the identity ceiling (~0.4–0.5 cosine drift at any scale where smile fires cleanly) is a property of the FluxSpace mechanism, not a property of any particular pair. If we need sub-0.2 identity drift, pair design is not the tool — we need ArcFace-loss-guided steering (FlowChef-style) outside the pair framework.

## Method-level lesson

Every "failure" in this session — iter_04 prompt-anchoring ablation, race_iter_01 contrastive pair, the chain-composition probe — was more informative than a direct win would have been. Each falsified a specific hypothesis cleanly and forced us toward the adjacent-not-diametric rule and the pair-averaging semantics. **The ablations were the product, not the winning iteration.**

Corollary: the pipeline should treat negative results as first-class output, logged to disk alongside winners, not as wasted compute.
