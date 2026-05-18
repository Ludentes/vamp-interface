---
status: live
topic: lam-chibi-recipe
---

# Splat → chibi: failure synthesis and the way forward

A consolidation of the whole chibi thread — what was tried, what each attempt
*proved*, the single root cause underneath all of them, and a recommended path
that also satisfies the live blendshape-driving constraint.

## The failure ladder

**Attempt 1 — `ChibiField` on the raw splat cloud** (concluded dead 2026-05-15).
Optimized a vertex-space landmark loss; moved only Gaussian `xyz`;
opacity / scale / rotation / SH frozen (`means2d` was `requires_grad=False`, so
the rasterizer was forward-only). A chibi-magnitude (~1.4×) stretch rescattered
the cloud — gaps, blur, iris-through-lid leak — because appearance is glued to
vertices at fixed density. **Proved:** a baked Gaussian cloud cannot survive
chibi-scale deformation under an `xyz`-only edit.

**Attempt 2 — splat → FLAME-mesh bake, v1** (gate failed 2026-05-16).
Geometry survived the bake cleanly. Per-vertex splat **color did not** — blotchy
raw-meat — because a splat's color is an alpha-blend / SH compositing color, not
an albedo. **Proved:** appearance cannot be read straight off splat parameters;
it must be re-derived from rendered pixels (multi-view render → UV atlas).

**Attempt 3 — v3 mesh + UV texture + staged geometry pipeline** (2026-05-17/18).
`Φ = HeadBlock ∘ ProportionRemap ∘ ReliefFlatten ∘ FeaturePrimitives`, eight
TDD tasks, 26 tests passing. The block head is coherent and fold-safe — stages
1–3 succeed. **Stage 4 (FeaturePrimitives) reads uncanny:** every geometric
metric (`eye_aspect`, `bridge_height`) passes, yet enlarging the eye aperture
stretches a small realistic dark eye texture into a hollow socket. **Proved:**
re-priming geometry under a frozen realistic texture cannot produce a chibi
face.

## The single root cause

Across all three attempts, **appearance was never an optimization variable — it
was always frozen and inherited.** The splat path froze SH/opacity; the v1 bake
copied splat color verbatim; v3 freezes the baked realistic texture. Every
method moved geometry while appearance stayed photoreal. But chibi-ness is half
appearance: a chibi eye is a flat painted oval with a highlight, not "the
realistic eye, enlarged." Stage 4 is that gap made visible. The
differentiable-loss-fishing note (`2026-05-18-chibi-differentiable-loss-fishing.md`)
names the same thing — "no image-space supervision, appearance-blind."

## What still stands

- **The v3 representation is sound.** FLAME mesh (5023 v) + 2048² UV atlas
  survived the geometry test. The bake failure was *splat-color → vertex-color*;
  the *multi-view-render → UV atlas* bake that v3 uses is the correct fix.
- **Stages 1–3 are keepers** — block / proportion / relief are
  appearance-tolerant and deterministic, each honestly gated by a metric.
- The differentiable surface already exists: pytorch3d renders the v3 textured
  mesh differentiably; `TexturesUV` makes the atlas a tensor.

## Representation verdict

Do **not** re-open the splat cloud. It has been falsified twice (rescatter,
then colour-bake blotchiness). The mesh + UV path survived the geometry test.
"Splatting into chibi" should mean: splat → mesh+UV (done) → fix appearance on
the mesh. A UV atlas is the most tractable appearance surface available — a 2D
image, known FLAME UV layout, known per-feature coordinates, directly
differentiable through `TexturesUV`.

## The live-driving constraint

The chibi avatar must remain drivable in real time via ARKit-52 blendshapes
(the iPhone live-demo path). LAM's animation is purely geometric: ARKit-52 →
FLAME blendshape basis → per-vertex `xyz` deltas; the identity net is never
touched by expression. So per frame:

```
displayed_verts = neutral + Σ wᵢ · Bᵢ        (Bᵢ = ARKit→FLAME blendshape i)
```

`Φ` is a *static* `verts → verts` remap of the **neutral** mesh. `Φ` cannot run
per frame — it fits a box and runs Laplacian smoothing; a per-frame box-fit
would jitter. The correct composition is the closed-form pull-back pattern
(`feedback_neural_renderer_override_audit`):

1. **Bake time, once:** fit the box on the *neutral* mesh and freeze it. With
   the box frozen, `Φ_frozen` is a smooth `verts → verts` map. Take its
   Jacobian `J` at neutral.
2. **Retarget the basis:** `B'ᵢ = J · Bᵢ` for each of the 52 blendshapes — 52
   vector-Jacobian products, seconds, once per avatar.
3. **Live, every frame:** `displayed_verts = Φ(neutral) + Σ wᵢ · B'ᵢ` — pure
   linear, trivially real-time.

A chibi blink then retargets through `J` so the lid moves correctly over the
enlarged, rounded chibi eye. The stages are already torch (grad was
deliberately stripped via `torch.no_grad()`); keeping grad live yields `J`.
Every stage is differentiable — superellipsoid projection, the exp-knot remap,
Laplacian smoothing, per-eye centroid scaling.

**Risk:** `J` is the linearization at neutral; very large expressions drift
from it. Bounded ARKit weights and LAM's own small-motion validity envelope
make this acceptable, but it must be gated — compare `Φ(neutral + ΣwB)` against
`Φ(neutral) + ΣwB'` on extreme frames and accept under a threshold.

The live constraint *reinforces* mesh + UV: UV coords are per-vertex static, so
a painted chibi feature rides the lid geometry for free — a static albedo atlas
is fully animation-compatible — and a 5023-vertex textured mesh rasterizes far
cheaper than LAM's 310 fps splat cloud.

## Recommended way forward — three components

1. **Geometry** — stages 1–3 (`HeadBlock`, `ProportionRemap`, `ReliefFlatten`),
   kept as built.
2. **Appearance** — a deterministic procedural texture compositor as the new
   Stage 4: paint chibi feature primitives (flat oval iris + highlight, strip
   mouth, button nose) into the v3 UV atlas at coordinates derived from the same
   FLAME masks `FeaturePrimitives` uses, so geometry and texture are
   co-registered by construction. No chibi target image, no diffusion prior, no
   CLIP/SDS — base chibi appearance is simple painted shapes at known UV
   coordinates, not a style-transfer problem. Generalizes across anchors for
   free (shared UV layout).
3. **Live driving** — Jacobian retarget of the ARKit→FLAME blendshape basis at
   bake time (`B'ᵢ = J · Bᵢ`); linear per-frame evaluation; extreme-frame gate.

The differentiable image-space machinery (directional-CLIP, NNFM, VQ-palette
through the rasterizer) is the right tool for a *later* anime cel-style track —
a genuine perceptual problem — and is held out of base-chibi scope.

## Open questions for review

- Is procedural feature paint actually sufficient, or does the relief-flattened
  realistic skin still read uncanny under painted features (does Stage 4 need a
  flat-skin appearance pass too)?
- Does `Φ`'s Jacobian retargeting hold for the eye region specifically, where
  `FeaturePrimitives` enlarges geometry most and the linearization is weakest?
- Is a single neutral-mesh box freeze safe, or does identity variation across
  anchors need a per-anchor `Φ` re-derivation?
