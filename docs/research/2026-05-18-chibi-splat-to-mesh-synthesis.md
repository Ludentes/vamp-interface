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

> **⚠️ Gated.** Adversarial review (see *Adversarial review* below) found three
> load-bearing claims here that are asserted, not verified. Treat this section
> as a *hypothesis to test*, not a plan to execute. Run the three gating spikes
> first.

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

## Adversarial review (2026-05-18)

An adversarial reviewer attacked this synthesis against the code. Three hits
are severe and the recommendation above does not survive without addressing
them. The full critique is sound; the key concessions:

**Root cause is over-unified.** "Appearance was never an optimization variable"
cleanly explains failure #3, only *partially* explains #1 (an `xyz`-only edit
tears a fixed-density cloud regardless of whether appearance is free — the
missing piece is *densification/coverage*, not colour), and *mis-describes* #2
(the v1 bake *moved* appearance, it just misread SH-DC as albedo). One true
observation about the latest failure was retro-projected onto two failures with
distinct causes — and the tidy root cause then launders the recommendation.

**Path B was strawmanned.** The "splats falsified twice" verdict conflates a
representation failure (#1) with a bake-method failure (#2). GaMeS /
GaussianAvatars triangle-rebinding — splats bound to FLAME triangles, coverage
preserved by construction under deformation — is refuted by *neither* failure
and is dismissed here without a sentence of real engagement. It remains a live
alternative for the geometry-coverage problem (though it does not by itself
solve the chibi *appearance* problem, and a 2D UV atlas is a far more tractable
appearance-edit surface than splat SH).

**Three gating spikes — run before any spec.**

1. **Is the v3 UV bake actually clean?** Everything downstream assumes it. The
   *geometry* survived the bake; the *texture* has never been shown
   artifact-free. *Spike:* render the v3 baked-textured neutral mesh vs the
   original LAM splat render, ≥4 views, eyeball for seams, baked-in lighting,
   and the sRGB/linear washout the topic index already flags. An afternoon;
   either unblocks or kills the plan.

2. **Does Jacobian retargeting hold at normal expression amplitude?** `Φ` is
   sharply nonlinear exactly where blendshapes animate — eye scale is taken
   about a *vert-dependent* centroid (`feature_primitives.py:38-46`), the
   superellipsoid has `abs()` gradient kinks on every box axis plane
   (`primitives.py:40-42`), and relief-flatten compresses panel z to ~0.2×
   (`relief_flatten.py:32-33`), so `J·B` silently flattens z-component
   blendshapes too. The proposed extreme-frame gate only catches large weights;
   the dangerous case is a *normal-amplitude blink* wrong because the eye is the
   most nonlinear region. *Spike:* compute `J`, retarget the 52-vector basis,
   render `Φ(neutral+ΣwB)` vs `Φ(neutral)+ΣwB'` for ~20 normal-amplitude ARKit
   frames (blink/smile/brow), measure per-vertex error inside the eye/lip mask
   specifically. Visible blink error ⇒ the linear retarget is dead.

3. **Is procedural paint sufficient, or sticker-on-a-photo?** The "no
   optimization needed" claim contradicts the project's own recorded belief
   (painter rule 8b — flat skin is a separate appearance edit) and dodges the
   inherited photoreal skin, baked lighting, hairline, and the paint↔skin seam.
   *Spike:* hand-paint flat chibi eyes into the v3 atlas in an image editor,
   render the mesh. If it reads as a sticker, the plan needs a deterministic
   de-light / flat-skin pass as a *peer* of the paint compositor — "no
   optimization" downgrades to "no *learned* optimization, but a de-light pass
   is required."

**Lesser but real:** pytorch3d `TexturesUV` (UV-seam interpolation, no
mipmapping) may not equal the nvdiffrast bake path — render-for-`J` and
render-for-bake are not assumed-identical; FLAME's 5023 verts are distributed
for *realistic* eye topology and may pinch when an aperture is enlarged 1.6× and
rounded — and the geometry aperture may then not co-register with the painted
oval; live mesh-rasterization plumbing into the iPhone demo is *new* work,
whereas the splat live path is already shipped.

**Sound as-is:** geometry stages 1-3 (deterministic, metric-gated); the
*structure* of basis retargeting (freeze box, compose into basis — only the
linearization error is unbudgeted); holding CLIP/NNFM/SDS out of base-chibi
scope.

## Next step

Run spike 1 first — it is the cheapest and the most foundational; a dirty v3
bake invalidates the entire mesh+UV appearance premise. Then spike 3 (the
hand-paint mock), then spike 2 (Jacobian validation). Only after all three is a
brainstorming → spec pass warranted.
