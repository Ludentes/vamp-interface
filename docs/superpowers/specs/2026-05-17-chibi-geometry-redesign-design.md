---
status: live
topic: lam-chibi-recipe
supersedes: 2026-05-15-chibi-differentiable-geometry-recipe-design.md
---

# Chibi Geometry Redesign — Staged Re-priming Pipeline — Design

**Date:** 2026-05-17
**Topic:** `lam-chibi-recipe`
**Supersedes:** `2026-05-15-chibi-differentiable-geometry-recipe-design.md` (the `ChibiField` scale-and-slide approach)

## Problem

The `ChibiField` deformation has two kinds of operation: a global vertical **remap** (slide feature lines up/down the head axis) and four region **scales** (eye/nose/mouth bigger/smaller). It can slide and scale — nothing else.

A chibi, per the painter-rules doc (`2026-05-15-chibi-painter-proportion-rules.md`), is mostly things slide-and-scale cannot do:

- the head is a rounded **block/sphere** — not a scaled realistic skull;
- eyes are huge **and round** — not enlarged almonds;
- the nose is a **button** — bridge ridge and nostril relief *deleted*, not shrunk;
- the skin is **flat** — cheek volume, nasolabial folds, brow ridge *smoothed away*.

Worse, the fit is blind to its own failure: it optimizes landmark-line positions and bbox sizes, which cannot measure "is the head a block" or "is the nose a button." So the optimizer reports success while the geometry stays a realistic face with its features slid around — the result reads as *a face with a deformation applied*, because that is literally what it is. (Confirmed twice this thread: the radial-oscillation wasp-waist passed every landmark target, and the post-fix coherent head still reads as non-chibi.)

The missing class of operation is geometric **re-priming**: blend a region toward a chibi *target primitive* (a block, a smoothed surface, a round lens, a button) — *replacing* the realistic geometry rather than transforming it.

## Scope

**In scope.** The chibi geometry deformation, rebuilt as a staged re-priming pipeline on the FLAME-topology mesh. Four stages (head→block, proportion remap, relief flatten, feature primitives), per-stage geometric verification, and the composed deformation's ARKit-basis rescaling.

**Out of scope.** Flat-*skin* appearance (rule 8b — a UV-texture edit, separate appearance track); hair (not in FLAME topology); the UV-texture bake (done — v3); a single joint optimizer over all stages (a possible later refinement — YAGNI now, the point of the redesign is staged inspectability).

## Approach

Approach A — **rebuild the operator set** — selected by the user over a chibi-base-mesh + identity-transfer alternative (which needs a new authored asset and re-rigging — a separate, larger project). Two shifts from the superseded design:

- **Transform operators → re-priming operators.** A scale preserves the realistic geometry; a re-priming operator blends it toward an explicit chibi target shape. This is what turns a warped face into a chibi.
- **One joint Adam fit → a staged, individually-inspectable pipeline.** Each stage is a named `verts → verts` operator with its own small knob set, its own geometric success metric, and a render after it. "Step by step": you see the head after Block, after Remap, after Relief, after Features — and a stage that misbehaves is caught at its own gate, not hidden inside a joint loss.

The paradigm is unchanged: a continuous per-vertex deformation of the FLAME mesh, ARKit-drivable, feeding the existing v3 UV-texture render path.

## Architecture

The deformation is a composition `Φ = FeaturePrimitives ∘ ReliefFlatten ∘ ProportionRemap ∘ HeadBlock`, run by a `ChibiPipeline`. Each stage moves verts only; faces, UV, and texture pass through (the deform is UV-invariant — established by the v3 work). Ordering rationale: global structure before local detail — block the head first so everything downstream operates on a chibi skull; remap band positions on that block; flatten the face panel *before* re-priming features so the features are rebuilt on a clean surface; re-prime features last.

### `src/chibi/primitives.py` — parametric target shapes

Pure geometry helpers, no chibi logic. `fit_rounded_box(points) → RoundedBox`: least-squares fit of an axis-aligned superellipsoid (box↔sphere via an exponent, plus a corner radius) to a head point cloud. `project_to_box(verts, box) → verts`: nearest-surface projection. `laplacian_smoothed(verts, faces, iters) → verts`: cotangent/uniform Laplacian smoothing, the "relief-free" reference surface. `RoundedBox.surface_normal`, etc. Each is independently unit-testable on synthetic shapes.

### `src/chibi/stages/head_block.py` — Stage 1: head → rounded block

Fit a `RoundedBox` to the head verts. Blend each vertex toward its projection on that box by a per-vertex strength: strong on the cranium/skull mask (the block silhouette), mild on the face panel (only flatten it toward the box front plane — features are re-primed later). Jaw and chin corners round off as a side effect of the box's corner radius (covers rule 1 and rule 7's "rounded, not pointed chin"). Knobs: box exponent (sphere↔block), corner radius, cranium blend strength, face-panel flatten strength. **Verification:** `blockiness` — fraction of head verts within ε of the fitted box surface — rises; the fitted box exponent is in the block range.

### `src/chibi/stages/proportion_remap.py` — Stage 2: quarter-grid remap

The monotone piecewise-linear vertical remap from the superseded design, kept verbatim — it is the one operator that worked. `u → u_chibi` placing the feature bands on the block: big empty forehead, eye band on the 1/2 line, mouth on the 3/4 line, generous rounded chin sweep (rules 2, 6, 7). Reuses `ChibiField`'s remap math and the `QUARTER_GRID_TARGETS` line positions. **Verification:** `landmark_lines` hits the quarter-grid within ±0.03 (the existing check).

### `src/chibi/stages/relief_flatten.py` — Stage 3: flatten facial relief

On the face-panel mask (cheek, nasolabial, brow, nose-bridge regions from `FLAME_masks`), blend verts toward `laplacian_smoothed` of themselves — erasing cheekbone volume, nasolabial folds, the brow ridge, the nose-bridge ridge (rule 8a). Strength per region (the bridge wants near-total flattening; the cheek wants partial). **Verification:** `relief_energy` — RMS deviation of masked verts from the smoothed surface — drops below a target fraction of its original value.

### `src/chibi/stages/feature_primitives.py` — Stage 4: re-prime eye, nose, mouth

On the now-flat panel, rebuild each feature toward a chibi primitive:
- **Eye → big round lens.** Per-eye centroid (not a shared one — this is the fix for the eye-pod ballooning), reshape the lid aperture toward a large circle in the face-tangent plane (no depth inflation). Rules 3.
- **Nose → button.** Collapse the remaining bridge toward the face plane, pull the tip toward a small sphere-cap, remove nostril relief; tip width below mouth width. Rule 4.
- **Mouth → strip.** Vertical compression of the lip region. Rule 5.

Each is a blend-toward-target with a strength knob. **Verification:** nose-bridge height → ~0; eye-aperture aspect ratio → ~1 (round); mouth height ratio at target.

### `src/chibi/pipeline.py` — `ChibiPipeline`

Composes the four stages, loads parameters from one JSON (`chibi_pipeline_params.json` — a dict of per-stage knob sets). `run(mesh, through=None) → mesh` applies stages in order, optionally stopping after a named stage for inspection. Exposes `deformation_secant(verts, basis)` — the composed-Φ secant for ARKit-basis rescaling (the existing `secant_basis` mechanism generalizes to a composition; the chord `Φ(v+bₖ) − Φ(v)` is exact regardless of how many stages compose Φ).

### `src/chibi/chibi_metrics.py` — per-stage geometric verification

`blockiness`, `relief_energy`, `bridge_height`, `eye_aspect`, plus the global coherence checks already built this thread (triangle fold-over count, multi-angle lit render). Each stage's plan task asserts its own metric — the loss can finally *see* the chibi-ness.

### Rewiring

`mesh_deform.apply_chibi` runs `ChibiPipeline` instead of `ChibiField.forward`; its `ChibiMesh`/`TexturedMesh` dispatch and the v3 render path are unchanged. `ChibiField` is retained only for its remap math (imported by Stage 2) — the radial-scale and region-similarity operators are dropped (they are the scale-only operators this redesign replaces).

## Data flow

```
v3 textured mesh (FLAME topology) + FLAME_masks + chibi_pipeline_params.json
        │
        ▼  ChibiPipeline.run
   HeadBlock → ProportionRemap → ReliefFlatten → FeaturePrimitives
        │            (each stage: render + geometric-metric gate)
        ▼
   chibi mesh  ──> v3 render path (render_textured turntable)
        │
        ▼  [ARKit drive: composed-Φ secant rescales the 52-channel basis]
```

## Error handling

- `fit_rounded_box` on a degenerate point cloud (collinear / too few points) → raise, do not return a zero-volume box.
- Each stage asserts vert-count invariance (a deformation must not re-topologise) and NaN-free output.
- `ChibiPipeline` asserts the triangle fold-over count stays near baseline after each stage — the wasp-waist class of failure is caught at the stage that introduces it, not at the end.
- Missing/empty FLAME mask region → fail fast with the region name.

## Testing & verification

TDD per unit. `primitives.py`: box fit recovers a known synthetic box; `laplacian_smoothed` reduces a known bump. Each stage: identity params leave verts unchanged; the stage moves its target metric in the right direction and leaves out-of-mask verts within falloff tolerance. `ChibiPipeline`: `through=` stops correctly; the composed secant matches a finite-difference of `run`. Integration: the staged turntable render — a coherent block-headed chibi, inspected after every stage, no fold-over, landmark quarter-grid met. The verdict is per-stage (does *this* stage's metric pass) plus a final eyeball against the painter-rules doc.

## Open questions

- **Block vs sphere.** The box exponent is a knob; start mid-range, tune to the reference. The painter doc itself leaves this a tunable.
- **Joint refinement.** After the staged pipeline works, a light joint optimization over all knobs could remove inter-stage interactions. Deferred — staged inspectability is the current priority, and the per-stage gates make joint coupling visible if it matters.
- **Relief flatten vs identity.** Aggressive relief flattening erases anatomy that also carries *identity*. The strength knobs trade chibi-ness against recognisability; the right setting is an eyeball call against the anchor — flagged for tuning, not fixed here.
