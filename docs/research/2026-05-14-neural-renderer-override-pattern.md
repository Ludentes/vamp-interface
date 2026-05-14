---
status: live
topic: lam-chibi-recipe
---

# The neural-renderer override-audit pattern

**Date:** 2026-05-14
**Status:** Methodology note distilled from the chibi splat-scale fix. Applies whenever we override one output of a neural renderer that emits a coupled set of per-vertex predictions.

## The rule

> When you override a geometric input to a neural renderer, audit which downstream predictions were conditioned on the un-overridden value, and inject the matching closed-form correction at the same hook point.

A neural renderer's per-vertex outputs are not independent — they're a coupled set produced from one shared geometric input. The API exposes them as if they were independent (`_gm.xyz`, `_gm.scaling`, `_gm.rotation`, …) but they share a hidden invariant: the network was trained on one mesh density, and each output is consistent with the others only at that density.

When you take over one output, you take on responsibility for every invariant the rest of the network is silently relying on. The network has no way to flag the inconsistency — it just renders whatever it predicted, and the surface tiles wrong.

## The chibi case as a worked example

LAM's `GSLayer` predicts three per-vertex things from FLAME geometry: positions, sigmas, colors. The chibi recipe overrode positions and left sigmas conditioned on the canonical (unstretched) mesh. Result: splats stayed at original size while the surface stretched ~1.4× under them. The lid sheet developed gaps; iris read through closed eyelids.

The fix is the closed-form pull-back of "splat size matches local vertex density" under the deformation: `r_i = mean_edge_chibi_i / mean_edge_original_i`, computed per vertex, multiplied into `_gm.scaling` at the same hook point as the position override. Cost: 1 s precompute, 0 ms per frame. Detail in [`2026-05-14-chibi-splat-scale-fix.md`](2026-05-14-chibi-splat-scale-fix.md).

## The audit checklist (for any new override)

When taking over one of a neural renderer's per-vertex outputs:

1. List every per-vertex output the network emits.
2. For each one, ask: what local geometric assumption was it conditioned on?
3. For each one, derive the closed-form pull-back under your deformation.
4. Inject the pull-back at the same hook point as your override.

The chibi audit, made explicit:

| Output | Conditioned on | Pull-back under chibi stretch | Hook landed? |
|---|---|---|---|
| `_gm.xyz` | Canonical FLAME mesh | The override itself | Yes (`LAM_EDIT_XYZ_OBJ`) |
| `_gm.scaling` | Canonical surface-area-per-splat | Per-vertex edge-length ratio | Yes (`LAM_CHIBI_SCALE_RATIO`) |
| `_gm.rotation` | Canonical vertex normals | New normals from chibi mesh (approximately tracked because rotations are relative to local frame) | Not yet — next predictable failure under more aggressive deformation |
| `_gm.opacity` | Identity-dependent only | Unchanged | N/A |
| SH-DC colors | Surface properties (photo bake) | Unchanged | N/A |

## Will this scale with deformation strength?

**The global compensation is monotone in strength; the visible artifacts are a sequence of local thresholds.** The scalar-edge-ratio fix is computed from the actual deformed mesh, so it grows with the deformation by construction. What does not scale is the secondary artifact ladder:

- At s=1.0–2.0: iris-through-lid was the dominant artifact. Fixed by the ratio.
- At s=2.0: lid-margin doubling emerged — upper-lash and lower-lash bands no longer merge because both rim splats got enlarged symmetrically.
- At s=3.0 (untested): likely nostril splats over-spreading; possibly rotation mismatch on the radial-stretch boundary.

Each artifact has its own threshold. There is no single fix that pre-emptively handles all of them; each requires its own diagnosis at its own strength regime.

## Will this extrapolate to other deformations?

| Deformation | Scalar edge-ratio fix applies? | Notes |
|---|---|---|
| Radial chibi stretch (our case) | ✅ | Smooth, near-isotropic |
| Anime-style oversized eyes | ✅ | Same recipe + extra eye-region multiplier |
| Furry/animal snout extension | ⚠️ | Snout verts have huge anisotropic stretch (long along snout axis, narrow across). Scalar ratio under-corrects; v3 anisotropic likely needed |
| Ghost/translucent | ❌ | Different problem. Modify SH-DC or opacity, not sigma |
| Human → fridge (non-FLAME morphology) | ❌ | Stretch ratios > 5 with strong anisotropy. Beyond the scalar model |

The scalar ratio is the isotropic special case of a 2×2 tangent-plane covariance pull-back. Whenever local stretch is large **and directional**, replace the scalar with the covariance (anisotropic v3, not currently implemented). The audit pattern stays the same; only the closed-form pull-back gets richer.

## Meta-lesson: prefer closed-form pull-backs over trained corrections

The chibi fix is a closed-form function of mesh geometry: `f(canonical_mesh, deformed_mesh) → per_vertex_ratio`. No training, no checkpoint, no forward pass. The asset is 80 KB; precompute is 1 s; runtime is one multiply at model load.

The alternative — train a small MLP to predict per-vertex sigma corrections from blendshape weights + chibi parameters — would give a checkpoint to manage, a corpus to curate, a forward pass at load time, and a quality regression risk on new anchors. The geometric pullback gives the same answer for free.

This is now three closed-form geometric assets stacked into the same pipeline:

| Asset | Closed-form input | Output |
|---|---|---|
| `chibi_arkit_bs.npy` | (canonical ARKit basis, deformed mesh) | Re-projected ARKit basis on chibi geometry |
| `chibi_textured_mesh.obj` | (canonical mesh, chibi_strength) | Stretched mesh |
| `chibi_scale_ratio.npy` | (canonical mesh edges, chibi mesh edges) | Per-vertex sigma multiplier |

**This is the recipe shape worth stealing for the next stylization axis** — anime eyes, furry head, whatever — before reaching for trained correction networks. Trained corrections are the fallback when no closed-form pull-back exists, not the default.

## Operational rule for next stylization recipe

When adding a new deformation (e.g. anime-eye boost, snout extension, ear morph):

1. Build the deformed mesh.
2. Run the audit checklist above on every `_gm.*` output.
3. For each output with a non-trivial pull-back, add an asset to `chibi_make_assets.py` (or its sibling) and wire an env var.
4. Three-moment visual gate before declaring shipped.
5. Document any artifact thresholds that emerged at the strength regime tested; do not claim coverage of untested strengths.
