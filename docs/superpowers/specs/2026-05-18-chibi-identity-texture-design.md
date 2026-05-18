---
status: live
topic: lam-chibi-recipe
supersedes: 2026-05-17-chibi-geometry-redesign-design.md
---

# Chibi Identity-in-Texture Pipeline — Design

**Goal:** Render any job anchor as a drivable chibi by baking a Flux-generated
chibi portrait onto the UV of a *fixed* off-the-shelf ARKit-rigged chibi mesh —
identity lives in the texture, geometry and rig are constant.

**Architecture:** A canonical chibi mesh (Koban Chibi Base Mesh — ships native
ARKit-52) is treated as immutable. Per anchor: Flux generates one flat-lit
frontal chibi portrait; it is landmark-registered and TPS-warped onto the
canonical mesh's frontal projection; a single-view projective bake writes it
into the mesh's UV atlas; symmetry-mirror + dilation + skin-fallback fill the
non-frontal texels. The textured mesh is then driven by the mesh's own ARKit-52
shape keys.

**Tech stack:** Python 3.12 / uv; Flux (existing project pipeline) + PuLID for
the portrait; insightface for landmark detection; nvdiffrast for UV-space
rasterisation; pytorch3d `TexturesUV` render; the existing `src/chibi/` bake
modules (`bake_points`, `rasterize_uv_attrs`, `camera_rig`, `mesh_render`).

---

## Why this supersedes the geometry-redesign track

The chibi thread spent three attempts trying to *deform* a FLAME/LAM head into
chibi proportions (splat edit → mesh deform → staged re-priming). Every attempt
failed on the same root cause: chibi-magnitude deformation rescatters a baked
appearance, and landmark losses are blind to the failure. The Koban mesh
removes the deformation problem entirely — it **is** a chibi, modelled by an
artist, and it ships the full ARKit-52 blendshape basis (61 shape keys verified
on `Chibi Base Mesh`). There is nothing to deform and nothing to retarget. The
only unsolved piece is appearance, and appearance on a fixed mesh is a texture
problem. The staged geometry-redesign spec
(`2026-05-17-chibi-geometry-redesign-design.md`) is therefore retired.

## Decision: Approach A — single-view projective bake

Three approaches were considered for getting a Flux portrait onto the canonical
UV. All three keep Flux as the identity source (LAM/MeshLAM cannot synthesise
identity from an embedding — they only reconstruct from an image).

**A — Single-view projective bake (chosen).** One flat-lit frontal Flux
portrait; landmark-detect; TPS-warp into the mesh's frontal-projection landmark
layout; rasterise the mesh UV to per-texel 3D position; project texels through
the frontal camera and sample the warped image; mirror + dilate + fallback for
the rest. Reuses ~90% of the existing FLAME-UV bake code. Deterministic, no
training, seconds per anchor. Load-bearing unknown: the TPS registration
(addressed by spike S1).

**B — Differentiable inverse-render fit.** Optimise the UV texture so a render
of the textured mesh matches the Flux portrait (photometric + LPIPS through the
rasteriser) — MeshLAM's texture branch without the learned net. Rejected as the
*first* step: it still needs landmark registration to initialise, runs a
per-anchor optimisation loop (minutes, not seconds), and adds failure modes. It
is a clean *refinement* to layer on A later if A's textures read soft.

**C — Texture-predictor net (embedding → UV).** Train a net mapping job
embedding → UV texture, ArcFace-supervised — the full "identity-in-texture"
vision. Rejected now as premature: its training corpus is exactly the
(embedding, UV) pairs that Approach A produces, so A is a prerequisite, not a
competitor. Revisit once A has generated a corpus.

A is the minimum that produces the artifact. B and C both depend on A.

## Requirements

- **R0 — Licensing gate. CLEARED (2026-05-18).** Koban Chibi Base Mesh is a
  paid Gumroad asset; the user confirmed use is fine for this local research
  project. The design remains written mesh-agnostically, but no swap is needed.
- **R1** — Geometry and rig are immutable. The canonical mesh is never deformed
  per-anchor; only its UV texture varies.
- **R2** — Per-anchor output is one UV texture PNG; identity is fully carried by
  that texture.
- **R3** — The textured mesh is drivable by ARKit-52 (blink, jawOpen, mouth/brow)
  via the mesh's native shape keys — no FLAME, no deformation transfer, no
  Jacobian retarget.
- **R4** — The pipeline is offline-batch and resumable (skip-if-exists per
  anchor), matching the project's pre-generate-and-cache model.
- **R5** — No double-shading: the Flux portrait is flat-lit and the verify
  render uses ambient-dominant (near-unlit) shading, per painter rule 8b.
- **R6** — Texture style matches geometry style: Flux generates a *chibi-styled*
  face (Flux + PuLID + style prompt), not a photoreal one. A photoreal texture
  on chibi geometry is the known uncanny-wrong failure.
- **R7** — Continuity (the project's core hypothesis) must be re-measured for
  the chibi style; it is not inherited from the photoreal Flux baseline. Out of
  scope for this spec — flagged for a follow-up measurement pass.

## Components

Each is a focused module under `src/chibi/`; drivers under `scripts/`.

**`koban_asset.py` — canonical-mesh prep (one-time).** Loads the Koban mesh
(from the VRM-export `.blend` via a headless Blender export script), **generates
a UV unwrap** (the Koban mesh ships none — see *UV unwrap* below), writes a
clean OBJ with UVs, verifies the ARKit-52 shape keys survive export, and stores:
the UV layout (reuse the `FlameUV` dataclass — it is just `vt` + `f v/vt`,
topology-agnostic), ~15 landmark vertex indices (eye corners, brow, nose tip,
mouth corners, chin), and one canonical frontal `View`. Output:
`koban_canonical/` (OBJ, `landmarks.json`, `view.json`, `arkit_keys.json`).

*UV unwrap (frontal projection).* Verified 2026-05-18: the Koban mesh — both
the `1.0.blend` and the VRM-export blend — has **no usable UV** (26 unique
coords across the whole mesh). The prep step generates one, computed directly
in Python (no viewport — headless-safe), not via a Blender viewport operator:
- *Face region* = the union of all polygons touching any vertex displaced
  (> ε) by any ARKit-52 shape key relative to `Basis`. The ARKit basis only
  moves face geometry, so this is a robust, rig-derived face mask — no manual
  selection, no material guessing.
- *Face UV* = frontal orthographic projection: each face-vertex's
  `(x, z)` (the two non-depth axes) normalised by the face region's frontal
  bounding box into the UV sub-rect `[0, 0.95]²`. The face UV island is then
  literally what the frontal camera sees — the projective bake is near-1:1 and
  TPS registration only has to correct the portrait's framing.
- *Non-face polygons* carry no identity; all their loops map to one reserved
  skin texel near `(0.98, 0.98)`. No second unwrap, no island packing, zero
  UV overlap. `skin_fallback` fills that texel with a flat skin tone.
This keeps geometry and the rig untouched (R1) — only UV coordinates are added.

**`chibi_portrait.py` — Flux portrait generation.** Given a job anchor
(embedding/seed), generates one 1024² flat-lit, front-facing, chibi-styled face
via the project Flux pipeline + PuLID + a flat-lighting style prompt. Resumable.

**`register.py` — portrait → canonical registration.** Detects 2D face
landmarks on the Flux portrait (insightface — Flux faces are photoreal enough
for a stock detector). Projects the canonical mesh's annotated landmark
vertices through the frontal `View` to 2D. Fits a thin-plate-spline warp
mapping portrait landmarks → canonical-projection landmarks, applies it, and
returns the warped portrait aligned to the canonical frontal projection.

**`koban_bake.py` — UV bake.** `rasterize_uv_attrs` (existing) rasterises the
canonical UV → per-texel 3D position + normal. `bake_points` (existing)
projects each texel through the frontal `View`, samples the warped portrait,
normal-weights the contribution. Then: UV-symmetric mirror-fill for texels the
frontal view missed but whose mirror partner is seen; `dilate_texture`
(existing) closes UV-gutter holes; a constant skin colour (median of seen
forehead texels) fills the genuinely unseen back-of-head texels (covered by
hair geometry anyway). Output: `<anchor>_texture.png`.

**`scripts/chibi_identity_render.py` — driver + verify.** End-to-end per
anchor: portrait → register → bake → write textured OBJ. Verify pass: render
the textured canonical mesh frontally (ambient-dominant) and drive a fixed
ARKit clip (neutral → blink → jawOpen → smile) to an mp4.

## Data flow

```
job anchor ─▶ chibi_portrait ─▶ portrait.png
portrait.png ─▶ register (insightface + TPS) ─▶ warped.png
canonical UV ─▶ rasterize_uv_attrs ─▶ texel pos/normal maps
warped.png + texel maps + frontal View ─▶ bake_points
  ─▶ mirror-fill ─▶ dilate ─▶ skin-fallback ─▶ <anchor>_texture.png
canonical OBJ + <anchor>_texture.png ─▶ TexturesUV mesh
  ─▶ frontal render + ARKit-52 driven clip ─▶ verify mp4
```

## Error handling

- **Landmark detection fails** on the Flux portrait → reject the anchor, log,
  continue the batch (R4 resumability); a regeneration with a new seed is a
  manual follow-up, not an automatic retry.
- **Texel unseen by the frontal view** → mirror partner if seen, else
  skin-fallback constant. Never left black.
- **R0 licensing unresolved** → implementation is gated; the plan's first task
  is the licensing check, and no rendering work starts until it clears.

## Spikes (resolve before/within implementation)

- **S1 — Registration quality.** Does insightface-on-portrait + TPS-to-canonical
  produce a bake that lands eyes/mouth/nose on the right canonical texels?
  Build it on one anchor, inspect the baked texture and a frontal render.
  *Acceptance:* we can describe whether TPS registration is sufficient or needs
  the Approach-B optimisation refinement.
- **S2 — Rig drives cleanly.** Set `eyeBlinkLeft/Right`, `jawOpen`,
  `mouthSmileLeft/Right` to 1.0 on the canonical mesh and render. *Acceptance:*
  we can describe whether the native ARKit-52 shape keys deform correctly with
  no broken geometry. (This is the blink/jawOpen test already offered.)
- **S3 — No double-shading.** Bake a flat-lit portrait, render ambient-dominant.
  *Acceptance:* we can describe whether residual Flux shading reads as
  baked-in relief on the chibi.

## Testing

- **`register.py`** — unit test: synthetic landmark pairs (a known affine +
  small jitter) → TPS warp recovers the affine within tolerance.
- **`koban_bake.py`** — unit test: a checkerboard "portrait" + a trivial planar
  patch UV → baked texture is a checkerboard (projection/sampling correctness);
  test mirror-fill on a half-masked texture; test dilation closes a 1-texel hole.
- **`koban_asset.py`** — test: exported OBJ round-trips through `FlameUV` with
  the expected vt/face counts; ARKit-52 key names are all present.
- **End-to-end** — verify render gated by visual inspection plus a metric:
  re-detect landmarks on the rendered canonical frontal and check they fall
  near the annotated targets.

## Out of scope

- Continuity re-measurement for the chibi style (R7) — separate pass.
- Approach B (inverse-render refinement) and C (texture-predictor net) — future
  iterations, both dependent on A.
- Multi-anchor batch tuning, UI integration, serving.
