---
status: live
topic: lam-chibi-recipe
supersedes: 2026-05-18-chibi-identity-texture-design
---

# Chibi identity-in-texture v2: render-to-UV direct sample

> **2026-05-20 update (v2.1):** Three upgrades pulled from the canonical-UV framing brainstorm
> (`docs/research/2026-05-20-uv-mapping-canonical-framing.md`):
>
> 1. **Portrait identity backbone: InfiniteYou, not PuLID.** Stronger identity injection in
>    the same Flux generation step. Plugs into the same ComfyUI workflow slot.
> 2. **Delight is solved at the portrait stage, not the bake.** Prompt tokens (*flat lighting,
>    matte vinyl, no shadows*), negative prompt (*harsh shadows, specular, glossy*), and a
>    canny derived from a **flat-shaded** canonical-mesh render. No separate delight pass,
>    no albedo estimation network.
> 3. **Inpaint pass after mirror-fill: UV-space SD inpaint.** Mirror-fill seeds the unseen
>    UV regions; a diffusion inpaint pass (run in UV space with the mirror-fill as initial
>    image and an unseen-region mask) cleans seams and unseen-region textures. No UniTEX
>    dependency — UniTEX+InfiniteYou stays a parallel watch for code release, not a v2 blocker.
>
> The rest of this spec stands as-is. The phase gates below now incorporate these upgrades:
> P2 uses a flat-shaded mesh render for the canny; P3 swaps in InfiniteYou + delight prompt;
> P4 adds a UV-SD-inpaint sub-step after mirror-fill.

## Why this exists

The v1 spec (`2026-05-18-chibi-identity-texture-design.md`) was executed end-to-end in spike S1 on 2026-05-20 and failed in three independent ways at once:

- **Identity collapse.** The Task-4 portrait prompt is identity-blind (`"chibi character face portrait, front view, big round eyes..."`). The chibi LoRA wins style, PuLID can't pull the subject through, and every anchor ends up a generic chibi girl. The earlier `chibi_highstr_sweep.py` (`highstr_2026-05-15`) demonstrated identity *does* survive when the prompt names adult/structural tokens — we silently dropped that work.
- **Wrong canny.** The canny was derived from the input photo (photoreal edges: beard, cap, sunglasses, text) and fed into a chibi-style target. At CN strength 0.5 it loses to the chibi LoRA and contributes only confusion. The plan called for a canny derived from the canonical chibi mesh's frontal render; we never built that.
- **Architecturally fragile registration.** The v1 pipeline was `portrait → detect 16 landmarks → TPS-warp portrait to landmark-projection of mesh → projectively bake from warped image`. Three sub-failures: 16 landmarks is far too sparse for TPS (and all of ours sit on the upper face, none on the perimeter); chibi-mesh proportions are nothing like human-portrait proportions, so the warp compresses faces into a sliver and extrapolates the rest catastrophically; and the projection math in `koban_asset._project_verts` shipped with three independent coordinate bugs (divide vs multiply by focal length, spurious sign flips, mesh-vs-landmark frame mismatch) — all of which Task 1's shape-only tests passed.

The process failure under all three: every v1 phase gate verified shapes and dtypes; nothing rendered the canonical mesh through its own view and looked at the image. By the time we did (Task 6 spike), all three layers were broken at once.

## What v2 changes

**One architectural change** (kills the TPS warp + landmark detection step):

```
v1:  portrait → landmarks → TPS-warp → projective bake
v2:  portrait → (already framed like canonical view) → rasterize-UV-map → direct sample
```

The portrait is generated **framed identically to the canonical mesh's frontal view**, so no warp is needed. The bake becomes: rasterize the mesh through the canonical view to get a per-screen-pixel `(u,v)` map, sample the portrait at each screen pixel, scatter into the UV atlas. Three sources of error (sparse landmarks, TPS extrapolation, projection math) collapse into one constraint: *the portrait is framed correctly*. That constraint is solvable in the portrait stage, not the bake stage.

**Two prompt-stage changes** (fix identity collapse):

- Adopt `chibi_highstr_sweep.py`'s prompt verbatim, including the adult-structural suffix (`defined jawline, prominent chin, mature facial structure, high cheekbones, ..., adult, grown-up person, mature, in their thirties, designer collectible figure, matte vinyl finish, painted toy eyes`).
- Replace the photo-derived canny with a **canonical-mesh canny**: render the canonical chibi mesh, extract its frontal-view canny edges, use that as the structural target. This makes the controlnet pull *toward* chibi-mesh face structure instead of the photo's photoreal edges.

**One process change** (prevent recurrence of "shape-only verified" bugs):

Every task must produce a visual artifact, and the controller must view it before marking the task complete. No more shape/dtype-only gates.

## Architecture

### Pipeline

```
                                                   ┌── canonical mesh frontal render (one-time)
                                                   │
anchor identity PNG  ─┐                            ▼
                      ├─► Flux+PuLID portrait ──► canny ──┐
chibi-mesh canny  ────┘   (framed like canon)             │
                                                          │
                              canonical view  ──► UV rasterizer ──► uv_map (H,W,2)
                                                          │
                              portrait + uv_map ─► direct sample ─► texture atlas
                                                          │
                              ARKit driver  ────────► drive textured mesh ─► verify render
```

### Components

**Canonical chibi mesh asset** (mostly reuses Task 1 v1 work, with one render-and-eyeball gate added).
Files: `exp_output/chibi_meshes/koban_canonical/{koban.obj,centroid.json,view.json,landmarks.json}` are kept as-is. The v1 fixes to `koban_asset.py` stay (projection math, recentre-on-load). The asset is correct iff a frontal render of it produces a chibi face visibly centred in the frame. Visual gate.

**Mesh-frontal-render utility** (`src/chibi/canon_render.py`).
Renders the canonical mesh through its own `View` using pytorch3d (or nvdiffrast — whichever has cleaner UV-attribute paths). Returns: `(H,W,3)` RGB image *and* `(H,W,2)` UV map (per-screen-pixel u,v) *and* `(H,W)` visibility mask. The same call drives the canny-derivation step (RGB) and the direct-bake step (UV map).

**Mesh canny** (`src/chibi/canon_canny.py`).
Takes the canonical-mesh frontal render → Canny edges → writes to ComfyUI input dir as `chibi_canon_canny.png`. One-time artifact, anchor-independent. Replaces the per-anchor photo-canny in v1.

**Portrait wrapper** (`src/chibi/chibi_portrait.py`, replace v1).
Generates `<anchor>.png` via ComfyUI Flux + PuLID + chibi LoRA + the *mesh* canny. Identity ref is the anchor's photo (unchanged from v1). Prompt + schedule + strength = `chibi_highstr_sweep` `highstr_2026-05-15` recipe. Output is framed by the canny to match the canonical view, so no downstream warp is needed.

**Direct UV bake** (`src/chibi/uv_direct_bake.py`).
Takes `portrait` and `uv_map` → for every screen pixel with `mask=True`, write `portrait[y,x]` into `texture[uv_map[y,x]]`. Handles the discretisation (multiple screen pixels can map to the same texel: average them; texels with no screen pixel: fill via `mirror_fill` + `dilate_texture` + `skin_fallback` as in v1).

**Driver and verify render** (`scripts/chibi_identity_render.py`, replace v1).
Orchestrates: portrait → bake → ARKit-driven render. Same external interface as v1 (`--anchor --identity --out`), so call-sites and Task 7 stay valid.

## Phases and visual gates

Every phase must produce a PNG; controller views it; only then mark complete.

**P1 — Canonical render**
Build `canon_render.py`. Produce one PNG: `exp_output/chibi_meshes/koban_canonical/frontal.png` showing the recentred chibi mesh through its canonical View. Gate: *can you see a chibi face centred in the frame?* If no — fix view fov/dist or recentre/landmark frames before continuing.

**P2 — Canonical canny**
Build `canon_canny.py`. Produce `exp_output/chibi_meshes/koban_canonical/canny.png`. Gate: *does the canny show chibi-face edges — big eyes, small nose, head outline?* (Not photo edges.)

**P3 — Portrait via mesh-canny**
Update `chibi_portrait.py` to use the mesh canny + highstr recipe. Render id_14. Gate: *does the output preserve identity-bearing features (glasses, beard-equivalent stylization, age) AND match the canonical view framing?* Compare side-by-side with `exp_output/chibi_score/highstr_id14_grid.png` (the prior working sweep).

**P4 — Direct UV bake**
Build `uv_direct_bake.py`. Bake id_14. Gate: *does the UV atlas show face features in the correct UV islands and skin elsewhere?* (Atlas displayed as 1024×1024 PNG.)

**P5 — Textured turntable render**
Run `render_textured` over the baked texture. Gate: *is the rendered chibi recognisable as id_14, not as a generic chibi?* (3 azimuth angles, side-by-side.)

**P6 — ARKit-driven verify render**
Same as v1 Task 7. Drive the textured canonical mesh with ARKit blendshapes (neutral → blink → jawOpen → smile). Gate: *do the expressions read cleanly?*

## Open questions / risks

- **Will the canny + PuLID combo actually produce a portrait framed like the canonical view?** Open. ControlNet canny constrains structure but not face *position* in the frame — it constrains structure *given* a position. If the canny is centred and at the right scale, the portrait usually follows; if not, we fall back to either (a) inpainting a masked frame, or (b) a thin per-anchor 2D translation pre-bake. The fallback is cheaper than v1's TPS, so even the failure mode is recoverable.
- **Single-frontal-view is still a constraint.** Ears, head sides, top of head still get filled by `mirror_fill` + skin fallback. We accept this for v2; a multi-view bake is a v3 problem.
- **The `chibi_highstr_sweep` recipe was tuned for matryoshka, not for a chibi mesh.** The prompt may over-stylize toward "designer collectible figure" at the expense of mesh-fitting structure. We expect to need to drop a few of the matryoshka-specific tokens; iteration in P3 is part of the work.
- **What about anchors that fail PuLID identity preservation?** id_14 is a clean case; harder anchors (heavy stylization, occlusions) may need higher PuLID weight or different scheduling. We address per-anchor knobs only after P5 passes on id_14 *and* one other anchor.

## What we don't do

- No multi-view bake. v2 is one frontal view.
- No swap of the chibi mesh asset. Koban is fixed for v2.
- No retraining of any model.
- No further v1-style "shape-only" tests — visual gate or it doesn't count.

## Migration

- v1 work to keep:
  - `koban_asset.py` (with today's three projection/coord fixes)
  - `koban.obj`, `centroid.json`, `view.json`, `landmarks.json` (landmarks now used only as sanity-check overlays in P1, not as TPS control points)
  - `mesh.py`, `mesh_render.py`, `texture_bake.py` (rasterize/dilate/mirror utilities are reusable)
- v1 work to delete:
  - `register.py` (TPS warp + landmark detection) — no longer in the pipeline
  - `tests/test_register.py`
  - `koban_bake.py`'s `bake_portrait_to_uv` (replaced by `uv_direct_bake.bake`)
- Branch: continue on `chibi-mesh-deform`. Commits should reference this spec.

## Done definition

Spike S1 is "pass" when:
- A turntable render of a baked id_14 texture on the canonical chibi mesh is visibly *id_14-as-chibi* (glasses, age, masculinity preserved at chibi scale) — not a generic chibi girl.
- The same pipeline produces a recognisable distinct chibi for a second anchor (e.g. id_08).
- An ARKit-driven sequence (neutral → blink → jawOpen → smile) renders without artifacts.

Spike S3 (no double-shading) folds into the same render: if the textured chibi looks correctly lit (no baked-in highlights from the portrait fighting the renderer's lighting), S3 passes implicitly. If we see double-shading, we add a flat-lit prompt token and a delight pre-pass in P3 — *not* a separate task.
