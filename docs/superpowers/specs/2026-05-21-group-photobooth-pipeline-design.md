---
status: live
topic: photobooth-sweep
---

# Group photobooth — multi-person → matryoshka group portrait

**Date:** 2026-05-21
**Scope:** Architecture for the next layer above the single-face photobooth: turn a photo of 1-5 people into a matryoshka group scene on a chosen background. Matryoshka-only; chibi is a later pivot that reactivates pose.

## Inputs and outputs

**Inputs:**
- A single photo with 1-5 people standing or sitting roughly upright.
- A background choice from a pre-rendered library (`assets/backgrounds/`), each with metadata describing its scene type.

**Output:**
- One composited PNG: chosen background with one matryoshka doll per detected person, positioned at the person's original location, sized to the person's apparent height.

**Not in scope (yet):**
- Pose-driven dolls (matryoshka is a rigid object — orientation is upright).
- Chibi pivot. When chibi replaces matryoshka, pose becomes load-bearing and a separate spec will cover it.
- Live/streaming. Output is offline.
- Background generation. Backgrounds are pre-rendered offline and shipped as assets.

## Locked decisions (from brainstorm with user)

1. **Single-face renderer:** Phase 3 "heavy mix" config (`swap_weight=0.10`, embedding pulled 28% toward doll, natural_1024 / canny / soft, demo_inject on, refine off). This is the per-face block — black box from the group pipeline's perspective.
2. **Background source:** Pre-rendered library with text descriptions. No on-demand background generation.
3. **Doll layout:** One doll per person, side-by-side at the person's original screen position. Not a nested set.
4. **Replacement region:** Full SAM-style person silhouette (not just the face / head bbox). Person mask defines the spatial footprint a doll should occupy.

## Approach selection

Three architectures considered. **A is recommended.**

### A — Per-face render, then rembg + composite (recommended)

Pipeline:
1. Detect + segment each person in the input photo.
2. Per person: crop face → run existing photobooth → 1024² doll portrait.
3. Per portrait: rembg → RGBA matryoshka silhouette.
4. Layout solver maps each person's mask to `(x, y, scale)` on the background canvas.
5. Paste dolls in painter's order (back-to-front by bottom-y) onto the background.
6. Optional polish: Lab color-match each doll to the background palette; drop shadow from the layout solver's foot anchor.

**Pros:** Cheap (no scene generation, no inpaint), deterministic, reuses the photobooth verbatim, backgrounds are fixed-quality. Each doll is rendered in isolation, so identity behaviour is exactly what we measured in Phase 3.

**Cons:** Lighting mismatch between doll and background must be patched by harmonization. Hard edges at silhouette boundary unless rembg + drop shadow + slight feather is done well.

### B — Single-pass scene generation with multi-face swap

Build a ControlNet image of N doll silhouettes at the person positions, generate the full scene (`background description + matryoshka dolls in <positions>`) with Z-Image + CN, then HyperSwap each face in the generated scene.

**Pros:** Unified lighting in a single pass. No compositing seams.

**Cons:** HyperSwap targets one face at a time and locating each target face inside the multi-face generated scene needs reliable per-person face detection + bbox tracking back to person identity. Generation cost scales with photo, not amortized over the background library. Less debuggable — failures span all stages.

### C — Per-person inpaint on the background

For each person, inpaint a matryoshka into the pre-rendered background at the person's mask location, using a reference doll portrait via IP-Adapter or ControlNet-reference.

**Pros:** Consistent lighting (inpaint sees the background). No rembg seams.

**Cons:** Identity injection through inpaint + reference is not a path we've validated for matryoshkas. Costs N inpaint passes per output. Mostly all the cost of A plus an inpaint that we'd have to characterise.

### Why A

A is the smallest delta over what already works. The face block is unchanged; the new code is detection + segmentation + cutout + 2D composition. B and C are downstream polish options if the composited output looks fake — we'll have ground-truth comparisons from A to decide whether the extra cost is worth it.

## Architecture

```
       Input photo                            Background library
            │                                         │
            ▼                                         │
  [A] Detect + segment people                         │
      • YOLOv8 person class → bboxes                  │
      • SAM2 (or YOLOv8-seg) → per-person mask        │
            │                                         │
            ▼                                         │
  [B] Per person:                                     │
      • Face crop via insightface buffalo_l           │
      • Skip person if no face is detected            │
            │                                         │
            ▼                                         │
  [C] Per person: photobooth (Phase-3 heavy mix)      │
      → 1024² doll portrait                           │
            │                                         │
            ▼                                         │
  [D] rembg → RGBA matryoshka silhouette              │
            │                                         │
            ▼                                         │
  [E] Layout solver                                   │
      person_mask → (x_center, y_bottom, height_px)   │
      on background-sized canvas                      │
            │                                         │
            └─────────────────────┐                   │
                                  ▼                   ▼
                  [F] Composite onto background
                      • painter's order: sort by y_bottom ASC
                      • alpha blend with feathered edges
                                  │
                                  ▼
                  [G] Polish (optional, can ship without)
                      • Lab color-match per doll to bg palette
                      • Drop shadow at foot anchor
                                  │
                                  ▼
                          Final group portrait
```

## Components

New code lives under `scripts/group_photobooth/`. Each module has one responsibility and a CLI for isolated testing.

| Module | Responsibility | Inputs → Outputs |
|---|---|---|
| `detect.py` | Run YOLOv8-person + SAM2 → per-person mask + bbox + face anchor | photo → `list[Person(mask, body_bbox, face_bbox)]` |
| `silhouette.py` | rembg cutout from a photobooth portrait | portrait → RGBA |
| `layout.py` | Map photo-space person geometry → background-canvas placements | `(person_list, photo_dims, bg_dims)` → `list[Placement(x, y, scale, z_order)]` |
| `composite.py` | Painter's-order alpha blend, optional Lab match + drop shadow | `(bg, placements, dolls)` → composited PNG |
| `driver.py` | Orchestrator. Reads input photo + background id, calls A-F, writes output. Resumable per-person cache. | CLI |

The single-face photobooth is invoked verbatim — we treat `swap_identity` + the Z-Image render as a black box reachable through a thin wrapper that takes a face crop and returns a portrait PNG.

The background library lives at `assets/backgrounds/<id>/`:

```
assets/backgrounds/
  manifest.json        # array of {id, description, dims, palette_lab_mean, lighting_hint}
  game_board/bg.png
  19th_century_house/bg.png
  blank_studio/bg.png
```

`manifest.json` carries:
- `id` — slug used by the CLI
- `description` — used by future LLM-driven background selection
- `dims` — `(W, H)` of the background PNG
- `palette_lab_mean` — `(L, a, b)` mean of background, for the Lab color match step
- `lighting_hint` — e.g. `"warm-overhead"`, `"cool-flat"`, used for the drop-shadow direction

## Layout solver

For each detected person:
- `body_mask` defines the silhouette in photo coords `(W_p, H_p)`.
- Take the mask's bounding box. Use its center-x and bottom-y as the doll's foot anchor in photo coords.
- The person's apparent height in pixels = bbox height in photo coords.
- Project to background coords by uniform scaling: `s = H_b / H_p` if we letterbox, or `s = min(W_b/W_p, H_b/H_p)` to fit. Letterbox is honest about the original aspect ratio; pick this. The non-matched dimension gets centered on the canvas with the background color sampled from the edge or simply cropped.
- Doll height in the background = `s × bbox_height`. Doll silhouette is resized to that height; width follows from its native aspect ratio.

**Occlusion:** Sort placements by `y_bottom` ascending (smaller = higher in image = farther back) and blit in that order. Standard 2D painter's algorithm.

**N=1 case:** Works identically. A single placement with one doll. Spec covers it without a special branch.

**Failure modes:**
- No people detected → output is just the background. Log + warn.
- A person mask whose bbox is wider than tall (lying down) → log a warning, still place a doll using the bbox center, but note that the result will look odd. v1 does not pose-correct.
- Two masks overlap heavily → painter's order handles overlap correctly; the back doll will be partially occluded by the front one, which is correct.

## Polish stage (optional, gated)

Implemented but off by default in v1. Two operations:

1. **Lab color match per doll.** For each doll silhouette, compute its Lab mean over visible (alpha>0) pixels, shift to `palette_lab_mean` of the background. Caps the shift at a small amount (e.g. `|ΔL|≤8`) so dolls don't lose their character.
2. **Drop shadow at foot anchor.** Render an elliptical shadow under each doll at `(x_center, y_bottom)`, blurred and dimmed, before the doll is composited. Direction taken from `lighting_hint`.

Both are pure post-processing — turning them off must produce a valid (uglier) output.

## Open implementation choices (resolved here, not deferred)

- **Detector:** YOLOv8 person class. Standard, fast, runs on CPU. No license issues.
- **Segmenter:** SAM2-base (CPU is fine, single image, ~1-2 s for the whole image). Alternative is YOLOv8-seg (faster, less precise edges). Pick SAM2 because edge quality matters for clean rembg-free silhouettes — even though we still run rembg on the doll portrait, the person mask defines the layout, and a clean mask gives a clean foot anchor.
- **rembg model:** `u2net` default. Backgrounds in matryoshka portraits are uniform (Z-Image renders dolls on flat-ish backgrounds), so u2net is sufficient.
- **CPU vs GPU:** Everything runs on CPU except the photobooth itself. The photobooth already orchestrates GPU usage; the group pipeline calls into it through the same path. Detection + segmentation + cutout + composite is CPU-fast (single-digit seconds per photo).

## Resumability + caching

Per-person face crops are cached at `<run>/people/<i>/face.png`. Photobooth outputs cache at `<run>/people/<i>/doll.png`. Silhouette cutouts cache at `<run>/people/<i>/doll_rgba.png`. The driver skips any stage whose output exists. This matches the photobooth's existing resumability discipline (`scores.parquet` + skip-if-exists).

## Testing

Three smoke tests, each runnable standalone:
1. **N=1 path:** Single-person input photo → group pipeline → output must visually match running the photobooth directly on the same face. Sanity check that the group pipeline degenerates to the single-face case.
2. **N=3 synthetic input:** Three person crops side-by-side on a known background, with hand-laid-out positions. Output should place three dolls at the expected `(x, scale)` triplets within a few pixels.
3. **Occlusion test:** Two persons with overlapping masks. Front doll must occlude back doll; reversing the painter's order produces a visibly wrong output (rear doll on top).

Visual smoke is enough at this scope. Numerical regressions would require ground truth we don't have. Per-person photobooth metrics (id_cos, clip_style) carry over unchanged from Phase 3.

## File layout

```
scripts/group_photobooth/
  __init__.py
  detect.py        # YOLOv8 + SAM2 → Person dataclass list
  silhouette.py    # rembg wrapper
  layout.py        # mask → background-canvas placement
  composite.py     # painter's-order alpha blend + polish
  driver.py        # CLI: --photo X.png --background blank_studio --out Y.png
assets/backgrounds/
  manifest.json
  blank_studio/bg.png        # ship with at least one to start
  ...
docs/research/2026-05-21-group-photobooth-architecture.md   # follow-up runbook after first end-to-end
```

## Cross-references

- Single-face renderer: `docs/research/2026-05-20-photobooth-architecture.md`
- Phase 3 swap_weight ladder verdict: `docs/research/2026-05-20-photobooth-phase2-findings.md` (and the open Phase 3 findings doc when written)
- Topic index: `docs/research/_topics/photobooth-sweep.md`
- Chibi pivot (later, reactivates pose): `docs/research/2026-05-20-uv-mapping-canonical-framing.md`
