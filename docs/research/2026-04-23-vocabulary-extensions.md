---
status: live
topic: metrics-and-direction-quality
---

# Vocabulary extensions — running list of beyond-blendshape confounds (2026-04-23)

Running log of effects we see in rendered outputs that are **real,
reproducible, and not captured by our current measurement stack**
(MediaPipe blendshapes + 11 NMF atoms + SigLIP-2 probes +
MiVOLO/FairFace/InsightFace age/gender/race + ArcFace identity + SigLIP
total-drift).

Each entry is an axis our current vocabulary is blind to. Over time
these get promoted to first-class vocabulary members — either by
adding a dedicated probe/classifier, by adding a SigLIP-2 probe pair,
or by building a specialized reader.

## Conventions

- **Added 2026-04-23** date stamps when an axis was first called out.
- **Source renders** point at the visual inspection that flagged it.
- **Proposed probe** is the quickest path to measurability — typically
  a SigLIP-2 probe pair (cheap, reuses existing infra) or a named
  classifier (richer, more setup).

## Open axes

### Skin smoothness / age-texture

- **Added 2026-04-23.** Seen on `overnight_beard/add` and
  `overnight_beard_rebalance/remove` for `european_m` and
  `asian_m` — skin texture smoothed, eye-bag shading reduced,
  pore-level detail dropped even at scale 1. mv_age picks this up
  imperfectly (~3-year drop not explained by explicit face shape).
  FairFace age_bin probably wraps this signal too.
- **Proposed probe** —
  SigLIP-2 pair `("a close-up photo with visible skin pores and fine wrinkles", "a close-up photo with smooth porcelain-like skin")`.
  Name `siglip_skin_smoothness_margin`.

### Hair — style / length / texture

- **Added 2026-04-23.** Seen on smile and beard renders on
  `young_european_f` — hair becomes wavier, sometimes fringe appears,
  sometimes parts shift. No current measurement exists. Likely a
  large contributor to `siglip_img_cos_to_base` drift but invisible
  in blendshape or classifier readouts.
- **Proposed probes (candidate list):**
  - `siglip_hair_length_margin`: `("long hair past the shoulders", "short cropped hair")`
    — already in our probe bank as `long_hair_margin` but not
    referenced systematically.
  - `siglip_hair_curly_margin`: `("curly wavy hair", "straight flat hair")`.
  - `siglip_hair_bangs_margin`: `("hair with bangs covering the forehead", "hair parted showing full forehead")`.

### Hair color

- **Added 2026-04-23.** Seen on `beard_rebalance` — background
  grey may have influenced a perceived hair darkening. Separate
  from skin tone. Currently uncaptured.
- **Proposed probe** — ordinal SigLIP pair
  `("dark black hair", "light blonde hair")` — ordinal axis, direction
  = dark→light. Or a dedicated hair-color classifier (InsightFace
  does not expose one; would need a purpose-built probe).

### Body pose / shoulder line / neck tilt

- **Added 2026-04-23.** Seen on smoke atom-inject renders at
  scale ≥ 25 — figure gains slight downward tilt, shoulders slope,
  body posture shifts from frontal to 3/4 view. Completely invisible
  to face-only detectors.
- **Proposed probe** — SigLIP pair
  `("a centered frontal portrait with level shoulders", "a tilted candid portrait with uneven posture")`.
  Name `siglip_posture_margin`.

### Image style / rendering quality

- **Added 2026-04-23.** Atom-inject at scale 25–50 produced
  *painterly* artifacts without shifting any face geometry.
  Important as a failure mode detector.
- **Proposed probe** — SigLIP pair
  `("a photorealistic photograph with sharp focus", "a painterly illustration with brush strokes")`.
  Name `siglip_photoreality_margin`. High magnitude = off-manifold
  editing, useful gate on composition outputs.

### Clothing visible / upper-body distraction

- **Added 2026-04-23.** Some renders produce visible shirt collars,
  necklines, tank tops. For a portrait-only study these count as
  confounds because they can pull age / gender / identity readings.
- **Proposed probe** — SigLIP pair
  `("a bare-shoulders portrait with no visible clothing", "a portrait with a visible shirt or top")`.
  Name `siglip_clothing_margin`.

### Background / scene

- **Added 2026-04-23.** Baseline prompt pins "plain grey background"
  but edits can introduce slight gradients, vignette, or color
  shifts.
- **Proposed probe** — SigLIP pair
  `("a plain uniform neutral grey studio background", "a background with color gradient or texture")`.
  Name `siglip_background_plainness_margin`.

### Expression intensity (beyond blendshape)

- **Added 2026-04-23.** Overnight smile rungs produce visibly
  different *intensities* of smile (faint → warm → broad → manic)
  that MediaPipe's linear 0–1 `mouthSmileL/R` doesn't fully capture
  because it saturates at broad before manic.
- **Proposed probe** — already covered by SigLIP `smiling_margin`,
  but worth double-checking whether manic-rung exceeds broad-rung
  on SigLIP cleanly.

## How this list evolves

1. **Add on inspection.** Any time a visual inspection surfaces a
   reproducible effect that isn't captured by existing readouts, add
   an entry here the same session.
2. **Promote on evidence.** If the same axis shows up in three
   separate inspections across different axes, build the probe and
   add it to `score_clip_probes.py` (SigLIP) or to a new classifier
   wrapper.
3. **Retire when covered.** When an axis is measured and surviving
   readouts confirm the effect, note the promotion at the bottom of
   its entry. Do not delete — it's provenance for why the probe
   exists.

## Promoted (none yet, placeholder for future)

_(axes here have moved into `score_clip_probes.py` or
`classifiers.py` — see the commit or the referenced PR for the
implementation.)_
