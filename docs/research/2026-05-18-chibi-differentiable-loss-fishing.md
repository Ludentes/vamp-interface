---
status: live
topic: lam-chibi-recipe
---

# Chibi differentiable-loss fishing — was our approach wrong?

Exploratory re-evaluation prompted by the question: *3DGS is differentiable optimization
— what losses could we add to bend the result, and did the chibi thread ever actually
use that lever?*

## The finding in one line

The chibi thread **never used the differentiable Gaussian rasterizer as an optimization
surface.** Every fit optimized a vertex-space landmark loss on the *baked, frozen* splat
cloud, moving only `xyz`. That loss is structurally blind to the failures that killed the
project. "Representation limit" is half the story; "wrong objective on the wrong surface"
is the other half.

## What the chibi thread actually optimized

`ChibiField` fit (splat path, concluded dead 2026-05-15):

```
Loss = L_landmark + λ_s·L_smooth + λ_r·L_reg + λ_c·L_curv
```

All four terms are functions of **vertex positions only**. `src/chibi/splat_render.py`
builds `means2d` with `requires_grad=False` — the rasterizer is forward-only. So:

- The optimizer moved `xyz`; opacity/scale/rotation/SH stayed frozen (mirroring LAM's
  own inference regime, which is validated only for *small* blendshape motion).
- The diagnosed failures — density rescatter (gaps on stretch), iris-through-lid leak,
  blotchy per-vertex color — are all **appearance-after-rasterization** effects. A
  vertex loss cannot measure them, so the fit reported success while renders fell apart.

The staged-redesign (2026-05-17) replaced the joint loss with per-stage geometric
metrics — better, but still vertex-space, still no image-space supervision, still
appearance-blind.

## Does LAM change the loss, or just use a foundational model?

LAM trains **its own** model with **its own** losses — it is not adapting a foundational
Gaussian model. From `LAM/configs/inference/lam-20k-8gpu.yaml` + `LAM/lam/losses/`:

| Term | Weight | Notes |
|---|---|---|
| masked-pixel (MSE) | 1.0 | photometric, matting-masked |
| LPIPS perceptual | 1.0 | VGG/AlexNet feature distance |
| mask / alpha | `1.0→0.5` over 10k steps | scheduled |
| offset regularization | 0.1 | keeps per-vertex Gaussian offset near FLAME |
| TV | disabled | available |

**But the released checkpoint is feed-forward inference only** — no test-time loss, no
per-subject optimization. At animation only `xyz` moves; opacity/scale/rotation/SH are
frozen identity predictions. LAM's `xyz`-only regime is validated for small FLAME
blendshape motion, *not* a 1.4× chibi stretch. We inherited that regime without
inheriting its validity envelope.

## Loss/regularizer research that solves our exact failures

- **GaussianAvatars** (CVPR 2024) — binds Gaussians to FLAME **triangles** (not
  vertices) and adds a **position loss + scaling loss**. Without them, "large spike- and
  blob-like artifacts appear wildly" under novel expressions — *our density-rescatter
  failure, fixed with regularizers.*
- **GaMeS** (Gaussian Mesh Splatting) — parameterizes each Gaussian by a mesh face
  (barycentric + local frame). Deform the mesh → Gaussians translate/rotate/**scale with
  the parent triangle automatically.** Coverage preserved *by construction*, no loss.
- **SuGaR** (CVPR 2024) — a **surface-alignment regularizer** (density / dn-consistency)
  that flattens Gaussians onto the surface so a mesh + texture extracts in seconds. This
  is literally "optimize the splats for future baking."
- **SC-GS** (CVPR 2024) — sparse control points + an **ARAP loss** for editable
  deformation with local rigidity.
- **StylizedGS / score-distillation editing** — optimization-based stylization driving
  splats with an image/style loss *through* the rasterizer.

## Regularizers we could add (answers the "extra members" question)

| Goal | Regularizer | Source |
|---|---|---|
| Limit splat count | L1 / entropy penalty on opacity + prune low-opacity | 3DGS densification |
| Prevent gaps under chibi stretch | coverage/density loss + adaptive densification (clone/split) | 3DGS, GaussianAvatars scaling loss |
| Pre-condition for UV bake | SuGaR surface-alignment + normal-consistency + opacity→binary | SuGaR |
| Keep splats mesh-bound during deform | position + scaling reg, or triangle rebinding | GaussianAvatars / GaMeS |
| Smooth deformation field | ARAP / Laplacian | SC-GS |

The bake-readiness angle is strong: if the end goal is splats → texture on a reference
head, a **SuGaR-style surface-alignment loss is the correct pre-conditioning** — it makes
splats flat and on-surface, which is exactly what makes the UV bake clean. The v1 mesh
gate failed because we baked splat *parameters* (`shs`) instead of a surface-aligned
cloud.

## Two paths forward

**Path A — image-space loss through the rasterizer.** Make `means2d` differentiable,
deform geometry, render the deformed splats, backprop a **photometric + LPIPS loss**
against a chibi target, with densification on and opacity/scale/SH *free*. The optimizer
can then re-densify and re-color to fix rescatter/leak — which a vertex loss never could.
Open question: the chibi target source (a chibi-stylized 2D render of the anchor,
multi-view; or SDS from a chibi diffusion prior).

**Path B — re-parameterize, GaMeS/GaussianAvatars-style.** Rebind LAM's per-vertex
splats to FLAME *triangles* (barycentric + local frame). Chibi stays pure geometry, but
splats scale with their triangle automatically — density preserved by construction, no
new coverage loss. Likely the cheaper, more robust win, and it directly unblocks the
mesh-bake pivot.

## Verdict

Was the approach wrong? Partly. "Chibi is too large for a baked splat representation" is
true *given* an `xyz`-only edit and a vertex-space loss. It is not known to be true under
an image-space loss that also optimizes opacity/scale/SH with densification, nor under a
triangle-bound (GaMeS) re-parameterization. Both are unexplored. The thread closed one
door and called it the building.

## Practical loss/regularizer catalogue

Framework: a **per-avatar test-time optimization loop** — no LAM retraining. Take LAM's
output splats (or a triangle-bound re-parameterization), mark parameters trainable,
render from sampled views through the differentiable rasterizer, backprop a weighted sum
of losses, Adam. Seconds-to-minutes per avatar.

Every loss is in one of three buckets, and that is how conflicts are reasoned about:

- **Geometry** — acts on vertex/splat *positions*. (chibi, matryoshka-ovoid, low-poly)
- **Appearance** — acts on the *rendered image*, through the rasterizer. (anime, ArcFace,
  Gram/style, color)
- **Structure** — acts on the *splat set itself*. (splat-count, densification,
  surface-alignment-for-baking)

### Anime

Composite, not one loss: chibi geometry + flat shading + quantized color, plus a style
anchor. Style anchor options:

- **Directional CLIP loss** (StyleGAN-NADA): minimize angle between
  `CLIP(render) − CLIP(photo_render)` and `CLIP("anime face") − CLIP("photo of a face")`.
  Cheap, no diffusion, controllable. We already use decoupled anime CLIP elsewhere.
- **SDS** from an anime diffusion LoRA — stronger, slower, risks identity drift + Janus
  multi-face. Fallback only.
- **VGG perceptual** to a single anime img2img of the anchor as a fixed target.

Schedule the style anchor in *after* geometry settles.

### Low poly

Fundamentally decimation (quadric edge-collapse) + flat per-face normals — an op, not a
loss. Soft/differentiable versions:

- **Opacity-L1 + pruning** on splats → fewer, larger blobs. This *is* the low-poly
  regularizer for the splat rep.
- **Planarity / developability regularizer** — cluster verts into patches, penalize
  within-patch normal variation → faceted surface with sharp creases.
- **Normal-map TV loss** through the rasterizer — L1 on normal gradients → piecewise-flat
  facets.
- **Vertex-position quantization** to a coarse lattice → blocky low-poly silhouette.

### ArcFace identity loss

`L_id = 1 − cos(ArcFace(render), ArcFace(anchor_photo))`, image-space. We already have
`insightface buffalo_l` installed. This is the **recognizability dial**.

Gotcha: ArcFace is trained on *real* faces; under heavy stylization the cosine collapses
and the loss fights the stylization, dragging output back toward realism. That is the
feature — it turns the chibi-ness ↔ recognizability tradeoff (the "relief flatten vs
identity" eyeball-call in `2026-05-17-chibi-geometry-redesign-design.md`) into an explicit
weight. Schedule: low weight early, ramp late; or compute on a lightly-destylized pass.

### Chibi

Geometry, staged — the redesign's per-stage metrics, but as differentiable losses with
the rasterizer in the loop so they also see appearance failures a vertex loss misses:

- **Proportion-remap loss** — feature lines hit the quarter grid.
- **Head-block loss** — superellipsoid fit on the cranium.
- **Relief-flatten loss** — face panel toward its Laplacian-smoothed surface.
- **Feature-primitive losses** — eye→circle, nose→button.
- **+ coverage/densification regularizers** — so the ~1.4× stretch does not tear splats.
  The piece the old thread never had.

### Matryoshka surface (largest doll)

The largest doll is a smooth ovoid solid with an essentially flat *painted* front.

- **Ovoid-SDF loss** — `L_ovoid = Σ |SDF_egg(v)|`, fit head to a parametric egg. Strong
  on back/sides, weak on the face panel. Same family as head-block, but a closed ovoid
  target and the whole head snaps to it.
- **Relief→0** on the face panel — matryoshka faces are painted, not sculpted.
- Face becomes pure texture on the egg → color (below) + style (below) do the real work.
  The "too glossy" feedback is a material/shading fix, not geometry.

### Colors

- **Palette / VQ loss** — soft-assign rendered colors to K palette centers + commitment
  penalty (differentiable k-means) → posterized cel blocks. Fix the palette for
  matryoshka; learn K≈8–16 for anime.
- **Color-TV loss** — L1 on color gradients → piecewise-constant flat regions.
- **Histogram / optimal-transport palette match** to a reference artwork.
- **Saturation regularizer** — push mean HSV saturation up (anime + matryoshka are
  saturated).
- **Albedo/shading split** — penalize shading variation so color reads as flat albedo;
  kills the glossy-matryoshka problem at the loss level.

### Style transfer / Gram matrix

`L_style = Σ_layers ‖Gram(VGG(render)) − Gram(VGG(style_img))‖²` + a content loss to hold
identity layout. This is exactly what **StylizedGS** does — optimization-based 3DGS
stylization through the rasterizer. Upgrades:

- **NNFM** (nearest-neighbor feature matching, from ARF) beats Gram for 3D style transfer
  — sharper, less wash-out. Prefer it.
- **Multi-view consistency** — Gram applied per-view independently can give inconsistent
  3D; sample multiple views per step and average. Watch for the SDS-style Janus artifact.

## Product recipes

**Matryoshka (active track):**

```
L = λ1·L_ovoid + λ2·L_relief→0 + λ3·L_palette(fixed) + λ4·L_gram(doll_painting)
    + λ5·L_id + λ_struct·L_surface_align
```

**Anime-chibi:**

```
L = chibi staged losses + λ·L_clip_dir(anime) + λ·L_VQ_palette + λ·L_color_TV
    + λ·L_id(scheduled) + λ_struct·(L_opacityL1 + L_surface_align)
```

Cross-cutting tension: ArcFace pulls toward realism; anime/chibi/matryoshka pull away.
Do not try to win both — `λ_id` is the explicit recognizability knob, a Pareto front,
the same uncanny-valley dial the project is built around. `L_surface_align` (SuGaR) is in
both recipes deliberately — it keeps splats bake-ready for the splats→UV-texture step.

## Sources

- SuGaR — https://arxiv.org/abs/2311.12775
- GaussianAvatars — https://arxiv.org/html/2312.02069v2
- GaMeS — https://arxiv.org/html/2402.01459v4
- SC-GS — https://arxiv.org/abs/2312.14937
- StylizedGS — https://arxiv.org/abs/2404.05220
- ARF (NNFM) — https://arxiv.org/abs/2206.06360
- LAM losses — `LAM/configs/inference/lam-20k-8gpu.yaml`, `LAM/lam/losses/`
