---
status: live
topic: lam-chibi-recipe
---

# Research: Splats → Mesh for the Chibi Pivot

**Date:** 2026-05-15
**Sources:** 9 sources — SuGaR (CVPR 2024), 2DGS, GeoAvatar (ICCV 2025), GaussianAvatars, Texture-GS, GStex (WACV 2025), HairGS, CGHair, Strands2Cards (SIGGRAPH Asia 2025).

---

## Executive Summary

We are not in the situation most "splat → mesh" papers solve. SuGaR/2DGS/GOF exist to *recover* a surface from an unstructured splat cloud — extract topology where there was none. The LAM avatar already **is** a FLAME-topology mesh (5023 → 20018 verts, one subdivision, with a UV layout shipped by FLAME). The Gaussians are per-vertex attributes hung on that mesh by the GSLayer MLP. So the conversion is not surface reconstruction; it is **appearance baking** — turn each splat's SH-DC color into a value in a FLAME UV texture, then throw the Gaussians away. That is a far smaller, far better-posed problem than the literature's headline task, and it is the reason the mesh pivot is cheap. The genuinely hard residuals are not the face: they are teeth (FLAME has none, LAM's checkpoint trained them off) and hair (FLAME has no scalp geometry; LAM fakes hair by scattering splats off the head verts — that geometry does not survive baking to a FLAME UV).

## Key Findings

### We already own the mesh — skip surface extraction entirely

SuGaR's whole pipeline (short 3DGS optimization → surface-alignment regularization → Poisson reconstruction → mesh) and 2DGS's TSDF-fusion-from-depth-maps both solve "I have a splat cloud, I need a surface" [1][4]. Neither applies to us. LAM's canonical representation is the FLAME mesh; the splats are emitted *from* its vertices. We have watertight, animation-ready, UV-unwrapped topology already, with the FLAME LBS + jaw + eyeball rig and the ARKit-52 blendshape basis attached. The chibi deformation is then ordinary mesh deformation of a known-topology mesh — exactly the "well-behaved mesh tools" regime the user asked for. The one thing to take from SuGaR is its *endpoint*, not its pipeline: the "Mesh + bound Gaussians" hybrid, where "mesh deformations automatically propagate to bound Gaussians" [2]. If we ever want to keep a Gaussian renderer, GaussianAvatars already gives us that binding for free — it initializes one Gaussian per FLAME triangle, rigidly parented to the triangle's frame [5]. But for chibi the point of the pivot is to *drop* the Gaussian renderer, so this is a fallback, not the plan.

### Baking: SH-DC → FLAME UV texture is the core operation

Standard 3DGS stores color as per-Gaussian spherical harmonics; the order-0 (DC) term is the view-independent diffuse color. Texture-GS and Texture-GS-like work show the move: "represent the view-independent appearance as a 2D texture map" instead of per-Gaussian attributes, keeping SH only for residual view-dependent gloss [3]. GStex does the per-primitive version — texturing 2D Gaussians so appearance is decoupled from primitive density [6]. For us the procedure is concrete and does not need any of their machinery:

1. For each of the 20018 mesh vertices, take the LAM-emitted splat's SH-DC, convert to linear RGB (`C0 = 0.28209479` factor — DC color is `0.5 + C0·sh_dc`).
2. Each vertex has a known FLAME UV coordinate. Rasterize the mesh in UV space; for every texel, barycentric-interpolate the three corner vertex colors. That is the diffuse texture.
3. Resolution: 1–2K is plenty for a 256×256 product render. Vertex colors are only 20018 samples, so the texture is interpolation-limited, not resolution-limited — going past ~1K buys nothing.

The decisive property, and the whole reason the pivot fixes chibi: a UV texture is **deformation-invariant**. Texture-GS notes that once geometry and texture are disentangled, editing geometry "automatically maintains view consistency" because the texture is indexed by UV, not by world position [3]. Chibi magnitude stops mattering — stretch the mesh however far, every triangle still samples the same patch of texture. This is exactly the failure the baked-splat representation could not survive (appearance glued to verts at fixed density → rescatter on deformation). Bake once, deform freely.

Two cautions. First, **bake from the canonical (rest) pose**, before any chibi or ARKit deformation — the texture must be authored in the undeformed UV layout. Second, LAM's SH-DC has lighting partially baked in; we are baking *shaded* color, not true albedo. For a flat-shaded stylized chibi (painter rule: chibi wants flat skin) that is acceptable and arguably correct. If later we want relighting, that is an inverse-rendering problem (GS-ID-class illumination decomposition) and explicitly out of scope now.

### Teeth: FLAME has none — add a static proxy, do not recover from splats

FLAME models "an articulated neck, jaw, and eyeballs" but no teeth or tongue [the FLAME model]. LAM's released checkpoint additionally has teeth trained *off* (prior inventory finding). So there is nothing to bake — there are no teeth splats to convert. GeoAvatar's relevance here is its diagnosis, not its method: it explicitly segments the head into a **rigid set, a flexible set, and a separate mouth-structure set**, treating the mouth interior as its own problem class precisely because the jaw region is where Gaussian-on-FLAME avatars break [7]. The mesh-world answer is the standard rigging answer: add a **static teeth proxy** — two simple arch meshes (upper/lower), upper rigid to the skull, lower rigid to the FLAME jaw joint so it follows `jawOpen`. This is how every game-engine FLAME/MetaHuman head does it. For a chibi the mouth is usually a flat simplified shape anyway (painter rule: mouth simplification), so a low-poly dark cavity + minimal teeth card may be all that is needed — possibly nothing at all if the chibi mouth is drawn closed/simplified.

### Eyelids: keep them as FLAME mesh — the chibi iris-leak is *fixed* by the pivot

The iris-through-lid leak that consumed two falsification rounds was a splat-density artifact: enlarging the eye region spread the lid splats thin enough to see the eyeball splats through them. On a mesh this failure mode does not exist — the lid is a closed surface, opaque by construction, regardless of how far the eye region is scaled. FLAME already has eyelid geometry and eyeball spheres; keep them, deform them with the rest of the head, let the rasterizer occlude. The entire `LAM_CHIBI_SCALE_RATIO` / J·SVD lid-mask apparatus exists only to fight a Gaussian-density problem and can be **deleted** in the mesh world. The eyeball is a separate small mesh parented to the FLAME eye joint; the enlarged-eye chibi move becomes: scale the lid opening (mesh deformation) + scale the eyeball proxy. No leak is geometrically possible.

### Hair: the real open loss — splats off the scalp don't bake to a FLAME UV

This is the one part with no clean answer. FLAME has no hair geometry; LAM represents hair as Gaussians scattered outward from the scalp/skull vertices. Those splats are *not* on the FLAME surface, so they have no FLAME UV coordinate — step 2 of the bake (rasterize in UV space) silently drops them. Three options, in increasing fidelity and cost:

- **Helmet hair (cheapest, recommended first).** Treat hair as a thin shell offset from the FLAME scalp verts — a displaced copy of the upper-head triangles, baked with the hair-colored splats projected onto it by nearest-scalp-vertex. Deforms with the head for free (it is parented to the same verts). Loses all wisp/strand structure; gives a solid stylized hair cap. This is *on-brand for chibi* — chibi hair is drawn as big simple volumes, not strands. Likely good enough and should be tried before anything harder.
- **Hair cards.** Strands2Cards (SIGGRAPH Asia 2025) and CGHair both go strand-cloud → textured polygon cards [8][9]: cluster strands into wisps, fit polygon strips, bake a texture per card. CGHair specifically projects strands onto card mesh faces to make the texture. This is the production-standard hair representation and deforms well as mesh. Cost: we would need strands first.
- **Strands from the hair splats.** HairGS converts a hair Gaussian cloud into "3D polylines" — variable-length strands — by merging each Gaussian into a 2-joint segment and growing under photometric supervision [the HairGS pipeline]. Output is polylines, not mesh; it does not itself export cards. So the full path is HairGS (splats → strands) → Strands2Cards (strands → cards). Heaviest option; only worth it if helmet hair looks too cheap.

Recommendation: ship helmet hair, hold cards in reserve. The chibi aesthetic forgives simplified hair more than it forgives a broken face.

## Comparison

| Concern | Generic splat→mesh paper | Our LAM situation |
|---|---|---|
| Topology | Must be reconstructed (SuGaR/2DGS/GOF) | Already FLAME, UV-unwrapped, rigged — skip |
| Appearance | Re-bake from multi-view images | Bake SH-DC of 20018 per-vertex splats → UV texture |
| Deformation | Re-bind Gaussians or re-extract | Ordinary mesh deformation; texture UV-locked |
| Rig | Usually none | FLAME LBS + jaw + eyeballs + ARKit-52 kept as-is |
| Teeth | In the scan | Absent — add static jaw-parented proxy |
| Eyelids | Density artifacts | Mesh = opaque by construction; leak gone |
| Hair | In the scan | Off-surface splats — does not bake; needs a shell or cards |

## Open Questions

- **Texture seam quality at the FLAME UV cuts.** FLAME's UV layout has seams (ears, back of head); barycentric-baked vertex color may show seam discontinuities. Likely minor at 256² output; verify on first bake.
- **Does LAM's SH have meaningful view-dependent terms?** If order ≥1 SH carries real specular/translucency, flat-baking the DC term loses skin sheen. Probably negligible for a stylized target; measure SH-DC vs full-SH render difference once.
- **Helmet-hair coverage.** Whether a scalp-offset shell can cover LAM's hair splat extent (long/voluminous styles) without gaps — unknown until tried on a real anchor. The `me` anchor with its deeper z-range is the stress case.
- **Renderer choice post-bake.** Rasterized mesh (lose the Gaussian look entirely) vs SuGaR-style bound-Gaussian hybrid (keep splat softness, accept the density coupling on hair only). The pivot's premise is "drop the renderer," but the hybrid is a documented fallback if flat rasterization looks too hard-edged.

## Sources

[1] Guédon & Lepetit. "SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction." CVPR 2024. https://arxiv.org/html/2311.12775v3 (Retrieved 2026-05-15)
[2] SuGaR project page. https://anttwo.github.io/sugar/ (Retrieved 2026-05-15)
[3] Xu et al. "Texture-GS: Disentangling the Geometry and Texture for 3D Gaussian Splatting Editing." https://arxiv.org/html/2403.10050v1 (Retrieved 2026-05-15)
[4] "How to Extract 3D Meshes from Gaussian Splats: SuGaR and 2DGS." Inverse Render blog. https://inverserender.com/blog/how-to-extract-meshes/ (Retrieved 2026-05-15)
[5] "GaussianAvatars: Photorealistic Head Avatars with Rigged 3D Gaussians." https://arxiv.org/html/2312.02069v2 (Retrieved 2026-05-15)
[6] Rong et al. "GStex: Per-Primitive Texturing of 2D Gaussian Splatting." WACV 2025. https://openaccess.thecvf.com/content/WACV2025/papers/Rong_GStex_Per-Primitive_Texturing_of_2D_Gaussian_Splatting_for_Decoupled_Appearance_WACV_2025_paper.pdf (Retrieved 2026-05-15)
[7] Moon et al. "GeoAvatar: Adaptive Geometrical Gaussian Splatting for 3D Head Avatar." ICCV 2025. https://openaccess.thecvf.com/content/ICCV2025/papers/Moon_GeoAvatar_Adaptive_Geometrical_Gaussian_Splatting_for_3D_Head_Avatar_ICCV_2025_paper.pdf (Retrieved 2026-05-15)
[8] Pan et al. "HairGS: Hair Strand Reconstruction based on 3D Gaussian Splatting." https://yimin-pan.github.io/hair-gs/ (Retrieved 2026-05-15)
[9] "CGHair: Compact Gaussian Hair Reconstruction with Card Clustering" / "Strands2Cards" SIGGRAPH Asia 2025. https://arxiv.org/abs/2604.03716 , https://dl.acm.org/doi/10.1145/3757377.3763864 (Retrieved 2026-05-15)
