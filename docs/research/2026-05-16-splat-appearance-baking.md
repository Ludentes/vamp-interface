---
status: live
topic: lam-chibi-recipe
supersedes: 2026-05-15-splats-to-mesh-conversion.md
---

# Research: Baking Gaussian-Splat Appearance onto a Mesh

**Date:** 2026-05-16
**Sources:** SuGaR (CVPR 2024), Gaussian Frosting (ECCV 2024), Texture-GS (ECCV 2024),
nvdiffrast (NVIDIA), KIRI Engine, 3DGS Render (Blender add-on), GSOPs (Houdini),
sam-3d-objects issue #75.

---

## Executive Summary

The Milestone-0 gate showed: splats→mesh **geometry** is clean (LAM's FLAME mesh
renders coherently), but **per-vertex splat color** (`f_dc`/`shs`) is blotchy —
splat color is alpha-blend color, not surface albedo. The accepted fix is not a
better per-splat color; it is **render-and-bake**: render the Gaussian splats
from many camera views, then fit a UV texture (or per-vertex color) by
differentiable rasterisation so the textured mesh's renders match the splat
renders. This is exactly what SuGaR's textured-mesh extraction does, in seconds,
with nvdiffrast. Our situation is *easier* than the literature's: every
splat→mesh paper must first reconstruct a mesh; we already own LAM's FLAME mesh
with a UV layout, so only the bake step remains.

## Key Findings

### Render-and-bake is the accepted method — SuGaR is the reference impl

SuGaR's `extract_refined_mesh_with_texture.py` (`--export_obj True` in the full
pipeline) computes a traditional color UV texture by rendering the scene from
the training viewpoints and baking the appearance into a UV atlas with
**nvdiffrast**, NVIDIA's differentiable rasteriser — "should just take a few
seconds." The texture is *optimised*, not copied: the UV texture is the variable,
the loss is photometric against the splat renders, nvdiffrast backprops to the
texels. This is the same primitive as "Joint UV Optimization and Texture Baking"
(ACM TOG 2023) and the standard nvdiffrast multi-view texture-fit recipe
(capture from many poses, differentiably fit a UV texture so renders match).

The crucial distinction from what v1 tried: v1 baked the *splat parameters*
(`shs`). The accepted method bakes the *splat renders* — the photoreal image the
overlapping Gaussians produce when alpha-composited. That image is what we want
on the mesh.

### Our problem is the easy half of SuGaR

Every generic splat→mesh method (SuGaR, Frosting, 2DGS, GOF) spends most of its
pipeline *reconstructing a surface* from an unstructured cloud. We skip all of
it: LAM's canonical representation already is a FLAME-topology mesh
(`shaped_mesh.obj` = `xyz − offset`, the clean 5023-vert / 9976-face base), and
FLAME ships a UV layout (`head_template_mesh.obj`, 5118 `vt`). What remains is
only SuGaR's last step — the nvdiffrast UV bake — run on a mesh we already have.

### Two bake variants, increasing cost

- **Direct projection bake (cheap, no optimisation).** The LAM splat renderer
  and the mesh share a coordinate frame. For each texel (or vertex), render the
  splats from N views, project the texel into each view, sample the rendered
  pixel, average with a visibility/normal-facing weight. No differentiable
  rendering needed. Lower quality at grazing angles and seams; a fine first cut.
- **nvdiffrast optimisation bake (SuGaR-grade).** Make the UV texture a
  learnable tensor; render the *mesh* with nvdiffrast from the same N views;
  L2 against the splat renders; optimise. Handles view parallax and seams
  better, still seconds. This is the target-quality method.

### Gotcha: linear vs sRGB colour space

sam-3d-objects issue #75 reports baked-mesh textures coming out washed-out and
desaturated versus the source splat — the bake renders observations in linear
colour space but saves them as sRGB without tone-mapping. LAM runs `gs_use_rgb`,
so its splat colour is already display-RGB-ish; whatever space the splat render
is read in must be the space the texture is written in. Verify on the first bake
by eye against a splat-render frame.

### Methods that do NOT fit, and why

- **Texture-GS, Neural Shell Texture Splatting, GStex, GTAvatar** — learn a UV
  texture *jointly during 3DGS training*. They need to train the splat model;
  our LAM checkpoint is frozen and pretrained. Not applicable post-hoc without
  retraining LAM. (Texture-GS remains the conceptual reference for *why* a UV
  texture is deformation-invariant — see prior doc.)
- **Gaussian Frosting** — keeps a hybrid mesh + bound Gaussian shell for
  rendering quality. The pivot's whole point is to *drop* the splat renderer, so
  Frosting re-imports the coupling we are escaping. Documented fallback only.
- **mesh2splat (EA)** — mesh → splat, the opposite direction.
- **KIRI Engine / 3DGS Render 3.0–4.0 / GSOPs / Polycam** — turnkey 3DGS→mesh
  with texture baking (3DGS Render auto-generates UV maps in Blender). Black-box
  and outside our Python/LAM pipeline, but confirm render-and-bake is the
  industry-standard operation and are a fallback if the in-pipeline bake stalls.

## Comparison

| Method | Needs mesh recon? | Needs 3DGS retrain? | Fits us | Note |
|---|---|---|---|---|
| SuGaR textured-mesh extract | yes (we skip) | no | **yes** | nvdiffrast UV bake — reference |
| Direct projection bake | no | no | **yes** | cheapest first cut |
| Texture-GS / GStex / Neural Shell | no | yes | no | joint-trained texture |
| Gaussian Frosting | yes | no | fallback | keeps splats |
| KIRI / 3DGS Render / GSOPs | yes | no | fallback | turnkey, out-of-pipeline |

## Open Questions

- **UV on the 20018 mesh.** FLAME's UV lives on the 5023+teeth base; LAM's
  subdivision does not carry `verts_uvs`. Bake on the 5023 base (UV exists) and
  treat 20018 as a render-time upsample, or derive subdivided UVs via
  `SubdivideMeshes` feature-carry. The 5023 base is the simpler v2 target.
- **How many views, which poses.** SuGaR uses training views; we author the
  camera set. A turntable ring + a few elevations should cover a head. Verify
  coverage on the back-of-head / under-chin.
- **Per-vertex vs UV texture.** 5023 verts at 256² output may be enough as
  per-vertex colour if the bake is done from renders (not `shs`). UV texture is
  strictly better and needed for v2 Blender hand-off. Measure both.

## Sources

[1] Guédon & Lepetit. "SuGaR." CVPR 2024. https://github.com/Anttwo/SuGaR ; https://arxiv.org/html/2311.12775v3 (Retrieved 2026-05-16)
[2] Guédon & Lepetit. "Gaussian Frosting." ECCV 2024. https://anttwo.github.io/frosting/ (Retrieved 2026-05-16)
[3] "Texture-GS." ECCV 2024. https://arxiv.org/html/2403.10050v1 (Retrieved 2026-05-16)
[4] NVlabs. "nvdiffrast." https://nvlabs.github.io/nvdiffrast/ (Retrieved 2026-05-16)
[5] "Joint UV Optimization and Texture Baking." ACM TOG 2023. https://dl.acm.org/doi/10.1145/3617683 (Retrieved 2026-05-16)
[6] facebookresearch/sam-3d-objects issue #75 — washed-out baked texture (linear/sRGB). https://github.com/facebookresearch/sam-3d-objects/issues/75 (Retrieved 2026-05-16)
[7] KIRI Engine. "What Is 3DGS To Mesh." https://www.kiriengine.app/blog/what-is-3dgs-to-mesh (Retrieved 2026-05-16)
[8] "3DGS Render 3.0." CG Channel. https://www.cgchannel.com/2025/03/3dgs-render-3-0-lets-you-paint-3d-gaussian-splats-in-blender/ (Retrieved 2026-05-16)
