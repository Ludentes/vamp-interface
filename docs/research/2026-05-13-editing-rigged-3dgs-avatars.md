---
status: live
topic: liveportrait-stylized
---

# Research: editing rigged / FLAME-bound 3DGS avatars

**Date:** 2026-05-13
**Sources:** 11 sources — LAM paper (arXiv 2502.17796), GaussianAvatars (CVPR 2024), SplattingAvatar README, GTAvatar (arXiv 2512.09162), Texture-GS (ECCV 2024), GaussianEditor (CVPR 2024), AvatarPerfect (arXiv 2412.15609), KIRI 3DGS Render 3.0 (CG Channel), TeGA (SIGGRAPH 2025), MeGA (CVPR 2025), SuperSplat repo.

---

## Executive Summary

There is **no off-the-shelf "open the .ply, paint, save, re-render" workflow** for FLAME-rigged Gaussian avatars whose canonical splats are stored as offsets from a parametric mesh (LAM, GaussianAvatars, SplattingAvatar). Every release in this family points to one of three accepted patterns: **edit in 2D and lift to 3D** (LAM's own recommendation [1], AvatarPerfect [7]), **train so splats are UV-texture-bound and edit the texture** (Texture-GS [3], GTAvatar [4], TeGA [9], MeGA [10]), or **delete-only cleanup in SuperSplat** (the documented role for editor tools on this avatar family [7]). Standard splat editors — SuperSplat, KIRI 3DGS Render — assume world-space, positive-scale, viewer-ready splats. They open mesh-bound offset-.plys, but the result is sub-pixel fog because per-splat scales are tuned for a runtime that adds back the FLAME mesh.

For LAM specifically, the paper itself (§4.3 "Stylize Editing of Animatable Gaussian Avatar") prescribes the 2D-edit-then-lift workflow [1]. No edit-the-.ply path is documented in the repo, in issues, or in the paper [11][2].

## Key Findings

### LAM's documented path is "edit the 2D render, lift to splats"

LAM §4.3 states: *"our framework can edit different styles of the 3D Gaussian avatar efficiently utilizing a 2D editing prior models to edit the avatar in the 2D image and then lift it to 3D Gaussian space"* [1]. The paper does not describe what the canonical .ply contains or whether it is meant to be viewer-readable [1]. The GitHub repo's README has no mention of `edit`, `color`, `blender`, `supersplat`, `texture`, or `appearance` workflows [11]. The closest export feature is "Avatar Export" for OpenAvatarChat — format conversion, not appearance editing [11].

The lift step is unspecified in §4.3 but the standard recipe in the literature is: render the canonical avatar to one or several views, edit those images (image-to-image diffusion, manual paint, ControlNet, etc.), then run a short optimization that fits the splat parameters to match the edited renders. This is the same pattern GaussianEditor, AvatarPerfect, and Instruct-GS2GS implement [6][7]. None of them edit the .ply directly.

### SuperSplat on a rigged avatar is delete-only

AvatarPerfect (Dec 2024) is the most concrete documentation we have on what SuperSplat can actually do to a rigged GaussianAvatars-style head, because they used SuperSplat as the baseline in their user study [7]. Quoting their findings:

- *"SuperSplat does not allow users to change the avatar's body pose"* [7]
- Available operations on a rigged avatar in SuperSplat: **Erase Center Tool** (removes Gaussians whose centers lie in selected regions) and **Erase Splat Tool** (removes overlapping Gaussians) [7]
- *"the system did not support recoloring or sculpting"* [7]
- *"Direct deletion can expose previously hidden anomalous-color Gaussians"* [7]

So SuperSplat is useful for one thing on this avatar family: pruning bad splats. That's it. The MIT-licensed local install we built (v2.25.1) reflects this — it's an editor, but for world-space scene splats, not for offset-encoded rigged avatars [11].

### KIRI 3DGS Render in Blender paints splats — but on world-space splats, not rigged ones

KIRI 3DGS Render 3.0 (March 2025) added the ability to *"paint directly onto 3DGS scans"* with solid colour or through an image texture, and *"automatically generates UV maps when importing 3DGS data"* [8]. This is the closest tool to "Blender enthusiast paints a splat head." However, the article (and follow-up v4 release notes) **describe no support for rigged or FLAME-bound splats** — the tool treats the .ply as a static world-space scene, applies its own UV unwrap to the displayed splat positions, and writes back per-splat colours [8].

Applied to a LAM canonical .ply where xyz holds *offsets-from-FLAME-vertices*, KIRI would unwrap the offset cloud, not the head, and any paint would land on a coordinate system the LAM runtime has no concept of. Limitations explicitly called out: Eevee-only rendering, Blender 4.2+, PLY-only [8]. Rigged-splat support not mentioned.

### Texture-space editing requires a model trained for it

The two papers that *do* offer a clean "open the texture, paint, done" workflow on parametric-mesh avatars — **Texture-GS** (ECCV 2024) [3] and **GTAvatar** (Dec 2025) [4] — both achieve it by training the avatar so that splat colour is parameterized by a 2D texture map.

Texture-GS adds a UV-mapping MLP that projects 3D Gaussian centres into 2D UV space and a learnable 2D texture; editing means modifying the texture, and changes propagate at render time [3]. *"Each 3D Gaussian often covers multiple pixels in practice, and mapping all pixels covered by a single Gaussian to the same UV location results in degenerated textures"* — Texture-GS solves this by allowing per-pixel UVs within a single splat's footprint [3]. The method *"assumes an opaque and smooth surface"* and *"does not address mesh-bound rigged avatars with parametric deformation"* explicitly [3].

GTAvatar is the directly-applicable analogue for FLAME rigs: *"enables intuitive editing of material and normals directly in the FLAME UV domain without requiring further optimization"* [4]. Supported edits: *"adding a star decal, changing hair, teeth or eye colors, removing skin imperfections or adding make-up"*, plus arbitrary PBR material maps [4]. **No retraining required at edit time** [4]. But — GTAvatar is a different reconstruction pipeline than LAM. Adopting it means retraining (or re-baking) avatars in their framework, not editing our LAM outputs. The paper doesn't release a "convert LAM .ply → GTAvatar UV bake" path [4].

TeGA (SIGGRAPH 2025) and MeGA (CVPR 2025) take similar texture-space routes — TeGA stores up to 4M Gaussians in a continuous UVD tangent space of the mesh, MeGA disentangles diffuse/dynamic/view-dependent texture components per pixel, *"natively supports hair alteration and texture editing"* [9][10]. Same constraint: different training pipeline, not a post-hoc upgrade for an existing LAM .ply.

### Text-driven editing (GaussianEditor / Instruct-GS2GS) is feasible but coarse

GaussianEditor (CVPR 2024) edits splats via text prompts: region-of-interest extraction → grounding-segmentation → diffusion-guided optimization on the splats inside the RoI, ~20 min on a V100 [6]. It works on arbitrary splat scenes, including rigged avatars in principle, because it does the editing *through* a 2D rendering loss. Output: a modified .ply with updated colours / geometry that still rasterises correctly.

For our specific case this is the most likely "Blender enthusiast doesn't need to learn 3DGS" path, but **(a)** it's an instruction-driven optimization, not interactive painting, and **(b)** the original GaussianEditor does not handle the FLAME-driven deformation gracefully — edits go onto canonical splats, but the underlying offset-to-FLAME-vertex correspondence isn't preserved across the edit, so animation may break.

### The reason our .ply looks like fog is structural

LAM's canonical save writes `xyz = offset` (displacement from FLAME canonical vertex), `f_dc = RGB` (not SH), scale in log space tuned for the runtime that re-derives effective scale per frame, opacity in logit space [11][1]. Mean log-scale −8 → 0.3 mm splat radius on a head of 20 cm extent → ~0.83 px at 512² framing. Mean opacity logit −2.5 → σ=0.08 transparency. Naïvely scaling either up turns the cloud into an undifferentiated sphere (we verified: scale+4 / opacity+4 produces a dense ball, not a head). **This is not a save bug — it's that the .ply stores model-internal parameters, not viewer-ready geometry.** Same shape will be true of any avatar where splats live in a per-mesh-vertex offset space (GaussianAvatars [2], SplattingAvatar [5], SuGaR-style surface-aligned splatting).

## Comparison: accepted ways to edit a rigged / FLAME-bound 3DGS avatar

| Pattern | Tooling | What user edits | Requires retraining? | Works on a vanilla LAM .ply? | Cited at |
|---|---|---|---|---|---|
| **2D-edit-then-lift** | Image editor + diffusion + short optimization | A rendered image | Short optimization (minutes) | Yes — this is LAM's own recipe | [1][7] |
| **Texture-space edit** | Blender/Photoshop/etc. on a 2D UV image | FLAME UV texture | Requires training in a UV-bound formulation (GTAvatar/TeGA/MeGA) | **No** — LAM doesn't expose a FLAME UV bake | [3][4][9][10] |
| **Text-driven (GaussianEditor)** | Prompt + RoI + diffusion-guided opt | Text instruction | Edit-time optimization (~20 min/V100) | Yes, but breaks the FLAME binding for animation | [6] |
| **Splat pruning** | SuperSplat / KIRI | Delete tool on canonical splats | No | Yes — but sub-pixel display makes selection hard; useful for invisible-splat cleanup, not for visible edits | [7][8] |
| **Direct paint on splats in Blender** | KIRI 3DGS Render 3.0+ | Paint / image-projection on splats | No | Theoretically yes; in practice no, because rigged-splat offset coordinates don't match what KIRI's UV-unwrap and runtime expect | [8] |
| **Convert to mesh first, edit mesh, re-bake** | 3DGS-to-Mesh (SuGaR / KIRI) + Blender + bake | Standard texture/sculpt on a mesh | No | Partially — we already have `_textured_mesh.obj` in our handoff; need a bake-back-to-splats script we don't have yet | [8] |

## Implications for our LAM handoff

The cleanest published path for our exact situation is **LAM §4.3**: render canonical to 2D, edit the image, lift back with a short splat optimization [1]. That's what the model authors recommend. It requires writing the lift step ourselves, which the paper doesn't release code for; it's a standard differentiable-rendering loop (~50-200 iterations of SGD on per-splat colour, optionally on opacity and position, against the edited image).

Concretely, "make the handoff actually editable" decomposes into three workable products with different effort/value trade-offs:

1. **Mesh-OBJ as the edit surface (low effort, low fidelity).** Ship only the `_textured_mesh.obj`. Colleague edits in Blender as a normal model. We write a bake script: edited per-vertex colour → nearest-neighbour onto LAM's canonical-splat indices → patched .ply. ~0.5 day on our side. Limitation: no per-splat granularity (multiple splats per FLAME vertex collapse to one colour).

2. **2D-render as the edit surface, lift on receipt (medium effort, high fidelity).** Ship 8-16 canonical views as PNGs. Colleague paints/inpaints them in Photoshop/Blender/Krita. We run a per-splat colour optimization (frozen geometry, frozen FLAME binding) to fit the edited views — Texture-GS-style but degenerate to texture=identity. ~1-2 days on our side, depending on how much we reuse LAM's training loss. This is the LAM-recommended path [1].

3. **GaussianEditor on top (medium effort, lower control).** Wire GaussianEditor [6] into our LAM canonical .ply pipeline. Colleague writes text prompts ("make the hair red", "turn the skin into porcelain") and the diffusion-guided optimizer modifies the splats. ~1 day to integrate. Risk: animation drift, because the optimization doesn't respect the FLAME offset binding.

We do **not** have a path that delivers "drag the .ply into SuperSplat and start painting." The literature confirms that path doesn't exist for this avatar family today; the closest formal solution (GTAvatar [4]) requires reconstructing the avatar in their framework instead of LAM's.

## Open Questions

- **Does LAM provide an internal canonical render path?** §4.3 implies the authors did this themselves; the released code under `lam/runners/infer/lam.py` only does motion-driven renders. Need to check whether the inference scripts can be coerced into a static-canonical render, or whether we have to write it from scratch.
- **Does GaussianEditor preserve LAM's offset-to-FLAME-vertex correspondence?** Likely not by default — its optimization writes to per-splat fields without knowledge of LAM's deformation model. Would need a constrained version that only modifies `f_dc_*` and `opacity`, leaving xyz/scale/rot frozen. Untested.
- **Cost of a per-splat colour-only optimization.** LAM has 20018 splats. Fitting `f_dc_*` only against 8 edited views is a 60k-parameter problem, trivially convex in colour space. Probably under a minute on RTX 5090. Worth a spike before committing to the 2D-edit-then-lift path.
- **What does AvatarPerfect's 2D image editing UI look like in practice?** It claims to address SuperSplat's pose/recolour limitations with a 2D-edit-and-lift workflow [7]. Code release status unverified.

## Sources

[1] Zhang et al. (2025). *LAM: Large Avatar Model for One-shot Animatable Gaussian Head*. SIGGRAPH 2025. https://arxiv.org/html/2502.17796 (Retrieved 2026-05-13). §4.3 quoted.

[2] Qian et al. (2024). *GaussianAvatars: Photorealistic Head Avatars with Rigged 3D Gaussians*. CVPR 2024. https://arxiv.org/html/2312.02069v2 (Retrieved 2026-05-13).

[3] Xu et al. (2024). *Texture-GS: Disentangling the Geometry and Texture for 3D Gaussian Splatting Editing*. ECCV 2024. https://arxiv.org/html/2403.10050v1 (Retrieved 2026-05-13).

[4] Anonymous (2026). *GTAvatar: Bridging Gaussian Splatting and Texture Mapping for Relightable and Editable Gaussian Avatars*. arXiv preprint. https://arxiv.org/html/2512.09162v1 (Retrieved 2026-05-13).

[5] initialneil. *SplattingAvatar: Realistic Real-Time Human Avatars with Mesh-Embedded Gaussian Splatting* (CVPR 2024). https://github.com/initialneil/SplattingAvatar (Retrieved 2026-05-13).

[6] Chen et al. (2024). *GaussianEditor: Swift and Controllable 3D Editing with Gaussian Splatting*. CVPR 2024. https://github.com/buaacyw/GaussianEditor (Retrieved 2026-05-13).

[7] Tojo et al. (2024). *AvatarPerfect: User-Assisted 3D Gaussian Splatting Avatar Refinement with Automatic Pose Suggestion*. arXiv preprint. https://arxiv.org/html/2412.15609 (Retrieved 2026-05-13). SuperSplat baseline analysis.

[8] CG Channel (2025-03). *3DGS Render 3.0 lets you paint 3D Gaussian Splats in Blender*. https://www.cgchannel.com/2025/03/3dgs-render-3-0-lets-you-paint-3d-gaussian-splats-in-blender/ (Retrieved 2026-05-13).

[9] Authors (2025). *TeGA: Texture Space Gaussian Avatars for High-Resolution Dynamic Head Modeling*. SIGGRAPH 2025. https://dl.acm.org/doi/10.1145/3721238.3730710 (Retrieved 2026-05-13).

[10] Authors (2025). *MeGA: Hybrid Mesh-Gaussian Head Avatar for High-Fidelity Rendering*. CVPR 2025. https://cg.cs.tsinghua.edu.cn/papers/CVPR-2025-MeGA.pdf (Retrieved 2026-05-13).

[11] aigc3d. *LAM official repository*. https://github.com/aigc3d/LAM (Retrieved 2026-05-13). README has no documented edit workflow.
