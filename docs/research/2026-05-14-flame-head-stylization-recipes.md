---
status: live
topic: flame-stylization
---

# Research: Community recipes for stylizing a FLAME-topology 3DGS head

**Date:** 2026-05-14
**Sources:** 17 sources (FLAME-Universe, ShapeKeyWrap, deformation_transfer_ARkit_blendshapes, Polywink, Goo Engine, GaussianEditor, StyleGaussian, StyleSplat, HeadStudio, GeoAvatar, Surface Deform docs, Wrap3D/Faceform docs, Polycount and Polywink writeups). Key URLs in Sources section.

---

## Executive Summary

**No automated "FLAME → chibi/anime/animal" recipe exists** — FLAME-Universe lists zero cartoon, anime, animal, furry, or non-human extensions of the model [1]. The community path is **assemble**, not pick-from-shelf: wrap a stylized base mesh onto FLAME topology with a Blender modifier or deformation-transfer script, then either repaint the SH-DC vertex colours or bake a stylized rendering pass back in. The blendshape-rescale problem is solved by `deformation_transfer_ARkit_blendshapes` [2] (free) or Polywink Blendshapes-on-Demand [3] (paid service, ~24h). For full-scene restyle, **StyleGaussian** [4] and **StyleSplat** [5] work on 3DGS but do not preserve animation drivers; **GaussianEditor** [6] supports text-prompt edits in 2–7 min but needs COLMAP scene data, not a single PLY. Easiest wins ranked: ghost (minutes), cel-shaded human (hours), chibi-proportioned human (one day), anime line-art (multi-day), furry/animal (multi-day, no shortcut), human-to-fridge (out of reach without abandoning FLAME).

---

## Key Findings

### No published FLAME extension covers chibi, anime, animal, or non-human

A direct read of the FLAME-Universe index — the canonical curated list of FLAME-derived projects maintained by FLAME's author Timo Bolkart — shows the entire ecosystem targets photorealistic human heads, with no caricature, AnimalFLAME, FLAMEinStyle, or similar branch [1]. The closest adjacency is DiffPoseTalk for *animation* style transfer, which is unrelated to the geometric/aesthetic problem [1]. This is a real gap, not a search miss: searches for "FLAMEinStyle" and "AnimalFLAME" return no academic hits. Any chibi/anime/furry result must therefore be built by composing generic tools.

### Topology transfer is the load-bearing recipe

The accepted community pattern when you have a source mesh with working blendshapes and a target mesh with the desired aesthetic is **deformation transfer plus shape-key wrap** [2][7][8]. Two free implementations dominate. `deformation_transfer_ARkit_blendshapes` [2] is a Python tool (numpy/scipy/numba dependencies) that takes a source mesh with the 52 ARKit blendshapes and a topologically-corresponded target mesh, and produces all 52 blendshapes on the target. Its README explicitly assumes pre-alignment via Wrap3D — a commercial tool — but the alignment can be substituted with Blender's free Surface Deform modifier [8] or with the MIT-licensed ShapeKeyWrap addon [7], which wraps Surface Deform into a shape-key-transfer workflow. ShapeKeyWrap shipped v0.5.0 on March 1, 2025 with 57 commits and is actively maintained [7]; the deformation-transfer repo has 14 commits with unclear recent activity [2].

The Surface Deform modifier itself is the cheap path when the target is a moderate FLAME deformation rather than an entirely different mesh: bind the deformed FLAME to the original FLAME, and shape keys flow through automatically. Blender's docs warn that bind quality degrades "the further a mesh deviates from the target mesh surface" [8], which means chibi-scale changes are at the edge of what Surface Deform handles cleanly.

### Polywink is the paid escape hatch

Polywink's Blendshapes-on-Demand generates 157 FACS blendshapes (a superset of ARKit-52) on any 3D character, including "scanned head, photorealistic human model, or cartoonish character" [3]. Delivery is FBX format in under 24 hours [3]. Per-character pricing is not published on the listing pages I retrieved [3][9] and the search results did not surface a current figure — an enquiry would be needed before estimating cost-vs-artist-time. This is the highest-leverage option if artist time is the bottleneck and the character is custom-authored.

### 3DGS post-hoc stylization tools exist but don't fit our animation pipeline

Three tools cover post-hoc Gaussian stylization at varying granularity. **GaussianEditor** [6] supports text-prompt edits to 3DGS scenes in 2–7 minutes per edit, but the README specifies inputs are "pretrained Gaussians and COLMAP outputs" — scene reconstructions, not isolated avatars — and its InstructPix2Pix backbone "only works on limited prompts." Linux/CUDA tested on Ubuntu 22 + RTX 3090/A5000/A6000 [6], so RTX 5090 should be compatible. **StyleGaussian** [4] does instant 3D style transfer from one reference image at 10 fps post-training, but each scene needs a multi-step training pass (reconstruction → feature embedding → style transfer) and the documentation makes no mention of blendshape or pose preservation [4] — the stylized splats are not the LAM splats, so the FLAME rig won't drive them. **StyleSplat** [5] localizes style transfer to specific segmented objects, useful for "stylize the hair only" but still doesn't preserve animation drivers.

The honest verdict: these tools restyle the *appearance* of a static 3DGS reconstruction, not the *animatable* SH-DC of a FLAME-rigged LAM head. You could in principle restyle a LAM head with StyleSplat and then bake the resulting SH-DC back into a new LAM-format PLY, but this is research-grade plumbing, not a recipe.

### Goo Engine is the practical anime-aesthetic path

For NPR / anime / cel-shaded look on any Blender mesh — including a deformed FLAME — Goo Engine is the maintained community fork [10][11]. It adds four Eevee shader nodes and curvature/light-group features tuned for cel shading [10], with NPR vertex-colour authoring patterns where "red-painted areas will be white no matter what while blue painted areas will be black no matter what" [10] — exactly the toolkit needed to bake stylized colours into our SH-DC channel. Distribution: free as open-source fork on GitHub, with pre-built Windows/macOS binaries behind a £4.50/month Patreon [11]; Linux users must build from source. CG Channel notes Blender itself has experimental NPR builds as of late 2024 [12], which may eventually fold these features in.

### Animal/furry FLAME has no community shortcut

Searches for animal-head FLAME adaptations return zero hits. The closest 3DMM analogues are entirely separate models (SMAL for animals, BARC for dogs) with no shared topology with FLAME. The viable furry path is to author or purchase a furry head mesh (BlenderKit, CGTrader, Daz Genesis fursona morph packs), then run the topology-transfer recipe above to inherit ARKit-52 — but the topology gap from human FLAME to canine skull is large enough that Surface Deform binds will be poor, and the deformation-transfer step risks expression corruption around the muzzle. **There is no easy win here.**

### Ghost / translucent is trivial

This is the one stylization that maps directly to existing LAM splat fields: lower the opacity logits on a vertex band, tint the SH-DC blue, and the head reads as a ghost. No topology change, no blendshape work, no external tool needed beyond our existing vertex-paint round-trip. **Minutes.**

### Chibi-human is the most interesting bet

Chibi proportions on a recognisable human face — big head, big eyes, small chin — sit in a sweet spot:

1. **Geometry** (Surface Deform from FLAME-neutral to a chibi-proportioned copy, or per-region scale): an afternoon's Blender work, no new tools.
2. **Blendshape rescale** (deformation_transfer_ARkit_blendshapes [2] or per-region scale of `flame_arkit_bs.npy` rows): half a day to a day.
3. **Appearance** (vertex paint cel-shaded skin on the SH2RGB-baked OBJ via the round-trip we already have, plus Goo Engine offline render to validate aesthetic): a day.
4. **Hair** (aux-splats for anime spikes/volumes, the path we proved with horns): variable depending on style.

Total: roughly two artist-days per character once the first one is wired up. **This is the recipe most likely to give a "wow" result with proportional effort.**

---

## Recipe matrix

Effort assumes the round-trip pipeline (`lam_bake_display_rgb_obj.py`, aux-splat injection hook) is already in place — i.e. only the new style work counts.

| Style | Tool stack | Works on FLAME? | Effort | Quality ceiling | Notes |
|---|---|---|---|---|---|
| **Ghost / translucent** | Existing vertex-paint + opacity edit | Native | **Minutes** | High | Direct map to SH-DC + opacity logit |
| **Recolour (hair, skin tone)** | Existing vertex-paint round-trip | Native | **Minutes** | High | Shipped 2026-05-13 |
| **Cel-shaded human (Goo render bake)** | Goo Engine [10][11] offline render → SH-DC bake | Yes — it's Blender on the deformed mesh | **Hours** | High for static lighting; SH-DC degree 0 means no view-dep | One-time per character |
| **Chibi-proportions human** | Surface Deform [8] + per-region blendshape rescale or `deformation_transfer_ARkit_blendshapes` [2] + cel-shade bake | Yes via wrap | **~1 day** | High if portrait-derived texture is restyled; uncanny if left photo | Aux-splats for anime hair add ~hours |
| **Anime line-art / full toon** | Chibi pipeline + manual lineart vertex-paint OR Goo Engine line modifier + bake | Yes via wrap | **Multi-day** | Medium — SH-DC degree 0 fights you on the line consistency | Goo Engine paywall is £4.50/mo or build-from-source |
| **Cosmetic appendages (horns, halo, antennae)** | Existing aux-splat injection | Native | **Hours per appendage** | High geometry, rigid attachment limit | Shipped 2026-05-13 |
| **Furry / animal head** | Author/buy furry mesh + Surface Deform [8] or NRICP + deformation transfer [2] + Polywink fallback [3] | **Marginal** — large topology gap from human FLAME | **Multi-day** + risk of expression corruption around muzzle | Variable | No FLAME-native option exists |
| **Non-human prop ("fridge")** | Abandon FLAME; reconstruct prop as separate 3DGS scene; drive separately | **No** — out of FLAME basis | **Days** + custom rig | N/A | Outside the LAM pipeline entirely |
| **Global style transfer (Ghibli-painterly look)** | StyleGaussian [4] or StyleSplat [5] post-bake → manual SH-DC re-import | Lossy — animation drivers don't survive | **Days** of research plumbing | Unverified | Not a recipe; a research path |
| **Text-prompt 3DGS edit ("anime")** | GaussianEditor [6] | Needs COLMAP wrap of single avatar | **Hours** of plumbing per edit | Unverified — InstructPix2Pix prompt-limited | Worth a spike but not a recipe |
| **Custom-authored stylized character with ARKit** | Author in Blender/ZBrush + Polywink Blendshapes-on-Demand [3] | Polywink-routed, not FLAME-routed | **Outsourced 24h** + author time | High | Best path when artist time is the bottleneck |

---

## The chibi spike, concretely

The recipe that the evidence above most supports as the next high-leverage experiment:

1. Duplicate the FLAME-canonical OBJ for one anchor (e.g. `asian_m`).
2. In Blender, apply per-region scale: eye region ×2.3, lower-face ×0.65, skull ×1.25. Use proportional editing with falloff to keep transitions smooth.
3. Save as `chibi_target.obj`. Bind the original FLAME-canonical to it via Surface Deform [8] to validate the deformation flows cleanly.
4. **Blendshape rescale** — two paths:
   - Quick: per-region scale the corresponding rows of `flame_arkit_bs.npy` (e.g. eye-blink rows scaled by 2.3 in the local eye frame). Half day.
   - Robust: feed source FLAME with ARKit-52 + chibi target into `deformation_transfer_ARkit_blendshapes` [2], pre-aligning with Surface Deform [8] or ShapeKeyWrap [7] in place of Wrap3D. One day.
5. Vertex-paint cel-shaded skin onto the SH2RGB-baked OBJ. Use Goo Engine's reference vertex-colour convention (red=lit, blue=shaded) [10] as guidance even if rendering with our existing Eevee bake.
6. Add aux-splat anime hair (~8 chains, anisotropic, rigid-attached to scalp — our shipped tool).
7. Render with the chibi blendshape basis swapped in. Drive with ARKit. Side-by-side vs the original anchor.

Pass criteria: (i) eyes visibly close on `eyeBlinkLeft`, (ii) jaw open doesn't tear the cheek, (iii) result reads as "stylized character" not "warped photo" in a one-second glance.

---

## Open questions

- Does swapping `flame_arkit_bs.npy` for a deformation-transferred chibi version inside LAM's runtime actually work, or does the GS net's identity head balk at the new vertex positions? Untested. The 2-line patch path (memory: lam_arkit_spike_resolved) reads the file at load time, so substitution should be transparent, but the GS attribute network was trained on real-face proportions — there's a non-zero risk of artifacts at extreme deformations.
- How does the SH-DC degree-0 limit interact with cel shading? Cel shaders rely on view-dependent banding; LAM-20K bakes one DC colour per splat. The bake-once recipe captures **one** lighting-pose's cel look — when the head rotates, the cel bands don't move. This may be acceptable (it's a stylistic choice readable as "drawn"), but the failure mode is unverified.
- Polywink pricing for a single chibi character — not published; an enquiry would resolve whether the outsourced path is competitive with the artist-day estimate above.
- Can `StyleGaussian` [4] or `StyleSplat` [5] be hooked into LAM's PLY format with the FLAME rig preserved? No published path; would require writing a custom SH-DC re-import. Tagged as a research thread, not a recipe.

---

## Sources

[1] Bolkart, T. "FLAME-Universe." https://github.com/TimoBolkart/FLAME-Universe (Retrieved: 2026-05-14)
[2] Katr, V. "deformation_transfer_ARkit_blendshapes." https://github.com/vasiliskatr/deformation_transfer_ARkit_blendshapes (Retrieved: 2026-05-14)
[3] Polywink. "Blendshapes on Demand - Automatically Generated Blend Shapes." https://polywink.com/en/9-automatic-expressions-blendshapes-on-demand.html (Retrieved: 2026-05-14)
[4] Liu, K. et al. "StyleGaussian: Instant 3D Style Transfer with Gaussian Splatting." SIGGRAPH Asia 2024. https://kunhao-liu.github.io/StyleGaussian/ and https://github.com/Kunhao-Liu/StyleGaussian (Retrieved: 2026-05-14)
[5] Jain, S. et al. "StyleSplat: 3D Object Style Transfer with Gaussian Splatting." arXiv:2407.09473. https://bernard0047.github.io/stylesplat/ (Retrieved: 2026-05-14)
[6] Chen, Y. et al. "GaussianEditor: Swift and Controllable 3D Editing with Gaussian Splatting." CVPR 2024. https://github.com/buaacyw/GaussianEditor (Retrieved: 2026-05-14)
[7] Petrenko, M. "ShapeKeyWrap." https://github.com/MykytaPetrenko/ShapeKeyWrap (Retrieved: 2026-05-14) — v0.5.0 released 2025-03-01
[8] Blender Foundation. "Surface Deform Modifier — Blender Manual." https://docs.blender.org/manual/en/dev/modeling/modifiers/deform/surface_deform.html (Retrieved: 2026-05-14)
[9] CG Channel. "Polywink creates blendshapes and rigs for any head model." 2018. https://www.cgchannel.com/2018/07/polywink-creates-blendshapes-and-rigs-for-any-head-model/ (Retrieved: 2026-05-14)
[10] CG Channel. "Check out Goo Engine: Blender for 3D anime." 2023. https://www.cgchannel.com/2023/07/check-out-goo-engine-blender-for-anime/ (Retrieved: 2026-05-14)
[11] DillonGoo Studios. "goo-engine." https://github.com/dillongoostudios/goo-engine (Retrieved: 2026-05-14)
[12] CG Channel. "Check out Blender's experimental anime rendering builds." 2024. https://www.cgchannel.com/2024/12/check-out-blenders-experimental-non-photorealistic-rendering-build/ (Retrieved: 2026-05-14)
[13] Wang, J. et al. "Fully Automatic Blendshape Generation for Stylized Characters." SJTU CharacterLab. https://sjtu-characterlab.github.io/files/Fully_Automatic_Blendshape_Generation_for_Stylized_Characters.pdf (Retrieved: 2026-05-14)
[14] Filmic Worlds. "Solving Blendshapes for ARKit." https://filmicworlds.com/blog/solving-face-scans-for-arkit/ (Retrieved: 2026-05-14)
[15] Qian, S. et al. "GaussianAvatars: Photorealistic Head Avatars with Rigged 3D Gaussians." arXiv:2312.02069. https://arxiv.org/html/2312.02069v2 (Retrieved: 2026-05-14)
[16] Zhou, Z. et al. "HeadStudio: Text to Animatable Head Avatars with 3D Gaussian Splatting." ECCV 2024. https://github.com/ZhenglinZhou/HeadStudio (Retrieved: 2026-05-14)
[17] Faceform. "BlendWrapping — Faceform Wrap documentation." https://docs.faceform.com/Wrap/Nodes/BlendWrapping/BlendWrapping.html (Retrieved: 2026-05-14)
