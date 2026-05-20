---
status: live
topic: lam-chibi-recipe
---

# Image → UV texture atlas for a fixed rigged head mesh — solution survey

**Date:** 2026-05-20
**Sources:** 18 papers + repos. Key sources: SOAP (SIGGRAPH 2025), TEXGen (NeurIPS 2024), Portrait3D / ID-Sculpt (2024), Make-A-Texture (Meta 2024), AUV-Net (CVPR 2022), Portrait3D, FF HQ-UV, MeGA (CVPR 2025), TF_FLAME, FaceScape.

---

## Executive summary

For "single portrait → UV texture for a fixed rigged head mesh", the literature splits cleanly into three families: **(1) projective/optimisation bakes from a fitted parametric face** (TF_FLAME, FaceScape, AvatarMe, FFHQ-UV) — photoreal-only, can't handle our chibi inputs; **(2) view-space diffusion + back-projection** (Portrait3D, Make-A-Texture, SOAP); and **(3) UV-space diffusion directly** (TEXGen, Point-UV Diffusion, TEXGen). The closest fit to our problem — *stylized portrait → FLAME-topology rigged mesh with UV texture, MIT-licensed, runnable today* — is **SOAP (Style-Omniscient Animatable Portraits, SIGGRAPH 2025) [1][2]**. It is essentially our v2 target with the multi-view diffusion and FLAME fit-and-rig already trained. The two real questions for adopting it: (a) does its FLAME output topology compose with our Koban chibi mesh, or does it replace it; (b) is the multi-view diffusion model's stylized output good enough on the bearded/middle-aged anchors we struggle with.

The fast alternative — **TEXGen [3]** — solves the *texture-only* problem directly (10 s on A100, arbitrary mesh topology, MIT-style release) and would let us keep the Koban chibi mesh exactly as-is, taking (mesh + one front portrait) → UV atlas. This is a strict drop-in for the v2 spec's "direct UV bake" step and is the lowest-risk path.

The recommendation is to validate TEXGen as a drop-in replacement for our hand-rolled bake first (lower switching cost, keeps Koban), and only if it fails on stylized portraits, swap the whole frontend to SOAP (which then also replaces the chibi mesh asset itself).

---

## Key findings

### SOAP — strongest end-to-end match, SIGGRAPH 2025

SOAP is "a style-omniscient framework to generate rigged, topology-consistent avatars from any portrait, leveraging a multiview diffusion model trained on 24K 3D heads with multiple styles and an adaptive optimization pipeline to deform the FLAME mesh while maintaining topology and rigging via differentiable rendering" [1]. Input: one image, realistic or cartoon. Output: a FLAME-topology mesh (~20K faces) with consistent UV texture, eyes, teeth, and skinning weights, animatable via FACS expressions. The pipeline is (image → Unique3D-initialised multi-view diffusion → multi-view renders → adaptive FLAME fit-and-rig via differentiable rendering → mesh + UV).

Code is at `github.com/TingtingLiao/soap` under MIT license [2]; weights at `huggingface.co/Luffuly/head-mvimage-diffuser`. Three inference scripts ship: `process.py` (preprocess), `main.py` (reconstruct), `gui.py` (interactive). Setup requires CUDA 11.8 or 12.1 and a Python 3.10 conda env. Wall-clock is ~6 minutes per portrait on what the project does not specify but appears to be a single high-end GPU [1]. The project page demonstrates results on cartoon, anime, and realistic portraits — the explicit design target is broad style coverage, which is exactly our v1 failure mode.

The architectural caveat for us: SOAP outputs *FLAME* topology. Our chibi pipeline targets the *Koban* chibi mesh (5662 verts, custom rig). If we adopt SOAP wholesale, we replace the Koban asset with SOAP's FLAME output — a real strategic shift, not a drop-in. The FLAME output has its own ARKit-blendshape correspondence (via the existing FLAME→ARKit linear basis we already use in `arkit_landmark_spike`), so we don't lose ARKit driving; we'd lose the chibi *proportions* baked into the Koban mesh, but SOAP's multi-view diffusion handles chibi/cartoon styles natively, so the chibi proportions may emerge from the diffusion output rather than from a fixed base mesh.

### TEXGen — texture-only, drop-in compatible with our existing mesh

TEXGen [3] is a 700M-parameter diffusion model that "can generate UV texture maps guided by text prompts and single-view images, and supports applications including text-guided texture inpainting, sparse-view texture completion, and text-driven texture synthesis" [3]. Critically: it takes (mesh + single image) → 1024×1024 UV atlas in under 10 seconds on a single A100, works on *arbitrary* meshes (uses xAtlas to re-unfold UVs as a preprocessing step), and the diffusion is done directly in UV space using a hybrid 2D-conv + 3D point-cloud architecture so the output respects the underlying 3D topology rather than relying on inverse-projection from view space [3].

Code at `github.com/CVMI-Lab/TEXGen`. This is the lowest-cost path for us: feed in the Koban chibi mesh + the chibi portrait we already generate, get a UV texture atlas in 10 s, skip the entire TPS-warp + projective-bake stack we shipped (and broke) in v1. The risks are: (a) the model was trained primarily on photoreal Objaverse-style geometry, not chibi heads — does it adapt to our anchor's chibi style when fed a chibi portrait? (b) does it handle the head-topology-specific UV layout of Koban or want a re-unwrap?

### Portrait3D / ID-Sculpt — overkill, wrong topology, too slow

Portrait3D [4] takes a single in-the-wild portrait and produces "a textured 3D mesh representation" using deformable tetrahedral grids (DMTet) at 512 resolution with a 1024×1024 UV texture map. The texture stage uses *progressive texture inpainting* — render the partially-textured mesh from sequential camera positions, back-project generated imagery, fill uncoloured UV regions front-to-back [4]. It supports stylization "with stylized diffusion models," but takes ~1 hour per head on a V100 and outputs a DMTet topology that is not FLAME and not riggable for ARKit blendshapes. Wrong shape for our problem.

### Make-A-Texture — fast but text-only

Make-A-Texture (Meta, 2024) [5] produces 1024×1024 UV textures from (mesh + text prompt) in 3.07 seconds on H100. The catch: no image-identity input. Text prompts only. So it solves the texturing-a-mesh problem but cannot encode a specific person's identity. Useful only if we treat the portrait stage as upstream and use Make-A-Texture for a default chibi skin tone bake — which is much less than what we need.

### Classical projective bakes (TF_FLAME, FaceScape, FFHQ-UV, AvatarMe, AlbedoMM, GANFit, PR3D)

The classical family fits a parametric 3D face (FLAME or BFM) to landmarks in the input image, then projects the image onto the fitted mesh to bake a UV [6][7][8][9]. TF_FLAME's `build_texture_from_image.py` demo is exactly this [6]. FFHQ-UV is the standard high-quality variant — it provides a normalised facial UV-texture dataset and a fitting pipeline that produces full head albedo UV maps from images. AvatarMe++, StyleUV, StyleFaceUV, UV-GAN, AlbedoMM, and GANFit are all variants of "fit 3DMM + bake UV" [9]. All are photoreal-only by construction (the textures they generate are tuned to human skin) and the FLAME licence is non-commercial. The whole family is what our v1 reinvented (badly): pre-trained versions exist but require photoreal inputs and won't accept a chibi portrait.

### View-to-UV neural-prediction (ROME, MeGA, AUV-Net)

ROME (ECCV 2022) [10] takes a single photograph and "estimates a person-specific head mesh and the associated neural texture, which encodes both local photometric and geometric details." It uses a *neural* texture (a feature volume rendered by a learned neural renderer) rather than an RGB UV atlas, so it doesn't compose with a standard rasterizer that expects an RGB texture. Cross-person reenactment works; stylized portrait support is not documented in the abstract.

MeGA (CVPR 2025) [11] uses an enhanced FLAME mesh and predicts a *UV displacement map* + texture from views. It is multi-view-based (not single-image) and aimed at high-fidelity photoreal rendering. Not a fit for single-portrait→texture.

AUV-Net (CVPR 2022) [12] is closer to AUV-Net learns to embed 3D surfaces into "a 2D aligned UV space, by mapping the corresponding semantic parts of different 3D shapes to the same location." It produces aligned UVs *across shapes* so cross-domain texture transfer becomes possible. License/code release not specified clearly. The technique would let us re-target a photo-derived texture onto a chibi mesh by aligning their UVs, but it's an additional pre-processing step rather than an end-to-end solution.

### Diffusion in UV space (TEXGen [3], Point-UV Diffusion [13], TexFusion [14], FlexPainter)

Point-UV Diffusion [13] is the architectural predecessor of TEXGen: two-stage coarse-to-fine, point diffusion → UV diffusion. TexFusion [14] (NVIDIA) takes text + mesh, denoises in UV space via fused multi-view sampling. FlexPainter (2025) generalises multi-view consistency across more conditioning inputs. None of these are designed for *identity preservation from a portrait* — they're text-and-mesh systems. TEXGen is the one with explicit single-image conditioning that we'd want.

### Differentiable-render inverse optimisation (ROSA, generic nvdiffrast/pytorch3d)

For completeness: ROSA [15] and the generic "optimise UV texture pixels by backpropagating photometric loss through a differentiable renderer" approach is always available. It needs multi-view supervision or a strong prior; with a single portrait and a head shape that doesn't match the portrait (chibi vs photo of a real person), the optimisation has no constraint on the back/sides of the head — same single-frontal-view limitation as our v1.

---

## Comparison

| Method | Input | Output | Topology | Style | Code/license | Time | Single-image identity | Fit for our problem |
|--------|-------|--------|----------|-------|--------------|------|----------------------|---------------------|
| **SOAP** [1][2] | 1 image, any style | FLAME mesh + UV + rig | FLAME (~20K faces) | Realistic + cartoon (24K-avatar trained) | MIT, weights public | ~6 min | Yes (the design goal) | ★★★ end-to-end replacement (replaces Koban with FLAME) |
| **TEXGen** [3] | mesh + 1 image (+ optional text) | 1024² UV atlas | Arbitrary | Trained on photoreal; stylized untested | github CVMI-Lab/TEXGen | <10 s/A100 | Partial (image-conditioned) | ★★★ drop-in for bake stage, keeps Koban |
| Portrait3D / ID-Sculpt [4] | 1 in-the-wild portrait | DMTet mesh + 1024² UV | DMTet (~512 res) | Photoreal; stylized claimed | Built on threestudio; release unclear | ~1 h/V100 | Yes | Too slow, wrong topology |
| Make-A-Texture [5] | mesh + text prompt | 1024² UV | Arbitrary | Limited | Meta, release unclear | 3 s/H100 | No (text only) | Wrong I/O |
| Point-UV Diffusion [13] | mesh + text | UV atlas | Arbitrary | Generic | Open | — | No | Wrong I/O |
| TexFusion [14] | mesh + text | UV atlas | Arbitrary | Generic | NVIDIA research | Slower than TEXGen | No | Wrong I/O |
| TF_FLAME projective [6] | 1 photo + landmarks | FLAME mesh + UV | FLAME | Photoreal only | CC-BY non-commercial | seconds | Yes for photoreal | Won't accept chibi input |
| FFHQ-UV [7] | 1 photo | Full-head albedo UV | FLAME | Photoreal only | Open | minutes | Yes | Wrong style |
| ROME [10] | 1 photo | mesh + neural texture | FLAME-like | Photoreal | Samsung Labs release | Inference fast | Yes | Neural-texture, not RGB UV |
| MeGA [11] | multi-view | FLAME + UV displacement | FLAME | Photoreal | CVPR 2025 | — | Multi-view, not single | Wrong input |
| AUV-Net [12] | mesh | aligned UV space | Arbitrary | — | NV-tlabs | — | Cross-domain transfer | Aux tool, not a solution |
| AvatarMe++ / AlbedoMM / GANFit [8] | 1 photo | FLAME UV (albedo) | FLAME | Photoreal | CC-BY-NC | minutes | Yes for photoreal | Wrong style |
| Inverse-render optimisation | 1+ images | optimised UV | Arbitrary | Generic | — | minutes | Yes with priors | Same limitation as v1 |

---

## TEXGen descendants (added 2026-05-20)

**UniTEX (arXiv 2505.23253, May 2025) [19].** Two stages: (1) two fine-tuned **Flux DiTs** generate 6 orthographic views at 512² conditioned on surface normals + canonical-coord-maps (first generates shaded views; second delights and produces diffuse); (2) **Large Texturing Model** predicts a **Texture Function** — continuous 3D-space color map indexed by closest-surface-point projection, topology-invariant, *escaping TEXGen's mandatory xAtlas re-unwrap*. Beats TEXGen on PSNR-UV (23.01 vs 20.47), CLIP-MMD (0.826 vs 1.196), user preference (65.9% vs 6.8%). Key win: handles non-canonical/fragmented UV layouts where UV-space methods fail. Has cartoon-style demos but no quantitative stylized eval, no face/portrait experiments. Code "will be" at `github.com/YixunLiang/UniTEX`; license unstated; inference time/GPU memory not reported.

**SeqTex (arXiv 2507.04285, SIGGRAPH Asia 2025) [20].** Reformulates texture generation as **sequence generation of multi-view renderings + UV texture jointly**, leveraging video-diffusion priors. Decoupled multi-view/UV branches with geometry-informed cross-attention; adaptive token resolution to preserve fine texture details. Claims SOTA on both image- and text-conditioned 3D texture generation. From VAST-AI-Research. Code at `github.com/VAST-AI-Research/SeqTex`. No face-specific evaluation in the paper.

**Field trajectory.** Post-TEXGen, the SOTA moved away from "UV is the latent" toward either (a) **topology-invariant 3D Texture Functions** (UniTEX) or (b) **video-diffusion priors** (SeqTex). Both are still object-centric general texturing, not face-specialized. None of TEXGen, UniTEX, or SeqTex publishes quantitative face/portrait benchmarks.

## Recommendation for v2 architecture

Two viable paths, lowest-risk first.

**Path A (drop-in, keep Koban): TEXGen as the bake stage.** Replace `koban_bake.bake_portrait_to_uv` + `register.fit_tps`/`warp_image` with `TEXGen.generate(koban_mesh, chibi_portrait)`. Keep the portrait generation stage (Flux + PuLID + chibi-highstr-sweep recipe). Keep the Koban mesh asset and the existing ARKit driving. Switching cost: install one repo, swap one function. If it works, we're done in a day. If it fails on stylized portraits (likely risk: model was trained on photoreal Objaverse-class geometry), fall back to Path B. *This is the recommended first attempt.*

**Path B (end-to-end replace, kill the chibi mesh asset): SOAP as the whole frontend.** Feed the anchor's photo (the *photo*, not the chibi portrait — SOAP does the stylization itself) into SOAP. Take SOAP's FLAME-topology mesh + UV texture as the avatar. Wire SOAP's FLAME output into our existing ARKit→FLAME driving (which we already have for LAM). Net effect: the entire chibi-mesh-pivot we've been building becomes SOAP's responsibility, and the v1 work in `chibi_portrait.py`, `koban_*`, `register.py` is retired. Switching cost: bigger; we'd lose the Koban chibi *proportions* control and trust SOAP's multi-view diffusion to produce chibi-styled views. The upside: this is the closest match in the literature, MIT-licensed, with released weights, and tested explicitly on cartoon styles which is exactly our failure mode.

**Decision rule.** Try Path A first. If TEXGen on (Koban mesh + a chibi portrait) produces an atlas where features land in the right UV islands and skin tone is preserved, ship Path A. If TEXGen distorts or produces photoreal-looking textures that fight the chibi mesh proportions, jump to Path B.

The v2 spec I wrote earlier today (`docs/superpowers/specs/2026-05-20-chibi-identity-texture-v2-design.md`) described a hand-rolled "render-to-UV-map → direct sample" architecture. That's still a valid v3 fallback if both Path A and Path B fail, but it's strictly inferior to TEXGen (less general, no learned prior for occluded regions, no inpainting) and to SOAP (no multi-view consistency, no rig). **Recommend retiring the hand-rolled architecture in favour of Path A first attempt.**

---

## Open questions

- **TEXGen on chibi-style input.** TEXGen's paper demonstrates photoreal Objaverse textures; we have no evidence it handles stylized portraits cleanly. One-day spike: install, run on (koban_mesh, id_14_chibi_portrait), look at the atlas.
- **SOAP cartoon-style range.** The project page shows cartoon results but doesn't quantify how far the style range extends. Need to verify on our actual id_14 (bearded white guy) and id_08 anchors.
- **SOAP's FLAME output and our ARKit driving.** Our existing ARKit→FLAME basis works for LAM. SOAP outputs FLAME — should compose cleanly, but the rig weights and FACS basis may differ. One-day spike: drive a SOAP avatar with ARKit-52 and see if expressions read.
- **Licence audit.** SOAP repo claims MIT; SOAP's *training data* includes 24K 3D heads — not our concern at inference time but worth confirming for the inference weights specifically.
- **Path A + B parallel evaluation.** A 2-day spike that runs both on the same 3 anchors (id_14, id_08, id_03) and produces a side-by-side comparison is the strongest decision input. Cost is bounded; we should do that before locking in a v2 architecture.

---

## Sources

[1] Liao, T. et al. "SOAP: Style-Omniscient Animatable Portraits." arXiv:2505.05022, SIGGRAPH 2025. https://arxiv.org/abs/2505.05022 (Retrieved 2026-05-20)
[2] SOAP project page and repo. https://tingtingliao.github.io/soap/ and https://github.com/TingtingLiao/soap (Retrieved 2026-05-20)
[3] Yu, X. et al. "TEXGen: a Generative Diffusion Model for Mesh Textures." NeurIPS 2024, arXiv:2411.14740. https://arxiv.org/html/2411.14740v1 and https://github.com/CVMI-Lab/TEXGen (Retrieved 2026-05-20)
[4] Wu, J. et al. "Portrait3D / ID-Sculpt: ID-aware 3D Head Generation from Single In-the-wild Portrait Image." arXiv:2406.16710. https://arxiv.org/html/2406.16710v1 (Retrieved 2026-05-20)
[5] Meta AI. "Make-A-Texture: Fast Shape-Aware Texture Generation in 3 Seconds." arXiv:2412.07766. https://arxiv.org/html/2412.07766v1 (Retrieved 2026-05-20)
[6] Bolkart, T. "TF_FLAME — Tensorflow framework for FLAME 3D head model." https://github.com/TimoBolkart/TF_FLAME (Retrieved 2026-05-20)
[7] Bai et al. "FFHQ-UV: Normalized Facial UV-Texture Dataset for 3D Face Reconstruction." https://www.researchgate.net/publication/373306448 (Retrieved 2026-05-20)
[8] "A Morphable Face Albedo Model" (AlbedoMM) and "GANFit." arXiv:2004.02711, https://github.com/barisgecer/GANFit (Retrieved 2026-05-20)
[9] "High-Fidelity Facial Albedo Estimation via Texture Quantization." arXiv:2406.13149 (Retrieved 2026-05-20)
[10] Khakhulin et al. "Realistic One-shot Mesh-based Head Avatars." ECCV 2022, arXiv:2206.08343. https://arxiv.org/abs/2206.08343 (Retrieved 2026-05-20)
[11] Wang et al. "MeGA: Hybrid Mesh-Gaussian Head Avatar." CVPR 2025, arXiv:2404.19026. https://arxiv.org/abs/2404.19026 (Retrieved 2026-05-20)
[12] Chen et al. "AUV-Net: Learning Aligned UV Maps for Texture Transfer and Synthesis." CVPR 2022, arXiv:2204.03105. https://arxiv.org/pdf/2204.03105 (Retrieved 2026-05-20)
[13] Yu et al. "Texture Generation on 3D Meshes with Point-UV Diffusion." arXiv:2308.10490 (Retrieved 2026-05-20)
[14] Cao et al. "TexFusion: Synthesizing 3D Textures with Text-Guided Image Diffusion Models." NVIDIA Toronto. https://research.nvidia.com/labs/toronto-ai/texfusion/ (Retrieved 2026-05-20)
[15] "ROSA: Reconstructing Object Shape and Appearance Textures by Adaptive Detail Transfer." arXiv:2501.18595 (Retrieved 2026-05-20)
[16] Yang et al. "FaceScape: 3D Facial Dataset and Benchmark for Single-View 3D Face Reconstruction." arXiv:2111.01082 and https://github.com/zhuhao-nju/facescape (Retrieved 2026-05-20)
[17] FLAME-Universe overview index. https://github.com/TimoBolkart/FLAME-Universe (Retrieved 2026-05-20)
[18] "Fully Automatic Blendshape Generation for Stylized Characters." https://sjtu-characterlab.github.io/files/Fully_Automatic_Blendshape_Generation_for_Stylized_Characters.pdf (Retrieved 2026-05-20)
[19] Liang et al. "UniTEX: Universal High Fidelity Generative Texturing for 3D Shapes." arXiv:2505.23253, May 2025. https://arxiv.org/abs/2505.23253 and https://github.com/YixunLiang/UniTEX (Retrieved 2026-05-20). PDF: `docs/papers/unitex-2505.23253.pdf`.
[20] Yuan et al. "SeqTex: Generate Mesh Textures in Video Sequence." SIGGRAPH Asia 2025, arXiv:2507.04285. https://github.com/VAST-AI-Research/SeqTex (Retrieved 2026-05-20). PDF: `docs/papers/seqtex-2507.04285.pdf`.
