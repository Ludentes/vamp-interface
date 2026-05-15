---
status: live
topic: stylized-renderer
---

# Research: FLAME extensions and alternatives for non-human / stylized anchors in a LAM-style pipeline

**Date:** 2026-05-12
**Sources:** 15 sources (LAM, GAGAvatar, GaussianAvatars, Portrait4D / Portrait4D-v2, VOODOO 3D, VOODOO XP, PAniC-3D, StyleMM, StyleMorpheus, CAP4D, PortraitGen, AniFaceGAN, VMagicMirror Perfect Sync, MoCap Online VTuber guide, Zhu 2024 retargeting review). URLs in *Sources*.

---

## Executive Summary

The realistic-tolerant one-shot Gaussian-splat literature (LAM, GAGAvatar, GaussianAvatars, Portrait4D, CAP4D) is **uniformly FLAME-locked at the preprocessor**: trackers built on FLAME / 3DMM-driven landmark detectors are the chokepoint, not the renderer. Stylized anchors (anime, orc, demon, cartoon-duck) fall off at face detection / FLAME tracking long before the Gaussian model gets a chance, which matches our observed VGGHead + FaceBoxes failure. Of the four strategic options the brief enumerates, only two are well-attested at 2026-05 maturity: **(ii) stack a stylized renderer on top of a FLAME-tracked human (our existing `project_stylized_renderer_thread` SPADE-LoRA path on LivePortrait)** and **(iv) hand stylized characters to the industry VTuber stack (VRoid + Perfect Sync ARKit 52)**. **Path (i) "fix LAM's preprocessor"** is single-paper territory — only PAniC-3D and the very recent StyleMM / StyleMorpheus address stylized tracking, and none of them is a drop-in for LAM. **Path (iii) "swap to a generic-mesh GS"** does not exist as a published system today; every "generalizable" one-shot head-Gaussian paper we found still pulls FLAME / 3DMM parameters somewhere in its tracker.

The recommendation is a **two-stack split** built around our existing capabilities: keep LAM for human and painted-human anchors (4/4 pass), and route stylized anchors (anime/orc/demon/cartoon) to the VTuber stack (path iv) for now, with the SPADE-LoRA renderer (path ii) as the medium-term alternative when a human-base anchor is acceptable. Treat path (i) as a research bet only — StyleMM is the single most interesting 2025 paper, but it is one paper, no LAM integration, and is texture/mesh-only, not Gaussian.

---

## Key Findings

### FLAME for anime / stylized faces — only two viable threads, both single-paper

The literature explicitly trying to put stylized inputs into a FLAME-like parametric pipeline is thin. **PAniC-3D** (CVPR 2023, Chen et al.) reconstructs a stylized volumetric radiance field directly from anime portraits, trained on 11.2k VRoid 3D models and 1k VTuber illustrations [1]. It does *not* extend FLAME — it learns its own line-fill plus volumetric field — so the output is a NeRF, not a riggable mesh, and the work was not subsequently extended into the Gaussian-splat or animatable-avatar literature.

**StyleMM** (Computer Graphics Forum 2025, arXiv 2508.11203) is the only 2025 work that produces an animatable stylized 3DMM with preserved vertex connectivity, which means expression/pose parameters from a video tracker can drive it directly [2]. It fine-tunes mesh-deformation and texture nets on stylized i2i targets from a diffusion model. Critically, it claims to *preserve the animatable structure of realistic 3DMM*, so a FLAME-tracked driver could in principle animate a StyleMM mesh. This is the most interesting single-paper result for path (i), but it is mesh+texture, **not Gaussian**, and there is no demonstrated integration with LAM-class one-shot Gaussian pipelines. **StyleMorpheus** (arXiv 2503.11792) is a parallel style-based 3DMM but with less ARKit-driver story [2].

AniFaceGAN (NeurIPS 2022) is sometimes cited in this neighbourhood but is a 3D-aware GAN (NeRF-style triplane), not a FLAME extension, and decomposes into a canonical template + deformation rather than emitting parametric blendshapes [3]. Not a drop-in.

### Generic-mesh Gaussian/NeRF head avatars — none are FLAME-free at inference

The "swap parametric model entirely" path (option iii in the brief) does not have a clean published winner. **LAM** (SIGGRAPH 2025) explicitly states its canonical Gaussian generator *uses FLAME canonical points as queries* and animates via standard LBS plus FLAME corrective blendshapes [4]; FLAME is load-bearing inside the model, not merely a preprocessing step. **GaussianAvatars** (CVPR 2024 Highlight) initialises one 3D Gaussian per FLAME triangle and rigs them rigidly to the FLAME mesh; tracking errors propagate to the Gaussians during optimisation [5]. **GAGAvatar** (NeurIPS 2024) "leverages global image features and the 3D morphable model to construct 3D Gaussians for controlling expressions" — its acknowledgements list FLAME, StyleMatte, EMICA, VGGHead as upstream tracking dependencies, the same VGGHead our LAM pipeline already fails on at the detector stage [6]. **Portrait4D / Portrait4D-v2** (CVPR 2024 / ECCV 2024) is the closest thing to FLAME-independence — Portrait4D-v2 explicitly *avoids reliance on inaccurate 3DMM reconstruction* by training on pseudo multi-view video, and the abstract claims it "accommodates human, stylized, and anthropomorphic forms" including accessories [7]. This is the single most credible candidate for stylized one-shot Gaussian-style output, but the project page does not show an extensive stylized gallery and the claim is single-source. **CAP4D** (CVPR 2025) uses a "morphable multi-view diffusion model" to hallucinate views, then fits 3D Gaussians; the morphable prior is still 3DMM-grounded [8]. **PortraitGen** (SIGGRAPH Asia 2024) lifts a portrait video into a 4D Gaussian field via SMPL-X tracking and then permits diffusion-prior stylisation at 100 fps — but it is a *video editor*, not a one-shot avatar, and it stylises an already-tracked human, which puts it firmly in path (ii), not (iii) [9].

The honest summary: **as of 2026-05 there is no published one-shot Gaussian-head method that is both (a) generic-mesh / FLAME-independent at inference and (b) demonstrated on anime/orc/cartoon inputs.** Portrait4D-v2 has the strongest claim but it is one paper with limited stylised demos.

### Expression / blendshape retargeting — well-attested but offline

Retargeting from a human (or ARKit) driver onto a non-human face is a well-trodden problem with mature, if labour-intensive, solutions. Zhu et al. 2024 [10] survey appearance-agnostic FACS-based pipelines that translate motion into AU coefficients (or ARKit blendshape weights) and then re-bind those to non-human rigs prepared with matching blendshape sets. The review's key caution: blendshape parameterisation "works for human, human-like, and nonhuman characters but requires intensive preparation of the facial poses." This matches industry practice — Mark Sagar's FACS pipeline at Weta was used on *Monster House* and *King Kong* [10]. For our use case this means the retargeting math is solved provided the **target character is hand-rigged with a FACS-equivalent blendshape set** (the ARKit 52 are functionally a FACS subset). This is exactly what VRoid + HANA_Tool + Perfect Sync gives us in path (iv).

### Recent one-shot Gaussian-splat papers explicitly supporting stylized inputs — Portrait4D-v2 is the only candidate

Cross-checking the brief's named candidates: **VOODOO 3D** (arXiv 2312.04651, ECCV 2024) is volumetric and avoids 3DMM for expression disentanglement, but the authors' own limitations section says "for very highly stylized portraits such as cartoons, the framework often produces photorealistic facial elements such as teeth which can be inconsistent in style" [11] — a direct admission that stylised inputs leak photoreal artefacts in the renderer, mirroring exactly our observed failure mode on `project_stylized_renderer_thread`. **VOODOO XP** (SIGGRAPH Asia 2024) extends VOODOO 3D for VR telepresence with monocular face capture, real-time and view-consistent, but the demo set is human telepresence; no stylised gallery [12]. **Portrait4D-v2** [7] is the only paper that explicitly claims stylized support in the abstract.

### VTuber / industry stack — production-grade, ARKit-native, what we already have

The VRoid + Perfect Sync stack is mature and matches our Live Link Face driver exactly. **VRoid Studio** exports a VRM model; **HANA_Tool** (Unity package) automatically generates the 52 ARKit blendshapes from VRoid's base shape keys; **VMagicMirror / VSeeFace / Warudo** receive iFacialMocap or Live Link Face ARKit streams and drive the 52 BlendShapeClips at runtime [13]. iFacialMocap is interchangeable with Live Link Face — both stream the same 52 ARKit coefficients [14]. The pipeline is plug-and-play for indie VTubers, no per-character training required, runs on consumer hardware, and head pose + ARKit blendshapes is exactly what our LLF UDP stream already provides. The constraint is that the character must be authored once in VRoid (or Blender, with 52 blendshapes set up manually) — it is not "one image in seconds" like LAM.

---

## Comparison

| Name | Family | Input domain | Output | Driver | License / code | Use-case fit (anime/orc/cartoon) |
| --- | --- | --- | --- | --- | --- | --- |
| LAM [4] | FLAME-extend (one-shot GS) | Real human photo | Animatable Gaussian head | FLAME params / ARKit via Audio2Expression | Apache-ish, code public | **Fails at detector**; only path forward is preprocessor hack (path i) |
| GaussianAvatars [5] | FLAME-extend (per-subject GS) | Multi-view human video | Rigged Gaussians on FLAME | FLAME tracker | Code public | Fails for same reason as LAM; also not one-shot |
| GAGAvatar [6] | FLAME-extend (one-shot GS) | Real human photo | Gaussian head | 3DMM params | MIT | Same FLAME / VGGHead chokepoint we already hit |
| Portrait4D-v2 [7] | Generic triplane (one-shot) | Human + claimed stylized/anthropomorphic | 4D triplane | Motion tokens | Code public | **Most credible single-paper candidate for path (iii)**; demos limited |
| VOODOO 3D / XP [11][12] | Volumetric tri-plane (one-shot) | Human portrait | Tri-plane radiance field | Driver video | Code public | Authors' own paper documents stylised failure: photoreal teeth leak |
| CAP4D [8] | Diffusion + 3DMM + GS | 1–100 reference images | 4D GS avatar | 3DMM motion | Code public | 3DMM-bound, not stylized-native |
| PortraitGen [9] | Video → 4D GS, multimodal edit | Real portrait *video* | 4D Gaussian field, stylised | SMPL-X + text/image prompt | Code public | **Path (ii)-shaped**: stylise a tracked human; 100 fps; not anchored-anime input |
| PAniC-3D [1] | Stylised single-view → NeRF | Anime illustration | Volumetric radiance field | Static (not animatable in the FLAME sense) | Code + datasets public | Reconstructs but doesn't animate via ARKit; 2023, no follow-up GS extension |
| StyleMM [2] | Stylized 3DMM (mesh+texture) | Real human + text style | Animatable stylized mesh + texture | Standard 3DMM expression params | arXiv + CGF 2025; code unclear | **Single most interesting 2025 paper** for path (i); mesh-only, not Gaussian, no LAM integration |
| StyleMorpheus [2] | Style-based 3D-aware MM | Implicit | Implicit MM | 3DMM params | arXiv 2025 | Parallel to StyleMM; less ARKit story |
| AniFaceGAN [3] | 3D-aware GAN (anime-tolerant) | 2D anime images | Triplane | Template+deformation | Code public; older (2022) | Not a FLAME or ARKit pipeline |
| VRoid + Perfect Sync + Live Link Face [13][14] | Industry VTuber rig | Pre-authored VRoid VRM | Real-time rigged stylised 3D head | ARKit 52 blendshapes via Live Link/iFacialMocap | Free / commercial-friendly; mature | **Production-ready for stylized**; not one-shot — requires authoring per character |
| FACS retargeting reviews [10] | Blendshape retargeting | Driver video / blendshapes | Any rigged target | FACS / ARKit 52 | n/a (theory) | Confirms ARKit→non-human works *if target is hand-rigged* |

---

## Open Questions

- **Does LAM's FLAME-points generator tolerate a stylized target if we hand it a hand-tracked FLAME state?** The brief assumes the failure is the detector. We have not verified that bypassing the detector with manually-set FLAME params would let LAM render a working orc or anime head — its training distribution is human FFHQ-class. Quick spike: feed LAM a known-FLAME-tracked stylised render and look at output.
- **Does Portrait4D-v2 actually pass our anime / orc / cartoon-duck stills?** The abstract claim is the strongest in the literature; the project page demo set is the weakest evidence. Worth one afternoon of running their public checkpoint on our four failure cases before committing to path (iv).
- **Is StyleMM's "preserve 3DMM animatable structure" claim load-bearing enough to drive a stylized mesh from our ARKit 52 stream?** The CGF paper is one paper, October 2025; no replication yet, code availability unclear.
- **Latency / FPS on Blackwell:** none of the cited papers report 5090 numbers; LAM and GAGAvatar quote A100 67 fps for GAGAvatar. PortraitGen quotes 100 fps but on pre-tracked input. Real-time at 512² on our target hardware is unverified for any of these.

---

## Conclusion: recommended path for the Rorschach VTuber product

Frame this against our concrete constraint stack: one-shot input (a single still of the character), 30 fps, 512², Blackwell-friendly, head-only, ARKit-52 driver from Live Link Face, and the **stylized anchors (anime + orc + demon + cartoon-duck) already fail at the FaceBoxes/VGGHead detector inside LAM's preprocessor**. The 4/4 human / painted-human anchors that *do* pass LAM are not the problem we need to solve.

The right move is **path (iv) for the stylized half of the product, with path (ii) as the medium-term R&D bet** — explicitly *not* path (i) or (iii):

- **Path (iv) — VTuber stack for stylized characters:** plug Live Link Face → VRoid VRM (with HANA_Tool Perfect Sync blendshapes) → VMagicMirror/Warudo. This already exists, is production-mature, drives the same ARKit 52 we already stream, and has zero training cost per character. The cost is that each new stylized character needs to be authored as a VRM once (artist time, or commission, or VRoid Studio user). Reframing the product accordingly: Rorschach's stylized hosts (anime/orc/demon/duck) become *authored* characters, not one-shot reconstructions. This is also how every shipping VTuber-class product works today, which is independent evidence that the path is real.
- **Path (ii) — SPADE-LoRA on LivePortrait, the existing `project_stylized_renderer_thread`:** keep this as the bet on "one-shot stylized from a still." It does not need FLAME at all because LivePortrait already handles its own implicit-keypoint tracking, and the SPADE-LoRA localises the photoreal prior. This is where the FFHQ+MetFaces+Ukiyoe paired-synthetic LoRA effort should continue. It is also the only published path that gets us *one-shot* stylized animation.
- **Path (iii) — generic-mesh GS swap:** parked. No published system delivers FLAME-independent one-shot Gaussian heads on stylized inputs today. Portrait4D-v2 is the only candidate worth even a single-day spike, and only because its abstract makes the claim — the rest of the literature is FLAME-locked.
- **Path (i) — fix LAM's preprocessor:** parked. PAniC-3D and StyleMM are interesting research, but neither is a drop-in LAM extension, and patching FaceBoxes/VGGHead confidence thresholds to admit anime inputs is a degrees-of-freedom problem we cannot validate without training-time data that doesn't exist.

Net: keep LAM as the human-anchor renderer (it works), ship the stylized anchors through VRoid + Perfect Sync (path iv), and continue the SPADE-LoRA LivePortrait research (path ii) as the bet on one-shot stylized. Path (i) and (iii) are research thread material, not product roadmap.

---

## Sources

[1] Chen et al. "PAniC-3D: Stylized Single-view 3D Reconstruction from Portraits of Anime Characters." CVPR 2023. https://arxiv.org/abs/2303.14587 ; https://github.com/ShuhongChen/panic3d-anime-reconstruction (retrieved 2026-05-12)
[2] Lee et al. "StyleMM: Stylized 3D Morphable Face Model via Text-Driven Aligned Image Translation." Computer Graphics Forum 2025 / arXiv 2508.11203. https://arxiv.org/abs/2508.11203 ; https://kwanyun.github.io/stylemm_page/ . Cross-ref: "StyleMorpheus: A Style-Based 3D-Aware Morphable Face Model." arXiv 2503.11792. https://arxiv.org/abs/2503.11792 (retrieved 2026-05-12)
[3] Wu et al. "AniFaceGAN: Animatable 3D-Aware Face Image Generation for Video Avatars." NeurIPS 2022 Spotlight. https://arxiv.org/abs/2210.06465 (retrieved 2026-05-12)
[4] He et al. "LAM: Large Avatar Model for One-shot Animatable Gaussian Head." SIGGRAPH 2025 / arXiv 2502.17796. https://arxiv.org/abs/2502.17796 ; https://github.com/aigc3d/LAM (retrieved 2026-05-12)
[5] Qian et al. "GaussianAvatars: Photorealistic Head Avatars with Rigged 3D Gaussians." CVPR 2024 Highlight. https://arxiv.org/abs/2312.02069 ; https://github.com/ShenhanQian/GaussianAvatars (retrieved 2026-05-12)
[6] Chu et al. "Generalizable and Animatable Gaussian Head Avatar (GAGAvatar)." NeurIPS 2024. https://arxiv.org/abs/2410.07971 ; https://github.com/xg-chu/GAGAvatar (retrieved 2026-05-12)
[7] Deng et al. "Portrait4D: Learning One-Shot 4D Head Avatar Synthesis using Synthetic Data" (CVPR 2024) and "Portrait4D-v2: Pseudo Multi-View Data Creates Better 4D Head Synthesizer" (ECCV 2024). https://arxiv.org/abs/2311.18729 ; https://yudeng.github.io/Portrait4D-v2/ ; https://github.com/YuDeng/Portrait-4D (retrieved 2026-05-12)
[8] Taubner et al. "CAP4D: Creating Animatable 4D Portrait Avatars with Morphable Multi-View Diffusion Models." CVPR 2025. https://arxiv.org/abs/2412.12093 ; https://felixtaubner.github.io/cap4d/ (retrieved 2026-05-12)
[9] USTC3DV. "PortraitGen: Portrait Video Editing Empowered by Multimodal Generative Priors." SIGGRAPH Asia 2024. https://arxiv.org/abs/2409.13591 ; https://ustc3dv.github.io/PortraitGen/ (retrieved 2026-05-12)
[10] Zhu et al. "A Facial Motion Retargeting Pipeline for Appearance Agnostic 3D Characters." Computer Animation and Virtual Worlds, 2024. https://onlinelibrary.wiley.com/doi/10.1002/cav.70001 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC11653099/ . Cross-ref: review article https://www.sciencedirect.com/science/article/pii/S0097849324001729 (retrieved 2026-05-12)
[11] Tran et al. "VOODOO 3D: Volumetric Portrait Disentanglement for One-Shot 3D Head Reenactment." CVPR 2024 / arXiv 2312.04651. https://arxiv.org/abs/2312.04651 ; https://p0lyfish.github.io/voodoo3d/ ; https://github.com/mbzuai-metaverse/VOODOO3D-official (retrieved 2026-05-12)
[12] Tran, Zakharov et al. "VOODOO XP: Expressive One-Shot Head Reenactment for VR Telepresence." SIGGRAPH Asia 2024 / arXiv 2405.16204. https://arxiv.org/abs/2405.16204 ; https://mbzuai-metaverse.github.io/voodooxp/ (retrieved 2026-05-12)
[13] Baku. "Perfect Sync — VMagicMirror." https://malaybaku.github.io/VMagicMirror/en/tips/perfect_sync/ . Cross-ref: VSeeFace manual https://github.com/emilianavt/VSeeFaceManual ; Warudo Handbook https://docs.warudo.app/docs/tutorials/3d-primer ; "Example Expressions from Blendshapes with VRoid Studio, HANA_Tool, and Unity," https://extra-ordinary.tv/2021/01/15/example-expressions-from-blendshapes-with-vroid-studio-hana_tool-and-unity/ (retrieved 2026-05-12)
[14] MoCap Online. "VTuber Motion Capture: Body Tracking and Avatar Setup." https://mocaponline.com/blogs/mocap-news/vtuber-motion-capture-guide ; "Face Capture for Game Dev: iPhone, ARKit, and MetaHuman." https://mocaponline.com/blogs/mocap-news/face-capture-game-dev-iphone-arkit-live-link-metahuman (retrieved 2026-05-12)
[15] ARKit Blendshapes reference. Apple ARKit 52 blendshape definitions: https://arkit-face-blendshapes.com/ ; https://github.com/suchipi/arkit-face-blendshapes (retrieved 2026-05-12)
