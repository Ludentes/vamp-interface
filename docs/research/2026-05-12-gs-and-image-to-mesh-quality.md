---
status: live
topic: stylized-renderer
---

# GS-to-Mesh and Image-to-Mesh Quality, Measured Against the Auto-Rig Bar

**Date:** 2026-05-12
**Sources:** 16 sources (arXiv primary papers, project pages, one practical comparison blog, one rigging workflow page). Key load-bearing sources: 2DGS [1], SuGaR [2], TRELLIS / TRELLIS.2 [3][4], Hunyuan3D-2.1 [5], Hunyuan3D Studio [6], InstantMesh [7], MeshFormer [8], CharacterGen [9], RigNet [10], Filmic Worlds ARKit solver [11], Reallusion / VSet3D Hunyuan-to-CC4 [12][13], Trellis-vs-Hunyuan comparison [14].

---

## Executive Summary

For our use case — single image of a stylized, non-humanoid anchor (anime girl, duck, demon) auto-converted to a riggable VRM with ARKit blendshapes painted on — **the current state of GS-to-mesh and image-to-mesh extraction is not good enough to remove the manual Blender step.** GS-to-mesh methods (SuGaR, 2DGS, GS2Mesh) produce sub-millimeter chamfer on object-centric DTU scenes ([1] reports 2DGS CD = 0.80 mm vs SuGaR 1.33 mm vs NeuS 0.84 mm) but are TSDF-fused triangle soup with no quad structure, no UV, no skeleton awareness, and dense (often 100k+ tri) output [1]. Image-to-mesh feed-forward models (TRELLIS, Hunyuan3D-2.x, InstantMesh, MeshFormer) generate watertight marching-cubes meshes that look good in screenshots but every paper that admits a limitations section calls out the same thing: thin geometry fails (hair, horns, glasses, tusks). InstantMesh states this plainly — "FlexiCubes is less effective on modeling tiny and thin structures" [7]. 2DGS states it plainly — "densification favors texture-rich over geometry-rich areas" [1]. Hunyuan3D 2.5 / PolyGen (July 2025) is the first model marketed as producing rig-ready topology, but it still outputs raw 100k–600k tri meshes that require its own retopology pass to get to 10k–50k quads [6][15], and Hunyuan's own auto-rig is documented as working only for humanoid T/A-pose inputs, "struggling with non-standard designs such as creatures" [16]. RigNet caps input at 5k triangles [10], which no image-to-3D model hits without aggressive decimation. **Bottom line: today's pipeline is `image → AI mesh → Blender remesh + retopo + UV + ARKit blendshape wrap → auto-rig`, and the middle step is not automatable for stylized non-humanoid anchors.**

## GS-to-Mesh Quality

The Gaussian-Splatting-to-mesh family is a multi-view-reconstruction technology, not a single-image one — it needs ~30–200 images of a real or rendered object. Useful as a baseline for "what is the ceiling of mesh extraction from a perfectly-fit GS scene?" Sub-table:

| Method | Year | Chamfer↓ (DTU, mm) | Mesh extraction | Watertight? | Known failure modes |
|---|---|---|---|---|---|
| 3DGS (vanilla) | 2023 | 1.96 [1] | Density iso-surface or post-hoc TSDF | No | Floaters; off-surface gaussians; surface is incidental [1] |
| SuGaR | CVPR 2024 | 1.33 [1] | Poisson reconstruction on surface-aligned points [2] | Mostly | Density-reg only suits object-centric scenes; SDF-reg needed for backgrounds [2]; no quad output |
| 2DGS | SIGGRAPH 2024 | **0.80** [1] | TSDF fusion (Open3D, voxel 0.004, trunc 0.02) [1] | Yes (TSDF) | Semi-transparent surfaces fail; fine geometric structures lost to texture bias; regularization over-smooths [1] |
| NeuS (neural SDF baseline) | 2021 | 0.84 [1] | Marching cubes on SDF | Yes | Hours of training; thin features still poor |
| GS2Mesh | ECCV 2024 | SOTA on TT/DTU [17] | Stereo matching → fused depth | Yes | Relies on stereo-net priors; pre-trained model bias on out-of-domain stylized inputs not evaluated |
| 2D-SuGaR | 2026 | 0.67 [1-followup] | 2DGS + mono depth/normal priors | Yes | Single-source claim; small follow-up paper |

**Poly count:** None of the GS-to-mesh papers report a stable poly count number. The TSDF voxel size in 2DGS (0.004 of unit cube) implies grids of ~250³ ≈ 15M candidate cells, with output mesh sizes typically in the 100k–500k triangle range for a single object — confirmed by inspection of SuGaR's own output meshes on the project page [2]. No paper in this family reports non-manifold edge count or hole count as a metric. The TSDF route produces watertight by construction (closed iso-surface); the Poisson route in SuGaR produces watertight on object-centric scenes but leaks on outdoor / background-heavy ones [2].

**For our anchors:** GS-to-mesh is not on the table for one-shot stylized image input — it needs a multi-view scene that doesn't exist for a single anime portrait. Mentioning it only as the upper bound of mesh-extraction quality when geometry input is unlimited.

## Image-to-Mesh Quality

The single-image (or sparse-view) feed-forward family is what's actually relevant for `image → VRM`. Sub-table:

| Method | Year | CD↓ / F-score↑ | Eval set | Output | Watertight | Stated limitations |
|---|---|---|---|---|---|---|
| TripoSR | 2024 | InstantMesh beats it on CD/FS [7] | GSO | Triplane NeRF → MC mesh | Yes | Back-side artifacts; "lacks imagination" outside Objaverse style [7] |
| LGM, CRM | 2024 | Beaten by InstantMesh [7] | GSO | Gaussian / triplane → MC | Yes | Multi-view inconsistency; low-res triplane |
| InstantMesh | 2024 | CD 0.180 / F 0.880 (GSO); CD 0.203 / F 0.864 (Omni3D) [7] | GSO, Omni3D | FlexiCubes mesh | Yes | 64×64 triplane is a resolution bottleneck; multi-view diffusion inconsistency; "FlexiCubes is less effective on tiny and thin structures" [7] |
| SF3D | 2024 | Improves over TripoSR | GSO | Direct mesh | Yes | Back-side artifacts (no intermediate multiview) [18] |
| Unique3D | NeurIPS 2024 | Visual SOTA at release [18] | (qualitative) | Multiview → mesh | Yes | Sensitive to facing direction; orthographic front + rest pose required; occlusions break it; longest-edge normalization causes squashing on non-canonical inputs [18] |
| Era3D | 2024 | not directly reported | — | Multiview → mesh | Yes | Not separately evaluated in our sources |
| MeshLRM | 2024 | High visual quality, mesh has "uneven artifacts on close inspection" [8] | GSO | Differentiable MC, end-to-end | Yes | Failed to converge in MeshFormer's 8-GPU 2-day reproduction [8] |
| MeshFormer | NeurIPS 2024 | "Most accurate fine-grained details" of the LRM family [8] | GSO | Sparse-voxel + transformer → mesh | Yes | No quad output; topology is voxel-derived; isotropy is acknowledged as missing [8] |
| CharacterGen | TOG 2024 | Quantitative + qualitative claimed [9] | 13.7k VRoid anime characters | A-pose mesh | Yes | Trained on VRoid only; A-pose canonicalization is the point; out-of-distribution stylized characters not evaluated [9] |
| TRELLIS / TRELLIS.2 | CVPR 2025 | CD 0.0083 / F 0.9999 (autoencoder reconstruction on Toys4k) [3][4] | Toys4k | SLAT → FlexiCubes 256³ | Yes | Autoencoder fidelity not end-to-end generation quality; limitations section truncated in v1 paper [3]; better on hollow/thin walls than peers [14] |
| Hunyuan3D-2.1 | 2025 | **No CD/F-score reported in paper [5]** | (ULIP, Uni3D similarity only) | Marching cubes on iso-surface [5] | "Watertight by construction" claim [5] | No limitations section discusses hair, thin, stylized; no UV parameterization details [5] |
| Hunyuan3D 2.5 / PolyGen | Apr/Jul 2025 | Not benchmarked vs prior CD/F in any source we found | 1024 geometric resolution; 10B params [6][15] | 50k–1.5M tris raw; PolyGen retopo → quad/tri 10k+ [6][15] | Yes (raw); retopo claims "deformation-aware edge flow" [6] | PolyGen claims are vendor-source; no independent benchmark; auto-rig "struggles with non-standard designs" [16] |
| Rodin Gen-2 | commercial | No published quality benchmark | — | — | — | Commercial; treat marketing claims with caution |

**Poly count is the load-bearing number for us, and almost no academic paper reports it.** What we have: Hunyuan3D 2.5 self-reports 50k–1.5M [15]; the practical comparison [14] describes Hunyuan generally producing "dense meshes (up to 600,000 triangles)" requiring 3–5 min retopo for AAA. InstantMesh / MeshFormer / TRELLIS output is FlexiCubes at 256³ which typically yields tens of thousands of triangles, but I did not find an explicit number in any of the four papers.

**Reconstruction-vs-generation gap.** The TRELLIS chamfer of 0.0083 [3] is its *VAE reconstruction* fidelity, not the quality of an image-conditioned generation. For image-to-3D end-to-end CD/F-score, the only directly-comparable numbers I found are InstantMesh's 0.180 CD / 0.880 F on GSO [7]. This is the cleanest absolute number from a primary source in this report.

## Hair, Thin, Non-Convex Geometry

This is our actual failure mode (anime hair, demon horns, orc tusks, fox ears, glasses, mic stands). What the papers say:

- **2DGS:** "Densification strategy favors texture-rich over geometry-rich areas, occasionally leading to less accurate representations of fine geometric structures" [1]. Plus full failure on semi-transparent surfaces. Anime hair is exactly the failure region — thin, layered, often translucent.
- **InstantMesh:** "FlexiCubes can improve smoothness but is less effective on modeling tiny and thin structures compared to NeRF" [7]. The mesh variant explicitly degrades on thin features vs the NeRF variant of the same model.
- **Unique3D:** Strong on canonical front-facing rest-pose only. Occluded geometry (anime hair behind head, anything behind tusks) "will cause worse reconstructions, since four views cannot cover the complete object" [18].
- **TRELLIS 2:** The Trellis-vs-Hunyuan-vs-Meshy artist comparison [14] reports TRELLIS 2's sparse voxel representation handles hollow interiors and thin walls better than alternatives — but the same comparison says both Trellis and Hunyuan produce "lower-quality face details that would require enhancement," and Meshy is preferred for hollow trigger guards on weapons. So even at the current 2025 SOTA, thin features are unreliable per artist testing.
- **SuGaR / SDF methods:** No explicit hair-failure discussion in [2], but the Poisson-on-surface-points approach has no mechanism for thin sheets — every sheet becomes a closed thin volume or merges into a blob.

There is no image-to-mesh paper in our source set that publishes a hair / thin-feature benchmark with numbers. All discussion is either limitation-section prose or third-party visual comparison.

## Topology and UV

Every method discussed produces **triangle soup, not quads**. Marching cubes (the iso-surface extractor behind 2DGS-TSDF, Hunyuan3D, and the FlexiCubes-using LRM family) produces by-the-book irregular triangulation oriented to the voxel grid, not to surface curvature. MeshFormer's own paper acknowledges this as a problem: "isotropy property is an important metric for mesh quality evaluation, holding significant value for applications such as texture UV-mapping, physical simulation, and discrete geometric analysis" [8] — i.e., they know isotropy matters and they don't have it.

The community-side picture from artist reviews [19]: AI-generated meshes are uniformly described as "dense, irregular triangle meshes" with "chaotic polygon distribution with no regard for joint deformation," requiring a remesh / retopo step before rigging will work. This is consistent across Hunyuan, TRELLIS, Tripo, Meshy outputs.

**The one apparent exception is Hunyuan3D-PolyGen (July 2025) [15]**, an autoregressive face-by-face model that takes a generated point cloud and emits quad/tri topology with "deformation-aware edge flow." Vendor claims include 70% creation-time reduction and 35% better topology neatness. There is no independent academic benchmark of these claims in our sources. The model is referenced as available via Tencent's API but is not in the open Hunyuan3D-2.1 release [20]. For our purposes: promising but not verified.

**UV unwrap.** No image-to-mesh paper we found explicitly evaluates UV chart quality, seam count, or distortion. Hunyuan3D ships PBR textures in baked space [5], but how the UVs are derived (xatlas? per-triangle?) is not stated in the paper. Practical impact: even if geometry is good, the texture-painting path for ARKit blendshape transfer needs clean UVs, and we have no guarantee.

## Stylized / Anime / Non-Human Input

Only **CharacterGen** [9] is built and benchmarked specifically on anime characters (13,746 VRoid characters, multiple poses and views). It generates A-pose meshes "suitable for downstream rigging and animation" and explicitly handles the cohesion failure that other methods produce on non-A-pose inputs (a character generated in mid-pose has limbs merging into the torso, which CharacterGen avoids by canonicalizing to A-pose first). However, CharacterGen is trained on VRoid-style anime only — humanoid bipedal characters with anime aesthetic. Duck / demon / orc / non-bipedal inputs are out-of-distribution.

For non-humanoid stylized (the duck, the dragon, the demon mask), no source in this report shows a method explicitly evaluated. The Trellis-vs-Hunyuan-vs-Meshy comparison [14] notes "for stylized characters, the open-source models (Hunyuan and TRELLIS) outperformed commercial solutions by better preserving the cartoon aesthetic" — but this is a visual claim, not a quantitative one, and "preserving the cartoon aesthetic" refers to surface look, not to whether the resulting mesh is rig-clean. The same source says Meshy dominates stylized cartoon characters on visual quality but doesn't auto-rig.

## Verdict Against Our Use Case

**For an anime girl / duck / demon image, the auto-extracted mesh today is not directly usable in an automated rig → ARKit → VRM pipeline.** Three concrete reasons:

1. **Topology is wrong for rigging.** Triangle soup from marching cubes / FlexiCubes / Poisson has no deformation-aware edge flow. RigNet's documented input cap is 5k triangles [10]; image-to-3D outputs are 30k–600k+ triangles [14][15]. Even modern auto-riggers (Mixamo, Hunyuan's built-in, AccuRIG, Auto-Rig Pro) work reliably only on humanoid T/A-pose with clean topology. The Reallusion / Vset3D Hunyuan-to-CC4 tutorial [12][13] is explicit that Hunyuan auto-rig "works best for humanoid characters in T/A-poses but may struggle with non-standard designs such as creatures, and manual rigging in Blender may be needed for complex animations." A duck is a creature.

2. **Thin features are unreliable.** Every paper that has a limitations section flags it [1][7][18]. Anime hair, horns, tusks, ears, glasses, hand fans — these are exactly where mesh extraction collapses into blobs or holes. PolyGen-style retopology can rebuild the topology but cannot rebuild geometry that wasn't extracted in the first place.

3. **ARKit blendshape transfer requires landmark-consistent face topology.** ARKit's reference face mesh is 1,220 verts / 2,304 tris [11], and deformation-transfer-based blendshape painting (NRICP via Wrap3D, EBFR, deformation transfer [11][21]) needs a target face region with anatomically-correct landmarks. An AI-generated duck has no canonical face topology to wrap onto. Even an AI-generated anime girl has hair occlusion and mesh fusion at the head-hair boundary that breaks NRICP.

The premise "one-shot image → fully automated rigged ARKit-driven VRM" is structurally unworkable today for non-human / stylized anchors. It is almost workable for canonical-anime humanoid inputs via CharacterGen + auto-rig, but only because CharacterGen specifically restricts to that distribution.

## What an Honest Pipeline Would Need to Add

If we keep the `image → mesh → rig → ARKit` shape, the manual / semi-manual steps that must sit between the AI stages:

1. **Decimation + retopology.** Either Hunyuan3D-PolyGen (vendor, unverified), or Quad Remesher (commercial, well-validated), or RetopoFlow / ZBrush ZRemesher / Blender's Quadriflow [19]. ~3–5 min auto, longer with manual edge-flow cleanup [14].
2. **Manual hair / thin-feature reconstruction.** Either accept low-fidelity hair, or rebuild hair as cards in Blender, or use a hair-specialized pipeline. No automated solution exists for arbitrary stylized hair.
3. **Skeleton placement.** Auto-riggers work for humanoid T/A-pose; creature rigs need manual joint placement. UniRig [22] (SIGGRAPH 2025) claims diverse skeleton support but is single-source and unverified for our stylized inputs.
4. **Face topology wrap + ARKit blendshape transfer.** Wrap3D NRICP + deformation transfer of a canonical 52-blendshape ARKit donor [11][21]. Possible to script for humanoid faces; fragile for non-human.
5. **VRM export with spring-bones, lookat, blendshape-clip mapping.** Mostly tooling, not AI.

A realistic pipeline for our bake-off is: `image → CharacterGen (humanoid anime only) or Hunyuan3D 2.5 + PolyGen (everything else) → Blender retopo step → manual face-region cleanup → Wrap3D ARKit blendshape transfer → UniRig / Mixamo → VRM`. Steps 3 and 4 are where the "automate it" premise breaks for non-humanoid; step 2 retopo is feasible but not zero-touch.

For a non-humanoid anchor (duck, demon, orc), the honest answer is **drop the universal-mesh pipeline and use anchor-specific approaches:** LAM-style FLAME-anchored GS for humanoid-ish heads; existing rigged VRoid / VRM creature templates with morph-driven shape blending for stylized creatures; LP-style 2D warp for the rest. The "automate image → VRM" promise is structurally one-shot only for the canonical-anime-humanoid corner of the input distribution.

## Open Questions

- **No primary source publishes CD / F-score for image-to-3D on stylized non-Objaverse anchors.** All numbers are on GSO / Toys4k (mostly toys and objects) or DTU / Tanks&Temples (multi-view real captures).
- **Hunyuan3D-PolyGen claims are vendor-side only.** No independent benchmark of its "deformation-aware edge flow" claim exists in our sources.
- **UV chart quality is uniformly unreported.** None of the image-to-3D papers we read evaluate UV seam count, chart count, or stretch.
- **Watertightness rates are reported as binary properties (the method "produces watertight meshes") rather than as success rates over a test set.** No method publishes "we are watertight on 87% of GSO inputs."
- **Hair / thin-feature failure has no benchmark.** All discussion is qualitative.

## Sources

[1] Huang, Yu, Chen, Geiger, Gao. "2D Gaussian Splatting for Geometrically Accurate Radiance Fields." SIGGRAPH 2024. https://arxiv.org/html/2403.17888v3 (Retrieved 2026-05-12)
[2] Guédon, Lepetit. "SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction." CVPR 2024. https://anttwo.github.io/sugar/ ; https://arxiv.org/abs/2311.12775 (Retrieved 2026-05-12)
[3] Microsoft. "TRELLIS: Structured 3D Latents for Scalable and Versatile 3D Generation." CVPR 2025 Spotlight. https://arxiv.org/html/2412.01506v1 (Retrieved 2026-05-12)
[4] Microsoft. "TRELLIS.2: Native and Compact Structured Latents for 3D Generation." 2025. https://microsoft.github.io/TRELLIS.2/ (Retrieved 2026-05-12)
[5] Tencent. "Hunyuan3D 2.1: From Images to High-Fidelity 3D Assets with Production-Ready PBR Material." 2025. https://arxiv.org/html/2506.15442v1 (Retrieved 2026-05-12)
[6] Tencent. "Hunyuan3D Studio: End-to-End AI Pipeline for Game-Ready 3D Asset Generation." 2025. https://arxiv.org/html/2509.12815v1 (Retrieved 2026-05-12)
[7] Xu et al. "InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models." 2024. https://arxiv.org/html/2404.07191v1 (Retrieved 2026-05-12)
[8] Liu et al. "MeshFormer: High-Quality Mesh Generation with 3D-Guided Reconstruction Model." NeurIPS 2024. https://meshformer3d.github.io/ ; https://arxiv.org/html/2408.10198v1 (Retrieved 2026-05-12)
[9] Peng et al. "CharacterGen: Efficient 3D Character Generation from Single Images with Multi-View Pose Canonicalization." TOG 2024. https://charactergen.github.io/ ; https://cg.cs.tsinghua.edu.cn/papers/TOG-2024-CharacterGen.pdf (Retrieved 2026-05-12)
[10] Xu et al. "RigNet: Neural Rigging for Articulated Characters." SIGGRAPH 2020. https://zhan-xu.github.io/rig-net/ ; https://arxiv.org/pdf/2005.00559 (Retrieved 2026-05-12)
[11] Filmic Worlds. "Solving Blendshapes for ARKit." https://filmicworlds.com/blog/solving-face-scans-for-arkit/ (Retrieved 2026-05-12)
[12] Reallusion Magazine. "From AI to Animation: 3 Ways to Bring Hunyuan3D Characters to Life in CC4." 2025. https://magazine.reallusion.com/2025/08/13/from-ai-to-animation-3-ways-to-bring-hunyuan3d-characters-to-life-in-cc4/ (Retrieved 2026-05-12)
[13] VSet3D. "Hunyuan 3D-2.5 – Create and Rig a 3D Character in 5 Steps." https://www.vset3d.com/hunyuan-3d-2-5-create-and-rig-a-3d-character-in-5-steps/ (Retrieved 2026-05-12)
[14] 3DAI Studio. "Trellis 2 vs Hunyuan 3D: Key Differences Explained." https://www.3daistudio.com/blog/trellis-2-vs-hunyuan-3d-differences-explained (Retrieved 2026-05-12)
[15] Scenario. "Best AI 3D Retopology Tool: Hunyuan Polygen 1.5." https://www.scenario.com/models/hunyuan-polygen-15 (Retrieved 2026-05-12)
[16] Yelzkizi. "How to Make a 3D VTuber Avatar." https://yelzkizi.org/make-a-3d-vtuber-avatar/ (Retrieved 2026-05-12)
[17] Wolf et al. "GS2Mesh: Surface Reconstruction from Gaussian Splatting via Novel Stereo Views." ECCV 2024. https://gs2mesh.github.io/ ; https://arxiv.org/abs/2404.01810 (Retrieved 2026-05-12)
[18] Wu et al. "Unique3D: High-Quality and Efficient 3D Mesh Generation from a Single Image." NeurIPS 2024. https://arxiv.org/html/2405.20343v1 (Retrieved 2026-05-12)
[19] Alpha3D. "AI Retopology for 3D Modeling." https://www.alpha3d.io/kb/3d-modelling/ai-retopology/ (Retrieved 2026-05-12)
[20] Tencent-Hunyuan / Hunyuan3D-2.1 issue 111. "Open-Source Plans for Hunyuan3D-2.5 and Hunyuan3D-PolyGen." https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/issues/111 (Retrieved 2026-05-12)
[21] Katr. "deformation_transfer_ARkit_blendshapes." https://github.com/vasiliskatr/deformation_transfer_ARkit_blendshapes (Retrieved 2026-05-12)
[22] VAST-AI-Research. "UniRig: One Model to Rig Them All." SIGGRAPH 2025. https://github.com/VAST-AI-Research/UniRig (Retrieved 2026-05-12)
