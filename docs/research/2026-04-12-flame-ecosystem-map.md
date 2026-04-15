# Research: FLAME Ecosystem Map — Live2D, Diffusion, ComfyUI, VTubing

**Date:** 2026-04-12

---

## Summary

FLAME is heavily used in research but barely present in production tooling. The ecosystem is split: research papers use FLAME as a principled parametric rig; production tools (VTubing, ComfyUI) use faster/simpler landmark-based systems. The bridges that do exist are mostly one-directional (photo → FLAME params) not generative (FLAME params → photo).

---

## FLAME + Live2D

**Gap: essentially nothing.**

Live2D is 2D skeletal/mesh animation driven by ~50-100 named scalar parameters (`PARAM_EYE_L_OPEN`, `PARAM_MOUTH_OPEN_Y`, etc.). Production face tracking for Live2D uses OpenSeeFace (MobileNetV3 landmark detector, CPU, 30-60 FPS) or ARKit (iPhone, 52 blendshapes, 60 FPS). Neither is FLAME-based.

The mismatch is fundamental: FLAME is 100-component PCA over 3D vertex positions; Live2D is ~50 hand-named 2D transform scalars. Projecting FLAME expression space into Live2D parameter space requires a learned mapping trained on paired data. No published tool provides this. No research paper addresses this specific conversion.

**Bottom line:** FLAME → Live2D does not exist anywhere in the public tooling.

---

## FLAME + Diffusion

The most active quadrant. Production-ready options exist.

### Arc2Face + Expression Adapter (ECCV 2024 + ICCVW 2025)
**Architecture:** ArcFace identity embedding (512-d) + FLAME blendshape coefficients (via SMIRK extraction) as two independent cross-attention channels in Stable Diffusion. Identity and expression are genuinely orthogonal.
**Input:** Reference face image (for ArcFace embedding) + target expression image (for FLAME coefficient extraction via SMIRK).
**Output:** Photorealistic face of same identity with target expression.
**Code:** https://github.com/foivospar/Arc2Face + https://huggingface.co/FoivosPar/Arc2Face
**Status:** Production-ready, public weights.

### RigFace (arXiv Feb 2025)
**Architecture:** Fully fine-tuned SD 1.5 with DECA/FLAME-derived 3D renders + expression coefficients + masks as conditioning. Separate Identity Encoder (full UNet clone). Controls expression, pose, and lighting independently.
**Output:** Photorealistic edited face at ~1-2s/image.
**Code:** https://github.com/weimengting/RigFace + https://huggingface.co/mengtingwei/rigface
**Status:** Production-ready, public weights (released June 2025).

### MorphFace (CVPR 2025)
**Architecture:** 3DMM-guided diffusion with context blending — identity styles emphasized at early denoising timesteps, expression styles at later timesteps.
**Purpose:** Synthetic training data generation for face recognition (not interactive editing).
**Status:** Research, code availability unconfirmed.

### Multimodal 3D Face Geometry (arXiv 2024)
**Architecture:** 403-d FLAME parameter vector → 3-layer MLP with Leaky ReLU → cross-attention conditioning in UNet. Accepts FLAME params directly as input vector, not rendered image.
**Output:** 3D facial geometry (mesh), not 2D image.
**Status:** Research.

### The End-to-End Pipeline (manually assembled, no single tool)
```
Photo → SMIRK/EMOCA → FLAME params
      → edit params (change expression, pose)
      → Arc2Face / RigFace → new photorealistic face
~2-3 seconds total, Python glue code required
```

---

## FLAME + ComfyUI

### What Exists

**Pixel3DMM** ([A043-studios/comfyui-pixel3dmm](https://github.com/A043-studios/comfyui-pixel3dmm)) — the only native FLAME integration in ComfyUI Manager. Nodes: `FaceReconstructor3D`, `Pixel3DMMLoader`. Does 2D image → FLAME parameter extraction → 3D mesh with UV/normals. Config uses FLAME dimension 101. Exports OBJ/PLY/STL. **Reconstruction only — extracts FLAME params from images, does not generate images from FLAME params.**

### What Does NOT Exist
No ComfyUI node that takes FLAME parameters as *input* and generates or edits a face image. To use RigFace or Arc2Face+expression inside ComfyUI requires a custom node (Python wrapper, ~200 lines) that doesn't exist in any public repository.

### The ComfyUI Face Tooling Landscape (all FLAME-free)

| Node / Package | What It Does | Face Representation |
|---|---|---|
| ComfyUI-LivePortraitKJ | Portrait animation from driving video | InsightFace / MediaPipe keypoints |
| ComfyUI-AdvancedLivePortrait | Fine-grained expression sliders (blink, brow, mouth, pupils) | Proprietary latent space |
| ComfyUI_InstantID | ID-consistent face generation | InsightFace antelopev2 embeddings |
| ComfyUI_IPAdapter_plus | Reference image style transfer | CLIP image embeddings |
| comfyui_controlnet_aux (MediaPipe) | Face mesh ControlNet preprocessing | MediaPipe 468 landmarks |
| ComfyUI-ReActor | Face swapping | InsightFace |

**Pattern:** ComfyUI's face ecosystem converged on InsightFace embeddings and MediaPipe landmarks. FLAME is absent. The 1.2M ComfyUI users have no FLAME generation node available.

---

## FLAME + VTubing

**Not present in production at all.**

### Production VTuber Stack

```
Face capture: ARKit (iPhone) / MediaPipe (webcam) / OpenSeeFace (CPU)
      ↓
Middleware: VTube Studio ($24.99) / VSeeFace (free)
      ↓
Avatar: Live2D model / VRM 3D model
      ↓
Streaming: OBS
```

FLAME appears nowhere. Reasons:
1. **Speed**: FLAME fitting requires per-frame optimization — 100-500ms. VTubers need ≤50ms. OpenSeeFace does 30-60 FPS on CPU; ARKit does 60 FPS on-device.
2. **No integration**: VTube Studio and VSeeFace have no FLAME input path.
3. **No need**: ARKit 52 blendshapes → Live2D parameters is well-solved without FLAME as intermediary.

### Production Tools

| Tool | Type | Tracking backend | Status |
|---|---|---|---|
| VTube Studio | 2D Live2D driver | OpenSeeFace / ARKit / NVIDIA | Production ($24.99) |
| VSeeFace | 3D VRM driver | OpenSeeFace | Production (free) |
| iFacialMocap | iOS capture app | ARKit | Production (paid) |
| Kalidoface 3D | Web-based VRM | MediaPipe | Production (free) |

### Research Prototypes (FLAME-based, not in production)

| Project | What | Speed | Status |
|---|---|---|---|
| StreamME | Real-time 3DGS avatar with FLAME rigging from live video | ~5min optimization then real-time | Research |
| SPARK | Self-supervised real-time FLAME capture + neural rendering | Real-time target | Research (GitHub: kelianb/SPARK) |
| PrismAvatar (arXiv 2502) | Neural head avatar at 60 FPS on mobile | 60 FPS on device | Research pre-print |
| HeadStudio | Text → FLAME-rigged 3DGS avatar at 40 FPS | 2h generation, 40 FPS render | Research |

### The Missing Bridge

The complete VTuber pipeline that doesn't exist:
```
ARKit 52 blendshapes (60 FPS iPhone)
    → solve to FLAME coefficients (~1ms, doable)
    → drive pre-built 3DGS avatar (Arc2Avatar path)
    → 3D Gaussian Blendshapes render (370 FPS)
    → photorealistic real-time VTuber avatar
```

All components exist in research code. No product packages them. MediaPipe→FLAME converter exists ([PeizhiYan/mediapipe-blendshapes-to-flame](https://github.com/PeizhiYan/mediapipe-blendshapes-to-flame)). 3D Gaussian Blendshapes code exists (SIGGRAPH 2024). Arc2Avatar (single image → 3DGS) code exists. The integration is engineering, not research.

---

## The Gap Summary

| Axis | Current State | Gap |
|---|---|---|
| FLAME → Live2D | Nothing | Full gap — no published converter |
| FLAME → Diffusion (generation) | Arc2Face + RigFace, production-ready | Integration glue needed; no single-tool pipeline |
| FLAME → ComfyUI (generation) | Pixel3DMM (reconstruction only) | Generation node needed (~200 lines Python) |
| FLAME → VTubing | Research prototypes only | Product packaging + ARKit integration needed |
| ARKit → FLAME | Python library exists | Speed tuning for real-time needed |
