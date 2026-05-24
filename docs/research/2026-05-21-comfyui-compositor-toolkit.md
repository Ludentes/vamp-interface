---
status: live
topic: photobooth-sweep
---

# Research: ComfyUI compositor toolkit (detection, segmentation, background removal, mask, layout, composite)

**Date:** 2026-05-21
**Sources:** 12 sources covering the six axes the group-photobooth pipeline needs. Authoritative: the four GitHub repos for ComfyUI-Impact-Pack, ComfyUI_LayerStyle, ComfyUI-RMBG, ComfyUI_essentials; secondary: comfyai.run / runcomfy.com / instasd.com node-reference indexes.

## Executive summary

Two packs cover most of what the group-photobooth needs: **ltdrdata's ComfyUI-Impact-Pack** for detection + mask-from-detector + mask-arithmetic, and **chflame163's ComfyUI_LayerStyle** for Photoshop-style compositing (`ImageBlendAdvance`, `DropShadow`, `ColorAdapter`) [3,5]. Both are actively maintained as of mid-2026. For SAM2 segmentation, **kijai/ComfyUI-segment-anything-2** is the canonical pack [2]. For background removal, **1038lab/ComfyUI-RMBG** is the umbrella node that ships every modern matter (RMBG-2.0, INSPYRENET, BiRefNet variants, BEN/BEN2, SDMatte, SAM2/3, GroundingDINO) under one interface and was last updated v3.0.0 on 2026-01-01 [4]. For per-person human body parsing (head/torso/arms/legs masks), **cozymantis/human-parser-comfyui-node** runs CPU-fine and ships LIP/Pascal/ATR weights [6]. ComfyUI_essentials (cubiq) went maintenance-only on 2025-04-14 — usable, but not a fresh dependency [7]. Masquerade Nodes (BadCafeCode) is officially deprecated by its author in favor of Impact Pack [5].

## Findings by axis

### Detection: GroundingDINO + YOLO via Impact-Pack or LayerStyle

The community has converged on two patterns. Impact-Pack ships **BBOX detectors** (Ultralytics-YOLO-based) and **SAM detectors** that emit SEGS — a SEG object containing bbox, mask, label, confidence — and a downstream `SAM2 Video Detector (SEGS)` for tracked segmentation [3]. LayerStyle bundles YOLO8, YOLO-World, Florence-2, and Qwen as model dependencies for its own detection-aware nodes [1]. The dedicated **PozzettiAndrea/ComfyUI-Grounding** pack exposes a Model Loader with 19+ heads including GroundingDINO, MM-GroundingDINO, OWLv2, Florence-2, YOLO-World, and a Florence-2/SA2VA mask path — useful when the input scene needs text-prompted detection (`"person"`) rather than a fixed YOLO class index [8].

For the group-photobooth: a YOLO-person BBOX detector from Impact-Pack is the cheapest path; GroundingDINO is overkill for the "person" class but useful later if we need to detect props (chair, table) to constrain matryoshka placement.

### Segmentation: SAM2 via Kijai pack, Impact-Pack for SEGS interop

**kijai/ComfyUI-segment-anything-2** is the install both the ComfyUI Manager and the InstaSD / RunDiffusion guides surface first [2]. Two less-common forks exist (neverbiasu/ComfyUI-SAM2, MicheleGuidi/ComfyUI-Contextual-SAM2 for Florence-2-bbox → SAM2-mask chaining) but Kijai is the canonical [2]. Models auto-download to `ComfyUI/models/sam2/` on first run. SAM2 inside ComfyUI-RMBG is also available, so for a pipeline that already pulls RMBG, no separate SAM2 install is needed [4].

Impact-Pack's `MASK to SEGS` and `SEGSToMaskList` bridge between the SEG (Impact's tagged-mask object) and plain ComfyUI MASK tensors, which is the glue layer when chaining detection → segmentation → mask ops → composite [3].

### Background removal: ComfyUI-RMBG (umbrella) with INSPYRENET or BiRefNet-portrait for humans

**1038lab/ComfyUI-RMBG** is the dominant pack — it ships RMBG-2.0, BiRefNet (general / portrait / matting / HR / lite), INSPYRENET, BEN/BEN2, SDMatte, SAM, SAM2, SAM3, and GroundingDINO behind a single node interface [4]. The pack's own docs single out **INSPYRENET** as "specialized in human portrait segmentation" with fast processing and good edge detection, and **BiRefNet-portrait** as the alternative when matting-quality edges matter [4]. v3.0.0 was released 2026-01-01, confirming active maintenance.

Licensing nuance: **BRIA RMBG-2.0** is built on BiRefNet but trained on a proprietary 15k-image dataset by BRIA AI [9]. RMBG-2.0 weights have a non-commercial research license that requires a BRIA commercial agreement for production use — verify before shipping any product-facing render. The plain **BiRefNet** weights are MIT-equivalent and have no such caveat. INSPYRENET is also research-permissive.

For our pipeline, doll portraits (Z-Image renders against flat-ish backgrounds) cut cleanly with INSPYRENET, BiRefNet-portrait, or even u2net (the rembg default). The Z-Image background is uniform enough that the model choice mostly affects edge softness, not topology.

### Mask tools: Impact-Pack is the canonical, LayerStyle for advanced edge work

Impact-Pack provides every mask operation the group-photobooth needs and is actively maintained — 888 commits, V8.24 current, GPL-3.0 [3]. The relevant nodes:
- **Dilate Mask** — supports negative values for erosion
- **Gaussian Blur Mask** — for feathering
- **Pixelwise AND/SUB/ADD** — boolean composition between masks
- **MASK ↔ SEGS** — interop between plain masks and detection-pack SEGS

LayerStyle's LayerMask group complements this with **MaskEdgeUltraDetail / V3** (refines mask edges via VitMatte), **MaskGrow**, **MaskEdgeShrink**, and **CropByMaskV2** [1]. Use these when matting-quality edges matter (closeup) rather than topology (bulk silhouette work).

Masquerade Nodes (BadCafeCode) is still installable and ships `Paste By Mask`, but its README now explicitly recommends migrating to Impact Pack: "the creator recommends using Impact Pack instead as it's a more feature-rich and well-maintained alternative" [5]. Treat Masquerade as legacy.

### Layout / placement: LayerStyle ImageBlendAdvance is the answer

**chflame163/ComfyUI_LayerStyle** is the closest thing ComfyUI has to a Photoshop compositor [1]. **ImageBlendAdvance** is the node that does what the group-photobooth needs in a single call: "composites layer images of different sizes with position control via x/y percentage, scale, rotation, and multiple blend modes" — it handles the entire "scale doll, place at (x, y) on background, with alpha + blend" step [1]. MIT licensed, 542 commits, active maintenance.

For finer control there is also `ImageCompositeHandleMask` with feathering + crop-region output, and `DropShadow` with offset/blur/growth/color [1]. The drop-shadow node maps 1:1 onto the spec's optional polish stage.

Alternatives — `mikey_nodes/ImagePaste`, `was-node-suite-comfyui/Image-Paste-Face`, `comfyui-art-venture/ImageAlphaComposite` — exist but are narrower in scope and don't bundle the polish ops. Core ComfyUI's built-in `Image Composite Masked` (and `PorterDuffImageComposite`) cover simple alpha compositing without scale/rotation but are dependency-free [10].

### Compositing polish: LayerStyle ColorAdapter + DropShadow

For the optional polish stage in the group-photobooth spec, LayerStyle provides direct equivalents [1]:
- **ColorAdapter** — automatic color-tone adjustment to a reference image. Hands the "Lab match doll → background palette" job over without writing it ourselves.
- **LAB** node — manual L*a*b channel adjustment when ColorAdapter is too aggressive.
- **DropShadow** — exactly the foot-anchor shadow the spec asked for; offset/blur/grow/color all exposed.

So polish can be implemented purely as a few extra LayerStyle nodes after `ImageBlendAdvance` per doll.

### Human body parsing (optional): Cozy Human Parser, runs on CPU

If we ever need per-region masks (head / torso / arms / legs) — e.g. for chibi when pose matters — **cozymantis/human-parser-comfyui-node** ships three trained heads (LIP / Pascal / ATR) and runs "on both CPU and CUDA" while being "fast, VRAM-light" [6]. Pascal gives big regions (head/torso/arms/legs), LIP and ATR give finer clothing splits. metal3d/ComfyUI_Human_Parts is an alternative using DeepLabV3+ ResNet50 [6]. Not needed for matryoshka v1 — body silhouette is enough — but it's the natural node to plug in for chibi pose-aware composition later.

## Comparison

| Axis | Recommended pack | Repo | License | Status | Key nodes for us |
|---|---|---|---|---|---|
| Detection (person) | ComfyUI-Impact-Pack | ltdrdata/ComfyUI-Impact-Pack [3] | GPL-3.0 | Active (V8.24) | UltralyticsDetectorProvider, BBOX Detector (SEGS) |
| Detection (text-prompt) | ComfyUI-Grounding | PozzettiAndrea/ComfyUI-Grounding [8] | (unstated) | Active 2025 | GroundingDINO / YOLO-World loaders |
| Segmentation | ComfyUI-segment-anything-2 | kijai/ComfyUI-segment-anything-2 [2] | Apache-2.0 (Kijai default) | Active | Sam2Segmentation |
| Background removal | ComfyUI-RMBG | 1038lab/ComfyUI-RMBG [4] | (per-model; INSPYRENET / BiRefNet permissive, RMBG-2.0 non-commercial) | Active (v3.0.0, 2026-01-01) | INSPYRENET or BiRefNet-portrait |
| Mask arithmetic | ComfyUI-Impact-Pack | ltdrdata/ComfyUI-Impact-Pack [3] | GPL-3.0 | Active | Dilate Mask, Gaussian Blur Mask, Pixelwise AND/SUB/ADD |
| Mask edge refinement | ComfyUI_LayerStyle | chflame163/ComfyUI_LayerStyle [1] | MIT | Active | MaskEdgeUltraDetailV3 |
| Layout + composite | ComfyUI_LayerStyle | chflame163/ComfyUI_LayerStyle [1] | MIT | Active | ImageBlendAdvance, ImageCompositeHandleMask |
| Polish (color/shadow) | ComfyUI_LayerStyle | chflame163/ComfyUI_LayerStyle [1] | MIT | Active | ColorAdapter, DropShadow |
| Human body parsing | human-parser-comfyui-node | cozymantis/human-parser-comfyui-node [6] | (research, unstated) | Active 2025 | Cozy Human Parser Pascal / LIP / ATR |
| Avoid / legacy | ComfyUI_essentials | cubiq/ComfyUI_essentials [7] | MIT | Maintenance-only since 2025-04-14 | — |
| Avoid / legacy | masquerade-nodes-comfyui | BadCafeCode/masquerade-nodes-comfyui [5] | — | Author defers to Impact Pack | — |

## Minimal install path for the group-photobooth

If we proceed with Approach A in the group-photobooth spec, the new `~/w/ComfyUI/custom_nodes/` adds are:

1. `git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack` — detection + mask arithmetic
2. `git clone https://github.com/kijai/ComfyUI-segment-anything-2` — SAM2 segmentation
3. `git clone https://github.com/chflame163/ComfyUI_LayerStyle` — layout + composite + polish
4. `git clone https://github.com/1038lab/ComfyUI-RMBG` — only if we want INSPYRENET/BiRefNet via a node graph instead of a Python rembg call

The Python-side alternative is to keep detection / segmentation / cutout / composite **outside** ComfyUI entirely — call `ultralytics`, `segment-anything-2`, `rembg`, and `Pillow.Image.alpha_composite` directly from our driver — and only hit ComfyUI for the Z-Image face render. That's what the spec already assumes, and matches our existing photobooth pattern (CPU helpers in Python, GPU work via the ComfyUI HTTP API). The custom-node packs above are the fallback if we want to stay inside ComfyUI graphs for the whole pipeline.

## Open questions

- **BRIA RMBG-2.0 licensing.** The pack ships the weights but BRIA's commercial-use policy needs explicit confirmation before we depend on RMBG-2.0 for any shipped product render. INSPYRENET and BiRefNet-base avoid the question entirely [4,9]. Single-source on the commercial restriction (HuggingFace model card) — not independently confirmed.
- **PozzettiAndrea/ComfyUI-Grounding license.** GitHub page does not surface a clear LICENSE file in the search snippet [8]. Worth checking the repo before depending on it.
- **CPU benchmarks.** SAM2's CPU latency at our typical resolution (1024-2048 px on the long side) is single-source — claims of "1-2 s" or "5-10 s" come from blog posts, not measured by us. If we go this route, benchmark on the actual input distribution before committing.

## Sources

[1] chflame163. "ComfyUI_LayerStyle". https://github.com/chflame163/ComfyUI_LayerStyle (Retrieved 2026-05-21)
[2] kijai. "ComfyUI-segment-anything-2". https://github.com/kijai/ComfyUI-segment-anything-2 (Retrieved 2026-05-21)
[3] ltdrdata. "ComfyUI-Impact-Pack". https://github.com/ltdrdata/ComfyUI-Impact-Pack (Retrieved 2026-05-21)
[4] 1038lab. "ComfyUI-RMBG". https://github.com/1038lab/ComfyUI-RMBG (Retrieved 2026-05-21)
[5] BadCafeCode. "masquerade-nodes-comfyui". https://github.com/BadCafeCode/masquerade-nodes-comfyui (Retrieved 2026-05-21)
[6] cozymantis. "human-parser-comfyui-node". https://github.com/cozymantis/human-parser-comfyui-node (Retrieved 2026-05-21)
[7] cubiq. "ComfyUI_essentials". https://github.com/cubiq/ComfyUI_essentials (Retrieved 2026-05-21)
[8] PozzettiAndrea. "ComfyUI-Grounding". https://github.com/PozzettiAndrea/ComfyUI-Grounding (Retrieved 2026-05-21)
[9] BRIA AI. "RMBG-2.0 model card". https://huggingface.co/briaai/RMBG-2.0 (Retrieved 2026-05-21)
[10] ComfyUI Wiki. "Image Composite Masked". https://comfyui-wiki.com/en/comfyui-nodes/image/image-composite-masked (Retrieved 2026-05-21)
[11] InstaSD. "comfyui-segment-anything-2 reference". https://www.instasd.com/comfyui/custom-nodes/comfyui-segment-anything-2 (Retrieved 2026-05-21)
[12] RunComfy. "ComfyUI Impact Pack detailed guide". https://www.runcomfy.com/comfyui-nodes/ComfyUI-Impact-Pack (Retrieved 2026-05-21)
