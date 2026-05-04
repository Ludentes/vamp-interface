---
status: live
topic: neural-deformation-control
---

# Neural Deformation for Blendshape Control — Practical Recipe (2026-05-03)

Goal: add ARKit-52 blendshape-driven expression control to existing Flux.1-Krea 512² portraits in 1–2 hours, without training per-axis sliders.

Scope: pick a tool we can install today, drive it from a 52-d MediaPipe blendshape vector, and chain it after our existing Flux render to produce a deformed face. Verdict at top, then per-tool recipes.

## TL;DR verdict

**Install AdvancedLivePortrait + LivePortraitKJ (Kijai) under our existing ComfyUI.** Two hours from zero to working. ~12 sliders close to ARKit semantics (blink, wink, eyebrow, pupil_x/y, aaa/eee/woo, smile, rot_pitch/yaw/roll). MediaPipe → 12-axis bridge has to be written by us (~50 LOC) since neither node ships an ARKit-52→slider mapper. Identity drift is the main risk and is real but characterizable. The `Bacis_Liveportrait_with_expression_editing.json` workflow is the starting point.

Falls back: Follow-Your-Emoji if we want continuous landmark control (heavier, video-driven), PersonaLive if we eventually need a streaming live loop (12 GB, Apache-2.0, landmark-driven). X-Portrait, IM-Animation, FG-Portrait are all video-driven academic releases — none accept a blendshape vector directly.

---

## LivePortrait + AdvancedLivePortrait — install & run

### Install

ComfyUI lives at `/home/newub/w/ComfyUI/`. Two custom node packs needed:

```bash
cd /home/newub/w/ComfyUI/custom_nodes
git clone https://github.com/kijai/ComfyUI-LivePortraitKJ
git clone https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait
cd ComfyUI-LivePortraitKJ && pip install -r requirements.txt
cd ../ComfyUI-AdvancedLivePortrait && pip install -r requirements.txt
```

LivePortraitKJ is the engine (warping module + image generator + stitcher). AdvancedLivePortrait provides the per-slider expression editor on top.

### Models (auto-downloaded)

LivePortraitKJ auto-pulls from `https://huggingface.co/Kijai/LivePortrait_safetensors` into `ComfyUI/models/liveportrait/` on first run. ~500 MB total (motion extractor, appearance extractor, warping, generator, stitching networks). Folder structure documented at the LivePortraitKJ README.

Face detector choice (LivePortraitKJ README quote): *"Insightface is strictly for NON-COMMERCIAL use. MediaPipe is a bit worse at detection, and can't run on GPU in Windows, though it's much faster on CPU compared to Insightface."* On Linux + 4090 the GPU restriction does not apply. Pick MediaPipe for license cleanliness.

### Sample workflow to load first

`/home/newub/w/ComfyUI/custom_nodes/ComfyUI-AdvancedLivePortrait/sample/workflows/Bacis_Liveportrait_with_expression_editing.json` — loads a portrait, runs through LivePortrait, exposes `Expression Editor (PHM)` sliders, outputs a deformed image. This is the minimal "static portrait → 12-axis slider vector → deformed portrait" workflow.

Other workflows in the same folder:
- `Bacis_Simple_expression_editing.json` — pure slider edit, no animation
- `Bacis_Extracting_expression_from_photo.json` — read sliders out of a face (useful for calibration)
- `Advanced_Animate_without_vid.json` — animate from a sequence of saved exp_data .pkl files
- `Advanced_Inserting_expressions_into_vid.json` — video-driven

**Concrete next step you can run today.** `cd /home/newub/w/ComfyUI && pnpm/uv-launched-comfy`, drag `Bacis_Liveportrait_with_expression_editing.json` into the canvas, replace the source image with a Flux portrait from `output/`, hit queue. ~1–2 s/render on 4090, <2 GB additional VRAM beyond Flux.

---

## ARKit-52 → AdvancedLivePortrait slider mapping

### What the node actually exposes

Verified from `nodes.py` in PowerHouseMan/ComfyUI-AdvancedLivePortrait (main branch):

| Slider | Min | Max | Step | ARKit channels (suggested mapping) |
|---|---|---|---|---|
| rotate_pitch | -20 | 20 | 0.5 | head pose pitch (not in 52-d, separate) |
| rotate_yaw | -20 | 20 | 0.5 | head pose yaw |
| rotate_roll | -20 | 20 | 0.5 | head pose roll |
| blink | -20 | 5 | 0.5 | `eyeBlinkLeft`+`eyeBlinkRight` average (negative = wide-open, 0 = neutral, +5 = closed) |
| eyebrow | -10 | 15 | 0.5 | `browInnerUp` minus `browDown_L/R` |
| wink | 0 | 25 | 0.5 | `eyeBlinkLeft - eyeBlinkRight` (signed asymmetry) |
| pupil_x | -15 | 15 | 0.5 | `eyeLookOut/InLeft+Right` combined |
| pupil_y | -15 | 15 | 0.5 | `eyeLookUp/Down*` combined |
| aaa | -30 | 120 | 1 | `jawOpen` (×100) |
| eee | -20 | 15 | 0.2 | `mouthStretchLeft/Right` |
| woo | -20 | 15 | 0.2 | `mouthFunnel`, `mouthPucker` |
| smile | -0.3 | 1.3 | 0.01 | `mouthSmileLeft+Right` average |

### Is there ARKit conversion code in the repo?

**No.** Verified by reading `nodes.py` — the `calc_fe()` method maps slider values to internal keypoint indices (0–20) and spatial dims (0–2). There is no ARKit name-lookup table, no MediaPipe blendshape ingestion. We have to write the bridge.

Coverage: ~12 sliders cover roughly 20–25 of the 52 ARKit channels well (eye open/close, gaze, jaw, smile, brow, pucker/funnel, head pose). Channels with no good slider home: `cheekPuff`, `cheekSquint*`, `noseSneer*`, `mouthDimple*`, `mouthFrown*`, `mouthRollLower/Upper`, `mouthShrugLower/Upper`, `tongueOut`. These will silently no-op if we send them.

For squint specifically: **AdvancedLivePortrait has no dedicated squint slider.** `eyeSquintLeft/Right` from ARKit needs to bridge to a combination of `blink` (small positive) + `eyebrow` (slight negative) — empirical fit needed. This is relevant given our active squint-slider thread.

**Concrete next step you can run today.** Write `src/blendshape_bridge/arkit_to_alp.py` (~50 LOC): load MediaPipe blendshape output from a webcam frame, apply the table above with empirically-tuned scale factors, emit a dict of slider values, POST as ComfyUI workflow params via the `/prompt` REST endpoint.

---

## Alternative tools — fitness summary

| Tool | License | VRAM | Direct blendshape input? | Verdict |
|---|---|---|---|---|
| LivePortrait (Kijai+PHM) | MIT (code), non-commercial Insightface; MediaPipe option | ~2 GB | No — slider bridge needed | **Pick this.** Cheapest install, immediate visual results. |
| Follow-Your-Emoji | not listed in README — verify before commercial use | not stated; 512² works on RTX 4090 per repo | No — landmark-driven (.npy from MediaPipe video) | Heavier diffusion model. Better for cartoon/stylized faces. Multi-second inference. |
| X-Portrait | not stated | "max 512×512", DDIM steps configurable, LCM-LoRA option | No — driving video required | Source-and-driving-video architecture. Wrong shape for "static blendshape vector → frame". |
| FG-Portrait (CVPR 2026) | n/a | n/a | No — driving video, 3D-flow guided | **Code release status: not found.** arXiv only. Skip until repo appears. |
| IM-Animation (arXiv:2602.07498) | unverified | unverified | unverified | **Did not surface a public repo in search.** Mark unconfirmed; do not plan around it. |
| PersonaLive (CVPR 2026) | Apache-2.0 | 12 GB (streaming) | No — landmark-driven | Best long-term play for "live webcam → animated Flux portrait" loop. ComfyUI wrapper exists at `okdalto/ComfyUI-PersonaLive`. ~7–22× faster than prior diffusion animators per paper. |
| Follow-Your-Emoji-Faster (IJCV 2025) | same as base | better than base | No | Faster variant of FYE, ~2× speedup. |

**Concrete next step you can run today.** Don't install any of these yet. Confirm the LivePortrait recipe works first. If smile/squint sliders prove too coarse for our finegrained sus-encoding palette, escalate to PersonaLive (landmark control surface is richer than 12 sliders).

---

## End-to-end "Flux portrait → LivePortrait deform → output" workflows

Two community workflows confirmed:

1. **OpenArt: FLUX Live Portrait Generator** (sekinvr) — chains Flux generation with LivePortrait expression editing. URL: `https://openart.ai/workflows/sekinvr/flux-live-portrait-generator/imxyow52lzAHWOzjNU0A`. Couldn't fetch the JSON directly (permission denied), but the listing confirms the chain exists and is downloadable.
2. **Civitai: FLUX Live Portrait Generator v1.0** — `https://civitai.com/models/642104/flux-live-portrait-generator`. Same author family, same chain.

For our pipeline, the chain is just two nodes plumbed in series in the same workflow: Flux KSampler → VAE Decode → `Expression Editor (PHM)` → SaveImage. Source image input on the editor is the Flux output. No special bridging needed at the ComfyUI level — only seed/prompt management.

**Concrete next step you can run today.** Take our existing Flux portrait workflow, add the AdvancedLivePortrait `Expression Editor (PHM)` node downstream of VAE Decode, wire the image. Manually slide `smile` to test that perturbation appears. ~30 min sanity check.

---

## Quality loss when warping a Flux render through LivePortrait

No controlled identity-drift benchmark surfaced in search for "Flux → LivePortrait" specifically. What I did find:

- **FLUXSynID (arXiv:2505.07530)** — uses LivePortrait to generate live-capture variants of document-style identity images. Quote (paraphrased from search summary): "LivePortrait closely matches high identity consistency with minimal appearance variation." This is the closest thing to a positive controlled report — but FLUXSynID's source images are flat document photos, not Flux-generated photoreal portraits, so transfer to our case is not guaranteed.
- **Common Face-Sim convention** (zenn.dev/taku_sid Face-Sim primer): face-swap pipelines are considered "high quality" at ArcFace cosine ≥ 0.8, aging applications at ≥ 0.7. Use these as our pass/fail thresholds.
- Anecdotal: stablediffusiontutorials.com and several Medium tutorials on LivePortrait note eye-region artifacts at high `blink` values and texture-bleed in the mouth interior at extreme `aaa` values. No numbers.

**No specific Flux→LivePortrait identity-drift benchmark was located. Mark this as unverified.**

**Concrete next step you can run today.** Burn 30 min on our own micro-benchmark: pick 20 Flux portraits from our existing corpus, run each through Expression Editor at 5 slider settings (neutral, smile=0.5, smile=1.0, blink=3, jawOpen=80), measure ArcFace cosine vs original using our existing `insightface buffalo_l` setup. If median drops below 0.7, LivePortrait warping is too destructive for our application and we escalate to PersonaLive or Follow-Your-Emoji which have stronger appearance encoders.

---

## Closed-loop "MediaPipe webcam → blendshape → LivePortrait sliders → animated Flux portrait"

FasterLivePortrait (`warmshao/FasterLivePortrait`) gives us the realtime engine: 30+ FPS on RTX 3090 (per README) end-to-end including pre/post, MIT license, MediaPipe option built in (`--mp` flag, `configs/trt_mp_infer.yaml`, `python run.py --src_image X --dri_video 0 --cfg configs/trt_mp_infer.yaml`).

But the bridge from "MediaPipe blendshape vector → AdvancedLivePortrait slider vector" is the missing public piece. FasterLivePortrait's MediaPipe path uses MediaPipe for **face detection / landmark extraction**, not for blendshape readout — the blendshape semantics never enter the warp.

**No public code or write-up demonstrates the full closed loop "MediaPipe blendshape → ALP slider → animated source" was located.** This is greenfield.

The realistic build path:
1. MediaPipe Face Landmarker task in Python emits 52-d blendshape vector at 30 FPS (verified — Google maintains this since April 2023, see `face_blendshapes_graph.cc`).
2. Apply our ARKit-52→12-slider bridge (the table above).
3. Drive AdvancedLivePortrait via direct Python import (skip ComfyUI overhead) using `motion_extractor` + `warping_module` + `image_generator` from LivePortraitKJ's bundled engine.
4. Push to a webrtc / cv2 window.

Alternatively: skip the slider abstraction, use FasterLivePortrait's existing implicit-keypoint motion-transfer path — feed it a webcam stream as `dri_video`, and our control surface becomes "the user's actual face on webcam" rather than "the blendshape vector". This is what FasterLivePortrait's `--realtime` flag already does. The blendshape vector is then only useful if we want a *programmatic* (non-webcam) driver, e.g., reading blendshapes from a TTS-driven model or from our `sus_factors`.

For our actual use case (sus_factors → blendshape vector → portrait), the webcam is irrelevant. We never need a true closed loop. We need: `sus_factors_16d → arkit_52d (learned) → alp_12_sliders (table) → warped Flux portrait`. The middle link is the only research piece.

**Concrete next step you can run today.** Skip the closed-loop framing. Build the offline pipeline first: `sus_factors → arkit_52d → 12 sliders → 1 frame`. Verify visual continuity (close points → close faces) holds through the warp. Only then think about realtime.

---

## Two-hour install order

1. (15 min) `git clone` both custom nodes, `pip install -r requirements.txt`, restart ComfyUI. Models auto-download on first run.
2. (15 min) Open `Bacis_Liveportrait_with_expression_editing.json`, swap source image to a Flux render, manually verify a `smile=0.8` slider produces a visible smile.
3. (30 min) Write `src/blendshape_bridge/arkit_to_alp.py` from the table above. Sanity-test with a hand-constructed `eyeSquintLeft=0.8, mouthSmileLeft=0.7` vector → expect ALP `blink≈+1.5, smile≈0.7`.
4. (30 min) Run our 20-portrait identity-drift micro-benchmark with insightface buffalo_l. Threshold pass/fail at cosine 0.7.
5. (30 min buffer) Iterate on slider scaling factors against 3–4 ARKit reference exemplars (open mouth, big smile, squint+brow-down, wink).

If step 4 fails, log the failure and pivot to PersonaLive — but only after confirming the Flux→ALP path is actually unrecoverable, not just under-tuned.

---

## Sources

- [PowerHouseMan/ComfyUI-AdvancedLivePortrait](https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait) — main repo; `nodes.py` confirms slider ranges.
- [PowerHouseMan ALP nodes.py (slider definitions, calc_fe)](https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait/blob/main/nodes.py)
- [PowerHouseMan ALP sample workflows directory](https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait/tree/main/sample/workflows)
- [kijai/ComfyUI-LivePortraitKJ](https://github.com/kijai/ComfyUI-LivePortraitKJ) — the underlying engine wrapper, MediaPipe option.
- [Kijai/LivePortrait_safetensors on HuggingFace](https://huggingface.co/Kijai/LivePortrait_safetensors/tree/main) — auto-downloaded model checkpoints.
- [KlingAIResearch/LivePortrait](https://github.com/KlingAIResearch/LivePortrait) — original Kuaishou release, paper `arXiv:2407.03168`.
- [warmshao/FasterLivePortrait](https://github.com/warmshao/FasterLivePortrait) — TensorRT realtime port, MIT, MediaPipe support.
- [mayuelala/FollowYourEmoji](https://github.com/mayuelala/FollowYourEmoji) — landmark-driven diffusion animator (SIGGRAPH Asia 2024 / IJCV 2025).
- [Follow-Your-Emoji project page](https://follow-your-emoji.github.io/)
- [bytedance/X-Portrait](https://github.com/bytedance/X-Portrait) — SIGGRAPH 2024 driving-video animator, Python 3.9 + CUDA 11.8.
- [akatz-ai/ComfyUI-X-Portrait-Nodes](https://github.com/akatz-ai/ComfyUI-X-Portrait-Nodes) — community ComfyUI wrapper.
- [GVCLab/PersonaLive](https://github.com/GVCLab/PersonaLive) — CVPR 2026, Apache-2.0, 12 GB streaming inference.
- [okdalto/ComfyUI-PersonaLive](https://github.com/okdalto/ComfyUI-PersonaLive) — community ComfyUI wrapper.
- [PersonaLive paper (arXiv:2512.11253)](https://arxiv.org/abs/2512.11253)
- [FG-Portrait (arXiv:2603.23381)](https://arxiv.org/abs/2603.23381) — code release not found.
- [Follow-Your-Emoji (arXiv:2406.01900)](https://arxiv.org/abs/2406.01900)
- [Follow-Your-Emoji-Faster (arXiv:2509.16630)](https://arxiv.org/html/2509.16630)
- [MediaPipe face_blendshapes_graph.cc](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/cc/vision/face_landmarker/face_blendshapes_graph.cc) — confirms 52-channel ARKit-aligned output.
- [yeemachine/kalidokit](https://github.com/yeemachine/kalidokit) — reference implementation of MediaPipe→blendshape→VRM-style mapping (TS, but the math transfers).
- [py-feat/mp_blendshapes on HuggingFace](https://huggingface.co/py-feat/mp_blendshapes)
- [ARKit blendshape catalog (Apple docs)](https://developer.apple.com/documentation/arkit/arfaceanchor/blendshapelocation)
- [arkit-face-blendshapes.com](https://arkit-face-blendshapes.com/) — visual reference for the 52 channels.
- [OpenArt: FLUX Live Portrait Generator workflow](https://openart.ai/workflows/sekinvr/flux-live-portrait-generator/imxyow52lzAHWOzjNU0A)
- [Civitai: FLUX Live Portrait Generator v1.0](https://civitai.com/models/642104/flux-live-portrait-generator)
- [RunComfy: Advanced Live Portrait workflow](https://www.runcomfy.com/comfyui-workflows/advanced-live-portrait-workflow-in-comfyui)
- [comfyai.run: Basic Liveportrait with Expression Editing workflow](https://comfyai.run/download/workflow/Basic%20Liveportrait%20with%20Expression%20Editing/ec45fd5f-ed63-0aa8-90ff-be0446daffb9)
- [FLUXSynID (arXiv:2505.07530)](https://arxiv.org/html/2505.07530v2) — uses LivePortrait for ID-preserving augmentation; the only quasi-benchmark of LivePortrait identity preservation found.
- [Face-Sim primer (zenn.dev)](https://zenn.dev/taku_sid/articles/20250511_face_sim_metric?locale=en) — convention for ArcFace cosine pass thresholds.
- [Animating Portraits with LivePortrait in ComfyUI (Medium)](https://medium.com/@emabyte/animating-portraits-with-liveportrait-in-comfyui-a-step-by-step-guide-d9e5bc37b806) — community walkthrough.
- [stablediffusiontutorials.com LivePortrait guide](https://www.stablediffusiontutorials.com/2024/07/live-portrait.html)
