---
status: live
topic: arkit-controlnet
---

# InfiniteYou local asset report

**Date:** 2026-05-16
**Purpose:** Inventory of every asset needed to run the InfiniteYou ComfyUI
workflow on the local box, with current location and status. Snapshot taken
after the tight (sim-fp8-only) re-download completed.

## One-line status

**All assets are in place and the workflow is runnable.** Every model weight is
present locally and the `ComfyUI_InfiniteYou` custom node is installed (deps in
the ComfyUI venv). Only a ComfyUI restart is needed to register the nodes.

## The node (bytedance/ComfyUI_InfiniteYou)

Official ByteDance ComfyUI-native node (ICCV 2025 Highlight). It is *native* —
`InfuseNetLoader` returns a standard `CONTROL_NET`, `InfuseNetApply` chains via
`set_previous_controlnet`, so InfiniteYou **stacks with a normal Canny
ControlNet** (the doll-silhouette structure path). Node classes:

- `IDEmbeddingModelLoader` → `(FACE_DETECTOR, ARCFACE_MODEL, IMAGE_PROJ_MODEL)`
- `ExtractIDEmbedding(face_detector, arcface, image_proj, id_image)` → identity CONDITIONING
- `ExtractFacePoseImage(face_detector, image, w, h)` → 5-keypoint pose IMAGE
- `InfuseNetLoader(controlnet_name)` → `CONTROL_NET`
- `InfuseNetApply(positive, id_embedding, control_net, image, strength, start%, end%)` → `(positive, negative)`

**Status:** ✅ installed at `~/w/ComfyUI/custom_nodes/ComfyUI_InfiniteYou/`
(cloned 2026-05-17). Requirements installed into the ComfyUI venv
(`~/w/ComfyUI/.venv`): facexlib 0.3.0, insightface 0.7.3, onnxruntime 1.26.0,
opencv-python 4.13.0, huggingface_hub — all verified importable. A ComfyUI
restart is needed to register the node classes.

## Asset matrix

| Asset | Where the node expects it | Current location | Status |
|---|---|---|---|
| **ComfyUI_InfiniteYou node** | `~/w/ComfyUI/custom_nodes/ComfyUI_InfiniteYou/` | cloned + venv deps installed | ✅ installed |
| **InfuseNet sim fp8** | `models/infinite_you/sim_stage1/infusenet_sim_fp8e4m3fn.safetensors` | symlink → `data/infiniteyou_dl/.../sim_stage1/` (2.95 GB) | ✅ staged |
| **image_proj_model** | `models/infinite_you/sim_stage1/image_proj_model.bin` | symlink → `data/infiniteyou_dl/.../sim_stage1/` (338 MB) | ✅ staged |
| **insightface antelopev2** | `models/insightface/models/antelopev2/*.onnx` | already in ComfyUI tree (5 onnx); fresh copy also in `data/infiniteyou_dl/supports/` | ✅ present |
| **FLUX.1-dev fp8** (base) | `models/diffusion_models/` | `models/diffusion_models/FLUX1/flux1-dev-fp8.safetensors` (11.9 GB) | ✅ present |
| **t5xxl fp8** | `models/text_encoders/` | `models/text_encoders/t5/t5xxl_fp8_e4m3fn.safetensors` | ✅ present |
| **clip_l** | `models/text_encoders/` | `models/text_encoders/clip_l.safetensors` | ✅ present |
| **FLUX VAE** | `models/vae/` | `models/vae/FLUX1/ae.safetensors` | ✅ present |
| **FLUX Canny ControlNet** (doll structure) | `models/controlnet/` | `models/controlnet/FLUX.1/instantx-union/` (InstantX Union — canny mode) | ✅ present |
| **arcface recog weight** | facexlib cache (auto) | — | ⏳ auto-downloads on first run (`init_recognition_model('arcface')`) |

## The InfiniteYou download

`data/infiniteyou_dl/` — 3.5 GB, tight re-download (sim variant, fp8 only):

```
infu_flux_v1.0/sim_stage1/image_proj_model.bin            338 MB
infu_flux_v1.0/sim_stage1/infusenet_sim_fp8e4m3fn.safetensors  2.95 GB
supports/insightface/models/antelopev2/*.onnx             5 files
README.md, config.json
```

**Variant choice:** `sim_stage1` (not `aes_stage2`). InfiniteYou ships two —
`sim` favours identity similarity, `aes` favours text-image aesthetics. The
matryoshka problem is identity-through-a-flat-painted-style, the hardest
identity case (PuLID failed it outright), so `sim` is the right arm. `aes` was
deliberately **not** downloaded.

**fp8, not bf16:** the box is a 32 GB card. InfiniteYou BF16 peaks ~43 GB
(won't fit); fp8 peaks ~24 GB (fits). The earlier broad download was pulling
bf16 ×2 + the redundant sharded `InfuseNetModel/` dir (~40 GB of waste); it was
killed and re-run with tight `allow_patterns`.

## Base-model note

InfiniteYou's InfuseNet was trained against **FLUX.1-dev**. The local box has
`flux1-dev-fp8.safetensors` — the correct base. The matryoshka sweep used
Flux-**Krea**; do **not** reuse Krea here, the InfuseNet residuals are
dev-aligned.

## Windows box

The Windows box (3090, ComfyUI host) is kept as a **secondary test host** for
InfiniteYou — viable at modest resolution (fp8 peaks ~24 GB; the 3090 is
exactly 24 GB, so it runs with ComfyUI offloading but is tight).

The broad download had actually completed there (~40 GB, all variants). On
2026-05-17 it was **trimmed to the test kit** — `models/infinite_you/sim_stage1/`
now holds only `infusenet_sim_fp8e4m3fn.safetensors` (2.75 GB) +
`image_proj_model.bin` (0.32 GB); antelopev2 is already at
`models/insightface/models/antelopev2/`. The ~37 GB of bf16 / aes / sharded
waste, the idle `InfiniteYouDL` scheduled task, and the `dl_infiniteyou.*`
scripts in `importer/` were all removed.

Still needed before Windows testing: install the `ComfyUI_InfiniteYou` custom
node there (only `ComfyUI-PuLID-Flux` is currently in its `custom_nodes/`).

## Remaining step to runnable

Restart ComfyUI so the InfiniteYou node classes register. Every weight and the
node are already in place.
