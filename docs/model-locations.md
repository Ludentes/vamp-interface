---
status: live
topic: arkit-controlnet
---

# Model locations

Greppable inventory of model weights on this machine that the CFM training run
(and downstream code) loads. Sizes are bf16 unless noted. Update when a new
checkpoint is downloaded; do not check weights into git.

## FLUX.1-dev (HuggingFace cache)

Loaded via `from_pretrained("black-forest-labs/FLUX.1-dev", subfolder=...)`.

Root: `~/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/<rev>/`

| Subfolder | Files | Size |
|---|---|---|
| `transformer/` | 3-shard safetensors (`diffusion_pytorch_model-0000{1,2,3}-of-00003.safetensors`) + `config.json` + index | ~23 GB |
| `vae/` | safetensors + config | ~170 MB |
| `text_encoder/` (CLIP-L) | safetensors + config | ~250 MB |
| `text_encoder_2/` (T5-XXL) | sharded safetensors + config | ~9 GB |
| `tokenizer/`, `tokenizer_2/`, `scheduler/` | small JSONs | <1 MB |

Total HF cache size: ~32 GB.

## InfiniteYou (bf16, downloaded for CFM)

Root: `data/infiniteyou_dl_bf16/infu_flux_v1.0/sim_stage1/`

| Path | Class | Size | Notes |
|---|---|---|---|
| `InfuseNetModel/` (sharded safetensors + `config.json`) | diffusers `FluxControlNetModel` | 11 GB | bf16 control branch; load with `FluxControlNetModel.from_pretrained(...)`. |
| `image_proj_model.bin` | InfiniteYou `Resampler` (vendored from ComfyUI custom node) | 323 MB | ArcFace 512-d → (1, 8, 4096) identity tokens; output_dim=4096 — *not* the Resampler default 1024. |

Download recipe (idempotent — `huggingface-cli` skips existing files):

```bash
huggingface-cli download ByteDance/InfiniteYou \
    --include "infu_flux_v1.0/sim_stage1/InfuseNetModel/*" \
              "infu_flux_v1.0/sim_stage1/image_proj_model.bin" \
    --local-dir data/infiniteyou_dl_bf16/
```

## ArcFace / face detection (insightface buffalo_l)

Auto-downloaded on first `insightface.app.FaceAnalysis(name='buffalo_l')`.

Root: `~/.insightface/models/buffalo_l/`

| File | Role |
|---|---|
| `w600k_r50.onnx` | ArcFace R50 recognition (512-d embedding — the CFM identity input) |
| `det_10g.onnx` | RetinaFace 10G detection |
| `2d106det.onnx` | 106-pt 2D landmarks |
| `1k3d68.onnx` | 68-pt 3D landmarks |
| `genderage.onnx` | gender + age (unused here) |

## ComfyUI weights (separate workflow stack — *not* used by CFM training)

Mentioned for grep completeness; the CFM trainer does not touch these.

- `/home/newub/w/ComfyUI/models/diffusion_models/flux1-krea-dev.safetensors` — FLUX.1-dev finetune used by the InfuseNet spike (single-file safetensors). The real training run uses FLUX.1-dev bf16 from the HF cache above.
- `/home/newub/w/ComfyUI/models/diffusion_models/flux1-krea-dev_fp8_scaled.safetensors` — fp8 ComfyUI variant.
- `/home/newub/w/ComfyUI/models/diffusion_models/fluxFillFP8_v10.safetensors` — Flux Fill (inpaint).
- `/home/newub/w/ComfyUI/custom_nodes/ComfyUI_InfiniteYou/` — Python source for the InfiniteYou ComfyUI nodes (Resampler reference implementation; we vendor a copy under `src/arkit_controlnet/cfm/resampler.py`).

## FLAME assets

Root: `data/flame_assets/` — checked into git-LFS-style external bucket per
existing project conventions. Used by `arkit_controlnet.flame_render`.

## What is *not* yet downloaded

- `aes_stage2/` weights from InfiniteYou — the "aesthetic stage 2" alternative
  InfuseNet checkpoint. We use `sim_stage1/` (similarity stage) per the spec.
  If a future ablation needs aes, swap the `--include` glob.
