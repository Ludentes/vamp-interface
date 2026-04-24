---
status: live
topic: demographic-pc-pipeline
---

# Step 1 — Inventory & hardware budget for Flux image-pair slider trainer

Formalises what we have and what the RTX 5090 can actually run, before any LoRA-theory or training-script work.

## Assets

### Corpus (training data)

- Root: `output/demographic_pc/fluxspace_metrics/crossdemo/<axis>/<axis>_inphase/<base>/s{seed}_a{0.00,0.25,0.50,0.75,1.00}.png`
- **Shape per axis:** 6 bases × 3 seeds × 5 α = 90 images
- **Endpoint pairs per axis:** 6 × 3 = **18** (α=0, α=1)
- **Multi-scale samples per axis:** 18 × 5 = 90 (all α), enabling SDXL-image-sliders-style `scales` training
- **Axes inventory** (under `crossdemo/`):
  - Strong/confirmed (slider-queue): `eye_squint`, `gaze_horizontal`, `brow_lift`, `age`, `gender`, `hair_style`, `hair_color`
  - Re-prompt-fixed v2: `mouth_stretch_v2`, `brow_furrow_v2`
  - Dropped: `skin_smoothness`, `nose_shape`
  - Other axes from prior corpora (not in this tree): `smile`, `anger`, `surprise`, `disgust`, `pucker`, `lip_press`, `glasses`, `jaw`
- **First axis to train:** `eye_squint` — strongest signal (effect +0.90), least base-dependence, Δ ≥ 0.71 on all 6 bases.

### Measurement infrastructure (metrics → potential losses)

- **MediaPipe 52-d ARKit blendshapes** (`score_blendshapes.py`): per-image AU vector
- **SigLIP-2 probes** (`score_overnight_siglip.py`, `google/siglip2-so400m-patch16-384`): text-concept margins
- **ArcFace IR101** (identity preservation): cosine vs anchor
- **NMF k=11 atoms** on attn-feature corpus (diagnostic basis, not loss)

All are **eval-side** tools. Aux-loss integration costs 5–10× wallclock + extra VRAM, so we defer and use flow-matching velocity loss as primary.

### Flux model weights on disk

Located under `/home/newub/w/ComfyUI/models/`:

| File | Path | Size | Use |
|---|---|---|---|
| `flux1-krea-dev_fp8_scaled.safetensors` | `diffusion_models/` | ~12 GB | base transformer (fp8) |
| `flux1-dev-kontext_fp8_scaled.safetensors` | `diffusion_models/` | ~12 GB | Kontext variant |
| `t5xxl_fp16.safetensors` | `text_encoders/t5/` | ~9.5 GB | T5-XXL (bf16 equiv) |
| `t5xxl_fp8_e4m3fn.safetensors` | `text_encoders/t5/` | ~4.7 GB | T5-XXL fp8 |
| `clip_l.safetensors` | `text_encoders/` | ~0.25 GB | CLIP-L |
| `ae.safetensors` | `vae/FLUX1/` | ~0.3 GB | Flux VAE |

**Gap:** no bf16 Flux-dev/schnell transformer on disk. Concept-Sliders notebook assumes bf16 base (~24 GB weights). Two paths:

1. **Download bf16 Flux-dev** (~24 GB, HF access already set up in ComfyUI) — matches reference recipe exactly.
2. **Train with fp8 base + bf16 LoRA** — fits easier in 32 GB but departs from reference; need to verify LoRA forward through fp8-quantised linears works cleanly (diffusers `quanto`/`torchao` path).

**Recommendation:** download bf16 Flux-dev once. 24 GB is not a bottleneck on a 2 TB disk, and it eliminates a risk vector.

## RTX 5090 hardware budget

- **Total VRAM:** 32 607 MiB
- **Currently free:** 12 507 MiB (ComfyUI is holding ~20 GB)
- **Actionable:** must shut ComfyUI down for training runs. Assume full 32 GB available.

### Memory math for training (bf16 Flux-dev, LoRA rank-16 xattn-only)

| Component | Size | Resident during train step? |
|---|---:|---|
| Flux-dev transformer (bf16, frozen) | 23.8 GB | yes |
| LoRA params (rank 16, xattn) | ~30 MB | yes |
| LoRA optimizer state (8-bit Adam) | ~60 MB | yes |
| Activations @ 512×512 w/ grad ckpt | 2–4 GB | yes |
| T5-XXL (encode once, then offload) | 9.5 GB | **no** (pre-encode) |
| CLIP-L | 0.25 GB | no |
| VAE (encode once, offload) | 0.3 GB | no |
| Latents + text embeds cache | ~0.5 GB | yes |
| **Estimated peak** | **~27–30 GB** | — |

**Fits on 5090, but tight.** Required tactics:
- Pre-compute and cache T5+CLIP embeddings for all training prompts (once).
- Pre-compute and cache VAE latents for all before/after images (once).
- Drop text encoders and VAE from GPU after precompute.
- Gradient checkpointing on transformer blocks.
- 8-bit Adam (bitsandbytes) for LoRA optimizer state.
- bf16 mixed precision — **not fp16** (diffusion underflows fp16 gradients).

### On fp16 vs bf16

User's constraint "fp16 is not an option" is correct for the **training** step but the reason is numerical, not hardware: bf16 has 8-bit exponent (same dynamic range as fp32), fp16 has 5-bit and underflows on small gradients typical of diffusion. 5090 supports all precisions (Blackwell SM 12.0, including fp8). We run the whole trainer in bf16 mixed precision.

## Constraints we are accepting

- **Single-GPU training.** No DDP, no FSDP. One slider at a time.
- **~1 h/axis target** (A100 reference) — 5090 is ~1.3× faster than A100 40 GB on bf16 matmul but memory-bandwidth-bound on LoRA; realistic estimate 60–90 min/axis.
- **10 axes in queue** → 10–15 h total training time (overnight-feasible).
- **No aux losses in v1.** Pure flow-matching velocity MSE. Metrics are eval-only.
- **Endpoint pairs first.** Multi-α scales training is v2 if endpoints underperform.

## Open questions (resolve in step 2–3, not now)

- Is vanilla LoRA the best choice, or does LoKr/DoRA/rsLoRA help on diffusion xattn? (step 2)
- Endpoint-only vs multi-scale α sampling? (step 4)
- Train per base or pooled across bases? Pooled risks demographic averaging; per-base means 6× more training runs. (step 3 — validation protocol decides)

## Next

Step 2: LoRA family refresher + pick parametrisation for diffusion xattn.
