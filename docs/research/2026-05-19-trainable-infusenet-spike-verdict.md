---
status: live
topic: arkit-controlnet
---

# Trainable bf16 InfuseNet — CFM Feasibility Spike Verdict

**Verdict: GO.** One CFM forward+backward step through a bf16 InfuseNet stacked
on a frozen FLUX (Krea-bf16) transformer ran cleanly on the RTX 5090: finite
loss, non-zero LoRA + control-stem gradients, peak VRAM well under budget.

## Measured

```
loss=447.30017  grad_norm=8.807e+00  peak_vram=30.7GB
GO: one CFM step completed, finite loss, non-zero gradients
```

- **Trainable params**: 148 tensors, ~4.92 M total — ~0.40 M from the two
  control-image input stems (`x_embedder` + `controlnet_x_embedder`),
  ~4.52 M from a peft LoRA (r=8, α=8) on all attention projections of the
  4 double + 10 single InfuseNet blocks.
- **Peak VRAM**: 30.7 GB with gradient checkpointing on the frozen FLUX
  trunk and a single packed-latent batch at 512² (16 × 64 × 64 unpacked).
  Headroom for batch=2 or a longer T5 sequence is tight but real; the
  training run plan should keep gradient checkpointing on InfuseNet too.

## Concrete API

The bf16 weights at
`data/infiniteyou_dl_bf16/infu_flux_v1.0/sim_stage1/InfuseNetModel/` are a
**diffusers `FluxControlNetModel`** checkpoint (config: `_class_name =
FluxControlNetModel`, `_diffusers_version = 0.31.0`, `_infu_flux_version =
1.0`, `num_layers = 4`, `num_single_layers = 10`, `in_channels = 64`,
`joint_attention_dim = 4096`, `pooled_projection_dim = 768`,
`guidance_embeds = true`). No custom class porting needed: diffusers ≥0.31
loads it directly via `from_pretrained`.

### Load calls (copy-pasteable)

```python
from diffusers import FluxTransformer2DModel, FluxControlNetModel

# Frozen base — Krea-bf16 single-file safetensors. (Krea is a fine-tune of
# FLUX.1-dev with identical architecture; we use it because dev-bf16 is not
# on disk and a one-step CFM spike is base-identity-insensitive.)
flux = FluxTransformer2DModel.from_single_file(
    "/home/newub/w/ComfyUI/models/diffusion_models/flux1-krea-dev.safetensors",
    torch_dtype=torch.bfloat16,
).to("cuda")
flux.requires_grad_(False)
flux.enable_gradient_checkpointing()
flux.eval()

# bf16 InfuseNet — diffusers sharded safetensors (2 shards in sim_stage1).
infusenet = FluxControlNetModel.from_pretrained(
    "data/infiniteyou_dl_bf16/infu_flux_v1.0/sim_stage1/InfuseNetModel",
    torch_dtype=torch.bfloat16,
).to("cuda")
infusenet.requires_grad_(False)
infusenet.enable_gradient_checkpointing()
```

### Trainable surface

- **peft LoRA** on InfuseNet's DiT-copy attention projections. The 4 double
  blocks expose `attn.{to_q, to_k, to_v, to_out.0, add_q_proj, add_k_proj,
  add_v_proj, to_add_out}`; the 10 single blocks expose `attn.{to_q, to_k,
  to_v}` and `proj_out`.
- **Control-image input stem**: in diffusers `FluxControlNetModel` the
  stem that ingests the packed control latent is `x_embedder` plus an
  auxiliary `controlnet_x_embedder` (both `nn.Linear`, ~0.2 M params each
  for `in_channels=64 × patch_size² → hidden_size=3072`). Both are set
  `requires_grad=True` for training.

### Forward signature (diffusers 0.37.1)

InfuseNet returns FLUX residuals — no monkey-patch:

```python
controlnet_block_samples, controlnet_single_block_samples = infusenet(
    hidden_states=z_t_packed,        # (B, H/2*W/2, 64) — noised image latent
    controlnet_cond=control_packed,  # (B, H/2*W/2, 64) — packed render latent
    conditioning_scale=1.0,
    encoder_hidden_states=eh,        # (B, 8 + T5_seq, 4096) — id tokens || T5
    pooled_projections=pooled,       # (B, 768)
    timestep=sigma,                  # (B,) — FLUX sigma (already shift-warped)
    img_ids=img_ids,                 # (H/2*W/2, 3)
    txt_ids=txt_ids,                 # (8 + T5_seq, 3) — zeros
    guidance=guidance,               # (B,) — guidance_embeds=true
    return_dict=False,
)
```

The 8 identity tokens (from InfiniteYou's `Resampler` over the ArcFace
embedding) are **concatenated to the T5 sequence** — InfuseNet's joint
attention reads them as additional text tokens. `txt_ids` is zeros.

Residual-injection into FLUX is the standard diffusers control path:

```python
v_packed = flux(
    hidden_states=z_t_packed,
    timestep=sigma,
    guidance=guidance,
    pooled_projections=pooled,
    encoder_hidden_states=eh,
    txt_ids=txt_ids,
    img_ids=img_ids,
    controlnet_block_samples=controlnet_block_samples,
    controlnet_single_block_samples=controlnet_single_block_samples,
    return_dict=False,
)[0]  # (B, H/2*W/2, 64)
```

### Identity tokens for the spike

The 8 identity tokens come from the InfiniteYou `Resampler` over a 512-d
ArcFace embedding (weights at
`data/infiniteyou_dl_bf16/infu_flux_v1.0/sim_stage1/image_proj_model.bin`,
not yet downloaded). The training plan will load this resampler frozen —
it's identity-projection only and never trained.

The spike **stubs** the identity tokens as `torch.randn(B, 8, 4096)` since
it only needs to exercise InfuseNet's DiT-copy + control stem under
gradient. This is justified: the resampler is frozen in the training run
anyway, so its absence does not change which params receive gradient.

## Caveats for the training-run plan

- **Base FLUX**: spike used Krea-bf16 (`flux1-krea-dev.safetensors`). For
  the real training run, decide whether to keep Krea or download
  `FLUX.1-dev` bf16 (~24 GB). Krea is a fine-tune of dev — InfuseNet was
  trained against dev — so the **safer choice for the actual run is
  dev-bf16** so InfuseNet residuals match the trunk they were trained
  against. The spike used Krea only because dev-bf16 isn't on disk; load
  shapes were identical, confirming the architectures match.
- **Gradient checkpointing**: required on the FLUX trunk to hit 30.7 GB;
  also enabled on InfuseNet. Without InfuseNet checkpointing, expect to
  rise into the 40-50 GB range at batch=1. Keep both on.
- **Resampler download**: before the real training run, fetch
  `image_proj_model.bin` (~50 MB) — `huggingface-cli download
  ByteDance/InfiniteYou infu_flux_v1.0/sim_stage1/image_proj_model.bin
  --local-dir data/infiniteyou_dl_bf16/`.
- **Guidance**: `guidance_embeds=true` on both transformer and InfuseNet.
  The CFM trainer should pass a real `guidance` tensor (training-time
  guidance scale, e.g. 3.5 or sampled from a range), not `None`.
- **LoRA targets**: spike used r=8 on all attention modules. The training
  plan should sweep r ∈ {8, 16, 32} early — InfuseNet only has 14 blocks
  (vs 57 in FLUX) so trainable count stays small.
- **Control image preprocessing**: in InfiniteYou inference the control
  image is the *5-keypoint* drawing (CFM training will swap this for the
  FLAME render — flat/normals/depth — per the render-modality A/B in the
  spec). The control input is packed via the same 2×2 unfold as the noised
  latent, then passed as `controlnet_cond`.

## Weights paths

- `data/infiniteyou_dl_bf16/infu_flux_v1.0/sim_stage1/InfuseNetModel/`
  (`config.json` + 2 safetensors shards, bf16, diffusers
  `FluxControlNetModel`).
- `/home/newub/w/ComfyUI/models/diffusion_models/flux1-krea-dev.safetensors`
  (FLUX transformer single-file, bf16) — replaceable with FLUX.1-dev bf16
  for the real training run.
- `~/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/` —
  VAE + CLIP-L + T5-XXL + tokenizers (HF id `black-forest-labs/FLUX.1-dev`).

## Reproducing

```
PYTHONPATH=src uv run python -m arkit_controlnet.cfm.spike_trainable_infusenet
```

Source: `src/arkit_controlnet/cfm/spike_trainable_infusenet.py`.
