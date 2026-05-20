"""Build the trainable CFM model (frozen FLUX.1-dev + LoRA-trained InfuseNet).

The concrete API was discovered by the gating spike at
`spike_trainable_infusenet.py`; this module is the library version. Differences
from the spike:
  - Base FLUX is dev-bf16 from HF cache, not Krea (InfuseNet was trained
    against dev — residual alignment matters for the long run).
  - Exposes `velocity()` as the trainer's single-shot forward function.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn import Parameter

DEVICE = "cuda"
DTYPE = torch.bfloat16

FLUX_HF_ID = "black-forest-labs/FLUX.1-dev"
INFUSE_DIR_DEFAULT = "data/infiniteyou_dl_bf16/infu_flux_v1.0/sim_stage1/InfuseNetModel"


@dataclass
class CfmModel:
    flux: object                    # FluxTransformer2DModel — frozen
    infusenet: object               # peft-wrapped FluxControlNetModel
    trainable_params: list[Parameter]


def pack_latents(z: torch.Tensor) -> torch.Tensor:
    Bn, C, Hn, Wn = z.shape
    z = z.view(Bn, C, Hn // 2, 2, Wn // 2, 2)
    z = z.permute(0, 2, 4, 1, 3, 5)
    return z.reshape(Bn, (Hn // 2) * (Wn // 2), C * 4)


def unpack_latents(z_packed: torch.Tensor, Hn: int, Wn: int) -> torch.Tensor:
    Bn, _, Cp = z_packed.shape
    C = Cp // 4
    z = z_packed.view(Bn, Hn // 2, Wn // 2, C, 2, 2)
    z = z.permute(0, 3, 1, 4, 2, 5)
    return z.reshape(Bn, C, Hn, Wn)


def prepare_latent_image_ids(Hn: int, Wn: int, device, dtype):
    ids = torch.zeros(Hn // 2, Wn // 2, 3)
    ids[..., 1] = ids[..., 1] + torch.arange(Hn // 2)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(Wn // 2)[None, :]
    return ids.reshape(-1, 3).to(device, dtype)


def prepare_text_ids(seq_len: int, device, dtype):
    return torch.zeros(seq_len, 3).to(device, dtype)


# Same module list as the spike. 4 double + 10 single blocks in InfuseNet.
_DOUBLE_TARGETS = [
    "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
    "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",
]
_SINGLE_TARGETS = ["attn.to_q", "attn.to_k", "attn.to_v", "proj_out"]


def build_model(
    flux_hf_id: str = FLUX_HF_ID,
    infusenet_dir: str = INFUSE_DIR_DEFAULT,
    lora_r: int = 8,
    lora_alpha: int = 8,
    device: str = DEVICE,
    dtype: torch.dtype = DTYPE,
) -> CfmModel:
    from diffusers import FluxTransformer2DModel, FluxControlNetModel
    from peft import LoraConfig, get_peft_model

    flux = FluxTransformer2DModel.from_pretrained(
        flux_hf_id, subfolder="transformer", torch_dtype=dtype
    ).to(device)
    flux.requires_grad_(False)
    flux.enable_gradient_checkpointing()
    flux.eval()

    infusenet = FluxControlNetModel.from_pretrained(
        infusenet_dir, torch_dtype=dtype
    ).to(device)
    infusenet.requires_grad_(False)
    if hasattr(infusenet, "enable_gradient_checkpointing"):
        infusenet.enable_gradient_checkpointing()

    n_double = len(infusenet.transformer_blocks)
    n_single = len(infusenet.single_transformer_blocks)
    targets = []
    for i in range(n_double):
        for suf in _DOUBLE_TARGETS:
            targets.append(f"transformer_blocks.{i}.{suf}")
    for i in range(n_single):
        for suf in _SINGLE_TARGETS:
            targets.append(f"single_transformer_blocks.{i}.{suf}")

    lora_cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha,
                          target_modules=targets,
                          lora_dropout=0.0, bias="none")
    infusenet = get_peft_model(infusenet, lora_cfg, adapter_name="cfm")

    # Control-image input stems are fully trainable (the FLAME-render modality
    # is novel; LoRA on the stem is too narrow to absorb the modality switch).
    for name in ("x_embedder", "controlnet_x_embedder"):
        mod = getattr(infusenet, name, None)
        if mod is None and hasattr(infusenet, "base_model"):
            mod = getattr(infusenet.base_model.model, name, None)
        if mod is None:
            raise RuntimeError(
                f"could not locate {name!r} on InfuseNet — peft/diffusers "
                f"layout may have changed; fix build_model() before training")
        for p in mod.parameters():
            p.requires_grad = True

    trainable_params = [p for p in infusenet.parameters() if p.requires_grad]
    return CfmModel(flux=flux, infusenet=infusenet,
                    trainable_params=trainable_params)


def velocity(
    model: CfmModel,
    z_t_packed: torch.Tensor,
    sigma: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    pooled: torch.Tensor,
    txt_ids: torch.Tensor,
    img_ids: torch.Tensor,
    control_packed: torch.Tensor,
    guidance: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One CFM forward: InfuseNet residuals -> frozen FLUX -> predicted velocity."""
    cn_double, cn_single = model.infusenet(
        hidden_states=z_t_packed,
        controlnet_cond=control_packed,
        conditioning_scale=1.0,
        encoder_hidden_states=encoder_hidden_states,
        pooled_projections=pooled,
        timestep=sigma,
        img_ids=img_ids,
        txt_ids=txt_ids,
        guidance=guidance,
        return_dict=False,
    )
    v_packed = model.flux(
        hidden_states=z_t_packed,
        timestep=sigma,
        guidance=guidance,
        pooled_projections=pooled,
        encoder_hidden_states=encoder_hidden_states,
        txt_ids=txt_ids,
        img_ids=img_ids,
        controlnet_block_samples=cn_double,
        controlnet_single_block_samples=cn_single,
        return_dict=False,
    )[0]
    return v_packed
