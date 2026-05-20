"""GATING feasibility spike: prove a trainable bf16 InfuseNet runs one CFM step.

Run: uv run python -m arkit_controlnet.cfm.spike_trainable_infusenet

Loads diffusers FLUX (frozen Krea-bf16 transformer as the base — same arch as
FLUX.1-dev) + bf16 InfuseNet (a `FluxControlNetModel` checkpoint released by
ByteDance), attaches a LoRA to InfuseNet's DiT-copy blocks, marks its
control-image input stem (`x_embedder`) trainable, and runs one
conditional-flow-matching forward+backward step on synthetic tensors.
Asserts the loss is finite and the trainable gradients are non-zero, then
prints peak VRAM. Exit code 0 = GO, non-zero = NO-GO.

Discovery: the bf16 weights at
`data/infiniteyou_dl_bf16/infu_flux_v1.0/sim_stage1/InfuseNetModel/` are a
diffusers `FluxControlNetModel` (config `_class_name=FluxControlNetModel`,
4 double + 10 single layers, in_channels=64). It loads via
`FluxControlNetModel.from_pretrained(...)` and its forward returns
`(controlnet_block_samples, controlnet_single_block_samples)` which the base
FLUX transformer's forward accepts directly as residuals — no monkey-patch.

The 8 identity tokens from the InfiniteYou image-proj resampler are
prepended to the T5 encoder_hidden_states. The spike stubs them as
`torch.randn(B, 8, 4096)` since identity projection is frozen in training
anyway.
"""
import sys
import math
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
INFUSE_DIR = (REPO / "data/infiniteyou_dl_bf16/infu_flux_v1.0/sim_stage1/"
              "InfuseNetModel")
FLUX_TRANSFORMER_SF = Path(
    "/home/newub/w/ComfyUI/models/diffusion_models/flux1-krea-dev.safetensors"
)
FLUX_HF_ID = "black-forest-labs/FLUX.1-dev"

DEVICE = "cuda"
DTYPE = torch.bfloat16

# Spike tensor shapes — small to keep VRAM under 32 GB on a single forward
# with gradient checkpointing on the frozen FLUX trunk.
B = 1
H = W = 64                   # latent H/W (image is 8x → 512x512)
SEQ_T5 = 64                  # short T5 seq for the spike (training uses 512)
ID_TOKENS = 8                # InfiniteYou identity tokens
T5_DIM = 4096
CLIP_POOLED = 768


# --- FLUX packing helpers (copied verbatim from train_flux_image_slider) ---

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


# --- Model build ---

def build_trainable_infusenet():
    """Load frozen FLUX + bf16 InfuseNet; attach a LoRA + trainable control
    stem; freeze everything else. Returns (flux, infusenet, trainable_params).
    """
    from diffusers import FluxTransformer2DModel, FluxControlNetModel
    from peft import LoraConfig, get_peft_model

    print(f"[load] FLUX transformer (frozen, bf16) from {FLUX_TRANSFORMER_SF}")
    flux = FluxTransformer2DModel.from_single_file(
        str(FLUX_TRANSFORMER_SF), torch_dtype=DTYPE
    ).to(DEVICE)
    flux.requires_grad_(False)
    flux.enable_gradient_checkpointing()
    flux.eval()

    print(f"[load] InfuseNet (FluxControlNetModel, bf16) from {INFUSE_DIR}")
    infusenet = FluxControlNetModel.from_pretrained(
        str(INFUSE_DIR), torch_dtype=DTYPE
    ).to(DEVICE)
    infusenet.requires_grad_(False)
    if hasattr(infusenet, "enable_gradient_checkpointing"):
        infusenet.enable_gradient_checkpointing()

    # LoRA on the DiT-copy double + single blocks inside InfuseNet.
    target_modules = [
        # double blocks
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.0.attn.to_k",
        "transformer_blocks.0.attn.to_v",
        "transformer_blocks.0.attn.to_out.0",
        "transformer_blocks.0.attn.add_q_proj",
        "transformer_blocks.0.attn.add_k_proj",
        "transformer_blocks.0.attn.add_v_proj",
        "transformer_blocks.0.attn.to_add_out",
        # single blocks
        "single_transformer_blocks.0.attn.to_q",
        "single_transformer_blocks.0.attn.to_k",
        "single_transformer_blocks.0.attn.to_v",
        "single_transformer_blocks.0.proj_out",
    ]
    # Expand to all layers.
    expanded = []
    n_double = len(infusenet.transformer_blocks)
    n_single = len(infusenet.single_transformer_blocks)
    print(f"[load] InfuseNet has {n_double} double + {n_single} single blocks")
    for tm in target_modules:
        if tm.startswith("transformer_blocks"):
            for i in range(n_double):
                expanded.append(tm.replace("transformer_blocks.0",
                                            f"transformer_blocks.{i}"))
        else:
            for i in range(n_single):
                expanded.append(tm.replace("single_transformer_blocks.0",
                                            f"single_transformer_blocks.{i}"))

    lora_cfg = LoraConfig(
        r=8, lora_alpha=8, target_modules=expanded,
        lora_dropout=0.0, bias="none",
    )
    infusenet = get_peft_model(infusenet, lora_cfg, adapter_name="cfm")

    # Mark the control-image input stem trainable. In diffusers
    # FluxControlNetModel, the input stem is `x_embedder` (a Linear that
    # ingests packed control latents). InfuseNet additionally has a
    # `controlnet_x_embedder` for the explicit control image; both share
    # this role in different diffusers versions.
    stem_params = []
    for name in ("x_embedder", "controlnet_x_embedder"):
        mod = getattr(infusenet, name, None)
        if mod is None and hasattr(infusenet, "base_model"):
            mod = getattr(infusenet.base_model.model, name, None)
        if mod is not None:
            for p in mod.parameters():
                p.requires_grad = True
                stem_params.append(p)
            print(f"[load] marked {name} trainable ({sum(p.numel() for p in mod.parameters())} params)")

    trainable_params = [p for p in infusenet.parameters() if p.requires_grad]
    n_lora = sum(p.numel() for p in trainable_params) - sum(p.numel() for p in stem_params)
    print(f"[load] trainable params: {len(trainable_params)} tensors "
          f"(~{sum(p.numel() for p in trainable_params)/1e6:.2f}M); "
          f"stem={sum(p.numel() for p in stem_params)/1e6:.2f}M, "
          f"lora≈{n_lora/1e6:.2f}M")

    return flux, infusenet, trainable_params


# --- CFM step ---

def forward_velocity(flux, infusenet, z_t, sigma, encoder_hidden_states,
                      pooled, txt_ids, img_ids, control_packed, guidance):
    """Run InfuseNet → residuals → frozen FLUX with residuals; return velocity."""
    # InfuseNet forward — returns (controlnet_block_samples,
    #                              controlnet_single_block_samples)
    cn_double, cn_single = infusenet(
        hidden_states=z_t,
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

    v_packed = flux(
        hidden_states=z_t,
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


def one_cfm_step(flux, infusenet, trainable_params):
    # Synthetic conditioning.
    z0 = torch.randn(B, 16, H, W, device=DEVICE, dtype=DTYPE)
    eps = torch.randn_like(z0)
    t = torch.sigmoid(torch.randn(B, device=DEVICE))
    shift = 3.0
    sigma = (shift * t) / (1 + (shift - 1) * t)
    s = sigma.view(-1, 1, 1, 1).to(DTYPE)
    z_t = (1 - s) * z0 + s * eps
    target = eps - z0

    # Pack to FLUX token shape (B, H/2*W/2, 64).
    z_t_packed = pack_latents(z_t)
    target_packed = pack_latents(target)
    # Control image: another packed latent, same shape (B, seq, 64).
    control_packed = pack_latents(
        torch.randn_like(z0)
    ).requires_grad_(True)
    # Force grad through frozen-+-checkpointed trunk; mirrors slider trainer.
    z_t_packed.requires_grad_(True)

    img_ids = prepare_latent_image_ids(H, W, DEVICE, DTYPE)

    # Conditioning: 8 identity tokens prepended to T5 sequence.
    t5 = torch.randn(B, SEQ_T5, T5_DIM, device=DEVICE, dtype=DTYPE)
    id_tokens = torch.randn(B, ID_TOKENS, T5_DIM, device=DEVICE, dtype=DTYPE)
    encoder_hidden_states = torch.cat([id_tokens, t5], dim=1)
    pooled = torch.randn(B, CLIP_POOLED, device=DEVICE, dtype=DTYPE)
    txt_ids = prepare_text_ids(encoder_hidden_states.shape[1], DEVICE, DTYPE)

    guidance = None
    if getattr(flux.config, "guidance_embeds", False):
        guidance = torch.full((B,), 3.5, device=DEVICE, dtype=DTYPE)

    opt = torch.optim.AdamW(trainable_params, lr=1e-4)
    opt.zero_grad()

    with torch.autocast("cuda", dtype=DTYPE):
        v_pred = forward_velocity(
            flux, infusenet, z_t_packed, sigma,
            encoder_hidden_states, pooled, txt_ids, img_ids,
            control_packed, guidance,
        )

    loss = torch.nn.functional.mse_loss(v_pred.float(), target_packed.float())
    loss.backward()
    grad_norm = torch.sqrt(
        sum((p.grad.float() ** 2).sum()
            for p in trainable_params if p.grad is not None)
    )
    opt.step()
    return loss.item(), grad_norm.item()


def main() -> int:
    torch.cuda.reset_peak_memory_stats()
    flux, infusenet, trainable_params = build_trainable_infusenet()
    assert len(trainable_params) > 0, "no trainable parameters"
    loss, grad_norm = one_cfm_step(flux, infusenet, trainable_params)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"loss={loss:.5f}  grad_norm={grad_norm:.3e}  peak_vram={peak_gb:.1f}GB")

    if not math.isfinite(loss):
        print("NO-GO: loss is not finite"); return 1
    if grad_norm == 0.0 or not math.isfinite(grad_norm):
        print("NO-GO: trainable gradients are zero / non-finite"); return 1
    print("GO: one CFM step completed, finite loss, non-zero gradients")
    return 0


if __name__ == "__main__":
    sys.exit(main())
