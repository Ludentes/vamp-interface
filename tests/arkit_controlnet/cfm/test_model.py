import math
import torch
import pytest
from arkit_controlnet.cfm.model import build_model, velocity, pack_latents


@pytest.fixture(scope="module")
def model():
    return build_model()


def test_build_model_has_only_lora_and_stems_trainable(model):
    """LoRA + x_embedder + controlnet_x_embedder are trainable, all else frozen."""
    trainable = [(n, p) for n, p in model.infusenet.named_parameters()
                 if p.requires_grad]
    assert len(trainable) > 0
    for n, _ in trainable:
        assert ("lora_" in n) or ("x_embedder" in n), f"unexpected trainable: {n}"
    # Both stems must have at least one trainable param (positive check; the
    # substring whitelist above would silently pass if a stem went missing).
    assert any("controlnet_x_embedder" in n and p.requires_grad
               for n, p in model.infusenet.named_parameters()), \
        "controlnet_x_embedder has no trainable params"
    # x_embedder check must exclude the controlnet_x_embedder substring match.
    assert any(("x_embedder" in n) and ("controlnet_x_embedder" not in n)
               and p.requires_grad
               for n, p in model.infusenet.named_parameters()), \
        "x_embedder has no trainable params"
    for n, p in model.flux.named_parameters():
        assert not p.requires_grad, f"FLUX param {n} unexpectedly trainable"


def test_velocity_output_shape_matches_input(model):
    B, H, W = 1, 64, 64
    DTYPE = torch.bfloat16
    z0 = torch.randn(B, 16, H, W, device="cuda", dtype=DTYPE)
    z_t = pack_latents(z0)
    sigma = torch.full((B,), 0.5, device="cuda", dtype=DTYPE)
    ctrl = pack_latents(torch.randn_like(z0))
    eh = torch.randn(B, 8 + 512, 4096, device="cuda", dtype=DTYPE)
    pooled = torch.randn(B, 768, device="cuda", dtype=DTYPE)
    from arkit_controlnet.cfm.model import (
        prepare_latent_image_ids, prepare_text_ids)
    img_ids = prepare_latent_image_ids(H, W, "cuda", DTYPE)
    txt_ids = prepare_text_ids(8 + 512, "cuda", DTYPE)
    guidance = torch.full((B,), 3.5, device="cuda", dtype=DTYPE)
    with torch.no_grad():
        v = velocity(model, z_t, sigma, eh, pooled, txt_ids, img_ids,
                     ctrl, guidance)
    assert v.shape == z_t.shape
    assert math.isfinite(v.float().abs().mean().item())
