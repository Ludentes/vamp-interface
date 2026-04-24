---
status: live
topic: demographic-pc-pipeline
---

# Step 2.5 — Flow explainer: how patching, MMDiT, LoRA, and flow-matching interplay

Pedagogical walk-through of one training step from the `(α=0.png, α=1.png)` pair to the LoRA gradient update. Companion to step 2 (family refresher) — this doc makes the signal flow explicit so improvement ideas (step 2.6+) can reason about *where* to intervene.

## Terminology note

Flux uses a **DiT** (Diffusion Transformer), not a ViT. But the **patchify-to-token-sequence mechanism is directly inherited from ViT** — that's the connection. "Token matching" here refers to how image and text tokens attend jointly in MMDiT blocks.

## Eleven-step flow (512×512 training)

**1. Image → latent (VAE, frozen, one-shot).**
```
RGB [3, 512, 512] ──VAE encoder──▶ z [16, 64, 64]
```
16-channel, 8× spatial downsample. Cache these — never recomputed.

**2. Latent → patch tokens (ViT heritage).**
```
z [16, 64, 64] ──patch 2×2──▶ [1024, 64] ──linear──▶ [1024, 3072]
```
32×32 = **1024 image tokens**, each 3072-d hidden. Position via RoPE at every attention call.

**3. Text → tokens (both encoders frozen, one-shot).**
```
prompt ──T5-XXL──▶ [≤512, 3072]   (enters attention as tokens)
prompt ──CLIP-L──▶ [768] pooled   (adaLN modulation vector)
```
Cache these per prompt.

**4. Noisy latent at random timestep (flow matching).**
```
t ~ U[0,1],  ε ~ N(0,I)
z_t = (1-t)·z_clean + t·ε
v_target = ε - z_clean   (straight-line velocity, constant along t)
```
`FlowMatchEulerDiscreteScheduler`. Network predicts velocity, not noise.

**5. Noisy latent → image tokens** (repeat step 2 on `z_t`).

**6. MMDiT double blocks (19 of them).** Two streams kept separate but
attention is joint:
```
q = concat(W_q·img, W_q_add·txt)   → [1536, 3072]
k = concat(W_k·img, W_k_add·txt)
v = concat(W_v·img, W_v_add·txt)
attn = softmax(qkᵀ/√d)·v
img_out, txt_out = split(attn, [1024, 512])
img_out → MLP_img;  txt_out → MLP_txt
```
**This is where text concepts ("squint") modulate image-token representations** — via the softmax routing in attention Q/K/V.

Each double block has **8 attention linears**:
`to_q, to_k, to_v, to_out.0, add_q_proj, add_k_proj, add_v_proj, to_add_out`.

**7. Single blocks (38 of them).** Streams concatenated into one
sequence (1536 tokens), standard self-attention + fused MLP. **4 attention
linears** per block: `to_q, to_k, to_v, to_out`.

**Total LoRA hooks:** 19×8 + 38×4 = **304 linear layers**.

**8. Extract prediction.** Drop text tokens, un-patch image tokens:
```
[1536, 3072] → [1024, 3072] → v̂ [16, 64, 64]
```

**9. Where LoRA lives.** Every one of those 304 linears runs:
```python
def lora_forward(x):
    base = x @ W₀              # frozen 3072×3072, ~9.4M params
    delta = (x @ Aᵀ) @ Bᵀ      # A [r=16, 3072], B [3072, r=16]
    return base + (α/r) * delta
```
`B` init zeros → `ΔW=0` at step 0 → **LoRA'd model identical to base Flux at init.** Scale `α/r = 1/16` damps injection.

**10. Loss — image-pair version.**
```
z_before, z_after ← cached VAE latents
v_target = z_after - z_before        # edit velocity
z_t = (1-t)·z_before + t·ε
v̂ = flux(z_t, t, base_prompt, LoRA=on)
loss = MSE(v̂, v_target)
```
**One DiT forward per step** vs the notebook's 4 (target + pos + neg teachers + student). 4× speedup from having image supervision.

**11. Backward.** Only A, B have `requires_grad=True`. Grad checkpointing recomputes frozen activations rather than storing them (VRAM trade). AdamW 8-bit updates ~30 MB of LoRA params. 24 GB of Flux weights are read-only throughout.

## Mental model — one-line summary per piece

| Piece | Role |
|---|---|
| VAE | pixel ↔ latent compressor, outside the loop |
| Patchify | ViT-style token sequence construction |
| T5 + CLIP | frozen text understanding |
| Flow matching | velocity-prediction loss (not noise-prediction) |
| MMDiT joint attention | where text modulates image via softmax routing |
| LoRA (xattn) | additive bypass wires on 304 attention projections |
| Training signal | "your current routing produces v̂; needs to produce z_after − z_before" |

## The one-paragraph answer to "how does it all interplay?"

A ViT-style patchifier turns the latent into a 1024-token sequence. Text tokens from T5 enter the DiT alongside image tokens. In each MMDiT block, joint attention is how text concepts reshape image tokens — Q/K/V routing is the interplay. LoRA inserts a tiny additive correction into every Q/K/V projection; `W₀` stays frozen. Flow matching defines the loss: at a random timestep along the straight line from `z_before` to noise, Flux (with LoRA on) must predict the velocity that transports `z_before` to `z_after`. Gradients only update the ~30 MB of LoRA params. The slider is a **304-point additive correction to Flux's attention routing**, tuned to reproduce our 18 image-pair velocities.

## Next

Step 2.6: improvement ideas given our specific corpus + metrics + goals.
