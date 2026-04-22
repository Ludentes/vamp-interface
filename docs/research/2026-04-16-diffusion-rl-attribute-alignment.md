---
status: archived
topic: archived-threads
summary: Vanilla DDPO diverges on Flux; use Flow-GRPO or DanceGRPO only if differentiable reward backprop fails, but prefer ReFL-style truncated gradients first.
---

# RL for attribute-alignment of a frozen Flux via a learned projection P

2026-04-16 — research note (no code). Target use: training upstream projection `P: qwen-1024 → Flux-cond` (and readout head `R`) against a scalar pattern-recovery reward computed from a frozen face encoder on Flux outputs.

## 1. TL;DR

RL is a **plausible but not the first-best tool** here. For a pipeline where (a) the reward is a smooth-ish scalar computed by a frozen encoder and (b) the only trainable parameters live *upstream* of the frozen Flux in a differentiable projection, the cleanest move is to try to keep it differentiable (ReFL-style truncated backprop through a 1–4-step rectified-flow sample, or a latent-space surrogate reward) before committing to policy gradient. If you do go RL, **the only algorithms that currently work stably on Flux are rectified-flow-specific**: **Flow-GRPO** (arXiv:2505.05470, NeurIPS 2025) and **DanceGRPO** (arXiv:2505.07818). Vanilla **DDPO diverges on rectified-flow SDEs** — this is reported both in the Flow-GRPO/DanceGRPO motivations and in follow-ups. TexForce (arXiv:2311.15657) is the exact architectural precedent for training only a conditioning module via DDPO, but it predates Flux and was built on SD1.x.

## 2. Per-algorithm summary

**DDPO** — arXiv:2305.13301 (Black et al., 2023). Reframes denoising as an MDP; DDPO-IS is basically PPO on the denoising trajectory. Reference implementations: authors' repo `jannerm/ddpo`, and `trl.DDPOTrainer` (HF). Compatible with "frozen-backbone + trainable conditioning" in principle — TexForce showed exactly this on SD1.5. But **it does not port to rectified-flow models**: the deterministic ODE sampler of Flux has no exploration noise, so the standard DDPO likelihood ratio is ill-defined. You can force it by converting ODE → SDE (the Flow-GRPO trick), at which point you're essentially running Flow-GRPO. Stability is known to be fragile: reward hacking and mode collapse are both reported when KL is absent.

**DPOK** — arXiv:2305.16381 (Fan et al., 2023, NeurIPS). DDPO + KL regularization to the pre-trained model. More stable than DDPO on text-to-image, prevents the "oversaturated / unnatural shape" degenerate solutions. Same rectified-flow-incompatibility problem: the KL term is defined over the original diffusion process, not over rectified flow. No known Flux adaptation.

**Diffusion-DPO** — arXiv:2311.12908 (Wallace et al., CVPR 2024). Preference-pair training, no scalar reward, no online sampling. Our scalar reward converts naturally: within each batch sample `N` generations per prompt, label the top-k as "chosen" and bottom-k as "rejected" by reward. This **does port to rectified flow** — SD3 was fine-tuned with Diffusion-DPO — and there are Flux.1-dev DPO variants in the wild. But Diffusion-DPO trains UNet/DiT weights, *not* the conditioning projection upstream of it. Not a clean fit for our P-only setup.

**ReFL** — arXiv:2304.05977 and later variants. Reward-weighted regression / directly backpropagating reward through a truncated denoising trajectory. **This is probably the best match to our problem shape** because our reward is already a differentiable function of the generated image pixels (frozen encoder → readout head), and P is differentiable. The question collapses to "can we push gradients through N rectified-flow steps" — Flow-GRPO reduces N to ~10 training steps, which keeps memory tractable.

**D3PO** — arXiv:2311.13231 (Yang et al., CVPR 2024). DPO-style, no reward model, treats the denoising chain as the preference comparison object. Same fit/non-fit profile as Diffusion-DPO. Useful if you can only generate pairs, not scalars.

**Flow-GRPO** — arXiv:2505.05470 (Liu et al., NeurIPS 2025). GRPO with two rectified-flow-specific tricks: ODE→SDE conversion to create exploration noise at each step; "Denoising Reduction" to train with T=10 while inferring with T=40. Reports SD3.5-M GenEval 63 → 95 (compositional attributes, spatial relations, fine-grained attributes — this is the closest published analogue to attribute-preservation reward on rectified flow). **Official repo supports FLUX.1-dev**. Stability good. GRPO needs no value network (group-relative advantage `A_i = (r_i − mean) / std`).

**DanceGRPO** — arXiv:2505.07818. Unified GRPO over diffusion + rectified flow; explicitly tests on FLUX.1-dev and SD3 with up to 181% reward-score improvement. Same algorithmic family as Flow-GRPO.

## 3. The conditioning-only fine-tuning gotcha

**Known-working**, but rarely in the configuration we want. TexForce (ECCV 2024, arXiv:2311.15657) is the canonical reference: freeze UNet, train a LoRA on the text encoder with DDPO. They report better semantic/attribute preservation than UNet-LoRA because the UNet likes to cheat (change appearance to raise reward) while the text encoder is forced to rearrange conditioning. This is exactly the inductive bias we want.

Our case is one step further upstream: not the text encoder itself but a projection `P` that feeds the text encoder's output-space (or CLIP pooled embedding, or T5 sequence — depending on where P lands). The gradient path `reward → R → encoder → Flux → P` is **identical in structure** to TexForce's `reward → ImageReward → UNet → text-encoder-LoRA`. Policy gradient of DDPO doesn't care which upstream parameters carry the score; as long as `P` receives gradient from `log π(a|s)` on every denoising step (it does, via the conditioning input to Flux), the update works.

Gotcha #1: **gradient reaches P only through the conditioning input to every denoising step**. With 10–40 denoising steps and Flux.1-dev's size, the autograd graph is large; LoRA on P helps, checkpointing is mandatory.

Gotcha #2: **a tiny P has low capacity to absorb reward variance**. If P is a 1024→4096 linear map (≈4M params), it may collapse to a single mode under reward pressure. Add KL to the pre-RL P (DPOK-style) or weight-decay to prior.

Gotcha #3: **rectified-flow divergence applies regardless of what you're tuning**. Even if gradients flow cleanly to P, the sampler still has to be made stochastic (ODE→SDE). Use Flow-GRPO's sampler as a drop-in.

## 4. Realistic compute budget (500-item corpus, 3×3 axes)

Reference points from published work:

- Flow-GRPO on SD3.5-M: **"at least 8 × 80 GB GPUs"** for full-parameter training, LoRA mode available for smaller rigs (VRAM not documented). Reward curves stabilize in ~1–2k iterations; a prompt batch of 64–128 with 16 denoising steps each is typical.
- DanceGRPO on FLUX.1-dev: **32 × H800** for the paper's headline numbers. Clearly overkill for a 500-prompt corpus.
- TexForce (LoRA on text encoder, DDPO, SD1.5): runs on 1 × A100 in published experiments; ~12–24 h to converge on a narrow task.

Rough extrapolation for our setup (1 × 24 GB, Flux.1-dev, LoRA on P only, Flow-GRPO sampler at T=10 training / T=28 inference, 500 prompts × 4 samples/prompt per iteration, ~500 iterations):

- Per-iteration cost: ~2000 Flux forward passes. At ~1.5 s/pass on a 4090 at 512² with T=10, that's ~50 min/iter raw sampling + ~10 min gradient update. **~1 GPU-hour per iteration.**
- To convergence: 200–500 iterations. **200–500 GPU-hours** order-of-magnitude, i.e. **1–3 weeks on a single 24 GB card**, or ~24 h on 8 × 80 GB.
- If you can make the reward differentiable (ReFL-style truncated backprop, T=4), you skip GRPO entirely and the budget drops to ~50–100 GPU-hours.

For a detective-puzzle-scale 3-axis × 3-level pattern (only ~9 latent modes to recover), this is heavy. Seriously consider: (i) caching all Flux generations once and training only R on them as a sanity check, (ii) trying differentiable ReFL-style before RL.

## 5. Open questions to answer empirically

1. **Does R train to ~100% accuracy on non-P Flux outputs?** If the frozen Flux + some hand-written conditioning already linearly separates the 9 modes, RL on P is unnecessary — P can be fit by supervised regression to any working conditioning. Do this first.
2. **Is reward differentiable through the sampler at T=4?** If yes, skip RL entirely. Measure gradient norm and noise.
3. **Does the ODE→SDE noise level wash out the signal P is supposed to carry?** Flow-GRPO adds stochasticity that may partially erase fine conditioning perturbations. Needs ablation.
4. **How does P interact with classifier-free guidance?** Flux.1-dev uses distilled guidance; conditioning perturbations get amplified. May need to RL-train with guidance disabled and re-enable at inference.
5. **Reward sparsity at start of training.** If R is untrained and reward is at chance, policy gradient has no signal. Warm-start R on random-P-generated images before RL.

## Key references (arXiv IDs)

- 2305.13301 DDPO
- 2305.16381 DPOK
- 2311.12908 Diffusion-DPO
- 2311.13231 D3PO
- 2311.15657 TexForce (conditioning-only RL, closest precedent)
- 2407.13734 Tutorial/review of RL-based diffusion fine-tuning (useful overview)
- 2505.05470 Flow-GRPO (first stable rectified-flow RL, supports Flux.1-dev)
- 2505.07818 DanceGRPO (unified rectified-flow GRPO, FLUX.1-dev results)
- 2503.11240 Sparse-reward alignment for diffusion (CVPR 2025)
- 2411.15247 Differentiable latent surrogate reward (avoids policy gradient entirely — may be preferable)
