---
status: archived
topic: archived-threads
summary: DRaFT-1 / DRaFT-LV (last-step backprop) are the cheapest differentiable paths for reward fine-tuning on Flux; Adjoint Matching provides theoretical grounding if needed.
---

# Differentiable Flux Sampling — How to Get Gradients Through Rectified-Flow ODE Integration

**Date:** 2026-04-16
**Question:** Given the pipeline `qwen-embedding → projection P → Flux conditioning → Flux sampling → face → face-encoder → scalar loss ℒ`, with Flux.1-dev frozen, how do we compute `∂ℒ/∂θ` where θ = params of P?
**Audience:** Us, before we spend GPU time. Base = Flux.1-dev, single 24 GB consumer GPU, ComfyUI inference works today.

---

## 1. TL;DR

**Start with DRaFT-1 / DRaFT-LV** (last-step-only backprop) on Flux.1-dev: cheapest memory, simplest code, proven on SD and proven to work for the *adjacent* task (reward fine-tuning). If the 1-step gradient is too biased to train P well, climb to **DRaFT-K with gradient checkpointing** (K≈4–10). If you want an exact, principled answer and are willing to read one hard paper, **Adjoint Matching (Domingo-Enrich et al., ICLR 2025, arXiv:2409.08861)** is the only published method designed for Flux/SD3 reward fine-tuning with memoryless noise schedules, and Microsoft ships working code. Don't start with neural-ODE adjoint or distillation — both are strictly more work for this project.

---

## 2. Primary techniques

### 2.1 Neural-ODE adjoint sensitivity (Chen et al. 2018, arXiv:1806.07366)

Flux's velocity field `v_θ(x_t, t, c)` is literally the RHS of an ODE, so in principle the adjoint method applies: solve an augmented backward ODE for `∂ℒ/∂c` with O(1) activation memory. **(a) Flux-specific?** No published work applies adjoint sensitivity to Flux or SD3; the Flux DiT is 12 B params and the backward ODE re-integration will call it ~50 more times. **(b) Memory:** O(1) in step count, so ~single-forward memory for the state — but you still hold the DiT and its backward graph for one step at a time. **(c) Compute:** ≈2–3× forward pass (backward integration re-runs the solver). **(d) Bias:** exact in the continuous limit, but numerical adjoint gradients are known to diverge from true backprop under adaptive solvers and sometimes under fixed solvers — see Symplectic Adjoint (Matsubara et al., NeurIPS 2021, arXiv:2102.09750) and "Efficient, Accurate and Stable Gradients for Neural ODEs" (arXiv:2410.11648) for gradient-fidelity fixes. **(e) Maturity:** `torchdiffeq` exists but has never been wired to Flux/SD3; you would be the first. **Verdict:** theoretically clean, empirically untested on rectified-flow transformers, high integration risk. Skip unless memory is the binding constraint.

### 2.2 Rectified-flow inversion (RF-Solver-Edit, FireFlow, RF-Inversion)

RF-Solver-Edit (Wang et al., ICML 2025, arXiv:2411.04746) and FireFlow (ICML 2025) provide high-precision deterministic inversion for Flux via higher-order Taylor corrections to the ODE step. **These give reconstruction, not differentiability.** Inversion recovers `x_T` from `x_0` (and vice versa) as a frozen, approximately-invertible map; it does not by itself produce `∂ℒ/∂c`. You *could* combine deterministic inversion with implicit-function-theorem gradients (treat the inversion map as an implicit layer), but no one has published this for Flux. Related: **FlowChef** (Patel et al., ICCV 2025, arXiv:2412.00100) explicitly *avoids* backprop through the ODE and uses rectified-flow straight-trajectory geometry to approximate gradients — "gradient skipping." FlowChef targets inference-time steering of a frozen Flux, not training a learned projection. **Verdict:** inversion ≠ differentiation; don't plan on this path for gradient computation. FlowChef is relevant as a *different* primitive (training-free steering), not as a gradient source for training P.

### 2.3 Consistency / few-step distillation (InstaFlow, Flux Schnell)

**Flux Schnell is already 1–4-step distilled** via latent adversarial diffusion distillation (LADD) and ships under Apache-2.0. If we accept Schnell as the base instead of Flux-dev, sampling is `Cond → Image` in 1–4 transformer calls and is trivially differentiable end-to-end with gradient checkpointing; memory and compute drop by ~12–50× vs. 50-step dev. **(a) Flux-specific?** Yes — Schnell *is* the Flux distillation. **(b) Memory:** at 4 steps w/ checkpointing, ~15–20 GB estimated for Flux-12B fp16 — tight on 24 GB but plausible; at 1 step, comfortable. **(c) Compute:** 1–4× forward per gradient vs 50× for dev. **(d) Bias:** exact gradients through the Schnell sampler, but Schnell's image distribution differs from dev's — if the final downstream tool (ComfyUI face pre-gen) uses dev, there is a train/eval distribution gap. **(e) Maturity:** Schnell is stable; distilling Flux *dev* yourself via InstaFlow-style reflow (Liu et al., ICLR 2024, arXiv:2309.06380) is a several-hundred-GPU-hour project and likely off-budget. **Verdict:** if you're willing to switch the base to Schnell, this is the cleanest path. If you must stay on dev, the distillation route is too expensive.

### 2.4 Truncated / last-step backprop (DRaFT-1, DRaFT-LV)

DRaFT (Clark et al., ICLR 2024, arXiv:2309.17400) backprops through only the last K steps of sampling. DRaFT-1 uses K=1, and DRaFT-LV reduces variance by averaging over noise draws at the last step. **(a) Flux-specific?** The paper uses SD; the mechanism is solver-agnostic and transfers to rectified-flow transformers — `ReFL` (Xu et al., NeurIPS 2023, ImageReward) and DRaFT-LV already target a randomly-picked latter step and work unchanged on flow matching. **(b) Memory:** ~1× forward pass (one DiT backward graph). Fits easily on 24 GB. **(c) Compute:** ~1× forward per update for DRaFT-1. **(d) Bias:** biased — the 1-step gradient ignores how earlier denoising steps would have reshaped the trajectory. Empirically DRaFT-1 and DRaFT-LV match full backprop on aesthetic/alignment rewards and sometimes beat it due to lower variance. **(e) Maturity:** production-grade; used in AlignProp and many follow-ups. **Verdict:** correct starting point for this project.

---

## 3. Adjacent survey methods

- **AlignProp** (Prabhudesai et al., 2023, arXiv:2310.03739). Full-trajectory reward backprop on SD with LoRA + gradient checkpointing; 25× more sample-efficient than PPO. Uses the same ingredients (checkpointing + LoRA adapter on the frozen base) we would use. Reference code at `github.com/mihirp1998/AlignProp`.
- **ReFL** (ImageReward, Xu et al., 2023, arXiv:2304.05977). Applies reward gradient at a single randomly-picked late step, with a quality anchor. Essentially DRaFT-1 with a random step choice. Proven stable on SD.
- **Adjoint Matching** (Domingo-Enrich et al., ICLR 2025, arXiv:2409.08861). The only published gradient-based fine-tuning method *explicitly validated on Flux.1 and SD 3.5*. Frames reward fine-tuning as stochastic optimal control and proves a **memoryless noise schedule** is required when fine-tuning flow-matching models (vanilla reward backprop is biased on rectified flows because of the noise-to-sample dependency). Reference code at `github.com/microsoft/soc-fine-tuning-sd`. This is the *theoretically correct* version of what DRaFT does on rectified flows.
- **ReNeg** (Li et al., CVPR 2025, arXiv:2412.19637). Learns negative conditioning embeddings via reward gradients. Direct structural analogue to "learn a projection P that produces Flux conditioning." Code at `github.com/AMD-AGI/ReNeg`. Closest precedent to our exact setup.
- **Value Gradient Guidance (VGG-Flow)** (arXiv:2512.05116, NeurIPS 2025). Gradient-matching alternative to DRaFT for flow matching; useful later, not to start with.

---

## 4. Ranking — easiest first on 24 GB with frozen Flux.1-dev

| Rank | Method | One-shot? | External deps | Why |
|---|---|---|---|---|
| 1 | **DRaFT-1 / DRaFT-LV** on Flux-dev | Yes | `diffusers` + LoRA on P, gradient checkpointing | ~1× forward memory, ~1× compute, trivial code, works on any flow model |
| 2 | **Adjoint Matching** on Flux-dev | Yes | `microsoft/soc-fine-tuning-sd` | Theoretically correct for rectified flows; reference Flux code exists |
| 3 | **Switch base to Flux Schnell, full backprop** | Yes | `diffusers` Schnell pipeline | Trivially differentiable, but train/eval distribution gap vs dev |
| 4 | **AlignProp-style full-trajectory backprop w/ checkpointing** on Flux-dev | Yes | `diffusers`, heavy checkpointing | Exact gradients but ~50× compute and tight on 24 GB |
| 5 | **Neural-ODE adjoint** | No | `torchdiffeq` + custom Flux wrapper | No reference implementation for Flux; unknown numerical stability on MM-DiT |
| 6 | **Distill Flux-dev ourselves via InstaFlow reflow** | No | Hundreds of GPU-hours | Out of budget for this project |

---

## 5. Flux-specific reference code

- **Adjoint Matching + Flux:** `github.com/microsoft/soc-fine-tuning-sd` — reward fine-tuning for SD3 and Flux with memoryless noise schedules. **This is the closest thing to a reference implementation for our exact problem** (gradients through Flux sampling), and the only repo we found that explicitly fine-tunes Flux with reward gradients.
- **AlignProp + SD (port to Flux):** `github.com/mihirp1998/AlignProp` — full reward backprop with LoRA + checkpointing. Needs porting to Flux's MM-DiT backbone.
- **ReNeg + SD (port to Flux):** `github.com/AMD-AGI/ReNeg` — learns a conditioning embedding via reward gradients, the exact mechanism we want for P.
- **RF-Solver-Edit:** `github.com/wangjiangshan0725/RF-Solver-Edit` — not a gradient source, but the reference high-precision ODE solver for Flux. Useful if we need inversion to construct a ground-truth trajectory for supervised training of P.
- **FlowChef:** `github.com/FlowChef/flowchef` — training-free steering on Flux, *gradient-skipping* through the solver. Not a gradient source; relevant as a zero-training fallback if learning P proves too costly.

---

## 6. What's unknown — we'd have to measure it ourselves

1. **Does DRaFT-1 give a useful gradient for training *projection* P (vs. fine-tuning the DiT)?** All published DRaFT/AlignProp/ReFL work optimizes DiT weights. We want to optimize a ~1024→(T5+CLIP-pooled) projection upstream of the conditioning. The gradient arrives at the conditioning tensor; we need to empirically verify that `∂ℒ/∂c` propagated through one Flux step has enough signal to train a small linear/MLP P.
2. **Memoryless noise schedule bias on Flux-dev specifically.** Adjoint Matching proves standard fine-tuning is biased on rectified flows without a memoryless schedule. DRaFT predates this observation and doesn't use one. Does the bias matter in practice for *our* loss (face-encoder similarity), or only for certain reward classes (aesthetic/alignment)? Needs an ablation.
3. **Flux-dev guidance-distilled CFG interaction.** Flux-dev is guidance-distilled; it does not accept a genuine CFG scale. The DRaFT paper assumes classical CFG. Gradients through Flux-dev's learned guidance-embedding conditioning path have not been published.
4. **Exact 24 GB memory headroom at 50 steps with LoRA + checkpointing on Flux-12B fp16/bf16.** Order-of-magnitude estimates say "should fit" but we have not measured. Expect to spend the first day of implementation on VRAM tuning.
5. **Whether inversion-based implicit differentiation on RF-Solver could replace DRaFT.** No one has published this. It is a 1–2-person-week research bet, not a drop-in.

---

## 7. Recommendation

Build it in this order, stop when it works:

1. Prototype **DRaFT-1** on Flux-dev with LoRA-parameterized P (or plain MLP on the conditioning). One batch, one loss evaluation. Measure: does P's gradient norm track ℒ?
2. If signal is too noisy, upgrade to **DRaFT-LV** (average last-step gradient over multiple noise draws) or **DRaFT-K** with K=4 and gradient checkpointing.
3. If DRaFT is visibly biased (ℒ plateaus or drifts), switch to **Adjoint Matching** with memoryless schedule, using the Microsoft reference code as the starting scaffold.
4. Only if none of the above work, consider **switching base to Flux Schnell** (cleanest gradients, different image distribution) or **full-trajectory AlignProp backprop** (exact, expensive). Do not implement neural-ODE adjoint unless steps 1–4 have all failed for identifiable reasons.

---

## Citations (arXiv IDs)

- Chen et al., Neural ODEs: **1806.07366**
- Esser et al., Scaling Rectified Flow Transformers (SD3/MM-DiT): **2403.03206**
- Liu et al., InstaFlow: **2309.06380**
- Clark et al., DRaFT: **2309.17400**
- Prabhudesai et al., AlignProp: **2310.03739**
- Xu et al., ImageReward / ReFL: **2304.05977**
- Wang et al., RF-Solver-Edit: **2411.04746**
- Patel et al., FlowChef: **2412.00100**
- Domingo-Enrich et al., Adjoint Matching: **2409.08861**
- Li et al., ReNeg: **2412.19637**
- Matsubara et al., Symplectic Adjoint: **2102.09750**
- Liu et al., Efficient Accurate Stable Gradients for Neural ODEs: **2410.11648**
- Value Gradient Guidance (VGG-Flow): **2512.05116**
