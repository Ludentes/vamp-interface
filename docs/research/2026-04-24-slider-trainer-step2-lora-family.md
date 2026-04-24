---
status: live
topic: demographic-pc-pipeline
---

# Step 2 — LoRA family refresher + Flux specifics

Formalises the parameter-efficient fine-tuning (PEFT) design space so the
slider hyperparameters (step 4) are chosen with intent, not by cargo-cult
copy of the reference notebook. Citations in `[N]` form, sources at end.

## LoRA mechanics (recap)

**Core identity** [1]:
```
W = W₀ + ΔW      (standard fine-tune)
W = W₀ + (α/r) · B·A    (LoRA — ΔW constrained to rank r)
```
where `W₀ ∈ ℝ^{d_out × d_in}` is frozen, `B ∈ ℝ^{d_out × r}` init zeros,
`A ∈ ℝ^{r × d_in}` init Kaiming uniform. Trainable params per layer drop
from `d_in·d_out` (e.g. 3072² ≈ 9.4M for a Flux attn projection) to
`r·(d_in+d_out)` (rank 16 → ~100k — **~100× less**).

**Forward pass:**
```python
y = x @ W₀ + (alpha/rank) * (x @ A.T) @ B.T
```

**Why it works:** Hu et al. argue task adaptation has *low intrinsic
rank* — the weight *delta*, not `W₀`, lives on a low-dimensional
manifold.

**Design levers:**

| Knob | Typical range | Effect |
|---|---|---|
| `rank (r)` | 4–128 | capacity; larger = more params, more overfit risk |
| `alpha (α)` | 1–2r | effective LR on the delta (since `scale = α/r`) |
| `target_modules` | `attn`, `mlp`, all | which layers get LoRA'd |
| dropout | 0–0.1 | regularisation on the LoRA-branch activations |

**Alpha convention:** PEFT ships `α = r` by default (scale = 1). The
Concept-Sliders notebook uses `α = 1, r = 16` → **scale = 1/16**. That's
a deliberately *conservative* injection — the LoRA contribution is 16×
smaller at init than a full fine-tune would be. Justification: sliders
are narrow edits; strong injection confuses the base model.

## The PEFT family (LoRA-and-relatives)

### LoRA — the baseline [1]
- Simplest, fastest, most-tested.
- Scaling `α/r` becomes unstable as `r` grows → the original paper
  reports rank 4–32 is "enough"; **rsLoRA** shows this saturation is an
  artefact of the scaling, not a real capacity limit.

### rsLoRA — rank-stabilised scaling [2]
- Swap `α/r` → `α/√r`.
- Lets you use rank 64–2048 stably, which genuinely helps on
  high-capacity tasks.
- Zero inference overhead (same param count, just different scalar).
- **Relevant if** rank 16 underperforms on a difficult slider axis and
  we want to push to rank 64/128 as a fallback.

### DoRA — weight-decomposed [3] (ICML 2024 oral)
- Decomposes `W = m · (V/||V||)` into magnitude `m ∈ ℝ^{d_out}` and
  direction `V ∈ ℝ^{d_out × d_in}`.
- Trains `m` fully + LoRA on direction.
- Consistent +1–4% improvements across LLM / VLM / diffusion / VL-BART.
- **Catch:** ~1.3–1.5× slower training (extra magnitude backward pass).
- **HF diffusers status:** supported via PEFT but explicitly "still
  experimental" for diffusion. Not derisked on Flux specifically.

### LoHa (LoRA Hadamard) [4]
- `ΔW = (A₁·B₁) ⊙ (A₂·B₂)` — Hadamard product of two LoRAs, four matrices total.
- 2× the params of same-rank LoRA but can express rank up to `r²`
  because Hadamard is multiplicative.
- Community verdict: good for *multi-concept* training (e.g. character
  LoRAs that need both face + outfit). Likely overkill for single-axis
  sliders.

### LoKr (LoRA Kronecker) [4, 5]
- `ΔW = A ⊗ B` — Kronecker product of two small matrices.
- Can be extremely parameter-efficient: a `d×d` delta from
  `(√d × √d) ⊗ (√d × √d)` is `2d` params instead of `d²`.
- `factor` parameter controls the split. `factor=16` + attention module
  is a common Flux config.
- Good for *style* LoRAs where the delta has structural regularity.
  Slider axes don't obviously have that structure.

### VeRA [6] (not covered above but relevant)
- Shares a single pair of frozen random A, B across all layers; only
  tiny scaling vectors `b, d` are trainable per layer.
- Extreme param efficiency (~100× fewer than LoRA) but noticeable
  quality drop. Skip.

### AdaLoRA / PiSSA
- Adaptive rank allocation, init from SVD of `W₀`. Mostly LLM
  territory, not derisked on diffusion. Skip for v1.

### Family cheat-sheet

| Method | Params (vs LoRA r=16) | Train speed | Inference | Sliders fit? |
|---|---|---|---|---|
| LoRA | 1× | 1× | merge → free | **baseline** |
| rsLoRA | same at same r | 1× | merge → free | fallback if r=16 underfits |
| DoRA | 1.05× | 0.7× | merge → free | **experimental on Flux**, try v2 |
| LoHa | 2× | 0.9× | no-merge (Hadamard) | overkill |
| LoKr | 0.1–1× | 0.95× | merge → free | style-tuned, not semantic |
| VeRA | 0.01× | 1× | no-merge | too restrictive |

**Recommendation:** LoRA v1, rsLoRA fallback, DoRA as v2 experiment on
one axis after v1 works end-to-end.

## Flux DiT specifics

### Architecture recap [7]

Flux-dev (and Krea-dev) has:
- **19 "double" transformer blocks** = MMDiT joint-attention blocks
  (image + text tokens attend together). Class
  `FluxTransformerBlock` in diffusers. Modules: `attn.to_q, to_k,
  to_v, to_out.0, add_q_proj, add_k_proj, add_v_proj, to_add_out` +
  separate MLP for text and image streams.
- **38 "single" transformer blocks** = post-concat single-stream
  blocks (image + text as one sequence). Class
  `FluxSingleTransformerBlock`. Modules: `attn.to_q, to_k, to_v, to_out` + `proj_mlp, proj_out` (fused MLP).
- **3072-dim hidden** state.
- Total ~12B params in bf16 (~23.8 GB).

### What Concept-Sliders-Flux actually trains [8]

Reading the notebook (`docs/research/external/train-flux-concept-sliders.ipynb`):

```python
alpha = 1
rank = 16
train_method = 'xattn'
```

The `LoRANetwork` class (ported from the SDXL implementation —
`DEFAULT_TARGET_REPLACE = ["Attention"]`) walks the transformer and:
- Matches any submodule whose class name is `Attention` (diffusers
  uses this name in the Flux attention class in current versions).
- Within each match, inserts LoRA on every `Linear` child.
- `train_method='xattn'` filters by the substring `"attn"` in the
  full module path — on Flux this matches **every attention module in
  both double and single blocks**. The SDXL name ("xattn" =
  cross-attention only) is misleading on Flux, where self/cross
  attention are fused in MMDiT. On Flux, `xattn` effectively means
  "all attention, no MLP".

**Net result on Flux:**
- 19 doubles × 8 attn linears + 38 singles × 4 attn linears = **304
  target linears**.
- Rank 16 + alpha 1 on each → **~30 MB of LoRA params** total.
- Attention-only leaves all MLP paths untouched — the base model's
  "knowledge" is preserved; only the *routing* of tokens changes.

### Alternatives we could pick instead

1. **`train_method='full'`** — all linears in all transformer blocks.
   ~5× more params (~150 MB). Higher capacity but higher overfit risk
   and ~2× more VRAM for activation state. Reference recipe says not
   needed for concepts.
2. **Single-blocks only** [9] — Gözükara shows training only single
   blocks preserves quality with less disk/VRAM. For sliders this is
   plausible but untested in the reference repo.
3. **Specific block indices** — "concept lives in middle layers"
   intuition from SDXL. Not established on Flux; skip.

### Why xattn is the right default for our slider use case

- Our axes are **semantic token routing edits** (smile, eye squint,
  gaze direction) — these are exactly the routing decisions attention
  makes.
- Leaving MLPs frozen preserves base-model identity / style quality,
  which matters for our downstream pipeline (we want axis-specific
  edits, not a general Flux mod).
- Inference compatibility: standard LoRA merge is a no-op at inference
  time; attention-only LoRA merges just as easily.
- Memory: MLP params dominate Flux — skipping them saves backward
  activation memory (~2–3 GB at 512×512 with grad ckpt).

## Parametric decision tree

```
v1 (this build):        LoRA, rank=16, α=1 (scale 1/16), xattn, bf16
                        → matches reference recipe, ~30 MB params,
                          60–90 min/axis on 5090

if v1 underfits on an axis:
  → try rsLoRA, rank=64, α=8 (scale 1/√64=0.125), xattn, bf16
    (same scale regime, 4× capacity)

if v1 overfits / axis leaks into other axes:
  → try LoRA rank=8 xattn (half capacity) OR dropout 0.1 on LoRA branch

v2 experiment (after v1 works):
  → DoRA rank=16 xattn on one axis (eye_squint), compare to v1 baseline
```

## What changes for our image-pair objective (preview of step 4)

Everything above is about the **parametrisation**. Our training signal
differs from the reference notebook: we have `(α=0.png, α=1.png)` pairs
instead of `(positive_prompt, negative_prompt)` teachers. The LoRA
architecture doesn't care — A·B-style rank adaptation is a property of
the *network*, not the *loss*. We keep the recipe, swap the loss.

Loss design is step 4 work.

## Gaps / open questions

- **Is `Attention` still the class name in current diffusers' Flux?**
  Recent diffusers renamed it `FluxAttention` in some versions. Must
  verify on the downloaded model before v1 run. If renamed,
  `DEFAULT_TARGET_REPLACE` must update to `["FluxAttention"]` or use
  duck-typing on the attribute name.
- **Does Flux Krea have the same architecture as Flux-dev?** Expected
  yes (it's a fine-tune of dev, not a new architecture), but confirm
  by inspecting the downloaded state dict shape.
- **LoKr with `factor=16` on attention:** community reports this works
  on Flux for styles, but no data on semantic sliders. Could be a v2
  experiment alongside DoRA.

## Next

Step 3: Validation protocol — hold-out splits, primary/guardrail
metrics, pass/fail thresholds.

## Sources

[1] Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685. [link](https://arxiv.org/abs/2106.09685)

[2] Kalajdzievski. *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA*. arXiv:2312.03732. HF blog: [link](https://huggingface.co/blog/damjan-k/rslora)

[3] Liu et al. *DoRA: Weight-Decomposed Low-Rank Adaptation*. arXiv:2402.09353. ICML 2024 oral. [NVIDIA blog](https://developer.nvidia.com/blog/introducing-dora-a-high-performing-alternative-to-lora-for-fine-tuning/)

[4] LyCORIS: [github.com/KohakuBlueleaf/LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) — LoHa, LoKr, LoCon, DyLoRA implementations.

[5] HF PEFT LoKr docs: [link](https://huggingface.co/docs/peft/en/package_reference/lokr)

[6] Kopiczko et al. *VeRA: Vector-based Random Matrix Adaptation*. arXiv:2310.11454.

[7] HF diffusers Flux architecture: 19 double + 38 single transformer blocks, 3072-d. See `FluxTransformer2DModel` source in diffusers repo.

[8] `docs/research/external/train-flux-concept-sliders.ipynb` cell 3: `rank=16, alpha=1, train_method='xattn'`. Uses `rohitgandikota/sliders/flux-sliders/utils/lora.py` port.

[9] Gözükara. *Single Block / Layer FLUX LoRA Training Research Results*. Medium. [link](https://medium.com/@furkangozukara/single-block-layer-flux-lora-training-research-results-and-lora-network-alpha-change-impact-with-e713cc89c567)
