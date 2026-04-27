---
status: live
topic: metrics-and-direction-quality
---

# Chaining two FluxSpaceEditPair nodes does not compose — the outer node wins

**2026-04-23 evening probe.** Tested on `elderly_latin_m`, seed 2026.

## Setup

Built `pair_compose_workflow` — a workflow that chains N `FluxSpaceEditPair` nodes in series before the KSampler:

```
UNet → FSEP(smile, s=A) → FSEP(race, s=B) → KSampler → VAE
```

Hypothesis: since the node returns a patched MODEL, stacking two should compose both edits additively.

## Result

A 2D sweep of smile ∈ {0.3, 0.5} × race ∈ {0.0, 0.1, 0.2, 0.3, 0.5} shows the same qualitative pattern in every row:

| race scale | visual result |
|---|---|
| 0.00 | smile present, Latino base unchanged |
| 0.10 | smile **gone**, Latino base unchanged |
| 0.20 | smile gone, Latino base unchanged |
| 0.30 | smile gone, face morphing to Indian |
| 0.50 | smile gone, strong Indian morph |

Changing the smile scale from 0.3 to 0.5 does not change the pattern — the smile is equally annihilated. This is not gradual interference; it is one-sided override.

Collage: `output/demographic_pc/probe_chain_sweep/collage.png`

## Why

Each `FluxSpaceEditPair.patch()` does:

```python
m = model.clone()
# … build ctx …
m.set_model_attn1_output_patch(attn_out_patch)
return (m,)
```

ComfyUI's `set_model_attn1_output_patch` is a single slot per model. When node1 patches its cloned model and passes it to node2, node2 clones again and calls `set_model_attn1_output_patch` — which replaces the patch node1 installed. Node1's `attn_out_patch` never runs. The race pair takes over entirely.

Race=0.0 escapes this because of the early-exit `if scale == 0.0: return (model,)` — the race node returns its input model unchanged, preserving the smile node's patch. That's why race=0 is the only cell with a visible smile.

## Implication

Simple graph chaining **cannot** compose two FluxSpace edits. This is a structural constraint of the node API, not a scale/weight tuning problem. The only paths forward:

1. **`FluxSpaceEditPairMulti` node.** Takes N `(cond_a, cond_b, scale)` triples. Runs N edit passes (caching attn per pair per block), then in the base pass combines all N pair-averages: `steered = attn + Σᵢ scaleᵢ·(edit_meanᵢ − attn)`. Single patch, single render. Clean.
2. **Patch-chaining inside the existing node.** `FluxSpaceEditPair.patch()` could read any currently-installed attn1 patch, store a reference, and call it from inside its own hook before applying its own steering. Enables arbitrary N via graph chaining without a new node. Subtle: need to handle `ctx["active"]` coordination across independent node instances.
3. **Sequential img2img.** Render pair 1 to PNG, re-encode as init latent, run pair 2 as img2img. Works today. Costs: 2× diffusion steps per composed edit, plus a VAE round-trip that loses fine detail.

Option 1 is the most direct; option 2 is clever but fragile. For the iterative corrective loop we need option 1 or 3.

## What this blocks

- `compose_iterative.py` — can't be built as a single-render loop without a multi-pair node.
- Any test of whether FluxSpace edits compose linearly in attention space — we don't know because we've never actually measured two edits simultaneously.
- Dictionary reuse across axes — each dictionary row assumes its effect is the effect; compositional behavior is untested.

## What this doesn't falsify

- FluxSpace as an edit mechanism (single edits still work).
- The iterative corrective loop concept (just needs a valid composition primitive).
- The dictionary-as-cache framing (still valid for single-axis lookup).
