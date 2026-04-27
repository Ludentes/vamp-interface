---
status: live
topic: demographic-pc-pipeline
---

# Flux MMDiT attention — what to train for a concept slider

**Date:** 2026-04-26

Context: `glasses_slider_v0` (ai-toolkit `concept_slider` extension on Flux Krea) blew up to pure noise at lr=2e-3 even with α-override compensation in place. Audit found three compounding errors — α-override, 494-module scope, two backwards/step. This note logically decomposes the attention layer to argue *which* of those modules actually carry the slider direction, and which are just bleed surface.

## Flux architecture, relevant slice

Flux is **MMDiT**, not a SD/SDXL UNet. There is **no separate cross-attention module** — the analog of "cross-attn" is the joint attention inside double_blocks.

| Stack | Count | Inside each block | Function |
|---|---|---|---|
| `transformer_blocks` (a.k.a. **double_blocks**) | 19 | Separate img + txt streams. `attn.to_{q,k,v,out.0}` (img), `attn.add_{q,k,v}_proj` + `to_add_out` (txt), `ff` (img MLP), `ff_context` (txt MLP) | Joint MMDiT attention — text and image tokens mix here. |
| `single_transformer_blocks` | 38 | Combined stream. `attn.to_{q,k,v}`, `proj_mlp`, `proj_out` | Self-attention over the already-mixed joint sequence. Refines rendering; no new text↔image conditioning. |

In each double block the joint attention is:

```
q_img,k_img,v_img = to_q(img), to_k(img), to_v(img)
q_txt,k_txt,v_txt = add_q_proj(txt), add_k_proj(txt), add_v_proj(txt)

Q = [q_img ; q_txt]       # concat along sequence axis
K = [k_img ; k_txt]
V = [v_img ; v_txt]
attn = softmax(Q K^T / √d) V
img_out = to_out.0(attn[:N_img])
txt_out = to_add_out(attn[N_img:])
```

The attention matrix decomposes into four quadrants:

```
            k_img             k_txt
       ┌──────────────┬──────────────────┐
q_img  │ img → img    │ img → txt        │
       │ (geometry)   │ (conditioning IN)│
       ├──────────────┼──────────────────┤
q_txt  │ txt → img    │ txt → txt        │
       │ (text reads  │ (text self-      │
       │  image)      │  refine)         │
       └──────────────┴──────────────────┘
```

For any concept slider, the load-bearing quadrant is **img → txt** — the only place where a prompt token actually reaches an image patch.

## What Q, K, V mean (smile example)

| Role | "Question" it answers | Concrete for `smiling` |
|---|---|---|
| `to_q` (q_img) | What is this image patch looking for? | A mouth-corner patch asks: "is there a smile cue to pull?" |
| `to_k` (k_img) | What does this image patch advertise? | "I am mouth geometry." |
| `to_v` (v_img) | What payload does this patch deliver when attended to? | Pixel-feature contribution to other patches. |
| `add_q_proj` (q_txt) | What does the text token read from the image? | Token `smiling` queries: "do you currently look smiling?" — updates the text rep across blocks. |
| `add_k_proj` (k_txt) | What does the text token advertise to image patches? | "I am the smile concept — patches looking for emotion cues, attend here." |
| `add_v_proj` (v_txt) | What payload does it deliver into attending img patches? | Smile semantics — lip curvature, cheek lift — added to the image residual. |

## Where the slider direction lives

Slider training compares two forwards at the same noisy latent and seed:

- positive: `"...person smiling..."`
- negative: `"...person not smiling..."`

What differs between the two passes at step 0:

| Term | pos vs neg diff | Why |
|---|---|---|
| q_img, k_img, v_img | **0** | Image stream sees identical noisy latent. |
| q_txt, k_txt, v_txt | **large** | Different text tokens → different projections. |
| `to_out`, MLPs | 0 at first call, diverges through residual stream | Compound effect of upstream divergence. |

The signal that distinguishes pos from neg at the entry point is **almost entirely in `add_k_proj` and `add_v_proj`** — the projections that turn the word `smiling` into an attention-routing key and a content payload. That is the minimal lever a slider could grab.

`add_q_proj` matters secondarily — it controls how the text representation updates block-to-block. Less direct but compounds across depth.

`to_q` matters because image patches need to learn "when the smile token is in scope, my query should orient toward it." Without `to_q` adaptation, the image side cannot route attention differently between pos and neg.

`to_k`, `to_v` are mostly img → img (geometry). Touching them lets the LoRA cheat by directly drawing smile pixels rather than going through text conditioning.

`to_out` / `to_add_out` and the MLPs are *post-mix rendering*. Training them = "memorize how a smile looks on these particular faces." High bleed risk.

`single_transformer_blocks.*.*` is *all* post-mix. Same bleed argument, more strongly.

## Scope ladder

Surgical → permissive, with module count for rank-16 LoRA across all 19 double_blocks:

| Scope | Modules / block | Total | Hypothesis |
|---|---|---|---|
| **Surgical**: `add_k_proj`, `add_v_proj` | 2 | 38 | Most direction-pure; smallest expressive bottleneck. May underfit visually-rich concepts. |
| **Targeted**: + `to_q` | 3 | 57 | Adds img-side query adaptation so patches learn to attend to the concept token. |
| **Notebook xattn**: + `add_q_proj`, `to_k`, `to_v`, `to_out.0`, `to_add_out` | 8 | ~152 | All attention I/O, both streams. Canonical Concept Sliders recipe. |
| **Default (current run)**: + `ff`, `ff_context`, + everything in single_transformer_blocks | ~26 | ~494 | Maximum capacity, maximum bleed. What blew up at lr=2e-3. |

The 6× parameter inflation between notebook and default is not just a count — at the same nominal LR the total update norm per step is ~6× larger, which is a material part of why the run blew up independently of the α-override and two-backwards issues.

## Concept-dependence of the right scope

Asymmetry worth noting:

- **Smiles** are mostly a semantic relabel of an existing facial config (the mouth is already there; we're modulating its shape). Surgical `add_k/v_proj`-only might suffice.
- **Glasses** require rendering a new object that wasn't in the unconditioned prediction. The surgical scope may underfit because none of the `to_*` projections get to learn glasses geometry.
- **Demographic axes** (age, gender, race) blend both — semantic relabel + new visual content (wrinkles, beard, skin texture).

So the "right" scope likely differs by axis. The notebook chose all-xattn as a one-size-fits-all middle ground. We may eventually want axis-specific scopes once we have a falsifiable test for "did the LoRA learn the concept direction or memorize the appearance."

## ai-toolkit knobs

ai-toolkit has no `train_method` parameter, but `network:` accepts:

- `only_if_contains: [list]` — substring OR-match (any clause must hit). Loose include filter.
- `ignore_if_contains: [list]` — substring OR-match (any clause is excluded).

To approximate notebook xattn on Flux without code changes:

```yaml
network:
  type: "lora"
  linear: 16
  linear_alpha: 1
  ignore_if_contains:
    - "single_transformer_blocks"   # drops the whole single stack
    - "ff"                          # drops img MLP
    - "ff_context"                  # drops txt MLP
    - "norm"                        # drops AdaLN modulation
```

This leaves `transformer_blocks.*.attn.*` only — both streams' Q/K/V/out — which is the notebook's xattn target.

For the surgical scope, swap to `only_if_contains: ["add_k_proj", "add_v_proj"]` (OR-mode is fine here because we want the union of the two).

For the targeted scope (+ `to_q`), `only_if_contains: ["add_k_proj", "add_v_proj", "to_q"]` works — but note `to_q` substring also matches `add_q_proj` (since `q_proj` ⊃ `to_q`? actually no: `to_q` is its own name, `add_q_proj` does not contain `to_q`). Quick verification needed before relying on this.

## Recommendation for current debug

After the `lr=8e-5` run finishes, if behavior is still off:

1. Drop scope to notebook xattn (`ignore_if_contains` filter above), bump LR back toward 1.25e-4. Tests the "scope, not LR" hypothesis.
2. If notebook xattn works, then run a controlled comparison: surgical vs targeted vs xattn at fixed compute budget, evaluate by axis-purity (does ±1.5 produce a clean glasses delta without identity drift?) rather than by training loss.
3. Result feeds into a per-axis scope policy for the broader slider corpus (glasses, smile, age, gender, race, facial-hair, ...).

## Open questions

- Does ai-toolkit's `peft_format` save path emit a valid safetensors when LoRA scope is non-default? Should be — it only writes adapters that exist — but worth a smoke-load before the inference sweep.
- Is the AdaLN modulation (`norm` modules) ever load-bearing for sliders? They modulate residual streams per-timestep; might matter for time-localized concept emergence. Not addressed here.
- Surgical scope (`add_k/v_proj` only) — is it real underfitting or does it just need higher rank to compensate? Open empirical question.
