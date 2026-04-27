---
status: live
topic: metrics-and-direction-quality
---

# Anchor-cache architecture for N-variation portrait generation

When the workload is "one anchor, many edit combinations" (smile 0.3 + beard 0.1; smile 0.4; smile 1.0 + AU4 0.7; smile 0.3 + gender 0.3; …), the current FluxSpaceEditPairMulti pays for full 2N+1 forward passes per variation. Most of that work is redundant across variations. A three-layer cache can collapse it.

## What is redundant

**`attn_base[step, block]`** — the base model's attention pattern per sampling step per transformer block. Depends only on `(base_prompt, seed, step_idx, block_idx)`. Identical across every variation on the anchor. Currently recomputed in the base pass every render.

**`attn_a[step, block]` and `attn_b[step, block]`** for a given pair — `edit_mean_i` on a given anchor is identical across scales and across the presence or absence of other slots. Render "smile 0.3" once; you already have everything the solver needs for "smile 0.4," "smile 1.0," "smile 0.3 + anything."

**Text embeddings** — ComfyUI caches within a workflow but not across; trivially hash prompt → (cross, pooled).

## The caching lifecycle

**Phase 1 — anchor warmup.** One full sampling run of the unedited anchor. Dump `attn_base[step, block]` tensors + final latent + seed + base_prompt to disk. Cost: one render.

**Phase 2 — axis library build.** For each unique pair on this anchor, one sampling run that captures `attn_a[step, block]` and `attn_b[step, block]`. Cost: one render per pair.

**Phase 3 — variations.** Each requested combination runs only the base pass with the attn1 patch injecting cached deltas:

```
steered = attn_base_cached + Σᵢ scaleᵢ · (edit_meanᵢ_cached − attn_base_cached)
```

No edit passes. No base recomputation. Each variation is ≈1 forward pass per step.

## Speedup arithmetic

For 20 variations on one anchor using a 4-axis pair library:

- Current: `20 × (2·2+1) = 100` forward passes per step (assuming avg 2 slots per variation).
- Cached: `1 (anchor) + 4 (library) + 20 × 1 (variations) = 25` per step.

~**4× speedup** for N=20 at 2-axis average; scales to ~7× at N=100 with the same library. Adding a new pair costs one extra library render, amortised across all variations using it.

## What does NOT transfer

- **Seed / noise-trajectory specificity.** `attn_a[step]` depends on the live latent `x_t` at step t, which depends on the anchor seed. A library is keyed on `(anchor_prompt, seed)`, not on the axis alone. A new anchor requires a new library.
- **Base-prompt specificity of pairs.** The same pair on a different anchor prompt will produce a different `edit_mean` because the cross-attention is computed against a different base latent. Confirmed by the counter-edit base-specificity finding (see `2026-04-23-session-findings.md`).
- **Scale values** are *not* cached — they remain a runtime parameter applied to the cached δ.

## What breaks the linearity assumption

The cached-δ injection assumes `(edit_meanᵢ − attn_base)` is the whole story and that scaling + summing reproduces what a live-rendered multi-pair would produce. This is **exactly the additivity claim** the smoke test validated at the pixel level (A = single slot s=0.5 vs B = two slots 0.25+0.25, visually identical). So the cache-inject path reduces to the same math as the live multi-pair path — it just reuses tensors instead of recomputing them.

One caveat: the *base-pass* attention in a live multi-pair run depends on what the edit passes deposit in model internals (KV caches, position embeddings touched by the same module). We assume stateless forwards; `FluxSpaceEditPair` already relies on this. Verified implicitly by the smoke test working.

## Implementation

Split `FluxSpaceEditPairMulti` into three cooperating nodes:

- **`FluxSpaceAnchorCache`** — one-shot sampling that writes `anchor.v1.npz` containing `attn_base[step, block, L, D]` and the final latent + metadata (seed, base_prompt, steps, sampler, scheduler).
- **`FluxSpacePairLibraryCache`** — takes an anchor reference + one pair, runs sampling with caching hooks (no delta injection), writes `pair_{label}.v1.npz` containing `attn_a[step, block, L, D]` and `attn_b[step, block, L, D]`.
- **`FluxSpaceEditPairMulti` cache-mode** — accept `anchor_cache_path` + list of `pair_cache_path + scale + mix_b`. Skip edit passes and the model-side base pass; instead, call apply_model with the attn1 patch reading from disk tensors per `(step, block)` lookup. Return the latent.

Cache format: same `.v1.npz` convention already used by the measurement dump. fp16, per-block keys, step keys. Expected size: ~15 MB per block × 16 blocks = ~250 MB per anchor, similar per pair. Library of 4 axes = ~1 GB of cache for one anchor; manageable.

## Memory vs I/O tradeoff

- **In-RAM cache.** Hold the entire anchor + library in GPU memory during Phase 3. Fastest, but ~1 GB extra VRAM for a 4-pair library.
- **Disk-streamed cache.** Load `(step, block)` slice on demand. Slower per-variation (disk latency × steps × blocks) but zero VRAM overhead.
- **Compromise.** Load per-step slice into RAM at the start of each step — one bulk read per variation per step.

Start with in-RAM; fall back to streamed if VRAM becomes an issue.

## Unblocks bonus threads

- **Cache-δ axis extraction** (milestone 8). With `attn_a`, `attn_b`, `attn_base` as explicit on-disk tensors (not buried inside render pkls), the Phase B ridge/PCA analysis becomes numpy on saved arrays — no parsing needed.
- **Scale sweeps become nearly free.** Re-rendering `smile @ {0.1, 0.2, 0.3, 0.5, 0.75, 1.0}` on the same anchor costs 6 base passes, not 6 full (2N+1) runs.
- **Variation-grid blog renders.** The dataset-scale portrait use case becomes practical.

## Non-goals

- Generalising across anchors (requires rebuilding library).
- Learning a transferable "direction" (separate thread: Phase B of the cache-geometry plan).
- Reducing the sampling step count (orthogonal to edit mechanism).

## When to build this

Not yet — current priority is finishing the editing procedure itself (slot-label metadata, cache-δ analysis, dictionary metadata). This architecture is the right move once (a) the multi-pair node is used in anger, (b) we have ≥5 axes in the dictionary, and (c) we actually want to batch-generate variation grids. Until then, single-render cost is acceptable and this is premature.

## Pointer

Core finding that makes this viable: the additivity test in `multi_smoke` (output/demographic_pc/multi_smoke/compare_ABC.png). A vs B at same total scale are visually identical → live multi-pair and cached-δ injection reduce to the same math.
