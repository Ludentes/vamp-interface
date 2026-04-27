---
status: live
topic: demographic-pc-pipeline
---

# Cached-δ replay falsified as an editing mechanism

## TL;DR

Across four experiments with progressively more information preserved in
the cache, replaying a captured FluxSpace delta **never produces a
visible edit** — while live `FluxSpaceEditPair` on the same (base, seed,
prompt pair) produces a strong edit. The most decisive test is the
on-latent sanity: cached replay on the **exact capture seed** at scale
1.0 produces a neutral face; live produces a dramatic smile. This
falsifies both option B (direct cached-δ replay) and option C (learn
`g(attn_base) → delta`), and points to a mechanistic reason they cannot
work in principle.

## What we tried

All four experiments target the same smile axis (`EDIT_A="A person with
a faint closed-mouth smile."` vs `EDIT_B="A person with a manic
wide-open grin, teeth bared."`) with `mix_b=0.5`, `scale=1.0`,
`start_percent=0.15`.

1. **Channel-mean cached-δ injection** (superseded path — earlier thread).
   `delta_mix.mean_d` of shape `(D,)` broadcast over all L tokens at
   K=20 or K=912 sites. Noise collapse at scales 5–100; no semantic
   edit at any scale.
2. **Full-tensor `(L,D)` cached-δ, K=20 sites**, renorm OFF. Noise
   collapse at scale 5+; no smile at low scales.
3. **Full-tensor `(L,D)` cached-δ, K=20 sites**, **renorm ON**
   (paper's per-token renormalization trick). No noise collapse at any
   tested scale (0–300). **No smile at any scale.**
4. **On-latent sanity test** — replay at the exact capture seeds
   (2026 / 4242 / 1337) on elderly_latin_m at scale=1.0. Compared
   side-by-side with the live `FluxSpaceEditPair` PNG from the same
   (base, seed). Live produces dramatic tooth-baring smiles across all
   three seeds; cached replay produces neutral faces across all three
   seeds.

See `docs/research/images/2026-04-23-phase3-onlatent-sanity.png` for the
decisive 3×3 comparison and `2026-04-23-phase3-huge-sweep.png` for the
scale-300 no-op behavior.

## Capture pipeline (what Phase 1 actually produced)

- Modified `FluxSpaceEditPair` in `~/w/ComfyUI/custom_nodes/demographic_pc_fluxspace/__init__.py`
  to stream full `(L=1280, D=3072)` attn_base + delta_mix tensors per
  (step, block) at fp16 to memmapped `.npy` files, gated by a
  `full_capture_sites_json` allow-list.
- Top-K=20 sites picked by channel-mean fro from
  `smile_inphase/delta_mix.npy` — all concentrated on
  `single_{31,33,34}` across steps 5–15 (peak per earlier probe A).
- 3 seeds (2026, 4242, 1337) × elderly_latin_m at mix_b=0.5, scale=1.0 →
  `output/demographic_pc/phase1_full_captures/` (~910 MB internal,
  sparse ext4).
- Seed-averaged into `models/blendshape_nmf/phase1_smile_delta_full.npz`
  (260 MB, shape `(11, 3, 1280, 3072)` fp16) via
  `phase1_build_library.py`.
- Replayed via a new node `FluxSpaceDirectionInjectFull` which adds the
  cached full `(L,D)` slice at matching (step, block) sites, with an
  optional per-token renormalization toggle.

The cache contents are correct: per-site fro ~2200–2600, unwritten
slots zero, step/block allow-list matches the JSON exactly, shapes and
dtypes consistent across seeds. The replay machinery works. The edit
just does not appear.

## The mechanism — why cached-δ injection cannot work

Live `FluxSpaceEditPair` at scale=1 does:

```python
steered = attn + 1.0 * (edit_mean_live - attn) = edit_mean_live
```

i.e. **it replaces** the current block's attn output with the edit
prompt's attn output, where `edit_mean_live` is computed against the
*current latent* in a live forward pass. This replacement happens at
**every** one of the 912 sites (16 steps × 57 blocks). Because Flux is
residual, block N+1's inputs include block N's (now edited) output.
**The edit compounds through the forward pass** — each replaced attn
contaminates the residual stream feeding downstream blocks, which then
themselves get replaced with their edit-prompt output at that
already-contaminated latent. By the VAE decoder the entire residual
stream is co-opted toward the edit semantics.

Our cached replay does:

```python
steered = attn_live + scale * δ_cached   # (+ optional per-token renorm)
```

where `δ_cached` is a static tensor captured from a previous render. At
non-cached sites (892 of 912 in the K=20 experiment, or zero of 912 in
a hypothetical full re-capture) the residual stream runs the *unedited*
path. Those blocks' outputs dominate and erase the cached
perturbation before it reaches the decoder.

Even with full 912-site coverage, cached replay cannot match live
because `edit_mean_live` depends on *what block N−1 produced under
editing*. That is a function of the latent trajectory, not a tensor.
Live editing is a **stateful cascade** through the network — each
block's edit changes the next block's input, which gets re-edited. A
static cache captures a single point along one such trajectory and
cannot reproduce the trajectory on a different latent (or even on the
same seed, because the unedited intermediate activations diverge
immediately at block 0).

The on-latent sanity test (experiment 4) is the cleanest falsification
of this. Same seed, same base, same capture delta — the network's
block-0 output diverges between "add δ_cached" and "replace with
edit_mean_live", and the two trajectories never reconverge.

## Implications

- **Editing primitive = live `FluxSpaceEditPair` / `FluxSpaceEditPairMulti`**.
  No cached alternative exists. Cost = `(2N+1) × forward passes per
  step` for N-axis composition. This is the ceiling and we have to live
  with it.
- **Axis library = prompt-pair text dictionary**, not a tensor
  dictionary. Each axis is `(EDIT_A text, EDIT_B text)` and is
  characterized by its downstream measurement signatures (blendshape
  responses, classifier margins, ArcFace drift), not by a captured δ.
- **Existing attention caches** (`models/blendshape_nmf/attn_cache/*`,
  ~315 GB of v1 pkls on external drive) remain valid as
  *measurement/interpretability* assets — ridge-R² predictions of
  blendshape responses, classifier probes, cluster analysis. They do
  **not** become a replay library; do not attempt to synthesize one
  from them.
- **Atom libraries** (`directions_resid_causal.npz`,
  `directions_k11.npz`, `au_inject_*.npz`,
  `phase1_smile_delta_full.npz`) — all rendered moot as edit
  mechanisms. Their measurement-side usefulness survives where it was
  already demonstrated independently.
- **Phase 3 of the AU-library thread** (composition via additive cached
  atoms) is killed. Composition must go through
  `FluxSpaceEditPairMulti` with live prompt pairs.
- **Stage 5 direction**: reverts to the pre-cache plan — build a
  prompt-pair dictionary, run multi-pair composition, measure, iterate.

## Code / artifact inventory (kept vs dead)

Kept (measurement + evidence):
- `docs/research/images/2026-04-23-phase3-*.png` — 5 collages including
  the decisive on-latent sanity.
- `output/demographic_pc/phase3_*/` — ~16 MB of test-render PNGs for
  the four experiments. Cheap, keep for reproducibility.
- `models/blendshape_nmf/phase1_smile_delta_full.npz` — 260 MB
  seed-averaged library. Useful as an example of what a "clean"
  cached-δ looks like (for any future paper claiming this should
  work — it doesn't).

Dead-thread code (kept in-tree for reproducibility; do not reanimate):
- `src/demographic_pc/phase1_full_capture.py`
- `src/demographic_pc/phase1_pick_topk_sites.py`
- `src/demographic_pc/phase1_build_library.py`
- `src/demographic_pc/phase3_*.py`
- `FluxSpaceDirectionInjectFull` in the ComfyUI node file, +
  `full_measure_path` / `full_capture_sites_json` inputs on
  `FluxSpaceEditPair`. All harmless — the editing-pipeline flow does
  not exercise them.

Raw Phase-1 captures:
- `output/demographic_pc/phase1_full_captures/` — 910 MB logical,
  sparse ~80 MB actual, contains 3 seed-specific memmapped .npy files
  + sidecars. Decision pending (nuke vs cold-archive).

## Related prior threads

- `docs/research/2026-04-23-atom-inject-visual-failure.md` — earlier
  scoped failure (`directions_resid_causal.npz` atom_16) that should
  now be read as an instance of the same mechanism falsified here.
- `docs/research/2026-04-23-au-library-hybrid-plan.md` — the NMF+ridge
  library plan. Measurement side survives; injection side falsified.
- `docs/research/2026-04-23-framework-procedure.md` — compaction-proof
  framework doc. The "dictionary of effect atoms" picture remains,
  but atoms are **measurement readouts**, never edit injectors.
  Prompt-pair primitives are the edit mechanism.
