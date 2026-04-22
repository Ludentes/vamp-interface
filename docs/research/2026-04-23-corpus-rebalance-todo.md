---
status: live
topic: manifold-geometry
summary: Next-session plan — current NMF corpus is smile-dominated (85%) and demographically skewed per prompt; rebalance with diverse expression prompts on every base before refitting Phase 2+3.
---

# Corpus rebalance — plan for 2026-04-23

**Context:** Today's NMF decomposition gave 11 channel-coherent atoms
with CV R² 0.82–0.97, but the top-activating-samples collage revealed
the atoms are corpus-correlation-pure, not AU-pure. Eight of eleven
atoms co-activate with smile; six have a single dominant demographic
at the top percentile. Phase 4 is premature on this basis.

See `2026-04-22-phase3-proper-directions.md` + the draft blog
`docs/blog/2026-04-22-nmf-atoms.md` §"Then we looked at the collage
carefully" for the diagnosis.

## Target corpus shape

Current (1941 samples): 85% smile-centric.

Target: ≤ 40% smile/jaw, rest distributed across five other
expression axes. Rough allocation:

- 300 samples — angry / brow-down / lid-tight cluster (AU4 + AU7 +
  AU5)
- 300 samples — surprise / brow-up / wide-eye cluster (AU1 + AU2 +
  AU5 + AU26 mild)
- 300 samples — disgust / sneer / lower-lip-depress (AU9 + AU15 +
  AU16)
- 300 samples — pucker proper (AU18) — current `mouthPucker` atom
  only has 956 top-site R²; needs more training signal
- 200 samples — lip press / lip suck proper (AU24 + AU28) — the
  fragile atom 8 from stability check
- Keep current 1941 smile/jaw samples as-is

Total new: ~1400 renders. At ~10s/render on one GPU that's ~4 hours
compute.

## Rendering requirements — key design rule

**Every new prompt pair must be rendered on every base.** This is
the fix for demographic entanglement. Use the six-base list from
`alpha_interp_attn` as the canonical set:

- asian_m
- black_f
- elderly_latin_m
- european_m
- young_european_f
- southasian_f

Per prompt pair: 6 bases × 2–3 seeds × 4–6 scales = 50–100 renders.

## Prompt design — avoid nuisance entanglement

Keep each prompt pair as close to the base prompt as possible,
differing only in expression words. Do NOT vary:

- Glasses / no-glasses (keeps pucker atom clean of eyewear)
- Age / hair style / attire (keeps expression atoms from loading
  onto identity)
- Scene / lighting / background

Expression word-list (A = neutral, B = expressive):

| Axis | A | B |
|------|---|---|
| anger | "with a neutral expression" | "with angry brows lowered and jaw tensed" |
| surprise | "with a neutral expression" | "with raised brows and mouth agape" |
| disgust | "with a neutral expression" | "with a sneer and wrinkled nose" |
| pucker | "with a neutral expression" | "with lips puckered forward" |
| lip press | "with a neutral expression" | "with lips pressed firmly together" |

## Pipeline for tomorrow

1. Write `src/demographic_pc/render_expression_corpus.py` — takes
   an expression axis name + base list, renders FluxSpace pair
   sweeps. Reuse `fluxspace_intensity_sweep.py` structure.
2. Launch renders in background (~4 hours).
3. Score new PNGs with `score_blendshapes.py`.
4. Refit `blendshape_decomposition.py` on combined corpus (new
   1400 + existing 1941 = 3341 samples). Expect effective rank to
   climb from 11 toward 15–20; sweep k ∈ {10, 12, 15, 18, 22}.
5. Re-run `nmf_stability.py` with new k.
6. Rebuild the attention-cache (`cache_attn_features.py`) on the
   new paired renders only (~1400 new pkls — 14 GB delta).
7. Refit `fit_nmf_directions.py` at new k, new corpus.
8. Regenerate `nmf_atom_collage.py` at 3 cols on new basis —
   verify smile co-activation drops and demographic variety within
   rows increases.
9. If collage passes honest inspection: proceed to Phase 4
   validation pilot on one atom per expression axis.

## Honest expectations

Rebalancing should:
- **Reduce CV R²** (current numbers are inflated by corpus homogeneity)
- **Reduce cross-atom leakage** at inference
- **Raise the effective rank** above 11
- Make collage rows demographically diverse by construction

Might still reveal residual issues:
- If NMF still smile-dominates after balancing → Flux's output
  distribution is itself smile-dominated regardless of prompt
  (Flux bias).
- If atoms still entangle with demographics → some ARKit channels
  have demographic-correlated baselines that NMF can't separate
  without explicit identity residualisation.

## Artefacts expected

- `output/demographic_pc/fluxspace_metrics/crossdemo/{expression}/...`
- `models/blendshape_nmf/W_nmf_k{new_k}.npy` (bigger, more balanced basis)
- `models/blendshape_nmf/directions_k{new_k}.npz` (refit directions)
- `docs/research/2026-04-23-corpus-rebalance-result.md` — report with
  post-rebalance collage, per-axis linearity, comparison to current
  skewed-corpus numbers.
- Blog post `docs/blog/2026-04-22-nmf-atoms.md` can be promoted from
  draft to live once the refit replaces the skewed figure.
