---
status: live
topic: liveportrait-stylized
---

# Stylized-Renderer Research Session Handoff

**Date:** 2026-05-08
**Purpose:** End-of-session summary. Read first on resume.

---

## What changed in our worldview this session

The previous session's adversarial-interview findings said *not photoreal of self* — and the user clarified empirically that **heavily-stylized anchors fail for both LP and PersonaLive**. Paintings work somewhat on PL but PL is too slow for consumer GPUs. The viable product gap is **a renderer that's LP-fast but tolerates stylized anchors**. This session was the research thread to figure out *how* to build that.

The thread produced four research documents, in order:

1. `docs/research/2026-05-08-stylized-liveportrait-renderer.md` — initial scoping of LP architecture, training data, animals-mode as existence proof, three-path proposal (A: per-anchor LoRA, B: per-anchor 10-min fit, C: full corpus retrain).
2. `docs/research/2026-05-08-face-image-vocabulary-latent.md` — foundations: what an image/face/mascot is, latent topology of face generators, whether FLAME/ARKit are natural axes of the data (verdict: no, partial-credit alignment only).
3. `docs/research/2026-05-08-stylization-locus-and-transfer.md` — mechanics: localizing the photoreal-prior to G's SPADE γ/β projections, transfer-learning recipes, layered surgical targets. **Supersedes doc 1.**
4. `_parts/*` — 7 sub-research-agent outputs feeding the above (latent-topology, flame-3dmm, arkit-facs, overlap-question, style-locus, face-domain-adaptation, peft-image-gen). ~75 sources total.

## Load-bearing findings (the durable knowledge)

**LP architecture decomposition** — F (appearance encoder) → M (motion extractor, consumes driver only) → W (warping, flow-field producer) → G (SPADE decoder). Stage-1 trains all four; stage-2 freezes them and trains stitch/retarget MLPs. LP-animals (~230K cat/dog frames) is a stage-1-style fine-tune of base modules; stitch/retarget were skipped per LP team's own statement.

**Stylization is channel-statistics, not geometry** — Gatys 2016: style = Gram matrices of feature activations (channel co-occurrence stats); content = raw spatial activations at deep layers. Stationarity is the load-bearing property — paintings perturb early-layer Gram stats massively while deep semantic features survive.

**StyleGAN resolution-to-attribute map** — coarse 4-8px = pose/shape; medium 16-32px = features/expression; fine 64-1024px = color/texture/micro-detail. The photoreal-skin prior lives in fine layers. Pinkney's Toonify swap (low-res from toon + high-res from photo = toonify; inverse = "deeply weird") is the cleanest validation.

**Where photoreal-prior lives in LP** — three converging arguments point at **G's SPADE γ/β projection layers**:
1. Architecturally — SPADE = spatially-adaptive AdaIN = channel-affine style injection.
2. Training corpus skew — LP-G saw ~69M photoreal frames vs. ~60K styled stills (~1000:1).
3. CustomDiffusion analogy — K/V projections are 3% of params and carry full DreamBooth-style personalization; SPADE γ/β are the architectural analog in encoder-decoder GANs.

F is partially robust (deep features survive style); W is a *structural* failure mode, not stylization (mosaic-tile tearing on non-anatomical regions); G is the texture-and-final-paint stage where the photoreal prior is dominant.

**Anatomical failures (long beards, elf ears, horns) are W's failure, not G's** — separate from the stylization-as-channel-stats problem. Paths: X1 anchor-segment + static composite, X2 rigid 2D extension transform (recommended for v1), X3-X5 progressively more expensive.

**Pinkney layer-swap is intra-architecture-family only** — works for {FFHQ, MetFaces, Ukiyoe, NeuralKuvshinov, More-Abstract-Art, DualStyleGAN heads} since all are FFHQ-StyleGAN2-transfer-trained. Does NOT cross StyleGAN ↔ LP. LP-human ↔ LP-animals is the only LP-family swap pair publicly available.

**The vocabulary question** — FLAME and ARKit/FACS are *engineered conventions*, not the natural axes of the data. Linear AU probes work in StyleGAN W but unsupervised methods do not reinvent FACS. **Direct parameter conditioning fails** (GIF 2020: "unsatisfactory"); rendered-intermediate conditioning works. Operational recommendation: drive the renderer with implicit keypoints (in-distribution); use FLAME/ARKit as measurement/anchoring/tooling only. **Our shipped architecture (ARKit consumed only by bridge, never by renderer) is architecturally correct — not an accident.**

## Available pre-trained models, layer-compatibility-grouped

**LP family** (architecturally identical, weight-swappable):
- LP base (human portrait), Kuaishou
- LP-animals (cat/dog fine-tune, 230K frames), Kuaishou
- **No stylized LP exists publicly — we are alone on that thread**

**FFHQ-StyleGAN2 family** (FFHQ-transfer-trained, Pinkney-swap-compatible among each other):
- FFHQ-StyleGAN2 (base photoreal)
- MetFaces — 1336 Met portrait paintings, 1024, official NVIDIA, **license-clean**
- Ukiyoe Faces — 5000 Japanese woodblock-print faces, 256, Pinkney, **license-clean** (out of copyright)
- **NeuralKuvshinov v2.1/2.2/2.3 + v1.27** — Ilya Kuvshinov-style anime/illustration, dobrosketchkun, **license-fraught** (research only, don't ship)
- More Abstract Art — 14K abstract paintings, 512, Saraev
- DualStyleGAN's 10 heads (Cartoon, Caricature, Anime, Arcane, Comic, Pixar, Slamdunk, Fantasy, Illustration, Impasto) — Yang et al. CVPR 2022, **mixed license**

**Not layer-compatible with the above** (different architectures, separate ecosystems):
- StyleGAN3 family (WikiArt-1024 Pinkney/Lambda etc.)
- SDXL LoRA painter packs (KappaNeuro Vrubel/Bilibin, Civitai Vrubel) — useful as text-prompt-conditioned synthetic data generators, not as swap partners

**Painter the user was remembering**: Ilya Kuvshinov, via NeuralKuvshinov_v2 by `dobrosketchkun` on GitHub. Russian-born, Japan-based contemporary illustrator. StyleGAN2-ADA-PyTorch, FFHQ-transfer-trained. Multiple checkpoint variants + male/megane/sketch specialty heads.

## The keystone pipeline opened up by FFHQ-family compatibility

Paired-synthetic-corpus generator, free, at arbitrary scale, identity-preserved:

```
For each random W in 100K-1M samples:
    photo  = FFHQ_StyleGAN2(W)        # photoreal of identity-W
    styled = StyleHead_StyleGAN2(W)   # MetFaces / Ukiyoe / DualStyleGAN / Kuvshinov
    pair = (photo, styled)            # same identity, two styles
→ Train LP-G LoRA on paired-image loss
```

This is the unifying mechanism. Each style head (MetFaces, Ukiyoe, DualStyleGAN-Cartoon, etc.) gives us a synthetic paired training set in its style, **with identity preserved by W-anchor**. The Kuvshinov head is the research probe (validation only, don't ship); MetFaces and Ukiyoe are the first shippable style packs.

## Anchored next-step spike ladder

| # | Spike | Cost | Tests |
|---|---|---|---|
| **0a** | Model-soup α between LP-human and LP-animals G | hours | Does any zero-training mixing produce stylization? |
| **0b** | Inverse Pinkney in LP-G (coarse human + fine animals) | hours | Animals fine-layer texture on Vrubel? |
| **0c** | Forward Pinkney (coarse animals + fine human) | hours | Animals coarse-layer for anatomical extensions (horns, beards)? |
| **0.5** | **Paired-synthetic + LoRA-G via NeuralKuvshinov + FFHQ** | 1-3 days | **Keystone: does the paired-pair pipeline produce a usable stylized LP?** |
| **1** | Single-anchor JoJoGAN-style 30-min LoRA fit | 3-5 days | Per-anchor path A |
| **2** | MetFaces-paired LoRA (first shippable) | 3-5 days | First license-clean style head |

**Spike 0.5 is the keystone.** If paired-synthetic produces a usable LoRA, the product mechanic shifts from "per-anchor fit at upload (30-min wait)" to "ship N style packs, instant load, free per-anchor inference." Better UX, cleaner license story.

## The open product-shape question

Three competing product mechanics, each implied by a different research thread:

- **A) Per-anchor LoRA fit at upload** (JoJoGAN-style, 30-min one-time per painting). Best fidelity per anchor; commits to a wait time.
- **B) Pre-shipped style packs** (DualStyleGAN-style menu of "Anime / Painted Portrait / Ukiyoe / ..."). Instant load; less anchor-specific; clean license story.
- **C) Hybrid** — auto-classify uploaded anchor to nearest pre-shipped pack, optional 5-min anchor-specific fine-tune.

**Not yet decided.** Awaiting user direction. The Spike 0.5 outcome determines whether B is even viable; A and C are the fallbacks if 0.5 fails.

## Pending tasks for next session

1. **Pick A/B/C product shape.** Gates everything else. Requires user input.
2. **Run Spike 0a (model soup)** — cheapest possible test, hours, no risk.
3. **Run Spike 0.5 (paired-synthetic + LoRA-G with NeuralKuvshinov)** — keystone. If this works, product is B-shaped; if it doesn't, product is A-shaped.
4. **Land round-3 update to `rorschach/docs/shaping/shaping.md`** — still pending from before this research thread. R10/R11/R12/R13/R15 from interviews + new R-stylized-default from this research.
5. **Verify the OpenReview PDF the user shared** — couldn't be fetched, needs title/forum URL from user.
6. **Translate to product story**: if Spike 0.5 succeeds, the v1 marketing copy becomes "Pick a style. Upload your painting. Stream." Much stronger than the per-anchor-fit story.

## Failed / dead-end findings (don't re-explore)

- **Direct FLAME/ARKit conditioning of a learned face generator** — GIF 2020 falsified this. Generators want pixel-space hints, not abstract parameter vectors. Our shipped architecture (ARKit → bridge → implicit keypoints → renderer) is already on the right side.
- **Diffusion-based stylized renderers as speed competitors** — X-Portrait and AniPortrait can do stylized but are inference-cost-disqualified. They remain useful as quality references and as synthetic-data generators.
- **No public stylized LP variant exists** — confirmed via search. LP-animals is the only LP-family fine-tune publicly available.
- **Pure CLIP-text supervision without image data (NADA pattern) on LP** — unexplored in literature; flagged as plausible Path D in doc 3 but no published evidence it works outside StyleGAN.

## Memory updates needed

Add a new memory file pointing at this handoff and the three research docs:

```
project_stylized_renderer_thread.md — pointer to 2026-05-08-{stylized-liveportrait-renderer, face-image-vocabulary-latent, stylization-locus-and-transfer}.md + this handoff. Key claims: G's SPADE γ/β = photoreal-prior locus; FFHQ-family pair-synthetic via NeuralKuvshinov is keystone pipeline; product shape A/B/C undecided.
```

Update the existing `🚢 LP RECIPE (2026-05-07)` memory with cross-reference to this new thread.

## Source bibliographies

Full per-section citations in `docs/research/_parts/`:
- `latent-topology.md` — 12 sources
- `flame-3dmm.md` — 13 sources
- `arkit-facs.md` — 11 sources
- `overlap-question.md` — 20 sources
- `style-locus.md` — 15 sources
- `face-domain-adaptation.md` — 17 sources
- `peft-image-gen.md` — 16 sources

Plus the four main synthesis docs above. Total: ~75 primary sources cited across the session.
