---
status: live
topic: liveportrait-stylized
---

# Flux → StyleGAN2 Style-Head Importer Bridge

**Date:** 2026-05-12
**Context:** Stylized-LP-renderer thread, follow-on to `2026-05-08-stylized-renderer-session-handoff.md`. Changes the keystone diagram by adding a manufacturing step for W-pair-compatible style heads.

## The problem this solves

The keystone pipeline from 2026-05-08 (`Spike 0.5`) is bottlenecked on the supply of FFHQ-StyleGAN2-transfer-trained style heads. The full public set is small, dated, and license-mixed: MetFaces, Ukiyoe, DualStyleGAN's 10 heads, More-Abstract-Art, NeuralKuvshinov (license-fraught). Past that handful, there is no live stylized-portrait research in the FFHQ-StyleGAN2 ecosystem — current frontier work has moved to Flux and SDXL, where the W-pair coupling does not exist.

This means product shape B ("pre-shipped style packs") had a structural ceiling at ~5 shippable styles. Anything beyond that required either (a) full re-fit of new StyleGAN2 heads from scratch on scraped corpora (expensive, license-murky, weeks per style), or (b) pivot to Track 2 (direct Flux + PuLID + ControlNet + LoRA paired generation), which trades identity coupling (cos ≈ 1.0 → 0.4–0.7) and ~50–100× per-pair compute for ecosystem reach.

## The bridge

A three-stage cascade in which the current Flux ecosystem is treated as the *style library*, JoJoGAN (or DualStyleGAN) as the *importer*, and StyleGAN2 W-pair sampling as the *clean training-signal channel for LP-G*:

```
Civitai Flux LoRA           (style catalog, unbounded, current frontier)
        │  Flux + PuLID + ControlNet, ~30 refs, varied identities
        ▼
20–50 FFHQ-aligned reference stills          (manual curation gate)
        │  JoJoGAN / DualStyleGAN fit, ~30 min on RTX 5090
        ▼
StyleHead_StyleGAN2.pkl                       (W-compatible with FFHQ-base)
        │  paired W sampling, hours, identity cos ≈ 1.0 (mild styles)
        ▼
~100K clean (photo, styled) pairs
        │  G-LoRA training, 1–3 days
        ▼
Stylized LP renderer for that style
```

Each new style becomes ~half a day of importer work upfront, then everything downstream is automated. The handoff's bottleneck shifts from "where do we get more style heads" (architectural) to "how many evenings of curation can we spare" (operational).

## Why this works architecturally

Three constraints align favorably:

**JoJoGAN does not need identity preservation in its references.** Unlike direct Track 2, where PuLID identity drift under heavy style LoRA contaminates the LP-G training pairs, JoJoGAN's training objective is to learn the *style transform*, not to memorize a face. Reference diversity across identities is a feature, not a bug. This relaxes the hardest constraint in Track 2.

**Style we need to import lives in fine layers; W-coupling lives in coarse layers.** LP-G's photoreal-prior is in SPADE γ/β projections — channel-wise affine, i.e. fine-layer texture statistics. That matches Pinkney's resolution-to-attribute map: fine = color/texture, coarse = pose/shape/identity. A *layer-restricted* JoJoGAN fit (freeze coarse, fine-tune fine only) imports exactly the part of style we want without touching the W-coupling on the identity-bearing coarse layers. Lucky alignment of two independent constraints.

**Identity coupling at the LP-G training stage is recovered.** Because W-pair sampling from the importer-produced head reuses the same W on FFHQ-base, the (photo, styled) pairs feeding LP-G LoRA have identity cos ≈ 1.0 (mild styles) or ≈ 0.9 (heavy styles after manifold drift). That preserves the keystone's clean training signal — the original reason Track 1 was the chosen path.

## Known failure modes

**JoJoGAN W-manifold drift on heavy styles.** Full JoJoGAN fine-tunes the synthesis network unrestricted, which pulls the W manifold itself. For mild stylization (oil painting, Ukiyoe-class, MetFaces-class), the new head's W manifold stays close to FFHQ-base — shared W → same identity. For heavy stylization with anatomy change (big-eye anime, caricature, chibi), the manifold drifts and identity coupling degrades from 1.0 toward 0.85–0.9. Two mitigations:

- *Layer-restricted JoJoGAN.* Only fine-tune fine layers (texture/color), freeze coarse. Pinkney swap variant. Preserves W-coupling on identity layers, but loses some "intensity" of stylization.
- *DualStyleGAN architecture.* Explicit two-branch design — separate intrinsic style branch with original FFHQ W preserved verbatim. Designed for the multi-style menu product shape. CVPR 2022 release has reference code + 10 pre-trained heads. **Probably the right production target if we end up shipping ≥3 style packs.** JoJoGAN is the cheaper first-spike importer.

**Reference set quality bounds output quality.** JoJoGAN with 30 misaligned, low-diversity, off-style references produces a mediocre style head. The curation gate is load-bearing.

**License inheritance is unclear.** A StyleGAN2 head fit from outputs of a Flux LoRA may inherit derivative restrictions from Flux.1-dev's non-commercial-derivatives license, the per-LoRA Civitai license, *and* the FFHQ base license (research-only). Per-style legal review before shipping. The bridge does not make this easier — it makes us responsible for both ecosystems' licenses simultaneously.

**Anthro/furry morphology breaks the entire bridge.** FFHQ alignment depends on human face landmarks. Anthropomorphic heads (muzzles, ears off skull, fur silhouettes) fail insightface 5-point detection and can't be FFHQ-warped. The bridge is human-style coverage only. Anthro/furry remains a separate, harder problem (likely a fork: LP-animals + furry fine-tune on that branch).

## Generation plan — one-week overnight schedule

Target: end-of-week deliverable of **5–7 curated reference sets**, each 30 FFHQ-aligned stills representing one candidate style pack, ready to feed into importer fits.

Estimated office-PC GPU budget: assume ~30–60s per Flux+PuLID+CN render at 1024². Each 30-still pack = ~25–60 min raw generation. Adding ~50% rejection rate after QA, generate ~60 raw per pack → ~50–120 min per pack per night. Fits comfortably in an overnight window with one pack per night.

### Setup (day 0, daytime, before tonight's run)

Lay down the harness so all nightly batches are deterministic and comparable:

- **Identity pool** — 30 FFHQ portraits (or 30 PuLID embeddings, equivalent). Balance 15 male / 15 female, varied age and ethnicity. Store as `data/importer/identities/id_{00..29}.jpg`.
- **Pose CN templates** — 5 openpose head-skeletons drawn at the FFHQ canonical positions: frontal, ±15° yaw, ±10° pitch. Store as `data/importer/pose_cn/pose_{0..4}.png`. Tight FFHQ-canonical placement (eyes at the canonical y, inter-pupillary distance fixed).
- **Prompt template** — `"<style_trigger>, portrait, neutral expression, plain background, looking at camera, FFHQ photo style"`. Negative: `"hands, body, full body, multiple people, watermark, text, low quality"`.
- **Style LoRA shortlist** — pick 5–7 candidates from Civitai. Suggested first batch (mild → heavy):
  1. *Oil painting portrait* (mild stylization, sanity-check pack)
  2. *Watercolor portrait* (mild)
  3. *Ghibli-style portrait* (medium)
  4. *Arcane / League cinematic* (medium)
  5. *Pixar-style 3D portrait* (medium-heavy)
  6. *Ilya Kuvshinov anime portrait* (heavy, parity with NeuralKuvshinov as research probe)
  7. *Vintage comic / pulp illustration* (medium, license-friendlier)
- **QA script** — `scripts/importer_qa.py`:
  - Insightface 5-point landmark detection
  - Similarity warp to FFHQ-canonical 1024×1024
  - Reject if landmark confidence < threshold, eye-line drift > N pixels, or alignment matrix singular
  - Output `data/importer/refs/<style_slug>/aligned/{kept,rejected}/`
- **Generation script** — `scripts/importer_generate.py`:
  - Loop: for each of 30 identities × 2 pose-variations = 60 raw renders per pack
  - PuLID weight ~0.7 (loose — identity diversity is the goal, not preservation)
  - ControlNet weight ~0.8 (firm pose lock so QA pass-rate stays high)
  - LoRA weight per-style (oil 0.7, Ghibli 0.9, Kuvshinov 1.0, tune per pack)
  - Resumable (skip-if-exists) per the project rule
- **Output structure**:
```
data/importer/
  identities/          # 30 ID source images
  pose_cn/             # 5 pose skeletons
  refs/<style_slug>/
    raw/               # 60 raw Flux outputs per pack
    aligned/kept/      # ≤60 FFHQ-aligned, passed QA (target ~30)
    aligned/rejected/  # for inspection
    manifest.json      # generation params, prompt, LoRA, weights, seed
```

### Daily schedule (7 nights)

Each night runs one pack end-to-end. Morning reviews the previous night before that evening's launch.

| Night | Style pack | Notes |
|---|---|---|
| 1 | Oil painting (mild) | **Sanity-check pack.** Easiest case. If this fails QA badly, the harness is broken and fix it before continuing. |
| 2 | Watercolor (mild) | Second mild; confirms harness stability across styles. |
| 3 | Ghibli (medium) | First style with real anime/anatomy bend. Watch FFHQ alignment pass-rate. |
| 4 | Pixar 3D (medium-heavy) | Cross-genre — tests whether the harness handles non-2D stylization. |
| 5 | Arcane / cinematic (medium) | Painted-realism mid-range. |
| 6 | Kuvshinov / anime (heavy) | Hard case. Expected to lose ≥30% to QA. Parity probe against NeuralKuvshinov-StyleGAN2 reference. |
| 7 | Buffer / re-runs | Whichever pack from nights 1–6 came in below 25 kept stills, re-run with adjusted CN/PuLID/LoRA weights. |

Morning review (~10 min each):
- Open the kept-aligned grid (16-thumbnail collage) — does it look like the LoRA's intended style?
- Check the rejected pile — are rejections principled (face occluded, landmarks lost) or harness-wrong (good faces being rejected)?
- Spot-check one alignment: open one raw + its aligned crop side by side, eyeball that the warp is sane.
- If pass-rate < 40%: adjust ControlNet weight up or pose template tighter, re-launch tomorrow night.
- Log outcome to `data/importer/log.md` (one line per night).

### Heavier-budget option (parallel during the week)

If office PC has spare daytime cycles, also generate the **direct-photoreal partner** of each reference: same PuLID embedding + same pose CN, no style LoRA. This gives a free side-deliverable — 30 photoreal-Flux portraits per pack — useful for:

- Sanity-checking PuLID identity quality outside any LoRA influence
- Building a Track-2 direct-paired-generation evaluation corpus for comparison against the importer-produced corpus, later
- Future eval anchors

Same harness, just an extra script invocation. ~zero extra design cost; pure cycles.

### End-of-week deliverable

By morning of day 8, the directory holds:

- 5–7 curated reference sets, each ≥25 FFHQ-aligned stills
- Manifest per pack with full generation params (reproducibility)
- Review-log noting which packs were strong / borderline / re-runs
- (Optional) parallel photoreal-Flux portrait sets

This is the input the importer (JoJoGAN spike + DualStyleGAN production fit) consumes. From there, the rest of the cascade is GPU-bound on the lab machine, not the office PC.

## What success looks like at end of week

Concrete go/no-go gates for each pack:

- **≥25 kept-aligned stills.** Below this JoJoGAN reference set is too small; redo or drop the pack.
- **Style-identity grid is visually consistent.** Across 30 different identities, the style transform should look uniform. If half the grid is "barely stylized" and half is "wildly off-style," LoRA strength was wrong; revisit weights.
- **Identity variety is preserved.** All 30 faces in the kept set should be visually distinct identities (we explicitly do not want JoJoGAN to overfit to one face). Spot-check via ArcFace pairwise cosines if uncertain — most pairs should be < 0.6.
- **FFHQ alignment passes QA at ≥40%.** Lower means harness needs adjustment (pose CN, prompt template, or face crop pipeline), not that the style is fundamentally bad.

Packs failing one gate are documented as "borderline" and held; we'll know whether they're salvageable after the first JoJoGAN fit on a *passing* pack tells us how forgiving the importer is to reference noise.

## Open questions, to resolve after first importer fit

These need empirical answers from the first JoJoGAN run, not pre-decision:

- Does layer-restricted JoJoGAN preserve enough style "intensity" for medium-heavy packs (Arcane, Ghibli, Kuvshinov), or does it underfit?
- Is DualStyleGAN's two-branch architecture worth the extra implementation cost on the first style, or only at pack ≥3?
- How sensitive is JoJoGAN to reference-set size between 20 and 50? Worth fewer-better refs vs more-noisier refs?
- Are the synthesized (photo, styled) pairs from the bridge-produced head clean enough at cos ≈ 0.9 to train LP-G, or do we still need an ArcFace filter pass on the paired output?

## Cross-references

- Parent thread handoff: `docs/research/2026-05-08-stylized-renderer-session-handoff.md`
- Stylization-locus mechanics: `docs/research/2026-05-08-stylization-locus-and-transfer.md`
- LP architecture decomposition (F/M/W/G): same doc
- FFHQ-family compatibility table: parent handoff, *Available pre-trained models* section
- LP-streaming recipe (target inference path for the eventual stylized renderer): memory `project_lp_streaming_recipe.md`

## Memory pointer to add

```
project_flux_stylegan2_importer.md — Flux→StyleGAN2 style-head importer bridge: Civitai LoRA →
Flux+PuLID+CN reference set (30 stills) → JoJoGAN/DualStyleGAN fit → W-pair-compatible style head →
clean paired corpus for LP-G LoRA. Unlocks unlimited style-pack supply for product shape B.
One-week office-PC generation plan in dated doc.
```
