# Demographic-PC Extraction Plan (Flux Krea v3)

**Date:** 2026-04-20
**Status:** Draft for review
**Goal:** Identify the directions in Flux Krea conditioning space that correlate with classifier-predicted demographic labels (age, gender, race), so we can orthogonalize against them when sampling Δ for the perception curriculum (`2026-04-20-ffa-progression-curriculum.md`).
**Grounding research:** `2026-04-20-demographic-classifiers.md`.

## What We're Actually Computing

Not "PCA of labels." The defensible pipeline is:

1. Sample N faces by drawing conditioning vectors `c_i ∈ R^d` from a diverse neutral-portrait prompt distribution.
2. Render via Flux Krea → image `x_i`.
3. Run three classifiers → label vector `y_i` (age, gender, race one-hots/continuous; ~18–22 dims stacked).
4. Fit a regression `y = f(c) + ε` (ridge or multinomial logistic per head).
5. For each label head, the top-k **predictive directions** in `c`-space are the ones to orthogonalize against.
6. Stack direction matrices across all heads → truncated SVD → `D = { d_1, ..., d_K }` is the demographic subspace we project out.

Naive PCA of label vectors gives directions *in label space*, not in conditioning space — unusable for projection. The regression-direction approach is the one that supports `c' = c − Σ (c·d_k) d_k`.

## Defining "Conditioning Space"

Flux Krea uses dual CLIP: CLIP-L pooled (768-d) + T5XXL per-token (~256 × 4096-d). We need to decide what `c` is.

**Decision:** use **concat[CLIP-L pooled (768-d), mean-pooled T5 (4096-d)] = 4864-d**.

**Why:** matches the standard "conditioning embedding" in the Flux editing literature; avoids full per-token storage cost (~1M floats/sample); mean-pool loses token-position information but demographic signal is prompt-global, not token-local.

**Capture mechanism:** bypass ComfyUI for the encoder pass — load CLIP-L + T5XXL once in a Python script, encode each prompt, save `(c_i, seed_i, prompt_i)`, then feed the same prompt+seed through the ComfyUI Flux workflow to get the image. Conditioning capture and image generation run in two passes over the same `(prompt, seed)` list.

*Alternative if encoder-load-in-python is painful:* add a `SaveTensor` node to the ComfyUI workflow and dump conditioning from there. Slightly more dev work but avoids duplicate encoder load. **Pick whichever is easier after a 30-min spike.**

## Sample Distribution

**N = 1800** (research doc said 1500–2000 for ~20-dim label regression in ~4864-d space; we'll be L2-regularized so the lower end is OK but we'll hit the upper).

**Prompt template:** neutral-portrait anchor varied along coarse demographic attributes so the classifier sees diverse faces:

```
A photorealistic portrait photograph of a [AGE] [GENDER] [ETHNICITY],
neutral expression, plain grey background, studio lighting, sharp focus.
```

Sampled factorially from:
- AGE: {child, young adult, adult, middle-aged, elderly} — 5
- GENDER: {man, woman, non-binary person} — 3
- ETHNICITY: {East Asian, Southeast Asian, South Asian, Black, White, Hispanic/Latino, Middle Eastern} — 7

Full grid: 105 combinations × ~17 seeds ≈ 1785 samples. Round to 1800.

**Why this distribution:** we want the classifiers to fire across their full label range, so the regression has signal in every label dim. If we only prompt "neutral portrait," we'd get a narrow age/gender band and race directions would be unidentifiable.

**Caveat:** the prompt distribution **is** the demographic distribution here. The directions we extract will be aligned with whatever Flux does when you ask for "elderly Black woman" vs "young White man." We're measuring Flux's *response* to demographic prompt variation, which is exactly what we want to subtract from curriculum sampling. But it means the extracted directions are *specific to this prompt family* — they're not a universal "age axis in Flux space."

## Pipeline Stages

### Stage 0: Environment setup
- `uv add fairface mivolo` → likely not on PyPI; source installs into `.venv`
- FairFace: clone `dchen236/FairFace`, install dlib (needs C++ toolchain), download Res34 weights
- DEX: pick `siriusdemon/pytorch-DEX` or similar; verify apparent-age weights (not real-age)
- MiVOLO: clone, Apache-2.0, relax pinned deps, download face-only `volo_d1` checkpoint, skip body branch
- **Smoke test each:** one face image through each classifier, print raw outputs. Commit to a Jupyter notebook `notebooks/demographic_classifiers_smoke.ipynb`.

### Stage 1: Synthetic-transfer sanity check (50 samples)
**Load-bearing:** all three classifiers trained on real photos; Flux-portrait transfer is unvalidated. Before trusting 1800-sample regression, we need evidence classifiers produce sensible outputs on Flux output.

- Generate 50 Flux portraits from the full grid
- For each, record: prompt-attributes (age/gender/ethnicity), three classifier predictions
- Report agreement with prompt-attribute by classifier, inter-classifier agreement (age: DEX vs MiVOLO vs FairFace-bin)
- **Gate:** if any classifier disagrees with prompt-attribute on >30% of samples *and* also disagrees with the other two classifiers, pause and investigate before committing to 1800-run.

Save to `output/demographic_pc/sanity_check_50.json` + a short writeup.

### Stage 2: Full generation run (1800 samples)
- Generate images through existing Flux Krea pipeline (v3 workflow in ComfyUI)
- **Resolution: 768×1024 portrait** (chosen 2026-04-20: ~44% of 1024² compute; Flux Krea still in-distribution; separates "classifier fails on Flux" from "classifier fails on low-res" in Stage 1)
- Anchor for img2img: generate a fresh 768×1024 neutral portrait anchor once, reuse across all 1800 samples
- Capture conditioning `c_i` per sample (chosen mechanism from "Defining Conditioning Space")
- Save: `output/demographic_pc/samples/{id}.png`, `output/demographic_pc/conditioning.npy` (shape 1800 × 4864), `output/demographic_pc/prompts.json`
- Expected time: ~7s/face × 1800 ≈ 3.5h on the existing Flux setup.

### Stage 3: Classifier inference
- Three classifiers × 1800 images
- Save: `output/demographic_pc/labels.parquet` with columns: `id, fairface_race (onehot×7), fairface_gender (onehot×2), fairface_age_bin (onehot×9), dex_age (continuous), mivolo_age (continuous), mivolo_gender (onehot×2)`
- ~22 label dims total
- Report: per-classifier coverage (any NaNs from face-detection failures?), inter-classifier agreement matrix

### Stage 4: Regression & direction extraction
- For continuous heads (DEX age, MiVOLO age): ridge regression `c → y`, extract top-5 right singular vectors of learned weight matrix
- For categorical heads (FairFace race/gender/age-bin, MiVOLO gender): multinomial logistic, extract directions via SVD of weight matrix per head
- Stack all direction vectors → `W ∈ R^{K × 4864}` where `K ≈ 30` (5 dims × 6 heads)
- Truncated SVD on `W` → keep components explaining >90% of variance → **demographic subspace `D` of dimension ~10–15**
- Report: variance-explained curve, cosine similarity between directions from different classifiers (do they rediscover the same axes?), per-head CV R² for continuous and accuracy for categorical

### Stage 5: Validation
- **Orthogonalization sanity:** take a held-out 200 samples, project their `c` onto `D`-complement, re-render through Flux, run classifiers → do demographic predictions collapse toward the dataset mean? (If not, `D` isn't capturing what we think it is.)
- **Curriculum readiness:** sample two random conditioning vectors `c_a, c_b` from `D`-complement with `‖c_a − c_b‖` = curriculum's σ unit → render → check inter-classifier demographic agreement says they look "same demographic." Go/no-go for curriculum Level 0.

## Outputs

- `output/demographic_pc/direction_matrix.npy` — `K × 4864`, the demographic subspace basis
- `output/demographic_pc/variance_explained.png` — scree-style plot
- `output/demographic_pc/classifier_agreement.md` — inter-classifier diagnostic writeup
- `output/demographic_pc/transfer_sanity_check.md` — Stage 1 report
- `src/demographic_pc/` — reusable code (not scripts, importable module)

## Open Questions / Assumptions to Resolve Before Starting

1. **Conditioning representation:** confirm mean-pooled T5 + CLIP-L pooled is the right `c`. Alternative is CLIP-L pooled only (768-d) — much cheaper, may be sufficient if T5 contribution to demographic signal is diffuse. **Spike:** fit regression on both, compare R². *Could add a day.*
2. **Non-binary prompts:** Flux's response to "non-binary person" is probably noisy. Keep in the grid for label-distribution coverage or drop? Default: keep, flag in writeup.
3. **FairFace race coverage:** 7 buckets is what the model outputs. FairFace's "Middle Eastern" ≈ "White" confusion is a known failure mode; our prompt grid includes both as separate prompts, so we'll see it in the data.
4. **Do we need ArcFace identity-preservation as a separate axis to subtract?** Probably not — identity is the *dependent variable* we want to preserve, not an axis to project out. But worth a sentence in the writeup.

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Classifiers don't transfer to Flux output | Stage 1 sanity check; gate the full run on it |
| Conditioning capture is painful | 30-min spike on both mechanisms before committing |
| 1800 samples still underpowered | L2-regularize; report CV R² per head; if any head has R² < 0.3, drop it from direction stacking |
| Directions are prompt-distribution-specific | Explicit caveat in output writeup; validate on Stage 5 held-out |
| FairFace race is single-sourced | Report race-subspace confidence separately from age/gender; treat as weakest component |

## Timeline Estimate

- Stage 0 (env setup + smoke): 1 day
- Stage 1 (50-sample sanity check): 0.5 day
- Stage 2 (1800-sample generation): ~3.5h unattended (768×1024)
- Stage 3 (classifier inference): 0.5 day
- Stage 4 (regression + direction extraction): 1 day
- Stage 5 (validation): 0.5 day

**Total active: ~3.5 days, plus one overnight.**

## Not Doing (Out Of Scope)

- Training our own demographic classifier on Flux output (future work if Stage 1 shows transfer failure)
- Fine-grained demographic axes (e.g., facial hair, hairstyle) — these can come later as separate confound controls
- Projecting out *identity* — that's a different axis and is preserved intentionally
- Adversarial robustness of the directions to prompt paraphrasing — noted as future work

---

**Next action after plan approval:** start Stage 0 environment setup. Write a short install-log doc as we go so the second attempt is cheaper.
