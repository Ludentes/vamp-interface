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

### Stage 4.5: Cross-method direction comparison (added 2026-04-20)
**Purpose:** before trusting our regression directions for the curriculum, measure them head-to-head against the two published alternatives on the **same held-out Flux portraits**. If someone else's method produces cleaner demographic directions on our backbone, we want to know before Stage 5, not after.

**Methods compared — three genuinely distinct paradigms:**

1. **Ours — regression → pre-computed direction → subtract at conditioning level** (Stage 4 output). Directions obtained by ridge/multinomial-logistic regression from pooled 4864-d `c` to classifier-predicted labels, SVD'd into a basis. Applied as `c' = c − Σ (c·d_k) d_k` before sampling.
2. **FluxSpace — prompt-pair contrast → pre-computed direction → orthogonal projection at MM-DiT attention-output** (Dalva et al. CVPR 2025). For each attribute, construct a prompt pair (`c_e` = "elderly portrait…", `φ` = "young adult portrait…") and take `d = ℓ_θ(c_e) − proj_φ ℓ_θ(c_e)` at a mid-depth MM-DiT block. Training-free; uses the FluxSpace public toolkit.
3. **FlowChef — no pre-computed direction → live classifier-guided steering during sampling** (Patel et al. ICCV 2025). The conceptually different method: rectified flow's straight-trajectory property lets you steer the velocity field during sampling using a classifier's gradient, without backprop through the ODE solver ("gradient skipping"). For our purpose, use **negative guidance** against all three demographic classifiers simultaneously — MiVOLO, FairFace, InsightFace — and ask FlowChef to render faces the classifiers can't read. Training-free on Flux-dev; we already own the classifiers. Flux-dev initial configs target 256×256 with AFHQ/CelebA annotations; we'd need to extend to our 768×1024 setting.
4. **~~Concept Sliders / SliderSpace~~ (deferred).** Gandikota et al.; ECCV 2024 / CVPR 2025. No pretrained Flux sliders released (per `2026-04-14-flux-edit-code-inventory.md`); running this method would require training our own LoRA per attribute at ~1–2 days and A6000-class compute per axis. Re-evaluate only if Stages 4/4.5/5 show the three primary methods all fail to give the curriculum what it needs.

**Why all three matter:** the curriculum's Level 0 *orthogonalization* requirement can be met either by pre-computing a direction and subtracting (paradigm 1 and 2), *or* by never extracting a direction and instead steering at sample time (paradigm 3). If FlowChef achieves demographic-collapse via live classifier guidance without a direction matrix, that reshapes Level 0 — we'd cache FlowChef-steered samples rather than compute orthogonalized `c`. Worth knowing before committing.

**Evaluation set:** reuse the Stage 1 50-sample portraits (we already have classifier labels and prompt-attribute ground truth on them). Add a second, unseen 50-sample draw from the full grid as a held-out set so we don't report Stage-1-train-on-train-test-on-train.

**Per-method, per-attribute metrics** (age, gender, 7-race one-hot):
- **Target-axis response curve:** apply direction `d_attr` at scales `λ ∈ {−1, −0.5, 0, +0.5, +1}` in appropriate normalized units; re-render through Flux; record classifier prediction on the target attribute at each λ. A clean direction gives monotone, near-linear response.
- **Identity drift:** ArcFace cosine between baseline render (λ=0) and edited render at each λ. Lower drift for the same target-response slope = more disentangled.
- **Off-axis drift:** classifier prediction delta on the *other* attributes (e.g. age edit → measure gender flip rate, race distribution shift). An age-direction that also shifts race is not an age direction.

**Reporting:** one table per attribute, three rows (us / FluxSpace / deferred), columns = target-slope, ID-drift at matched target-slope, off-axis drift at matched target-slope. Plus per-method qualitative sheets (9 portraits × 5 λ = 45 renders per attribute for visual inspection).

**Time & cost:** FluxSpace implementation = 0.5–1 day (project toolkit is public but needs wiring to our ComfyUI path or a `diffusers` hook). Generation = 3 × attributes × 5 λ × 50 portraits ≈ 750 renders per method; ~1.5h at 7s/render. Classification = negligible.

**Decision rule after Stage 4.5:**
- If ours is Pareto-dominated on all three metrics for all attributes → switch curriculum to use FluxSpace directions. Stage 5 runs on FluxSpace output.
- If ours and FluxSpace trade off (ours cleaner on age, FluxSpace cleaner on race, say) → keep ours for the orthogonalization step (it's what Stage 5 was designed for) and note FluxSpace as the editor we'd use if/when the curriculum needs attribute *manipulation* rather than attribute *subtraction*.
- If both fail the off-axis-drift bar → that's when Stage 4+ (fine-space regression) becomes load-bearing, not optional.

### Stage 5: Validation
- **Orthogonalization sanity:** take a held-out 200 samples, project their `c` onto `D`-complement, re-render through Flux, run classifiers → do demographic predictions collapse toward the dataset mean? (If not, `D` isn't capturing what we think it is.)
- **Curriculum readiness:** sample two random conditioning vectors `c_a, c_b` from `D`-complement with `‖c_a − c_b‖` = curriculum's σ unit → render → check inter-classifier demographic agreement says they look "same demographic." Go/no-go for curriculum Level 0.
- **FluxSpace-borrowed orthogonality check (added 2026-04-20):** after truncated SVD of the stacked weight matrix `W`, also report (a) pairwise cosine similarities between the retained direction vectors — close to 0 after SVD by construction, but worth confirming no numerical surprises; (b) **cosine similarity between same-attribute directions extracted by different classifiers** (e.g. MiVOLO age direction vs FairFace age direction vs InsightFace age direction). If they rediscover the same axis, cosines should be high (≳0.7). If they point different ways, the "age direction" is classifier-specific and the Stage 4 stacking is combining signals that shouldn't be combined. This is the sanity check FluxSpace uses for its own disentanglement claim (Dalva et al. CVPR 2025, §4.2) and is free to add here.

### Stage 4+ (future work): fine-space regression on MM-DiT attention outputs
**Trigger:** do this only if Stage 4's per-head CV R² on the 4864-d pre-transformer `c` is weaker than hoped (e.g. age R² < 0.5 with regularization tuned), *or* if Stage 5 orthogonalization doesn't collapse classifier predictions enough to unblock curriculum Level 0.

**What:** re-extract directions in FluxSpace's "fine" space — the joint-attention output tensors `ℓ_θ(x, c, t)` inside the MM-DiT blocks of Flux at a chosen timestep and layer. FluxSpace (Dalva et al. CVPR 2025) reports stronger disentanglement of facial attributes (age, gender, smile, eyeglasses) at fine-space than at coarse/pooled levels, and their qualitative ablations put the sweet spot at mid-depth layers (≈15 of 38 in Flux-dev). This is the same orthogonal-projection algebra we already use, just applied one layer further in.

**What it costs:** we currently have neither a ComfyUI node that dumps attention outputs nor a Python-side Flux inference path. Either:
- write a small ComfyUI custom node (`SaveAttentionOutput`) that caches the tensor at the selected block+timestep, or
- drop ComfyUI for this stage and run Flux via `diffusers` directly, registering a forward-hook on the target MM-DiT block.

Both are 1–2 days of work. Not worth doing unless Stage 4's coarse-space regression disappoints.

**What we'd gain:** finer disentanglement of demographic axes, especially for gender/race sub-structure where the pre-transformer pooled `c` has less geometry to work with. The fine-space directions would also be directly usable with FluxSpace's inference-time edit mechanism if we ever want to do attribute ablations outside the curriculum's `Δ`-sampling pipeline.

**What we don't do regardless of Stage 4 outcome:** don't switch to FluxSpace's prompt-pair contrastive direction extraction (e.g. "smile vs no smile"). That method answers "what direction changes this labeled attribute" — useful if we want editing; our question is "what direction does *a classifier* pick up on" — the two diverge when the classifier has bias the prompt-pair doesn't know about (e.g. InsightFace's +15y age bias on adult/elderly prompts). Classifier-predicted labels remain the right target for the curriculum's orthogonalization purpose.

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
- Stage 4.5 (cross-method comparison vs FluxSpace): 1–1.5 days
- Stage 5 (validation): 0.5 day

**Total active: ~3.5 days, plus one overnight.**

## Not Doing (Out Of Scope)

- Training our own demographic classifier on Flux output (future work if Stage 1 shows transfer failure)
- Fine-grained demographic axes (e.g., facial hair, hairstyle) — these can come later as separate confound controls
- Projecting out *identity* — that's a different axis and is preserved intentionally
- Adversarial robustness of the directions to prompt paraphrasing — noted as future work

---

**Next action after plan approval:** start Stage 0 environment setup. Write a short install-log doc as we go so the second attempt is cheaper.
