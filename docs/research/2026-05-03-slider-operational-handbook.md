---
status: live
topic: arc-distill
---

# Slider thread — operational handbook

**Date:** 2026-05-03
**Purpose:** Complete recipe for building, training, and iterating Concept-Slider LoRAs with classifier-loss critics on Flux Krea. Captures everything we learned from 2026-04-23 (framework procedure) through 2026-04-30 (v1k working recipe). Replaces the shorter parking note `2026-05-03-slider-thread-recipe.md` (which stays as TL;DR).

This is the doc to load on a fresh session if you want to ship a new axis.

## How to use this doc

Read top-to-bottom on first load. On returning, jump to the section you need:
- *Solver taxonomy* — which solver to use for which axis
- *Dataset construction* — build the training corpus
- *Critic training* — train / re-train a noise-conditional bs / arc / sg head
- *Loss specification* — wire and weight loss terms in `ConceptSliderTrainer`
- *Slider training procedure* — phased plan, gates, hyperparameters
- *Iteration loop* — eval → curate → re-train cadence
- *Falsified approaches* — what NOT to redo (with provenance)

## Solver taxonomy

We have four solvers (A, B, C, plus paths C-surrogate and D direct-extraction). Pick by axis behaviour, not by preference.

| Solver | What it produces | When to use | Failure mode | Code |
|---|---|---|---|---|
| **A — FluxSpace prompt-pair** | Online attention-cache δ injection during render via `FluxSpaceEditPair`/`Multi` | First attempt on any axis. Composition via `compose_iterative` cancels confounds. | Identity drift ceiling ~0.4–0.5 cosine; pair halves must be **adjacent** (not diametric) on confound axis or one attractor wins | `~/w/ComfyUI/custom_nodes/demographic_pc_fluxspace/__init__.py`; `src/demographic_pc/compose_iterative.py` |
| **B — Image-pair Concept Sliders LoRA** | LoRA trained to learn `δv ≈ (z_neg − z_pos)/(2w)` on same-noise pairs | After Solver C produces a clean pair set. Production path for shippable sliders. | Per-axis training cost; classifier-fooling if critic is weak | `~/w/ai-toolkit/extensions_built_in/concept_slider/ConceptSliderTrainer.py` |
| **C — Inverse training-pair selection** | ~200–1000 pairs `(i,j)` with large Δintended, small `‖Δconfound‖_w` | Always run before Solver B. Breaks stereotype bundles in the corpus. | Corpus-distribution can be fundamentally bundled (e.g. Flux squint failed eff-rank 3/31 — Duchenne PC0 = smile+blink+cheek). FFHQ rescue worked at 16/21. | `src/demographic_pc/solver_c_squint_feasibility{,_ffhq}.py` |
| **C-surrogate (Path C)** | Tiny CNN φ predicting metric from latent → `−t·∇_z φ(x̂_0)` per-input gradient | Only after FFHQ rescue fails on an axis | Currently demoted; future thread. | `2026-04-26-measurement-grounded-slider-plan.md:189-244` |
| **D — Direct latent-direction extraction** | Closed-form ridge `d* = (AᵀA+λI)⁻¹Aᵀm` from cached attention | Validation only, not editing | **Cached-δ replay falsified 2026-04-23** — Flux editing is a stateful residual cascade, static δ injection produces no visible edit | `2026-04-23-cached-delta-replay-falsified.md` |

**Default order for a new axis:** A (sanity-check direction) → C (clean pair set) → B (train LoRA) → critic-loss head if axis has a measurable target.

## Dataset construction

### Sources

- **FFHQ reverse_index** — `output/reverse_index/reverse_index.parquet`, 70k rows, ~26k pass ArcFace `det_thresh=0.5`. Joins ArcFace 512-d + 70k MediaPipe blendshapes + 12 SigLIP-2 probes + MiVOLO + FairFace + InsightFace. Used by Solver C, mediapipe_distill v1, siglip_distill (`compact_siglip.pt`, 26,108 rows, 100% coverage).
- **Flux solver_a grid** — `output/demographic_pc/`, 7,652–8,115 rendered rows in `models/blendshape_nmf/sample_index.parquet`; per-row blendshapes/atoms/SigLIP/ArcFace/MiVOLO/FairFace/InsightFace.
- **Rendered slider corpus (v2c augmentation)** — additional 7,281 rows from slider sweeps, joined into `compact_rendered.pt`. Combined with FFHQ → 31,839 train + 2,003 val for v2c bs head.

### Solver C feasibility (run this first)

Per `2026-04-28-solver-c-{squint-feasibility-result,ffhq-rescue}.md`:

1. Cell key: `(ethnicity, gender, age)` for Flux, `(ff_race, ff_gender, ff_age_bin)` for FFHQ.
2. η-vector (confound features): blendshape scalars + base one-hots + ArcFace PCs. Flux: 6+17+8 = 31 dims. FFHQ: 5+16 = 21 dims.
3. Pair restriction: within-cell only. FFHQ adds top-50 ArcFace cosine KNN per row (FFHQ identities are unique).
4. Score `J = (Δθ)² / (Δη^T Σ_η⁻¹ Δη + σ_θ² · floor)`, Tikhonov λ=1e-3.
5. Diagnose top eigenvectors of selected-Δη covariance.
6. **Pass card:** ≥200 pairs above p90 of J; σ_θ² rank Spearman ≥0.90; eff-rank of selected Δη ≥ max(4, η-dim/4).
7. If pass → top-1000 selected pairs to `output/solver_c_ffhq/<axis>_pairs.parquet` with `selected=True` → VAE-encode at train time for Path B.

### Eye / region masks

For region-localized axes (squint, brow, mouth):

- Masks at `output/squint_path_b/eye_masks/` (one per train image).
- Construction: 2D Gaussian centered on eye region in image coords, normalized to `[0, peak]`.
- **Settled values for FFHQ-pair training:** `peak = 2.5–3.0`, `y0 = 0.41`, `x0 = 0.50`, `sigma_y = 0.06`, `sigma_x = 0.18`. (We used `peak = 5.0` early — over-concentrates gradient, per `distill-loss-sanity-plan.md:121`. For FFHQ pairs the per-image supervision means peak need not be hot; default 2.5.)

### Contamination filters (must be applied before training)

- ArcFace `det_thresh ≥ 0.5` (drops ~40% of FFHQ).
- `found` mask required (avoids the `aligned14 polluted-row` bug from arc-distill lessons).
- Drop empty-FairFace cells (~120 rows).
- Skip cells with <30 rows.
- Exclude degenerate-target channels (`val_std < 1e-6` like `_neutral`) from `bs_loss_channels`.
- **Per-channel R² gates** classify channels into four shippability buckets — `confident_ship` / `ship` / `do_not_ship` / `degenerate` — never use the aggregate `r2_mean` (the degenerate channels blow it up).
  - v2c bs head distribution: 23 confident_ship + 10 ship + 14 do_not_ship + 5 degenerate.

### Curation (the eyeball gate)

Contact sheet 7-col strengths × 3-row seeds per checkpoint via `src/demographic_pc/build_ffhq_contact_sheets.py`. Find `m_cliff` = lowest strength m where ≥2/3 seeds show iris occlusion / line-shaped lids / lid skin touching. **Quality-then-quantity rule:** when a new corpus is strictly better at given α, drop the prior corpus from that α range. Manifest is curated, not accumulated.

## Critic / loss-head training

We have three teacher-distilled student heads. Each is callable from `ConceptSliderTrainer` as a classifier-loss term.

### BlendshapeStudent (mediapipe_distill, `bs_a` family)

- **Code:** `src/mediapipe_distill/student.py`. Variant `bs_a` = `LatentStemFull64Native` (ConvT stride-2 → 112×112) + ResNet-18 trunk + Linear(512, 52) + sigmoid. 11.47 M params.
- **Training:** AdamW, lr 1e-3, cosine schedule, 20 epochs. v1 val MSE 0.0028, median R² 0.761.
- **Variants on disk** (`models/mediapipe_distill/`):
  - `v1`, `v2c` (FFHQ + rendered combined, warm-started), `v2b` (8-d NMF atom target, decoded via fixed H), `v2d` (two-stage latent → 106 landmark → 52-d), `v2e` (additional sanity).
  - `bs_v3_t/final.pt` — **canonical noise-conditional critic**. Trained with random-t noise augmentation. Justification: a clean-trained student fed Tweedie x̂_0 at high t injects high-variance random gradient; v1d froze.
  - `bs_v4_pgd/final.pt` — PGD-adversarial-trained robust critic (commits f30bdcb / 8df1fa0 / 4c91973 / a39cd35). **Validation pending** — gates G1–G4 in `docs/superpowers/plans/2026-04-30-bs-v4-pgd-robust-critic.md`.
- **t_max safe range:** 0.4 (preserve mode); 0.5 ("plenty" with R²=0.89 jawOpen); 1.0 catastrophic for clean-student v1d. Cap noise-conditional v3_t at ~0.6.

### AdapterStudent (arc_distill, `latent_a2_full_native_shallow`)

- **Code:** `src/arc_distill/`, predicts 512-d ArcFace embedding from Flux VAE latent `(16, 64, 64)`. 43.6 M params.
- **Architecture:** ConvT stride-2 stem + ResNet-18 + L2-normalized 512-d output (`stems.py:LatentStemFull64Native`).
- **Training:** Distilled from frozen IResNet50 ArcFace backbone on FFHQ. Pixel-A baseline 0.96 cos teacher, then full distill.
- **Gate:** held-out cosine R² ≥ 0.95; achieved ~0.881.
- **Use:** identity anchor at `id_loss_*` knobs in `ConceptSliderTrainer.py:46-59`, called from pos/neg halves at lines 519, 572. ArcFace's inductive bias (invariant to expression/pose, sensitive to identity) is exactly the inverse of what we want to penalize → it works as anchor where SigLIP failed.

### SigLIPStudent (siglip_distill, `sg_b`)

- **Code:** `src/siglip_distill/`. Same trunk as v2c (ResNet-18 → 512), head `Linear(512, 1152)` no activation, output L2-normed at inference.
- **Loss:** `0.5 · MSE + 0.5 · (1 − cos)`.
- **Train data:** `output/siglip_distill/compact_siglip.pt` (26,108 rows, 1152-d fp16, L2-normed).
- **Validation:** holdout cosine ≥ 0.85; per-text-probe margin R² ≥ 0.5 on the 12 SigLIP probes.
- **Use as anchor:** **failed** at w=5000 in v1i — too soft (VAE-bottleneck distill not strict enough about pixel-level identity vs ArcFace's identity-vs-expression bias). Use as semantic anchor for non-identity tasks; not as identity anchor.

### Distill-loss sanity protocol

Before launching a new slider with a critic loss, run the sanity protocol from `2026-04-30-distill-loss-sanity-plan.md`:

1. **arc_distill sanity** — `glasses_slider_v9_idloss_sanity.yaml`: id_loss weight 1.0, t_max 0.4, manual kill at step ~600. Pass: id_loss finite/non-NaN/descending; glasses engagement at m=±1.5 matches v8 step 600; ArcFace cos to m=0 anchor at m=±1.5 **higher** than v8 baseline.
2. **bs_loss preserve sanity** — `glasses_slider_v10_bsloss_sanity.yaml`: bs_loss preserve on `[mouthSmileLeft, mouthSmileRight, jawOpen, jawForward, browDownLeft, browDownRight]` — channels in confident_ship / ship and unrelated to glasses edit. Clean preservation gate.
3. Then proceed to engage / target on the actual axis.

## Loss specification

All loss heads are wired in `~/w/ai-toolkit/extensions_built_in/concept_slider/ConceptSliderTrainer.py`.

### Knobs and validated ranges

| Knob | Validated values | Notes |
|---|---|---|
| `id_loss_weight` | 0.5–1.0 sanity; **5000** in v1k working recipe | High weight only with arc_distill — not sg_b |
| `id_loss_t_max` | 0.4–0.5 | Match anchor's training distribution |
| `id_loss_t_norm` | 1000.0 | |
| `id_loss_checkpoint` | `models/arc_distill/checkpoint.pt` | |
| `id_loss_variant` | `latent_a2_full_native_shallow` | |
| `bs_loss_weight` | 1.0 sanity; 10000 in v1j/v1k working | High weight needs companion identity anchor |
| `bs_loss_t_max` | 0.4–0.5 | Match bs_v3_t / bs_v4_pgd training distribution. **1.0 catastrophic** with clean-student v1d |
| `bs_loss_t_norm` | 1000.0 | |
| `bs_loss_mode` | `preserve` (penalize change), `engage` (drive to target), `target` (drive channel to scalar) | preserve sanity-validated; engage used in v1c–v1k |
| `bs_loss_channels` | confident_ship/ship only | Never `_neutral`, `cheekSquintL/R`, `noseSneerL/R`, any `val_std < 1e-6` |
| `bs_loss_engage_target` | 0.5 (squint, but baseline overlap), **0.9** (jawOpen, confirmed) | Rule: target must be ≥ 2× the FFHQ baseline for that channel |
| `bs_loss2_*` | (ensemble distillation, second bs head) | Path 2c, not yet wired |
| `sg_loss_weight` | **falsified at 5000** (v1i) | Don't use as identity anchor; use ArcFace |
| Slider polarity | `+w` / `−w` per step at LoRA scale | `bs_only_mode: true` skips polarity (used in v1h–v1k) |
| `anchor_class` / `anchor_strength` | "a portrait photograph of a person" / 1.0 | Base prompt anchoring (do NOT anchor on edit halves — race-anchoring fails, see Falsified) |
| `mask_dir` / `eye_mask_peak` | peak 2.5–3.0 (was 5.0 over-concentrating) | masks at `output/squint_path_b/eye_masks` |

### Geometric latent mask (Path 1a fallback)

If identity anchor alone is insufficient:

```
L_geo = w · mean( (1 − M) ⊙ (z_lora_on − z_lora_off)² )
```

Forbids LoRA perturbation outside the mask region. Surgical, ~30 min to wire. Limitation: assumes mask is correct; deformable expressions may need soft masks. Best as additive belt-and-suspenders combined with ArcFace anchor (Path 1c).

### The v1k working recipe (canonical for new bs_only axes)

```yaml
slider:
  bs_only_mode: true                 # no slider polarity, no neg pass
  bs_loss_weight: 10000
  bs_loss_t_max: 0.5
  bs_loss_checkpoint: models/mediapipe_distill/bs_v3_t/final.pt   # or bs_v4_pgd once validated
  bs_loss_variant: bs_a
  bs_loss_mode: engage
  bs_loss_engage_target: 0.9         # well above any FFHQ baseline
  bs_loss_channels: [<single channel>]

  id_loss_weight: 5000
  id_loss_t_max: 0.5
  id_loss_checkpoint: models/arc_distill/checkpoint.pt
  id_loss_variant: latent_a2_full_native_shallow

  mask_dir: output/squint_path_b/eye_masks
  eye_mask_peak: 2.5
  eye_mask_y0: 0.41
  eye_mask_x0: 0.50
  eye_mask_sigma_y: 0.06
  eye_mask_sigma_x: 0.18
```

200 steps, batch 1, lr 1.25e-4, adamw8bit. Sample renders at multiplier=1.0 will look identity-collapsed until ~step 100; this is normal — arc anchor catches up.

## Slider training procedure (phased)

Per `2026-04-24-slider-trainer-phased-plan.md` and `step1-inventory` / `step2-lora-family` / `step3-validation` / `step4-hyperparams`.

### v1.0 single-axis smoke

- LoRA `r=16, α=1, ignore_if_contains=[single_transformer_blocks, ff, ff_context, norm]` → xattn-only (304 linears, ~30 MB).
- bf16. AdamW 8-bit, lr 2e-3, β₂=0.999, wd=0, grad clip max_norm=1.0.
- Constant LR + 200-step linear warmup. 1000 steps. Batch 1, grad accum 4.
- 512×512.
- Multi-α ∈ {0.25, 0.5, 0.75, 1.0}. **Note:** canonical Concept-Slider recipes train at binary endpoints `--scales '1,-1'`; our 8-value α grid is unprecedented and sublinear. For new axes prefer canonical text-pair (eta=2) or binary image-pair `--scales '1,-1'` ~56–84 cells.
- Logit-normal timestep `u ~ N(0.5, 1), t = sigmoid(u)` (peak ~0.62). **Note:** canonical Flux text-pair uses uniform; our μ=0.5 mid-timestep bias may entangle edit with face structure. Try uniform first on new axes.
- Eval shape: 5 train bases × 3 seeds × 5 α = 75 samples; LOBO hold-out.

### Glasses-track production hyperparameters (v8)

- `r=32, α=1, η=4, lr=1.25e-4 constant`, ~2000 steps, save_every=50.
- Surgical scope: `add_k_proj` + `add_v_proj` only (38 modules). Drop `proj_out` (downstream of all block interactions, perturbs residual-stream identity).

### v1.5 joint 10-axis composable

- 10 LoRA adapters loaded via PEFT `set_adapter`, ~10000 steps.
- Orthogonality reg: `λ · Σ_{i≠j} ‖A_i · A_jᵀ‖²_F / r²` on down-projections.
- P4 additivity ≤ 0.15. P5 interference ≥ 0.7×.

### Pass / ship gates

Per `step3-validation.md` (table at lines 108–118):

| Gate | Threshold |
|---|---|
| P1 target-Δ | ≥ 0.8× prompt-pair effect (e.g. eye_squint ≥ 0.72) |
| P2 monotonicity ρ | ≥ 0.85 |
| P3 R² | ≥ 0.80 |
| G1 ArcFace cos to anchor | ≥ 0.70 |
| G2 off-target L¹ | ≤ 0.03 |
| G3 anchor PSNR (s=0) | ≥ 40 dB |

**Ship rule:** all P pass + ≥1 G in pass + no G in fail.

### Inference — step-gated

`bs_only` LoRAs trained at `t_max=0.5` are **structural by construction**. At inference apply only at early-mid Flux denoising steps:

- `start_percent ≈ 0.0–0.05`
- `end_percent ≈ 0.75–0.85`

Without step-gating the LoRA perturbs detail bands it never trained on → identity drift / lighting collapse at multiplier=1.0. ComfyUI `LoRAControl` / ModelSamplingFlux step-gate covers this.

## Iteration loop

After a training run:

### Eval

1. **`measure_slider.py --phase render`** — strengths {-2.5, -1.5, -1.0, -0.5, 0, +0.5, +1.0, +1.5, +2.5} × ≥3 seeds × (in-distribution 9 + held-out ≥6) prompts. Output `models/sliders/<slider_name>/<ckpt_tag>/eval.parquet`. Fusion: `pd.concat([sample_index, lora_eval], join="outer")`.
2. **`scripts/sanity_bs_critic.py`** — multi-checkpoint critic readout at t=0 to Flux VAE encoding of rendered JPGs. Reports per-demographic eyeSquint / eyeBlink / jawOpen.
3. **finegrained_eval** (per `squint-slider-bs-loss-finegrained-eval-plan.md`) — 7 strengths × 3 seeds (2026, 4242, 7777 disjoint from train seed 1337) × 1 european_man prompt × 5 ckpts = 105 renders ~50 min. Build 7×3 contact sheet per ckpt; identify `m_cliff`.
4. **Fooling-collage** — every 25 steps for 3 demographics. Tracks identity collapse independent of bs-target satisfaction.
5. **ArcFace cosine to m=0 anchor** — identity_pass_075 ≥ 0.75.

### Curate back

- `selected=True` rows from solver C output → Path B trainer corpus.
- Quality-then-quantity rule: drop prior corpus from any α range where new corpus is strictly better.

### Triggers

| Symptom | Action |
|---|---|
| Engagement passes + bundle creep (off-axis SigLIP probes drifting positive, identity cosine sliding toward 0.4) | Branch from chosen ckpt; drop LR ~5–10× (1.25e-4 → 2e-5); cosine to small floor 5e-6 over 200–400 steps. **Do NOT cosine→0** (v6 falsified) |
| All 5 criteria pass on held-out | SHIP |
| No engagement at all | Falsification clause: structural failure. Change loss / mask / data, not schedule |

### Path 1 (regularizer) vs Path 2 (retrain critic) escalation

Per `2026-04-30-bs-loss-classifier-fooling.md:73-101`:

- **Path 1** — keep critic, stronger anchor:
  - 1a geometric latent mask (~30 min wire)
  - 1b ArcFace-on-x0 anchor (most likely to solve, 3–5× slower per step) — **this is the v1k recipe**
  - 1c combine 1a + 1b (production end-state)
- **Path 2** — retrain bs critic to be Lipschitz:
  - 2a adversarial-perturbation training (~1 h compute)
  - 2b PGD-style adversarial examples — **`bs_v4_pgd` is committed**, validation pending
  - 2c ensemble distillation, K critics with min-readings target

**Recommended sequence:** v1k = Path 1b first (working). In parallel queue Path 2a/2b — produces a reusable robust critic any future bs-loss benefits from. Geometric mask (1a) as additive fallback.

## Falsified approaches — do NOT redo

Each entry: claim, doc that falsified it, mechanism.

- **Cached-δ replay** (static direction injection from cached attention δs) — `2026-04-23-cached-delta-replay-falsified.md`. 4 experiments (channel-mean, full (L,D) K=20, full + renorm to scale=300, on-latent sanity) all produced zero visible edit. Mechanism: Flux editing is stateful residual cascade. Falsifies Path D static injection AND Option C learn-g(attn_base)→delta.
- **Atom injection (specific scope)** — `directions_resid_causal.npz` atom_16 — `2026-04-23-atom-inject-visual-failure.md`. **NOT falsified at general level:** `directions_k11.npz`, `directions_resid.npz`, prompt-pair edits remain valid. Run Option-C gating pilot on `directions_k11.npz` at scales calibrated to that file's tensor norms before declaring atoms dead generally.
- **v6 lion** — `2026-04-27-v6-lion-falsified.md`. Cosine→0 collapses gradient budget faster than schedule predicts; training stalls before refinement. Use small floor (5e-6), not zero.
- **sg-only anchor at w=5000** (v1i) — `bs-loss-classifier-fooling.md:65`. sg_b distilled from VAE bottleneck not strict enough about pixel identity. Cosine preserved numerically while pixel identity drifted; no squint emerged. Use ArcFace.
- **"Same person" identity anchor language (v5_identity_anchor)** — `framework-procedure.md:241`. Did not meaningfully anchor identity.
- **Race contrastive pair-averaging** (race/iter_01) — `framework-procedure.md:212`. Pair-averaging interprets `pos=Latin/neg=East Asian` as a *mixture*, not subtraction. **Pair halves must be ADJACENT, not diametric, on the confound axis.**
- **`{ethnicity}` anchoring inside edit pair pos+neg** (smile_iter_04) — `2026-04-23-smile-iter04-race-anchoring-fails.md`. Both halves carry the anchor → cancels in δ. Anchor base prompt only; compose counter-edits for δ-side drift.
- **v0_overshoot lr=2e-3 ai-toolkit notebook port** — silent α-override, 494-module scope, 2-backward-per-step → ~192× update inflation. Use `linear_alpha:1` retained but understand the override; ai-toolkit defaults are NOT notebook defaults.
- **v1d clean-trained student at t_max=1.0** — memory `project_v1d_slider_falsified_by_clean_student`. High-variance random gradient drowns base loss. Direct fix: noise-conditional bs_v3_t with z_t (not Tweedie), cap t_max≈0.6, per-channel R² mask.
- **`alpha=rank`** — set `alpha=1` at `rank=16` (effective 1/16 magnitude). Our alpha=16 was 16× notebook update magnitude.
- **`xattn + proj_out` target modules** — drop proj_out, xattn only. proj_out is downstream of all block interactions; perturbs residual-stream identity.
- **Anchors (Ostris-style)** — explicitly flagged convergence footgun. Don't add until simpler recipe is debugged.
- **Ridge atoms as input** — atoms are always *output*, never input (`project_editing_framework_principle`).
- **Solver C on Flux corpus for squint** — corpus-distribution problem (Duchenne bundled). FFHQ rescue is the route; corpus rebalance unnecessary.
- **`r2_mean` aggregate metric** — degenerate channels (`_neutral`, etc.) blow it up. Always per-channel buckets.

## Failure ladder we climbed (squint → jaw → arc anchor)

| Run | Change | Result | Lesson |
|---|---|---|---|
| v1c–v1g | classic slider+bs | eyeSquint↔eyeBlink coupled, closure at high m | FFHQ correlation prevents isolation with bs alone |
| v1h | bs_only_mode (no polarity) | critic satisfied, render unchanged | Classifier-fooling at training scale; off-manifold direction |
| v1i | v1h + SigLIP anchor (sg_b, w=5000) | critic still fooled, sg cosine preserved, identity drifts | SigLIP-on-latent too soft for pixel-identity preservation |
| v1j | v1h with single jawOpen→0.9 (sanity) | mouths visibly open, severe identity collapse | bs_only infrastructure works; identity drift is universal failure mode |
| v1k | v1j + arc_distill anchor (w=5000) | mouths open, identity preserved across all 3 demos | **Working recipe.** ArcFace inductive bias is the missing piece. |

## Pending work (resume sequence)

```bash
# 1. Validate v4_pgd vs the foolability gates (G1–G4 in the v4_pgd plan)
.venv/bin/python scripts/sanity_bs_critic.py   # multi-checkpoint side-by-side (commit a001e99)

# 2. (If G4 passes) draft v1l_squint_arc.yaml from lora_v1k_jaw_arc.yaml:
#    - bs_loss_channels: [eyeSquintLeft, eyeSquintRight]
#    - bs_loss_engage_target: 0.9   (raised from 0.5 to clear FFHQ baseline overlap)
#    - bs_loss_checkpoint: models/mediapipe_distill/bs_v4_pgd/final.pt
#    - keep id_loss_* at 5000

# 3. Launch via ai-toolkit (NOT through `uv run` — direct venv to avoid oyaml miss):
/home/newub/w/ai-toolkit/.venv/bin/python /home/newub/w/ai-toolkit/run.py \
  /home/newub/w/ai-toolkit/config/lora_v1l_squint_arc.yaml

# 4. Step-gated render at inference:
#    ComfyUI LoRAControl with start_percent=0.0, end_percent=0.8

# 5. Evaluate via measure_slider + sanity_bs_critic + finegrained_eval contact sheets
#    Identify m_cliff. If clean (no closure), promote to ship.
```

Open question to test once v1l ships: does v4_pgd compose multiplicatively with arc anchor (cleaner v1l with both, or does one dominate)? 2×2 ablation if compute permits.

## Cross-references

- Parking note (TL;DR): `2026-05-03-slider-thread-recipe.md`
- Neural-deformation alternative: `2026-05-03-neural-deformation-synthesis.md`
- Master procedure: `2026-04-23-framework-procedure.md`
- Diagnosis of classifier-fooling: `2026-04-30-bs-loss-classifier-fooling.md`
- Critic training plan: `2026-04-30-mediapipe-distill-plan.md`, `…-handoff.md`
- Distill loss sanity: `2026-04-30-distill-loss-sanity-plan.md`
- arc_distill lessons: `2026-04-30-arc-distill-lessons.md`
- Solver C: `2026-04-28-solver-c-{squint-feasibility-result,ffhq-rescue}.md`
- Slider trainer phased plan: `2026-04-24-slider-trainer-phased-plan.md` + step1–4
- Eval / measurement: `2026-04-26-slider-quality-measurement.md`, `…-experiments-journal.md`
- v4_pgd gates: `docs/superpowers/plans/2026-04-30-bs-v4-pgd-robust-critic.md`
