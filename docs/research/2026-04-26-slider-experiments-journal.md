---
status: live
topic: demographic-pc-pipeline
---

# Flux concept-slider experiments — journal

Append-only log of slider training runs. Each entry captures hypothesis, config, data, observed result, verdict. Source of truth for configs is the YAML in `~/w/ai-toolkit/config/`; this file documents the *why* and the outcome. New runs append at the bottom.

Trainer: ai-toolkit `concept_slider` extension on Flux Krea (text-pair Concept Sliders objective: `gt = neutral + η·(positive_pred − negative_pred)`, two backward passes per step at LoRA multiplier ±1.0).

## Schema

For each run, document:

- **ID** — matches output dir name and YAML filename
- **Hypothesis** — what we're testing
- **Config delta** — change vs the previous run (full config in YAML)
- **Data** — training images / pairs used
- **Run** — start, end, wall-clock, GPU
- **Result** — loss trajectory + sample-collage observation
- **Verdict** — pass / partial / fail, and *why*
- **Next** — what the next run should change

## Glossary

- **Module scope** — which Linear modules in the Flux transformer get LoRA adapters. See [`2026-04-26-flux-attention-slider-scope.md`](2026-04-26-flux-attention-slider-scope.md). Default = ai-toolkit `transformer_blocks` filter, 494 modules. Notebook xattn ≈ 152 modules. Surgical = 38 modules (`add_k_proj` + `add_v_proj` only).
- **α-override** — ai-toolkit's `lora_special.py:282-285` forces `alpha = rank` when peft format is detected, ignoring user `linear_alpha`. With nominal `lr` and forced `alpha=rank`, effective per-weight update = `lr · (rank/rank) = lr`, vs notebook's `lr · (1/rank)` = `lr/16`.
- **Effective LR inflation** = (α-override factor) × (module-count factor) × (backwards-per-step factor). For ai-toolkit default vs notebook: 16 × ~6 × 2 ≈ 32× ×6× = ~192× total update-norm inflation per step.

---

## glasses_slider_v0_overshoot_lr2e-3 — FAILED

**Hypothesis.** Notebook recipe (lr=2e-3, rank=16, alpha=1, η=2, 1000 steps) ports to ai-toolkit unchanged.

**Config delta.** First slider run, no prior. Followed published Flux Concept Sliders notebook hyperparameters verbatim.

**Data.** Text-pair only (no image supervision). Positive: `"a portrait photograph of a person wearing eyeglasses, glasses on face, eyewear"`. Negative: `"a portrait photograph of a person without glasses, no eyewear, bare face"`. Target class: `"a portrait photograph of a person"`. No anchor.

Noisy-latent source: `datasets/ai_toolkit_glasses_v1/train` (provides shape only, supervision comes from prompt triple).

**Run.** Single A6000, killed mid-training after step 400 produced pure noise at sample sweeps.

**Result.** Loss curves looked superficially fine (1e-2 → low single-digit e-3) but ±1.5 sweep samples were pure RGB noise.

**Verdict.** FAIL — three compounding errors discovered post-mortem:

1. **α-override.** `linear_alpha: 1` was silently overridden to `alpha = rank = 16` by ai-toolkit's peft-format path. Effective per-weight LR was 16× notebook.
2. **Module scope.** 494 modules trained vs notebook's ~76 (xattn-only on Flux). 6× more parameters at the same nominal LR → ~6× larger update-norm per step.
3. **Two backwards per step.** ConceptSliderTrainer calls `loss.backward()` twice (lines 263, 293 of `ConceptSliderTrainer.py`) — at multiplier +1 and -1. Doubles the effective optimizer step.

Combined effective LR inflation vs notebook: ~32× per weight, applied to ~6× more weights. Total update norm per step ~192× notebook's intended.

**Next.** v0 launch with conservative LR.

Output archived: `output/ai_toolkit_runs/glasses_slider_v0_overshoot_lr2e-3/`.

---

## glasses_slider_v0 — IN PROGRESS (started 2026-04-26 ~14:00 local)

**Hypothesis.** Cutting LR to 8e-5 (~16× below notebook nominal, compensating for α-override) lets the default ai-toolkit module scope train without blowing up. Conservative single-knob change.

**Config delta vs v0_overshoot.**

| Knob | overshoot | v0 |
|---|---|---|
| `lr` | 2e-3 | **8e-5** |
| Everything else | — | unchanged |

`linear_alpha: 1` retained in YAML for documentation but still overridden to 16 by peft-format path.

**Data.** Same text-pair as overshoot. Same noisy-latent source.

**Run.** Single A6000, ~37 min budget at 1000 steps × ~2.2 s/it. Started ~14:00 local. Sample sweeps at every 200 steps, save every 200.

**Mid-run observation (step 600).** Loss trajectory clean: 1e-2 (step ~50) → 2-4e-3 (step 200+), no spikes, no noise. Step-600 collage at `output/ai_toolkit_runs/glasses_slider_v0/samples/collage_progress.png`.

**No glasses emerging.** The +1.5 sweep across asian_m / black_f / european_m shows identity perturbations (mustache appears step 200/400, stubble grows on european_m) but no eyewear. Slider is finding *a* signed direction that satisfies `pos_pred > neg_pred` but it's drifting through facial-hair / identity space rather than locking onto the glasses concept.

This is the "degenerate direction" failure mode: low loss + no visible concept. Predicted by the architecture decomposition (494-module scope lets LoRA satisfy guidance objective via MLPs / single_blocks, which are rendering layers not conditioning layers).

**Verdict.** Pending final inference sweep at training end. Expected: PARTIAL FAIL — training stable, concept not learned.

**Next.** v1 should fix scope, not LR.

---

## glasses_slider_v1 — PLANNED

**Hypothesis.** Restricting LoRA scope to attention-only in `transformer_blocks` (notebook xattn analog on Flux MMDiT) forces the gradient signal through the text→image conditioning pathway, where the slider direction actually lives. Removes degenerate-direction escape hatches in MLPs and single_blocks.

**Config delta vs v0.**

| Knob | v0 | v1 |
|---|---|---|
| `network.ignore_if_contains` | not set | `[single_transformer_blocks, ff, ff_context, norm]` |
| `lr` | 8e-5 | 1.25e-4 (or 2e-3 — needs preflight) |
| `ema_config.use_ema` | true | **false** |

`ignore_if_contains` is OR-mode (subtract by substring match) — drops the entire single stack, both MLPs, and AdaLN modulation. Leaves `transformer_blocks.*.attn.*` only — both img and txt streams' Q/K/V/out — matching notebook's `train_method='xattn'` target on Flux.

EMA off because at 0.99 decay × 1000 steps, the saved weights heavily average over early-training noise. Notebook uses no EMA.

LR question: with α-override active and 3× fewer modules, the per-step update norm vs v0 is ~3× smaller at the same nominal LR. Going from 8e-5 to 1.25e-4 (≈1.5×) keeps the per-step update similar and lets us test scope independently of LR. If 1.25e-4 underfits, jump to 2e-3 and accept the α-override absorbs 16× of the budget.

**Data.** Identical to v0.

**Pre-launch checks.**

- Verify `ignore_if_contains` substring matching doesn't accidentally drop `attn` modules (e.g., does "norm" appear inside any attention module name? Quick grep on the loaded LoRA name list at startup will confirm.)
- Verify saved safetensors contains expected ~152 entries, not ~494.

**Pass criteria.**

- ±1.5 sweep at step 1000: clear glasses on +1.5, clean baseline on 0, visibly fewer glasses (or anti-eyewear cue) on -1.5.
- Identity preserved across sweep (no demographic drift, no skin/hair changes beyond the eyewear region).
- Holds across 4 demographics in inference sweep, not just the 3 train-prompt demos.

---

## 2026-04-26 — Flux-specific structural finding: guidance distillation weakens slider signal

After v0 underlearning, did wider literature search on Flux concept slider failures. Key finding:

**Slider algebra survives velocity parameterization.** `target = neutral + η·(pos - neg)` works in ε-space (SDXL) or v-space (Flux) because the `z` baseline cancels in differences. Not the issue.

**Guidance distillation IS the issue.** Flux dev / Krea was trained to mimic CFG-amplified outputs in a single forward pass via a learned `guidance_embedding`. ai-toolkit's `ConceptSliderTrainer.py` hardcodes `guidance_embedding_scale=1.0` for the three concept-prediction forwards, which is **out-of-distribution** for Flux's distillation training (typical range ~1.5-5.5). At scale=1.0 the model produces less prompt-sensitive predictions, so `(pos_pred - neg_pred)` has smaller magnitude than the equivalent `(ε_pos - ε_neg)` on SDXL. Slider direction is real but gradient signal is weaker per step → LoRA satisfies MSE by drifting in any direction loosely correlated with the (small) prompt delta, including degenerate identity perturbations.

**Sources confirming this is a known structural issue:**
- John Shi, "Why Flux LoRA So Hard to Train" — guidance distillation breaks standard LoRA techniques without computational headroom.
- Official Concept Sliders repo README — *"FLUX is not designed the same way as SDXL... Flux slider support is experimental and may not work as well as SDXL"*.
- `nyanko7/flux-dev-de-distill` exists specifically to restore real CFG and rescue training that breaks under distilled guidance.
- nyanko7 discussion: *"The key reason for failure in training Flux-dev lies in the cfg guidance scale... can destroy the guidance embedding."*

**Implication for v1 plan.** Scope tightening + EMA off addresses degenerate-direction symptom but not the underlying weak-signal cause. v1 should also:

- **Test guidance_embedding_scale=3.5 during training** (patch ai-toolkit or override). Each of pos/neg/neutral is then internally CFG-amplified, increasing (pos - neg) magnitude. η then re-amplifies; risk is compounded artifacts. Cheap experiment.
- **Validate the structural fix on a known-easy axis (smile) before tackling glasses.** If guidance=3.5 + xattn scope + no EMA learns smile cleanly, we've isolated the Flux-specific axis. If it still fails on smile, the issue is deeper (e.g., needs de-distilled base).

**Fallback if ai-toolkit + Flux dev/Krea remains uncooperative:**
- Switch base to `nyanko7/flux-dev-de-distill` — restores real CFG behavior; cleanest slider math but breaks compatibility with our existing `glasses_krea_v1` concept LoRA.
- Switch trainer to upstream notebook directly (its hyperparameters are validated end-to-end on Flux).
- Accept Flux is the wrong base for sliders and run sliders on SDXL as a second editing pipeline alongside our Flux measurement work.

## glasses_slider_v2 — STAGED (waiting for v1 GPU)

**Hypothesis.** Training Flux at `guidance_embedding_scale=3.5` (in-distribution for Krea's distillation regime) makes the (positive_pred − negative_pred) delta large enough to drive a clean slider direction, isolating the signal-magnitude question from v1's scope/EMA fix.

**Branch / patch.** `~/w/ai-toolkit` branch `vamp/flux-slider-guidance-3p5`, commit `237edee`. Patches `extensions_built_in/concept_slider/ConceptSliderTrainer.py` at all three `predict_noise` call sites in `get_guided_loss`:

- target combo forward (positive + neutral + negative + optional anchor) — line 162
- student forward at multiplier +1.0 — line 244
- student forward at multiplier −1.0 — line 278

All three changed from `guidance_embedding_scale=1.0` to `=3.5`. Both target and student must run at the same scale, otherwise student tries to match an in-distribution target while running OOD.

**Config delta vs v1.** Identical except `name: glasses_slider_v2`. The guidance change is in code, not YAML.

**Run.** Will launch as soon as v1 finishes (sequential on single A6000).

**Pass criteria.** Same as v1 — clear glasses on +1.5 sweep, clean baseline on 0, anti-glasses cue on −1.5, identity preserved.

**Decision matrix.**

| v1 result | v2 result | Inferred cause |
|---|---|---|
| Pass | (skip) | Was scope/EMA. Stop here. |
| Fail | Pass | Was guidance signal magnitude. v3 = combine both fixes; ablate later. |
| Pass | Pass | Either fix sufficient; can compose for stronger sliders or simplify. |
| Fail | Fail | Deeper Flux issue — consider de-distilled base or upstream notebook directly. |

If v2 overshoots (samples look distorted from too-strong amplification compounding η=2 with in-distribution prompt sensitivity), follow up with v2.1 dropping `guidance_strength: 2.0 → 1.5`.

## glasses_slider_v1 — UNDERLEARN (killed at step 400)

**Result.** No glasses emergence at +1.5 by step 200/400. Different failure than v0 — where v0 *drifted* (mustache, stubble — wrong direction), v1 with xattn scope was *stable but immobile*. Confirmed scope hypothesis: removing escape valves into MLPs / single_blocks eliminated the degenerate-direction failure but exposed a weak-signal floor where the LoRA simply has too little gradient magnitude to move.

**Verdict.** Failure mode different from v0 in informative way. The scope fix was correct but insufficient on its own — guidance distillation magnitude problem becomes the bottleneck. Killed at step 400 to free GPU for v2.

Output archived: `output/ai_toolkit_runs/glasses_slider_v1_underlearn_killed_step400/`.

---

## glasses_slider_v2 — OVERSHOOT (killed at step 600)

**Hypothesis.** Patching `ConceptSliderTrainer.py` to train at `guidance_embedding_scale=3.5` (in-distribution for Krea distillation regime) fixes the magnitude problem v1 exposed.

**Branch.** `vamp/flux-slider-guidance-3p5` commit `237edee` — patches all three `predict_noise` call sites in `get_guided_loss`.

**Config delta vs v1.** Identical YAML; guidance fix is in code.

**Result.** **The patch worked.** At step 200 +1.5 sweep, glasses visible on asian_m and european_m (clear thin-rim frames). v1 had nothing at step 200. Confirms in-distribution training recovered the (pos − neg) signal magnitude predicted by the wider-search literature.

**But also overshoot.** By step 400 the glasses were gone. Asian_m at +1.5 reverted to baseline; european_m grew stubble (v0's failure mode at +1 reappeared). The 0 column (slider neutral) also drifted across steps — mustache appeared on asian_m at m=0, beard heavier on european_m. The LoRA at multiplier=0 should be a no-op, so this signals the LoRA is pushing baseline appearance, not just the +/- direction.

**Why overshoot happened.**

1. **EMA was off** (v1 inheritance — turned off because v0 EMA=0.99 buried early signal). Without EMA the saved checkpoint reflects whatever instantaneous configuration the optimizer is in. Step 200 happened to land on glasses; step 400 wandered off.
2. **Guidance × η compounded amplification.** `guidance_embedding_scale=3.5` × `guidance_strength=2.0` → strong target → large per-step gradients → overshoot without smoothing.
3. **Glasses is one of many low-loss directions.** Stubble, age drift, demographic shift all satisfy `pos_pred > neg_pred` at LoRA +1. Without EMA averaging across the wander, individual checkpoints capture whichever direction the optimizer is currently passing through.

**Verdict.** Patch confirmed working (signal magnitude problem real). v3 needs to add stabilization to lock onto glasses.

**Useful artifact.** `glasses_slider_v2_000000200.safetensors` — the step-200 checkpoint has glasses on 2/3 demographics. That's already a usable slider, even if no later step beats it. Kept under archive `glasses_slider_v2_overshoot_killed_step600/`.

Output archived: `output/ai_toolkit_runs/glasses_slider_v2_overshoot_killed_step600/`.

---

## glasses_slider_v3 — IN PROGRESS

**Hypothesis.** Re-enabling EMA at low decay (0.95 — averages over ~last 20 steps) dampens v2's overshoot without burying signal. Should let glasses persist past step 200 and converge across the full run.

**Config delta vs v2.** EMA back on with `ema_decay: 0.95`. Everything else identical (same xattn scope, same guidance=3.5 patch via branch, same lr=1.25e-4, same η=2).

**EMA decay rationale.**
- 0.99 (v0): too slow — at decay 0.99 the EMA copy is heavily weighted toward early-training noise; concept signal that emerges at step 200 takes ~500 more steps to dominate the EMA. v0 used this and it buried any signal.
- 0.95 (v3): averages over ~last 20 steps. Dampens single-step gradient noise; recently-found directions (like glasses at step 200) become dominant within ~50 steps.
- Off (v1, v2): no smoothing — saved checkpoint = optimizer instant. Caused v2 overshoot.

**Pass criteria.**
- Glasses emerge by step 200-400 (consistent with v2's discovery)
- Glasses *persist* past step 400 (v3's stabilization fix)
- Identity preserved at m=0 across all steps (the v2 baseline drift goes away with EMA)
- ±1.5 sweep produces clean signed gradient at step 1000

**Decision matrix.**

| v3 result | Inferred verdict |
|---|---|
| Glasses persist + clean direction | Pipeline works. Move to other axes (smile, age, gender) and validate. |
| Glasses emerge then re-fade | EMA insufficient; try lower η (1.5) or lower lr (6e-5) in v4. |
| No glasses at all | EMA decay 0.95 still too slow given small-by-comparison gradients in xattn scope. Try decay 0.85 or 0.9 in v4. |
| Different failure | Reassess. |

## Backlog

Hypotheses worth running once the basic recipe lands:

- **Surgical scope** (`only_if_contains: [add_k_proj, add_v_proj]`, 38 modules) — does the most direction-pure scope underfit on glasses (visual concept) or work fine?
- **Same surgical scope on smile** — smiles are semantic relabel not new visual content; surgical scope should work better there than on glasses. Direct test of the per-axis-scope hypothesis from [`2026-04-26-flux-attention-slider-scope.md`](2026-04-26-flux-attention-slider-scope.md).
- **Anchor class** (e.g., `target_class="person"`, `anchor_class="hat"`) — currently null; paper recommends starting null but anchor regularization is a documented quality lever once the basic slider works.
- **Image-pair Path B** — `image_reference_slider_trainer` ported to Flux flow-matching. Higher-fidelity supervision (real α=0.4 glasses images) but requires ~30-50 line port. Postponed until A passes.
