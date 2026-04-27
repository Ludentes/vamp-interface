---
status: live
topic: demographic-pc-pipeline
---

# Slider experiments — resume note (2026-04-26)

Paused to free GPU. Everything below is staged and ready to launch.

## State on pause

- **v1 concept LoRA** (`glasses_krea_v1.safetensors`): trained, validated. Clean — glasses appear with trigger, no composition bleed without trigger. ai-toolkit + Krea + LoRA-rank-8 + 500 steps + paired-anchor regularization works end-to-end.
- **Path A slider config** (`~/w/ai-toolkit/config/glasses_slider_v0.yaml`): written + reviewed against published Flux Concept Sliders notebook. Ready to launch.
- **Plot twist found:** ai-toolkit ships its own `concept_slider` extension implementing the canonical 4-forward Concept Sliders objective — Path A is now config-only, no code. `image_reference_slider_trainer` (Path B) also exists but its loss is SD/SDXL-era (DDPM noise prediction, not Flux flow-matching velocity) — needs ~30-50 line port to work on Flux.

## Resume checklist

When GPU is free again, in order:

1. **Launch Path A** (~80 min, 5 forwards/step × 1000 steps):
   ```bash
   cd ~/w/ai-toolkit && source .venv/bin/activate && \
     python run.py config/glasses_slider_v0.yaml
   ```
   Output: `output/ai_toolkit_runs/glasses_slider_v0/glasses_slider_v0.safetensors`
   Sample sweeps with `--m -1.5 / 0 / +1.5` saved every 200 steps.

2. **Render Path A inference sweep** at strength {-1.5, -0.5, 0, 0.5, 1.0, 1.5} across 4 demographics. Adapt `inference_glasses_v1.py`:
   - Change LORA path to `glasses_slider_v0.safetensors`
   - Sweep `STRENGTHS = [-1.5, -0.5, 0.0, 0.5, 1.0, 1.5]`
   - **Drop the trigger phrase from prompts** (sliders activate from strength alone)

3. **Build collage + verdict.** Success criteria:
   - At strength −1.5: visibly *less* glasses than baseline (slider goes both ways)
   - At strength 0: matches baseline Krea
   - At strength +1.5: clear glasses, identity preserved, no style bleed
   - Composition (clothing, lighting, backdrop) stable across the strength sweep

4. **If A passes → start Path B port.** Patch `extensions_built_in/image_reference_slider_trainer/ImageReferenceSliderTrainerProcess.py:hook_train_loop` to:
   - Replace `noise_scheduler.add_noise()` with Flux flow-matching `x_t = (1-t)*z + t*noise`
   - Replace `target = noise` with `target = noise - z` (velocity)
   - Use logit-normal timestep sampling + Flux sigma shift (see our `train_flux_image_slider.py:658-666` for the formula)
   - Build paired side-by-side image dataset from `datasets/ai_toolkit_glasses_v1/{train,reg}/` (concat horizontally)

## Files staged

| Path | Purpose |
|---|---|
| `~/w/ai-toolkit/config/glasses_slider_v0.yaml` | Path A config (text-pair Concept Slider) |
| `~/w/ai-toolkit/config/glasses_krea_v1.yaml` | Validated v1 concept LoRA recipe (golden baseline) |
| `~/w/ai-toolkit/inference_glasses_v1.py` | Inference sweep template (modify for slider) |
| `datasets/ai_toolkit_glasses_v1/train/` | 7 α=0.4 with-glasses PNGs + captions |
| `datasets/ai_toolkit_glasses_v1/reg/` | 7 α=0.0 anchors + captions (matched composition) |
| `datasets/ai_toolkit_glasses_v1/inspect_pairs.png` | Pre-flight collage |
| `output/ai_toolkit_runs/glasses_krea_v1/glasses_krea_v1.safetensors` | Validated v1 concept LoRA |
| `output/ai_toolkit_runs/glasses_krea_v1/inference_test/collage_v1.png` | v1 verdict collage |
| `docs/research/external/train-flux-concept-sliders.ipynb` | Canonical recipe (review reference) |
| `docs/research/external/sdxl-train-lora-scale.py` | SDXL image-pair script (Path B reference) |

## Review of `glasses_slider_v0.yaml` vs the published notebook

Already done — see chat transcript. Summary:
- ✅ Match: model arch, steps=1000, lr=2e-3, eta=2, rank=16, alpha=1, height=512, dtype=bf16, batch=1, loss math
- ⚠ Different (probably benign): ai-toolkit's default timestep weighting (logit-normal) vs notebook's uniform; ai-toolkit's normalize-to-neutral vs notebook's normalize-to-positive
- ⭐ Better: ai-toolkit explicitly trains both LoRA polarities (+1, -1) per step; notebook only amplifies positive
- ❓ Unspecified: lr_warmup_steps (notebook = 200), lr_scheduler (notebook = constant), train_method (notebook = xattn-only)
