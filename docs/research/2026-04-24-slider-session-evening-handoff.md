---
status: live
topic: demographic-pc-pipeline
supersedes: 2026-04-24-slider-session-handoff
---

# Slider session — evening handoff (post-power-loss)

This doc supersedes `2026-04-24-slider-session-handoff.md` for current
state; that one captures the morning/early-afternoon arc.

## What changed this afternoon

**v3 corpus completed (Strategy C, pair-averaging, original prompts).**
400 PNGs at `output/demographic_pc/fluxspace_metrics/crossdemo_v3/eye_squint/`.
Identity-pass curve via ArcFace cos≥0.75:

| α | pass rate |
|---|---|
| 0.00 | 100% |
| 0.15 | 100% |
| 0.30 | 98% |
| 0.45 | 82% |
| 0.60 | 61% |
| 0.75 | 39% |
| 0.90 | 8% |
| 1.00 | 6% |

Visually (`preview_eye_squint_v3.png`): two cosmetic drifts inherited
from prompt phrasing —
- **Smile drift** from pair 2's "as if smiling warmly".
- **Side-look drift** from pair 1's "alert eyes".

Plus `adult_middle_f` mostly veiled by Flux's "Middle Eastern woman"
prior, costing pixel area and weakening the edit on that base.

## v3.1 — sanitized prompts (PARTIAL)

Started rendering `crossdemo_v3_1/eye_squint/` with three fixes:

1. Pair 1 → `"wide-open eyes looking directly at the camera"` (no "alert")
2. Pair 2 → `"softly squinted eyes as if shielding against bright sunlight, looking at the camera"` (no "smiling")
3. `adult_middle_f` overridden to `"a Lebanese woman with long dark wavy hair flowing past her shoulders, olive skin, dark almond eyes, ..."` — re-anchors demographic without invoking veil prior.

**Power loss interrupted at 148/400 renders (4 of 10 bases complete).**
4 PNGs were truncated mid-write (`adult_european_m/s7777_a0.{00,15,30,45}`),
deleted on cleanup. **144 valid PNGs ingested into the index.**

Even on this partial corpus, v3.1 dramatically beats v3 at high α
(n=18 base×seed cells per α, all from `adult_asian_m`, `adult_black_f`,
`adult_european_m` partial, `adult_latin_f`):

| α | v3 pass% | v3.1 pass% |
|---|---|---|
| 0.45 | 82 | 100 |
| 0.60 | 61 | 89 |
| 0.75 | 39 | **72** |
| 0.90 |  8 | **72** |
| 1.00 |  6 | **67** |

The smile/sidelook drift was costing more identity than the squint
itself. With them removed, identity holds usable up to α=1.00.

## Curator built — `src/demographic_pc/curate_slider_training.py`

Scatter–gather: pulls rows from all corpora (v1, v2, v3, v3.1) where:
- `identity_pass_075 == True`
- `Δ(eye_squint) ≥ τ_edit` (default 0.15) vs (source, base, seed, α=0) anchor
- `|Δ(smile)| ≤ τ_smile` (default 0.15)
- `|Δ(off-axis-gaze)| ≤ τ_gaze` (default 0.10)

Anchors (α=0 cells) included unconditionally so the trainer has neutral
references.

Output: `models/flux_sliders/training_manifest_eye_squint.parquet` —
**297 rows** post-cleanup.

α distribution of the manifest (this is the curated training set):

| α | # cells | dominant source |
|---|---|---|
| 0.00 | 136 | (anchors, all corpora) |
| 0.15 | 4 | v3.1 + v3 |
| 0.20 | 2 | v2 |
| 0.25 | 3 | v1 |
| 0.30 | 33 | v3, v3.1, v2 |
| 0.40 | 16 | v2 |
| 0.45 | 31 | v3, v3.1 |
| 0.50 | 10 | v1 |
| 0.60 | 26 | v3, v3.1 |
| 0.75 | 15 | v3.1 dominant |
| 0.90 | 11 | v3.1 dominant |
| 1.00 | 10 | v3.1 dominant |

For the first time we have usable training cells out to α=1.00.

## Power-loss event (2026-04-24 ~17:40 CDT)

Catastrophic loss. Diagnosed via journalctl, dmesg, `last -x`,
`nvidia-smi` post-reboot:

- **`last -x`** marks the prior session **`crash`** (no clean shutdown).
- Journal trails off mid-`sysstat-collect` at 17:40:13, no kernel
  panic/oops/Xid/NVRM/EDAC/MCE/thermal events anywhere in the prior
  boot.
- Post-reboot idle nvidia-smi: 47–49 W, 55–56 °C, **pviol=0, tviol=0,
  dbecc=0, PCIe errs=0** over 10s window.

Most consistent with **wall power loss**, not GPU/PSU/thermal trip.
Caveat: a fast PSU OCP can cut without logging if the disk buffer dies
first; can't 100% rule out. **No further heavy GPU loads today** —
ComfyUI down, v3.1 render not resumed, no training launches.

Hardening for next session:
```bash
sudo apt install lm-sensors && sudo sensors-detect --auto
nvidia-smi dmon -s pucvmet -c 600 > /tmp/dmon.log &  # log during next render
```

## Index state

`models/blendshape_nmf/sample_index.parquet` — **7332 rows × 91 cols**.
Includes:
- v1 corpora (overnight axes, reprompt v2, original crossdemo)
- v2 (`v2_eye_squint`, Strategy A compressed mix_b)
- v3 (`v3_eye_squint`, Strategy C full sweep)
- **v3.1** (`v3_1_eye_squint`, sanitized prompts, partial 144 rows)

Backup parquets at `*.bak.20260424_*` from each save.

`extend_sample_index_v2.py` updated to discover `crossdemo_v3_1/` —
re-run extends without changes when v3.1 render is completed later.

## Files that materialised this afternoon

```
src/demographic_pc/curate_slider_training.py       NEW — scatter–gather curator
src/demographic_pc/expand_corpus_v3_multipair.py   updated — sanitized AXIS_PAIRS, BASE_PROMPT_OVERRIDES, v3_1 output
src/demographic_pc/extend_sample_index_v2.py       updated — knows v3 + v3_1 sources
output/demographic_pc/fluxspace_metrics/crossdemo_v3/eye_squint/    400 PNGs (complete)
output/demographic_pc/fluxspace_metrics/crossdemo_v3_1/eye_squint/  144 PNGs (partial, deferred)
models/flux_sliders/collages/preview_eye_squint_v3.png   10×8 visual
models/flux_sliders/training_manifest_eye_squint.parquet 297 curated rows
models/blendshape_nmf/sample_index.parquet               7332 rows × 91 cols
```

## Next-session moves (hardware permitting)

In order:

1. **Hardware sanity** — install lm-sensors, check PSU rails. If
   uncertain about root cause, try a low-power render (one short
   v3.1 batch) with `nvidia-smi dmon` recording in another shell.
   Abort if voltages drift or pviol/tviol climb.
2. **Resume v3.1 render** — script is resumable (skip-if-exists), so
   just re-launch:
   ```
   uv run python src/demographic_pc/expand_corpus_v3_multipair.py --axis eye_squint --alphas 0.0 0.15 0.30 0.45 0.60 0.75 0.90 1.00
   ```
   Picks up the remaining ~256 PNGs (~64 min ETA).
3. **Re-extend index + re-curate** (CPU-only, safe).
4. **Eyeball v3.1 collage** at full coverage.
5. **Train eye_squint v1.1** on `training_manifest_eye_squint.parquet`.
   Trainer needs to be taught to read the manifest (currently it walks
   a directory). Small change: filter by manifest `img_path` set.
6. **Compare** v1.0 (drifted) vs v1.1 (curated) sliders side-by-side.

## Operational invariants (unchanged)

- ComfyUI + training are VRAM-incompatible on 32 GB. Render needs
  ComfyUI up; train needs it killed.
- Render scripts use **flat prefixes** (`expand_v3_1_<axis>_<base>_...`)
  not `/`-separated.
- Classifier_scores embeddings must be float32 for pyarrow.
- `extend_sample_index_v2.py` rescores when source rows lack
  embeddings in `>30%` of cases.
