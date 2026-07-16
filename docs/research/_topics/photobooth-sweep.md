## Photobooth sweep

**Status:** Phases 3–4 run (2026-05-20) and human-judged (2026-07-16).
**Production recipe:** `tight_1024 + canny/aggressive @0.85,
refine_denoise=0.00, demo_inject=on, swap_weight=0.40` (light HyperSwap
embedding mix — beat the 0.5 baseline on *both* id_cos and clip_style).
Mask erode is a dead dial (kills identity, buys no style); feather is a
seam-repair knob only. Caveats: the two winning picks were never run
together, and `tight_1024` carries a ~40 % no-face nan rate that needs a
fallback before productisation. Per-photo response remains bimodal —
4/20 photos rock-solid, 4/20 structurally hard (beard, glasses, hat,
depth-collapse); adaptive lookup is still the open axis.

**Pipeline:** Z-Image Turbo + Z-Image-Turbo-Fun-CN-Union + HyperSwap-1c,
running on Windows RTX 3090 shard via `scripts/photobooth_sweep/driver.py`.

### Read-first

- [`2026-05-20-arcface-topology.md`](../2026-05-20-arcface-topology.md)
  — ArcFace embedding-space manifold / linearity / OOD survey. Justifies
  the lerp+renorm math we use, flags that the matryoshka target is OOD
  (TPR collapse from 0.765→0.372 on stylized [StyleID]), and proposes
  two unexplored levers (average-target-embedding for stability;
  constructed `mean(doll)−mean(photo)` stylization direction).
- [`2026-05-20-hyperswap-parameters.md`](../2026-05-20-hyperswap-parameters.md)
  — HyperSwap inference parameter survey. Identifies the one knob we
  missed (`face_swapper_weight`, the embedding mix), the ONNX runtime
  contract (only `source` + `target`, no hidden weight tensor), and the
  recommended Phase 3 sweep direction (lower w → more matryoshka).
- [`2026-07-16-photobooth-phase3-4-findings.md`](../2026-07-16-photobooth-phase3-4-findings.md)
  — current state. Phase 3 swap_weight sweep + Phase 4 mask-geometry
  sweep + human judging verdicts + production recipe + open items
  (combined cell untested; tight_1024 nan fallback needed;
  `face_renderer.py` still hardcodes heavy mix). **Start here.**
- [`2026-05-20-photobooth-phase2-findings.md`](../2026-05-20-photobooth-phase2-findings.md)
  — 360 cells, 20 identities × 6 configs × 3 seeds.
  Identifies tight_1024 high-risk/high-reward, photo-bimodal failure
  pattern, 5 named failure modes.
- [`2026-05-20-photobooth-phase1-findings.md`](../2026-05-20-photobooth-phase1-findings.md)
  — LHS pilot. Locked-in `refine_denoise=0.00`, `demo_inject=on`,
  `cn_strength≈0.85`. Pruned `natural_768` and refine-as-axis.

### Live artefacts

- Driver: `scripts/photobooth_sweep/driver.py`
- Axes catalog + phase grids: `scripts/photobooth_sweep/axes.py`
- Contact-sheet generator: `scripts/photobooth_sweep/contact_sheet.py`
- Judging-sheet generator: `scripts/photobooth_sweep/judging_sheet.py`
  (column sets `p3` = swap_weight axis, `p4` = all-phases survey)
- Per-cell intermediates: `exp_output/photobooth_phase{1..4}/cells/<cell_id>/`
  (local-only, gitignored)
- Score tables: `exp_output/photobooth_phase{1..4}/scores.parquet`
- Judging sheets: `exp_output/photobooth_judging/sheet_p{3,4}.png`
  (local-only, gitignored)
- HyperSwap knobs (`weight`, `mask_erode_px`, `mask_feather_px`):
  `scripts/swap_core.py`

### Spec / plan

- Spec: `docs/superpowers/specs/2026-05-19-photobooth-phase1-sweep-design.md`
- Plan: `docs/superpowers/plans/2026-05-19-photobooth-phase1-sweep.md`
- **Group-photobooth spec (2026-05-21):** `docs/superpowers/specs/2026-05-21-group-photobooth-pipeline-design.md` — multi-person photo → matryoshka group portrait on chosen background. Approach A: per-face render + rembg + composite.
- **ComfyUI compositor toolkit research (2026-05-21):** `docs/research/2026-05-21-comfyui-compositor-toolkit.md` — preferred node packs per axis (Impact-Pack, LayerStyle, kijai SAM2, 1038lab RMBG, cozymantis human-parser). Recommended adds for group-photobooth.
- **Group-photobooth plan (2026-05-21):** `docs/superpowers/plans/2026-05-21-group-photobooth.md`
- **Group-photobooth runbook (2026-05-21):** `docs/research/2026-05-21-group-photobooth-architecture.md` — pipeline diagram, CLI, per-person cache layout, smoke tests, background-library schema.

### Cross-thread

- [[reference-comfyui-shard-runbook]] — the Windows 3090 host
- [[feedback-sweep-avoid-local-gpu]] — silent-OOM root cause
- [[project_matryoshka_bakeoff]] — prior swap baseline (Phase 0)
