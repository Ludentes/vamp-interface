## Photobooth sweep

**Status:** Phase 2 complete (2026-05-20). Robust default identified
(`cn_strength=0.85, refine_denoise=0.00, demo_inject=on,
face_pixel_budget=natural_1024, cn_condition=canny`). Per-photo response
is bimodal — 4/20 photos are rock-solid, 4/20 are structurally hard
(beard, glasses, hat, depth-collapse). Adaptive lookup is the next axis.

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
- [`2026-05-20-photobooth-phase2-findings.md`](../2026-05-20-photobooth-phase2-findings.md)
  — current state. 360 cells, 20 identities × 6 configs × 3 seeds.
  Identifies tight_1024 high-risk/high-reward, photo-bimodal failure
  pattern, 5 named failure modes. **Start here.**
- [`2026-05-20-photobooth-phase1-findings.md`](../2026-05-20-photobooth-phase1-findings.md)
  — LHS pilot. Locked-in `refine_denoise=0.00`, `demo_inject=on`,
  `cn_strength≈0.85`. Pruned `natural_768` and refine-as-axis.

### Live artefacts

- Driver: `scripts/photobooth_sweep/driver.py`
- Axes catalog + Phase 2 grid: `scripts/photobooth_sweep/axes.py`
- Contact-sheet generator: `scripts/photobooth_sweep/contact_sheet.py`
- Per-cell intermediates: `exp_output/photobooth_phase2/cells/<cell_id>/`
- Score table: `exp_output/photobooth_phase2/scores.parquet`
- Visual sheets: `exp_output/photobooth_phase2/grid_s{0,1,2}.png`,
  `stages.png`

### Spec / plan

- Spec: `docs/superpowers/specs/2026-05-19-photobooth-phase1-sweep-design.md`
- Plan: `docs/superpowers/plans/2026-05-19-photobooth-phase1-sweep.md`
- **Group-photobooth spec (2026-05-21):** `docs/superpowers/specs/2026-05-21-group-photobooth-pipeline-design.md` — multi-person photo → matryoshka group portrait on chosen background. Approach A: per-face render + rembg + composite.
- **ComfyUI compositor toolkit research (2026-05-21):** `docs/research/2026-05-21-comfyui-compositor-toolkit.md` — preferred node packs per axis (Impact-Pack, LayerStyle, kijai SAM2, 1038lab RMBG, cozymantis human-parser). Recommended adds for group-photobooth.

### Cross-thread

- [[reference-comfyui-shard-runbook]] — the Windows 3090 host
- [[feedback-sweep-avoid-local-gpu]] — silent-OOM root cause
- [[project_matryoshka_bakeoff]] — prior swap baseline (Phase 0)
