---
status: live
topic: metrics-and-direction-quality
summary: Post-compact resume plan — launch the overnight v2-format render batch, document drift framework as next work, skip the finished direction-injection dead-end.
---

# Post-compact state (2026-04-23)

## Where we are

Direction injection (both ridge and causal) **failed** at inference
(1/16 atoms with cos>0.3). Conclusion: linear attention-space
directions computed from training are not controllers. Two architectural
paths forward:

1. **Atoms-as-objective + EditPair-as-mechanism** — keep the atom
   decomposition as targets, use FluxSpaceEditPair's text-pair edits as
   the actual steering. Validated today: `smile_calibration_v2` showed
   FluxSpaceEditPair produces reliable smile edits (just needs per-base
   calibration for target-hitting).
2. **Drift framework** (formalized today, `docs/research/2026-04-23-drift-framework.md`) —
   edits pull identity / age / beard / hair / eye-closure along with the
   target channel. Measure drift with vendored ArcFace IR101 + MiVOLO +
   DEX + FairFace + CLIP probes, build a drift matrix, apply L1
   (preservation-clause prompts) → L2 (counter-edit composition).

## Immediate next step: launch overnight v2-format render batch

Script: `src/demographic_pc/overnight_render_batch.py`

Renders ~1440 samples across 3 purposes:
- **smile ladder densification**: 6 bases × 4 rungs × 4 scales × 8 seeds
  = 768. Fills the per-seed variance gap observed today
  (`smile_calibration_v2` showed ±0.2 smile per-seed on same recipe).
- **beard axis**: 2 add (asian_m, european_m) + 1 subtract (elderly_latin_m)
  × 4 scales × 8 seeds = 96. New axis — gender-aware: male bases only.
- **rebalance re-seed** (anger/surprise/pucker, the confirmed-usable
  three): 3 × 6 × 3 scales × 8 seeds = 432. Fills per-cell sample count.

Disk: ~12 GB total (~290 KB PNG + ~11 MB uncompressed fp16 `.v2.npz`
per edit render). Current free ~300 GB.

Launch command (after ComfyUI restart, which user has already done):

```bash
nohup uv run python -u -m src.demographic_pc.overnight_render_batch \
    --run --score > /tmp/vamp_overnight.log 2>&1 &
```

`--score` runs MediaPipe scoring after renders complete.

## Node state — current (reviewed)

`/home/newub/w/ComfyUI/custom_nodes/demographic_pc_fluxspace/__init__.py`
- v1 pickle writer preserved (backward compat)
- v2 `.v2.npz` writer added: uncompressed `np.savez` for speed
- Measurement dispatches on file extension (`.npz` → v2 slim, else v1)
- Dump frequency: tail-of-active-window only (last ~5% of window OR
  exact-sigma_end-boundary) → ~1–2 writes per render, not 16×
- Code-reviewed three times, edge cases (tmp-file suffix, double-dump,
  empty-steps overwrite, narrow-window silent-loss) all addressed

## Tomorrow — drift scorer build (when user says go)

1. `score_drift.py` — one-pass enrichment: ArcFace IR101, MiVOLO age,
   DEX age, FairFace race/gender/age, CLIP probe bank (bearded,
   long_hair, wrinkled, glasses, open_mouth, …)
2. `build_drift_index.py` — adds drift columns to
   `sample_index.parquet`
3. `analyze_drift.py` — per-(axis, base, scale) drift matrix + top
   offenders report
4. `compose_preserving_prompt.py` — L1 preservation-clause generator

Verify ArcFace/MiVOLO/DEX/FairFace weights load correctly before
committing to the full build (they're vendored under
`vendor/{FairFace,MiVOLO,pytorch-DEX}`).

## Key artefacts for resume

- `models/blendshape_nmf/sample_index.parquet` — 3502 samples × 87 cols
  (H_nmf_resid exact for 2842, pinv approx for 660)
- `models/blendshape_nmf/calibration_table.parquet` — (atom × axis ×
  base) linear response + usable flag
- `output/demographic_pc/direction_inject_broad/validation_report.json`
  — Phase-4 validation results (1/16 wins at cos>0.3)
- `output/demographic_pc/smile_calibration_v2/blendshapes.json` —
  baseline smile-edit sanity data (2/12 within ±0.1 of 0.7 target,
  but mechanism confirmed reliable)
- `docs/research/2026-04-23-drift-framework.md` — drift framework spec

## Memory entries to recall

- `feedback_review_node_changes.md` — ALWAYS review node edits
- `feedback_v2_npz_per_step_dumps.md` — fix applied today (tail-only dump)
- `feedback_measurement_format_versioning.md` — `.v2.npz` extension marks v2
- `reference_comfyui_paths.md` — node location (not repo's comfyui/)
- `reference_external_pkl_archive.md` — old pkls on external drive
