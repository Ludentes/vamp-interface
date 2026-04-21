# demographic_pc_edit — ComfyUI custom node

Injects a precomputed direction into Flux CONDITIONING. Used by Stage 4.5 to
compare conditioning-level edits (Ours vs FluxSpace-coarse).

## Install

Copy (or symlink) this directory into `ComfyUI/custom_nodes/` and restart ComfyUI.

```bash
ln -s "$(pwd)/comfyui/custom_nodes/demographic_pc_edit" /path/to/ComfyUI/custom_nodes/demographic_pc_edit
```

## Inputs

- `conditioning` — CONDITIONING from CLIPTextEncode.
- `edit_npz_path` — path to a `.npz` with keys `pooled_delta` (768,) and `seq_delta` (4096,).
- `strength` — float. If 0, node is a no-op.

## Graph placement

`CLIPTextEncode → ApplyConditioningEdit → FluxGuidance → KSampler`.

Produces: CLIP-L pooled gets `strength · pooled_delta` added; T5 sequence gets
`strength · seq_delta` broadcast to every token (so the T5 mean-pool shifts by
`seq_delta`, matching the regression setup in `stage4_regression.py`).

## Producing .npz files

See `src/demographic_pc/build_age_edits.py` in this repo — builds both
`age_ours.npz` (ridge-regression direction) and `age_fluxspace_coarse.npz`
(prompt-pair contrast).
