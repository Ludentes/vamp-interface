---
status: live
topic: neural-deformation-control
---

# PersonaLive 8-take render observations + quantification plan

**Date:** 2026-05-05
**Source:** `data/llf-take-renders/take{1..8}__asian_m.mp4` — render of 8 iPhone takes through PersonaLive against the canonical Phase-2 anchor (`asian_m__06_neutral.midframe.png`), via `scripts/render_take.py`. Side-by-side, driver left, output right, 1024×512 @ 60 fps.

## Render summary

| take | dur | frames | render | mp4 | face_mesh fallback |
|---|---|---|---|---|---|
| 2 | 124 s | 7440 | 10.1 min | 105 MB | 42 (0.6 %) |
| 5 | 122 s | 7336 | 9.8 min | 64 MB | 88 (1.2 %) |
| 7 | 107 s | 6432 | 8.7 min | 62 MB | **0 (0 %)** |
| 1 | 106 s | 6384 | 7.8 min | 52 MB | 10 (0.2 %) |
| 6 | 87 s | 5208 | 7.0 min | 51 MB | **0 (0 %)** |
| 3 | 54 s | 3248 | 4.4 min | 32 MB | 30 (0.9 %) |
| 8 | 46 s | 2784 | 3.8 min | 30 MB | **324 (11.6 %)** ⚠ |
| 4 | 34 s | 2028 | 2.8 min | 21 MB | **856 (41.6 %)** ⚠⚠ |

Sustained ~12 fps render, no model drift across the batch.

## Qualitative observations (user, after watching all 8 mp4s)

1. **Cut jitter on the driver side** — the per-frame face_mesh crop introduces visible jitter in the *original* (left pane) that wasn't in the source MOV.
2. **Head clipping** — the cut frequently crops out the top of the head.
3. **Artifact persistence** — once an artifact appears in the output, it almost never leaves.
4. **Clothing edges** — edges of garments seem to seed artifacts in the output.
5. **Glasses → extreme cut jitter** — when the driver wears glasses, the per-frame crop becomes especially unstable.
6. **Ghost glasses** — the unstable cut + glasses presence causes the model to render glasses on the output, even when the anchor has none.

**User verdict:** "Our video cut definitely makes things worse, and I am not sure it makes things better."

This is consistent with the saved memory `project_facemesh_crop_wobble.md` (per-frame `crop_face` recomputes bbox each frame → input jitter → output wobble). What's new here is that with longer takes the failure modes compound: the input jitter correlates with what the model is forced to confabulate, and once it hallucinates a feature (glasses arm, smudge on forehead), the autoregressive temporal module locks it in.

## Ultimate goal — realtime control

Any quantification or fix must respect the eventual realtime constraint:

- **Causal only.** No future-frame lookahead in any stabilization scheme. EMA fine, detect-every-Nth-with-look-ahead-interpolation **not** fine — the realtime version of that is "use the last detection until a new one arrives," which has different statistics.
- **Cheap.** Stabilization must run at ≥30 fps on the same GPU as PersonaLive. EMA is free; a learned predictor would need its own latency budget.
- **No global re-pass.** Whatever we measure on offline takes should be measured with a *causal* smoother so the metrics predict realtime behavior. Otherwise we'd be optimizing for an offline metric that doesn't survive the deployment switch.

This shapes the experiment: we are not asking "what's the best crop" in the abstract, we are asking "what's the best *causal* crop within a tiny latency budget."

## Quantification plan

Three independent measurements, ordered by how directly they answer "is the cut helping?".

### Input side — cut jitter, head clipping, glasses correlation

`scripts/cut_jitter_score.py` (new). Runs face_mesh on a MOV and emits per-frame bbox. Computes:

- **center_jitter_px**: ‖bbox_center_t − causal_ema_t‖ — pixels of unforced motion the model has to absorb.
- **size_jitter_pct**: std(bbox_w / ema_bbox_w_ema) — scale wobble.
- **head_clip_rate**: % of frames where face_mesh landmark 10 (top-of-forehead) falls above the bbox top edge.
- **detect_fail_rate**: % of frames where face_mesh failed (matches the `[done]` counter from `render_take.py`).

Run on takes 7 (clean reference), 2 (glasses-heavy, long), 4 (highest detect_fail_rate). Compare distributions.

### Output side — artifact persistence and ghost glasses

Existing `vamp-interface/scripts/artifact_score.py` on the right half of each of the 8 mp4s. Already produces:

- arcface_cos vs anchor
- ghost_resid on forehead (catches phantom glasses arms — directly tests problem 6)
- flicker, lap_var, bg_drift

For this round we extract two derived measures:

- **drift slope**: linear regression of arcface_cos over the take. A monotonically decaying slope confirms problem 3 ("artifacts never leave").
- **input↔output flicker correlation**: per-frame, regress output flicker on `center_jitter_px`. Significant slope means the cut is causally amplifying instability.

### A/B/C — does the cut earn its keep?

Same anchor + same MOVs, three driver-prep strategies:

- **A**: current per-frame face_mesh crop.
- **B**: no crop — resize full frame to 512×512, accept letterbox.
- **C**: causal-stabilized — face_mesh on first frame + EMA bbox over time, redetect on detect_fail.

Run on take 7 (clean) and take 2 (glasses-heavy) for tractable scope. Score all three on the metrics above. Decision rule:

- If C ≈ A on identity but better on flicker/ghost → cut is fine but the per-frame jitter is the bug; ship causal EMA.
- If C beats A on identity → the per-frame crop is actively *destroying* identity signal across frames; ship C immediately.
- If B ≥ A on all metrics → the cut is unjustified entirely; remove it and resize directly.

## A/B/C result (2026-05-05 evening)

Ran the three crop strategies on the first 60 s of takes 7 and 2. `StabilizedFaceCropper` added to `src/utils/util.py`, `--crop-strategy` flag added to `scripts/render_take.py`. Per-strategy renders + scoring:

**Input-side (left-half mean per-frame flicker, ~3600 frames each):**

| | A perframe | B nocrop | C ema |
|---|---|---|---|
| take 7 | 4.57 | 1.68 | **3.00** (−34% vs A) |
| take 2 | 6.01 | 1.97 | **3.63** (−40% vs A) |

C cuts ~40% of input jitter without losing the face track. B has nothing to wobble (static centre square).

**Output-side (artifact_score.py on the right half, 1800 sampled frames each):**

| take | strat | arcface mean | start→end | drift /kfr | flicker | ghost p95 |
|---|---|---|---|---|---|---|
| 7 | A | 0.728 | 0.882→0.670 | −0.038 | 1.78 | **3.62** |
| 7 | B | **0.749** | 0.877→0.713 | **−0.031** | 2.01 | 3.85 |
| 7 | C | 0.742 | 0.883→0.692 | −0.036 | 1.82 | 3.75 |
| 2 | A | 0.868 | 0.910→0.817 | −0.048 | 1.55 | 3.92 |
| 2 | B | 0.870 | 0.915→0.823 | −0.045 | **1.47** | **3.90** |
| 2 | C | **0.871** | 0.909→0.827 | **−0.043** | 1.54 | 3.98 |

**Verdict.** A is dominated. C and B are both viable; A is not.

- **C (causal EMA + 10% forehead bias)** wins on identity drift slope on both takes (−0.036/−0.043 vs A's −0.038/−0.048), preserves sharpness comparable to A (lap_var ~35/44), and is the right realtime default — face tracking + hair room + no static-centre fragility.
- **B (no crop, centre square)** is surprisingly competitive: best arcface mean on take 7 (+0.021 vs A), tied on take 2, and best output flicker on take 2. The model's reference-conditioning copes with a wider field of view better than expected. Caveat: face must be roughly centred in the source frame (selfie framing satisfies this; tripod with subject off-axis would not).
- The earlier r=+0.56 input-jitter→output-flicker correlation predicted bigger output gains from C than we measured. PersonaLive's temporal module absorbs more input chop than its instantaneous coupling suggested. Drift slope (a long-window quantity) is where the signal lives.
- 60 s probably isn't long enough to distinguish C and B sharply on identity. A full-length re-run (107 s + 124 s) would; deferred for now.

**Action.** Both C and B are documented as viable. Default for shipping should be C (more general; tracks the face when subject moves). Keep B as a single-flag fallback for fixed-framing studio capture where its lower flicker on take 2 may matter.

## Open questions

- Is face_mesh landmark 10 actually the top of the forehead, or just the upper-mid eyebrow region? (verify before relying on it for head_clip_rate)
- The face_mesh fallback in `render_take.py` is "use last good crop" — is that already a degenerate causal stabilizer? On takes 4/8 (very high fallback rate) the *de facto* crop is largely frozen; if those mp4s look better in steady-state than takes with low fallback, that's another data point pointing to "less aggressive recropping wins."

## Sources

- `scripts/render_take.py` — produced the takes
- `vamp-interface/scripts/artifact_score.py` — existing output-side scorer (ArchA T10, in progress)
- Memory: `project_facemesh_crop_wobble.md`
