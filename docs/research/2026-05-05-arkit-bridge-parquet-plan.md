---
status: live
topic: arkit-bridge
---
# arkit-bridge diagnostic parquet plan (2026-05-05)

Replace the scattered `*_compare.json`, `channel_recovery_v3_*.json`,
`teacher_mf*.npz`, `health_summary.txt` files with **four parquet tables** that
together let any future student be evaluated with a single query, and any two
students compared by a `pl.join`.

## Why

- Aggregation across runs currently means glob+JSON-load+per-key dict munging.
- Per-channel queries on the corpus require Python loops over 100K pickles.
- Adding a new student should not require re-running `student_axis_report.py` against a
  fixed file layout — it should be `append student predictions → re-query`.
- Two students differ not in pixel space but in m_f cells and recovered
  blendshapes. Storing those as columns is the natural representation.

## Tables

### `frames.parquet` — per-frame ground truth (build once)

Row = one extracted training/eval frame.

| col | dtype | source |
|---|---|---|
| `take` | uint8 | `data/arkit_bridge_pairs/all/<NAME>_frame_<idx>.pkl` parsing |
| `frame_idx` | uint32 | pkl filename |
| `csv_idx` | uint32 | mapping to original LLF CSV row (for cross-ref) |
| `split` | enum(train, val, test) | takes 4 + 7 = test (locked, see test-split doc) |
| `b_expr` | list<f32>[58] | pkl `b_expr` |
| `head_ypr` | list<f32>[3] | pkl `head_ypr` (if present) |
| `m_f_teacher` | list<f32>[512] | pkl `m_f` (32×16 flattened) |

Source: `data/arkit_bridge_pairs/all/*.pkl` + LLF CSVs.
Estimated size: ~100K rows × ~580 floats ≈ 230 MB raw → 80–120 MB parquet.

### `predictions.parquet` — per-student m_f (append per ckpt)

Row = (ckpt × frame).

| col | dtype |
|---|---|
| `ckpt_tag` | string (e.g. `v3_lam10`, `v4_causal`) |
| `take` | uint8 |
| `frame_idx` | uint32 |
| `m_f_student` | list<f32>[512] |
| `m_f_l2_err` | f32 (vs teacher) |
| `m_f_cos` | f32 (vs teacher) |

Build: `scripts/build_predictions_parquet.py --ckpt <pt>` runs the student forward
on every row in `frames.parquet`, appends. ~1 min per ckpt on 5090.

### `anchors.parquet` — per-anchor neutral baseline (tiny)

Row = one anchor portrait used as PersonaLive reference image.

| col | dtype |
|---|---|
| `anchor_stem` | string (e.g. `asian_m__06_neutral.midframe`) |
| `anchor_path` | string |
| `mp_blendshapes` | list<f32>[52] |
| `mp_landmarks` | list<list<f32>[3]>[478] (optional) |

Build: `scripts/build_anchors_parquet.py` runs mediapipe FaceLandmarker on each
anchor portrait. Trivial.

### `render_metrics.parquet` — per-render closed-loop signal (append per render run)

Row = (run_tag × take × frame × mode).

| col | dtype |
|---|---|
| `run_tag` | string (e.g. `v3_full`, `v4_full`) |
| `take` | uint8 |
| `frame_idx` | uint32 |
| `mode` | enum(`teacher_full`, `bridge`, `personalive_rgb`) |
| `preprocessing` | enum(`perframe`, `nocrop`, `ema`, `n/a`) — driver-side crop strategy (only varies for `personalive_rgb`) |
| `pane` | enum(`right`, `left`, `full`) — for side-by-side mp4s, which half this row scores |
| `anchor_stem` | string (FK to anchors) |
| `mp4_path` | string |
| `mp_blendshapes` | list<f32>[52] |
| `abs_diff_global` | f32 (vs teacher mp4 same frame, NaN for teacher rows) |
| `region_brow` | f32 |
| `region_eye_l` | f32 |
| `region_eye_r` | f32 |
| `region_mouth` | f32 |

`personalive_rgb` rows are the **upstream ceiling**: real motion_encoder driven by raw RGB. See `2026-05-05-llf-take-renders-catalog.md` for the existing 8-take + abc-ablation render set.

Build: `scripts/build_render_metrics_parquet.py --run_tag v3_full` reads each
mp4, runs mediapipe per frame, computes pixel-region diffs against the matching
`teacher_full` mp4 (cache-resolved), appends.

**Teacher rows are reusable across student retrains** — only the bridge rows
need recomputing per student.

## Closed-loop signals derived from the four tables

Single neutral reference `mp(anchor)`. Five derived metrics per (channel, frame,
ckpt):

| signal | formula | meaning |
|---|---|---|
| `delta_input` | `b_expr − mp(anchor)` (52-d slice of b_61 vs anchor) | what the user asked for |
| `delta_teacher` | `mp(teacher_full[t]) − mp(anchor)` | what PersonaLive *can* produce when fed real motion_encoder + the ARKit-derived pose — ARKit-path ceiling |
| `delta_bridge` | `mp(bridge[t]) − mp(anchor)` | what we produced |
| `delta_rgb_in`  | `mp(personalive_rgb.left[t]) − mp(anchor)` | what the camera/driver actually expressed (RGB→mediapipe, no ARKit) |
| `delta_rgb_out` | `mp(personalive_rgb.right[t]) − mp(anchor)` | what PersonaLive produces from RGB end-to-end — **upstream ceiling** |
| `amp_personalive` | `‖delta_rgb_out‖ / ‖delta_rgb_in‖` per channel | PersonaLive's own attenuation (irreducible, RGB-pure) |
| `amp_arkit_path` | `‖delta_teacher‖ / ‖delta_input‖` per channel | additional attenuation from going through ARKit b_61 instead of RGB |
| `amp_student`   | `‖delta_bridge‖ / ‖delta_teacher‖` per channel | **fair student score** — divides out everything that isn't the student |

The current `amp_vs_driving` is `‖bridge‖ / ‖input‖` which conflates the
two attenuations. After the parquet, we report `amp_student` and stop blaming
the student for PersonaLive's irreducible non-injectivity.

## Build order

1. **`anchors.parquet`** — one mediapipe call. Build first, no deps.
2. **`frames.parquet`** — read `data/arkit_bridge_pairs/all/*.pkl`. No render dep.
3. **`predictions.parquet`** for v3_lam10 — student forward over `frames`.
4. **`render_metrics.parquet`** for v3_full — runs once `v3_full` render finishes.
   - Subset `render_metrics_partial.parquet` can be built **now** from existing
     60-frame teacher and bridge mp4s for takes {2, 3, 8} for sanity-check.

## Storage

- Location: `exp_output/arkit_bridge/parquet/` (gitignored, mirrors mp4 cache).
- Symlink to `data/parquet/` for convenience; do not commit.

## Replacements

Once the parquets exist, these become superseded:
- `student_axis_report.py` → re-implement as ~30-line Polars query against the
  three parquets; old version kept as reference.
- `summarize_student_health.py` → `predictions.parquet.group_by('ckpt').agg(...)`.
- `channel_recovery_v3_take*.json` → `render_metrics.parquet.filter(run_tag=...)`.

## Open decisions

- **List-form vs exploded columns** for the 58-d / 52-d / 512-d arrays:
  list-form (chosen) — compact and Polars handles it well. Add a helper
  `with_channel(name) -> pl.Expr` for ergonomic access.
- **Mediapipe model**: keep the same FaceLandmarker we already use in
  `compare_teacher_vs_bridge.py` for consistency. Pin its version (mediapipe
  0.10.20) — the 0.10.21 protobuf bug is the kind of thing that silently
  changes the recovered blendshape distribution.

## Pre-render starter (process what's already on disk)

These exist before the v3_full render finishes; build a partial
`render_metrics.parquet` against them so the schema and pipeline are
exercised:

| mp4 | frames | mode | take |
|---|---|---|---|
| `teacher_full_cache/...take_2...n1200_s2_o0.mp4` | 1200 | teacher_full | 2 |
| `v3_compare/20260505_MySlate_2_teacher_full.mp4` | 60 | teacher_full | 2 |
| `v3_compare/20260505_MySlate_3_teacher_full.mp4` | 60 | teacher_full | 3 |
| `v3_compare/20260505_MySlate_8_teacher_full.mp4` | 60 | teacher_full | 8 |
| `v3_compare/20260505_MySlate_2_bridge.mp4` | 60 | bridge | 2 |
| `v3_compare/20260505_MySlate_3_bridge.mp4` | 60 | bridge | 3 |
| `v3_compare/20260505_MySlate_8_bridge.mp4` | 60 | bridge | 8 |

Plus the duplicate 60-frame teachers under `teacher_rot/` and `teacher_full/`
(redundant, pick one source per take).

**Plus the full RGB-driven baseline set** (already on disk, no render dep) —
documented separately in `2026-05-05-llf-take-renders-catalog.md`:

| mp4 | take | preprocessing | mode | pane=right frames |
|---|---|---|---|---|
| `data/llf-take-renders/take{1..8}__asian_m.mp4` | 1..8 | perframe | personalive_rgb | full take (2028..7440) |
| `data/llf-take-renders/abc/take{2,7}_A_perframe.mp4` | 2,7 | perframe | personalive_rgb | 3600 |
| `data/llf-take-renders/abc/take{2,7}_B_nocrop.mp4`   | 2,7 | nocrop   | personalive_rgb | 3600 |
| `data/llf-take-renders/abc/take{2,7}_C_ema.mp4`      | 2,7 | ema      | personalive_rgb | 3600 |

Side-by-side 1024×512 — slice `[:, :, 512:]` for `pane=right`, `[:, :, :512]` for `pane=left` before mediapipe.

Once the v3_full batch completes (in flight, ~3–4 hr ETA after fix to memory
cap), append the 1200-frame rows for takes {2, 3, 4, 5, 6, 7, 8}, modes
{teacher_full, bridge}.
