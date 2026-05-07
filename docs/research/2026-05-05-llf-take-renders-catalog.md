---
status: live
topic: arkit-bridge
---

# llf-take-renders catalog (2026-05-05)

The set of PersonaLive renders produced *before* the bridge work, when the
driver was raw RGB into PersonaLive's own motion_encoder. These are the
**natural ceiling** for the bridge: any place where bridge mp4 differs from
its matching `personalive_rgb` mp4 is a bridge-side bug, not a PersonaLive
limitation.

Renders live under `data/llf-take-renders/` and are already on disk —
no re-render needed.

## Layout

```
data/llf-take-renders/
├── take{1..8}__asian_m.mp4          # full take, perframe crop (default), 1024×512 @ 60 fps
└── abc/
    ├── take{2,7}_A_perframe.mp4     # 60 s slice, --crop-strategy perframe
    ├── take{2,7}_B_nocrop.mp4       # 60 s slice, --crop-strategy nocrop
    └── take{2,7}_C_ema.mp4          # 60 s slice, --crop-strategy ema (causal EMA + forehead bias)
```

All mp4s are SIDE-BY-SIDE: **left half (0..512) = driver crop**, **right half (512..1024) = PersonaLive output**. Mediapipe blendshape extraction must run on the right half.

## Render parameters

Common across all rows:
- driver: `data/llf-takes/20260505_MySlate_<N>/<NAME>.MOV`
- anchor: `data/llf-phase2/asian_m__06_neutral.midframe.png` (same as bridge runs)
- model: PersonaLive offline (real motion_encoder, real pose_guider), 4-frame chunks, SDPA
- output: 512×512 right half, 60 fps
- script: `~/w/PersonaLive/scripts/render_take.py`

Per-row variation: `crop_strategy` (preprocessing flavor) and `start_frame`/`max_frames` for the abc slices.

## Catalog table

| file | take | preprocessing | start_frame | n_frames | duration | face_mesh fallback |
|---|---|---|---|---|---|---|
| `take1__asian_m.mp4`     | 1 | perframe | 0 | 6384 | 106.4 s | 10 (0.2 %) |
| `take2__asian_m.mp4`     | 2 | perframe | 0 | 7440 | 124.0 s | 42 (0.6 %) |
| `take3__asian_m.mp4`     | 3 | perframe | 0 | 3248 | 54.1 s  | 30 (0.9 %) |
| `take4__asian_m.mp4`     | 4 | perframe | 0 | 2028 | 33.8 s  | 856 (41.6 %) ⚠⚠ |
| `take5__asian_m.mp4`     | 5 | perframe | 0 | 7336 | 122.3 s | 88 (1.2 %) |
| `take6__asian_m.mp4`     | 6 | perframe | 0 | 5208 | 86.8 s  | 0 (0 %) |
| `take7__asian_m.mp4`     | 7 | perframe | 0 | 6432 | 107.2 s | 0 (0 %) |
| `take8__asian_m.mp4`     | 8 | perframe | 0 | 2784 | 46.4 s  | 324 (11.6 %) ⚠ |
| `abc/take2_A_perframe.mp4` | 2 | perframe | 0 | 3600 | 60.0 s | — |
| `abc/take2_B_nocrop.mp4`   | 2 | nocrop   | 0 | 3600 | 60.0 s | — |
| `abc/take2_C_ema.mp4`      | 2 | ema      | 0 | 3600 | 60.0 s | — |
| `abc/take7_A_perframe.mp4` | 7 | perframe | 0 | 3600 | 60.0 s | — |
| `abc/take7_B_nocrop.mp4`   | 7 | nocrop   | 0 | 3600 | 60.0 s | — |
| `abc/take7_C_ema.mp4`      | 7 | ema      | 0 | 3600 | 60.0 s | — |

Note: the `take{N}__asian_m.mp4` fallback rates are stress-test signal — takes 4 & 8 the driver pose escaped face_mesh frequently, the cropper held the last good box, and that "frozen crop" is *itself* a low-end-of-the-knob preprocessing variant. (Cf. open question in `2026-05-05-personalive-take-render-observations.md`.)

## Implication for the parquet

Adds a fourth `mode`-equivalent and a new `preprocessing` column to `render_metrics.parquet`:

| col | dtype | values |
|---|---|---|
| `mode` | enum | `teacher_full` (real motion_encoder, ARKit-driven via apply_bridge), `bridge` (student m_f, ARKit-driven), `personalive_rgb` (real motion_encoder, RGB-driven, full PersonaLive offline), `none` (raw driver crop, no model) |
| `preprocessing` | enum | `perframe`, `nocrop`, `ema`, `n/a` |
| `pane` | enum | `left` (driver), `right` (output), `full` (single-pane mp4s like teacher_full / bridge) |

For `mode=personalive_rgb` we extract two rows per frame: `pane=right` (model output, contributes blendshapes + region diffs) and `pane=left` (driver's actual mediapipe blendshapes — the "ground truth" the bridge is asking PersonaLive to mimic via ARKit b_61). The left-pane signal lets us close the loop between *what the camera saw* and *what PersonaLive produced* without going through ARKit at all — orthogonal anchor for `delta_input`.

## Closed-loop ceiling

The fair ceiling for `amp_student` is now:

```
amp_student = ‖delta_bridge‖ / ‖delta_personalive_rgb_right‖   (per channel)
```

with `delta_personalive_rgb_right = mp(personalive_rgb.right[t]) − mp(anchor)`. This divides out **both** PersonaLive's irreducible attenuation **and** the ARKit b_61 → motion_encoder bridge's task itself, leaving only the student's contribution.

The earlier `delta_teacher = mp(teacher_full[t]) − mp(anchor)` is still useful — it's the per-frame ARKit-driven ceiling — but `personalive_rgb` is the *upstream* ceiling and is the one we should report in any publication.

## Build implications

- `render_metrics.parquet` builder must consume side-by-side mp4s (slice right half before mediapipe).
- A `preprocessing=perframe` baseline exists for **all 8 takes** at full duration. No re-render required to populate the upstream ceiling for the diag set (2,3,5,6,8) and the held-out test set (4,7).
- The abc slices (60 s, takes 2 & 7 only) give us a 3-way preprocessing comparison row for free; we don't need to re-render with C ema preprocessing on the bridge side just to measure — unless the **abc verdict** itself becomes the contribution we're publishing, in which case bridge-side perframe vs ema becomes a separate experiment.

## Sources
- `~/w/PersonaLive/scripts/render_take.py` produced the mp4s
- `2026-05-05-personalive-take-render-observations.md` — observations + abc verdict
- `2026-05-05-arkit-bridge-parquet-plan.md` — schema being amended
