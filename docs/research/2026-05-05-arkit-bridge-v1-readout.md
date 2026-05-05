---
status: live
topic: neural-deformation-control
---

# ARKit→PersonaLive bridge v1 — readout (2026-05-05)

End-of-cycle write-up for the v1 plan
(`2026-05-05-arkit-bridge-v1-plan.md`). Captures what we built, what the
viability gates said, what the renders actually showed, and what we'd
change in v2. Read this first on resume.

## What was built

| Task | Output |
|---|---|
| 1–2 | Closed-form ARKit Euler → LivePortrait `k_d` (`src/arkit_bridge/closed_form_pose.py`) — calibrated `EULER_SIGNS=(+1,-1,-1)` |
| 3 | Teacher loaders for `motion_encoder`, `motion_extractor`, `pose_guider` from PersonaLive |
| 4 | `MotEncoderStudent` (58 → 256³ SiLU MLP → 32×16, zero-init head, 278 K params) |
| 5–6 | Pair extraction over 7 takes (16 522 pairs at stride=2) + per-frame MSE distill, AdamW, eval-every-2k with R²-mask + ratio dual best ckpts |
| 7 | Tier 1/2 viability eval + per-channel sensitivity sweep |
| 8 | Trained 30 K steps, batch 128, lr 5e-4 — final ratio=0.00664, R²≥0.7=0.979 (both Tier-1 gates clear with margin) |
| 9 | 8-way Euler sign calibration (takes 2/5/7) — agreement on (+1,-1,-1); take-5 yaw was a sub-noise tie |
| 10 | Drop-in PersonaLive smoke render: monkey-patches `pose_encoder.{interpolate_kps_online,get_kps}` and `motion_encoder.forward` (T==1 keeps real path; T≥2 uses student) |
| 11 | Per-component perf bench (RTX 5090 fp16 T=4): bridge replaces 10.82 ms with 0.37 ms — 28.9× speedup; UNet now the firm bottleneck |
| 12 | This readout |

## Viability gates

**Tier 1 (distill fidelity, locked thresholds in `2026-05-05-arkit-bridge-v1-viability.md`)**

| metric | gate | measured | margin |
|---|---|---|---|
| ratio_mean | < 0.10 | **0.00664** | 15× |
| R²≥0.7 fraction | ≥ 0.80 | **0.979** | +0.18 |

**Tier 2 (per-input-category)**: passed for all 7 ARKit categories with
ratios all under 0.05; see `runs/student_v1/eval_step030000.json`.

**Tier 3 (perceptual)** — *not gated by Tier 1*. Three smoke renders
analysed via `scripts/diagnose_render_expression.py`:

| Take | User verdict | ArcFace cos | LPIPS r vs bnorm_expr | bs_cos render↔CSV | bs_cos render↔driving (mp) | ypr_err render→CSV (rad) |
|---|---|---|---|---|---|---|
| 2 | "decent" | **0.918** | +0.54 | 0.318 | 0.564 | 0.328 |
| 3 | flat / lifeless | 0.820 | **−0.64** | **0.252** (lowest) | 0.498 | 0.223 |
| 8 | artifacts | **0.732** (worst) | +0.92 | 0.349 | 0.591 | 0.491 |

Sanity floor: mediapipe-vs-ARKit ypr disagreement on driving frames =
0.29–0.49 rad (16–28°). Render ypr_err is at or below this floor on
takes 2 and 3; take 8 marginally above. **Pose path is fine.**

## Failure mode (confirmed)

**Tier 1 cleared, Tier 3 partially failed.** The student passes per-cell
R² because most cells track the dominant neutral cluster well, but
attenuates the high-amplitude tail. Output-side renders confirm:

- m_f std ratio (`mf_attenuation_v1.json`): median 0.91, p10 0.87 —
  **9–13 % shrinkage uniform across cells**.
- Tail recovery on |teacher_z|>2 samples: median 0.84, p10 0.77 —
  **16–23 % attenuation at the cells that carry expression amplitude**.
- LPIPS r(bnorm_expr) on take 3 = **−0.64** with r(head_ypr) = +0.03 —
  the negative correlation is driven by **expression amplitude itself**,
  not by extreme head pose. **Falsifies hypothesis #2** (closed-form pose
  failure at extreme angles); confirms hypothesis #1 (student tail
  attenuation).
- LPIPS r(bnorm_expr) on take 2 = +0.54 (in-distribution, faithful) and
  on take 8 = +0.92 (in-distribution amplitude-wise but identity drift
  drives most of the LPIPS variance).

**Take 8 is a separate failure mode**: low ArcFace (0.732) with
relatively high blendshape recovery (0.349). Identity drifts despite
expression reaching pixels. Likely the catalogued ghost-glasses /
clothing-edge artifacts (see
`2026-05-05-personalive-take-render-observations.md`) plus the take's
11.6 % source face_mesh fallback. **Out of scope for the bridge** —
fixing it requires PersonaLive-side work.

## Decision

**Bridge v1 is viable as a baseline** for moderate-amplitude inputs
(take 2 regime). It is **not viable** as a drop-in replacement for the
full motion_encoder until the tail-attenuation is fixed. Numerical
quality is good (R²=0.98), perf is excellent (28.9× on the replaced
components, no UNet-side cost), but the loss objective is the wrong
one — see v2 redesign below.

Recommendation: **proceed to v2 distill** with a tail-aware loss before
any wider use. Do **not** retrain on more data alone; the architecture
and corpus are already capable, the optimisation target was the issue.

## Artifacts to keep

- `runs/student_v1/student_best.pt` — best ratio_mean ckpt
- `runs/student_v1/student_best_r2.pt` — best R²-mask ckpt
- `runs/student_v1/eval_step030000.json` — final Tier 1+2 numbers
- `exp_output/arkit_bridge/calibration/myslate_{2,5,7}.json` — Euler sign votes
- `exp_output/arkit_bridge/diagnostics/mf_attenuation_v1.{json,cells.npz}` — m_f-side attenuation
- `exp_output/arkit_bridge/diagnostics/render_expression_take{2,3,8}_final.json` — output-side round-trip + ArcFace
- `exp_output/arkit_bridge/render/smoke/take{2..8}_bridge_v1.mp4` — 60-frame stride-4 smoke
- `exp_output/arkit_bridge/render/full/take{2..8}_bridge_v1_full.mp4` — full-length stride-2 renders (some
  takes had a tiny final segment fail on `max() arg empty sequence`; the produced concat is the bulk of
  the take and visually representative)
- `exp_output/perf/personalive-component-budget.json` — RTX 5090 fp16 T=4 budget

Loss/training redesign: `2026-05-05-arkit-bridge-v2-loss-redesign.md`.

## Cross-references

- Plan: `2026-05-05-arkit-bridge-v1-plan.md`
- Viability framework: `2026-05-05-arkit-bridge-v1-viability.md`
- Architecture notes: `2026-05-05-personalive-architecture-notes.md`
- LLF takes catalog: `2026-05-05-llf-takes-catalog.md`
- Render observations: `2026-05-05-personalive-take-render-observations.md`
- Topic index: `_topics/neural-deformation-control.md`
