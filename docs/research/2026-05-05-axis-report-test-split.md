---
status: live
topic: arkit-bridge
---
# axis-report test split (2026-05-05)

Locked split for student retraining cycles, so the axis report does not become an overfit target.

## takes

| take | rows | role |
|---|---|---|
| 1 | 9578  | **DROPPED** (no Live Link Face CSV — only frame_log) |
| 2 | 14881 | diag |
| 3 | 6461  | diag |
| 4 | 2877  | **HELDOUT TEST** (shortest) |
| 5 | 14678 | diag — **rendered on Windows 3090** |
| 6 | 10417 | diag |
| 7 | 12869 | **HELDOUT TEST** (medium-long) |
| 8 | 5383  | diag |

**Diagnostic set** (used in `axis_report.json`): 2, 3, 5, 6, 8.
**Held-out test set** (untouched until final retrain comparison): 4, 7.

## render params (fixed across all runs for comparability)

- `--n_frames 1200 --stride 2 --start_frame 0`
- anchor: `data/llf-phase2/asian_m__06_neutral.midframe.png`
- modes: `teacher_full` AND `bridge`

## existing m_f corpora (no re-render needed for m_f-space diagnostics)

- `data/arkit_bridge_pairs/all/` — full training corpus, per-frame `(b_expr, m_f)` pairs (63 MB)
- `data/arkit_bridge_pairs/holdout_v3/` — 3.3 MB held-out slice
- `exp_output/arkit_bridge/teacher_mf{,_rot}/take{2,3,8}_teacher_mf.npz` — full driving_mf traces from 60-frame compare runs
- New: every cache-bound `teacher_full` render now dumps a sidecar `.mf.npz` next to the mp4 (added 2026-05-05). One-time cost; reused across student retrains.

## anchor diversity

Phase 1 uses one anchor only. Phase 2 (after first retrain shows improvement) renders 2-3 additional anchors on the diag set to disambiguate student-side failures from anchor-coupling.

## comparability rules

- Same `--n_frames`, `--stride`, `--start_frame`, anchor and seed across runs
- Rebuild `axis_report.json` with the same `student_axis_report.py` invocation pattern
- `diff_axis_reports.py old_report new_report` is the canonical comparison
- Test takes (4, 7) are NOT included in diag axis report; they are scored separately and only consulted at end-of-cycle to confirm gains generalize
