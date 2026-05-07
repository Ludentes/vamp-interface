"""Per-axis head-amp ratio: bridge (student_v2_120k) vs teacher_full.

For each (take, axis) pair, compute the 95th-percentile of |angle| on
teacher and on bridge; report ratio.

**Gate is wide because head under-rotation vs teacher_full is an
accepted limitation of the bridge** — the student was trained against
this, hit a structural floor (ratio_mean 0.0114), and rendered
amplitude shortfall on high-motion takes (worst seen: take 8 pitch
0.585, roll 0.507) is the known shipped state, not a regression. The
gate exists to catch *new* regressions vs that shipped baseline (e.g.
the OLD config that crushed pitch to 13%, or a stale parquet pointing
at pre-fix renders).

  HARD FAIL : ratio < 0.45 or > 1.55  (well below shipped state)
  WARN      : ratio < 0.70 or > 1.30  (attention worth paying)
  PASS      : otherwise

Schema: render_metrics.parquet uses a single `ypr` numpy-array column
(yaw, pitch, roll), selects bridge/teacher by `(run_tag, mode)`. Bridge
= run_tag 'v2_120k' (mode 'bridge'); teacher = run_tag
'teacher_full_cache' (mode 'teacher_full').
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

PARQUET = Path("exp_output/arkit_bridge/parquet/render_metrics.parquet")
OUT = Path("exp_output/arkit_bridge/diagnostics/head_attenuation_2026-05-06.json")

BRIDGE_RUN_TAG = "v2_120k"
TEACHER_RUN_TAG = "teacher_full_cache"
AXES = ("yaw", "pitch", "roll")


def axis_abs(sub: pd.DataFrame, axis_idx: int) -> np.ndarray:
    arr = np.stack(sub["ypr"].to_numpy())
    return np.abs(arr[:, axis_idx])


def main():
    df = pd.read_parquet(PARQUET)
    bridge_df = df[(df["run_tag"] == BRIDGE_RUN_TAG) & (df["mode"] == "bridge")]
    teacher_df = df[(df["run_tag"] == TEACHER_RUN_TAG) & (df["mode"] == "teacher_full")]

    common_takes = sorted(set(bridge_df["take"].unique()) & set(teacher_df["take"].unique()))
    rows = []
    for take in common_takes:
        b_sub = bridge_df[bridge_df["take"] == take]
        t_sub = teacher_df[teacher_df["take"] == take]
        if len(b_sub) == 0 or len(t_sub) == 0:
            continue
        for axis_idx, axis in enumerate(AXES):
            teacher = axis_abs(t_sub, axis_idx)
            bridge = axis_abs(b_sub, axis_idx)
            tp = float(np.nanpercentile(teacher, 95))
            bp = float(np.nanpercentile(bridge, 95))
            ratio = bp / tp if tp > 1e-6 else float("nan")
            rows.append({
                "take": int(take),
                "axis": axis,
                "teacher_p95": tp,
                "bridge_p95": bp,
                "ratio": ratio,
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "bridge_run_tag": BRIDGE_RUN_TAG,
        "teacher_run_tag": TEACHER_RUN_TAG,
        "rows": rows,
    }, indent=2))

    HARD_LO, HARD_HI = 0.45, 1.55
    WARN_LO, WARN_HI = 0.70, 1.30
    bad = [r for r in rows if not (HARD_LO <= r["ratio"] <= HARD_HI)]
    warn = [r for r in rows if (HARD_LO <= r["ratio"] <= HARD_HI)
                            and not (WARN_LO <= r["ratio"] <= WARN_HI)]
    for r in rows:
        if not (HARD_LO <= r["ratio"] <= HARD_HI):
            flag = "  <-- HARD FAIL"
        elif not (WARN_LO <= r["ratio"] <= WARN_HI):
            flag = "  <-- WARN (known under-rotation regime)"
        else:
            flag = ""
        print(f"  take={r['take']:>2}  {r['axis']:6s}  "
              f"teacher_p95={r['teacher_p95']:.4f}  bridge_p95={r['bridge_p95']:.4f}  "
              f"ratio={r['ratio']:.3f}{flag}")
    if bad:
        print(f"\nFAIL: {len(bad)}/{len(rows)} axis-take pairs out of "
              f"[{HARD_LO}, {HARD_HI}]")
        sys.exit(1)
    if warn:
        print(f"\nPASS with {len(warn)}/{len(rows)} pairs in WARN band "
              f"(<{WARN_LO} or >{WARN_HI}; expected on high-motion takes)")
    else:
        print(f"\nPASS: all {len(rows)} pairs within [{WARN_LO}, {WARN_HI}]")


if __name__ == "__main__":
    main()
