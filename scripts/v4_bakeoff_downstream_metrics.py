"""Aggregate per-tag rendered-quality metrics from render_metrics.parquet.

Compares each `bridge` run-tag (v2_120k, v4a, v4b, v4c, ...) to the
teacher_full reference on the same takes. Emits five rendered-side
metrics that escape offline `ratio_mean`:

  bs_cos_to_teacher    — mean cosine(bridge_blendshapes_t,
                                     teacher_blendshapes_t)
  bs_amp_ratio         — ‖bridge_bs - bs_anchor‖ / ‖teacher_bs - bs_anchor‖
                          per frame, then median over frames. <1 = damped,
                          >1 = exaggerated.
  ypr_l1_err           — mean |bridge_ypr - teacher_ypr| (radians)
  region_energy_diff   — mean(|bridge_pixel - teacher_pixel|) over four
                          face regions, summed
  bs_temporal_jitter   — mean ‖bs_t - bs_{t-1}‖ for bridge — direct
                          measure of the wobble that offline ratio_mean
                          misses

Usage:
  python scripts/v4_bakeoff_downstream_metrics.py \
    --parquet exp_output/arkit_bridge/parquet/render_metrics.parquet \
    --out exp_output/arkit_bridge/parquet/v4_bakeoff_downstream.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


TAGS = ["v2_120k", "v4a", "v4b", "v4c"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parquet",
        default="exp_output/arkit_bridge/parquet/render_metrics.parquet",
    )
    ap.add_argument(
        "--out",
        default="exp_output/arkit_bridge/parquet/v4_bakeoff_downstream.json",
    )
    ap.add_argument("--tags", nargs="+", default=TAGS)
    args = ap.parse_args()

    df = pl.read_parquet(args.parquet)

    # Anchor blendshape baseline = first teacher_full frame per take. Used
    # to compute "amplitude" of expression deltas.
    teacher = df.filter(pl.col("mode") == "teacher_full")
    bridge = df.filter(pl.col("mode") == "bridge")

    out: dict[str, dict] = {}
    for tag in args.tags:
        b_tag = bridge.filter(pl.col("run_tag") == tag)
        if b_tag.height == 0:
            out[tag] = {"error": "no bridge rows"}
            continue
        per_take_rows = []
        for take in sorted(b_tag["take"].unique().to_list()):
            b_take = b_tag.filter(pl.col("take") == take).sort("mp4_frame_idx")
            t_take = teacher.filter(pl.col("take") == take).sort(
                "mp4_frame_idx")
            if t_take.height == 0:
                continue
            n = min(b_take.height, t_take.height)
            if n < 5:
                continue
            b_bs = np.array(
                b_take["mp_blendshapes"].to_list()[:n], dtype=np.float32)
            t_bs = np.array(
                t_take["mp_blendshapes"].to_list()[:n], dtype=np.float32)
            b_ypr = np.array(b_take["ypr"].to_list()[:n], dtype=np.float32)
            t_ypr = np.array(t_take["ypr"].to_list()[:n], dtype=np.float32)

            # Drop frames where either side has no detection (zero vector)
            valid = (np.linalg.norm(b_bs, axis=1) > 0) & (
                np.linalg.norm(t_bs, axis=1) > 0)
            if valid.sum() < 5:
                continue
            b_bs = b_bs[valid]; t_bs = t_bs[valid]
            b_ypr = b_ypr[valid]; t_ypr = t_ypr[valid]

            # bs_cos_to_teacher
            num = (b_bs * t_bs).sum(axis=1)
            den = (np.linalg.norm(b_bs, axis=1)
                   * np.linalg.norm(t_bs, axis=1) + 1e-9)
            bs_cos = (num / den).mean()

            # bs_amp_ratio: anchor = mean of first 5 teacher frames
            anchor_bs = t_bs[:5].mean(axis=0)
            t_amp = np.linalg.norm(t_bs - anchor_bs, axis=1)
            b_amp = np.linalg.norm(b_bs - anchor_bs, axis=1)
            ok = t_amp > 1e-3
            amp_ratio = float(np.median(b_amp[ok] / t_amp[ok])) if ok.any() \
                else float("nan")

            # ypr_l1_err
            ypr_l1 = float(np.mean(np.abs(b_ypr - t_ypr)))

            # region energy: read region_* columns averaged over frames
            region_cols = [
                "region_brow", "region_eye_l", "region_eye_r", "region_mouth",
            ]
            region_vals = []
            for c in region_cols:
                m = b_take[c].drop_nulls().mean()
                region_vals.append(float(m) if m is not None  # type: ignore[arg-type]
                                   else 0.0)
            region_energy_sum = float(np.sum(region_vals))

            # bs_temporal_jitter: mean ‖Δbs‖ frame-to-frame on bridge side
            jitter = float(np.mean(np.linalg.norm(np.diff(b_bs, axis=0),
                                                  axis=1)))

            per_take_rows.append({
                "take": int(take),
                "n_frames": int(valid.sum()),
                "bs_cos_to_teacher": float(bs_cos),
                "bs_amp_ratio": amp_ratio,
                "ypr_l1_err": ypr_l1,
                "region_energy_diff": region_energy_sum,
                "bs_temporal_jitter": jitter,
            })

        if not per_take_rows:
            out[tag] = {"error": "no usable takes"}
            continue
        agg = {}
        for k in [
            "bs_cos_to_teacher", "bs_amp_ratio", "ypr_l1_err",
            "region_energy_diff", "bs_temporal_jitter",
        ]:
            vals = np.array([r[k] for r in per_take_rows
                             if not np.isnan(r[k])])
            if len(vals) == 0:
                agg[k] = None
            else:
                agg[k] = {
                    "mean": float(vals.mean()),
                    "median": float(np.median(vals)),
                }
        out[tag] = {"per_take": per_take_rows, "aggregate": agg}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # Print scorecard
    cols = [
        ("bs_cos_to_teacher", "BS cos↑"),
        ("bs_amp_ratio", "amp ratio (1=match)"),
        ("ypr_l1_err", "ypr L1 (rad)↓"),
        ("region_energy_diff", "region E diff↓"),
        ("bs_temporal_jitter", "BS jitter↓"),
    ]
    print(f"\n{'tag':10s}  " + "  ".join(f"{label:>20s}" for _, label in cols))
    for tag in args.tags:
        if tag not in out or "aggregate" not in out[tag]:
            print(f"{tag:10s}  (no data)")
            continue
        a = out[tag]["aggregate"]
        cells = []
        for key, _ in cols:
            v = a.get(key)
            cells.append(f"{v['median']:>20.4f}" if v is not None else
                         f"{'-':>20s}")
        print(f"{tag:10s}  " + "  ".join(cells))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
