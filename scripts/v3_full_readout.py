"""Print a compact 7-take readout from v3_full compare jsons.

Reads `exp_output/arkit_bridge/render/v3_full/<TAKE>_compare.json` and
prints per-take and aggregate stats. Cheap; no parquet dependency.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="exp_output/arkit_bridge/render/v3_full")
    args = ap.parse_args()
    d = Path(args.dir)
    files = sorted(d.glob("*_compare.json"))
    print(f"# v3_full readout — {len(files)} takes\n")
    print(f"{'take':<28} {'n':>5} {'absMean':>8} {'absP90':>8} "
          f"{'brow_p90':>10} {'eyeL_p90':>10} {'eyeR_p90':>10} {'mouth_p90':>10}")

    rows = []
    for f in files:
        s = json.loads(f.read_text())
        rows.append({
            "take":     s["take"],
            "n":        s["n_frames_compared"],
            "abs_mean": s["mean_abs_per_frame"]["mean"],
            "abs_p90":  s["mean_abs_per_frame"]["p90"],
            "brow":     s["region_energy"]["brow"]["p90"],
            "eye_l":    s["region_energy"]["eye_l"]["p90"],
            "eye_r":    s["region_energy"]["eye_r"]["p90"],
            "mouth":    s["region_energy"]["mouth"]["p90"],
        })

    for r in rows:
        print(f"{r['take']:<28} {r['n']:>5} "
              f"{r['abs_mean']:>8.2f} {r['abs_p90']:>8.2f} "
              f"{r['brow']:>10.2f} {r['eye_l']:>10.2f} "
              f"{r['eye_r']:>10.2f} {r['mouth']:>10.2f}")

    print()
    diag = [r for r in rows if int(r["take"].split("_")[-1]) in {2, 3, 5, 6, 8}]
    test = [r for r in rows if int(r["take"].split("_")[-1]) in {4, 7}]
    for label, group in [("DIAG (2,3,5,6,8)", diag), ("TEST (4,7)", test), ("ALL", rows)]:
        if not group: continue
        for k in ("abs_mean", "abs_p90", "brow", "eye_l", "eye_r", "mouth"):
            vals = [r[k] for r in group]
            print(f"  {label:<22} {k:<10} mean={statistics.mean(vals):>7.2f}  med={statistics.median(vals):>7.2f}")
        print()


if __name__ == "__main__":
    main()
