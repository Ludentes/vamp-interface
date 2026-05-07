"""Diff two student axis_report.json files: which axes improved / regressed.

Usage:
  python scripts/diff_axis_reports.py --old runs/student_v3_lam10/axis_report.json \
                                     --new runs/student_v4_xxx/axis_report.json
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    o = json.loads(Path(args.old).read_text())
    n = json.loads(Path(args.new).read_text())

    print(f"old: {o['run_tag']}   new: {n['run_tag']}")
    print()
    print("=== summary ===")
    for k in ["score", "amp_geomean", "amp_pass_frac", "r_pass_frac",
              "r2_ge_0p7_frac", "region_p90_mean_per255"]:
        ov, nv = o["summary"][k], n["summary"][k]
        arrow = "↑" if nv > ov else ("↓" if nv < ov else "=")
        print(f"  {k:<28s}  {ov:.4f}  →  {nv:.4f}   {arrow}{abs(nv-ov):.4f}")
    print()

    o_ch = {c["name"]: c for c in o["channels"]}
    n_ch = {c["name"]: c for c in n["channels"]}
    rows = []
    for name in sorted(set(o_ch) | set(n_ch)):
        oc = o_ch.get(name, {}); nc = n_ch.get(name, {})
        if not oc or not nc: continue
        if oc["status"] == "INACTIVE" or nc["status"] == "INACTIVE": continue
        rows.append({
            "name": name,
            "status_change": f"{oc['status']}→{nc['status']}",
            "amp_old": oc["amp_med"], "amp_new": nc["amp_med"],
            "amp_delta": nc["amp_med"] - oc["amp_med"],
            "r_old": oc["abs_r_med"], "r_new": nc["abs_r_med"],
        })
    rows.sort(key=lambda r: r["amp_delta"])  # most regressed first
    print(f"=== top {args.top} regressed channels (amp_med decrease) ===")
    for r in rows[:args.top]:
        print(f"  {r['name']:<22s} {r['status_change']:<14s} "
              f"amp {r['amp_old']:.3f}→{r['amp_new']:.3f} (Δ{r['amp_delta']:+.3f}) "
              f"|r| {r['r_old']:.2f}→{r['r_new']:.2f}")
    print()
    print(f"=== top {args.top} improved channels (amp_med increase) ===")
    for r in rows[::-1][:args.top]:
        print(f"  {r['name']:<22s} {r['status_change']:<14s} "
              f"amp {r['amp_old']:.3f}→{r['amp_new']:.3f} (Δ{r['amp_delta']:+.3f}) "
              f"|r| {r['r_old']:.2f}→{r['r_new']:.2f}")
    print()
    print("=== regions (mean_avg pixel abs / 255) ===")
    for reg in o["regions"]:
        if reg not in n["regions"]: continue
        ov = o["regions"][reg]["mean_avg"]; nv = n["regions"][reg]["mean_avg"]
        arrow = "↑" if nv > ov else ("↓" if nv < ov else "=")
        print(f"  {reg:<8s}  {ov:.2f}  →  {nv:.2f}   {arrow}{abs(nv-ov):.2f}")


if __name__ == "__main__":
    main()
