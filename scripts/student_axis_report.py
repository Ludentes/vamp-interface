"""Canonical per-axis health report for a trained student.

Produces ONE JSON + ONE markdown table covering three axes:
  A. Input channels (58 ARKit dims)   — amp ratios + correlations, aggregated over takes
  B. Output cells   (32x16 m_f cells) — per-cell R²
  C. Face regions   (brow / eye_l / eye_r / mouth) — pixel diff vs teacher_full

Plus a single scalar `summary_score` so two checkpoints can be ranked.

Why: when we retrain we need to know which specific axes improved or regressed,
not just "global ratio looks better". A run-to-run diff requires fixed schema +
deterministic aggregation. This is that schema.

Schema (axis_report_<run_tag>.json):
{
  "run_tag": "...", "ckpt": "...", "n_takes": 3, "thresholds": {...},
  "summary": {
      "amp_geomean": ...,
      "amp_pass_frac": ...,
      "r_pass_frac": ...,
      "r2_ge_0p7_frac": ...,
      "region_p90_mean_per255": ...,
      "score": ...                      # weighted blend, higher = better
  },
  "channels":  [ {name, category, amp_med, amp_iqr, r_med, status, ...}, x58 ],
  "cells":     {"r2_grid": [[...]], "bottom_k": [{row,col,r2}, x15], "frac_*": ...},
  "regions":   { "brow": {mean, p90, max, ...}, ... },
  "per_take":  {"take_name": {region_means}, ...}
}

Usage:
  python scripts/student_axis_report.py \
      --run_tag v3_lam10 \
      --ckpt runs/student_v3_lam10/student_best.pt \
      --eval_r2 runs/student_v3_lam10/eval_step060000.per_cell_r2.npz \
      --channel_jsons exp_output/arkit_bridge/diagnostics/channel_recovery_v3_take{2,3,8}.json \
      --compare_jsons exp_output/arkit_bridge/render/v3_compare/20260505_MySlate_{2,3,8}_compare.json \
      --out runs/student_v3_lam10/axis_report.json
"""
import argparse
import json
from pathlib import Path
import numpy as np


# Status thresholds (kept in JSON output so future runs use SAME definitions)
THRESHOLDS = {
    "amp_pass": 0.5,        # amp_vs_driving >= this = good
    "amp_weak": 0.2,        # in [weak, pass) = WEAK; below = BROKEN
    "r_pass":   0.3,        # |r_render_vs_driving| >= this = correlated
    "r2_pass":  0.7,        # cell R² >= this = good
    "region_ok_per255": 25, # region mean abs diff <= this = OK
}


def aggregate_channels(channel_jsons):
    """Aggregate per-take channel_recovery JSON files into per-channel rows."""
    by_name = {}  # name -> {category, amps:[], rs:[], driving_stds:[], csv_stds:[]}
    for p in channel_jsons:
        d = json.loads(Path(p).read_text())
        for ch in d["all_channels"]:
            n = ch["name"]
            r = by_name.setdefault(n, {
                "name": n, "category": ch["category"],
                "amp_vs_driving": [], "amp_vs_csv": [],
                "r_render_vs_driving": [], "r_driving_vs_csv": [],
                "driving_std": [], "rendered_std": [], "csv_std": [],
            })
            r["amp_vs_driving"].append(ch["amp_vs_driving"])
            r["amp_vs_csv"].append(ch["amp_vs_csv"])
            r["r_render_vs_driving"].append(ch["r_render_vs_driving"])
            r["r_driving_vs_csv"].append(ch["r_driving_vs_csv"])
            r["driving_std"].append(ch["driving_mp_std"])
            r["rendered_std"].append(ch["rendered_mp_std"])
            r["csv_std"].append(ch["csv_std"])

    rows = []
    for n in sorted(by_name):
        r = by_name[n]
        amp = np.array(r["amp_vs_driving"], dtype=float)
        rr = np.array(r["r_render_vs_driving"], dtype=float)
        amp_med = float(np.nanmedian(amp))
        amp_iqr = float(np.nanpercentile(amp, 75) - np.nanpercentile(amp, 25))
        r_med = float(np.nanmedian(np.abs(rr)))
        # status — uses thresholds for stable categorisation
        if amp_med >= THRESHOLDS["amp_pass"] and r_med >= THRESHOLDS["r_pass"]:
            status = "PASS"
        elif amp_med >= THRESHOLDS["amp_weak"]:
            status = "WEAK"
        else:
            status = "BROKEN"
        # but if driving_std is ~0 across takes, channel is INACTIVE in this corpus
        if float(np.median(r["driving_std"])) < 0.01:
            status = "INACTIVE"
        rows.append({
            "name": n,
            "category": r["category"],
            "n_takes": len(amp),
            "amp_med": amp_med,
            "amp_iqr": amp_iqr,
            "amp_min": float(np.nanmin(amp)),
            "amp_max": float(np.nanmax(amp)),
            "abs_r_med": r_med,
            "driving_std_med": float(np.median(r["driving_std"])),
            "rendered_std_med": float(np.median(r["rendered_std"])),
            "status": status,
        })
    return rows


def aggregate_cells(eval_r2_path):
    """Per-output-cell R² distribution."""
    r2 = np.load(eval_r2_path)["r2"]   # (32,16)
    flat = r2.flatten()
    order = np.argsort(flat)
    bottom_k = []
    for i in order[:15]:
        bottom_k.append({"row": int(i // 16), "col": int(i % 16),
                         "r2": float(flat[i])})
    return {
        "shape": list(r2.shape),
        "median": float(np.median(flat)),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "frac_ge_0p9": float((flat >= 0.9).mean()),
        "frac_ge_0p7": float((flat >= 0.7).mean()),
        "frac_ge_0p5": float((flat >= 0.5).mean()),
        "frac_lt_0p3": float((flat < 0.3).mean()),
        "bottom_k": bottom_k,
        "r2_grid": r2.tolist(),
    }


def aggregate_regions(compare_jsons):
    """Aggregate per-region pixel diffs across compare runs."""
    region_names = ["brow", "eye_l", "eye_r", "mouth"]
    per_take = {}
    pooled = {r: {"mean": [], "p90": [], "max": []} for r in region_names}
    for p in compare_jsons:
        d = json.loads(Path(p).read_text())
        take = d["take"]
        per_take[take] = {}
        for r in region_names:
            re = d["region_energy"].get(r)
            if re is None: continue
            per_take[take][r] = {
                "mean": re["mean"], "p90": re["p90"], "argmax": re["argmax"]
            }
            pooled[r]["mean"].append(re["mean"])
            pooled[r]["p90"].append(re["p90"])
        # also pixel-global mean abs
        per_take[take]["_global_mean_abs"] = float(d["mean_abs_per_frame"]["mean"])

    summary = {}
    for r in region_names:
        m = pooled[r]["mean"]; p = pooled[r]["p90"]
        if not m: continue
        mean_avg = float(np.mean(m))
        summary[r] = {
            "n_takes": len(m),
            "mean_avg": mean_avg,
            "p90_avg": float(np.mean(p)),
            "mean_max": float(np.max(m)),
            "status": "OK" if mean_avg <= THRESHOLDS["region_ok_per255"] else "HIGH",
        }
    return summary, per_take


def compute_summary(channels, cells, regions):
    amps = [c["amp_med"] for c in channels if c["status"] != "INACTIVE"]
    amps_pos = [a for a in amps if a > 0]
    amp_geomean = float(np.exp(np.mean(np.log(amps_pos)))) if amps_pos else 0.0
    amp_pass_frac = float(np.mean([c["status"] == "PASS" for c in channels
                                   if c["status"] != "INACTIVE"]))
    r_pass_frac = float(np.mean([c["abs_r_med"] >= THRESHOLDS["r_pass"]
                                 for c in channels if c["status"] != "INACTIVE"]))
    region_p90_mean = float(np.mean([r["p90_avg"] for r in regions.values()])) \
        if regions else 0.0
    # Higher = better. Score in [0,1] roughly.
    # weight: amp_geomean (clamped to 1.0) 0.3, pass_frac 0.3, r2_ge_0p7 0.2, region 0.2
    region_term = max(0.0, 1.0 - region_p90_mean / 80.0)  # 80/255 ≈ 31% pixel diff = floor
    score = (
        0.3 * min(1.0, amp_geomean) +
        0.3 * amp_pass_frac +
        0.2 * cells["frac_ge_0p7"] +
        0.2 * region_term
    )
    return {
        "amp_geomean": amp_geomean,
        "amp_pass_frac": amp_pass_frac,
        "r_pass_frac": r_pass_frac,
        "r2_ge_0p7_frac": cells["frac_ge_0p7"],
        "region_p90_mean_per255": region_p90_mean,
        "score": float(score),
    }


def render_markdown(report, out_md):
    L = []
    L.append(f"# axis report: {report['run_tag']}")
    L.append("")
    L.append(f"ckpt: `{report['ckpt']}`")
    L.append("")
    L.append("## summary")
    for k, v in report["summary"].items():
        L.append(f"- **{k}**: {v:.4f}" if isinstance(v, float) else f"- **{k}**: {v}")
    L.append("")
    L.append("## input channels (sorted by amp_med ascending — worst first)")
    L.append("")
    L.append("| status | name | category | amp_med | amp_iqr | abs_r_med | driv_std | rend_std |")
    L.append("|---|---|---|---|---|---|---|---|")
    chs = sorted(report["channels"], key=lambda c: (c["status"] == "INACTIVE", c["amp_med"]))
    for c in chs:
        L.append(f"| {c['status']} | {c['name']} | {c['category']} | "
                 f"{c['amp_med']:.3f} | {c['amp_iqr']:.3f} | {c['abs_r_med']:.3f} | "
                 f"{c['driving_std_med']:.4f} | {c['rendered_std_med']:.4f} |")
    L.append("")
    L.append("## bottom-15 output cells by R²")
    L.append("")
    L.append("| row | col | R² |")
    L.append("|---|---|---|")
    for c in report["cells"]["bottom_k"]:
        L.append(f"| {c['row']} | {c['col']} | {c['r2']:+.4f} |")
    L.append("")
    L.append("## face regions (pixel abs diff /255, teacher vs bridge, averaged over takes)")
    L.append("")
    L.append("| region | mean_avg | p90_avg | status |")
    L.append("|---|---|---|---|")
    for r, v in report["regions"].items():
        L.append(f"| {r} | {v['mean_avg']:.2f} | {v['p90_avg']:.2f} | {v['status']} |")
    L.append("")
    L.append("## per-take regions")
    L.append("")
    L.append("| take | brow | eye_l | eye_r | mouth | global |")
    L.append("|---|---|---|---|---|---|")
    for t, v in report["per_take"].items():
        b = v.get("brow", {}).get("mean", float("nan"))
        el = v.get("eye_l", {}).get("mean", float("nan"))
        er = v.get("eye_r", {}).get("mean", float("nan"))
        mo = v.get("mouth", {}).get("mean", float("nan"))
        g = v.get("_global_mean_abs", float("nan"))
        L.append(f"| {t} | {b:.2f} | {el:.2f} | {er:.2f} | {mo:.2f} | {g:.2f} |")

    Path(out_md).write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_tag", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval_r2", required=True)
    ap.add_argument("--channel_jsons", nargs="+", required=True)
    ap.add_argument("--compare_jsons", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    channels = aggregate_channels(args.channel_jsons)
    cells = aggregate_cells(args.eval_r2)
    regions, per_take = aggregate_regions(args.compare_jsons)
    summary = compute_summary(channels, cells, regions)

    report = {
        "run_tag": args.run_tag,
        "ckpt": args.ckpt,
        "n_takes_channel": len({Path(p).stem for p in args.channel_jsons}),
        "n_takes_compare": len({Path(p).stem for p in args.compare_jsons}),
        "thresholds": THRESHOLDS,
        "summary": summary,
        "channels": channels,
        "cells": cells,
        "regions": regions,
        "per_take": per_take,
    }

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    md = out.with_suffix(".md")
    render_markdown(report, md)

    # Console summary
    print(f"=== axis report: {args.run_tag} ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<28s}  {v:.4f}")
        else:
            print(f"  {k:<28s}  {v}")
    print()
    print("worst-5 channels (by amp_med, excluding INACTIVE):")
    chs = [c for c in channels if c["status"] != "INACTIVE"]
    for c in sorted(chs, key=lambda x: x["amp_med"])[:5]:
        print(f"  [{c['status']:<7s}] {c['name']:<22s} amp={c['amp_med']:.3f} "
              f"|r|={c['abs_r_med']:.3f} ({c['category']})")
    print()
    print(f"BROKEN: {sum(1 for c in chs if c['status']=='BROKEN')}/"
          f"{len(chs)}    WEAK: {sum(1 for c in chs if c['status']=='WEAK')}/"
          f"{len(chs)}    PASS: {sum(1 for c in chs if c['status']=='PASS')}/{len(chs)}")
    print()
    print(f"wrote {out}")
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
