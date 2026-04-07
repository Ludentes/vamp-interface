#!/usr/bin/env python3
"""
compare_scorers.py — Compare gemma3 vs gemma4 rescoring results.

Usage:
    uv run src/compare_scorers.py data/test_dataset_gemma3.json data/test_dataset_gemma4.json
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> tuple[dict, list[dict]]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    meta = d.get("rescored_with", {})
    return meta, d["jobs"]


def sus_category(level: int) -> str:
    if level <= 20: return "safe"
    if level <= 40: return "low"
    if level <= 60: return "medium"
    if level <= 80: return "high"
    return "critical"


def cohort_stats(jobs: list[dict], field: str) -> dict[str, dict]:
    by_cohort: dict[str, list[int]] = defaultdict(list)
    for j in jobs:
        v = None
        if field == "original":
            v = j.get("sus_level")
        elif field == "rescore":
            v = (j.get("rescore") or {}).get("sus_level")
        if v is not None:
            by_cohort[j["cohort"]].append(v)
    return {
        c: {
            "n": len(vals),
            "avg": sum(vals) / len(vals),
            "min": min(vals),
            "max": max(vals),
        }
        for c, vals in by_cohort.items()
    }


def suspicious_delivery_rate(jobs: list[dict], field: str) -> dict[str, float]:
    """What fraction of each cohort has suspicious_delivery=true in rescore factors."""
    by_cohort: dict[str, list] = defaultdict(list)
    for j in jobs:
        if field == "rescore":
            factors = (j.get("rescore") or {}).get("factors", {})
        else:
            factors = j.get("sus_factors", {})
        sd = factors.get("suspicious_delivery")
        if sd is not None:
            by_cohort[j["cohort"]].append(bool(sd))
    return {
        c: sum(vals) / len(vals)
        for c, vals in by_cohort.items() if vals
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: compare_scorers.py <gemma3_file> <gemma4_file>")
        sys.exit(1)

    path3, path4 = Path(sys.argv[1]), Path(sys.argv[2])
    meta3, jobs3 = load(path3)
    meta4, jobs4 = load(path4)

    orig_stats = cohort_stats(jobs3, "original")
    stats3 = cohort_stats(jobs3, "rescore")
    stats4 = cohort_stats(jobs4, "rescore")

    sd3 = suspicious_delivery_rate(jobs3, "rescore")
    sd4 = suspicious_delivery_rate(jobs4, "rescore")
    sd_orig = suspicious_delivery_rate(jobs3, "original")

    all_cohorts = sorted(set(orig_stats) | set(stats3) | set(stats4))

    print(f"\n{'='*90}")
    print(f"Sus Scoring Comparison: gemma3 ({meta3.get('model','?')}) vs gemma4 ({meta4.get('model','?')})")
    print(f"{'='*90}")
    print(f"\n{'Cohort':25s} {'orig':>6} {'g3':>6} {'Δ3':>6} {'g4':>6} {'Δ4':>6}  {'sd_orig':>7} {'sd_g3':>7} {'sd_g4':>7}")
    print("─" * 90)

    for cohort in all_cohorts:
        o  = orig_stats.get(cohort, {}).get("avg", 0)
        g3 = stats3.get(cohort, {}).get("avg", 0)
        g4 = stats4.get(cohort, {}).get("avg", 0)
        sdo  = sd_orig.get(cohort, 0)
        sdg3 = sd3.get(cohort, 0)
        sdg4 = sd4.get(cohort, 0)
        flag3 = " ◄" if abs(g3 - o) > 20 else ""
        flag4 = " ◄" if abs(g4 - o) > 20 else ""
        print(f"  {cohort:25s} {o:6.1f} {g3:6.1f} {g3-o:+6.1f}{flag3:2s} {g4:6.1f} {g4-o:+6.1f}{flag4:2s}  "
              f"{sdo:7.2f} {sdg3:7.2f} {sdg4:7.2f}")

    print("─" * 90)

    # Per-model accuracy vs original (MAE and category match)
    print("\nAccuracy vs original prod scores:")
    for label, jobs in [("gemma3", jobs3), ("gemma4", jobs4)]:
        orig_vals, new_vals = [], []
        bucket_match = 0
        for j in jobs:
            ov = j.get("sus_level")
            nv = (j.get("rescore") or {}).get("sus_level")
            if ov is not None and nv is not None:
                orig_vals.append(ov)
                new_vals.append(nv)
                if sus_category(ov) == sus_category(nv):
                    bucket_match += 1
        if not orig_vals:
            continue
        mae = sum(abs(o - n) for o, n in zip(orig_vals, new_vals)) / len(orig_vals)
        bucket_acc = bucket_match / len(orig_vals) * 100
        print(f"  {label:8s}: MAE={mae:.1f}  bucket_acc={bucket_acc:.1f}%  n={len(orig_vals)}")

    # suspicious_delivery false positive focus
    print("\nsuspicious_delivery flag rate (legit cohorts only):")
    legit_cohorts = [c for c in all_cohorts if "legit" in c]
    for cohort in legit_cohorts:
        sdo  = sd_orig.get(cohort, 0)
        sdg3 = sd3.get(cohort, 0)
        sdg4 = sd4.get(cohort, 0)
        print(f"  {cohort:25s}: prod={sdo:.2f}  gemma3={sdg3:.2f}  gemma4={sdg4:.2f}")

    print()
    print("Legend: Δ3/Δ4 = drift vs original prod score. ◄ = drift >20pts.")
    print("sd = suspicious_delivery factor rate (false positives on legit cohorts = bad).")


if __name__ == "__main__":
    main()
