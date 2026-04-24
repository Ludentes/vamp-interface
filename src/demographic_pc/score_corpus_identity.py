"""Stage 0 diagnostic — identity drift in existing corpus.

For each (axis, base, seed) triple in the crossdemo corpus, compute
ArcFace cosine similarity between the α=0 image (anchor) and each
α>0 image. Reports:

  - per-α distribution (mean, std, median, min)
  - per-base distribution (which demographics drift more)
  - fraction of pairs below identity threshold τ=0.75

Usage:
    uv run python src/demographic_pc/score_corpus_identity.py \\
        --axis eye_squint --threshold 0.75

Output: CSV per (base, seed, α) + printed summary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.demographic_pc.classifiers import InsightFaceClassifier  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "output/demographic_pc/fluxspace_metrics/crossdemo"

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
BASES = ["asian_m", "black_f", "european_m", "elderly_latin_m",
         "young_european_f", "southasian_f"]
SEEDS = [1337, 2026, 4242]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--axis", required=True)
    p.add_argument("--subdir", default=None,
                   help="default: <axis>_inphase")
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--threshold", type=float, default=0.75)
    p.add_argument("--output", default=None,
                   help="CSV output path (default: alongside corpus)")
    return p.parse_args()


def main():
    args = parse_args()
    subdir = args.subdir or f"{args.axis}_inphase"
    axis_root = Path(args.root) / args.axis / subdir
    if not axis_root.exists():
        raise SystemExit(f"axis root not found: {axis_root}")

    print(f"[arcface] loading InsightFace w/ recognition head")
    clf = InsightFaceClassifier(ctx_id=0, with_embedding=True)

    # Score every image
    print(f"[score] walking {axis_root}")
    embs: dict[tuple[str, int, float], np.ndarray] = {}
    missing: list[tuple[str, int, float]] = []
    for base in BASES:
        base_dir = axis_root / base
        if not base_dir.exists():
            continue
        for seed in SEEDS:
            for a in ALPHAS:
                path = base_dir / f"s{seed}_a{a:.2f}.png"
                if not path.exists():
                    missing.append((base, seed, a))
                    continue
                img = cv2.imread(str(path))
                if img is None:
                    missing.append((base, seed, a))
                    continue
                pred = clf.predict(img)
                if not pred["detected"] or pred["embedding"] is None:
                    embs[(base, seed, a)] = None  # no face detected
                else:
                    embs[(base, seed, a)] = pred["embedding"]

    print(f"[score] {len(embs)} images scored, {len(missing)} missing, "
          f"{sum(1 for v in embs.values() if v is None)} no-face")

    # Compute per-pair cosine vs α=0 anchor
    rows = []
    for base in BASES:
        for seed in SEEDS:
            anchor = embs.get((base, seed, 0.0))
            if anchor is None:
                continue
            for a in ALPHAS:
                if a == 0.0:
                    continue
                emb = embs.get((base, seed, a))
                if emb is None:
                    rows.append({"base": base, "seed": seed, "alpha": a,
                                 "cos_vs_anchor": None, "face_detected": False})
                    continue
                cos = float(np.dot(anchor, emb))  # both are normed
                rows.append({"base": base, "seed": seed, "alpha": a,
                             "cos_vs_anchor": cos, "face_detected": True})

    # Summary stats
    print(f"\n=== Identity drift summary for axis={args.axis} ===")
    print(f"Threshold τ = {args.threshold}")
    print(f"\nPer-α distribution:")
    print(f"{'α':<6} {'n':<4} {'mean':<8} {'median':<8} {'min':<8} {'<τ':<6}")
    for a in ALPHAS[1:]:
        alpha_rows = [r for r in rows if r["alpha"] == a and r["cos_vs_anchor"] is not None]
        if not alpha_rows:
            continue
        cosines = np.array([r["cos_vs_anchor"] for r in alpha_rows])
        below = (cosines < args.threshold).sum()
        print(f"{a:<6.2f} {len(alpha_rows):<4d} "
              f"{cosines.mean():<8.3f} {np.median(cosines):<8.3f} "
              f"{cosines.min():<8.3f} {below}/{len(cosines)}")

    print(f"\nPer-base distribution (mean cos across α,seeds):")
    for base in BASES:
        base_rows = [r for r in rows if r["base"] == base and r["cos_vs_anchor"] is not None]
        if not base_rows:
            continue
        cosines = np.array([r["cos_vs_anchor"] for r in base_rows])
        below = (cosines < args.threshold).sum()
        print(f"  {base:<20s} n={len(cosines):<3d} mean={cosines.mean():.3f} "
              f"min={cosines.min():.3f} <τ={below}/{len(cosines)}")

    # Overall
    all_cos = np.array([r["cos_vs_anchor"] for r in rows if r["cos_vs_anchor"] is not None])
    overall_below = (all_cos < args.threshold).sum()
    print(f"\nOverall: n={len(all_cos)} mean={all_cos.mean():.3f} "
          f"median={np.median(all_cos):.3f} "
          f"below-τ={overall_below}/{len(all_cos)} ({100 * overall_below / len(all_cos):.1f}%)")

    # Write CSV
    out_path = Path(args.output) if args.output else (
        ROOT / f"output/demographic_pc/stage0_identity_{args.axis}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    with open(out_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["base", "seed", "alpha", "cos_vs_anchor", "face_detected"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[write] {out_path}")


if __name__ == "__main__":
    main()
