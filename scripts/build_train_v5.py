"""Build the v5 training set from `all/` + `flipped_raw/`.

v5 dataset = v3 train (`all/`) ∪ horizontal-mirror augmentation.

Steps:
  1. Hardlink every pkl in `all/` into `all_v5/` (the "real" pairs).
  2. For each pkl in `flipped_raw/`, hardlink into `all_v5/` ONLY if its
     un-flipped sibling is in `all/` (preserves the holdout split — we
     do not want a flipped version of a holdout-v3 frame in training).
  3. Optional --filter_p99: drop pkls (originals + flipped) whose ‖m_f‖
     exceeds the p99 of the input corpus norms.

Output: `data/arkit_bridge_pairs/all_v5/` with hardlinks, plus a
`build_log.json` summarizing counts.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_dir", default="data/arkit_bridge_pairs/all")
    ap.add_argument("--flip_dir", default="data/arkit_bridge_pairs/flipped_raw")
    ap.add_argument("--holdout_dir", default="data/arkit_bridge_pairs/holdout_v3")
    ap.add_argument("--out_dir", default="data/arkit_bridge_pairs/all_v5")
    ap.add_argument("--filter_p99", action="store_true",
                    help="Drop pkls whose ‖m_f‖ > p99 over the input corpus.")
    args = ap.parse_args()

    orig = Path(args.orig_dir)
    flip = Path(args.flip_dir)
    hold = Path(args.holdout_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    orig_files = sorted(orig.glob("*frame_*.pkl"))
    flip_files = sorted(flip.glob("*frame_*_flip.pkl"))
    orig_names = {p.name for p in orig_files}

    # Determine which flipped frames are train-eligible: un-flipped name
    # (stripping the trailing "_flip") must be in `all/` (i.e. not holdout).
    eligible_flip = []
    skipped_holdout = 0
    for fp in flip_files:
        unflipped = fp.name.replace("_flip.pkl", ".pkl")
        if unflipped in orig_names:
            eligible_flip.append(fp)
        else:
            skipped_holdout += 1

    # Optional norm filter — load m_f norms across both sets together.
    keep_orig, keep_flip = orig_files, eligible_flip
    p99 = None
    n_dropped = 0
    if args.filter_p99:
        norms = []
        for p in orig_files + eligible_flip:
            with open(p, "rb") as f:
                d = pickle.load(f)
            norms.append(float(np.linalg.norm(d["m_f"])))
        norms = np.asarray(norms)
        p99 = float(np.percentile(norms, 99))
        kept = norms <= p99
        all_files = orig_files + eligible_flip
        keep_orig = [all_files[i] for i in range(len(orig_files))
                     if kept[i]]
        keep_flip = [all_files[i] for i in range(len(orig_files),
                                                  len(all_files))
                     if kept[i]]
        n_dropped = int(len(all_files) - kept.sum())

    n_link = 0
    for src in keep_orig + keep_flip:
        dst = out / src.name
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
        n_link += 1

    info = {
        "orig_dir": str(orig), "flip_dir": str(flip), "out_dir": str(out),
        "holdout_dir": str(hold),
        "n_orig_input": len(orig_files),
        "n_flip_input": len(flip_files),
        "n_flip_eligible": len(eligible_flip),
        "n_flip_skipped_as_holdout_source": skipped_holdout,
        "filter_p99": args.filter_p99,
        "p99_norm": p99,
        "n_dropped_p99": n_dropped,
        "n_kept_orig": len(keep_orig),
        "n_kept_flip": len(keep_flip),
        "n_total_linked": n_link,
    }
    with open(out / "build_log.json", "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
