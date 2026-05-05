"""Random 10% holdout split of pair pkls via hardlinks (no disk doubling)."""

import argparse
import os
import random
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir of frame_*.pkl")
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--holdout_dir", required=True)
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src)
    train = Path(args.train_dir)
    hold = Path(args.holdout_dir)
    train.mkdir(parents=True, exist_ok=True)
    hold.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("*frame_*.pkl"))
    if not files:
        print(f"no pkls in {src}", file=sys.stderr); sys.exit(1)

    rng = random.Random(args.seed)
    n_hold = max(1, int(round(len(files) * args.frac)))
    hold_set = set(rng.sample(files, n_hold))

    n_t = n_h = 0
    for f in files:
        dst = (hold if f in hold_set else train) / f.name
        if dst.exists():
            dst.unlink()
        os.link(f, dst)
        if f in hold_set:
            n_h += 1
        else:
            n_t += 1
    print(f"src={len(files)} train={n_t} holdout={n_h} (frac={args.frac}, seed={args.seed})")


if __name__ == "__main__":
    main()
