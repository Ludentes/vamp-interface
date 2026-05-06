"""Multiply existing sample_weights by 1/sqrt(take_size).

Reads a stats npz (produced by precompute_v4_stats.py), groups its
`paths` by take_id (filename prefix up to the `_frame_` token, with
optional trailing `_flip` stripped), and rewrites `sample_weights` as
`sample_weights * (1 / sqrt(take_size))`. Saves to `--out`.

Note: under-represented takes get higher per-sample weight; takes are
the natural unit of expressive content (one MySlate clip ≈ one
expression sequence).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


_FRAME_RE = re.compile(r"_frame_\d+(?:_flip)?\.pkl$")


def take_id(name: str) -> str:
    return _FRAME_RE.sub("", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npz", required=True)
    ap.add_argument("--out_npz", required=True)
    args = ap.parse_args()

    src = np.load(args.in_npz, allow_pickle=True)
    paths = [str(p) for p in src["paths"]]
    sw = src["sample_weights"].astype(np.float32)

    take_of = [take_id(p) for p in paths]
    sizes: dict[str, int] = {}
    for t in take_of:
        sizes[t] = sizes.get(t, 0) + 1

    factor = np.array(
        [1.0 / np.sqrt(sizes[t]) for t in take_of], dtype=np.float32,
    )
    new_sw = sw * factor
    # Re-normalize so the mean weight is 1.0 (preserves overall LR).
    new_sw = new_sw / new_sw.mean()

    out = {k: src[k] for k in src.files}
    out["sample_weights"] = new_sw

    Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_npz, **out)

    by_size = sorted(sizes.items(), key=lambda kv: kv[1])
    print(f"takes: {len(sizes)}  smallest={by_size[0]}  largest={by_size[-1]}")
    print(f"sample_weights post-reweight: min={new_sw.min():.3f} "
          f"median={np.median(new_sw):.3f} max={new_sw.max():.3f}")
    print(f"wrote {args.out_npz}")


if __name__ == "__main__":
    main()
