"""Project per-row blendshape targets through the NMF basis pseudo-inverse
to produce 8-d atom targets aligned to compact / compact_rendered SHA order.

Output schema:
  atoms: (N, 8) float32
  detected: (N,) bool — copied from bs source
  atom_tags: list[str] (8 names, e.g. 'smile_inphase', ...)
  shas: list[str]
  format_version: 1

Atoms are computed via Y @ H_pinv (least-squares solution; can be slightly
negative — that's fine, NMF non-negativity is only on the basis side).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bs", type=Path, required=True,
                   help="compact_blendshapes.pt or compact_rendered.pt — anything with "
                        "'blendshapes', 'detected', 'shas', 'channel_names'")
    p.add_argument("--atom-library", type=Path,
                   default=Path("models/blendshape_nmf/au_library.npz"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    blob = torch.load(args.bs, map_location="cpu", weights_only=False)
    Y = blob["blendshapes"].numpy().astype(np.float32)
    detected = blob["detected"]
    channel_names = blob["channel_names"]
    shas = blob["shas"]

    lib = np.load(args.atom_library)
    H = lib["H"].astype(np.float32)               # (8, 52)
    tags = list(lib["tags"])
    lib_names = [str(s) for s in lib["names"]]
    if lib_names != channel_names:
        raise ValueError(f"channel order mismatch: lib first={lib_names[:3]}, ours first={channel_names[:3]}")

    H_pinv = np.linalg.pinv(H).astype(np.float32)  # (52, 8)
    atoms = Y @ H_pinv                              # (N, 8)
    print(f"atoms: shape={atoms.shape} range=[{atoms.min():.4f}, {atoms.max():.4f}] mean={atoms.mean():.4f}")
    print(f"per-atom std: {atoms.std(axis=0)}")
    print(f"tags: {tags}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "atoms": torch.from_numpy(atoms),
        "detected": detected,
        "atom_tags": [str(t) for t in tags],
        "shas": shas,
        "format_version": 1,
    }, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
