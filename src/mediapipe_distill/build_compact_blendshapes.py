"""Build `compact_blendshapes.pt` aligned to `compact.pt`'s SHA order.

`output/reverse_index/reverse_index.parquet` already has 52-d ARKit
blendshapes (columns `bs_*`) for all 70,000 FFHQ source rows, including
the 26,108 we VAE-encoded into `arc_full_latent/compact.pt`. So Step 1 of
the MediaPipe distillation plan is a join, not a re-extraction.

Output schema:
  blendshapes  (N, 52) fp32     — same channel order as MediaPipe's output
  detected     (N,)   bool      — bs_detected flag from reverse_index
  channel_names list[str] of 52 — for downstream interpretability
  shas         list[str] of N   — must match compact.pt's SHA order exactly
  format_version 1

The 52 channels are the standard ARKit blendshapes (excluding the
`bs__neutral` column, which is a redundant probability-of-neutral signal).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch


# 52 channels matching MediaPipe FaceLandmarker output exactly: 51 ARKit
# blendshapes + 1 `_neutral` (probability-of-neutral, output independently
# by MediaPipe). Order matches the reverse_index parquet schema after the
# `bs__neutral` prefix is stripped.
BLENDSHAPE_CHANNELS = [
    "_neutral",
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
]
# The reverse_index includes one extra: bs_jawOpen is right between mouth
# rows; we accept whatever columns are in the parquet that match `bs_<channel>`.
# Final canonical list comes from BLENDSHAPE_CHANNELS above (length-checked
# against the parquet at runtime).


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reverse-index", type=Path,
                   default=Path("output/reverse_index/reverse_index.parquet"))
    p.add_argument("--shas-json", type=Path, required=True,
                   help="JSON list of compact.pt SHAs in canonical order "
                        "(produce via: torch.load(compact.pt)['shas'])")
    p.add_argument("--out", type=Path, required=True,
                   help="output .pt path, e.g. compact_blendshapes.pt")
    args = p.parse_args()

    assert len(BLENDSHAPE_CHANNELS) == 52, "expected 52 ARKit blendshapes"

    print(f"loading reverse_index: {args.reverse_index}")
    df = pd.read_parquet(args.reverse_index)
    df = df[df["source"] == "ffhq"].copy()
    print(f"  ffhq rows: {len(df)}")

    bs_cols = [f"bs_{c}" for c in BLENDSHAPE_CHANNELS]
    missing = [c for c in bs_cols if c not in df.columns]
    if missing:
        raise ValueError(f"reverse_index missing columns: {missing[:5]}...")
    if "bs_detected" not in df.columns:
        raise ValueError("reverse_index missing bs_detected column")

    print(f"loading shas: {args.shas_json}")
    shas = json.load(args.shas_json.open())
    print(f"  N: {len(shas)}")

    # Index reverse_index by sha for an O(N) lookup (vs O(N²) iterrows).
    df = df.set_index("image_sha256")
    sub = df.loc[shas]  # raises KeyError if any sha missing — fail loud
    print(f"  joined: {len(sub)}/{len(shas)}")
    if len(sub) != len(shas):
        raise ValueError("join failed: missing SHAs in reverse_index")

    blendshapes = torch.from_numpy(sub[bs_cols].to_numpy(dtype="float32"))
    detected = torch.from_numpy(sub["bs_detected"].to_numpy(dtype=bool))
    print(f"  blendshapes: {tuple(blendshapes.shape)} {blendshapes.dtype}")
    print(f"  detected: {int(detected.sum())}/{detected.numel()} "
          f"({float(detected.float().mean()) * 100:.1f}%)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "blendshapes": blendshapes,
        "detected": detected,
        "channel_names": BLENDSHAPE_CHANNELS,
        "shas": shas,
        "format_version": 1,
    }, args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
