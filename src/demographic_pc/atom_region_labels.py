"""Auto-generate a per-atom region label from top-channel loadings.

Useful for morning analysis: filter validation results by facial region to
see if injection works only for certain atom categories (e.g. mouth atoms
but not brow atoms, suggesting region-specific Flux response).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NMF_DIR = ROOT / "models/blendshape_nmf"

REGIONS = {
    "mouth-pucker": ("mouthPucker", "mouthFunnel", "mouthClose"),
    "mouth-smile":  ("mouthSmile",),
    "mouth-lower":  ("mouthLowerDown", "mouthShrugLower", "mouthRoll"),
    "mouth-upper":  ("mouthUpperUp", "mouthShrugUpper"),
    "mouth-press":  ("mouthPress", "mouthDimple", "mouthFrown"),
    "mouth-stretch": ("mouthStretch", "mouthLeft", "mouthRight"),
    "brow-up":      ("browOuterUp", "browInnerUp"),
    "brow-down":    ("browDown",),
    "eye-squint":   ("eyeSquint",),
    "eye-blink":    ("eyeBlink",),
    "eye-gaze-h":   ("eyeLookOut", "eyeLookIn"),
    "eye-gaze-v":   ("eyeLookUp", "eyeLookDown"),
    "eye-wide":     ("eyeWide",),
    "jaw":          ("jaw", "cheekPuff"),
}


def classify(top_channels: list[str]) -> str:
    score: dict[str, int] = {}
    for ch in top_channels[:3]:
        for region, prefixes in REGIONS.items():
            if any(p in ch for p in prefixes):
                score[region] = score.get(region, 0) + 1
                break
    if not score:
        return "other"
    return max(score, key=lambda r: score[r])


def main():
    W = np.load(NMF_DIR / "W_nmf_resid.npy")  # (k, 2C)
    manifest = json.loads((NMF_DIR / "manifest_resid.json").read_text())
    channels_raw = manifest["channels_raw"]
    stack = manifest["channels_stack"]  # length 2C, e.g. "mouthSmile(+)", "mouthSmile(-)"
    print(f"{'atom':>4}  {'region':>14}  top-3 channels")
    out = []
    for k in range(W.shape[0]):
        if W[k].sum() < 1e-4:
            out.append({"atom": k, "region": "dead"})
            continue
        top_idx = np.argsort(-W[k])[:3]
        tops = [stack[i] for i in top_idx]
        # Strip (+)/(-) suffix for region matching
        top_channels = [t.split("(")[0] for t in tops]
        region = classify(top_channels)
        out.append({"atom": k, "region": region, "top_channels": tops})
        print(f"  #{k:02d}  {region:>14}  {', '.join(tops)}")

    (NMF_DIR / "atom_regions.json").write_text(json.dumps(out, indent=2))
    print(f"\n[save] → {NMF_DIR}/atom_regions.json")


if __name__ == "__main__":
    main()
