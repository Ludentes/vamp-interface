"""Score the overnight axis corpus with SigLIP-2 probes targeting the 6
non-blendshape-measurable axes (age, gender, hair_style, hair_color,
skin_smoothness, nose_shape).

Reuses the Siglip2Backend class from score_clip_probes.py.

Run:
  uv run python -m src.demographic_pc.score_overnight_siglip --run
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from src.demographic_pc.score_clip_probes import Siglip2Backend, score_image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "output/demographic_pc/fluxspace_metrics/crossdemo"
OUT = ROOT / "output/demographic_pc/overnight_siglip_probes.parquet"

# (probe_name, positive_prompt, negative_prompt)
PROBES: list[tuple[str, str, str]] = [
    ("elderly",
     "a photo of an elderly person in their eighties with deep wrinkles and grey hair",
     "a photo of a youthful person in their mid-twenties with smooth skin"),
    ("feminine",
     "a photo of a feminine face with soft features and long eyelashes",
     "a photo of a masculine face with a strong jawline and stubble"),
    ("long_hair",
     "a photo of a person with long flowing hair past the shoulders",
     "a photo of a person with a very short cropped military buzz-cut hairstyle"),
    ("black_hair",
     "a photo of a person with jet black hair",
     "a photo of a person with platinum blonde hair"),
    ("rough_skin",
     "a photo of a person with rough textured skin, visible pores and imperfections",
     "a photo of a person with very smooth flawless porcelain skin"),
    ("aquiline_nose",
     "a photo of a person with a prominent aquiline hooked nose",
     "a photo of a person with a small delicate button nose"),
]


@torch.no_grad()
def run() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backend = Siglip2Backend(device)
    # Re-encode only the probes we care about for this run.
    feats = {}
    for name, pos, neg in PROBES:
        toks = backend.processor(text=[pos, neg], padding="max_length",
                                  return_tensors="pt",
                                  truncation=True).to(device)  # type: ignore[attr-defined]
        tf = backend._as_tensor(backend.model.get_text_features(**toks))  # type: ignore[attr-defined]
        tf = tf / tf.norm(dim=-1, keepdim=True)
        feats[name] = tf

    pngs = sorted(SRC.rglob("*.png"))
    # Only our 11 new axes — skip the other crossdemo corpora.
    AXES = {"age", "gender", "hair_style", "hair_color", "skin_smoothness",
            "nose_shape", "eye_squint", "brow_lift", "brow_furrow",
            "gaze_horizontal", "mouth_stretch"}
    pngs = [p for p in pngs if p.relative_to(SRC).parts[0] in AXES]
    print(f"[siglip-overnight] scoring {len(pngs)} images")
    rows = []
    t0 = time.time()
    for i, p in enumerate(pngs):
        r = score_image(backend, feats, p)
        r["rel"] = str(p.relative_to(SRC))
        rows.append(r)
        if (i + 1) % 200 == 0:
            dt = time.time() - t0
            print(f"  [{i+1}/{len(pngs)}] {(i+1)/dt:.1f} img/s")
    df = pd.DataFrame(rows)
    cols = ["rel"] + [c for c in df.columns if c.endswith("_margin")]
    df = df[cols]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False, compression="zstd")
    print(f"[save] → {OUT}  rows={len(df)} cols={df.shape[1]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if args.run:
        run()
    else:
        print(f"Would score ~990 PNGs with SigLIP-2 on {len(PROBES)} probes")


if __name__ == "__main__":
    main()
