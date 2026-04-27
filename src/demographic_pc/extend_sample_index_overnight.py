"""Append the overnight_drift renders (1440 samples) to sample_index.parquet.

Layout (per overnight_render_batch.py):
    overnight_drift/smile/{rung}/{base}/seed{N}_s{scale}.png         (768)
    overnight_drift/beard/{polarity}/{base}/seed{N}_s{scale}.png     (96)
    overnight_drift/rebalance_reseed/{axis}/{base}/seed{N}_s{scale}.png (576)

One blendshapes.json per top-level axis dir keyed by `{sub}/{base}/{fname}`.

Atoms are computed via pinv projection (these samples are not in the NMF fit).
`source` is a new tag per axis; a new `rung` / `polarity` column goes under
`start_pct` in the metadata block but we reuse `start_pct` -> NaN and store
the rung/polarity in a new column `subtag` (added non-destructively).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.demographic_pc.build_sample_index import (
    BASE_META, load_nmf, project_sample, OUT_PARQUET,
)

# Bearded variants of the male bases used in beard_rebalance. Same
# (ethnicity, gender, age) as the clean base — the only difference is
# the forced-bearded starting prompt. Recorded as separate bases so the
# same-identity bearded→unbearded pair is queryable.
BEARDED_BASE_META = {
    "asian_m_bearded":         ("asian",    "m", "adult"),
    "european_m_bearded":      ("european", "m", "adult"),
    "elderly_latin_m_bearded": ("latin",    "m", "elderly"),
}
ALL_BASE_META = {**BASE_META, **BEARDED_BASE_META}

ROOT = Path(__file__).resolve().parents[2]
OVERNIGHT = ROOT / "output/demographic_pc/overnight_drift"

SEED_SCALE_RE = re.compile(r"seed(?P<seed>\d+)_s(?P<scale>[+-]?[0-9.]+)")

# (source_tag, blendshapes.json, axis_label_or_None_means_from_subtag)
OVERNIGHT_SOURCES = [
    ("overnight_smile",           OVERNIGHT / "smile" / "blendshapes.json",           "smile"),
    ("overnight_beard",           OVERNIGHT / "beard" / "blendshapes.json",           "beard"),
    ("overnight_beard_rebalance", OVERNIGHT / "beard_rebalance" / "blendshapes.json", "beard_rebalance"),
    ("overnight_rebalance",       OVERNIGHT / "rebalance_reseed" / "blendshapes.json", None),
]


def parse_rel(rel: str) -> dict | None:
    parts = rel.split("/")
    if len(parts) != 3:
        return None
    subtag, base, fname = parts
    if base not in ALL_BASE_META:
        return None
    m = SEED_SCALE_RE.search(fname.rsplit(".", 1)[0])
    if not m:
        return None
    return {
        "subtag": subtag,
        "base": base,
        "seed": int(m["seed"]),
        "scale": float(m["scale"]),
    }


def main():
    print("[overnight-index] loading NMF basis")
    (W, W_pinv, H_fit, h_lookup, mu, sigma, channels_raw, channels_full,
     prune_mask, base_idx) = load_nmf()
    bs_cols = [f"bs_{c}" for c in channels_full]
    atom_cols = [f"atom_{k:02d}" for k in range(W.shape[0])]

    rows = []
    for tag, bs_path, axis_fixed in OVERNIGHT_SOURCES:
        if not bs_path.exists():
            print(f"  [skip] missing {tag}: {bs_path}")
            continue
        data = json.loads(bs_path.read_text())
        n_added = n_bad = 0
        img_dir = bs_path.parent
        for rel, scores in data.items():
            meta = parse_rel(rel)
            if meta is None:
                n_bad += 1
                continue
            base = meta["base"]
            eth, gen, age = ALL_BASE_META[base]
            axis = axis_fixed if axis_fixed is not None else meta["subtag"]
            # Bearded variants share mu/sigma with the clean base (same
            # demographic); strip the suffix for the base_idx lookup.
            base_for_norm = base.removesuffix("_bearded")
            atoms = project_sample(scores, channels_full, prune_mask, mu, sigma,
                                   base_idx[base_for_norm], W_pinv)
            row = {
                "source": tag, "rel": rel, "base": base,
                "ethnicity": eth, "gender": gen, "age": age,
                "axis": axis, "scale": meta["scale"], "seed": meta["seed"],
                "start_pct": float("nan"),
                "has_attn": False, "attn_tag": None, "attn_row": -1,
                "atom_source": "pinv",
                "img_path": str((img_dir / rel).relative_to(ROOT)),
                "subtag": meta["subtag"],
            }
            for c in channels_full:
                row[f"bs_{c}"] = float(scores.get(c, 0.0))
            for k in range(len(atoms)):
                row[f"atom_{k:02d}"] = float(atoms[k])
            rows.append(row)
            n_added += 1
        print(f"  [src] {tag:<22} +{n_added:>4} samples  (bad={n_bad})")

    new_df = pd.DataFrame(rows)
    meta_cols = ["source", "rel", "base", "ethnicity", "gender", "age",
                 "axis", "scale", "seed", "start_pct", "has_attn",
                 "attn_tag", "attn_row", "atom_source", "img_path", "subtag"]
    new_df = new_df[meta_cols + bs_cols + atom_cols]

    # Merge with existing parquet. Add `subtag` column (null for old rows).
    old_df = pd.read_parquet(OUT_PARQUET)
    if "subtag" not in old_df.columns:
        old_df["subtag"] = None
    # Reorder old cols to match new layout.
    old_df = old_df[meta_cols + bs_cols + atom_cols]

    # Drop any overnight rows that might already be in old_df (re-runs).
    old_df = old_df[~old_df["source"].isin([t for t, *_ in OVERNIGHT_SOURCES])]

    combined = pd.concat([old_df, new_df], ignore_index=True)
    combined.to_parquet(OUT_PARQUET, index=False, compression="zstd")

    print(f"\n[save] → {OUT_PARQUET}  rows={len(combined)} (old={len(old_df)}, new={len(new_df)})")
    print("\n[sanity] by source (new only):")
    print(new_df.groupby("source").size().to_string())
    print("\n[sanity] overnight axis × subtag × base coverage:")
    print(new_df.groupby(["axis", "subtag", "base"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
