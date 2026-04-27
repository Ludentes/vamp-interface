"""Build a per-sample reverse index across all scored corpora.

Reads every blendshapes.json we have, parses filename metadata, residualises
and projects onto the NMF atom basis, and emits a single parquet at
`models/blendshape_nmf/sample_index.parquet`.

Each row = one scored sample. Columns:

    source        {bootstrap_v1, alpha_interp, smile_inphase, jaw_inphase,
                   intensity_full, anger_rebalance, surprise_rebalance,
                   disgust_rebalance, pucker_rebalance, lip_press_rebalance}
    rel           path-relative-to-source key used inside its blendshapes.json
    base          parsed base id (asian_m / black_f / ...)
    ethnicity     parsed ethnic component from BASE_META
    gender        m / f
    age           adult / young / elderly / teenage
    axis          expression axis text pair (anger/surprise/... or "smile" /
                   "jaw" / "glasses" inferred from the source)
    scale         float, parsed from filename if encoded; else NaN
    seed          int, parsed from filename; else -1
    start_pct     float, parsed from filename; else NaN
    has_attn      bool — has a row in an attn_cache/*/meta.json
    attn_tag      cache tag (e.g. "anger_rebalance"); null if no cache
    attn_row      row index into that cache's delta_mix.npy; -1 if none
    bs_*          52 raw MediaPipe blendshape channels (float)
    atom_NN       20 residualised atom coefficients (float), post-clip

Query examples:

    df = pd.read_parquet("models/blendshape_nmf/sample_index.parquet")
    df.nlargest(10, "atom_17")
    df[df.base == "young_european_f"].nlargest(10, "atom_17")
    df.groupby(["axis", "base"])[[f"atom_{k:02d}" for k in range(20)]].mean()
    df[df.has_attn & (df.axis == "pucker")][["attn_tag", "attn_row"]]
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "output/demographic_pc/fluxspace_metrics"
NMF_DIR = ROOT / "models/blendshape_nmf"
CACHE_ROOT = NMF_DIR / "attn_cache"
OUT_PARQUET = NMF_DIR / "sample_index.parquet"

# (source_tag, blendshapes.json path, image dir, axis label, attn_cache_tag)
SOURCES: list[tuple[str, Path, Path, str | None, str | None]] = [
    ("bootstrap_v1",       METRICS / "bootstrap_v1/blendshapes.json",
                            METRICS / "bootstrap_v1", None, None),
    ("alpha_interp",       METRICS / "crossdemo/smile/alpha_interp/blendshapes.json",
                            METRICS / "crossdemo/smile/alpha_interp", "smile_alpha", None),
    ("smile_inphase",      METRICS / "crossdemo/smile/smile_inphase/blendshapes.json",
                            METRICS / "crossdemo/smile/smile_inphase", "smile_inphase", "smile_inphase"),
    ("jaw_inphase",        METRICS / "crossdemo/smile/jaw_inphase/blendshapes.json",
                            METRICS / "crossdemo/smile/jaw_inphase", "jaw_inphase", "jaw_inphase"),
    ("intensity_full",     METRICS / "crossdemo/smile/intensity_full/blendshapes.json",
                            METRICS / "crossdemo/smile/intensity_full", "smile_intensity", None),
    ("alpha_interp_attn",  METRICS / "crossdemo/smile/alpha_interp_attn/blendshapes.json",
                            METRICS / "crossdemo/smile/alpha_interp_attn", "smile_alpha_attn", "alpha_interp_attn"),
    ("anger_rebalance",    METRICS / "crossdemo/anger/rebalance/blendshapes.json",
                            METRICS / "crossdemo/anger/rebalance", "anger", "anger_rebalance"),
    ("surprise_rebalance", METRICS / "crossdemo/surprise/rebalance/blendshapes.json",
                            METRICS / "crossdemo/surprise/rebalance", "surprise", "surprise_rebalance"),
    ("disgust_rebalance",  METRICS / "crossdemo/disgust/rebalance/blendshapes.json",
                            METRICS / "crossdemo/disgust/rebalance", "disgust", "disgust_rebalance"),
    ("pucker_rebalance",   METRICS / "crossdemo/pucker/rebalance/blendshapes.json",
                            METRICS / "crossdemo/pucker/rebalance", "pucker", "pucker_rebalance"),
    ("lip_press_rebalance",METRICS / "crossdemo/lip_press/rebalance/blendshapes.json",
                            METRICS / "crossdemo/lip_press/rebalance", "lip_press", "lip_press_rebalance"),
]

BASE_META = {
    "asian_m":          ("asian",     "m", "adult"),
    "black_f":          ("black",     "f", "adult"),
    "european_m":       ("european",  "m", "adult"),
    "elderly_latin_m":  ("latin",     "m", "elderly"),
    "young_european_f": ("european",  "f", "young"),
    "southasian_f":     ("southasian","f", "adult"),
}

# Pre-compile filename regexps; parsers try each in order.
SCALE_PATTERNS = [
    re.compile(r"s(?P<seed>\d+)_a(?P<alpha>[0-9.+-]+)"),                         # smile alpha
    re.compile(r"seed(?P<seed>\d+)_sp(?P<sp>[0-9.]+)_s(?P<scale>[+-]?[0-9.]+)"),  # rebalance
    re.compile(r"sp(?P<sp>[0-9.]+)_s(?P<scale>[+-]?[0-9.]+)"),                   # intensity_full (no seed in stem)
]


def parse_base(rel: str) -> str | None:
    top = rel.split("/", 1)[0]
    if top in BASE_META:
        return top
    fname = rel.split("/")[-1]
    for b in BASE_META:
        if fname.startswith(b + "_"):
            return b
    return None


def parse_filename_meta(rel: str) -> dict:
    """Extract scale, seed, start_percent from a relative filename, if
    encoded. Missing fields come back as NaN / -1."""
    out = {"scale": float("nan"), "seed": -1, "start_pct": float("nan"),
           "alpha": float("nan")}
    fname = rel.split("/")[-1]
    stem = fname.rsplit(".", 1)[0]
    for pat in SCALE_PATTERNS:
        m = pat.search(stem)
        if not m:
            continue
        gd = m.groupdict()
        if "seed" in gd and gd.get("seed") is not None:
            try:
                out["seed"] = int(gd["seed"])
            except Exception:
                pass
        if "scale" in gd and gd.get("scale") is not None:
            try:
                out["scale"] = float(gd["scale"])
            except Exception:
                pass
        if "sp" in gd and gd.get("sp") is not None:
            try:
                out["start_pct"] = float(gd["sp"])
            except Exception:
                pass
        if "alpha" in gd and gd.get("alpha") is not None:
            try:
                out["alpha"] = float(gd["alpha"])
            except Exception:
                pass
        break
    return out


def load_attn_cache_rowmap() -> dict[str, dict[str, int]]:
    """Map cache_tag → {rel_in_cache: row_idx}."""
    out: dict[str, dict[str, int]] = {}
    for meta_path in CACHE_ROOT.glob("*/meta.json"):
        tag = meta_path.parent.name
        m = json.loads(meta_path.read_text())
        rels = m.get("rels", [])
        out[tag] = {r: i for i, r in enumerate(rels)}
    return out


def load_nmf():
    """Return NMF artefacts + exact H lookup for fit-time samples.

    For samples that were in the residualised NMF fit, we return the exact
    `H_nmf_resid` row (from NMF's NNLS-constrained fit_transform). For
    samples NOT in the fit (e.g. intensity_full, which wasn't part of the
    decomposition corpus), we fall back to the pinv approximation.
    """
    W = np.load(NMF_DIR / "W_nmf_resid.npy")
    H_fit = np.load(NMF_DIR / "H_nmf_resid.npy")
    manifest = json.loads((NMF_DIR / "manifest_resid.json").read_text())
    mu = np.load(NMF_DIR / "mu_base_resid.npy")
    sigma = np.load(NMF_DIR / "sigma_base_resid.npy")
    channels_raw = manifest["channels_raw"]
    channels_full = manifest["channels_full"]
    unique_bases = manifest["unique_bases"]
    fit_sample_ids = manifest["sample_ids"]  # order matches H_fit rows
    h_lookup = {sid: i for i, sid in enumerate(fit_sample_ids)}
    raw_set = set(channels_raw)
    prune_mask = np.array([c in raw_set for c in channels_full])
    W_pinv = np.linalg.pinv(W)
    base_idx = {b: i for i, b in enumerate(unique_bases)}
    return (W, W_pinv, H_fit, h_lookup, mu, sigma, channels_raw,
            channels_full, prune_mask, base_idx)


def project_sample(scores: dict, channels_full: list[str],
                   prune_mask: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                   base_idx_val: int, W_pinv: np.ndarray) -> np.ndarray:
    x = np.array([scores.get(c, 0.0) for c in channels_full])
    sigma_safe = np.where(sigma < 1e-4, 1.0, sigma)
    x_res = (x - mu[base_idx_val]) / sigma_safe[base_idx_val]
    x_res = x_res[prune_mask]
    x_pos = np.clip(x_res, 0.0, None)
    x_neg = np.clip(-x_res, 0.0, None)
    stacked = np.concatenate([x_pos, x_neg])
    return np.clip(stacked @ W_pinv, 0.0, None)


def _fit_sid(tag: str, rel: str) -> str:
    """Build the same 'tag/<optional subtag>/rel' key used in
    fit_nmf_residualised.load_and_residualise so we can match rows in
    H_nmf_resid."""
    # fit_nmf_residualised tag logic: grandparent/parent when parent=="rebalance",
    # else parent. For the build_sample_index sources, the grandparent/parent
    # reconstruction is: anger_rebalance → "anger/rebalance", similarly for
    # all 5 rebalance sources; otherwise the source tag is the parent.
    if tag.endswith("_rebalance"):
        axis = tag.removesuffix("_rebalance")
        return f"{axis}/rebalance/{rel}"
    return f"{tag}/{rel}"


def main():
    print("[index] loading NMF basis + attn-cache row maps")
    (W, W_pinv, H_fit, h_lookup, mu, sigma, channels_raw, channels_full,
     prune_mask, base_idx) = load_nmf()
    cache_rowmap = load_attn_cache_rowmap()
    print(f"  NMF: {W.shape[0]} atoms, {W.shape[1]} stacked channels, "
          f"{len(channels_full)} raw channels, {len(channels_raw)} post-prune")
    print(f"  attn caches: {list(cache_rowmap.keys())}")

    # Discover the full channel order for BS columns (use channels_full so
    # parquet has every MediaPipe channel we ever see, not just post-prune).
    bs_cols = [f"bs_{c}" for c in channels_full]
    atom_cols = [f"atom_{k:02d}" for k in range(W.shape[0])]

    rows = []
    n_unresolved_base = 0
    n_exact_atoms = 0
    n_pinv_atoms = 0
    missing_channel_counts: dict[str, int] = {}
    for tag, bs_path, img_dir, axis, cache_tag in SOURCES:
        if not bs_path.exists():
            print(f"  [skip] missing {tag}: {bs_path}")
            continue
        data = json.loads(bs_path.read_text())
        cache_rowmap_for_tag = cache_rowmap.get(cache_tag, {}) if cache_tag else {}
        n_added = 0
        for rel, scores in data.items():
            base = parse_base(rel)
            if base is None:
                n_unresolved_base += 1
                continue
            eth, gen, age = BASE_META[base]
            meta = parse_filename_meta(rel)
            # Prefer exact H from the NMF fit; fall back to pinv approx.
            sid = _fit_sid(tag, rel)
            h_row = h_lookup.get(sid)
            if h_row is not None:
                atoms = H_fit[h_row]
                atom_source = "exact"
                n_exact_atoms += 1
            else:
                atoms = project_sample(scores, channels_full, prune_mask, mu,
                                       sigma, base_idx[base], W_pinv)
                atom_source = "pinv"
                n_pinv_atoms += 1
            # Track channels missing from MediaPipe score (silently zeroed).
            missing_here = [c for c in channels_full if c not in scores]
            if missing_here:
                for c in missing_here:
                    missing_channel_counts[c] = missing_channel_counts.get(c, 0) + 1
            row = {
                "source": tag, "rel": rel, "base": base,
                "ethnicity": eth, "gender": gen, "age": age,
                "axis": axis, "scale": meta["scale"], "seed": meta["seed"],
                "start_pct": meta["start_pct"],
                "has_attn": rel in cache_rowmap_for_tag,
                "attn_tag": cache_tag if rel in cache_rowmap_for_tag else None,
                "attn_row": int(cache_rowmap_for_tag.get(rel, -1)),
                "atom_source": atom_source,
                "img_path": str((img_dir / rel).relative_to(ROOT)),
            }
            for c in channels_full:
                row[f"bs_{c}"] = float(scores.get(c, 0.0))
            for k in range(len(atoms)):
                row[f"atom_{k:02d}"] = float(atoms[k])
            rows.append(row)
            n_added += 1
        print(f"  [src] {tag:<22} +{n_added:>4} samples  "
              f"(axis={axis}, attn_cache={cache_tag})")

    if n_unresolved_base:
        print(f"  [warn] {n_unresolved_base} samples dropped (unparseable base)")
    if missing_channel_counts:
        top_missing = sorted(missing_channel_counts.items(), key=lambda x: -x[1])[:5]
        print(f"  [warn] {len(missing_channel_counts)} MediaPipe channels "
              f"missing in some samples (zero-filled); top: {top_missing}")
    print(f"  [atoms] exact H_nmf_resid match: {n_exact_atoms}  "
          f"pinv approximation (not in fit): {n_pinv_atoms}")

    df = pd.DataFrame(rows)
    # Enforce column order: metadata first, then bs, then atoms
    meta_cols = ["source", "rel", "base", "ethnicity", "gender", "age",
                 "axis", "scale", "seed", "start_pct", "has_attn",
                 "attn_tag", "attn_row", "atom_source", "img_path"]
    df = df[meta_cols + bs_cols + atom_cols]
    df.to_parquet(OUT_PARQUET, index=False, compression="zstd")
    print(f"\n[index] rows={len(df)}  cols={df.shape[1]}")
    print(f"[save] → {OUT_PARQUET}  "
          f"({OUT_PARQUET.stat().st_size / 1024 / 1024:.2f} MB)")

    # Quick sanity summary
    print("\n[sanity] by source:")
    print(df.groupby("source").size().to_string())
    print("\n[sanity] axis coverage:")
    print(df.groupby("axis", dropna=False).size().to_string())
    print("\n[sanity] attention-cached samples:")
    print(f"  {df.has_attn.sum()} / {len(df)} linkable to fp16 attn cache")


if __name__ == "__main__":
    main()
