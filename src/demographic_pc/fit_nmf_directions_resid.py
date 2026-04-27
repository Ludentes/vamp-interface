"""Phase 3-proper — attention-space directions for residualised NMF atoms.

Same pipeline as `fit_nmf_directions.py` but:
  * Targets are z-score-residualised atom coefficients (using per-base mean +
    std from `mu_base_resid.npy` / `sigma_base_resid.npy`), reprojected onto
    the signed atom basis `W_nmf_resid.npy`.
  * Training set spans old 3 attention-cache sources (smile_inphase,
    jaw_inphase, alpha_interp_attn) plus 5 new rebalance sources
    (anger/surprise/disgust/pucker/lip_press).

Output: `models/blendshape_nmf/directions_resid.npz`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "output/demographic_pc/fluxspace_metrics"
CACHE_ROOT = ROOT / "models/blendshape_nmf/attn_cache"
NMF_DIR = ROOT / "models/blendshape_nmf"
OUT_PATH = NMF_DIR / "directions_resid.npz"

# (cache_tag, path to blendshapes.json used when caching)
SOURCES: list[tuple[str, Path]] = [
    ("smile_inphase",        METRICS / "crossdemo/smile/smile_inphase/blendshapes.json"),
    ("jaw_inphase",          METRICS / "crossdemo/smile/jaw_inphase/blendshapes.json"),
    ("alpha_interp_attn",    METRICS / "crossdemo/smile/alpha_interp_attn/blendshapes.json"),
    ("anger_rebalance",      METRICS / "crossdemo/anger/rebalance/blendshapes.json"),
    ("surprise_rebalance",   METRICS / "crossdemo/surprise/rebalance/blendshapes.json"),
    ("disgust_rebalance",    METRICS / "crossdemo/disgust/rebalance/blendshapes.json"),
    ("pucker_rebalance",     METRICS / "crossdemo/pucker/rebalance/blendshapes.json"),
    ("lip_press_rebalance",  METRICS / "crossdemo/lip_press/rebalance/blendshapes.json"),
]

BASE_META = {
    "asian_m":          ("asian",     "m", "adult"),
    "black_f":          ("black",     "f", "adult"),
    "european_m":       ("european",  "m", "adult"),
    "elderly_latin_m":  ("latin",     "m", "elderly"),
    "young_european_f": ("european",  "f", "young"),
    "southasian_f":     ("southasian","f", "adult"),
}

K_SITES = 24
RIDGE_ALPHA = 1.0


def parse_base(rel: str) -> str | None:
    # rel = "asian_m/..." for every cached source we ever write.
    head = rel.split("/", 1)[0]
    return head if head in BASE_META else None


def load_sources():
    """Load concatenated (delta_mix, residualised-atom-coefficients Y) across
    all cached sources that exist on disk. Skip any source whose cache isn't
    built yet."""
    W_nmf = np.load(NMF_DIR / "W_nmf_resid.npy")            # (k, 2C)
    mu    = np.load(NMF_DIR / "mu_base_resid.npy")          # (nB, C)
    sigma = np.load(NMF_DIR / "sigma_base_resid.npy")       # (nB, C)
    manifest = json.loads((NMF_DIR / "manifest_resid.json").read_text())
    channels_raw = manifest["channels_raw"]
    channels_full = manifest["channels_full"]
    unique_bases = manifest["unique_bases"]
    base_idx = {b: i for i, b in enumerate(unique_bases)}
    W_pinv = np.linalg.pinv(W_nmf)                          # (2C, k)

    sigma_safe = np.where(sigma < 1e-4, 1.0, sigma)

    # mu/sigma are stored at the full pre-prune channel shape; the NMF basis
    # is in the post-prune (channels_raw) space. Build the prune mask from
    # channels_full → channels_raw membership (same order preserved).
    raw_set = set(channels_raw)
    prune_mask = np.array([c in raw_set for c in channels_full])
    assert prune_mask.sum() == len(channels_raw), \
        f"prune mask size {prune_mask.sum()} != channels_raw {len(channels_raw)}"

    all_dm = []
    all_ab = []
    all_Y = []
    meta_rows = []
    step_keys = None
    block_keys = None
    total_skipped = 0
    for tag, bs_path in SOURCES:
        cache_dir = CACHE_ROOT / tag
        meta_path = cache_dir / "meta.json"
        if not meta_path.exists():
            print(f"  [skip] no cache yet: {tag}")
            continue
        m = json.loads(meta_path.read_text())
        step_keys = m["step_keys"]
        block_keys = m["block_keys"]
        dm = np.load(cache_dir / "delta_mix.npy", mmap_mode="r")  # (N, S, B, D)
        ab = np.load(cache_dir / "attn_base.npy", mmap_mode="r")
        scores = json.loads(bs_path.read_text())
        rels = m["rels"]

        # Residualise raw blendshapes at 52-channel shape, then prune.
        X = np.zeros((len(rels), len(channels_full)))
        base_list = []
        keep = []
        for i, rel in enumerate(rels):
            base = parse_base(rel)
            if base is None or base not in base_idx:
                continue
            s = scores.get(rel, {})
            for j, c in enumerate(channels_full):
                X[i, j] = s.get(c, 0.0)
            base_list.append(base)
            keep.append(i)
        if not keep:
            print(f"  [skip] {tag}: zero parseable samples")
            continue

        keep = np.array(keep)
        dm_k = dm[keep]
        ab_k = ab[keep]
        X_k = X[keep]
        bi = np.array([base_idx[b] for b in base_list])
        X_res_full = (X_k - mu[bi]) / sigma_safe[bi]            # (N, 52)
        X_res = X_res_full[:, prune_mask]                       # (N, len(channels_raw))
        X_pos = np.clip(X_res, 0.0, None)
        X_neg = np.clip(-X_res, 0.0, None)
        X_stack = np.concatenate([X_pos, X_neg], axis=1)        # (N, 2*len(channels_raw))
        Y = np.clip(X_stack @ W_pinv, 0.0, None)                # (N, k)

        skipped = len(rels) - len(keep)
        total_skipped += skipped
        all_dm.append(dm_k)
        all_ab.append(ab_k)
        all_Y.append(Y)
        for r, b in zip([rels[i] for i in keep], base_list):
            meta_rows.append({"tag": tag, "rel": r, "base": b})
        print(f"  [load] {tag}: N={len(keep)} (skipped {skipped})")

    if not all_dm:
        raise SystemExit("No cached sources found — run cache_attn_features first.")
    delta_mix = np.concatenate(all_dm, axis=0)
    attn_base = np.concatenate(all_ab, axis=0)
    Y = np.concatenate(all_Y, axis=0)
    print(f"  [load] total N={delta_mix.shape[0]}  delta_mix={delta_mix.shape}  "
          f"Y={Y.shape}  total_skipped={total_skipped}")
    return delta_mix, attn_base, Y, meta_rows, step_keys, block_keys


def per_site_screening(delta_mix: np.ndarray, Y: np.ndarray) -> np.ndarray:
    N, S, B, D = delta_mix.shape
    n_atoms = Y.shape[1]
    n_sites = S * B
    flat = delta_mix.reshape(N, n_sites, D)
    r2 = np.zeros((n_atoms, n_sites))
    print(f"  [screen] {n_sites} sites × {n_atoms} atoms")
    tss = ((Y - Y.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    for s_idx in range(n_sites):
        X = flat[:, s_idx].astype(np.float32)
        if X.std() < 1e-6:
            continue
        m = Ridge(alpha=RIDGE_ALPHA, random_state=0).fit(X, Y)
        pred = m.predict(X)
        rss = ((Y - pred) ** 2).sum(axis=0)
        r2[:, s_idx] = 1.0 - rss / (tss + 1e-12)
        if (s_idx + 1) % 200 == 0:
            print(f"    site {s_idx+1}/{n_sites}")
    return r2


def fit_direction_for_atom(delta_mix, y, site_indices):
    N, S, B, D = delta_mix.shape
    flat = delta_mix.reshape(N, S * B, D)
    X = flat[:, site_indices].reshape(N, -1).astype(np.float32)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    r2_cv = []
    for tr, te in kf.split(X):
        m = Ridge(alpha=RIDGE_ALPHA, random_state=0).fit(X[tr], y[tr])
        pred = m.predict(X[te])
        rss = ((y[te] - pred) ** 2).sum()
        tss = ((y[te] - y[te].mean()) ** 2).sum() + 1e-12
        r2_cv.append(float(1.0 - rss / tss))
    final = Ridge(alpha=RIDGE_ALPHA, random_state=0).fit(X, y)
    pred = final.predict(X)
    rss = ((y - pred) ** 2).sum()
    tss_full = ((y - y.mean()) ** 2).sum() + 1e-12
    return {"direction": final.coef_.reshape(len(site_indices), D),
            "cv_r2_mean": float(np.mean(r2_cv)),
            "cv_r2_std": float(np.std(r2_cv)),
            "in_sample_r2": float(1.0 - rss / tss_full),
            "intercept": float(final.intercept_)}


def main():
    print("[dir-resid] loading cache + residualised atom targets")
    delta_mix, attn_base, Y, meta, step_keys, block_keys = load_sources()
    N, S, B, D = delta_mix.shape
    n_atoms = Y.shape[1]

    print(f"\n[dir-resid] per-site screening — {S*B} sites, {n_atoms} atoms")
    r2_site = per_site_screening(delta_mix, Y)
    top_sites = np.argsort(-np.abs(r2_site), axis=1)[:, :K_SITES]

    print(f"\n[dir-resid] fitting direction per atom @ K={K_SITES}")
    print(f"  {'#':>3}  {'CV R²':>7}  {'in R²':>7}  {'top-site R²':>12}")
    out_atoms = {}
    for k in range(n_atoms):
        sites = top_sites[k]
        res = fit_direction_for_atom(delta_mix, Y[:, k], sites)
        out_atoms[f"atom_{k:02d}_direction"] = res["direction"]
        out_atoms[f"atom_{k:02d}_sites"] = sites
        out_atoms[f"atom_{k:02d}_cv_r2_mean"] = np.array([res["cv_r2_mean"]])
        out_atoms[f"atom_{k:02d}_cv_r2_std"] = np.array([res["cv_r2_std"]])
        out_atoms[f"atom_{k:02d}_intercept"] = np.array([res["intercept"]])
        print(f"  {k:>3}  {res['cv_r2_mean']:>7.3f}  {res['in_sample_r2']:>7.3f}  "
              f"{float(r2_site[k, sites].max()):>12.3f}")

    out_atoms["step_keys"] = np.array(step_keys)
    out_atoms["block_keys"] = np.array(block_keys, dtype=object)
    out_atoms["site_r2_by_atom"] = r2_site
    out_atoms["K_sites"] = np.array([K_SITES])
    out_atoms["ridge_alpha"] = np.array([RIDGE_ALPHA])

    np.savez(OUT_PATH, **out_atoms)
    print(f"\n[dir-resid] saved → {OUT_PATH}  (K={K_SITES}, {n_atoms} atoms, N={N})")


if __name__ == "__main__":
    main()
