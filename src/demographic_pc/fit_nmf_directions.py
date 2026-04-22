"""Phase 3-proper — fit FluxSpace-usable edit directions per NMF atom.

For each atom, picks top-K (step, block) sites from the diagnostic
weights, loads the full 3072-d `delta_mix.mean_d` tensors at those
sites across 1320 training renders, and fits ridge to produce a
direction matrix (K, 3072) per atom. These directions can be applied
at inference via FluxSpace by injecting `scale * direction` at the
corresponding (step, block) locations.

Input: cached `delta_mix.npy` + `attn_base.npy` per source (from
cache_attn_features.py), plus the diagnostic ridge report to select
sites.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = ROOT / "models/blendshape_nmf/attn_cache"
NMF_DIR = ROOT / "models/blendshape_nmf"
OUT_PATH = NMF_DIR / "directions_k11.npz"
SOURCES = ["smile_inphase", "jaw_inphase", "alpha_interp_attn"]

K_SITES = 24       # top sites per atom
RIDGE_ALPHA = 1.0
FNAME_RE = re.compile(r"(?P<base>[^/]+)/s(?P<seed>\d+)_a(?P<alpha>[0-9.]+)\.png$")


def load_cache_and_blendshapes():
    """Load concatenated delta_mix and blendshape projections across
    all sources. Returns:
        delta_mix: (N, S, B, D) fp16
        attn_base: (N, S, B, D) fp16
        Y:        (N, k) atom coefficients
        meta_rows: list of {source, rel, base, seed, alpha}
        step_keys, block_keys
    """
    import json as _json
    W_nmf = np.load(NMF_DIR / "W_nmf_k11.npy")  # (k, C)
    meta = _json.loads((NMF_DIR / "manifest.json").read_text())
    channels = meta["channels"]

    all_dm = []
    all_ab = []
    all_Y = []
    meta_rows = []
    step_keys = None
    block_keys = None
    for tag in SOURCES:
        cache_dir = CACHE_ROOT / tag
        m = _json.loads((cache_dir / "meta.json").read_text())
        step_keys = m["step_keys"]
        block_keys = m["block_keys"]
        dm = np.load(cache_dir / "delta_mix.npy", mmap_mode="r")  # mmap
        ab = np.load(cache_dir / "attn_base.npy", mmap_mode="r")
        # Load blendshapes for this source and project to atoms
        bs_json = ROOT / f"output/demographic_pc/fluxspace_metrics/crossdemo/smile/{tag}/blendshapes.json"
        scores = _json.loads(bs_json.read_text())
        rels = m["rels"]
        X_bs = np.zeros((len(rels), len(channels)))
        for i, rel in enumerate(rels):
            s = scores.get(rel, {})
            for j, c in enumerate(channels):
                X_bs[i, j] = s.get(c, 0.0)
        Y = np.clip(X_bs @ np.linalg.pinv(W_nmf), 0.0, None)  # (N, k)
        all_dm.append(dm)
        all_ab.append(ab)
        all_Y.append(Y)
        for rel in rels:
            mo = FNAME_RE.search(rel)
            meta_rows.append({
                "tag": tag, "rel": rel,
                "base": mo.group("base") if mo else "",
                "seed": int(mo.group("seed")) if mo else -1,
                "alpha": float(mo.group("alpha")) if mo else 0.0,
            })
        print(f"  [load] {tag}: N={len(rels)}")
    delta_mix = np.concatenate(all_dm, axis=0)
    attn_base = np.concatenate(all_ab, axis=0)
    Y = np.concatenate(all_Y, axis=0)
    print(f"  [load] total N={delta_mix.shape[0]}  delta_mix={delta_mix.shape}")
    return delta_mix, attn_base, Y, meta_rows, step_keys, block_keys


def load_site_importance():
    """Return per-atom per-site importance scores derived from the
    diagnostic ridge report.

    Returns: ndarray (n_atoms, n_sites) with non-negative importance.
    """
    rep = json.loads((NMF_DIR / "phase3_ridge_report.json").read_text())
    # Diagnostic feature layout: idx = step_idx * (n_blocks * 6) + block_idx * 6 + feat_type
    # Feature weights for each atom are stored by fitting Ridge again on full
    # data post-CV; we don't have them in the report JSON directly.
    # Rebuild: compute per-atom site-importance by re-running a quick
    # diagnostic ridge and summing |coef| across feat_type per site.
    #
    # Simpler: in the original diagnostic each atom had only top-5 feature
    # indices printed. That's sparse info. Instead, here we'll just take
    # all 912 sites into account, let the per-atom full-vector ridge decide.
    # Return uniform importance (equivalent to "let ridge figure it out").
    return None


def select_top_sites(W_coef_per_atom_site: np.ndarray, K: int) -> np.ndarray:
    """W_coef_per_atom_site shape (n_atoms, n_sites). Return (n_atoms, K)
    int indices of top-K sites per atom."""
    return np.argsort(-np.abs(W_coef_per_atom_site), axis=1)[:, :K]


def per_site_screening(delta_mix: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Quick per-site screening: for each (step, block) site s, fit a ridge
    on the 3072-d vector against all atoms jointly, report per-site
    per-atom in-sample R². Shape (n_atoms, n_sites).

    This is cheaper than per-site CV and gets us a reliable top-K ranking.
    """
    N, S, B, D = delta_mix.shape
    n_atoms = Y.shape[1]
    n_sites = S * B
    flat = delta_mix.reshape(N, n_sites, D)
    r2 = np.zeros((n_atoms, n_sites))
    print(f"  [screen] per-site ridge screening — {n_sites} sites, {n_atoms} atoms")
    tss = ((Y - Y.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)  # (k,)
    for s_idx in range(n_sites):
        X = flat[:, s_idx].astype(np.float32)
        if X.std() < 1e-6:
            continue
        m = Ridge(alpha=RIDGE_ALPHA, random_state=0).fit(X, Y)
        pred = m.predict(X)
        rss = ((Y - pred) ** 2).sum(axis=0)  # (k,)
        r2[:, s_idx] = 1.0 - rss / (tss + 1e-12)
        if (s_idx + 1) % 100 == 0:
            print(f"    site {s_idx+1}/{n_sites}")
    return r2


def fit_direction_for_atom(delta_mix: np.ndarray, y: np.ndarray,
                           site_indices: np.ndarray) -> dict:
    """Fit CV ridge for a single atom on concatenated site tensors.

    site_indices: (K,) int, flat indices into S*B dimension.
    Returns {'direction': (K, D), 'cv_r2': float, 'in_sample_r2': float}.
    """
    N, S, B, D = delta_mix.shape
    flat = delta_mix.reshape(N, S * B, D)
    X = flat[:, site_indices].reshape(N, -1).astype(np.float32)  # (N, K*D)
    # CV
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
    in_sample = float(1.0 - rss / tss_full)
    direction = final.coef_.reshape(len(site_indices), D)
    return {"direction": direction,
            "cv_r2_mean": float(np.mean(r2_cv)),
            "cv_r2_std": float(np.std(r2_cv)),
            "in_sample_r2": in_sample,
            "intercept": float(final.intercept_)}


def main() -> None:
    print("[phase3p] loading cache + blendshape atoms")
    delta_mix, attn_base, Y, meta, step_keys, block_keys = load_cache_and_blendshapes()
    N, S, B, D = delta_mix.shape
    n_atoms = Y.shape[1]

    print(f"\n[phase3p] per-site screening — {S*B} sites")
    r2_site = per_site_screening(delta_mix, Y)  # (n_atoms, S*B)
    print(f"  [screen] done. max per-atom in-sample R²: "
          f"{[f'{r2_site[k].max():.3f}' for k in range(n_atoms)]}")

    # Top-K sites per atom by in-sample R² from screening
    top_sites = select_top_sites(r2_site, K_SITES)  # (n_atoms, K)

    print(f"\n[phase3p] fitting direction per atom @ K={K_SITES} sites")
    print(f"  {'#':>3}  {'CV R²':>7}  {'in-sample R²':>12}  "
          f"{'top site R²':>12}")
    out_atoms = {}
    for k in range(n_atoms):
        sites = top_sites[k]
        res = fit_direction_for_atom(delta_mix, Y[:, k], sites)
        out_atoms[f"atom_{k:02d}_direction"] = res["direction"]
        out_atoms[f"atom_{k:02d}_sites"] = sites
        out_atoms[f"atom_{k:02d}_cv_r2_mean"] = np.array([res["cv_r2_mean"]])
        out_atoms[f"atom_{k:02d}_cv_r2_std"] = np.array([res["cv_r2_std"]])
        top_site_r2 = float(r2_site[k, sites].max())
        print(f"  {k:>3}  {res['cv_r2_mean']:>7.3f}  "
              f"{res['in_sample_r2']:>12.3f}  {top_site_r2:>12.3f}")

    # Metadata
    out_atoms["step_keys"] = np.array(step_keys)
    out_atoms["block_keys"] = np.array(block_keys, dtype=object)
    out_atoms["site_r2_by_atom"] = r2_site  # (n_atoms, S*B)
    out_atoms["K_sites"] = np.array([K_SITES])
    out_atoms["ridge_alpha"] = np.array([RIDGE_ALPHA])

    np.savez(OUT_PATH, **out_atoms)
    print(f"\n[phase3p] saved → {OUT_PATH}  (K={K_SITES}, {n_atoms} atoms)")


if __name__ == "__main__":
    main()
