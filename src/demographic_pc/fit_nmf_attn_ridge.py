"""Phase 3 (diagnostic) — ridge-fit attention-cache scalar features
against NMF-atom coefficients.

Input features per render: delta_mix Frobenius norm at each (step, block)
from the captured attention pkl — 16 steps × 57 blocks ≈ 912 scalars.
Adds a small number of global scalars (delta_a_fro, delta_b_fro, cos_ab)
per (step, block) for richer signal.

Targets per render: 11 NMF-atom coefficients computed by projecting the
measured 39-channel blendshape vector onto the canonical basis
(W_nmf_k11.npy).

Purpose: (a) quantify how predictable each atom is from the observed
edit geometry, (b) localize which DiT blocks/steps drive each atom.
This does NOT yet produce FluxSpace-usable edit directions — that
requires fitting on the full 3072-d delta_mix.mean_d vectors, which is
Phase 3-proper. This diagnostic gates whether to go there.
"""

from __future__ import annotations

import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "output/demographic_pc/fluxspace_metrics"
NMF_DIR = ROOT / "models/blendshape_nmf"
OUT_DIR = ROOT / "models/blendshape_nmf"
ANALYSIS_DIR = METRICS / "analysis"

# (bs_json, attn_dir, tag) triples — only sources that have BOTH scored
# images AND captured attention pkls.
PAIRED_SOURCES = [
    (METRICS / "crossdemo/smile/smile_inphase/blendshapes.json",
     METRICS / "crossdemo/smile/smile_inphase", "smile_inphase"),
    (METRICS / "crossdemo/smile/jaw_inphase/blendshapes.json",
     METRICS / "crossdemo/smile/jaw_inphase", "jaw_inphase"),
    (METRICS / "crossdemo/smile/alpha_interp_attn/blendshapes.json",
     METRICS / "crossdemo/smile/alpha_interp_attn", "alpha_interp_attn"),
]

FNAME_RE = re.compile(r"^s(?P<seed>\d+)_a(?P<alpha>[0-9.]+)\.png$")


def load_nmf_basis():
    W = np.load(NMF_DIR / "W_nmf_k11.npy")
    meta = json.loads((NMF_DIR / "manifest.json").read_text())
    channels = meta["channels"]
    return W, channels


def bs_to_vec(scores: dict, channels: list[str]) -> np.ndarray:
    return np.array([scores.get(c, 0.0) for c in channels], dtype=np.float64)


def project_to_atoms(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Least-squares projection of data (N, C) onto basis (k, C).
    Returns (N, k) non-negative atom coefficients.

    For the canonical NMF basis W (non-negative), we use the standard
    right-pseudoinverse: H = X W⁺. Project → clamp to non-negative to
    preserve NMF semantics.
    """
    H = X @ np.linalg.pinv(W)  # X (N, C) @ pinv (C, k) → (N, k)
    return np.clip(H, 0.0, None)


def extract_features(pkl_path: Path) -> np.ndarray | None:
    """Per-(step, block) scalar summary features → 1D array.

    Features per (step, block): [delta_mix_fro, attn_base_fro,
    attn_base_max_abs, delta_a_fro, delta_b_fro, cos_ab] = 6 scalars.
    Stable block ordering via sorted names.
    """
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except Exception:
        return None
    steps = data.get("steps", {})
    if not steps:
        return None
    step_keys = sorted(steps.keys())
    # Determine block names from first step; assume consistent
    first = steps[step_keys[0]]
    block_names = sorted(first.keys())
    feats = []
    for sk in step_keys:
        step_data = steps[sk]
        for bn in block_names:
            b = step_data.get(bn, {})
            base = b.get("attn_base", {})
            dm = b.get("delta_mix", {})
            feats.extend([
                float(dm.get("fro", 0.0)) if isinstance(dm, dict) else 0.0,
                float(base.get("fro", 0.0)) if isinstance(base, dict) else 0.0,
                float(base.get("max_abs", 0.0)) if isinstance(base, dict) else 0.0,
                float(b.get("delta_a_fro", 0.0)),
                float(b.get("delta_b_fro", 0.0)),
                float(b.get("cos_ab", 0.0)),
            ])
    return np.array(feats, dtype=np.float64)


def build_dataset(W: np.ndarray, channels: list[str]):
    """Walk all paired sources, collect (features, atom_coeffs, metadata)."""
    X_feat = []
    Y_atoms = []
    meta_rows = []
    for bs_json, attn_dir, tag in PAIRED_SOURCES:
        if not bs_json.exists():
            print(f"  [skip] {tag}: missing {bs_json.name}")
            continue
        scores_by_rel = json.loads(bs_json.read_text())
        n_paired = 0
        n_skipped = 0
        for rel, scores in scores_by_rel.items():
            png_path = attn_dir / rel
            pkl_path = png_path.with_suffix(".pkl")
            if not pkl_path.exists():
                n_skipped += 1
                continue
            f = extract_features(pkl_path)
            if f is None:
                n_skipped += 1
                continue
            bs_vec = bs_to_vec(scores, channels)  # (C,)
            atom_vec = project_to_atoms(bs_vec[None, :], W)[0]  # (k,)
            X_feat.append(f)
            Y_atoms.append(atom_vec)
            # Parse base/seed/alpha from rel = base/stem.png
            base_name, fname = rel.split("/")
            m = FNAME_RE.match(fname)
            meta_rows.append({
                "tag": tag, "base": base_name,
                "seed": int(m.group("seed")) if m else -1,
                "alpha": float(m.group("alpha")) if m else 0.0,
                "rel": rel,
            })
            n_paired += 1
        print(f"  [pair] {tag}: {n_paired} paired, {n_skipped} skipped")
    X = np.stack(X_feat)
    Y = np.stack(Y_atoms)
    return X, Y, meta_rows


def fit_cv_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0, n_splits: int = 5) -> dict:
    """Fit ridge with cross-validation, return mean test R² and trained
    weights on full data."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    r2_scores = []
    for tr, te in kf.split(X):
        m = Ridge(alpha=alpha, random_state=0)
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        ss_res = ((y[te] - pred) ** 2).sum()
        ss_tot = ((y[te] - y[te].mean()) ** 2).sum() + 1e-12
        r2_scores.append(float(1.0 - ss_res / ss_tot))
    final = Ridge(alpha=alpha, random_state=0).fit(X, y)
    return {
        "cv_r2_mean": float(np.mean(r2_scores)),
        "cv_r2_std": float(np.std(r2_scores)),
        "cv_r2_folds": r2_scores,
        "weights": final.coef_,
        "intercept": float(final.intercept_),
    }


def main() -> None:
    print("[phase3] loading NMF canonical basis")
    W, channels = load_nmf_basis()
    print(f"  W shape={W.shape}, n_channels={len(channels)}")

    print("\n[phase3] building paired dataset (attention features + atom coeffs)")
    X, Y, meta = build_dataset(W, channels)
    print(f"  dataset: X={X.shape}  Y={Y.shape}  N={len(meta)}")

    # Standardize features (ridge is scale-sensitive)
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-8
    Xn = (X - X_mean) / X_std

    # Atom names from manifest
    manifest = json.loads((NMF_DIR / "manifest.json").read_text())
    atom_names = []
    for a in manifest["nmf_atoms"]:
        top = a.get("top_channels") or []
        atom_names.append(top[0] if top else "(dead)")

    print(f"\n[phase3] ridge CV per atom (k={Y.shape[1]}, alpha=1.0, 5-fold)")
    print(f"  {'#':>3}  {'atom top':<28}  {'baseline σ':>10}  "
          f"{'CV R² mean':>11}  {'CV R² std':>10}")
    results = []
    for j in range(Y.shape[1]):
        res = fit_cv_ridge(Xn, Y[:, j], alpha=1.0)
        top = atom_names[j][:26] if j < len(atom_names) else f"atom{j}"
        sigma = float(Y[:, j].std())
        print(f"  {j:>3}  {top:<28}  {sigma:>10.4f}  "
              f"{res['cv_r2_mean']:>11.3f}  {res['cv_r2_std']:>10.3f}")
        results.append({
            "atom_index": j,
            "sigma": sigma,
            "cv_r2_mean": res["cv_r2_mean"],
            "cv_r2_std": res["cv_r2_std"],
            "cv_r2_folds": res["cv_r2_folds"],
        })

    # Localization: per atom, which features matter most?
    print(f"\n[phase3] top-5 feature indices per atom (|weight| ranked)")
    # Feature layout: for each (step, block) we have 6 scalars in order
    # [delta_mix_fro, attn_base_fro, attn_base_max_abs, delta_a_fro, delta_b_fro, cos_ab]
    feat_names = ["delta_mix_fro", "attn_base_fro", "attn_base_max_abs",
                  "delta_a_fro", "delta_b_fro", "cos_ab"]
    # We don't have step/block labels from the pkl walk without re-opening one;
    # skip labelling for now — just report feature indices.
    # (Block names are stable across pkls; TODO: wire them through.)
    for j in range(Y.shape[1]):
        w = results[j]["cv_r2_folds"]  # unused here; we refit final below
    # Single final fit on full dataset per atom for weight inspection
    for j in range(Y.shape[1]):
        m = Ridge(alpha=1.0, random_state=0).fit(Xn, Y[:, j])
        w = m.coef_
        top5 = np.argsort(-np.abs(w))[:5]
        top_desc = ", ".join(f"f{int(i)}({w[i]:+.3f})" for i in top5)
        print(f"  atom {j:>2}: {top_desc}")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUT_DIR / "phase3_ridge_report.json"
    save_payload = {
        "dataset": {"N": int(X.shape[0]), "n_features": int(X.shape[1]),
                    "n_atoms": int(Y.shape[1])},
        "n_feature_groups_per_block": len(feat_names),
        "feature_names_per_block": feat_names,
        "atom_results": results,
    }
    save_path.write_text(json.dumps(save_payload, indent=2))
    print(f"\n[phase3] wrote → {save_path}")


if __name__ == "__main__":
    main()
