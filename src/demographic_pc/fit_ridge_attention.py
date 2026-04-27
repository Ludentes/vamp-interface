"""Ridge fit in attention-output space.

Fits per-(block, step) linear map from a structured demographic feature vector
(parsed from calibration filenames) to FluxSpaceBaseMeasure attn_base.mean_d.
This gives a 15-direction basis W in attention space — the ridge analogue of
FluxSpace's text-prompt-pair directions, but:
  * learned from 300 varied samples instead of one prompt pair,
  * directly in the layer where linearity holds,
  * simultaneously covering age × gender × ethnicity × expression.

Outputs:
  * W tensor (K_features, B_blocks, T_steps, D=3072) pickled.
  * Per-(block, step) R² and explained-variance tables.
  * Cosine similarity of each W-direction with the existing glasses/smile
    pair-averaging δ (sanity: does ridge recover the pair direction for
    the axis that has a corresponding categorical feature?).
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CAL_DIR = ROOT / "output" / "demographic_pc" / "fluxspace_metrics" / "calibration_expanded"
OUT_DIR = ROOT / "output" / "demographic_pc" / "fluxspace_metrics" / "ridge_attn"
PAIR_DIR = ROOT / "output" / "demographic_pc" / "fluxspace_metrics" / "crossdemo"

# Categorical levels used by expand_calibration.py. Reference level per
# category is the first entry — dropped from one-hot to avoid collinearity.
AGE_LEVELS = ["young", "adult", "middle-aged", "elderly"]
GENDER_LEVELS = ["m", "f"]
ETHNICITY_LEVELS = ["east_asian", "south_asian", "black", "european",
                    "latin_american", "middle_eastern", "southeast_asian",
                    "mixed_race"]
EXPRESSION_LEVELS = ["neutral", "slight-smile", "pensive", "serious"]

# Filename: "NNN_{age}_{ethnicity_snake}_{gender}_{expr}_s{seed}.pkl"
# The ethnicity may contain underscores (e.g., "east_asian"); we match greedily
# using suffix constraints for gender and expression.
FNAME_RE = re.compile(
    r"^(?P<idx>\d{3})_(?P<age>young|adult|middle-aged|elderly)_"
    r"(?P<eth>.+)_(?P<gender>m|f)_(?P<expr>neutral|slight-smile|pensive|serious)_"
    r"s(?P<seed>\d+)$"
)


def parse_tag(stem: str) -> dict | None:
    m = FNAME_RE.match(stem)
    return m.groupdict() if m else None


def build_design_matrix(tags: list[dict]) -> tuple[np.ndarray, list[str]]:
    """One-hot encode with reference levels dropped. Returns (X, feature_names)."""
    names = ["intercept"]
    for lvl in AGE_LEVELS[1:]:
        names.append(f"age={lvl}")
    for lvl in GENDER_LEVELS[1:]:
        names.append(f"gender={lvl}")
    for lvl in ETHNICITY_LEVELS[1:]:
        names.append(f"eth={lvl}")
    for lvl in EXPRESSION_LEVELS[1:]:
        names.append(f"expr={lvl}")
    N = len(tags)
    K = len(names)
    X = np.zeros((N, K), dtype=np.float64)
    X[:, 0] = 1.0
    for i, t in enumerate(tags):
        col = 1
        for lvl in AGE_LEVELS[1:]:
            if t["age"] == lvl:
                X[i, col] = 1.0
            col += 1
        for lvl in GENDER_LEVELS[1:]:
            if t["gender"] == lvl:
                X[i, col] = 1.0
            col += 1
        for lvl in ETHNICITY_LEVELS[1:]:
            if t["eth"] == lvl:
                X[i, col] = 1.0
            col += 1
        for lvl in EXPRESSION_LEVELS[1:]:
            if t["expr"] == lvl:
                X[i, col] = 1.0
            col += 1
    return X, names


def load_pkls_to_tensor(pkl_paths: list[Path]) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Stack all calibration pkls into Y of shape (N, B*T, D).
    Returns (Y, keys) where keys = list of (block, step) tuples aligned to the
    middle axis."""
    first = pickle.load(pkl_paths[0].open("rb"))
    keys = [(bk, st) for st, blocks in first["steps"].items() for bk in blocks.keys()]
    keys.sort()
    D = next(iter(first["steps"].values())).__iter__().__next__()
    # Re-get D properly:
    any_step = next(iter(first["steps"]))
    any_block = next(iter(first["steps"][any_step]))
    arr = first["steps"][any_step][any_block]["attn_base"]["mean_d"]
    D = arr.numpy().shape[0] if hasattr(arr, "numpy") else arr.shape[0]
    print(f"[ridge] N={len(pkl_paths)} keys={len(keys)} D={D}")

    Y = np.zeros((len(pkl_paths), len(keys), D), dtype=np.float32)
    for i, p in enumerate(pkl_paths):
        d = pickle.load(p.open("rb"))
        for j, (bk, st) in enumerate(keys):
            entry = d["steps"].get(st, {}).get(bk)
            if entry is None:
                continue
            v = entry["attn_base"]["mean_d"]
            Y[i, j] = v.numpy() if hasattr(v, "numpy") else v
        if (i + 1) % 50 == 0:
            print(f"[ridge] loaded {i+1}/{len(pkl_paths)}")
    return Y, keys


def fit_ridge_per_key(X: np.ndarray, Y: np.ndarray, lam: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Ridge per middle-axis slice: β = (X'X + λI)⁻¹ X'Y.
    X: (N, K), Y: (N, B*T, D) → β: (K, B*T, D), R²: (B*T,).
    Computes R² per (block, step) as 1 - RSS/TSS aggregated over D dims.
    """
    N, K = X.shape
    _, BT, D = Y.shape
    XtX = X.T @ X
    reg = XtX + lam * np.eye(K)
    inv = np.linalg.inv(reg)
    # β_j = inv @ X.T @ Y[:, j, :]; stack across j.
    # Y_flat: (N, BT*D); XtY: (K, BT*D); β: (K, BT*D)
    Y_flat = Y.reshape(N, BT * D)
    XtY = X.T @ Y_flat
    B_flat = inv @ XtY
    B = B_flat.reshape(K, BT, D)

    # R² per (block, step)
    Y_pred = X @ B_flat  # (N, BT*D)
    residual = Y_flat - Y_pred
    rss = (residual.reshape(N, BT, D) ** 2).sum(axis=(0, 2))  # (BT,)
    tss = ((Y_flat - Y_flat.mean(axis=0, keepdims=True)).reshape(N, BT, D) ** 2).sum(axis=(0, 2))
    r2 = 1.0 - rss / np.maximum(tss, 1e-12)
    return B, r2


def load_pair_delta(pkl_path: Path) -> dict[tuple[str, int], np.ndarray]:
    """Extract delta_mix per (block, step) from a cross-demo measurement pkl."""
    d = pickle.load(pkl_path.open("rb"))
    out = {}
    for st, blocks in d["steps"].items():
        for bk, e in blocks.items():
            v = e.get("delta_mix", {}).get("mean_d")
            if v is None:
                continue
            arr = v.numpy() if hasattr(v, "numpy") else v
            out[(bk, st)] = arr.astype(np.float32)
    return out


def cos_direction_vs_pair(W_dir: np.ndarray, keys: list, pair: dict) -> dict:
    """Cosine(W_dir[b,t], pair[b,t]) aggregated; returns mean, p95, per-key stats."""
    cos_list = []
    for j, key in enumerate(keys):
        d = pair.get(key)
        if d is None:
            continue
        w = W_dir[j]
        num = float(w @ d)
        den = float(np.linalg.norm(w) * np.linalg.norm(d) + 1e-12)
        cos_list.append(num / den)
    a = np.array(cos_list)
    return {
        "n_compared": len(a),
        "mean": float(a.mean()) if len(a) else None,
        "abs_mean": float(np.abs(a).mean()) if len(a) else None,
        "abs_p95": float(np.percentile(np.abs(a), 95)) if len(a) else None,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pkl_paths = sorted(CAL_DIR.glob("*.pkl"))
    png_paths = sorted(CAL_DIR.glob("*.png"))
    print(f"[ridge] {len(pkl_paths)} pkls, {len(png_paths)} pngs")

    tags, kept_paths = [], []
    for p in pkl_paths:
        t = parse_tag(p.stem)
        if t is None:
            print(f"  skip unparseable: {p.stem}")
            continue
        tags.append(t)
        kept_paths.append(p)
    print(f"[ridge] parsed {len(tags)} tags")

    X, feat_names = build_design_matrix(tags)
    print(f"[ridge] X: {X.shape}, features: {feat_names}")

    Y, keys = load_pkls_to_tensor(kept_paths)
    print(f"[ridge] Y: {Y.shape}")

    # Ridge fit with modest regularisation
    B, r2 = fit_ridge_per_key(X, Y, lam=1.0)
    print(f"[ridge] β: {B.shape}  R² (BT): min={r2.min():.3f} mean={r2.mean():.3f} "
          f"median={float(np.median(r2)):.3f} max={r2.max():.3f}")

    # Per-feature direction norm aggregated across keys
    print("\n[ridge] per-feature ‖β[k]‖ (mean over block×step, RMS over D):")
    for k, name in enumerate(feat_names):
        norm_bt = np.linalg.norm(B[k], axis=-1)  # (BT,)
        print(f"  {name:>20}  mean={norm_bt.mean():.4f}  p95={np.percentile(norm_bt, 95):.4f}")

    # Compare ridge expression=slight-smile direction to pair-averaged smile δ.
    smile_pair = PAIR_DIR / "smile" / "measurement" / "asian_m_meas.pkl"
    glasses_pair = PAIR_DIR / "measurement" / "asian_m_meas.pkl"
    for axis_label, pair_pkl in [("smile_pair(asian_m)", smile_pair),
                                  ("glasses_pair(asian_m)", glasses_pair)]:
        if not pair_pkl.exists():
            print(f"\n[ridge] missing {pair_pkl}, skipping cos comparison")
            continue
        pair = load_pair_delta(pair_pkl)
        print(f"\n[ridge] cos(ridge_feature, pair_delta) vs {axis_label}:")
        for k, name in enumerate(feat_names):
            stats = cos_direction_vs_pair(B[k], keys, pair)
            if stats["abs_p95"] is None:
                continue
            print(f"  {name:>20}  |cos|_p95={stats['abs_p95']:.3f}  mean={stats['mean']:+.3f}")

    # Save
    out = {
        "feature_names": feat_names,
        "keys": keys,
        "W": B.astype(np.float32),  # (K, BT, D)
        "r2_per_key": r2,
        "lam": 1.0,
        "n_samples": len(tags),
    }
    with (OUT_DIR / "ridge_basis.pkl").open("wb") as f:
        pickle.dump(out, f)
    print(f"\n[ridge] saved → {OUT_DIR / 'ridge_basis.pkl'}")


if __name__ == "__main__":
    main()
