"""Stage 4 — regression + demographic-subspace extraction.

For each classifier head:
  - ridge regression (continuous targets: mivolo_age, insightface_age)
  - multinomial logistic (categorical: fairface_race, *_gender, fairface_age_bin)
Extract the top-k right singular vectors of the fitted weight matrix per head.
Stack across heads → truncated SVD → retain components until cumulative variance > 0.90.
That orthonormal basis is the demographic subspace D used for Level-0 orthogonalization.

Also reports:
  - per-head CV score (R² / accuracy)
  - cosine similarities between per-classifier same-attribute directions
    (the FluxSpace-borrowed cross-classifier check)

Outputs:
  output/demographic_pc/
    direction_matrix.npy         stacked W, shape (K, 4864), float32
    direction_matrix_rows.json   metadata per row (head, class_idx, singular_value)
    subspace_D.npy               orthonormal basis, shape (d, 4864)
    subspace_report.json         variance-explained, per-head metrics, cross-cos

Usage:
    uv run python -m src.demographic_pc.stage4_regression
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

CV_SEED = 0
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output" / "demographic_pc"
COND_PATH = OUT_DIR / "conditioning.npy"
IDS_PATH = OUT_DIR / "conditioning_ids.json"
LABELS_PATH = OUT_DIR / "labels.parquet"

TOP_K_PER_HEAD = 5            # right singular vectors retained per head
VAR_EXPLAINED_TARGET = 0.90   # cumulative variance threshold for D
MIN_R2_CONTINUOUS = 0.3       # drop continuous head if CV R² below this
MIN_ACC_CATEGORICAL = 0.4     # drop categorical head if CV accuracy below this


def continuous_head(X: np.ndarray, y: np.ndarray, name: str) -> dict | None:
    mask = ~np.isnan(y)
    Xm, ym = X[mask], y[mask]
    model = RidgeCV(alphas=np.logspace(-2, 4, 13))
    cv = KFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    r2_cv = cross_val_score(model, Xm, ym, cv=cv, scoring="r2").mean()
    model.fit(Xm, ym)
    w = model.coef_.reshape(1, -1)  # (1, 4864)
    print(f"  [{name}] n={mask.sum()}  CV R²={r2_cv:.3f}  alpha={model.alpha_:.3g}")
    if r2_cv < MIN_R2_CONTINUOUS:
        print(f"    DROP (R² below {MIN_R2_CONTINUOUS})")
        return None
    # For a 1-row W, "right singular vectors" = w / ||w||
    U, S, Vt = np.linalg.svd(w, full_matrices=False)
    return {
        "name": name, "kind": "continuous",
        "cv_score": float(r2_cv), "n": int(mask.sum()),
        "W": w, "U": U, "S": S, "Vt": Vt,   # Vt rows are the directions
    }


def categorical_head(X: np.ndarray, y_raw, name: str) -> dict | None:
    # drop None/NaN
    y_raw = np.asarray(y_raw, dtype=object)
    mask = np.array([v is not None and not (isinstance(v, float) and np.isnan(v)) for v in y_raw])
    if mask.sum() < 20:
        print(f"  [{name}] too few labeled rows ({mask.sum()}), skipping")
        return None
    Xm = X[mask]
    ym = y_raw[mask].astype(str)
    n_classes = len(np.unique(ym))
    if n_classes < 2:
        print(f"  [{name}] only 1 class, skipping")
        return None
    model = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    try:
        acc_cv = cross_val_score(model, Xm, ym, cv=cv, scoring="accuracy").mean()
    except Exception as e:
        print(f"  [{name}] CV failed: {e}")
        return None
    model.fit(Xm, ym)
    # For binary, coef_ is (1, 4864); for multiclass (C, 4864).
    w = model.coef_
    print(f"  [{name}] n={mask.sum()}  classes={list(model.classes_)}  CV acc={acc_cv:.3f}")
    if acc_cv < MIN_ACC_CATEGORICAL:
        print(f"    DROP (acc below {MIN_ACC_CATEGORICAL})")
        return None
    U, S, Vt = np.linalg.svd(w, full_matrices=False)
    return {
        "name": name, "kind": "categorical",
        "cv_score": float(acc_cv), "n": int(mask.sum()),
        "classes": list(model.classes_),
        "W": w, "U": U, "S": S, "Vt": Vt,
    }


def main() -> None:
    print("[stage4] loading conditioning + labels")
    C = np.load(COND_PATH)                          # (1785, 4864)
    ids = json.loads(IDS_PATH.read_text())
    df = pd.read_parquet(LABELS_PATH).set_index("sample_id").reindex(ids)
    assert len(df) == len(C), (len(df), len(C))
    print(f"  conditioning: {C.shape}  labels: {df.shape}")

    # Standardize features — ridge/logistic both benefit
    scaler = StandardScaler().fit(C)
    Xn = scaler.transform(C)

    heads: list[dict] = []
    print("\n[stage4] fitting heads")

    # Continuous
    heads_maybe = [
        continuous_head(Xn, df.mivolo_age.to_numpy(dtype=float), "mivolo_age"),
        continuous_head(Xn, df.insightface_age.to_numpy(dtype=float), "insightface_age"),
    ]
    # Categorical
    heads_maybe.extend([
        categorical_head(Xn, df.mivolo_gender.values, "mivolo_gender"),
        categorical_head(Xn, df.fairface_gender.values, "fairface_gender"),
        categorical_head(Xn, df.insightface_gender.values, "insightface_gender"),
        categorical_head(Xn, df.fairface_age_bin.values, "fairface_age_bin"),
        categorical_head(Xn, df.fairface_race.values, "fairface_race"),
    ])
    heads = [h for h in heads_maybe if h is not None]
    print(f"\n[stage4] kept {len(heads)} heads")

    # Stack top-k right singular vectors weighted by singular value
    rows, meta_rows = [], []
    for h in heads:
        k = min(TOP_K_PER_HEAD, h["Vt"].shape[0])
        for j in range(k):
            rows.append(h["Vt"][j] * h["S"][j])
            meta_rows.append({
                "head": h["name"], "kind": h["kind"],
                "class_idx": j, "singular_value": float(h["S"][j]),
                "cv_score": h["cv_score"],
            })
    W_stack = np.stack(rows, axis=0).astype(np.float32)  # (K, 4864)
    print(f"\n[stage4] stacked W: {W_stack.shape}")

    # Truncated SVD of W_stack → demographic subspace D
    Uw, Sw, Vtw = np.linalg.svd(W_stack, full_matrices=False)
    var = (Sw ** 2)
    var /= var.sum()
    cum = np.cumsum(var)
    d = int(np.searchsorted(cum, VAR_EXPLAINED_TARGET) + 1)
    D = Vtw[:d].astype(np.float32)   # (d, 4864), orthonormal rows
    print(f"  d={d}  cum_var@d={cum[d-1]:.3f}  total singular values: {len(Sw)}")

    # Cross-classifier cosine (same attribute, different classifier)
    def head_basis(head_name: str) -> np.ndarray | None:
        for h in heads:
            if h["name"] == head_name:
                return h["Vt"][: min(TOP_K_PER_HEAD, h["Vt"].shape[0])]
        return None

    def subspace_cos(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
        if a is None or b is None:
            return None
        # Principal angle: largest singular value of a @ b.T (both orthonormal)
        M = a @ b.T
        s = np.linalg.svd(M, compute_uv=False)
        return float(s[0])

    cross_cos = {
        "age_mivolo_vs_insightface": subspace_cos(head_basis("mivolo_age"), head_basis("insightface_age")),
        "age_mivolo_vs_fairface_bin": subspace_cos(head_basis("mivolo_age"), head_basis("fairface_age_bin")),
        "age_insightface_vs_fairface_bin": subspace_cos(head_basis("insightface_age"), head_basis("fairface_age_bin")),
        "gender_mivolo_vs_fairface": subspace_cos(head_basis("mivolo_gender"), head_basis("fairface_gender")),
        "gender_mivolo_vs_insightface": subspace_cos(head_basis("mivolo_gender"), head_basis("insightface_gender")),
        "gender_fairface_vs_insightface": subspace_cos(head_basis("fairface_gender"), head_basis("insightface_gender")),
    }
    print("\n[stage4] cross-classifier principal-angle cosines (top singular of A@B.T, 1.0 = coplanar):")
    for k, v in cross_cos.items():
        print(f"  {k:40s} {'None' if v is None else f'{v:.3f}'}")

    # Save outputs
    np.save(OUT_DIR / "direction_matrix.npy", W_stack)
    with open(OUT_DIR / "direction_matrix_rows.json", "w") as f:
        json.dump(meta_rows, f, indent=2)
    np.save(OUT_DIR / "subspace_D.npy", D)

    report = {
        "conditioning_shape": list(C.shape),
        "n_heads_kept": len(heads),
        "heads": [
            {"name": h["name"], "kind": h["kind"], "cv_score": h["cv_score"],
             "n": h["n"], "classes": h.get("classes")}
            for h in heads
        ],
        "stack_shape": list(W_stack.shape),
        "subspace_dim": d,
        "cum_var_at_d": float(cum[d - 1]),
        "variance_spectrum": var.tolist(),
        "cross_classifier_cos": cross_cos,
    }
    with open(OUT_DIR / "subspace_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[stage4] wrote direction_matrix.npy, subspace_D.npy (d={d}), subspace_report.json")


if __name__ == "__main__":
    main()
