"""Stage 4.5 — evaluate Ours vs FluxSpace-coarse on 20 × 5 λ grid.

Metrics (all reported at *matched target-slope*, i.e. after rescaling each
method's λ axis so that d(MiVOLO_age)/dλ_effective = 1 year / unit):

  - on-axis slope (raw):        d(MiVOLO_age) / dλ, per portrait, averaged.
  - identity drift:             1 − cos(ArcFace_IR101(baseline), ArcFace(edit))
                                averaged across portraits at each normalised |λ|.
  - Attribute Dependency (AD):  |slope_k(λ_eff)| for each off-axis continuous head
                                (gender-prob, race-prob), normalised so on-axis
                                slope = 1. Report max and mean across k.
  - off-axis flip rate:         fraction of portraits whose argmax(gender) or
                                argmax(race) differs between λ=0 and λ_max.

Outputs:
  output/demographic_pc/stage4_5/eval.parquet       per-render classifier rows
  output/demographic_pc/stage4_5/eval_summary.json  metric table

Usage:
    uv run python -m src.demographic_pc.stage4_5_evaluate
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from torchvision import transforms

from src.demographic_pc.classifiers import (
    FairFaceClassifier,
    InsightFaceClassifier,
    MiVOLOClassifier,
    predict_all,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output" / "demographic_pc" / "stage4_5"
MANIFEST = OUT_DIR / "render_manifest.json"
EVAL_PARQUET = OUT_DIR / "eval.parquet"
EVAL_JSON = OUT_DIR / "eval_summary.json"

DEMO_PC_DIR = ROOT / "output" / "demographic_pc"
CONDITIONING_NPY = DEMO_PC_DIR / "conditioning.npy"
EDITS_DIR = DEMO_PC_DIR / "edits"
CLIP_DIM = 768

ARCFACE_HF = "minchul/cvlface_arcface_ir101_webface4m"
ARCFACE_SIZE = 112

FAIRFACE_GENDER = ["Male", "Female"]
FAIRFACE_RACE = ["White", "Black", "Latino_Hispanic", "East Asian",
                 "Southeast Asian", "Indian", "Middle Eastern"]


def load_arcface():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    local = snapshot_download(ARCFACE_HF)
    old = os.getcwd()
    sys.path.insert(0, local)
    os.chdir(local)
    try:
        from wrapper import ModelConfig, CVLFaceRecognitionModel  # type: ignore
        model = CVLFaceRecognitionModel(ModelConfig()).eval().to(device)
    finally:
        os.chdir(old)
        if local in sys.path:
            sys.path.remove(local)
    return model, device


_ARC_TF = transforms.Compose([
    transforms.Resize((ARCFACE_SIZE, ARCFACE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def arcface_embed(model, device, path: Path) -> np.ndarray:
    t = _ARC_TF(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        e = model(t)
    e = e / e.norm(p=2, dim=-1, keepdim=True)
    return e.squeeze(0).float().cpu().numpy()


def mahalanobis_direction_analysis() -> dict:
    """Compute Mahalanobis norm of each method's edit direction under the
    training-conditioning covariance. Low value → direction aligned with
    high-variance (manifold) axes; high value → direction exits the manifold fast.

    Uses Ledoit-Wolf shrinkage for Σ since with N=1785, p=4864 the sample
    covariance is rank-deficient. The shift from any base by s·w has
    squared Mahalanobis norm s² · wᵀ Σ⁻¹ w, so sqrt(wᵀ Σ⁻¹ w) is the
    per-unit-strength Mahalanobis step size.
    """
    from sklearn.covariance import LedoitWolf

    C = np.load(CONDITIONING_NPY).astype(np.float64)  # (1785, 4864)
    mu = C.mean(axis=0)
    Cc = C - mu
    print(f"[mahal] fitting Ledoit-Wolf on ({C.shape[0]}, {C.shape[1]})…")
    lw = LedoitWolf(store_precision=True).fit(Cc)
    prec = lw.precision_  # Σ⁻¹ (p, p)
    print(f"[mahal]   shrinkage α = {lw.shrinkage_:.4f}")

    eucl_mu_offset = float(np.sqrt((mu @ prec @ mu)))  # d_M(0, μ) sanity check; should be 0

    results: dict = {"shrinkage": float(lw.shrinkage_)}
    for method, fname in [("ours", "age_ours.npz"), ("fluxspace", "age_fluxspace_coarse.npz")]:
        data = np.load(EDITS_DIR / fname)
        pooled_delta = data["pooled_delta"].astype(np.float64)    # (768,)
        seq_delta = data["seq_delta"].astype(np.float64)          # (4096,)
        # Ambient-space shift vector: pooled (768) concat (4096)
        # since stored conditioning = concat[CLIP-pool, T5-mean].
        w = np.concatenate([pooled_delta, seq_delta])             # (4864,)
        mahal_sq = float(w @ prec @ w)
        mahal_norm = float(np.sqrt(mahal_sq))
        eucl_norm = float(np.linalg.norm(w))
        pooled_norm = float(np.linalg.norm(pooled_delta))
        seq_norm = float(np.linalg.norm(seq_delta))
        results[method] = {
            "euclidean_norm_4864": eucl_norm,
            "pooled_norm_768": pooled_norm,
            "seq_norm_4096": seq_norm,
            "mahalanobis_norm_per_unit_strength": mahal_norm,
            "mahalanobis_over_euclidean": mahal_norm / max(eucl_norm, 1e-9),
            # ratio of Mahalanobis to Euclidean — low → direction in high-variance subspace;
            # high → direction crosses many low-variance (off-manifold) axes.
        }

    # Report ratio between methods (the core theoretical prediction)
    r_ours = results["ours"]["mahalanobis_over_euclidean"]
    r_flux = results["fluxspace"]["mahalanobis_over_euclidean"]
    results["flux_over_ours_mahal_per_eucl"] = r_flux / max(r_ours, 1e-9)
    print(f"[mahal] Ours      √(wᵀΣ⁻¹w) = {results['ours']['mahalanobis_norm_per_unit_strength']:.4g}   (eucl {results['ours']['euclidean_norm_4864']:.4g})")
    print(f"[mahal] FluxSpace √(wᵀΣ⁻¹w) = {results['fluxspace']['mahalanobis_norm_per_unit_strength']:.4g}   (eucl {results['fluxspace']['euclidean_norm_4864']:.4g})")
    print(f"[mahal] Mahal/Eucl ratio: ours={r_ours:.4g}  flux={r_flux:.4g}  (flux/ours = {r_flux/max(r_ours,1e-9):.3g}×)")
    print(f"[mahal] mu·Σ⁻¹·mu offset (sanity): {eucl_mu_offset:.3g}")
    return results


def classify_all(manifest: list[dict]) -> pd.DataFrame:
    print("[eval] loading classifiers…")
    mv = MiVOLOClassifier()
    ff = FairFaceClassifier()
    ins = InsightFaceClassifier()
    print("[eval] loading ArcFace IR101…")
    arc_model, arc_dev = load_arcface()

    rows: list[dict] = []
    t0 = time.time()
    for i, j in enumerate(manifest, 1):
        p = Path(j["dest"])
        if not p.exists():
            print(f"[eval] MISSING {p}")
            continue
        bgr = cv2.imread(str(p))
        rec = predict_all(bgr, mv, ff, ins)
        emb = arcface_embed(arc_model, arc_dev, p)
        rows.append({
            "portrait_id": j["portrait_id"],
            "method": j["method"],
            "lam": float(j["lam"]),
            "strength": float(j["strength"]),
            "mivolo_age": rec.mivolo_age,
            "mivolo_gender": rec.mivolo_gender,
            "fairface_gender_probs": rec.fairface_gender_probs,
            "fairface_race_probs": rec.fairface_race_probs,
            "fairface_race": rec.fairface_race,
            "fairface_gender": rec.fairface_gender,
            "insightface_age": rec.insightface_age,
            "arcface_emb": emb,
        })
        if i % 20 == 0 or i == len(manifest):
            dt = time.time() - t0
            print(f"  [{i}/{len(manifest)}] rate={i/dt:.2f}/s")
    return pd.DataFrame.from_records(rows)


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    """Global linear slope via polyfit."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return float("nan")
    return float(np.polyfit(x[m], y[m], 1)[0])


def _local_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Slope near λ=0 via central difference between the two smallest non-zero λ."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    pos = x > 0
    neg = x < 0
    if not pos.any() or not neg.any():
        return float("nan")
    x_p = x[pos].min()
    x_n = x[neg].max()  # largest (closest to 0) negative
    y_p = y[pos][x[pos] == x_p][0]
    y_n = y[neg][x[neg] == x_n][0]
    return float((y_p - y_n) / (x_p - x_n))


def _linear_r2(x: np.ndarray, y: np.ndarray) -> float:
    """R² of linear fit — how linear is y(λ)?"""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return float("nan")
    a, b = np.polyfit(x, y, 1)
    y_hat = a * x + b
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _inner_outer_slopes(x: np.ndarray, y: np.ndarray, inner_abs: float = 1.0) -> tuple[float, float]:
    """Slopes on |λ|≤inner_abs vs |λ|>inner_abs — ratio diagnoses saturation."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    inner = np.abs(x) <= inner_abs
    outer = np.abs(x) > inner_abs
    s_in = float(np.polyfit(x[inner], y[inner], 1)[0]) if inner.sum() >= 2 else float("nan")
    s_out = float(np.polyfit(x[outer], y[outer], 1)[0]) if outer.sum() >= 2 else float("nan")
    return s_in, s_out


def summarize(df: pd.DataFrame) -> dict:
    # For each method, compute per-portrait slope on the method's own λ grid
    # (baseline=λ=0 is method="baseline" and shared — merge it into each method).
    baseline = df[df.method == "baseline"].set_index("portrait_id")
    out: dict = {"methods": {}}

    for method in ["ours", "fluxspace"]:
        mdf = df[df.method == method].copy()
        # Add baseline rows as λ=0 for this method
        b = baseline.reset_index().assign(method=method, lam=0.0, strength=0.0)
        mdf = pd.concat([mdf, b[mdf.columns]], ignore_index=True)

        # Per-portrait slopes — both global (linear assumption) and local
        # (tangent at λ=0, robust to saturation/sigmoid shape).
        age_slopes_global: list[float] = []
        age_slopes_local: list[float] = []
        age_linear_r2: list[float] = []
        age_inner, age_outer = [], []
        # For AD we use signed continuous scores:
        #   gender: P(male) − P(female)  (∈ [−1,1])
        #   race: a 7-vector of per-class probs; we track each class slope
        male_slopes_local, race_slopes_local = [], []
        id_drifts: dict[float, list[float]] = {}

        for _pid, g in mdf.groupby("portrait_id"):
            g = g.sort_values("lam")
            lam = g.lam.to_numpy()
            age = g.mivolo_age.to_numpy(dtype=float)
            age_slopes_global.append(_slope(lam, age))
            age_slopes_local.append(_local_slope(lam, age))
            age_linear_r2.append(_linear_r2(lam, age))
            si, so = _inner_outer_slopes(lam, age, inner_abs=1.0)
            age_inner.append(si); age_outer.append(so)

            # FairFace probs can be None when face-detect failed; replace with NaN vectors
            def _pad(vals, dim):
                out = np.full((len(vals), dim), np.nan, dtype=float)
                for i, v in enumerate(vals):
                    if v is not None and hasattr(v, "__len__") and len(v) == dim:
                        out[i] = np.asarray(v, dtype=float)
                return out

            ff_gp = _pad(g.fairface_gender_probs.to_list(), 2)
            male_minus_female = ff_gp[:, 0] - ff_gp[:, 1]
            male_slopes_local.append(_local_slope(lam, male_minus_female))

            ff_rp = _pad(g.fairface_race_probs.to_list(), 7)
            race_slopes_local.append(
                np.array([_local_slope(lam, ff_rp[:, k]) for k in range(7)])
            )

            # ArcFace identity drift vs λ=0
            base_row = g[g.lam == 0.0]
            if len(base_row) == 0:
                continue
            base_emb = np.asarray(base_row.arcface_emb.iloc[0], dtype=float)
            for _, row in g.iterrows():
                if row.lam == 0.0:
                    continue
                e = row.arcface_emb
                if e is None:
                    continue
                cos = float(np.dot(base_emb, np.asarray(e, dtype=float)))
                id_drifts.setdefault(float(row.lam), []).append(1.0 - cos)

        age_slope_global = float(np.nanmean(age_slopes_global))
        age_slope_local = float(np.nanmean(age_slopes_local))
        age_lin_r2_mean = float(np.nanmean(age_linear_r2))
        inner_mean = float(np.nanmean(age_inner))
        outer_mean = float(np.nanmean(age_outer))
        saturation_ratio = outer_mean / inner_mean if inner_mean else float("nan")

        male_slope_mean = float(np.nanmean(male_slopes_local))
        race_slope_mean = np.nanmean(np.stack(race_slopes_local), axis=0)  # (7,)

        # AD at matched target-slope: use local (tangent) slopes so the comparison
        # is meaningful even if response saturates at extreme λ.
        ad_gender = abs(male_slope_mean) / max(abs(age_slope_local), 1e-9)
        ad_race = np.abs(race_slope_mean) / max(abs(age_slope_local), 1e-9)

        # Off-axis flip rates at extreme |λ|
        extreme = mdf.lam.abs().max()
        ext = mdf[mdf.lam.abs() == extreme]
        base_gender = baseline.fairface_gender.to_dict()
        base_race = baseline.fairface_race.to_dict()
        g_flip = float(np.mean([
            r.fairface_gender != base_gender.get(r.portrait_id)
            for _, r in ext.iterrows()
        ]))
        r_flip = float(np.mean([
            r.fairface_race != base_race.get(r.portrait_id)
            for _, r in ext.iterrows()
        ]))

        id_drift_mean = {str(k): float(np.mean(v)) for k, v in sorted(id_drifts.items())}

        out["methods"][method] = {
            "on_axis_age_slope_local": age_slope_local,     # tangent at λ=0
            "on_axis_age_slope_global": age_slope_global,   # polyfit over full λ range
            "linearity_r2_mean": age_lin_r2_mean,           # 1.0 = perfectly linear
            "slope_inner_abs_lam_le_1": inner_mean,
            "slope_outer_abs_lam_gt_1": outer_mean,
            "saturation_ratio_outer_over_inner": saturation_ratio,  # <1 = saturating, >1 = accelerating
            "off_axis": {
                "AD_gender_male_minus_female": ad_gender,
                "AD_race_max": float(ad_race.max()),
                "AD_race_mean": float(ad_race.mean()),
                "AD_race_by_class": {FAIRFACE_RACE[k]: float(ad_race[k]) for k in range(7)},
            },
            "flip_rate_at_lam_extreme": {
                "lam": float(extreme),
                "gender": g_flip,
                "race": r_flip,
            },
            "identity_drift_by_lam": id_drift_mean,
        }
    return out


def main() -> None:
    # Mahalanobis is pure direction analysis — no renders needed, do it first.
    print("\n== Mahalanobis direction analysis ==")
    mahal = mahalanobis_direction_analysis()

    manifest = json.loads(MANIFEST.read_text())
    print(f"\n[eval] manifest entries: {len(manifest)}")
    df = classify_all(manifest)
    df.to_parquet(EVAL_PARQUET, index=False)
    print(f"[eval] wrote {EVAL_PARQUET}  rows={len(df)}")

    summary = summarize(df)
    summary["mahalanobis"] = mahal
    EVAL_JSON.write_text(json.dumps(summary, indent=2, default=float))
    print(f"[eval] wrote {EVAL_JSON}")
    for method, s in summary["methods"].items():
        print(f"\n--- {method} ---")
        print(f"  age slope local (yr/λ, tangent):  {s['on_axis_age_slope_local']:+.3f}")
        print(f"  age slope global (yr/λ, linear):  {s['on_axis_age_slope_global']:+.3f}")
        print(f"  linearity R²:                     {s['linearity_r2_mean']:.3f}")
        print(f"  slope inner(|λ|≤1) / outer(|λ|>1): {s['slope_inner_abs_lam_le_1']:+.2f} / {s['slope_outer_abs_lam_gt_1']:+.2f}  (sat={s['saturation_ratio_outer_over_inner']:.2f})")
        print(f"  AD gender (|slope|/age):    {s['off_axis']['AD_gender_male_minus_female']:.4f}")
        print(f"  AD race max / mean:         {s['off_axis']['AD_race_max']:.4f} / {s['off_axis']['AD_race_mean']:.4f}")
        flip = s["flip_rate_at_lam_extreme"]
        print(f"  flip@|λ|={flip['lam']}:  gender={flip['gender']:.2f}  race={flip['race']:.2f}")
        print(f"  identity drift by λ: {s['identity_drift_by_lam']}")


if __name__ == "__main__":
    main()
