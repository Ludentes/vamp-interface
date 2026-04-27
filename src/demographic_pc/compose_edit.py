"""Composition solver — target vector in vocabulary space → prompt-pair weights.

v0: reads `output/demographic_pc/effect_matrix_dictionary.parquet`, accepts
a target YAML spec (Level 1/2/3 per 2026-04-23-solver-design.md), runs
L1-regularized NNLS over the (readouts × pairs) matrix, outputs the
recommended prompt-pair weights + predicted effect vector + residual
table.

This tool does not render anything. Feed its output to `promptpair_iterate`
(or a future multi-pair render tool) to actually execute the composition.

Usage:
    uv run python -m src.demographic_pc.compose_edit --target specs/target_smile.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[2]
DICTIONARY = ROOT / "output/demographic_pc/effect_matrix_dictionary.parquet"
OUT_ROOT = ROOT / "output/demographic_pc/compose_edit"

# Natural-language intents → (target dict, default constraints). Minimal v0 table.
#
# Note on constraint semantics (2026-04-23):
# Targets and constraints both act on the *predicted slope-sum* — i.e. the
# change in a readout per unit effective scale, summed over picked pairs.
#
# For a readout whose base value at scale=0 is known (e.g. identity_cos_to_base
# starts at 1.0), a constraint like `identity_cos_to_base: {">=": -0.7}` reads
# as: "identity cosine drops by at most 0.7 at unit effective scale."
# Equivalently, you can use the convenience shorthand `identity_drift_abs`
# (value = −slope_identity_cos_to_base; rate of identity loss per unit scale)
# which is always positive and easier to constrain with abs<=.
INTENT_TABLE: dict[str, dict] = {
    "smile": {
        "target": {
            "siglip_smiling_margin": 0.07,
            "bs_mouthSmileLeft": 0.4,
            "bs_mouthSmileRight": 0.4,
        },
        "constraints": {
            "mv_age": {"abs<=": 5.0},
            "identity_cos_to_base": {">=": -0.7},  # identity drifts ≤ 0.7 per unit effective scale
        },
    },
    "anger": {
        "target": {"siglip_angry_margin": 0.08},
        "constraints": {
            "mv_age": {"abs<=": 10.0},
            "identity_cos_to_base": {">=": -0.7},
        },
    },
}


DERIVED_READOUTS = {
    # name → (source_column_in_matrix, transform_fn)
    # identity_drift_abs = |1 - identity_cos| if realized, or −slope if slope-space
    "identity_drift_abs": ("identity_cos_to_base", lambda x: -x),
    "total_drift_abs":    ("siglip_img_cos_to_base", lambda x: -x),
}


def resolve_target(spec: dict) -> tuple[dict, dict]:
    """Resolve Level 1/2/3 spec into (target, constraints)."""
    target: dict = {}
    constraints: dict = {}
    if "intent" in spec:
        entry = INTENT_TABLE.get(spec["intent"].lower())
        if entry is None:
            raise SystemExit(f"unknown intent '{spec['intent']}'; known: {list(INTENT_TABLE)}")
        target.update(entry["target"])
        constraints.update(entry["constraints"])
    if "target" in spec:
        target.update(spec["target"])
    if "constraints" in spec:
        constraints.update(spec["constraints"])
    if not target:
        raise SystemExit("spec must provide intent or target")
    return target, constraints


def build_matrix(df: pd.DataFrame, readouts: list[str]) -> tuple[np.ndarray, list[str]]:
    """Return (readouts × pairs) matrix A and pair labels.

    A pair here = one dictionary row (axis/iter/variant/base). Its column in A
    is the per-unit-scale slope for each readout.
    """
    slope_cols = [f"slope_{r}" for r in readouts]
    missing = [r for r, c in zip(readouts, slope_cols) if c not in df.columns]
    if missing:
        raise SystemExit(f"dictionary missing slope columns: {missing}")
    A = df[slope_cols].to_numpy(dtype=np.float64).T  # (readouts, pairs)
    labels = [f"{r['axis']}/{r['iteration_id'].split('/')[-1]}/{r['variant']}@{r['base']}"
              for _, r in df.iterrows()]
    return A, labels


def _resolve_constraint_value(vec: np.ndarray, readouts: list[str], rname: str) -> tuple[float | None, str]:
    """Return (value, label) for a constraint readout, resolving derived readouts."""
    key = rname.replace("slope_", "").replace("_slope", "")
    if key in DERIVED_READOUTS:
        src, fn = DERIVED_READOUTS[key]
        for j, r in enumerate(readouts):
            if r == src:
                return float(fn(vec[j])), f"{key} (derived)"
        return None, key
    for j, r in enumerate(readouts):
        if r == key or r == rname:
            return float(vec[j]), key
    return None, rname


def check_constraints(vec: np.ndarray, readouts: list[str], constraints: dict) -> list[str]:
    """Return list of violated constraint messages; empty if all pass.

    Values are predicted slope-sums — i.e. change per unit effective scale.
    """
    bad = []
    for rname, rule in constraints.items():
        val, label = _resolve_constraint_value(vec, readouts, rname)
        if val is None:
            bad.append(f"{rname}: constraint references readout not in target set")
            continue
        for op, thr in rule.items():
            if op == ">=" and not (val >= thr):
                bad.append(f"{label} = {val:+.3f} (need ≥ {thr})")
            elif op == "<=" and not (val <= thr):
                bad.append(f"{label} = {val:+.3f} (need ≤ {thr})")
            elif op == "abs<=" and not (abs(val) <= thr):
                bad.append(f"|{label}| = {abs(val):.3f} (need ≤ {thr})")
    return bad


def compose(target: dict, constraints: dict,
            lambda_l1: float = 0.1, max_pairs: int = 3,
            scale_cap: float = 1.0,
            base: str | None = None) -> dict:
    df = pd.read_parquet(DICTIONARY)
    if base is not None:
        df = df[df["base"] == base].reset_index(drop=True)
        if len(df) == 0:
            raise SystemExit(f"no dictionary entries for base '{base}'")
    # Build readout set: target keys + constraint keys (resolving derived names to their source).
    readouts_needed = list(target)
    for rname in constraints:
        key = rname.replace("slope_", "").replace("_slope", "")
        if key in DERIVED_READOUTS:
            readouts_needed.append(DERIVED_READOUTS[key][0])
        else:
            readouts_needed.append(key)
    readouts = sorted(set(readouts_needed))
    A, labels = build_matrix(df, readouts)

    t = np.zeros(len(readouts), dtype=np.float64)
    target_mask = np.zeros(len(readouts), dtype=bool)
    for i, r in enumerate(readouts):
        if r in target:
            t[i] = float(target[r])
            target_mask[i] = True

    # only fit on target readouts; constraint readouts are checked after.
    A_fit = A[target_mask]; t_fit = t[target_mask]

    # augmented NNLS with L1 penalty: append sqrt(lambda)*I row and 0 row to A/t
    n_pairs = A_fit.shape[1]
    l1_rows = np.sqrt(lambda_l1) * np.eye(n_pairs)
    A_aug = np.vstack([A_fit, l1_rows])
    t_aug = np.concatenate([t_fit, np.zeros(n_pairs)])
    w, residual = nnls(A_aug, t_aug)

    # scale cap: if any weight is implausibly large, divide the whole w by the max so the effective scale stays within cap
    if w.max() > scale_cap:
        w = w * (scale_cap / w.max())

    # keep top-k by weight
    if max_pairs and (w > 1e-6).sum() > max_pairs:
        keep = np.argsort(w)[-max_pairs:]
        mask = np.zeros_like(w, dtype=bool); mask[keep] = True
        w = np.where(mask, w, 0.0)

    predicted = A @ w
    violations = check_constraints(predicted, readouts, constraints)

    # rank by weight
    nz = np.argsort(w)[::-1]
    nz = [i for i in nz if w[i] > 1e-6]
    picks = []
    for i in nz:
        picks.append({
            "pair": labels[i],
            "weight": float(w[i]),
            "pos_prompt": df.iloc[i]["prompt_pos"],
            "neg_prompt": df.iloc[i]["prompt_neg"],
        })

    return {
        "readouts": readouts,
        "target_vec": t.tolist(),
        "target_mask": target_mask.tolist(),
        "predicted_vec": predicted.tolist(),
        "residual_on_targets": float(np.linalg.norm(A_fit @ w - t_fit)),
        "picks": picks,
        "constraint_violations": violations,
    }


def write_report(spec: dict, result: dict, out_path: Path) -> None:
    readouts = result["readouts"]
    t = result["target_vec"]; tm = result["target_mask"]
    p = result["predicted_vec"]
    lines = [
        "---", "status: live", "topic: metrics-and-direction-quality", "---", "",
        "# Composition — solver output", "",
    ]
    if "intent" in spec:
        lines.append(f"**Intent:** {spec['intent']}")
    lines += ["", "## Target and prediction", "",
              "| readout | target | predicted | Δ |",
              "|---|---|---|---|"]
    for i, r in enumerate(readouts):
        tgt = f"{t[i]:+.3f}" if tm[i] else "—"
        lines.append(f"| {r} | {tgt} | {p[i]:+.3f} | {p[i] - (t[i] if tm[i] else 0):+.3f} |")
    lines += ["", f"Target residual: `{result['residual_on_targets']:.4f}`"]

    if result["constraint_violations"]:
        lines += ["", "### ⚠ constraint violations"]
        for v in result["constraint_violations"]:
            lines.append(f"- {v}")
    else:
        lines += ["", "### ✓ all constraints satisfied"]

    lines += ["", "## Recommended composition", ""]
    if not result["picks"]:
        lines.append("No pairs selected (target may be infeasible with current dictionary).")
    else:
        lines.append("| weight | pair | pos prompt |")
        lines.append("|---|---|---|")
        for p in result["picks"]:
            lines.append(f"| {p['weight']:.3f} | `{p['pair']}` | `{p['pos_prompt'][:60]}` |")
    lines += ["", "## Notes",
              "- Weights encode both scale and mixing. For a single-pair execution, "
              "fire that pair at `scale = weight`. For multi-pair, either use a "
              "parallel FluxSpace call (node extension required) or render "
              "sequentially with residual re-measurement.",
              "- Residual = L2 norm of predicted minus target over the target readouts. "
              "Small = dictionary covers this target well; large = coverage gap "
              "(consider new `promptpair_iterate` iteration to span the residual direction)."]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, type=Path, help="target spec YAML")
    ap.add_argument("--out", type=Path, default=None, help="report path (default: auto)")
    args = ap.parse_args()
    spec = yaml.safe_load(args.target.read_text())
    target, constraints = resolve_target(spec)
    solver_cfg = spec.get("solver", {})
    result = compose(
        target, constraints,
        lambda_l1=solver_cfg.get("lambda_l1", 0.1),
        max_pairs=solver_cfg.get("max_pairs", 3),
        scale_cap=solver_cfg.get("scale_cap", 1.0),
        base=spec.get("base"),
    )
    out = args.out or (OUT_ROOT / f"{args.target.stem}.md")
    write_report(spec, result, out)
    print(f"[save] → {out}")
    print()
    # console preview
    print(f"Target: {target}")
    print(f"Constraints: {constraints}")
    print(f"Residual: {result['residual_on_targets']:.4f}")
    print(f"Violations: {result['constraint_violations'] or 'none'}")
    print(f"Picks:")
    for p in result["picks"]:
        print(f"  {p['weight']:.3f}  {p['pair']}")


if __name__ == "__main__":
    main()
