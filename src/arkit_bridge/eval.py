"""Viability eval for the MotEncoder student.

Produces a single JSON with three groups, mapped to the four-tier viability
framework in `docs/research/2026-05-05-arkit-bridge-v1-viability.md`:

  Tier 1 (distill fidelity):
    - held_out_ratio_mean, held_out_ratio_p95
    - per_cell_r2: (32, 16) array
    - r2_above_0_7_fraction
    - sensitivity sweep (per-input-channel response magnitude)

  Tier 2 (input-category coverage):
    - category_counts and category_ratio for each ARKit-input category

Tier 3 / Tier 4 are pipeline-level evaluations and live in
`scripts/run_personalive_with_student.py` (Task 10), not here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from arkit_bridge.dataset import PairDataset
from arkit_bridge.student import MotEncoderStudent

ARKIT_NAMES = [
    "EyeBlinkLeft", "EyeLookDownLeft", "EyeLookInLeft", "EyeLookOutLeft",
    "EyeLookUpLeft", "EyeSquintLeft", "EyeWideLeft",
    "EyeBlinkRight", "EyeLookDownRight", "EyeLookInRight", "EyeLookOutRight",
    "EyeLookUpRight", "EyeSquintRight", "EyeWideRight",
    "JawForward", "JawLeft", "JawRight", "JawOpen",
    "MouthClose", "MouthFunnel", "MouthPucker", "MouthLeft", "MouthRight",
    "MouthSmileLeft", "MouthSmileRight", "MouthFrownLeft", "MouthFrownRight",
    "MouthDimpleLeft", "MouthDimpleRight", "MouthStretchLeft", "MouthStretchRight",
    "MouthRollLower", "MouthRollUpper", "MouthShrugLower", "MouthShrugUpper",
    "MouthPressLeft", "MouthPressRight", "MouthLowerDownLeft", "MouthLowerDownRight",
    "MouthUpperUpLeft", "MouthUpperUpRight",
    "BrowDownLeft", "BrowDownRight", "BrowInnerUp",
    "BrowOuterUpLeft", "BrowOuterUpRight",
    "CheekPuff", "CheekSquintLeft", "CheekSquintRight",
    "NoseSneerLeft", "NoseSneerRight", "TongueOut",
]
EYE_NAMES = ["LeftEyeYaw", "LeftEyePitch", "LeftEyeRoll",
             "RightEyeYaw", "RightEyePitch", "RightEyeRoll"]


def _bs_idx(name: str) -> int:
    return ARKIT_NAMES.index(name)


def _stack(dataset: PairDataset) -> tuple[torch.Tensor, torch.Tensor]:
    bs = torch.stack([dataset[i][0] for i in range(len(dataset))])
    ms = torch.stack([dataset[i][1] for i in range(len(dataset))])
    return bs, ms


def _student_predict(student: torch.nn.Module, b: torch.Tensor,
                     device: str, batch_size: int = 256) -> torch.Tensor:
    student.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(b), batch_size):
            out.append(student(b[i:i + batch_size].to(device)).cpu())
    return torch.cat(out, dim=0)


def held_out_ratios(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Per-frame MSE / per-frame Var(target). Returns mean and p95."""
    diff_sq = (pred - target).pow(2).flatten(1).mean(dim=1)
    var = target.flatten(1).var(dim=1, unbiased=False).clamp_min(1e-8)
    ratio = (diff_sq / var).numpy()
    return {
        "ratio_mean": float(ratio.mean()),
        "ratio_p95": float(np.quantile(ratio, 0.95)),
        "var_mean": float(var.mean()),
        "mse_mean": float(diff_sq.mean()),
    }


def per_cell_r2(pred: torch.Tensor, target: torch.Tensor) -> tuple[np.ndarray, float]:
    """Per-cell R² over the (32, 16) output, computed across the held-out set.

    R² = 1 - SS_res / SS_tot per cell. SS_tot computed against per-cell mean
    of target. Cells with zero variance return NaN.
    """
    p = pred.squeeze(1).numpy()
    t = target.squeeze(1).numpy()
    ss_res = ((p - t) ** 2).sum(axis=0)
    mean = t.mean(axis=0, keepdims=True)
    ss_tot = ((t - mean) ** 2).sum(axis=0)
    r2 = np.where(ss_tot > 1e-12, 1.0 - ss_res / np.maximum(ss_tot, 1e-12), np.nan)
    valid = ~np.isnan(r2)
    above = (r2[valid] >= 0.7).sum() / max(valid.sum(), 1)
    return r2, float(above)


def categorize(b_expr_all: torch.Tensor) -> dict[str, np.ndarray]:
    """Boolean masks (N,) over the held-out set for the Tier-2 categories."""
    b = b_expr_all.numpy()
    smile_l = b[:, _bs_idx("MouthSmileLeft")]
    smile_r = b[:, _bs_idx("MouthSmileRight")]
    jaw_open = b[:, _bs_idx("JawOpen")]
    brow_inner = b[:, _bs_idx("BrowInnerUp")]
    brow_l = b[:, _bs_idx("BrowOuterUpLeft")]
    brow_r = b[:, _bs_idx("BrowOuterUpRight")]
    blink_l = b[:, _bs_idx("EyeBlinkLeft")]
    blink_r = b[:, _bs_idx("EyeBlinkRight")]
    sq_l = b[:, _bs_idx("EyeSquintLeft")]
    sq_r = b[:, _bs_idx("EyeSquintRight")]
    eye_y = np.maximum(np.abs(b[:, 52]), np.abs(b[:, 55]))
    eye_p = np.maximum(np.abs(b[:, 53]), np.abs(b[:, 56]))
    bs_l1 = np.abs(b[:, :52]).sum(axis=1)
    return {
        "neutral":     bs_l1 < 1.5,
        "broad_smile": (smile_l + smile_r) > 1.0,
        "speech":      jaw_open > 0.3,
        "brow_raise":  (brow_inner > 0.4) | ((brow_l + brow_r) > 0.6),
        "blink_squint": np.maximum(np.maximum(blink_l, blink_r),
                                   np.maximum(sq_l, sq_r)) > 0.4,
        "asym_smile":  np.abs(smile_l - smile_r) > 0.2,
        "eye_gaze":    np.maximum(eye_y, eye_p) > 0.2,
    }


def category_ratios(pred: torch.Tensor, target: torch.Tensor,
                    cats: dict[str, np.ndarray]) -> dict:
    diff_sq = (pred - target).pow(2).flatten(1).mean(dim=1).numpy()
    var = target.flatten(1).var(dim=1, unbiased=False).clamp_min(1e-8).numpy()
    out = {}
    for name, mask in cats.items():
        n = int(mask.sum())
        if n == 0:
            out[name] = {"n": 0, "ratio_mean": None}
            continue
        out[name] = {
            "n": n,
            "ratio_mean": float((diff_sq[mask] / var[mask]).mean()),
            "ratio_p95": float(np.quantile(diff_sq[mask] / var[mask], 0.95)),
            "passes_0_15": bool((diff_sq[mask] / var[mask]).mean() < 0.15),
        }
    return out


def channel_sensitivity(student: torch.nn.Module, b_expr_all: torch.Tensor,
                        device: str = "cuda") -> dict[str, float]:
    """RMS response of student output to perturbing each ARKit input."""
    student.eval()
    b_neutral = b_expr_all.median(dim=0).values.to(device)
    out = {}
    with torch.no_grad():
        ref = student(b_neutral.unsqueeze(0))
        for i in range(52):
            b_hi = b_neutral.clone(); b_hi[i] = 1.0
            out[ARKIT_NAMES[i]] = (student(b_hi.unsqueeze(0)) - ref).pow(2).mean().sqrt().item()
        for i, name in enumerate(EYE_NAMES):
            b_hi = b_neutral.clone()
            b_hi[52 + i] = b_neutral[52 + i] + 0.5
            out[name] = (student(b_hi.unsqueeze(0)) - ref).pow(2).mean().sqrt().item()
    return out


def main(ckpt: str, pairs_dir: str, out_path: str, device: str = "cuda"):
    s = MotEncoderStudent().to(device)
    s.load_state_dict(torch.load(ckpt, map_location=device))
    ds = PairDataset(pairs_dir)
    b_all, m_all = _stack(ds)
    pred = _student_predict(s, b_all, device=device)

    tier1 = held_out_ratios(pred, m_all)
    r2_grid, r2_above = per_cell_r2(pred, m_all)
    cats = categorize(b_all)
    tier2 = category_ratios(pred, m_all, cats)
    sens = channel_sensitivity(s, b_all, device=device)
    ranked = sorted(sens.items(), key=lambda kv: -kv[1])

    payload = {
        "ckpt": str(ckpt),
        "n": len(ds),
        "tier1": {
            **tier1,
            "r2_above_0_7_fraction": r2_above,
            "per_cell_r2_shape": list(r2_grid.shape),
            "per_cell_r2_min": float(np.nanmin(r2_grid)),
            "per_cell_r2_median": float(np.nanmedian(r2_grid)),
            "passes_ratio_0_10": bool(tier1["ratio_mean"] < 0.10),
            "passes_r2_mask": bool(r2_above >= 0.80),
        },
        "tier2": tier2,
        "sensitivity": sens,
        "ranked_sensitivity": ranked,
    }
    op = Path(out_path)
    op.parent.mkdir(parents=True, exist_ok=True)
    with open(op, "w") as f:
        json.dump(payload, f, indent=2)
    np.savez(op.with_suffix(".per_cell_r2.npz"), r2=r2_grid)

    print(f"\n=== Tier 1 ===")
    print(f"  ratio mean: {tier1['ratio_mean']:.4f} (gate <0.10) "
          f"[{'PASS' if payload['tier1']['passes_ratio_0_10'] else 'FAIL'}]")
    print(f"  ratio p95:  {tier1['ratio_p95']:.4f}")
    print(f"  R² ≥ 0.7 fraction: {r2_above:.3f} (gate ≥0.80) "
          f"[{'PASS' if payload['tier1']['passes_r2_mask'] else 'FAIL'}]")
    print(f"  R² median: {payload['tier1']['per_cell_r2_median']:.3f}")
    print(f"\n=== Tier 2 (per-category) ===")
    for name, st in tier2.items():
        if st["n"] == 0:
            print(f"  {name:14s} n=0 (no frames in category)")
        elif st["n"] < 100:
            print(f"  {name:14s} n={st['n']:4d} ratio={st['ratio_mean']:.4f} "
                  f"(below 100-frame floor — out of v1 scope)")
        else:
            mark = "PASS" if st["passes_0_15"] else "FAIL"
            print(f"  {name:14s} n={st['n']:4d} ratio={st['ratio_mean']:.4f} "
                  f"(gate <0.15) [{mark}]")
    print(f"\n=== Top sensitivity (descending RMS response) ===")
    for n, v in ranked[:10]:
        print(f"  {n:24s} {v:.5f}")
    print(f"=== Bottom sensitivity ===")
    for n, v in ranked[-5:]:
        print(f"  {n:24s} {v:.5f}")
