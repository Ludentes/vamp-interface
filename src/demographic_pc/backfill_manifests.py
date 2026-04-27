"""One-shot backfill: write manifest.json into each existing experiment
directory so future sessions can tell which data to use."""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FLUX_ROOT = ROOT / "output" / "demographic_pc" / "fluxspace_metrics"

NOW = _dt.datetime.now(_dt.timezone.utc).isoformat()


def _manifest(name: str, purpose: str, date: str, params: dict,
              related: list | None = None, notes: str | None = None,
              backfilled: bool = True) -> dict:
    return {
        "name": name,
        "purpose": purpose,
        "date_iso": date,
        "backfilled": backfilled,
        "backfill_date_iso": NOW if backfilled else None,
        "parameters": params,
        "measurement": notes,
        "related_to": related or [],
    }


# Entries — each is (relative_path, manifest_dict)
entries = [
    (
        "calibration",
        _manifest(
            name="calibration",
            purpose="Initial 10-prompt diverse-demographic FluxSpaceBaseMeasure corpus for on-manifold μ/σ estimation in the glasses collapse-prediction pipeline. Used as the calibration reference for max_env threshold fitting.",
            date="2026-04-21",
            params={
                "n_prompts": 10,
                "seed_base": 10000,
                "prompts": "CALIBRATION_PROMPTS in fluxspace_metrics.py",
                "workflow": "calibration_workflow (base-only, FluxSpaceBaseMeasure attn capture)",
            },
            related=[],
            notes="Superseded by calibration_expanded for ridge fits (this corpus is too small).",
        ),
    ),
    (
        "calibration_expanded",
        _manifest(
            name="calibration_expanded",
            purpose="Expanded 150-prompt × 2-seed calibration corpus with attention capture, designed to enable ridge-in-attention-space fitting against a 15-feature categorical design matrix (age × gender × ethnicity × expression). Fuels combo-1 ridge fit.",
            date="2026-04-21",
            params={
                "n_prompts": 150,
                "seeds": [20260, 20261],
                "total_renders": 300,
                "features": "age(4)×gender(2)×ethnicity(8)×expression(4)",
                "script": "src/demographic_pc/expand_calibration.py",
            },
            related=["ridge_attn"],
            notes="Filenames encode structured labels parseable by fit_ridge_attention.py.",
        ),
    ),
    (
        "ridge_attn",
        _manifest(
            name="ridge_attn",
            purpose="Ridge-regression fit from calibration_expanded's 15-d categorical design matrix to per-(block,step) attention states. Produces a 15-direction basis W in attention-output space. Compares directions against FluxSpace pair-averaging δ via |cos|_p95.",
            date="2026-04-22",
            params={
                "inputs": "calibration_expanded (300 pkls)",
                "features": ["intercept", "age×3", "gender×1", "ethnicity×7", "expression×3"],
                "ridge_lambda": 1.0,
                "output": "ridge_basis.pkl",
            },
            related=["calibration_expanded", "paired_ridge"],
            notes="R² mean 0.53. Demographic directions strong (norm 5-8); expression directions weak (norm 1-2). |cos|_p95 with pair δ only ~0.20 → motivated paired-contrast bootstrap.",
        ),
    ),
    (
        "crossdemo",
        _manifest(
            name="crossdemo (glasses axis)",
            purpose="Cross-demographic confirmation that FluxSpaceEditPair glasses edit generalizes beyond the initial Latin-f base. 6 diverse bases × 1 seed × scale sweep straddling predicted collapse edges.",
            date="2026-04-21",
            params={
                "bases": ["asian_m", "black_f", "european_m", "elderly_latin_m", "young_european_f", "southasian_f"],
                "seed": 2026,
                "axis": "glasses",
                "edit_a": "A person wearing thick-rimmed eyeglasses.",
                "mix_b": 0.5, "scale": 1.0,
            },
            related=["calibration"],
            notes="See measurement/ for attn + δ_mix per base. verify/ has scale-sweep PNGs.",
        ),
    ),
    (
        "crossdemo/smile",
        _manifest(
            name="crossdemo (smile axis)",
            purpose="Second-axis confirmation that FluxSpace pair-averaging recipe generalizes beyond glasses. Same 6 bases × smile edit. Tests whether collapse predictor and cos|p95| geometry transfer.",
            date="2026-04-21",
            params={
                "bases": "same 6 as glasses crossdemo",
                "seed": 2026,
                "axis": "smile",
                "edit_a": "A person smiling warmly.",
                "mix_b": 0.5, "scale": 1.0,
            },
            related=["crossdemo"],
            notes="cos|p95| ranking identical to glasses (strongest cross-axis invariant found). T_ratio over-predicts upper edge ≥0.2 on 5/6 bases.",
        ),
    ),
    (
        "crossdemo/smile/intensity",
        _manifest(
            name="intensity (local MVP)",
            purpose="Smile intensity dial MVP: 4-rung B-ladder (faint/warm/broad/manic) × 5 scales × 3 start_percents × 3 bases = 180 renders. Tests whether content-graded prompts give cleaner intensity control than scale magnitude.",
            date="2026-04-21",
            params={
                "bases": ["elderly_latin_m", "asian_m", "european_m"],
                "ladder": 4, "scales": 5, "start_pcts": 3, "seeds": 1,
                "total": 180,
            },
            related=["crossdemo/smile", "crossdemo/smile/intensity_full"],
            notes="Finding: B-ladder works at (sp=0.15, s=1.0); scale >1.4 saturates; sp=0.40 kills edit on smile.",
        ),
    ),
    (
        "crossdemo/smile/intensity_full",
        _manifest(
            name="intensity_full (shard)",
            purpose="Full 6-demographic intensity sweep on Windows shard. 4-rung ladder × 7 scales × 3 start_percents × 6 bases = 504 renders, plus 42 fifth-rung (cackle) extension. Tests the MVP findings at full coverage.",
            date="2026-04-21/22",
            params={
                "bases": 6, "ladder": 4, "scales": 7, "start_pcts": 3, "seeds": 1,
                "total_4rung": 504, "total_5rung": 42,
                "hardware": "Windows shard @ 192.168.87.25:8188 (COMFY_URL)",
            },
            related=["crossdemo/smile/intensity"],
            notes="Rung 5 (cackle) only rendered at sp=0.15 due to scope.",
        ),
    ),
    (
        "crossdemo/smile/alpha_interp",
        _manifest(
            name="alpha_interp (Mona Lisa ↔ Joker)",
            purpose="Linear mix_b interpolation between Mona Lisa (faint closed-smile) and Joker (manic grin) endpoint prompts. 11 alpha values × 10 seeds × 6 bases = 660 renders. Tests whether attention-cache linear interpolation produces a smooth blendshape trajectory.",
            date="2026-04-22",
            params={
                "edit_a": "A person with a faint closed-mouth smile.",
                "edit_b": "A person with a manic wide-open grin, teeth bared.",
                "alphas": [round(i * 0.1, 2) for i in range(11)],
                "seeds": [2026, 4242, 1337, 8080, 9999, 31415, 27182, 16180, 55555, 12345],
                "scale": 1.0, "start_percent": 0.15,
                "attn_capture": False,
            },
            related=["crossdemo/smile/intensity_full"],
            notes="No attention caches saved (measure_path=None). PNGs only. Phase boundary discovered at α≈0.45 for jawOpen (step function) and non-monotonic mouthSmile peaking mid-range.",
        ),
    ),
    (
        "bootstrap_v1",
        _manifest(
            name="bootstrap_v1",
            purpose="Paired-contrast bootstrap data: 6 bases × 3 axes (smile/age/glasses) × 8 levels × 2 seeds = 288 renders with FluxSpaceBaseMeasure attention capture. Fixed seed across each ladder enables clean Δattn computation within (base, seed) groups. Fuels paired_ridge fit.",
            date="2026-04-22",
            params={
                "bases": 6, "axes": ["smile", "age", "glasses"],
                "levels_per_axis": 8, "seeds": [2026, 4242],
                "total": 288,
                "workflow": "calibration_workflow (base-only, attn capture)",
            },
            related=["paired_ridge", "paired_ridge_pca"],
            notes="All 288 have attn pkls. MediaPipe blendshape scores in blendshapes.json (281/288 face-detected, cackle level failing).",
        ),
    ),
    (
        "paired_ridge",
        _manifest(
            name="paired_ridge",
            purpose="Multi-channel blendshape-target ridge fit on bootstrap_v1 paired differences. Tests whether continuous blendshape labels improve direction alignment with FluxSpace pair-averaging δ.",
            date="2026-04-22",
            params={
                "inputs": "bootstrap_v1 + blendshapes.json",
                "targets": 8,
                "method": "multi-output ridge with relative λ=1e-3",
            },
            related=["bootstrap_v1", "paired_ridge_pca"],
            notes="R² jumped 33× vs categorical. But multicollinearity in L/R blendshapes yields antiparallel per-channel directions — motivates PCA decorrelation.",
        ),
    ),
    (
        "paired_ridge_pca",
        _manifest(
            name="paired_ridge_pca",
            purpose="PCA-decorrelated version of paired_ridge with double-blocks-only restriction for cleaner directions. PC1-5 from 5-channel target. |cos|_p95 with pair-averaging δ reaches 0.37 on european_m.",
            date="2026-04-22",
            params={
                "targets_pre_pca": ["mouthSmileL", "mouthSmileR", "jawOpen", "mouthStretchL", "mouthStretchR"],
                "pc_explained_var": [0.49, 0.26, 0.19, 0.06, 0.003],
                "restriction": "double blocks only for comparison",
            },
            related=["paired_ridge"],
            notes="PC1 = overall smile; PC3 = pure jawOpen; PC4/5 = noise. R² reporting has intercept bug, but β directions valid.",
        ),
    ),
    (
        "alpha_linearity",
        _manifest(
            name="alpha_linearity",
            purpose="Linearity / phase-boundary analysis of alpha_interp. 60 (base, seed) trajectories fit with linear vs cubic polynomials. Detects the jawOpen step function at α≈0.45 and non-monotonic mouthSmile.",
            date="2026-04-22",
            params={
                "inputs": "alpha_interp/blendshapes.json",
                "bases": 6, "seeds_per_base": 10, "alphas": 11,
            },
            related=["crossdemo/smile/alpha_interp"],
            notes="0/60 trajectories monotonic in smile. jawOpen shows phase boundary (Lipschitz singularity). See docs/research/2026-04-22-alpha-interp-phase-boundary.md.",
        ),
    ),
]


def main() -> None:
    for rel, m in entries:
        d = FLUX_ROOT / rel
        if not d.exists():
            print(f"[backfill] skip (not found): {rel}")
            continue
        dest = d / "manifest.json"
        if dest.exists():
            # Still update a record that we checked
            print(f"[backfill] already present: {rel}")
            continue
        with dest.open("w") as f:
            json.dump(m, f, indent=2, default=str)
        print(f"[backfill] wrote: {rel}/manifest.json")


if __name__ == "__main__":
    main()
