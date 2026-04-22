"""Phase-1 falsification test for the blendshape-bridge hypothesis.

Hypothesis: the non-monotonic mouthSmile and jawOpen phase-cliff in the
Mona→Joker (cross-phase) sweep are caused by prompt-embedding mixture
of AU axes, NOT by attention-cache curvature. Prediction: same-phase
sweeps (smile_inphase = AU12-only; jaw_inphase = AU26-only) should be
monotonic in the corresponding blendshape with no cliff.

Compares monotonicity rate (Kendall's τ and sign-of-consecutive-diffs)
between three datasets:
  - smile_inphase  (AU12 endpoint pair)
  - jaw_inphase    (AU26 endpoint pair)
  - alpha_interp   (Mona→Joker cross-phase — known non-monotonic)
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[2]
SMILE_DIR = ROOT / "output/demographic_pc/fluxspace_metrics/crossdemo/smile"

DATASETS = {
    "smile_inphase": SMILE_DIR / "smile_inphase/blendshapes.json",
    "jaw_inphase": SMILE_DIR / "jaw_inphase/blendshapes.json",
    "alpha_interp": SMILE_DIR / "alpha_interp/blendshapes.json",
}

CHANNELS = {
    "smile": ["mouthSmileLeft", "mouthSmileRight"],
    "jaw": ["jawOpen"],
    "stretch": ["mouthStretchLeft", "mouthStretchRight"],
}

FNAME_RE = re.compile(r"^s(?P<seed>\d+)_a(?P<alpha>[0-9.]+)$")


def avg(bs: dict, keys: list[str]) -> float:
    return float(np.mean([bs.get(k, 0.0) for k in keys]))


def group_trajectories(bs_json: dict) -> dict:
    """Return {(base, seed): {alpha: {channel: value}}}."""
    groups: dict = defaultdict(dict)
    for rel, bs in bs_json.items():
        base, fname = rel.split("/")
        m = FNAME_RE.match(Path(fname).stem)
        if not m:
            continue
        key = (base, int(m.group("seed")))
        alpha = float(m.group("alpha"))
        groups[key][alpha] = {name: avg(bs, keys) for name, keys in CHANNELS.items()}
    return groups


def analyze_dataset(name: str, path: Path) -> dict:
    if not path.exists():
        print(f"  [skip] {name}: {path} not found")
        return {}
    bs = json.loads(path.read_text())
    groups = group_trajectories(bs)
    if not groups:
        return {}

    # For each trajectory, per channel compute:
    #  - Kendall's τ over α
    #  - strictly monotonic (sign of all consecutive diffs equal, with slack)
    #  - max single-step jump ratio (detects cliffs)
    per_channel: dict = {ch: {"tau": [], "strict_mono": [], "max_step": []}
                         for ch in CHANNELS}

    for (_base, _seed), pts in groups.items():
        alphas = np.array(sorted(pts.keys()))
        for ch in CHANNELS:
            y = np.array([pts[a][ch] for a in alphas])
            tau = kendalltau(alphas, y).statistic if len(alphas) > 2 else 0.0
            dy = np.diff(y)
            strict = bool(np.all(dy >= -1e-4) or np.all(dy <= 1e-4))
            y_range = (y.max() - y.min()) + 1e-9
            max_step = float(np.abs(dy).max() / y_range) if len(dy) else 0.0
            per_channel[ch]["tau"].append(float(tau))
            per_channel[ch]["strict_mono"].append(strict)
            per_channel[ch]["max_step"].append(max_step)

    out = {"n_trajectories": len(groups), "channels": {}}
    for ch, rec in per_channel.items():
        tau_arr = np.array(rec["tau"])
        mono_arr = np.array(rec["strict_mono"])
        step_arr = np.array(rec["max_step"])
        out["channels"][ch] = {
            "tau_mean": float(tau_arr.mean()),
            "abs_tau_mean": float(np.abs(tau_arr).mean()),
            "abs_tau_ge_0.9": float((np.abs(tau_arr) >= 0.9).mean()),
            "strict_mono_rate": float(mono_arr.mean()),
            "max_step_mean": float(step_arr.mean()),
            "max_step_p95": float(np.percentile(step_arr, 95)),
        }
    return out


def main() -> None:
    print("=" * 72)
    print("Phase-1 falsification: in-phase vs cross-phase monotonicity")
    print("=" * 72)

    results = {name: analyze_dataset(name, path) for name, path in DATASETS.items()}

    # Print per-dataset summary table
    channels = list(CHANNELS.keys())
    for ch in channels:
        print(f"\n--- channel: {ch} ---")
        print(f"  {'dataset':<16} {'n':>4} {'<|τ|>':>7} {'|τ|≥0.9':>10} "
              f"{'strict':>7} {'step_mean':>10} {'step_p95':>9}")
        for name, res in results.items():
            if not res:
                continue
            c = res["channels"][ch]
            print(f"  {name:<16} {res['n_trajectories']:>4} "
                  f"{c['abs_tau_mean']:>7.3f} "
                  f"{c['abs_tau_ge_0.9']*100:>9.1f}% "
                  f"{c['strict_mono_rate']*100:>6.1f}% "
                  f"{c['max_step_mean']:>10.3f} "
                  f"{c['max_step_p95']:>9.3f}")

    # Falsification verdict
    print("\n" + "=" * 72)
    print("Falsification verdict")
    print("=" * 72)

    hypothesis_confirmed = True
    reasons = []
    for dset, primary_ch, expected_flat in [
        ("smile_inphase", "smile", "jaw"),
        ("jaw_inphase", "jaw", "smile"),
    ]:
        r = results.get(dset)
        if not r:
            continue
        active = r["channels"][primary_ch]
        flat = r["channels"][expected_flat]
        print(f"\n  {dset}:")
        print(f"    primary ({primary_ch}): |τ|≥0.9 in {active['abs_tau_ge_0.9']*100:.1f}% "
              f"of trajectories; strict-monotonic in {active['strict_mono_rate']*100:.1f}%")
        print(f"    flat ({expected_flat}):  max-step p95 = {flat['max_step_p95']:.3f}")
        if active["abs_tau_ge_0.9"] < 0.8:
            hypothesis_confirmed = False
            reasons.append(f"{dset}: only {active['abs_tau_ge_0.9']*100:.1f}% high-|τ| on primary (<80%)")

    cross = results.get("alpha_interp")
    if cross:
        smile = cross["channels"]["smile"]
        jaw = cross["channels"]["jaw"]
        print(f"\n  alpha_interp (cross-phase reference):")
        print(f"    smile: strict-monotonic in {smile['strict_mono_rate']*100:.1f}%")
        print(f"    jaw:   step_p95 = {jaw['max_step_p95']:.3f}")

    print("\n" + "=" * 72)
    if hypothesis_confirmed:
        print("RESULT: hypothesis SUPPORTED — in-phase sweeps are monotonic on")
        print("the primary axis. Non-monotonicity in cross-phase α-interp is")
        print("consistent with mixture-of-AUs mechanism, not attention-cache")
        print("curvature. Proceed to Phase 2 (PCA→ICA).")
    else:
        print("RESULT: hypothesis NOT supported:")
        for r in reasons:
            print(f"  - {r}")
        print("Revisit Phase 5 / local-metric estimation thread.")
    print("=" * 72)

    # Save artefact
    out_path = ROOT / "output/demographic_pc/fluxspace_metrics/analysis/inphase_monotonicity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved → {out_path}")


if __name__ == "__main__":
    main()
