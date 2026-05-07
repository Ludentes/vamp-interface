"""Per-input-channel sensitivity audit.

For each ARKit input dim i, perturb b along axis i by δ and measure how
much m_f moves vs the teacher's response. Channels where the student is
"deaf" (ratio < ~0.5) are exactly the inputs that disappear in renders.

Output: 58-element per-channel ratio + summary.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from arkit_bridge.student import MotEncoderStudent  # noqa: E402
from arkit_bridge.llf_csv import ARKIT_BLENDSHAPE_NAMES  # noqa: E402

# 58 = 52 ARKit + 6 eye rotations (LE+RE yaw/pitch/roll). Names match the
# pair-extraction layout (b_expr, no head ypr).
INPUT_NAMES = list(ARKIT_BLENDSHAPE_NAMES) + [
    "leftEyeYaw", "leftEyePitch", "leftEyeRoll",
    "rightEyeYaw", "rightEyePitch", "rightEyeRoll",
]
assert len(INPUT_NAMES) == 58


def load_teacher_pairs(pairs_dir, n):
    paths = sorted(Path(pairs_dir).glob("*frame_*.pkl"))
    rng = np.random.default_rng(0)
    if n < len(paths):
        idx = rng.choice(len(paths), size=n, replace=False)
        paths = [paths[i] for i in idx]
    bs, ms = [], []
    for p in paths:
        with open(p, "rb") as f:
            d = pickle.load(f)
        bs.append(np.asarray(d["b_expr"], dtype=np.float32))
        ms.append(np.asarray(d["m_f"], dtype=np.float32).squeeze(0))
    return np.stack(bs), np.stack(ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--holdout_dir", required=True,
                    help="dir of pkl pairs to seed perturbations from")
    ap.add_argument("--stats", required=True,
                    help="teacher_stats.npz (for b_p95)")
    ap.add_argument("--n_samples", type=int, default=200)
    ap.add_argument("--delta_frac", type=float, default=0.5,
                    help="δ as fraction of b_p95 per channel")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    stats = np.load(args.stats, allow_pickle=True)
    b_p95 = stats["b_p95"].astype(np.float32)  # (58,)
    delta = args.delta_frac * b_p95              # (58,)

    s = MotEncoderStudent().to(args.device).eval()
    s.load_state_dict(torch.load(args.ckpt, map_location=args.device))

    # Seed batch — random subset of holdout, treated as "current b" baselines.
    bs, _ = load_teacher_pairs(args.holdout_dir, args.n_samples)
    n = bs.shape[0]
    print(f"loaded {n} seeds, perturbing 58 channels each", flush=True)

    b_t = torch.from_numpy(bs).to(args.device)
    with torch.no_grad():
        m0 = s(b_t).squeeze(1)  # (n, 32, 16)

    # The "teacher response" we don't have directly without rerunning the
    # PersonaLive teacher, but for the v1→v2 comparison we use the *v1
    # student's* response as the reference baseline ratio: a channel where
    # both v1 and v2 are deaf gets ratio≈1 (no improvement to flag); a
    # channel where v2 is more responsive than v1 gets ratio>1. To get an
    # absolute measure, optionally compare to a held-out teacher_step_resp
    # dataset (not built here). Document this caveat in the output.
    ratios = np.zeros(58, dtype=np.float32)
    student_norms = np.zeros(58, dtype=np.float32)
    for i in range(58):
        b_p = b_t.clone()
        b_p[:, i] = b_p[:, i] + delta[i]
        with torch.no_grad():
            m1 = s(b_p).squeeze(1)
        diff = (m1 - m0).reshape(n, -1)               # (n, 512)
        norms = diff.norm(dim=1).cpu().numpy()        # (n,)
        student_norms[i] = float(norms.mean())
        # Per-cell normalised by teacher_std for a comparable amplitude:
        # || (m1 - m0) / sigma ||
        sigma = stats["teacher_std"].astype(np.float32).reshape(-1)  # (512,)
        norms_n = (
            (m1 - m0).reshape(n, -1).cpu().numpy() / np.maximum(sigma, 1e-3)
        )
        ratios[i] = float(np.linalg.norm(norms_n, axis=1).mean())

    out = {
        "n_seeds": n,
        "delta_frac": args.delta_frac,
        "b_p95": b_p95.tolist(),
        "per_channel": [
            {
                "i": i,
                "name": INPUT_NAMES[i],
                "delta": float(delta[i]),
                "student_response_norm": float(student_norms[i]),
                "student_response_norm_sigma_normalised": float(ratios[i]),
            }
            for i in range(58)
        ],
        "stats": {
            "median_response": float(np.median(student_norms)),
            "mean_response": float(np.mean(student_norms)),
            "frac_below_10pct_of_max": float(
                (student_norms < 0.10 * student_norms.max()).mean()
            ),
            "deafest_channels_top10": [
                {"i": int(i), "name": INPUT_NAMES[int(i)],
                 "response": float(student_norms[i])}
                for i in np.argsort(student_norms)[:10]
            ],
            "loudest_channels_top10": [
                {"i": int(i), "name": INPUT_NAMES[int(i)],
                 "response": float(student_norms[i])}
                for i in np.argsort(-student_norms)[:10]
            ],
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    print(f"  median student response: {out['stats']['median_response']:.4f}")
    print(f"  deafest channels: " + ", ".join(
        f"{c['name']}({c['response']:.3f})" for c in out["stats"]["deafest_channels_top10"][:5]
    ))
    print(f"  loudest channels: " + ", ".join(
        f"{c['name']}({c['response']:.3f})" for c in out["stats"]["loudest_channels_top10"][:5]
    ))


if __name__ == "__main__":
    main()
