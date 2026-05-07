"""One-page health summary for a trained student checkpoint.

Combines:
  - per-cell R² from existing eval npz   (output-side fit quality)
  - per-input-channel sensitivity audit (input-side responsiveness)
  - holdout cos(student_mf, teacher_mf)  (alignment to teacher)

Why: training prints aggregate ratio and category gates but burying
the bottom-K channels makes silent failures (e.g. brow direction
inversion) easy to miss. This dumps the sorted lists in plain text so
problem channels surface immediately.

Usage:
  python summarize_student_health.py \
      --ckpt runs/student_v3_lam10/student_best.pt \
      --eval_r2 runs/student_v3_lam10/eval_step060000.per_cell_r2.npz \
      --holdout_dir data/arkit_bridge_pairs/holdout_v3 \
      --stats runs/teacher_stats_v3.npz \
      --out runs/student_v3_lam10/health_summary.txt
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkit_bridge.student import MotEncoderStudent           # noqa: E402
from arkit_bridge.llf_csv import ARKIT_BLENDSHAPE_NAMES      # noqa: E402

INPUT_NAMES = list(ARKIT_BLENDSHAPE_NAMES) + [
    "leftEyeYaw", "leftEyePitch", "leftEyeRoll",
    "rightEyeYaw", "rightEyePitch", "rightEyeRoll",
]
assert len(INPUT_NAMES) == 58


def load_holdout(pdir):
    bs, ms = [], []
    for p in sorted(Path(pdir).glob("*.pkl")):
        with open(p, "rb") as f:
            d = pickle.load(f)
        bs.append(np.asarray(d["b_expr"], dtype=np.float32))
        ms.append(np.asarray(d["m_f"], dtype=np.float32).reshape(32, 16))
    return np.stack(bs), np.stack(ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval_r2", required=True, help="per_cell_r2.npz")
    ap.add_argument("--holdout_dir", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    stats = np.load(args.stats)
    b_p95 = stats["b_p95"].astype(np.float32)
    sigma = stats["teacher_std"].astype(np.float32).reshape(-1)

    s = MotEncoderStudent().to(args.device).eval()
    s.load_state_dict(torch.load(args.ckpt, map_location=args.device))

    bs, ms = load_holdout(args.holdout_dir)
    n = len(bs)
    with torch.no_grad():
        m_pred = s(torch.from_numpy(bs).to(args.device)).squeeze(1).cpu().numpy()
    # cos & R² to teacher per holdout sample
    cos_per = []
    for i in range(n):
        a = m_pred[i].reshape(-1)
        b = ms[i].reshape(-1)
        cos_per.append(float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b)+1e-9)))
    cos_per = np.array(cos_per)

    # per-channel sensitivity (output norm change when you bump input dim by p95)
    delta = 0.5 * b_p95
    bt = torch.from_numpy(bs).to(args.device)
    with torch.no_grad():
        m0 = s(bt).squeeze(1)
    sens = np.zeros(58, dtype=np.float32)
    for i in range(58):
        bp = bt.clone()
        bp[:, i] = bp[:, i] + delta[i]
        with torch.no_grad():
            m1 = s(bp).squeeze(1)
        diff = (m1 - m0).reshape(n, -1).cpu().numpy()
        # sigma-normalised so all channels are in comparable units
        sens[i] = float(np.linalg.norm(diff / np.maximum(sigma, 1e-3), axis=1).mean())

    # eval R² per output cell
    r2 = np.load(args.eval_r2)["r2"]   # (32, 16)
    r2_flat = r2.reshape(-1)

    out = []
    out.append("=== student health summary ===")
    out.append(f"ckpt: {args.ckpt}")
    out.append(f"holdout: {args.holdout_dir} (n={n})")
    out.append("")
    out.append(f"alignment cos(student, teacher) on holdout:")
    out.append(f"  mean={cos_per.mean():.4f}  median={np.median(cos_per):.4f}  "
               f"p10={np.percentile(cos_per,10):.4f}  min={cos_per.min():.4f}")
    out.append("")
    out.append(f"output-cell R²:  ≥0.7 frac={(r2_flat>=0.7).mean():.3f}  "
               f"≥0.5 frac={(r2_flat>=0.5).mean():.3f}  median={np.median(r2_flat):.3f}  "
               f"min={r2_flat.min():.3f}")
    out.append("")
    out.append("=== bottom-15 output cells by R² ===")
    worst = np.argsort(r2_flat)[:15]
    for i in worst:
        row, col = i // 16, i % 16
        out.append(f"  cell[{row:>2d},{col:>2d}]  R²={r2_flat[i]:+.4f}")
    out.append("")
    out.append("=== bottom-15 input channels by sensitivity ===")
    order = np.argsort(sens)
    for i in order[:15]:
        out.append(f"  ch{i:>2d} {INPUT_NAMES[i]:<22s}  sens={sens[i]:.3f}  b_p95={b_p95[i]:.3f}")
    out.append("")
    out.append("=== top-15 input channels by sensitivity ===")
    for i in order[::-1][:15]:
        out.append(f"  ch{i:>2d} {INPUT_NAMES[i]:<22s}  sens={sens[i]:.3f}  b_p95={b_p95[i]:.3f}")
    out.append("")
    out.append("=== brow channels ===")
    for i, name in enumerate(INPUT_NAMES):
        if "brow" in name.lower() or "Brow" in name:
            out.append(f"  ch{i:>2d} {name:<22s}  sens={sens[i]:.3f}  b_p95={b_p95[i]:.3f}")

    text = "\n".join(out)
    Path(args.out).write_text(text)
    print(text)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
