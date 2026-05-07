"""One-pass over the train corpus to precompute v2 stats.

Outputs a single .npz with:
  teacher_mean : (32, 16)  — m_f mean over all train pairs
  teacher_std  : (32, 16)  — m_f std (used as sigma in varnorm-MSE)
  b_p95        : (58,)     — 95th percentile of |b_expr| per input dim
  sample_weights : (N,)    — per-pair active-channel sampling weight
                              max_i(|b_i| / b_p95_i), clipped to [0.1, 10.0]
  paths        : (N,)      — basename order matching sample_weights
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = sorted(Path(args.pairs_dir).glob("*frame_*.pkl"))
    n = len(paths)
    if n == 0:
        raise SystemExit(f"no pkls under {args.pairs_dir}")
    print(f"loading {n} pairs from {args.pairs_dir}")

    b_all = np.zeros((n, 58), dtype=np.float32)
    m_sum = np.zeros((32, 16), dtype=np.float64)
    m_sum2 = np.zeros((32, 16), dtype=np.float64)
    for i, p in enumerate(tqdm(paths)):
        with open(p, "rb") as f:
            d = pickle.load(f)
        b_all[i] = np.asarray(d["b_expr"], dtype=np.float32)
        mf = np.asarray(d["m_f"], dtype=np.float64).squeeze(0)  # (32,16)
        m_sum += mf
        m_sum2 += mf * mf

    teacher_mean = (m_sum / n).astype(np.float32)
    teacher_var = (m_sum2 / n) - (m_sum / n) ** 2
    teacher_std = np.sqrt(np.clip(teacher_var, 0, None)).astype(np.float32)

    b_p95 = np.percentile(np.abs(b_all), 95, axis=0).astype(np.float32)
    b_p95 = np.maximum(b_p95, 1e-3)

    # active-channel weight: max over inputs of |b_i| / b_p95_i. Frames
    # with any input near its 95th-percentile get weight ≥ 1.0; neutral
    # frames get weight much less. Clip to avoid one-frame domination.
    norm = np.abs(b_all) / b_p95[None, :]
    weights = norm.max(axis=1).astype(np.float32)
    weights = np.clip(weights, 0.1, 10.0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        teacher_mean=teacher_mean,
        teacher_std=teacher_std,
        b_p95=b_p95,
        sample_weights=weights,
        paths=np.array([p.name for p in paths]),
    )
    print(f"wrote {out}")
    print(f"  teacher_std : mean={teacher_std.mean():.4f} "
          f"min={teacher_std.min():.4f} max={teacher_std.max():.4f}")
    print(f"  b_p95       : min={b_p95.min():.4f} max={b_p95.max():.4f}")
    print(f"  weights     : p10={np.percentile(weights, 10):.3f} "
          f"p50={np.percentile(weights, 50):.3f} "
          f"p90={np.percentile(weights, 90):.3f} "
          f"max={weights.max():.3f}")


if __name__ == "__main__":
    main()
