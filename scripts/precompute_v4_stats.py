"""Precompute v4 stats: teacher_mean/std, b_p95, sample_weights, freq_k,
and empirical coupling matrix C[512, 58].

C[j, k] = empirical |dm_f[j] / db_k| from finite-difference probe pairs
(top-k_high vs bottom-k_low along each channel). Used for cell weighting
in variant A+B and as the magnitude target ‖C[:, k]‖ in variant C.

Output schema (strict superset of compute_teacher_stats.py's npz so
existing v2/v3 consumers can read it unchanged):

  teacher_mean    (32, 16)   m_f mean over training pairs
  teacher_std     (32, 16)   m_f std (sigma in varnorm-MSE)
  b_p95           (58,)      95th percentile of |b_expr| per input dim
  sample_weights  (N,)       per-pair active-channel weight, clipped [0.1,10]
  paths           (N,)       basenames matching sample_weights
  freq_k          (58,)      mean(b_k > active_thresh) over corpus, floored
                              at 1/N to avoid 0 in 1/freq_k weighting
  C               (512, 58)  per-cell × per-channel coupling magnitude
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_corpus(pairs_dir: Path):
    paths = sorted(pairs_dir.glob("*frame_*.pkl"))
    n = len(paths)
    if n == 0:
        raise SystemExit(f"no pkls under {pairs_dir}")
    print(f"loading {n} pairs from {pairs_dir}")
    b_all = np.zeros((n, 58), dtype=np.float32)
    m_all = np.zeros((n, 32, 16), dtype=np.float32)
    for i, p in enumerate(tqdm(paths)):
        with open(p, "rb") as f:
            d = pickle.load(f)
        b_all[i] = np.asarray(d["b_expr"], dtype=np.float32)
        mf = np.asarray(d["m_f"], dtype=np.float32).squeeze(0)
        m_all[i] = mf
    return paths, b_all, m_all


def compute_coupling(b_all: np.ndarray, m_all: np.ndarray, k: int = 32):
    """For each input channel, average |Δm/Δb| over top-k vs bottom-k pair
    differences (rank-paired). Returns C with shape (32*16, 58)."""
    n_b = b_all.shape[1]
    m_flat = m_all.reshape(m_all.shape[0], -1)  # (N, 512)
    C = np.zeros((m_flat.shape[1], n_b), dtype=np.float32)
    for ch in range(n_b):
        order = np.argsort(b_all[:, ch])
        lo = order[:k]
        hi = order[-k:]
        # b_diff = b[hi] - b[lo] with hi/lo from argsort, so b_diff >= 0 by
        # construction. Floor at eps to guard against degenerate constant
        # channels where many ranks share the same value.
        eps = 1e-6
        b_diff = np.maximum(b_all[hi, ch] - b_all[lo, ch], eps)
        m_diff = m_flat[hi] - m_flat[lo]  # (k, 512)
        secant = m_diff / b_diff[:, None]  # (k, 512)
        C[:, ch] = np.abs(secant).mean(axis=0)
    return C


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--active_thresh", type=float, default=0.05,
                    help="b_k > active_thresh counts as 'active' for freq_k")
    ap.add_argument("--coupling_k", type=int, default=32,
                    help="top-k vs bottom-k pair count for coupling secant")
    args = ap.parse_args(argv)

    pairs_dir = Path(args.pairs_dir)
    paths, b_all, m_all = load_corpus(pairs_dir)
    n = len(paths)

    teacher_mean = m_all.mean(0).astype(np.float32)
    teacher_std = m_all.std(0).astype(np.float32)

    b_abs = np.abs(b_all)
    b_p95 = np.maximum(np.percentile(b_abs, 95, axis=0), 1e-3).astype(np.float32)
    weights = np.clip((b_abs / b_p95[None, :]).max(axis=1), 0.1, 10.0).astype(np.float32)

    freq_k = (b_all > args.active_thresh).mean(axis=0).astype(np.float32)
    freq_k = np.maximum(freq_k, 1.0 / n)  # avoid div-by-zero downstream

    print("computing coupling matrix C[512, 58]...")
    C = compute_coupling(b_all, m_all, k=args.coupling_k)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        teacher_mean=teacher_mean,
        teacher_std=teacher_std,
        b_p95=b_p95,
        sample_weights=weights,
        paths=np.array([p.name for p in paths]),
        freq_k=freq_k,
        C=C,
    )
    print(f"wrote {out}")
    print(f"  freq_k: min={freq_k.min():.4f} median={np.median(freq_k):.4f} "
          f"max={freq_k.max():.4f}")
    Cn = np.linalg.norm(C, axis=0)
    print(f"  C norms per channel: min={Cn.min():.4f} median={np.median(Cn):.4f} "
          f"max={Cn.max():.4f}")


if __name__ == "__main__":
    main()
