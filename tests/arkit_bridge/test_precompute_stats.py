"""Smoke test for v4 stats precompute on a synthetic 2-pair corpus."""

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def make_pair(out_path: Path, b_expr: np.ndarray, m_f: np.ndarray):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"b_expr": b_expr.astype(np.float32),
                     "m_f": m_f.astype(np.float32)}, f)


def test_smoke(tmp_path):
    """Synthetic corpus with orthogonal one-hot probes per channel.

    Each pair activates exactly one b channel at one of two levels (low or
    high). m_f only responds to channel 7, with slope 2.0 in column 0. With
    orthogonal probes, top-k vs bottom-k of any channel ch≠7 pulls samples
    where b[7]=0 in both groups → near-zero secant for column 0.
    """
    pairs_dir = tmp_path / "pairs"
    out = tmp_path / "stats.npz"
    n_b = 58
    levels = (0.1, 1.0)  # low / high amplitude
    counter = 0
    for ch in range(n_b):
        for level in levels:
            for _ in range(2):  # 2 reps per (ch, level) so coupling_k=4 has support
                b = np.zeros(n_b, dtype=np.float32)
                b[ch] = level
                m = np.zeros((1, 32, 16), dtype=np.float32)
                if ch == 7:
                    m[0, :, 0] = b[7] * 2.0
                make_pair(pairs_dir / f"frame_{counter:06d}.pkl", b, m)
                counter += 1

    from precompute_v4_stats import main as precompute_main
    precompute_main([
        "--pairs_dir", str(pairs_dir),
        "--out", str(out),
        "--coupling_k", "4",
    ])

    s = np.load(out, allow_pickle=True)
    # strict superset of compute_teacher_stats.py keys
    for key in ("teacher_mean", "teacher_std", "b_p95",
                "sample_weights", "paths"):
        assert key in s.files, f"missing v2 key: {key}"
    assert s["teacher_mean"].shape == (32, 16)
    assert s["teacher_std"].shape == (32, 16)
    assert s["b_p95"].shape == (58,)
    assert s["freq_k"].shape == (58,)
    assert s["C"].shape == (32 * 16, 58)
    C = s["C"].reshape(32, 16, 58)
    # Ground truth slope is 2.0 on channel 7, column 0; everywhere else 0.
    col0 = C[:, 0, :].mean(0)
    assert abs(col0[7] - 2.0) < 0.05, f"col0[7]={col0[7]}, expected ~2.0"
    other = np.delete(col0, 7)
    assert other.max() < 0.05, f"non-7 leakage too high: {other.max()}"
    # Other m_f columns should be near zero everywhere
    other_cols = C[:, 1:, :]
    assert np.abs(other_cols).max() < 0.05
