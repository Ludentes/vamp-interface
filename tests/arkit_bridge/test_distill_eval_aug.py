"""Unit test for distill.train()'s eval-during-training augmentation.

Monkey-patches eval.main with a controlled ratio sequence and checks:
  - student_best.pt + student_best_r2.pt written when respective metric improves
  - eval_log.json contains one row per eval
  - early-stop fires after `early_stop_patience` consecutive non-improvements
"""

import json
import pickle
from pathlib import Path

import numpy as np


def _write_pair(d: Path, name: str, b_dim=58, l=32, c=16):
    rng = np.random.default_rng(int(name.split("_")[-1]))
    with open(d / f"{name}.pkl", "wb") as f:
        pickle.dump({
            "b_expr": rng.standard_normal(b_dim).astype(np.float32),
            "m_f": rng.standard_normal((1, l, c)).astype(np.float16),
            "frame_idx": 0,
        }, f)


def test_eval_aug_writes_best_and_early_stops(tmp_path, monkeypatch):
    pairs = tmp_path / "train"
    pairs.mkdir()
    holdout = tmp_path / "holdout"
    holdout.mkdir()
    out = tmp_path / "out"
    for i in range(64):
        _write_pair(pairs, f"t1_frame_{i:06d}")
    _write_pair(holdout, "t1_frame_999000")

    # Controlled ratio + r2 sequence per eval call.
    ratios = [0.20, 0.18, 0.19, 0.21, 0.22]   # improves twice, then no-improve x3
    r2s    = [0.50, 0.55, 0.54, 0.53, 0.52]
    calls = {"n": 0}

    def fake_eval_main(ckpt, holdout_dir, out_path, device="cpu"):
        i = calls["n"]; calls["n"] += 1
        payload = {
            "tier1": {
                "ratio_mean": ratios[i],
                "r2_above_0_7_fraction": r2s[i],
                "passes_ratio_0_10": ratios[i] < 0.10,
                "passes_r2_mask": r2s[i] >= 0.80,
            }
        }
        Path(out_path).write_text(json.dumps(payload))

    import arkit_bridge.eval as ev_mod
    monkeypatch.setattr(ev_mod, "main", fake_eval_main)

    from arkit_bridge.distill import train
    train(
        pairs_dir=pairs, out_dir=out, holdout_dir=holdout,
        batch_size=8, lr=1e-3, steps=10_000,
        log_every=10_000, ckpt_every=10,
        device="cpu", early_stop_patience=3,
    )

    assert (out / "student_best.pt").exists()
    assert (out / "student_best_r2.pt").exists()
    log = json.loads((out / "eval_log.json").read_text())
    assert len(log) >= 4  # early-stop after 3 non-improvements past best
    assert log[0]["ratio_mean"] == ratios[0]
    # Stopped before consuming all 5 sentinel values.
    assert calls["n"] <= 5
