"""Diagnose attenuation in MotEncoderStudent vs teacher m_f.

Hypothesis: low Tier-1 ratio_mean (0.0066) is partly driven by the student
predicting the dense neutral cluster well while smoothing high-amplitude
expression cells. Symptom in renders: subdued expression despite passing
distill metrics.

Per-cell metrics over a holdout dir:
  - std_ratio   = std(student) / std(teacher)            ; <0.7 => attenuation
  - amp_ratio   = mean|student - mean| / mean|teacher - mean|
  - explained_var = 1 - var(teacher - student) / var(teacher)  (matches per-cell R²)
  - tail_recovery = std(student[teacher_z>2]) / std(teacher[teacher_z>2])
                    isolating the high-magnitude tail behaviour

Output JSON + per-cell .npz; report worst-attenuated cells and overall
fraction below threshold.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from arkit_bridge.dataset import PairDataset
from arkit_bridge.student import MotEncoderStudent


def collect(student_path, holdout_dir, device="cuda", batch_size=256):
    ds = PairDataset(holdout_dir)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=2)
    s = MotEncoderStudent().to(device).eval()
    s.load_state_dict(torch.load(student_path, map_location=device))

    teach_chunks, stud_chunks = [], []
    with torch.no_grad():
        for b, m in dl:
            b = b.to(device)
            stud = s(b).squeeze(1).cpu().float().numpy()  # (B, 32, 16)
            teach = m.squeeze(1).cpu().float().numpy()    # (B, 32, 16)
            stud_chunks.append(stud)
            teach_chunks.append(teach)
    return np.concatenate(teach_chunks, 0), np.concatenate(stud_chunks, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--holdout_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tail_z", type=float, default=2.0,
                    help="Z-score threshold for tail-recovery metric")
    args = ap.parse_args()

    teach, stud = collect(args.ckpt, args.holdout_dir, device=args.device)
    print(f"loaded {len(teach)} holdout samples; cell shape {teach.shape[1:]}",
          flush=True)

    # Per-cell stats
    eps = 1e-8
    t_mean = teach.mean(0)
    t_std = teach.std(0)
    s_std = stud.std(0)
    std_ratio = s_std / np.maximum(t_std, eps)

    t_dev = teach - t_mean
    s_dev = stud - t_mean
    amp_ratio = np.mean(np.abs(s_dev), 0) / np.maximum(np.mean(np.abs(t_dev), 0), eps)

    var_resid = np.var(teach - stud, 0)
    var_teach = np.var(teach, 0)
    expl_var = 1 - var_resid / np.maximum(var_teach, eps)

    # Tail recovery: std on samples where |teacher_z| > tail_z (per cell).
    z = (teach - t_mean) / np.maximum(t_std, eps)
    tail_mask = np.abs(z) > args.tail_z   # (N, 32, 16)
    tail_t_std = np.zeros_like(t_std)
    tail_s_std = np.zeros_like(s_std)
    tail_n = np.zeros_like(t_std, dtype=np.int64)
    L, C = teach.shape[1], teach.shape[2]
    for l in range(L):
        for c in range(C):
            m = tail_mask[:, l, c]
            tail_n[l, c] = int(m.sum())
            if m.sum() >= 5:  # need a few tail samples
                tail_t_std[l, c] = teach[m, l, c].std()
                tail_s_std[l, c] = stud[m, l, c].std()
    tail_recovery = tail_s_std / np.maximum(tail_t_std, eps)
    tail_recovery[tail_n < 5] = np.nan

    # Headline numbers
    summary = {
        "n_holdout": int(len(teach)),
        "tail_z": float(args.tail_z),
        "std_ratio": {
            "mean": float(np.mean(std_ratio)),
            "median": float(np.median(std_ratio)),
            "p10": float(np.percentile(std_ratio, 10)),
            "p90": float(np.percentile(std_ratio, 90)),
            "frac_below_0_7": float((std_ratio < 0.7).mean()),
            "frac_below_0_5": float((std_ratio < 0.5).mean()),
        },
        "amp_ratio": {
            "mean": float(np.mean(amp_ratio)),
            "median": float(np.median(amp_ratio)),
            "frac_below_0_7": float((amp_ratio < 0.7).mean()),
        },
        "explained_var": {
            "mean": float(np.mean(expl_var)),
            "median": float(np.median(expl_var)),
            "frac_above_0_7": float((expl_var > 0.7).mean()),
        },
        "tail_recovery": {
            "n_cells_with_tail": int((tail_n >= 5).sum()),
            "median": float(np.nanmedian(tail_recovery)),
            "mean": float(np.nanmean(tail_recovery)),
            "p10": float(np.nanpercentile(tail_recovery, 10)),
            "frac_below_0_7": float(np.nanmean(tail_recovery < 0.7)),
        },
    }

    # Worst 10 cells by std_ratio
    flat = std_ratio.flatten()
    worst_idx = np.argsort(flat)[:10]
    summary["worst_10_cells_by_std_ratio"] = [
        {
            "l": int(i // C), "c": int(i % C),
            "std_ratio": float(flat[i]),
            "teacher_std": float(t_std.flatten()[i]),
            "student_std": float(s_std.flatten()[i]),
            "tail_recovery": (float(tail_recovery.flatten()[i])
                              if not np.isnan(tail_recovery.flatten()[i]) else None),
        }
        for i in worst_idx
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    np.savez(
        out_path.with_suffix(".cells.npz"),
        std_ratio=std_ratio, amp_ratio=amp_ratio,
        explained_var=expl_var, tail_recovery=tail_recovery,
        teacher_std=t_std, student_std=s_std,
    )
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path} and {out_path.with_suffix('.cells.npz')}",
          flush=True)


if __name__ == "__main__":
    main()
