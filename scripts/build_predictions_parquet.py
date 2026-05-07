"""Build predictions.parquet — per-frame student m_f_pred for one or more ckpts.

Reads frames.parquet (b_expr + m_f_teacher per frame), runs the student forward
in batched torch on GPU, writes one row per (run_tag, take, frame_idx) with
m_f_pred (512-d flat) plus pre-computed scalar diagnostics.

Idempotent over (run_tag, take, frame_idx): reruns replace prior rows for the
same run_tag. Keeps prior runs for other run_tags so multiple students can
coexist.

Schema:
  run_tag        str       checkpoint identifier (e.g. "student_v3_lam10")
  take           uint8     1..8
  frame_idx      uint32    canonical pkl frame index
  split          str       "test" | "diag" | "excluded"
  m_f_pred       list[f32] flat 512-d (32 x 16 row-major), matches m_f_teacher
  mse            f32       mean((pred - teacher)^2)
  cos            f32       cosine similarity between pred and teacher
  l2_ratio       f32       ||pred|| / ||teacher||  (NaN if ||teacher||=0)

Cost: ~5–60s per ckpt depending on n_frames; CPU is fine.

Usage:
  python scripts/build_predictions_parquet.py \
      --ckpt runs/student_v3_lam10/student_best.pt --run_tag student_v3_lam10
  python scripts/build_predictions_parquet.py \
      --ckpt runs/student_v2_lam10/student_best.pt --run_tag student_v2_lam10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

sys.path.insert(0, "src")
from arkit_bridge.student import MotEncoderStudent  # noqa: E402


def forward_all(ckpt: Path, b_expr: np.ndarray, *, device: str, batch: int) -> np.ndarray:
    """b_expr (N, 58) -> m_f_pred (N, 512)."""
    s = MotEncoderStudent().to(device)
    s.load_state_dict(torch.load(ckpt, map_location=device))
    s.eval()
    out = np.empty((len(b_expr), 512), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(b_expr), batch):
            x = torch.from_numpy(b_expr[i:i + batch]).to(device)
            y = s(x)  # (B, 1, 32, 16)
            out[i:i + batch] = y.reshape(y.shape[0], -1).cpu().numpy()
    return out


def per_row_stats(pred: np.ndarray, teacher: np.ndarray) -> dict[str, np.ndarray]:
    """Pred / teacher are (N, 512) f32."""
    diff = pred - teacher
    mse = (diff * diff).mean(axis=1)
    p_norm = np.linalg.norm(pred, axis=1)
    t_norm = np.linalg.norm(teacher, axis=1)
    cos = np.where(
        (p_norm > 0) & (t_norm > 0),
        (pred * teacher).sum(axis=1) / np.clip(p_norm * t_norm, 1e-12, None),
        np.nan,
    )
    l2_ratio = np.where(t_norm > 0, p_norm / np.clip(t_norm, 1e-12, None), np.nan)
    return {"mse": mse.astype(np.float32),
            "cos": cos.astype(np.float32),
            "l2_ratio": l2_ratio.astype(np.float32)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--run_tag", required=True,
                    help="identifier in the parquet, e.g. student_v3_lam10")
    ap.add_argument("--frames", default="exp_output/arkit_bridge/parquet/frames.parquet")
    ap.add_argument("--out",    default="exp_output/arkit_bridge/parquet/predictions.parquet")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=4096)
    args = ap.parse_args()

    frames = pl.read_parquet(args.frames)
    print(f"frames: {len(frames)} rows from {args.frames}")
    b_expr = np.asarray(frames["b_expr"].to_list(), dtype=np.float32)
    m_teacher = np.asarray(frames["m_f_teacher"].to_list(), dtype=np.float32)
    assert b_expr.shape[1] == 58, b_expr.shape
    assert m_teacher.shape[1] == 512, m_teacher.shape

    print(f"forwarding {args.ckpt} (tag={args.run_tag}) on {args.device}...")
    pred = forward_all(args.ckpt, b_expr, device=args.device, batch=args.batch)
    stats = per_row_stats(pred, m_teacher)
    print(f"  mse mean={stats['mse'].mean():.6f}  cos mean={np.nanmean(stats['cos']):.4f}  "
          f"l2_ratio mean={np.nanmean(stats['l2_ratio']):.4f}")

    new_df = pl.DataFrame({
        "run_tag":   [args.run_tag] * len(frames),
        "take":      frames["take"],
        "frame_idx": frames["frame_idx"],
        "split":     frames["split"],
        "m_f_pred":  pred.tolist(),
        "mse":       stats["mse"],
        "cos":       stats["cos"],
        "l2_ratio":  stats["l2_ratio"],
    }, schema={
        "run_tag":   pl.String,
        "take":      pl.UInt8,
        "frame_idx": pl.UInt32,
        "split":     pl.String,
        "m_f_pred":  pl.List(pl.Float32),
        "mse":       pl.Float32,
        "cos":       pl.Float32,
        "l2_ratio":  pl.Float32,
    })

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        prior = pl.read_parquet(out)
        prior = prior.filter(pl.col("run_tag") != args.run_tag)
        merged = pl.concat([prior, new_df], how="vertical_relaxed")
    else:
        merged = new_df
    merged = merged.sort(["run_tag", "take", "frame_idx"])
    merged.write_parquet(out, compression="zstd")
    print(f"wrote {out}  rows={len(merged)} ({len(new_df)} new for {args.run_tag})  "
          f"size={out.stat().st_size/1e6:.1f} MB")
    print(merged.group_by("run_tag").agg(
        pl.len().alias("n"),
        pl.col("cos").mean().alias("cos_mean"),
        pl.col("mse").mean().alias("mse_mean"),
    ).sort("run_tag"))


if __name__ == "__main__":
    main()
