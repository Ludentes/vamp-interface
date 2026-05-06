"""CLI wrapper around arkit_bridge.distill.train."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkit_bridge.distill import train  # noqa: E402


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--holdout_dir", default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--ckpt_every", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--early_stop_patience", type=int, default=3)
    ap.add_argument("--loss_mode", default="plain",
                    choices=["plain", "varnorm", "varnorm_std_tail",
                             "weighted_mse", "varnorm_jvp"])
    ap.add_argument("--sampler", default="uniform",
                    choices=["uniform", "active_channel"])
    ap.add_argument("--stats", default=None,
                    help="path to teacher_stats.npz (required for varnorm/active_channel)")
    ap.add_argument("--lam_std", type=float, default=1.0)
    ap.add_argument("--lam_tail", type=float, default=0.5)
    ap.add_argument("--tail_z", type=float, default=2.0)
    ap.add_argument("--lam_jvp", type=float, default=0.1,
                    help="weight on JVP-norm regularizer (varnorm_jvp only)")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="exponent on 1/freq_k channel weighting "
                         "(weighted_mse and varnorm_jvp)")
    args = ap.parse_args()
    train(
        args.pairs_dir, args.out_dir,
        holdout_dir=args.holdout_dir,
        batch_size=args.batch_size, lr=args.lr,
        steps=args.steps, ckpt_every=args.ckpt_every,
        device=args.device,
        early_stop_patience=args.early_stop_patience,
        loss_mode=args.loss_mode,
        sampler=args.sampler,
        stats_path=args.stats,
        lam_std=args.lam_std, lam_tail=args.lam_tail, tail_z=args.tail_z,
        lam_jvp=args.lam_jvp, alpha=args.alpha,
    )
