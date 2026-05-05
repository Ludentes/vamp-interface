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
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    train(
        args.pairs_dir, args.out_dir,
        batch_size=args.batch_size, lr=args.lr,
        steps=args.steps, device=args.device,
    )
