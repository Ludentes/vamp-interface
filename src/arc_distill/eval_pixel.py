"""Evaluate trained arc_distill checkpoint on FFHQ held-out.

Gate: per-row cosine mean > 0.9 → step 2 passes, proceed to arc_latent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from arc_distill.dataset import CompactFFHQDataset
from arc_distill.model import ArcStudentResNet18


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--compact", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    val = CompactFFHQDataset(args.compact, split="val")
    loader = DataLoader(val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    model = ArcStudentResNet18(pretrained=False).to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    cosines = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            p = model(x)
            yn = F.normalize(y, dim=-1)
            c = (p * yn).sum(dim=-1).cpu().numpy()
            cosines.append(c)
    cosines = np.concatenate(cosines)

    out = {
        "n": int(cosines.size),
        "cosine_mean": float(cosines.mean()),
        "cosine_median": float(np.median(cosines)),
        "cosine_p05": float(np.quantile(cosines, 0.05)),
        "cosine_p95": float(np.quantile(cosines, 0.95)),
        "cosine_min": float(cosines.min()),
        "frac_above_0p9": float((cosines > 0.9).mean()),
        "gate_passed_mean_gt_0p9": bool(cosines.mean() > 0.9),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
