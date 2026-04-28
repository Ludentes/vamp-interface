"""Evaluate an AdapterStudent checkpoint on the held-out val split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .adapter import AdapterStudent
from .dataset import CompactFFHQDataset, CompactLatentDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True,
                   choices=["pixel_a", "latent_a_up", "latent_a_native", "latent_a2_shallow"])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--compact", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--onnx-path", type=Path, default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    if args.variant == "pixel_a":
        ds = CompactFFHQDataset(args.compact, "val", normalisation="arcface")
    else:
        ds = CompactLatentDataset(args.compact, "val")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    onnx_path = args.onnx_path if args.onnx_path else None
    m = AdapterStudent(args.variant) if onnx_path is None else AdapterStudent(args.variant, onnx_path)
    m = m.to(device)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    m.load_state_dict(ck["model"])
    m.eval()

    cos_all = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            z = m(x)
            y = F.normalize(y, dim=-1)
            cos_all.append((z * y).sum(dim=-1).cpu())
    cos = torch.cat(cos_all)
    out = {
        "n": int(cos.numel()),
        "cosine_mean": float(cos.mean()),
        "cosine_median": float(cos.median()),
        "cosine_p05": float(cos.kthvalue(max(1, int(0.05 * cos.numel()))).values),
        "cosine_p95": float(cos.kthvalue(max(1, int(0.95 * cos.numel()))).values),
        "cosine_min": float(cos.min()),
        "frac_above_0p9": float((cos > 0.9).float().mean()),
        "variant": args.variant,
        "checkpoint_epoch": int(ck.get("epoch", -1)),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
