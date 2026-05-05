"""Per-channel sensitivity match between student and teacher."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from arkit_bridge.dataset import DistillPairDataset
from arkit_bridge.encoder import ARKitParametricPoseGuider
from arkit_bridge.extractors import ARKIT_BLENDSHAPE_NAMES


def held_out_mse(student, dataset, device="cuda", batch_size=32):
    student.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = [dataset[j] for j in range(i, min(i+batch_size, len(dataset)))]
            b = torch.stack([x[0] for x in batch]).to(device)
            T = torch.stack([x[1] for x in batch]).to(device)
            pred = student(b)
            total += F.mse_loss(pred, T, reduction="sum").item()
            n += T.numel()
    return total / max(n, 1)


def channel_sensitivity(student, dataset, device="cuda"):
    """Per-channel L2 of d output / d channel value."""
    student.eval()
    bs = torch.stack([dataset[i][0] for i in range(len(dataset))])
    b_neutral = bs.median(dim=0).values.to(device)
    out = {}
    with torch.no_grad():
        ref = student(b_neutral.unsqueeze(0))
        for i in range(52):
            b_hi = b_neutral.clone()
            b_hi[i] = 1.0
            r_hi = student(b_hi.unsqueeze(0))
            delta = (r_hi - ref).pow(2).mean().sqrt().item()
            out[ARKIT_BLENDSHAPE_NAMES[i]] = delta
    return out


def main(ckpt: str, pairs_dir: str, out_path: str, device: str = "cuda"):
    student = ARKitParametricPoseGuider().to(device)
    student.load_state_dict(torch.load(ckpt, map_location=device))
    ds = DistillPairDataset(pairs_dir)
    mse = held_out_mse(student, ds, device=device)
    sens = channel_sensitivity(student, ds, device=device)
    report = {
        "ckpt": ckpt,
        "pairs_dir": pairs_dir,
        "n_samples": len(ds),
        "held_out_mse": mse,
        "channel_sensitivity": sens,
        "channel_sensitivity_ranked": sorted(
            sens.items(), key=lambda kv: -kv[1]),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"held-out MSE: {mse:.5f}")
    print("top 10 most-responsive channels:")
    for name, val in report["channel_sensitivity_ranked"][:10]:
        print(f"  {name:24s} {val:.5f}")
    print("bottom 5 least-responsive channels:")
    for name, val in report["channel_sensitivity_ranked"][-5:]:
        print(f"  {name:24s} {val:.5f}")
