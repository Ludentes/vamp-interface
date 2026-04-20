"""FairFace 7-race smoke test.

Uses res34_fair_align_multi_7 weights (from yakhyo/fairface-onnx mirror).
Skips dlib face detection — Flux portraits are already face-centered, just
resize to 224 and feed. For production we'd add dlib face-align for proper
per-sample crops, but smoke test establishes the classifier works.

Usage:
    uv run python src/demographic_pc/smoke_fairface.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "vendor" / "weights" / "fairface" / "res34_fair_align_multi_7_20190809.pt"

RACE_7 = ["White", "Black", "Latino_Hispanic", "East Asian",
          "Southeast Asian", "Indian", "Middle Eastern"]
GENDERS = ["Male", "Female"]
AGE_BINS = ["0-2", "3-9", "10-19", "20-29", "30-39",
            "40-49", "50-59", "60-69", "70+"]


def list_samples() -> list[Path]:
    out = [ROOT / "output" / "phase1" / "phase1_anchor.png"]
    for d in ["courier_legit", "office_legit", "scam_critical"]:
        p = ROOT / "output" / "phase1" / d
        if p.is_dir():
            imgs = sorted(p.glob("*.png"))
            if imgs:
                out.append(imgs[0])
    return out


def load_model(device: str) -> nn.Module:
    model = tvm.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 18)
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    return model.to(device).eval()


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    model = load_model(device)
    trans = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for p in list_samples():
        if not p.exists():
            print(f"skip: {p}")
            continue
        img = Image.open(p).convert("RGB")
        x = trans(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x).cpu().numpy().squeeze()
        race = np.exp(out[:7]); race /= race.sum()
        gender = np.exp(out[7:9]); gender /= gender.sum()
        age = np.exp(out[9:18]); age /= age.sum()
        print(
            f"{p.name:40s}  "
            f"race={RACE_7[race.argmax()]} ({race.max():.2f})  "
            f"gender={GENDERS[gender.argmax()]} ({gender.max():.2f})  "
            f"age={AGE_BINS[age.argmax()]} ({age.max():.2f})"
        )


if __name__ == "__main__":
    main()
