"""MiVOLO face-only smoke test.

Loads volo_d1 weights via timm directly, bypassing MiVOLO's create_timm_model
shim (which is broken against timm 1.0). Tests on Flux portraits.

Usage:
    uv run python src/demographic_pc/smoke_mivolo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import timm
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor" / "MiVOLO"))

# register the MiVOLO model class with timm
from mivolo.model.mivolo_model import *  # noqa: F401, F403, E402

CKPT = ROOT / "vendor" / "weights" / "mivolo_volo_d1_face_age_gender_imdb.pth.tar"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def list_samples() -> list[Path]:
    out = [ROOT / "output" / "phase1" / "phase1_anchor.png"]
    for d in ["courier_legit", "office_legit", "scam_critical"]:
        p = ROOT / "output" / "phase1" / d
        if p.is_dir():
            imgs = sorted(p.glob("*.png"))
            if imgs:
                out.append(imgs[0])
    return out


def letterbox(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=img.dtype)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def load_model(device: str) -> tuple[torch.nn.Module, dict]:
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    meta = {
        "min_age": state["min_age"],
        "max_age": state["max_age"],
        "avg_age": state["avg_age"],
        "only_age": state.get("no_gender", False),
    }
    sd = state["state_dict"]
    # face-only: in_chans=3, num_classes=3 ([gender_0, gender_1, age_norm])
    in_chans = 6 if "patch_embed.conv1.0.weight" in sd else 3
    meta["in_chans"] = in_chans
    num_classes = 1 if meta["only_age"] else 3
    input_size = sd["pos_embed"].shape[1] * 16
    meta["input_size"] = input_size

    # filter FDS (feature-distribution smoothing) keys
    sd = {k: v for k, v in sd.items() if not k.startswith("fds.")}

    model = timm.create_model(
        f"mivolo_d1_{input_size}",
        num_classes=num_classes,
        in_chans=in_chans,
        pretrained=False,
    )
    incompat = model.load_state_dict(sd, strict=False)
    print(f"missing: {len(incompat.missing_keys)}  unexpected: {len(incompat.unexpected_keys)}")
    if incompat.missing_keys:
        print(f"  missing sample: {incompat.missing_keys[:3]}")
    if incompat.unexpected_keys:
        print(f"  unexpected sample: {incompat.unexpected_keys[:3]}")
    model = model.to(device).eval()
    return model, meta


def predict(model: torch.nn.Module, img_path: Path, size: int, device: str) -> tuple[float, str, float]:
    bgr = cv2.imread(str(img_path))
    box = letterbox(bgr, size)
    rgb = cv2.cvtColor(box, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
    age_norm = out[0, 2].item()
    gender_probs = torch.softmax(out[0, :2], dim=-1)
    gender = "male" if gender_probs[0].item() > gender_probs[1].item() else "female"
    gconf = max(gender_probs[0].item(), gender_probs[1].item())
    return age_norm, gender, gconf


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    model, meta = load_model(device)
    print(f"meta: {meta}")
    size = meta["input_size"]
    for p in list_samples():
        if not p.exists():
            print(f"skip: {p}")
            continue
        age_norm, gender, gconf = predict(model, p, size, device)
        age = age_norm * (meta["max_age"] - meta["min_age"]) + meta["avg_age"]
        print(f"{p.name:40s}  age={age:5.1f}  gender={gender} ({gconf:.2f})")


if __name__ == "__main__":
    main()
