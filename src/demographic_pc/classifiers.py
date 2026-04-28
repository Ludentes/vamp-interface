"""Unified wrapper around MiVOLO, FairFace, InsightFace.

Exposes one `predict(image_bgr)` per classifier returning a dict with age,
gender, and (FairFace only) race. Handles dlib alignment for FairFace since
the model was trained on aligned crops and refuses to be informative
without it.

Usage:
    from src.demographic_pc.classifiers import MiVOLOClassifier, FairFaceClassifier, InsightFaceClassifier
    mv = MiVOLOClassifier(); ff = FairFaceClassifier(); ins = InsightFaceClassifier()
    out = mv.predict(bgr_image)  # {"age": float, "gender": "M"/"F", "gender_conf": float}
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import dlib
import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor" / "MiVOLO"))
from mivolo.model.mivolo_model import *  # noqa: F401, F403, E402


# ── MiVOLO ────────────────────────────────────────────────────────────────────

MIVOLO_CKPT = ROOT / "vendor" / "weights" / "mivolo_volo_d1_face_age_gender_imdb.pth.tar"


class MiVOLOClassifier:
    """face-only volo_d1, trained on IMDB-cleaned."""

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(MIVOLO_CKPT, map_location="cpu", weights_only=False)
        sd = {k: v for k, v in state["state_dict"].items() if not k.startswith("fds.")}
        self.input_size = sd["pos_embed"].shape[1] * 16
        self.min_age = state["min_age"]
        self.max_age = state["max_age"]
        self.avg_age = state["avg_age"]
        self.model = timm.create_model(
            f"mivolo_d1_{self.input_size}",
            num_classes=3, in_chans=3, pretrained=False,
        )
        self.model.load_state_dict(sd, strict=False)
        self.model = self.model.to(self.device).eval()
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _prep(self, bgr: np.ndarray) -> torch.Tensor:
        h, w = bgr.shape[:2]
        s = self.input_size / max(h, w)
        nw, nh = int(w * s), int(h * s)
        r = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((self.input_size, self.input_size, 3), dtype=bgr.dtype)
        canvas[(self.input_size - nh) // 2:(self.input_size - nh) // 2 + nh,
               (self.input_size - nw) // 2:(self.input_size - nw) // 2 + nw] = r
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self._mean) / self._std
        return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

    def predict(self, bgr: np.ndarray) -> dict[str, Any]:
        x = self._prep(bgr)
        with torch.no_grad():
            out = self.model(x)[0]
        age_norm = out[2].item()
        age = age_norm * (self.max_age - self.min_age) + self.avg_age
        gp = torch.softmax(out[:2], dim=-1).cpu().numpy()
        gender = "M" if gp[0] > gp[1] else "F"
        return {"age": float(age), "gender": gender, "gender_conf": float(max(gp))}


# ── FairFace ──────────────────────────────────────────────────────────────────

FAIRFACE_CKPT = ROOT / "vendor" / "weights" / "fairface" / "res34_fair_align_multi_7_20190809.pt"
DLIB_DET = ROOT / "vendor" / "FairFace" / "dlib_models" / "mmod_human_face_detector.dat"
DLIB_SP = ROOT / "vendor" / "FairFace" / "dlib_models" / "shape_predictor_5_face_landmarks.dat"

RACE_7 = ["White", "Black", "Latino_Hispanic", "East Asian",
          "Southeast Asian", "Indian", "Middle Eastern"]
FAIRFACE_GENDER = ["M", "F"]
AGE_BINS = ["0-2", "3-9", "10-19", "20-29", "30-39",
            "40-49", "50-59", "60-69", "70+"]


class FairFaceClassifier:
    """res34 FairFace 7-race model. Uses dlib detect + 5-landmark align.

    `use_hog=False` (default) uses the dlib CNN MMOD detector — most accurate
    but 1-15 seconds per image on CPU.

    `use_hog=True` uses the dlib HOG frontal-face detector — ~50ms per image,
    suitable for high-quality frontal datasets like FFHQ. Less accurate on
    profile / occluded / small faces.
    """

    def __init__(self, device: str | None = None, use_hog: bool = False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_hog = use_hog
        model = tvm.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 18)
        model.load_state_dict(torch.load(FAIRFACE_CKPT, map_location="cpu", weights_only=True))
        self.model = model.to(self.device).eval()
        if use_hog:
            self.detector = dlib.get_frontal_face_detector()
        else:
            self.detector = dlib.cnn_face_detection_model_v1(str(DLIB_DET))
        self.sp = dlib.shape_predictor(str(DLIB_SP))
        self.trans = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _align(self, bgr: np.ndarray, size: int = 300, padding: float = 0.25) -> np.ndarray | None:
        """dlib detect + 5-landmark align. Returns aligned RGB crop or None."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        max_side = 800
        if max(h, w) > max_side:
            s = max_side / max(h, w)
            rgb = cv2.resize(rgb, (int(w * s), int(h * s)))
        dets = self.detector(rgb, 1)
        if not dets:
            return None
        if self.use_hog:
            # HOG returns dlib.rectangle directly
            det = max(dets, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))
            rect = det
        else:
            # CNN returns dlib.mmod_rectangle (.rect attr)
            det = max(dets, key=lambda d: (d.rect.right() - d.rect.left()) * (d.rect.bottom() - d.rect.top()))
            rect = det.rect
        faces = dlib.full_object_detections()
        faces.append(self.sp(rgb, rect))
        chips = dlib.get_face_chips(rgb, faces, size=size, padding=padding)
        return chips[0] if chips else None

    def predict(self, bgr: np.ndarray) -> dict[str, Any]:
        aligned = self._align(bgr)
        if aligned is None:
            return {"age_bin": None, "gender": None, "race": None,
                    "age_probs": None, "gender_probs": None, "race_probs": None,
                    "detected": False}
        img = Image.fromarray(aligned)
        x = self.trans(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(x).cpu().numpy().squeeze()
        race = np.exp(out[:7]); race /= race.sum()
        gender = np.exp(out[7:9]); gender /= gender.sum()
        age = np.exp(out[9:18]); age /= age.sum()
        return {
            "age_bin": AGE_BINS[int(age.argmax())],
            "gender": FAIRFACE_GENDER[int(gender.argmax())],
            "race": RACE_7[int(race.argmax())],
            "age_probs": age.astype(np.float32),
            "gender_probs": gender.astype(np.float32),
            "race_probs": race.astype(np.float32),
            "detected": True,
        }


# ── InsightFace ───────────────────────────────────────────────────────────────

class InsightFaceClassifier:
    """buffalo_l: SCRFD detection + genderage (+ optional ArcFace r50 recognition).

    With `with_embedding=True` the w600k_r50 recognition head is enabled and
    `predict()` returns a normalised 512-d ArcFace embedding under "embedding"
    for identity-drift cosines.
    """

    def __init__(self, ctx_id: int = 0, with_embedding: bool = False):
        from insightface.app import FaceAnalysis
        modules = ["detection", "genderage"] + (["recognition"] if with_embedding else [])
        self.app = FaceAnalysis(name="buffalo_l", allowed_modules=modules)
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        self.with_embedding = with_embedding

    def predict(self, bgr: np.ndarray) -> dict[str, Any]:
        faces = self.app.get(bgr)
        if not faces:
            return {"age": None, "gender": None, "bbox": None, "detected": False,
                    "embedding": None}
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        emb = None
        if self.with_embedding and getattr(f, "normed_embedding", None) is not None:
            emb = np.asarray(f.normed_embedding, dtype=np.float32)
        return {
            "age": float(f.age),
            "gender": "M" if f.sex == "M" else "F",
            "bbox": tuple(int(v) for v in f.bbox),
            "detected": True,
            "embedding": emb,
        }


# ── Combined record ───────────────────────────────────────────────────────────

@dataclass
class ClassifierRecord:
    mivolo_age: float | None
    mivolo_gender: str | None
    mivolo_gender_conf: float | None

    fairface_age_bin: str | None
    fairface_gender: str | None
    fairface_race: str | None
    fairface_age_probs: list[float] | None
    fairface_gender_probs: list[float] | None
    fairface_race_probs: list[float] | None
    fairface_detected: bool

    insightface_age: float | None
    insightface_gender: str | None
    insightface_detected: bool


def predict_all(
    bgr: np.ndarray,
    mv: MiVOLOClassifier,
    ff: FairFaceClassifier,
    ins: InsightFaceClassifier,
) -> ClassifierRecord:
    m = mv.predict(bgr)
    f = ff.predict(bgr)
    i = ins.predict(bgr)
    return ClassifierRecord(
        mivolo_age=m["age"], mivolo_gender=m["gender"], mivolo_gender_conf=m["gender_conf"],
        fairface_age_bin=f["age_bin"], fairface_gender=f["gender"], fairface_race=f["race"],
        fairface_age_probs=f["age_probs"].tolist() if f["detected"] else None,
        fairface_gender_probs=f["gender_probs"].tolist() if f["detected"] else None,
        fairface_race_probs=f["race_probs"].tolist() if f["detected"] else None,
        fairface_detected=f["detected"],
        insightface_age=i["age"], insightface_gender=i["gender"], insightface_detected=i["detected"],
    )
