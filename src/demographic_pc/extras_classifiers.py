"""Extras classifiers for non-demographic axes.

- MediaPipe Face Landmarker → ARKit-style blendshapes (52 dials, continuous).
  Exposes a few high-signal aggregates used as axis targets:
    smile        = mouthSmileLeft + mouthSmileRight   ∈ [0, 2]
    brow_raise   = browInnerUp + 0.5·(browOuterUpLeft + browOuterUpRight)
    eye_open     = 1 − 0.5·(eyeBlinkLeft + eyeBlinkRight)
    jaw_open     = jawOpen

- CLIP zero-shot → binary glasses probability via open_clip's ViT-B-32
  pretrained on laion2b_s34b_b79k. Returns P(glasses) in [0, 1].

Both classifiers take BGR numpy arrays (OpenCV convention), mirroring the
interface of classifiers.py so a future merge is mechanical.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
FACE_LANDMARKER = ROOT / "vendor" / "weights" / "mediapipe" / "face_landmarker.task"

BLENDSHAPE_INDEX: dict[str, int] = {
    # ARKit-order indices used by MediaPipe's face_landmarker v1.
    # Set at first predict() since the task file ships its own order.
}


class MediaPipeBlendshapes:
    """Thin wrapper over mediapipe.tasks.FaceLandmarker that returns the
    52-dim blendshape vector and a few named aggregates.
    """

    def __init__(self) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(FACE_LANDMARKER)),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def predict(self, bgr: np.ndarray) -> dict:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_img)
        if not result.face_blendshapes:
            return {
                "detected": False, "blendshapes": None,
                "smile": float("nan"), "brow_raise": float("nan"),
                "eye_open": float("nan"), "jaw_open": float("nan"),
            }
        bs = result.face_blendshapes[0]
        if not BLENDSHAPE_INDEX:
            for i, c in enumerate(bs):
                BLENDSHAPE_INDEX[c.category_name] = i
        scores = np.array([c.score for c in bs], dtype=np.float32)  # (52,)
        get = lambda name: float(scores[BLENDSHAPE_INDEX[name]])
        smile = get("mouthSmileLeft") + get("mouthSmileRight")
        brow_raise = (
            get("browInnerUp")
            + 0.5 * (get("browOuterUpLeft") + get("browOuterUpRight"))
        )
        eye_open = 1.0 - 0.5 * (get("eyeBlinkLeft") + get("eyeBlinkRight"))
        jaw_open = get("jawOpen")
        return {
            "detected": True,
            "blendshapes": scores,
            "smile": smile,
            "brow_raise": brow_raise,
            "eye_open": eye_open,
            "jaw_open": jaw_open,
        }


class CLIPZeroShotGlasses:
    """ViT-B-32 CLIP scoring the text pair
        prompts = ["a photo of a person wearing glasses",
                   "a photo of a person not wearing glasses"]
    and returning the softmax P(glasses).
    """

    def __init__(self) -> None:
        import open_clip

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k",
        )
        self.model = self.model.eval().to(device)
        tok = open_clip.get_tokenizer("ViT-B-32")
        prompts = [
            "a photo of a person wearing glasses",
            "a photo of a person not wearing glasses",
        ]
        with torch.no_grad():
            text = tok(prompts).to(device)
            tfeat = self.model.encode_text(text)
            tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
        self.text_feat = tfeat  # (2, d)

    def predict(self, bgr: np.ndarray) -> dict:
        from PIL import Image

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        x = self.preprocess(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(x)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            logits = (feat @ self.text_feat.T) * 100.0  # CLIP scale
            probs = logits.softmax(dim=-1).squeeze(0).cpu().numpy()
        return {"glasses_prob": float(probs[0])}
