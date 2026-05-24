"""Background library — pre-rendered scenes + metadata.

Manifest lives at `assets/backgrounds/manifest.json`. Each background's
PNG is at `assets/backgrounds/<id>/bg.png`.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # From src/group_photobooth/backgrounds.py to root
BG_DIR = ROOT / "assets" / "backgrounds"


@lru_cache(maxsize=1)
def _manifest() -> list[dict]:
    with open(BG_DIR / "manifest.json") as f:
        return json.load(f)


def list_backgrounds() -> list[dict]:
    """Return the manifest list as-is."""
    return list(_manifest())


def load_background(bg_id: str) -> tuple[np.ndarray, dict]:
    """Return (image_bgr, metadata_dict) for `bg_id`.

    Raises KeyError if the id is not in the manifest.
    """
    for entry in _manifest():
        if entry["id"] == bg_id:
            img = cv2.imread(str(BG_DIR / bg_id / "bg.png"))
            if img is None:
                raise FileNotFoundError(
                    f"manifest lists {bg_id} but bg.png is missing")
            return img, dict(entry)
    raise KeyError(f"unknown background id: {bg_id}")
