"""FFHQ pixel + ArcFace teacher dataset for arc_pixel distillation."""
from __future__ import annotations


def is_held_out(image_sha256: str) -> bool:
    """Deterministic 6.25% held-out split: SHA prefix 'f' is val."""
    return image_sha256[:1].lower() == "f"
