"""Tests for swap_core.collapse_eyes -- the doll-eye pre-collapse step."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from swap_core import collapse_eyes  # noqa: E402

import cv2  # noqa: E402


def _doll_with_big_black_eyes():
    """A 256x256 skin-tone canvas with two big black eye discs."""
    img = np.full((256, 256, 3), (180, 200, 230), dtype=np.uint8)  # BGR skin
    cv2.circle(img, (96, 110), 26, (8, 8, 8), -1)    # left eye
    cv2.circle(img, (160, 110), 26, (8, 8, 8), -1)   # right eye
    # arcface kps: eyeL, eyeR, nose, mouthL, mouthR
    kps = np.array([[96, 110], [160, 110], [128, 150],
                    [104, 190], [152, 190]], dtype=np.float32)
    return img, kps


def test_big_black_eyes_are_lightened():
    """The dark pixel at each eye centre is repainted toward skin colour."""
    img, kps = _doll_with_big_black_eyes()
    before = img.copy()
    out = collapse_eyes(img, kps)
    # the big black disc is mostly gone: pixels a bit off-centre (clear of the
    # small replacement dot) are now light, not near-black.
    for cx in (96, 160):
        off = out[110, cx + 15]            # inside old disc, outside new dot
        assert int(off.mean()) > 120, f"eye blob at {cx} not lightened: {off}"
    # input image was not mutated
    assert np.array_equal(img, before), "collapse_eyes mutated its input"


def test_small_dot_is_drawn_at_eye_centre():
    """A small dark dot replaces the big eye at the centre."""
    img, kps = _doll_with_big_black_eyes()
    out = collapse_eyes(img, kps)
    for cx in (96, 160):
        assert int(out[110, cx].mean()) < 90, "no dark dot at eye centre"


def test_small_painted_eyes_are_left_untouched():
    """A doll that already has small folk-art eyes is not collapsed."""
    img = np.full((256, 256, 3), (180, 200, 230), dtype=np.uint8)
    cv2.circle(img, (96, 110), 5, (8, 8, 8), -1)     # small left eye
    cv2.circle(img, (160, 110), 5, (8, 8, 8), -1)    # small right eye
    kps = np.array([[96, 110], [160, 110], [128, 150],
                    [104, 190], [152, 190]], dtype=np.float32)
    out = collapse_eyes(img, kps)
    assert np.array_equal(out, img), "small eyes should be left untouched"


def test_degenerate_kps_returns_unchanged_copy():
    """kps with both eyes coincident -> unchanged copy, no crash."""
    img, _ = _doll_with_big_black_eyes()
    kps = np.array([[128, 128]] * 5, dtype=np.float32)
    out = collapse_eyes(img, kps)
    assert np.array_equal(out, img)
    assert out is not img
