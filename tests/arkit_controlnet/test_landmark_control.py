from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from arkit_controlnet.landmark_control import (
    REVERSE_INDEX,
    render_depth_map,
    render_landmark_mesh,
    select_exemplars,
    select_neutral,
)

pytestmark = pytest.mark.skipif(
    not REVERSE_INDEX.exists(), reason="reverse_index parquet not present"
)

_FIXTURE = Path("tests/arkit_controlnet/fixtures/face.png")


def test_select_exemplars_returns_k_existing_pngs():
    paths = select_exemplars("smile", k=3)
    assert len(paths) == 3
    assert all(p.exists() and p.suffix == ".png" for p in paths)


def test_smile_exemplars_score_above_corpus_median():
    # the picked exemplars must actually be high-smile, not arbitrary rows
    df = pd.read_parquet(
        REVERSE_INDEX, columns=["image_sha256", "bs_mouthSmileLeft", "bs_mouthSmileRight"]
    )
    median = (df["bs_mouthSmileLeft"] + df["bs_mouthSmileRight"]).median()
    picked = {p.stem for p in select_exemplars("smile", k=3)}
    rows = df[df["image_sha256"].isin(picked)]
    score = rows["bs_mouthSmileLeft"] + rows["bs_mouthSmileRight"]
    assert (score > median).all()


def test_select_neutral_returns_k_existing_pngs():
    paths = select_neutral(k=3)
    assert len(paths) == 3
    assert all(p.exists() for p in paths)


@pytest.mark.skipif(not _FIXTURE.exists(), reason="fixture face.png not present")
def test_render_landmark_mesh_shape_and_ink():
    mesh = render_landmark_mesh(_FIXTURE)
    assert mesh.shape == (1152, 864, 3)        # (H, W, 3)
    assert mesh.dtype == np.uint8
    white = int((mesh > 0).any(axis=2).sum())
    assert white > 2000                         # mesh edges actually drawn


@pytest.mark.skipif(not _FIXTURE.exists(), reason="fixture face.png not present")
def test_render_is_centered_in_canvas():
    # the drawn mesh's centroid sits near the canvas centre, not a corner
    mesh = render_landmark_mesh(_FIXTURE)
    ys, xs = np.where((mesh > 0).any(axis=2))
    assert abs(xs.mean() - 864 / 2) < 864 * 0.15
    assert abs(ys.mean() - 1152 * 0.42) < 1152 * 0.15


@pytest.mark.skipif(not _FIXTURE.exists(), reason="fixture face.png not present")
def test_render_depth_map_shape_and_dtype():
    depth = render_depth_map(_FIXTURE)
    assert depth.shape == (1152, 864, 3)
    assert depth.dtype == np.uint8


@pytest.mark.skipif(not _FIXTURE.exists(), reason="fixture face.png not present")
def test_render_depth_map_has_near_and_far_pixels():
    depth = render_depth_map(_FIXTURE)
    g = depth[:, :, 0]
    assert g.max() > 200          # a near (nose-tip-bright) region exists
    assert (g == 0).sum() > 0     # black far background exists


@pytest.mark.skipif(not _FIXTURE.exists(), reason="fixture face.png not present")
def test_render_depth_map_blob_is_centred():
    depth = render_depth_map(_FIXTURE)
    ys, _ = np.where(depth[:, :, 0] > 30)
    # the painted face blob sits near _CENTER_Y (0.42) of the 1152-px canvas
    assert 0.30 < ys.mean() / 1152 < 0.55
