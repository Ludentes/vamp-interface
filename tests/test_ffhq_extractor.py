"""Unit tests for the FFHQ extractor's pure logic.

Heavy GPU-dependent paths (mivolo/fairface/insightface/siglip2/mediapipe) are
covered by the Windows-side smoke run, not here. These tests hit only the
deterministic plumbing: hashing, image decode + resize, NMF atom projection.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from demographic_pc.extract_ffhq_metrics import (
    decode_and_resize,
    compute_image_sha256,
    project_blendshapes_to_atoms,
)


FIXTURE = Path(__file__).parent / "fixtures" / "ffhq_smoke.parquet"
AU_LIBRARY = Path(__file__).resolve().parents[1] / "models" / "blendshape_nmf" / "au_library.npz"
EXPECTED = Path(__file__).parent / "fixtures" / "ffhq_smoke_expected.json"


def test_compute_image_sha256_matches_hashlib():
    payload = b"some png bytes"
    expected = hashlib.sha256(payload).hexdigest()
    assert compute_image_sha256(payload) == expected


def test_decode_and_resize_produces_expected_shape():
    table = pq.read_table(FIXTURE, columns=["image"])
    row = table.column("image").to_pylist()[0]
    rgb = decode_and_resize(row["bytes"], 512)
    assert rgb.shape == (512, 512, 3)
    assert rgb.dtype == np.uint8


def test_project_blendshapes_to_atoms_zero_input_gives_zero_atoms():
    H = np.load(AU_LIBRARY)["H"].astype(np.float32)
    y = np.zeros(52, dtype=np.float32)
    atoms = project_blendshapes_to_atoms(y, H)
    assert atoms.shape == (8,)
    assert np.allclose(atoms, 0.0)


def test_project_blendshapes_reconstruction_roughly_recovers_atom_basis():
    """If we feed in one row of H (an atom pattern), the projection should put
    most of the loading on that atom."""
    H = np.load(AU_LIBRARY)["H"].astype(np.float32)
    for k in range(H.shape[0]):
        y = H[k].copy()
        atoms = project_blendshapes_to_atoms(y, H)
        assert atoms[k] > 0.5 * atoms.sum(), \
            f"atom {k} not dominant: atoms={atoms}"


@pytest.mark.skipif(
    not Path("/tmp/ffhq_smoke_out/ffhq_smoke.pt").exists(),
    reason="run the smoke extractor first (Task 5 step 1)",
)
def test_smoke_output_matches_golden():
    import torch
    p = torch.load("/tmp/ffhq_smoke_out/ffhq_smoke.pt", weights_only=False)
    expected = json.loads(EXPECTED.read_text())
    assert p["image_sha256"] == expected["image_sha256"]
    assert p["format_version"] == expected["format_version"]
