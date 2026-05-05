import numpy as np
from arkit_bridge.extractors import (
    ARKIT_BLENDSHAPE_NAMES,
    extract_arkit_61,
)


def test_arkit_names_length():
    assert len(ARKIT_BLENDSHAPE_NAMES) == 52


def test_extractor_returns_61_or_none():
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    result = extract_arkit_61(img)
    assert result is None or (
        isinstance(result, np.ndarray) and result.shape == (61,)
    )
