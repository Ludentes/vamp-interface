from group_photobooth.backgrounds import load_background, list_backgrounds


def test_list_backgrounds_includes_blank_studio():
    ids = [b["id"] for b in list_backgrounds()]
    assert "blank_studio" in ids


def test_load_background_returns_image_and_metadata():
    img, meta = load_background("blank_studio")
    assert img.shape == (1024, 1024, 3)
    assert meta["id"] == "blank_studio"
    assert tuple(meta["dims"]) == (1024, 1024)
    assert "palette_lab_mean" in meta


def test_load_background_unknown_id_raises():
    import pytest
    with pytest.raises(KeyError):
        load_background("does_not_exist")
