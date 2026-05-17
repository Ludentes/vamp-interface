from arkit_controlnet.axes import AXES, Axis


def test_smile_axis_is_defined_and_well_formed():
    smile = AXES["smile"]
    assert isinstance(smile, Axis)
    assert smile.edit_prompt_a and smile.edit_prompt_b
    assert smile.mix_b == 0.5
    # scale band is ascending and inside the verified smile window
    assert smile.scale_band == sorted(smile.scale_band)
    assert min(smile.scale_band) >= 0.0 and max(smile.scale_band) <= 2.3
    # the ARKit channels this axis is expected to move
    assert "mouthSmileLeft" in smile.target_channels
    assert "mouthSmileRight" in smile.target_channels


def test_three_axes_present():
    assert set(AXES) == {"smile", "pucker", "surprise"}
